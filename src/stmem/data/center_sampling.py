"""Deterministic CPU selection of observed-point visual anchors.

Selection uses only observed xyzrgb and a calibrated table-height prior. Prepare
and store the returned original-cloud indices once per frame; packing/encoders
reuse them rather than running clustering on the training GPU. Both current and
historical representations use the same frame-local geometry. RGB is used only
to break geometric ties canonically; object labels are never accepted.

The balanced selector refines stratified FPS seeds with weighted Lloyd updates,
snapping each weighted centroid to an observed point in its assigned cluster.
Each occupied 10 cm world XYZ cell contributes total weight one. Foreground
centers are refined first, then fixed while global centers address residual
coverage. The default iteration ceiling is 20 PER STAGE, not 20 in total.
"""

from __future__ import annotations

import math
import hashlib
import json
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor


CENTER_SAMPLING_VERSION = "cell_balanced_observed_medoids_v1"
DEFAULT_CENTER_SETTINGS = {
    "method": "balanced_cluster",
    "foreground_fraction": 0.75,
    "table_z": 0.0,
    "height_margin": 0.005,
    "cell_size": 0.1,
    "max_iterations": 20,
}


def center_sampler_settings(config: Mapping | None = None) -> dict:
    """Resolve the full config's model.encoder_options.center_sampling settings."""
    config = config or {}
    model = config.get("model", {})
    options = model.get("encoder_options", {})
    return resolve_center_sampling(options.get("center_sampling"))


def resolve_center_sampling(spec: Mapping | None = None) -> dict:
    """Normalize a selector spec; None resolves to the new balanced default."""
    configured = {} if spec is None else spec
    if not isinstance(configured, Mapping):
        raise ValueError("model.encoder_options.center_sampling must be a mapping")
    unknown = configured.keys() - DEFAULT_CENTER_SETTINGS.keys()
    if unknown:
        raise ValueError(f"Unknown center sampling settings: {sorted(unknown)}")
    settings = {**DEFAULT_CENTER_SETTINGS, **configured}
    for key in ("foreground_fraction", "table_z", "height_margin", "cell_size"):
        settings[key] = float(settings[key])
    _validate_parameters(0, **settings)
    return settings


def center_sampling_signature(spec: Mapping | None = None) -> str:
    """Fingerprint implementation version and normalized selection settings."""
    payload = {"version": CENTER_SAMPLING_VERSION, "settings": resolve_center_sampling(spec)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def center_counts(config: Mapping | None = None) -> list[int]:
    """Distinct positive current/history per-frame quotas for the configured model."""
    model = (config or {}).get("model", {})
    current = model.get("current_tokens", 32)
    block = model.get("block_size", 8)
    visual = model.get("visual_tokens", 256)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (current, block, visual)):
        raise ValueError("current_tokens, block_size and visual_tokens must be positive integers")
    if visual < block:
        raise ValueError("visual_tokens must be at least block_size")
    low, remainder = divmod(visual, block)
    return sorted({current, low, low + bool(remainder)})


def _validate_parameters(
    count, *, method, foreground_fraction, table_z, height_margin, cell_size, max_iterations
):
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if method not in ("balanced_cluster", "fps"):
        raise ValueError(f"Unknown center sampling method: {method!r}")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not all(math.isfinite(x) for x in (foreground_fraction, table_z, height_margin, cell_size)):
        raise ValueError("Finite geometry parameters required")
    if not 0 <= foreground_fraction <= 1 or cell_size <= 0:
        raise ValueError("Invalid foreground fraction or cell size")


def select_stratified_fps_centers(
    points: Tensor,
    count: int,
    *,
    foreground_fraction: float = 0.75,
    table_z: float = 0.0,
    height_margin: float = 0.005,
) -> Tensor:
    """Foreground-first FPS initialization; return distinct original indices.

    Work is always performed on CPU, even if an external diagnostic supplies a
    CUDA tensor. No RNG is used. This is stratified FPS, not the old unstratified
    encoder fallback.
    """
    _validate_parameters(
        count,
        method="fps",
        foreground_fraction=foreground_fraction,
        table_z=table_z,
        height_margin=height_margin,
        cell_size=0.1,
        max_iterations=20,
    )
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError("Expected observed xyzrgb [N,6]")
    cpu_points = points.detach().cpu().float()
    usable = torch.where(torch.isfinite(cpu_points).all(-1))[0]
    for axis in reversed(range(6)):
        usable = usable[torch.argsort(cpu_points[usable, axis], stable=True)]
    p = cpu_points[usable]
    count = min(count, len(p))
    if not count:
        return usable[:0].to(points.device)
    foreground = p[:, 2] > table_z + height_margin
    quota = min(int(foreground.sum()), math.floor(count * foreground_fraction))
    chosen = torch.zeros(len(p), dtype=torch.bool)
    distance = torch.full((len(p),), float("inf"))
    result = []
    for i in range(count):
        eligible = (~chosen) & (foreground if i < quota else torch.ones_like(foreground))
        index = distance.masked_fill(~eligible, -1.0).argmax()
        result.append(index)
        chosen[index] = True
        distance = torch.minimum(distance, (p[:, :3] - p[index, :3]).square().sum(-1))
    return usable[torch.stack(result)].to(points.device)


def _cell_weights(xyz: np.ndarray, cell_size: float) -> np.ndarray:
    """Each occupied world-grid XYZ cell contributes total weight exactly one."""
    if not len(xyz):
        return np.empty(0, dtype=np.float64)
    cells = np.floor(xyz / cell_size)
    limit = np.iinfo(np.int64)
    if (cells <= limit.min).any() or (cells >= limit.max).any():
        raise ValueError("Coordinates/cell size exceed supported grid integer range")
    _, inverse, counts = np.unique(cells.astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def _assign(xyz: np.ndarray, anchors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = ((xyz[:, None, :] - xyz[anchors][None, :, :]) ** 2).sum(-1)
    assignment = distance.argmin(axis=1)
    # Unique anchors own their exact observed points. This only resolves
    # zero-distance ties and keeps clusters nonempty even at coincident XYZ.
    assignment[anchors] = np.arange(len(anchors))
    return assignment, distance[np.arange(len(xyz)), assignment]


def _lloyd(
    xyz: np.ndarray,
    weights: np.ndarray,
    anchors: np.ndarray,
    *,
    fixed_count: int,
    max_iterations: int,
) -> tuple[np.ndarray, dict]:
    anchors = anchors.copy()
    if not len(anchors):
        return anchors, {"iterations": 0, "converged": True, "weighted_squared_distortion": []}
    _, distance = _assign(xyz, anchors)
    objective = [float(np.dot(weights, distance))]
    converged = False
    iterations = 0
    for _ in range(max_iterations):
        assignment, _ = _assign(xyz, anchors)
        updated = anchors.copy()
        for cluster in range(fixed_count, len(anchors)):
            members = np.flatnonzero(assignment == cluster)
            if not len(members):
                # Defensive only: anchor ownership above normally prevents this.
                # Preserve the unique old anchor rather than duplicate another.
                continue
            centroid = np.average(xyz[members], axis=0, weights=weights[members])
            residual = ((xyz[members] - centroid) ** 2).sum(-1)
            # Members retain canonical xyzrgb order, so equal distances have a
            # deterministic, input-order-independent tie break.
            updated[cluster] = members[residual.argmin()]
        if len(np.unique(updated)) != len(updated):
            raise RuntimeError("Disjoint cluster update produced duplicate anchors")
        iterations += 1
        stable = np.array_equal(updated, anchors)
        anchors = updated
        _, distance = _assign(xyz, anchors)
        objective.append(float(np.dot(weights, distance)))
        if stable:
            converged = True
            break
    return anchors, {
        "iterations": iterations,
        "converged": converged,
        "weighted_squared_distortion": objective,
    }


def _select(
    points: Tensor,
    count: int,
    *,
    foreground_fraction: float,
    table_z: float,
    height_margin: float,
    cell_size: float,
    max_iterations: int,
) -> tuple[Tensor, dict]:
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError("Expected observed xyzrgb [N,6]")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not all(math.isfinite(x) for x in (foreground_fraction, table_z, height_margin, cell_size)):
        raise ValueError("Finite geometry parameters required")
    if not 0 <= foreground_fraction <= 1 or cell_size <= 0:
        raise ValueError("Invalid foreground fraction or cell size")

    # Match the project's FP32 geometry contract and the compared point clouds.
    raw = points.detach().cpu().float().numpy()
    original = np.flatnonzero(np.isfinite(raw).all(axis=1))
    finite = raw[original]
    if len(finite):
        order = np.lexsort(tuple(finite[:, axis] for axis in reversed(range(6))))
        original, finite = original[order], finite[order]
    count = min(count, len(finite))
    report = {
        "method": "cell_balanced_observed_medoid_lloyd",
        "point_weighting": "one_total_weight_per_occupied_world_xyz_cell",
        "grid_origin": [0.0, 0.0, 0.0],
        "cell_size_m": float(cell_size),
        "valid_points": len(finite),
        "centers": count,
        "max_iterations_per_stage": max_iterations,
        "privileged_information_used": False,
    }
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=points.device), report

    xyz = finite[:, :3].astype(np.float64)
    foreground_indices = np.flatnonzero(finite[:, 2] > table_z + height_margin)
    foreground_count = min(len(foreground_indices), math.floor(count * foreground_fraction))
    seeds = (
        select_stratified_fps_centers(
            torch.from_numpy(finite),
            count,
            foreground_fraction=foreground_fraction,
            table_z=table_z,
            height_margin=height_margin,
        )
        .cpu()
        .numpy()
    )
    if len(seeds) != count or len(np.unique(seeds)) != count:
        raise RuntimeError("FPS initialization must return distinct observed points")
    foreground_weights = _cell_weights(xyz[foreground_indices], cell_size)
    if foreground_count:
        inverse = np.full(len(finite), -1, dtype=np.int64)
        inverse[foreground_indices] = np.arange(len(foreground_indices))
        local_seeds = inverse[seeds[:foreground_count]]
        if (local_seeds < 0).any():
            raise RuntimeError("Foreground seed outside geometric foreground")
        local_anchors, foreground_report = _lloyd(
            xyz[foreground_indices],
            foreground_weights,
            local_seeds,
            fixed_count=0,
            max_iterations=max_iterations,
        )
        foreground_anchors = foreground_indices[local_anchors]
    else:
        foreground_anchors = np.empty(0, dtype=np.int64)
        foreground_report = {"iterations": 0, "converged": True, "weighted_squared_distortion": []}

    # A foreground medoid may coincide with a formerly global FPS seed. Keep
    # every other global seed; replace collisions using weighted residual error.
    global_weights = _cell_weights(xyz, cell_size)
    fixed = set(foreground_anchors.tolist())
    retained = [int(index) for index in seeds[foreground_count:] if int(index) not in fixed]
    used = set(retained) | fixed
    collisions = count - foreground_count - len(retained)
    while len(retained) < count - foreground_count:
        existing = np.asarray(foreground_anchors.tolist() + retained, dtype=np.int64)
        if len(existing):
            distance = ((xyz[:, None, :] - xyz[existing][None, :, :]) ** 2).sum(-1).min(axis=1)
        else:
            distance = np.ones(len(xyz), dtype=np.float64)
        score = global_weights * distance
        score[np.asarray(sorted(used), dtype=np.int64)] = -np.inf
        index = int(score.argmax())
        retained.append(index)
        used.add(index)
    global_seeds = np.asarray(retained, dtype=np.int64)
    combined = np.concatenate([foreground_anchors, global_seeds])
    if len(global_seeds):
        final, global_report = _lloyd(
            xyz, global_weights, combined, fixed_count=foreground_count, max_iterations=max_iterations
        )
    else:
        final = combined
        global_report = {"iterations": 0, "converged": True, "weighted_squared_distortion": []}
    if len(final) != count or len(np.unique(final)) != count:
        raise RuntimeError("Expected exact distinct center budget")
    if not np.array_equal(final[:foreground_count], foreground_anchors):
        raise RuntimeError("Global refinement moved fixed foreground centers")
    report.update(
        foreground_center_quota=foreground_count,
        global_center_quota=count - foreground_count,
        foreground_stage=foreground_report,
        global_stage=global_report,
        repaired_global_seed_collisions=collisions,
        foreground_threshold_z_m=float(table_z + height_margin),
    )
    result = torch.as_tensor(original[final].copy(), dtype=torch.long, device=points.device)
    return result, report


def select_cluster_centers(
    points: Tensor,
    count: int,
    *,
    foreground_fraction: float = 0.75,
    table_z: float = 0.0,
    height_margin: float = 0.005,
    cell_size: float = 0.1,
    max_iterations: int = 20,
) -> Tensor:
    """Return distinct original-cloud indices; output device matches input device.

    Each of the foreground and fixed-foreground/global refinement stages runs
    for at most max_iterations; neither stage uses future frames or any labels.
    """
    indices, _ = _select(
        points,
        count,
        foreground_fraction=foreground_fraction,
        table_z=table_z,
        height_margin=height_margin,
        cell_size=cell_size,
        max_iterations=max_iterations,
    )
    return indices


def select_cluster_centers_with_diagnostics(
    points: Tensor,
    count: int,
    *,
    foreground_fraction: float = 0.75,
    table_z: float = 0.0,
    height_margin: float = 0.005,
    cell_size: float = 0.1,
    max_iterations: int = 20,
) -> tuple[Tensor, dict]:
    """Same selection, plus per-stage convergence and geometric objective only."""
    return _select(
        points,
        count,
        foreground_fraction=foreground_fraction,
        table_z=table_z,
        height_margin=height_margin,
        cell_size=cell_size,
        max_iterations=max_iterations,
    )


def select_center_indices(
    points: Tensor,
    count: int = 32,
    *,
    method: str = "balanced_cluster",
    foreground_fraction: float = 0.75,
    table_z: float = 0.0,
    height_margin: float = 0.005,
    cell_size: float = 0.1,
    max_iterations: int = 20,
) -> Tensor:
    """Return min(count, finite-point count) unique ORIGINAL-cloud indices.

    No padding is fabricated. Storage callers can pad the result with -1 and
    keep that sentinel masked. Run on acquired/prepared observations or loader
    workers, then reuse the cached indices in both encoder paths.
    """
    _validate_parameters(
        count,
        method=method,
        foreground_fraction=foreground_fraction,
        table_z=table_z,
        height_margin=height_margin,
        cell_size=cell_size,
        max_iterations=max_iterations,
    )
    if method == "fps":
        return select_stratified_fps_centers(
            points,
            count,
            foreground_fraction=foreground_fraction,
            table_z=table_z,
            height_margin=height_margin,
        )
    return select_cluster_centers(
        points,
        count,
        foreground_fraction=foreground_fraction,
        table_z=table_z,
        height_margin=height_margin,
        cell_size=cell_size,
        max_iterations=max_iterations,
    )


def prepare_center_indices(
    points: Tensor,
    valid: Tensor,
    counts,
    settings: Mapping | None = None,
) -> dict[int, Tensor]:
    """Prepare reusable fixed-shape center indices; -1 denotes unavailable slots.

    Masked finite padding is excluded before selection. Returned nonnegative
    indices refer to the original padded cloud, never the compact valid subset.
    Both acquisition and HDF5 preparation call this same CPU geometry routine.
    """
    if points.ndim != 2 or points.shape[-1] != 6:
        raise ValueError("Expected observed xyzrgb [N,6]")
    if valid.shape != points.shape[:1] or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [N] with True meaning observed")
    resolved = resolve_center_sampling(settings)
    cpu_points = points.detach().cpu().float()
    observed = torch.where(valid.detach().cpu() & torch.isfinite(cpu_points).all(-1))[0]
    cloud = cpu_points[observed]
    result = {}
    for count in counts:
        _validate_parameters(count, **resolved)
        chosen = select_center_indices(cloud, count, **resolved)
        indices = torch.full((count,), -1, dtype=torch.long)
        indices[: len(chosen)] = observed[chosen]
        result[count] = indices.to(points.device)
    return result
