"""Deterministic visible RGB-D geometry using CV world-to-camera extrinsics."""

from __future__ import annotations
from collections.abc import Mapping
import math
import numpy as np
import torch
from torch import Tensor

# World tabletop is z=0, Panda base x=-0.615; geometry-derived, task-independent.
DEFAULT_CROP = ((-0.9, -0.75, -0.05), (0.75, 0.75, 1.25))
CALIBRATION_VERSION = "cv_world_to_camera_mm_v1"
POINT_PROVENANCE_VERSION = "selected_xyzrgb_camera_uv_v1"
CAMERA_NAMES = ("front", "wrist")
# This only validates a source pixel for an already stored FP32 point. It never
# authorizes replacing that point, changing its order, or resampling geometry.
CANONICAL_PIXEL_XYZ_TOLERANCE_M = 1e-6
POINT_SAMPLING_VERSION = "above_table_xy_balanced_v2"
DEFAULT_POINT_SETTINGS = {
    "sampling": "balanced",
    "foreground_fraction": 0.75,
    "table_z": 0.0,
    "height_margin": 0.005,
    "cell_size": 0.1,
}


def resolve_point_sampling(spec: Mapping | None = None) -> dict:
    """Resolve geometry-only point priorities, with no object/segmentation inputs."""
    configured = {} if spec is None else spec
    if not isinstance(configured, Mapping):
        raise ValueError("geometry.point_sampling must be a mapping")
    unknown = configured.keys() - DEFAULT_POINT_SETTINGS.keys()
    if unknown:
        raise ValueError(f"Unknown point sampling settings: {sorted(unknown)}")
    settings = {**DEFAULT_POINT_SETTINGS, **configured}
    for key in ("foreground_fraction", "table_z", "height_margin", "cell_size"):
        settings[key] = float(settings[key])
    if settings["sampling"] not in ("balanced", "uniform"):
        raise ValueError(f"Unknown point sampling method: {settings['sampling']!r}")
    if not all(
        math.isfinite(settings[key])
        for key in ("foreground_fraction", "table_z", "height_margin", "cell_size")
    ):
        raise ValueError("Point sampling geometry must be finite")
    if not 0 <= settings["foreground_fraction"] <= 1 or settings["cell_size"] <= 0:
        raise ValueError("Invalid point foreground fraction or cell size")
    return settings


def point_sampler_settings(config: Mapping | None = None) -> dict:
    """Resolve the full configuration's geometry.point_sampling mapping."""
    geometry = (config or {}).get("geometry", {})
    return resolve_point_sampling(geometry.get("point_sampling"))


def backproject(rgb, depth_mm, intrinsic, extrinsic, *, crop=DEFAULT_CROP, max_depth_m=4.0) -> Tensor:
    """Return valid xyzrgb points; reject signed-depth invalids before projection."""
    return _backproject_with_pixels(rgb, depth_mm, intrinsic, extrinsic, crop=crop, max_depth_m=max_depth_m)[
        0
    ]


def _backproject_with_pixels(rgb, depth_mm, intrinsic, extrinsic, *, crop=DEFAULT_CROP, max_depth_m=4.0):
    """Also return original integer (u,v), following exactly the XYZRGB filtering."""
    depth = torch.as_tensor(np.asarray(depth_mm).copy(), dtype=torch.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    color = torch.as_tensor(np.asarray(rgb).copy(), dtype=torch.float32)
    k = torch.as_tensor(np.asarray(intrinsic).copy(), dtype=torch.float32).reshape(3, 3)
    e = torch.as_tensor(np.asarray(extrinsic).copy(), dtype=torch.float32).reshape(3, 4)
    if depth.ndim != 2 or color.shape != (*depth.shape, 3):
        raise ValueError("RGB/depth shapes must be H,W,3 and H,W[,1]")
    if not torch.isfinite(k).all() or not torch.isfinite(e).all():
        raise ValueError("nonfinite camera calibration")
    if not torch.allclose(e[:, :3] @ e[:, :3].T, torch.eye(3), atol=2e-3):
        raise ValueError("extrinsic rotation is not orthonormal")
    z = depth * 0.001
    valid = torch.isfinite(z) & (z > 0) & (z < max_depth_m)
    v, u = torch.where(valid)
    if len(u) == 0:
        return torch.empty((0, 6), dtype=torch.float32), torch.empty((0, 2), dtype=torch.long)
    pixels = torch.stack((u.float(), v.float(), torch.ones_like(u).float()), -1)
    camera = (pixels @ torch.linalg.inv(k).T) * z[v, u, None]
    world = (camera - e[:, 3]) @ e[:, :3]
    points = torch.cat((world, color[v, u] / 255.0), -1)
    good = torch.isfinite(points).all(-1)
    if crop is not None:
        lo, hi = map(lambda a: torch.as_tensor(a, dtype=torch.float32), crop)
        good &= ((world >= lo) & (world <= hi)).all(-1)
    return points[good], torch.stack((u, v), dim=-1)[good]


def _canonical_voxel_indices(points: Tensor, voxel_size: float) -> np.ndarray:
    """Indices of exactly the original canonical voxel representatives."""
    p = points.detach().cpu().float().numpy()
    indices = np.flatnonzero(np.isfinite(p).all(1))
    p = p[indices]
    if len(p):
        order = np.lexsort(tuple(p[:, i] for i in reversed(range(6))))
        p, indices = p[order], indices[order]
        if voxel_size > 0:
            vox = np.floor(p[:, :3] / voxel_size).astype(np.int64)
            _, idx = np.unique(vox, axis=0, return_index=True)
            indices = indices[np.sort(idx)]
    return indices


def _even(indices: np.ndarray, count: int) -> np.ndarray:
    if count >= len(indices):
        return indices.copy()
    if count <= 0:
        return indices[:0].copy()
    return indices[np.linspace(0, len(indices) - 1, count, dtype=np.int64)]


def _balanced_xy_cells(xyz: np.ndarray, candidates: np.ndarray, count: int, cell_size: float):
    """Equal occupied-cell quotas with capacity redistribution and canonical ties."""
    if count >= len(candidates):
        return candidates.copy()
    if count == 0:
        return candidates[:0].copy()
    cells = np.floor(xyz[candidates, :2] / cell_size).astype(np.int64)
    _, inverse, capacities = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    quotas = np.zeros(len(capacities), dtype=np.int64)
    remaining = count
    while remaining:
        eligible = np.flatnonzero(quotas < capacities)
        if remaining < len(eligible):
            quotas[_even(eligible, remaining)] += 1
            break
        increment = min(remaining // len(eligible), int((capacities[eligible] - quotas[eligible]).min()))
        quotas[eligible] += increment
        remaining -= increment * len(eligible)
    selected = [_even(candidates[inverse == cell], int(quota)) for cell, quota in enumerate(quotas) if quota]
    return np.sort(np.concatenate(selected))


def sample_points(
    points: Tensor,
    count: int = 2048,
    *,
    voxel_size: float = 0.005,
    sampling: str = "balanced",
    foreground_fraction: float = 0.75,
    table_z: float = 0.0,
    height_margin: float = 0.005,
    cell_size: float = 0.1,
    return_indices: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Voxelize, prioritize elevated geometry, then retain broad scene coverage.

    The default reserves up to 75% of slots for points above calibrated table z
    plus 5 mm, balancing their counts across occupied 10 cm XY cells. Remaining
    slots sample all unselected voxel representatives, including table surfaces.
    Sparse foreground transfers unused slots to whole-scene coverage. Return
    canonical FP32 xyzrgb and a valid mask; padding never duplicates observations.
    Explicit ``sampling='uniform'`` reproduces the previous sampling algorithm.
    """
    if points.ndim != 2 or points.shape[1] != 6:
        raise ValueError("Expected observed xyzrgb [N,6]")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not math.isfinite(voxel_size) or voxel_size < 0:
        raise ValueError("voxel_size must be finite and nonnegative")
    settings = resolve_point_sampling(
        {
            "sampling": sampling,
            "foreground_fraction": foreground_fraction,
            "table_z": table_z,
            "height_margin": height_margin,
            "cell_size": cell_size,
        }
    )
    raw_indices = _canonical_voxel_indices(points, voxel_size)
    raw = points.detach().cpu().float().numpy()
    p = raw[raw_indices]
    if settings["sampling"] == "uniform":
        selected = _even(np.arange(len(p)), count)
    else:
        foreground = np.flatnonzero(p[:, 2] > settings["table_z"] + settings["height_margin"])
        quota = min(len(foreground), math.floor(count * settings["foreground_fraction"]))
        chosen = _balanced_xy_cells(p[:, :3], foreground, quota, settings["cell_size"])
        unused = np.ones(len(p), dtype=bool)
        unused[chosen] = False
        remainder = _even(np.flatnonzero(unused), min(count - len(chosen), int(unused.sum())))
        selected = np.sort(np.concatenate([chosen, remainder]))
    p = p[selected]
    out = torch.zeros((count, 6), dtype=torch.float32)
    mask = torch.zeros(count, dtype=torch.bool)
    if len(p):
        out[: len(p)] = torch.from_numpy(p.copy())
        mask[: len(p)] = True
    if return_indices:
        indices = torch.full((count,), -1, dtype=torch.long)
        indices[: len(p)] = torch.from_numpy(raw_indices[selected].copy())
        return out, mask, indices
    return out, mask


def rgbd_to_points(
    observation: dict,
    calibration: dict,
    *,
    count=2048,
    crop=DEFAULT_CROP,
    voxel_size=0.005,
    max_depth_m=4.0,
    sampling: str = "balanced",
    foreground_fraction: float = 0.75,
    table_z: float = 0.0,
    height_margin: float = 0.005,
    cell_size: float = 0.1,
):
    clouds = [
        backproject(
            observation[f"{camera}_rgb"],
            observation[f"{camera}_depth"],
            calibration[f"{camera}_camera_intrinsic"],
            observation[f"{camera}_camera_extrinsic"],
            crop=crop,
            max_depth_m=max_depth_m,
        )
        for camera in ("front", "wrist")
    ]
    return sample_points(
        torch.cat(clouds),
        count,
        voxel_size=voxel_size,
        sampling=sampling,
        foreground_fraction=foreground_fraction,
        table_z=table_z,
        height_margin=height_margin,
        cell_size=cell_size,
    )


def rgbd_to_points_with_provenance(observation: dict, calibration: dict, **kwargs):
    """Canonical XYZRGB plus observing-camera index and original integer pixels.

    Returns ``points[N,6], valid[N], camera[N], uv[N,2]``. Front is 0 and
    wrist is 1. Padding camera/uv are -1. Provenance is carried by selection
    indices, never concatenated into the geometry used to select points.
    Consequently voxel tie-breaking and point order are identical to
    :func:`rgbd_to_points`, including overlapping camera observations.
    """
    crop = kwargs.pop("crop", DEFAULT_CROP)
    max_depth_m = kwargs.pop("max_depth_m", 4.0)
    clouds, cameras, pixels = [], [], []
    for index, camera in enumerate(CAMERA_NAMES):
        points, uv = _backproject_with_pixels(
            observation[f"{camera}_rgb"],
            observation[f"{camera}_depth"],
            calibration[f"{camera}_camera_intrinsic"],
            observation[f"{camera}_camera_extrinsic"],
            crop=crop,
            max_depth_m=max_depth_m,
        )
        clouds.append(points)
        cameras.append(torch.full((len(points),), index, dtype=torch.long))
        pixels.append(uv)
    points, valid, indices = sample_points(torch.cat(clouds), return_indices=True, **kwargs)
    source = torch.full((len(points),), -1, dtype=torch.long)
    uv = torch.full((len(points), 2), -1, dtype=torch.long)
    source[valid] = torch.cat(cameras)[indices[valid]]
    uv[valid] = torch.cat(pixels)[indices[valid]]
    return points, valid, source, uv


@torch.no_grad()
def canonical_point_provenance(
    observation: dict,
    calibration: dict,
    points: Tensor,
    valid: Tensor,
    *,
    max_depth_m: float = 4.0,
    voxel_size: float = 0.005,
) -> tuple[Tensor, Tensor]:
    """Recover observing camera/pixel for immutable, already selected points.

    Re-running voxelization can select different points after tiny CPU BLAS
    rounding changes. Instead, project each canonical XYZ into each camera in
    FP64, then validate its nearest integer pixel against that pixel's actual
    depth-backprojected FP32 XYZRGB. RGB must match exactly and every XYZ
    component must agree within one micrometre. The input points and mask are
    never modified, reordered or resampled.

    A point needs one matching camera. If both match, only bitwise-identical
    source XYZRGB is accepted when voxelization was enabled: its original
    stable merge order selected front before wrist in that exact tie. Without
    voxelization both duplicate sources survive and remain ambiguous. Other
    ambiguity and absent correspondence fail closed. Invalid padding returns
    camera/UV -1. Results are CPU tensors.
    """
    if not isinstance(points, Tensor) or points.ndim != 2 or points.shape[1] != 6:
        raise ValueError("Canonical points must be a Tensor [N,6]")
    if points.dtype != torch.float32:
        raise ValueError("Canonical points must retain their stored FP32 dtype")
    if not isinstance(valid, Tensor) or valid.dtype != torch.bool or valid.shape != (len(points),):
        raise ValueError("Canonical valid mask must be bool [N]")
    if not torch.isfinite(points).all():
        raise ValueError("Canonical points contain nonfinite values")
    if not math.isfinite(max_depth_m) or max_depth_m <= 0:
        raise ValueError("max_depth_m must be finite and positive")
    if not math.isfinite(voxel_size) or voxel_size < 0:
        raise ValueError("voxel_size must be finite and nonnegative")
    canonical = points.detach().cpu()
    valid_indices = torch.where(valid.detach().cpu())[0]
    source = torch.full((len(points),), -1, dtype=torch.long)
    uv = torch.full((len(points), 2), -1, dtype=torch.long)
    if not len(valid_indices):
        return source, uv
    selected = canonical[valid_indices]
    matches = torch.zeros((len(selected), len(CAMERA_NAMES)), dtype=torch.bool)
    candidate_points = torch.zeros((len(selected), len(CAMERA_NAMES), 6), dtype=torch.float32)
    candidate_uv = torch.full((len(selected), len(CAMERA_NAMES), 2), -1, dtype=torch.long)
    for camera_index, camera in enumerate(CAMERA_NAMES):
        rgb, depth = observation[f"{camera}_rgb"], observation[f"{camera}_depth"]
        intrinsic = calibration[f"{camera}_camera_intrinsic"]
        extrinsic = observation[f"{camera}_camera_extrinsic"]
        raw_points, raw_uv = _backproject_with_pixels(
            rgb, depth, intrinsic, extrinsic, crop=None, max_depth_m=max_depth_m
        )
        if not len(raw_points):
            continue
        height, width = np.asarray(rgb).shape[:2]
        # Match the calibration values used by FP32 backprojection, while doing
        # the inverse projection arithmetic in FP64 to choose an integer pixel.
        k = torch.as_tensor(np.asarray(intrinsic).copy(), dtype=torch.float32).double().reshape(3, 3)
        e = torch.as_tensor(np.asarray(extrinsic).copy(), dtype=torch.float32).double().reshape(3, 4)
        # Invert the actual row-vector backprojection, even when a recorded
        # rotation has small accepted departures from exact orthonormality.
        camera_xyz = selected[:, :3].double() @ torch.linalg.inv(e[:, :3]) + e[:, 3]
        projected = camera_xyz @ k.T
        projectable = torch.isfinite(projected).all(-1) & (projected[:, 2] > 0)
        pixel = torch.zeros((len(selected), 2), dtype=torch.float64)
        pixel[projectable] = projected[projectable, :2] / projected[projectable, 2:3]
        rounded = pixel.round()
        projectable &= (
            (rounded[:, 0] >= 0) & (rounded[:, 0] < width) & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
        )
        selected_indices = torch.where(projectable)[0]
        if not len(selected_indices):
            continue
        integer_uv = rounded[selected_indices].long()
        lookup = torch.full((height, width), -1, dtype=torch.long)
        lookup[raw_uv[:, 1], raw_uv[:, 0]] = torch.arange(len(raw_points))
        raw_indices = lookup[integer_uv[:, 1], integer_uv[:, 0]]
        present = raw_indices >= 0
        selected_indices, integer_uv, raw_indices = (
            selected_indices[present],
            integer_uv[present],
            raw_indices[present],
        )
        candidate = raw_points[raw_indices]
        expected = selected[selected_indices]
        agrees = (candidate[:, 3:] == expected[:, 3:]).all(-1) & (
            (candidate[:, :3] - expected[:, :3]).abs() <= CANONICAL_PIXEL_XYZ_TOLERANCE_M
        ).all(-1)
        selected_indices, candidate, integer_uv = (
            selected_indices[agrees],
            candidate[agrees],
            integer_uv[agrees],
        )
        matches[selected_indices, camera_index] = True
        candidate_points[selected_indices, camera_index] = candidate
        candidate_uv[selected_indices, camera_index] = integer_uv
    absent = ~matches.any(-1)
    duplicate = matches.all(-1)
    identical_sources = (
        candidate_points[:, 0].contiguous().view(torch.int32)
        == candidate_points[:, 1].contiguous().view(torch.int32)
    ).all(-1)
    ambiguous = duplicate & (~identical_sources if voxel_size > 0 else torch.ones_like(duplicate))
    if absent.any() or ambiguous.any():
        raise ValueError(
            "Cannot recover canonical point provenance: "
            f"missing={int(absent.sum())}, ambiguous={int(ambiguous.sum())}; "
            f"first missing indices={valid_indices[absent][:8].tolist()}, "
            f"first ambiguous indices={valid_indices[ambiguous][:8].tolist()}. "
            "Requires exact RGB and unique raw-pixel XYZ agreement within "
            f"{CANONICAL_PIXEL_XYZ_TOLERANCE_M:g} m; canonical geometry was not changed."
        )
    chosen = matches.to(torch.int64).argmax(-1)
    source[valid_indices] = chosen
    uv[valid_indices] = candidate_uv[torch.arange(len(selected)), chosen]
    return source, uv


def proprioception(observation: dict) -> Tensor:
    q = torch.as_tensor(np.asarray(observation["joint_state"]).copy(), dtype=torch.float32).flatten()[:7]
    ee = torch.as_tensor(np.asarray(observation["eef_state"]).copy(), dtype=torch.float32).flatten()
    if len(q) != 7 or len(ee) != 6:
        raise ValueError("expected observed joint_state7/eef_state6")
    # Extrinsic XYZ Euler = quaternion qz(yaw)*qy(pitch)*qx(roll), wxyz.
    r, p, y = ee[3:] / 2
    cr, cp, cy = torch.cos(torch.stack((r, p, y)))
    sr, sp, sy = torch.sin(torch.stack((r, p, y)))
    quat = torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )
    # A deterministic hemisphere removes physically equivalent sign ambiguity.
    quat = quat * torch.where(quat[:1] < 0, -1.0, 1.0)
    grip = (
        torch.as_tensor(np.asarray(observation["gripper_state"]).copy(), dtype=torch.float32)
        .flatten()
        .mean()
        .reshape(1)
    )
    out = torch.cat((q, ee[:3], quat, grip))
    if not torch.isfinite(out).all():
        raise ValueError("nonfinite observed robot state")
    return out
