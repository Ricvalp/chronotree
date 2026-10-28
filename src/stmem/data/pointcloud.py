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
