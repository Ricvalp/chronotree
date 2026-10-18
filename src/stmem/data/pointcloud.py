"""Deterministic visible RGB-D geometry using CV world-to-camera extrinsics."""

from __future__ import annotations
import numpy as np
import torch
from torch import Tensor

# World tabletop is z=0, Panda base x=-0.615; geometry-derived, task-independent.
DEFAULT_CROP = ((-0.9, -0.75, -0.05), (0.75, 0.75, 1.25))
CALIBRATION_VERSION = "cv_world_to_camera_mm_v1"


def backproject(rgb, depth_mm, intrinsic, extrinsic, *, crop=DEFAULT_CROP, max_depth_m=4.0) -> Tensor:
    """Return valid xyzrgb points; reject signed-depth invalids before projection."""
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
        return torch.empty((0, 6), dtype=torch.float32)
    pixels = torch.stack((u.float(), v.float(), torch.ones_like(u).float()), -1)
    camera = (pixels @ torch.linalg.inv(k).T) * z[v, u, None]
    world = (camera - e[:, 3]) @ e[:, :3]
    points = torch.cat((world, color[v, u] / 255.0), -1)
    good = torch.isfinite(points).all(-1)
    if crop is not None:
        lo, hi = map(lambda a: torch.as_tensor(a, dtype=torch.float32), crop)
        good &= ((world >= lo) & (world <= hi)).all(-1)
    return points[good]


def sample_points(points: Tensor, count: int, *, voxel_size: float = 0.005) -> tuple[Tensor, Tensor]:
    """Lexicographic voxel representative, then evenly spaced deterministic sampling.

    Input order independent; no duplicated occupied points to pad the shape.
    """
    if count < 1:
        raise ValueError("count must be positive")
    p = points.detach().cpu().float().numpy()
    p = p[np.isfinite(p).all(1)]
    if len(p):
        order = np.lexsort(tuple(p[:, i] for i in reversed(range(6))))
        p = p[order]
        if voxel_size > 0:
            vox = np.floor(p[:, :3] / voxel_size).astype(np.int64)
            _, idx = np.unique(vox, axis=0, return_index=True)
            p = p[np.sort(idx)]
        if len(p) > count:
            p = p[np.linspace(0, len(p) - 1, count, dtype=np.int64)]
    out = torch.zeros((count, 6), dtype=torch.float32)
    mask = torch.zeros(count, dtype=torch.bool)
    if len(p):
        out[: len(p)] = torch.from_numpy(p.copy())
        mask[: len(p)] = True
    return out, mask


def rgbd_to_points(
    observation: dict, calibration: dict, *, count=1024, crop=DEFAULT_CROP, voxel_size=0.005, max_depth_m=4.0
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
    return sample_points(torch.cat(clouds), count, voxel_size=voxel_size)


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
