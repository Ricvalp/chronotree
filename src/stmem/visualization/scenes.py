"""Small current-observation scenes with kinematic joint-command TCP paths.

The payload follows W&B's documented Object3D ``lidar/beta`` RGB points and
labeled boxes, without relying on undocumented vector colors:
https://docs.wandb.ai/models/ref/python/data-types/object3d

This module does not import W&B, render a simulator, observe future point clouds,
or modify model state. Its CPU copies belong only at an explicitly scheduled
diagnostic boundary, never in an online decision timing region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from stmem.data.canonical import Decision
from stmem.evaluation.robomme_adapter import physical_action
from stmem.visualization.kinematics import (
    panda_joint_limits_valid,
    panda_observation_errors,
    panda_tcp_positions,
)


PREDICTED_RGB = (64, 255, 96)
RECORDED_RGB = (255, 145, 0)
CURRENT_RGB = (255, 64, 255)
POSITION_TOLERANCE_M = 1e-4
ROTATION_TOLERANCE_RAD = 1e-3


@dataclass(frozen=True)
class ActionScene:
    """Pass payload/caption directly to ``wandb.Object3D`` at logging time."""

    payload: dict[str, Any]
    caption: str
    metadata: dict[str, Any]


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _prefix_length(valid):
    invalid = np.flatnonzero(~valid)
    return int(invalid[0]) if len(invalid) else len(valid)


def _sample_cloud(points, valid, count):
    observed = points[valid & np.isfinite(points).all(axis=-1)].astype(np.float64, copy=True)
    before = len(observed)
    if before:
        # Independent of input point ordering and model/training random state.
        order = np.lexsort(tuple(observed[:, i] for i in reversed(range(6))))
        observed = observed[order]
        if before > count:
            observed = observed[np.linspace(0, before - 1, count, dtype=np.int64)]
        observed[:, 3:] = np.rint(np.clip(observed[:, 3:], 0, 1) * 255)
    return observed, before


def _colored_points(xyz, color):
    return np.concatenate((xyz, np.broadcast_to(color, (len(xyz), 3))), axis=-1)


def _polyline_points(vertices, count):
    """Dotted straight connections clarify target order, not physical motion."""
    if len(vertices) < 2:
        return np.empty((0, 3), dtype=np.float64)
    nonzero = np.r_[True, np.linalg.norm(np.diff(vertices, axis=0), axis=1) > 1e-12]
    vertices = vertices[nonzero]
    if len(vertices) == 1:
        return vertices
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(vertices, axis=0), axis=1))]
    positions = np.linspace(0, distance[-1], min(count, (len(vertices) - 1) * 16 + 1))
    return np.column_stack([np.interp(positions, distance, vertices[:, axis]) for axis in range(3)])


def _marker(center, color, label, size=0.012):
    # Box corner ordering follows W&B's documented box3d convention.
    offsets = np.asarray(
        [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)],
        dtype=np.float64,
    )
    return {
        "corners": (center + offsets * size / 2).tolist(),
        "label": label,
        "color": (np.asarray(color, dtype=np.float64) / 255).tolist(),
    }


def build_action_scene(
    decision: Decision,
    predicted_actions,
    *,
    max_points: int = 1024,
    path_samples: int = 128,
    base_pose=None,
) -> ActionScene:
    """Compare denormalized [H,8] commands on the current world-coordinate cloud.

    The first seven command components are absolute Panda joint targets in
    radians. TCP positions come from task-specific, pinned robot kinematics.
    Neither RGB-D from future frames nor the gripper command is visualized.

    Both paths end before the first missing/nonfinite/out-of-limit reference
    arm target. Predicted commands use deployment's exact joint clipping and
    task-specific gripper mapping. Nonfinite predictions are rejected clearly.
    Gripper-only reference missingness does not invalidate an arm position.
    Kinematically inconsistent observed robot state raises ValueError, so an
    optional caller can record a diagnostic skip instead of displaying a false
    coordinate alignment.
    """
    if not isinstance(max_points, int) or isinstance(max_points, bool) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    if not isinstance(path_samples, int) or isinstance(path_samples, bool) or path_samples < 2:
        raise ValueError("path_samples must be an integer of at least two")
    predicted = _numpy(predicted_actions).astype(np.float64, copy=False)
    recorded = _numpy(decision.target_chunk).astype(np.float64, copy=False)
    valid = _numpy(decision.target_valid)
    if predicted.ndim != 2 or predicted.shape[-1] != 8 or predicted.shape != recorded.shape:
        raise ValueError("predicted_actions and recorded targets must have the same [H,8] shape")
    if valid.shape != recorded.shape or valid.dtype != np.bool_:
        raise ValueError("target_valid must be boolean [H,8]")
    if not np.isfinite(predicted).all():
        raise ValueError("Cannot display nonfinite predicted action commands")
    commanded = (
        np.stack([physical_action(action, decision.task) for action in predicted])
        if len(predicted)
        else np.empty((0, 8), dtype=np.float32)
    )
    joint_clipped = commanded[:, :7] != predicted[:, :7].astype(np.float32)
    proprio = _numpy(decision.current.proprio).astype(np.float64, copy=False)
    errors = panda_observation_errors(proprio, task=decision.task, base_pose=base_pose)
    if (
        errors["position_error_m"] > POSITION_TOLERANCE_M
        or errors["rotation_error_rad"] > ROTATION_TOLERANCE_RAD
    ):
        raise ValueError(
            "Current observed TCP is inconsistent with pinned Panda world kinematics: "
            f"position={errors['position_error_m']:.6g} m, rotation={errors['rotation_error_rad']:.6g} rad"
        )
    if not bool(panda_joint_limits_valid(proprio[:7])):
        raise ValueError("Current observed Panda joints are outside the pinned robot limits")
    points = _numpy(decision.current.points)
    point_valid = _numpy(decision.current.valid)
    cloud, observed_count = _sample_cloud(points, point_valid, max_points)
    mask_prefix = _prefix_length(valid[:, :7].all(axis=-1))
    recorded_valid = panda_joint_limits_valid(recorded[:, :7])
    recorded_count = min(mask_prefix, _prefix_length(recorded_valid))
    predicted_count = recorded_count
    current = proprio[7:10]
    recorded_xyz = panda_tcp_positions(recorded[:recorded_count, :7], task=decision.task, base_pose=base_pose)
    predicted_xyz = panda_tcp_positions(
        commanded[:predicted_count, :7], task=decision.task, base_pose=base_pose
    )
    recorded_path = np.concatenate((current[None], recorded_xyz), axis=0)
    predicted_path = np.concatenate((current[None], predicted_xyz), axis=0)
    parts = [
        cloud,
        _colored_points(_polyline_points(recorded_path, path_samples), RECORDED_RGB),
        _colored_points(_polyline_points(predicted_path, path_samples), PREDICTED_RGB),
        _colored_points(current[None], CURRENT_RGB),
    ]
    boxes = [_marker(current, CURRENT_RGB, "Observed current TCP")]
    if recorded_count:
        boxes.append(_marker(recorded_xyz[-1], RECORDED_RGB, f"Recorded target +{recorded_count}"))
    if predicted_count:
        boxes.append(_marker(predicted_xyz[-1], PREDICTED_RGB, f"Predicted target +{predicted_count}"))
    horizon = len(recorded)
    metadata = {
        "task": decision.task,
        "episode_id": decision.episode_id,
        "decision_index": int(decision.index),
        "event_id": decision.current.event_id,
        "coordinate_frame": "world",
        "units": "meters; input absolute joint targets in radians",
        "cloud_source": "current_observation_only",
        "observed_valid_points": observed_count,
        "displayed_cloud_points": len(cloud),
        "max_cloud_points": max_points,
        "max_points_per_path": path_samples,
        "horizon": horizon,
        "valid_mask_prefix": mask_prefix,
        "recorded_targets": recorded_count,
        "predicted_targets": predicted_count,
        "recorded_truncated": recorded_count < horizon,
        "predicted_truncated": predicted_count < horizon,
        "recorded_invalid_joint_index": recorded_count if recorded_count < mask_prefix else None,
        "predicted_joint_clipped_components": int(joint_clipped.sum()),
        "predicted_joint_clipped_steps": int(joint_clipped.any(axis=-1).sum()),
        "predicted_actions_raw": predicted.tolist(),
        "predicted_actions_commanded": commanded.tolist(),
        "recorded_actions_raw": [
            [float(value) if np.isfinite(value) else None for value in row] for row in recorded
        ],
        "target_valid": valid.tolist(),
        "calibration": {key: float(value) for key, value in errors.items()},
        "legend": {
            "predicted": list(PREDICTED_RGB),
            "recorded": list(RECORDED_RGB),
            "current": list(CURRENT_RGB),
        },
    }
    caption = (
        f"{decision.task} / {decision.episode_id} / decision {decision.index}. "
        f"Green: predicted ({predicted_count}/{horizon}); orange: recorded ({recorded_count}/{horizon}); "
        "magenta: observed current TCP. Current RGB-D cloud only, world meters. "
        "Dotted lines connect commanded joint-target TCP positions, not simulated motion. "
        f"Predicted joints use deployment clipping ({int(joint_clipped.sum())} components). "
        "Paths stop at missing/invalid reference arm targets; gripper opening is not shown."
    )
    return ActionScene(
        {"type": "lidar/beta", "points": np.concatenate(parts), "boxes": np.asarray(boxes, dtype=object)},
        caption,
        metadata,
    )
