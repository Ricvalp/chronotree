"""Simulator-free Panda TCP kinematics for diagnostic world-coordinate overlays.

Geometry is transcribed from ManiSkill's ``panda_v3.urdf`` / ``panda_stick.urdf``
at revision 07be6fbc66350ddca200abfb0a11b692f078f7fd (Apache-2.0; license retained
under third_party/licenses). RoboMME 6503b26e00ff338611fceb039417fbc64826ccc8 uses
panda_wristcam except for PatternLock/RouteStick, which use panda_stick. Its
DemonstrationWrapper observes ``agent.tcp.pose``, i.e. world ``panda_hand_tcp``.
All sixteen pinned task loaders use the fixed base pose [-0.615, 0, 0], identity
rotation. An explicit base transform supports independently calibrated scenes;
this module never fits a transform to make inconsistent observations agree.

These are TCP positions of absolute joint targets, not simulated trajectories.
No dynamics, collision checks, joint clipping, or gripper geometry are implied.
"""

from __future__ import annotations

import numpy as np

from stmem.data.download import TASKS

MANISKILL_REVISION = "07be6fbc66350ddca200abfb0a11b692f078f7fd"
TCP_LINK = "panda_hand_tcp"
JOINT_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
JOINT_LOW.setflags(write=False)
JOINT_HIGH.setflags(write=False)

# URDF origin translation and roll, followed by each joint's local z rotation.
_ORIGINS = (
    ((0.0, 0.0, 0.333), 0.0),
    ((0.0, 0.0, 0.0), -np.pi / 2),
    ((0.0, -0.316, 0.0), np.pi / 2),
    ((0.0825, 0.0, 0.0), np.pi / 2),
    ((-0.0825, 0.384, 0.0), -np.pi / 2),
    ((0.0, 0.0, 0.0), np.pi / 2),
    ((0.088, 0.0, 0.0), np.pi / 2),
)


def panda_variant(task: str) -> str:
    """Resolve only the audited benchmark tasks; never guess an unknown robot."""
    if task not in TASKS:
        raise ValueError(f"No verified Panda kinematics for task {task!r}")
    return "panda_stick" if task in ("PatternLock", "RouteStick") else "panda_wristcam"


def _joints(joints) -> np.ndarray:
    values = np.asarray(joints, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 7:
        raise ValueError("Absolute arm joint positions must have shape [...,7] in radians")
    return values


def panda_joint_limits_valid(joints) -> np.ndarray:
    """Validate at the controller's FP32 precision without modifying FK targets.

    Decimal URDF limits need not be exactly representable as FP32. Comparing
    stored/clipped FP32 commands against FP64 decimal limits falsely rejects
    some valid boundary commands. Use the same representation as deployment.
    """
    values = _joints(joints)
    with np.errstate(over="ignore"):
        commands = values.astype(np.float32)
    return (
        np.isfinite(commands)
        & (commands >= JOINT_LOW.astype(np.float32))
        & (commands <= JOINT_HIGH.astype(np.float32))
    ).all(axis=-1)


def _base_transform(base_pose) -> np.ndarray:
    if base_pose is None:
        base = np.eye(4)
        base[0, 3] = -0.615
        return base
    base = np.asarray(base_pose, dtype=np.float64)
    if base.shape != (4, 4) or not np.isfinite(base).all():
        raise ValueError("base_pose must be a finite [4,4] world-from-base transform")
    rotation = base[:3, :3]
    if (
        not np.allclose(base[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
        or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError("base_pose must be a rigid right-handed transform")
    return base


def _z_rotation(angle) -> np.ndarray:
    angle = np.asarray(angle)
    matrix = np.zeros((*angle.shape, 4, 4))
    c, s = np.cos(angle), np.sin(angle)
    matrix[..., 0, 0] = matrix[..., 1, 1] = c
    matrix[..., 0, 1], matrix[..., 1, 0] = -s, s
    matrix[..., 2, 2] = matrix[..., 3, 3] = 1
    return matrix


def panda_tcp_pose(joints, *, task: str, base_pose=None) -> np.ndarray:
    """Map finite absolute arm targets [...,7] to world TCP transforms [...,4,4].

    ``base_pose`` is world-from-base, not its inverse. Out-of-limit finite targets
    remain unchanged so diagnostic overlays can report raw prediction errors.
    Gripper commands are excluded because neither robot's TCP depends on them.
    """
    variant = panda_variant(task)
    values = _joints(joints)
    if not np.isfinite(values).all():
        raise ValueError("Cannot project nonfinite joint targets")
    transform = np.broadcast_to(_base_transform(base_pose), (*values.shape[:-1], 4, 4)).copy()
    for i, (xyz, roll) in enumerate(_ORIGINS):
        origin = np.eye(4)
        c, s = np.cos(roll), np.sin(roll)
        origin[1:3, 1:3] = [[c, -s], [s, c]]
        origin[:3, 3] = xyz
        transform = transform @ origin @ _z_rotation(values[..., i])
    hand = _z_rotation(-np.pi / 4)
    hand[2, 3] = 0.107  # fixed panda_joint8, then panda_hand_joint
    transform = transform @ hand
    tcp = np.eye(4)
    tcp[2, 3] = 0.15 if variant == "panda_stick" else 0.1034
    return transform @ tcp


def panda_tcp_positions(joints, *, task: str, base_pose=None) -> np.ndarray:
    """Return [...,3] world coordinates in meters for seven absolute joint targets."""
    return panda_tcp_pose(joints, task=task, base_pose=base_pose)[..., :3, 3]


def panda_observation_errors(proprio, *, task: str, base_pose=None) -> dict[str, float]:
    """Compare FK to the current observed world TCP, without fitting any transform.

    Canonical proprio is qpos7, xyz3, quaternion wxyz4, finger opening1. Callers
    should skip an overlay with an explicit reason if these errors exceed their
    tolerances. Synthetic/unphysical observations need not satisfy this check.
    """
    state = np.asarray(proprio, dtype=np.float64)
    if state.shape != (15,) or not np.isfinite(state).all():
        raise ValueError("Expected finite canonical proprioception [15]")
    norm = np.linalg.norm(state[10:14])
    if norm < 1e-8:
        raise ValueError("Observed TCP quaternion has zero norm")
    w, x, y, z = state[10:14] / norm
    observed_rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    expected = panda_tcp_pose(state[:7], task=task, base_pose=base_pose)
    relative_rotation = expected[:3, :3].T @ observed_rotation
    angle = np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))
    return {
        "position_error_m": float(np.linalg.norm(expected[:3, 3] - state[7:10])),
        "rotation_error_rad": float(angle),
    }
