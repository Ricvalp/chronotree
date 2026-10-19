"""World TCP alignment without loading a simulator, physics engine, or GPU."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from stmem.visualization.kinematics import (
    JOINT_LOW,
    JOINT_HIGH,
    panda_joint_limits_valid,
    panda_observation_errors,
    panda_tcp_pose,
    panda_tcp_positions,
)

REST = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])


def test_rest_world_pose_and_stick_tcp_are_not_hand_link():
    standard = panda_tcp_pose(REST, task="VideoUnmask")
    np.testing.assert_allclose(standard[:3, 3], [-0.0605, 0, 0.5211], atol=1e-12)
    np.testing.assert_allclose(standard[:3, :3], np.diag([1, -1, -1]), atol=1e-12)
    for task in ("PatternLock", "RouteStick"):
        stick = panda_tcp_pose(REST, task=task)
        np.testing.assert_allclose(stick[:3, 3], [-0.0605, 0, 0.4745], atol=1e-12)
        np.testing.assert_allclose(stick[:3, :3], standard[:3, :3], atol=1e-12)


def test_world_from_base_rotation_translation_and_batched_targets():
    base = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=float)
    targets = np.stack([REST, REST + np.array([0.4, 0, 0, 0, 0, 0, 0])])
    positions = panda_tcp_positions(targets, task="VideoUnmask", base_pose=base)
    np.testing.assert_allclose(positions[0], [1, 2.5545, 3.5211], atol=1e-12)
    np.testing.assert_allclose(
        positions[1], [1 - 0.5545 * np.sin(0.4), 2 + 0.5545 * np.cos(0.4), 3.5211], atol=1e-12
    )
    assert panda_tcp_pose(np.empty((0, 7)), task="VideoUnmask").shape == (0, 4, 4)


def test_observation_validation_detects_wrong_frame_and_rotation_without_fitting():
    state = np.r_[REST, [-0.0605, 0, 0.5211], [0, 1, 0, 0], 0.04]
    errors = panda_observation_errors(state, task="VideoUnmask")
    assert errors["position_error_m"] < 1e-12
    assert errors["rotation_error_rad"] < 1e-7
    state[10:14] *= -1
    assert panda_observation_errors(state, task="VideoUnmask") == errors
    state[7] += 0.615  # a base-relative position is not a world position
    state[10:14] = [1, 0, 0, 0]
    errors = panda_observation_errors(state, task="VideoUnmask")
    assert errors["position_error_m"] == pytest.approx(0.615)
    assert errors["rotation_error_rad"] == pytest.approx(np.pi)


def test_invalid_geometry_rejected_and_out_of_limit_targets_never_clipped():
    assert panda_joint_limits_valid(REST)
    assert panda_joint_limits_valid(JOINT_LOW) and panda_joint_limits_valid(JOINT_HIGH)
    targets = np.stack([REST, REST, REST])
    targets[1, 0] = JOINT_HIGH[0] + 0.2
    targets[2, 0] = np.nan
    assert panda_joint_limits_valid(targets).tolist() == [True, False, False]
    raw = panda_tcp_positions(targets[1], task="VideoUnmask")
    clipped = panda_tcp_positions(np.clip(targets[1], JOINT_LOW, JOINT_HIGH), task="VideoUnmask")
    assert np.linalg.norm(raw - clipped) > 0.05
    for values in ([0] * 8, targets[2]):
        with pytest.raises(ValueError):
            panda_tcp_pose(values, task="VideoUnmask")
    with pytest.raises(ValueError, match="verified"):
        panda_tcp_pose(REST, task="unknown")
    for base in (np.zeros((4, 4)), np.diag([-1, 1, 1, 1]), np.eye(3)):
        with pytest.raises(ValueError, match="base_pose"):
            panda_tcp_pose(REST, task="VideoUnmask", base_pose=base)
    with pytest.raises(ValueError, match="quaternion"):
        panda_observation_errors(np.r_[REST, np.zeros(8)], task="VideoUnmask")


@pytest.mark.skipif(
    not Path("data/prepared/VideoUnmask.h5").exists(),
    reason="optional explicitly prepared real-data diagnostic subset is unavailable",
)
def test_local_prepared_observations_match_fk_in_world_coordinates():
    # Measured qpos and observed TCP are independent of this implementation.
    # Validate every locally prepared frame, including conditioning and execution.
    with h5py.File("data/prepared/VideoUnmask.h5", "r") as source:
        assert len(source) > 0
        for group in source.values():
            for state in group["proprio"]:
                errors = panda_observation_errors(state, task="VideoUnmask")
                assert errors["position_error_m"] < 1e-5
                assert errors["rotation_error_rad"] < 1e-4
