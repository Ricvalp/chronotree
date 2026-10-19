"""3D diagnostics preserve geometry, causal masks, and the training RNG."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from stmem.data.canonical import Decision, Frame
from stmem.evaluation.robomme_adapter import physical_action
from stmem.visualization.kinematics import JOINT_HIGH, panda_tcp_pose, panda_tcp_positions
from stmem.visualization.scenes import CURRENT_RGB, PREDICTED_RGB, RECORDED_RGB, build_action_scene


def decision_fixture():
    q = np.asarray([0.0, -0.2, 0.0, -1.9, 0.0, 1.6, 0.7])
    pose = panda_tcp_pose(q, task="VideoUnmask")
    quat = Rotation.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
    proprio = torch.from_numpy(np.r_[q, pose[:3, 3], quat, 0.04]).float()
    cloud = torch.tensor(
        [
            [0.3, 0.1, 0.1, 0.2, 0.4, 0.6],
            [0.2, 0.2, 0.2, 1.0, 0.5, 0.0],
            [0.1, 0.3, 0.3, 0.0, 0.0, 1.0],
            [999, 999, 999, 1, 1, 1],
            [float("nan"), 0, 0, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    current = Frame(
        cloud,
        torch.tensor([True, True, True, False, True]),
        proprio,
        torch.zeros(8),
        False,
        3,
        "execution",
        "fixed-current-3",
    )
    target = torch.from_numpy(np.tile(np.r_[q, 0.04], (4, 1))).float()
    target[:, 0] += torch.tensor([0.03, 0.06, 0.09, 0.12])
    # Objects deliberately lack Frame attributes: historical clouds must not be read.
    return Decision(
        current,
        [object()],
        [object()],
        target,
        torch.ones((4, 8), dtype=torch.bool),
        "fixture",
        "VideoUnmask",
        "episode_0",
        3,
    )


def box_center(box):
    return np.asarray(box["corners"]).mean(axis=0)


def test_current_cloud_sampling_is_deterministic_bounded_and_rng_neutral():
    decision = decision_fixture()
    prediction = decision.target.clone()
    prediction[:, 0] += 0.05
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    scene = build_action_scene(decision, prediction, max_points=2, path_samples=24)
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    torch.testing.assert_close(torch.rand(3), expected_rng[2], rtol=0, atol=0)
    assert scene.payload["type"] == "lidar/beta"
    assert scene.metadata["observed_valid_points"] == 3
    assert scene.metadata["displayed_cloud_points"] == 2
    assert scene.metadata["cloud_source"] == "current_observation_only"
    assert scene.metadata["recorded_targets"] == scene.metadata["predicted_targets"] == 4
    assert scene.metadata["coordinate_frame"] == "world"
    assert np.isfinite(scene.payload["points"]).all()
    assert len(scene.payload["points"]) <= 2 + 24 * 2 + 1
    np.testing.assert_allclose(scene.payload["points"][:2, :3], decision.current.points[[2, 0], :3])
    np.testing.assert_array_equal(scene.payload["points"][:2, 3:], [[0, 0, 255], [51, 102, 153]])
    assert (scene.payload["points"][:, :3] < 999).all()
    perm = torch.tensor([2, 0, 4, 3, 1])
    reordered = replace(
        decision,
        current=replace(
            decision.current, points=decision.current.points[perm], valid=decision.current.valid[perm]
        ),
    )
    repeated = build_action_scene(reordered, prediction, max_points=2, path_samples=24)
    np.testing.assert_array_equal(scene.payload["points"], repeated.payload["points"])
    assert json.dumps(scene.metadata, allow_nan=False)
    assert "not simulated motion" in scene.caption
    assert "gripper opening is not shown" in scene.caption


def test_target_paths_stop_at_first_arm_gap_and_preserve_table_commands():
    decision = decision_fixture()
    decision.target_valid[0, 7] = False  # Missing gripper does not mask observed arm commands.
    decision.target_valid[2, 3] = False
    prediction = decision.target.clone()
    prediction[:, 7] = -0.2
    scene = build_action_scene(decision, prediction)
    assert scene.metadata["valid_mask_prefix"] == 2
    assert scene.metadata["recorded_targets"] == 2
    assert scene.metadata["predicted_targets"] == 2
    boxes = scene.payload["boxes"].tolist()
    assert [box["label"] for box in boxes] == [
        "Observed current TCP",
        "Recorded target +2",
        "Predicted target +2",
    ]
    np.testing.assert_allclose(box_center(boxes[0]), decision.current.proprio[7:10], atol=1e-12)
    np.testing.assert_allclose(
        box_center(boxes[1]),
        panda_tcp_positions(decision.target[1, :7].numpy(), task=decision.task),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        box_center(boxes[2]), panda_tcp_positions(prediction[1, :7].numpy(), task=decision.task), atol=1e-12
    )
    assert np.isfinite(scene.payload["points"]).all()
    assert scene.metadata["predicted_actions_raw"] == prediction.tolist()
    assert scene.metadata["recorded_actions_raw"] == decision.target.tolist()
    assert scene.metadata["target_valid"] == decision.target_valid.tolist()
    commanded = np.asarray(scene.metadata["predicted_actions_commanded"])
    np.testing.assert_array_equal(commanded, [physical_action(row, decision.task) for row in prediction])
    assert (commanded[:, 7] == -1).all()


def test_joint_limit_violation_uses_deployment_clipping_and_records_raw_values():
    decision = decision_fixture()
    prediction = decision.target.clone()
    prediction[0, 0] = float(JOINT_HIGH[0] + 0.5)
    decision.target_chunk[1, 1] = float(JOINT_HIGH[1] + 0.5)
    scene = build_action_scene(decision, prediction)
    assert scene.metadata["recorded_targets"] == 1
    assert scene.metadata["recorded_invalid_joint_index"] == 1
    assert scene.metadata["predicted_targets"] == 1
    assert scene.metadata["predicted_joint_clipped_components"] == 1
    assert scene.metadata["predicted_joint_clipped_steps"] == 1
    assert prediction[0, 0] > JOINT_HIGH[0]
    assert scene.metadata["predicted_actions_raw"] == prediction.tolist()
    commanded = physical_action(prediction[0], decision.task)
    expected = panda_tcp_positions(commanded[:7], task=decision.task)
    predicted_box = next(box for box in scene.payload["boxes"] if box["label"].startswith("Predicted"))
    np.testing.assert_allclose(box_center(predicted_box), expected, atol=1e-12)
    assert np.any(np.all(scene.payload["points"][:, 3:] == PREDICTED_RGB, axis=1))
    assert "deployment clipping (1 components)" in scene.caption


@pytest.mark.parametrize("component", [0, 7])
def test_nonfinite_prediction_is_explicitly_rejected(component):
    decision = decision_fixture()
    prediction = decision.target.clone()
    prediction[1, component] = float("nan")
    with pytest.raises(ValueError, match="nonfinite predicted"):
        build_action_scene(decision, prediction)


def test_float32_controller_limit_targets_are_not_truncated():
    from stmem.evaluation.robomme_adapter import JOINT_LOW, JOINT_HIGH

    decision = decision_fixture()
    decision.target_chunk[:, :7] = torch.from_numpy(np.stack([JOINT_LOW, JOINT_HIGH] * 2))
    original = decision.target_chunk.clone()
    scene = build_action_scene(decision, original)
    assert scene.metadata["recorded_targets"] == scene.metadata["predicted_targets"] == 4
    assert scene.metadata["recorded_invalid_joint_index"] is None
    assert scene.metadata["predicted_joint_clipped_components"] == 0
    torch.testing.assert_close(decision.target_chunk, original, rtol=0, atol=0)


def test_stick_task_command_table_uses_zero_gripper():
    decision = decision_fixture()
    decision.task = "PatternLock"
    pose = panda_tcp_pose(decision.current.proprio[:7], task=decision.task)
    decision.current.proprio[7:10] = torch.as_tensor(pose[:3, 3])
    scene = build_action_scene(decision, decision.target)
    assert np.asarray(scene.metadata["predicted_actions_commanded"])[:, 7].tolist() == [0] * 4


def test_masked_empty_targets_have_only_current_geometry():
    decision = decision_fixture()
    decision.target_valid[:] = False
    scene = build_action_scene(decision, decision.target)
    assert scene.metadata["recorded_targets"] == scene.metadata["predicted_targets"] == 0
    assert len(scene.payload["boxes"]) == 1
    assert len(scene.payload["points"]) == 4  # Three observed points plus current TCP.
    np.testing.assert_array_equal(scene.payload["points"][-1, 3:], CURRENT_RGB)
    assert not np.any(np.all(scene.payload["points"][:, 3:] == RECORDED_RGB, axis=1))


def test_inconsistent_observation_and_bad_shapes_are_rejected():
    decision = decision_fixture()
    with pytest.raises(ValueError, match=r"same \[H,8\] shape"):
        build_action_scene(decision, decision.target[:2])
    with pytest.raises(ValueError, match="max_points"):
        build_action_scene(decision, decision.target, max_points=0)
    with pytest.raises(ValueError, match="path_samples"):
        build_action_scene(decision, decision.target, path_samples=1)
    bad = decision.current.proprio.clone()
    bad[7] += 0.01
    with pytest.raises(ValueError, match="inconsistent"):
        build_action_scene(replace(decision, current=replace(decision.current, proprio=bad)), decision.target)
    bad = decision.current.proprio.clone()
    bad[10:14] = torch.tensor([1.0, 0, 0, 0])
    with pytest.raises(ValueError, match="inconsistent"):
        build_action_scene(replace(decision, current=replace(decision.current, proprio=bad)), decision.target)


def test_real_wandb_object3d_serializes_scene_without_a_run():
    import wandb

    decision = decision_fixture()
    scene = build_action_scene(decision, decision.target, max_points=3, path_samples=16)
    obj = wandb.Object3D(scene.payload, caption=scene.caption)
    serialized = json.loads(Path(obj._path).read_text())
    assert serialized["type"] == "lidar/beta"
    assert serialized["vectors"] == []
    assert len(serialized["points"]) == len(scene.payload["points"])
    assert len(serialized["boxes"]) == 3
    assert serialized["boxes"][1]["label"] == "Recorded target +4"
    assert wandb.run is None
