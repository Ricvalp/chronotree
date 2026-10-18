import io
import tarfile
import numpy as np
import pytest
import torch
from stmem.data.dataset import synthetic_episode
from stmem.data.download import safe_extract
from stmem.data.pointcloud import backproject, sample_points, proprioception
from stmem.data.prepare import numeric_keys
from stmem.evaluation.robomme_adapter import RoboMMEAdapter, physical_action


def observation():
    return {
        "front_rgb": np.full((3, 3, 3), 128, dtype=np.uint8),
        "wrist_rgb": np.full((3, 3, 3), 64, dtype=np.uint8),
        "front_depth": np.full((3, 3, 1), 1000, dtype=np.int16),
        "wrist_depth": np.full((3, 3, 1), 1000, dtype=np.int16),
        "front_camera_extrinsic": np.c_[np.eye(3), np.zeros(3)],
        "wrist_camera_extrinsic": np.c_[np.eye(3), np.zeros(3)],
        "joint_state": np.zeros(7),
        "eef_state": np.zeros(6),
        "gripper_state": np.ones(2) * 0.04,
    }


def batch(n):
    obs = observation()
    return {k + "_list": [v.copy() for _ in range(n)] for k, v in obs.items()}


class FakeEnvironment:
    control_freq = 20
    sim_freq = 100
    max_steps_without_demonstration = 12

    @property
    def unwrapped(self):
        return self

    def reset(self):
        return batch(3), {
            "task_goal": ["test goal"],
            "front_camera_intrinsic": np.eye(3),
            "wrist_camera_intrinsic": np.eye(3),
            "object_hidden_pose": "must not leak",
        }

    def step(self, action):
        self.command = action
        return {}, 0, True, False, {"status": "error", "error_message": "bounded fake terminal"}

    def close(self):
        pass


def test_post_action_targets_past_history_terminal_and_gap_masks():
    ep = synthetic_episode(length=6, conditioning_length=2)
    ep.actions = torch.arange(48, dtype=torch.float32).reshape(6, 8)
    ep.action_valid[-1] = False
    decision = ep.decision(2, 4)
    assert torch.equal(decision.target_chunk[:3], ep.actions[3:])
    assert decision.target_valid.sum() == 16
    assert len(decision.execution_prefix) == 2
    assert not any(f.action_available for f in decision.conditioning)
    assert all(not f.previous_action.any() for f in decision.conditioning)
    ep.action_valid[4, :7] = False
    assert ep.decision(2, 4).target_valid.sum() == 8  # unavailable intermediate command breaks remainder
    ep.alignment = "unresolved"
    with pytest.raises(RuntimeError, match="alignment unresolved"):
        ep.decision(0, 1)


def test_geometry_world_to_camera_and_invalid_signed_depth():
    rgb = np.full((2, 2, 3), 255, np.uint8)
    depth = np.array([[1000, -1], [0, 2000]], np.int16)
    # camera x = world x+1; inverse must subtract camera translation.
    e = np.c_[np.eye(3), np.array([1, 0, 0])]
    cloud = backproject(rgb, depth, np.eye(3), e, crop=None)
    assert torch.allclose(cloud[:, :3], torch.tensor([[-1.0, 0, 1], [1.0, 2, 2]]))
    assert torch.equal(cloud[:, 3:], torch.ones(2, 3))
    empty = backproject(rgb, np.zeros((2, 2), np.int16), np.eye(3), e, crop=None)
    sampled, valid = sample_points(empty, 8)
    assert not valid.any() and not sampled.any()
    p = torch.rand(100, 6)
    a, ma = sample_points(p, 32)
    b, mb = sample_points(p.flip(0), 32)
    assert torch.equal(a, b) and torch.equal(ma, mb)
    assert proprioception(observation()).shape == (15,)


def test_protocol_retains_conditioning_masks_reset_actions_and_empty_terminal():
    env = FakeEnvironment()
    adapter = RoboMMEAdapter("VideoUnmask", env=env, points=8, crop=None)
    reset = adapter.reset()
    assert len(reset.conditioning) == 2 and reset.current.source == "execution"
    assert reset.current.time == 0 and not reset.current.action_available
    assert all(not f.action_available for f in reset.conditioning)
    assert not hasattr(reset.current, "object_hidden_pose")
    assert adapter.ingest(batch(3), {}, event_id=reset.current.event_id.rsplit("/", 1)[0], reset=True) == []
    action = np.array([0, 0, 0, -1.5, 0, 1.5, 0.7, 0.2], np.float32)
    terminal = adapter.step(action)
    assert terminal.done and terminal.status == "error" and terminal.frames == []
    assert env.command[-1] == 1
    with pytest.raises(ValueError):
        physical_action(np.full(8, np.nan), "VideoUnmask")
    with pytest.raises(ValueError):
        physical_action(np.zeros(7), "VideoUnmask")
    assert physical_action(action, "PatternLock")[-1] == 0


def test_numeric_sort_and_archive_traversal(tmp_path):
    assert numeric_keys({"episode_10": 0, "episode_2": 0, "episode_0": 0}, "episode_") == [
        "episode_0",
        "episode_2",
        "episode_10",
    ]
    for name, kind in [("../outside", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]:
        path = tmp_path / "unsafe.tar.xz"
        with tarfile.open(path, "w:xz") as t:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "/tmp/outside"
            member.size = 1 if kind == tarfile.REGTYPE else 0
            t.addfile(member, io.BytesIO(b"x") if member.size else None)
        with pytest.raises(ValueError, match="Unsafe"):
            safe_extract(path, tmp_path / "out")
    assert not (tmp_path / "outside").exists()
