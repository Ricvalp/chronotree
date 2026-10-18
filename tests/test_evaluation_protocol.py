"""Run the real chunk runner against a deterministic fake benchmark protocol."""

import numpy as np
import pytest
import torch
from stmem.evaluation.robomme_adapter import RoboMMEAdapter
from stmem.evaluation.runner import run_episode


def batch(n):
    values = {
        "front_rgb": np.zeros((2, 2, 3), np.uint8),
        "wrist_rgb": np.zeros((2, 2, 3), np.uint8),
        "front_depth": np.full((2, 2, 1), 1000, np.int16),
        "wrist_depth": np.full((2, 2, 1), 1000, np.int16),
        "front_camera_extrinsic": np.c_[np.eye(3), np.zeros(3)],
        "wrist_camera_extrinsic": np.c_[np.eye(3), np.zeros(3)],
        "joint_state": np.zeros(7),
        "eef_state": np.zeros(6),
        "gripper_state": np.zeros(2),
    }
    return {k + "_list": [v.copy() for _ in range(n)] for k, v in values.items()}


class FakeBenchmark:
    max_steps_without_demonstration = 10
    control_freq = 20
    sim_freq = 100

    @property
    def unwrapped(self):
        return self

    def reset(self):
        self.commands = []
        return batch(3), {
            "task_goal": ["Use the supplied video."],
            "front_camera_intrinsic": np.eye(3),
            "wrist_camera_intrinsic": np.eye(3),
        }

    def step(self, command):
        self.commands.append(command)
        if len(self.commands) == 2:
            return {}, 0.0, True, False, {"status": "success"}
        return batch(1), 0.0, False, False, {"status": "ongoing"}

    def close(self):
        pass


class FakeOnline:
    method = "CURRENT"
    intervention = "none"
    budget = 16
    policy = type("Policy", (), {"device": "cpu"})()
    cfg = {"seed": 0, "model": {"horizon": 4, "execute": 2}, "evaluation": {"max_steps": 10}}

    def __init__(self, mode="valid"):
        self.mode = mode
        self.decision_calls = 0
        self.observed = []

    def reset(self, reset, seed):
        assert len(reset.conditioning) == 2
        assert all(not f.action_available for f in reset.conditioning)
        assert not reset.current.action_available
        self.reset_encoding_seconds = 0.0

    def decide(self):
        self.decision_calls += 1
        chunk = torch.tensor([[0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.7, 1.0]]).repeat(4, 1)
        if self.mode == "shape":
            chunk = chunk[:, :7]
        if self.mode == "nan":
            chunk[0, 0] = float("nan")
        return chunk, {"complete_online_seconds": 0.01}

    def observe(self, frame, preprocessing_seconds=0.0):
        assert preprocessing_seconds > 0
        self.observed.append(frame)


def test_real_runner_conditioning_chunking_and_empty_terminal():
    environment = FakeBenchmark()
    adapter = RoboMMEAdapter("VideoUnmask", env=environment, points=8, crop=None)
    online = FakeOnline()
    result = run_episode(adapter, online, task="VideoUnmask", episode_idx=0, split="val", seed=0)
    assert result["status"] == "success" and result["primitive_commands"] == 2
    assert len(environment.commands) == 2 and online.decision_calls == 1 and len(online.observed) == 1
    assert online.observed[0].action_available
    assert torch.equal(online.observed[0].previous_action, torch.as_tensor(environment.commands[0]))
    assert result["error"] is None


@pytest.mark.parametrize("mode", ["shape", "nan"])
def test_invalid_policy_chunk_is_failed_planned_episode(mode):
    environment = FakeBenchmark()
    adapter = RoboMMEAdapter("VideoUnmask", env=environment, points=8, crop=None)
    result = run_episode(adapter, FakeOnline(mode), task="VideoUnmask", episode_idx=3, split="val", seed=0)
    assert result["task"] == "VideoUnmask" and result["episode_idx"] == 3
    assert result["status"] == "policy_error" and result["primitive_commands"] == 0
    assert "nonfinite or incorrectly shaped" in result["error"]
    assert len(environment.commands) == 0
