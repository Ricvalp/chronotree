"""Synthetic protocol diagnostics for sampled observations and held commands."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from stmem.config import load_config
from stmem.data.dataset import synthetic_episode
from stmem.evaluation.robomme_adapter import ResetResult, RoboMMEAdapter
from stmem.evaluation.runner import OnlinePolicy, run_episode
from stmem.models.policy import MemoryPolicy
from test_evaluation_protocol import FakeBenchmark, batch


class CadenceBenchmark(FakeBenchmark):
    def __init__(self, terminal_at=6, missing_at=None):
        self.terminal_at, self.missing_at = terminal_at, missing_at

    def reset(self):
        _, info = super().reset()
        return batch(6), info  # five conditioning frames and execution frame zero

    def step(self, command):
        self.commands.append(command.copy())
        n = len(self.commands)
        if n == self.terminal_at:
            return {}, 0.0, True, False, {"status": "success"}
        return {} if n == self.missing_at else batch(1), 0.0, False, False, {"status": "ongoing"}


class CadenceOnline:
    method, intervention, budget = "CURRENT", "none", 16
    policy = type("Policy", (), {"device": "cpu"})()
    cfg = {
        "seed": 0,
        "model": {"horizon": 16, "execute": 2, "action_stride": 2},
        "data": {"observation_stride": 2, "nominal_control_hz": 20},
        "evaluation": {"max_steps": 10},
    }

    def __init__(self):
        self.observed, self.decisions = [], []

    def reset(self, reset, seed):
        self.conditioning = reset.conditioning
        self.current = reset.current
        self.reset_encoding_seconds = 0.0

    def observe(self, frame, preprocessing_seconds=0.0):
        self.observed.append(frame)
        self.current = frame

    def decide(self):
        self.decisions.append(self.current.source_index)
        chunk = torch.tensor([[0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.7, 1.0]]).repeat(16, 1)
        chunk[:, 0] = len(self.decisions) / 10
        chunk[1, 0], chunk[1, 7] = len(self.decisions) / 10 + 0.05, -1
        return chunk, {"complete_online_seconds": 0.01}


def rollout(terminal_at=6, missing_at=None):
    env = CadenceBenchmark(terminal_at, missing_at)
    adapter = RoboMMEAdapter("VideoUnmask", env=env, points=8, crop=None, observation_stride=2)
    online = CadenceOnline()
    result = run_episode(adapter, online, task="VideoUnmask", episode_idx=0, split="val", seed=0)
    return env, adapter, online, result


def test_coarse_commands_hold_native_rate_and_preserve_replan_interval():
    env, adapter, online, result = rollout()
    assert result["status"] == "success" and result["primitive_commands"] == 6
    assert online.decisions == [0, 4]
    for start in (0, 2, 4):
        np.testing.assert_array_equal(env.commands[start], env.commands[start + 1])
    assert not np.array_equal(env.commands[0], env.commands[2])
    assert env.commands[2][-1] == -1
    assert [f.source_index for f in online.conditioning] == [0, 2, 4]
    assert [f.time for f in online.observed] == [2, 4]
    for frame in online.observed:
        np.testing.assert_array_equal(frame.previous_action.numpy(), env.commands[frame.source_index - 1])
    assert result["environment"]["conditioning_frames_recorded"] == 5
    assert result["environment"]["conditioning_frames"] == 3
    assert result["temporal_sampling"]["native_control_hz"] == 20
    assert result["temporal_sampling"]["action_hz"] == 10
    assert result["temporal_sampling"]["observation_hz"] == 10
    assert result["temporal_sampling"]["replan_primitive_steps"] == 4


def test_terminal_on_first_hold_step_never_executes_second_copy():
    env, _, online, result = rollout(terminal_at=5)
    assert result["status"] == "success"
    assert result["primitive_commands"] == len(env.commands) == 5
    assert online.decisions == [0, 4]


def test_empty_ongoing_observation_cannot_be_treated_as_cadence_skip():
    _, _, _, result = rollout(missing_at=1)
    assert result["status"] == "error"
    assert result["primitive_commands"] == 1
    assert "returned no allowed observation" in result["error"]


def test_dropped_frames_keep_raw_video_and_clocks_without_feature_work(monkeypatch):
    class SyntheticExtractor:
        signature = "synthetic-test-only"

        def __init__(self):
            self.calls = 0

        def sample(self, observation, camera, uv, valid):
            self.calls += 1
            return torch.zeros(len(valid), 256)

    extractor = SyntheticExtractor()
    adapter = RoboMMEAdapter(
        "VideoUnmask",
        env=CadenceBenchmark(),
        points=8,
        crop=None,
        observation_stride=2,
        feature_extractor=extractor,
    )
    reset = adapter.reset()
    assert len(reset.conditioning) == 3 and extractor.calls == 4
    assert len(adapter.last_rgb) == 6
    action = torch.tensor([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.7, 1.0])
    skipped = adapter.step(action)
    assert skipped.observation_skipped and not skipped.frames
    assert extractor.calls == 4 and len(adapter.last_rgb) == 1
    assert adapter.last_raw_observation and adapter._clock["execution"] == 2
    kept = adapter.step(action)
    assert not kept.observation_skipped and kept.frames[0].source_index == 2
    assert extractor.calls == 5 and len(adapter.last_rgb) == 1
    # Event deduplication is distinct from intentional temporal subsampling.
    repeated = adapter.ingest(adapter.last_raw_observation, {}, event_id=f"{adapter._reset_id}/step/2")
    assert not repeated and not adapter.last_observation_skipped
    assert adapter._clock["execution"] == 3


def test_online_policy_filters_injected_full_rate_frames_and_matches_offline_evidence():
    torch.set_num_threads(1)
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["data"]["observation_stride"] = 2
    cfg["model"].update(width=16, heads=4, layers=1, text_layers=1, block_size=8, visual_tokens=16)
    policy = MemoryPolicy(cfg).eval()
    episode = synthetic_episode(length=18, conditioning_length=5, points=8)
    online = OnlinePolicy(policy, cfg, "FULL", cfg["model"]["budget"])
    online.reset(ResetResult(episode.conditioning, episode.execution[0], episode.goal), seed=0)
    for frame in episode.execution[1:17]:
        online.observe(frame)
    assert online.memory.current.source_index == 16
    assert online.memory.execution_observations == 9
    assert online.clocks["conditioning"] == 4
    decision = replace(
        episode.decision(16, cfg["model"]["horizon"]),
        conditioning=episode.conditioning[::2],
        execution_prefix=episode.execution[:16:2],
    )
    expected = policy.encode(decision).archive.leaves
    actual = online.memory.archive.leaves
    assert len(actual) == len(expected)
    for found, wanted in zip(actual, expected):
        assert (found.time_min, found.time_max) == (wanted.time_min, wanted.time_max)
        torch.testing.assert_close(found.content, wanted.content, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("stride", [0, -1, True, 1.5])
def test_adapter_rejects_invalid_sampling_stride(stride):
    with pytest.raises(ValueError, match="observation_stride"):
        RoboMMEAdapter("VideoUnmask", env=CadenceBenchmark(), observation_stride=stride)
