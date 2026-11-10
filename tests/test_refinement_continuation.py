"""Synthetic protocol tests; no simulator, task oracle or GPU required."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stmem.inspection import refinement_continuation as module


def test_case_selection_is_gain_blind_and_preserves_original_partition():
    states = [{"task": "VideoUnmask", "episode_id": "episode_0", "decision_index": step,
               "state_id": str(step), "cache_eligible": eligible, "partition": "validation"}
              for step, eligible in ((16, True), (32, False), (36, True), (52, True))]
    selected = module.predeclared_state({"states": list(reversed(states))})
    assert selected["decision_index"] == 36 and selected["partition"] == "validation"
    with pytest.raises(ValueError, match="No eligible"):
        module.predeclared_state({"states": states[:2]})


def test_first_chunk_executes_ten_targets_with_two_native_holds():
    actions = np.arange(128).reshape(16, 8)
    commands = module.held_commands(actions)
    assert commands.shape == (20, 8)
    np.testing.assert_array_equal(commands[::2], actions[:10])
    np.testing.assert_array_equal(commands[1::2], actions[:10])
    with pytest.raises(ValueError, match="finite"):
        module.held_commands(np.full((16, 8), np.nan))


def test_replay_mismatch_is_rejected():
    assert module.verify_joint_state(np.zeros(8), np.zeros(8), "fixture") == 0.
    with pytest.raises(ValueError, match="mismatch"):
        module.verify_joint_state(np.ones(8), np.zeros(8), "fixture")


def test_first_selected_chunk_is_used_once_and_later_history_is_updated(monkeypatch, tmp_path):
    prefix = 36
    cap = prefix + 40
    first = np.arange(128, dtype=np.float32).reshape(16, 8)
    followup = first + 1000
    active = {}

    def frame(step):
        return SimpleNamespace(proprio=torch.zeros(18), source_index=step)

    class Adapter:
        def __init__(self, *args, **kwargs):
            self.step_index = 0
            self.last_preprocess_seconds = 0.
            self.last_raw_observation = {}
            self.commands = []
            active["adapter"] = self
        def reset(self):
            self.last_rgb = [np.zeros((16, 16, 3), dtype=np.uint8)]
            self.frame = frame(0)
            return SimpleNamespace(goal="goal", current=self.frame, conditioning=[])
        def step(self, action):
            self.step_index += 1
            self.commands.append(action.numpy().copy())
            self.frame = frame(self.step_index)
            self.last_rgb = []
            return SimpleNamespace(frames=[self.frame] if self.step_index % 2 == 0 else [],
                done=False, status="ongoing", observation_skipped=self.step_index % 2 == 1,
                command=action.numpy())
        def close(self):
            pass

    class Online:
        def __init__(self, policy, cfg, reader, budget):
            assert reader == "UNIFORM" and budget == 128
            self.memory = SimpleNamespace(archive=SimpleNamespace(leaves=[]))
            self.update_seconds = 0.
            self.calls = []
            active["online"] = self
        def reset(self, reset, seed):
            self.memory.current = reset.current
            self.memory.execution_observations = 1
        def observe(self, observation, seconds):
            self.memory.current = observation
            self.memory.execution_observations += 1
        def decide(self):
            self.calls.append(self.memory.current.source_index)
            assert self.memory.execution_observations == (prefix + 20) // 2 + 1
            return torch.from_numpy(followup), {}

    monkeypatch.setattr(module, "RoboMMEAdapter", Adapter)
    monkeypatch.setattr(module, "OnlinePolicy", Online)
    monkeypatch.setattr(module, "raw_frame", lambda adapter, old, step: (adapter.frame, True))
    monkeypatch.setattr(module, "observed_choice", lambda adapter, observed: {
        "target_height_m": .05, "grasped_container_indices": [], "tcp_nearest_container": 0})
    monkeypatch.setattr(module.imageio, "get_writer", lambda *args, **kwargs:
                        SimpleNamespace(append_data=lambda *_: None, close=lambda: None))
    cfg = {"model": {"points": 32}, "evaluation": {"max_steps": 100},
           "geometry": {"crop": [0, 1], "voxel_size": .01, "max_depth": 4}}
    monkeypatch.setattr(module, "point_sampler_settings", lambda _: {})
    monkeypatch.setattr(module, "center_sampler_settings", lambda _: {})
    monkeypatch.setattr(module, "center_counts", lambda _: (32,))
    source = {"goal": "goal", "reset_proprio": np.zeros(18), "conditioning_proprio": [],
              "commands": np.zeros((cap, 8)), "step_proprio": np.zeros((cap, 18))}
    record = {"identity": "synthetic", "state": {"decision_index": prefix}, "wall_seconds_per_branch": 5,
              "continuation_noise_seed": 123, "first_chunk_noise_seed": 456,
              "cap_total_native_commands": cap, "contexts": {"packet_completed": "fixed_first_context"}}
    result = module.run_branch(None, cfg, None, source, record, "packet_completed", first, tmp_path)
    assert result["status"] == "stopped_at_declared_diagnostic_cap", result.get("error")
    assert active["online"].calls == [prefix + 20]
    commands = np.asarray(active["adapter"].commands)
    np.testing.assert_array_equal(commands[prefix:prefix+20], module.held_commands(first))
    np.testing.assert_array_equal(commands[prefix+20:], module.held_commands(followup))
    assert len(result["prefix_replay_checks"]) == prefix
    assert result["timeline"][-1]["execution_observations"] == cap // 2 + 1
