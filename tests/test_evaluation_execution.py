"""Synthetic CPU checks for an explicit rollout execution-length ablation."""

from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from stmem import evaluate
from stmem.config import load_config
from stmem.evaluation.execution import evaluation_execution_config
from stmem.evaluation.robomme_adapter import RoboMMEAdapter
from stmem.evaluation.runner import run_episode
from stmem.models.policy import MemoryPolicy
from stmem.training.checkpoints import file_sha256
from stmem.training.runtime import validate_policy_config
from test_temporal_rollout import CadenceBenchmark, CadenceOnline
from test_tracking_evaluation import FakeTracker


def config(stride=1):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"].update(horizon=16, execute=4 if stride == 1 else 2, action_stride=stride)
    cfg["data"].update(observation_stride=stride, process_every_observation=stride == 1)
    return cfg


@pytest.mark.parametrize("stride", [1, 2])
def test_override_changes_only_execution_length_and_preserves_checkpoint(stride):
    checkpoint_cfg = config(stride)
    cfg = deepcopy(checkpoint_cfg)
    before = deepcopy(cfg)
    effective, record = evaluation_execution_config(checkpoint_cfg, cfg, execute_actions=8)
    assert checkpoint_cfg == cfg == before
    assert effective["model"]["execute"] == record["execute_actions"] == 8
    assert record["checkpoint_execute_actions"] == (4 if stride == 1 else 2)
    assert record["replan_primitive_steps"] == 8 * stride
    assert record["checkpoint_replan_primitive_steps"] == 4
    effective["model"]["execute"] = cfg["model"]["execute"]
    assert effective == cfg


def test_default_and_global_compatibility_remain_strict():
    cfg = config()
    changed = deepcopy(cfg)
    changed["model"]["execute"] = 8
    for validator in (validate_policy_config, evaluation_execution_config):
        with pytest.raises(ValueError, match="model.execute"):
            validator(cfg, changed)


@pytest.mark.parametrize("value", [0, -1, 17, True, 2.5])
def test_override_must_be_a_positive_target_count_within_predicted_horizon(value):
    cfg = config()
    with pytest.raises(ValueError, match="between 1 and checkpoint horizon 16"):
        evaluation_execution_config(cfg, cfg, execute_actions=value)


@pytest.mark.parametrize("field,value", [("horizon", 32), ("action_stride", 2), ("points", 64),
                                         ("condition_on_previous_commands", False)])
def test_execution_override_does_not_authorize_other_model_changes(field, value):
    cfg = config()
    changed = deepcopy(cfg)
    changed["model"][field] = value
    with pytest.raises(ValueError, match=field):
        evaluation_execution_config(cfg, changed, execute_actions=8)


def test_execution_override_must_replan_on_a_retained_observation():
    cfg = config()
    cfg["data"].update(observation_stride=2, process_every_observation=False)
    with pytest.raises(ValueError, match="retained observation"):
        evaluation_execution_config(cfg, cfg, execute_actions=3)


def test_test_split_remains_frozen_and_rejects_override_before_checkpoint_access(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["stmem.evaluate", "--config", "missing.yaml", "--split", "test",
                                     "--execute-actions", "8"])
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2
    cfg = config()
    with pytest.raises(ValueError, match="validation-only"):
        evaluation_execution_config(cfg, cfg, execute_actions=8, split="test")


@pytest.mark.parametrize("stride", [1, 2])
def test_actual_executor_replans_after_eight_distinct_targets_at_recorded_cadence(stride):
    cfg, _ = evaluation_execution_config(config(stride), config(stride), execute_actions=8)
    env = CadenceBenchmark(terminal_at=8 * stride + 1)
    env.max_steps_without_demonstration = 40
    adapter = RoboMMEAdapter("VideoUnmask", env=env, points=8, crop=None, observation_stride=stride)

    class DistinctTargets(CadenceOnline):
        def decide(self):
            self.decisions.append(self.current.source_index)
            chunk = torch.tensor([[0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.7, 1.0]]).repeat(16, 1)
            chunk[:, 0] = torch.arange(16) / 100 + (len(self.decisions) - 1)
            chunk[1::2, 7] = -1
            return chunk, {"complete_online_seconds": 0.01}

    online = DistinctTargets()
    online.cfg = cfg
    result = run_episode(adapter, online, task="VideoUnmask", episode_idx=0, split="val", seed=0)
    assert result["status"] == "success", result.get("error")
    assert online.decisions == [0, 8 * stride]
    commands = np.array(env.commands)
    assert len(commands) == 8 * stride + 1
    np.testing.assert_allclose(commands[:8 * stride, 0], np.repeat(np.arange(8) / 100, stride))
    assert commands[-1, 0] == 1  # new chunk, not target nine of the previous chunk
    for index in range(0, 8 * stride, stride):
        for offset in range(stride):
            np.testing.assert_array_equal(commands[index], commands[index + offset])
    assert result["temporal_sampling"]["executed_actions_per_chunk"] == 8
    assert result["temporal_sampling"]["replan_primitive_steps"] == 8 * stride
    assert result["temporal_sampling"]["action_hz"] == 20 / stride


def test_cli_records_override_with_real_checkpoint_loading_and_rejects_mixed_resume(tmp_path, monkeypatch):
    """Tiny synthetic checkpoint and fake episodes; no simulator/GPU/training."""
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = config()
    cfg["model"].update(width=16, heads=4, layers=1, text_layers=1, points=8)
    cfg["data"]["tasks"] = ["VideoUnmask"]
    cfg["evaluation"].update(episode_count=1, video_cap=0)
    cfg["training"]["precision"] = "fp32"
    checkpoint = tmp_path / "checkpoint.pt"
    policy = MemoryPolicy(cfg)
    torch.save({"config": cfg, "stage": "teacher", "model": policy.state_dict()}, checkpoint)
    original_hash = file_sha256(checkpoint)
    config_path = tmp_path / "evaluation.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "evaluation"
    FakeTracker.instances = []
    monkeypatch.setattr(evaluate, "RunTracker", FakeTracker)
    monkeypatch.setattr(evaluate, "environment_manifest", lambda: {"scope": "synthetic CPU diagnostic"})
    monkeypatch.setattr(evaluate, "source_revision", lambda: {"source_sha256": "fixture"})
    observed_configs = []

    def online(found_policy, effective, *args, **kwargs):
        for name, tensor in policy.state_dict().items():
            torch.testing.assert_close(found_policy.state_dict()[name], tensor, rtol=0, atol=0)
        assert found_policy.config["model"]["execute"] == 4
        assert effective["model"]["execute"] == 8
        observed_configs.append(deepcopy(effective))
        return SimpleNamespace(cfg=effective)

    monkeypatch.setattr(evaluate, "OnlinePolicy", online)
    monkeypatch.setattr(evaluate, "RoboMMEAdapter", lambda *a, **kw: SimpleNamespace(close=lambda: None))
    calls = []

    def episode(*args, **kwargs):
        calls.append(kwargs["episode_idx"])
        return {"task": "VideoUnmask", "episode_idx": 0, "status": "success", "actions": []}

    monkeypatch.setattr(evaluate, "run_episode", episode)
    argv = ["stmem.evaluate", "--config", str(config_path), "--checkpoint", str(checkpoint),
            "--output", str(output), "--execute-actions", "8"]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate.main()
    metadata = json.loads((output / "manifest.json").read_text())
    override = metadata["execution_override"]
    assert override["checkpoint_execute_actions"] == 4 and override["execute_actions"] == 8
    assert metadata["config"]["model"]["execute"] == 8
    assert FakeTracker.instances[0].options["metadata"]["evaluation_manifest"]["execution_override"] == override
    assert FakeTracker.instances[0].summaries["evaluation/execution_override"] == override
    saved = json.loads((output / "episodes/VideoUnmask_000.json").read_text())
    assert saved["execution_override"] == override
    assert file_sha256(checkpoint) == original_hash
    assert yaml.safe_load(config_path.read_text())["model"]["execute"] == 4
    evaluate.main()
    assert calls == [0]  # matching resume reuses the durable episode
    assert len(observed_configs) == 2
    monkeypatch.setattr(sys, "argv", argv[:-1] + ["4"])
    with pytest.raises(ValueError, match="different inputs"):
        evaluate.main()
    assert calls == [0]
    assert file_sha256(checkpoint) == original_hash
