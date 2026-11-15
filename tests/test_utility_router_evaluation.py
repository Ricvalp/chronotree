"""Synthetic CPU checks for the immutable utility-router rollout protocol."""

from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from stmem import evaluate
from stmem.evaluation.execution import evaluation_execution_config
from stmem.evaluation.runner import OnlinePolicy
from stmem.evaluation.utility_router import evaluation_settings
from stmem.memory.utility_routing import build_scorer
from stmem.models.policy import MemoryPolicy
from stmem.training import utility_routers
from stmem.training.checkpoints import file_sha256
from test_evaluation_execution import config
from test_tracking_evaluation import FakeTracker


def settings():
    return evaluation_settings({
        "schema": "utility_router_checkpoint_v1",
        "utility_protocol": {"budget": 128, "execute_targets": 8, "flow_steps": 8,
                             "candidate_limit": 8, "packet_max_cost": 8},
    })


def utility_config(stride=2):
    checkpoint_cfg = config(stride)
    checkpoint_cfg["model"].update(flow_steps=8, budget=128)
    cfg = deepcopy(checkpoint_cfg)
    cfg["evaluation"]["utility_routing"] = settings()
    return checkpoint_cfg, cfg


@pytest.mark.parametrize("stride", [1, 2])
def test_utility_execution_defaults_to_recorded_targets_and_preserves_sampling(stride):
    checkpoint_cfg, cfg = utility_config(stride)
    original = deepcopy(cfg)
    effective, record = evaluation_execution_config(checkpoint_cfg, cfg)
    explicit, explicit_record = evaluation_execution_config(checkpoint_cfg, cfg, execute_actions=8)
    assert cfg == original
    assert checkpoint_cfg["model"]["execute"] == (4 if stride == 1 else 2)
    assert effective == explicit and record == explicit_record
    assert effective["model"]["execute"] == record["execute_actions"] == 8
    assert effective["model"]["flow_steps"] == 8
    assert record["replan_primitive_steps"] == 8 * stride
    assert record["command_interpolation"] == "zero_order_hold"


@pytest.mark.parametrize("execute", [4, 10, 16])
def test_utility_pilot_rejects_execution_ablation_outside_its_frozen_protocol(execute):
    checkpoint_cfg, cfg = utility_config()
    with pytest.raises(ValueError, match="executed targets"):
        evaluation_execution_config(checkpoint_cfg, cfg, execute_actions=execute)


@pytest.mark.parametrize("which", ["checkpoint", "evaluation"])
def test_both_checkpoint_and_evaluation_flow_integrators_must_match_protocol(which):
    checkpoint_cfg, cfg = utility_config()
    (checkpoint_cfg if which == "checkpoint" else cfg)["model"]["flow_steps"] = 4
    with pytest.raises(ValueError, match="flow integration"):
        evaluation_execution_config(checkpoint_cfg, cfg)


def test_utility_protocol_does_not_authorize_test_or_changed_action_horizon():
    checkpoint_cfg, cfg = utility_config()
    with pytest.raises(ValueError, match="validation-only"):
        evaluation_execution_config(checkpoint_cfg, cfg, split="test")
    cfg["model"]["horizon"] = 32
    with pytest.raises(ValueError, match="horizon"):
        evaluation_execution_config(checkpoint_cfg, cfg)


@pytest.mark.parametrize("field,value", [("execute", 4), ("flow_steps", 4)])
def test_direct_online_policy_cannot_bypass_frozen_execution_settings(field, value):
    _, cfg = utility_config()
    cfg["model"].update(width=16, heads=4, layers=1, text_layers=1, points=8, execute=8)
    cfg["training"]["precision"] = "fp32"
    policy = MemoryPolicy(cfg)
    router = build_scorer(16, 8).eval().requires_grad_(False)
    valid = OnlinePolicy(policy, cfg, "TREE", 128, router=router)
    assert valid.utility_router is not None and valid.scorer is None
    changed = deepcopy(cfg)
    changed["model"][field] = value
    with pytest.raises(ValueError, match="frozen execution and integration"):
        OnlinePolicy(policy, changed, "TREE", 128, router=router)


def test_single_evaluation_records_inherited_protocol_before_manifest_and_reuses_it(tmp_path, monkeypatch):
    """Actual compact checkpoint loading; synthetic episode, no simulator or GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    checkpoint_cfg, cfg = utility_config()
    checkpoint_cfg["model"].update(width=16, heads=4, layers=1, text_layers=1, points=8)
    checkpoint_cfg["data"]["tasks"] = ["VideoUnmask"]
    checkpoint_cfg["evaluation"].update(episode_count=1, video_cap=0)
    checkpoint_cfg["training"]["precision"] = "fp32"
    # Test preflight binding from the router header when the input config is old.
    cfg = deepcopy(checkpoint_cfg)
    policy = MemoryPolicy(cfg)
    checkpoint = tmp_path / "compact.pt"
    torch.save({"config": cfg, "stage": "compact", "model": policy.state_dict()}, checkpoint)
    checkpoint_hash = file_sha256(checkpoint)
    router_path = tmp_path / "router.pt"
    header = {"schema": "utility_router_checkpoint_v1", "reader": "tree",
              "utility_protocol": settings()["protocol"], "provenance": {"compact": checkpoint_hash}}
    torch.save(header, router_path)
    router = build_scorer(16, 8).eval().requires_grad_(False)
    loaded = []

    def load_router(path, **kwargs):
        assert path == str(router_path)
        assert kwargs["expected_student_sha256"] == checkpoint_hash
        assert kwargs["expected_reader"] == "tree"
        loaded.append(path)
        return router, header

    monkeypatch.setattr(utility_routers, "load_utility_router", load_router)
    monkeypatch.setattr(evaluate, "RunTracker", FakeTracker)
    monkeypatch.setattr(evaluate, "environment_manifest", lambda: {"scope": "synthetic CPU diagnostic"})
    monkeypatch.setattr(evaluate, "source_revision", lambda: {"source_sha256": "synthetic"})
    monkeypatch.setattr(evaluate, "RoboMMEAdapter", lambda *a, **kw: SimpleNamespace(close=lambda: None))
    observed = []

    def episode(adapter, online, **kwargs):
        assert isinstance(online, OnlinePolicy) and online.utility_router is not None
        assert online.cfg["model"]["execute"] == 8
        assert online.cfg["model"]["flow_steps"] == 8
        observed.append(kwargs["episode_idx"])
        return {"task": "VideoUnmask", "episode_idx": 0, "status": "success", "actions": []}

    monkeypatch.setattr(evaluate, "run_episode", episode)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", ["stmem.evaluate", "--config", str(config_path),
                                     "--checkpoint", str(checkpoint), "--router-checkpoint", str(router_path),
                                     "--method", "TREE", "--output", str(output)])
    evaluate.main()
    metadata = json.loads((output / "manifest.json").read_text())
    assert metadata["config"]["evaluation"]["utility_routing"] == settings()
    assert metadata["config"]["model"]["execute"] == 8
    assert metadata["execution_override"]["checkpoint_execute_actions"] == 2
    assert metadata["execution_override"]["replan_primitive_steps"] == 16
    evaluate.main()
    assert observed == [0] and len(loaded) == 2
    assert file_sha256(checkpoint) == checkpoint_hash
    assert yaml.safe_load(config_path.read_text())["model"]["execute"] == 2
