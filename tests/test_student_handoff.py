"""Synthetic CPU invariants for opt-in compact training-scene handoffs."""
from copy import deepcopy
import inspect
import json

import pytest
import torch

from stmem.config import load_config
from stmem.data.dataset import synthetic_episode
from stmem.evaluation.robomme_adapter import ResetResult
from stmem.evaluation.runner import OnlinePolicy
from stmem.inspection import handoff
from stmem.inspection.handoff_validation import validate_trial
from stmem.inspection.student_handoff import TrainingSceneCompactPolicy, diagnostic_config
from stmem.models.policy import MemoryPolicy


def fixture_cfg():
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"].update(width=16, heads=4, layers=1, text_layers=1, block_size=8, visual_tokens=128,
                        horizon=16, execute=2, action_stride=2, flow_steps=8, budget=128)
    cfg["data"].update(observation_stride=2, process_every_observation=False)
    cfg["stage"] = "compact"
    cfg["evaluation"]["split"] = "train"
    cfg["training"]["precision"] = "fp32"
    return cfg


def test_diagnostic_factory_preserves_memory_clocks_and_current_and_legacy_full_default():
    torch.set_num_threads(1)
    cfg = fixture_cfg()
    policy = MemoryPolicy(cfg).eval()
    episode = synthetic_episode(length=18, conditioning_length=5, points=32)
    reset = ResetResult(episode.conditioning, episode.execution[0], episode.goal)
    native = OnlinePolicy(policy, cfg, "UNIFORM", 128)
    diagnostic = TrainingSceneCompactPolicy(policy, cfg, "TREE", 128, selection_family="tree")
    for online in (native, diagnostic):
        online.reset(reset, 2701)
        for frame in episode.execution[1:17]:
            online.observe(frame)
    assert diagnostic.method == "TREE" and diagnostic.selection_family == "tree"
    assert diagnostic.scorer is None
    assert diagnostic.memory.current is native.memory.current
    assert diagnostic.memory.execution_observations == native.memory.execution_observations
    assert diagnostic.clocks == native.clocks and diagnostic.text.equal(native.text)
    assert diagnostic.generator.get_state().equal(native.generator.get_state())
    assert diagnostic.selection_rng.getstate() == native.selection_rng.getstate()
    a, b = diagnostic.memory.archive.leaves, native.memory.archive.leaves
    assert len(a) == len(b) > 128
    for found, wanted in zip(a, b):
        assert found.leaf_ids == wanted.leaf_ids
        torch.testing.assert_close(found.content_sum, wanted.content_sum, rtol=0, atol=0)
    signature = inspect.signature(handoff.run)
    assert signature.parameters["method"].default == "FULL"
    assert signature.parameters["online_factory"].default is OnlinePolicy
    assert signature.parameters["absolute_time_noise"].default is False
    # Production train selection remains forbidden; only the diagnostic wrapper is opt-in.
    with pytest.raises(ValueError, match="validation-only"):
        OnlinePolicy(policy, cfg, "TREE", 128, selection_family="tree")


@pytest.mark.parametrize("method", ["TREE", "FLAT", "FULL"])
def test_transport_parity_uses_compact_selection_and_does_not_consume_routing_rng(method):
    torch.set_num_threads(1)
    cfg = fixture_cfg()
    policy = MemoryPolicy(cfg).eval()
    episode = synthetic_episode(length=18, conditioning_length=5, points=32)
    online = (OnlinePolicy(policy, cfg, "FULL", 128) if method == "FULL" else
              TrainingSceneCompactPolicy(policy, cfg, method, 128, selection_family=method.lower()))
    online.reset(ResetResult(episode.conditioning, episode.execution[0], episode.goal), 2701)
    for frame in episode.execution[1:17]:
        online.observe(frame)
    before = online.selection_rng.getstate()
    result = handoff.transport_parity(online, handoff.fast_transfer(policy))
    assert result["context_bit_identical"] and result["actions_bit_identical"]
    assert result["method"] == method and online.selection_rng.getstate() == before
    assert result["selected_records"] == (result["history_leaves"] if method == "FULL" else 128)


def test_explicit_execution_override_does_not_change_saved_config_or_learning_contract():
    cfg = fixture_cfg()
    original = deepcopy(cfg)
    effective, override = diagnostic_config({"stage": "compact", "config": cfg}, {"clip_checkpoint": "/local/clip.pt"})
    assert cfg == original and effective["model"]["execute"] == 8
    assert effective["model"]["horizon"] == 16 and effective["model"]["action_stride"] == 2
    assert effective["evaluation"]["split"] == "train"
    assert override["scope"] == "training_scene_diagnostic_not_benchmark"
    assert override["checkpoint_execute"] == 2 and override["primitive_steps_per_replan"] == 16
    with pytest.raises(ValueError, match="compact"):
        diagnostic_config({"stage": "teacher", "config": cfg}, {"clip_checkpoint": "/local/clip.pt"})


def test_absolute_time_noise_pairs_common_times_without_reader_or_checkpoint_keys():
    a = torch.Generator().manual_seed(handoff.seed_for_primitive_time("VideoUnmask", 0, 32))
    b = torch.Generator().manual_seed(handoff.seed_for_primitive_time("VideoUnmask", 0, 32))
    assert torch.equal(torch.randn((16, 8), generator=a), torch.randn((16, 8), generator=b))
    flow = handoff.seed_for_primitive_time("VideoUnmask", 0, 32)
    assert flow != handoff.seed_for_primitive_time("VideoUnmask", 0, 32, "selection")
    assert flow != handoff.seed_for_primitive_time("VideoUnmask", 0, 48)
    assert flow != handoff.seed_for_primitive_time("VideoUnmask", 1, 32)


def test_generalized_saved_trace_accepts_execute8_and_checks_seed_and_reader(tmp_path, monkeypatch):
    from test_inspection_handoff import _fixture
    path = tmp_path / "result.json"
    data, source, cfg = _fixture(path)
    cfg["model"].update(execute=8, action_stride=2)
    data["temporal_sampling"].update(action_stride=2, execute=8)
    data.update(absolute_time_noise=True, method="TREE", budget=128)
    data["decisions"][0].update(flow_seed=handoff.seed_for_primitive_time("PickXtimes", 0, 0))
    data["decisions"][0]["timing"].update(method="TREE", history_tokens=0, represented_leaf_counts=[])
    path.write_text(json.dumps(data))
    monkeypatch.setattr("imageio_ffmpeg.count_frames_and_secs", lambda _: (3, 0.15))
    validate_trial(path, source, cfg)
    data["decisions"][0]["flow_seed"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        validate_trial(path, source, cfg)
