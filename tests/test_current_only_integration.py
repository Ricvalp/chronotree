"""Current-only ablation integration; synthetic fixtures, not benchmark scores."""

from copy import deepcopy
import json

import h5py
import pytest
import torch

from stmem.data.dataset import EpisodeDataset, synthetic_episode
from stmem.evaluation.robomme_adapter import ResetResult, RoboMMEAdapter
from stmem.evaluation.runner import OnlinePolicy
from stmem.models.policy import MemoryPolicy
from stmem.training.attention import PolicyAttention, context_groups
from stmem.training.data import TrainingData
from stmem.training.runtime import validate_policy_config
from stmem.visualization.diagnostics import PolicyScenes
from test_current_only_policy import config
from test_training_startup import _fixture_config
from test_frozen_feature_cache import cache_input, prepare  # noqa: F401 -- reusable pytest fixture
from test_attention_training import Tracker
from test_temporal_rollout import CadenceBenchmark


@pytest.mark.parametrize("alignment", ["pre_action_v1", "post_action_v1"])
@pytest.mark.parametrize("stride", [1, 2])
def test_selective_reader_preserves_supervision_and_reads_only_current_rows(
    tmp_path, monkeypatch, alignment, stride
):
    cfg = _fixture_config(tmp_path, alignment=alignment)
    cfg["model"].update(
        history_mode="none",
        proprioception_mode="none",
        condition_on_previous_commands=False,
        action_stride=stride,
    )
    cfg["data"].update(observation_stride=stride, decision_stride=stride)
    dataset = EpisodeDataset(tmp_path / "prepared", visual_features_config=cfg)
    full = dataset.load_episode(0)
    indices = [4, 0, 4, 2]
    wanted = [full.decision(i, cfg["model"]["horizon"]) for i in indices]
    original = h5py.Dataset.__getitem__
    reads = []

    def selective(dataset, selection):
        if dataset.name.endswith(("/points", "/valid", "/proprio")) or "/center_indices/" in dataset.name:
            assert isinstance(selection, list), (dataset.name, selection)
            assert selection == [1, 4, 7]  # Interleaved source fixture: execution indices 0,2,4.
            reads.append(dataset.name)
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", selective)
    monkeypatch.setattr(dataset, "load_episode", lambda *_: pytest.fail("loaded history"))
    actual = dataset.load_current_decisions(0, indices, cfg["model"]["horizon"])
    assert reads
    for got, expected in zip(actual, wanted):
        assert not got.conditioning and not got.execution_prefix
        assert not got.current.action_available and not got.current.previous_action.any()
        assert got.index == expected.index and got.current.event_id == expected.current.event_id
        for field in ("points", "valid", "proprio"):
            torch.testing.assert_close(
                getattr(got.current, field), getattr(expected.current, field), rtol=0, atol=0
            )
        torch.testing.assert_close(got.target_chunk, expected.target_chunk, rtol=0, atol=0)
        torch.testing.assert_close(got.target_valid, expected.target_valid, rtol=0, atol=0)


def test_training_sampler_uses_selective_reads_without_episode_loading(tmp_path, monkeypatch):
    cfg = _fixture_config(tmp_path)
    cfg["model"]["history_mode"] = "none"
    data = TrainingData(cfg)
    positions = [9, 0, 9, 2, 7]
    expected = [data.decision(*data.sample_reference(p)) for p in positions]
    monkeypatch.setattr(data, "episode", lambda *_: pytest.fail("loaded full episode"))
    actual = data.sample_positions(positions)
    for got, wanted in zip(actual, expected):
        assert (got.task, got.episode_id, got.index) == (wanted.task, wanted.episode_id, wanted.index)
        assert not got.conditioning and not got.execution_prefix
        torch.testing.assert_close(got.target_chunk, wanted.target_chunk, rtol=0, atol=0)


def test_selective_frozen_features_keep_exact_point_correspondence(cache_input):  # noqa: F811
    cfg, _, canonical, _ = cache_input
    prepare(cache_input)
    dataset = EpisodeDataset(canonical, visual_features_config=cfg)
    full = dataset.load_episode(0)
    selected = dataset.load_current_decisions(0, [3, 0, 3])
    for decision in selected:
        wanted = full.execution[decision.index]
        assert decision.current.visual_features == wanted.visual_features
        for field in ("points", "valid", "point_camera", "point_uv"):
            torch.testing.assert_close(
                getattr(decision.current, field), getattr(wanted, field), rtol=0, atol=0
            )


def test_online_current_only_never_encodes_or_retains_history_and_matches_offline(monkeypatch):
    torch.set_num_threads(1)
    cfg = config()
    cfg["training"]["precision"] = "fp32"
    cfg["data"]["observation_stride"] = 2
    policy = MemoryPolicy(cfg).eval()
    episode = synthetic_episode(19, 6, 8)
    online = OnlinePolicy(policy, cfg, "FULL", 1)
    monkeypatch.setattr(
        policy.encoder, "encode_frames", lambda *_args, **_kwargs: pytest.fail("encoded history")
    )
    online.reset(ResetResult([object()], episode.execution[0], episode.goal), seed=27)
    for frame in episode.execution[1:17]:
        online.observe(frame)
    assert online.memory.current is episode.execution[16]
    assert not online.memory.archive.leaves
    assert online.memory.storage["archive_payload_bytes"] == 0
    assert online.memory.storage["pending_raw_bytes"] == 0
    assert online.memory.storage["encoding_calls"] == 0
    evidence = policy.encode(episode.decision(16, cfg["model"]["horizon"]))
    with torch.no_grad():
        expected = policy.predict(evidence, [], generator=torch.Generator().manual_seed(27))
        actual, stats = online.decide()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert stats["history_tokens"] == stats["archive_fine_leaves"] == 0
    assert online.clocks == {"conditioning": 0.0, "execution": 0.0}
    online.reset(ResetResult([], episode.execution[0], episode.goal), seed=27)
    assert online.memory.execution_observations == 1
    with pytest.raises(ValueError, match="FULL/CURRENT"):
        OnlinePolicy(policy, cfg, "UNIFORM", 32)


def test_current_only_adapter_skips_conditioning_feature_extraction():
    class Extractor:
        signature = "synthetic-only"
        calls = 0

        def sample(self, obs, camera, uv, valid):
            self.calls += 1
            return torch.zeros(len(valid), 256)

    extractor = Extractor()
    adapter = RoboMMEAdapter(
        "VideoUnmask",
        env=CadenceBenchmark(),
        points=8,
        crop=None,
        feature_extractor=extractor,
        include_conditioning=False,
    )
    reset = adapter.reset()
    assert not reset.conditioning and extractor.calls == 1
    assert reset.metadata["conditioning_frames_recorded"] == 5
    assert reset.metadata["conditioning_frames"] == 0
    assert not reset.metadata["conditioning_included"]


def test_current_only_attention_probe_matches_context_and_has_no_history_mass(tmp_path):
    cfg = config()
    cfg["model"]["points"] = 8
    cfg["tracking"].update(enabled=True, system_stats=False, mode="offline")
    data = TrainingData(cfg)
    policy = MemoryPolicy(cfg).eval()
    tracker = Tracker(tmp_path)
    scenes = PolicyScenes(cfg, data, policy, tracker, stage="teacher")
    attention = PolicyAttention(cfg, policy, tracker, scenes, stage="teacher", every=10)
    attention.log(10)
    row = json.loads((tmp_path / "attention.jsonl").read_text())
    for example, decision in zip(row["examples"], scenes.decisions):
        evidence = policy.encode(decision)
        groups = context_groups(evidence, [], policy.device, current_state_enabled=False, clock_enabled=False)
        assert len(groups) == len(evidence.current_records) + 1
        assert example["context_tokens"] == len(groups)
        assert example["metrics"]["mass/history"] == 0
        assert example["metrics"]["mass/clock"] == 0


def test_checkpoint_contract_rejects_history_mode_change():
    cfg = config()
    cfg["model"]["encoder_options"]["center_sampling"] = {"method": "fps"}
    changed = deepcopy(cfg)
    changed["model"]["history_mode"] = "full"
    with pytest.raises(ValueError, match="history_mode"):
        validate_policy_config(cfg, changed)
