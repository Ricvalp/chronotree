"""Synthetic temporal-sampling invariants; no benchmark performance claims."""

from copy import deepcopy

import h5py
import numpy as np
import pytest
import torch

from stmem.config import load_config
from stmem.data.canonical import retained_observations, valid_decision_indices
from stmem.data.dataset import EpisodeDataset, synthetic_episode
from stmem.data.feature_cache import FeatureRow
from stmem.models.action_flow import ActionNormalizer
from stmem.training.data import TrainingData
from test_frozen_feature_cache import cache_input, prepare  # noqa: F401 -- imported pytest fixture
from test_training_startup import _fixture_config, _forbid_visual_loading


def episode(*, alignment="synthetic_post_action_v1", length=40):
    result = synthetic_episode(length=length, conditioning_length=7, points=4)
    result.actions = torch.arange(length * 8, dtype=torch.float32).reshape(length, 8)
    result.alignment = alignment
    result.observation_stride = 2
    result.action_stride = 2
    return result


@pytest.mark.parametrize("alignment,offset", [("synthetic_post_action_v1", 1), ("pre_action_v1", 0)])
def test_coarse_targets_preserve_alignment_and_native_clocks(alignment, offset):
    ep = episode(alignment=alignment)
    decision = ep.decision(4, 16)
    assert decision.index == 4 and decision.current is ep.execution[4]
    assert [f.source_index for f in decision.conditioning] == [0, 2, 4, 6]
    assert [f.time for f in decision.conditioning] == [0, 2, 4, 6]
    assert [f.source_index for f in decision.execution_prefix] == [0, 2]
    assert [f.time for f in decision.execution_prefix] == [0, 2]
    assert all(f is not decision.current for f in decision.execution_prefix)
    expected = ep.actions[4 + offset : 4 + offset + 32 : 2]
    torch.testing.assert_close(decision.target_chunk, expected, rtol=0, atol=0)
    assert decision.target_valid.all()
    assert all(not f.action_available for f in decision.conditioning)
    assert all(not f.previous_action.any() for f in decision.conditioning)
    with pytest.raises(IndexError, match="retained observation cadence"):
        ep.decision(3, 16)


def test_observation_filter_uses_source_clock_without_renumbering():
    ep = episode()
    frames = [ep.execution[i] for i in (0, 1, 4, 5, 8)]
    selected = retained_observations(frames, 2)
    assert [f.source_index for f in selected] == [0, 4, 8]
    assert [f.time for f in selected] == [0, 4, 8]
    assert all(frame is original for frame, original in zip(selected, frames[::2]))


def test_coarse_targets_never_skip_an_invalid_intervening_joint_command():
    ep = episode()
    # Selected rows are 1,3,5,...; row4 is not itself a target, but its
    # invalid command prevents the second hold and every later target.
    ep.action_valid[4, 2] = False
    decision = ep.decision(0, 16)
    assert decision.target_valid[0].all()
    assert not decision.target_valid[1:].any()
    ep.action_valid[2, 0] = False
    assert not ep.decision(0, 16).target_valid.any()
    assert 0 not in valid_decision_indices(ep.action_valid, ep.alignment, horizon=16, action_stride=2)


def test_coarse_gripper_mask_checks_intervening_commands_without_hiding_valid_joints():
    ep = episode()
    ep.action_valid[2, 7] = False
    decision = ep.decision(0, 16)
    assert decision.target_valid[:, :7].all()
    assert not decision.target_valid[0, 7]
    assert decision.target_valid[1:, 7].all()


@pytest.mark.parametrize("length,valid_targets", [(1, 0), (2, 0), (3, 1), (4, 1), (5, 2)])
def test_coarse_target_requires_a_complete_native_hold_interval(length, valid_targets):
    ep = episode(length=length)
    decision = ep.decision(0, 4)
    assert decision.target_valid[:valid_targets].all()
    assert not decision.target_valid[valid_targets:].any()
    assert not decision.target_chunk[valid_targets:].any()


@pytest.mark.parametrize("action_stride", [1, 2, 3])
@pytest.mark.parametrize("alignment", ["pre_action_v1", "post_action_v1"])
def test_action_only_candidate_index_matches_coarse_decision_oracle(action_stride, alignment):
    ep = episode(alignment=alignment, length=15)
    ep.observation_stride = 1
    ep.action_stride = action_stride
    ep.action_valid[4, 0] = False
    ep.action_valid[8, 7] = False
    ep.action_valid[-1] = False
    for decision_stride in (1, 2, 4):
        expected = [
            index for index in range(0, len(ep), decision_stride) if ep.decision(index, 4).target_valid.any()
        ]
        actual = valid_decision_indices(
            ep.action_valid, alignment, horizon=4, stride=decision_stride, action_stride=action_stride
        )
        assert actual == expected


def test_downsampled_inputs_are_causal_and_keep_lazy_features():
    ep = episode()
    for frame in ep.conditioning + ep.execution:
        frame.visual_features = FeatureRow("unused.h5", frame.source, frame.source_index, 4, "diagnostic", 0)
    before = ep.decision(4, 16)
    original_conditioning = [f.event_id for f in before.conditioning]
    original_prefix = [f.event_id for f in before.execution_prefix]
    for frame in ep.execution[5:]:
        frame.points.fill_(12345)
        frame.previous_action.fill_(-12345)
    ep.actions[10:] += 1000
    after = ep.decision(4, 16)
    assert [f.event_id for f in after.conditioning] == original_conditioning
    assert [f.event_id for f in after.execution_prefix] == original_prefix
    assert after.current is before.current
    assert all(isinstance(f.visual_features, FeatureRow) for f in after.conditioning + after.execution_prefix)
    assert [f.visual_features.row for f in after.execution_prefix] == [0, 2]


def test_native_default_keeps_consecutive_targets_and_every_observation():
    ep = synthetic_episode(length=8, conditioning_length=5, points=4)
    decision = ep.decision(3, 4)
    torch.testing.assert_close(decision.target_chunk, ep.actions[4:8], rtol=0, atol=0)
    assert len(decision.conditioning) == 5 and len(decision.execution_prefix) == 3
    assert decision.target_valid.all()


def test_dataset_cadence_reuses_unchanged_geometry_and_feature_cache(cache_input):  # noqa: F811 -- pytest fixture
    cfg, _, canonical, output = cache_input
    prepare(cache_input)
    before = canonical.read_bytes(), output.read_bytes()
    cfg["data"]["observation_stride"] = 2
    cfg["model"]["action_stride"] = 2
    dataset = EpisodeDataset(canonical, visual_features_config=cfg)
    dataset.validate_preprocessing(cfg)
    ep = dataset.load_episode(0)
    assert (ep.observation_stride, ep.action_stride) == (2, 2)
    assert len(ep.execution) == dataset.load_episode_actions(0).execution_length
    decision = ep.decision(2, 4)
    assert [f.source_index for f in decision.execution_prefix] == [0]
    assert [f.source_index for f in decision.conditioning] == [0]
    assert isinstance(decision.current.visual_features, FeatureRow)
    assert decision.current.visual_features.row == 4
    assert before == (canonical.read_bytes(), output.read_bytes())


def test_training_sampler_and_normalizer_keep_native_statistics(tmp_path, monkeypatch):
    cfg = _fixture_config(tmp_path)
    cfg["data"]["decision_stride"] = 4
    # Make full held intervals available; the fixture is deliberately synthetic.
    for path in (tmp_path / "prepared").glob("*.h5"):
        with h5py.File(path, "r+") as handle:
            for group in handle.values():
                source = np.asarray(group["source"])
                masks = np.ones_like(group["action_valid"][:])
                masks[source] = False
                masks[np.flatnonzero(~source)[0]] = False
                group["action_valid"][:] = masks
    native = TrainingData(cfg)
    native_normalizer = ActionNormalizer()
    native.fit_normalizer(native_normalizer)
    sampled_cfg = deepcopy(cfg)
    sampled_cfg["data"]["observation_stride"] = 2
    sampled_cfg["model"]["action_stride"] = 2
    with monkeypatch.context() as guarded:
        _forbid_visual_loading(guarded)
        sampled = TrainingData(sampled_cfg)
    assert sampled.identity != native.identity
    sampled_normalizer = ActionNormalizer()
    sampled.fit_normalizer(sampled_normalizer)
    for key in ("mean", "std", "count"):
        torch.testing.assert_close(
            getattr(sampled_normalizer, key), getattr(native_normalizer, key), rtol=0, atol=0
        )
    for task, candidates in sampled.candidates.items():
        assert candidates
        for episode_index, index in candidates:
            assert index % 4 == 0
            decision = sampled.decision(episode_index, index)
            assert decision.target_valid.any()
            assert all(f.source_index % 2 == 0 for f in decision.execution_prefix + decision.conditioning)


def test_synthetic_training_respects_same_temporal_contract():
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["data"].update(observation_stride=2, decision_stride=4, synthetic_execution_frames=40)
    cfg["model"].update(horizon=16, action_stride=2)
    data = TrainingData(cfg)
    decision = data.sample_position(0)
    assert decision.target_chunk.shape == (16, 8)
    assert all(f.source_index % 2 == 0 for f in decision.execution_prefix + decision.conditioning)
    cfg["data"]["decision_stride"] = 3
    with pytest.raises(ValueError, match="multiple"):
        TrainingData(cfg)
