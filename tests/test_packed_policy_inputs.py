"""Synthetic diagnostics for worker-packed, causal policy inputs."""

from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import pickle
import random

import pytest
import torch

from stmem.config import load_config
from stmem.data.dataset import synthetic_episode
from stmem.models.action_flow import masked_mse
from stmem.models.batched_policy import pack_policy_batch
from stmem.models.policy import MemoryPolicy, choose_records


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(713)


def _config(checkpoint_blocks=False):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"].update(
        width=16,
        heads=4,
        layers=1,
        text_layers=1,
        max_text_bytes=24,
        block_size=3,
        current_tokens=5,
        visual_tokens=7,
        horizon=4,
        checkpoint_blocks=checkpoint_blocks,
        encoder_options={"neighbor_cap": 4, "center_chunk": 3},
        budget=12,
    )
    cfg["training"]["encoder_block_batch_size"] = 2
    return cfg


def _decisions():
    # Empty input, completed/partial conditioning, completed/pending execution,
    # differing point counts, unavailable actions and invalid visual rows.
    decisions = [
        synthetic_episode(5, 0, 0, seed=714).decision(0, 4),
        synthetic_episode(9, 5, 7, seed=715).decision(7, 4),
        synthetic_episode(6, 1, 4, seed=716).decision(2, 4),
    ]
    for i, decision in enumerate(decisions):
        decision.goal = ["", "Move the blue object.", "é"][i]
        decision.task = f"task_{i}"
        decision.episode_id = f"episode_{i + 1}"
        for frame in (*decision.conditioning, *decision.execution_prefix, decision.current):
            # Preserve non-integer clocks and source indices across transport.
            frame.time = frame.time * 1.125 + 0.123456789
            frame.source_index += 4
    decisions[1].execution_prefix[1] = replace(
        decisions[1].execution_prefix[1],
        points=decisions[1].execution_prefix[1].points[:2],
        valid=decisions[1].execution_prefix[1].valid[:2],
    )
    decisions[2].target_valid[:, 7] = False
    for frame in (*decisions[2].conditioning, *decisions[2].execution_prefix, decisions[2].current):
        frame.valid.zero_()
        frame.points.fill_(float("nan"))
    return decisions


def _assert_decision_equal(expected, actual):
    assert expected.goal == actual.goal
    assert expected.task == actual.task
    assert expected.episode_id == actual.episode_id
    assert expected.index == actual.index
    torch.testing.assert_close(expected.target_chunk, actual.target_chunk, rtol=0, atol=0)
    torch.testing.assert_close(expected.target_valid, actual.target_valid, rtol=0, atol=0)
    for left_bank, right_bank in (
        (expected.conditioning, actual.conditioning),
        (expected.execution_prefix, actual.execution_prefix),
        ([expected.current], [actual.current]),
    ):
        assert len(left_bank) == len(right_bank)
        for left, right in zip(left_bank, right_bank):
            for field in fields(left):
                a, b = getattr(left, field.name), getattr(right, field.name)
                if isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b, atol=0, rtol=0, equal_nan=True)
                elif isinstance(a, dict):
                    assert a.keys() == b.keys()
                    for key in a:
                        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
                else:
                    assert a == b


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif is_dataclass(value):
        for field in fields(value):
            yield from _tensors(getattr(value, field.name))
    elif isinstance(value, (list, tuple)):
        for entry in value:
            yield from _tensors(entry)
    elif isinstance(value, dict):
        for entry in value.values():
            yield from _tensors(entry)


def test_packed_cpu_transport_has_fixed_tensor_count_and_exact_canonical_roundtrip():
    original = _decisions()
    packed = pack_policy_batch(original, 3)
    assert len(list(_tensors(packed))) == 17
    assert not list(_tensors(packed.metadata))
    assert all(t.device.type == "cpu" and t.is_contiguous() for t in _tensors(packed))
    assert packed.history_lengths == [0, 12, 3]
    assert [m.conditioning_frames for m in packed.metadata] == [0, 5, 1]
    assert [m.execution_history_frames for m in packed.metadata] == [0, 7, 2]
    # Python metadata is serializable; no closures, file handles or device state.
    transported = pickle.loads(pickle.dumps(packed))
    for expected, actual in zip(original, transported.decisions()):
        _assert_decision_equal(expected, actual)
    rebuilt = packed.decisions()[1]
    assert rebuilt.execution_prefix[0].points.untyped_storage().data_ptr() == (
        packed.history.points.untyped_storage().data_ptr()
    )
    assert rebuilt.current.proprio.untyped_storage().data_ptr() == (
        packed.current_state.untyped_storage().data_ptr()
    )
    larger = pack_policy_batch([synthetic_episode(100, 25, 4).decision(95, 4)], 3)
    assert len(list(_tensors(larger))) == 17


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_worker_packed_teacher_matches_predictions_and_gradients_without_repacking(
    checkpoint_blocks, monkeypatch
):
    from stmem.models import batched_policy

    model = MemoryPolicy(_config(checkpoint_blocks))
    reference = deepcopy(model)
    decisions = _decisions()
    packed = pack_policy_batch(decisions, 3)
    noisy = torch.randn(3, 4, 8)
    times = torch.tensor([0.2, 0.4, 0.8])
    target = torch.randn_like(noisy)
    expected, expected_stats = reference(decisions, noisy, times, return_stats=True)

    def cannot_repack(*args, **kwargs):
        raise AssertionError("Prepacked teacher must not collate frames in the training process")

    monkeypatch.setattr(batched_policy, "pack_blocks", cannot_repack)
    prediction, stats = model(list(packed.metadata), noisy, times, packed_batch=packed, return_stats=True)
    torch.testing.assert_close(prediction, expected, atol=0, rtol=0)
    for name in stats:
        torch.testing.assert_close(stats[name], expected_stats[name], atol=0, rtol=0)
    masked_mse(prediction, target, packed.target_valid).backward()
    masked_mse(expected, target, packed.target_valid).backward()
    for (name, actual), (_, wanted) in zip(model.named_parameters(), reference.named_parameters()):
        assert (actual.grad is None) == (wanted.grad is None), name
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, wanted.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize("perceiver", [False, True])
def test_worker_packed_frozen_readers_match_canonical_inputs(perceiver):
    model = MemoryPolicy(_config())
    if perceiver:
        model.add_perceiver()
    model.freeze_encoders()
    decisions = _decisions()
    packed = pack_policy_batch(decisions, 3)
    reconstructed = packed.decisions()
    with torch.no_grad():
        evidence = [model.encode(d) for d in decisions]
        records = [
            choose_records(e, "PERCEIVER" if perceiver else "FLAT", 12, rng=random.Random(i)).records
            for i, e in enumerate(evidence)
        ]
        noisy = torch.randn(3, 4, 8)
        times = torch.tensor([0.2, 0.4, 0.8])
        expected = model(decisions, noisy, times, evidence=evidence, records=records, perceiver=perceiver)
        actual = model(
            reconstructed,
            noisy,
            times,
            evidence=evidence,
            records=records,
            perceiver=perceiver,
            packed_batch=packed,
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_packed_supervision_and_bookkeeping_do_not_enter_policy_features():
    model = MemoryPolicy(_config()).eval()
    packed = pack_policy_batch(_decisions(), 3)
    noisy = torch.randn(3, 4, 8)
    times = torch.tensor([0.2, 0.4, 0.8])
    with torch.no_grad():
        before = model(packed.metadata, noisy, times, packed_batch=packed)
        packed.target_chunk.fill_(10000)
        packed.target_valid.zero_()
        packed.metadata = [
            replace(info, task="different task ID", episode_id="different episode", index=999)
            for info in packed.metadata
        ]
        after = model(packed.metadata, noisy, times, packed_batch=packed)
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_packed_policy_rejects_mismatched_cardinality_and_block_size():
    model = MemoryPolicy(_config())
    decisions = _decisions()
    packed = pack_policy_batch(decisions, 3)
    noisy = torch.randn(3, 4, 8)
    times = torch.tensor([0.2, 0.4, 0.8])
    with pytest.raises(ValueError, match="one entry per decision"):
        model(decisions[:2], noisy, times, packed_batch=packed)
    with pytest.raises(ValueError, match="different encoder block size"):
        model(decisions, noisy, times, packed_batch=pack_policy_batch(decisions, 2))
    with pytest.raises(ValueError, match="list of decisions"):
        model(decisions[0], noisy[0], times[0], packed_batch=packed)
    with pytest.raises(ValueError, match="at least one decision"):
        pack_policy_batch([], 3)
    with pytest.raises(ValueError, match="positive"):
        pack_policy_batch(decisions, 0)


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_prepared_cluster_centers_roundtrip_and_match_serial_without_reselection(
    checkpoint_blocks, monkeypatch
):
    from stmem.data import center_sampling

    cfg = _config(checkpoint_blocks)
    cfg["model"]["encoder_options"]["center_sampling"] = {"method": "balanced_cluster"}
    policy = MemoryPolicy(cfg)
    serial = deepcopy(policy)
    decisions = _decisions()
    packed = pack_policy_batch(decisions, 3, model_config=cfg["model"])
    rebuilt = packed.decisions()
    for original, actual in zip(decisions, rebuilt):
        _assert_decision_equal(original, actual)
    assert set(packed.history.center_indices) == {2, 3}
    assert set(packed.current_center_indices) == {5}
    # Each quota adds one contiguous storage, not one tensor per history frame.
    assert len(list(_tensors(packed))) == 20

    def cannot_select(*args, **kwargs):
        raise AssertionError("Prepared center indices must be reused, including checkpoint recomputation")

    monkeypatch.setattr(center_sampling, "select_center_indices", cannot_select)
    again = pack_policy_batch(decisions, 3, model_config=cfg["model"])
    assert again.history.center_sampling_signature == packed.history.center_sampling_signature
    noisy, times = torch.randn(3, 4, 8), torch.tensor([0.2, 0.4, 0.8])
    expected = torch.stack([serial(d, noisy[i], times[i]) for i, d in enumerate(rebuilt)])
    actual = policy(packed.metadata, noisy, times, packed_batch=packed)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=3e-5)
    target = torch.randn_like(noisy)
    masked_mse(actual, target, packed.target_valid).backward()
    masked_mse(expected, target, packed.target_valid).backward()
    for (name, left), (_, right) in zip(policy.named_parameters(), serial.named_parameters()):
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, atol=4e-6, rtol=2e-4, msg=name)


def test_prepared_center_settings_mismatch_is_rejected_before_encoding():
    cfg = _config()
    cfg["model"]["encoder_options"]["center_sampling"] = {"method": "balanced_cluster"}
    packed = pack_policy_batch(_decisions(), 3, model_config=cfg["model"])
    cfg["model"]["encoder_options"]["center_sampling"]["height_margin"] = 0.02
    policy = MemoryPolicy(cfg)
    with pytest.raises(ValueError, match="center settings do not match"):
        policy(packed.metadata, torch.randn(3, 4, 8), torch.rand(3), packed_batch=packed)


def test_frame_center_cache_is_recomputed_when_selector_settings_change(monkeypatch):
    from stmem.data import center_sampling
    from stmem.models.spacetime_encoder import cached_frame_centers

    frame = synthetic_episode(3, 0, 40).execution[0]
    select = center_sampling.select_center_indices
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return select(*args, **kwargs)

    monkeypatch.setattr(center_sampling, "select_center_indices", counted)
    settings = {"method": "balanced_cluster"}
    first = cached_frame_centers(frame, 5, settings)
    assert cached_frame_centers(frame, 5, settings) is first
    assert len(calls) == 1
    old_signature = frame.center_sampling_signature
    cached_frame_centers(frame, 5, {**settings, "height_margin": 0.02})
    assert len(calls) == 2
    assert frame.center_sampling_signature != old_signature
