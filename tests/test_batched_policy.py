"""Synthetic diagnostics: padded policy training matches causal serial reads."""

from copy import deepcopy
from dataclasses import replace
import random

import pytest
import torch

from stmem.config import load_config
from stmem.data.dataset import synthetic_episode
from stmem.models.action_flow import masked_mse
from stmem.models.batched_policy import batch_context, pack_blocks, records_bank
from stmem.models.metadata_adapter import metadata
from stmem.models.policy import MemoryPolicy, choose_records


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(204)


def _config(*, checkpoint_blocks=False, variant="joint"):
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
        points=7,
        encoder=variant,
        checkpoint_blocks=checkpoint_blocks,
        encoder_options={"neighbor_cap": 4, "center_chunk": 3},
    )
    cfg["training"]["encoder_block_batch_size"] = 2
    return cfg


def _decisions():
    empty = synthetic_episode(5, 0, 0, seed=401).decision(0, 4)
    empty.goal = ""
    complete = synthetic_episode(8, 5, 7, seed=402).decision(6, 4)
    complete.goal = "Move the blue object."
    pending = synthetic_episode(5, 1, 4, seed=403).decision(2, 4)
    pending.goal = "é"
    # Zero occupied visual leaves must not remove observed robot-event leaves.
    for frame in [pending.current, *pending.conditioning, *pending.execution_prefix]:
        frame.valid.zero_()
        frame.points.fill_(float("nan"))
    pending.target_valid[:, 7] = False
    return [empty, complete, pending]


def _draws(decisions):
    noisy = torch.randn(len(decisions), 4, 8)
    times = torch.linspace(0.15, 0.75, len(decisions))
    targets = torch.randn_like(noisy)
    valid = torch.stack([decision.target_valid for decision in decisions])
    return noisy, times, targets, valid


def _compare_gradients(actual, expected):
    expected_parameters = dict(expected.named_parameters())
    for name, parameter in actual.named_parameters():
        reference = expected_parameters[name]
        assert (parameter.grad is None) == (reference.grad is None), name
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, reference.grad, atol=4e-6, rtol=2e-4, msg=name)


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
@pytest.mark.parametrize("variant", ["joint", "framewise"])
def test_batched_teacher_matches_serial_predictions_gradients_and_leaf_counts(checkpoint_blocks, variant):
    model = MemoryPolicy(_config(checkpoint_blocks=checkpoint_blocks, variant=variant))
    separate = deepcopy(model)
    decisions = _decisions()
    noisy, times, targets, valid = _draws(decisions)
    prediction, stats = model(decisions, noisy, times, return_stats=True)
    evidences = [separate.encode(decision) for decision in decisions]
    expected = torch.stack(
        [
            separate(decision, noisy[i], times[i], evidence=evidences[i])
            for i, decision in enumerate(decisions)
        ]
    )
    torch.testing.assert_close(prediction, expected, atol=2e-6, rtol=3e-5)
    assert stats["history_leaves"].tolist() == [len(evidence.archive.leaves) for evidence in evidences]
    assert stats["current_visual_tokens"].tolist() == [
        len(evidence.current_records) for evidence in evidences
    ]
    assert stats["history_leaves"][0] == 0
    assert stats["history_leaves"][2] == 3  # One conditioning + two execution event leaves.
    assert stats["current_visual_tokens"][0] == stats["current_visual_tokens"][2] == 0
    masked_mse(prediction, targets, valid).backward()
    torch.stack(
        [masked_mse(expected[i], targets[i], valid[i]) for i in range(len(decisions))]
    ).mean().backward()
    _compare_gradients(model, separate)


def test_batched_policy_independent_of_batch_padding_and_other_examples():
    model = MemoryPolicy(_config()).eval()
    decisions = _decisions()
    noisy, times, _, _ = _draws(decisions)
    with torch.no_grad():
        original = model(decisions, noisy, times)
        single = model([decisions[2]], noisy[2:3], times[2:3])
        reordered = model(decisions[::-1], noisy.flip(0), times.flip(0))
        perturbed = deepcopy(decisions)
        for frame in [perturbed[1].current, *perturbed[1].conditioning, *perturbed[1].execution_prefix]:
            frame.points.add_(100)
            frame.proprio.add_(10)
        perturbed[1].goal = "A completely different instruction"
        altered = model(perturbed, noisy, times)
    torch.testing.assert_close(original[2:3], single, atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(original, reordered.flip(0), atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(original[[0, 2]], altered[[0, 2]], atol=1e-6, rtol=2e-5)
    assert not torch.allclose(original[1], altered[1])


def test_batched_teacher_future_observations_actions_and_targets_cannot_change_prediction():
    model = MemoryPolicy(_config()).eval()
    episode = synthetic_episode(9, 2, 6, seed=414)
    early = episode.decision(2, 4)
    companion = synthetic_episode(7, 4, 3, seed=415).decision(5, 4)
    noisy, times, _, _ = _draws([early, companion])
    with torch.no_grad():
        before = model([early, companion], noisy, times)
        changed = deepcopy(episode)
        for frame in changed.execution[3:]:
            frame.points.fill_(float("nan"))
            frame.proprio.add_(100)
            frame.previous_action.fill_(99)
        changed.actions[3:] = 999
        changed.action_valid[3:] = False
        after = model([changed.decision(2, 4), companion], noisy, times)
    torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("methods", [("TREE", "FLAT", "CURRENT"), ("PERCEIVER",) * 3])
def test_cached_compact_and_perceiver_batches_match_serial_readers(methods):
    cfg = _config()
    cfg["model"]["budget"] = 12
    model = MemoryPolicy(cfg)
    perceiver = methods[0] == "PERCEIVER"
    if perceiver:
        model.add_perceiver()
    model.freeze_encoders()
    separate = deepcopy(model)
    decisions = _decisions()
    # Give both TREE and FLAT a nonempty archive, including residual/node masses.
    decisions = [decisions[1], synthetic_episode(9, 3, 6, seed=425).decision(7, 4), decisions[0]]
    with torch.no_grad():
        evidences = [model.encode(decision) for decision in decisions]
        records = [
            choose_records(evidence, method, 12, rng=random.Random(i)).records
            for i, (evidence, method) in enumerate(zip(evidences, methods))
        ]
    noisy, times, targets, valid = _draws(decisions)
    prediction, stats = model(
        decisions,
        noisy,
        times,
        evidence=evidences,
        records=records,
        perceiver=perceiver,
        return_stats=True,
    )
    assert stats["history_leaves"].tolist() == [len(evidence.archive.leaves) for evidence in evidences]
    assert stats["history_context_tokens"].tolist() == [
        (cfg["model"]["budget"] if selected else 0) if perceiver else len(selected) for selected in records
    ]
    expected = torch.stack(
        [
            separate(
                decision, noisy[i], times[i], evidence=evidences[i], records=records[i], perceiver=perceiver
            )
            for i, decision in enumerate(decisions)
        ]
    )
    torch.testing.assert_close(prediction, expected, atol=2e-6, rtol=3e-5)
    masked_mse(prediction, targets, valid).backward()
    masked_mse(expected, targets, valid).backward()
    _compare_gradients(model, separate)
    if not perceiver:
        assert any(record.count > 1 for record in records[0])
        assert any(record.disconnected for record in records[1])


def test_packed_record_metadata_matches_reference_for_nodes_residuals_and_empty_rows():
    model = MemoryPolicy(_config()).eval()
    decision = synthetic_episode(9, 4, 5, seed=435).decision(7, 4)
    with torch.no_grad():
        evidence = model.encode(decision)
    records = [
        choose_records(evidence, "TREE", 12, rng=random.Random(0)).records,
        [],
        choose_records(evidence, "FLAT", 12, rng=random.Random(1)).records,
    ]
    clocks = torch.tensor([[3.0, 7.0], [0.0, 0.0], [3.0, 7.0]])
    bank = records_bank(model, records, clocks)
    for i, selected in enumerate(records):
        count = len(selected)
        assert bank.valid[i].sum() == count
        if count:
            expected = torch.stack([metadata(record, evidence.clocks) for record in selected])
            torch.testing.assert_close(bank.metadata[i, :count], expected, atol=0, rtol=0)
            torch.testing.assert_close(bank.content[i, :count], torch.stack([r.content for r in selected]))
    assert not bank.valid[1].any()


@pytest.mark.parametrize("count", [2, 4])
def test_cached_batched_inputs_reject_mismatched_bank_cardinality(count):
    model = MemoryPolicy(_config()).eval()
    decisions = _decisions()
    with torch.no_grad():
        evidences = [model.encode(decision) for decision in decisions]
    noisy, times, _, _ = _draws(decisions)
    records = ([evidence.archive.leaves for evidence in evidences] + [[]])[:count]
    with pytest.raises(ValueError):
        model(decisions, noisy, times, evidence=evidences, records=records)
    with pytest.raises(ValueError):
        records_bank(model, records, torch.zeros(3, 2))
    with pytest.raises(ValueError):
        model(decisions, noisy, times, evidence=(evidences + [evidences[0]])[:count])


def test_padding_remains_masked_after_metadata_adapter_bias():
    model = MemoryPolicy(_config()).eval()
    decisions = _decisions()
    with torch.no_grad():
        context, bias, _ = batch_context(model, decisions)
    assert torch.isfinite(context).all()
    assert torch.count_nonzero(context[~torch.isfinite(bias)]) == 0
    assert int(torch.isfinite(bias[0]).sum()) == 3  # Proprioception, instruction, clock only.


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("perceiver", [False, True])
def test_singleton_batch_with_completely_empty_visual_and_history_banks(cached, perceiver):
    model = MemoryPolicy(_config())
    if perceiver:
        model.add_perceiver()
    decision = synthetic_episode(5, 0, 0, seed=438).decision(0, 4)
    noisy, times, targets, valid = _draws([decision])
    with torch.no_grad():
        evidence = model.encode(decision) if cached else None
    kwargs = {"evidence": [evidence]} if cached else {}
    prediction, stats = model([decision], noisy, times, perceiver=perceiver, return_stats=True, **kwargs)
    expected = model(decision, noisy[0], times[0], evidence=evidence, perceiver=perceiver)
    assert prediction.shape == (1, 4, 8)
    assert stats["history_leaves"].tolist() == stats["current_visual_tokens"].tolist() == [0]
    torch.testing.assert_close(prediction[0], expected, atol=1e-6, rtol=2e-5)
    masked_mse(prediction, targets, valid).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("violation", ["current_in_history", "future", "duplicate", "source", "clock"])
def test_batched_prefix_rejects_invalid_source_or_causal_boundaries(violation):
    decision = synthetic_episode(6, 2, 3, seed=440).decision(3, 4)
    if violation == "current_in_history":
        decision.execution_prefix.append(decision.current)
    elif violation == "future":
        decision.execution_prefix[-1] = replace(decision.execution_prefix[-1], time=4)
    elif violation == "duplicate":
        decision.conditioning[1] = replace(
            decision.conditioning[1], event_id=decision.conditioning[0].event_id
        )
    elif violation == "source":
        decision.execution_prefix[0] = replace(decision.execution_prefix[0], source="conditioning")
    else:
        decision.current = replace(decision.current, time=float("nan"))
    with pytest.raises(ValueError):
        pack_blocks([decision], 3)
