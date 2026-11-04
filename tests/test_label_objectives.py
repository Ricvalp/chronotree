"""Synthetic CPU scientific checks for matched utility objectives and batching."""

# ruff: noqa: F811
from copy import deepcopy

import pytest
import torch

from test_cached_context_loader import fixture  # noqa: F401
from test_packed_context import indexed as indexed
from test_label_data import _configure, _reference
from stmem.training.gain_labels import paired_gains
from stmem.training.label_data import LabelDataReader, LabelPlan
from stmem.training.label_objectives import evaluate_batch


def policies(teacher):
    compact = deepcopy(teacher)
    with torch.no_grad():
        compact.adapter.meta[0].weight.mul_(0.7)
        compact.decoder.out[-1].bias.add_(0.03)
    for model in (teacher, compact):
        model.freeze_encoders()
        model.eval().requires_grad_(False)
    return teacher, compact


@pytest.mark.parametrize("steps", [4, 8])
def test_batched_two_objectives_match_independent_reference(indexed, steps):
    cfg, data, teacher, cache, root = indexed
    _configure(cfg)
    teacher, compact = policies(teacher)
    plan = LabelPlan(data, cfg).items()[:6]
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        payloads = [reader.prepare(item) for item in plan]
    got, timings = evaluate_batch(teacher, compact, payloads, integration_steps=steps,
                                  teacher_token_budget=100, compact_batch_size=7)
    assert timings["teacher_seconds"] > 0 and timings["compact_seconds"] > 0
    for position, (item, payload) in enumerate(zip(plan, payloads)):
        decision = data.episode(item["episode_index"]).decision(item["index"], data.horizon)
        evidence = cache.evidence(decision)
        generator = torch.Generator().manual_seed(item["seed"])
        noise = torch.randn(decision.target_chunk.shape, generator=generator)
        flow_time = torch.rand((), generator=generator)
        for family in ("tree", "flat"):
            before, after, _ = _reference(cfg, item, evidence, family)
            if not after:
                continue
            for objective in ("velocity", "final_action"):
                expected, _ = paired_gains(
                    teacher, compact, evidence, before, after, decision.target_chunk, decision.target_valid,
                    noise=noise, time=flow_time, objective=objective, integration_steps=steps,
                )
                torch.testing.assert_close(got[objective][family][position], torch.stack(expected),
                                           rtol=3e-4, atol=2e-6)


def test_noise_and_gains_are_independent_of_processing_order_and_microbatch(indexed):
    cfg, data, teacher, cache, root = indexed
    _configure(cfg)
    teacher, compact = policies(teacher)
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        payloads = [reader.prepare(item) for item in LabelPlan(data, cfg).items()[:5]]
    batched, _ = evaluate_batch(teacher, compact, payloads, integration_steps=4, compact_batch_size=64)
    reversed_result, _ = evaluate_batch(teacher, compact, payloads[::-1], integration_steps=4,
                                       teacher_token_budget=1, compact_batch_size=1)
    for objective in batched:
        for family in batched[objective]:
            for before, after in zip(batched[objective][family], reversed_result[objective][family][::-1]):
                if before is None:
                    assert after is None
                else:
                    torch.testing.assert_close(before, after, rtol=3e-4, atol=2e-6)


def test_final_action_labels_never_use_expert_targets_as_initial_noise(indexed):
    cfg, data, teacher, cache, root = indexed
    _configure(cfg)
    teacher, compact = policies(teacher)
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        payloads = [reader.prepare(item) for item in LabelPlan(data, cfg).items()[:4]]
    altered = deepcopy(payloads)
    for item in altered:
        item["target"] += 100
    before, _ = evaluate_batch(teacher, compact, payloads, integration_steps=4)
    after, _ = evaluate_batch(teacher, compact, altered, integration_steps=4)
    velocity_changes = []
    for family in ("tree", "flat"):
        for left, right, lv, rv in zip(before["final_action"][family], after["final_action"][family],
                                      before["velocity"][family], after["velocity"][family]):
            if left is not None:
                torch.testing.assert_close(left, right, rtol=0, atol=0)
                velocity_changes.append(not torch.allclose(lv, rv, atol=1e-6))
    assert any(velocity_changes)


def test_normalization_mismatch_cannot_generate_labels(indexed):
    cfg, data, teacher, cache, root = indexed
    _configure(cfg)
    teacher, compact = policies(teacher)
    compact.normalizer.std.mul_(2)
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        payload = reader.prepare(LabelPlan(data, cfg).items()[0])
    with pytest.raises(ValueError, match="normalization"):
        evaluate_batch(teacher, compact, [payload])
