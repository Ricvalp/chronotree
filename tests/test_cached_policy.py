"""Synthetic CPU parity checks for worker-packed compact policy contexts."""

from copy import deepcopy
import random
from types import SimpleNamespace

import pytest
import torch

from stmem.config import load_config
from stmem.data.dataset import synthetic_episode
from stmem.memory.frontier import validate_frontier
from stmem.models.action_flow import masked_mse
from stmem.models.batched_policy import batch_context, records_bank
from stmem.models.policy import MemoryPolicy, choose_records
from stmem.training.cached_policy import CachedPolicyForward


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(403)


def _config(proprioception, mass_bias):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"].update(width=16, heads=4, layers=1, text_layers=1, max_text_bytes=24,
                        current_tokens=3, points=7, horizon=4, block_size=4, visual_tokens=8,
                        proprioception_mode=proprioception, condition_on_previous_commands=False,
                        action_stride=2, mass_bias=mass_bias)
    cfg["data"].update(observation_stride=2, process_every_observation=False)
    return cfg


def _decisions():
    result = []
    # Reset has no history. The other decisions include a three-frame pending
    # block and a complete block followed by a two-frame pending prefix.
    for seed, conditioning, time in ((810, 0, 0), (811, 5, 6), (812, 5, 12), (813, 3, 8)):
        episode = synthetic_episode(22, conditioning, 7, seed=seed)
        episode.observation_stride = episode.action_stride = 2
        result.append(episode.decision(time, 4))
    result[0].current.valid.zero_()
    result[0].current.points.fill_(float("nan"))
    result[2].target_valid[-1, -1] = False
    return result


def _packed(decisions, evidences, records):
    clocks = torch.tensor([[d.conditioning[-1].time if d.conditioning else 0, d.current.time]
                           for d in decisions], dtype=torch.float32)
    cpu = SimpleNamespace(device=torch.device("cpu"), config={"model": {"width": evidences[0].text.numel()}})
    states = torch.stack([torch.cat([d.current.proprio, d.current.previous_action,
                                    torch.tensor([float(d.current.action_available)])]) for d in decisions])
    return SimpleNamespace(
        positions=tuple(range(len(decisions))),
        current=records_bank(cpu, [e.current_records for e in evidences], clocks),
        history=records_bank(cpu, records, clocks),
        clocks=clocks,
        text=torch.stack([e.text for e in evidences]),
        current_state=states,
        history_leaf_counts=torch.tensor([len(e.archive.leaves) for e in evidences]),
    )


@pytest.mark.parametrize("proprioception", ["all", "none"])
@pytest.mark.parametrize("mass_bias", [True, False])
@pytest.mark.parametrize("current_only_contexts", [False, True])
def test_worker_context_forward_matches_reference_predictions_and_gradients(
    proprioception, mass_bias, current_only_contexts,
):
    policy = MemoryPolicy(_config(proprioception, mass_bias))
    policy.freeze_encoders()
    policy.train()
    reference = deepcopy(policy)
    wrapper = CachedPolicyForward(policy)
    decisions = _decisions()
    with torch.no_grad():
        evidences = [reference.encode(d) for d in decisions]
        methods = ["CURRENT"] * 4 if current_only_contexts else ["CURRENT", "TREE", "FLAT", "UNIFORM"]
        records = [choose_records(evidence, method, 12, rng=random.Random(index)).records
                   for index, (evidence, method) in enumerate(zip(evidences, methods))]
    for method, selected, evidence in zip(methods, records, evidences):
        if method != "CURRENT":
            validate_frontier(selected, evidence.archive.leaves, 12)
    if not current_only_contexts:
        assert any(r.count > 1 for selected in records for r in selected)
        assert any(not r.completed for e in evidences for r in e.archive.leaves)
    batch = _packed(decisions, evidences, records)
    actual_context, actual_bias, actual_stats = wrapper.context(batch)
    expected_context, expected_bias, expected_stats = batch_context(
        reference, decisions, evidences=evidences, record_lists=records,
    )
    torch.testing.assert_close(actual_context, expected_context, rtol=0, atol=0)
    torch.testing.assert_close(actual_bias, expected_bias, rtol=0, atol=0)
    for key in expected_stats:
        torch.testing.assert_close(actual_stats[key], expected_stats[key], rtol=0, atol=0)
    assert actual_stats["history_leaves"][0] == 0
    assert actual_stats["current_visual_tokens"][0] == 0
    if current_only_contexts:
        assert actual_stats["history_context_tokens"].sum() == 0

    noisy = torch.randn(len(decisions), 4, 8)
    time = torch.tensor([0.12, 0.34, 0.56, 0.78])
    targets = torch.randn_like(noisy)
    valid = torch.stack([d.target_valid for d in decisions])
    actual, stats = wrapper(batch, noisy, time)
    expected = reference(decisions, noisy, time, evidence=evidences, records=records)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    masked_mse(actual, targets, valid).backward()
    masked_mse(expected, targets, valid).backward()
    ref_parameters = dict(reference.named_parameters())
    for name, parameter in policy.named_parameters():
        other = ref_parameters[name]
        assert (parameter.grad is None) == (other.grad is None), name
        if name.startswith(("encoder.", "text_encoder.")):
            assert parameter.grad is None and not parameter.requires_grad
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0, msg=name)
    for key in stats:
        torch.testing.assert_close(stats[key], expected_stats[key], rtol=0, atol=0)


def test_cached_wrapper_rejects_trainable_encoders_and_no_history_policy():
    cfg = _config("none", True)
    with pytest.raises(ValueError, match="frozen full-history"):
        CachedPolicyForward(MemoryPolicy(cfg))
    cfg["model"]["history_mode"] = "none"
    policy = MemoryPolicy(cfg)
    policy.freeze_encoders()
    with pytest.raises(ValueError, match="frozen full-history"):
        CachedPolicyForward(policy)
