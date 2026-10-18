"""Checks that keep action-derived utility and routing conditioning causal."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from stmem.data.canonical import Episode, Frame
from stmem.memory.frontier import select_tree, split_frontier
from stmem.models.policy import MemoryPolicy
from stmem.models.router import CachedScorer, LocalRouter
from stmem.models.action_flow import masked_mse
from stmem.training.gain_labels import paired_gains
from stmem.analyze import (
    intervene_evidence,
    exhaustive_tree_frontiers,
    exhaustive_flat_contexts,
    SearchSpaceExceeded,
    routing_diagnostic,
)
from stmem.memory.records import leaf_record
from stmem.memory.hierarchy import local_tree
from stmem.memory.frontier import validate_frontier


def tiny_policy():
    return MemoryPolicy(
        {
            "model": {
                "width": 16,
                "visual_tokens": 4,
                "block_size": 2,
                "current_tokens": 4,
                "heads": 2,
                "layers": 1,
                "horizon": 3,
                "text_layers": 1,
                "max_text_bytes": 24,
                "flow_steps": 2,
                "encoder_options": {"neighbor_cap": 8, "center_chunk": 4},
            }
        }
    )


def tiny_episode():
    generator = torch.Generator().manual_seed(34)

    def frame(i, source):
        points = torch.rand(10, 6, generator=generator)
        points[:, :3] *= 0.1
        return Frame(
            points,
            torch.ones(10, dtype=torch.bool),
            torch.rand(15, generator=generator),
            torch.rand(8, generator=generator),
            source == "execution" and i > 0,
            float(i),
            source,
            f"{source}:{i}",
        )

    return Episode(
        [frame(i, "conditioning") for i in range(3)],
        [frame(i, "execution") for i in range(7)],
        torch.randn(7, 8, generator=generator),
        torch.ones(7, 8, dtype=torch.bool),
        "pick the seen object",
        alignment="synthetic_post_action_v1",
    )


def test_router_cache_prepares_once_and_traversal_runs_without_key_tower():
    torch.manual_seed(3)
    policy = tiny_policy().eval().requires_grad_(False)
    evidence = policy.encode(tiny_episode().decision(3, 3))
    router = LocalRouter(16, 8).eval().requires_grad_(False)
    cached = CachedScorer(router, "test-frozen-checkpoint")
    query = policy.routing_query(evidence, 10)
    cached.set_query(query)
    with pytest.raises(RuntimeError, match="prepare"):
        cached(evidence.archive.roots)
    cached.prepare(evidence.archive.records, chunk_size=3)
    count = cached.keys_encoded
    cached.prepare(evidence.archive.records)
    assert cached.keys_encoded == count == len(evidence.archive.records)
    expected = router(query, router.candidate_input(evidence.archive.roots))
    torch.testing.assert_close(torch.tensor(cached(evidence.archive.roots)), expected)

    def fail(*args):
        raise AssertionError("Traversal may not run a key tower or transfer new keys")

    original = router.key.forward
    router.key.forward = fail
    selection = select_tree(evidence.archive.roots, 10, cached)
    assert len(selection) == 10
    router.key.forward = original
    cached.retain(evidence.archive.roots)
    assert set(cached.keys) == {r.node_id for r in evidence.archive.roots}
    assert cached.storage_bytes > 0
    # The complete immutable key input excludes read-time relative ages.
    features = router.candidate_input(evidence.archive.roots)
    assert torch.equal(features[:, -23 + 11 : -23 + 13], torch.zeros(len(features), 2))


def test_future_targets_observations_and_actions_cannot_change_query_or_prediction():
    torch.manual_seed(10)
    policy = tiny_policy().eval().requires_grad_(False)
    episode = tiny_episode()
    decision = episode.decision(3, 3)
    altered = deepcopy(episode)
    altered.actions[4:] *= 1000
    for frame in altered.execution[4:]:
        frame.points *= -50
        frame.proprio += 800
        frame.previous_action *= 500
    changed = altered.decision(3, 3)
    assert not torch.equal(decision.target_chunk, changed.target_chunk)
    first, second = policy.encode(decision), policy.encode(changed)
    torch.testing.assert_close(policy.routing_query(first, 12), policy.routing_query(second, 12))
    router = LocalRouter(16, 8).eval().requires_grad_(False)

    def select(evidence):
        scorer = CachedScorer(router, "fixed")
        scorer.prepare(evidence.archive.records)
        scorer.set_query(policy.routing_query(evidence, 12))
        return select_tree(evidence.archive.roots, 12, scorer).records

    a, b = select(first), select(second)
    assert [r.node_id for r in a] == [r.node_id for r in b]
    noise = torch.randn(3, 8)
    torch.testing.assert_close(policy.predict(first, a, noise=noise), policy.predict(second, b, noise=noise))
    assert all(r.time_max < decision.current.time for r in first.archive.leaves if r.source == "execution")


def test_no_history_and_all_invalid_current_points_still_have_finite_policy_output():
    policy = tiny_policy().eval().requires_grad_(False)
    decision = tiny_episode().decision(0, 3)
    current = replace(
        decision.current,
        valid=torch.zeros_like(decision.current.valid),
        points=torch.full_like(decision.current.points, torch.nan),
    )
    decision = replace(decision, current=current, conditioning=[], execution_prefix=[])
    evidence = policy.encode(decision)
    assert not evidence.current_records and not evidence.archive.leaves
    assert torch.isfinite(policy.predict(evidence, [], noise=torch.zeros(3, 8))).all()


@pytest.mark.parametrize("candidate_count", [1, 2])
def test_gain_labels_share_one_flow_draw_and_match_direct_losses(candidate_count):
    torch.manual_seed(12)
    teacher = tiny_policy().eval().requires_grad_(False)
    compact = deepcopy(teacher).eval().requires_grad_(False)
    # Distinct compact adaptation makes gains nontrivial without changing encoders.
    with torch.no_grad():
        compact.adapter.content.weight.add_(0.02 * torch.randn_like(compact.adapter.content.weight))
    decision = tiny_episode().decision(3, 3)
    evidence = teacher.encode(decision)
    before = evidence.archive.roots
    candidates = [r for r in before if r.children][:candidate_count]
    after = [split_frontier(before, r) for r in candidates]
    captures = []
    handles = [
        model.decoder.register_forward_pre_hook(
            lambda module, inputs: captures.append((inputs[0].detach().clone(), inputs[1].detach().clone()))
        )
        for model in (teacher, compact)
    ]
    noise, time = torch.randn(3, 8), torch.tensor(0.37)
    gains, draw = paired_gains(
        teacher,
        compact,
        evidence,
        before,
        after,
        decision.target_chunk,
        decision.target_valid,
        noise=noise,
        time=time,
    )
    for handle in handles:
        handle.remove()
    assert len(captures) == 3  # teacher, compact before, one batch of candidate afters
    for x, u in captures:
        if x.ndim == 3:
            for candidate in x:
                torch.testing.assert_close(candidate, draw["noisy_action"])
        else:
            torch.testing.assert_close(x, draw["noisy_action"])
        assert torch.all(u == time)
    reference = teacher(None, draw["noisy_action"], time, evidence=evidence)
    base = masked_mse(
        compact(None, draw["noisy_action"], time, evidence=evidence, records=before),
        reference,
        decision.target_valid,
    )
    for gain, records in zip(gains, after):
        new = masked_mse(
            compact(None, draw["noisy_action"], time, evidence=evidence, records=records),
            reference,
            decision.target_valid,
        )
        torch.testing.assert_close(gain, (base - new) / (len(records) - len(before)), atol=2e-7, rtol=2e-5)
    with pytest.raises(ValueError, match="positive token"):
        paired_gains(
            teacher,
            compact,
            evidence,
            before,
            [before],
            decision.target_chunk,
            decision.target_valid,
            noise=noise,
            time=time,
        )


def test_exact_tiny_search_enumerates_and_refuses_to_claim_capped_optimum():
    leaves = [
        leaf_record(
            str(i),
            torch.tensor([float(i), 1.0]),
            torch.tensor([float(i), 0.0, 0.0]),
            float(i),
            "execution",
            "visual",
        )
        for i in range(4)
    ]
    roots = [local_tree(leaves)]
    frontiers = exhaustive_tree_frontiers(roots, 3)
    flat = exhaustive_flat_contexts(leaves, 3)
    assert len(frontiers) == 2
    assert len(flat) == 6  # every two selected leaves plus the other-two residual
    for context in frontiers + flat:
        validate_frontier(context, leaves, 3)
    assert len(exhaustive_flat_contexts(leaves, 4)) == 1
    assert len(exhaustive_tree_frontiers(roots, 4)) == 1
    with pytest.raises(SearchSpaceExceeded, match="No oracle result"):
        exhaustive_flat_contexts(leaves, 3, max_states=2)
    with pytest.raises(SearchSpaceExceeded):
        exhaustive_tree_frontiers(roots, 3, max_states=2)


def test_memory_interventions_preserve_current_conditioning_and_exact_mass():
    torch.manual_seed(56)
    policy = tiny_policy().eval().requires_grad_(False)
    evidence = policy.encode(tiny_episode().decision(3, 3))
    for mode in ("erase_execution", "permute_execution", "erase_conditioning"):
        altered = intervene_evidence(evidence, mode, seed=4)
        assert altered.current is evidence.current
        assert altered.current_records is evidence.current_records
        assert altered.text is evidence.text and altered.clocks is evidence.clocks
        validate_frontier(altered.archive.roots, altered.archive.leaves, 100)
        if mode != "erase_conditioning":
            a = [r for r in evidence.archive.leaves if r.source == "conditioning"]
            b = [r for r in altered.archive.leaves if r.source == "conditioning"]
            assert [r.node_id for r in a] == [r.node_id for r in b]
            for left, right in zip(a, b):
                torch.testing.assert_close(left.content, right.content)
        if mode.startswith("erase"):
            removed = mode.removeprefix("erase_")
            assert not any(r.source == removed for r in altered.archive.leaves)
        else:
            a = [r for r in evidence.archive.leaves if r.source == "execution"]
            b = [r for r in altered.archive.leaves if r.source == "execution"]
            assert len(a) == len(b)
            assert not {r.node_id for r in a}.intersection(r.node_id for r in b)
            torch.testing.assert_close(
                torch.stack([r.content for r in a]).sum(0), torch.stack([r.content for r in b]).sum(0)
            )
            assert sorted(r.time_anchor for r in a) == sorted(r.time_anchor for r in b)
            repeated = intervene_evidence(evidence, mode, seed=4)
            assert [r.node_id for r in repeated.archive.leaves] == [r.node_id for r in altered.archive.leaves]


def test_routing_diagnostic_uses_all_evidence_and_explicit_tiny_cap():
    policy = tiny_policy().eval().requires_grad_(False)
    decision = tiny_episode().decision(1, 3)
    evidence = policy.encode(decision)
    result = routing_diagnostic(
        policy,
        deepcopy(policy),
        decision,
        budget=len(evidence.archive.roots),
        max_leaves=24,
        max_states=64,
        seed=3,
    )
    assert result["families"]["tree"]["exact"] and result["families"]["flat"]["exact"]
    assert result["compact_full_disagreement"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError, match="never silently shortened"):
        routing_diagnostic(policy, deepcopy(policy), decision, budget=8, max_leaves=2)
