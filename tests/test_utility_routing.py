"""Synthetic CPU contracts for matched packet routing; no benchmark evidence."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from test_refinement_candidates import forest
from stmem.inspection.refinement_candidates import IndexedArchive
from stmem.memory.utility_routing import (
    build_scorer, feature_spec, propose_packets, query_features, route,
)
from stmem.models.batched_policy import TokenBank


def bank(values, width=4):
    contents = torch.as_tensor(values, dtype=torch.float32).reshape(1, -1, width)
    n = contents.shape[1]
    return TokenBank(contents, torch.zeros(1, n, 23), torch.ones(1, n, dtype=torch.bool), torch.zeros(1, n))


@pytest.mark.parametrize("reader", ["TREE", "FLAT"])
@pytest.mark.parametrize("mode", ["packet", "local"])
def test_negative_scores_fill_budget_without_omitting_evidence(reader, mode):
    index, roots, leaves = forest((64, 48), geometry=True)
    archive = IndexedArchive(index, roots, leaves)
    seen = []
    def scores(context, entries):
        assert archive.validate(context) == context
        assert len({entry["cost"] for entry in entries}) == 1
        for entry in entries:
            assert archive.validate(entry["final"])["token_count"] == context["token_count"] + entry["cost"]
            assert entry["final"]["coverage_hash"] == archive.coverage
        seen.append(context["token_count"])
        return -np.arange(1, len(entries) + 1)
    selected, stats = route(archive, reader, 37, scores, mode=mode, seed=17)
    assert selected["token_count"] == 37
    assert stats["actual_count"] == 37 and stats["candidates_scored"] > 0
    assert stats["index_rows_scanned"] >= stats["candidates_scored"]
    assert sum(row["cost"] for row in stats["steps"]) == 37 - len(roots)
    assert seen == sorted(seen)


@pytest.mark.parametrize("reader", ["TREE", "FLAT"])
def test_packet_pool_is_gain_blind_reproducible_equal_cost_and_bounded(reader):
    index, roots, leaves = forest((128, 128), geometry=True)
    archive = IndexedArchive(index, roots, leaves)
    context = archive.ordinary(reader, 30, seed=8)
    a, _ = propose_packets(archive, context, 37, seed=93, candidate_limit=4)
    b, _ = propose_packets(archive, deepcopy(context), 37, seed=93, candidate_limit=4)
    assert len(a) <= 4
    assert [entry["candidate_id"] for entry in a] == [entry["candidate_id"] for entry in b]
    assert len({entry["candidate_id"] for entry in a}) == len(a)
    assert len({entry["cost"] for entry in a}) == 1
    assert a[0]["cost"] <= 7
    for entry in a:
        for step, prefix in entry["prefixes"].items():
            assert archive.validate(prefix)["token_count"] == 30 + step


def test_query_depends_on_observed_resolution_and_budget_without_privileged_inputs():
    scene = bank([[1, 2, 3, 4], [2, 3, 4, 5]])
    text = torch.arange(4.)
    history = bank([[0, 1, 2, 3], [4, 5, 6, 7]])
    q = query_features(scene, text, history, [4, 8], width=4, budget=16, packet_cost=3)
    assert q.shape == (feature_spec(4)["query_dim"],)
    torch.testing.assert_close(q[:4], scene.content.mean(1)[0])
    torch.testing.assert_close(q[4:8], text)
    torch.testing.assert_close(q[8:12], history.content.mean(1)[0])
    assert not torch.equal(q, query_features(scene, text, history, [4, 8], width=4, budget=32, packet_cost=3))
    assert not torch.equal(q, query_features(scene, text, bank([[0, 1, 2, 3]]), [4, 8],
                                           width=4, budget=16, packet_cost=3))
    with pytest.raises(TypeError):
        query_features(scene, text, history, [4, 8], width=4, budget=16, packet_cost=3,
                       expert_action=torch.zeros(16, 8))
    with pytest.raises(ValueError, match="exceed"):
        query_features(scene, text, history, [4, 8], width=4, budget=3, packet_cost=3)


def test_matching_scorers_have_identical_capacity_and_finite_gradients():
    torch.manual_seed(9)
    first = build_scorer(4, 8)
    torch.manual_seed(9)
    packet = build_scorer(4, 8)
    assert first.state_dict().keys() == packet.state_dict().keys()
    assert sum(p.numel() for p in first.parameters()) == sum(p.numel() for p in packet.parameters())
    spec = feature_spec(4)
    q, c = torch.randn(5, spec["query_dim"]), torch.randn(5, spec["candidate_dim"])
    torch.testing.assert_close(first(q, c), packet(q, c))
    loss = (first(q, c) - torch.linspace(-1, 1, 5)).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in first.parameters())


def test_empty_history_and_budget_violation_are_explicit():
    index, roots, leaves = forest((16, 16))
    archive = IndexedArchive(index, [], [])
    for reader in ("TREE", "FLAT"):
        selected, stats = route(archive, reader, 8, lambda *args: pytest.fail("No score needed"))
        assert selected["rows"] == [] and stats["actual_count"] == 0
    archive = IndexedArchive(index, roots, leaves)
    with pytest.raises(ValueError, match="roots"):
        route(archive, "TREE", 1, lambda *args: [])
