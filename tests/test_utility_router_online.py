"""CPU synthetic online/cached packet parity, coverage and causal boundaries."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from stmem.evaluation.utility_router import LiveIndexedArchive, UtilityOnlineRouter, evaluation_settings
from stmem.memory.hierarchy import Archive
from stmem.memory.records import leaf_record
from stmem.memory.utility_routing import build_scorer, candidate_features, query_features, route
from stmem.models.batched_policy import records_bank
from stmem.models.policy import Evidence
from stmem.training.packed_context import _take


def evidence(width=8):
    archive = Archive()
    for group in ("conditioning", "execution"):
        records = [leaf_record(f"{group}-{i:03d}", torch.full((width,), i / 20),
                               torch.tensor([i / 20, i % 3, .2]), i / 20, group, "visual") for i in range(24)]
        archive.add_block(records, group)
    current = [leaf_record("current", torch.ones(width), torch.zeros(3), 10., "execution", "visual")]
    return Evidence(archive, current, torch.ones(width), None, {"conditioning": 2., "execution": 10.})


@pytest.mark.parametrize("reader", ["TREE", "FLAT"])
def test_online_hydration_and_scoring_match_cached_columns(reader):
    ev = evidence()
    clocks = torch.tensor([2., 10.])
    live = LiveIndexedArchive(ev.archive, 8, clocks)
    model = build_scorer(8, 16).eval().requires_grad_(False)
    policy = SimpleNamespace(config={"model": {"width": 8}}, device=torch.device("cpu"))
    current = records_bank(policy, [ev.current_records], clocks[None])

    def score(context, packets):
        selected_records = live.selected_records(context)
        reference = records_bank(policy, [selected_records], clocks[None])
        cached = live.bank(context)
        torch.testing.assert_close(reference.content, cached.content)
        torch.testing.assert_close(reference.metadata, cached.metadata)
        torch.testing.assert_close(reference.log_count, cached.log_count)
        q = query_features(current, ev.text, reference, clocks, width=8, budget=16, packet_cost=packets[0]["cost"])
        q_cached = query_features(current, ev.text, cached, clocks, width=8, budget=16, packet_cost=packets[0]["cost"])
        torch.testing.assert_close(q, q_cached)
        c = candidate_features(_take(live.columns, np.asarray([p["region"] for p in packets])),
                               width=8, cost=packets[0]["cost"], budget=16)
        return model(q, c).detach().numpy()

    final, _ = route(live.archive, reader, 16, score, seed=23)
    online = UtilityOnlineRouter(model, reader=reader, budget=16,
                                 protocol={"budget": 16, "candidate_limit": 8, "packet_max_cost": 8})
    result = online.select(ev, seed=23)
    assert [r.node_id for r in result.records] == [r.node_id for r in live.selected_records(final)]
    covered = [name for r in result.records for name in r.leaf_ids]
    assert len(result.records) == 16 and len(covered) == len(set(covered)) == 48
    assert set(covered) == {r.node_id for r in ev.archive.leaves}
    assert result.stats["utility_index_rows_scanned"] > 0


@pytest.mark.parametrize("reader", ["TREE", "FLAT"])
def test_empty_and_saturated_history_do_not_fabricate_or_drop_evidence(reader):
    ev = evidence()
    ev.archive.clear()
    model = build_scorer(8, 16).eval().requires_grad_(False)
    online = UtilityOnlineRouter(model, reader=reader, budget=16,
                                 protocol={"budget": 16, "candidate_limit": 8, "packet_max_cost": 8})
    assert online.select(ev, seed=5).records == []
    ev.archive.add_block([leaf_record("single", torch.ones(8), torch.zeros(3), 1., "execution", "visual")])
    assert [r.node_id for r in online.select(ev, seed=5).records] == ["single"]
    ev.clocks["execution"] = 1.
    with pytest.raises(ValueError, match="Current/future"):
        online.select(ev, seed=5)


def test_packet_controls_and_budget_override_are_explicit():
    checkpoint = {"schema": "utility_router_checkpoint_v1", "utility_protocol": {"budget": 128}}
    assert evaluation_settings(checkpoint)["controller"] == "packet"
    assert evaluation_settings(checkpoint, controller="local")["comparison"] == "controller"
    assert evaluation_settings(checkpoint, selection="random")["selection"] == "random"
    with pytest.raises(ValueError, match="budget"):
        evaluation_settings(checkpoint, budget=64)
    with pytest.raises(ValueError, match="expert-utility"):
        evaluation_settings({"stage": "router"}, selection="random")
