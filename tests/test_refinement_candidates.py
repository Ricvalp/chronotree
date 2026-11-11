"""Synthetic topology tests of causal coverage and equal-cost utility panels."""
from collections import Counter

import numpy as np
import pytest

from stmem.inspection.refinement_candidates import IndexedArchive, training_routes


def forest(sizes=(1024, 512), *, geometry=False):
    rows, roots, leaves = [], [], []
    def build(size, group):
        if size == 1:
            row = len(rows)
            rows.append(dict(node_id=f"leaf_{group}_{row:06d}", children=(-1, -1), count=1,
                             source=group, modality="visual"))
            leaves.append(row)
        else:
            left = build(size // 2, group)
            right = build(size - size // 2, group)
            row = len(rows)
            rows.append(dict(node_id=f"parent_{group}_{row:06d}", children=(left, right), count=size,
                             source=group, modality="visual"))
        return row
    for i, size in enumerate(sizes):
        roots.append(build(size, "conditioning" if i == 0 else "execution"))
    index = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    if geometry:
        index["anchor"] = np.asarray([[i % 19, i % 7, i % 3] for i in range(len(rows))], float)
        index["time_anchor"] = np.arange(len(rows), dtype=float)
    return index, roots, leaves


def empty_causal_forest(empty_storage):
    # At the first decision, the sidecar can already contain future nodes even
    # though none are yet causal. Also cover a physically empty indexed cache.
    index, _, _ = forest((16,), geometry=True)
    if empty_storage:
        index = {key: values[:0] for key, values in index.items()}
    return index, [], []


@pytest.mark.parametrize("empty_storage", [False, True])
def test_empty_flat_archive_has_typed_selection_and_exact_zero_coverage(empty_storage):
    index, roots, leaves = empty_causal_forest(empty_storage)
    archive = IndexedArchive(index, roots, leaves)
    context = archive.coarse("FLAT")
    assert archive.validate(context) == context
    assert context["rows"] == [] and archive.records_count(context) == 0
    mask, codes, groups = archive.flat_arrays(context)
    assert mask.shape == codes.shape == (0,) and mask.dtype == np.bool_
    assert np.issubdtype(codes.dtype, np.integer) and groups == []
    # This is the ordinary FLAT token-accounting operation used by the audit
    # and training loader; an empty float mask cannot be inverted or indexed.
    assert int(mask.sum()) + len(set(codes[~mask])) == 0
    assert archive.leaves[mask].shape == (0,)
    description = archive.describe(context)
    assert description["residuals"] == [] and description["source_leaf_counts"] == {}
    assert description["maximum_record_mass"] == description["maximum_depth"] == 0
    assert description["leaf_fraction"] == 0
    assert archive.coverage_hash(context) == archive.coverage_hash(archive.coarse("TREE"))
    panel = archive.diagnostic_panel(seed=37)
    assert panel["leaf_count"] == panel["root_count"] == 0
    assert panel["contexts"] == {} and panel["families"] == []
    assert len(panel["eligibility"]) == 6
    assert all(not row["eligible"] and row["reason"] for row in panel["eligibility"])
    with pytest.raises(ValueError, match="noncausal"):
        archive.context("FLAT", [0])


@pytest.mark.parametrize("empty_storage", [False, True])
@pytest.mark.parametrize("reader", ["TREE", "FLAT"])
def test_empty_history_routes_are_saturated_without_moves_or_fabricated_records(empty_storage, reader):
    index, roots, leaves = empty_causal_forest(empty_storage)
    archive = IndexedArchive(index, roots, leaves)
    initial = archive.coarse(reader)
    routes = training_routes(index, roots, leaves, reader, target_budget=128, seed=37)
    assert routes["eligible"] and routes["saturated"]
    assert routes["initial_count"] == routes["target_count"] == 0
    for route in (routes["stage"], routes["packet"]):
        assert route["contexts"] == {0: initial}
        assert route["moves"] == route["packets"] == []
    assert archive.positive_moves(initial) == []
    assert archive.feasible_regions(initial, 1) == []
    assert np.issubdtype(archive.node_group_codes.dtype, np.integer)


@pytest.mark.parametrize("empty_storage", [False, True])
def test_empty_flat_hydration_keeps_zero_history_tokens(empty_storage):
    from stmem.training.packed_context import _empty_columns, columns_bank, flat_residual_columns

    index, roots, leaves = empty_causal_forest(empty_storage)
    archive = IndexedArchive(index, roots, leaves)
    context = archive.coarse("FLAT")
    columns = flat_residual_columns(_empty_columns(4), *archive.flat_arrays(context))
    bank = columns_bank(columns, clocks=(0., 0.), width=4)
    assert bank.content.shape == (1, 0, 4)
    assert bank.metadata.shape == (1, 0, 23)
    assert bank.valid.shape == bank.log_count.shape == (1, 0)
    assert not bank.valid.any()
    assert len(columns["count"]) == context["token_count"] == 0


@pytest.mark.parametrize("reader", ["TREE", "FLAT"])
def test_full_routes_cover_each_leaf_once_at_every_integer_count(reader):
    index, roots, leaves = forest((160, 140), geometry=True)
    pair = training_routes(index, roots, leaves, reader, 73, seed=19)
    assert pair["eligible"]
    assert set(pair["stage"]["contexts"]) == set(pair["packet"]["contexts"]) == set(range(2, 74))
    archive = IndexedArchive(index, roots, leaves)
    for route in (pair["stage"], pair["packet"]):
        for count, context in route["contexts"].items():
            assert archive.validate(context)["token_count"] == count
            assert archive.coverage_hash(context) == archive.coverage
        assert sum(packet["cost"] for packet in route["packets"]) == 71
    assert pair["packet"]["packets"][-1]["cost"] <= 8
    assert pair["stage"]["contexts"][73]["context_id"] != pair["packet"]["contexts"][73]["context_id"]


def test_unavailable_future_and_overlapping_prefixes_are_rejected():
    index, roots, leaves = forest((32, 32))
    archive = IndexedArchive(index, roots[:1], leaves[:32])
    with pytest.raises(ValueError, match="noncausal"):
        archive.context("TREE", roots)
    with pytest.raises(ValueError, match="overlap"):
        archive.context("TREE", [roots[0], leaves[0]])
    with pytest.raises(ValueError, match="exact causal"):
        IndexedArchive(index, roots[:1], leaves)
    with pytest.raises(ValueError, match="cycle/shared"):
        IndexedArchive(index, [roots[0], leaves[0]])


def test_tree_packet_exact_prefixes_and_protected_pairing():
    index, roots, leaves = forest()
    archive = IndexedArchive(index, roots, leaves)
    panel = archive.diagnostic_panel(46)
    families = [family for family in panel["families"] if family["reader"] == "TREE" and family["panel"] == "packet"]
    assert len(families) == 2 and all(f["paired"] for f in families)
    coarse, rich = families
    assert coarse["background"] == 56 and rich["background"] == 120
    c = {candidate["candidate_id"]: candidate for candidate in coarse["candidates"]}
    r = {candidate["candidate_id"]: candidate for candidate in rich["candidates"]}
    assert c.keys() == r.keys() and len(c) >= 2
    for key in c:
        assert c[key]["moves"] == r[key]["moves"]
        assert c[key]["support_hash"] == r[key]["support_hash"]
        for family, candidate in ((coarse, c[key]), (rich, r[key])):
            for step, context_id in candidate["prefixes"].items():
                context = panel["contexts"][context_id]
                assert archive.records_count(context) == family["background"] + int(step)
            for control in family["controls"].values():
                assert panel["contexts"][control]["token_count"] == family["final_budget"]


def test_flat_packet_retains_residuals_and_identical_support_across_backgrounds():
    index, roots, leaves = forest()
    archive = IndexedArchive(index, roots, leaves)
    panel = archive.diagnostic_panel(11)
    families = [family for family in panel["families"] if family["reader"] == "FLAT" and family["panel"] == "packet"]
    assert len(families) == 2 and all(f["paired"] for f in families)
    for coarse, rich in zip(families[0]["candidates"], families[1]["candidates"]):
        assert coarse["candidate_id"] == rich["candidate_id"]
        assert coarse["moves"] == rich["moves"]
    for family in families:
        base = panel["contexts"][family["base"]]
        for candidate in family["candidates"]:
            for row in candidate["moves"]:
                assert row not in base["rows"]
            for step, context_id in candidate["prefixes"].items():
                context = panel["contexts"][context_id]
                selected, codes, groups = archive.flat_arrays(context)
                assert selected.sum() + sum(np.any((codes == i) & ~selected) for i in range(len(groups))) == context["token_count"]
                assert context["token_count"] == family["background"] + int(step)


def test_flat_singletons_normalize_without_counting_zero_cost_promotions():
    index, roots, leaves = forest((2, 1))
    archive = IndexedArchive(index, roots, leaves)
    initial = archive.coarse("FLAT")
    assert initial["token_count"] == 2
    assert leaves[-1] in initial["rows"]
    final = archive.move(initial, leaves[0])
    assert final["token_count"] == 3 and len(final["rows"]) == 3
    with pytest.raises(ValueError, match="zero actual token cost"):
        archive.move(final, leaves[1])
    with pytest.raises(ValueError, match="residual mass"):
        archive.packet(initial, roots[0], 2)


def test_audit_survives_packet_ineligibility_and_has_unique_positive_moves():
    index, roots, leaves = forest((128,))
    panel = IndexedArchive(index, roots, leaves).diagnostic_panel(98)
    audit = [family for family in panel["families"] if family["panel"] == "immediate_audit"]
    assert len(audit) == 2
    for family in audit:
        assert family["background"] == 127
        assert family["candidates"]
        assert len({c["candidate_id"] for c in family["candidates"]}) == len(family["candidates"])
        for candidate in family["candidates"]:
            assert panel["contexts"][candidate["prefixes"]["1"]]["token_count"] == 128
    assert any(not item["eligible"] for item in panel["eligibility"] if item["panel"] == "packet")


def test_determinism_and_missing_capacity_reported_without_replacing_state():
    index, roots, leaves = forest((35, 10))
    archive = IndexedArchive(index, roots, leaves)
    first = archive.diagnostic_panel(12)
    assert first == archive.diagnostic_panel(12)
    assert not first["contexts"] and not first["families"]
    assert len(first["eligibility"]) == 6
    assert all(not item["eligible"] for item in first["eligibility"])
    route = training_routes(index, roots, leaves, "TREE", 128, 0)
    assert route["eligible"] and route["saturated"] and route["target_count"] == 45
    assert set(route["stage"]["contexts"]) == set(route["packet"]["contexts"])


def test_packet_sampler_balances_eligible_groups_and_reduces_final_cost():
    index, roots, leaves = forest((70, 70))
    archive = IndexedArchive(index, roots, leaves)
    counts = Counter()
    for seed in range(100):
        route = archive.route("TREE", 10, seed, mode="packet")
        counts[tuple(route["packets"][0]["group"])] += 1
    assert len(counts) == 2 and all(30 < count < 70 for count in counts.values())
    route = archive.route("TREE", 140, 12, mode="packet")
    assert route["target_count"] == 140
    assert any(packet["cost"] < packet["requested_cost"] for packet in route["packets"])


def test_corrupt_stored_context_identity_and_mass_rejected():
    index, roots, leaves = forest((32,))
    archive = IndexedArchive(index, roots, leaves)
    context = archive.coarse("TREE")
    with pytest.raises(ValueError, match="token_count"):
        archive.validate(dict(context, token_count=2))
    index["count"][roots[0]] += 1
    with pytest.raises(ValueError, match="descendant-leaf count"):
        IndexedArchive(index, roots, leaves)


def test_protected_background_fails_explicitly_when_capacity_is_exhausted():
    index, roots, leaves = forest((128,))
    archive = IndexedArchive(index, roots, leaves)
    for reader in ("TREE", "FLAT"):
        with pytest.raises(ValueError, match="Protected"):
            archive._extend(archive.coarse(reader), 56, protected=roots)


def test_flat_hydration_matches_existing_residual_implementation_exactly():
    import torch
    from stmem.memory.flat import FlatState
    from stmem.memory.records import leaf_record
    from stmem.training.packed_context import (
        SCALAR_FIELDS, STRING_FIELDS, TENSOR_FIELDS, columns_bank, flat_residual_columns,
    )
    index, roots, leaves = forest((19, 13), geometry=True)
    archive = IndexedArchive(index, roots, leaves)
    records = [leaf_record(index["node_id"][row], torch.tensor([row, row + 1, -row], dtype=torch.float32),
                           torch.as_tensor(index["anchor"][row]).float(), float(row),
                           index["source"][row], index["modality"][row]) for row in leaves]
    columns = {key: torch.stack([getattr(record, key) for record in records]) for key in TENSOR_FIELDS}
    columns.update({key: np.asarray([getattr(record, key) for record in records])
                    for key in (*SCALAR_FIELDS, *STRING_FIELDS)})
    route = archive.route("FLAT", 31, 92, mode="packet")
    for context in route["contexts"].values():
        mask, codes, groups = archive.flat_arrays(context)
        actual = flat_residual_columns(columns, mask, codes, groups)
        expected = FlatState(tuple(records), {index["node_id"][row] for row in context["rows"]}).records
        assert list(actual["node_id"]) == [record.node_id for record in expected]
        for key in TENSOR_FIELDS:
            torch.testing.assert_close(actual[key], torch.stack([getattr(record, key) for record in expected]),
                                       atol=0, rtol=0)
        for key in (*SCALAR_FIELDS, *STRING_FIELDS):
            assert list(actual[key]) == [getattr(record, key) for record in expected]
        bank = columns_bank(actual, (100., 100.), 3)
        torch.testing.assert_close(bank.log_count[0], torch.tensor([np.log(record.count) for record in expected]).float())
        manifest = archive.describe(context)
        expected_residuals = [record.node_id for record in expected if record.residual_leaf_ids]
        assert [record["node_id"] for record in manifest["residuals"]] == expected_residuals


def test_flat_regions_survive_tree_minimum_background_ineligibility():
    index, roots, leaves = forest(tuple([16] * 65))
    archive = IndexedArchive(index, roots, leaves)
    panel = archive.diagnostic_panel(4)
    assert any(f["reader"] == "FLAT" and f["panel"] == "packet" and f["background"] == 56
               for f in panel["families"])
    assert not any(f["reader"] == "TREE" and f["panel"] == "packet" and f["background"] == 56
                   for f in panel["families"])
