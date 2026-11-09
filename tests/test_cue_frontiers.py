"""Synthetic topology diagnostics: causal availability and exact coverage only."""

import numpy as np
import pytest

from stmem.inspection.cue_frontiers import (
    assert_frontier_coverage, frontier_leaf_rows, prioritized_frontier,
)


@pytest.fixture
def tree():
    # Eight leaves, four pairs, two halves, one root. Rows15..17 are a
    # different (future/prefix) branch which is present in storage, not active.
    children = np.full((18, 2), -1, dtype=np.int64)
    children[8:15] = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]
    children[17] = [15, 16]
    return {"children": children, "count": np.asarray([1]*8 + [2]*4 + [4, 4, 8, 1, 1, 2]),
            "node_id": np.asarray([f"n{i:02d}" for i in range(18)])}


def test_preference_changes_resolution_without_changing_budget_or_evidence(tree):
    first = prioritized_frontier(tree, [14], 4, preferred_leaves=[0])
    other = prioritized_frontier(tree, [14], 4, preferred_leaves=[7])
    assert len(first) == len(other) == 4
    assert 0 in first and 7 in other
    assert not np.array_equal(first, other)
    for selected in (first, other):
        assert_frontier_coverage(tree, [14], selected)
        assert np.array_equal(frontier_leaf_rows(tree, selected), np.arange(8))
        assert not np.any(selected >= 15)


def test_annotation_order_and_repeated_calls_cannot_change_selection(tree):
    first = prioritized_frontier(tree, [14], 5, preferred_leaves=[0, 1, 6])
    second = prioritized_frontier(tree, [14], 5, preferred_leaves=[6, 1, 0])
    assert np.array_equal(first, second)


def test_uniform_fill_and_full_capacity_cover_every_leaf(tree):
    assert set(prioritized_frontier(tree, [14], 4)) == {8, 9, 10, 11}
    assert np.array_equal(prioritized_frontier(tree, [14], 8, [0]), np.arange(8))
    assert np.array_equal(prioritized_frontier(tree, [14], 100, [0]), np.arange(8))


def test_protected_cue_stays_coarse_while_other_region_fills_budget(tree):
    selected = prioritized_frontier(tree, [14], 5, protected_nodes=[12])
    assert set(selected) == {12, 4, 5, 6, 7}
    assert_frontier_coverage(tree, [14], selected)
    with pytest.raises(ValueError, match="prevent filling"):
        prioritized_frontier(tree, [14], 6, protected_nodes=[12])
    with pytest.raises(ValueError, match="inside a protected"):
        prioritized_frontier(tree, [14], 5, preferred_leaves=[0], protected_nodes=[12])


def test_separate_sources_are_never_dropped_for_small_budget(tree):
    with pytest.raises(ValueError, match="all causal roots"):
        prioritized_frontier(tree, [12, 13], 1)
    selected = prioritized_frontier(tree, [12, 13], 4, [0])
    assert len(selected) == 4
    assert_frontier_coverage(tree, [12, 13], selected)


@pytest.mark.parametrize("argument,value", [
    ("preferred_leaves", [15]), ("preferred_leaves", [8]), ("protected_nodes", [17]),
])
def test_future_or_nonterminal_preferences_are_rejected(tree, argument, value):
    with pytest.raises(ValueError):
        prioritized_frontier(tree, [14], 4, **{argument: value})


@pytest.mark.parametrize("frontier", [[14, 0], [12], [12, 17], [0, 0]])
def test_overlap_omission_and_future_frontiers_fail(tree, frontier):
    with pytest.raises(ValueError):
        assert_frontier_coverage(tree, [14], frontier)


def test_malformed_active_topology_and_stale_counts_fail(tree):
    invalid = {**tree, "children": tree["children"].copy()}
    invalid["children"][8] = [0, -1]
    with pytest.raises(ValueError, match="two valid children"):
        prioritized_frontier(invalid, [14], 4)
    invalid["children"][8] = [14, 1]
    with pytest.raises(ValueError, match="cycle"):
        prioritized_frontier(invalid, [14], 4)
    invalid = {**tree, "count": tree["count"].copy()}
    invalid["count"][14] = 9
    with pytest.raises(ValueError, match="Stored count"):
        prioritized_frontier(invalid, [14], 4)


def test_empty_archive_and_invalid_arguments(tree):
    assert prioritized_frontier(tree, [], 128).shape == (0,)
    with pytest.raises(ValueError, match="nonnegative integer"):
        prioritized_frontier(tree, [14], 1.5)
    with pytest.raises(ValueError, match="valid integer"):
        prioritized_frontier(tree, [-1], 4)
    with pytest.raises(ValueError, match="overlap"):
        prioritized_frontier(tree, [14, 14], 4)
