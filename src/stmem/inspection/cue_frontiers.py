"""Deterministic, exact-coverage frontiers for offline cue-resolution probes.

Preferred/protected regions are diagnostic annotations, never router or policy
features. This module changes only which already-available tree records are read;
it does not change cached content, introduce future leaves, or drop evidence.
"""

from __future__ import annotations

import heapq
from numbers import Integral

import numpy as np


def _rows(values, size, name):
    result = []
    for value in values:
        if not isinstance(value, Integral) or not 0 <= int(value) < size:
            raise ValueError(f"{name} must contain valid integer hierarchy rows")
        result.append(int(value))
    return result


def _active_tree(index, roots):
    """Validate only the causal forest reachable from the requested roots."""
    children = np.asarray(index["children"])
    if children.ndim != 2 or children.shape[1] != 2 or children.dtype.kind not in "iu":
        raise ValueError("Hierarchy children must be an integer [N,2] array")
    size = len(children)
    roots = _rows(roots, size, "Roots")
    seen, leaves, postorder, parents = set(), [], [], {}
    pending = [(row, False) for row in reversed(roots)]
    while pending:
        row, finish = pending.pop()
        if finish:
            postorder.append(row)
            continue
        if row in seen:
            raise ValueError("Causal roots overlap, or the active hierarchy has a cycle/shared child")
        seen.add(row)
        left, right = map(int, children[row])
        if left == right == -1:
            leaves.append(row)
            postorder.append(row)
            continue
        if not (0 <= left < size and 0 <= right < size):
            raise ValueError("Each active parent must have exactly two valid children")
        if left == right:
            raise ValueError("An active parent cannot repeat a child")
        pending.extend(((row, True), (right, False), (left, False)))
        parents[left] = parents[right] = row
    counts = {}
    for row in postorder:
        left, right = map(int, children[row])
        counts[row] = 1 if left < 0 else counts[left] + counts[right]
        if "count" in index and index["count"][row] != counts[row]:
            raise ValueError("Stored count differs from exact active descendant-leaf count")
    return children, roots, seen, leaves, postorder, parents, counts


def frontier_leaf_rows(index, roots):
    """Return the unique fine rows available through these causal roots."""
    return np.asarray(sorted(_active_tree(index, roots)[3]), dtype=np.int64)


def assert_frontier_coverage(index, roots, frontier):
    """Reject overlap, missing evidence, and any leaf outside the causal roots."""
    _, _, active, expected, _, _, _ = _active_tree(index, roots)
    frontier = _rows(frontier, len(index["children"]), "Frontier")
    if not set(frontier).issubset(active):
        raise ValueError("Frontier contains a node outside the causal archive")
    actual = frontier_leaf_rows(index, frontier)
    if not np.array_equal(actual, np.asarray(sorted(expected), dtype=np.int64)):
        raise ValueError("Frontier must cover every causal leaf exactly once")


def prioritized_frontier(index, roots, budget, preferred_leaves=(), protected_nodes=()):
    """Allocate detail to an annotated cue while retaining exact causal coverage.

    Preferred rows must be available fine leaves. Among splittable nodes with
    preferred descendants, split the highest preferred fraction first, then the
    highest preferred count, then the largest region. Once no preferred branch
    remains, fill capacity by largest descendant count. Stable node IDs and row
    numbers break ties. This is a spatial diagnostic heuristic, not gain search.

    A protected node may be read as a summary but may never be split. A selected
    ancestor can also summarize it. Preferred leaves strictly inside protected
    nodes are conflicting requests. Return exactly min(budget, available leaves)
    records; raise if roots or protected regions make that count impossible.
    """
    if not isinstance(budget, Integral) or budget < 0:
        raise ValueError("Budget must be a nonnegative integer")
    children, roots, active, leaves, postorder, parents, counts = _active_tree(index, roots)
    preferred = set(_rows(preferred_leaves, len(children), "Preferred leaves"))
    protected = set(_rows(protected_nodes, len(children), "Protected nodes"))
    if not preferred.issubset(set(leaves)):
        raise ValueError("Preferred rows must be active fine leaves")
    if not protected.issubset(active):
        raise ValueError("Protected nodes must belong to the causal archive")
    for leaf in preferred:
        row = parents.get(leaf)
        while row is not None:
            if row in protected:
                raise ValueError("A preferred leaf is inside a protected coarse node")
            row = parents.get(row)
    target = min(int(budget), len(leaves))
    if len(roots) > target:
        raise ValueError("Budget cannot cover all causal roots")
    preference_counts = {}
    for row in postorder:
        left, right = map(int, children[row])
        preference_counts[row] = (int(row in preferred) if left < 0 else
                                  preference_counts[left] + preference_counts[right])
    names = index.get("node_id")
    frontier, heap = set(roots), []

    def enqueue(row):
        if children[row, 0] < 0 or row in protected:
            return
        preferred_count, count = preference_counts[row], counts[row]
        name = str(names[row]) if names is not None else str(row)
        priority = (not bool(preferred_count), -preferred_count / count,
                    -preferred_count, -count, name, row)
        heapq.heappush(heap, priority)

    for row in roots:
        enqueue(row)
    while len(frontier) < target and heap:
        row = heapq.heappop(heap)[-1]
        frontier.remove(row)
        for child in map(int, children[row]):
            frontier.add(child)
            enqueue(child)
    if len(frontier) != target:
        raise ValueError("Protected nodes prevent filling the requested equal token budget")
    result = np.asarray(sorted(frontier), dtype=np.int64)
    assert_frontier_coverage(index, roots, result)
    return result
