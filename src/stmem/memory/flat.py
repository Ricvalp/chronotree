"""Exact flat scan with one honest residual per nonempty source/modality group."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .frontier import ScoreFunction, Selection, scores_to_list
from .hierarchy import stable_id
from .records import MemoryRecord, aggregate, record_bytes


@dataclass
class FlatState:
    leaves: tuple[MemoryRecord, ...]
    selected_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.leaves = tuple(self.leaves)
        ids = {r.node_id for r in self.leaves}
        if len(ids) != len(self.leaves) or not self.selected_ids <= ids:
            raise ValueError("Flat state requires unique leaves and valid selected IDs")
        if any(not r.is_leaf or r.count != 1 for r in self.leaves):
            raise ValueError("Flat retrieval consumes the exact unit-mass archived leaves")

    @property
    def unselected(self) -> list[MemoryRecord]:
        return [r for r in self.leaves if r.node_id not in self.selected_ids]

    @property
    def records(self) -> list[MemoryRecord]:
        selected = [r for r in self.leaves if r.node_id in self.selected_ids]
        groups: dict[tuple[str, str], list[MemoryRecord]] = defaultdict(list)
        for record in self.unselected:
            groups[record.group].append(record)
        for group in sorted(groups):
            remaining = groups[group]
            # A singleton residual and a selected singleton are the same evidence.
            # Preserve its actual leaf identity; normalize_singletons handles state.
            if len(remaining) == 1:
                selected.append(remaining[0])
            else:
                selected.append(
                    aggregate(
                        remaining, stable_id("residual", (r.node_id for r in remaining)), disconnected=True
                    )
                )
        return selected

    @property
    def token_count(self) -> int:
        return len(self.selected_ids) + len({r.group for r in self.unselected})

    def move_cost(self, leaf: MemoryRecord | str) -> int:
        leaf_id = leaf if isinstance(leaf, str) else leaf.node_id
        candidates = {r.node_id: r for r in self.unselected}
        if leaf_id not in candidates:
            raise ValueError("Move must select a currently unselected fine leaf")
        group = candidates[leaf_id].group
        return int(sum(r.group == group for r in candidates.values()) > 1)

    def move(self, leaf: MemoryRecord | str) -> FlatState:
        leaf_id = leaf if isinstance(leaf, str) else leaf.node_id
        self.move_cost(leaf_id)
        return FlatState(self.leaves, self.selected_ids | {leaf_id})

    def normalize_singletons(self) -> FlatState:
        """Zero-cost moves are explicit and never divided by zero for gain labels."""
        groups: dict[tuple[str, str], list[MemoryRecord]] = defaultdict(list)
        for r in self.unselected:
            groups[r.group].append(r)
        singleton_ids = {rs[0].node_id for rs in groups.values() if len(rs) == 1}
        return FlatState(self.leaves, self.selected_ids | singleton_ids)


def select_flat(leaves: Iterable[MemoryRecord], budget: int, score_fn: ScoreFunction) -> Selection:
    leaves = tuple(leaves)
    state = FlatState(leaves)
    groups = {r.group for r in leaves}
    if budget < len(groups):
        raise ValueError(f"History budget {budget} cannot cover all {len(groups)} source/modality residuals")
    if budget < 0:
        raise ValueError("Budget must be nonnegative")
    if len(leaves) <= budget:
        records = list(leaves)
        return Selection(
            records,
            {
                "nodes_scored": 0,
                "leaves_scored": 0,
                "active_tokens": len(records),
                "selected_leaves": len(records),
                "residuals": 0,
                "records_fetched": len(records),
                "selected_payload_bytes": record_bytes(records),
            },
        )
    # One batched scorer call. No repeated per-leaf neural network invocations.
    values = scores_to_list(score_fn, leaves)
    ranked = sorted(zip(leaves, values), key=lambda pair: (-pair[1], pair[0].node_id))
    remaining_counts: dict[tuple[str, str], int] = defaultdict(int)
    for leaf in leaves:
        remaining_counts[leaf.group] += 1
    selected: set[str] = set()
    token_count = len(groups)
    for leaf, _ in ranked:
        cost = int(remaining_counts[leaf.group] > 1)
        if token_count + cost <= budget:
            selected.add(leaf.node_id)
            remaining_counts[leaf.group] -= 1
            token_count += cost
    state = FlatState(leaves, selected).normalize_singletons()
    records = state.records
    assert len(records) == min(budget, len(leaves))
    return Selection(
        records,
        {
            "nodes_scored": 0,
            "leaves_scored": len(leaves),
            "active_tokens": len(records),
            "selected_leaves": len(state.selected_ids),
            "residuals": sum(bool(r.residual_leaf_ids) for r in records),
            "records_fetched": len(records),
            "selected_payload_bytes": record_bytes(records),
        },
    )
