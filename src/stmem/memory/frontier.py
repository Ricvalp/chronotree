"""Budget-filling frontier reads with fixed-query, once-per-node scores."""

from __future__ import annotations

import heapq
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import torch

from .records import MemoryRecord, record_bytes

ScoreFunction = Callable[[Sequence[MemoryRecord]], Sequence[float] | torch.Tensor]


@dataclass
class Selection:
    records: list[MemoryRecord]
    stats: dict[str, int | float] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


def scores_to_list(score_fn: ScoreFunction, records: Sequence[MemoryRecord]) -> list[float]:
    if not records:
        return []
    scores = score_fn(records)
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().float().cpu().reshape(-1).tolist()
    scores = [float(x) for x in scores]
    if len(scores) != len(records) or not all(math.isfinite(x) for x in scores):
        raise ValueError("Scorer must return one finite scalar per candidate")
    return scores


def split_frontier(frontier: Sequence[MemoryRecord], node: MemoryRecord | str) -> list[MemoryRecord]:
    node_id = node if isinstance(node, str) else node.node_id
    matches = [r for r in frontier if r.node_id == node_id]
    if len(matches) != 1 or len(matches[0].children) != 2:
        raise ValueError("Move must split one active binary parent")
    return [child for r in frontier for child in (r.children if r.node_id == node_id else (r,))]


def select_tree(roots: Iterable[MemoryRecord], budget: int, score_fn: ScoreFunction) -> Selection:
    roots = list(roots)
    if budget < len(roots):
        raise ValueError(
            f"History budget {budget} cannot cover all {len(roots)} source/modality/prefix roots"
        )
    if budget < 0:
        raise ValueError("Budget must be nonnegative")
    frontier = {r.node_id: r for r in roots}
    if len(frontier) != len(roots):
        raise ValueError("Duplicate root IDs")
    heap: list[tuple[float, str]] = []
    scored, splits = 0, 0

    def enqueue(records: Sequence[MemoryRecord]) -> None:
        nonlocal scored
        candidates = [r for r in records if r.children]
        values = scores_to_list(score_fn, candidates)
        scored += len(candidates)
        for record, value in zip(candidates, values):
            heapq.heappush(heap, (-value, record.node_id))

    if len(frontier) < budget:
        enqueue(roots)
    while len(frontier) < budget and heap:
        _, node_id = heapq.heappop(heap)
        node = frontier.pop(node_id)
        for child in node.children:
            frontier[child.node_id] = child
        splits += 1
        if len(frontier) < budget:
            enqueue(node.children)
    result = list(frontier.values())
    return Selection(
        result,
        {
            "nodes_scored": scored,
            "leaves_scored": 0,
            "splits": splits,
            "active_tokens": len(result),
            "roots": len(roots),
            "records_fetched": len(result),
            "selected_payload_bytes": record_bytes(result),
        },
    )


def select_uniform(roots: Iterable[MemoryRecord], budget: int) -> Selection:
    """Refine greatest leaf coverage first; IDs deterministically resolve ties."""
    return select_tree(roots, budget, lambda records: [float(r.count) for r in records])


def select_random(roots: Iterable[MemoryRecord], budget: int, rng: random.Random) -> Selection:
    return select_tree(roots, budget, lambda records: [rng.random() for _ in records])


def validate_frontier(records: Iterable[MemoryRecord], leaves: Iterable[MemoryRecord], budget: int) -> None:
    records, leaves = list(records), list(leaves)
    if len(records) > budget:
        raise AssertionError(f"{len(records)} tokens exceed budget {budget}")
    expected = Counter(r.node_id for r in leaves)
    actual = Counter(i for record in records for i in record.leaf_ids)
    if any(n != 1 for n in expected.values()) or actual != expected:
        raise AssertionError("Context must cover each available fine leaf exactly once")
    if any(r.count != len(r.leaf_ids) for r in records):
        raise AssertionError("Record mass is not its descendant leaf count")
