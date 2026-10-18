"""Balanced local trees and an append-only binary-counter temporal forest."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import torch

from .records import MemoryRecord, aggregate, record_bytes


def stable_id(prefix: str, parts: Iterable[str]) -> str:
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]
    return f"{prefix}:{digest}"


def local_tree(leaves: Iterable[MemoryRecord]) -> MemoryRecord | None:
    leaves = list(leaves)
    if not leaves:
        return None
    if len({r.group for r in leaves}) != 1:
        raise ValueError("A local tree must have one source and modality")
    # One metadata transfer for a whole block, not a synchronization per sort key.
    coordinates = torch.stack([r.anchor.detach() for r in leaves]).cpu().tolist()
    by_id = {r.node_id: xyz for r, xyz in zip(leaves, coordinates)}

    def build(members: list[MemoryRecord]) -> MemoryRecord:
        if len(members) == 1:
            return members[0]
        if members[0].modality == "visual":
            extent = [
                max(by_id[r.node_id][axis] for r in members) - min(by_id[r.node_id][axis] for r in members)
                for axis in range(3)
            ]
            axis = max(range(3), key=lambda axis: extent[axis])
            members.sort(key=lambda r: (by_id[r.node_id][axis], r.time_anchor, r.node_id))
        else:
            members.sort(key=lambda r: (r.time_anchor, r.node_id))
        middle = len(members) // 2
        left, right = build(members[:middle]), build(members[middle:])
        return aggregate(
            (left, right), stable_id("local", (left.node_id, right.node_id)), children=True, scale=0
        )

    return build(leaves)


def walk_records(roots: Iterable[MemoryRecord]) -> list[MemoryRecord]:
    result, seen = [], set()
    stack = list(reversed(list(roots)))
    while stack:
        record = stack.pop()
        if record.node_id in seen:
            continue
        seen.add(record.node_id)
        result.append(record)
        stack.extend(reversed(record.children))
    return result


@dataclass
class Archive:
    """All completed blocks, plus a replaceable observed prefix per source.

    Structural state is CPU/Python. Tensor contents may remain differentiable on
    an accelerator during teacher training; streaming passes detached CPU leaves.
    """

    forests: dict[tuple[str, str], list[MemoryRecord]] = field(default_factory=dict)
    pending: dict[str, tuple[MemoryRecord, ...]] = field(default_factory=dict)
    completed_blocks: dict[str, tuple[MemoryRecord, ...]] = field(default_factory=dict)

    def add_block(self, leaves: Iterable[MemoryRecord], block_id: str | None = None) -> None:
        leaves = tuple(leaves)
        if not leaves:
            return
        if len({r.source for r in leaves}) != 1:
            raise ValueError("A block cannot cross conditioning/execution boundary")
        if not all(r.completed for r in leaves):
            raise ValueError("Completed block contains temporary prefix evidence")
        source = leaves[0].source
        block_id = block_id or stable_id("block", (r.node_id for r in leaves))
        if block_id in self.completed_blocks:
            raise ValueError(f"Block {block_id} already archived")
        self.pending.pop(source, None)
        self.completed_blocks[block_id] = leaves
        groups: dict[tuple[str, str], list[MemoryRecord]] = defaultdict(list)
        for leaf in leaves:
            if not leaf.is_leaf:
                raise ValueError("add_block requires fine leaves")
            groups[leaf.group].append(leaf)
        for group, members in sorted(groups.items()):
            root = local_tree(members)
            assert root is not None
            forest = self.forests.setdefault(group, [])
            if forest and root.time_min < forest[-1].time_min:
                raise ValueError("Blocks must be appended in source-local chronological order")
            while forest and forest[-1].scale == root.scale:
                previous = forest.pop()
                root = aggregate(
                    (previous, root),
                    stable_id("temporal", (previous.node_id, root.node_id)),
                    children=True,
                    scale=root.scale + 1,
                )
            forest.append(root)

    def set_pending(self, source: str, leaves: Iterable[MemoryRecord]) -> None:
        leaves = tuple(leaves)
        if any(r.source != source or r.completed for r in leaves):
            raise ValueError("Pending leaves must be incomplete and belong to the given source")
        self.pending[source] = leaves

    @property
    def roots(self) -> list[MemoryRecord]:
        roots = [r for key in sorted(self.forests) for r in self.forests[key]]
        for source in sorted(self.pending):
            groups: dict[tuple[str, str], list[MemoryRecord]] = defaultdict(list)
            for leaf in self.pending[source]:
                groups[leaf.group].append(leaf)
            for group in sorted(groups):
                root = local_tree(groups[group])
                if root is not None:
                    roots.append(root)
        return roots

    @property
    def leaves(self) -> list[MemoryRecord]:
        return [r for leaves in self.completed_blocks.values() for r in leaves] + [
            r for source in sorted(self.pending) for r in self.pending[source]
        ]

    @property
    def records(self) -> list[MemoryRecord]:
        return walk_records(self.roots)

    @property
    def storage_bytes(self) -> int:
        return record_bytes(self.records)

    def clear(self) -> None:
        self.forests.clear()
        self.pending.clear()
        self.completed_blocks.clear()


def maximum_root_count(
    max_execution_frames: int, max_conditioning_frames: int, block_size: int, modalities: int = 2
) -> int:
    """Worst count over *any* prefix, including the pending block, not just final time."""
    if block_size < 1 or min(max_execution_frames, max_conditioning_frames) < 0:
        raise ValueError("Nonnegative lengths and positive block size required")
    execution = max(
        ((n // block_size).bit_count() + int(n % block_size != 0) for n in range(max_execution_frames + 1)),
        default=0,
    )
    # Conditioning is complete: its final partial block is a completed block.
    conditioning = max(
        (((n + block_size - 1) // block_size).bit_count() for n in range(max_conditioning_frames + 1)),
        default=0,
    )
    return modalities * (execution + conditioning)
