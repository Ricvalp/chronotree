"""Typed evidence records. Aggregation acts on latent content, before read adapters."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import torch
from torch import Tensor

SOURCES = ("conditioning", "execution")
MODALITIES = ("visual", "event")


@dataclass(frozen=True)
class MemoryRecord:
    node_id: str
    content_sum: Tensor
    count: int
    anchor: Tensor
    bounds_min: Tensor
    bounds_max: Tensor
    time_min: float
    time_max: float
    source: str
    modality: str
    max_content: Tensor
    children: tuple[MemoryRecord, ...] = ()
    depth: int = 0
    scale: int = 0
    completed: bool = True
    disconnected: bool = False
    valid: bool = True
    time_anchor: float = 0.0
    occupancy: float = 1.0
    # Only residuals carry explicit coverage; tree coverage is recursive, not O(N log N).
    residual_leaf_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source not in SOURCES or self.modality not in MODALITIES:
            raise ValueError(f"Unknown source/modality: {self.source}/{self.modality}")
        if self.count < 1 or not self.valid:
            raise ValueError("Archive records must represent at least one valid unit-mass leaf")
        if self.content_sum.dtype != torch.float32 or self.max_content.dtype != torch.float32:
            raise TypeError("Aggregation sums and maxima must remain FP32")
        if len(self.children) not in (0, 2):
            raise ValueError("Hierarchical records have zero or two children")
        if self.time_min > self.time_max:
            raise ValueError("Invalid observed-time interval")

    @property
    def content(self) -> Tensor:
        return self.content_sum / self.count

    @property
    def is_leaf(self) -> bool:
        return not self.children and not self.residual_leaf_ids

    @property
    def leaf_ids(self) -> tuple[str, ...]:
        if self.residual_leaf_ids:
            return self.residual_leaf_ids
        if not self.children:
            return (self.node_id,)
        return tuple(x for child in self.children for x in child.leaf_ids)

    @property
    def group(self) -> tuple[str, str]:
        return self.source, self.modality

    def tensors(self) -> tuple[Tensor, ...]:
        return (self.content_sum, self.max_content, self.anchor, self.bounds_min, self.bounds_max)

    def to(self, device: torch.device | str, *, detach: bool = False) -> MemoryRecord:
        """Move one record; selected-record transfers do not recursively fetch descendants."""

        def move(x: Tensor) -> Tensor:
            return (x.detach() if detach else x).to(device=device, dtype=torch.float32)

        return replace(
            self,
            content_sum=move(self.content_sum),
            max_content=move(self.max_content),
            anchor=move(self.anchor),
            bounds_min=move(self.bounds_min),
            bounds_max=move(self.bounds_max),
        )


def leaf_record(
    node_id: str,
    content: Tensor,
    anchor: Tensor,
    time: float,
    source: str,
    modality: str,
    *,
    bounds_min: Tensor | None = None,
    bounds_max: Tensor | None = None,
    time_min: float | None = None,
    time_max: float | None = None,
    completed: bool = True,
    occupancy: float = 1.0,
) -> MemoryRecord:
    content, anchor = content.float(), anchor.float()
    return MemoryRecord(
        node_id,
        content,
        1,
        anchor,
        anchor if bounds_min is None else bounds_min.float(),
        anchor if bounds_max is None else bounds_max.float(),
        time if time_min is None else time_min,
        time if time_max is None else time_max,
        source,
        modality,
        content,
        completed=completed,
        time_anchor=time,
        occupancy=occupancy,
    )


def aggregate(
    records: Iterable[MemoryRecord],
    node_id: str,
    *,
    children: bool = False,
    scale: int | None = None,
    disconnected: bool = False,
) -> MemoryRecord:
    records = tuple(records)
    if not records:
        raise ValueError("Cannot aggregate an empty set")
    if len({r.group for r in records}) != 1:
        raise ValueError("Aggregation may not cross sources or modalities")
    if children and len(records) != 2:
        raise ValueError("A binary parent requires exactly two children")
    count = sum(r.count for r in records)
    # Float casts explicitly preserve sensitive accumulation under autocast.
    total = torch.stack([r.content_sum.float() for r in records]).sum(0, dtype=torch.float32)
    anchor = torch.stack([r.anchor.float() * r.count for r in records]).sum(0, dtype=torch.float32) / count
    return MemoryRecord(
        node_id=node_id,
        content_sum=total,
        count=count,
        anchor=anchor,
        bounds_min=torch.stack([r.bounds_min for r in records]).amin(0),
        bounds_max=torch.stack([r.bounds_max for r in records]).amax(0),
        time_min=min(r.time_min for r in records),
        time_max=max(r.time_max for r in records),
        source=records[0].source,
        modality=records[0].modality,
        max_content=torch.stack([r.max_content for r in records]).amax(0),
        children=records if children else (),
        depth=1 + max(r.depth for r in records),
        scale=max(r.scale for r in records) if scale is None else scale,
        completed=all(r.completed for r in records),
        disconnected=disconnected,
        time_anchor=sum(r.time_anchor * r.count for r in records) / count,
        occupancy=sum(r.occupancy * r.count for r in records) / count,
        residual_leaf_ids=() if children else tuple(i for r in records for i in r.leaf_ids),
    )


def record_bytes(records: Iterable[MemoryRecord]) -> int:
    """Tensor storage plus explicit numeric/ID payload; Python object overhead is separate."""
    seen: set[str] = set()
    total = 0
    for record in records:
        if record.node_id in seen:
            continue
        seen.add(record.node_id)
        total += sum(t.numel() * t.element_size() for t in record.tensors())
        total += 88 + len(record.node_id.encode()) + sum(len(i.encode()) for i in record.residual_leaf_ids)
        total += sum(len(c.node_id.encode()) for c in record.children)
    return total
