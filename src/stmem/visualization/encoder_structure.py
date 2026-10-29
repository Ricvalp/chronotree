"""Exact, bounded-detail inspection of the causal encoder and its binary archive.

The graph comes from a real encoder forward. This module never reads an episode's
expert actions or target masks, and never serializes learned feature vectors.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor

from stmem.data.canonical import Episode, Frame
from stmem.memory.hierarchy import walk_records
from stmem.memory.records import MemoryRecord
from stmem.memory.streaming import build_archive
from stmem.models.spacetime_encoder import BlockEncoder


def _values(value: Tensor) -> list:
    value = value.detach().cpu()
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError("Encoder structure contains nonfinite numeric values")
    return value.tolist()


def _node(record: MemoryRecord) -> dict[str, Any]:
    return {
        "id": record.node_id,
        "source": record.source,
        "modality": record.modality,
        "kind": "leaf" if record.is_leaf else ("temporal" if record.scale else "local"),
        "children": [child.node_id for child in record.children],
        "leaf_ids": list(record.leaf_ids),
        "count": record.count,
        "anchor": _values(record.anchor),
        "bounds_min": _values(record.bounds_min),
        "bounds_max": _values(record.bounds_max),
        "time_min": record.time_min,
        "time_max": record.time_max,
        "time_anchor": record.time_anchor,
        "depth": record.depth,
        "scale": record.scale,
        "completed": record.completed,
        "occupancy": record.occupancy,
    }


def _frame(frame: Frame, quota: int) -> dict[str, Any]:
    return {
        "event_id": frame.event_id,
        "source_index": frame.source_index,
        "time": float(frame.time),
        "input_points": len(frame.points),
        "valid_points": int((frame.valid & torch.isfinite(frame.points).all(-1)).sum()),
        "quota": quota,
    }


class _Collector:
    """Detach only at the inspection boundary, outside the encoder's math."""

    def __init__(self) -> None:
        self.trace: dict[str, Any] | None = None

    def __call__(self, values: dict[str, Any]) -> None:
        self.trace = {
            key: _values(value) if isinstance(value, Tensor) else list(value) for key, value in values.items()
        }

    def details(
        self, block_id: str, frames: Sequence[Frame], records: Sequence[MemoryRecord]
    ) -> dict[str, Any]:
        if self.trace is None:
            raise RuntimeError("The visual encoder did not emit its requested structure trace")
        trace = self.trace
        centers = []
        for index, valid in enumerate(trace["valid"]):
            point_index = trace["center_point_indices"][index]
            centers.append(
                {
                    "index": index,
                    "leaf_id": f"{block_id}:visual:{index}" if valid else None,
                    "point_index": point_index if point_index >= 0 else None,
                    "frame_index": trace["center_frame_indices"][index],
                    "time": trace["center_times"][index],
                    "anchor": trace["anchor"][index],
                    "candidate_valid": trace["center_candidate_valid"][index],
                    "valid": valid,
                    "neighbor_indices": trace["neighbor_indices"][index],
                    "neighbor_mask": trace["neighbor_mask"][index],
                    "weights": trace["weights"][index],
                }
            )
        events = []
        for index, record in enumerate(record for record in records if record.modality == "event"):
            frame = frames[index]
            events.append(
                {
                    "leaf_id": record.node_id,
                    "frame_index": index,
                    "event_id": frame.event_id,
                    "time": float(frame.time),
                    "anchor": _values(record.anchor),
                    "proprio": _values(frame.proprio),
                    "action_available": bool(frame.action_available) and frame.source == "execution",
                }
            )
        return {
            "points": trace["points"],
            "point_frame_indices": trace["point_frame_indices"],
            "point_times": trace["point_times"],
            "centers": centers,
            "events": events,
        }


def _block(
    block_id: str,
    source: str,
    frames: Sequence[Frame],
    quotas: Sequence[int],
    records: Sequence[MemoryRecord],
    completed: bool,
    collector: _Collector | None,
    *,
    current: bool = False,
) -> dict[str, Any]:
    return {
        "id": block_id,
        "source": source,
        "kind": "current" if current else "history",
        "completed": completed,
        "detailed": collector is not None,
        "frames": [_frame(frame, quota) for frame, quota in zip(frames, quotas)],
        "leaf_ids": [record.node_id for record in records],
        "visual_leaf_ids": [record.node_id for record in records if record.modality == "visual"],
        "event_leaf_ids": [record.node_id for record in records if record.modality == "event"],
        "local_roots": {"visual": None, "event": None},
        "details": collector.details(block_id, frames, records) if collector is not None else None,
    }


def build_structure(
    episode: Episode,
    encoder: BlockEncoder,
    decision_index: int,
    *,
    detail_blocks: int = 2,
) -> dict[str, Any]:
    """Inspect all causal history, detailing only its last N blocks per source.

    Current observation details are always included, even when detail_blocks=0.
    The caller owns encoder placement/checkpoint loading; no device, parameter,
    model mode, or RNG state is changed here. Training gradients are not recorded.
    The standard streaming builder owns block identities, pending semantics, and
    local/temporal tree construction, so the inspector cannot invent a hierarchy.
    """
    if isinstance(decision_index, bool) or not isinstance(decision_index, int):
        raise ValueError("decision_index must be an integer execution-frame index")
    if not 0 <= decision_index < len(episode.execution):
        raise ValueError("decision_index must select an existing execution observation")
    if isinstance(detail_blocks, bool) or not isinstance(detail_blocks, int) or detail_blocks < 0:
        raise ValueError("detail_blocks must be a nonnegative integer")
    conditioning, execution = (
        (episode.conditioning, episode.execution[:decision_index]) if encoder.history_enabled else ([], [])
    )
    current = episode.execution[decision_index]
    if current.source != "execution":
        raise ValueError("The current observation must belong to the execution source")
    if any(frame.event_id == current.event_id for frame in execution):
        raise ValueError("The current observation also occurs in the historical execution prefix")
    if execution and current.time <= execution[-1].time:
        raise ValueError("The current source-local clock must follow the historical execution prefix")
    blocks: list[dict[str, Any]] = []
    source_lengths = {"conditioning": len(conditioning), "execution": len(execution)}

    class ObservedEncoder:
        block_size = encoder.block_size

        def __init__(self) -> None:
            self.counts = {"conditioning": 0, "execution": 0}

        def encode_frames(self, frames, source, block_id, *, completed):
            block_index = self.counts[source]
            self.counts[source] += 1
            block_count = (source_lengths[source] + self.block_size - 1) // self.block_size
            collector = _Collector() if block_index >= block_count - detail_blocks else None
            records = encoder.encode_frames(frames, source, block_id, completed=completed, trace=collector)
            quotas = [
                encoder.visual_tokens // self.block_size
                + int(index < encoder.visual_tokens % self.block_size)
                for index in range(len(frames))
            ]
            blocks.append(_block(block_id, source, frames, quotas, records, completed, collector))
            return records

    with torch.no_grad():
        archive = build_archive(conditioning, execution, ObservedEncoder())
        roots = archive.roots
        records = walk_records(roots)
        nodes = [_node(record) for record in records]
        owner = {leaf_id: block["id"] for block in blocks for leaf_id in block["leaf_ids"]}
        by_block = {block["id"]: block for block in blocks}
        # A block-local root is the highest actual tree node owned by one block.
        # Preorder visits parents first; temporal parents have multiple owners.
        for node in nodes:
            owners = {owner[leaf_id] for leaf_id in node["leaf_ids"]}
            node["block_id"] = next(iter(owners)) if len(owners) == 1 else None
            if node["block_id"] is not None:
                block = by_block[node["block_id"]]
                if block["local_roots"][node["modality"]] is None:
                    block["local_roots"][node["modality"]] = node["id"]
        current_collector = _Collector()
        current_records = encoder.current(current, trace=current_collector)
        current_id = f"current:{current.event_id}"
        blocks.append(
            _block(
                current_id,
                "execution",
                [current],
                [encoder.current_tokens],
                current_records,
                True,
                current_collector,
                current=True,
            )
        )
        visual = encoder.visual
        return {
            "schema_version": 1,
            "episode": {
                "task": episode.task,
                "episode_id": episode.episode_id,
                "decision_index": decision_index,
                "conditioning_frames": len(conditioning),
                "execution_prefix_frames": len(execution),
            },
            "config": {
                "variant": encoder.variant,
                "block_size": encoder.block_size,
                "visual_tokens": encoder.visual_tokens,
                "current_tokens": encoder.current_tokens,
                "proprioception_mode": encoder.proprioception_mode,
                "history_enabled": encoder.history_enabled,
                "condition_on_previous_commands": (
                    encoder.event.condition_on_previous_commands if encoder.event is not None else False
                ),
                "history_events_enabled": encoder.history_events_enabled,
                "tau_x": float(visual.tau_x.detach()),
                "tau_t": float(visual.tau_t.detach()),
                "tau_x_units": "metres squared",
                "tau_t_units": "recorded frames squared",
                "spatial_radius": visual.spatial_radius,
                "temporal_radius": visual.temporal_radius,
                "neighbor_cap": visual.neighbor_cap,
            },
            "blocks": blocks,
            "archive": {
                "roots": [record.node_id for record in roots],
                "leaf_ids": [record.node_id for record in archive.leaves],
                "nodes": nodes,
            },
            "current": {
                "block_id": current_id,
                "leaf_ids": [record.node_id for record in current_records],
                "nodes": [_node(record) for record in current_records],
                "proprio": _values(current.proprio),
                "action_available": bool(current.action_available),
                "raw_state_diagnostic_only": encoder.proprioception_mode == "none",
            },
            "provenance": {
                "trace_source": "Actual SpacetimeEncoder.forward neighborhood indices, masks, and weights",
                "detail_blocks_per_source": detail_blocks,
                "point_index_space": "Concatenated valid finite points, canonically ordered within each frame",
                "gaussian": "exp(-squared_xyz_distance/tau_x - squared_recorded_frame_distance/tau_t)",
                "normalization": "Softmax over allowed capped neighbors; masked slots have zero weight",
                "causality": (
                    "All conditioning and execution strictly before current; current stays outside archive"
                    if encoder.history_enabled
                    else "Current observation only; both historical banks are omitted"
                ),
                "scope": (
                    "Full causal binary hierarchy; point details only for the last requested blocks per source and current"
                    if encoder.history_enabled
                    else "Current visual pooling only; no history, clock, proprioception or previous-command tokens"
                ),
                "features": "Learned features, expert targets, and future execution observations are omitted",
                "events": (
                    "Event tokens disabled; visual quotas are unchanged"
                    if not encoder.history_events_enabled
                    else "Event content and anchors follow the configured proprioception/command switches; raw state fields shown here are diagnostic values, not proof of model conditioning"
                ),
            },
        }
