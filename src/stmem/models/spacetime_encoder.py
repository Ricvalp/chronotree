"""Local Gaussian space-time evidence, without a global register bottleneck.

The conceptual Gaussian pooling ancestry is ICIL's SpacetimeEventEncoder at
3af009b458c74aa66e695e35b02f9f8801f52138; this is an independent Torch implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from stmem.data.center_sampling import (
    center_sampling_signature,
    prepare_center_indices,
    resolve_center_sampling,
)
from stmem.memory.hierarchy import stable_id
from stmem.memory.records import MemoryRecord, leaf_record
from .batched_spacetime import canonical_nearest, forward_batched
from .event_encoder import EventEncoder


VisualTrace = Callable[[dict[str, Any]], None]


@dataclass
class EncodedLeaves:
    content: Tensor
    valid: Tensor
    anchor: Tensor
    bounds_min: Tensor
    bounds_max: Tensor
    time_anchor: Tensor
    time_min: Tensor
    time_max: Tensor
    occupancy: Tensor


def _lexicographic_indices(points: Tensor) -> Tensor:
    order = torch.arange(len(points), device=points.device)
    for axis in reversed(range(points.shape[-1])):
        order = order[torch.argsort(points[order, axis], stable=True)]
    return order


def _lexicographic(points: Tensor) -> Tensor:
    """Canonical point ordering makes center and nearest-neighbor ties invariant."""
    return points[_lexicographic_indices(points)]


def cached_frame_centers(frame: Any, count: int, settings: dict, signature: str | None = None) -> Tensor:
    """Reuse prepared geometry, or compute it once for a CPU diagnostic frame.

    Real prepared datasets validate and supply this cache before training. The
    fallback serves synthetic inputs and individual acquired/diagnostic frames;
    it never runs inside the batched CUDA encoder. Cache identity includes the
    selector implementation and settings, independently of the center quota.
    """
    signature = signature or center_sampling_signature(settings)
    if getattr(frame, "center_sampling_signature", None) != signature:
        frame.center_indices = {}
        frame.center_sampling_signature = signature
    if count not in frame.center_indices:
        frame.center_indices.update(prepare_center_indices(frame.points, frame.valid, [count], settings))
    return frame.center_indices[count]


def _fps(points: Tensor, count: int) -> Tensor:
    """Deterministic modest FPS on canonically ordered points; never duplicates."""
    count = min(count, len(points))
    if not count:
        return torch.empty(0, device=points.device, dtype=torch.long)
    indices = torch.empty(count, device=points.device, dtype=torch.long)
    distance = torch.full((len(points),), float("inf"), device=points.device)
    next_index = torch.zeros((), device=points.device, dtype=torch.long)
    chosen = torch.zeros(len(points), device=points.device, dtype=torch.bool)
    for i in range(count):
        indices[i] = next_index
        chosen[next_index] = True
        distance = torch.minimum(distance, (points[:, :3] - points[next_index, :3]).square().sum(-1))
        distance = distance.masked_fill(chosen, -1.0)
        next_index = distance.argmax()
    return indices


class SpacetimeEncoder(nn.Module):
    def __init__(
        self,
        width: int = 256,
        *,
        tau_x: float = 0.01,
        tau_t: float = 4.0,
        spatial_radius: float = 0.3,
        temporal_radius: float = 8.0,
        neighbor_cap: int = 128,
        center_chunk: int = 16,
        learnable_bandwidth: bool = True,
        framewise: bool = False,
        center_sampling: dict | None = None,
    ):
        super().__init__()
        if (
            min(width, neighbor_cap, center_chunk) < 1
            or min(tau_x, tau_t, spatial_radius, temporal_radius) <= 0
        ):
            raise ValueError("Encoder dimensions, locality limits and Gaussian denominators must be positive")
        self.width, self.neighbor_cap, self.center_chunk = width, neighbor_cap, center_chunk
        self.spatial_radius, self.temporal_radius, self.framewise = spatial_radius, temporal_radius, framewise
        # None preserves the explicit global-FPS reference used by low-level
        # invariants. Configured selectors consume CPU-prepared point indices.
        self.center_sampling = None if center_sampling is None else resolve_center_sampling(center_sampling)
        self.center_sampling_signature = (
            None if self.center_sampling is None else center_sampling_signature(self.center_sampling)
        )
        self.min_bandwidth = 1e-6

        # tau_x has metre^2 units; tau_t has recorded-frame^2 units.
        def inverse_softplus(value: float) -> float:
            return math.log(math.expm1(max(value - self.min_bandwidth, self.min_bandwidth)))

        self.raw_tau_x = nn.Parameter(
            torch.tensor(inverse_softplus(tau_x)), requires_grad=learnable_bandwidth
        )
        self.raw_tau_t = nn.Parameter(
            torch.tensor(inverse_softplus(tau_t)), requires_grad=learnable_bandwidth
        )
        self.point_mlp = nn.Sequential(nn.Linear(6, width), nn.SiLU(), nn.Linear(width, width))
        self.edge_mlp = nn.Sequential(nn.Linear(width + 4, width), nn.SiLU(), nn.Linear(width, width))

    @property
    def tau_x(self) -> Tensor:
        return F.softplus(self.raw_tau_x.float()) + self.min_bandwidth

    @property
    def tau_t(self) -> Tensor:
        return F.softplus(self.raw_tau_t.float()) + self.min_bandwidth

    def forward_batched(
        self,
        points: Tensor,
        valid: Tensor,
        times: Tensor,
        centers_per_frame: int | Sequence[int],
        *,
        center_indices: Tensor | None = None,
    ) -> EncodedLeaves:
        """Vectorized independent local blocks; masked leaves retain fixed slots."""
        return forward_batched(self, points, valid, times, centers_per_frame, center_indices=center_indices)

    def forward(
        self,
        points: Sequence[Tensor] | Tensor,
        valid: Sequence[Tensor] | Tensor,
        times: Tensor | Sequence[float],
        centers_per_frame: int | Sequence[int],
        *,
        trace: VisualTrace | None = None,
        center_indices: Sequence[Tensor] | Tensor | None = None,
    ) -> EncodedLeaves:
        """Pool observed points; an optional observer receives the actual pooling graph.

        Trace tensors are borrowed from this forward, with no hidden features.
        Observers must detach/copy any values they retain and must not mutate them.
        The ordinary training path does not collect or transfer trace tensors.
        """
        if len(points) != len(valid) or len(points) != len(times):
            raise ValueError("Points, masks and source-local timestamps must describe the same frames")
        if isinstance(centers_per_frame, int):
            quotas = [centers_per_frame] * len(points)
        else:
            quotas = list(centers_per_frame)
        if len(quotas) != len(points) or any(x < 0 for x in quotas):
            raise ValueError("One nonnegative temporal-stratum center quota is required per frame")
        if center_indices is not None and len(center_indices) != len(points):
            raise ValueError("One prepared center-index row is required per frame")
        device = self.raw_tau_x.device
        times = torch.as_tensor(times, dtype=torch.float32, device=device)
        if not torch.isfinite(times).all() or (len(times) > 1 and (times[1:] < times[:-1]).any()):
            raise ValueError("Use finite chronological source-local timestamps")
        ordered, frame_indices, point_times = [], [], []
        anchors, center_times, center_frames, center_valid = [], [], [], []
        center_point_indices = [] if trace is not None else None
        trace_neighbors, trace_masks, trace_weights = [], [], []
        # Selection is discrete and content-independent except observed coordinates/RGB.
        for frame_index, (frame_points, frame_valid, quota) in enumerate(zip(points, valid, quotas)):
            frame_points = frame_points.to(device=device, dtype=torch.float32)
            frame_valid = frame_valid.to(device=device, dtype=torch.bool)
            if (
                frame_points.ndim != 2
                or frame_points.shape[-1] != 6
                or frame_valid.shape != frame_points.shape[:1]
            ):
                raise ValueError("Expected points [N,6] and True-for-valid mask [N]")
            frame_valid = frame_valid & torch.isfinite(frame_points).all(-1)
            observed = torch.where(frame_valid)[0]
            original_order = observed[_lexicographic_indices(frame_points[observed])]
            cloud = frame_points[original_order]
            supplied = None if center_indices is None else center_indices[frame_index]
            if supplied is None and self.center_sampling is not None:
                # Direct raw-tensor visualizations may request selection without
                # a Frame cache. This explicit scalar fallback can copy to CPU;
                # training's packed path always supplies prepared indices.
                supplied = prepare_center_indices(frame_points, frame_valid, [quota], self.center_sampling)[
                    quota
                ]
            if supplied is None:
                selected = _fps(cloud.detach(), quota)
                chosen = torch.cat([selected, selected.new_full((quota - len(selected),), -1)])
            else:
                supplied = torch.as_tensor(supplied, device=device)
                if supplied.shape != (quota,) or supplied.dtype != torch.long:
                    raise ValueError("Prepared center indices must be int64 with the exact frame quota")
                inverse = torch.full((len(frame_points),), -1, device=device, dtype=torch.long)
                inverse[original_order] = torch.arange(len(cloud), device=device)
                if len(frame_points):
                    in_range = (supplied >= 0) & (supplied < len(frame_points))
                    chosen = inverse[supplied.clamp(0, len(frame_points) - 1)]
                    chosen = torch.where(in_range, chosen, -1)
                else:
                    chosen = supplied.new_full((quota,), -1)
            occupied = chosen >= 0
            if center_point_indices is not None:
                offset = sum(len(previous) for previous in ordered)
                center_point_indices.append(torch.where(occupied, chosen + offset, -1))
            ordered.append(cloud)
            frame_indices.append(torch.full((len(cloud),), frame_index, device=device, dtype=torch.long))
            point_times.append(times[frame_index].expand(len(cloud)))
            frame_anchors = torch.zeros(quota, 3, device=device)
            frame_anchors[occupied] = cloud[chosen[occupied], :3]
            anchors.append(frame_anchors)
            center_times.append(times[frame_index].expand(quota))
            center_frames.append(torch.full((quota,), frame_index, device=device, dtype=torch.long))
            center_valid.append(occupied)
        m = sum(quotas)
        if not m:
            z = torch.zeros(0, device=device)
            result = EncodedLeaves(
                torch.zeros(0, self.width, device=device),
                z.bool(),
                torch.zeros(0, 3, device=device),
                torch.zeros(0, 3, device=device),
                torch.zeros(0, 3, device=device),
                z,
                z,
                z,
                z,
            )
            if trace is not None:
                empty_index = torch.empty(0, device=device, dtype=torch.long)
                self._emit_trace(
                    trace,
                    result,
                    quotas,
                    times,
                    torch.cat(ordered) if ordered else torch.zeros(0, 6, device=device),
                    torch.cat(frame_indices) if frame_indices else empty_index,
                    torch.cat(point_times) if point_times else z,
                    empty_index,
                    empty_index,
                    z.bool(),
                    torch.empty(0, 0, device=device, dtype=torch.long),
                    torch.empty(0, 0, device=device, dtype=torch.bool),
                    torch.empty(0, 0, device=device),
                )
            return result
        anchor, ct, cf, cv = map(torch.cat, (anchors, center_times, center_frames, center_valid))
        cloud, pt, pf = map(torch.cat, (ordered, point_times, frame_indices))
        if not len(cloud):
            z = torch.zeros(m, device=device)
            result = EncodedLeaves(
                torch.zeros(m, self.width, device=device),
                cv,
                anchor,
                anchor.clone(),
                anchor.clone(),
                ct,
                ct.clone(),
                ct.clone(),
                z,
            )
            if trace is not None:
                self._emit_trace(
                    trace,
                    result,
                    quotas,
                    times,
                    cloud,
                    pf,
                    pt,
                    torch.cat(center_point_indices),
                    cf,
                    cv,
                    torch.empty(m, 0, device=device, dtype=torch.long),
                    torch.empty(m, 0, device=device, dtype=torch.bool),
                    torch.empty(m, 0, device=device),
                )
            return result
        features = self.point_mlp(cloud)
        contents, valids, minima, maxima, tmins, tmaxs, occupancies = [], [], [], [], [], [], []
        for start in range(0, m, self.center_chunk):
            end = min(start + self.center_chunk, m)
            # Distances/reductions are explicitly FP32 even under AMP.
            dx = cloud[None, :, :3] - anchor[start:end, None, :]
            dt = pt[None, :] - ct[start:end, None]
            spatial = dx.float().square().sum(-1)
            temporal = dt.float().square()
            allowed = (spatial <= self.spatial_radius**2) & (dt.abs() <= self.temporal_radius)
            allowed = allowed & cv[start:end, None]
            if self.framewise:
                allowed = allowed & (pf[None, :] == cf[start:end, None])
            distance = spatial / self.tau_x + temporal / self.tau_t
            nearest = canonical_nearest(distance, allowed, self.neighbor_cap)
            mask = allowed.gather(1, nearest)
            row_valid = mask.any(-1)
            logits = -distance.gather(1, nearest)
            logits = logits.masked_fill(~mask, -torch.inf)
            logits = torch.where(row_valid[:, None], logits, torch.zeros_like(logits))
            weights = torch.softmax(logits.float(), dim=-1) * mask.float()
            if trace is not None:
                trace_neighbors.append(nearest)
                trace_masks.append(mask)
                trace_weights.append(weights)
            relative = torch.cat(
                [dx.gather(1, nearest[..., None].expand(-1, -1, 3)), dt.gather(1, nearest)[..., None]], dim=-1
            )
            edge = self.edge_mlp(torch.cat([features[nearest], relative], dim=-1)).float()
            contents.append((weights[..., None] * edge).sum(1, dtype=torch.float32))
            neighbors = cloud[nearest, :3]
            lo = neighbors.masked_fill(~mask[..., None], torch.inf).amin(1)
            hi = neighbors.masked_fill(~mask[..., None], -torch.inf).amax(1)
            minima.append(torch.where(row_valid[:, None], lo, torch.zeros_like(lo)))
            maxima.append(torch.where(row_valid[:, None], hi, torch.zeros_like(hi)))
            neighbor_times = pt[nearest]
            lo_t = neighbor_times.masked_fill(~mask, torch.inf).amin(1)
            hi_t = neighbor_times.masked_fill(~mask, -torch.inf).amax(1)
            tmins.append(torch.where(row_valid, lo_t, ct[start:end]))
            tmaxs.append(torch.where(row_valid, hi_t, ct[start:end]))
            occupancies.append((torch.exp(-distance.gather(1, nearest)) * mask).sum(-1))
            valids.append(row_valid)
        result = EncodedLeaves(
            torch.cat(contents),
            torch.cat(valids),
            anchor,
            torch.cat(minima),
            torch.cat(maxima),
            ct,
            torch.cat(tmins),
            torch.cat(tmaxs),
            torch.cat(occupancies),
        )
        if trace is not None:
            self._emit_trace(
                trace,
                result,
                quotas,
                times,
                cloud,
                pf,
                pt,
                torch.cat(center_point_indices),
                cf,
                cv,
                torch.cat(trace_neighbors),
                torch.cat(trace_masks),
                torch.cat(trace_weights),
            )
        return result

    @staticmethod
    def _emit_trace(
        trace: VisualTrace,
        result: EncodedLeaves,
        quotas: list[int],
        times: Tensor,
        points: Tensor,
        point_frames: Tensor,
        point_times: Tensor,
        center_points: Tensor,
        center_frames: Tensor,
        center_candidates: Tensor,
        neighbors: Tensor,
        masks: Tensor,
        weights: Tensor,
    ) -> None:
        trace(
            {
                "frame_quotas": tuple(quotas),
                "frame_times": times,
                "points": points,
                "point_frame_indices": point_frames,
                "point_times": point_times,
                "center_point_indices": center_points,
                "center_frame_indices": center_frames,
                "center_candidate_valid": center_candidates,
                "anchor": result.anchor,
                "center_times": result.time_anchor,
                "valid": result.valid,
                "neighbor_indices": neighbors,
                "neighbor_mask": masks,
                "weights": weights,
            }
        )


class BlockEncoder(nn.Module):
    """Shared current/history point machinery, plus a separate robot-event encoder."""

    def __init__(
        self,
        width: int = 256,
        visual_tokens: int = 32,
        block_size: int = 8,
        current_tokens: int = 64,
        proprio_dim: int = 15,
        action_dim: int = 8,
        variant: str = "joint",
        checkpoint_blocks: bool = False,
        condition_on_previous_joint_commands: bool = True,
        **visual_kwargs: Any,
    ):
        super().__init__()
        if min(visual_tokens, block_size, current_tokens) < 1 or visual_tokens < block_size:
            raise ValueError("At least one center candidate per historical observation is required")
        if variant not in ("joint", "framewise"):
            raise ValueError("Encoder variant must be joint or framewise")
        self.width, self.visual_tokens, self.block_size, self.current_tokens = (
            width,
            visual_tokens,
            block_size,
            current_tokens,
        )
        self.variant = variant
        self.checkpoint_blocks = checkpoint_blocks
        self.visual = SpacetimeEncoder(width, framewise=variant == "framewise", **visual_kwargs)
        self.event = EventEncoder(
            width,
            proprio_dim,
            action_dim,
            condition_on_previous_joint_commands=condition_on_previous_joint_commands,
        )

    def encode_frames(
        self,
        frames: Sequence[Any],
        source: str | None = None,
        block_id: str | None = None,
        *,
        completed: bool = True,
        trace: VisualTrace | None = None,
    ) -> list[MemoryRecord]:
        if not frames:
            return []
        if len(frames) > self.block_size:
            raise ValueError("Local encoder may not materialize an entire episode")
        source = source or frames[0].source
        if any(f.source != source for f in frames):
            raise ValueError("A local block may not cross source boundaries")
        block_id = block_id or stable_id(
            source + (":complete" if completed else ":prefix"), (f.event_id for f in frames)
        )
        quotas = [
            self.visual_tokens // self.block_size + int(i < self.visual_tokens % self.block_size)
            for i in range(len(frames))
        ]
        points, valid, times = [f.points for f in frames], [f.valid for f in frames], [f.time for f in frames]
        center_indices = None
        if self.visual.center_sampling is not None:
            center_indices = [
                cached_frame_centers(
                    f, quota, self.visual.center_sampling, self.visual.center_sampling_signature
                )
                for f, quota in zip(frames, quotas)
            ]
        if self.checkpoint_blocks and self.training and torch.is_grad_enabled():

            def pool():
                output = self.visual(points, valid, times, quotas, trace=trace, center_indices=center_indices)
                return tuple(getattr(output, name) for name in EncodedLeaves.__dataclass_fields__)

            pooled = EncodedLeaves(*checkpoint(pool, use_reentrant=False))
        else:
            pooled = self.visual(points, valid, times, quotas, trace=trace, center_indices=center_indices)
        records = self._visual_records(pooled, source, block_id, completed)
        device = self.visual.raw_tau_x.device
        state = torch.stack([f.proprio for f in frames]).to(device=device, dtype=torch.float32)
        action = torch.stack([f.previous_action for f in frames]).to(device=device, dtype=torch.float32)
        available = torch.tensor(
            [bool(f.action_available) and source == "execution" for f in frames], device=device
        )
        events = self.event(state, action, available)
        for index, (frame, content) in enumerate(zip(frames, events)):
            records.append(
                leaf_record(
                    f"{block_id}:event:{index}",
                    content,
                    state[index, 7:10],
                    frame.time,
                    source,
                    "event",
                    completed=completed,
                )
            )
        return records

    def current(self, frame: Any, *, trace: VisualTrace | None = None) -> list[MemoryRecord]:
        indices = None
        if self.visual.center_sampling is not None:
            indices = [
                cached_frame_centers(
                    frame,
                    self.current_tokens,
                    self.visual.center_sampling,
                    self.visual.center_sampling_signature,
                )
            ]
        pooled = self.visual(
            [frame.points],
            [frame.valid],
            [frame.time],
            [self.current_tokens],
            trace=trace,
            center_indices=indices,
        )
        return self._visual_records(pooled, frame.source, f"current:{frame.event_id}", True)

    @staticmethod
    def _visual_records(
        pooled: EncodedLeaves, source: str, block_id: str, completed: bool
    ) -> list[MemoryRecord]:
        valid = pooled.valid.detach().cpu().tolist()
        metadata = (
            torch.stack([pooled.time_anchor, pooled.time_min, pooled.time_max, pooled.occupancy], dim=-1)
            .detach()
            .cpu()
            .tolist()
        )
        return [
            leaf_record(
                f"{block_id}:visual:{i}",
                pooled.content[i],
                pooled.anchor[i],
                metadata[i][0],
                source,
                "visual",
                bounds_min=pooled.bounds_min[i],
                bounds_max=pooled.bounds_max[i],
                time_min=metadata[i][1],
                time_max=metadata[i][2],
                completed=completed,
                occupancy=metadata[i][3],
            )
            for i, ok in enumerate(valid)
            if ok
        ]
