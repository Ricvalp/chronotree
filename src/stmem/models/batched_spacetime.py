"""Vectorized local point-cloud pooling with explicit frame/point padding.

The batch axis contains independent local blocks, never one concatenated episode.
All geometric arithmetic remains FP32 under autocast. No tensor value is copied
to the CPU, and no Python loop scales with the number of blocks or frames.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .spacetime_encoder import EncodedLeaves, SpacetimeEncoder


def canonical_nearest(distance: Tensor, allowed: Tensor, cap: int) -> Tensor:
    """Select by (nonnegative FP32 distance, canonical point order), exactly.

    Positive IEEE float bit patterns have the same order as their values. Packing
    those 32-bit patterns and a 32-bit point index into an int64 gives topk an
    exact secondary key, without perturbing distances or sorting the whole cloud.
    Unlike ordinary topk, ties cannot change when padded point counts change.
    The geometry/weights still use the original differentiable FP32 distances.
    """
    count = distance.shape[-1]
    if count >= 2**32:
        raise ValueError("A local block must contain fewer than 2**32 points")
    with torch.no_grad():
        masked = distance.detach().float().masked_fill(~allowed, torch.inf).contiguous()
        bits = masked.view(torch.int32).to(torch.int64)
        index = torch.arange(count, device=distance.device, dtype=torch.int64)
        key = (bits << 32) | index
        return torch.topk(key, min(cap, count), largest=False, sorted=True).indices


def _canonical_cloud(points: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Lexicographically order each padded frame and put invalid points last."""
    valid = valid & torch.isfinite(points).all(-1)
    points = torch.where(valid[..., None], points, torch.zeros_like(points))
    count = points.shape[-2]
    order = torch.arange(count, device=points.device).expand(points.shape[:-1])
    # Sorting indices is discrete: observed content remains differentiable after
    # gathering, but ordering must not retain an unnecessary autograd graph.
    with torch.no_grad():
        for axis in reversed(range(6)):
            values = points[..., axis].gather(-1, order)
            order = order.gather(-1, torch.argsort(values, dim=-1, stable=True))
        ordered_valid = valid.gather(-1, order)
        order = order.gather(-1, torch.argsort(ordered_valid.to(torch.int8), descending=True, stable=True))
    return points.gather(-2, order[..., None].expand(*order.shape, 6)), valid.gather(-1, order), order


def _batched_fps(points: Tensor, valid: Tensor, count: int) -> Tensor:
    """Return FPS point indices for all [block, frame] pairs simultaneously."""
    shape = valid.shape
    indices = torch.zeros((*shape[:-1], count), device=points.device, dtype=torch.long)
    distance = torch.full(shape, torch.inf, device=points.device)
    chosen = torch.zeros_like(valid)
    nearest = torch.zeros(shape[:-1], device=points.device, dtype=torch.long)
    with torch.no_grad():
        xyz = points[..., :3].detach()
        for index in range(count):
            indices[..., index] = nearest
            chosen.scatter_(-1, nearest[..., None], True)
            center = xyz.gather(-2, nearest[..., None, None].expand(*nearest.shape, 1, 3))
            distance = torch.minimum(distance, (xyz - center).square().sum(-1))
            distance = distance.masked_fill(chosen | ~valid, -1.0)
            nearest = distance.argmax(-1)
    return indices


def forward_batched(
    encoder: SpacetimeEncoder,
    points: Tensor,
    valid: Tensor,
    times: Tensor,
    centers_per_frame: int | Sequence[int],
    *,
    center_indices: Tensor | None = None,
) -> EncodedLeaves:
    """Pool independent padded local blocks [K,F,N,6] into leaves [K,M,...].

    Callers supply chronological source-local timestamps and pad frames using an
    all-False point mask. Padding values, including NaNs, are ignored. A missing
    frame's candidate slots remain invalid; no occupied leaf is manufactured.
    """
    from .spacetime_encoder import EncodedLeaves

    if points.ndim != 4 or points.shape[-1] != 6 or valid.shape != points.shape[:-1]:
        raise ValueError("Expected points [K,F,N,6] and True-for-valid mask [K,F,N]")
    k, frames, point_count, _ = points.shape
    if times.shape != (k, frames):
        raise ValueError("Source-local timestamps must have shape [K,F]")
    quotas = [centers_per_frame] * frames if isinstance(centers_per_frame, int) else list(centers_per_frame)
    if len(quotas) != frames or any(not isinstance(q, int) or q < 0 for q in quotas):
        raise ValueError("One nonnegative integer center quota is required per frame")
    device = encoder.raw_tau_x.device
    points, valid = points.to(device=device, dtype=torch.float32), valid.to(device=device, dtype=torch.bool)
    times = times.to(device=device, dtype=torch.float32)
    finite_times = torch.isfinite(times)
    valid = valid & finite_times[..., None]
    times = torch.where(finite_times, times, torch.zeros_like(times))
    m = sum(quotas)
    frame_index = torch.tensor([f for f, q in enumerate(quotas) for _ in range(q)], device=device)
    frame_index = frame_index.to(torch.long)
    center_time = times[:, frame_index]
    if not m or not point_count or not k:
        z = torch.zeros((k, m), device=device)
        xyz = torch.zeros((k, m, 3), device=device)
        return EncodedLeaves(
            torch.zeros((k, m, encoder.width), device=device),
            z.bool(),
            xyz,
            xyz.clone(),
            xyz.clone(),
            center_time,
            center_time.clone(),
            center_time.clone(),
            z,
        )
    cloud, point_valid, order = _canonical_cloud(points, valid)
    if center_indices is None:
        if encoder.center_sampling is not None:
            raise ValueError("Configured center sampling requires CPU-prepared indices in batched pooling")
        chosen = _batched_fps(cloud, point_valid, max(quotas))
        candidate_valid = (
            torch.arange(max(quotas), device=device)[None, None, :] < point_valid.sum(-1)[..., None]
        )
    else:
        if center_indices.shape != (k, frames, max(quotas)) or center_indices.dtype != torch.long:
            raise ValueError("Prepared center indices require int64 [blocks,frames,max_frame_quota]")
        center_indices = center_indices.to(device)
        in_range = (center_indices >= 0) & (center_indices < point_count)
        inverse = torch.empty_like(order)
        inverse.scatter_(-1, order, torch.arange(point_count, device=device).expand_as(order))
        chosen = inverse.gather(-1, center_indices.clamp(0, point_count - 1))
        candidate_valid = in_range & point_valid.gather(-1, chosen)
    quota_index = torch.tensor([i for q in quotas for i in range(q)], device=device)
    selected = chosen[:, frame_index, quota_index]
    cloud = cloud.flatten(1, 2)
    selected = selected + frame_index[None, :] * point_count
    anchor = cloud[..., :3].gather(1, selected[..., None].expand(k, m, 3))
    center_valid = candidate_valid[:, frame_index, quota_index]
    anchor = torch.where(center_valid[..., None], anchor, torch.zeros_like(anchor))
    point_valid = point_valid.flatten(1, 2)
    point_time = times[:, :, None].expand(k, frames, point_count).flatten(1, 2)
    point_frame = torch.arange(frames, device=device).repeat_interleave(point_count)
    features = encoder.point_mlp(cloud)
    batch_index = torch.arange(k, device=device)[:, None, None]
    contents, valids, minima, maxima, tmins, tmaxs, occupancies = [], [], [], [], [], [], []
    for start in range(0, m, encoder.center_chunk):
        end = min(start + encoder.center_chunk, m)
        dx = cloud[:, None, :, :3] - anchor[:, start:end, None, :]
        dt = point_time[:, None, :] - center_time[:, start:end, None]
        spatial, temporal = dx.square().sum(-1), dt.square()
        allowed = (spatial <= encoder.spatial_radius**2) & (dt.abs() <= encoder.temporal_radius)
        allowed = allowed & center_valid[:, start:end, None] & point_valid[:, None, :]
        if encoder.framewise:
            allowed = allowed & (point_frame[None, None, :] == frame_index[None, start:end, None])
        distance = spatial / encoder.tau_x + temporal / encoder.tau_t
        nearest = canonical_nearest(distance, allowed, encoder.neighbor_cap)
        mask = allowed.gather(-1, nearest)
        row_valid = mask.any(-1)
        selected_distance = distance.gather(-1, nearest)
        logits = (-selected_distance).masked_fill(~mask, -torch.inf)
        logits = torch.where(row_valid[..., None], logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1) * mask.float()
        relative = torch.cat(
            [dx.gather(2, nearest[..., None].expand(*nearest.shape, 3)), dt.gather(-1, nearest)[..., None]],
            dim=-1,
        )
        edge = encoder.edge_mlp(torch.cat([features[batch_index, nearest], relative], dim=-1)).float()
        contents.append((weights[..., None] * edge).sum(2, dtype=torch.float32))
        neighbors = cloud[batch_index, nearest, :3]
        lo = neighbors.masked_fill(~mask[..., None], torch.inf).amin(2)
        hi = neighbors.masked_fill(~mask[..., None], -torch.inf).amax(2)
        minima.append(torch.where(row_valid[..., None], lo, torch.zeros_like(lo)))
        maxima.append(torch.where(row_valid[..., None], hi, torch.zeros_like(hi)))
        neighbor_time = point_time[batch_index, nearest]
        lo_t = neighbor_time.masked_fill(~mask, torch.inf).amin(2)
        hi_t = neighbor_time.masked_fill(~mask, -torch.inf).amax(2)
        tmins.append(torch.where(row_valid, lo_t, center_time[:, start:end]))
        tmaxs.append(torch.where(row_valid, hi_t, center_time[:, start:end]))
        occupancies.append((torch.exp(-selected_distance) * mask).sum(-1))
        valids.append(row_valid)
    return EncodedLeaves(
        torch.cat(contents, dim=1),
        torch.cat(valids, dim=1),
        anchor,
        torch.cat(minima, dim=1),
        torch.cat(maxima, dim=1),
        center_time,
        torch.cat(tmins, dim=1),
        torch.cat(tmaxs, dim=1),
        torch.cat(occupancies, dim=1),
    )
