"""Tensor banks for padded policy batches, without per-leaf Python teacher trees.

The full-history teacher reads every valid leaf, so parent construction is not
needed for its loss. Evaluation and compact reader selection retain the exact
archive implementation. All metadata fields and current/history boundaries are
shared with that reference path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from stmem.models.metadata_adapter import METADATA_DIM
from stmem.models.spacetime_encoder import EncodedLeaves, cached_frame_centers


@dataclass
class TokenBank:
    content: torch.Tensor
    metadata: torch.Tensor
    valid: torch.Tensor
    log_count: torch.Tensor

    def adapt(self, adapter):
        # Mask both inputs and outputs: padding cannot carry NaNs or adapter bias.
        content = self.content.masked_fill(~self.valid[..., None], 0)
        meta = self.metadata.masked_fill(~self.valid[..., None], 0)
        result = adapter.norm(adapter.content(content) + adapter.meta(meta))
        return result.masked_fill(~self.valid[..., None], 0)


@dataclass
class PackedBlocks:
    points: torch.Tensor
    valid: torch.Tensor
    times: torch.Tensor
    frame_valid: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    available: torch.Tensor
    source: torch.Tensor
    completed: torch.Tensor
    owner: torch.Tensor
    slot: torch.Tensor
    blocks_per_example: list[int]
    center_indices: dict[int, torch.Tensor] = field(default_factory=dict)
    center_sampling_signature: str | None = None


@dataclass(frozen=True)
class PackedFrameInfo:
    """Non-tensor identity needed to rebuild a canonical CPU frame."""

    point_count: int
    time: float
    source: str
    event_id: str
    source_index: int
    action_available: bool
    center_counts: tuple[int, ...] = ()
    center_sampling_signature: str | None = None

    @classmethod
    def from_frame(cls, frame):
        return cls(
            len(frame.points),
            frame.time,
            frame.source,
            frame.event_id,
            frame.source_index,
            frame.action_available,
            tuple(sorted(frame.center_indices)),
            frame.center_sampling_signature,
        )


@dataclass(frozen=True)
class PackedDecisionInfo:
    """Bookkeeping only; these IDs and counts are never model features."""

    goal: str
    task: str
    episode_id: str
    index: int
    conditioning: tuple[PackedFrameInfo, ...]
    execution_prefix: tuple[PackedFrameInfo, ...]
    current: PackedFrameInfo

    @property
    def conditioning_frames(self):
        return len(self.conditioning)

    @property
    def execution_history_frames(self):
        return len(self.execution_prefix)


@dataclass
class PackedPolicyBatch:
    """Raw CPU inputs carried between workers and the training process.

    Historical frames occupy a fixed number of contiguous tensor storages,
    independent of episode length. In particular, do not attach the original
    Decisions here: their thousands of individual tensors would all be sent
    through multiprocessing. No learned features are retained in this payload.
    Targets are separate supervision and are never consumed by batch_context.
    """

    history: PackedBlocks
    current_points: torch.Tensor
    current_valid: torch.Tensor
    current_state: torch.Tensor
    clocks: torch.Tensor
    target_chunk: torch.Tensor
    target_valid: torch.Tensor
    metadata: list[PackedDecisionInfo]
    block_size: int
    current_center_indices: dict[int, torch.Tensor] = field(default_factory=dict)
    current_center_sampling_signature: str | None = None

    @property
    def history_lengths(self):
        return [m.conditioning_frames + m.execution_history_frames for m in self.metadata]

    def decisions(self):
        """Rebuild canonical views for consumers of frozen episode caches.

        The teacher uses metadata directly and never needs this Python frame
        reconstruction. Returned tensors share the packed CPU storage; exact
        original point counts, identities and Python clocks are restored.
        """
        from stmem.data.canonical import Decision, Frame

        def frame(info, points, valid, state, action, centers):
            return Frame(
                points=points[: info.point_count],
                valid=valid[: info.point_count],
                proprio=state,
                previous_action=action,
                action_available=info.action_available,
                time=info.time,
                source=info.source,
                event_id=info.event_id,
                source_index=info.source_index,
                center_indices={quota: centers[quota] for quota in info.center_counts},
                center_sampling_signature=info.center_sampling_signature,
            )

        result, block = [], 0
        for owner, info in enumerate(self.metadata):
            banks = []
            for entries in (info.conditioning, info.execution_prefix):
                frames = []
                for start in range(0, len(entries), self.block_size):
                    for offset, identity in enumerate(entries[start : start + self.block_size]):
                        frames.append(
                            frame(
                                identity,
                                self.history.points[block, offset],
                                self.history.valid[block, offset],
                                self.history.state[block, offset],
                                self.history.action[block, offset],
                                {
                                    quota: values[block, offset]
                                    for quota, values in self.history.center_indices.items()
                                },
                            )
                        )
                    block += 1
                banks.append(frames)
            current = frame(
                info.current,
                self.current_points[owner, 0],
                self.current_valid[owner, 0],
                self.current_state[owner, :15],
                self.current_state[owner, 15:23],
                {quota: values[owner, 0] for quota, values in self.current_center_indices.items()},
            )
            result.append(
                Decision(
                    current,
                    banks[0],
                    banks[1],
                    self.target_chunk[owner],
                    self.target_valid[owner],
                    info.goal,
                    info.task,
                    info.episode_id,
                    info.index,
                )
            )
        return result


def _validate_banks(decisions):
    for decision in decisions:
        seen = set()
        for source, frames in (
            ("conditioning", decision.conditioning),
            ("execution", decision.execution_prefix),
        ):
            previous = -math.inf
            for frame in frames:
                key = source, frame.event_id
                if (
                    frame.source != source
                    or key in seen
                    or not math.isfinite(frame.time)
                    or frame.time <= previous
                ):
                    raise ValueError(
                        "Causal source banks require unique observations and increasing finite clocks"
                    )
                if source == "execution" and (
                    frame.event_id == decision.current.event_id or frame.time >= decision.current.time
                ):
                    raise ValueError(
                        "Execution history must strictly precede the separate current observation"
                    )
                seen.add(key)
                previous = frame.time
        if decision.current.source != "execution" or not math.isfinite(decision.current.time):
            raise ValueError("The current observation must have a finite execution clock")


def _center_cache_state(frames):
    frames = list(frames)
    counts = sorted({quota for frame in frames for quota in frame.center_indices})
    signatures = {frame.center_sampling_signature for frame in frames}
    signature = next(iter(signatures)) if len(signatures) == 1 else None
    return counts, signature


def pack_blocks(decisions, block_size, *, center_sampling=None, centers_per_frame=None):
    """Collate raw CPU frames once; no padding across an entire episode's points."""
    if block_size < 1:
        raise ValueError("History block size must be positive")
    _validate_banks(decisions)
    blocks, owners, slots, sources, completed, lengths = [], [], [], [], [], []
    for owner, decision in enumerate(decisions):
        slot = 0
        for source, frames in ((0, decision.conditioning), (1, decision.execution_prefix)):
            for start in range(0, len(frames), block_size):
                part = frames[start : start + block_size]
                blocks.append(part)
                owners.append(owner)
                slots.append(slot)
                sources.append(source)
                completed.append(source == 0 or len(part) == block_size)
                slot += 1
        lengths.append(slot)
    n = max((len(frame.points) for part in blocks for frame in part), default=0)
    k = len(blocks)
    if center_sampling is not None:
        if centers_per_frame is None or len(centers_per_frame) != block_size:
            raise ValueError("Configured center sampling requires one quota per block position")
        from stmem.data.center_sampling import center_sampling_signature

        signature = center_sampling_signature(center_sampling)
        for part in blocks:
            for position, frame in enumerate(part):
                cached_frame_centers(frame, centers_per_frame[position], center_sampling, signature)
    center_counts, signature = _center_cache_state(frame for part in blocks for frame in part)
    # Data preparation supplies CPU tensors. Also accept device-resident frames
    # for invariant checks; copy_ moves them to this bounded host staging buffer.
    points = torch.zeros(k, block_size, n, 6)
    valid = torch.zeros(k, block_size, n, dtype=torch.bool)
    times = torch.zeros(k, block_size)
    frame_valid = torch.zeros(k, block_size, dtype=torch.bool)
    state = torch.zeros(k, block_size, 15)
    action = torch.zeros(k, block_size, 8)
    available = torch.zeros(k, block_size, dtype=torch.bool)
    centers = {quota: torch.full((k, block_size, quota), -1, dtype=torch.long) for quota in center_counts}
    for i, part in enumerate(blocks):
        for j, frame in enumerate(part):
            count = len(frame.points)
            points[i, j, :count].copy_(frame.points)
            valid[i, j, :count].copy_(frame.valid)
            times[i, j] = frame.time
            frame_valid[i, j] = True
            state[i, j].copy_(frame.proprio)
            action[i, j].copy_(frame.previous_action)
            available[i, j] = frame.action_available and sources[i] == 1
            for quota, indices in frame.center_indices.items():
                centers[quota][i, j].copy_(indices)
        times[i, len(part) :] = part[-1].time
    return PackedBlocks(
        points,
        valid,
        times,
        frame_valid,
        state,
        action,
        available,
        torch.tensor(sources, dtype=torch.long),
        torch.tensor(completed, dtype=torch.bool),
        torch.tensor(owners, dtype=torch.long),
        torch.tensor(slots, dtype=torch.long),
        lengths,
        centers,
        signature,
    )


def pack_policy_batch(decisions, block_size, *, model_config=None):
    """Collate all policy inputs on CPU, suitable for a background worker."""
    decisions = list(decisions)
    if not decisions:
        raise ValueError("A policy batch must contain at least one decision")
    center_sampling = (
        None if model_config is None else model_config.get("encoder_options", {}).get("center_sampling")
    )
    quotas = None
    if center_sampling is not None:
        visual_tokens = model_config["visual_tokens"]
        quotas = [
            visual_tokens // block_size + int(i < visual_tokens % block_size) for i in range(block_size)
        ]
        from stmem.data.center_sampling import center_sampling_signature

        signature = center_sampling_signature(center_sampling)
        for decision in decisions:
            cached_frame_centers(decision.current, model_config["current_tokens"], center_sampling, signature)
    history = pack_blocks(decisions, block_size, center_sampling=center_sampling, centers_per_frame=quotas)
    center_counts, signature = _center_cache_state(d.current for d in decisions)
    n = max(len(d.current.points) for d in decisions)
    points = torch.zeros(len(decisions), 1, n, 6)
    valid = torch.zeros(len(decisions), 1, n, dtype=torch.bool)
    states = torch.zeros(len(decisions), 24)
    centers = {quota: torch.full((len(decisions), 1, quota), -1, dtype=torch.long) for quota in center_counts}
    for i, decision in enumerate(decisions):
        current = decision.current
        count = len(current.points)
        points[i, 0, :count].copy_(current.points)
        valid[i, 0, :count].copy_(current.valid)
        states[i, :15].copy_(current.proprio)
        states[i, 15:23].copy_(current.previous_action)
        states[i, 23] = float(current.action_available)
        for quota, indices in current.center_indices.items():
            centers[quota][i, 0].copy_(indices)
    return PackedPolicyBatch(
        history=history,
        current_points=points,
        current_valid=valid,
        current_state=states,
        clocks=_clocks(decisions, "cpu"),
        target_chunk=torch.stack([d.target_chunk for d in decisions]).cpu(),
        target_valid=torch.stack([d.target_valid for d in decisions]).cpu(),
        metadata=[
            PackedDecisionInfo(
                d.goal,
                d.task,
                d.episode_id,
                d.index,
                tuple(PackedFrameInfo.from_frame(frame) for frame in d.conditioning),
                tuple(PackedFrameInfo.from_frame(frame) for frame in d.execution_prefix),
                PackedFrameInfo.from_frame(d.current),
            )
            for d in decisions
        ],
        block_size=block_size,
        current_center_indices=centers,
        current_center_sampling_signature=signature,
    )


def _leaf_metadata(anchor, lower, upper, time_min, time_max, now, source, visual, completed):
    """Tensor equivalent of metadata() for unit-mass depth/scale-zero leaves."""
    shape = time_min.shape
    zero = torch.zeros_like(time_min)
    one = torch.ones_like(time_min)
    now = now.expand(shape)
    source = source.expand(shape)
    completed = completed.expand(shape)
    visual = torch.as_tensor(visual, device=anchor.device).expand(shape)
    scalars = torch.stack(
        [
            time_min / 100,
            time_max / 100,
            (now - time_min) / 100,
            (now - time_max) / 100,
            zero,
            zero,
            zero,
            completed.float(),
            zero,
            (source == 0).float(),
            (source == 1).float(),
            visual.float(),
            (~visual.bool()).float(),
            one,
        ],
        dim=-1,
    )
    return torch.cat([anchor, lower, upper, scalars], dim=-1).float()


def _clocks(decisions, device):
    return torch.tensor(
        [[d.conditioning[-1].time if d.conditioning else 0, d.current.time] for d in decisions],
        device=device,
        dtype=torch.float32,
    )


def encode_history_bank(policy, decisions, clocks, *, packed=None):
    encoder, device = policy.encoder, policy.device
    quotas = [
        encoder.visual_tokens // encoder.block_size + int(i < encoder.visual_tokens % encoder.block_size)
        for i in range(encoder.block_size)
    ]
    packed = (
        packed
        if packed is not None
        else pack_blocks(
            decisions,
            encoder.block_size,
            center_sampling=encoder.visual.center_sampling,
            centers_per_frame=quotas,
        )
    )
    b, k = len(decisions), len(packed.owner)
    width = encoder.width
    if not k:
        return TokenBank(
            torch.zeros(b, 0, width, device=device),
            torch.zeros(b, 0, METADATA_DIM, device=device),
            torch.zeros(b, 0, dtype=torch.bool, device=device),
            torch.zeros(b, 0, device=device),
        )
    prepared_centers = None
    if encoder.visual.center_sampling is not None:
        if packed.center_sampling_signature != encoder.visual.center_sampling_signature:
            raise ValueError("Packed historical center settings do not match the encoder")
        if any(quota not in packed.center_indices for quota in quotas):
            raise ValueError(
                "Packed history is missing a required center quota; prepare centers during CPU collation"
            )
        prepared_centers = torch.full((k, encoder.block_size, max(quotas)), -1, dtype=torch.long)
        for position, quota in enumerate(quotas):
            prepared_centers[:, position, :quota].copy_(packed.center_indices[quota][:, position])
    chunk_size = policy.config["training"].get("encoder_block_batch_size", 32)
    contents, metas, masks = [], [], []

    def pool(points, valid, times, indices):
        result = encoder.visual.forward_batched(points, valid, times, quotas, center_indices=indices)
        return tuple(getattr(result, field) for field in EncodedLeaves.__dataclass_fields__)

    for start in range(0, k, chunk_size):
        end = min(start + chunk_size, k)
        points = packed.points[start:end].to(device)
        valid = packed.valid[start:end].to(device)
        times = packed.times[start:end].to(device)
        indices = None if prepared_centers is None else prepared_centers[start:end].to(device)
        if encoder.checkpoint_blocks and encoder.training and torch.is_grad_enabled():
            visual = EncodedLeaves(*checkpoint(pool, points, valid, times, indices, use_reentrant=False))
        else:
            visual = EncodedLeaves(*pool(points, valid, times, indices))
        state = packed.state[start:end].to(device)
        events = encoder.event(
            state, packed.action[start:end].to(device), packed.available[start:end].to(device)
        )
        source = packed.source[start:end].to(device)
        owner = packed.owner[start:end].to(device)
        now = clocks[owner, source][:, None]
        done = packed.completed[start:end].to(device)[:, None]
        visual_meta = _leaf_metadata(
            visual.anchor,
            visual.bounds_min,
            visual.bounds_max,
            visual.time_min.detach(),
            visual.time_max.detach(),
            now,
            source[:, None],
            True,
            done,
        )
        anchor = state[..., 7:10]
        event_meta = _leaf_metadata(anchor, anchor, anchor, times, times, now, source[:, None], False, done)
        contents.append(torch.cat([visual.content, events], dim=1))
        metas.append(torch.cat([visual_meta, event_meta], dim=1))
        masks.append(torch.cat([visual.valid, packed.frame_valid[start:end].to(device)], dim=1))
    tokens = encoder.visual_tokens + encoder.block_size
    max_length = max(packed.blocks_per_example) * tokens
    indices = (
        ((packed.owner * max_length + packed.slot * tokens)[:, None] + torch.arange(tokens))
        .flatten()
        .to(device)
    )

    def scatter(value):
        shape = (b * max_length,) + value.shape[2:]
        result = value.new_zeros(shape).index_copy(0, indices, value.flatten(0, 1))
        return result.reshape(b, max_length, *value.shape[2:])

    return TokenBank(
        scatter(torch.cat(contents)),
        scatter(torch.cat(metas)),
        scatter(torch.cat(masks)),
        torch.zeros(b, max_length, device=device),
    )


def encode_current_bank(policy, decisions, clocks, *, packed_batch=None):
    encoder = policy.encoder
    prepared_centers = None
    if packed_batch is None:
        n = max(len(d.current.points) for d in decisions)
        points = torch.zeros(len(decisions), 1, n, 6)
        valid = torch.zeros(len(decisions), 1, n, dtype=torch.bool)
        for i, decision in enumerate(decisions):
            count = len(decision.current.points)
            points[i, 0, :count].copy_(decision.current.points)
            valid[i, 0, :count].copy_(decision.current.valid)
        if encoder.visual.center_sampling is not None:
            prepared_centers = torch.stack(
                [
                    cached_frame_centers(
                        d.current,
                        encoder.current_tokens,
                        encoder.visual.center_sampling,
                        encoder.visual.center_sampling_signature,
                    ).cpu()
                    for d in decisions
                ]
            )[:, None]
    else:
        points, valid = packed_batch.current_points, packed_batch.current_valid
        if encoder.visual.center_sampling is not None:
            if packed_batch.current_center_sampling_signature != encoder.visual.center_sampling_signature:
                raise ValueError("Packed current center settings do not match the encoder")
            if encoder.current_tokens not in packed_batch.current_center_indices:
                raise ValueError("Packed current input is missing its required center quota")
            prepared_centers = packed_batch.current_center_indices[encoder.current_tokens]
    visual = policy.encoder.visual.forward_batched(
        points.to(policy.device),
        valid.to(policy.device),
        clocks[:, 1:2],
        policy.encoder.current_tokens,
        center_indices=None if prepared_centers is None else prepared_centers.to(policy.device),
    )
    meta = _leaf_metadata(
        visual.anchor,
        visual.bounds_min,
        visual.bounds_max,
        visual.time_min.detach(),
        visual.time_max.detach(),
        clocks[:, 1:2],
        torch.ones(len(decisions), 1, device=policy.device),
        True,
        torch.ones(len(decisions), 1, device=policy.device),
    )
    return TokenBank(visual.content, meta, visual.valid, torch.zeros_like(visual.time_anchor))


def records_bank(policy, record_lists, clocks):
    """Pack selected frozen records with one transfer per padded tensor."""
    if len(record_lists) != len(clocks):
        raise ValueError("One record list is required per decision")
    width = policy.config["model"]["width"]
    # Frozen caches normally live on CPU; choose the records' device to preserve
    # differentiability in parity tests without per-record device transfers.
    record_device = next(
        (r.content_sum.device for records in record_lists for r in records), torch.device("cpu")
    )
    content_rows, meta_rows, counts, valid_rows = [], [], [], []
    clock_rows = clocks.detach().cpu().tolist()
    for records, clock in zip(record_lists, clock_rows):
        if not records:
            content_rows.append(torch.zeros(0, width, device=record_device))
            meta_rows.append(torch.zeros(0, METADATA_DIM, device=record_device))
            counts.append(torch.zeros(0, device=record_device))
            valid_rows.append(torch.zeros(0, device=record_device, dtype=torch.bool))
            continue
        content_rows.append(torch.stack([r.content for r in records]).float())
        geometry = torch.stack([torch.cat([r.anchor, r.bounds_min, r.bounds_max]) for r in records]).float()
        scalars = []
        for r in records:
            now = clock[0 if r.source == "conditioning" else 1]
            scalars.append(
                [
                    r.time_min / 100,
                    r.time_max / 100,
                    (now - r.time_min) / 100,
                    (now - r.time_max) / 100,
                    math.log(r.count),
                    r.depth / 16,
                    math.log1p(r.scale),
                    float(r.completed),
                    float(r.disconnected),
                    float(r.source == "conditioning"),
                    float(r.source == "execution"),
                    float(r.modality == "visual"),
                    float(r.modality == "event"),
                    float(r.count == 1),
                ]
            )
        meta_rows.append(torch.cat([geometry, torch.tensor(scalars, device=record_device)], dim=-1))
        counts.append(torch.tensor([math.log(r.count) for r in records], device=record_device))
        valid_rows.append(torch.ones(len(records), device=record_device, dtype=torch.bool))
    return TokenBank(
        *(
            pad_sequence(rows, batch_first=True).to(policy.device)
            for rows in (content_rows, meta_rows, valid_rows, counts)
        )
    )


def batch_context(
    policy,
    decisions,
    *,
    evidences=None,
    record_lists=None,
    perceiver=False,
    progress=None,
    packed_batch=None,
):
    """Encode/pack all examples, then adapt padded tokens with batched GEMMs."""
    device = policy.device
    if packed_batch is not None:
        if len(packed_batch.metadata) != len(decisions):
            raise ValueError("Packed inputs must contain one entry per decision")
        if packed_batch.block_size != policy.encoder.block_size:
            raise ValueError("Packed history uses a different encoder block size")
        clocks = packed_batch.clocks.to(device)
    else:
        clocks = _clocks(decisions, device)
    if evidences is None:
        if progress is not None:
            progress.update("encoding_history")
        history = encode_history_bank(
            policy, decisions, clocks, packed=None if packed_batch is None else packed_batch.history
        )
        current = encode_current_bank(policy, decisions, clocks, packed_batch=packed_batch)
        text = policy.text_encoder(
            [d.goal for d in (decisions if packed_batch is None else packed_batch.metadata)]
        )
    else:
        if len(evidences) != len(decisions):
            raise ValueError("One cached evidence bank is required per decision")
        current = records_bank(policy, [e.current_records for e in evidences], clocks)
        histories = [e.archive.leaves for e in evidences] if record_lists is None else record_lists
        history = records_bank(policy, histories, clocks)
        text = torch.stack([e.text for e in evidences]).to(device)
    if progress is not None:
        progress.update("forward")
    states = (
        packed_batch.current_state.to(device)
        if packed_batch is not None
        else torch.stack(
            [
                torch.cat(
                    [
                        d.current.proprio,
                        d.current.previous_action,
                        d.current.proprio.new_tensor([float(d.current.action_available)]),
                    ]
                )
                for d in decisions
            ]
        )
        .to(device)
        .float()
    )
    current_tokens = torch.cat(
        [
            current.adapt(policy.adapter),
            policy.proprio(states)[:, None],
            text[:, None],
            policy.clock(clocks / 100)[:, None],
        ],
        dim=1,
    )
    current_valid = torch.cat(
        [current.valid, torch.ones(len(decisions), 3, device=device, dtype=torch.bool)], dim=1
    )
    history_tokens = history.adapt(policy.adapter)
    history_valid = history.valid
    bias = (
        history.log_count
        if policy.config["model"].get("mass_bias", True)
        else torch.zeros_like(history.log_count)
    )
    bias = bias.masked_fill(~history.valid, -torch.inf)
    if perceiver:
        if policy.resampler is None:
            raise RuntimeError("PERCEIVER requires its separately adapted checkpoint")
        history_tokens, history_valid = policy.resampler(history_tokens, bias, return_mask=True)
        bias = torch.zeros_like(history_valid, dtype=torch.float32).masked_fill(~history_valid, -torch.inf)
    current_bias = torch.zeros_like(current_valid, dtype=torch.float32).masked_fill(
        ~current_valid, -torch.inf
    )
    return (
        torch.cat([current_tokens, history_tokens], dim=1),
        torch.cat([current_bias, bias], dim=1),
        {
            "history_leaves": history.valid.sum(dim=1)
            if evidences is None
            else torch.tensor(
                [len(e.archive.leaves) for e in evidences],
                device=device,
            ),
            "history_context_tokens": history_valid.sum(dim=1),
            "current_visual_tokens": current.valid.sum(dim=1),
        },
    )
