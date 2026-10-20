"""Tensor banks for padded policy batches, without per-leaf Python teacher trees.

The full-history teacher reads every valid leaf, so parent construction is not
needed for its loss. Evaluation and compact reader selection retain the exact
archive implementation. All metadata fields and current/history boundaries are
shared with that reference path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from stmem.models.metadata_adapter import METADATA_DIM
from stmem.models.spacetime_encoder import EncodedLeaves


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


def pack_blocks(decisions, block_size):
    """Collate raw CPU frames once; no padding across an entire episode's points."""
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
    # Data preparation supplies CPU tensors. Also accept device-resident frames
    # for invariant checks; copy_ moves them to this bounded host staging buffer.
    points = torch.zeros(k, block_size, n, 6)
    valid = torch.zeros(k, block_size, n, dtype=torch.bool)
    times = torch.zeros(k, block_size)
    frame_valid = torch.zeros(k, block_size, dtype=torch.bool)
    state = torch.zeros(k, block_size, 15)
    action = torch.zeros(k, block_size, 8)
    available = torch.zeros(k, block_size, dtype=torch.bool)
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


def encode_history_bank(policy, decisions, clocks):
    encoder, device = policy.encoder, policy.device
    packed = pack_blocks(decisions, encoder.block_size)
    b, k = len(decisions), len(packed.owner)
    width = encoder.width
    if not k:
        return TokenBank(
            torch.zeros(b, 0, width, device=device),
            torch.zeros(b, 0, METADATA_DIM, device=device),
            torch.zeros(b, 0, dtype=torch.bool, device=device),
            torch.zeros(b, 0, device=device),
        )
    quotas = [
        encoder.visual_tokens // encoder.block_size + int(i < encoder.visual_tokens % encoder.block_size)
        for i in range(encoder.block_size)
    ]
    chunk_size = policy.config["training"].get("encoder_block_batch_size", 32)
    contents, metas, masks = [], [], []

    def pool(points, valid, times):
        result = encoder.visual.forward_batched(points, valid, times, quotas)
        return tuple(getattr(result, field) for field in EncodedLeaves.__dataclass_fields__)

    for start in range(0, k, chunk_size):
        end = min(start + chunk_size, k)
        points = packed.points[start:end].to(device)
        valid = packed.valid[start:end].to(device)
        times = packed.times[start:end].to(device)
        if encoder.checkpoint_blocks and encoder.training and torch.is_grad_enabled():
            visual = EncodedLeaves(*checkpoint(pool, points, valid, times, use_reentrant=False))
        else:
            visual = EncodedLeaves(*pool(points, valid, times))
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


def encode_current_bank(policy, decisions, clocks):
    n = max(len(d.current.points) for d in decisions)
    points = torch.zeros(len(decisions), 1, n, 6)
    valid = torch.zeros(len(decisions), 1, n, dtype=torch.bool)
    for i, decision in enumerate(decisions):
        count = len(decision.current.points)
        points[i, 0, :count].copy_(decision.current.points)
        valid[i, 0, :count].copy_(decision.current.valid)
    visual = policy.encoder.visual.forward_batched(
        points.to(policy.device),
        valid.to(policy.device),
        clocks[:, 1:2],
        policy.encoder.current_tokens,
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


def batch_context(policy, decisions, *, evidences=None, record_lists=None, perceiver=False, progress=None):
    """Encode/pack all examples, then adapt padded tokens with batched GEMMs."""
    device = policy.device
    clocks = _clocks(decisions, device)
    if evidences is None:
        if progress is not None:
            progress.update("encoding_history")
        history = encode_history_bank(policy, decisions, clocks)
        current = encode_current_bank(policy, decisions, clocks)
        text = policy.text_encoder([d.goal for d in decisions])
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
        torch.stack(
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
