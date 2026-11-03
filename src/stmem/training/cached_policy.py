"""Training-only forward path for context tensors assembled by CPU workers.

The encoders, token schema, adapter and decoder are unchanged. This wrapper is
inside DDP so gradients through the shared adapter/clock/decoder are synchronized;
checkpoints still serialize the underlying MemoryPolicy, without wrapper prefixes.
"""

from __future__ import annotations

import torch
from torch import nn

from stmem.models.batched_policy import TokenBank


def _to_device(bank, device):
    return TokenBank(*(value.to(device, non_blocking=True) for value in (
        bank.content, bank.metadata, bank.valid, bank.log_count,
    )))


class CachedPolicyForward(nn.Module):
    """Numerically equivalent to batch_context's cached compact-policy branch."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        if not policy.encoders_frozen or not policy.history_enabled:
            raise ValueError("Cached compact training requires frozen full-history encoders")

    def context(self, batch):
        policy = self.policy
        device = policy.device
        current = _to_device(batch.current, device)
        history = _to_device(batch.history, device)
        clocks = batch.clocks.to(device, non_blocking=True)
        text = batch.text.to(device, non_blocking=True)
        size = len(batch.positions)
        pieces = [current.adapt(policy.adapter)]
        if policy.current_state_enabled:
            states = batch.current_state.to(device, non_blocking=True).float()
            pieces.append(policy.encode_current_state(states)[:, None])
        pieces.extend([text[:, None], policy.clock(clocks / 100)[:, None]])
        current_tokens = torch.cat(pieces, dim=1)
        current_valid = torch.cat([
            current.valid,
            torch.ones(size, 2 + int(policy.current_state_enabled), device=device, dtype=torch.bool),
        ], dim=1)
        history_tokens = history.adapt(policy.adapter)
        history_bias = history.log_count if policy.config["model"].get("mass_bias", True) else (
            torch.zeros_like(history.log_count)
        )
        history_bias = history_bias.masked_fill(~history.valid, -torch.inf)
        current_bias = torch.zeros_like(current_valid, dtype=torch.float32).masked_fill(
            ~current_valid, -torch.inf
        )
        stats = {
            "history_leaves": batch.history_leaf_counts.to(device, non_blocking=True),
            "history_context_tokens": history.valid.sum(dim=1),
            "current_visual_tokens": current.valid.sum(dim=1),
        }
        return (
            torch.cat([current_tokens, history_tokens], dim=1),
            torch.cat([current_bias, history_bias], dim=1),
            stats,
        )

    def forward(self, batch, noisy_action, time, *, progress=None):
        if progress is not None:
            progress.update("cached_context_transfer")
        context, bias, stats = self.context(batch)
        if progress is not None:
            progress.update("forward")
        return self.policy.decoder(noisy_action, time, context, bias), stats
