"""Global B-token learned resampler; full O(BN) read is included in timing."""

import torch
from torch import nn
from stmem.models.action_flow import Attention


class Perceiver(nn.Module):
    def __init__(self, width, heads, budget):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(budget, width) * 0.02)
        self.attn = Attention(width, heads)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, history, bias, *, return_mask=False):
        """Resample padded histories, keeping empty examples explicitly invalid.

        ``return_mask=True`` returns (content, valid). A batched empty history
        occupies ``budget`` zero, invalid slots; it never creates learned
        history tokens for an example which has no history.
        """
        unbatched = history.ndim == 2
        if unbatched and not len(history):
            valid = torch.zeros(0, device=history.device, dtype=torch.bool)
            return (history, valid) if return_mask else history
        if unbatched:
            history, bias = history[None], bias[None]
        if history.ndim != 3 or bias.shape != history.shape[:2]:
            raise ValueError("Perceiver requires history [N,D] or [B,N,D] and matching bias")
        valid = torch.isfinite(bias).any(-1, keepdim=True).expand(-1, len(self.latents))
        x = self.latents[None].expand(history.shape[0], -1, -1)
        x = x + self.attn(x, history, bias)
        x = x + self.ff(self.norm(x))
        x = x.masked_fill(~valid[..., None], 0)
        if unbatched:
            x, valid = x[0], valid[0]
        return (x, valid) if return_mask else x
