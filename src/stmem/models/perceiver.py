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

    def forward(self, history, bias):
        if not len(history):
            return history
        x = self.latents[None]
        x = x + self.attn(x, history[None], bias[None])
        x = x + self.ff(self.norm(x))
        return x.squeeze(0)
