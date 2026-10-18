"""Action-query flow matching: no dense archive self-attention."""

from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class ActionNormalizer(nn.Module):
    def __init__(self, dimensions: int = 8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimensions))
        self.register_buffer("std", torch.ones(dimensions))
        self.register_buffer("count", torch.zeros(dimensions, dtype=torch.long))

    @torch.no_grad()
    def fit(self, actions: torch.Tensor, valid: torch.Tensor):
        a = actions.float()
        m = valid.bool() & torch.isfinite(a)
        count = m.sum(0)
        mean = a.masked_fill(~m, 0).sum(0) / count.clamp_min(1)
        var = ((a - mean).square().masked_fill(~m, 0)).sum(0) / count.clamp_min(1)
        self.mean.copy_(mean)
        self.std.copy_(var.sqrt().clamp_min(0.01))
        self.count.copy_(count)

    def normalize(self, action):
        return (action.float() - self.mean) / self.std

    def denormalize(self, action):
        return action.float() * self.std + self.mean


def masked_mse(prediction, target, valid):
    """Every valid command dimension has equal weight; padding has zero gradient."""
    mask = valid.bool()
    error = (prediction.float() - target.float()).square().masked_fill(~mask, 0)
    denom = mask.sum().clamp_min(1)
    return error.sum() / denom


def flow_sample(actions, valid, normalizer, *, noise=None, time=None, generator=None):
    clean = normalizer.normalize(actions).masked_fill(~valid.bool(), 0)
    if noise is None:
        noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    if time is None:
        time = torch.rand((), device=clean.device, generator=generator)
    x = ((1 - time) * noise + time * clean).masked_fill(~valid.bool(), 0)
    target = (clean - noise).masked_fill(~valid.bool(), 0)
    return x, time, target, noise


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x, context, bias=None):
        b, n, d = x.shape
        q = self.q(x).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(context).chunk(2, -1)
        k = k.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        mask = None if bias is None else bias[:, None, None, :].to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class FlowBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.self_attn = Attention(width, heads)
        self.cross_attn = Attention(width, heads)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x, context, bias):
        y = self.norm1(x)
        x = x + self.self_attn(y, y)
        x = x + self.cross_attn(self.norm2(x), context, bias)
        return x + self.ff(self.norm3(x))


class ActionFlow(nn.Module):
    def __init__(self, width=256, heads=8, layers=4, horizon=16, dimensions=8):
        super().__init__()
        self.horizon = horizon
        self.dimensions = dimensions
        self.action_in = nn.Linear(dimensions, width)
        self.position = nn.Parameter(torch.randn(horizon, width) * 0.01)
        self.time_in = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        self.layers = nn.ModuleList([FlowBlock(width, heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dimensions))

    def forward(self, noisy_action, time, context, bias):
        if noisy_action.ndim == 2:
            noisy_action = noisy_action[None]
        if context.ndim == 2:
            context = context[None]
            bias = bias[None]
        freq = torch.exp(torch.linspace(0, math.log(1000), 16, device=noisy_action.device))
        phase = time.reshape(-1, 1).float() * freq[None] * math.pi
        t = self.time_in(torch.cat([phase.sin(), phase.cos()], -1))
        x = self.action_in(noisy_action) + self.position[None] + t[:, None]
        for layer in self.layers:
            x = layer(x, context, bias)
        return self.out(x).squeeze(0)

    @torch.no_grad()
    def integrate(self, context, bias, *, steps=8, noise=None, generator=None):
        """Fixed-step explicit Euler, from noise (u=0) to action (u=1)."""
        if steps < 1:
            raise ValueError("At least one flow integration step is required")
        if noise is None:
            noise = torch.randn(self.horizon, self.dimensions, device=context.device, generator=generator)
        x = noise.float().clone()
        for i in range(steps):
            t = torch.tensor(i / steps, device=x.device)
            x = x + self(x, t, context, bias).float() / steps
        return x


class RegressionHead(nn.Module):
    """Optional debug-only deterministic action head with the same context API."""

    def __init__(self, width=256, heads=8, layers=4, horizon=16, dimensions=8):
        super().__init__()
        self.horizon = horizon
        self.dimensions = dimensions
        self.queries = nn.Parameter(torch.randn(horizon, width) * 0.01)
        self.layers = nn.ModuleList([FlowBlock(width, heads) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dimensions))

    def forward(self, noisy_action, time, context, bias):
        if context.ndim == 2:
            context = context[None]
            bias = bias[None]
        x = self.queries[None].expand(context.shape[0], -1, -1)
        for layer in self.layers:
            x = layer(x, context, bias)
        return self.out(x).squeeze(0)

    @torch.no_grad()
    def integrate(self, context, bias, **kwargs):
        return self(None, None, context, bias)
