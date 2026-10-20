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


def masked_mse(prediction, target, valid, *, reduction="mean"):
    """Average each example's valid dimensions, then average examples equally.

    Inputs are [H,D] or [B,H,D]. Padding has zero gradient. Equal example
    weighting matches gradient accumulation even when terminal chunks have
    different numbers of valid commands. ``reduction='none'`` exposes those
    per-example means (a scalar for the unbatched API).
    """
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Prediction, target and validity shapes must match")
    if target.ndim not in (2, 3):
        raise ValueError("Action losses require [H,D] or [B,H,D] tensors")
    if reduction not in ("mean", "none"):
        raise ValueError("Action loss reduction must be mean or none")
    mask = valid.bool()
    error = (prediction.float() - target.float()).masked_fill(~mask, 0).square()
    dims = (-2, -1)
    losses = error.sum(dims) / mask.sum(dims).clamp_min(1)
    return losses.mean() if reduction == "mean" else losses


def flow_sample(actions, valid, normalizer, *, noise=None, time=None, generator=None):
    if actions.ndim not in (2, 3) or valid.shape != actions.shape:
        raise ValueError("Actions and validity require matching [H,D] or [B,H,D] shapes")
    clean = normalizer.normalize(actions).masked_fill(~valid.bool(), 0)
    if noise is None:
        noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    elif noise.shape != clean.shape:
        raise ValueError("Flow noise must match the action shape")
    if time is None:
        shape = (actions.shape[0],) if actions.ndim == 3 else ()
        time = torch.rand(shape, device=clean.device, generator=generator)
    time = torch.as_tensor(time, device=clean.device)
    if actions.ndim == 3:
        if time.numel() == 1:
            time = time.reshape(1).expand(actions.shape[0])
        elif time.shape != (actions.shape[0],):
            raise ValueError("Batched flow time must be scalar or [B]")
        interpolation_time = time[:, None, None]
    else:
        if time.numel() != 1:
            raise ValueError("Unbatched flow time must be scalar")
        time = time.reshape(())
        interpolation_time = time
    x = ((1 - interpolation_time) * noise + interpolation_time * clean).masked_fill(~valid.bool(), 0)
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
        if context.ndim != 3 or context.shape[0] != b or context.shape[2] != d:
            raise ValueError("Attention context must have shape [B,N,D] matching queries")
        if context.shape[1] == 0:
            return torch.zeros_like(x)
        empty = None
        if bias is not None:
            if bias.shape != context.shape[:2]:
                raise ValueError("Attention bias must have shape [B,N]")
            allowed = torch.isfinite(bias)
            bias = bias.masked_fill(~allowed, -torch.inf)
            # Zero invalid values before projection: a NaN padding value must
            # not contaminate K/V products even though its attention is masked.
            context = context.masked_fill(~allowed[..., None], 0)
            empty = ~allowed.any(-1)
            # An entirely masked row attends to a zero sentinel, then is zeroed
            # below. SDPA never receives an all-invalid normalization row.
            bias = torch.cat([torch.where(empty, 0.0, bias[:, 0])[:, None], bias[:, 1:]], dim=1)
        q = self.q(x).reshape(b, n, self.heads, d // self.heads).transpose(1, 2)
        k, v = self.kv(context).chunk(2, -1)
        k = k.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        v = v.reshape(b, -1, self.heads, d // self.heads).transpose(1, 2)
        mask = None if bias is None else bias[:, None, None, :].to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        result = self.out(y.transpose(1, 2).reshape(b, n, d))
        return result if empty is None else result.masked_fill(empty[:, None, None], 0)


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
        unbatched = noisy_action.ndim == 2
        if noisy_action.ndim == 2:
            noisy_action = noisy_action[None]
        elif noisy_action.ndim != 3:
            raise ValueError("Noisy actions require [H,D] or [B,H,D] tensors")
        if context.ndim == 2:
            context = context[None]
            bias = bias[None] if bias is not None else None
        freq = torch.exp(torch.linspace(0, math.log(1000), 16, device=noisy_action.device))
        phase = time.reshape(-1, 1).float() * freq[None] * math.pi
        t = self.time_in(torch.cat([phase.sin(), phase.cos()], -1))
        x = self.action_in(noisy_action) + self.position[None] + t[:, None]
        for layer in self.layers:
            x = layer(x, context, bias)
        result = self.out(x)
        return result[0] if unbatched else result

    @torch.no_grad()
    def integrate(self, context, bias, *, steps=8, noise=None, generator=None):
        """Fixed-step explicit Euler, from noise (u=0) to action (u=1)."""
        if steps < 1:
            raise ValueError("At least one flow integration step is required")
        if noise is None:
            shape = (self.horizon, self.dimensions)
            if context.ndim == 3:
                shape = (context.shape[0], *shape)
            noise = torch.randn(shape, device=context.device, generator=generator)
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
        unbatched = context.ndim == 2 if noisy_action is None else noisy_action.ndim == 2
        if context.ndim == 2:
            context = context[None]
            bias = bias[None] if bias is not None else None
        x = self.queries[None].expand(context.shape[0], -1, -1)
        for layer in self.layers:
            x = layer(x, context, bias)
        result = self.out(x)
        return result[0] if unbatched else result

    @torch.no_grad()
    def integrate(self, context, bias, **kwargs):
        return self(None, None, context, bias)
