"""Bounded, detached summaries of decoder cross-attention.

These are observational diagnostics, not an alternative attention implementation.
Hooks reuse the Q/K projections from the actual decoder forward pass. Attention
scores are reduced in key chunks, so no full [heads, queries, context] map is
retained. Entropies are computed per example/head/query before averaging.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import torch
from torch import Tensor


GROUP_NAMES = (
    "current_visual",
    "current_state",
    "goal",
    "clock",
    "history_visual",
    "history_event",
)
GROUP_UNIONS = {
    "current": ("current_visual", "current_state"),
    "history": ("history_visual", "history_event"),
}


def _mean_over(value: Tensor, valid: Tensor) -> Tensor:
    valid = valid.expand_as(value)
    return value.masked_fill(~valid, 0).sum() / valid.sum().clamp_min(1)


@torch.no_grad()
def cross_attention_statistics(
    queries: Tensor,
    keys: Tensor,
    bias: Tensor | None,
    group_ids: Tensor,
    group_names: Sequence[str] = GROUP_NAMES,
    *,
    key_chunk_size: int = 2048,
    group_unions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Tensor]:
    """Summarize projected [B,heads,Q/K,head_width] attention in FP32.

    ``bias`` is the actual additive [B,K] SDPA bias, including log-mass. Its
    nonfinite entries are excluded exactly as in ``Attention.forward``.
    ``group_ids`` is [K] or [B,K]; -1 is permitted only for excluded keys.
    ``relative_mass`` compares each row's group mass with its uniform-token
    baseline before averaging. Conditional entropy is normalized by log of
    the group's valid token count, with singleton entropy defined as zero.
    Empty groups retain mass=uniform_mass=0 but omit undefined relative mass
    and conditional entropy. All-padding examples are excluded from averages.

    Group-local shifted exponential moments keep entropy stable even when a
    group's total attention mass underflows relative to another group.
    """
    if queries.ndim != 4 or keys.ndim != 4:
        raise ValueError("Projected queries and keys require [B,heads,Q/K,width]")
    b, heads, query_count, width = queries.shape
    if keys.shape[:2] != (b, heads) or keys.shape[-1] != width:
        raise ValueError("Projected query/key batch, heads and width must match")
    if b < 1 or heads < 1 or query_count < 1 or width < 1:
        raise ValueError("Attention diagnostics require nonempty queries")
    if queries.device != keys.device:
        raise ValueError("Projected queries and keys must be on the same device")
    if not isinstance(key_chunk_size, int) or isinstance(key_chunk_size, bool) or key_chunk_size < 1:
        raise ValueError("key_chunk_size must be a positive integer")
    names = tuple(group_names)
    if not names or len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ValueError("group_names must be distinct nonempty strings")
    k_count, groups = keys.shape[2], len(names)
    labels = torch.as_tensor(group_ids, device=queries.device)
    if labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex():
        raise ValueError("group_ids must be integer labels")
    if labels.shape == (k_count,):
        labels = labels[None].expand(b, -1)
    if labels.shape != (b, k_count):
        raise ValueError("group_ids must have shape [K] or [B,K]")
    labels = labels.long()
    if bias is None:
        allowed = torch.ones((b, k_count), device=queries.device, dtype=torch.bool)
        additive = torch.zeros((b, k_count), device=queries.device, dtype=torch.float32)
    else:
        if bias.shape != (b, k_count):
            raise ValueError("Attention bias must have shape [B,K]")
        allowed = torch.isfinite(bias).to(queries.device)
        # Match the production cast to Q dtype before the FP32 score reduction.
        additive = bias.to(device=queries.device, dtype=queries.dtype).float()
    if ((labels < -1) | (labels >= groups) | (allowed & (labels < 0))).any():
        raise ValueError("Every allowed context token must have a valid group label")
    unions = (
        {name: members for name, members in GROUP_UNIONS.items() if all(m in names for m in members)}
        if group_unions is None
        else dict(group_unions)
    )
    if any(
        name in names
        or not members
        or len(set(members)) != len(members)
        or any(m not in names for m in members)
        for name, members in unions.items()
    ):
        raise ValueError("Group unions require distinct existing members and a new name")

    # Excluded keys use one extra scatter bin, discarded from every statistic.
    labels = torch.where(allowed, labels, groups)
    counts = torch.zeros((b, groups + 1), device=queries.device, dtype=torch.float32)
    counts.scatter_add_(1, labels, allowed.float())
    counts = counts[:, :groups]
    total_count = counts.sum(-1)
    row_valid = (total_count > 0)[:, None, None]
    shape = (b, heads, query_count, groups + 1)
    maximum = torch.full(shape, -torch.inf, device=queries.device)
    weight_sum = torch.zeros(shape, device=queries.device)
    shifted_moment = torch.zeros(shape, device=queries.device)

    with torch.autocast(queries.device.type, enabled=False):
        q = queries.detach().float()
        for start in range(0, k_count, key_chunk_size):
            stop = min(start + key_chunk_size, k_count)
            keep = allowed[:, start:stop]
            key = keys[:, :, start:stop].detach().float().masked_fill(~keep[:, None, :, None], 0)
            scores = (q @ key.transpose(-2, -1)) / math.sqrt(width)
            scores = (scores + additive[:, None, None, start:stop]).masked_fill(
                ~keep[:, None, None, :], -torch.inf
            )
            index = labels[:, None, None, start:stop].expand_as(scores)
            local_max = torch.full_like(maximum, -torch.inf)
            local_max.scatter_reduce_(-1, index, scores, reduce="amax", include_self=True)
            shift = scores - local_max.gather(-1, index)
            shift = torch.where(keep[:, None, None, :], shift, 0)
            weights = shift.exp().masked_fill(~keep[:, None, None, :], 0)
            local_sum = torch.zeros_like(weight_sum).scatter_add_(-1, index, weights)
            local_moment = torch.zeros_like(shifted_moment).scatter_add_(-1, index, weights * shift)

            combined_max = torch.maximum(maximum, local_max)
            old_shift = torch.where(torch.isfinite(maximum), maximum - combined_max, 0)
            new_shift = torch.where(torch.isfinite(local_max), local_max - combined_max, 0)
            old_scale = old_shift.exp().masked_fill(~torch.isfinite(maximum), 0)
            new_scale = new_shift.exp().masked_fill(~torch.isfinite(local_max), 0)
            shifted_moment = old_scale * (shifted_moment + weight_sum * old_shift) + new_scale * (
                local_moment + local_sum * new_shift
            )
            weight_sum = old_scale * weight_sum + new_scale * local_sum
            maximum = combined_max

        maximum, weight_sum, shifted_moment = (
            value[..., :groups] for value in (maximum, weight_sum, shifted_moment)
        )
        log_weight = weight_sum.clamp_min(torch.finfo(torch.float32).tiny).log()
        log_partition = maximum + log_weight
        within_entropy = (log_weight - shifted_moment / weight_sum.clamp_min(1e-30)).clamp_min(0)
        within_entropy = within_entropy.masked_fill(weight_sum == 0, 0)
        total_partition = torch.logsumexp(log_partition, -1, keepdim=True)
        mass = (log_partition - total_partition).exp().masked_fill(~row_valid[..., None], 0)
        mass = mass.masked_fill(weight_sum == 0, 0)

        def entropy_of(selected):
            selected_log = log_partition[..., selected]
            selected_total = torch.logsumexp(selected_log, -1, keepdim=True)
            relative = (selected_log - selected_total).exp()
            relative = relative.masked_fill(~torch.isfinite(selected_log), 0)
            log_relative = torch.where(relative > 0, selected_log - selected_total, 0)
            return (relative * (within_entropy[..., selected] - log_relative)).sum(-1).clamp_min(0)

        all_entropy = entropy_of(list(range(groups)))
        all_normalized = all_entropy / total_count.clamp_min(2).log()[:, None, None]
        all_normalized = all_normalized.masked_fill((total_count <= 1)[:, None, None], 0).clamp(0, 1)
        result = {"entropy/all": _mean_over(all_normalized, row_valid)}
        selections = {name: [i] for i, name in enumerate(names)}
        selections.update(
            {name: [names.index(member) for member in members] for name, members in unions.items()}
        )
        # One small transfer establishes which group metrics are defined. No
        # per-key or per-chunk CPU synchronization is performed.
        present = counts.sum(0).gt(0).cpu().tolist()
        for name, selected in selections.items():
            group_count = counts[:, selected].sum(-1)
            group_mass = mass[..., selected].sum(-1)
            uniform = group_count / total_count.clamp_min(1)
            result[f"mass/{name}"] = _mean_over(group_mass, row_valid)
            result[f"uniform_mass/{name}"] = _mean_over(uniform[:, None, None], row_valid)
            if not any(present[i] for i in selected):
                continue
            group_valid = (group_count > 0)[:, None, None]
            result[f"relative_mass/{name}"] = _mean_over(
                group_mass / uniform.clamp_min(1e-30)[:, None, None], group_valid
            )
            entropy = entropy_of(selected) / group_count.clamp_min(2).log()[:, None, None]
            entropy = entropy.masked_fill((group_count <= 1)[:, None, None], 0).clamp(0, 1)
            result[f"entropy/{name}"] = _mean_over(entropy, group_valid)
    return result


def _python_scalars(values: dict[str, Tensor]) -> dict[str, float]:
    if not values:
        return {}
    names = list(values)
    scalars = torch.stack([values[name].detach().float() for name in names]).cpu().tolist()
    return dict(zip(names, scalars))


class CrossAttentionDiagnostics:
    """Observe one chosen decoder evaluation, without replacing SDPA or weights.

    Example: ``capture_on_call=8`` observes the final evaluation in an eight-step
    Euler integration (u=7/8). All heads, action queries and supplied examples
    contribute equally. Metrics are pooled equally over decoder layers; optional
    per-layer summaries are available without additional attention computation.
    """

    def __init__(
        self,
        decoder,
        group_ids: Tensor,
        group_names: Sequence[str] = GROUP_NAMES,
        *,
        capture_on_call: int = 1,
        key_chunk_size: int = 2048,
    ):
        if not isinstance(capture_on_call, int) or isinstance(capture_on_call, bool) or capture_on_call < 1:
            raise ValueError("capture_on_call must be a positive integer")
        self.decoder = decoder
        self.group_ids, self.group_names = group_ids, tuple(group_names)
        self.capture_on_call, self.key_chunk_size = capture_on_call, key_chunk_size
        self._handles = []
        self._state = {}
        self._layers: dict[int, dict[str, Tensor]] = {}

    def __enter__(self):
        if self._handles:
            raise RuntimeError("Attention capture is already active")
        self._state.clear()
        self._layers.clear()
        try:
            for index, layer in enumerate(self.decoder.layers):
                attention = layer.cross_attn
                self._state[index] = {"calls": 0, "active": False}

                def before(module, args, index=index):
                    state = self._state[index]
                    state["calls"] += 1
                    state["active"] = state["calls"] == self.capture_on_call

                def query(module, args, output, index=index):
                    if self._state[index]["active"]:
                        self._state[index]["q"] = output.detach()

                def key(module, args, output, index=index):
                    if self._state[index]["active"]:
                        self._state[index]["k"] = output[..., : output.shape[-1] // 2].detach()

                def after(module, args, output, index=index):
                    state = self._state[index]
                    if not state["active"]:
                        return
                    if args[1].shape[1] == 0:
                        # Attention.forward returns zero before its projections
                        # for an empty context. There is no attention to invent.
                        q, k = torch.zeros_like(args[0]), args[1]
                    else:
                        q, k = state.pop("q"), state.pop("k")
                    b, _, width = q.shape
                    q = q.reshape(b, -1, module.heads, width // module.heads).transpose(1, 2)
                    k = k.reshape(b, -1, module.heads, width // module.heads).transpose(1, 2)
                    bias = args[2] if len(args) > 2 else None
                    self._layers[index] = cross_attention_statistics(
                        q,
                        k,
                        bias,
                        self.group_ids,
                        self.group_names,
                        key_chunk_size=self.key_chunk_size,
                    )

                self._handles.extend(
                    [
                        attention.register_forward_pre_hook(before),
                        attention.q.register_forward_hook(query),
                        attention.kv.register_forward_hook(key),
                        attention.register_forward_hook(after),
                    ]
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._state.clear()
        return False

    def layer_metrics(self) -> dict[int, dict[str, float]]:
        self._require_complete()
        return {index: _python_scalars(values) for index, values in self._layers.items()}

    def metrics(self) -> dict[str, float]:
        self._require_complete()
        names = set.intersection(*(set(layer) for layer in self._layers.values()))
        return _python_scalars(
            {
                name: torch.stack([layer[name] for layer in self._layers.values()]).mean()
                for name in sorted(names)
            }
        )

    def _require_complete(self):
        if not self._layers or len(self._layers) != len(self.decoder.layers):
            raise RuntimeError("Requested attention evaluation was not captured for every decoder layer")
