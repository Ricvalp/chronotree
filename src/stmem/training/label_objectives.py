"""Batched, paired utility measurements for frozen memory-reader experiments.

Velocity labels use a shared noised expert chunk at one flow time. Final-action
labels integrate from the same *unmasked* Gaussian initialization, exactly as
inference does; expert validity masks are used only when measuring discrepancy.
Both objectives compare normalized model outputs and preserve signed gains.
"""

from __future__ import annotations

from contextlib import contextmanager
import time
from types import SimpleNamespace

import torch
from torch.nn.utils.rnn import pad_sequence

from stmem.models.action_flow import flow_sample, masked_mse
from stmem.models.batched_policy import TokenBank
from stmem.training.cached_policy import CachedPolicyForward


OBJECTIVES = ("velocity", "final_action")


def variant_name(objective, integration_steps):
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown label objective: {objective}")
    if not isinstance(integration_steps, int) or isinstance(integration_steps, bool) or integration_steps < 1:
        raise ValueError("Label integration_steps must be a positive integer")
    return "velocity" if objective == "velocity" else f"final_action_{integration_steps}"


def pack_banks(banks):
    """Pad actual evidence without ever truncating full teacher history."""
    if not banks:
        raise ValueError("Cannot pack an empty list of evidence banks")
    return TokenBank(*(pad_sequence([getattr(bank, field)[0] for bank in banks], batch_first=True)
                       for field in ("content", "metadata", "valid", "log_count")))


def policy_context(policy, items, histories):
    """Use the existing cached policy's adapter, mass bias and current boundary."""
    batch = SimpleNamespace(
        positions=range(len(items)), current=pack_banks([item["current"] for item in items]),
        history=pack_banks(histories), text=torch.stack([item["text"] for item in items]),
        current_state=torch.stack([item["current_state"] for item in items]),
        clocks=torch.stack([torch.as_tensor(item["clocks"], dtype=torch.float32) for item in items]),
        history_leaf_counts=torch.tensor([item["history_leaves"] for item in items]),
    )
    if policy.device.type == "cuda":
        for bank in (batch.current, batch.history):
            for name in ("content", "metadata", "valid", "log_count"):
                setattr(bank, name, getattr(bank, name).pin_memory())
        for name in ("text", "current_state", "clocks", "history_leaf_counts"):
            setattr(batch, name, getattr(batch, name).pin_memory())
    # All inference models are frozen, including the teacher's encoders. The
    # wrapper changes neither weights nor the forward definition.
    return CachedPolicyForward(policy).context(batch)[:2]


@contextmanager
def _timed(timings, key, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    yield
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings[key] = timings.get(key, 0.0) + time.perf_counter() - started


def _teacher_groups(items, token_budget):
    """Length-sort locally and bound padded tokens; oversize examples run alone."""
    group, maximum = [], 0
    for index in sorted(range(len(items)), key=lambda i: items[i]["full"].content.shape[1]):
        length = items[index]["full"].content.shape[1] + items[index]["current"].content.shape[1] + 3
        if group and (len(group) + 1) * max(maximum, length) > token_budget:
            yield group
            group, maximum = [], 0
        group.append(index)
        maximum = max(maximum, length)
    if group:
        yield group


@torch.inference_mode()
def evaluate_batch(teacher, compact, items, *, objectives=OBJECTIVES, integration_steps=8,
                   teacher_token_budget=32768, compact_batch_size=64):
    """Return ``{objective: {reader_key: [gain vectors by item]}}`` and timings.

    Initial reader keys are tree/flat and used by both objectives. Refresh keys
    are objective/tree or objective/flat; each objective then uses its own frozen
    router's selected context. Teacher evaluations remain shared in either case.
    All noises are independently seeded by context ID, not worker/batch order.
    """
    objectives = tuple(objectives)
    if not objectives or len(set(objectives)) != len(objectives):
        raise ValueError("Specify distinct label objectives")
    for objective in objectives:
        variant_name(objective, integration_steps)
    if teacher_token_budget < 1 or compact_batch_size < 1:
        raise ValueError("Inference batch limits must be positive")
    if not items:
        return {}, {}
    device = teacher.device
    if compact.device != device:
        raise ValueError("Paired teacher and compact policy must use the same device")
    if teacher.training or compact.training:
        raise ValueError("Utility labels require deterministic eval-mode policies")
    for field in ("mean", "std"):
        if not torch.equal(getattr(teacher.normalizer, field), getattr(compact.normalizer, field)):
            raise ValueError("Teacher and compact action normalization differs")
    noises, flow_times = [], []
    for item in items:
        generator = torch.Generator(device=device).manual_seed(int(item["seed"]))
        noises.append(torch.randn(item["target"].shape, device=device, generator=generator))
        flow_times.append(torch.rand((), device=device, generator=generator))
    noise, u = torch.stack(noises), torch.stack(flow_times)
    actions = torch.stack([item["target"] for item in items]).to(device)
    valid = torch.stack([item["valid"] for item in items]).to(device).bool()
    x, _, _, _ = flow_sample(actions, valid, teacher.normalizer, noise=noise, time=u)
    references = {objective: torch.empty_like(noise) for objective in objectives}
    timings = {}
    with _timed(timings, "teacher_seconds", device):
        for indices in _teacher_groups(items, teacher_token_budget):
            selected = [items[i] for i in indices]
            context, bias = policy_context(teacher, selected, [item["full"] for item in selected])
            for objective in objectives:
                if objective == "velocity":
                    prediction = teacher.decoder(x[indices], u[indices], context, bias)
                else:
                    prediction = teacher.decoder.integrate(context, bias, steps=integration_steps, noise=noise[indices])
                references[objective][indices] = prediction.float()
            del context, bias
    output = {objective: {} for objective in objectives}
    entries = []
    # Pack before and counterfactual contexts together. Context adaptation is
    # shared by the two objectives, but never between teacher and compact weights.
    for owner, item in enumerate(items):
        for key, reader in item["readers"].items():
            applicable = [obj for obj in objectives if "/" not in key or key.startswith(obj + "/")]
            if not applicable or not len(reader["after"]):
                continue
            if len(reader["after"]) != len(reader["costs"]) or torch.any(reader["costs"] <= 0):
                raise ValueError("Candidate refinements must have strictly positive token costs")
            for obj in applicable:
                output[obj].setdefault(key, [None] * len(items))
            for move, bank in enumerate([reader["before"], *reader["after"]]):
                entries.append((owner, key, move, bank, applicable))
    errors = {}
    with _timed(timings, "compact_seconds", device):
        # Group by applicability so refresh never evaluates irrelevant objectives.
        entries.sort(key=lambda entry: (tuple(entry[4]), entry[3].content.shape[1]))
        start = 0
        while start < len(entries):
            end = min(start + compact_batch_size, len(entries))
            while end > start + 1 and entries[end - 1][4] != entries[start][4]:
                end -= 1
            batch = entries[start:end]
            owners = [entry[0] for entry in batch]
            context, bias = policy_context(compact, [items[i] for i in owners], [entry[3] for entry in batch])
            for objective in batch[0][4]:
                if objective == "velocity":
                    prediction = compact.decoder(x[owners], u[owners], context, bias)
                else:
                    prediction = compact.decoder.integrate(context, bias, steps=integration_steps, noise=noise[owners])
                losses = masked_mse(prediction, references[objective][owners], valid[owners], reduction="none")
                for entry, loss in zip(batch, losses):
                    errors[objective, entry[0], entry[1], entry[2]] = loss
            start = end
    for objective, readers in output.items():
        for key, values in readers.items():
            for owner, item in enumerate(items):
                reader = item["readers"].get(key)
                if reader is None or not len(reader["after"]):
                    continue
                base = errors[objective, owner, key, 0]
                after = torch.stack([errors[objective, owner, key, move + 1] for move in range(len(reader["after"]))])
                gains = (base - after) / reader["costs"].to(device).float()
                if not torch.isfinite(gains).all():
                    raise FloatingPointError("Nonfinite paired utility gains")
                values[owner] = gains.cpu()
    return output, timings
