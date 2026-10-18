"""Bounded frozen-checkpoint memory sensitivity and tiny exact routing diagnostics.

This command never launches training, Slurm jobs, or a benchmark campaign.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random
import time
from typing import Iterable

import torch

from stmem.memory.flat import FlatState
from stmem.memory.frontier import split_frontier, select_uniform
from stmem.memory.hierarchy import Archive, stable_id
from stmem.memory.records import MemoryRecord

INTERVENTIONS = ("none", "erase_execution", "permute_execution", "erase_conditioning")


def intervene_evidence(evidence, mode: str, seed: int = 0):
    """Return an evidence intervention while preserving current input and clocks.

    The permutation swaps execution leaf payloads across their recorded-time
    slots, separately within each modality. It is an encoded-evidence sensitivity
    diagnostic, not a claim that the resulting observations form a physically
    possible trajectory. Simply permuting token order would be ineffectual for
    this attention policy. Conditioning is held fixed for execution interventions.
    """
    if mode not in INTERVENTIONS:
        raise ValueError(f"Unknown memory intervention {mode}")
    if mode == "none":
        return evidence
    archive = evidence.archive
    mapping = {}
    if mode == "permute_execution":
        groups = defaultdict(list)
        for leaf in archive.leaves:
            if leaf.source == "execution":
                groups[leaf.modality].append(leaf)
        rng = random.Random(seed)
        for modality in sorted(groups):
            slots = sorted(groups[modality], key=lambda r: (r.time_anchor, r.node_id))
            payloads = slots.copy()
            rng.shuffle(payloads)
            for slot, payload in zip(slots, payloads):
                # IDs change when payload changes so frozen key caches cannot
                # accidentally reuse unperturbed evidence under the same key.
                node_id = stable_id(f"permuted:{seed}", (slot.node_id, payload.node_id))
                mapping[slot.node_id] = replace(
                    slot,
                    node_id=node_id,
                    content_sum=payload.content_sum,
                    max_content=payload.max_content,
                    anchor=payload.anchor,
                    bounds_min=payload.bounds_min,
                    bounds_max=payload.bounds_max,
                    occupancy=payload.occupancy,
                )
    erased_source = {"erase_execution": "execution", "erase_conditioning": "conditioning"}.get(mode)
    changed = Archive()
    for block_id, leaves in archive.completed_blocks.items():
        keep = [mapping.get(r.node_id, r) for r in leaves if r.source != erased_source]
        if keep:
            changed.add_block(keep, block_id)
    for source, leaves in archive.pending.items():
        if source != erased_source:
            changed.set_pending(source, [mapping.get(r.node_id, r) for r in leaves])
    return replace(evidence, archive=changed)


class SearchSpaceExceeded(RuntimeError):
    pass


def _check_size(size: int, limit: int) -> None:
    if size > limit:
        raise SearchSpaceExceeded(
            f"Exact enumeration exceeds max_states={limit}; choose a smaller causal prefix "
            "or explicitly raise the diagnostic cap. No oracle result was produced."
        )


def exhaustive_tree_frontiers(roots: Iterable[MemoryRecord], budget: int, *, max_states: int = 4096):
    """Enumerate every valid frontier at min(B, N); refuse an incomplete search."""
    roots = list(roots)
    if budget < len(roots):
        raise ValueError("Budget cannot cover all source/modality roots")
    target = min(budget, sum(r.count for r in roots))
    states = {tuple(sorted(r.node_id for r in roots)): roots}
    visited = len(states)
    _check_size(visited, max_states)
    size = len(roots)
    while size < target:
        following = {}
        for frontier in states.values():
            for record in frontier:
                if not record.children:
                    continue
                refined = split_frontier(frontier, record)
                key = tuple(sorted(r.node_id for r in refined))
                if key not in following:
                    following[key] = refined
                    _check_size(visited + len(following), max_states)
        if not following:
            raise AssertionError("A full-coverage tree exhausted splits below its leaf count")
        visited += len(following)
        states = following
        size += 1
    return list(states.values())


def exhaustive_flat_contexts(leaves: Iterable[MemoryRecord], budget: int, *, max_states: int = 4096):
    """Enumerate exact selected-plus-residual contexts with singleton accounting."""
    initial = FlatState(tuple(leaves)).normalize_singletons()
    if budget < initial.token_count:
        raise ValueError("Budget cannot cover all source/modality residuals")
    target = min(budget, len(initial.leaves))
    states = {tuple(sorted(initial.selected_ids)): initial}
    visited = len(states)
    _check_size(visited, max_states)
    size = initial.token_count
    while size < target:
        following = {}
        for state in states.values():
            for leaf in state.unselected:
                if state.move_cost(leaf) == 0:
                    raise AssertionError("Singleton normalization left a zero-cost move")
                moved = state.move(leaf).normalize_singletons()
                key = tuple(sorted(moved.selected_ids))
                if key not in following:
                    following[key] = moved
                    _check_size(visited + len(following), max_states)
        if not following:
            raise AssertionError("Flat enumeration exhausted moves below leaf count")
        visited += len(following)
        states = following
        size += 1
    return [state.records for state in states.values()]


@torch.no_grad()
def routing_diagnostic(
    teacher, compact, decision, *, budget, max_leaves=24, max_states=4096, seed=0, learned_routers=None
):
    from stmem.models.action_flow import flow_sample, masked_mse
    from stmem.models.policy import choose_records
    from stmem.models.router import CachedScorer

    if teacher.training or compact.training:
        raise ValueError("Routing diagnostics require deterministic eval-mode checkpoints")
    evidence = teacher.encode(decision)
    if len(evidence.archive.leaves) > max_leaves:
        raise ValueError(
            f"Archive has {len(evidence.archive.leaves)} leaves, exceeding the explicit tiny-diagnostic "
            f"cap {max_leaves}. Select an earlier decision; history is never silently shortened."
        )
    generator = torch.Generator(device=teacher.device).manual_seed(seed)
    x, u, _, _ = flow_sample(
        decision.target_chunk.to(teacher.device),
        decision.target_valid.to(teacher.device),
        teacher.normalizer,
        generator=generator,
    )
    reference = teacher(None, x, u, evidence=evidence)
    valid = decision.target_valid.to(teacher.device)

    def disagreement(records):
        return float(masked_mse(compact(None, x, u, evidence=evidence, records=records), reference, valid))

    started = time.perf_counter()
    families = {
        "tree": exhaustive_tree_frontiers(evidence.archive.roots, budget, max_states=max_states),
        "flat": exhaustive_flat_contexts(evidence.archive.leaves, budget, max_states=max_states),
    }
    summary = {
        "diagnostic": "exact tiny-archive teacher velocity agreement; not closed-loop success or a control oracle",
        "seed": seed,
        "budget": budget,
        "leaves": len(evidence.archive.leaves),
        "roots": len(evidence.archive.roots),
        "decision_index": decision.index,
        "teacher_flow_time": float(u),
        "compact_full_disagreement": disagreement(evidence.archive.leaves),
        "uniform_disagreement": disagreement(select_uniform(evidence.archive.roots, budget).records),
        "families": {},
    }
    for family, contexts in families.items():
        values = [disagreement(records) for records in contexts]
        best = min(range(len(values)), key=lambda i: values[i])
        summary["families"][family] = {
            "enumerated_contexts": len(contexts),
            "exact": True,
            "best_disagreement": values[best],
            "mean_disagreement": sum(values) / len(values),
            "best_node_ids": [r.node_id for r in contexts[best]],
        }
        if learned_routers and family in learned_routers:
            router, checkpoint_hash = learned_routers[family]
            scorer = CachedScorer(router, checkpoint_hash)
            scorer.prepare(evidence.archive.records if family == "tree" else evidence.archive.leaves)
            scorer.set_query(compact.routing_query(evidence, budget))
            selected = choose_records(evidence, family, budget, scorer=scorer)
            summary["families"][family]["learned_disagreement"] = disagreement(selected.records)
            summary["families"][family]["learned_stats"] = selected.stats
    summary["seconds"] = time.perf_counter() - started
    return summary


@torch.no_grad()
def memory_sensitivity(policy, decision, *, budget, seed=0):
    """Cheap fixed-noise action changes; success sensitivity requires val rollouts."""
    policy.eval()
    evidence = policy.encode(decision)
    generator = torch.Generator(device=policy.device).manual_seed(seed)
    noise = torch.randn(policy.config["model"]["horizon"], 8, device=policy.device, generator=generator)
    results = {}
    reference = None
    for mode in INTERVENTIONS:
        changed = intervene_evidence(evidence, mode, seed)
        selected = select_uniform(changed.archive.roots, budget)
        prediction = policy.predict(changed, selected.records, noise=noise)
        if reference is None:
            reference = prediction
        results[mode] = {
            "command_mse_change": float((prediction - reference).square().mean()),
            "active_tokens": len(selected),
            "archive_leaves": len(changed.archive.leaves),
        }
    return {
        "diagnostic": "fixed-noise action sensitivity on one available prefix; not task success or proof of memory dependence",
        "reader": "UNIFORM",
        "seed": seed,
        "decision_index": decision.index,
        "budget": budget,
        "permutation": "execution leaf payloads across time slots within modality",
        "interventions": results,
    }


def main():
    from stmem.config import load_config, dependency
    from stmem.training.data import TrainingData
    from stmem.training.runtime import load_policy, load_router
    from stmem.training.checkpoints import atomic_json, file_sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["routing", "memory"], required=True)
    parser.add_argument("--teacher")
    parser.add_argument("--compact")
    parser.add_argument("--tree-router")
    parser.add_argument("--flat-router")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--decision-index", type=int, default=1)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--max-leaves", type=int, default=24)
    parser.add_argument("--max-states", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = TrainingData(cfg)
    episode = data.episode(args.episode_index)
    decision = episode.decision(args.decision_index, cfg["model"]["horizon"])
    compact_path = args.compact or dependency(cfg, "compact")
    compact, _ = load_policy(compact_path, device)
    compact.eval().requires_grad_(False)
    budget = args.budget if args.budget is not None else cfg["model"]["budget"]
    if args.mode == "memory":
        summary = memory_sensitivity(compact, decision, budget=budget, seed=args.seed)
    else:
        teacher_path = args.teacher or dependency(cfg, "teacher")
        teacher, _ = load_policy(teacher_path, device)
        teacher.eval().requires_grad_(False)
        routers = {}
        for family, checkpoint in (("tree", args.tree_router), ("flat", args.flat_router)):
            if checkpoint:
                router, _ = load_router(checkpoint, device)
                routers[family] = (router, file_sha256(checkpoint))
        summary = routing_diagnostic(
            teacher,
            compact,
            decision,
            budget=budget,
            max_leaves=args.max_leaves,
            max_states=args.max_states,
            seed=args.seed,
            learned_routers=routers,
        )
        summary["teacher_sha256"] = file_sha256(teacher_path)
    summary["task"] = episode.task
    summary["episode_id"] = episode.episode_id
    summary["compact_sha256"] = file_sha256(compact_path)
    summary["synthetic"] = bool(cfg["data"].get("synthetic", False))
    atomic_json(summary, Path(args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
