"""Observable packet scoring shared by expert-label fitting and deployment.

The packet constructor is the one used by the student utility study.  No expert
actions, flow inputs, losses, task IDs or eventual episode lengths enter these
features.  Unlike the original fixed-query router, its small query is refreshed
after each allocation because it includes the selected representation summary.
Its scans, packet proposals and repeated scoring are explicitly counted.
"""
from __future__ import annotations

from collections import defaultdict
import random
import time

import numpy as np
import torch
from torch import nn

from stmem.config import digest
from stmem.models.metadata_adapter import METADATA_DIM
from stmem.models.router import ArchitectureRouter


FEATURE_SCHEMA = "observable_expert_packet_router_v1"
CONTROLLER_SCHEMA = "equal_cost_indexed_packet_controller_v1"


def feature_spec(width):
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("Router feature width must be positive")
    return {
        "schema": FEATURE_SCHEMA,
        "width": width,
        "query_dim": 4 * width + 6,
        "candidate_dim": 2 * width + METADATA_DIM + 2,
        "query": ["current_content_mean", "task_embedding", "selected_content_mean",
                  "selected_content_max", "conditioning_clock/100", "execution_clock/100",
                  "selected_record_count/budget", "remaining_count/budget", "budget/128", "packet_cost/8"],
        "candidate": ["region_content_mean", "region_content_max", "packet_cost/8",
                      "packet_cost/budget", "immutable_geometry_time_type_metadata"],
        "context_summary": "unweighted selected-record contents, including exact residual records",
    }


class UtilityRouter(ArchitectureRouter):
    """The existing matched interaction MLP, with explicit new observable inputs."""

    def __init__(self, width=256, key_width=128):
        super().__init__(width, key_width, architecture="mlp")
        spec = feature_spec(width)
        self.query_dim = spec["query_dim"]
        self.candidate_dim = spec["candidate_dim"]
        self.query = nn.Sequential(nn.Linear(self.query_dim, key_width), nn.SiLU(),
                                   nn.Linear(key_width, key_width), nn.LayerNorm(key_width))
        self.key = nn.Sequential(nn.Linear(self.candidate_dim, key_width), nn.SiLU(),
                                 nn.Linear(key_width, key_width), nn.LayerNorm(key_width))
        self.feature_schema = FEATURE_SCHEMA
        self.key_width = key_width


def build_scorer(width, key_width=128):
    return UtilityRouter(width, key_width)


def _bank_contents(bank, width):
    if bank.content.ndim != 3 or bank.content.shape[0] != 1 or bank.content.shape[-1] != width:
        raise ValueError("Router features require one unbatched token bank")
    if bank.valid.shape != bank.content.shape[:2] or bank.valid.dtype != torch.bool:
        raise ValueError("Token validity does not match observable contents")
    contents = bank.content.detach().float().cpu()[0][bank.valid.detach().cpu()[0]]
    if not torch.isfinite(contents).all():
        raise ValueError("Nonfinite observed token content")
    return contents


def query_features(current, text, selected, clocks, *, width, budget, packet_cost):
    """Create only observable inputs; future-target-bearing items are not accepted."""
    if budget < 1 or not 1 <= packet_cost <= budget:
        raise ValueError("Query requires positive budget and feasible cost")
    scene, history = _bank_contents(current, width), _bank_contents(selected, width)
    if len(history) + packet_cost > budget:
        raise ValueError("Packet would exceed the declared historical budget")
    text = torch.as_tensor(text).detach().float().cpu().reshape(-1)
    clocks = torch.as_tensor(clocks).detach().float().cpu().reshape(-1)
    if text.shape != (width,) or clocks.shape != (2,):
        raise ValueError("Task embedding/source clocks have incorrect shape")
    zeros = torch.zeros(width)
    values = torch.cat((scene.mean(0) if len(scene) else zeros, text,
                        history.mean(0) if len(history) else zeros,
                        history.amax(0) if len(history) else zeros,
                        clocks / 100,
                        torch.tensor([len(history) / budget, (budget - len(history)) / budget,
                                      budget / 128, packet_cost / 8])))
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite observable query features")
    return values


def candidate_features(columns, *, width, cost, budget):
    """Identical online/offline region keys; relative ages remain query-dependent."""
    from stmem.training.label_data import candidate_features as original_features

    if budget < 1 or not 1 <= cost <= budget:
        raise ValueError("Candidate requires positive feasible packet cost")
    original = original_features(columns, width).detach().float().cpu()
    costs = torch.tensor([cost / 8, cost / budget]).expand(len(original), 2)
    # Keep metadata last for the inherited metadata-bias head.
    result = torch.cat((original[:, :2 * width], costs, original[:, 2 * width:]), -1)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite observable candidate features")
    return result


def _proposal_seed(seed, context, mode):
    return int(digest([CONTROLLER_SCHEMA, int(seed), context["context_id"], mode])[:15], 16)


def propose_packets(archive, context, budget, *, seed=0, candidate_limit=8,
                    packet_max_cost=8, mode="packet"):
    """A bounded, gain-blind source-balanced pool of equal actual-cost candidates.

    Both credit objectives use the same packet pool. A separate local controller
    proposes positive-cost single moves, so its comparison is explicitly a
    controller intervention rather than a supervision-only comparison.
    """
    if candidate_limit < 1 or packet_max_cost < 1 or mode not in ("local", "packet"):
        raise ValueError("Invalid proposal limit, packet size or controller mode")
    context = archive.validate(context)
    target = min(int(budget), len(archive.leaves))
    if context["token_count"] > target:
        raise ValueError("Minimum representation exceeds requested budget")
    started = time.perf_counter()
    remaining = target - context["token_count"]
    stats = {"index_rows_scanned": 0, "eligible_candidates": 0, "candidate_proposals": 0,
             "packet_moves_constructed": 0, "descendant_supports_examined": 0,
             "proposal_seconds": 0.0}
    if not remaining:
        return [], stats
    q = 1 if mode == "local" else min(packet_max_cost, remaining)
    requested = q
    while True:
        rows = archive.positive_moves(context) if mode == "local" else archive.feasible_regions(context, q)
        stats["index_rows_scanned"] += (len(context["rows"]) if context["reader"] == "TREE"
                                        else len(archive.active) + len(archive.leaves))
        if rows or q == 1:
            break
        q -= 1
    if not rows:
        raise ValueError("No feasible positive-cost proposal before attainable target")
    grouped = defaultdict(list)
    for row in rows:
        grouped[archive.groups[row]].append(row)
    rng = random.Random(_proposal_seed(seed, context, mode))
    # Select a group uniformly, then a unique region within it. Smaller groups
    # do not disappear behind the much larger execution-visual pool.
    pools = {group: sorted(members, key=lambda row: archive.names[row])
             for group, members in grouped.items()}
    chosen = []
    while pools and len(chosen) < candidate_limit:
        group = rng.choice(sorted(pools))
        members = pools[group]
        chosen.append(members.pop(rng.randrange(len(members))))
        if not members:
            del pools[group]
    entries = []
    for row in chosen:
        # Count repeated support traversals inside the shared constructor too,
        # rather than presenting only the final packet's leaf mass as work.
        support_method = archive.support
        had_override = "support" in archive.__dict__
        def counted_support(node):
            members = support_method(node)
            stats["descendant_supports_examined"] += len(members)
            return members
        archive.support = counted_support
        try:
            if mode == "local":
                final = archive.move(context, row)
                support = archive.support(row)
                packet = {"region": row, "region_id": archive.names[row], "group": list(archive.groups[row]),
                          "support_count": len(support), "cost": 1, "moves": [row],
                          "prefixes": {1: final}, "final": final, "balancing": archive.balancing}
            else:
                packet = archive.packet(context, row, q)
        finally:
            if had_override:
                archive.support = support_method
            else:
                del archive.support
        if packet["final"]["token_count"] != context["token_count"] + q:
            raise AssertionError("Packet proposal has unequal actual cost")
        packet["candidate_id"] = digest([context["context_id"], packet["region_id"],
                                          [archive.names[move] for move in packet["moves"]]])[:24]
        packet["requested_cost"] = requested
        entries.append(packet)
        stats["packet_moves_constructed"] += len(packet["moves"])
    stats.update(eligible_candidates=len(rows), candidate_proposals=len(entries),
                 proposal_seconds=time.perf_counter() - started)
    return entries, stats


def route(archive, reader, budget, scorer, *, seed=0, candidate_limit=8, packet_max_cost=8,
          mode="packet"):
    """Construct a deployable exact-coverage route and report all proposal work.

    ``scorer(context, packets)`` returns finite scores from observable features.
    Its model/transfer costs can be accumulated separately by the caller. Every
    visited context and final count are validated, including negative-score runs.
    """
    started = time.perf_counter()
    current = archive.coarse(reader)
    target = min(int(budget), len(archive.leaves))
    if current["token_count"] > target:
        raise ValueError("All causal roots/residuals must fit the target budget")
    steps, counts = [], {"index_rows_scanned": 0, "eligible_candidates": 0, "candidate_proposals": 0,
                         "packet_moves_constructed": 0, "descendant_supports_examined": 0,
                         "proposal_seconds": 0., "scoring_seconds": 0., "scorer_calls": 0,
                         "candidates_scored": 0}
    while current["token_count"] < target:
        entries, work = propose_packets(archive, current, budget, seed=seed,
                                       candidate_limit=candidate_limit, packet_max_cost=packet_max_cost, mode=mode)
        for key, value in work.items():
            counts[key] += value
        begin = time.perf_counter()
        values = np.asarray(scorer(current, entries), dtype=np.float64)
        counts["scoring_seconds"] += time.perf_counter() - begin
        if values.shape != (len(entries),) or not np.isfinite(values).all():
            raise ValueError("Router scores must contain one finite value per proposal")
        selected = min(range(len(entries)), key=lambda i: (-values[i], entries[i]["candidate_id"]))
        chosen = entries[selected]
        counts["scorer_calls"] += 1
        counts["candidates_scored"] += len(entries)
        steps.append({"before": current["context_id"], "before_count": current["token_count"],
                      "candidate_id": chosen["candidate_id"], "cost": chosen["cost"],
                      "eligible": work["eligible_candidates"], "proposed": len(entries),
                      "moves": [archive.names[row] for row in chosen["moves"]]})
        current = archive.validate(chosen["final"])
    counts["route_seconds"] = time.perf_counter() - started
    return current, {"schema": CONTROLLER_SCHEMA, "mode": mode, "budget": budget,
                     "actual_count": current["token_count"], "saturated": target < budget,
                     "coverage_hash": archive.coverage, "steps": steps, **counts}
