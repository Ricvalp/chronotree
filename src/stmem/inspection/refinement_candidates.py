"""Gain-blind, exact-coverage allocation constructions for the utility study.

This module operates on the existing indexed causal hierarchy, not on raw scenes
or model predictions. The same packet constructor serves training, diagnostic
panels, and callable-score controllers. Contexts contain only immutable row IDs;
callers hydrate them through ``PackedContextReader`` and its existing residual
aggregation. No hidden labels, future actions, or measured gains are accepted.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from numbers import Integral
import random

import numpy as np

from stmem.inspection.cue_frontiers import _active_tree
from stmem.memory.hierarchy import stable_id

RECIPE = "causal_positive_cost_packets_v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _positive(value, name):
    if not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class IndexedArchive:
    """Validated view of just the forest available at one causal decision.

    ``index`` uses the hierarchy sidecar's row numbering. ``leaves`` may retain
    its canonical payload/reduction order; its set must match the supplied roots.
    Optional ``anchor`` and ``time_anchor`` arrays enable explicit space-time
    balanced FLAT ordering. Without them, ordered subtree interleaving is the
    recorded topology-only fallback, never disguised as geometric sampling.
    """

    def __init__(self, index, roots, leaves=None):
        self.index = index
        (self.children, self.roots, self.active, dfs_leaves, postorder,
         self.parents, self.counts) = _active_tree(index, roots)
        n = len(self.children)
        for key in ("node_id", "source", "modality"):
            if len(index[key]) != n:
                raise ValueError(f"{key} does not match indexed hierarchy size")
        self.names = np.asarray(index["node_id"]).astype(str)
        if len({self.names[row] for row in self.active}) != len(self.active):
            raise ValueError("Active hierarchy IDs must be unique")
        self.groups = {row: (str(index["source"][row]), str(index["modality"][row]))
                       for row in self.active}
        for row in self.active:
            if self.children[row, 0] >= 0:
                if any(self.groups[int(child)] != self.groups[row] for child in self.children[row]):
                    raise ValueError("A hierarchy node crosses source/modality boundaries")
        self.dfs_leaves = tuple(dfs_leaves)
        leaf_position = {row: i for i, row in enumerate(dfs_leaves)}
        self.intervals = {}
        self.depths = {}
        for row in postorder:
            left, right = map(int, self.children[row])
            if left < 0:
                self.intervals[row] = (leaf_position[row], leaf_position[row] + 1)
                self.depths[row] = 0
            else:
                self.intervals[row] = (self.intervals[left][0], self.intervals[right][1])
                self.depths[row] = 1 + max(self.depths[left], self.depths[right])
        if leaves is None:
            leaves = dfs_leaves
        self.leaves = np.asarray(list(leaves), dtype=np.int64)
        if len(set(map(int, self.leaves))) != len(self.leaves) or set(self.leaves) != set(dfs_leaves):
            raise ValueError("Indexed fine rows differ from the exact causal forest")
        self.leaf_set = set(map(int, self.leaves))
        self.leaf_positions = {int(row): i for i, row in enumerate(self.leaves)}
        self.group_names = sorted({self.groups[row] for row in self.leaves})
        code = {group: i for i, group in enumerate(self.group_names)}
        self.group_codes = np.asarray([code[self.groups[row]] for row in self.leaves], dtype=np.int64)
        self.group_leaves = {group: tuple(int(row) for row in self.leaves if self.groups[row] == group)
                             for group in self.group_names}
        self.group_sizes = {group: len(rows) for group, rows in self.group_leaves.items()}
        self.sorted_active = sorted(self.active, key=lambda row: self.names[row])
        self.sorted_leaves = sorted(self.leaf_set, key=lambda row: self.names[row])
        self.interval_array = np.asarray([self.intervals[row] for row in self.sorted_active], dtype=np.int64).reshape(-1, 2)
        self.group_numbers = {group: i for i, group in enumerate(self.group_names)}
        self.node_group_codes = np.asarray(
            [self.group_numbers[self.groups[row]] for row in self.sorted_active], dtype=np.int64)
        self.coverage = _hash(sorted(self.names[row] for row in self.leaves))
        self.balancing = ("observed_space_time_balanced_partition" if "anchor" in index and "time_anchor" in index
                          else "observed_hierarchy_interleaving")
        self._order_cache = {}

    def support(self, row):
        if int(row) not in self.active:
            raise ValueError("Region is outside the causal archive")
        lo, hi = self.intervals[int(row)]
        return self.dfs_leaves[lo:hi]

    def _rows(self, rows, allowed):
        values = list(rows)
        if any(not isinstance(row, Integral) or int(row) not in allowed for row in values):
            raise ValueError("Context has invalid or noncausal rows")
        values = list(map(int, values))
        if len(set(values)) != len(values):
            raise ValueError("Context repeats a record")
        return values

    def context(self, reader, rows):
        reader = str(reader).upper()
        if reader not in ("TREE", "FLAT"):
            raise ValueError("Allocation reader must be TREE or FLAT")
        rows = self._rows(rows, self.active if reader == "TREE" else self.leaf_set)
        if reader == "TREE":
            intervals = sorted(self.intervals[row] for row in rows)
            cursor = 0
            for lo, hi in intervals:
                if lo != cursor:
                    raise ValueError("TREE context overlaps or omits causal leaves")
                cursor = hi
            if cursor != len(self.leaves):
                raise ValueError("TREE context omits causal leaves")
            tokens = len(rows)
        else:
            selected = set(rows)
            # Normalize the identity of singleton residuals without counting a
            # nonexistent extra token or recording a zero-cost promotion.
            selected_counts = Counter(self.groups[row] for row in selected)
            for group, members in self.group_leaves.items():
                if len(members) - selected_counts[group] == 1:
                    singleton = next(row for row in members if row not in selected)
                    selected.add(singleton)
                    selected_counts[group] += 1
            rows = list(selected)
            tokens = len(rows) + sum(selected_counts[group] < size for group, size in self.group_sizes.items())
        rows.sort(key=lambda row: self.names[row])
        ids = [self.names[row] for row in rows]
        return {"reader": reader, "rows": rows, "token_count": tokens,
                "context_id": _hash([reader, self.coverage, ids])[:24], "coverage_hash": self.coverage}

    def validate(self, context):
        """Revalidate a persisted context against this exact causal archive."""
        canonical = self.context(context["reader"], context["rows"])
        for key in ("context_id", "coverage_hash", "token_count"):
            if key in context and context[key] != canonical[key]:
                raise ValueError(f"Context {key} disagrees with causal allocation")
        return canonical

    def records_count(self, context):
        return self.context(context["reader"], context["rows"])["token_count"]

    def coverage_hash(self, context):
        return self.context(context["reader"], context["rows"])["coverage_hash"]

    def flat_arrays(self, context):
        if context["reader"] != "FLAT":
            raise ValueError("flat_arrays requires a FLAT context")
        canonical = self.context("FLAT", context["rows"])
        selected = set(canonical["rows"])
        # An episode can legitimately start without any historical leaves.
        # NumPy otherwise infers float64 for [], which cannot be inverted or
        # used as the Boolean mask required by residual aggregation.
        mask = np.asarray([int(row) in selected for row in self.leaves], dtype=bool)
        return mask, self.group_codes.copy(), list(self.group_names)

    def describe(self, context):
        """Serializable allocation metadata, including actual residual identities."""
        context = self.validate(context)
        result = dict(context, node_ids=[self.names[row] for row in context["rows"]], residuals=[])
        selected = set(context["rows"])
        if context["reader"] == "FLAT":
            for group, leaves in self.group_leaves.items():
                remaining = [row for row in leaves if row not in selected]
                if remaining:
                    names = [self.names[row] for row in remaining]
                    result["residuals"].append({
                        "node_id": names[0] if len(names) == 1 else stable_id("residual", names),
                        "source": group[0], "modality": group[1], "count": len(names),
                        "support_hash": _hash(sorted(names)),
                    })
        counts = [self.counts[row] for row in context["rows"]]
        counts.extend(item["count"] for item in result["residuals"])
        result.update(leaf_fraction=sum(value == 1 for value in counts) / max(1, len(counts)),
                      maximum_record_mass=max(counts, default=0),
                      maximum_depth=max((self.depths[row] for row in context["rows"]), default=0),
                      source_leaf_counts={source: sum(self.groups[row][0] == source for row in self.leaves)
                                          for source in sorted({group[0] for group in self.group_names})})
        return result

    def coarse(self, reader):
        return self.context(reader, self.roots if str(reader).upper() == "TREE" else [])

    def positive_moves(self, context):
        if context["reader"] == "TREE":
            return [row for row in context["rows"] if self.children[row, 0] >= 0]
        selected = set(context["rows"])
        selected_counts = Counter(self.groups[row] for row in selected)
        allowed_groups = {group for group, size in self.group_sizes.items() if size - selected_counts[group] > 1}
        return [row for row in self.sorted_leaves if row not in selected and self.groups[row] in allowed_groups]

    def move(self, context, row):
        row = int(row)
        if context["reader"] == "TREE":
            valid = row in context["rows"] and self.children[row, 0] >= 0
        else:
            selected = set(context["rows"])
            valid = row in self.leaf_set and row not in selected
            if valid:
                group = self.groups[row]
                valid = self.group_sizes[group] - sum(self.groups[value] == group for value in selected) > 1
        if not valid:
            raise ValueError("Move is unavailable or has zero actual token cost")
        if context["reader"] == "TREE":
            rows = [value for value in context["rows"] if value != row] + self.children[row].tolist()
        else:
            rows = list(context["rows"]) + [row]
        after = self.context(context["reader"], rows)
        if after["token_count"] != context["token_count"] + 1:
            raise ValueError("Refinement must add exactly one historical record")
        return after

    def _balanced_order(self, region):
        """Spread observed coordinates/time; stable IDs resolve all equal ties."""
        region = int(region)
        if region in self._order_cache:
            return self._order_cache[region]
        leaves = list(self.support(region))
        if self.balancing == "observed_space_time_balanced_partition":
            ordered = sorted(leaves, key=lambda row: self.names[row])
            xyz = np.asarray(self.index["anchor"], dtype=np.float64)[ordered]
            times = np.asarray(self.index["time_anchor"], dtype=np.float64)[ordered, None]
            points = np.concatenate((xyz, times), axis=1)
            if not np.isfinite(points).all():
                raise ValueError("Observed space-time coordinates must be finite")
            extent = np.ptp(points, axis=0)
            points = (points - points.min(0)) / np.where(extent > 0, extent, 1.0)

            def balanced(indices):
                if len(indices) <= 1:
                    return indices.tolist()
                axis = int(np.argmax(np.ptp(points[indices], axis=0)))
                ranked = indices[np.lexsort((indices, points[indices, axis]))]
                left = balanced(ranked[:len(ranked) // 2])
                right = balanced(ranked[len(ranked) // 2:])
                return [values[i] for i in range(max(len(left), len(right)))
                        for values in (left, right) if i < len(values)]

            leaves = [ordered[i] for i in balanced(np.arange(len(ordered))) ]
        else:
            def interleave(node):
                left, right = map(int, self.children[node])
                if left < 0:
                    return [node]
                a, b = interleave(left), interleave(right)
                return [value for i in range(max(len(a), len(b)))
                        for values in (a, b) if i < len(values) for value in (values[i],)]
            leaves = interleave(region)
        self._order_cache[region] = tuple(leaves)
        return self._order_cache[region]

    def packet(self, context, region, cost):
        """Deterministic coherent completion, with every intermediate count kept."""
        cost = _positive(cost, "Packet cost")
        region = int(region)
        support = set(self.support(region))
        current, moves, prefixes = context, [], {}
        if context["reader"] == "TREE":
            if region not in context["rows"] or self.counts[region] < cost + 1:
                raise ValueError("TREE anchor cannot support the requested packet")
            inside = {region}
            for step in range(1, cost + 1):
                candidates = [row for row in current["rows"] if row in inside and self.children[row, 0] >= 0]
                if not candidates:
                    raise ValueError("TREE packet lacks positive-cost descendants")
                row = min(candidates, key=lambda value: (-self.counts[value], self.names[value]))
                current = self.move(current, row)
                inside.remove(row)
                inside.update(map(int, self.children[row]))
                moves.append(row)
                prefixes[step] = current
        else:
            order = self._balanced_order(region)
            for step in range(1, cost + 1):
                selected = set(current["rows"])
                remaining = self.group_sizes[self.groups[region]] - sum(
                    self.groups[value] == self.groups[region] for value in selected)
                row = next((row for row in order if row not in selected), None) if remaining > 1 else None
                if row is None:
                    raise ValueError("FLAT region lacks positive-cost promotions retaining residual mass")
                current = self.move(current, row)
                moves.append(row)
                prefixes[step] = current
        if any(not set(self.support(row)).issubset(support) for row in moves):
            raise AssertionError("Packet escaped its protected support")
        return {"region": region, "region_id": self.names[region], "group": list(self.groups[region]),
                "support_hash": _hash(sorted(self.names[row] for row in support)),
                "support_count": len(support), "cost": cost, "moves": moves,
                "prefixes": prefixes, "final": current, "balancing": self.balancing}

    def feasible_regions(self, context, cost):
        cost = _positive(cost, "Packet cost")
        if context["reader"] == "TREE":
            return [row for row in context["rows"] if self.counts[row] >= cost + 1]
        selected = set(context["rows"])
        selected_counts = Counter(self.groups[row] for row in selected)
        remaining = np.asarray(
            [self.group_sizes[group] - selected_counts[group] for group in self.group_names], dtype=np.int64)
        # Every subtree is an interval in DFS leaf order. After deleting selected
        # leaves, equal prefix-index intervals identify exactly equal descendant
        # support sets. This deduplicates regions without materializing O(N log N)
        # leaf tuples at every controller decision.
        prefix = np.concatenate(([0], np.cumsum([row not in selected for row in self.dfs_leaves])))
        lower, upper = prefix[self.interval_array[:, 0]], prefix[self.interval_array[:, 1]]
        eligible = (upper - lower >= cost) & (remaining[self.node_group_codes] >= cost + 1)
        seen, result = set(), []
        for i in np.flatnonzero(eligible):
            support = (int(lower[i]), int(upper[i]))
            if support not in seen:
                seen.add(support)
                result.append(self.sorted_active[i])
        return result

    def route(self, reader, budget, seed, *, mode="local", packet_size=8, start=None, scorer=None):
        """Complete roots-to-budget construction; no step skips or discards leaves.

        A callable scorer receives ``(context, packets)`` and returns one finite
        score per feasible equal-cost packet. It is used only by separately
        provenance-tracked deployment, never diagnostic manifest generation.
        Local random priorities persist once assigned to a node/leaf, matching
        the established random single-move rule. Packet region sampling chooses
        a source/modality group uniformly, then a stable ordered eligible region.
        """
        if mode not in ("local", "packet"):
            raise ValueError("Route mode must be local or packet")
        budget = _positive(budget, "Budget")
        packet_size = _positive(packet_size, "Packet size")
        current = self.coarse(reader) if start is None else self.context(reader, start["rows"])
        initial = current["token_count"]
        target = min(budget, len(self.leaves))
        if initial > target:
            raise ValueError("Minimum representation exceeds requested budget")
        rng = random.Random(int(seed))
        contexts, moves, packets, priorities = {initial: current}, [], [], {}
        flat_heap = None
        if mode == "local" and current["reader"] == "FLAT" and scorer is None:
            flat_heap = [(-rng.random(), self.names[row], row) for row in self.positive_moves(current)]
            heapq.heapify(flat_heap)
        while current["token_count"] < target:
            before = current["token_count"]
            if mode == "local":
                if flat_heap is not None:
                    selected = set(current["rows"])
                    while flat_heap and flat_heap[0][2] in selected:
                        heapq.heappop(flat_heap)
                    if not flat_heap:
                        raise ValueError("No positive-cost move before attainable target")
                    row = heapq.heappop(flat_heap)[2]
                    candidates = None
                    candidate_count = len(self.leaves) - len(selected)
                else:
                    candidates = self.positive_moves(current)
                    candidate_count = len(candidates)
                    if not candidates:
                        raise ValueError("No positive-cost move before attainable target")
                    for row in candidates:
                        if row not in priorities:
                            priorities[row] = rng.random()
                if scorer is None and flat_heap is None:
                    row = min(candidates, key=lambda row: (-priorities[row], self.names[row]))
                elif scorer is not None:
                    entries = [{"region": row, "region_id": self.names[row], "cost": 1,
                                "moves": [row], "final": self.move(current, row)} for row in candidates]
                    scores = np.asarray(scorer(current, entries), dtype=float)
                    if scores.shape != (len(entries),) or not np.isfinite(scores).all():
                        raise ValueError("Controller scores must be one finite value per candidate")
                    row = min(range(len(entries)), key=lambda i: (-scores[i], entries[i]["region_id"]))
                    row = candidates[row]
                current = self.move(current, row)
                moves.append(row)
                contexts[current["token_count"]] = current
                packets.append({"start_count": before, "cost": 1, "region_id": self.names[row],
                                "group": list(self.groups[row]), "eligible_pool_size": candidate_count})
            else:
                q = min(packet_size, target - before)
                regions = self.feasible_regions(current, q)
                requested = q
                while not regions and q > 1:
                    q -= 1
                    regions = self.feasible_regions(current, q)
                if not regions:
                    raise ValueError("No feasible positive-cost packet before attainable target")
                grouped = defaultdict(list)
                for row in regions:
                    grouped[self.groups[row]].append(row)
                if scorer is None:
                    group = rng.choice(sorted(grouped))
                    pool = sorted(grouped[group], key=lambda row: self.names[row])
                    region = rng.choice(pool)
                    packet = self.packet(current, region, q)
                else:
                    alternatives = [self.packet(current, region, q) for region in regions]
                    scores = np.asarray(scorer(current, alternatives), dtype=float)
                    if scores.shape != (len(alternatives),) or not np.isfinite(scores).all():
                        raise ValueError("Controller scores must be one finite value per equal-cost packet")
                    chosen = min(range(len(alternatives)), key=lambda i: (-scores[i], alternatives[i]["region_id"]))
                    packet = alternatives[chosen]
                    region, group = packet["region"], self.groups[packet["region"]]
                    pool = grouped[group]
                for step, context in packet["prefixes"].items():
                    contexts[before + step] = context
                current = packet["final"]
                moves.extend(packet["moves"])
                packets.append({"start_count": before, "cost": q, "requested_cost": requested,
                                "region_id": self.names[region], "group": list(group),
                                "eligible_pool_size": len(regions), "eligible_group_size": len(pool),
                                "eligible_groups": len(grouped)})
        return {"contexts": contexts, "moves": moves, "packets": packets, "initial_count": initial,
                "target_count": target, "saturated": target < budget, "seed": int(seed),
                "mode": mode, "recipe": RECIPE, "balancing": self.balancing}

    def _extend(self, context, target, *, protected=(), mode="uniform", seed=0, region_rows=None):
        if target < context["token_count"] or target > len(self.leaves):
            raise ValueError("Requested exact count is outside feasible archive sizes")
        protected = set(map(int, protected))
        forbidden = set(leaf for region in protected for leaf in self.support(region))
        current, rng, touched, operations = context, random.Random(seed), Counter(), []
        region_rows = list(region_rows or context["rows"] if context["reader"] == "TREE" else region_rows or self.roots)
        owner = {}
        for region in region_rows:
            for leaf in self.support(region):
                owner[leaf] = region
        priority = {}
        flat_order = []
        # Round-robin source roots, then interleaved geometric/time locations.
        orders = [self._balanced_order(root) for root in self.roots]
        for i in range(max(map(len, orders), default=0)):
            flat_order.extend(values[i] for values in orders if i < len(values))
        flat_rank = {row: i for i, row in enumerate(flat_order)}
        while current["token_count"] < target:
            candidates = [row for row in self.positive_moves(current)
                          if not (set(self.support(row)) & forbidden)]
            if not candidates:
                raise ValueError("Protected regions prevent requested exact background count")
            def region(row):
                return owner.get(self.support(row)[0], row)
            if mode == "uniform":
                if current["reader"] == "TREE":
                    chosen = min(candidates, key=lambda row: (-self.counts[row], self.names[row]))
                else:
                    chosen = min(candidates, key=lambda row: (flat_rank[row], self.names[row]))
            elif mode == "dispersed":
                for row in candidates:
                    priority.setdefault(row, rng.random())
                chosen = min(candidates, key=lambda row: (touched[region(row)], priority[row], self.names[row]))
            elif mode == "random":
                for row in candidates:
                    if row not in priority:
                        priority[row] = rng.random()
                chosen = min(candidates, key=lambda row: (-priority[row], self.names[row]))
            else:
                raise ValueError("Unknown gain-blind continuation")
            current = self.move(current, chosen)
            touched[region(chosen)] += 1
            operations.append(chosen)
        return current, {"moves": operations, "distinct_regions_touched": len(touched),
                         "dispersed_meaningful": len(touched) > 1, "mode": mode}

    def ordinary(self, reader, budget, seed=0):
        """Existing persistent-priority random positive-cost allocation."""
        return self._extend(self.coarse(reader), min(int(budget), len(self.leaves)), mode="random", seed=seed)[0]

    def _pool(self, frontier, seed, cost, maximum):
        groups = defaultdict(list)
        for row in frontier["rows"]:
            if self.counts[row] >= cost + 1:
                groups[(*self.groups[row], int(math.log2(self.counts[row])))].append(row)
        rng = random.Random(seed)
        for values in groups.values():
            values.sort(key=lambda row: self.names[row])
            rng.shuffle(values)
        strata = sorted(groups)
        rng.shuffle(strata)
        ordered = [values[i] for i in range(max(map(len, groups.values()), default=0))
                   for group in strata for values in (groups[group],) if i < len(values)]
        return ordered[:maximum]

    def diagnostic_panel(self, seed, *, budgets=(64, 128), packet_size=8,
                         max_candidates=8, audit_candidates=4, prefixes=(1, 2, 4, 8)):
        """Common protected backgrounds, positive-cost controls, independent A2.

        All outcomes are topology-derived, including explicit eligibility
        shortfalls. The A2 audit is always attempted even when no packet exists.
        Returned arrays/dicts are JSON serializable and contain no policy losses.
        """
        packet_size = _positive(packet_size, "Packet size")
        max_candidates = _positive(max_candidates, "Candidate cap")
        audit_candidates = _positive(audit_candidates, "Audit candidate cap")
        budgets = tuple(sorted(set(_positive(b, "Budget") for b in budgets)))
        if any(b <= packet_size for b in budgets):
            raise ValueError("Final budget must exceed packet cost")
        saved_prefixes = sorted(set((1, packet_size, *(p for p in prefixes if 0 < p <= packet_size))))
        contexts, families, eligibility = {}, [], []

        def save(context):
            context = self.context(context["reader"], context["rows"])
            if context["context_id"] not in contexts:
                contexts[context["context_id"]] = self.describe(context)
            return context["context_id"]

        def family(reader, background, pool, final_budget, paired, reason=None, regions=None):
            candidates = []
            for anchor in pool:
                try:
                    packet = self.packet(background, anchor, packet_size)
                except ValueError:
                    continue
                candidate = {key: packet[key] for key in ("region_id", "support_hash", "support_count", "moves", "balancing")}
                candidate.update(candidate_id=_hash([reader, packet["support_hash"], packet["moves"]])[:24],
                                 source=self.groups[anchor][0], modality=self.groups[anchor][1],
                                 depth=self.depths[anchor], cost=packet_size,
                                 prefixes={str(step): save(packet["prefixes"][step]) for step in saved_prefixes})
                candidates.append(candidate)
            controls, control_metadata = {}, {}
            for mode in ("uniform", "dispersed"):
                control, meta = self._extend(background, final_budget, mode=mode,
                                             seed=seed + final_budget, region_rows=regions)
                controls[mode], control_metadata[mode] = save(control), meta
            families.append({"reader": reader, "panel": "packet", "background": background["token_count"],
                             "final_budget": final_budget, "base": save(background), "candidates": candidates,
                             "controls": controls, "control_metadata": control_metadata,
                             "paired": paired, "pairing_reason": reason, "ranking_eligible": len(candidates) >= 2})
            eligibility.append({"reader": reader, "panel": "packet", "final_budget": final_budget,
                                "eligible": bool(candidates), "ranking_eligible": len(candidates) >= 2,
                                "candidate_count": len(candidates), "paired": paired, "reason": reason})

        coarse_b, rich_b = budgets[0] - packet_size, budgets[-1] - packet_size
        tree_coarse, original_pool = None, []
        try:
            if len(self.leaves) < budgets[0]:
                raise ValueError("Insufficient leaves for exact final packet budget")
            tree_coarse = self._extend(self.coarse("TREE"), coarse_b)[0]
            original_pool = self._pool(tree_coarse, seed, packet_size, max_candidates)
            if not original_pool:
                raise ValueError("No anchor supports a positive-cost packet")
        except ValueError as exc:
            tree_reason = str(exc)
        else:
            tree_reason = None
        for reader in ("TREE", "FLAT"):
            pool, paired_contexts = list(original_pool), None
            region_frontier = tree_coarse
            reason = tree_reason
            if reader == "FLAT" and not pool and len(self.roots) > coarse_b:
                # FLAT residuals can fit even when a TREE frontier cannot. Its
                # diagnostic region proposal may still use the canonical roots.
                region_frontier = self.coarse("TREE")
                pool = self._pool(region_frontier, seed, packet_size, max_candidates)
            # Deterministically release the largest protected support until all
            # budgets can share one unchanged pool. Never inspect model gains.
            while pool:
                try:
                    if reader == "TREE":
                        base = tree_coarse
                        rich = self._extend(base, rich_b, protected=pool)[0]
                    else:
                        base = self._extend(self.coarse(reader), coarse_b, protected=pool)[0]
                        rich = self._extend(base, rich_b, protected=pool)[0]
                    for anchor in pool:
                        a, b = self.packet(base, anchor, packet_size), self.packet(rich, anchor, packet_size)
                        if a["moves"] != b["moves"] or a["support_hash"] != b["support_hash"]:
                            raise ValueError("Protected operation changed across backgrounds")
                    paired_contexts = base, rich
                    break
                except ValueError as exc:
                    reason = str(exc)
                    if len(pool) <= 2:
                        break
                    remove = max(pool, key=lambda row: (self.counts[row], self.names[row]))
                    pool.remove(remove)
            if paired_contexts is not None:
                for budget in budgets:
                    background = paired_contexts[0] if budget == budgets[0] else self._extend(
                        paired_contexts[0], budget - packet_size, protected=pool)[0]
                    family(reader, background, pool, budget, True, regions=region_frontier["rows"])
            else:
                # No paired pool: preserve each feasible single-budget stratum.
                for budget in budgets:
                    try:
                        if len(self.leaves) < budget:
                            raise ValueError("Insufficient leaves for exact final packet budget")
                        tree = (self.coarse("TREE") if reader == "FLAT" and len(self.roots) > budget - packet_size
                                else self._extend(self.coarse("TREE"), budget - packet_size)[0])
                        single_pool = self._pool(tree, seed + budget, packet_size, max_candidates)
                        if not single_pool:
                            raise ValueError("No packet anchor at this background")
                        if reader == "TREE":
                            background = tree
                        else:
                            while single_pool:
                                try:
                                    background = self._extend(self.coarse("FLAT"), budget - packet_size,
                                                              protected=single_pool)[0]
                                    for anchor in single_pool:
                                        self.packet(background, anchor, packet_size)
                                    break
                                except ValueError:
                                    single_pool.pop()
                            if not single_pool:
                                raise ValueError("No residual-preserving protected FLAT packet background")
                        family(reader, background, single_pool, budget, False,
                               reason or "Paired backgrounds infeasible", regions=tree["rows"])
                    except ValueError as exc:
                        eligibility.append({"reader": reader, "panel": "packet", "final_budget": budget,
                                            "eligible": False, "candidate_count": 0, "paired": False,
                                            "reason": str(exc)})
            # Unrestricted immediate audit must not depend on packet eligibility.
            try:
                if len(self.leaves) < 128:
                    raise ValueError("Fewer than 128 causal leaves for B127-to-B128 audit")
                base = self.ordinary(reader, 127, seed + 1789)
                candidates = self.positive_moves(base)
                rng = random.Random(seed + 1811)
                candidates = rng.sample(candidates, min(audit_candidates, len(candidates)))
                if not candidates:
                    raise ValueError("No positive-cost unrestricted move at B127")
                entries = []
                for row in candidates:
                    after = self.move(base, row)
                    entries.append({"candidate_id": _hash([reader, self.names[row], "A2"])[:24],
                                    "region_id": self.names[row], "support_hash": _hash(sorted(self.names[v] for v in self.support(row))),
                                    "support_count": self.counts[row], "source": self.groups[row][0],
                                    "modality": self.groups[row][1], "depth": self.depths[row], "moves": [row],
                                    "cost": 1, "prefixes": {"1": save(after)}})
                families.append({"reader": reader, "panel": "immediate_audit", "background": 127,
                                 "final_budget": 128, "base": save(base), "candidates": entries,
                                 "controls": {}, "paired": False, "ranking_eligible": len(entries) >= 2,
                                 "sampler": "persistent_random_priority_background_then_uniform_valid_moves"})
                eligibility.append({"reader": reader, "panel": "immediate_audit", "eligible": True,
                                    "candidate_count": len(entries), "ranking_eligible": len(entries) >= 2})
            except ValueError as exc:
                eligibility.append({"reader": reader, "panel": "immediate_audit", "eligible": False,
                                    "candidate_count": 0, "reason": str(exc)})
        return {"recipe": RECIPE, "seed": int(seed), "contexts": contexts, "families": families,
                "eligibility": eligibility, "leaf_count": len(self.leaves), "root_count": len(self.roots),
                "coverage_hash": self.coverage, "flat_ordering": self.balancing}


def training_routes(index, roots, leaves, reader, target_budget, seed, packet_size=8):
    """Common topology-only feasibility for paired local/packet student exposure."""
    archive = IndexedArchive(index, roots, leaves)
    initial = archive.coarse(reader)["token_count"]
    target = min(int(target_budget), len(archive.leaves))
    result = {"initial_count": initial, "target_count": target, "saturated": target < target_budget,
              "eligible": False, "reason": None, "stage": None, "packet": None}
    try:
        packet = archive.route(reader, target_budget, seed, mode="packet", packet_size=packet_size)
        stage = archive.route(reader, target_budget, seed, mode="local", packet_size=packet_size)
    except ValueError as exc:
        result["reason"] = str(exc)
        return result
    if set(stage["contexts"]) != set(packet["contexts"]):
        raise AssertionError("Paired routes have different attainable prefix schedules")
    result.update(eligible=True, stage=stage, packet=packet)
    return result
