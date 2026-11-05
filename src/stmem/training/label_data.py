"""Bounded, causal CPU preparation for paired TREE/FLAT utility labels.

The plan preserves the reference label sampler. Indexed hierarchy rows are
bookkeeping, never policy inputs; only a decision's completed blocks and causal
prefix enter its full teacher bank or compact contexts.
"""

from __future__ import annotations

from collections import defaultdict
import heapq
import random
import time
from types import SimpleNamespace

import numpy as np
import torch

from stmem.data.cache import episode_key
from stmem.models.router import query_features
from stmem.training.cached_loader import LightweightDecisions, _Files
from stmem.training.packed_context import (
    PackedContextReader, _flat_selection, _take, _tree_indices,
    columns_bank, flat_residual_columns,
)


class LabelPlan:
    """Action-index-only immutable sampling, independent of worker/shard order."""

    def __init__(self, data, cfg, refresh=False):
        self.data, self.cfg, self.refresh = data, cfg, bool(refresh)
        self.by_episode = defaultdict(list)
        for task in data.tasks:
            for episode, index in data.candidates[task]:
                self.by_episode[episode].append(index)
        self.contexts = int(cfg["labels"]["refresh_contexts" if refresh else "contexts"])
        if self.contexts < 1 or not cfg["labels"]["budgets"]:
            raise ValueError("Label generation requires positive contexts and configured budgets")
        self.history_counts = {}
        # Read only source/index sidecars for byte bounds. The point-cloud and
        # frozen-vision payloads are never opened. Synthetic fixture clocks are
        # explicitly phase-zero source-local ranges.
        files = _Files()
        try:
            for episode in self.by_episode:
                if data.synthetic:
                    execution = np.arange(cfg["data"].get("synthetic_execution_frames", 12))
                    conditioning = np.arange(cfg["data"].get("synthetic_conditioning_frames", 5))
                else:
                    ref = data.dataset.episodes[data.refs[episode]]
                    group = files.get(ref.path)[ref.key]
                    source = np.asarray(group["source"])
                    indices = np.asarray(group["source_index"])
                    execution, conditioning = indices[~source], indices[source]
                self.history_counts[episode] = (
                    int((conditioning % data.observation_stride == 0).sum()),
                    np.flatnonzero(execution % data.observation_stride == 0),
                )
        finally:
            files.close()

    def _size_bound(self, episode, index, budget):
        conditioning, execution_rows = self.history_counts[episode]
        execution = int(np.searchsorted(execution_rows, index))
        model = self.cfg["model"]
        block = model["block_size"]
        blocks = (conditioning + block - 1) // block + (execution + block - 1) // block
        events = model.get("proprioception_mode", "all") == "all" or model.get("condition_on_previous_commands", True)
        leaves = blocks * model["visual_tokens"] + int(events) * (conditioning + execution)
        # Full/current banks plus every possible compact variant and all scorer
        # inputs. Four readers safely cover the two objective-specific refreshes.
        readers = 4 if self.refresh else 2
        candidates = self.cfg["labels"]["candidates"]
        width = model["width"]
        tokens = leaves + model["current_tokens"] + readers * (candidates + 1) * budget
        size = tokens * (4 * (width + 24) + 1)
        size += readers * candidates * (4 * (2 * width + 23) + 4)
        size += 4 * (3 * width + 27 + 24 + 2) + model["horizon"] * 8 * 5
        return leaves, size

    def items(self, context_ids=None, *, shard_index=0, shard_count=1):
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("Invalid label shard index/count")
        ids = range(self.contexts) if context_ids is None else context_ids
        result = []
        for context_id in ids:
            if not 0 <= context_id < self.contexts:
                raise IndexError("Label context ID is outside the declared sample plan")
            if context_id % shard_count != shard_index:
                continue
            task = self.data.tasks[context_id % len(self.data.tasks)]
            episodes = self.data.by_task[task]
            episode = episodes[(context_id // len(self.data.tasks)) % len(episodes)]
            times = self.by_episode[episode]
            if not times:
                raise ValueError(f"Label episode {episode} has no eligible aligned decisions")
            budgets = self.cfg["labels"]["budgets"]
            stratum = (context_id // len(budgets)) % 3
            lo, hi = len(times) * stratum // 3, len(times) * (stratum + 1) // 3
            choices = times[lo:hi] or times
            seed = self.cfg["labels"]["seed"] + context_id + (1000000 if self.refresh else 0)
            index = int(random.Random(seed).choice(choices))
            budget = int(budgets[context_id % len(budgets)])
            leaf_bound, size_bound = self._size_bound(episode, index, budget)
            result.append({
                "context_id": int(context_id), "episode_index": int(episode),
                "index": index, "budget": budget, "seed": int(seed),
                "time_choices": len(choices), "refresh": self.refresh,
                "history_leaf_bound": leaf_bound, "estimated_bytes": size_bound,
            })
        return result


class _ReferenceData:
    """Reuse the canonical lightweight constructor for an explicit planned row."""

    def __init__(self, data):
        self.data, self.reference = data, None

    def __getattr__(self, name):
        return getattr(self.data, name)

    def sample_reference(self, _position):
        return self.reference


def candidate_features(columns, width):
    """LocalRouter.candidate_input in columns, including immutable zero ages."""
    bank = columns_bank(columns, (0.0, 0.0), width)
    metadata = bank.metadata[0].clone()
    metadata[:, 11:13] = 0
    return torch.cat((bank.content[0], columns["max_content"].float(), metadata), dim=-1)


def _records_for_scoring(columns):
    """Nonrecursive temporary views compatible with CachedScorer's public API."""
    for row, name in enumerate(columns["node_id"]):
        values = {key: column[row] for key, column in columns.items()}
        values["node_id"] = str(name)
        values["content"] = values["content_sum"] / int(values["count"])
        yield SimpleNamespace(**values)


class LabelDataReader:
    """One worker's bounded HDF5 handles/topology LRU and small action arrays.

    Frozen caches must be validated by the job before workers start, as for the
    compact-policy loader. This reader additionally checks every requested
    prefix and exact active-leaf count. No model, raw point cloud, or learned
    encoder is needed here. Refresh scorers use their ordinary frozen API.
    """

    def __init__(self, data, cfg, cache_root, hierarchy_root):
        self.data, self.cfg = data, cfg
        self.width = cfg["model"]["width"]
        self.reference = _ReferenceData(data)
        self.decisions = LightweightDecisions(self.reference)
        self.packed = PackedContextReader(cfg, cache_root, hierarchy_root)
        self.partitions = {index: data.label_partition(index) for index in range(len(data.refs))}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.decisions.close()
        self.packed.close()

    @staticmethod
    def _rng(item):
        rng = random.Random(item["seed"])
        # Replay exactly the RNG consumption of the already planned time draw.
        rng.randrange(item["time_choices"])
        return rng

    def _score(self, scorer, hierarchy, index, rows):
        rows = np.asarray(rows, dtype=np.int64)
        values = []
        for start in range(0, len(rows), 4096):
            columns = self.packed._columns(
                hierarchy["nodes"], rows[start:start + 4096], include_max=True, index=index,
            )
            records = list(_records_for_scoring(columns))
            child_features = None
            if getattr(scorer.router, "requires_children", False):
                pairs = index["children"][rows[start:start + 4096]]
                if np.any(pairs < 0):
                    raise ValueError("Child-aware scoring requires binary parent candidates")
                child_columns = self.packed._columns(
                    hierarchy["nodes"], pairs.reshape(-1), include_max=True, index=index,
                )
                child_features = candidate_features(child_columns, self.width).reshape(len(records), 2, -1)
            scorer.prepare(records, children=child_features)
            values.extend(scorer(records))
        if not np.isfinite(values).all():
            raise ValueError("Router scores must be finite")
        return values

    def _tree(self, hierarchy, index, roots, cap, rng, scorer):
        if scorer is None:
            return _tree_indices(index, roots, cap, rng)
        names, children = index["node_id"], index["children"]
        frontier = {str(names[row]): int(row) for row in roots}
        if len(frontier) != len(roots):
            raise ValueError("Duplicate hierarchy roots")
        heap = []

        def enqueue(rows):
            rows = [int(row) for row in rows if int(children[row, 0]) >= 0]
            scores = self._score(scorer, hierarchy, index, rows)
            for row, score in zip(rows, scores):
                heapq.heappush(heap, (-score, str(names[row])))

        if len(frontier) < cap:
            enqueue(roots)
        while len(frontier) < cap and heap:
            _, name = heapq.heappop(heap)
            row = frontier.pop(name)
            pair = children[row]
            if min(pair) < 0:
                raise ValueError("Indexed hierarchy parent must have two children")
            for child in pair:
                frontier[str(names[child])] = int(child)
            if len(frontier) < cap:
                enqueue(pair)
        return np.asarray(list(frontier.values()), dtype=np.int64)

    def _flat(self, hierarchy, index, leaves, cap, rng, scorer):
        if scorer is None:
            return _flat_selection(index, leaves, cap, rng)
        groups = list(zip(index["source"][leaves], index["modality"][leaves]))
        names = sorted(set(groups))
        lookup = {name: row for row, name in enumerate(names)}
        codes = np.asarray([lookup[group] for group in groups], dtype=np.int64)
        if cap < len(names):
            raise ValueError("Flat budget cannot cover source/modality residuals")
        if len(leaves) <= cap:
            return np.ones(len(leaves), dtype=bool), codes, names
        scores = self._score(scorer, hierarchy, index, leaves)
        ranked = sorted(range(len(leaves)), key=lambda row: (-scores[row], str(index["node_id"][leaves[row]])))
        counts = np.bincount(codes, minlength=len(names))
        selected = np.zeros(len(leaves), dtype=bool)
        tokens = len(names)
        for row in ranked:
            cost = int(counts[codes[row]] > 1)
            if tokens + cost <= cap:
                selected[row] = True
                counts[codes[row]] -= 1
                tokens += cost
        for group in range(len(names)):
            remaining = (~selected) & (codes == group)
            if int(remaining.sum()) == 1:
                selected[remaining] = True
        return selected, codes, names

    @torch.no_grad()
    def prepare(self, item, scorers=None, *, router_inputs=False, inputs_only=False, reader_family=None):
        if inputs_only and not router_inputs:
            raise ValueError("Input-only preparation requires router_inputs=True")
        if reader_family not in (None, "tree", "flat"):
            raise ValueError("Unknown router input reader")
        started = time.perf_counter()
        self.reference.reference = (item["episode_index"], item["index"])
        decision = self.decisions.decision(0)
        load_seconds = time.perf_counter() - started
        packed, info = self.packed, decision.info
        packed.read_seconds = packed.read_bytes = 0
        key = episode_key(info.task, info.episode_id)
        hierarchy = packed.hierarchy_files.get(packed.hierarchy_root / f"{info.task}.h5")[key]
        entry = hierarchy["decisions"][str(info.index)]
        nfull, pending = divmod(info.execution_history_frames, self.cfg["model"]["block_size"])
        expected_prefix = f"{nfull * self.cfg['model']['block_size']}_{pending}" if pending else ""
        prefix = entry.attrs["prefix_key"]
        prefix = prefix.decode() if isinstance(prefix, bytes) else str(prefix)
        if int(entry.attrs["nfull"]) != nfull or prefix != expected_prefix:
            raise ValueError("Hierarchy decision cadence differs from the requested prefix")
        roots = packed._read(entry["roots"])
        budget = int(item["budget"])
        if len(roots) > budget:
            raise ValueError("Label budget cannot cover roots")
        leaves = packed._active_leaves(hierarchy, entry)
        if len(leaves) != int(entry.attrs["history_leaf_count"]):
            raise ValueError("Indexed label leaf coverage differs from causal decision metadata")
        index = packed._index(hierarchy)
        leaf_columns = None if inputs_only else packed._columns(hierarchy["nodes"], leaves, index=index)
        base = packed.base_files.get(packed.cache_root / f"{info.task}.h5")[key]
        current_columns = packed._columns(base["current"][str(info.index)])
        current = columns_bank(current_columns, decision.clocks, self.width)
        full = None if inputs_only else columns_bank(leaf_columns, decision.clocks, self.width)
        text = torch.from_numpy(packed._read(base["text"])).float()
        state, clocks = decision.state.float(), torch.tensor(decision.clocks, dtype=torch.float32)
        observed = SimpleNamespace(proprio=state[:15], previous_action=state[15:23],
                                   action_available=bool(state[23]))
        mean = current.content[0].mean(0) if current.content.shape[1] else torch.zeros(self.width)
        model = self.cfg["model"]
        query = query_features(
            mean, text, observed, dict(zip(("conditioning", "execution"), decision.clocks)), budget,
            condition_on_previous_joint_commands=model.get("condition_on_previous_joint_commands", True),
            condition_on_previous_commands=model.get("condition_on_previous_commands", True),
            proprioception_mode=model.get("proprioception_mode", "all"),
        )
        if item.get("refresh") and not scorers:
            raise ValueError("Refresh label preparation requires frozen reader scorers")
        readers = {}
        selection_started = time.perf_counter()
        for name, frozen in (scorers.items() if scorers else (("tree", None), ("flat", None))):
            reader = name.rsplit("/", 1)[-1]
            if reader_family is not None and reader != reader_family:
                continue
            if reader not in ("tree", "flat"):
                raise ValueError(f"Unknown label reader: {name}")
            rng = self._rng(item)
            learned = frozen is not None and rng.random() >= self.cfg["labels"]["exploration_fraction"]
            scorer = frozen if learned else None
            if scorer is not None:
                # Each context may belong to a different episode/prefix. Keep
                # immutable keys only during this preparation; no unbounded LRU.
                scorer.retain(())
                scorer.set_query(query, current=torch.cat((current.content[0], current.metadata[0]), -1),
                                 current_valid=current.valid[0])
            cap = rng.randint(max(len(roots), 1), max(len(roots), budget - 1))
            if reader == "tree":
                before_rows = self._tree(hierarchy, index, roots, cap, rng, scorer)
                candidates = [int(row) for row in before_rows if index["children"][row, 0] >= 0]
                rng.shuffle(candidates)
                candidates = candidates[:self.cfg["labels"]["candidates"]] if len(before_rows) < budget else []
                after_rows = [np.asarray([child for row in before_rows for child in (
                    index["children"][row] if row == candidate else (row,)
                )], dtype=np.int64) for candidate in candidates]
                child_rows = index["children"][candidates].reshape(-1)
                union = (np.unique(np.concatenate([np.asarray(candidates, dtype=np.int64), child_rows]))
                         if inputs_only else np.unique(np.concatenate([before_rows, *after_rows])))
                columns = packed._columns(hierarchy["nodes"], union, include_max=True, index=index)
                lookup = {int(row): position for position, row in enumerate(union)}
                def take(rows):
                    return _take(columns, [lookup[int(row)] for row in rows])
                before = None if inputs_only else columns_bank(take(before_rows), decision.clocks, self.width)
                after = [] if inputs_only else [columns_bank(take(rows), decision.clocks, self.width) for rows in after_rows]
                candidate = candidate_features(take(candidates), self.width)
                costs = torch.ones(len(candidates), dtype=torch.float32)
                candidate_ids = [str(index["node_id"][row]) for row in candidates]
                frontier_ids = [str(index["node_id"][row]) for row in before_rows]
                children = (candidate_features(take(child_rows), self.width).reshape(len(candidates), 2, 2 * self.width + 23)
                            if router_inputs else None)
            else:
                selected, codes, groups = self._flat(hierarchy, index, leaves, cap, rng, scorer)
                if not inputs_only:
                    columns = flat_residual_columns(leaf_columns, selected, codes, groups)
                    before = columns_bank(columns, decision.clocks, self.width)
                    before_count = before.content.shape[1]
                else:
                    before = None
                    before_count = int(selected.sum()) + len(set(codes[~selected]))
                # Singleton residuals are normalized by selection. One group
                # count pass replaces the reference O(N^2) move_cost filter.
                counts = np.bincount(codes[~selected], minlength=len(groups))
                candidates = [int(row) for row in np.flatnonzero(~selected) if counts[codes[row]] > 1]
                rng.shuffle(candidates)
                candidates = candidates[:self.cfg["labels"]["candidates"]] if before_count < budget else []
                after, cost_values = [], []
                for row in candidates:
                    moved = selected.copy()
                    moved[row] = True
                    for group in range(len(groups)):
                        remaining = (~moved) & (codes == group)
                        if int(remaining.sum()) == 1:
                            moved[remaining] = True
                    after_count = int(moved.sum()) + len(set(codes[~moved]))
                    cost_values.append(after_count - before_count)
                    if not inputs_only:
                        after.append(columns_bank(flat_residual_columns(leaf_columns, moved, codes, groups),
                                                  decision.clocks, self.width))
                candidate = candidate_features(packed._columns(
                    hierarchy["nodes"], leaves[candidates], include_max=True, index=index,
                ), self.width)
                costs = torch.tensor(cost_values, dtype=torch.float32)
                candidate_ids = [str(index["node_id"][leaves[row]]) for row in candidates]
                frontier_ids = [str(index["node_id"][row]) for row in leaves[selected]]
                children = None
            if (costs <= 0).any() or any(bank.content.shape[1] > budget for bank in after):
                raise ValueError("Label moves require positive cost within the target budget")
            readers[name] = {"before": before, "after": after, "candidate": candidate, "costs": costs}
            if router_inputs:
                readers[name].update(children=children, candidate_ids=candidate_ids,
                                     frontier_ids=frontier_ids)
        return {
            "context_id": item["context_id"], "seed": item["seed"], "budget": budget,
            "decision_index": int(info.index),
            "episode_index": item["episode_index"], "episode": f"{info.task}/{info.episode_id}",
            "partition": int(self.partitions[item["episode_index"]] == "validation"),
            "target": decision.target.float(), "valid": decision.target_valid,
            "current": current, "full": full, "text": text, "current_state": state,
            "clocks": clocks, "query": query, "readers": readers, "history_leaves": len(leaves),
            "timings": {"load_seconds": load_seconds, "read_seconds": packed.read_seconds,
                        "read_bytes": packed.read_bytes,
                        "selection_seconds": time.perf_counter() - selection_started,
                        "prepare_seconds": time.perf_counter() - started},
        }
