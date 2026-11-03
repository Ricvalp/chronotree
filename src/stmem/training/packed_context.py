"""Family-first compact contexts from indexed immutable hierarchy sidecars.

Topology is CPU bookkeeping only. TREE/UNIFORM gather selected records after
selection; CURRENT never reads historical payload; FLAT scans its actual causal
leaves and aggregates the remaining leaves in their reference order. No runtime
tree construction, recursive records, or model weights are involved.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import heapq
import math
from pathlib import Path
import random
import sys
import time

import h5py
import numpy as np
import torch

from stmem.data.cache import episode_key
from stmem.memory.hierarchy import stable_id
from stmem.models.batched_policy import TokenBank


TENSOR_FIELDS = ("content_sum", "anchor", "bounds_min", "bounds_max", "max_content")
SCALAR_FIELDS = (
    "count", "time_min", "time_max", "depth", "scale", "completed", "disconnected", "time_anchor", "occupancy",
)
STRING_FIELDS = ("node_id", "source", "modality")


def _strings(value):
    return np.asarray([row.decode() if isinstance(row, bytes) else str(row) for row in value], dtype=object)


def _bytes(value):
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        return sum(len(str(row).encode()) for row in value.flat)
    return value.nbytes


def _resident_bytes(value):
    """Include object pointers and Python strings in the bounded topology LRU."""
    return value.nbytes + (sum(sys.getsizeof(row) for row in value.flat) if value.dtype.kind == "O" else 0)


class _Handles:
    def __init__(self, capacity=16):
        self.capacity = capacity
        self.values = OrderedDict()

    def get(self, path):
        key = str(path)
        handle = self.values.pop(key, None)
        if handle is None:
            handle = h5py.File(path, "r")
        self.values[key] = handle
        while len(self.values) > self.capacity:
            _, stale = self.values.popitem(last=False)
            stale.close()
        return handle

    def close(self):
        for handle in self.values.values():
            handle.close()
        self.values.clear()


@dataclass
class PackedContext:
    current: TokenBank
    history: TokenBank
    text: torch.Tensor
    leaf_count: int
    read_seconds: float
    selection_seconds: float
    residual_seconds: float
    read_bytes: int
    index_hits: int
    index_misses: int
    debug: dict | None = None


def context_family(cfg, position, root_count):
    """Reproduce the global-position family/budget/RNG protocol exactly."""
    rng = random.Random(cfg["seed"] * 1000003 + position)
    mixture = cfg["training"]["context_mix"]
    families = [name.upper() for name, fraction in mixture.items() for _ in range(round(fraction * 20))]
    random.Random(cfg["seed"] + position // 20).shuffle(families)
    method = families[position % len(families)]
    budget = cfg["model"]["budgets"][(position // len(families)) % len(cfg["model"]["budgets"])]
    if root_count > budget:
        raise ValueError(f"{root_count} roots exceed budget {budget}")
    if method != "CURRENT" and rng.random() < cfg["training"]["intermediate_probability"]:
        budget = rng.randint(max(root_count, 1), budget)
    return method, budget, rng


def _tree_indices(index, roots, budget, rng, *, uniform=False):
    names, children, counts = index["node_id"], index["children"], index["count"]
    frontier = {str(names[node]): int(node) for node in roots}
    if len(frontier) != len(roots):
        raise ValueError("Duplicate hierarchy roots")
    heap = []

    def enqueue(nodes):
        for node in nodes:
            node = int(node)
            if int(children[node, 0]) >= 0:
                score = float(counts[node]) if uniform else rng.random()
                heapq.heappush(heap, (-score, str(names[node])))

    if len(frontier) < budget:
        enqueue(roots)
    while len(frontier) < budget and heap:
        _, name = heapq.heappop(heap)
        node = frontier.pop(name)
        left, right = map(int, children[node])
        if min(left, right) < 0:
            raise ValueError("Indexed hierarchy parent must have two children")
        frontier[str(names[left])] = left
        frontier[str(names[right])] = right
        if len(frontier) < budget:
            enqueue((left, right))
    return np.asarray(list(frontier.values()), dtype=np.int64)


def _flat_selection(index, leaves, budget, rng):
    names = index["node_id"][leaves]
    groups = list(zip(index["source"][leaves], index["modality"][leaves]))
    group_names = sorted(set(groups))
    group_numbers = {name: number for number, name in enumerate(group_names)}
    codes = np.asarray([group_numbers[group] for group in groups], dtype=np.int64)
    if budget < len(group_names):
        raise ValueError("Flat budget cannot cover source/modality residuals")
    if len(leaves) <= budget:
        return np.ones(len(leaves), dtype=bool), codes, group_names
    # Exactly one RNG draw per leaf in original archive.leaves order. Sorting
    # ties by stable leaf ID matches select_flat's score ordering.
    scores = [rng.random() for _ in leaves]
    ranked = sorted(range(len(leaves)), key=lambda i: (-scores[i], str(names[i])))
    remaining = np.bincount(codes, minlength=len(group_names))
    selected = np.zeros(len(leaves), dtype=bool)
    tokens = len(group_names)
    for row in ranked:
        group = codes[row]
        cost = int(remaining[group] > 1)
        if tokens + cost <= budget:
            selected[row] = True
            remaining[group] -= 1
            tokens += cost
    for group in range(len(group_names)):
        members = (~selected) & (codes == group)
        if int(members.sum()) == 1:
            selected[members] = True
    return selected, codes, group_names


def _empty_columns(width):
    result = {key: torch.empty(0, width if key in ("content_sum", "max_content") else 3)
              for key in TENSOR_FIELDS}
    result.update({key: np.empty(0) for key in SCALAR_FIELDS})
    result.update({key: np.empty(0, dtype=object) for key in STRING_FIELDS})
    return result


def columns_bank(columns, clocks, width):
    """Identical metadata to records_bank, without constructing MemoryRecords."""
    count = np.asarray(columns["count"])
    n = len(count)
    if not n:
        return TokenBank(torch.empty(1, 0, width), torch.empty(1, 0, 23),
                         torch.empty(1, 0, dtype=torch.bool), torch.empty(1, 0))
    source = columns["source"]
    now = np.where(source == "conditioning", float(clocks[0]), float(clocks[1]))
    log_count = torch.tensor([math.log(int(value)) for value in count], dtype=torch.float32)
    scalars = np.stack([
        columns["time_min"] / 100, columns["time_max"] / 100,
        (now - columns["time_min"]) / 100, (now - columns["time_max"]) / 100,
        np.asarray([math.log(int(value)) for value in count]), columns["depth"] / 16,
        np.asarray([math.log1p(int(value)) for value in columns["scale"]]),
        columns["completed"].astype(float), columns["disconnected"].astype(float),
        (source == "conditioning").astype(float), (source == "execution").astype(float),
        (columns["modality"] == "visual").astype(float), (columns["modality"] == "event").astype(float),
        (count == 1).astype(float),
    ], axis=1)
    geometry = torch.cat([columns[key] for key in ("anchor", "bounds_min", "bounds_max")], dim=-1)
    content = columns["content_sum"] / torch.as_tensor(count, dtype=torch.float32)[:, None]
    metadata = torch.cat([geometry, torch.tensor(scalars, dtype=torch.float32)], dim=-1)
    return TokenBank(content[None], metadata[None], torch.ones(1, n, dtype=torch.bool), log_count[None])


def _take(columns, indices):
    return {key: value[indices] for key, value in columns.items()}


def flat_residual_columns(columns, selected, group_codes, group_names):
    """Aggregate only actual unselected leaves, preserving reference reduction order."""
    selected_rows = np.flatnonzero(selected)
    pieces = [_take(columns, selected_rows)]
    if np.any(np.asarray(columns["count"]) != 1):
        raise ValueError("Flat contexts require unit-mass fine leaves")
    for number, (source, modality) in enumerate(group_names):
        rows = np.flatnonzero((~selected) & (group_codes == number))
        if not len(rows):
            continue
        if len(rows) == 1:
            pieces.append(_take(columns, rows))
            continue
        values = _take(columns, rows)
        count = len(rows)
        row = {
            "node_id": np.asarray([stable_id("residual", map(str, values["node_id"]))], dtype=object),
            "source": np.asarray([source], dtype=object), "modality": np.asarray([modality], dtype=object),
            "content_sum": values["content_sum"].sum(0, dtype=torch.float32)[None],
            "anchor": (values["anchor"].sum(0, dtype=torch.float32) / count)[None],
            "bounds_min": values["bounds_min"].amin(0)[None],
            "bounds_max": values["bounds_max"].amax(0)[None],
            "count": np.asarray([count]), "time_min": np.asarray([min(values["time_min"])]),
            "time_max": np.asarray([max(values["time_max"])]),
            "depth": np.asarray([1 + max(values["depth"])]), "scale": np.asarray([max(values["scale"])]),
            "completed": np.asarray([all(values["completed"])]), "disconnected": np.asarray([True]),
            "time_anchor": np.asarray([sum(map(float, values["time_anchor"])) / count]),
            "occupancy": np.asarray([sum(map(float, values["occupancy"])) / count]),
        }
        if "max_content" in values:
            row["max_content"] = values["max_content"].amax(0)[None]
        pieces.append(row)
    return {key: torch.cat([piece[key] for piece in pieces]) if key in TENSOR_FIELDS else (
        np.concatenate([piece[key] for piece in pieces])
    ) for key in columns}


class PackedContextReader:
    """Read indexed contexts with bounded per-worker topology caching."""

    def __init__(self, cfg, cache_root, hierarchy_root, *, maximum_indices=4, maximum_index_bytes=128 * 1024 * 1024):
        self.cfg = cfg
        self.cache_root, self.hierarchy_root = Path(cache_root), Path(hierarchy_root)
        self.width = cfg["model"]["width"]
        self.maximum_indices, self.maximum_index_bytes = maximum_indices, maximum_index_bytes
        self.indices = OrderedDict()
        self.index_bytes = 0
        self.base_files, self.hierarchy_files = _Handles(), _Handles()
        self.read_seconds = self.read_bytes = self.hits = self.misses = 0

    def _read(self, dataset, rows=None):
        started = time.perf_counter()
        if rows is None:
            array = np.asarray(dataset)
        elif not len(rows):
            array = np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
        else:
            # h5py requires strictly increasing fancy indices. Preserve the
            # reader's arbitrary selected ordering through the inverse map.
            unique, inverse = np.unique(np.asarray(rows, dtype=np.int64), return_inverse=True)
            cuts = np.flatnonzero(np.diff(unique) != 1) + 1
            if len(unique) > 128 and len(cuts) < len(unique) // 4:
                # Dense leaf runs are faster than thousands of HDF5 point
                # selections and never touch gaps containing other records.
                runs = np.split(unique, cuts)
                array = np.concatenate([np.asarray(dataset[int(run[0]):int(run[-1]) + 1])
                                        for run in runs])[inverse]
            else:
                array = np.asarray(dataset[unique])[inverse]
        if array.dtype.kind in ("O", "S", "U"):
            array = _strings(array)
        self.read_seconds += time.perf_counter() - started
        self.read_bytes += _bytes(array)
        return array

    def _index(self, group):
        key = group.file.filename, group.name
        found = self.indices.pop(key, None)
        if found is not None:
            self.hits += 1
            self.indices[key] = found
            return found[0]
        self.misses += 1
        nodes = group["nodes"]
        values = {key: self._read(nodes[key]) for key in ("node_id", "children", "count", "source", "modality")}
        size = sum(_resident_bytes(value) for value in values.values())
        if self.maximum_indices and size <= self.maximum_index_bytes:
            self.indices[key] = values, size
            self.index_bytes += size
            while len(self.indices) > self.maximum_indices or self.index_bytes > self.maximum_index_bytes:
                _, (_, removed) = self.indices.popitem(last=False)
                self.index_bytes -= removed
        return values

    def _columns(self, group, rows=None, *, include_max=False, index=None):
        if int(group.attrs.get("length", -1)) == 0:
            result = _empty_columns(self.width)
            if not include_max:
                result.pop("max_content")
            return result
        result = {}
        for key in (*TENSOR_FIELDS, *SCALAR_FIELDS, *STRING_FIELDS):
            if key == "max_content" and not include_max:
                continue  # Persisted for router statistics; unused by Stage P.
            value = index[key][rows] if index is not None and key in index else self._read(group[key], rows)
            result[key] = torch.from_numpy(value).float() if key in TENSOR_FIELDS else value
        return result

    @staticmethod
    def _active_leaves(group, decision):
        blocks = group["blocks"]
        chosen = [blocks["conditioning"][key] for key in sorted(blocks["conditioning"], key=int)]
        nfull = int(decision.attrs["nfull"])
        execution = sorted(blocks["execution"], key=int)
        if len(execution) < nfull:
            raise ValueError("Hierarchy does not cover the requested completed execution blocks")
        chosen.extend(blocks["execution"][key] for key in execution[:nfull])
        prefix = decision.attrs["prefix_key"]
        prefix = prefix.decode() if isinstance(prefix, bytes) else str(prefix)
        if prefix:
            chosen.append(blocks["prefix"][prefix])
        ranges = [np.arange(int(block.attrs["leaf_start"]), int(block.attrs["leaf_start"]) +
                            int(block.attrs["leaf_count"]), dtype=np.int64) for block in chosen]
        return np.concatenate(ranges) if ranges else np.empty(0, dtype=np.int64)

    def prepare(self, decision, position, *, debug=False):
        self.read_seconds = self.read_bytes = 0
        before_hits, before_misses = self.hits, self.misses
        info = decision.info
        key = episode_key(info.task, info.episode_id)
        hierarchy = self.hierarchy_files.get(self.hierarchy_root / f"{info.task}.h5")[key]
        entry = hierarchy["decisions"][str(info.index)]
        roots = self._read(entry["roots"])
        leaf_count = int(entry.attrs["history_leaf_count"])
        block_size = self.cfg["model"]["block_size"]
        nfull, pending = divmod(info.execution_history_frames, block_size)
        expected_prefix = f"{nfull * block_size}_{pending}" if pending else ""
        prefix = entry.attrs["prefix_key"]
        prefix = prefix.decode() if isinstance(prefix, bytes) else str(prefix)
        if int(entry.attrs["nfull"]) != nfull or prefix != expected_prefix:
            raise ValueError("Indexed hierarchy decision cadence differs from the requested prefix")
        started = time.perf_counter()
        method, budget, rng = context_family(self.cfg, position, len(roots))
        selection_seconds = time.perf_counter() - started
        residual_seconds = 0.0
        selected = _empty_columns(self.width)
        leaves = selected_nodes = None
        if method in ("TREE", "UNIFORM"):
            index = self._index(hierarchy)
            started = time.perf_counter()
            selected_nodes = _tree_indices(index, roots, budget, rng, uniform=method == "UNIFORM")
            selection_seconds += time.perf_counter() - started
            selected = self._columns(hierarchy["nodes"], selected_nodes, include_max=debug, index=index)
        elif method == "FLAT":
            index = self._index(hierarchy)
            started = time.perf_counter()
            leaves = self._active_leaves(hierarchy, entry)
            if len(leaves) != leaf_count:
                raise ValueError("Indexed flat leaf coverage differs from causal decision metadata")
            chosen, group_codes, group_names = _flat_selection(index, leaves, budget, rng)
            selection_seconds += time.perf_counter() - started
            all_leaves = self._columns(hierarchy["nodes"], leaves, include_max=debug, index=index)
            started = time.perf_counter()
            selected = flat_residual_columns(all_leaves, chosen, group_codes, group_names)
            residual_seconds = time.perf_counter() - started
        elif method != "CURRENT":
            raise ValueError(f"Unsupported indexed compact context family: {method}")
        base = self.base_files.get(self.cache_root / f"{info.task}.h5")[key]
        current = self._columns(base["current"][str(info.index)])
        text = torch.from_numpy(self._read(base["text"])).float()
        clocks = torch.tensor(decision.clocks, dtype=torch.float32).tolist()
        return PackedContext(
            current=columns_bank(current, clocks, self.width), history=columns_bank(selected, clocks, self.width),
            text=text, leaf_count=leaf_count, read_seconds=self.read_seconds,
            selection_seconds=selection_seconds, residual_seconds=residual_seconds, read_bytes=self.read_bytes,
            index_hits=self.hits - before_hits, index_misses=self.misses - before_misses,
            debug={"method": method, "budget": budget, "columns": selected, "roots": roots,
                   "selected_nodes": selected_nodes, "leaves": leaves} if debug else None,
        )

    def close(self):
        self.base_files.close()
        self.hierarchy_files.close()
        self.indices.clear()
        self.index_bytes = 0
