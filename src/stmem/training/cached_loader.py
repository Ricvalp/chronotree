"""Bounded CPU preparation of frozen compact-policy contexts.

Workers read supervision and cached learned leaves, build/reuse causal block
trees, and select contexts before transferring only flat tensors to the trainer.
No raw point clouds, CLIP descriptors, model parameters, or recursive records
cross the multiprocessing queue. This consumes the existing encoder-cache format.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from stmem.data.cache import episode_key
from stmem.data.canonical import aligned_action_chunk, action_alignment_offset
from stmem.memory.hierarchy import Archive, local_tree, stable_id, walk_records
from stmem.memory.records import MemoryRecord, aggregate, record_bytes
from stmem.models.batched_policy import TokenBank, records_bank
from stmem.models.policy import Evidence, choose_records
from stmem.training.loader import PolicyBatchLoader, _Microbatches, _identity, _initialize_worker


@dataclass(frozen=True)
class CachedDecisionInfo:
    task: str
    episode_id: str
    index: int
    conditioning_frames: int
    execution_history_frames: int


@dataclass
class LoadedCachedBatch:
    positions: tuple[int, ...]
    target_chunk: torch.Tensor
    target_valid: torch.Tensor
    current: TokenBank
    history: TokenBank
    text: torch.Tensor
    current_state: torch.Tensor
    clocks: torch.Tensor
    history_lengths: tuple[int, ...]
    history_leaf_counts: torch.Tensor
    metadata: tuple[CachedDecisionInfo, ...]
    load_seconds: float
    pack_seconds: float
    worker_pid: int
    cache_seconds: float
    context_seconds: float
    block_cache_hits: int
    block_cache_misses: int
    hierarchy_read_seconds: float = 0.0
    hierarchy_select_seconds: float = 0.0
    hierarchy_residual_seconds: float = 0.0
    hierarchy_read_bytes: int = 0
    hierarchy_enabled: bool = False

    def pin_memory(self):
        def bank(value):
            return TokenBank(*(getattr(value, key).pin_memory()
                               for key in ("content", "metadata", "valid", "log_count")))

        return replace(
            self,
            **{key: getattr(self, key).pin_memory() for key in (
                "target_chunk", "target_valid", "text", "current_state", "clocks", "history_leaf_counts"
            )},
            current=bank(self.current), history=bank(self.history),
        )


@dataclass
class _SmallEpisode:
    task: str
    episode_id: str
    actions: torch.Tensor
    action_valid: torch.Tensor
    proprio: torch.Tensor
    execution_indices: np.ndarray
    conditioning_indices: np.ndarray
    alignment: str


@dataclass
class _CachedDecision:
    info: CachedDecisionInfo
    target: torch.Tensor
    target_valid: torch.Tensor
    state: torch.Tensor
    clocks: tuple[float, float]


class _Files:
    def __init__(self, maximum=16):
        self.maximum = maximum
        self.handles = OrderedDict()

    def get(self, path):
        path = str(path)
        handle = self.handles.pop(path, None)
        if handle is None:
            handle = h5py.File(path, "r")
        self.handles[path] = handle
        while len(self.handles) > self.maximum:
            _, stale = self.handles.popitem(last=False)
            stale.close()
        return handle

    def close(self):
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def _read_records(group):
    """Bulk conversion of immutable columns; tensors retain contiguous storage.

    The reference reader copies five tiny NumPy arrays for every leaf. Here each
    column owns one array/tensor storage and scalar columns convert once. Records
    expose views of that storage; all downstream adapters/aggregations create new
    tensors, so cached leaf contents remain immutable.
    """
    length = int(group.attrs["length"])
    if not length:
        return []
    tensor_names = {"content_sum", "anchor", "bounds_min", "bounds_max", "max_content"}
    values = {}
    for key, dataset in group.items():
        array = np.asarray(dataset)
        if key in tensor_names:
            values[key] = torch.from_numpy(array).float()
        else:
            rows = array.tolist()
            values[key] = [value.decode() if isinstance(value, bytes) else value for value in rows]
    return [MemoryRecord(**{key: column[index] for key, column in values.items()})
            for index in range(length)]


class LightweightDecisions:
    """Read small immutable supervision/state arrays, never observation payloads."""

    def __init__(self, data, *, maximum_episodes=16):
        self.data = data
        self.maximum_episodes = maximum_episodes
        self.episodes = OrderedDict()
        self.files = _Files()

    def _episode(self, index):
        episode = self.episodes.pop(index, None)
        if episode is None:
            if self.data.synthetic:
                original = self.data.episode(index)
                episode = _SmallEpisode(
                    original.task, original.episode_id, original.actions, original.action_valid,
                    torch.stack([frame.proprio for frame in original.execution]),
                    np.asarray([frame.source_index for frame in original.execution]),
                    np.asarray([frame.source_index for frame in original.conditioning]), original.alignment,
                )
            else:
                dataset = self.data.dataset
                ref = dataset.episodes[self.data.refs[index]]
                dataset._require_feature_episode(ref)
                alignment = dataset._alignment_for(ref)
                action_alignment_offset(alignment)
                group = self.files.get(ref.path)[ref.key]
                dataset._validate_episode_layout(group)
                source = np.asarray(group["source"])
                execution = ~source
                source_indices = np.asarray(group["source_index"])
                episode = _SmallEpisode(
                    ref.task, ref.key,
                    torch.from_numpy(np.asarray(group["actions"])[execution]),
                    torch.from_numpy(np.asarray(group["action_valid"])[execution]),
                    torch.from_numpy(np.asarray(group["proprio"])[execution]),
                    source_indices[execution], source_indices[source], alignment,
                )
        self.episodes[index] = episode
        while len(self.episodes) > self.maximum_episodes:
            self.episodes.popitem(last=False)
        return episode

    def decision(self, position):
        episode_index, index = self.data.sample_reference(position)
        episode = self._episode(episode_index)
        stride = self.data.observation_stride
        if not 0 <= index < len(episode.actions) or episode.execution_indices[index] % stride:
            raise IndexError("Cached decision must use the validated retained observation cadence")
        target, valid = aligned_action_chunk(
            episode.actions, episode.action_valid, index, self.data.horizon,
            episode.alignment, self.data.action_stride,
        )
        conditioning = episode.conditioning_indices[episode.conditioning_indices % stride == 0]
        previous_execution = episode.execution_indices[:index]
        execution_count = int((previous_execution % stride == 0).sum())
        previous = torch.zeros_like(episode.actions[index])
        available = False
        if index > 0:
            command_index = index if episode.alignment in ("post_action_v1", "synthetic_post_action_v1") else index - 1
            available = bool(episode.action_valid[command_index, :7].all())
            if available:
                previous = episode.actions[command_index]
        state = torch.cat([episode.proprio[index], previous, previous.new_tensor([float(available)])]).float()
        return _CachedDecision(
            CachedDecisionInfo(episode.task, episode.episode_id, index, len(conditioning), execution_count),
            target, valid, state,
            (float(conditioning[-1]) if len(conditioning) else 0.0, float(episode.execution_indices[index])),
        )

    def close(self):
        self.files.close()
        self.episodes.clear()


@dataclass(frozen=True)
class _Block:
    leaves: tuple
    roots: tuple
    estimated_bytes: int


@dataclass
class _PreparedArchive(Archive):
    pending_roots: dict = field(default_factory=dict)

    @property
    def roots(self):
        roots = [record for key in sorted(self.forests) for record in self.forests[key]]
        for source in sorted(self.pending_roots):
            roots.extend(root for _, root in self.pending_roots[source])
        return roots


class CachedArchiveReader:
    """Per-worker HDF5 handles and bounded reusable immutable block trees.

    Limits bound both retained block count and estimated CPU payload/object bytes.
    A decision's archive is temporary and released immediately after selected
    tensors are packed. Completed and prefix variants have different cache keys.
    """

    def __init__(self, cfg, root, *, maximum_blocks=64, maximum_bytes=128 * 1024 * 1024):
        self.cfg, self.root = cfg, Path(root)
        self.maximum_blocks, self.maximum_bytes = maximum_blocks, maximum_bytes
        self.blocks = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0
        self.files = _Files()

    def _block(self, group, name, key):
        cache_key = group.file.filename, group.name, name, str(key)
        block = self.blocks.pop(cache_key, None)
        if block is not None:
            self.hits += 1
            self.blocks[cache_key] = block
            return block
        self.misses += 1
        leaves = tuple(_read_records(group[name][str(key)]))
        groups = defaultdict(list)
        for leaf in leaves:
            groups[leaf.group].append(leaf)
        roots = tuple((kind, local_tree(members)) for kind, members in sorted(groups.items()))
        nodes = walk_records(root for _, root in roots)
        estimated = record_bytes(nodes) + 1024 * len(nodes)
        block = _Block(leaves, roots, estimated)
        if self.maximum_blocks and estimated <= self.maximum_bytes:
            self.blocks[cache_key] = block
            self.bytes += estimated
            while len(self.blocks) > self.maximum_blocks or self.bytes > self.maximum_bytes:
                _, evicted = self.blocks.popitem(last=False)
                self.bytes -= evicted.estimated_bytes
        return block

    @staticmethod
    def _append(archive, block, block_id):
        if not block.leaves:
            return
        if len({leaf.source for leaf in block.leaves}) != 1 or not all(leaf.completed for leaf in block.leaves):
            raise ValueError("Completed cached block crosses source boundaries or contains prefix leaves")
        if block_id in archive.completed_blocks:
            raise ValueError("Duplicate cached completed block")
        archive.completed_blocks[block_id] = block.leaves
        for kind, root in block.roots:
            forest = archive.forests.setdefault(kind, [])
            if forest and root.time_min < forest[-1].time_min:
                raise ValueError("Cached blocks must be in source-local chronological order")
            while forest and forest[-1].scale == root.scale:
                previous = forest.pop()
                root = aggregate(
                    (previous, root), stable_id("temporal", (previous.node_id, root.node_id)),
                    children=True, scale=root.scale + 1,
                )
            forest.append(root)

    def evidence(self, decision):
        info = decision.info
        group = self.files.get(self.root / f"{info.task}.h5")[episode_key(info.task, info.episode_id)]
        archive = _PreparedArchive()
        for key in sorted(group["conditioning"], key=int):
            self._append(archive, self._block(group, "conditioning", key), f"conditioning/{key}")
        block_size = self.cfg["model"]["block_size"]
        nfull, pending = divmod(info.execution_history_frames, block_size)
        for index in range(nfull):
            self._append(
                archive, self._block(group, "execution", index * block_size), f"execution/{index}"
            )
        if pending:
            block = self._block(group, "prefix", f"{nfull * block_size}_{pending}")
            archive.set_pending("execution", block.leaves)
            archive.pending_roots["execution"] = block.roots
        current = _read_records(group["current"][str(info.index)])
        text = torch.from_numpy(np.asarray(group["text"]).copy())
        current_frame = SimpleNamespace(
            proprio=decision.state[:15], previous_action=decision.state[15:23],
            action_available=bool(decision.state[23]), time=decision.clocks[1],
        )
        return Evidence(
            archive, current, text, current_frame,
            {"conditioning": decision.clocks[0], "execution": decision.clocks[1]},
        )

    def close(self):
        self.files.close()
        self.blocks.clear()
        self.bytes = 0


def selected_context(evidence, cfg, position):
    """The reference context_sample protocol, without a model dependency."""
    rng = random.Random(cfg["seed"] * 1000003 + position)
    mixture = cfg["training"]["context_mix"]
    families = [name.upper() for name, fraction in mixture.items() for _ in range(round(fraction * 20))]
    random.Random(cfg["seed"] + position // 20).shuffle(families)
    method = families[position % len(families)]
    budget = cfg["model"]["budgets"][(position // len(families)) % len(cfg["model"]["budgets"])]
    roots = len(evidence.archive.roots)
    if roots > budget:
        raise ValueError(f"{roots} roots exceed budget {budget}")
    if method != "CURRENT" and rng.random() < cfg["training"]["intermediate_probability"]:
        budget = rng.randint(max(roots, 1), budget)
    return choose_records(evidence, method, budget, rng=rng).records


def _combine_banks(rows):
    return TokenBank(*(pad_sequence([getattr(row, key)[0] for row in rows], batch_first=True)
                       for key in ("content", "metadata", "valid", "log_count")))


class CachedPolicyDataset(Dataset):
    """One item is one microbatch, with selected tensors bounded by context budget."""

    def __init__(self, data, cache_root):
        self.data, self.cache_root = data, Path(cache_root)
        self._decisions = self._reader = None

    def __getstate__(self):
        result = self.__dict__.copy()
        result["_decisions"] = result["_reader"] = None
        return result

    def __getitem__(self, positions):
        if self._reader is None:
            hierarchy_root = self.data.cfg["training"].get("hierarchy_cache_root")
            if hierarchy_root:
                from stmem.training.packed_context import PackedContextReader

                self._reader = PackedContextReader(self.data.cfg, self.cache_root, hierarchy_root)
            else:
                self._reader = CachedArchiveReader(self.data.cfg, self.cache_root)
            self._decisions = LightweightDecisions(self.data)
        reader = self._reader
        hierarchy_enabled = bool(self.data.cfg["training"].get("hierarchy_cache_root"))
        hits, misses = reader.hits, reader.misses
        policy = SimpleNamespace(config=self.data.cfg, device=torch.device("cpu"))
        current, history, text, decisions, leaves = [], [], [], [], []
        load_seconds = cache_seconds = context_seconds = pack_seconds = 0.0
        hierarchy_read_seconds = hierarchy_select_seconds = hierarchy_residual_seconds = 0.0
        hierarchy_read_bytes = 0
        for position in positions:
            started = time.perf_counter()
            decision = self._decisions.decision(position)
            load_seconds += time.perf_counter() - started
            started = time.perf_counter()
            if hierarchy_enabled:
                prepared = reader.prepare(decision, position)
                cache_seconds += time.perf_counter() - started
                current.append(prepared.current)
                history.append(prepared.history)
                text.append(prepared.text)
                leaves.append(prepared.leaf_count)
                decisions.append(decision)
                hierarchy_read_seconds += prepared.read_seconds
                hierarchy_select_seconds += prepared.selection_seconds
                hierarchy_residual_seconds += prepared.residual_seconds
                hierarchy_read_bytes += prepared.read_bytes
                context_seconds += prepared.selection_seconds
                continue
            evidence = reader.evidence(decision)
            cache_seconds += time.perf_counter() - started
            started = time.perf_counter()
            records = selected_context(evidence, self.data.cfg, position)
            context_seconds += time.perf_counter() - started
            started = time.perf_counter()
            clocks = torch.tensor([decision.clocks], dtype=torch.float32)
            current.append(records_bank(policy, [evidence.current_records], clocks))
            history.append(records_bank(policy, [records], clocks))
            text.append(evidence.text)
            leaves.append(len(evidence.archive.leaves))
            decisions.append(decision)
            # Parent records retain their entire descendant tree. Drop them
            # before the next example; only the compact tensors accumulate.
            del evidence, records
            pack_seconds += time.perf_counter() - started
        started = time.perf_counter()
        result = LoadedCachedBatch(
            positions=tuple(positions), target_chunk=torch.stack([row.target for row in decisions]),
            target_valid=torch.stack([row.target_valid for row in decisions]),
            current=_combine_banks(current), history=_combine_banks(history), text=torch.stack(text),
            current_state=torch.stack([row.state for row in decisions]),
            clocks=torch.tensor([row.clocks for row in decisions], dtype=torch.float32),
            history_lengths=tuple(row.info.conditioning_frames + row.info.execution_history_frames
                                  for row in decisions),
            history_leaf_counts=torch.tensor(leaves, dtype=torch.long),
            metadata=tuple(row.info for row in decisions),
            load_seconds=load_seconds, pack_seconds=0.0, worker_pid=os.getpid(),
            cache_seconds=cache_seconds, context_seconds=context_seconds,
            block_cache_hits=reader.hits - hits, block_cache_misses=reader.misses - misses,
            hierarchy_enabled=hierarchy_enabled, hierarchy_read_seconds=hierarchy_read_seconds,
            hierarchy_select_seconds=hierarchy_select_seconds, hierarchy_residual_seconds=hierarchy_residual_seconds,
            hierarchy_read_bytes=hierarchy_read_bytes,
        )
        result.pack_seconds = pack_seconds + time.perf_counter() - started
        return result

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._decisions.close()
            self._reader = self._decisions = None


class CachedPolicyBatchLoader(PolicyBatchLoader):
    def __init__(self, data, cache, *, stage="compact", pin_memory=False, **kwargs):
        if stage != "compact":
            raise ValueError("Bounded selected-context loader currently supports compact training only")
        super().__init__(data, **kwargs)
        self.cache_root = cache.root
        self.pin_memory = pin_memory
        self._dataset = None

    def __enter__(self):
        if self._closed or self._loader is not None:
            raise RuntimeError("A cached policy loader can only be started once")
        sampler = _Microbatches(self.start, self.steps, self.batch, self.rank, self.world, self.microbatch)
        kwargs = {}
        if self.workers:
            kwargs.update(
                multiprocessing_context="spawn", persistent_workers=True,
                prefetch_factor=self.prefetch_factor, worker_init_fn=_initialize_worker,
            )
        self._dataset = CachedPolicyDataset(self.data, self.cache_root)
        generator = torch.Generator().manual_seed(self.data.cfg["seed"] + self.rank)
        self._loader = DataLoader(
            self._dataset, batch_size=None, sampler=sampler, collate_fn=_identity,
            num_workers=self.workers, generator=generator, in_order=True,
            pin_memory=self.pin_memory, **kwargs,
        )
        if self.steps is None or len(sampler):
            try:
                self._iterator = iter(self._loader)
            except BaseException:
                self.close()
                raise
        return self

    def close(self):
        try:
            super().close()
        finally:
            if self._dataset is not None:
                self._dataset.close()
