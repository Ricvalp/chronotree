#!/usr/bin/env python3
"""Bounded local real-cache diagnostic; no model updates or cache mutations.

This deliberately uses one existing training episode. Repeated-prefix timings
favor immutable-block reuse and are not an estimate of a 1,600-episode HPC run.
The reference timing isolates the formerly synchronous cache/context/packing
work, excluding raw observation loading. Only the requested report is written.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import dataclasses
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from stmem.config import source_revision
from stmem.data.cache import FrozenCache
from stmem.data.canonical import valid_decision_indices
from stmem.data.dataset import EpisodeDataset
from stmem.models.batched_policy import records_bank
from stmem.train import context_sample
from stmem.training.cached_loader import (
    CachedArchiveReader, CachedPolicyBatchLoader, CachedPolicyDataset, LightweightDecisions,
)
from stmem.training.data import TrainingData
from stmem.training.checkpoints import file_sha256
from stmem.training.loader import _Microbatches, _identity, _initialize_worker
from stmem.training.runtime import load_checkpoint


class NoReuseDataset(CachedPolicyDataset):
    """Diagnostic-only variant approximating a cold block on every sample."""

    def __getitem__(self, positions):
        if self._reader is None:
            hierarchy = self.data.cfg["training"].get("hierarchy_cache_root")
            if hierarchy:
                from stmem.training.packed_context import PackedContextReader

                self._reader = PackedContextReader(
                    self.data.cfg, self.cache_root, hierarchy, maximum_indices=0,
                )
            else:
                self._reader = CachedArchiveReader(self.data.cfg, self.cache_root, maximum_blocks=0)
            self._decisions = LightweightDecisions(self.data)
        return super().__getitem__(positions)


class NoReuseLoader(CachedPolicyBatchLoader):
    """Same bounded sampler/worker ordering, with only block reuse disabled."""

    def __enter__(self):
        sampler = _Microbatches(self.start, self.steps, self.batch, self.rank, self.world, self.microbatch)
        kwargs = {}
        if self.workers:
            kwargs.update(multiprocessing_context="spawn", persistent_workers=True,
                          prefetch_factor=self.prefetch_factor, worker_init_fn=_initialize_worker)
        self._dataset = NoReuseDataset(self.data, self.cache_root)
        self._loader = DataLoader(
            self._dataset, batch_size=None, sampler=sampler, collate_fn=_identity,
            num_workers=self.workers, generator=torch.Generator().manual_seed(self.data.cfg["seed"] + self.rank),
            in_order=True, pin_memory=False, **kwargs,
        )
        self._iterator = iter(self._loader)
        return self


def bounded_data(args):
    state = load_checkpoint(args.checkpoint)
    cfg = deepcopy(state["config"])
    cfg["stage"] = "compact"
    if getattr(args, "hierarchy_root", None):
        cfg["training"]["hierarchy_cache_root"] = str(args.hierarchy_root.resolve())
    cfg["paths"].update(data_root=str(args.prepared.parent.resolve()),
                        cache_root=str(args.cache_root.resolve()))
    cfg["data"]["alignment_path"] = str(args.alignment.resolve())
    cfg["model"]["visual_features"]["cache_dir"] = str(args.feature_root.resolve())
    dataset = EpisodeDataset(args.prepared, tasks=[args.task], horizon=cfg["model"]["horizon"],
                             alignment_path=cfg["data"]["alignment_path"], visual_features_config=cfg)
    dataset.validate_preprocessing(cfg)
    episode = dataset.load_episode(0)
    times = list(valid_decision_indices(
        episode.action_valid, episode.alignment, horizon=cfg["model"]["horizon"],
        stride=cfg["data"]["decision_stride"], action_stride=cfg["model"].get("action_stride", 1),
    ))
    # Bypass the all-task training catalog constructor only for this explicitly
    # labeled one-episode diagnostic. Sampling remains TrainingData's real code.
    data = TrainingData.__new__(TrainingData)
    data.cfg, data.dataset = cfg, dataset
    data.tasks, data.refs, data.by_task = [args.task], [0], {args.task: [0]}
    data.synthetic, data.history_enabled = False, True
    data.horizon, data.stride = cfg["model"]["horizon"], cfg["data"]["decision_stride"]
    data.observation_stride = cfg["data"].get("observation_stride", 1)
    data.action_stride = cfg["model"].get("action_stride", 1)
    data.candidates = {args.task: [(0, t) for t in times]}
    cache = FrozenCache.__new__(FrozenCache)
    cache.cfg, cache.root = cfg, args.cache_root
    return data, cache, episode, state["step"]


def reference_batch(data, cache, episode, positions):
    policy = SimpleNamespace(config=data.cfg, device=torch.device("cpu"))
    decisions = [episode.decision(data.sample_reference(position)[1], data.horizon) for position in positions]
    started = time.perf_counter()
    evidence = [cache.evidence(decision) for decision in decisions]
    records = [context_sample(None, item, data.cfg, position) for item, position in zip(evidence, positions)]
    clocks = torch.tensor([[item.clocks["conditioning"], item.clocks["execution"]] for item in evidence])
    current = records_bank(policy, [item.current_records for item in evidence], clocks)
    history = records_bank(policy, records, clocks)
    elapsed = time.perf_counter() - started
    return SimpleNamespace(
        positions=tuple(positions), current=current, history=history, clocks=clocks,
        text=torch.stack([item.text for item in evidence]),
        target_chunk=torch.stack([item.target_chunk for item in decisions]),
        target_valid=torch.stack([item.target_valid for item in decisions]),
        current_state=torch.stack([torch.cat([item.current.proprio, item.current.previous_action,
            item.current.proprio.new_tensor([float(item.current.action_available)])]) for item in decisions]),
        history_leaf_counts=torch.tensor([len(item.archive.leaves) for item in evidence]),
        history_lengths=tuple(len(item.conditioning) + len(item.execution_prefix) for item in decisions),
    ), elapsed


def tensor_payload_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if dataclasses.is_dataclass(value):
        return sum(tensor_payload_bytes(getattr(value, field.name)) for field in dataclasses.fields(value))
    if isinstance(value, (tuple, list)):
        return sum(tensor_payload_bytes(item) for item in value)
    return 0


def assert_parity(reference, actual):
    assert reference.positions == actual.positions
    assert reference.history_lengths == actual.history_lengths
    for name in ("current", "history"):
        for field in ("content", "metadata", "valid", "log_count"):
            torch.testing.assert_close(getattr(getattr(reference, name), field),
                                       getattr(getattr(actual, name), field), atol=0, rtol=0)
    for name in ("text", "target_chunk", "target_valid", "clocks", "current_state", "history_leaf_counts"):
        torch.testing.assert_close(getattr(reference, name), getattr(actual, name), atol=0, rtol=0)


def summarize(seconds):
    ordered = sorted(seconds)
    return {"batches": len(seconds), "seconds": sum(seconds),
            "batch_seconds_median": statistics.median(seconds),
            "batch_seconds_p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]}


def benchmark(args):
    torch.set_num_threads(1)
    provenance = {
        "source": source_revision(), "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "inputs": {str(path.resolve()): file_sha256(path) for path in (
            args.checkpoint, args.prepared, args.cache_root / f"{args.task}.h5", args.alignment
        )},
    }
    data, cache, episode, checkpoint_step = bounded_data(args)
    count = len(data.candidates[args.task])
    unique_positions = tuple(range(min(count, args.microbatch * 2)))
    # A cold reader with reuse disabled separates parallelization from warm-LRU
    # benefits. Every sampled decision is unique in this first pass.
    reference, old_cold = reference_batch(data, cache, episode, unique_positions)
    cold = CachedPolicyDataset(data, cache.root)
    if not args.hierarchy_root:
        # Initialization owns lightweight action metadata; then disable block reuse.
        first = cold[unique_positions[:1]]
        del first
        cold._reader.close()
        cold._reader = CachedArchiveReader(data.cfg, cache.root, maximum_blocks=0)
    else:
        from stmem.training.packed_context import PackedContextReader

        cold._reader = PackedContextReader(data.cfg, cache.root, args.hierarchy_root, maximum_indices=0)
        cold._decisions = LightweightDecisions(data)
    started = time.perf_counter()
    new_cold = cold[unique_positions]
    new_cold_seconds = time.perf_counter() - started
    assert_parity(reference, new_cold)
    cold.close()
    report = {
        "kind": "bounded_real_episode_cached_loader_diagnostic_not_training_or_benchmark_result",
        "provenance": provenance,
        "task": args.task, "episode": episode.episode_id, "checkpoint_step": checkpoint_step,
        "unique_available_decisions": count, "examples": args.examples, "microbatch": args.microbatch,
        "block_reuse_disabled": args.disable_block_reuse,
        "index_reuse_disabled": args.disable_index_reuse,
        "hierarchy_root": str(args.hierarchy_root.resolve()) if args.hierarchy_root else None,
        "reference_scope": "cache read/rebuild, context selection and CPU packing; raw point-cloud reads excluded",
        "new_scope": "lightweight supervision, cached context preparation, CPU packing and IPC; no neural forward",
        "limitations": ["single episode; repeated prefixes favor block reuse",
                        "cold means no application block reuse; OS filesystem caches are not flushed",
                        "local storage and CPU timings are not HPC/shared-storage throughput",
                        "worker startup is reported separately; CUDA transfer/compute is not timed"],
        "cold_unique_no_block_reuse": {
            "examples": len(unique_positions), "reference_seconds": old_cold,
            "new_seconds": new_cold_seconds, "parity": "bit-exact selected tensors, targets, state and clocks",
        },
        "runs": [],
    }
    timings = []
    for start in range(0, args.examples, args.microbatch):
        _, elapsed = reference_batch(data, cache, episode, tuple(range(start, start + args.microbatch)))
        timings.append(elapsed)
    report["reference_serial"] = {**summarize(timings), "examples_per_second": args.examples / sum(timings)}
    print(json.dumps({"phase": "reference_done", **report["reference_serial"]}), flush=True)
    for workers in args.workers:
        data.cfg["training"].update(workers=workers, prefetch_factor=1)
        started = time.perf_counter()
        loader_type = NoReuseLoader if args.disable_block_reuse or args.disable_index_reuse else CachedPolicyBatchLoader
        loader = loader_type(
            data, cache, stage="compact", start=0, steps=args.examples // args.microbatch,
            batch=args.microbatch, rank=0, world=1, microbatch=args.microbatch,
        )
        waits, payloads, hits, misses, worker_cache_seconds, worker_pids = [], [], 0, 0, 0.0, set()
        with loader:
            entered = time.perf_counter()
            for step in range(args.examples // args.microbatch):
                waiting = time.perf_counter()
                batches = list(loader.read_update(step))
                waits.append(time.perf_counter() - waiting)
                batch = batches[0]
                payloads.append(tensor_payload_bytes(batch))
                hits += batch.block_cache_hits
                misses += batch.block_cache_misses
                worker_cache_seconds += batch.cache_seconds
                worker_pids.add(batch.worker_pid)
            completed = time.perf_counter()
        closed = time.perf_counter()
        # Initial dispatch may initialize one dataset per worker. Define steady
        # results explicitly after at least one microbatch from every worker.
        warm_batches = max(1, workers)
        steady = waits[warm_batches:]
        result = {
            "workers": workers, "worker_processes_observed": len(worker_pids),
            "loader_enter_seconds": entered - started,
            "first_batch_wait_seconds": waits[0], "consume_seconds": completed - entered,
            "shutdown_seconds": closed - completed, "total_seconds": closed - started,
            "consumed_examples_per_second": args.examples / (completed - entered),
            "warmup_batches_excluded": warm_batches,
            "steady_wait": summarize(steady) if steady else None,
            "selected_tensor_payload_bytes_max": max(payloads),
            ("index_cache_hits" if args.hierarchy_root else "block_cache_hits"): hits,
            ("index_cache_misses" if args.hierarchy_root else "block_cache_misses"): misses,
            "worker_cache_seconds_sum": worker_cache_seconds,
        }
        report["runs"].append(result)
        print(json.dumps({"phase": "workers_done", **result}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Report:", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt"))
    parser.add_argument("--prepared", type=Path, default=Path("data/inspection/VideoUnmask.h5"))
    parser.add_argument("--cache-root", type=Path, default=Path("runs/cache_checks/fv_20261102"))
    parser.add_argument("--hierarchy-root", type=Path,
                        help="Existing derived hierarchy for the same diagnostic leaf cache")
    parser.add_argument("--feature-root", type=Path, default=Path("data/inspection/features"))
    parser.add_argument("--alignment", type=Path, default=Path("data/alignment.json"))
    parser.add_argument("--task", default="VideoUnmask")
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--disable-block-reuse", action="store_true",
                        help="Diagnostic only: every block is rebuilt, approximating low-hit-rate workloads")
    parser.add_argument("--disable-index-reuse", action="store_true",
                        help="Diagnostic only: reread topology on every example, approximating low-hit-rate workloads")
    parser.add_argument("--output", type=Path,
                        default=Path("runs/compact_loader_diagnostic_20261103/report.json"))
    args = parser.parse_args()
    if not 1 <= args.microbatch <= 32 or not 1 <= args.examples <= 512 or args.examples % args.microbatch:
        parser.error("Require microbatch1–32 and examples1–512 divisible by microbatch")
    if any(not 0 <= count <= 16 for count in args.workers):
        parser.error("Workers must be between0 and16")
    if args.hierarchy_root and args.disable_block_reuse:
        parser.error("--disable-block-reuse applies only to the legacy tree-building reader")
    if args.disable_index_reuse and not args.hierarchy_root:
        parser.error("--disable-index-reuse requires --hierarchy-root")
    benchmark(args)


if __name__ == "__main__":
    main()
