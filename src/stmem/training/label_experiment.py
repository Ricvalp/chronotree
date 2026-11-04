"""Resumable independent GPU shards for matched velocity/final-action labels.

One shard shares immutable evidence and teacher predictions across both readers
and label objectives. No distributed gradient communication or simulator is used.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from copy import copy
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from stmem.config import dependency, digest, load_config, source_revision
from stmem.data.cache import FrozenCache
from stmem.data.hierarchy_cache import validate_hierarchy_cache
from stmem.models.router import CachedScorer
from stmem.tracking import RunTracker
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.label_data import LabelDataReader, LabelPlan
from stmem.training.label_objectives import OBJECTIVES, evaluate_batch, variant_name
from stmem.training.runtime import load_policy, load_router, validate_policy_config, validate_selection


_READER = None
_SCORERS = None
ROW_FIELDS = ("query", "candidate", "gain", "context_id", "partition", "budget", "episode")


def output_path(cfg, objective, reader, phase, shard):
    variant = variant_name(objective, cfg["labels"]["integration_steps"])
    root = Path(cfg["labels"]["experiment_root"])
    return root / variant / f"seed{cfg['seed']}" / f"router_{reader}" / f"labels_{phase}" / f"shard_{shard:05d}.h5"


def _episode_name(data, index):
    if data.synthetic:
        task, episode, _ = data.refs[index]
        return f"{task}/episode_{episode}"
    ref = data.dataset.episodes[data.refs[index]]
    return f"{ref.task}/{ref.key}"


def validate_models(cfg, teacher, teacher_state, compact, compact_state):
    """A new decoder objective must not silently change cached evidence."""
    if teacher_state["stage"] != "teacher" or compact_state["stage"] != "compact":
        raise ValueError("Expected a full-history teacher and shared compact checkpoint")
    for state in (teacher_state, compact_state):
        validate_policy_config(state["config"], cfg)
    if cfg["model"].get("decoder", "flow") != "flow" or cfg["model"].get("history_mode", "full") != "full":
        raise ValueError("Router utilities require a full-history flow decoder")
    for key, tensor in teacher.state_dict().items():
        if key.startswith(("encoder.", "text_encoder.", "normalizer.")):
            other = compact.state_dict().get(key)
            if other is None or not torch.equal(tensor.cpu(), other.cpu()):
                raise ValueError(f"Compact checkpoint changed frozen encoder/normalization: {key}")


def _load_scorers(cfg, expected_identity=None):
    scorers = {}
    dependency_hashes = {name: (expected_identity[name] if expected_identity is not None
                                else file_sha256(dependency(cfg, name))) for name in ("teacher", "compact")}
    for key, path in cfg["labels"]["refresh_router_checkpoints"].items():
        objective, reader = key.split("/")
        if objective not in cfg["labels"]["objectives"] or reader not in ("tree", "flat"):
            raise ValueError(f"Unknown refresh reader key: {key}")
        router, state = load_router(path, "cpu")
        recorded = state["config"].get("labels", {})
        if (state.get("reader", state["config"].get("reader")) != reader
                or recorded.get("objective", "velocity") != objective
                or recorded.get("integration_steps", 8) != cfg["labels"]["integration_steps"]):
            raise ValueError(f"Refresh router objective/reader/solver mismatch: {path}")
        provenance = state.get("provenance", {})
        for name in ("teacher", "compact"):
            if provenance.get(name) != dependency_hashes[name]:
                raise ValueError(f"Refresh router uses a different frozen {name}: {path}")
        if expected_identity is not None:
            actual_identity = state.get("label_identity", {})
            for name in ("teacher", "compact", "cache", "data", "partition", "source_sha256"):
                if actual_identity.get(name) != expected_identity[name]:
                    raise ValueError(f"Refresh router label {name} identity differs: {path}")
        scorers[key] = CachedScorer(router, file_sha256(path))
    expected = {f"{obj}/{reader}" for obj in cfg["labels"]["objectives"] for reader in ("tree", "flat")}
    if set(scorers) != expected:
        raise ValueError("Refresh requires one initial router checkpoint per objective and reader")
    return scorers


def _initialize_worker(data, cfg, cache_root, hierarchy_root, refresh):
    global _READER, _SCORERS
    torch.set_num_threads(1)
    _READER = LabelDataReader(data, cfg, cache_root, hierarchy_root)
    _SCORERS = _load_scorers(cfg) if refresh else None


def _prepare(item):
    return _READER.prepare(item, scorers=_SCORERS)


def _payload_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if hasattr(value, "content") and hasattr(value, "metadata"):
        return sum(_payload_bytes(getattr(value, key)) for key in ("content", "metadata", "valid", "log_count"))
    if isinstance(value, dict):
        return sum(_payload_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_bytes(item) for item in value)
    return 0


def prepared_items(data, cfg, cache_root, hierarchy_root, plan, *, refresh, workers, prefetch_bytes):
    """Bound in-flight payload by conservative planned bytes and worker count.

    An individually oversized history is allowed alone and is never truncated.
    Estimates bound result buffers; per-worker topology caches are separately
    bounded by LabelDataReader and recorded in the resource documentation.
    """
    if not workers:
        reader = LabelDataReader(data, cfg, cache_root, hierarchy_root)
        scorers = _load_scorers(cfg) if refresh else None
        try:
            for item in plan:
                yield reader.prepare(item, scorers=scorers)
        finally:
            reader.close()
        return
    # Fallback bounds use declared maximum history sizes, not empirical means.
    m = cfg["model"]
    max_frames = m.get("max_execution_frames", 1300) + m.get("max_conditioning_frames", 1300)
    max_leaves = max_frames * (m["visual_tokens"] + m["block_size"]) // m["block_size"]
    fallback = (max_leaves + 20 * (max(cfg["labels"]["budgets"]) + m["current_tokens"])) * (m["width"] + 30) * 4
    # The parent has already validated/catalogued the dataset. Workers need only
    # its small episode references, feature manifests and partitions, not a new
    # full index scan or a replica of every (episode, decision) candidate tuple.
    worker_data = copy(data)
    worker_data.candidates = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_initialize_worker,
                             initargs=(worker_data, cfg, str(cache_root), str(hierarchy_root), refresh)) as pool:
        pending, cursor, reserved = deque(), 0, 0
        while cursor < len(plan) or pending:
            while cursor < len(plan) and len(pending) < workers * 2:
                estimate = int(plan[cursor].get("estimated_bytes", fallback))
                if pending and reserved + estimate > prefetch_bytes:
                    break
                pending.append((pool.submit(_prepare, plan[cursor]), estimate))
                reserved += estimate
                cursor += 1
            future, estimate = pending.popleft()
            result = future.result()
            reserved -= estimate
            result.setdefault("timings", {})["payload_bytes"] = _payload_bytes(result)
            yield result


class LabelShard:
    """Single-writer append transactions with atomic completion and exact IDs."""

    def __init__(self, path, identity, width):
        self.path, self.identity = Path(path), identity
        self.expected_ids = set(range(identity["shard_index"], identity["contexts"], identity["shard_count"]))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(f"Label shard already has a writer: {self.path}") from None
        self.partial = self.path.with_suffix(".partial.h5")
        self.completed = self.path.exists()
        self.file = h5py.File(self.path if self.completed else self.partial, "r" if self.completed else "a")
        f = self.file
        try:
            if "identity" in f.attrs:
                if json.loads(f.attrs["identity"]) != identity:
                    raise ValueError(f"Label shard inputs changed: {self.path}")
                if self.completed and not f.attrs.get("complete"):
                    raise ValueError(f"Published label shard is incomplete: {self.path}")
            else:
                f.attrs["identity"] = json.dumps(identity, sort_keys=True)
                f.attrs["committed_rows"] = 0
                f.attrs["committed_contexts"] = 0
                f.attrs["commit"] = json.dumps([0, 0])
                for name, shape, dtype in [
                    ("query", (2 * width + 27,), "f4"), ("candidate", (2 * width + 23,), "f4"),
                    ("gain", (), "f4"), ("context_id", (), "i8"), ("partition", (), "i1"),
                    ("budget", (), "i8"), ("episode", (), h5py.string_dtype()),
                    ("attempted_context_ids", (), "i8"),
                ]:
                    f.create_dataset(name, shape=(0, *shape), maxshape=(None, *shape), chunks=True, dtype=dtype)
            rows, contexts = json.loads(f.attrs.get("commit", json.dumps([
                int(f.attrs["committed_rows"]), int(f.attrs["committed_contexts"]),
            ])))
            for name in (*ROW_FIELDS, "attempted_context_ids"):
                length = contexts if name == "attempted_context_ids" else rows
                if len(f[name]) < length:
                    raise ValueError(f"Damaged committed label shard: {self.partial}")
                if not self.completed:
                    f[name].resize(length, axis=0)
            self.done = set(map(int, np.asarray(f["attempted_context_ids"])))
            if len(self.done) != contexts or not self.done <= self.expected_ids:
                raise ValueError("Duplicate or foreign attempted context IDs")
            if self.completed and self.done != self.expected_ids:
                raise ValueError("Completed label shard does not cover its planned contexts")
        except BaseException:
            self.close()
            raise

    def append(self, items, gains, reader_key):
        if len(items) != len(gains):
            raise ValueError("Every attempted context requires one gain vector or an explicit skip")
        if self.completed:
            return
        f = self.file
        arrays = {name: [] for name in ROW_FIELDS}
        attempted = []
        for item, gain in zip(items, gains):
            context = int(item["context_id"])
            if context in self.done:
                continue
            if context not in self.expected_ids:
                raise ValueError("Cannot append a context owned by another shard")
            attempted.append(context)
            reader = item["readers"].get(reader_key)
            if gain is None:
                if reader is not None and len(reader["after"]):
                    raise ValueError("Missing gain for an eligible counterfactual")
                continue
            n = len(gain)
            if n != len(reader["candidate"]) or not torch.isfinite(gain).all():
                raise ValueError("Malformed candidate label row")
            values = {
                "query": item["query"].numpy()[None].repeat(n, axis=0),
                "candidate": reader["candidate"].numpy(), "gain": gain.numpy(),
                "context_id": [context] * n, "partition": [int(item["partition"])] * n,
                "budget": [item["budget"]] * n, "episode": [item["episode"]] * n,
            }
            for name, value in values.items():
                arrays[name].append(np.asarray(value))
        if not attempted:
            return
        rows = len(f["gain"])
        count = sum(len(value) for value in arrays["gain"])
        if count:
            for name in ROW_FIELDS:
                f[name].resize(rows + count, axis=0)
                f[name][rows:] = np.concatenate(arrays[name])
        old = len(f["attempted_context_ids"])
        f["attempted_context_ids"].resize(old + len(attempted), axis=0)
        f["attempted_context_ids"][old:] = attempted
        # Publish the joint row/context commit only after payloads are durable.
        f.flush()
        f.attrs["committed_rows"] = rows + count
        f.attrs["committed_contexts"] = old + len(attempted)
        f.attrs["attempted_contexts"] = old + len(attempted)
        f.attrs["commit"] = json.dumps([rows + count, old + len(attempted)])
        f.flush()
        self.done.update(attempted)

    def finish(self):
        if self.done != self.expected_ids:
            raise ValueError("Cannot complete a partially generated label shard")
        if self.completed:
            return
        gains = np.asarray(self.file["gain"])
        summary = {"contexts": len(self.done), "labels": len(gains),
                   "positive": int((gains > 1e-6).sum()), "negative": int((gains < -1e-6).sum()),
                   "near_zero": int((np.abs(gains) <= 1e-6).sum())}
        self.file.attrs["complete"] = True
        self.file.attrs["summary"] = json.dumps(summary, sort_keys=True)
        self.file.flush()
        self.file.close()
        self.file = None
        os.replace(self.partial, self.path)
        atomic_json({"status": "complete", "identity": self.identity, **summary}, self.path.with_suffix(".json"))
        self.completed = True

    def close(self):
        if getattr(self, "file", None) is not None:
            self.file.close()
            self.file = None
        if getattr(self, "lock", None) is not None:
            self.lock.close()
            self.lock = None


def generate(cfg, *, phase="initial", shard_index=0, shards=1, device="cuda", workers=4,
             batch_size=4, teacher_token_budget=32768, compact_batch_size=64,
             prefetch_mb=512, stop_file=None):
    if phase not in ("initial", "refresh") or not 0 <= shard_index < shards:
        raise ValueError("Invalid label phase or shard assignment")
    if workers < 0 or min(batch_size, teacher_token_budget, compact_batch_size, prefetch_mb) < 1:
        raise ValueError("Invalid label worker/batching limits")
    objectives = tuple(cfg["labels"].get("objectives", OBJECTIVES))
    if not objectives or len(set(objectives)) != len(objectives):
        raise ValueError("Specify distinct label objectives")
    integration_steps = cfg["labels"]["integration_steps"]
    for objective in objectives:
        variant_name(objective, integration_steps)
    refresh = phase == "refresh"
    torch.set_num_threads(1 if device.startswith("cuda") else min(4, os.cpu_count() or 1))
    teacher_path, compact_path = (dependency(cfg, name) for name in ("teacher", "compact"))
    startup = time.perf_counter()
    def report_startup(row):
        print(json.dumps({"event": "label_startup", "shard": shard_index, **row}), flush=True)
    data = TrainingData(cfg, progress=report_startup)
    for path in (teacher_path, compact_path):
        validate_selection(path, debug=bool(cfg.get("debug") and data.synthetic))
    cache = FrozenCache(cfg, teacher_path, data)
    hierarchy_root = cfg["training"]["hierarchy_cache_root"]
    validate_hierarchy_cache(cache.root, hierarchy_root)
    teacher, teacher_state = load_policy(teacher_path, device)
    compact, compact_state = load_policy(compact_path, device)
    validate_models(cfg, teacher, teacher_state, compact, compact_state)
    for model in (teacher, compact):
        model.freeze_encoders()
        model.eval().requires_grad_(False)
    plan = LabelPlan(data, cfg, refresh=refresh)
    all_items = plan.items()
    owned = [item for item in all_items if item["context_id"] % shards == shard_index]
    partition = {_episode_name(data, i): data.label_partition(i) for i in range(len(data.refs))}
    shared = {
        "schema_version": 2, "teacher": file_sha256(teacher_path), "compact": file_sha256(compact_path),
        "source_sha256": source_revision()["source_sha256"],
        "cache": digest(cache.manifest), "data": data.identity, "partition": partition,
        "contexts": len(all_items), "sample_plan": digest(all_items), "refresh": refresh,
        "shard_index": shard_index, "shard_count": shards, "integration_steps": integration_steps,
        "label_sampling": {key: cfg["labels"][key] for key in
                           ("seed", "budgets", "candidates", "exploration_fraction", "split_fraction")},
        "precision": "fp32", "action_output": "normalized_joint_targets",
        "final_action_noise": "unmasked_gaussian", "solver": "euler",
    }
    if refresh:
        # Validate every initial scorer's fixed scientific dependencies before work.
        checked = _load_scorers(cfg, shared)
        del checked
    with ExitStack() as stack:
        writers = {}
        for objective in objectives:
            for reader in ("tree", "flat"):
                identity = {**shared, "reader": reader, "objective": objective,
                            "router": file_sha256(cfg["labels"]["refresh_router_checkpoints"][f"{objective}/{reader}"])
                            if refresh else None}
                writer = LabelShard(output_path(cfg, objective, reader, phase, shard_index), identity, cfg["model"]["width"])
                stack.callback(writer.close)
                writers[objective, reader] = writer
        pending = [item for item in owned if any(item["context_id"] not in writer.done for writer in writers.values())]
        if not pending:
            for writer in writers.values():
                writer.finish()
            return 0
        tracking_dir = Path(cfg["labels"]["experiment_root"]) / "tracking" / f"labels_{phase}_{shard_index:05d}"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        with RunTracker(cfg, tracking_dir, job_type=f"labels_{phase}",
                        name=f"seed{cfg['seed']}-labels-{phase}-shard{shard_index}",
                        metadata={**shared, "objectives": objectives}, resume=any(writer.done for writer in writers.values())) as tracker:
            tracker.summary({"startup_seconds": time.perf_counter() - startup})
            started, completed = time.perf_counter(), len(owned) - len(pending)
            stream = prepared_items(data, cfg, cache.root, hierarchy_root, pending,
                                    refresh=refresh, workers=workers, prefetch_bytes=prefetch_mb * 1024**2)
            stack.callback(stream.close)
            iterator, exhausted, consumed = iter(stream), False, 0
            while not exhausted:
                if stop_file and Path(stop_file).exists():
                    tracker.finish(status="checkpointed_partial", exit_code=75)
                    return 75
                batch, loading, batch_bytes = [], time.perf_counter(), 0
                for _ in range(batch_size):
                    if consumed < len(pending):
                        estimate = int(pending[consumed].get("estimated_bytes", 0))
                        if batch and batch_bytes + estimate > prefetch_mb * 1024**2:
                            break
                    try:
                        payload = next(iterator)
                        batch.append(payload)
                        batch_bytes += _payload_bytes(payload)
                        consumed += 1
                    except StopIteration:
                        exhausted = True
                        break
                if not batch:
                    break
                wait_seconds = time.perf_counter() - loading
                gains, timings = evaluate_batch(
                    teacher, compact, batch, objectives=objectives, integration_steps=integration_steps,
                    teacher_token_budget=teacher_token_budget, compact_batch_size=compact_batch_size,
                )
                writing = time.perf_counter()
                for (objective, reader), writer in writers.items():
                    key = f"{objective}/{reader}" if refresh else reader
                    writer.append(batch, gains.get(objective, {}).get(key, [None] * len(batch)), key)
                completed += len(batch)
                elapsed = time.perf_counter() - started
                row = {"event": "label_progress", "phase": phase, "shard": shard_index,
                       "contexts_completed": completed, "contexts_planned": len(owned),
                       "labels_written": sum(len(writer.file["gain"]) if writer.file else 0 for writer in writers.values()),
                       "elapsed_seconds": elapsed, "data_wait_seconds": wait_seconds,
                       "write_seconds": time.perf_counter() - writing, **timings,
                       "paired_contexts_per_second": (completed - (len(owned) - len(pending))) / max(elapsed, 1e-9),
                       "history_leaves_max": max(item["history_leaves"] for item in batch),
                       "payload_bytes": sum(_payload_bytes(item) for item in batch),
                       "allocated_gpu_hours": elapsed * int(os.environ.get(
                           "STMEM_ALLOCATED_GPUS", "1" if teacher.device.type == "cuda" else "0")) / 3600}
                for key in ("load_seconds", "read_seconds", "read_bytes", "selection_seconds", "prepare_seconds"):
                    row["worker_" + key] = sum(item.get("timings", {}).get(key, 0) for item in batch)
                row["eta_seconds"] = (len(owned) - completed) / max(row["paired_contexts_per_second"], 1e-9)
                with (tracking_dir / "metrics.jsonl").open("a") as out:
                    out.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                tracker.log(row, step=completed)
            for writer in writers.values():
                writer.finish()
            tracker.summary({"status": "complete", "contexts": len(owned), "elapsed_seconds": time.perf_counter() - started})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("initial", "refresh"), default="initial")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--teacher-token-budget", type=int, default=32768)
    parser.add_argument("--compact-batch-size", type=int, default=64)
    parser.add_argument("--prefetch-mb", type=int, default=512)
    parser.add_argument("--stop-file", default=os.environ.get("STMEM_STOP_FILE"))
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    raise SystemExit(generate(cfg, phase=args.phase, shard_index=args.shard_index, shards=args.shards,
                             device=args.device, workers=args.workers, batch_size=args.batch_size,
                             teacher_token_budget=args.teacher_token_budget, compact_batch_size=args.compact_batch_size,
                             prefetch_mb=args.prefetch_mb, stop_file=args.stop_file))


if __name__ == "__main__":
    main()
