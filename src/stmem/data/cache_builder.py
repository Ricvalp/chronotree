"""Task-parallel, episode-resumable construction of immutable encoder caches.

Only final per-task shards are consumed. Private episode commits bound restart
loss to one episode without relying on a partially written HDF5 remaining valid
following SIGKILL. A complete manifest is published only after all task hashes
have been verified. Temporary episode files are removed after shard publication.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import re
import time

import h5py
import torch

from stmem.config import digest, run_dir, source_revision
from stmem.data.cache_source import cache_code_fingerprint
from stmem.training.checkpoints import atomic_json, file_sha256


def read_json(path):
    return json.loads(Path(path).read_text())


def progress(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


@contextmanager
def locked(path, *, wait=False):
    with Path(path).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f"Another encoder-cache writer owns {path}") from error
        yield


def require_source(identity):
    if cache_code_fingerprint() != identity["code"]:
        raise ValueError(
            "Source changed during encoder-cache construction; use the original checkout or a new cache"
        )


def require_file(path, receipt, *, identity, kind):
    if (
        receipt.get("identity_sha256") != digest(identity)
        or receipt.get("kind") != kind
        or receipt.get("status") != "complete"
    ):
        raise ValueError(f"Stale encoder-cache receipt for {path}")
    if (
        not path.is_file()
        or path.stat().st_size != receipt.get("bytes")
        or file_sha256(path) != receipt.get("sha256")
    ):
        raise ValueError(f"Missing or corrupted immutable encoder-cache file: {path}")


def episode_ref(data, index):
    if data.synthetic:
        task, episode, _ = data.refs[index]
        return task, f"episode_{episode}"
    ref = data.dataset.episodes[data.refs[index]]
    return ref.task, ref.key


def hydrate_frames(frames, reader):
    """Load at most one local block of lazy descriptors, never a whole episode."""
    from stmem.data.feature_cache import FeatureRow

    lazy = [(i, f.visual_features) for i, f in enumerate(frames) if isinstance(f.visual_features, FeatureRow)]
    if not lazy:
        return frames
    values = reader.read([row for _, row in lazy])
    result = list(frames)
    for value, (i, row) in zip(values, lazy):
        result[i] = replace(frames[i], visual_features=value[: row.point_count])
    return result


class CachedDescriptorEncoder:
    """Reuse descriptor file mappings while preserving the encoder's FP32 path."""

    def __init__(self, encoder, reader):
        self.encoder, self.reader = encoder, reader
        self.block_size = encoder.block_size

    def encode_frames(self, frames, *args, **kwargs):
        return self.encoder.encode_frames(hydrate_frames(frames, self.reader), *args, **kwargs)

    def current(self, frame):
        return self.encoder.current(hydrate_frames([frame], self.reader)[0])


def write_episode(path, episode, policy, data, identity, check_stop):
    from types import SimpleNamespace
    from stmem.data.cache import episode_key, write_episode_cache
    from stmem.data.feature_cache import FeatureRowReader

    temporary = path.with_suffix(".partial.h5")
    try:
        with FeatureRowReader() as reader, h5py.File(temporary, "w") as out:
            out.attrs["identity_sha256"] = digest(identity)
            proxy = SimpleNamespace(
                encoder=CachedDescriptorEncoder(policy.encoder, reader), text_encoder=policy.text_encoder
            )
            group = out.create_group(episode_key(episode.task, episode.episode_id))
            write_episode_cache(group, episode, proxy, data.stride, data.horizon, check_stop=check_stop)
            out.flush()
        require_source(identity)
        check_stop()
        receipt = {
            "kind": "encoder_cache_episode",
            "status": "complete",
            "identity_sha256": digest(identity),
            "task": episode.task,
            "episode_id": episode.episode_id,
            "sha256": file_sha256(temporary),
            "bytes": temporary.stat().st_size,
        }
        # Receipt before rename permits recovery of a validated temporary file
        # if a signal lands between these two atomic operations.
        atomic_json(receipt, path.with_suffix(".json"))
        os.replace(temporary, path)
        return receipt
    finally:
        # Keep a receipt-backed pending commit for recovery; discard partial
        # uncommitted bytes on graceful interruption only.
        if not path.with_suffix(".json").exists():
            temporary.unlink(missing_ok=True)


def recover_commit(path, receipt_path, temporary, identity, kind):
    if not receipt_path.exists():
        if path.exists():
            raise ValueError(f"Immutable encoder-cache file has no receipt: {path}")
        return None
    receipt = read_json(receipt_path)
    candidate = path if path.exists() else temporary
    require_file(candidate, receipt, identity=identity, kind=kind)
    if candidate != path:
        os.replace(candidate, path)
    return receipt


def build_task(root, task, data, policy, identity, check_stop):
    from stmem.data.cache import episode_key

    with locked(root / f"{task}.lock"):
        target = root / f"{task}.h5"
        receipt_path = root / f"{task}.complete.json"
        temporary = root / f"{task}.assembling.h5"
        existing = recover_commit(target, receipt_path, temporary, identity, "encoder_cache_task")
        if existing is not None:
            expected = {episode_key(*episode_ref(data, index)) for index in data.by_task[task]}
            if (
                existing.get("task") != task
                or existing.get("episodes") != len(expected)
                or set(existing.get("episode_sha256", {})) != expected
            ):
                raise ValueError("Task-cache receipt differs from the configured episode catalog")
            progress("cache_task_reused", task=task, episodes=existing["episodes"], path=str(target))
            return
        work = root / f".{task}.episodes"
        work.mkdir(exist_ok=True)
        episodes = []
        for index in data.by_task[task]:
            check_stop()
            actual_task, episode_id = episode_ref(data, index)
            if actual_task != task:
                raise ValueError("Training catalog task identity changed")
            key = episode_key(task, episode_id)
            path = work / f"{key}.h5"
            record = recover_commit(
                path,
                path.with_suffix(".json"),
                path.with_suffix(".partial.h5"),
                identity,
                "encoder_cache_episode",
            )
            started = time.perf_counter()
            reused = record is not None
            if not reused:
                episode = data.episode(index)
                record = write_episode(path, episode, policy, data, identity, check_stop)
                del episode
            if record["task"] != task or record["episode_id"] != episode_id:
                raise ValueError("Episode-cache receipt has a different task/episode")
            episodes.append((key, path, record))
            progress(
                "cache_episode_complete",
                task=task,
                episode_id=episode_id,
                episodes_done=len(episodes),
                episodes_total=len(data.by_task[task]),
                reused=reused,
                seconds=time.perf_counter() - started,
            )
        require_source(identity)
        # Rebuilding a killed assembly costs an HDF5 copy, not GPU encoding.
        with h5py.File(temporary, "w") as out:
            out.attrs["identity_sha256"] = digest(identity)
            out.attrs["task"] = task
            for key, path, record in episodes:
                check_stop()
                require_file(path, record, identity=identity, kind="encoder_cache_episode")
                with h5py.File(path, "r") as source:
                    if set(source) != {key} or source.attrs.get("identity_sha256") != digest(identity):
                        raise ValueError("Episode-cache HDF5 identity differs from its receipt")
                    source.copy(key, out)
            out.flush()
        require_source(identity)
        check_stop()
        receipt = {
            "kind": "encoder_cache_task",
            "status": "complete",
            "task": task,
            "identity_sha256": digest(identity),
            "sha256": file_sha256(temporary),
            "bytes": temporary.stat().st_size,
            "episodes": len(episodes),
            "episode_sha256": {key: record["sha256"] for key, _, record in episodes},
        }
        atomic_json(receipt, receipt_path)
        os.replace(temporary, target)
        # Every deleted intermediate is named by this task's verified receipt.
        for _, path, _ in episodes:
            path.unlink()
            path.with_suffix(".json").unlink()
        work.rmdir()
        progress(
            "cache_task_complete",
            task=task,
            episodes=len(episodes),
            path=str(target),
            bytes=receipt["bytes"],
            sha256=receipt["sha256"],
        )


def finalize_cache(root, cfg, checkpoint, identity, tasks, seconds):
    from stmem.training.runtime import environment_manifest

    with locked(root / ".manifest.lock", wait=True):
        require_source(identity)
        missing = [
            task
            for task in tasks
            if not (root / f"{task}.h5").exists() or not (root / f"{task}.complete.json").exists()
        ]
        if missing:
            progress(
                "cache_tasks_pending", completed=len(tasks) - len(missing), total=len(tasks), missing=missing
            )
            return False
        shards = {}
        for task in tasks:
            path = root / f"{task}.h5"
            record = read_json(root / f"{task}.complete.json")
            require_file(path, record, identity=identity, kind="encoder_cache_task")
            if record.get("task") != task:
                raise ValueError("Task-cache receipt name mismatch")
            shards[path.name] = {
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "episodes": record["episodes"],
            }
        if file_sha256(checkpoint) != identity["teacher_sha256"]:
            raise ValueError("Selected teacher changed while building its cache")
        require_source(identity)
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if (
                previous["identity"] != identity
                or previous["shards"] != shards
                or previous["status"] != "complete"
            ):
                raise ValueError("Complete encoder-cache manifest changed")
        else:
            atomic_json(
                {
                    "identity": identity,
                    "status": "complete",
                    "tasks": tasks,
                    "environment": environment_manifest(),
                    "source": source_revision(),
                    "finalizing_job_seconds": seconds,
                    "shards": shards,
                },
                manifest_path,
            )
        atomic_json(
            {
                "status": "complete",
                "cache_path": str(root.resolve()),
                "cache_manifest_sha256": file_sha256(manifest_path),
                "teacher_sha256": identity["teacher_sha256"],
                "identity_sha256": digest(identity),
            },
            run_dir(cfg, "cache") / "complete.json",
        )
        progress("cache_complete", cache=str(root), tasks=len(tasks))
        return True


def cache_status(cfg, checkpoint):
    """Quick filesystem progress only; builders validate hashes before reuse."""
    location = run_dir(cfg, "cache") / "cache_location.json"
    if not location.exists():
        return {"status": "not_started", "location_record": str(location)}
    record = read_json(location)
    if record["teacher_sha256"] != file_sha256(checkpoint):
        raise ValueError("Recorded cache belongs to a different selected teacher")
    root = Path(record["cache_path"])
    completed = [
        task
        for task in record["tasks"]
        if (root / f"{task}.h5").exists() and (root / f"{task}.complete.json").exists()
    ]
    return {
        "status": "complete" if (root / "manifest.json").exists() else "incomplete",
        "cache_path": str(root),
        "completed_tasks": completed,
        "missing_tasks": [task for task in record["tasks"] if task not in completed],
        "note": "Recorded filesystem progress; content hashes are checked by cache construction/reuse.",
    }


@torch.no_grad()
def build_cache(cfg, checkpoint, *, task=None, task_index=None, device=None, threads=None, stop_file=None):
    from stmem.data.cache import cache_identity
    from stmem.models.policy import MemoryPolicy
    from stmem.training.data import TrainingData
    from stmem.training.runtime import load_checkpoint, validate_policy_config, validate_selection

    tasks = list(cfg["data"]["tasks"])
    if (
        not tasks
        or len(tasks) != len(set(tasks))
        or any(not re.fullmatch(r"[A-Za-z0-9_-]+", t) for t in tasks)
    ):
        raise ValueError("Cache tasks must be unique safe task names")
    if task is not None and task_index is not None:
        raise ValueError("Choose --task or --task-index")
    if task_index is not None:
        if isinstance(task_index, bool) or not 0 <= task_index < len(tasks):
            raise ValueError("Task index is outside the configured task list")
        task = tasks[task_index]
    if task is not None and task not in tasks:
        raise ValueError("Requested cache task is not in the configured task list")
    if threads is not None:
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        torch.set_num_threads(threads)
    source_before = cache_code_fingerprint()
    checkpoint_before = file_sha256(checkpoint)
    validate_selection(checkpoint, debug=cfg.get("debug", False) and cfg["data"].get("synthetic", False))
    state = load_checkpoint(checkpoint)
    if file_sha256(checkpoint) != checkpoint_before:
        raise ValueError("Selected teacher changed while loading its state")
    validate_policy_config(state["config"], cfg)
    if state.get("stage") != "teacher" or cfg["model"].get("history_mode", "full") != "full":
        raise ValueError("Encoder-cache construction requires a full-history teacher")
    stop = stop_file or os.environ.get("STMEM_STOP_FILE")

    def check_stop():
        if stop and Path(stop).exists():
            raise InterruptedError("Stop requested; committed episodes will be reused on resubmission")

    check_stop()
    progress("cache_startup", phase="catalog", task=task)
    data = TrainingData(cfg, progress=lambda fields: progress("cache_startup", **fields))
    identity = cache_identity(cfg, checkpoint, data.identity)
    if identity["teacher_sha256"] != checkpoint_before or identity["code"] != source_before:
        raise ValueError("Selected teacher or source changed during cache initialization")
    root = Path(cfg["paths"]["cache_root"]) / digest(identity)[:24]
    root.mkdir(parents=True, exist_ok=True)
    location = run_dir(cfg, "cache") / "cache_location.json"
    location.parent.mkdir(parents=True, exist_ok=True)
    with locked(location.with_suffix(".lock"), wait=True):
        record = {
            "cache_path": str(root.resolve()),
            "identity_sha256": digest(identity),
            "teacher_sha256": identity["teacher_sha256"],
            "tasks": tasks,
        }
        if location.exists() and read_json(location) != record:
            raise ValueError("Run already records a different encoder cache; use a new run root")
        atomic_json(record, location)
    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate from the supplied configuration so rebased descriptor paths are
    # respected; strict compatibility above protects all scientific settings.
    policy = MemoryPolicy(cfg)
    policy.load_state_dict(state["model"], strict=True)
    del state
    policy.to(chosen_device).float().eval().requires_grad_(False)
    progress("cache_startup", phase="encoding", task=task, cache=str(root), device=str(chosen_device))
    start = time.perf_counter()
    for name in [task] if task else tasks:
        build_task(root, name, data, policy, identity, check_stop)
    finalize_cache(root, cfg, checkpoint, identity, tasks, time.perf_counter() - start)
    return root
