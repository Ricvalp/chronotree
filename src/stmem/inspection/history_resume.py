"""Durable identities and single-writer coordination for history diagnostics.

Resume fills missing planned records only. Published policy failures are data;
published error records require diagnosis and are never automatically retried.
The original manifest/config remain immutable when worker concurrency changes.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from stmem.config import digest
from stmem.training.checkpoints import atomic_json, file_sha256


SCHEMA = 1


def case_path(output, case):
    return Path(output) / "results" / case["arm"] / "episodes" / f'{case["task"]}_{case["episode_idx"]:03d}.json'


def protocol(manifest):
    """Concurrency and Git bookkeeping do not change the scientific protocol."""
    value = deepcopy(manifest)
    value.pop("workers", None)
    value["source"] = {"source_sha256": value["source"]["source_sha256"]}
    return value


def seal_plan(output, manifest, cfg):
    output = Path(output)
    identity = {
        "schema": SCHEMA,
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "config_sha256": file_sha256(output / "config.json"),
        "protocol_sha256": digest(protocol(manifest)),
        "checkpoint_sha256": file_sha256(manifest["checkpoint"]["path"]),
        "clip_sha256": file_sha256(cfg["model"]["visual_features"]["checkpoint"]),
    }
    if identity["checkpoint_sha256"] != manifest["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint changed while preparing the history diagnostic")
    atomic_json(identity, output / "identity.json")
    return identity


def validate_plan(output, requested_manifest, requested_cfg):
    """Verify stored inputs before any GPU initialization or file mutation."""
    output = Path(output)
    identity_path = output / "identity.json"
    if not identity_path.is_file():
        raise ValueError("This output lacks a resumable identity seal; use a new output directory")
    identity = json.loads(identity_path.read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    cfg = json.loads((output / "config.json").read_text())
    expected = {
        "schema": SCHEMA,
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "config_sha256": file_sha256(output / "config.json"),
        "protocol_sha256": digest(protocol(manifest)),
        "checkpoint_sha256": file_sha256(manifest["checkpoint"]["path"]),
        "clip_sha256": file_sha256(cfg["model"]["visual_features"]["checkpoint"]),
    }
    if identity != expected or identity["checkpoint_sha256"] != manifest["checkpoint"]["sha256"]:
        raise ValueError("Stored history diagnostic inputs changed after planning")
    if protocol(manifest) != protocol(requested_manifest) or cfg != requested_cfg:
        raise ValueError("Resume checkpoint, source, configuration or scientific protocol differs")
    return manifest, cfg


def result_seed(cfg, case):
    return cfg["evaluation"]["seed"] + cfg["data"]["tasks"].index(case["task"]) * 1000 + case["episode_idx"]


def seal_result(record):
    value = deepcopy(record)
    value.pop("resume_integrity", None)
    video = value.get("video_path")
    value["resume_integrity"] = {
        "payload_sha256": digest(value),
        "video_sha256": file_sha256(video) if video and Path(video).is_file() else None,
    }
    return value


def validate_result(path, case, manifest, cfg, *, require_seal=True, verify_video=True, video_root=None):
    """Return a valid durable trial, rejecting errors and wall-clock cutoffs."""
    path = Path(path)
    row = json.loads(path.read_text())
    if row.get("status") not in {"success", "fail", "timeout"} or any(
        row.get(key) for key in ("error", "close_error", "traceback")
    ):
        raise RuntimeError(f"Recorded error requires diagnosis, not automatic retry: {path}: {row.get('error')}")
    expected = {
        **case, "checkpoint_sha256": manifest["checkpoint"]["sha256"], "split": "val",
        "method": manifest["arms"][case["arm"]]["method"],
        "noise_seed": result_seed(cfg, case), "execution_override": manifest["execution_override"],
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Result identity/protocol mismatch: {path}")
    if row.get("primitive_commands") != len(row.get("actions", [])):
        raise ValueError(f"Result command accounting mismatch: {path}")
    if row["status"] == "timeout" and (
        row.get("terminal_status") != "timeout"
        or row["primitive_commands"] != cfg["evaluation"]["max_steps"] + 1
    ):
        raise RuntimeError(f"Recorded timeout is not the normal benchmark limit: {path}")
    integrity = row.get("resume_integrity")
    if require_seal:
        payload = {key: value for key, value in row.items() if key != "resume_integrity"}
        if not integrity or integrity.get("payload_sha256") != digest(payload):
            raise ValueError(f"Published result changed or lacks its integrity seal: {path}")
    if verify_video:
        video = Path(row.get("video_path") or "")
        if (not row.get("video_path") or not video.is_file() or not video.stat().st_size
                or (video_root is not None and not video.resolve().is_relative_to(Path(video_root).resolve()))):
            raise ValueError(f"Published result video is absent or outside its output: {path}")
        if require_seal and file_sha256(video) != integrity.get("video_sha256"):
            raise ValueError(f"Published result video changed: {path}")
    return row


def validate_existing(output, manifest, cfg, *, verify_video=True):
    output = Path(output)
    planned = {case_path(output, case): case for case in manifest["planned_cases"]}
    if len(planned) != len(manifest["planned_cases"]):
        raise ValueError("Duplicated planned history diagnostic case")
    observed = set((output / "results").glob("*/episodes/*.json"))
    if observed - planned.keys():
        raise ValueError("Unplanned history diagnostic result files exist")
    return [validate_result(path, planned[path], manifest, cfg, verify_video=verify_video,
                            video_root=output) for path in sorted(observed)]


@contextmanager
def run_lock(output):
    """Pass this fd to workers so parent death cannot release a live-run lock."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".history.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another history coordinator or worker is active in {output}") from exc
        yield stream.fileno()
        # Closing (rather than explicit LOCK_UN) preserves ownership if a child
        # survived an abrupt parent exit while retaining this open description.


def verify_worker_lock(output, fd):
    if fd is None or Path(os.readlink(f"/proc/self/fd/{fd}")).resolve() != (Path(output) / ".history.lock").resolve():
        raise RuntimeError("History workers must be launched by the locked coordinator")


@contextmanager
def interrupt_signals():
    signals = (signal.SIGINT, signal.SIGTERM)

    def interrupted(signum, _frame):
        # Repeated signals must not interrupt process-group cleanup.
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)
        raise InterruptedError(f"History diagnostic interrupted by signal {signum}")

    old = {sig: signal.signal(sig, interrupted) for sig in signals}
    try:
        yield
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def stop_workers(processes):
    """Stop whole worker sessions, including renderer/loader descendants."""
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        process.wait()


@contextmanager
def worker_lifetime(processes):
    """Use inside tracking context so worker cleanup precedes network teardown."""
    try:
        yield
    finally:
        stop_workers(processes)
