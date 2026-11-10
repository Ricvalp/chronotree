"""CPU-only synthetic fixtures for safe diagnostic recovery, not policy evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal
from types import SimpleNamespace

import pytest

from stmem.inspection.history_resume import (
    case_path, run_lock, seal_plan, seal_result, stop_workers,
    validate_existing, validate_plan, validate_result,
)
from stmem.training.checkpoints import atomic_json, file_sha256


def fixture(tmp_path):
    checkpoint, clip = tmp_path / "checkpoint.pt", tmp_path / "clip.pt"
    checkpoint.write_bytes(b"Synthetic checkpoint identity fixture")
    clip.write_bytes(b"Synthetic feature identity fixture")
    output = tmp_path / "diagnostic"
    output.mkdir()
    case = {"task": "SwingXtimes", "episode_idx": 0, "arm": "full"}
    cfg = {"model": {"visual_features": {"checkpoint": str(clip)}},
           "evaluation": {"seed": 17, "max_steps": 4}, "data": {"tasks": ["SwingXtimes"]}}
    manifest = {
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint), "step": 120000},
        "source": {"source_sha256": "source-A", "git": "revision-A", "dirty": True},
        "workers": 2, "planned_cases": [case], "planned_scenes": [["SwingXtimes", 0]],
        "arms": {"full": {"method": "FULL", "intervention": "none"}},
        "execution_override": {"execute_actions": 10}, "probe_decisions": [0, 2, 5, 10, 20],
    }
    atomic_json(manifest, output / "manifest.json")
    atomic_json(cfg, output / "config.json")
    seal_plan(output, manifest, cfg)
    return output, manifest, cfg, case


def publish(output, manifest, case, *, status="success"):
    video = output / "videos/full/SwingXtimes_000.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"Synthetic video bytes; decoding is tested separately")
    row = {**case, "checkpoint_sha256": manifest["checkpoint"]["sha256"],
           "split": "val", "method": "FULL", "noise_seed": 17,
           "execution_override": manifest["execution_override"], "status": status,
           "primitive_commands": 5, "actions": [[0] * 8] * 5, "error": None,
           "terminal_status": status, "video_path": str(video)}
    path = case_path(output, case)
    atomic_json(seal_result(row), path)
    return path


def test_worker_count_change_does_not_change_saved_inputs(tmp_path):
    output, manifest, cfg, case = fixture(tmp_path)
    path = publish(output, manifest, case)
    hashes = {p: file_sha256(p) for p in (path, output / "manifest.json", output / "config.json")}
    requested = deepcopy(manifest)
    requested["workers"] = 4
    requested["source"].update(git="new-docs-commit", dirty=False)
    saved, saved_cfg = validate_plan(output, requested, cfg)
    assert saved["workers"] == 2 and saved_cfg == cfg
    assert len(validate_existing(output, saved, saved_cfg)) == 1
    assert all(file_sha256(p) == value for p, value in hashes.items())


@pytest.mark.parametrize("mutation", ["source", "execute", "probe", "configuration", "checkpoint", "clip", "manifest"])
def test_changed_science_or_assets_rejected_before_resume(tmp_path, mutation):
    output, manifest, cfg, _ = fixture(tmp_path)
    requested = deepcopy(manifest)
    requested_cfg = deepcopy(cfg)
    if mutation == "source":
        requested["source"]["source_sha256"] = "source-B"
    elif mutation == "execute":
        requested["execution_override"]["execute_actions"] = 8
    elif mutation == "probe":
        requested["probe_decisions"].append(30)
    elif mutation == "configuration":
        requested_cfg["evaluation"]["seed"] = 18
    elif mutation == "checkpoint":
        Path(manifest["checkpoint"]["path"]).write_bytes(b"Changed checkpoint")
    elif mutation == "clip":
        Path(cfg["model"]["visual_features"]["checkpoint"]).write_bytes(b"Changed CLIP")
    else:
        altered = deepcopy(manifest)
        altered["workers"] = 4
        atomic_json(altered, output / "manifest.json")
    with pytest.raises(ValueError):
        validate_plan(output, requested, requested_cfg)


@pytest.mark.parametrize("status", ["success", "fail", "timeout"])
def test_success_failure_and_benchmark_timeout_are_preserved(tmp_path, status):
    output, manifest, cfg, case = fixture(tmp_path)
    path = publish(output, manifest, case, status=status)
    before = path.read_bytes()
    assert validate_result(path, case, manifest, cfg)["status"] == status
    assert path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["error", "walltime", "payload", "video", "seed", "unexpected", "truncated"])
def test_error_or_corrupt_results_are_never_silently_retried(tmp_path, mutation):
    output, manifest, cfg, case = fixture(tmp_path)
    path = publish(output, manifest, case)
    row = json.loads(path.read_text())
    if mutation == "error":
        row.update(status="infrastructure_error", error="Synthetic error")
    elif mutation == "walltime":
        row.update(status="timeout", terminal_status=None)
    elif mutation == "payload":
        row["actions"][0][0] = 123
    elif mutation == "video":
        Path(row["video_path"]).write_bytes(b"Truncated media")
    elif mutation == "seed":
        row["noise_seed"] += 1
    elif mutation == "unexpected":
        atomic_json(row, path.with_name("unexpected_000.json"))
    else:
        path.write_text('{"partial":')
    if mutation not in {"video", "unexpected", "truncated"}:
        atomic_json(row, path)
    before = path.read_bytes()
    with pytest.raises((ValueError, RuntimeError)):
        validate_existing(output, manifest, cfg)
    assert path.read_bytes() == before


def test_lock_survives_coordinator_fd_close_while_worker_holds_it(tmp_path):
    output = tmp_path / "locked"
    child = None
    try:
        with run_lock(output) as fd:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                     pass_fds=(fd,), start_new_session=True)
            with pytest.raises(RuntimeError, match="active"):
                with run_lock(output):
                    pass
        # The parent closed its descriptor, but the worker still owns the lock.
        with pytest.raises(RuntimeError, match="active"):
            with run_lock(output):
                pass
    finally:
        if child is not None:
            stop_workers([child])
    assert child.poll() is not None
    with run_lock(output):
        pass


def test_complete_resume_does_not_launch_workers_and_retains_wandb_identity(tmp_path, monkeypatch):
    from stmem.inspection import history_use
    import stmem.inspection.history_use_report as reports

    output, manifest, cfg, case = fixture(tmp_path)
    path = publish(output, manifest, case)
    before = path.read_bytes()
    seen = []

    class Tracker:
        run = None

        def __init__(self, *args, **kwargs):
            seen.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def log(self, *args, **kwargs):
            pass

        def summary(self, *args, **kwargs):
            pass

    def forbidden(*args, **kwargs):
        pytest.fail("Complete resume must not spawn any workers")

    monkeypatch.setattr(history_use.subprocess, "Popen", forbidden)
    monkeypatch.setattr(reports, "write_report", lambda directory: directory / "index.html")
    args = SimpleNamespace(resume=True, workers=4, recent_swing=False, permuted_swing=False)
    with run_lock(output) as fd:
        history_use.run_coordinator(output, args, manifest, cfg, fd, Tracker)
    assert seen[0]["resume"] is True
    assert path.read_bytes() == before
    state = json.loads((output / "run_status.json").read_text())
    assert state["status"] == "complete" and state["recorded"] == 1
    assert state["worker_exit_codes"] == []


def test_completed_worker_slice_returns_before_cuda_setup(tmp_path, monkeypatch):
    import torch
    import stmem.config
    from stmem.inspection.history_use import run_worker

    output, manifest, cfg, case = fixture(tmp_path)
    path = publish(output, manifest, case)
    before = path.read_bytes()
    monkeypatch.setattr(stmem.config, "source_revision", lambda: manifest["source"])
    def forbidden(*args, **kwargs):
        pytest.fail("A completed worker slice must not initialize CUDA")
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    run_worker(output, 0, worker_count=4)
    run_worker(output, 3, worker_count=4)
    assert path.read_bytes() == before


def test_cleanup_kills_descendant_even_after_worker_leader_exits():
    code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True)"
    leader = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                              text=True, start_new_session=True)
    child_pid = int(leader.stdout.readline())
    leader.wait(timeout=5)
    try:
        os.kill(child_pid, 0)
        stop_workers([leader])
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            stat = Path(f"/proc/{child_pid}/stat")
            if not stat.exists() or stat.read_text().split()[2] == "Z":
                break
            time.sleep(.02)
        else:
            pytest.fail("Renderer descendant survived cleanup")
    finally:
        try:
            os.kill(child_pid, 9)
        except ProcessLookupError:
            pass
        leader.stdout.close()


def test_followup_reference_copies_are_hashed_and_never_rewritten(tmp_path):
    from stmem.inspection.history_use import copy_references

    reference, manifest, cfg, case = fixture(tmp_path)
    path = publish(reference, manifest, case)
    original = json.loads(path.read_text())
    manifest["reference_results"] = {str(path): file_sha256(path)}
    manifest["reference_videos"] = {original["video_path"]: file_sha256(original["video_path"])}
    output = tmp_path / "followup"
    (output / "videos/full").mkdir(parents=True)
    copy_references(output, [(path, deepcopy(original))], manifest, cfg)
    copied = case_path(output, case)
    row = validate_result(copied, case, manifest, cfg, video_root=output)
    assert row["reused_reference_result"] == str(path)
    assert Path(row["video_path"]).read_bytes() == Path(original["video_path"]).read_bytes()
    # Resume preserves both bytes and mtime of the published row/media.
    before = {p: (file_sha256(p), p.stat().st_mtime_ns) for p in (copied, Path(row["video_path"]))}
    copy_references(output, [(path, deepcopy(original))], manifest, cfg)
    assert all((file_sha256(p), p.stat().st_mtime_ns) == values for p, values in before.items())
    # A reference changed between planning and copying must not be adopted.
    other = tmp_path / "other_followup"
    (other / "videos/full").mkdir(parents=True)
    Path(original["video_path"]).write_bytes(b"Changed reference media")
    with pytest.raises(ValueError, match="Reference video changed"):
        copy_references(other, [(path, original)], manifest, cfg)
    assert not case_path(other, case).exists()


def test_cleanup_has_one_shared_grace_period_for_all_worker_groups(monkeypatch):
    import stmem.inspection.history_resume as module

    now = [0.0]
    signals, waits = [], []
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    class Worker:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

        def wait(self, timeout=None):
            waits.append((self.pid, timeout))
            if timeout is not None:
                now[0] += timeout
                raise subprocess.TimeoutExpired("synthetic worker ignores TERM", timeout)
            assert (self.pid, module.signal.SIGKILL) in signals
            return -9

    processes = [Worker(pid) for pid in range(4)]
    stop_workers(processes)
    assert now[0] == 10  # Not ten seconds per worker.
    assert signals[:4] == [(pid, module.signal.SIGTERM) for pid in range(4)]
    assert signals[4:] == [(pid, module.signal.SIGKILL) for pid in range(4)]
    assert [wait[1] for wait in waits[:4]] == [10, 0, 0, 0]
    assert waits[4:] == [(pid, None) for pid in range(4)]


def test_interruption_stops_workers_before_potentially_blocking_wandb_finish(tmp_path, monkeypatch):
    from stmem.inspection import history_use, history_resume

    output, manifest, cfg, _ = fixture(tmp_path)
    workers, events = [], []

    class Tracker:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            assert len(workers) == 2 and all(worker.returncode == -15 for worker in workers)
            events.append("wandb finish")

    def spawn(*args, **kwargs):
        process = SimpleNamespace(returncode=None)
        workers.append(process)
        return process

    def collect(*args):
        if not workers:
            return [], {}
        raise InterruptedError("Synthetic coordinator signal")

    def stop(processes):
        assert processes == workers
        for process in processes:
            process.returncode = -15
        events.append("workers stopped")

    monkeypatch.setattr(history_use.subprocess, "Popen", spawn)
    monkeypatch.setattr(history_use, "collect", collect)
    monkeypatch.setattr(history_resume, "stop_workers", stop)
    args = SimpleNamespace(resume=True, workers=2, recent_swing=False, permuted_swing=False)
    with run_lock(output) as fd, pytest.raises(InterruptedError):
        history_use.run_coordinator(output, args, manifest, cfg, fd, Tracker)
    assert events == ["workers stopped", "wandb finish"]
    assert json.loads((output / "run_status.json").read_text())["status"] == "interrupted"


def test_repeated_signals_cannot_interrupt_cleanup_and_handlers_restore():
    from stmem.inspection.history_resume import interrupt_signals

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with interrupt_signals():
        handler = signal.getsignal(signal.SIGTERM)
        with pytest.raises(InterruptedError):
            handler(signal.SIGTERM, None)
        assert all(signal.getsignal(sig) == signal.SIG_IGN for sig in previous)
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())
