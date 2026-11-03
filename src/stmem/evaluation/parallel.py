"""One tracked validation run with independent simulator processes across GPUs."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from stmem.config import load_config, source_revision
from stmem.evaluation.execution import evaluation_execution_config
from stmem.evaluation.selection import validate_selection_protocol
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import environment_manifest, load_checkpoint
from stmem.tracking import RunTracker

REPO = Path(__file__).resolve().parents[3]
_spawning = False
_pending_signal = None


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


class EvaluationInterrupted(KeyboardInterrupt):
    def __init__(self, signum=None):
        self.signum = signum
        reason = signal.Signals(signum).name if signum is not None else "stop request"
        super().__init__(f"Parallel evaluation interrupted by {reason}")


@contextmanager
def termination_signals():
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1)
    old = {value: signal.getsignal(value) for value in signals}

    def stop(signum, _frame):
        global _pending_signal
        for value in signals:
            signal.signal(value, signal.SIG_IGN)
        if _spawning:
            _pending_signal = signum
            return
        raise EvaluationInterrupted(signum)

    try:
        for value in signals:
            signal.signal(value, stop)
        yield
    finally:
        for value, handler in old.items():
            signal.signal(value, handler)


@contextmanager
def registered_spawn():
    """Defer interruption until Popen's process group is registered for cleanup."""
    global _spawning, _pending_signal
    _spawning = True
    try:
        yield
    finally:
        _spawning = False
        if _pending_signal is not None:
            signum, _pending_signal = _pending_signal, None
            raise EvaluationInterrupted(signum)


@contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".parallel.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another parallel evaluation is writing {output}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def canonical_manifest(value):
    value = deepcopy(value)
    value["source"] = value["source"]["source_sha256"]
    # Physical allocation details and telemetry do not choose episode inputs.
    value.pop("environment", None)
    value["config"].pop("tracking", None)
    return value


def save_manifest(output, manifest, *, resume):
    path = output / "manifest.json"
    if path.exists():
        if not resume:
            raise FileExistsError(f"Evaluation output already exists: {output}; use --resume")
        previous = read_json(path)
        if canonical_manifest(previous) != canonical_manifest(manifest):
            raise ValueError("Parallel evaluation inputs/layout changed; choose a new output directory")
        # A killed initial writer may have committed the manifest just before
        # writing its derived worker config; reconstruct only a missing file.
        if not (output / "evaluation.yaml").exists():
            (output / "evaluation.yaml").write_text(yaml.safe_dump(previous["config"], sort_keys=False))
        return previous
    existing = [p for p in output.iterdir() if p.name != ".parallel.lock"]
    if existing:
        raise FileExistsError(f"Nonempty output lacks its parallel manifest: {output}")
    atomic_json(manifest, path)
    (output / "evaluation.yaml").write_text(yaml.safe_dump(manifest["config"], sort_keys=False))
    return manifest


def cpu_assignments(available, workers, threads):
    available = sorted(set(available))
    if workers < 1 or threads < 1 or len(available) < workers * threads:
        raise ValueError(
            f"Need at least {workers * threads} allocated CPU cores for {workers} workers "
            f"with {threads} threads each; only {len(available)} are visible"
        )
    size, extra = divmod(len(available), workers)
    assignments, start = [], 0
    for index in range(workers):
        stop = start + size + (index < extra)
        assignments.append(available[start:stop])
        start = stop
    return assignments


@dataclass
class Worker:
    index: int
    command: list[str]
    env: dict[str, str]
    directory: Path
    cpus: list[int]
    device: dict


def worker_plan(args, output, manifest, devices, *, environment=None, available_cpus=None):
    from stmem.evaluation.devices import worker_device_environment

    num_workers = manifest["num_shards"]
    available = os.sched_getaffinity(0) if available_cpus is None else available_cpus
    assignments = cpu_assignments(available, num_workers, args.threads_per_worker)
    environment = dict(os.environ if environment is None else environment)
    workers = []
    for index, cpus in enumerate(assignments):
        device = devices[index % len(devices)]
        env = dict(environment)
        for key in tuple(env):
            if key.startswith("STMEM_") or key in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"}:
                env.pop(key)
        env.update(
            PYTHONPATH=str(REPO / "src"), WANDB_MODE="disabled",
            OMP_NUM_THREADS=str(args.threads_per_worker), MKL_NUM_THREADS=str(args.threads_per_worker),
            OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", OMP_DYNAMIC="FALSE",
        )
        env.update(worker_device_environment(device))
        directory = output / "shards" / f"shard{index:03d}"
        forwarded = [
            "--config", str(output / "evaluation.yaml"), "--checkpoint", str(args.checkpoint.resolve()),
            "--method", args.method, "--split", args.split, "--output", str(directory),
            "--episode-count", str(args.episode_count), "--num-shards", str(num_workers),
            "--shard-index", str(index), "--video-cap", str(args.video_cap),
            "--no-tracking", "--render-device", "cuda:0",
        ]
        if args.execute_actions is not None:
            forwarded += ["--execute-actions", str(args.execute_actions)]
        if args.budget is not None:
            forwarded += ["--budget", str(args.budget)]
        if args.router_checkpoint is not None:
            forwarded += ["--router-checkpoint", str(args.router_checkpoint.resolve())]
        if getattr(args, "selection_family", None) is not None:
            forwarded += ["--selection-family", args.selection_family]
        if args.targeting_diagnostics:
            forwarded.append("--targeting-diagnostics")
        command = [sys.executable, "-u", "-m", "stmem.evaluation.parallel_worker",
                   "--cpu-affinity", ",".join(map(str, cpus)), "--", *forwarded]
        workers.append(Worker(index, command, env, directory, cpus, device))
    return workers


def completed_shard(directory):
    marker = directory / "complete.json"
    manifest = directory / "manifest.json"
    if not marker.exists():
        return False
    value = read_json(marker)
    if not manifest.exists() or value.get("manifest_sha256") != file_sha256(manifest):
        raise ValueError(f"Invalid completed shard identity: {directory}")
    declared = read_json(manifest).get("planned", [])
    return (value.get("status") == "complete" and isinstance(value.get("recorded"), int)
            and value["recorded"] == value.get("planned") == len(declared))


def terminate_workers(processes, *, grace_seconds=10):
    # Signal all groups before waiting, so cleanup time does not scale with W.
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_seconds
    for process in processes:
        try:
            process.wait(timeout=max(0, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        # Even an exited leader can have a renderer/encoder descendant alive.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        process.wait()


def supervise(workers, output, progress, *, resume=False, poll_seconds=15, stop_file=None):
    """Run a fixed shard plan; failures preserve outcomes and stop sibling work."""
    state = {"status": "running", "started_utc": utc_now(), "workers": {}}
    path = output / "parallel_status.json"
    logs = output / "workers"
    logs.mkdir(exist_ok=True)
    processes, handles = {}, []
    started = time.monotonic()
    try:
        progress()
        for worker in workers:
            if stop_file and Path(stop_file).exists():
                raise EvaluationInterrupted()
            entry = {"status": "starting", "cpu_affinity": worker.cpus,
                     "device": worker.device, "command": worker.command,
                     "stdout": str(logs / f"shard{worker.index:03d}.out"),
                     "stderr": str(logs / f"shard{worker.index:03d}.err")}
            state["workers"][str(worker.index)] = entry
            if resume and completed_shard(worker.directory):
                entry.update(status="reused", exit_code=0)
                continue
            stdout = Path(entry["stdout"]).open("a" if resume else "w")
            stderr = Path(entry["stderr"]).open("a" if resume else "w")
            handles.extend([stdout, stderr])
            # Delay parent interruption until the new process is registered;
            # otherwise a signal during Popen could orphan an untracked group.
            with registered_spawn():
                process = subprocess.Popen(worker.command, cwd=REPO, env=worker.env,
                                           stdout=stdout, stderr=stderr, start_new_session=True)
                processes[worker.index] = process
                entry.update(status="running", pid=process.pid)
            atomic_json(state, path)
        last_progress = time.monotonic()
        while True:
            if stop_file and Path(stop_file).exists():
                raise EvaluationInterrupted()
            running = False
            for index, process in processes.items():
                returncode = process.poll()
                entry = state["workers"][str(index)]
                if returncode is None:
                    running = True
                    continue
                entry.update(status="complete" if returncode == 0 else "failed", exit_code=returncode)
                if returncode:
                    raise RuntimeError(
                        f"Evaluation worker {index} exited {returncode}; see {entry['stderr']}"
                    )
            now = time.monotonic()
            if not running or now-last_progress >= poll_seconds:
                progress()
                atomic_json(state, path)
                last_progress = now
            if not running:
                break
            time.sleep(min(0.5, poll_seconds))
        missing = [worker.index for worker in workers if not completed_shard(worker.directory)]
        if missing:
            raise RuntimeError(f"Workers exited without valid completion markers: {missing}")
        # Successful leaders can leave encoder/renderer descendants behind.
        terminate_workers(list(processes.values()))
        progress()
        state.update(status="complete", finished_utc=utc_now(), elapsed_seconds=time.monotonic()-started)
        atomic_json(state, path)
        return state
    except BaseException as error:
        terminate_workers(list(processes.values()))
        for index, process in processes.items():
            entry = state["workers"][str(index)]
            if entry.get("status") == "running":
                entry.update(status="stopped", exit_code=process.returncode)
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        state.update(status=status, finished_utc=utc_now(), elapsed_seconds=time.monotonic()-started,
                     error=f"{type(error).__name__}: {error}")
        atomic_json(state, path)
        try:
            progress()
        except Exception as aggregation_error:
            state["aggregation_error"] = f"{type(aggregation_error).__name__}: {aggregation_error}"
            atomic_json(state, path)
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--router-checkpoint", type=Path)
    parser.add_argument("--selection-family", choices=("tree", "flat"),
                        help="Validation diagnostic with random compact-reader selection; no learned router")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=("FULL", "TREE", "FLAT", "UNIFORM", "PERCEIVER", "CURRENT"), default="FULL")
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--episode-count", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--video-cap", type=int, default=10)
    parser.add_argument("--execute-actions", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--targeting-diagnostics", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path, default=os.environ.get("STMEM_STOP_FILE"))
    args = parser.parse_args()
    if (not 1 <= args.episode_count <= 50 or args.video_cap < 0 or args.num_gpus < 1
            or args.workers_per_gpu < 1 or args.threads_per_worker < 1
            or not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0):
        parser.error("Episode count must be 1..50; video cap nonnegative; worker/GPU/thread/poll values positive")
    if args.budget is not None and args.budget < 1:
        parser.error("--budget must be positive")
    import torch
    from stmem.evaluation.devices import discover_devices
    from stmem.evaluation.aggregation import aggregate_shards
    from stmem.evaluation.workload import planned_episodes, video_episodes

    torch.set_num_threads(1)
    output = args.output.resolve()
    with output_lock(output), termination_signals():
        cfg = load_config(args.config)
        checkpoint_state = load_checkpoint(args.checkpoint)
        checkpoint_cfg = checkpoint_state["config"]
        validate_selection_protocol(
            args.method, args.selection_family, args.router_checkpoint,
            checkpoint_stage=checkpoint_state.get("stage"), split=args.split,
        )
        cfg, execution_override = evaluation_execution_config(
            checkpoint_cfg, cfg, execute_actions=args.execute_actions, split=args.split,
        )
        cfg["evaluation"].update(episode_count=args.episode_count, video_cap=args.video_cap, split=args.split)
        all_planned = planned_episodes(cfg)
        if not all_planned:
            raise ValueError("No planned validation episodes")
        devices = discover_devices(args.num_gpus)
        checkpoints = {str(args.checkpoint.resolve()): file_sha256(args.checkpoint)}
        if args.router_checkpoint:
            checkpoints[str(args.router_checkpoint.resolve())] = file_sha256(args.router_checkpoint)
        requested = args.num_gpus * args.workers_per_gpu
        manifest = {
            "schema_version": 1, "kind": "parallel_validation_evaluation", "config": cfg,
            "source": source_revision(), "environment": environment_manifest(), "checkpoints": checkpoints,
            "policy_checkpoint_sha256": checkpoints[str(args.checkpoint.resolve())],
            "method": args.method, "split": args.split, "all_planned": all_planned, "planned": all_planned,
            "video_episodes": video_episodes(cfg), "num_shards": min(requested, len(all_planned)),
            "budget": args.budget or cfg["model"]["budget"], "targeting_diagnostics": args.targeting_diagnostics,
            "selection_family": args.selection_family, "intervention": "none",
            "parallel": {"num_gpus": args.num_gpus, "workers_per_gpu": args.workers_per_gpu,
                         "threads_per_worker": args.threads_per_worker, "requested_workers": requested},
        }
        if execution_override is not None:
            manifest["execution_override"] = execution_override
        manifest = json.loads(json.dumps(manifest))
        manifest = save_manifest(output, manifest, resume=args.resume)
        # A stored config is an immutable input, not a file to update on resume.
        if yaml.safe_load((output / "evaluation.yaml").read_text()) != manifest["config"]:
            raise ValueError("Saved worker evaluation configuration changed")
        workers = worker_plan(args, output, manifest, devices)
        atomic_json({"observed_utc": utc_now(), "devices": devices,
                     "available_cpu_affinity": sorted(os.sched_getaffinity(0)),
                     "active_workers": len(workers), "requested_workers": requested}, output / "allocation.json")
        print(json.dumps({"event": "parallel_evaluation_started", "workers": len(workers),
                          "gpus": len(devices), "episodes": len(all_planned), "output": str(output)}), flush=True)
        selection_label = (f"-random-selection-step{checkpoint_state['step']}"
                           if args.selection_family is not None else "")
        with RunTracker(cfg, output, job_type="evaluation",
                        name=f"{args.method}{selection_label}-{args.split}-seed{cfg['seed']}-parallel",
                        metadata={"evaluation_manifest": manifest,
                                  "manifest_sha256": file_sha256(output / "manifest.json")},
                        resume=args.resume) as tracker:
            last_count = -1
            starting_count = None

            def progress():
                nonlocal last_count, starting_count
                records, summary = aggregate_shards(output, manifest)
                overall = summary["overall"]
                count = overall["recorded"]
                if starting_count is None:
                    starting_count = count
                if count != last_count:
                    tracker.log({
                        "evaluation/success_rate_so_far_percent": 100*overall["successes"]/count if count else None,
                        "evaluation/progress_percent": 100*count/overall["planned"],
                    }, step=count)
                    tracker.summary({"evaluation/status": summary["status"],
                                     "evaluation/progress": {**overall, "new_episodes": count-starting_count,
                                                             "reused_episodes": starting_count,
                                                             "scope": "complete parallel validation run"}})
                    last_count = count
                print(json.dumps({"event": "parallel_evaluation_progress", "recorded": count,
                                  "planned": overall["planned"], "successes": overall["successes"],
                                  "errors": overall["errors"]}), flush=True)
                return records, summary

            try:
                supervise(workers, output, progress, resume=args.resume,
                          poll_seconds=args.poll_seconds, stop_file=args.stop_file)
                records, summary = progress()
                if summary["status"] != "complete":
                    raise RuntimeError("Completed workers did not cover the full planned denominator")
                if source_revision()["source_sha256"] != manifest["source"]["source_sha256"]:
                    raise ValueError("Source changed during evaluation; choose a new output directory")
                if any(file_sha256(path) != sha for path, sha in checkpoints.items()):
                    raise ValueError("Checkpoint changed during evaluation; choose a new output directory")
                atomic_json({"status": "complete", "planned": len(all_planned), "recorded": len(records),
                             "manifest_sha256": file_sha256(output / "manifest.json"),
                             "num_shards": len(workers)}, output / "complete.json")
                tracker.summary({"evaluation/status": "complete", "evaluation/task_results": summary["tasks"],
                                 "evaluation/macro_success": summary["macro_success"]})
            except BaseException as error:
                status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
                status_path = output / "parallel_status.json"
                state = read_json(status_path) if status_path.exists() else {}
                state.update(status=status, finished_utc=utc_now(), error=f"{type(error).__name__}: {error}")
                atomic_json(state, status_path)
                (output / "complete.json").unlink(missing_ok=True)
                tracker.summary({"evaluation/status": status, "evaluation/error": f"{type(error).__name__}: {error}"})
                raise


if __name__ == "__main__":
    try:
        main()
    except EvaluationInterrupted as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(128+error.signum if error.signum is not None else 75)
