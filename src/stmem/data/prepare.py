"""Prepare explicitly limited episodes from whole task HDF5 archives."""

from __future__ import annotations
import argparse
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import re
import signal
import time
import h5py
import numpy as np
from .download import TASKS, REVISION, sha256
from .pointcloud import (
    DEFAULT_CROP,
    CALIBRATION_VERSION,
    POINT_SAMPLING_VERSION,
    rgbd_to_points,
    proprioception,
    point_sampler_settings,
    resolve_point_sampling,
)
from .center_sampling import (
    CENTER_SAMPLING_VERSION,
    center_sampler_settings,
    resolve_center_sampling,
    center_counts as configured_center_counts,
    prepare_center_indices,
)
from .preparation_state import preparation_lock, verify_prepared
from stmem.training.checkpoints import atomic_json

FORMAT_VERSION = "canonical_balanced_v2"
PREPARED_SUBDIR = "prepared_balanced_v2"


def _print_progress(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def numeric_keys(group, prefix):
    keys = [k for k in group if re.fullmatch(re.escape(prefix) + r"\d+", k)]
    return sorted(keys, key=lambda k: int(k[len(prefix) :]))


def text_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_goal(setup):
    raw = np.asarray(setup["task_goal"][()]).reshape(-1)
    return text_value(raw[0]) if len(raw) else ""


def raw_episode_metadata(group):
    steps = numeric_keys(group, "timestep_")
    demos = [bool(np.asarray(group[k]["info"]["is_video_demo"][()]).reshape(-1)[0]) for k in steps]
    if any(demos[i] and not demos[i - 1] for i in range(1, len(demos))):
        raise ValueError("conditioning resumes after execution; unsupported source boundary")
    completed = [bool(np.asarray(group[k]["info"].get("is_completed", False)).reshape(-1)[0]) for k in steps]
    return steps, demos, completed


def prepare_task(
    raw_path: Path,
    output: Path,
    task: str,
    episodes: int | None,
    *,
    points=2048,
    crop=DEFAULT_CROP,
    voxel_size=0.005,
    max_depth_m=4.0,
    point_sampling=None,
    center_sampling=None,
    center_counts=(32,),
    dataset_revision=REVISION,
    expected_raw_sha256=None,
    skip_existing=False,
    progress=None,
):
    """Prepare one independently opened task file; never change alignment rules.

    CLI parallelism lives outside this function. A task has exactly one writer,
    and its previous completed file survives any failure before atomic replace.
    """
    raw_path, output = Path(raw_path), Path(output)
    point_sampling = resolve_point_sampling(point_sampling)
    center_sampling = resolve_center_sampling(center_sampling)
    if not center_counts or any(
        isinstance(c, bool) or not isinstance(c, int) or c < 1 for c in center_counts
    ):
        raise ValueError("center_counts must contain positive integer quotas")
    center_counts = sorted(set(center_counts))
    started = time.monotonic()
    progress = progress or _print_progress

    def emit(event, **values):
        progress({"event": event, "task": task, "pid": os.getpid(), **values})

    raw_bytes = raw_path.stat().st_size
    emit("task_started", phase="hashing", raw_bytes=raw_bytes)
    last_hash_progress = started

    def hash_progress(processed):
        nonlocal last_hash_progress
        now = time.monotonic()
        if now - last_hash_progress >= 10 or processed == raw_bytes:
            emit("hash_progress", bytes_done=processed, bytes_total=raw_bytes, elapsed_seconds=now - started)
            last_hash_progress = now

    raw_hash = sha256(raw_path, progress=hash_progress)
    if expected_raw_sha256 is not None and raw_hash != expected_raw_sha256:
        raise ValueError(f"Extracted HDF5 failed its download-manifest SHA256: {raw_path}")
    emit(
        "raw_integrity",
        sha256=raw_hash,
        download_manifest_verified=expected_raw_sha256 is not None,
        message=(
            "Extracted file matches the download-manifest SHA256"
            if expected_raw_sha256 is not None
            else "Legacy manifest has no extracted-file SHA256; current raw digest recorded for provenance"
        ),
    )
    report = {
        "format": FORMAT_VERSION,
        "task": task,
        "dataset_revision": dataset_revision,
        "raw_sha256": raw_hash,
        "raw_path": str(raw_path.resolve()),
        "points": points,
        "crop": crop,
        "voxel_size": voxel_size,
        "max_depth_m": max_depth_m,
        "calibration": CALIBRATION_VERSION,
        "point_sampling_version": POINT_SAMPLING_VERSION,
        "point_sampling": point_sampling,
        "center_sampling_version": CENTER_SAMPLING_VERSION,
        "center_sampling": center_sampling,
        "center_counts": center_counts,
        "clock": "source_local_recorded_frame",
        "physical_continuity_verified": False,
        "nominal_control_hz": 20,
        "alignment": "unresolved",
        "episodes": [],
    }
    with preparation_lock(output), h5py.File(raw_path, "r") as src:
        if output.exists():
            try:
                with h5py.File(output, "r") as existing_file:
                    previous_format = json.loads(existing_file.attrs["manifest"]).get("format")
            except (OSError, KeyError, ValueError, AttributeError):
                previous_format = None
            if previous_format is not None and previous_format != FORMAT_VERSION:
                raise ValueError(
                    f"Refusing to overwrite prepared format {previous_format} at {output}. "
                    f"Use data.prepared_subdir: {PREPARED_SUBDIR} or a new --prepared-subdir."
                )
        keys = numeric_keys(src, "episode_")
        if episodes is not None:
            if len(keys) < episodes:
                raise ValueError(f"{task}: requested {episodes} episodes, found only {len(keys)}")
            keys = keys[:episodes]
        if not keys:
            raise ValueError(f"{task}: no raw episodes")
        layout = {key: len(numeric_keys(src[key], "timestep_")) for key in keys}
        if not all(layout.values()):
            raise ValueError(f"{task}: empty raw episode")
        if skip_existing:
            existing = verify_prepared(output, report, layout)
            if existing is not None:
                sidecar = output.with_suffix(".manifest.json")
                current_output = {
                    "output": str(output.resolve()),
                    "prepared_bytes": output.stat().st_size,
                }
                expected_sidecar = {**existing, **current_output}
                try:
                    saved = json.loads(sidecar.read_text())
                    matches = all(saved.get(key) == value for key, value in expected_sidecar.items())
                except (OSError, ValueError, AttributeError):
                    matches = False
                if not matches:
                    atomic_json(expected_sidecar, sidecar)
                return {
                    **existing,
                    "skipped": True,
                    "verification_seconds": time.monotonic() - started,
                    **current_output,
                }
        emit(
            "task_preparing",
            episodes_total=len(keys),
            frames_total=sum(layout.values()),
            rebuilding=output.exists(),
        )
        return _write_prepared(src, output, keys, report, layout, started, emit)


def _write_prepared(src, output, keys, report, layout, started, emit):
    task, points, crop, voxel_size, max_depth_m = (
        report[key] for key in ("task", "points", "crop", "voxel_size", "max_depth_m")
    )
    temporary = output.with_suffix(".partial.h5")
    processed_frames = 0
    total_frames = sum(layout.values())
    processing_started = last_progress = time.monotonic()

    def frame_progress(event, **values):
        elapsed = time.monotonic() - processing_started
        emit(
            event,
            frames_done=processed_frames,
            frames_total=total_frames,
            frames_per_second=processed_frames / max(elapsed, 1e-9),
            eta_seconds=elapsed * (total_frames - processed_frames) / max(processed_frames, 1),
            elapsed_seconds=time.monotonic() - started,
            **values,
        )

    with h5py.File(temporary, "w") as dst:
        for key in keys:
            raw = src[key]
            setup = raw["setup"]
            steps, demos, completed = raw_episode_metadata(raw)
            calib = {f"{c}_camera_intrinsic": setup[f"{c}_camera_intrinsic"][()] for c in ("front", "wrist")}
            group = dst.create_group(key)
            group.attrs.update(goal=read_goal(setup), task=task, episode_id=key)
            n = len(steps)
            for name, shape, dtype in [
                ("points", (n, points, 6), "f4"),
                ("valid", (n, points), "?"),
                ("proprio", (n, 15), "f4"),
                ("actions", (n, 8), "f4"),
                ("action_valid", (n, 8), "?"),
            ]:
                group.create_dataset(
                    name,
                    shape=shape,
                    dtype=dtype,
                    compression="gzip",
                    compression_opts=1,
                    chunks=(1, *shape[1:]),
                )
            group.create_dataset("source", data=np.asarray(demos, dtype=bool))
            group.create_dataset("completed", data=np.asarray(completed, dtype=bool))
            group.create_dataset("raw_index", data=[int(k[9:]) for k in steps])
            centers = group.create_group("center_indices")
            for count in report["center_counts"]:
                centers.create_dataset(
                    str(count),
                    shape=(n, count),
                    dtype="i8",
                    compression="gzip",
                    compression_opts=1,
                    chunks=(1, count),
                )
            seen_complete = False
            source_index = {"conditioning": 0, "execution": 0}
            source_indices = []
            for i, key_step in enumerate(steps):
                step = raw[key_step]
                obs = {
                    k: step["obs"][k][()]
                    for k in (
                        "front_rgb",
                        "wrist_rgb",
                        "front_depth",
                        "wrist_depth",
                        "front_camera_extrinsic",
                        "wrist_camera_extrinsic",
                        "joint_state",
                        "eef_state",
                        "gripper_state",
                    )
                }
                cloud, valid = rgbd_to_points(
                    obs,
                    calib,
                    count=points,
                    crop=crop,
                    voxel_size=voxel_size,
                    max_depth_m=max_depth_m,
                    **report["point_sampling"],
                )
                group["points"][i] = cloud.numpy()
                group["valid"][i] = valid.numpy()
                selected = prepare_center_indices(
                    cloud, valid, report["center_counts"], report["center_sampling"]
                )
                for count, indices in selected.items():
                    centers[str(count)][i] = indices.numpy()
                group["proprio"][i] = proprioception(obs).numpy()
                action = np.zeros(8, dtype=np.float32)
                mask = np.zeros(8, dtype=bool)
                try:
                    a = np.asarray(step["action"]["joint_action"][()], dtype=np.float32).reshape(-1)
                    if len(a) not in (7, 8):
                        raise ValueError("joint action requires 7 or 8 dimensions")
                    action[: len(a)] = a
                    mask[: len(a)] = np.isfinite(a)
                except (ValueError, KeyError, TypeError):
                    pass
                if demos[i] or seen_complete:
                    mask[:] = False
                if not demos[i] and (i == 0 or demos[i - 1]):
                    mask[:] = False  # reset's automatic initial command
                if (
                    i > 0
                    and not demos[i]
                    and not demos[i - 1]
                    and int(key_step[9:]) - int(steps[i - 1][9:]) != 1
                ):
                    mask[:] = False
                if task in ("PatternLock", "RouteStick"):
                    mask[7] = False
                    action[7] = 0
                # The first completed observation follows its final valid command.
                if not demos[i] and completed[i]:
                    seen_complete = True
                action[~np.isfinite(action)] = 0
                group["actions"][i] = action
                group["action_valid"][i] = mask
                source = "conditioning" if demos[i] else "execution"
                source_indices.append(source_index[source])
                source_index[source] += 1
                processed_frames += 1
                if time.monotonic() - last_progress >= 10:
                    frame_progress("frame_progress", episode_id=key, frame=i + 1, episode_frames=n)
                    last_progress = time.monotonic()
            group.create_dataset("source_index", data=source_indices)
            episode_report = {
                "episode_id": key,
                "conditioning_frames": sum(demos),
                "execution_frames": len(demos) - sum(demos),
                "first_completed_raw_index": next(
                    (int(steps[i][9:]) for i, c in enumerate(completed) if c), None
                ),
                "terminal_suffix_records": sum(1 for i, c in enumerate(completed) if c) - 1
                if any(completed)
                else 0,
                "seed": int(setup["seed"][()]),
                "difficulty": text_value(setup["difficulty"][()]),
                "raw_indices_contiguous": all(int(b[9:]) - int(a[9:]) == 1 for a, b in zip(steps, steps[1:])),
            }
            report["episodes"].append(episode_report)
            frame_progress(
                "episode_complete",
                episodes_done=len(report["episodes"]),
                episodes_total=len(keys),
                **episode_report,
            )
        dst.attrs["manifest"] = json.dumps(report)
    temporary.replace(output)
    report.update(
        elapsed_seconds=time.monotonic() - started,
        prepared_bytes=output.stat().st_size,
        output=str(output.resolve()),
    )
    atomic_json(report, output.with_suffix(".manifest.json"))
    return report


def _worker(job, threads, connection):
    """Fresh process and independent HDF5 handles, with parent-owned progress."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        result = prepare_task(**job, progress=connection.send)
        connection.send({"event": "_result", "report": result})
    except BaseException as exc:
        # Also detect abrupt death in the parent: OOM/SIGKILL cannot send this.
        connection.send({"event": "_error", "task": job["task"], "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        connection.close()


def _cpu_budget():
    limits = [len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1]
    if os.environ.get("SLURM_CPUS_PER_TASK"):
        value = int(os.environ["SLURM_CPUS_PER_TASK"])
        if value < 1:
            raise ValueError("SLURM_CPUS_PER_TASK must be positive")
        limits.append(value)
    return min(limits)


def run_preparation(jobs, *, workers=1, threads_per_worker=1):
    """Bound task concurrency and cancel all child writers on any failed task.

    Per-child pipes avoid a shared Queue becoming unreadable if a worker dies
    mid-message. No HDF5 handles or point-cloud tensors cross process boundaries.
    """
    if not jobs:
        return []
    active_count = min(workers, len(jobs))
    if workers < 1 or threads_per_worker < 1:
        raise ValueError("workers and threads_per_worker must be positive")
    if active_count * threads_per_worker > _cpu_budget():
        raise ValueError(
            f"{active_count} active workers x {threads_per_worker} threads exceeds "
            f"the {_cpu_budget()} available/allocated CPUs; reduce --workers or --threads-per-worker"
        )
    # Set before spawning: NumPy/BLAS/Torch may initialize at module import.
    environment = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {key: os.environ.get(key) for key in environment}
    context = mp.get_context("spawn")
    active, reports = {}, []
    remaining = iter(jobs)
    pending = True
    started = time.monotonic()
    cancellation = None
    old_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def interrupted(signum, _frame):
        # Raising inside Process.start() could leave a live child unregistered.
        # Defer cancellation until the parent can account for every writer.
        nonlocal cancellation
        cancellation = signum

    def check_interruption():
        if cancellation is not None:
            raise SystemExit(128 + cancellation)

    def consume(state):
        connection = state["connection"]
        if state["eof"] or not connection.poll():
            return
        try:
            event = connection.recv()
        except EOFError:
            state["eof"] = True
            return
        if event["event"] == "_error":
            raise RuntimeError(f"Preparation failed for {event['task']}: {event['error']}")
        if event["event"] == "_result":
            state["report"] = event["report"]
        else:
            _print_progress(event)

    try:
        for signum in old_handlers:
            signal.signal(signum, interrupted)
        for key in environment:
            os.environ[key] = str(threads_per_worker)
        _print_progress(
            {
                "event": "preparation_started",
                "tasks": len(jobs),
                "workers": active_count,
                "threads_per_worker": threads_per_worker,
                "cpu_budget": _cpu_budget(),
            }
        )
        while active or pending:
            check_interruption()
            while pending and len(active) < active_count:
                check_interruption()
                try:
                    job = next(remaining)
                except StopIteration:
                    pending = False
                    break
                receive, send = context.Pipe(duplex=False)
                process = context.Process(target=_worker, args=(job, threads_per_worker, send))
                try:
                    process.start()
                except BaseException:
                    receive.close()
                    process.close()
                    raise
                finally:
                    send.close()
                active[process.pid] = {
                    "process": process,
                    "connection": receive,
                    "task": job["task"],
                    "report": None,
                    "eof": False,
                }
                check_interruption()
            if not active:
                break
            wait(
                [item["connection"] for item in active.values() if not item["eof"]]
                + [item["process"].sentinel for item in active.values()],
                timeout=0.2,
            )
            check_interruption()
            for pid, state in list(active.items()):
                while not state["eof"] and state["connection"].poll():
                    consume(state)
                    check_interruption()
                process = state["process"]
                if process.exitcode is None:
                    continue
                process.join()
                # The final result can arrive between poll() and exitcode.
                # Once the child exits, drain its pipe before checking success.
                while not state["eof"] and state["connection"].poll():
                    consume(state)
                    check_interruption()
                if process.exitcode or state["report"] is None:
                    raise RuntimeError(
                        f"Preparation worker for {state['task']} exited with code {process.exitcode} "
                        "without completing its task"
                    )
                report = state["report"]
                reports.append(report)
                _print_progress(
                    {
                        "event": "task_skipped" if report.get("skipped") else "task_complete",
                        "task": state["task"],
                        "pid": pid,
                        "episodes": len(report["episodes"]),
                        "output": report["output"],
                        "prepared_bytes": report["prepared_bytes"],
                        "elapsed_seconds": report.get("verification_seconds", report.get("elapsed_seconds")),
                        "tasks_done": len(reports),
                        "tasks_total": len(jobs),
                    }
                )
                state["connection"].close()
                process.close()
                del active[pid]
        check_interruption()
        _print_progress(
            {
                "event": "preparation_complete",
                "tasks": len(reports),
                "skipped": sum(bool(report.get("skipped")) for report in reports),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        return reports
    finally:
        # Keep deferred signal handlers installed until every child has exited.
        try:
            for state in active.values():
                process = state["process"]
                if process.is_alive():
                    process.terminate()
            for state in active.values():
                process = state["process"]
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                state["connection"].close()
                process.close()
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", choices=TASKS)
    p.add_argument("--config")
    p.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Read only this many episodes per task (archive download is still whole-task)",
    )
    p.add_argument("--data-root", default=os.getenv("STMEM_DATA_ROOT", "data"))
    p.add_argument(
        "--prepared-subdir", help="Versioned output directory beneath data-root (overrides config)"
    )
    p.add_argument("--points", type=int)
    p.add_argument("--voxel-size", type=float)
    p.add_argument(
        "--workers", type=int, default=1, help="Concurrent task files (default: 1; at most selected tasks)"
    )
    p.add_argument(
        "--threads-per-worker", type=int, default=1, help="CPU threads per task worker (default: 1)"
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse completed files only after raw checksum and layout/config verification",
    )
    args = p.parse_args(argv)
    if args.episodes is not None and args.episodes < 1:
        p.error("--episodes must be positive")
    if args.workers < 1 or args.threads_per_worker < 1:
        p.error("--workers and --threads-per-worker must be positive")
    config = {}
    if args.config:
        from stmem.config import load_config, require_sampling_contract

        config = load_config(args.config)
        require_sampling_contract(config)
        args.data_root = config["paths"]["data_root"]
        args.tasks = args.tasks or config["data"]["tasks"]
        if args.episodes is None:
            args.episodes = config["data"]["episodes_per_task"]
    if not args.tasks:
        p.error("--tasks or --config is required")
    if len(set(args.tasks)) != len(args.tasks):
        p.error("Duplicate task names would create concurrent writers; each task must appear once")
    if any(task not in TASKS for task in args.tasks):
        p.error("Configuration contains unknown tasks")
    if args.episodes is not None and args.episodes < 1:
        p.error("--episodes/config episodes_per_task must be positive")
    geometry = config.get("geometry", {})
    points = args.points if args.points is not None else config.get("model", {}).get("points", 2048)
    voxel_size = args.voxel_size if args.voxel_size is not None else geometry.get("voxel_size", 0.005)
    root = Path(args.data_root)
    subdir = args.prepared_subdir or config.get("data", {}).get("prepared_subdir", PREPARED_SUBDIR)
    relative_output = Path(subdir)
    if relative_output.is_absolute() or ".." in relative_output.parts or relative_output == Path("."):
        p.error(
            "--prepared-subdir/data.prepared_subdir must be a nonempty relative directory below data-root"
        )
    point_sampling = point_sampler_settings(config)
    center_sampling = center_sampler_settings(config)
    center_counts = configured_center_counts(config)
    if points < 1:
        p.error("--points must be positive")
    jobs = []
    for task in args.tasks:
        candidates = list((root / "raw").rglob(f"record_dataset_{task}.h5"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one raw file for {task}, found {candidates}")
        manifest_path = root / "manifests" / f"{task}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing download integrity manifest {manifest_path}; use download --archive PATH to verify an existing archive"
            )
        download_manifest = json.loads(manifest_path.read_text())
        revision = download_manifest["revision"]
        expected_raw_sha256 = next(
            (
                member.get("sha256")
                for member in download_manifest.get("members", [])
                if Path(member["path"]).name == candidates[0].name
            ),
            None,
        )
        if expected_raw_sha256 is not None and (
            not isinstance(expected_raw_sha256, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_raw_sha256)
        ):
            raise ValueError(f"Invalid extracted-file SHA256 in {manifest_path}; reverify the task archive")
        jobs.append(
            dict(
                raw_path=candidates[0],
                output=root / relative_output / f"{task}.h5",
                task=task,
                episodes=args.episodes,
                points=points,
                voxel_size=voxel_size,
                crop=geometry.get("crop", DEFAULT_CROP),
                max_depth_m=geometry.get("max_depth", 4.0),
                point_sampling=point_sampling,
                center_sampling=center_sampling,
                center_counts=center_counts,
                dataset_revision=revision,
                expected_raw_sha256=expected_raw_sha256.lower() if expected_raw_sha256 is not None else None,
                skip_existing=args.skip_existing,
            )
        )
    # Preflight all paths/manifests before starting any task. This command does
    # not download missing data or submit any scheduler job.
    try:
        return run_preparation(jobs, workers=args.workers, threads_per_worker=args.threads_per_worker)
    except (KeyboardInterrupt, SystemExit):
        _print_progress(
            {"event": "preparation_interrupted", "message": "Workers stopped; completed task files retained"}
        )
        raise


if __name__ == "__main__":
    main()
