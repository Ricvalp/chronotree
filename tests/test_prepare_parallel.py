"""Small synthetic HDF5 diagnostics for isolated, deterministic task preparation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time

import h5py
import numpy as np
import pytest

from stmem.data.download import REVISION, sha256
from stmem.data.dataset import EpisodeDataset
from stmem.data import prepare
from stmem.data.prepare import prepare_task


TASKS = ("VideoUnmask", "PatternLock")
RAW_STEPS = (10, 2, 0, 8, 1, 7, 3, 6, 5)
pytestmark = pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity") or len(os.sched_getaffinity(0)) < 2,
    reason="parallel preparation integration requires Linux with at least two available CPUs",
)


def write_raw(root, task, *, episode_ids=(10, 2)):
    """No downloaded data: cameras and causal masks are explicit fixture values."""
    raw = root / "raw" / f"record_dataset_{task}.h5"
    raw.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(raw, "w", track_order=True) as target:
        for episode_id in episode_ids:
            episode = target.create_group(f"episode_{episode_id}", track_order=True)
            setup = episode.create_group("setup")
            setup["task_goal"] = np.asarray([b"synthetic preparation diagnostic"])
            setup["seed"] = 6000 + episode_id
            setup["difficulty"] = np.bytes_("fixture")
            camera = np.asarray([[12, 0, 5.5], [0, 12, 3.5], [0, 0, 1]], dtype=np.float32)
            for name in ("front", "wrist"):
                setup[f"{name}_camera_intrinsic"] = camera
            for index in RAW_STEPS:
                step = episode.create_group(f"timestep_{index}")
                obs = step.create_group("obs")
                colors = np.empty((8, 12, 3), dtype=np.uint8)
                colors[..., 0] = np.arange(12)[None] * 17
                colors[..., 1] = np.arange(8)[:, None] * 23
                colors[..., 2] = 30 + episode_id + index
                depth = np.full((8, 12, 1), 1000 + index, dtype=np.int16)
                depth[0, 0], depth[0, 1] = -1, 0
                for name in ("front", "wrist"):
                    obs[f"{name}_rgb"] = colors
                    obs[f"{name}_depth"] = depth
                    translation = np.array([0.02 if name == "wrist" else 0, 0, 0])
                    obs[f"{name}_camera_extrinsic"] = np.c_[np.eye(3), translation]
                obs["joint_state"] = np.asarray([0, 0, 0, -1.5, 0, 1.5, 0.7], dtype=np.float32)
                obs["eef_state"] = np.asarray([0.1, 0, 0.2, 0.1, 0.2, -0.1], dtype=np.float32)
                obs["gripper_state"] = np.asarray([0.04, 0.04], dtype=np.float32)
                action = step.create_group("action")
                command = np.linspace(0, 0.6, 7 if task == "PatternLock" else 8, dtype=np.float32)
                command += index / 100
                if index == 6:
                    command[2] = np.nan
                action["joint_action"] = command
                info = step.create_group("info")
                info["is_video_demo"] = index < 2
                info["is_completed"] = index >= 7
    write_manifest(root, task)
    return raw


def write_manifest(root, task):
    raw = root / "raw" / f"record_dataset_{task}.h5"
    manifest = root / "manifests" / f"{task}.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "task": task,
                "revision": REVISION,
                "members": [{"path": str(raw), "bytes": raw.stat().st_size, "sha256": sha256(raw)}],
            }
        )
    )


@pytest.fixture
def raw_root(tmp_path):
    root = tmp_path / "data"
    for task in TASKS:
        write_raw(root, task)
    return root


def cli_command(root, *extra, tasks=TASKS):
    return [
        sys.executable,
        "-m",
        "stmem.data.prepare",
        "--data-root",
        str(root),
        "--prepared-subdir",
        "prepared",
        "--tasks",
        *tasks,
        "--points",
        "16",
        "--threads-per-worker",
        "1",
        *map(str, extra),
    ]


def cli_environment():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
    env.pop("STMEM_DATA_ROOT", None)
    env.pop("SLURM_CPUS_PER_TASK", None)
    return env


def run_cli(root, *extra, tasks=TASKS, success=True):
    process = subprocess.Popen(
        cli_command(root, *extra, tasks=tasks),
        env=cli_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    if success:
        assert process.returncode == 0, stderr + stdout
    else:
        assert process.returncode != 0, stdout
    # Parallel progress must remain machine-readable, one complete JSON object per line.
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr), events


def initial_prepare(root, *, points=16, episodes=None):
    for task in TASKS:
        raw = root / "raw" / f"record_dataset_{task}.h5"
        prepare_task(
            raw,
            root / "prepared" / f"{task}.h5",
            task,
            episodes,
            points=points,
            expected_raw_sha256=sha256(raw),
        )


def output_snapshot(root, task):
    files = [root / "prepared" / f"{task}.h5", root / "prepared" / f"{task}.manifest.json"]
    return {
        path.name: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in files
    }


def scientific_manifest(manifest):
    result = deepcopy(manifest)
    for key in ("raw_path", "output", "elapsed_seconds", "prepared_bytes"):
        result.pop(key, None)
    return result


def test_parallel_cli_matches_serial_arrays_masks_and_manifest(raw_root, tmp_path):
    parallel = tmp_path / "parallel"
    shutil.copytree(raw_root, parallel)
    _, serial_events = run_cli(raw_root, "--workers", 1)
    _, parallel_events = run_cli(parallel, "--workers", 2)
    assert serial_events[0]["event"] == parallel_events[0]["event"] == "preparation_started"
    assert serial_events[0]["workers"] == 1 and parallel_events[0]["workers"] == 2
    assert {event["task"] for event in parallel_events if event["event"] == "task_complete"} == set(TASKS)
    assert len({event["pid"] for event in parallel_events if event["event"] == "task_started"}) == 2
    for task in TASKS:
        with (
            h5py.File(raw_root / "prepared" / f"{task}.h5", "r") as left,
            h5py.File(parallel / "prepared" / f"{task}.h5", "r") as right,
        ):
            assert set(left) == set(right) == {"episode_2", "episode_10"}
            lm, rm = json.loads(left.attrs["manifest"]), json.loads(right.attrs["manifest"])
            assert scientific_manifest(lm) == scientific_manifest(rm)
            assert [row["episode_id"] for row in lm["episodes"]] == ["episode_2", "episode_10"]
            for episode in left:
                assert dict(left[episode].attrs) == dict(right[episode].attrs)
                assert set(left[episode]) == set(right[episode])
                for name in left[episode]:
                    if name == "center_indices":
                        assert set(left[episode][name]) == set(right[episode][name])
                        for quota in left[episode][name]:
                            np.testing.assert_array_equal(
                                left[episode][name][quota][()], right[episode][name][quota][()]
                            )
                    else:
                        np.testing.assert_array_equal(left[episode][name][()], right[episode][name][()])
                group = left[episode]
                np.testing.assert_array_equal(group["raw_index"], [0, 1, 2, 3, 5, 6, 7, 8, 10])
                np.testing.assert_array_equal(group["source_index"], [0, 1, 0, 1, 2, 3, 4, 5, 6])
                np.testing.assert_array_equal(
                    group["source"], [True, True, False, False, False, False, False, False, False]
                )
                masks = group["action_valid"][()]
                assert not masks[[0, 1, 2, 4, 7, 8]].any()
                assert masks[[3, 6], :7].all()
                assert not masks[5, 2] and masks[5, [0, 1, 3, 4, 5, 6]].all()
                assert group["actions"][5, 2] == 0
                if task == "PatternLock":
                    assert not masks[:, 7].any()
                    assert not group["actions"][:, 7].any()
                assert np.isfinite(group["points"][()]).all()
        for root in (raw_root, parallel):
            sidecar = json.loads((root / "prepared" / f"{task}.manifest.json").read_text())
            assert scientific_manifest(sidecar) == scientific_manifest(lm)


def test_skip_existing_preserves_files_and_rebuilds_incompatible_geometry_and_episode_limit(raw_root):
    initial_prepare(raw_root, episodes=1)
    before = {task: output_snapshot(raw_root, task) for task in TASKS}
    run_cli(raw_root, "--workers", 2, "--episodes", 1, "--skip-existing")
    assert {task: output_snapshot(raw_root, task) for task in TASKS} == before
    run_cli(raw_root, "--workers", 2, "--episodes", 2, "--skip-existing")
    for task in TASKS:
        with h5py.File(raw_root / "prepared" / f"{task}.h5", "r") as source:
            assert set(source) == {"episode_2", "episode_10"}
    run_cli(raw_root, "--workers", 2, "--episodes", 2, "--points", 24, "--skip-existing")
    for task in TASKS:
        with h5py.File(raw_root / "prepared" / f"{task}.h5", "r") as source:
            assert source["episode_2/points"].shape == (9, 24, 6)
            assert json.loads(source.attrs["manifest"])["points"] == 24


def test_skip_still_rehashes_raw_member_and_refuses_changed_input(raw_root):
    initial_prepare(raw_root)
    before = output_snapshot(raw_root, "VideoUnmask")
    with h5py.File(raw_root / "raw/record_dataset_VideoUnmask.h5", "a") as source:
        source["episode_2/timestep_3/obs/front_rgb"][0, 0, 0] = 250
    result, _ = run_cli(raw_root, "--workers", 2, "--skip-existing", success=False)
    assert "SHA256" in result.stderr + result.stdout
    assert output_snapshot(raw_root, "VideoUnmask") == before


@pytest.mark.parametrize("invalid", ["duplicate", "malformed_checksum"])
def test_invalid_task_plan_is_rejected_before_any_output(raw_root, invalid):
    tasks = TASKS
    if invalid == "duplicate":
        tasks = ("VideoUnmask", "VideoUnmask")
    else:
        path = raw_root / "manifests/PatternLock.json"
        manifest = json.loads(path.read_text())
        manifest["members"][0]["sha256"] = "invalid-not-a-sha256"
        path.write_text(json.dumps(manifest))
    run_cli(raw_root, "--workers", 2, tasks=tasks, success=False)
    assert not list((raw_root / "prepared").glob("*.h5"))


def test_legacy_manifest_without_member_digest_remains_usable_with_explicit_provenance(raw_root):
    path = raw_root / "manifests/VideoUnmask.json"
    manifest = json.loads(path.read_text())
    manifest["members"][0].pop("sha256")
    path.write_text(json.dumps(manifest))
    _, events = run_cli(raw_root, "--workers", 2)
    integrity = {event["task"]: event for event in events if event["event"] == "raw_integrity"}
    assert integrity["VideoUnmask"]["download_manifest_verified"] is False
    assert integrity["PatternLock"]["download_manifest_verified"] is True
    raw = raw_root / "raw/record_dataset_VideoUnmask.h5"
    digest = sha256(raw)
    assert integrity["VideoUnmask"]["sha256"] == digest
    with h5py.File(raw_root / "prepared/VideoUnmask.h5", "r") as source:
        assert json.loads(source.attrs["manifest"])["raw_sha256"] == digest
    before = output_snapshot(raw_root, "VideoUnmask")
    _, reused = run_cli(raw_root, "--workers", 2, "--skip-existing")
    assert any(event["event"] == "task_skipped" and event["task"] == "VideoUnmask" for event in reused)
    assert output_snapshot(raw_root, "VideoUnmask") == before


@pytest.mark.parametrize("invalid_digest", ["", 0, False])
def test_falsey_provided_digest_is_rejected_before_orchestration(raw_root, monkeypatch, invalid_digest):
    path = raw_root / "manifests/PatternLock.json"
    manifest = json.loads(path.read_text())
    manifest["members"][0]["sha256"] = invalid_digest
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        prepare, "run_preparation", lambda *args, **kwargs: pytest.fail("invalid digest spawned workers")
    )
    with pytest.raises(ValueError, match="Invalid extracted-file SHA256"):
        prepare.main(["--data-root", str(raw_root), "--tasks", *TASKS, "--workers", "2"])
    assert not (raw_root / "prepared").exists()


def test_worker_failure_preserves_last_complete_output_and_sidecar(raw_root):
    initial_prepare(raw_root)
    before = output_snapshot(raw_root, "VideoUnmask")
    with h5py.File(raw_root / "raw/record_dataset_VideoUnmask.h5", "a") as source:
        del source["episode_10/timestep_7/obs/front_depth"]
    write_manifest(raw_root, "VideoUnmask")  # integrity succeeds; conversion fails partway through.
    result, _ = run_cli(raw_root, "--workers", 2, "--skip-existing", success=False)
    assert "front_depth" in result.stderr + result.stdout
    assert output_snapshot(raw_root, "VideoUnmask") == before


def test_reader_excludes_interrupted_files_and_rejects_explicit_partial_input(raw_root):
    initial_prepare(raw_root, episodes=1)
    partial = raw_root / "prepared/VideoUnmask.partial.h5"
    partial.write_bytes(b"interrupted HDF5 header, deliberately unreadable")
    dataset = EpisodeDataset(raw_root / "prepared")
    assert len(dataset) == 2
    assert set(dataset.by_task) == set(TASKS)
    assert all(ref.path != partial for ref in dataset.episodes)
    with pytest.raises(ValueError, match=r"incomplete.*partial\.h5"):
        EpisodeDataset(partial)


@pytest.mark.parametrize("interruption", ["sigint", "sigterm", "worker_death"])
def test_cancellation_or_dead_worker_joins_peers_and_preserves_outputs(raw_root, tmp_path, interruption):
    initial_prepare(raw_root)
    before = {task: output_snapshot(raw_root, task) for task in TASKS}
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    gates = tmp_path / "gates"
    gates.mkdir()
    # Only these subprocesses install the gate. Actual raw checksums still run;
    # workers then pause before HDF5 reads so cancellation has no timing race.
    (hooks / "sitecustomize.py").write_text(
        "import os, pathlib, time, h5py\n"
        "class GatedFile(h5py.File):\n"
        "    def __init__(self, name, *args, **kwargs):\n"
        f"        if str(name).startswith({str(raw_root / 'raw')!r}):\n"
        f"            pathlib.Path({str(gates)!r}, str(os.getpid())).touch()\n"
        "            while True:\n"
        "                time.sleep(0.01)\n"
        "        super().__init__(name, *args, **kwargs)\n"
        "h5py.File = GatedFile\n"
    )
    env = cli_environment()
    env["PYTHONPATH"] = str(hooks) + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen(
        cli_command(raw_root, "--workers", 2),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    workers = set()
    pending = b""
    deadline = time.monotonic() + 20
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while len(workers) < 2 and time.monotonic() < deadline:
                if process.poll() is not None:
                    _, stderr = process.communicate()
                    pytest.fail(f"preparation exited before workers started: {stderr}")
                for key, _ in selector.select(timeout=0.1):
                    pending += os.read(key.fd, 65536)
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        event = json.loads(line)
                        if event.get("event") == "task_started":
                            workers.add(event["pid"])
        assert len(workers) == 2, "both workers should start before cancellation"
        while not all((gates / str(pid)).exists() for pid in workers) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert all((gates / str(pid)).exists() for pid in workers), "workers did not reach fixture HDF5 gate"
        if interruption == "worker_death":
            os.kill(min(workers), signal.SIGKILL)
        else:
            process.send_signal(signal.SIGINT if interruption == "sigint" else signal.SIGTERM)
        _, stderr = process.communicate(timeout=10)
        assert process.returncode != 0, stderr
        assert all(not Path(f"/proc/{pid}").exists() for pid in workers), "preparation left worker processes"
        assert {task: output_snapshot(raw_root, task) for task in TASKS} == before
    finally:
        # A failing cleanup assertion must not itself leave blocked test workers.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_cancellation_during_start_registers_and_cleans_worker_before_exiting(monkeypatch, signum):
    events, processes, endpoints = [], [], []
    variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "5")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("NUMEXPR_NUM_THREADS", raising=False)
    previous_environment = {name: os.environ.get(name) for name in variables}

    class Endpoint:
        closed = False

        def close(self):
            self.closed = True

    class Process:
        pid = 12345
        alive = False
        closed = False
        exitcode = None

        def __init__(self, **_):
            processes.append(self)

        def start(self):
            assert all(os.environ[name] == "1" for name in variables)
            self.alive = True
            events.append("start")
            signal.raise_signal(signum)
            events.append("start_returned")

        def is_alive(self):
            return self.alive

        def terminate(self):
            events.append("terminate")
            self.alive = False
            self.exitcode = -signal.SIGTERM

        def join(self, timeout=None):
            assert not self.alive
            events.append("join")

        def close(self):
            assert not self.alive, "launch-time cancellation must not leave an unregistered live worker"
            self.closed = True
            events.append("close")

    class Context:
        def Pipe(self, *, duplex):
            assert duplex is False
            pair = Endpoint(), Endpoint()
            endpoints.extend(pair)
            return pair

        def Process(self, **kwargs):
            return Process(**kwargs)

    monkeypatch.setattr(prepare.mp, "get_context", lambda name: Context() if name == "spawn" else None)
    monkeypatch.setattr(prepare, "_cpu_budget", lambda: 2)
    monkeypatch.setattr(prepare, "wait", lambda *args, **kwargs: pytest.fail("cancel before waiting"))
    actual_handlers = {item: signal.getsignal(item) for item in (signal.SIGINT, signal.SIGTERM)}

    def harmless_original_handler(received, _frame):
        events.append(f"unhandled_signal_{received}")

    try:
        for item in actual_handlers:
            signal.signal(item, harmless_original_handler)
        # Requested workers exceed available CPUs, but only two tasks can run.
        with pytest.raises(SystemExit) as raised:
            prepare.run_preparation([{"task": task} for task in TASKS], workers=8, threads_per_worker=1)
        assert raised.value.code == 128 + signum
        assert events == ["start", "start_returned", "terminate", "join", "close"]
        assert len(processes) == 1 and processes[0].closed
        assert len(endpoints) == 2 and all(endpoint.closed for endpoint in endpoints)
        assert {name: os.environ.get(name) for name in variables} == previous_environment
        assert all(signal.getsignal(item) is harmless_original_handler for item in actual_handlers)
    finally:
        for item, handler in actual_handlers.items():
            signal.signal(item, handler)


@pytest.mark.parametrize(
    "affinity,allocation,workers,threads,budget",
    [(8, "2", 2, 2, 2), (1, "8", 2, 1, 1), (3, None, 2, 2, 3)],
)
def test_cpu_allocation_rejects_oversubscription_before_spawn(
    monkeypatch, affinity, allocation, workers, threads, budget
):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(affinity)))
    if allocation is None:
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    else:
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", allocation)
    assert prepare._cpu_budget() == budget
    monkeypatch.setattr(
        prepare.mp, "get_context", lambda *_: pytest.fail("invalid allocation spawned workers")
    )
    environment = dict(os.environ)
    handlers = {item: signal.getsignal(item) for item in (signal.SIGINT, signal.SIGTERM)}
    with pytest.raises(ValueError, match="exceeds.*available/allocated CPUs"):
        prepare.run_preparation(
            [{"task": task} for task in TASKS], workers=workers, threads_per_worker=threads
        )
    assert dict(os.environ) == environment
    assert {item: signal.getsignal(item) for item in handlers} == handlers
