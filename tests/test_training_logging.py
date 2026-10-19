"""Bounded synthetic diagnostics for observable, reproducible training progress."""

from copy import deepcopy
import json
import os
import random
import subprocess
import sys
import threading

import numpy as np
import pytest
import torch
import yaml

from stmem.config import load_config


def _assert_identical(left, right):
    """Compare checkpoint state, including nested RNG and optimizer arrays."""
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_identical(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_identical(a, b)
    else:
        assert left == right


def _tiny_config():
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"].update(
        width=32,
        heads=4,
        layers=1,
        current_tokens=4,
        visual_tokens=4,
        horizon=4,
        points=16,
        checkpoint_blocks=False,
    )
    cfg["training"].update(
        steps=3,
        effective_batch=2,
        checkpoint_every=3,
        checkpoint_steps=[3],
        precision="fp32",
        log_every=50,
    )
    cfg["data"].update(synthetic_execution_frames=5, synthetic_conditioning_frames=2)
    return cfg


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_logging_overrides_preserve_training_and_allow_resume(tmp_path):
    """Changing observation frequency must not change exposure or checkpoint RNG."""
    cfg = _tiny_config()
    roots = {name: tmp_path / name for name in ("baseline", "frequent", "resumed")}
    configs = {}
    for name, root in roots.items():
        copy = deepcopy(cfg)
        copy["paths"]["run_root"] = str(root)
        configs[name] = tmp_path / f"{name}.yaml"
        configs[name].write_text(yaml.safe_dump(copy))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "WANDB_MODE": "offline",
    }
    # Ambient HPC paths must not redirect these tiny diagnostic runs.
    for key in ("STMEM_RUN_ROOT", "STMEM_DATA_ROOT", "STMEM_CACHE_ROOT", "STMEM_STOP_FILE"):
        env.pop(key, None)

    def run(name, *args):
        return subprocess.run(
            [sys.executable, "-m", "stmem.train", "--config", str(configs[name]), *map(str, args)],
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )

    baseline = run("baseline")
    assert baseline.returncode == 0, baseline.stderr
    frequent = run("frequent", "--log-every", 1, "--progress-every-seconds", 0.01)
    assert frequent.returncode == 0, frequent.stderr
    directories = {name: root / "seed0/teacher" for name, root in roots.items()}
    assert [row["step"] for row in _rows(directories["baseline"] / "metrics.jsonl")] == [1, 3]
    metrics = _rows(directories["frequent"] / "metrics.jsonl")
    assert [row["step"] for row in metrics] == [1, 2, 3]
    assert all(row["step_seconds"] > 0 for row in metrics)
    assert all(0 <= row["data_loading_seconds_rank"] <= row["step_seconds"] for row in metrics)
    history = _rows(directories["frequent"] / "tracking_history.jsonl")
    completed = [row["metrics"] for row in history if "train/step_seconds" in row["metrics"]]
    assert [row["progress/step"] for row in completed] == [1, 2, 3]
    assert all(row["train/step_seconds"] > 0 for row in completed)
    progress = _rows(directories["frequent"] / "progress.jsonl")
    assert progress and all(row["event"] == "training_progress" for row in progress)
    assert progress[-1]["phase"] == "complete"
    assert progress[-1]["optimizer_steps_completed"] == 3
    assert progress[-1]["optimizer_step_in_progress"] is None

    stop = tmp_path / "stop"
    stop.touch()
    partial = run("resumed", "--stop-file", stop, "--log-every", 1)
    assert partial.returncode == 75, partial.stderr
    stop.unlink()
    resumed = run(
        "resumed",
        "--resume",
        directories["resumed"] / "last.pt",
        "--stop-file",
        stop,
        "--log-every",
        2,
        "--progress-every-seconds",
        0.01,
    )
    assert resumed.returncode == 0, resumed.stderr
    before = torch.load(directories["baseline"] / "last.pt", weights_only=False, map_location="cpu")
    for name in ("frequent", "resumed"):
        after = torch.load(directories[name] / "last.pt", weights_only=False, map_location="cpu")
        for key in (
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "rng_by_rank",
            "sampler",
            "provenance",
            "step",
        ):
            _assert_identical(before[key], after[key])
        assert after["config"]["training"] == cfg["training"]
        resolved = json.loads((directories[name] / "config.json").read_text())
        assert resolved["training"] == cfg["training"]
        assert "progress_every_seconds" not in resolved["training"]
        expected_logging = {"log_every": 1 if name == "frequent" else 2, "progress_every_seconds": 0.01}
        manifest = json.loads((directories[name] / "manifest.json").read_text())
        assert after["logging"] == manifest["logging"] == expected_logging


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--log-every", "0"),
        ("--log-every", "-1"),
        ("--log-every", "1.5"),
        ("--log-every", "nan"),
        ("--progress-every-seconds", "0"),
        ("--progress-every-seconds", "-1"),
        ("--progress-every-seconds", "nan"),
        ("--progress-every-seconds", "inf"),
        ("--progress-every-seconds", "-inf"),
    ],
)
def test_invalid_logging_options_fail_before_configuration_or_tracking(monkeypatch, capsys, flag, value):
    from stmem import train

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid CLI logging options reached configuration loading or W&B setup")

    monkeypatch.setattr(train, "load_config", forbidden)
    monkeypatch.setattr(train, "RunTracker", forbidden)
    monkeypatch.setattr(sys, "argv", ["stmem.train", "--config", "unused.yaml", f"{flag}={value}"])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
    assert flag in capsys.readouterr().err


def test_heartbeat_observes_a_blocked_phase_without_gpu_or_rng_work(tmp_path, monkeypatch):
    from stmem.training import progress as progress_module

    emitted = threading.Event()
    records = []
    worker_ids = set()
    main_id = threading.get_ident()
    original_print = progress_module.print_json

    def capture(row):
        original_print(row)
        if row.get("phase") == "encoding" and threading.get_ident() != main_id:
            records.append(row)
            worker_ids.add(threading.get_ident())
            if len(records) >= 3:
                emitted.set()

    def forbidden(*args, **kwargs):
        pytest.fail("The progress worker must not synchronize CUDA or interact with W&B")

    monkeypatch.setattr(progress_module, "print_json", capture)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    from stmem.tracking import RunTracker

    monkeypatch.setattr(RunTracker, "log", forbidden)
    monkeypatch.setattr(RunTracker, "summary", forbidden)
    rng_before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with progress_module.TrainingProgress(tmp_path, 0.01) as progress:
        progress.update("previous_example", global_sample_position=127)
        progress.begin_update(4, 32)
        progress.update("encoding", task="VideoUnmask", examples_completed_rank=1)
        # Main-thread work is deliberately blocked, as a slow encode can be.
        assert emitted.wait(timeout=2), "The heartbeat stopped while the main thread was blocked"
    _assert_identical(rng_before, (random.getstate(), np.random.get_state(), torch.get_rng_state()))
    assert threading.get_ident() not in worker_ids
    assert not any(thread.ident in worker_ids for thread in threading.enumerate())
    assert len(records) >= 3
    for row in records:
        assert row["event"] == "training_progress"
        assert row["optimizer_steps_completed"] == 4
        assert row["optimizer_step_in_progress"] == 5
        assert row["examples_completed_rank"] == 1
        assert row["task"] == "VideoUnmask"
        assert row["global_sample_position"] is None
        assert 0 <= row["phase_seconds"] <= row["elapsed_seconds"]
    assert records[-1]["phase_seconds"] > records[0]["phase_seconds"]
    assert records[-1]["elapsed_seconds"] > records[0]["elapsed_seconds"]
    assert all(row["event"] == "training_progress" for row in _rows(tmp_path / "progress.jsonl"))


@pytest.mark.parametrize("interval,rank", [(None, 0), (0.01, 1)])
def test_disabled_or_nonprimary_progress_has_no_worker_or_output(tmp_path, capsys, interval, rank):
    from stmem.training.progress import TrainingProgress

    threads = {thread.ident for thread in threading.enumerate()}
    with TrainingProgress(tmp_path, interval, rank=rank) as progress:
        progress.begin_update(0, 32)
        progress.update("encoding", task="VideoUnmask")
        assert {thread.ident for thread in threading.enumerate()} == threads
    assert not (tmp_path / "progress.jsonl").exists()
    assert not capsys.readouterr().out


def test_progress_thread_exits_when_training_raises(tmp_path, monkeypatch):
    from stmem.training import progress as progress_module

    observed = threading.Event()
    worker_ids = set()

    def capture(row):
        if threading.current_thread() is not threading.main_thread():
            worker_ids.add(threading.get_ident())
            observed.set()

    monkeypatch.setattr(progress_module, "print_json", capture)
    with pytest.raises(RuntimeError, match="synthetic diagnostic failure"):
        with progress_module.TrainingProgress(tmp_path, 0.01):
            assert observed.wait(timeout=2)
            raise RuntimeError("synthetic diagnostic failure")
    assert worker_ids
    assert not any(thread.ident in worker_ids for thread in threading.enumerate())
    final = _rows(tmp_path / "progress.jsonl")[-1]
    assert final["phase"] == "failed" and final["error_type"] == "RuntimeError"


@pytest.mark.parametrize("training_failed", [False, True])
def test_final_heartbeat_io_error_preserves_training_exception(tmp_path, monkeypatch, training_failed):
    from stmem.training.progress import TrainingProgress

    expected_type = RuntimeError if training_failed else OSError
    expected_message = "training failed" if training_failed else "final flush failed"
    with pytest.raises(expected_type, match=expected_message):
        with TrainingProgress(tmp_path, 30) as progress:
            original_emit = progress._emit

            def fail_final_emit():
                if threading.current_thread() is threading.main_thread():
                    raise OSError("final flush failed")
                original_emit()

            monkeypatch.setattr(progress, "_emit", fail_final_emit)
            if training_failed:
                raise RuntimeError("training failed")
    assert not progress._thread.is_alive()
    assert progress._stream.closed
