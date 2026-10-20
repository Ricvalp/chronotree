"""Synthetic CPU integration checks for deterministic background input workers."""

from copy import deepcopy
import json
import subprocess
import sys

import pytest
import torch

from test_batched_training import _assert_identical, _config, _cpu_env, _rows, _run, _write_config


def _worker_config(root, *, workers, batch=5, steps=3):
    cfg = _config(root, batch=batch, steps=steps)
    cfg["training"].update(workers=workers, prefetch_factor=1)
    return cfg


def _checkpoint(root):
    return torch.load(root / "seed0/teacher/last.pt", weights_only=False, map_location="cpu")


def _assert_loader_metrics(rows, workers):
    for row in rows:
        assert row["data_prefetch_workers_rank"] == workers
        assert row["data_prefetch_capacity_rank"] == workers  # prefetch_factor=1
        assert row["data_loading_seconds_rank"] == row["data_wait_seconds_rank"]
        assert row["data_wait_seconds_rank"] >= 0
        assert row["data_worker_load_seconds_rank"] > 0
        assert row["data_worker_pack_seconds_rank"] > 0


def test_background_workers_preserve_samples_gradients_and_main_process_rng(tmp_path):
    """Worker scheduling/iterator seeds must not alter flow noise or updates."""
    states = {}
    for workers in (0, 2):
        root = tmp_path / f"workers_{workers}"
        cfg = _worker_config(root, workers=workers)
        result = _run(_write_config(tmp_path / f"workers_{workers}.yaml", cfg))
        assert result.returncode == 0, result.stderr
        states[workers] = _checkpoint(root)
        rows = _rows(root / "seed0/teacher/metrics.jsonl")
        assert [row["step"] for row in rows] == [1, 2, 3]
        assert all(row["microbatches_per_update_rank"] == 3 for row in rows)
        _assert_loader_metrics(rows, workers)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng_by_rank", "sampler", "step"):
        _assert_identical(states[0][key], states[2][key])
    # The worker count is an explicit configuration change, recorded in provenance.
    assert states[0]["provenance"]["training"] != states[2]["provenance"]["training"]


def test_prefetched_training_stop_resume_is_bitwise_deterministic(tmp_path):
    """Discarding future prefetched data cannot move the sampler resume boundary."""
    cfg = _worker_config(tmp_path / "complete", workers=2)
    complete_path = _write_config(tmp_path / "complete.yaml", cfg)
    resumed_cfg = deepcopy(cfg)
    resumed_cfg["paths"]["run_root"] = str(tmp_path / "resumed")
    resumed_path = _write_config(tmp_path / "resumed.yaml", resumed_cfg)
    complete = _run(complete_path)
    assert complete.returncode == 0, complete.stderr

    stop = tmp_path / "stop"
    stop.touch()
    partial = _run(resumed_path, "--stop-file", stop)
    assert partial.returncode == 75, partial.stderr
    checkpoint = tmp_path / "resumed/seed0/teacher/last.pt"
    partial_state = _checkpoint(tmp_path / "resumed")
    assert partial_state["step"] == 1
    assert partial_state["sampler"]["next_global_position"] == 5
    assert not (checkpoint.parent / "complete.json").exists()
    stop.unlink()

    resumed = _run(resumed_path, "--resume", checkpoint, "--stop-file", stop)
    assert resumed.returncode == 0, resumed.stderr
    expected = _checkpoint(tmp_path / "complete")
    actual = _checkpoint(tmp_path / "resumed")
    for key in ("model", "optimizer", "scheduler", "scaler", "rng_by_rank", "sampler", "provenance", "step"):
        _assert_identical(expected[key], actual[key])
    assert actual["sampler"]["next_global_position"] == 15
    assert [row["step"] for row in _rows(checkpoint.parent / "metrics.jsonl")] == [1, 2, 3]
    assert (checkpoint.parent / "complete.json").exists()
    assert not (checkpoint.parent / "partial.json").exists()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed is unavailable")
def test_two_cpu_ddp_ranks_each_use_two_background_workers(tmp_path):
    """Spawn workers after DDP setup, preserving rank coverage and encoder gradients."""
    from stmem.models.policy import MemoryPolicy

    root = tmp_path / "ddp"
    cfg = _worker_config(root, workers=2, batch=8, steps=2)
    cfg["training"]["weight_decay"] = 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg["seed"])
        initial = MemoryPolicy(cfg).state_dict()
    path = _write_config(tmp_path / "ddp.yaml", cfg)
    result = _run(path, world=2)
    assert result.returncode == 0, result.stderr
    saved = _checkpoint(root)
    assert saved["step"] == 2
    assert saved["provenance"]["world_size"] == 2
    assert saved["config"]["training"]["workers"] == 2
    assert len(saved["rng_by_rank"]) == 2
    assert saved["sampler"]["next_global_position"] == 16
    assert all(torch.isfinite(value).all() for value in saved["model"].values())
    for name in ("encoder.visual.point_mlp.0.weight", "encoder.event.state.0.weight"):
        assert not torch.equal(saved["model"][name], initial[name]), name
    rows = _rows(root / "seed0/teacher/metrics.jsonl")
    assert [row["step"] for row in rows] == [1, 2]
    _assert_loader_metrics(rows, 2)
    assert all(row["effective_batch"] == 8 for row in rows)
    assert all(row["microbatch_size_rank"] == row["microbatches_per_update_rank"] == 2 for row in rows)
    assert (root / "seed0/teacher/complete.json").exists()


@pytest.fixture(scope="module")
def frozen_prefetch_pipeline(tmp_path_factory):
    """Build a real cache from a tiny explicitly synthetic teacher checkpoint."""
    root = tmp_path_factory.mktemp("prefetch_frozen_pipeline")
    cfg = _worker_config(root / "teacher", workers=2, steps=1)
    cfg["paths"]["cache_root"] = str(root / "cache")
    path = _write_config(root / "teacher.yaml", cfg)
    result = _run(path)
    assert result.returncode == 0, result.stderr
    checkpoint = root / "teacher/seed0/teacher/last.pt"
    checkpoint.with_suffix(".selection.json").write_text(
        json.dumps(
            {
                "synthetic_debug_only": True,
                "rule": "Tiny synthetic loader diagnostic, not a scientific checkpoint selection.",
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "stmem.data.cache",
            "--config",
            str(path),
            "--checkpoint",
            str(checkpoint),
        ],
        env=_cpu_env(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    cfg["checkpoints"]["teacher"] = str(checkpoint)
    cfg["training"].update(steps=2, checkpoint_every=2, checkpoint_steps=[2])
    return cfg, _checkpoint(root / "teacher")


@pytest.mark.parametrize("stage", ["compact", "perceiver"])
def test_prefetch_preserves_frozen_cache_training(stage, frozen_prefetch_pipeline, tmp_path):
    """Frozen-cache consumers reconstruct canonical decisions without changing results."""
    cfg, teacher = frozen_prefetch_pipeline
    states = {}
    for workers in (0, 2):
        run = tmp_path / f"{stage}_{workers}"
        student = deepcopy(cfg)
        student["paths"]["run_root"] = str(run)
        student["stage"] = stage
        student["training"]["workers"] = workers
        path = _write_config(tmp_path / f"{stage}_{workers}.yaml", student)
        result = _run(path)
        assert result.returncode == 0, result.stderr
        directory = run / f"seed0/{stage}"
        states[workers] = torch.load(directory / "last.pt", weights_only=False, map_location="cpu")
        _assert_loader_metrics(_rows(directory / "metrics.jsonl"), workers)
        assert (directory / "complete.json").exists()
        # Cached visual/event encoders remain the exact selected teacher weights.
        for name in ("encoder.visual.point_mlp.0.weight", "encoder.event.state.0.weight"):
            _assert_identical(states[workers]["model"][name], teacher["model"][name])
    for key in ("model", "optimizer", "scheduler", "scaler", "rng_by_rank", "sampler", "step"):
        _assert_identical(states[0][key], states[2][key])
