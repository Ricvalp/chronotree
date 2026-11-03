"""Tiny synthetic integration checks for cached contexts inside DDP/resume."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_batched_training import _assert_identical, _cpu_env, _rows, _run, _write_config
from test_training_prefetch import frozen_prefetch_pipeline as _frozen_prefetch_pipeline

cached_pipeline = _frozen_prefetch_pipeline


def _student(cfg, root):
    result = deepcopy(cfg)
    result["stage"] = "compact"
    result["paths"]["run_root"] = str(root)
    result["training"].update(workers=2, effective_batch=8, microbatch=2, steps=2,
                              checkpoint_every=2, checkpoint_steps=[2])
    return result


def _indexed(cfg):
    """Derive an index from the synthetic fixture without any new encoding."""
    cfg = deepcopy(cfg)
    manifests = list(Path(cfg["paths"]["cache_root"]).glob("*/manifest.json"))
    assert len(manifests) == 1
    base = manifests[0].parent
    output = base / "integration_hierarchy"
    result = subprocess.run(
        [sys.executable, "-m", "stmem.data.hierarchy_cache", "--cache-root", str(base),
         "--output-root", str(output)], env=_cpu_env(), capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    cfg["training"]["hierarchy_cache_root"] = str(output)
    return cfg


@pytest.mark.skipif(not torch.distributed.is_available(), reason="DDP unavailable")
@pytest.mark.parametrize("indexed", [False, True], ids=["legacy", "indexed"])
def test_cached_context_workers_with_two_ddp_ranks(tmp_path, cached_pipeline, indexed):
    cfg, teacher = cached_pipeline
    if indexed:
        cfg = _indexed(cfg)
    root = tmp_path / "ddp"
    student = _student(cfg, root)
    result = _run(_write_config(tmp_path / "ddp.yaml", student), world=2)
    assert result.returncode == 0, result.stderr
    state = torch.load(root / "seed0/compact/last.pt", weights_only=False, map_location="cpu")
    assert state["provenance"]["training_engine"] == "cached_context_batch_v1"
    assert state["step"] == 2 and state["provenance"]["world_size"] == 2
    assert len(state["rng_by_rank"]) == 2
    assert state["sampler"]["next_global_position"] == 16
    assert state["model"].keys() == teacher["model"].keys()
    for name, value in teacher["model"].items():
        if name.startswith(("encoder.", "text_encoder.")):
            torch.testing.assert_close(state["model"][name], value, rtol=0, atol=0)
    assert not torch.equal(state["model"]["adapter.content.weight"], teacher["model"]["adapter.content.weight"])
    rows = _rows(root / "seed0/compact/metrics.jsonl")
    assert [row["step"] for row in rows] == [1, 2]
    assert all(row["cache_worker_seconds_rank"] > 0 for row in rows)
    misses_key = "hierarchy_index_misses_rank" if indexed else "cache_block_misses_rank"
    assert sum(row[misses_key] for row in rows) > 0
    assert all(row["data_wait_seconds_max_rank"] >= row["data_wait_seconds_rank"] for row in rows)
    if indexed:
        assert "hierarchy" in state["provenance"]["dependencies"]
        assert all(row["hierarchy_read_bytes_rank"] > 0 and row["hierarchy_read_seconds_rank"] > 0 for row in rows)


@pytest.mark.parametrize("indexed", [False, True], ids=["legacy", "indexed"])
def test_cached_worker_stop_resume_matches_uninterrupted(tmp_path, cached_pipeline, indexed):
    cfg, _ = cached_pipeline
    if indexed:
        cfg = _indexed(cfg)
    full = _student(cfg, tmp_path / "full")
    resumed = _student(cfg, tmp_path / "resumed")
    full_path = _write_config(tmp_path / "full.yaml", full)
    resumed_path = _write_config(tmp_path / "resumed.yaml", resumed)
    result = _run(full_path)
    assert result.returncode == 0, result.stderr
    stop = tmp_path / "stop"
    stop.touch()
    result = _run(resumed_path, "--stop-file", stop)
    assert result.returncode == 75, result.stderr
    checkpoint = tmp_path / "resumed/seed0/compact/last.pt"
    partial = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert partial["step"] == 1
    stop.unlink()
    result = _run(resumed_path, "--resume", checkpoint, "--stop-file", stop)
    assert result.returncode == 0, result.stderr
    actual = torch.load(checkpoint, weights_only=False, map_location="cpu")
    expected = torch.load(tmp_path / "full/seed0/compact/last.pt", weights_only=False, map_location="cpu")
    for key in ("model", "optimizer", "scheduler", "scaler", "rng_by_rank", "sampler", "provenance"):
        _assert_identical(actual[key], expected[key])
