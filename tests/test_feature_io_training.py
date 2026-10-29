"""Synthetic fixture diagnostic: runtime I/O changes preserve training/resume."""

from copy import deepcopy
import json
import os
import shutil
import subprocess
import sys

import torch

from test_frozen_feature_cache import cache_input, prepare  # noqa: F401
from test_training_logging import _assert_identical


def test_prefetch_and_staged_path_resume_preserve_optimizer_state(cache_input, tmp_path):  # noqa: F811
    cfg, _, _, sidecar = cache_input
    prepare(cache_input)
    cfg["model"].update(width=32, heads=4, layers=1, horizon=4, execute=2, checkpoint_blocks=False)
    cfg["training"].update(
        steps=3, effective_batch=2, microbatch=2, encoder_block_batch_size=1,
        checkpoint_every=3, checkpoint_steps=[3], precision="fp32", workers=0,
    )
    cfg["tracking"].update(mode="offline", system_stats=False)
    cfg["tracking"]["scenes"]["enabled"] = False
    staged = tmp_path / "staged"
    staged.mkdir()
    shutil.copy2(sidecar, staged / sidecar.name)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "WANDB_MODE": "offline",
    }
    for key in (
        "STMEM_RUN_ROOT", "STMEM_DATA_ROOT", "STMEM_CACHE_ROOT", "STMEM_STOP_FILE",
        "STMEM_FEATURE_CACHE_ROOT", "STMEM_FEATURE_PREFETCH_CHUNKS",
    ):
        env.pop(key, None)
    configs, directories = {}, {}
    for name in ("baseline", "resumed"):
        configuration = deepcopy(cfg)
        configuration["paths"]["run_root"] = str(tmp_path / name)
        configs[name] = tmp_path / f"{name}.json"
        configs[name].write_text(json.dumps(configuration))
        directories[name] = tmp_path / name / "seed0/teacher"

    def run(name, depth, *extra, override=None):
        environment = {**env, "STMEM_FEATURE_PREFETCH_CHUNKS": str(depth)}
        if override is not None:
            environment["STMEM_FEATURE_CACHE_ROOT"] = str(override)
        return subprocess.run(
            [sys.executable, "-m", "stmem.train", "--config", str(configs[name]),
             "--log-every", "1", "--attention-every", "0", *map(str, extra)],
            env=environment, capture_output=True, text=True, timeout=90,
        )

    baseline = run("baseline", 0)
    assert baseline.returncode == 0, baseline.stderr
    stop = tmp_path / "stop"
    stop.touch()
    interrupted = run("resumed", 0, "--stop-file", stop)
    assert interrupted.returncode == 75, interrupted.stderr
    stop.unlink()
    resumed = run("resumed", 2, "--resume", directories["resumed"] / "last.pt", override=staged)
    assert resumed.returncode == 0, resumed.stderr
    before = torch.load(directories["baseline"] / "last.pt", weights_only=False, map_location="cpu")
    after = torch.load(directories["resumed"] / "last.pt", weights_only=False, map_location="cpu")
    for key in ("model", "optimizer", "scheduler", "scaler", "rng_by_rank", "sampler", "provenance", "step"):
        _assert_identical(before[key], after[key])
    assert after["config"]["training"] == cfg["training"]
    manifest = json.loads((directories["resumed"] / "manifest.json").read_text())
    assert manifest["feature_io"]["prefetch_chunks"] == 2
    assert manifest["feature_io"]["cache_root_override"] == str(staged)
    metrics = [json.loads(row) for row in (directories["resumed"] / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in metrics] == [1, 2, 3]
    for row in metrics:
        assert row["feature_bytes_rank"] > 0
        assert row["feature_chunks_rank"] > 0
        assert row["feature_read_seconds_rank"] > 0
        assert row["feature_wait_seconds_rank"] >= 0
        assert row["feature_transfer_seconds_rank"] == row["feature_transfer_bytes_rank"] == 0
        assert row["feature_read_seconds_max_rank"] == row["feature_read_seconds_rank"]
        assert row["step_seconds_max_rank"] == row["step_seconds"]
    history = [json.loads(row)["metrics"] for row in
               (directories["resumed"] / "tracking_history.jsonl").read_text().splitlines()]
    assert any(row.get("train/feature_bytes_rank", 0) > 0 for row in history)
