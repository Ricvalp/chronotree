"""Bounded CPU diagnostics for true batches, remainder weights, resume and DDP."""

from copy import deepcopy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from stmem.config import load_config


def _config(root, *, batch=5, steps=3):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["paths"]["run_root"] = str(root)
    cfg["model"].update(
        width=16,
        heads=4,
        layers=1,
        text_layers=1,
        max_text_bytes=16,
        current_tokens=4,
        visual_tokens=4,
        horizon=4,
        points=6,
        checkpoint_blocks=True,
        encoder_options={**cfg["model"]["encoder_options"], "neighbor_cap": 4},
    )
    cfg["training"].update(
        steps=steps,
        effective_batch=batch,
        microbatch=2,
        encoder_block_batch_size=2,
        checkpoint_every=steps,
        checkpoint_steps=[steps],
        precision="fp32",
        log_every=1,
        length_bucket_size=1,
    )
    cfg["data"].update(synthetic_execution_frames=6, synthetic_conditioning_frames=1)
    cfg["tracking"].update(enabled=False, scenes={"enabled": False})
    return cfg


def _cpu_env():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
    for key in (
        "STMEM_RUN_ROOT",
        "STMEM_DATA_ROOT",
        "STMEM_CACHE_ROOT",
        "STMEM_STOP_FILE",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        env.pop(key, None)
    return env


def _write_config(path, cfg):
    path.write_text(yaml.safe_dump(cfg))
    return path


def _run(path, *args, world=1):
    command = [sys.executable]
    if world > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={world}"]
    command += ["-m", "stmem.train", "--config", str(path), *map(str, args)]
    return subprocess.run(command, env=_cpu_env(), capture_output=True, text=True, timeout=90)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _assert_identical(left, right):
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


def test_true_batches_and_short_remainder_match_accumulated_example_gradients(tmp_path, monkeypatch):
    from stmem import train

    monkeypatch.setattr(train, "distributed_setup", lambda: (torch.device("cpu"), 0, 1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train, "environment_manifest", lambda: {"diagnostic": "synthetic CPU test"})
    monkeypatch.delenv("RANK", raising=False)
    for key in ("STMEM_RUN_ROOT", "STMEM_DATA_ROOT", "STMEM_CACHE_ROOT"):
        monkeypatch.delenv(key, raising=False)
    original_flow = train.flow_sample

    def fixed_flow(actions, valid, normalizer):
        return original_flow(
            actions, valid, normalizer, noise=torch.zeros_like(actions), time=torch.tensor(0.4)
        )

    monkeypatch.setattr(train, "flow_sample", fixed_flow)
    original_clip = torch.nn.utils.clip_grad_norm_
    gradients, batch_shapes = [], []

    def capture_gradients(parameters, *args, **kwargs):
        parameters = list(parameters)
        gradients.append([None if p.grad is None else p.grad.detach().clone() for p in parameters])
        return original_clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_gradients)
    original_forward = train.MemoryPolicy.forward

    def capture_batches(self, decisions, *args, **kwargs):
        batch_shapes.append(len(decisions) if isinstance(decisions, (list, tuple)) else 1)
        return original_forward(self, decisions, *args, **kwargs)

    monkeypatch.setattr(train.MemoryPolicy, "forward", capture_batches)
    cfg = _config(tmp_path / "serial", steps=1)
    cfg["training"]["microbatch"] = 1
    assert train.train_policy(cfg, "teacher") == 0
    assert batch_shapes == [1] * 5
    batch_shapes.clear()
    cfg["paths"]["run_root"] = str(tmp_path / "batched")
    cfg["training"]["microbatch"] = 2
    assert train.train_policy(cfg, "teacher") == 0
    assert batch_shapes == [2, 2, 1]
    assert len(gradients) == 2
    for serial, batched in zip(*gradients):
        assert (serial is None) == (batched is None)
        if serial is not None:
            assert torch.isfinite(batched).all()
            torch.testing.assert_close(batched, serial, atol=4e-6, rtol=4e-4)
    row = _rows(tmp_path / "batched/seed0/teacher/metrics.jsonl")[0]
    assert row["effective_batch"] == row["examples"] == 5
    assert row["microbatch_size_rank"] == 2
    assert row["microbatches_per_update_rank"] == 3
    assert row["gradient_norm_before_clip"] > 0


@pytest.mark.parametrize("history_mode", ["full", "none"])
def test_batched_training_checkpoint_resume_is_bitwise_deterministic(tmp_path, history_mode):
    cfg = _config(tmp_path / "complete")
    if history_mode == "none":
        cfg["model"].update(
            history_mode="none",
            proprioception_mode="none",
            condition_on_previous_commands=False,
        )
    complete_cfg = _write_config(tmp_path / "complete.yaml", cfg)
    resumed_cfg = deepcopy(cfg)
    resumed_cfg["paths"]["run_root"] = str(tmp_path / "resumed")
    resumed_path = _write_config(tmp_path / "resumed.yaml", resumed_cfg)
    complete = _run(complete_cfg)
    assert complete.returncode == 0, complete.stderr
    stop = tmp_path / "stop"
    stop.touch()
    partial = _run(resumed_path, "--stop-file", stop)
    assert partial.returncode == 75, partial.stderr
    checkpoint = tmp_path / "resumed/seed0/teacher/last.pt"
    saved_partial = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert saved_partial["step"] == 1
    assert saved_partial["sampler"]["next_global_position"] == 5
    stop.unlink()
    resumed = _run(resumed_path, "--resume", checkpoint, "--stop-file", stop)
    assert resumed.returncode == 0, resumed.stderr
    before = torch.load(tmp_path / "complete/seed0/teacher/last.pt", weights_only=False, map_location="cpu")
    after = torch.load(checkpoint, weights_only=False, map_location="cpu")
    for key in ("model", "optimizer", "scheduler", "scaler", "rng_by_rank", "sampler", "provenance", "step"):
        _assert_identical(before[key], after[key])
    assert after["sampler"]["next_global_position"] == 15
    assert [row["step"] for row in _rows(checkpoint.parent / "metrics.jsonl")] == [1, 2, 3]


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed is unavailable")
@pytest.mark.parametrize("history_mode", ["full", "none"])
def test_two_rank_cpu_ddp_runs_true_microbatches_and_writes_one_checkpoint(tmp_path, history_mode):
    from stmem.models.policy import MemoryPolicy

    batch = 128 if history_mode == "none" else 8
    cfg = _config(tmp_path / "ddp", batch=batch, steps=2)
    if history_mode == "none":
        cfg["model"].update(
            history_mode="none",
            proprioception_mode="none",
            condition_on_previous_commands=False,
        )
        # Exact requested allocation semantics: 64 examples/rank, two ranks,
        # one microbatch each, therefore 128 examples per optimizer update.
        cfg["training"]["microbatch"] = 64
    cfg["training"]["weight_decay"] = 0  # Encoder changes must come from gradients.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg["seed"])
        initial = MemoryPolicy(cfg).state_dict()
    path = _write_config(tmp_path / "ddp.yaml", cfg)
    result = _run(path, world=2)
    assert result.returncode == 0, result.stderr
    directory = tmp_path / "ddp/seed0/teacher"
    state = torch.load(directory / "last.pt", weights_only=False, map_location="cpu")
    assert state["step"] == 2 and state["provenance"]["world_size"] == 2
    assert len(state["rng_by_rank"]) == 2
    assert state["sampler"]["next_global_position"] == batch * 2
    assert all(torch.isfinite(value).all() for value in state["model"].values())
    learned = ["encoder.visual.point_mlp.0.weight"]
    if history_mode == "full":
        learned.append("encoder.event.state.0.weight")
    for name in learned:
        assert not torch.equal(state["model"][name], initial[name]), name
    rows = _rows(directory / "metrics.jsonl")
    assert [row["step"] for row in rows] == [1, 2]
    assert all(row["effective_batch"] == batch for row in rows)
    if history_mode == "none":
        assert all(row["microbatch_size_rank"] == 64 for row in rows)
        assert all(row["microbatches_per_update_rank"] == 1 for row in rows)
        assert not any(name.startswith(("encoder.event.", "proprio.", "clock.")) for name in state["model"])
    else:
        assert all(row["microbatch_size_rank"] == row["microbatches_per_update_rank"] == 2 for row in rows)
    assert all(
        np.isfinite(row["training_action_loss"]) and row["gradient_norm_before_clip"] > 0 for row in rows
    )
    assert (directory / "complete.json").exists()
