"""Bounded synthetic CPU checks for an explicitly open-ended teacher resume."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path

import pytest
import torch

from stmem.training.checkpoints import file_sha256
from stmem.training.continuation import continuation_record
from test_batched_training import _assert_identical, _config, _rows, _run, _write_config
from test_tracking import fake_wandb  # noqa: F401


def _tiny_config(root, *, tracking=False):
    cfg = _config(root, batch=2, steps=3)
    cfg["training"].update(
        lr=0.0001, warmup_steps=1, checkpoint_every=1, workers=0,
    )
    cfg["model"]["checkpoint_blocks"] = False
    cfg["tracking"].update(enabled=tracking, mode="online")
    return cfg


def _directory(cfg):
    return Path(cfg["paths"]["run_root"]) / "seed0/teacher"


def _read(cfg):
    return torch.load(_directory(cfg) / "last.pt", map_location="cpu", weights_only=False)


@pytest.fixture
def cpu_training(monkeypatch):
    from stmem import train

    for key in tuple(os.environ):
        if key.startswith(("STMEM_", "WANDB_", "SLURM_")) or key in {
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
        }:
            monkeypatch.delenv(key)
    monkeypatch.setattr(train, "distributed_setup", lambda: (torch.device("cpu"), 0, 1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train, "environment_manifest", lambda: {"diagnostic": "synthetic CPU"})
    monkeypatch.setattr(train, "source_revision", lambda: {"synthetic_test": True})
    return train


def _bounded_call(train, monkeypatch, cfg, *, stop_at=None, resume=False, continuation=False):
    """Exercise the real stop-file path; an overrun fails before another update."""
    directory = _directory(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    stop = directory / "test.stop"
    stop.unlink(missing_ok=True)
    original = train.TrainingProgress.begin_update

    def begin_update(progress, completed, examples):
        if stop_at is not None:
            assert completed < stop_at, "Open-ended test ignored its bounded stop request"
            if completed + 1 == stop_at:
                stop.touch()
        original(progress, completed, examples)

    before = deepcopy(cfg)
    with monkeypatch.context() as patch:
        patch.setattr(train.TrainingProgress, "begin_update", begin_update)
        result = train.train_policy(
            cfg,
            "teacher",
            resume=directory / "last.pt" if resume else None,
            stop_file=stop,
            attention_every=0,
            train_until_stopped=continuation,
        )
    assert cfg == before
    return result


def _assert_training_identical(before, after):
    for key in (
        "model", "optimizer", "scheduler", "scaler", "rng_by_rank", "sampler", "provenance", "step",
    ):
        _assert_identical(before[key], after[key])


def test_open_ended_resume_keeps_exact_training_across_schedule_endpoint(cpu_training, monkeypatch, tmp_path):
    """Interrupt both before and after the old endpoint without moving samples or LR."""
    configs = {name: _tiny_config(tmp_path / name) for name in ("single_tail", "resumed_tail", "finite")}
    train = cpu_training
    for name in ("single_tail", "resumed_tail"):
        assert _bounded_call(train, monkeypatch, configs[name], stop_at=1) == 75
    initial = _read(configs["single_tail"])
    _assert_training_identical(initial, _read(configs["resumed_tail"]))

    assert _bounded_call(
        train, monkeypatch, configs["single_tail"], stop_at=6, resume=True, continuation=True,
    ) == 75
    # The first continuation saves before the old endpoint; omitting the flag
    # must already fail here, rather than silently reverting to finite training.
    resumed_cfg = configs["resumed_tail"]
    assert _bounded_call(train, monkeypatch, resumed_cfg, stop_at=2, resume=True, continuation=True) == 75
    with pytest.raises(ValueError, match="resume with --train-until-stopped"):
        _bounded_call(train, monkeypatch, resumed_cfg, stop_at=3, resume=True)
    assert _read(resumed_cfg)["step"] == 2

    for endpoint in (3, 4, 6):
        assert _bounded_call(
            train, monkeypatch, resumed_cfg, stop_at=endpoint, resume=True, continuation=True,
        ) == 75
        assert not (_directory(resumed_cfg) / "complete.json").exists()
        assert _read(resumed_cfg)["step"] == endpoint

    before, after = _read(configs["single_tail"]), _read(resumed_cfg)
    _assert_training_identical(before, after)
    assert after["provenance"] == initial["provenance"]
    assert after["sampler"]["next_global_position"] == 12
    assert after["config"] == resumed_cfg
    assert after["continuation"] == before["continuation"] == {
        "mode": "constant_lr_tail_v1",
        "enabled_at_step": 1,
        "original_steps": 3,
        "schedule_steps": 3,
        "final_lr": 0.1 * resumed_cfg["training"]["lr"],
    }
    assert after["scheduler"]["last_epoch"] == 6
    floor = 0.1 * resumed_cfg["training"]["lr"]
    assert after["optimizer"]["param_groups"][0]["lr"] == floor
    assert after["scheduler"]["_last_lr"] == [floor]

    single_rows = _rows(_directory(configs["single_tail"]) / "metrics.jsonl")
    resumed_rows = _rows(_directory(resumed_cfg) / "metrics.jsonl")
    assert [row["step"] for row in resumed_rows] == list(range(1, 7))
    for left, right in zip(single_rows, resumed_rows, strict=True):
        assert left["training_action_loss"] == right["training_action_loss"]
        assert left["learning_rate"] == right["learning_rate"]
    # Expected post-update LR is the original warmup/cosine schedule followed
    # by its constant floor, with no restarted warmup or stretched decay.
    expected = [0.0001, 0.000055, floor, floor, floor, floor]
    assert [row["learning_rate"] for row in single_rows] == pytest.approx(expected)

    assert _bounded_call(train, monkeypatch, configs["finite"]) == 0
    finite = _read(configs["finite"])
    for name in ("single_tail", "resumed_tail"):
        candidate = torch.load(
            _directory(configs[name]) / "step_00000003.pt", map_location="cpu", weights_only=False,
        )
        _assert_training_identical(finite, candidate)
        assert candidate["continuation"]["enabled_at_step"] == 1


def test_finished_teacher_continues_same_wandb_run_and_preserves_candidate(
    cpu_training, monkeypatch, tmp_path, fake_wandb,  # noqa: F811
):
    cfg = _tiny_config(tmp_path / "finished", tracking=True)
    assert _bounded_call(cpu_training, monkeypatch, cfg) == 0
    directory = _directory(cfg)
    assert (directory / "complete.json").exists()
    initial = _read(cfg)
    original_candidate_hash = file_sha256(directory / "step_00000003.pt")
    initial_id = json.loads((directory / "tracking.json").read_text())["id"]

    for endpoint in (4, 5):
        assert _bounded_call(
            cpu_training, monkeypatch, cfg, stop_at=endpoint, resume=True, continuation=True,
        ) == 75
        assert not (directory / "complete.json").exists()
        assert file_sha256(directory / "step_00000003.pt") == original_candidate_hash
        saved = _read(cfg)
        assert saved["step"] == endpoint
        assert saved["provenance"] == initial["provenance"]
        assert saved["continuation"]["enabled_at_step"] == 3
        assert saved["tracking"]["id"] == initial_id
        tracking = json.loads((directory / "tracking.json").read_text())
        assert tracking["id"] == initial_id
        assert tracking["status"] == "checkpointed_partial"
        assert tracking["summary"]["training/intended_steps"] is None
        assert tracking["summary"]["training/original_intended_steps"] == 3
        assert tracking["summary"]["training/train_until_stopped"] is True
        assert json.loads((directory / "partial.json").read_text())["step"] == endpoint

    assert len(fake_wandb) == 3
    assert {run.kwargs["id"] for run in fake_wandb} == {initial_id}
    assert all(run.kwargs["resume"] == "allow" for run in fake_wandb)
    assert [run.finished for run in fake_wandb] == [[0], [75], [75]]
    history = _rows(directory / "tracking_history.jsonl")
    assert [row["history_step"] for row in history] == list(range(len(history)))
    updates = [row["metrics"]["progress/step"] for row in history if "train/step" in row["metrics"]]
    assert updates == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("stage,resume", [("teacher", None), ("compact", "last.pt"), ("perceiver", "last.pt")])
def test_continuation_requires_teacher_resume(cpu_training, tmp_path, stage, resume):
    with pytest.raises(ValueError, match="requires teacher training with --resume"):
        cpu_training.train_policy(
            _tiny_config(tmp_path), stage, resume=resume, train_until_stopped=True,
        )


@pytest.mark.parametrize(
    "replacement",
    [
        [],
        {"mode": "unknown"},
        {"enabled_at_step": -1},
        {"enabled_at_step": True},
        {"enabled_at_step": 4},
        {"enabled_at_step": 1.5},
        {"original_steps": 4},
        {"schedule_steps": 4},
        {"final_lr": math.nan},
        {"final_lr": 0.1},
        {"unexpected": "field"},
    ],
)
def test_continuation_rejects_inconsistent_saved_metadata(tmp_path, replacement):
    cfg = _tiny_config(tmp_path)
    saved = {"step": 3}
    previous = continuation_record(cfg, saved, requested=True)
    saved["continuation"] = {**previous, **replacement} if isinstance(replacement, dict) else replacement
    with pytest.raises(ValueError, match="[Cc]ontinuation"):
        continuation_record(cfg, saved, requested=True)


@pytest.mark.skipif(not torch.distributed.is_available(), reason="PyTorch distributed is unavailable")
def test_two_rank_cpu_continuation_passes_endpoint_and_checkpoints_both_ranks(tmp_path):
    cfg = _tiny_config(tmp_path / "ddp")
    path = _write_config(tmp_path / "ddp.yaml", cfg)
    initial = _run(path, "--attention-every", "0", world=2)
    assert initial.returncode == 0, initial.stdout + initial.stderr
    checkpoint = _directory(cfg) / "last.pt"
    assert _read(cfg)["step"] == 3
    stop = tmp_path / "ddp.stop"
    stop.touch()
    for expected in (4, 5):
        result = _run(
            path, "--resume", checkpoint, "--train-until-stopped", "--stop-file", stop,
            "--attention-every", "0", world=2,
        )
        # torchrun reports a ChildFailedError for the intentional partial exit;
        # each rank's actual trainer return code must still be 75.
        assert result.returncode != 0
        assert "exitcode: 75" in result.stderr or "exitcode  : 75" in result.stderr, result.stderr
        saved = _read(cfg)
        assert saved["step"] == expected
        assert saved["provenance"]["world_size"] == 2
        assert len(saved["rng_by_rank"]) == 2
        assert saved["sampler"]["next_global_position"] == expected * cfg["training"]["effective_batch"]
        assert saved["continuation"]["enabled_at_step"] == 3
        assert saved["optimizer"]["param_groups"][0]["lr"] == 0.1 * cfg["training"]["lr"]
        assert not (_directory(cfg) / "complete.json").exists()
