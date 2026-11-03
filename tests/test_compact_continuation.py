"""Bounded synthetic CPU checks; continuation must preserve real optimizer state."""

from copy import deepcopy
import json

import pytest
import torch

from stmem.train import continuation_record
from stmem.training.checkpoints import file_sha256
from test_batched_training import _assert_identical, _rows, _run, _write_config
from test_cached_training import _indexed, _student
from test_training_continuation import cpu_training  # noqa: F401
from test_training_prefetch import frozen_prefetch_pipeline as cached_pipeline  # noqa: F401
from test_tracking import fake_wandb  # noqa: F401


def _config(cfg, root):
    cfg = _student(cfg, root)
    cfg["training"].update(schedule_steps=2, warmup_steps=1, lr=1e-4,
                           workers=0, effective_batch=4, checkpoint_every=1)
    return cfg


def test_completed_compact_tail_preserves_rng_optimizer_and_wandb(
    tmp_path, cached_pipeline, cpu_training, monkeypatch, fake_wandb,  # noqa: F811
):
    cfg, _ = cached_pipeline
    cfg = _indexed(cfg)
    final_states = []
    for name, stops in (("uninterrupted", ()), ("resumed", (3,))):
        config = _config(cfg, tmp_path / name)
        config["tracking"].update(enabled=True, mode="online")
        directory = tmp_path / name / "seed0/compact"
        checkpoint = directory / "last.pt"
        assert cpu_training.train_policy(config, "compact", attention_every=0) == 0
        initial = torch.load(checkpoint, weights_only=False)
        candidate = directory / "step_00000002.pt"
        candidate_hash = file_sha256(candidate)
        tracking_id = initial["tracking"]["id"]
        original = deepcopy(config)
        assert (directory / "complete.json").exists()
        stop = directory / "test.stop"
        for endpoint in (*stops, 5):
            stop.unlink(missing_ok=True)
            begin = cpu_training.TrainingProgress.begin_update

            def begin_update(progress, completed, examples):
                assert completed < endpoint
                if endpoint != 5 and completed + 1 == endpoint:
                    stop.touch()
                begin(progress, completed, examples)

            with monkeypatch.context() as patch:
                patch.setattr(cpu_training.TrainingProgress, "begin_update", begin_update)
                result = cpu_training.train_policy(
                    config, "compact", resume=checkpoint, continue_to_step=5,
                    stop_file=stop, attention_every=0,
                )
            assert result == (0 if endpoint == 5 else 75)
            assert (directory / "complete.json").exists() == (endpoint == 5)
            state = torch.load(checkpoint, weights_only=False)
            assert state["step"] == endpoint
            assert state["tracking"]["id"] == tracking_id
            assert state["provenance"] == initial["provenance"]
            assert state["config"] == config == original
            assert file_sha256(candidate) == candidate_hash
            assert state["optimizer"]["param_groups"][0]["lr"] == 1e-5
            assert state["scheduler"]["last_epoch"] == endpoint
            assert state["sampler"]["next_global_position"] == endpoint * 4
        assert state["continuation"] == {
            "mode": "bounded_constant_lr_tail_v1", "original_steps": 2,
            "schedule_steps": 2, "final_lr": 1e-5, "enabled_at_step": 2, "intended_final_step": 5,
        }
        rows = _rows(directory / "metrics.jsonl")
        assert [r["step"] for r in rows] == [1, 2, 3, 4, 5]
        assert [r["learning_rate"] for r in rows[2:]] == [1e-5] * 3
        tracking = json.loads((directory / "tracking.json").read_text())
        assert tracking["summary"]["training/intended_steps"] == 5
        assert tracking["summary"]["training/original_intended_steps"] == 2
        assert tracking["summary"]["training/schedule_steps"] == 2
        assert {run.kwargs["id"] for run in fake_wandb if run.kwargs["id"] == tracking_id} == {tracking_id}
        assert len([run for run in fake_wandb if run.kwargs["id"] == tracking_id]) == 2 + len(stops)
        final_states.append(state)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng_by_rank", "sampler", "provenance"):
        _assert_identical(final_states[0][key], final_states[1][key])


@pytest.mark.skipif(not torch.distributed.is_available(), reason="DDP unavailable")
def test_bounded_compact_tail_two_rank_cli_resumes_and_finishes(tmp_path, cached_pipeline):  # noqa: F811
    cfg, _ = cached_pipeline
    cfg = _config(cfg, tmp_path / "ddp")
    cfg["training"].update(workers=2, effective_batch=8)
    path = _write_config(tmp_path / "ddp.yaml", cfg)
    checkpoint = tmp_path / "ddp/seed0/compact/last.pt"
    result = _run(path, "--attention-every", "0", world=2)
    assert result.returncode == 0, result.stderr
    candidate = checkpoint.with_name("step_00000002.pt")
    original_hash = file_sha256(candidate)
    stop = tmp_path / "stop"
    stop.touch()
    result = _run(path, "--resume", checkpoint, "--continue-to-step", "4",
                  "--stop-file", stop, "--attention-every", "0", world=2)
    assert result.returncode != 0 and "exitcode: 75" in result.stderr, result.stderr
    partial = torch.load(checkpoint, weights_only=False)
    assert partial["step"] == 3 and len(partial["rng_by_rank"]) == 2
    assert not checkpoint.with_name("complete.json").exists()
    stop.unlink()
    result = _run(path, "--resume", checkpoint, "--continue-to-step", "4",
                  "--attention-every", "0", world=2)
    assert result.returncode == 0, result.stderr
    state = torch.load(checkpoint, weights_only=False)
    assert state["step"] == 4 and state["sampler"]["next_global_position"] == 32
    assert state["optimizer"]["param_groups"][0]["lr"] == 1e-5
    assert len(state["rng_by_rank"]) == 2
    assert checkpoint.with_name("complete.json").exists()
    assert file_sha256(candidate) == original_hash


@pytest.mark.parametrize("stage,resume,endpoint", [
    ("teacher", "last.pt", 4), ("compact", None, 4), ("perceiver", "last.pt", 4),
    ("compact", "last.pt", 2), ("compact", "last.pt", True),
])
def test_bounded_extension_is_explicit_and_compact_only(cpu_training, tmp_path, stage, resume, endpoint):  # noqa: F811
    from test_batched_training import _config as tiny_config

    cfg = tiny_config(tmp_path, steps=2)
    with pytest.raises(ValueError, match="--continue-to-step"):
        cpu_training.train_policy(cfg, stage, resume=resume, continue_to_step=endpoint)


def test_bounded_metadata_retains_original_schedule_and_requires_explicit_endpoint():
    cfg = {"training": {"steps": 30000, "schedule_steps": 30000, "lr": 1e-4}}
    saved = {"step": 30000}
    record = continuation_record(cfg, saved, requested=False, continue_to_step=60000)
    saved.update(step=40000, continuation=record)
    assert continuation_record(cfg, saved, requested=False, continue_to_step=60000) == record
    with pytest.raises(ValueError, match="resume with --continue-to-step"):
        continuation_record(cfg, saved, requested=False)
    with pytest.raises(ValueError, match="shorten"):
        continuation_record(cfg, saved, requested=False, continue_to_step=50000)
    extended = continuation_record(cfg, saved, requested=False, continue_to_step=90000)
    assert extended["enabled_at_step"] == 30000 and extended["intended_final_step"] == 90000
    changed = deepcopy(cfg)
    changed["training"]["schedule_steps"] = 40000
    with pytest.raises(ValueError, match="record|schedule"):
        continuation_record(changed, saved, requested=False, continue_to_step=60000)
    with pytest.raises(ValueError, match="original training and LR decay"):
        continuation_record(cfg, {"step": 29000}, requested=False, continue_to_step=60000)


def test_deployed_encoder_cache_identity_survives_runtime_continuation_and_eval_changes():
    from stmem.data.cache_source import cache_code_fingerprint, _AUDITED_LEGACY_SHA256

    # Production-source regression: synthetic-only fingerprint fixtures missed a
    # validation helper being edited in a conservatively protected source file.
    assert cache_code_fingerprint() == _AUDITED_LEGACY_SHA256
