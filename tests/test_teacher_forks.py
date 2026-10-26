"""Synthetic CPU diagnostics for matched teacher continuations, not policy quality."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from stmem.config import digest, load_config, validate_config
from stmem.training.checkpoints import file_sha256
from stmem.training.forks import load_fork_checkpoint, teacher_fork_config, validate_teacher_fork


FLAG = "condition_on_previous_joint_commands"


def _tiny_config(root):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["paths"]["run_root"] = str(root)
    cfg["tracking"]["scenes"]["enabled"] = False
    cfg["model"].update(
        width=16,
        heads=4,
        layers=1,
        text_layers=1,
        max_text_bytes=24,
        current_tokens=4,
        visual_tokens=4,
        horizon=4,
        points=12,
        checkpoint_blocks=False,
    )
    cfg["training"].update(
        steps=8,
        effective_batch=2,
        microbatch=2,
        checkpoint_every=2,
        checkpoint_steps=[2, 4, 6, 8],
        precision="fp32",
        workers=0,
        lr=0.001,
        warmup_steps=2,
    )
    cfg["data"].update(episodes_per_task=2, synthetic_execution_frames=8, synthetic_conditioning_frames=2)
    return cfg


def _call(cfg, directory, *arguments):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("STMEM_", "WANDB_", "SLURM_"))
        and key not in {"RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
    }
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", WANDB_MODE="offline")
    return subprocess.run(
        [sys.executable, "-m", "stmem.train", "--config", str(config_path), *map(str, arguments)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _read(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_equal(before, after):
    if isinstance(before, torch.Tensor):
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            _assert_equal(before[key], after[key])
    elif isinstance(before, (tuple, list)):
        assert len(before) == len(after)
        for left, right in zip(before, after):
            _assert_equal(left, right)
    else:
        assert before == after


@pytest.fixture(scope="module")
def parent_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_teacher_forks")
    cfg = _tiny_config(root / "parent")
    result = _call(cfg, root / "parent_command")
    assert result.returncode == 0, result.stdout + result.stderr
    checkpoint = root / "parent/seed0/teacher/step_00000002.pt"
    return root, cfg, checkpoint, _read(checkpoint)


def _fork_config(parent_run, name, enabled):
    root, _, _, saved = parent_run
    return teacher_fork_config(
        saved,
        run_root=root / name,
        additional_steps=4,
        condition_on_previous_joint_commands=enabled,
    )


def _provenance(saved, cfg):
    return {
        **saved["provenance"],
        "model": digest(cfg["model"]),
        "training": digest(cfg["training"]),
    }


def test_control_fork_matches_uninterrupted_training_and_has_new_tracking(parent_run):
    root, _, source, saved = parent_run
    cfg = _fork_config(parent_run, "control", True)
    before_hash = file_sha256(source)
    result = _call(cfg, root / "control_command", "--fork-from", source, "--fork-sha256", before_hash)
    assert result.returncode == 0, result.stdout + result.stderr
    directory = root / "control/seed0/teacher"
    actual = _read(directory / "last.pt")
    expected = _read(root / "parent/seed0/teacher/step_00000006.pt")
    for key in ("model", "optimizer", "scheduler", "scaler", "sampler"):
        _assert_equal(expected[key], actual[key])
    for left, right in zip(expected["rng_by_rank"], actual["rng_by_rank"]):
        torch.testing.assert_close(left["torch"], right["torch"], atol=0, rtol=0)
        assert left["python"] == right["python"]
    assert actual["step"] == 6
    assert actual["config"]["training"]["steps"] == 6
    assert actual["config"]["training"]["schedule_steps"] == 8
    # Step 6 lies in cosine decay: a mistaken six-step horizon would already
    # give the final 0.1 * peak LR, whereas the original eight-step schedule
    # gives 0.325 * peak LR here. This must not be a warmup-only parity check.
    assert actual["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.000325)
    assert (directory / "step_00000004.pt").exists()
    assert (directory / "step_00000006.pt").exists()
    marker = json.loads((directory / "complete.json").read_text())
    assert marker["completed_steps"] == marker["intended_steps"] == 6
    assert actual["tracking"]["id"] != saved["tracking"]["id"]
    assert actual["tracking"]["resume_requested"] is False
    lineage = actual["fork"]
    assert lineage["source_checkpoint"] == str(source.resolve())
    assert lineage["source_sha256"] == before_hash
    assert lineage["source_step"] == 2
    assert lineage["source_tracking_id"] == saved["tracking"]["id"]
    assert lineage["schedule_steps"] == 8
    assert lineage["intended_final_step"] == 6
    assert json.loads((directory / "manifest.json").read_text())["fork"] == lineage
    assert file_sha256(source) == before_hash
    metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    reference = {
        row["step"]: row
        for row in (
            json.loads(line)
            for line in (root / "parent/seed0/teacher/metrics.jsonl").read_text().splitlines()
        )
    }
    assert [row["step"] for row in metrics] == [3, 4, 5, 6]
    for row in metrics:
        assert row["learning_rate"] == reference[row["step"]]["learning_rate"]
        assert row["training_action_loss"] == reference[row["step"]]["training_action_loss"]


def test_ablated_fork_changes_learning_without_resetting_schedule_or_normalization(parent_run):
    root, _, source, saved = parent_run
    cfg = _fork_config(parent_run, "ablated", False)
    result = _call(cfg, root / "ablated_command", "--fork-from", source)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = _read(root / "ablated/seed0/teacher/last.pt")
    baseline = _read(root / "parent/seed0/teacher/step_00000006.pt")
    assert actual["step"] == 6
    assert actual["config"]["model"][FLAG] is False
    _assert_equal(actual["scheduler"], baseline["scheduler"])
    _assert_equal(actual["sampler"], baseline["sampler"])
    for key in saved["model"]:
        if key.startswith("normalizer."):
            _assert_equal(saved["model"][key], actual["model"][key])
    assert any(not torch.equal(actual["model"][key], baseline["model"][key]) for key in actual["model"])
    assert actual["fork"]["changes"][FLAG] == {"before": True, "after": False}


def test_fork_interruption_and_normal_resume_preserve_lineage_and_training(parent_run):
    root, _, source, _ = parent_run
    cfg = _fork_config(parent_run, "resumed_branch", True)
    stop = root / "continuation.stop"
    stop.touch()
    result = _call(cfg, root / "resumed_command", "--fork-from", source, "--stop-file", stop)
    assert result.returncode == 75, result.stdout + result.stderr
    checkpoint = root / "resumed_branch/seed0/teacher/last.pt"
    partial = _read(checkpoint)
    assert partial["step"] == 3
    assert not checkpoint.with_name("complete.json").exists()
    stop.unlink()
    result = _call(cfg, root / "resumed_command", "--resume", checkpoint, "--stop-file", stop)
    assert result.returncode == 0, result.stdout + result.stderr
    final = _read(checkpoint)
    baseline = _read(root / "parent/seed0/teacher/step_00000006.pt")
    for key in ("model", "optimizer", "scheduler", "sampler"):
        _assert_equal(baseline[key], final[key])
    assert final["fork"] == partial["fork"]
    assert final["tracking"]["id"] == partial["tracking"]["id"]


def test_fork_helper_does_not_mutate_source_and_preserves_original_schedule(parent_run):
    _, _, _, saved = parent_run
    before = deepcopy(saved["config"])
    cfg = _fork_config(parent_run, "helper", False)
    assert saved["config"] == before
    assert cfg["training"]["schedule_steps"] == before["training"]["steps"] == 8
    assert cfg["training"]["steps"] == 6
    assert cfg["training"]["checkpoint_steps"] == [4, 6]
    assert cfg["model"][FLAG] is False
    validate_config(cfg)
    validate_teacher_fork(saved, cfg, _provenance(saved, cfg))


@pytest.mark.parametrize("change", ["lr", "batch", "warmup", "schedule", "model", "data", "world"])
def test_fork_rejects_unmatched_experimental_changes(parent_run, change):
    _, _, _, saved = parent_run
    cfg = _fork_config(parent_run, f"invalid_{change}", False)
    if change == "lr":
        cfg["training"]["lr"] *= 2
    elif change == "batch":
        cfg["training"]["effective_batch"] *= 2
    elif change == "warmup":
        cfg["training"]["warmup_steps"] += 1
    elif change == "schedule":
        cfg["training"]["schedule_steps"] += 1
    elif change == "model":
        cfg["model"]["points"] *= 2
    elif change == "data":
        cfg["data"]["decision_stride"] *= 2
    provenance = _provenance(saved, cfg)
    if change == "world":
        provenance["world_size"] *= 2
    with pytest.raises(ValueError):
        validate_teacher_fork(saved, cfg, provenance)


def test_fork_accepts_legacy_missing_command_flag_and_rejects_bad_parent_digest(parent_run):
    _, _, _, parent = parent_run
    saved = deepcopy(parent)
    saved["config"]["model"].pop(FLAG)
    saved["provenance"]["model"] = digest(saved["config"]["model"])
    cfg = teacher_fork_config(
        saved, run_root="unused", additional_steps=4, condition_on_previous_joint_commands=False
    )
    validate_teacher_fork(saved, cfg, _provenance(saved, cfg))
    saved["provenance"]["model"] = "incorrect parent model digest"
    with pytest.raises(ValueError):
        validate_teacher_fork(saved, cfg, _provenance(saved, cfg))


@pytest.mark.parametrize("additional_steps", [0, -1, 1.5, True, 7])
def test_fork_helper_rejects_invalid_or_beyond_schedule_budget(parent_run, additional_steps):
    _, _, _, saved = parent_run
    with pytest.raises(ValueError):
        teacher_fork_config(
            saved,
            run_root="unused",
            additional_steps=additional_steps,
            condition_on_previous_joint_commands=False,
        )


def test_fork_rejects_wrong_source_hash_before_training(parent_run):
    root, _, source, _ = parent_run
    cfg = _fork_config(parent_run, "wrong_hash", False)
    result = _call(cfg, root / "wrong_hash_command", "--fork-from", source, "--fork-sha256", "0" * 64)
    assert result.returncode != 0
    assert not (root / "wrong_hash/seed0/teacher/last.pt").exists()
    assert "hash" in result.stderr.lower() or "sha256" in result.stderr.lower()


def test_fork_load_uses_the_hashed_file_despite_atomic_filename_replacement(tmp_path, monkeypatch):
    from stmem.training import runtime

    source = tmp_path / "last.pt"
    replacement = tmp_path / "next.pt"
    torch.save({"step": 1, "value": torch.tensor([1.0])}, source)
    torch.save({"step": 2, "value": torch.tensor([2.0])}, replacement)
    source_hash = file_sha256(source)
    original_load = runtime.load_checkpoint

    def replace_before_deserialization(stream):
        assert hasattr(stream, "read") and stream.tell() == 0
        os.replace(replacement, source)
        return original_load(stream)

    monkeypatch.setattr(runtime, "load_checkpoint", replace_before_deserialization)
    loaded = load_fork_checkpoint(source, source_hash)
    assert loaded["step"] == 1
    torch.testing.assert_close(loaded["value"], torch.tensor([1.0]), atol=0, rtol=0)
    assert _read(source)["step"] == 2
    assert file_sha256(source) != source_hash


def test_fork_load_rejects_wrong_hash_before_deserialization(tmp_path, monkeypatch):
    from stmem.training import runtime

    source = tmp_path / "checkpoint.pt"
    torch.save({"step": 1}, source)
    monkeypatch.setattr(
        runtime,
        "load_checkpoint",
        lambda *_: pytest.fail("A mismatched source must not be deserialized"),
    )
    with pytest.raises(ValueError, match="SHA256"):
        load_fork_checkpoint(source, "0" * 64)


def test_fork_rejects_destination_with_existing_tracking_identity(parent_run):
    root, _, source, _ = parent_run
    cfg = _fork_config(parent_run, "used_destination", True)
    directory = root / "used_destination/seed0/teacher"
    directory.mkdir(parents=True)
    tracking = directory / "tracking.json"
    tracking.write_text('{"id": "existing-run"}\n')
    result = _call(cfg, root / "used_destination_command", "--fork-from", source)
    assert result.returncode != 0
    assert json.loads(tracking.read_text()) == {"id": "existing-run"}
    assert not (directory / "last.pt").exists()


@pytest.mark.parametrize("value", [0, -1, 2, 1.5, True, "8"])
def test_schedule_horizon_validation(value):
    cfg = _tiny_config("unused")
    cfg["training"]["schedule_steps"] = value
    with pytest.raises(ValueError, match="schedule_steps"):
        validate_config(cfg)
