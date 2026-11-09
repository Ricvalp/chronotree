"""Synthetic CPU checks: runtime saving changes must preserve exact training."""

from copy import deepcopy
import argparse
import json
import sys

import pytest
import torch

from stmem.training.checkpoints import file_sha256, positive_checkpoint_every
from test_batched_training import _run, _write_config
from test_training_continuation import (
    _assert_training_identical,
    _directory,
    _read,
    _tiny_config,
    cpu_training,  # noqa: F401
)
from test_tracking import fake_wandb  # noqa: F401


def test_runtime_cadence_retains_snapshots_and_resumes_exactly(
    cpu_training, monkeypatch, tmp_path, fake_wandb,  # noqa: F811
):
    train = cpu_training
    configs = {name: _tiny_config(tmp_path / name, tracking=True) for name in ("reference", "changed")}

    def run(name, *, interval=None, resume=False, stop_at=None):
        cfg = configs[name]
        before = deepcopy(cfg)
        stop = _directory(cfg) / "requested.stop"
        stop.unlink(missing_ok=True)
        original = train.TrainingProgress.begin_update

        def begin(progress, completed, examples):
            if stop_at is not None:
                assert completed < stop_at, "Bounded synthetic diagnostic overran its stop"
                if completed + 1 == stop_at:
                    stop.touch()
            original(progress, completed, examples)

        with monkeypatch.context() as patch:
            patch.setattr(train.TrainingProgress, "begin_update", begin)
            result = train.train_policy(
                cfg, "teacher", resume=_directory(cfg) / "last.pt" if resume else None,
                checkpoint_every=interval, attention_every=0, stop_file=stop,
                train_until_stopped=resume,
            )
        assert cfg == before
        return result

    assert run("reference") == 0
    assert run("changed", interval=2) == 0
    directory = _directory(configs["changed"])
    assert {p.name for p in directory.glob("step_*.pt")} == {"step_00000002.pt", "step_00000003.pt"}
    assert torch.load(directory / "last.pt.previous", weights_only=False)["step"] == 2
    _assert_training_identical(_read(configs["reference"]), _read(configs["changed"]))
    candidate_hash = file_sha256(directory / "step_00000003.pt")
    tracking_id = json.loads((directory / "tracking.json").read_text())["id"]

    # Continue past the original decay horizon; change saving cadence again on
    # the second resume. Stop saves must work even off the periodic boundary.
    assert run("reference", resume=True, stop_at=7) == 75
    assert run("changed", interval=2, resume=True, stop_at=5) == 75
    assert _read(configs["changed"])["step"] == 5
    assert (directory / "step_00000004.pt").is_file()
    assert not (directory / "step_00000005.pt").exists()
    assert run("changed", interval=3, resume=True, stop_at=7) == 75
    after = _read(configs["changed"])
    _assert_training_identical(_read(configs["reference"]), after)
    assert after["optimizer"]["param_groups"][0]["lr"] == .1 * configs["changed"]["training"]["lr"]
    assert after["config"]["training"] == configs["changed"]["training"]
    assert file_sha256(directory / "step_00000003.pt") == candidate_hash
    assert (directory / "step_00000006.pt").is_file()
    assert not (directory / "step_00000007.pt").exists()
    assert json.loads((directory / "tracking.json").read_text())["id"] == tracking_id
    assert json.loads((directory / "partial.json").read_text())["step"] == 7
    expected = {"periodic_every": 3, "retain_periodic": True, "step_basis": "absolute_completed_optimizer_steps"}
    assert after["checkpointing"] == expected
    assert json.loads((directory / "manifest.json").read_text())["checkpointing"] == expected


def test_two_rank_cli_cadence_and_checkpointed_resume(tmp_path):
    cfg = _tiny_config(tmp_path / "ddp")
    path = _write_config(tmp_path / "config.yaml", cfg)
    stop = tmp_path / "stop"
    stop.touch()
    initial = _run(path, "--checkpoint-every", 2, "--stop-file", stop, "--attention-every", 0, world=2)
    assert initial.returncode != 0 and "exitcode: 75" in initial.stderr, initial.stderr
    directory = _directory(cfg)
    assert _read(cfg)["step"] == 1
    stop.unlink()
    resumed = _run(path, "--resume", directory / "last.pt", "--checkpoint-every", 2,
                   "--attention-every", 0, world=2)
    assert resumed.returncode == 0, resumed.stderr
    for step in (2, 3):
        saved = torch.load(directory / f"step_{step:08d}.pt", weights_only=False, map_location="cpu")
        assert saved["step"] == step and len(saved["rng_by_rank"]) == 2
        assert saved["config"]["training"] == cfg["training"]
        assert saved["checkpointing"]["periodic_every"] == 2


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "nan"])
def test_invalid_cli_cadence_fails_before_loading_or_tracking(monkeypatch, value):
    from stmem import train

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid checkpoint interval reached configuration loading or W&B")

    monkeypatch.setattr(train, "load_config", forbidden)
    monkeypatch.setattr(train, "RunTracker", forbidden)
    monkeypatch.setattr(sys, "argv", ["stmem.train", "--config", "unused", f"--checkpoint-every={value}"])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2


@pytest.mark.parametrize("value", [True, 0, -2, 1.5])
def test_invalid_direct_api_cadence(value):
    with pytest.raises(argparse.ArgumentTypeError):
        positive_checkpoint_every(value)
