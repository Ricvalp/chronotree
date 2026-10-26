"""Synthetic metadata checks for isolated, matched HPC continuation jobs."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import torch
import yaml

from stmem.config import digest, load_config
from stmem.training.checkpoints import file_sha256


REPO = Path(__file__).resolve().parents[1]
FLAG = "condition_on_previous_joint_commands"


@pytest.fixture
def continuation_launcher(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith(("STMEM_", "WANDB_", "SLURM_")):
            monkeypatch.delenv(key)
    spec = importlib.util.spec_from_file_location(
        "teacher_continuation_launcher", REPO / "hpc/continue_teacher.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    repo = tmp_path / "repository with spaces"
    (repo / "src").mkdir(parents=True)
    (repo / "src/identity.py").write_text('"""Synthetic source fingerprint fixture."""\n')
    run_root = repo / "runs/continuation_pair"
    generated = repo / "hpc/generated/continuation_pair"
    original = repo / "runs/original"
    checkpoint = original / "seed0/teacher/last.pt"
    checkpoint.parent.mkdir(parents=True)
    cfg = load_config(REPO / "configs/teacher_b200_b128_m8_blocks32.yaml")
    cfg["paths"]["run_root"] = str(original)
    cfg["model"].pop(FLAG, None)  # The real parent checkpoint predates the new flag.
    source_step = 49978
    saved = {
        "config": cfg,
        "step": source_step,
        "stage": "teacher",
        "provenance": {
            "data": "synthetic-fixture-provenance-only",
            "dependencies": {},
            "model": digest(cfg["model"]),
            "training": digest(cfg["training"]),
            "seed": 0,
            "world_size": 4,
            "effective_batch": 128,
            "training_engine": "padded_policy_batch_v1",
            "sampler": "synthetic-sampler-fixture",
            "rank_assignment": "rotating_rank_v1",
        },
        "sampler": {
            "next_global_position": source_step * 128,
            "boundary": "optimizer_step",
            "version": "synthetic-sampler-fixture",
            "rank_assignment": "rotating_rank_v1",
        },
        "rng_by_rank": [{"synthetic_fixture_only": True} for _ in range(4)],
        "tracking": {"id": "parent-run-must-not-be-reused"},
        "model": {"synthetic_fixture_only": torch.tensor([1.0])},
    }
    torch.save(saved, checkpoint)
    site = tmp_path / "site.env"
    values = {
        "STMEM_REPO_ROOT": str(repo),
        "STMEM_DATA_ROOT": str(tmp_path / "prepared data"),
        "STMEM_CACHE_ROOT": str(tmp_path / "cache"),
        "STMEM_RUN_ROOT": str(original),
        "STMEM_PYTHON": sys.executable,
        "STMEM_PROFILE": "b200_4gpu",
        "STMEM_PARTITION": "aiq",
        "STMEM_GPU_TYPE": "",
        "STMEM_CPUS": "64",
        "STMEM_MEM": "128G",
        "STMEM_WALLTIME": "48:00:00",
        "WANDB_MODE": "online",
        "WANDB_PROJECT": "chronotree",
    }
    site.write_text("\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n")

    def invoke(*arguments):
        launcher.main(
            [
                "--site",
                str(site),
                "--checkpoint",
                str(checkpoint),
                "--run-root",
                "runs/continuation_pair",
                "--output-dir",
                str(generated),
                *arguments,
            ]
        )
        return json.loads((generated / "plan.json").read_text())

    return SimpleNamespace(
        launcher=launcher,
        invoke=invoke,
        repo=repo,
        run_root=run_root,
        generated=generated,
        original=original,
        checkpoint=checkpoint,
        saved=saved,
        site=site,
        values=values,
    )


def test_dry_run_pins_one_parent_and_matches_training_recipe(continuation_launcher, monkeypatch):
    fixture = continuation_launcher
    source_hash = file_sha256(fixture.checkpoint)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            fixture.launcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry-run submitted")
        )
        plan = fixture.invoke()
    assert plan["job_count"] == plan["maximum_concurrent_allocations"] == 2
    assert plan["maximum_total_gpus"] == 8
    assert plan["source_step"] == 49978 and plan["end_step"] == 59978
    assert plan["schedule_steps"] == 100000
    assert plan["source_sha256"] == source_hash == file_sha256(plan["snapshot"])
    assert file_sha256(fixture.checkpoint) == source_hash
    assert not (fixture.generated / "submitted.json").exists()
    assert not (fixture.run_root / "control").exists()
    assert not (fixture.run_root / "no_previous_joint_commands").exists()
    configs = {name: yaml.safe_load(Path(path).read_text()) for name, path in plan["configs"].items()}
    for name, enabled in fixture.launcher.BRANCHES.items():
        cfg = configs[name]
        assert cfg["model"][FLAG] is enabled
        assert cfg["paths"]["run_root"] == str(fixture.run_root / name)
        assert cfg["paths"]["data_root"] == fixture.values["STMEM_DATA_ROOT"]
        assert cfg["data"] == fixture.saved["config"]["data"]
        assert cfg["geometry"] == fixture.saved["config"]["geometry"]
        assert cfg["training"]["steps"] == 59978
        assert cfg["training"]["schedule_steps"] == 100000
        assert cfg["training"]["checkpoint_steps"] == [54978, 59978]
        assert cfg["training"]["lr"] == 0.0001
        assert cfg["training"]["effective_batch"] == 128
        assert cfg["training"]["microbatch"] == 8
        assert cfg["training"]["encoder_block_batch_size"] == 32
    left, right = deepcopy(configs["control"]), deepcopy(configs["no_previous_joint_commands"])
    right["model"][FLAG] = True
    right["paths"]["run_root"] = left["paths"]["run_root"]
    assert left == right
    for job, branch in zip(plan["jobs"], fixture.launcher.BRANCHES, strict=True):
        command = job["commands"][0]
        assert command[command.index("--fork-from") + 1] == plan["snapshot"]
        assert command[command.index("--fork-sha256") + 1] == source_hash
        assert "--resume" not in command and not job["deps"]
        assert job["inputs"] == [plan["snapshot"]]
        assert job["outputs"] == [str(fixture.run_root / branch / "seed0/teacher/step_00059978.pt")]
        script = Path(job["script"]).read_text()
        for directive in [
            "--gres=gpu:4",
            "--partition=aiq",
            "--time=48:00:00",
            "--cpus-per-task=64",
            "--mem=128G",
        ]:
            assert directive in script
        assert script.count("--nproc-per-node=4") == 1
        assert f"export STMEM_RUN_ROOT={shlex.quote(str(fixture.run_root / branch))}" in script
        assert "unset WANDB_RUN_ID WANDB_RESUME" in script
        assert "parent-run-must-not-be-reused" not in script
        subprocess.run(["bash", "-n", job["script"]], check=True)


def test_repeated_dry_run_and_submit_reuse_snapshot_when_source_advances(continuation_launcher, monkeypatch):
    fixture = continuation_launcher
    plan = fixture.invoke()
    snapshot = Path(plan["snapshot"])
    before = snapshot.stat().st_mtime_ns
    assert fixture.invoke()["source_sha256"] == plan["source_sha256"]
    assert snapshot.stat().st_mtime_ns == before
    changed = deepcopy(fixture.saved)
    changed["model"]["synthetic_fixture_only"].add_(1)
    changed["step"] += 1000
    changed["sampler"]["next_global_position"] = changed["step"] * 128
    torch.save(changed, fixture.checkpoint)
    assert fixture.invoke()["source_step"] == 49978
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"{60000 + len(calls)}\n")

    monkeypatch.setattr(fixture.launcher.subprocess, "run", sbatch)
    submitted = fixture.invoke("--submit")
    assert submitted["source_step"] == 49978 and submitted["end_step"] == 59978
    assert len(calls) == 2
    assert file_sha256(snapshot) == plan["source_sha256"]
    assert snapshot.stat().st_mtime_ns == before


def test_corrupted_snapshot_is_rejected_before_submission(continuation_launcher, monkeypatch):
    fixture = continuation_launcher
    plan = fixture.invoke()
    snapshot = Path(plan["snapshot"])
    snapshot.write_bytes(snapshot.read_bytes() + b"corrupted fixture")
    monkeypatch.setattr(fixture.launcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("Submitted"))
    with pytest.raises(SystemExit):
        fixture.invoke("--submit")
    assert not (fixture.generated / "submitted.json").exists()


def test_orphan_snapshot_is_not_silently_replaced(continuation_launcher):
    fixture = continuation_launcher
    snapshot = fixture.generated / "inputs/teacher.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"Unreviewed original artifact")
    with pytest.raises(SystemExit):
        fixture.invoke()
    assert snapshot.read_bytes() == b"Unreviewed original artifact"


@pytest.mark.parametrize(
    "arguments",
    [
        ["--profile", "b200_1gpu"],
        ["--additional-steps", "0"],
        ["--additional-steps", "60000"],
        ["--run-root", "runs/original/branch"],
        ["--run-root", "runs"],
    ],
)
def test_invalid_requests_fail_before_artifact_creation(continuation_launcher, arguments):
    fixture = continuation_launcher
    with pytest.raises(SystemExit):
        fixture.invoke(*arguments)
    assert not fixture.generated.exists()
    assert not fixture.run_root.exists()


def test_source_must_have_joint_commands_and_consistent_provenance(continuation_launcher):
    fixture = continuation_launcher
    changed = deepcopy(fixture.saved)
    changed["config"]["model"][FLAG] = False
    changed["provenance"]["model"] = digest(changed["config"]["model"])
    torch.save(changed, fixture.checkpoint)
    with pytest.raises(SystemExit):
        fixture.invoke()
    assert not fixture.generated.exists()
    torch.save(fixture.saved, fixture.checkpoint)
    changed = deepcopy(fixture.saved)
    changed["config"]["training"]["lr"] = 0.001
    torch.save(changed, fixture.checkpoint)
    with pytest.raises(SystemExit):
        fixture.invoke()
    assert not fixture.generated.exists()


@pytest.mark.parametrize("artifact", ["tracking.json", "last.pt"])
def test_existing_branch_outputs_are_not_reused(continuation_launcher, artifact):
    fixture = continuation_launcher
    output = fixture.run_root / "control/seed0/teacher"
    output.mkdir(parents=True)
    existing = output / artifact
    existing.write_text("existing-run")
    with pytest.raises(SystemExit):
        fixture.invoke()
    assert existing.read_text() == "existing-run"
    assert not (fixture.generated / "inputs/teacher.pt").exists()


def test_submit_is_explicit_and_receipts_prevent_duplicate_jobs(continuation_launcher, monkeypatch):
    fixture = continuation_launcher
    plan = fixture.invoke()
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        assert command == ["sbatch", "--parsable", plan["jobs"][len(calls) - 1]["script"]]
        return SimpleNamespace(stdout=f"{60000 + len(calls)};fixture-cluster\n")

    monkeypatch.setattr(fixture.launcher.subprocess, "run", sbatch)
    fixture.invoke("--submit")
    receipt = json.loads((fixture.generated / "submitted.json").read_text())
    assert receipt["status"] == "submitted"
    assert [v["job_id"] for v in receipt["jobs"].values()] == ["60001", "60002"]
    assert receipt == json.loads((fixture.run_root / "continuation_submission.json").read_text())
    with pytest.raises(SystemExit):
        fixture.invoke("--submit")
    with pytest.raises(SystemExit):
        fixture.invoke("--submit", "--output-dir", str(fixture.generated.with_name("alternate")))
    assert len(calls) == 2


def test_partial_submission_failure_retains_first_job_and_blocks_duplicate(
    continuation_launcher, monkeypatch
):
    fixture = continuation_launcher
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        if len(calls) == 2:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout="60001\n")

    monkeypatch.setattr(fixture.launcher.subprocess, "run", sbatch)
    with pytest.raises(SystemExit):
        fixture.invoke("--submit")
    receipt = json.loads((fixture.generated / "submitted.json").read_text())
    assert receipt["status"] == "submission_incomplete"
    assert len(receipt["jobs"]) == 1
    assert next(iter(receipt["jobs"].values()))["job_id"] == "60001"
    with pytest.raises(SystemExit):
        fixture.invoke("--submit")
    assert len(calls) == 2


def test_concurrent_launches_with_same_run_root_cannot_duplicate(continuation_launcher, monkeypatch):
    fixture = continuation_launcher
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=5)
        return SimpleNamespace(stdout=f"{60000 + len(calls)}\n")

    def second():
        with pytest.raises(SystemExit):
            fixture.invoke("--submit", "--output-dir", str(fixture.generated.with_name("alternate")))

    monkeypatch.setattr(fixture.launcher.subprocess, "run", sbatch)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(fixture.invoke, "--submit")
        assert entered.wait(timeout=5)
        other = executor.submit(second)
        release.set()
        first.result(timeout=10)
        other.result(timeout=10)
    assert len(calls) == 2
