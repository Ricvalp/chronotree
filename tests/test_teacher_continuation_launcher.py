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


@pytest.fixture
def interrupted_pair(continuation_launcher):
    """Synthetic checkpoint metadata for two interrupted, already forked jobs."""
    fixture = continuation_launcher
    plan = fixture.invoke("--additional-steps", "15000")
    branches = {}
    for branch, enabled in fixture.launcher.BRANCHES.items():
        cfg = yaml.safe_load(Path(plan["configs"][branch]).read_text())
        checkpoint = fixture.run_root / branch / "seed0/teacher/last.pt"
        checkpoint.parent.mkdir(parents=True)
        saved = deepcopy(fixture.saved)
        saved.update(config=cfg, step=54000)
        saved["provenance"].update(model=digest(cfg["model"]), training=digest(cfg["training"]))
        saved["sampler"]["next_global_position"] = 54000 * 128
        saved["fork"] = {
            "source_checkpoint": plan["snapshot"],
            "source_sha256": plan["source_sha256"],
            "source_step": 49978,
            "source_tracking_id": fixture.saved["tracking"]["id"],
            "changes": {FLAG: {"before": True, "after": enabled}},
            "schedule_steps": 100000,
            "intended_final_step": 64978,
        }
        saved["tracking"] = {
            "id": f"saved-{branch}",
            "project": "chronotree",
            "entity": None,
            "job_type": "teacher",
            "attempt": 1,
        }
        torch.save(saved, checkpoint)
        tracking = checkpoint.with_name("tracking.json")
        tracking.write_text(json.dumps({**saved["tracking"], "attempt": 2, "next_history_step": 1234}))
        branches[branch] = SimpleNamespace(saved=saved, checkpoint=checkpoint, tracking=tracking)
    receipt = {
        "status": "submitted",
        "generated": str(fixture.generated),
        "submitted_utc": "2026-10-26T14:45:00+00:00",
        "jobs": {
            job["name"]: {
                "job_id": str(60797 + index),
                "script": job["script"],
                "signature": job["signature"],
            }
            for index, job in enumerate(plan["jobs"])
        },
    }
    (fixture.run_root / "continuation_submission.json").write_text(json.dumps(receipt))
    (fixture.generated / "submitted.json").write_text(json.dumps(receipt))

    def resume(*arguments):
        fixture.launcher.main(
            [
                "--site",
                str(fixture.site),
                "--profile",
                "b200_4gpu",
                "--run-root",
                str(fixture.run_root),
                "--output-dir",
                str(fixture.generated),
                "--resume",
                *arguments,
            ]
        )
        return json.loads((fixture.generated / "resume/plan.json").read_text())

    return SimpleNamespace(fixture=fixture, plan=plan, branches=branches, receipt=receipt, resume=resume)


def test_resume_keeps_branch_state_endpoint_tracking_and_original_plan(interrupted_pair, monkeypatch):
    pair = interrupted_pair
    fixture = pair.fixture
    original = {
        path: path.read_bytes()
        for path in [
            fixture.generated / "plan.json",
            fixture.generated / "submitted.json",
            *[Path(path) for path in pair.plan["configs"].values()],
            *[Path(job["script"]) for job in pair.plan["jobs"]],
            *[branch.tracking for branch in pair.branches.values()],
            *[branch.checkpoint for branch in pair.branches.values()],
        ]
    }
    with monkeypatch.context() as scoped:
        scoped.setattr(
            fixture.launcher.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry-run queried Slurm")
        )
        plan = pair.resume("--dry-run")
    assert plan["resume_existing_runs"] is True
    assert plan["source_step"] == 49978 and plan["end_step"] == 64978
    assert plan["additional_steps"] == 15000 and plan["schedule_steps"] == 100000
    assert plan["maximum_total_gpus"] == 8 and plan["job_count"] == 2
    for branch, job in zip(fixture.launcher.BRANCHES, plan["jobs"], strict=True):
        saved = pair.branches[branch].saved
        assert yaml.safe_load(Path(plan["configs"][branch]).read_text()) == saved["config"]
        progress = plan["resume_progress"][branch]
        assert progress["checkpoint_step"] == 54000 and progress["remaining_steps"] == 10978
        assert progress["wandb_run_id"] == saved["tracking"]["id"]
        command = job["commands"][0]
        assert command[command.index("--resume") + 1] == str(pair.branches[branch].checkpoint)
        assert "--fork-from" not in command and "--fork-sha256" not in command
        assert job["inputs"] == [str(pair.branches[branch].checkpoint)]
        assert job["outputs"] == [str(pair.branches[branch].checkpoint.with_name("step_00064978.pt"))]
        script = Path(job["script"]).read_text()
        assert "--time=48:00:00" in script and "--gres=gpu:4" in script
        assert "--partition=aiq" in script and "--nproc-per-node=4" in script
        subprocess.run(["bash", "-n", job["script"]], check=True)
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize(
    "problem",
    ["missing", "config", "lineage", "world", "sampler", "tracking", "tracking_project", "budget", "flag"],
)
def test_resume_rejects_wrong_branch_checkpoints_before_rendering(interrupted_pair, problem):
    pair = interrupted_pair
    item = pair.branches["no_previous_joint_commands"]
    if problem == "missing":
        item.checkpoint.unlink()
    elif problem == "tracking":
        item.tracking.write_text(json.dumps({**item.saved["tracking"], "id": "another-run"}))
    elif problem == "tracking_project":
        item.tracking.write_text(json.dumps({**item.saved["tracking"], "project": "another-project"}))
    else:
        saved = deepcopy(item.saved)
        if problem == "config":
            saved["config"]["training"]["lr"] = 0.001
        elif problem == "lineage":
            saved["fork"]["source_sha256"] = "wrong-parent"
        elif problem == "world":
            saved["provenance"]["world_size"] = 8
        elif problem == "sampler":
            saved["sampler"]["next_global_position"] += 1
        elif problem == "budget":
            saved["step"] = 64978
        elif problem == "flag":
            saved["config"]["model"][FLAG] = True
        torch.save(saved, item.checkpoint)
    with pytest.raises(SystemExit):
        pair.resume()
    assert not (pair.fixture.generated / "resume").exists()


@pytest.mark.parametrize("arguments", [["--additional-steps", "15000"], ["--profile", "b200_1gpu"]])
def test_resume_cannot_extend_budget_or_change_world(interrupted_pair, arguments):
    with pytest.raises(SystemExit):
        interrupted_pair.resume(*arguments)


def test_resume_submit_checks_queue_and_retains_prior_receipts(interrupted_pair, monkeypatch):
    pair = interrupted_pair
    calls, active = [], "99999|RUNNING\n"  # An unrelated active job does not prevent resumption.

    def slurm(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            assert "--user" in command and "--jobs" not in command
            return SimpleNamespace(stdout=active)
        return SimpleNamespace(stdout=f"{60800 + sum(call[0] == 'sbatch' for call in calls)}\n")

    monkeypatch.setattr(pair.fixture.launcher.subprocess, "run", slurm)
    plan = pair.resume("--submit")
    assert [call[0] for call in calls] == ["squeue", "sbatch", "sbatch"]
    path = pair.fixture.run_root / "continuation_submission.json"
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "submitted" and receipt["resume_existing_runs"] is True
    assert receipt["history"] == [pair.receipt]
    assert sorted(value["job_id"] for value in receipt["jobs"].values()) == ["60801", "60802"]
    assert receipt == json.loads((pair.fixture.generated / "submitted.json").read_text())
    assert {value["script"] for value in receipt["jobs"].values()} == {job["script"] for job in plan["jobs"]}
    active = "60801|RUNNING\n60802|PENDING\n"
    with pytest.raises(SystemExit):
        pair.resume("--submit")
    assert [call[0] for call in calls] == ["squeue", "sbatch", "sbatch", "squeue"]
    assert json.loads(path.read_text()) == receipt


@pytest.mark.parametrize("failure", ["active", "query_error", "ambiguous"])
def test_resume_blocks_active_or_unverifiable_submissions(interrupted_pair, monkeypatch, failure):
    pair = interrupted_pair
    if failure == "ambiguous":
        receipt = {**pair.receipt, "status": "submission_incomplete"}
        for path in [
            pair.fixture.generated / "submitted.json",
            pair.fixture.run_root / "continuation_submission.json",
        ]:
            path.write_text(json.dumps(receipt))

    def slurm(command, **kwargs):
        assert command[0] == "squeue", "Unsafe duplicate submitted"
        if failure == "query_error":
            raise subprocess.CalledProcessError(1, command)
        assert failure != "ambiguous", "Ambiguous submission should fail without querying"
        return SimpleNamespace(stdout="60797|RUNNING\n")

    monkeypatch.setattr(pair.fixture.launcher.subprocess, "run", slurm)
    with pytest.raises(SystemExit):
        pair.resume("--submit")
    assert not (pair.fixture.generated / "resume").exists()


def test_resume_partial_submission_keeps_history_and_prevents_retry(interrupted_pair, monkeypatch):
    pair = interrupted_pair
    calls = []

    def slurm(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return SimpleNamespace(stdout="")
        if len(calls) == 3:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout="60801\n")

    monkeypatch.setattr(pair.fixture.launcher.subprocess, "run", slurm)
    with pytest.raises(SystemExit):
        pair.resume("--submit")
    receipt = json.loads((pair.fixture.generated / "submitted.json").read_text())
    assert receipt["status"] == "submission_incomplete" and receipt["history"] == [pair.receipt]
    assert len(receipt["jobs"]) == 1
    with pytest.raises(SystemExit):
        pair.resume("--submit")
    assert len(calls) == 3
