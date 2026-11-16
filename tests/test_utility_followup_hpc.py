"""CPU-only launch contracts; fake weights/receipts are explicitly diagnostics."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc import refinement_generated as generated
from hpc import utility_jobs as jobs
from hpc import utility_routers as routers
from stmem.config import digest


@pytest.fixture
def launch(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    parent = repo / "parent.json"
    parent.write_text("{}")
    checkpoint = repo / "student.pt"
    checkpoint.write_bytes(b"synthetic diagnostic checkpoint")
    sources = []
    for name in ("original", "current"):
        root = repo / name
        (root / "src").mkdir(parents=True)
        (root / "src/fake.py").write_text(f"# {name} diagnostic source\n")
        (root / "hpc").mkdir()
        (root / "hpc/plan.py").write_text("# diagnostic administrative file\n")
        (root / ".utility-source.json").write_text("{}")
        sources.append(root)
    site = {"STMEM_REPO_ROOT": str(repo), "STMEM_PYTHON": sys.executable,
            "WANDB_PROJECT": "must-be-overridden"}
    monkeypatch.setattr(routers, "read_site", lambda _: site)
    monkeypatch.setattr(generated, "read_site", lambda _: site)
    monkeypatch.setattr(routers, "create_study_source", lambda *_: sources[0])
    monkeypatch.setattr(generated, "create_study_source", lambda *_: sources[0])
    monkeypatch.setattr(routers, "current_source_snapshot", lambda *_: sources[1])
    monkeypatch.setattr(generated, "runtime_manifest", lambda *_: {})
    calls = []
    shell = subprocess.run

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["bash", "-n"]:
            return shell(command, **kwargs)
        if command[0] == "sbatch":
            return SimpleNamespace(stdout=str(100 + sum(c[0] == "sbatch" for c in calls)) + "\n")
        if "stmem.inspection.refinement_generated" in command or any(str(value).endswith("/planner.py") for value in command):
            mode = command[command.index("--mode") + 1]
            if mode == "plan":
                path = Path(command[command.index("--plan") + 1])
                path.parent.mkdir(parents=True, exist_ok=True)
                plan = {"plan_id": "fixture", "limited_check": "--limited-check" in command,
                        "cases": [{"state_id": "fixture-state"}],
                        "students": {s: {"checkpoint": str(checkpoint), "sha256": jobs.sha256(checkpoint)}
                                     for s in generated.STUDENTS}}
                path.write_text(json.dumps(plan))
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            return SimpleNamespace(returncode=0)
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr(subprocess, "run", run)
    args = ["--name", "pilot", "--parent-manifest", str(parent), "--checkpoint", str(checkpoint),
            "--development-step", "95000"]
    return SimpleNamespace(repo=repo, parent=parent, checkpoint=checkpoint, sources=sources,
                           calls=calls, args=args)


def test_router_dry_run_has_typed_gpu_dependency_dag_and_no_submission(launch):
    routers.main([*launch.args, "--dry-run"])
    plan = jobs.read_json(launch.repo / "hpc/generated/pilot/plan.json")
    assert plan["code_root"] == str(launch.sources[1])
    assert plan["parent_source_root"] == str(launch.sources[0])
    assert plan["jobs"]["labels"]["dependencies"] == ["prepare"]
    assert len(routers.selected_jobs("all")) == 6
    label_plan = jobs.read_json(plan["jobs"]["labels"]["plan"])
    text = Path(label_plan["script"]).read_text()
    assert "#SBATCH --gres=gpu:b200:1" in text
    assert "#SBATCH --partition=aiq" in text
    assert "#SBATCH --array=0-3%4" in text
    assert "--exclusive" not in text
    assert "WANDB_PROJECT=chronotree-routers" in text
    assert '"${SLURM_ARRAY_TASK_ID:-0}"' in text
    assert len(plan["jobs"]["fit_tree_packet"]["dependencies"]) == 1
    fit = jobs.read_json(plan["jobs"]["fit_tree_packet"]["plan"])
    assert len(fit["commands"]) == 2
    assert fit["commands"][0][-3:] == ["--mode", "train", "--resume"]
    assert fit["commands"][1][-2:] == ["--mode", "evaluate"]
    assert fit["commands"][0][fit["commands"][0].index("--device") + 1] == "cuda"
    assert all(command[0] != "sbatch" for command in launch.calls)


def test_router_submit_chains_prepare_labels_and_four_fits(launch):
    routers.main([*launch.args, "--submit"])
    submitted = [command for command in launch.calls if command[0] == "sbatch"]
    assert len(submitted) == 6
    assert not any(option.startswith("--dependency") for option in submitted[0])
    assert "--dependency=afterok:101" in submitted[1]
    assert all("--dependency=afterok:102" in command for command in submitted[2:])
    assert all("--kill-on-invalid-dep=yes" in command for command in submitted)


def test_resume_keeps_pinned_runtime_after_main_checkout_changes(launch, monkeypatch):
    routers.main([*launch.args, "--submit"])
    monkeypatch.setattr(routers, "active_user_job_ids", lambda: {str(i) for i in range(101, 107)})
    monkeypatch.setattr(routers, "current_source_snapshot", lambda *_: (_ for _ in ()).throw(
        AssertionError("Resume must retain the original snapshot instead of exporting new HEAD")))
    before = sum(c[0] == "sbatch" for c in launch.calls)
    routers.main([*launch.args, "--submit", "--resume"])
    assert sum(c[0] == "sbatch" for c in launch.calls) == before
    with pytest.raises(ValueError, match="Immutable --steps"):
        routers.main(["--name", "pilot", "--steps", "10000", "--dry-run"])


def test_late_stage_requires_complete_or_active_dependency(launch):
    routers.main([*launch.args, "--dry-run"])
    with pytest.raises(ValueError, match="requires incomplete labels"):
        routers.main(["--name", "pilot", "--stage", "train", "--submit"])
    assert not any(command[0] == "sbatch" for command in launch.calls)


def test_generated_jobs_pin_original_source_and_same_case_plan(launch):
    args = ["--name", "generated", "--manifest", str(launch.parent), "--limited-check", "--checkpoints",
            *(f"{name}={launch.checkpoint}" for name in generated.STUDENTS)]
    generated.main([*args, "--submit"])
    plan = jobs.read_json(launch.repo / "hpc/generated/generated/plan.json")
    assert plan["code_root"] == str(launch.sources[0])
    assert len(plan["jobs"]) == 3
    for student, item in plan["jobs"].items():
        job = jobs.read_json(item["plan"])
        command = job["commands"][0]
        assert command[3] == plan["case_plan"]
        assert command[4] == student
        assert "fcntl.LOCK_EX" in command[2]  # Archive's presentation writes cannot race.
        assert job["inputs"][str(launch.checkpoint)] == jobs.sha256(launch.checkpoint)
    assert len([c for c in launch.calls if c[0] == "sbatch"]) == 3
    adapters = list((launch.repo / "hpc/generated/generated/planning").glob("*/planner.py"))
    assert len(adapters) == 1
    receipts = list(adapters[0].parent.glob("*.receipt.json"))
    assert len(receipts) == 1
    receipt = jobs.read_json(receipts[0])
    assert receipt["adapter_sha256"] == jobs.sha256(adapters[0])
    assert receipt["case_plan_sha256"] == jobs.sha256(Path(plan["case_plan"]))
    assert receipt["original_code_root"] == str(launch.sources[0])
    progress = [json.loads(line) for line in
                (launch.repo / "hpc/generated/generated/preparation_progress.jsonl").read_text().splitlines()]
    assert progress[0]["phase"] == "validating_source_snapshot"
    assert sum(row["phase"] == "job_submitted_or_reused" for row in progress) == 3


def test_status_does_not_create_output_or_submit(launch, capsys):
    routers.main(["--name", "not_created", "--stage", "status"])
    generated.main(["--name", "not_created", "--stage", "status"])
    assert "not_prepared" in capsys.readouterr().out
    assert not (launch.repo / "hpc/generated/not_created").exists()
    assert launch.calls == []


def test_planning_adapter_cannot_be_modified_after_snapshotting(tmp_path):
    path, fingerprint = generated.planning_adapter(tmp_path)
    assert jobs.sha256(path) == fingerprint
    path.write_text("# altered administrative adapter\n")
    with pytest.raises(ValueError, match="Frozen planning adapter changed"):
        generated.planning_adapter(tmp_path)


def test_generated_preparation_failure_does_not_submit_jobs(launch, monkeypatch):
    original_run = subprocess.run

    def fail_planning(command, **kwargs):
        if any(str(value).endswith("/planner.py") for value in command):
            raise subprocess.CalledProcessError(1, command)
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_planning)
    with pytest.raises(subprocess.CalledProcessError):
        generated.main(["--name", "failed_plan", "--manifest", str(launch.parent),
                        "--checkpoints", *(f"{name}={launch.checkpoint}" for name in generated.STUDENTS),
                        "--limited-check", "--submit"])
    assert not any(command[0] == "sbatch" for command in launch.calls)


def test_completion_rejects_changed_student_identity(launch):
    routers.main([*launch.args, "--dry-run"])
    plan = jobs.read_json(launch.repo / "hpc/generated/pilot/plan.json")
    output = Path(plan["output"])
    output.mkdir(parents=True)
    manifest = {"schema": "utility_router_study_v1", "student": {"sha256": "wrong"}}
    manifest["experiment_id"] = digest(manifest)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="another frozen student"):
        routers.completed(plan, "prepare")


@pytest.fixture
def evaluation_fixture(launch, monkeypatch):
    routers.main([*launch.args, "--dry-run"])
    plan = jobs.read_json(launch.repo / "hpc/generated/pilot/plan.json")
    output = Path(plan["output"])
    frozen = output / "frozen/student.pt"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(launch.checkpoint.read_bytes())
    cfg = {"model": {"flow_steps": 8, "horizon": 16, "solver": "euler", "action_stride": 2, "budget": 256}}
    protocol = {"budget": 128, "flow_steps": 8, "execute_targets": 8, "fit": {"steps": 5000}}
    study = {"schema": "utility_router_study_v1", "configuration": cfg, "protocol": protocol,
             "parent_manifest": plan["parent_manifest"],
             "student": {"sha256": jobs.sha256(frozen), "checkpoint": str(frozen), "step": 95000},
             "dependencies": {"teacher_sha256": "synthetic-teacher"}}
    study["experiment_id"] = digest(study)
    (output / "manifest.json").write_text(json.dumps(study))
    states = {str(frozen): {"stage": "compact", "step": 95000, "config": cfg}}
    for reader, target in routers.VARIANTS:
        directory = routers.fit_directory(output, reader, target)
        (directory / "evaluation").mkdir(parents=True)
        checkpoint = directory / "last.pt"
        checkpoint.write_bytes(f"synthetic {reader} {target} checkpoint".encode())
        checkpoint_hash = jobs.sha256(checkpoint)
        identity = {"experiment_id": study["experiment_id"], "student_sha256": jobs.sha256(frozen),
                    "teacher_sha256": "synthetic-teacher", "reader": reader, "target": target,
                    "protocol": protocol}
        states[str(checkpoint)] = {"schema": "utility_router_checkpoint_v1", "stage": "router",
                                   "step": 5000, "identity": identity,
                                   "utility_protocol": protocol, "protocol": protocol}
        receipt = {"status": "complete", "checkpoint_sha256": checkpoint_hash, "completed_steps": 5000,
                   "intended_steps": 5000, "experiment_id": study["experiment_id"], "reader": reader,
                   "target": target}
        summary = {"status": "complete", "router_sha256": checkpoint_hash, "router_step": 5000,
                   "experiment_id": study["experiment_id"], "reader": reader, "target": target,
                   "student": study["student"]}
        (directory / "complete.json").write_text(json.dumps(receipt))
        (directory / "evaluation/summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(routers, "evaluation_study", lambda _: study)
    monkeypatch.setattr(routers, "load_checkpoint", lambda path: states[str(path)])
    commands = []

    def evaluate(command, **kwargs):
        assert command[1] == str(launch.repo / "hpc/evaluate.py")
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", evaluate)
    return SimpleNamespace(plan=plan, study=study, frozen=frozen, states=states, commands=commands, launch=launch)


def test_evaluation_uses_frozen_child_policy_and_exact_protocol(evaluation_fixture):
    fixture = evaluation_fixture
    fixture.launch.checkpoint.unlink()  # The original source is no longer required for screening.
    routers.main(["--name", "pilot", "--stage", "evaluate", "--submit"])
    assert len(fixture.commands) == 14
    assert all("--dry-run" in command for command in fixture.commands[:7])
    assert all("--submit" in command for command in fixture.commands[7:])
    for command in fixture.commands:
        assert command[command.index("--checkpoint") + 1] == str(fixture.frozen)
        assert command[command.index("--budget") + 1] == "128"
        assert command[command.index("--execute-actions") + 1] == "8"
        config = jobs.read_json(command[command.index("--config") + 1])
        assert config["model"]["flow_steps"] == 8
        assert config["model"]["action_stride"] == 2
        assert config["model"]["budget"] == 128


def test_larger_evaluation_has_fresh_names_and_preserves_all_seven_selectors(evaluation_fixture):
    """Synthetic command rendering: no GPU, checkpoint inference or Slurm calls."""
    fixture = evaluation_fixture
    original_plan = Path(fixture.plan["repo"]) / "hpc/generated/pilot/plan.json"
    original_bytes = original_plan.read_bytes()
    routers.main(["--name", "pilot", "--stage", "evaluate", "--dry-run"])
    initial_commands = list(fixture.commands)
    initial_names = {command[command.index("--name") + 1] for command in initial_commands}
    assert initial_names == {f"pilot_{variant}_packet_eval" for variant in (
        "tree_immediate", "tree_packet", "flat_immediate", "flat_packet", "tree_random", "flat_random", "uniform",
    )}
    fixture.commands.clear()
    routers.main(["--name", "pilot", "--stage", "evaluate", "--eval-name", "pilot_160ep",
                  "--episode-count", "10", "--eval-gpus", "4", "--workers-per-gpu", "4",
                  "--eval-cpus", "64", "--video-cap", "10", "--eval-walltime", "12:00:00", "--submit"])
    assert original_plan.read_bytes() == original_bytes
    assert len(fixture.commands) == 14
    assert all("--dry-run" in command for command in fixture.commands[:7])
    assert all("--submit" in command for command in fixture.commands[7:])
    new_names = {command[command.index("--name") + 1] for command in fixture.commands}
    assert len(new_names) == 7 and new_names.isdisjoint(initial_names)
    for before, after in zip(initial_commands, fixture.commands[:7]):
        assert after[after.index("--name") + 1] == before[before.index("--name") + 1].replace(
            "pilot_", "pilot_160ep_", 1)
        for flag in ("--checkpoint", "--config", "--method", "--budget", "--execute-actions"):
            assert before[before.index(flag) + 1] == after[after.index(flag) + 1]
        for flag in ("--router-checkpoint", "--router-controller", "--router-selection"):
            assert (flag in before) == (flag in after)
            if flag in before:
                assert before[before.index(flag) + 1] == after[after.index(flag) + 1]
        for flag, expected in {"--episode-count": "10", "--gpus": "4", "--workers-per-gpu": "4",
                               "--cpus": "64", "--mem": "128G", "--walltime": "12:00:00",
                               "--partition": "aiq", "--gpu-type": "b200",
                               "--wandb-project": "chronotree-routers"}.items():
            assert after[after.index(flag) + 1] == expected
        assert "--exclusive" not in after


def test_evaluation_name_cannot_rename_training(launch):
    with pytest.raises(SystemExit):
        routers.main([*launch.args, "--eval-name", "other", "--dry-run"])
    assert not (launch.repo / "hpc/generated/pilot/plan.json").exists()


@pytest.mark.parametrize("fault", ["partial", "checkpoint_changed", "wrong_credit_target", "flow_steps"])
def test_all_evaluation_inputs_verified_before_first_dry_run(evaluation_fixture, fault):
    fixture = evaluation_fixture
    directory = routers.fit_directory(fixture.plan["output"], "FLAT", "packet")
    if fault == "partial":
        (directory / "complete.json").unlink()
    elif fault == "checkpoint_changed":
        (directory / "last.pt").write_bytes(b"modified synthetic checkpoint")
    elif fault == "wrong_credit_target":
        fixture.states[str(directory / "last.pt")]["identity"]["target"] = "immediate"
    else:
        fixture.states[str(fixture.frozen)]["config"]["model"]["flow_steps"] = 4
    with pytest.raises(ValueError):
        routers.main(["--name", "pilot", "--stage", "evaluate", "--submit"])
    assert fixture.commands == []


def test_evaluation_parent_validation_uses_frozen_runtime_on_cpu(launch, monkeypatch):
    routers.main([*launch.args, "--dry-run"])
    plan = jobs.read_json(launch.repo / "hpc/generated/pilot/plan.json")
    study_path = Path(plan["study"])
    study_path.parent.mkdir(parents=True)
    study = {"student": {"sha256": plan["checkpoint_sha256"]}, "parent_manifest": plan["parent_manifest"]}
    study_path.write_text(json.dumps(study))
    verified = []
    monkeypatch.setattr(routers, "verify_study_source", lambda root, path: verified.append((root, path)))

    def read(command, **kwargs):
        assert "verify_study" in command[2]
        assert kwargs["cwd"] == launch.sources[1]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONPATH"] == str(launch.sources[1] / "src")
        return SimpleNamespace(stdout=json.dumps(study))

    monkeypatch.setattr(subprocess, "run", read)
    assert routers.evaluation_study(plan) == study
    assert verified == [(launch.sources[1], study_path)]


@pytest.mark.parametrize("flag,value", [("--gpu-type", ""), ("--partition", "aiq\n#SBATCH --exclusive"),
                                        ("--exclude", "node01; rm"), ("--exclude", "node01,,node02")])
def test_resource_injection_rejected(flag, value):
    with pytest.raises(ValueError):
        routers.main([flag, value])
