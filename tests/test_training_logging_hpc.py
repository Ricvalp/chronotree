"""Runtime training telemetry survives planning without changing stage identity."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from stmem.config import validate_config


@pytest.fixture
def planning(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith(("STMEM_", "WANDB_")):
            monkeypatch.delenv(key)
    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("training_logging_hpc_planner", project / "hpc/plan.py")
    planner = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, planner)
    spec.loader.exec_module(planner)
    repo = tmp_path / "isolated repo"
    shutil.copytree(project / "configs", repo / "configs")
    (repo / "src").mkdir()
    (repo / "src/identity.py").write_text('"""Stable fixture identity; never executed."""\n')
    run_root = tmp_path / "runs"
    generated = tmp_path / "generated"
    site = tmp_path / "site.env"
    site.write_text(
        "\n".join(
            f"{key}={shlex.quote(str(value))}"
            for key, value in {
                "STMEM_REPO_ROOT": repo,
                "STMEM_RUN_ROOT": run_root,
                "STMEM_PYTHON": sys.executable,
                "STMEM_PROFILE": "h200_1gpu",
            }.items()
        )
        + "\n"
    )

    def invoke(*options):
        def forbidden_subprocess(*args, **kwargs):
            pytest.fail(f"Dry-run must not execute external commands: {args}")

        with monkeypatch.context() as scoped:
            scoped.setattr(planner.subprocess, "run", forbidden_subprocess)
            scoped.setattr(
                sys,
                "argv",
                [
                    str(project / "hpc/plan.py"),
                    "--dry-run",
                    "--site",
                    str(site),
                    "--output-dir",
                    str(generated),
                    *options,
                ],
            )
            planner.main()
        return json.loads((generated / "plan.json").read_text())

    return planner, invoke, repo, run_root, generated


def option(command, name):
    return command[command.index(name) + 1]


@pytest.mark.parametrize("plan_name", ("core_seed0", "central_3seed"))
def test_teacher_plan_matches_training_budget_and_isolates_site_run_root(planning, tmp_path, plan_name):
    planner, invoke, repo, run_root, generated = planning
    site_path = tmp_path / "site.env"
    site_path.write_text(
        site_path.read_text() + "STMEM_CPUS=64\nWANDB_MODE=online\nWANDB_PROJECT=chronotree\n"
    )
    old_run = tmp_path / "previous experiment/seed0/teacher/last.pt"
    old_run.parent.mkdir(parents=True)
    old_run.write_bytes(b"Existing training artifacts must remain untouched")
    original_config = (repo / "configs/teacher.yaml").read_bytes()
    original_site = site_path.read_bytes()
    plan = invoke(
        "--plan", plan_name, "--profile", "h200_4gpu", "--only", "seed0_teacher", "--log-every", "1"
    )
    assert plan["job_count"] == plan["maximum_concurrent_allocations"] == 1
    assert plan["remaining_stage_work"] == ["seed0_teacher"]
    job = plan["jobs"][0]
    assert job["name"] == "seed0_teacher" and job["stage"] == "teacher"
    assert job["gpus"] == 4 and not job["deps"] and not job["inputs"]
    cfg_path = Path(option(job["commands"][0], "--config"))
    assert cfg_path == generated / "configs/seed0_teacher.yaml"
    cfg = planner.resolved_config(cfg_path)
    validate_config(cfg)
    assert cfg["training"]["steps"] == 100000
    assert cfg["training"]["effective_batch"] == 128
    assert cfg["training"]["microbatch"] == 64
    assert cfg["training"]["encoder_block_batch_size"] == 128
    assert cfg["paths"]["run_root"] == str(run_root)
    assert cfg["training"]["effective_batch"] // job["gpus"] == 32
    manifest = yaml.safe_load((repo / "configs" / f"experiment_{plan_name}.yaml").read_text())
    candidates = manifest["teacher_candidates"]
    assert candidates == cfg["training"]["checkpoint_steps"] == [50000, 100000]
    assert candidates[-1] == cfg["training"]["steps"]
    expected_outputs = [str(run_root / "seed0/teacher" / f"step_{step:08d}.pt") for step in candidates]
    assert job["outputs"] == expected_outputs
    script = Path(job["script"]).read_text()
    assert Path(job["script"]) == generated / "jobs/seed0_teacher.sbatch"
    assert "#SBATCH --gres=gpu:4" in script and "#SBATCH --cpus-per-task=64" in script
    assert "--nproc-per-node=4" in script and "stmem.train --stage teacher" in script
    assert f"export STMEM_RUN_ROOT={shlex.quote(str(run_root))}" in script
    assert "export WANDB_MODE=online" in script and "export WANDB_PROJECT=chronotree" in script
    assert "B:USR1@180" in script and "--stop-file" in script and "--log-every 1" in script
    subprocess.run(["bash", "-n", job["script"]], check=True)
    assert (repo / "configs/teacher.yaml").read_bytes() == original_config
    assert site_path.read_bytes() == original_site
    assert old_run.read_bytes() == b"Existing training artifacts must remain untouched"
    assert not run_root.exists() and not (generated / "submitted.json").exists()

    # Every seed's selection stage must evaluate precisely the checkpoints its
    # teacher will actually produce; downstream cache use still requires selection.
    full_plan = invoke("--plan", plan_name, "--profile", "h200_4gpu")
    jobs = {item["name"]: item for item in full_plan["jobs"]}
    for seed in manifest["seeds"]:
        teacher = jobs[f"seed{seed}_teacher"]
        select = jobs[f"seed{seed}_select_teacher"]
        assert select["deps"] == [f"seed{seed}_teacher_val_{step}" for step in candidates]
        for step, output in zip(candidates, teacher["outputs"], strict=True):
            validation = jobs[f"seed{seed}_teacher_val_{step}"]
            assert option(validation["commands"][0], "--checkpoint") == output
            assert validation["deps"] == [teacher["name"]]
        assert jobs[f"seed{seed}_cache"]["deps"] == [select["name"]]


def test_single_teacher_submission_and_resume_use_established_planner(planning, monkeypatch, tmp_path):
    planner, invoke, repo, run_root, generated = planning
    options = ["--plan", "core_seed0", "--profile", "h200_4gpu", "--only", "seed0_teacher"]
    dry_run = invoke(*options)
    job = dry_run["jobs"][0]
    calls = []

    def fake_scheduler(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            assert command == ["squeue", "-h", "-j", "12345", "-o", "%T"]
            return SimpleNamespace(stdout="RUNNING\n")
        assert command == ["sbatch", "--parsable", job["script"]]
        assert (repo / "hpc/logs").is_dir()
        return SimpleNamespace(stdout="12345;fixture-cluster\n")

    def submit(*extra):
        monkeypatch.setattr(planner.subprocess, "run", fake_scheduler)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hpc/plan.py",
                "--site",
                str(tmp_path / "site.env"),
                "--output-dir",
                str(generated),
                *options,
                "--submit",
                *extra,
            ],
        )
        planner.main()

    submit()
    receipt = json.loads((generated / "submitted.json").read_text())
    assert set(receipt) == {"seed0_teacher"}
    assert receipt["seed0_teacher"]["job_id"] == "12345"
    assert receipt["seed0_teacher"]["signature"] == job["signature"]
    assert len(calls) == 1
    checkpoint = run_root / "seed0/teacher/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"Presence-only checkpoint fixture; never loaded")
    submit("--resume")
    resumed = json.loads((generated / "plan.json").read_text())["jobs"][0]
    assert option(resumed["commands"][0], "--resume") == str(checkpoint)
    assert resumed["signature"] == job["signature"]
    assert [command[0] for command in calls] == ["sbatch", "squeue"]


def test_dry_run_forwards_runtime_flags_without_changing_configs_or_signatures(planning):
    _, invoke, _, _, generated = planning
    initial = invoke()
    original_configs = {path.name: path.read_bytes() for path in (generated / "configs").glob("*.yaml")}
    original_jobs = {job["name"]: job for job in initial["jobs"]}
    for job in initial["jobs"]:
        for command in job["commands"]:
            assert "--log-every" not in command and "--progress-every-seconds" not in command
    changed = invoke("--log-every", "1", "--progress-every-seconds", "2.5")
    assert {
        path.name: path.read_bytes() for path in (generated / "configs").glob("*.yaml")
    } == original_configs
    assert {job["stage"] for job in changed["jobs"]} >= {"teacher", "compact", "perceiver", "router"}
    for job in changed["jobs"]:
        assert job["signature"] == original_jobs[job["name"]]["signature"]
        for command in job["commands"]:
            if job["stage"] in ("teacher", "compact", "perceiver", "router"):
                assert option(command, "--log-every") == "1"
                if job["stage"] == "router":
                    assert "--progress-every-seconds" not in command
                else:
                    assert option(command, "--progress-every-seconds") == "2.5"
                script = Path(job["script"]).read_text()
                assert "--log-every 1" in script
                assert ("--progress-every-seconds 2.5" in script) == (job["stage"] != "router")
                subprocess.run(["bash", "-n", job["script"]], check=True)
            else:
                assert "--log-every" not in command and "--progress-every-seconds" not in command
    assert not (generated / "submitted.json").exists()


def test_resume_second_render_retains_runtime_flags_and_original_stage_identity(planning):
    _, invoke, _, run_root, generated = planning
    before = invoke("--only", "seed0_teacher", "seed0_router_tree_final")
    original_jobs = {job["name"]: job for job in before["jobs"]}
    for stage in ("teacher", "router_tree"):
        checkpoint = run_root / "seed0" / stage / "last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"Presence-only checkpoint fixture; never loaded")
    resumed = invoke(
        "--only",
        "seed0_teacher",
        "seed0_router_tree_final",
        "--resume",
        "--log-every",
        "3",
        "--progress-every-seconds",
        "0.25",
    )
    assert len(resumed["jobs"]) == 2
    for job in resumed["jobs"]:
        stage = "teacher" if job["stage"] == "teacher" else "router_tree"
        checkpoint = run_root / "seed0" / stage / "last.pt"
        command = job["commands"][0]
        assert command.count("--resume") == 1
        assert option(command, "--resume") == str(checkpoint)
        assert option(command, "--log-every") == "3"
        assert ("--progress-every-seconds" in command) == (stage == "teacher")
        assert job["signature"] == original_jobs[job["name"]]["signature"]
        script = Path(job["script"]).read_text()
        assert f"--signature {job['signature']}" in script
        assert "--log-every 3" in script and f"--resume {checkpoint}" in script
        assert ("--progress-every-seconds 0.25" in script) == (stage == "teacher")
        subprocess.run(["bash", "-n", job["script"]], check=True)
    assert not (generated / "submitted.json").exists()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--log-every", "0"),
        ("--log-every", "-1"),
        ("--log-every", "1.5"),
        ("--log-every", "nan"),
        ("--progress-every-seconds", "0"),
        ("--progress-every-seconds", "-0.5"),
        ("--progress-every-seconds", "nan"),
        ("--progress-every-seconds", "inf"),
        ("--progress-every-seconds", "1e309"),
        ("--progress-every-seconds", "nope"),
    ],
)
def test_invalid_runtime_flags_fail_before_any_plan_output(planning, flag, value):
    _, invoke, repo, run_root, generated = planning
    with pytest.raises(SystemExit) as error:
        invoke(flag, value)
    assert error.value.code == 2
    assert not generated.exists() and not run_root.exists() and not (repo / "hpc/logs").exists()


def test_signature_normalization_removes_only_valid_exact_training_pairs(planning):
    planner, _, _, _, _ = planning
    command = [sys.executable, "-m", "stmem.train", "--stage", "teacher"]
    runtime = ["--log-every", "5", "--progress-every-seconds", "1.5"]
    assert planner.scientific_training_command(command + runtime) == command
    assert planner.scientific_training_command(command + ["--log-every", "--config", "x"]) == (
        command + ["--log-every", "--config", "x"]
    )
    assert planner.scientific_training_command(command + ["--log-every", "0"]) == (
        command + ["--log-every", "0"]
    )
    scientific = ["--config", "/tmp/--log-every.yaml", "--stage", "perceiver"]
    assert planner.scientific_training_command(command + scientific) == command + scientific
    other = [sys.executable, "-m", "stmem.evaluate", *runtime]
    assert planner.scientific_training_command(other) == other
