"""Runtime training telemetry survives planning without changing stage identity."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


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
