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


def test_teacher_lr_variant_changes_only_peak_lr_and_isolates_all_run_paths(planning, monkeypatch, tmp_path):
    planner, invoke, repo, run_root, generated = planning
    variant = repo / "configs/teacher_lr001.yaml"
    original_variant = variant.read_bytes()
    original_teacher = (repo / "configs/teacher.yaml").read_bytes()
    site = tmp_path / "site.env"
    original_site = site.read_bytes()
    options = ["--profile", "h200_4gpu", "--only", "seed0_teacher"]
    baseline = invoke(*options)
    baseline_job = baseline["jobs"][0]
    baseline_config = planner.resolved_config(Path(option(baseline_job["commands"][0], "--config")))
    old_checkpoint = run_root / "seed0/teacher/last.pt"
    old_checkpoint.parent.mkdir(parents=True)
    old_checkpoint.write_bytes(b"Existing low-learning-rate run")
    monkeypatch.chdir(tmp_path)

    # Relative overrides resolve from the site repository, not the caller's cwd.
    variant_root = repo / "runs/teacher lr001"
    changed = invoke(
        *options,
        "--teacher-config",
        "configs/teacher_lr001.yaml",
        "--run-root",
        "runs/teacher lr001",
    )
    job = changed["jobs"][0]
    cfg = planner.resolved_config(Path(option(job["commands"][0], "--config")))
    validate_config(cfg)
    assert changed["teacher_config"] == str(variant)
    assert changed["run_root"] == str(variant_root)
    assert cfg["training"] == {**baseline_config["training"], "lr": 0.01}
    assert cfg["training"]["warmup_steps"] == 1000
    assert cfg["training"]["steps"] == 100000
    assert cfg["training"]["effective_batch"] == 128
    assert cfg["training"]["microbatch"] == 64
    assert cfg["training"]["encoder_block_batch_size"] == 128
    assert cfg["paths"] == {**baseline_config["paths"], "run_root": str(variant_root)}
    for key in baseline_config.keys() - {"training", "paths"}:
        assert cfg[key] == baseline_config[key]
    assert job["gpus"] == 4
    assert job["signature"] != baseline_job["signature"]
    assert job["outputs"] == [
        str(variant_root / "seed0/teacher/step_00050000.pt"),
        str(variant_root / "seed0/teacher/step_00100000.pt"),
    ]
    assert job["marker"] == str(variant_root / "seed0/.stages/seed0_teacher.json")
    script = Path(job["script"]).read_text()
    assert f"export STMEM_RUN_ROOT={shlex.quote(str(variant_root))}" in script
    assert "--nproc-per-node=4" in script
    subprocess.run(["bash", "-n", job["script"]], check=True)
    assert variant.read_bytes() == original_variant
    assert (repo / "configs/teacher.yaml").read_bytes() == original_teacher
    assert site.read_bytes() == original_site
    assert old_checkpoint.read_bytes() == b"Existing low-learning-rate run"
    assert not variant_root.exists() and not (generated / "submitted.json").exists()


def test_teacher_override_reaches_validation_and_cache_without_changing_compact(planning, tmp_path):
    planner, invoke, repo, _, _ = planning
    root = tmp_path / "lr001 outputs"
    plan = invoke(
        "--teacher-config",
        str(repo / "configs/teacher_lr001.yaml"),
        "--run-root",
        str(root),
    )
    jobs = {job["name"]: job for job in plan["jobs"]}
    teacher_cfg = option(jobs["seed0_teacher"]["commands"][0], "--config")
    for name in ("seed0_teacher_val_50000", "seed0_teacher_val_100000", "seed0_cache"):
        assert option(jobs[name]["commands"][0], "--config") == teacher_cfg
    compact_cfg = Path(option(jobs["seed0_compact"]["commands"][0], "--config"))
    compact = planner.resolved_config(compact_cfg)
    assert yaml.safe_load(compact_cfg.read_text())["extends"] == str(repo / "configs/compact.yaml")
    assert compact["training"] == planner.resolved_config(repo / "configs/compact.yaml")["training"]
    assert compact["paths"]["run_root"] == str(root)
    assert jobs["seed0_cache"]["deps"] == ["seed0_select_teacher"]
    assert all(job["marker"].startswith(str(root) + "/") for job in jobs.values())


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "compact"},
        {"data": {"synthetic": True}},
        {"data": {"split": "val"}},
        {"training": {"checkpoint_steps": [50000]}},
        {"training": {"steps": 50000}},
    ],
)
def test_incompatible_teacher_override_is_rejected_before_plan_writes(planning, changes):
    _, invoke, repo, run_root, generated = planning
    variant = repo / "configs/invalid_teacher.yaml"
    variant.write_text(yaml.safe_dump({"extends": "teacher.yaml", **changes}))
    with pytest.raises(SystemExit) as error:
        invoke("--teacher-config", str(variant))
    assert error.value.code == 2
    assert not generated.exists() and not run_root.exists() and not (repo / "hpc/logs").exists()


def test_frozen_protocol_records_teacher_override_and_refuses_different_variant(planning, tmp_path):
    planner, invoke, repo, run_root, generated = planning
    variant = repo / "configs/teacher_lr001.yaml"
    manifest = yaml.safe_load((repo / "configs/experiment_core_seed0.yaml").read_text())
    for stage, filename in (
        ("teacher", "selected.pt"),
        ("compact", "selected.pt"),
        ("router_tree", "last.pt"),
        ("router_flat", "last.pt"),
        ("perceiver", "last.pt"),
    ):
        checkpoint = run_root / "seed0" / stage / filename
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"Hash-only frozen-protocol fixture; never loaded")
    destination = tmp_path / "frozen.json"
    planner.freeze_protocol(manifest, run_root, destination, repo, teacher_config=variant)
    frozen = json.loads(destination.read_text())
    assert frozen["teacher_config"] == str(variant)
    for path in (variant, repo / "configs/teacher.yaml", repo / "configs/base.yaml"):
        assert frozen["configuration_hashes"][str(path)] == planner.sha256(path)

    # Even a byte-identical alternate path is a different declared variant.
    other_variant = repo / "configs/other_teacher.yaml"
    other_variant.write_bytes(variant.read_bytes())
    for extra in ([], ["--teacher-config", str(other_variant)]):
        with pytest.raises(SystemExit) as error:
            invoke("--phase", "test", "--freeze-manifest", str(destination), *extra)
        assert error.value.code == 2
        assert not generated.exists() and not (repo / "hpc/logs").exists()

    plan = invoke(
        "--phase",
        "test",
        "--freeze-manifest",
        str(destination),
        "--teacher-config",
        str(variant),
    )
    assert plan["phase"] == "test" and plan["teacher_config"] == str(variant)
    teacher_cfg = planner.resolved_config(generated / "configs/seed0_teacher.yaml")
    assert teacher_cfg["training"]["lr"] == 0.01
    assert teacher_cfg["evaluation"]["test_frozen_manifest"] == str(destination)

    variant.write_text(variant.read_text() + "\n# Changed after the protocol was frozen.\n")
    stale_output = tmp_path / "stale protocol output"
    with pytest.raises(SystemExit) as error:
        invoke(
            "--phase",
            "test",
            "--freeze-manifest",
            str(destination),
            "--teacher-config",
            str(variant),
            "--output-dir",
            str(stale_output),
        )
    assert error.value.code == 2
    assert not stale_output.exists()
