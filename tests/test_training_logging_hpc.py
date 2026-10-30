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


def test_teacher_until_stopped_preserves_saved_config_and_reopens_completed_stage(planning, tmp_path):
    planner, invoke, repo, run_root, generated = planning
    site = tmp_path / "site.env"
    site.write_text(
        site.read_text()
        + "STMEM_PARTITION=aiq\nSTMEM_GPU_TYPE=\nSTMEM_CPUS=64\nSTMEM_MEM=128G\n"
        + "STMEM_WALLTIME=48:00:00\n"
    )
    config = planner.resolved_config(
        repo / "configs/teacher_b200_b128_m8_blocks32_no_previous_joint_commands_h32.yaml"
    )
    # Match the historical run: no new model defaults may be injected on resume.
    config["model"].pop("history_mode")
    snapshot = tmp_path / "resume_teacher.json"
    snapshot.write_text(json.dumps(config))
    original_snapshot = snapshot.read_bytes()
    options = [
        "--teacher-config", str(snapshot), "--only", "seed0_teacher", "--profile", "b200_4gpu",
    ]
    original = invoke(*options)["jobs"][0]
    for path in original["outputs"]:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"Immutable candidate fixture; never loaded")
    checkpoint = run_root / "seed0/teacher/last.pt"
    checkpoint.write_bytes(b"Presence-only final checkpoint fixture; never loaded")
    marker = Path(original["marker"])
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({
        "status": "complete", "signature": original["signature"],
        "output_sha256": {path: planner.sha256(Path(path)) for path in original["outputs"]},
    }))
    marker_before = marker.read_bytes()
    assert invoke(*options, "--resume")["completed_skipped"] == ["seed0_teacher"]

    plan = invoke(*options, "--resume", "--train-until-stopped", "--attention-every", "10")
    assert plan["job_count"] == 1 and plan["completed_skipped"] == []
    job = plan["jobs"][0]
    command = job["commands"][0]
    assert command.count("--train-until-stopped") == 1
    assert option(command, "--resume") == str(checkpoint)
    assert job["signature"] != original["signature"]
    assert job["outputs"] == original["outputs"] and job["deps"] == []
    restored = planner.resolved_config(Path(option(command, "--config")))
    for key in ("model", "training", "data", "geometry"):
        assert restored[key] == config[key]
    assert "history_mode" not in restored["model"]
    assert restored["training"]["steps"] == 100000
    assert plan["training_continuation"] == {
        "mode": "until_stopped", "original_intended_steps": 100000,
        "schedule_steps": 100000, "final_learning_rate": 0.00001,
        "automatic_resubmission": False,
    }
    script = Path(job["script"]).read_text()
    assert "#SBATCH --gres=gpu:4" in script and "gpu:b200" not in script
    assert "#SBATCH --partition=aiq" in script and "#SBATCH --time=48:00:00" in script
    assert "--train-until-stopped" in script and f"--resume {checkpoint}" in script
    assert "#SBATCH --signal=B:USR1@180" in script
    assert marker.read_bytes() == marker_before and snapshot.read_bytes() == original_snapshot
    assert checkpoint.read_bytes() == b"Presence-only final checkpoint fixture; never loaded"
    subprocess.run(["bash", "-n", job["script"]], check=True)
    again = invoke(*options, "--resume", "--train-until-stopped")
    assert again["jobs"][0]["signature"] == job["signature"]


@pytest.mark.parametrize("options", [
    ("--only", "seed0_teacher"),
    ("--resume",),
    ("--resume", "--only", "seed0_compact"),
    ("--resume", "--only", "seed0_teacher", "evaluate"),
    ("--resume", "--only", "seed0_teacher", "--phase", "test"),
    ("--resume", "--only", "seed0_teacher", "--freeze-protocol", "unused.json"),
])
def test_teacher_until_stopped_rejects_fresh_or_downstream_plans(planning, options, capsys):
    _, invoke, _, _, generated = planning
    with pytest.raises(SystemExit) as error:
        invoke(*options, "--train-until-stopped")
    assert error.value.code == 2
    assert "--train-until-stopped requires --resume and --only teacher jobs" in capsys.readouterr().err
    assert not generated.exists()


@pytest.mark.parametrize("empty_file", (False, True))
def test_teacher_until_stopped_requires_existing_nonempty_checkpoint(planning, empty_file, capsys):
    _, invoke, _, run_root, generated = planning
    if empty_file:
        checkpoint = run_root / "seed0/teacher/last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
    with pytest.raises(SystemExit) as error:
        invoke("--resume", "--only", "seed0_teacher", "--train-until-stopped")
    assert error.value.code == 2
    assert "needs an existing checkpoint" in capsys.readouterr().err
    assert not generated.exists()


@pytest.mark.parametrize("queue_output", ("12345|RUNNING\n", ""))
def test_teacher_until_stopped_submission_preserves_active_job_protection(
    planning, monkeypatch, tmp_path, queue_output,
):
    planner, invoke, _, run_root, generated = planning
    original = invoke("--only", "seed0_teacher")["jobs"][0]
    receipt = {"seed0_teacher": {"job_id": "12345", "signature": original["signature"]}}
    (generated / "submitted.json").write_text(json.dumps(receipt))
    checkpoint = run_root / "seed0/teacher/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"Presence-only checkpoint fixture; never loaded")
    calls = []

    def scheduler(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return SimpleNamespace(stdout=queue_output)
        assert command == ["sbatch", "--parsable", original["script"]]
        return SimpleNamespace(stdout="23456\n")

    monkeypatch.setattr(planner.subprocess, "run", scheduler)
    monkeypatch.setattr(sys, "argv", [
        "hpc/plan.py", "--site", str(tmp_path / "site.env"),
        "--output-dir", str(generated), "--only", "seed0_teacher", "--resume",
        "--train-until-stopped", "--submit",
    ])
    planner.main()
    updated = json.loads((generated / "submitted.json").read_text())
    if queue_output:
        assert [command[0] for command in calls] == ["squeue"]
        assert updated == receipt
    else:
        assert [command[0] for command in calls] == ["squeue", "sbatch"]
        assert updated["seed0_teacher"]["job_id"] == "23456"
    planned = json.loads((generated / "plan.json").read_text())["jobs"][0]
    assert "--train-until-stopped" in planned["commands"][0]
    assert option(planned["commands"][0], "--resume") == str(checkpoint)


def test_current_only_teacher_renders_two_h200s_without_affecting_other_runs(planning, tmp_path):
    planner, invoke, repo, run_root, generated = planning
    site = tmp_path / "site.env"
    site.write_text(
        site.read_text()
        + "STMEM_PARTITION=gpuq\nSTMEM_GPU_TYPE=\nSTMEM_CPUS=32\nSTMEM_MEM=128G\n"
        + "STMEM_WALLTIME=48:00:00\nWANDB_MODE=online\nWANDB_PROJECT=chronotree\n"
    )
    config = repo / "configs/teacher_current_only_fv_no_prop_10hz.yaml"
    original_config = config.read_bytes()
    plan = invoke(
        "--plan", "core_seed0", "--profile", "h200_2gpu", "--only", "seed0_teacher",
        "--teacher-config", str(config), "--log-every", "1", "--attention-every", "10",
    )
    assert plan["job_count"] == 1 and plan["profile"] == "h200_2gpu"
    job = plan["jobs"][0]
    assert job["gpus"] == 2 and job["deps"] == [] and job["inputs"] == []
    assert job["walltime"] == "48:00:00"
    cfg = planner.resolved_config(Path(option(job["commands"][0], "--config")))
    validate_config(cfg)
    assert cfg["model"]["history_mode"] == cfg["model"]["proprioception_mode"] == "none"
    assert cfg["model"]["condition_on_previous_commands"] is False
    assert cfg["training"]["effective_batch"] == 128
    assert cfg["training"]["microbatch"] == 64
    assert cfg["training"]["microbatch"] * job["gpus"] == cfg["training"]["effective_batch"]
    assert cfg["paths"]["run_root"] == str(run_root)
    assert option(job["commands"][0], "--attention-every") == "10"
    script = Path(job["script"]).read_text()
    assert "#SBATCH --gres=gpu:2" in script and "gpu:h200" not in script
    assert "#SBATCH --partition=gpuq" in script and "#SBATCH --cpus-per-task=32" in script
    assert "#SBATCH --ntasks=1" in script and script.count("--nproc-per-node=2") == 1
    assert "export WANDB_MODE=online" in script
    assert config.read_bytes() == original_config
    assert not run_root.exists() and not (generated / "submitted.json").exists()
    subprocess.run(["bash", "-n", job["script"]], check=True)


def test_current_only_teacher_cannot_launch_full_memory_pipeline(planning):
    _, invoke, repo, _, _ = planning
    config = repo / "configs/teacher_current_only_fv_no_prop_10hz.yaml"
    for extra in ((), ("--only", "seed0_compact")):
        with pytest.raises(SystemExit) as error:
            invoke("--teacher-config", str(config), *extra)
        assert error.value.code == 2
    plan = invoke("--teacher-config", str(config), "--only", "seed0_teacher_val_50000", "seed0_select_teacher")
    assert {job["name"] for job in plan["jobs"]} == {"seed0_teacher_val_50000", "seed0_select_teacher"}


@pytest.mark.parametrize("plan_name", ("core_seed0", "central_3seed"))
@pytest.mark.parametrize("profile, gpu_type", (("h200_4gpu", ""), ("b200_4gpu", "customgpu")))
def test_teacher_plan_matches_training_budget_and_isolates_site_run_root(
    planning, tmp_path, plan_name, profile, gpu_type,
):
    planner, invoke, repo, run_root, generated = planning
    site_path = tmp_path / "site.env"
    site_path.write_text(
        site_path.read_text()
        + f"STMEM_CPUS=64\nSTMEM_GPU_TYPE={gpu_type}\nWANDB_MODE=online\nWANDB_PROJECT=chronotree\n"
    )
    old_run = tmp_path / "previous experiment/seed0/teacher/last.pt"
    old_run.parent.mkdir(parents=True)
    old_run.write_bytes(b"Existing training artifacts must remain untouched")
    original_config = (repo / "configs/teacher.yaml").read_bytes()
    original_site = site_path.read_bytes()
    plan = invoke(
        "--plan", plan_name, "--profile", profile, "--only", "seed0_teacher", "--log-every", "1"
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
    assert cfg["training"]["effective_batch"] == 256
    assert cfg["training"]["microbatch"] == 64
    assert cfg["training"]["encoder_block_batch_size"] == 128
    assert cfg["profile"] == profile
    assert cfg["paths"]["run_root"] == str(run_root)
    assert cfg["training"]["effective_batch"] // job["gpus"] == 64
    assert cfg["data"]["prepared_subdir"] == "prepared_balanced_v2"
    assert cfg["model"]["points"] == 2048
    assert cfg["model"]["visual_tokens"] // cfg["model"]["block_size"] == 32
    assert cfg["model"]["current_tokens"] == 32
    assert cfg["model"]["encoder_options"]["center_sampling"]["method"] == "balanced_cluster"
    manifest = yaml.safe_load((repo / "configs" / f"experiment_{plan_name}.yaml").read_text())
    candidates = manifest["teacher_candidates"]
    assert candidates == cfg["training"]["checkpoint_steps"] == [50000, 100000]
    assert candidates[-1] == cfg["training"]["steps"]
    expected_outputs = [str(run_root / "seed0/teacher" / f"step_{step:08d}.pt") for step in candidates]
    assert job["outputs"] == expected_outputs
    script = Path(job["script"]).read_text()
    assert Path(job["script"]) == generated / "jobs/seed0_teacher.sbatch"
    gres = f"gpu:{gpu_type + ':' if gpu_type else ''}4"
    assert f"#SBATCH --gres={gres}" in script and "#SBATCH --cpus-per-task=64" in script
    assert "#SBATCH --ntasks=1" in script and "srun --ntasks=1" in script
    assert script.count("--nproc-per-node=4") == 1 and "stmem.train --stage teacher" in script
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
    full_plan = invoke("--plan", plan_name, "--profile", profile)
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
            assert command == [
                "squeue", "--noheader", "--all", "--states=all",
                "--user", str(os.getuid()), "--format=%A|%T",
            ]
            return SimpleNamespace(stdout="12345|RUNNING\n")
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


@pytest.mark.parametrize("queue_output", ["", "99999|RUNNING\n", "60793|FAILED\n"])
def test_resume_resubmits_expired_job_id_without_querying_that_id(
    planning, monkeypatch, tmp_path, queue_output,
):
    planner, invoke, _, run_root, generated = planning
    original = invoke("--only", "seed0_teacher")["jobs"][0]
    receipt = {"seed0_teacher": {"job_id": "60793", "signature": original["signature"]}}
    (generated / "submitted.json").write_text(json.dumps(receipt))
    checkpoint = run_root / "seed0/teacher/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"Presence-only checkpoint fixture; never loaded")
    calls = []
    monkeypatch.setenv("SQUEUE_PARTITION", "unrelated-partition")

    def scheduler(command, **kwargs):
        calls.append(command)
        assert kwargs["check"] is True
        if command[0] == "squeue":
            assert "60793" not in command and "--user" in command
            assert "SQUEUE_PARTITION" not in kwargs["env"]
            return SimpleNamespace(stdout=queue_output)
        assert command == ["sbatch", "--parsable", original["script"]]
        return SimpleNamespace(stdout="61234\n")

    monkeypatch.setattr(planner.subprocess, "run", scheduler)
    monkeypatch.setattr(
        sys, "argv", [
            "hpc/plan.py", "--site", str(tmp_path / "site.env"),
            "--output-dir", str(generated), "--only", "seed0_teacher", "--resume", "--submit",
        ],
    )
    planner.main()
    assert [command[0] for command in calls] == ["squeue", "sbatch"]
    updated = json.loads((generated / "submitted.json").read_text())
    assert updated["seed0_teacher"]["job_id"] == "61234"
    resumed = json.loads((generated / "plan.json").read_text())["jobs"][0]
    assert option(resumed["commands"][0], "--resume") == str(checkpoint)
    assert resumed["signature"] == original["signature"]


@pytest.mark.parametrize("failure", ["scheduler_error", "malformed_output"])
def test_resume_refuses_submission_when_queue_status_cannot_be_verified(
    planning, monkeypatch, tmp_path, capsys, failure,
):
    planner, invoke, _, _, generated = planning
    invoke("--only", "seed0_teacher")
    receipt = json.dumps({"seed0_teacher": {"job_id": "60793", "signature": "fixture"}})
    (generated / "submitted.json").write_text(receipt)
    calls = []

    def scheduler(command, **kwargs):
        calls.append(command)
        assert command[0] == "squeue"
        if failure == "scheduler_error":
            raise subprocess.CalledProcessError(1, command, stderr="controller unavailable")
        return SimpleNamespace(stdout="unexpected controller response\n")

    monkeypatch.setattr(planner.subprocess, "run", scheduler)
    monkeypatch.setattr(
        sys, "argv", [
            "hpc/plan.py", "--site", str(tmp_path / "site.env"),
            "--output-dir", str(generated), "--only", "seed0_teacher", "--resume", "--submit",
        ],
    )
    with pytest.raises(SystemExit) as error:
        planner.main()
    assert error.value.code == 2
    assert "Cannot verify active jobs; no jobs submitted" in capsys.readouterr().err
    assert len(calls) == 1
    assert (generated / "submitted.json").read_text() == receipt


def test_resume_checks_all_known_jobs_with_one_queue_query(planning, monkeypatch, tmp_path):
    planner, invoke, _, _, generated = planning
    options = ["--plan", "central_3seed", "--only", "seed0_teacher", "seed1_teacher"]
    jobs = invoke(*options)["jobs"]
    (generated / "submitted.json").write_text(json.dumps({
        job["name"]: {"job_id": str(61000 + index), "signature": job["signature"]}
        for index, job in enumerate(jobs)
    }))
    calls = []

    def scheduler(command, **kwargs):
        calls.append(command)
        assert command[0] == "squeue"
        return SimpleNamespace(stdout="61000|SUSPENDED\n61001|REQUEUE_HOLD\n")

    monkeypatch.setattr(planner.subprocess, "run", scheduler)
    monkeypatch.setattr(
        sys, "argv", [
            "hpc/plan.py", "--site", str(tmp_path / "site.env"),
            "--output-dir", str(generated), *options, "--resume", "--submit",
        ],
    )
    planner.main()
    assert len(calls) == 1


def test_teacher_resume_can_use_frozen_yaml_after_source_defaults_change(planning):
    planner, invoke, repo, run_root, generated = planning
    options = [
        "--profile", "b200_4gpu", "--only", "seed0_teacher", "--teacher-config",
        "configs/teacher_b200_b128_m8_blocks32_no_previous_joint_commands.yaml",
    ]
    initial = invoke(*options)["jobs"][0]
    saved = planner.resolved_config(Path(option(initial["commands"][0], "--config")))
    checkpoint = run_root / "seed0/teacher/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"Presence-only checkpoint fixture; never loaded")
    # Export the checkpoint config as YAML. Feeding a JSON snapshot through the
    # YAML loader directly can turn JSON scientific-notation floats into strings.
    snapshot = generated / "inputs/teacher_resume.yaml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(yaml.safe_dump(json.loads(json.dumps(saved)), sort_keys=False))
    source = repo / "configs/teacher_b200_b128_m8_blocks32.yaml"
    cfg = yaml.safe_load(source.read_text())
    cfg["training"]["lr"] = 0.123
    source.write_text(yaml.safe_dump(cfg))
    resumed = invoke(
        "--profile", "b200_4gpu", "--only", "seed0_teacher", "--resume",
        "--teacher-config", str(snapshot),
    )["jobs"][0]
    restored = planner.resolved_config(Path(option(resumed["commands"][0], "--config")))
    assert restored == saved
    assert restored["training"]["lr"] == 0.0001
    assert restored["training"]["steps"] == 100000
    assert restored["model"]["condition_on_previous_joint_commands"] is False
    assert option(resumed["commands"][0], "--resume") == str(checkpoint)
    assert resumed["signature"] == initial["signature"]


@pytest.mark.parametrize("suffix,horizon", [("", 16), ("_h32", 32)])
def test_b200_command_ablation_resume_enables_attention_without_replacing_run(
    planning, tmp_path, suffix, horizon,
):
    planner, invoke, repo, _, generated = planning
    stem = "teacher_b200_b128_m8_blocks32_no_previous_joint_commands" + suffix
    run_root = repo / "runs" / stem
    site = tmp_path / "site.env"
    site.write_text(site.read_text() + "STMEM_PARTITION=aiq\nSTMEM_GPU_TYPE=\n"
                    "STMEM_CPUS=64\nSTMEM_MEM=128G\nSTMEM_WALLTIME=48:00:00\n")
    options = [
        "--profile", "b200_4gpu", "--only", "seed0_teacher",
        "--teacher-config", f"configs/{stem}.yaml", "--run-root", f"runs/{stem}",
    ]
    initial = invoke(*options)["jobs"][0]
    initial_config = planner.resolved_config(Path(option(initial["commands"][0], "--config")))
    directory = run_root / "seed0/teacher"
    directory.mkdir(parents=True)
    checkpoint = directory / "last.pt"
    checkpoint.write_bytes(b"Presence-only fixture; training checkpoint validation runs in trainer")
    tracking = directory / "tracking.json"
    tracking.write_text('{"id":"existing-wandb-run"}\n')
    before = {p: p.read_bytes() for p in (checkpoint, tracking)}

    resumed = invoke(*options, "--resume", "--attention-every", "10")
    assert resumed["job_count"] == 1
    job = resumed["jobs"][0]
    command = job["commands"][0]
    assert option(command, "--resume") == str(checkpoint)
    assert option(command, "--attention-every") == "10"
    assert job["signature"] == initial["signature"]
    cfg = planner.resolved_config(Path(option(command, "--config")))
    assert cfg == initial_config
    assert cfg["model"]["horizon"] == horizon and cfg["model"]["execute"] == 4
    assert cfg["model"]["condition_on_previous_joint_commands"] is False
    assert cfg["training"]["effective_batch"] == 128 and cfg["training"]["microbatch"] == 8
    assert cfg["training"]["encoder_block_batch_size"] == 32
    assert cfg["training"]["steps"] == 100000 and cfg["training"]["lr"] == 0.0001
    script = Path(job["script"]).read_text()
    for directive in (
        "--partition=aiq", "--gres=gpu:4", "--cpus-per-task=64", "--mem=128G", "--time=48:00:00",
    ):
        assert f"#SBATCH {directive}" in script
    assert "--nproc-per-node=4" in script and "--attention-every 10" in script
    subprocess.run(["bash", "-n", job["script"]], check=True)
    assert {p: p.read_bytes() for p in before} == before
    assert not (generated / "submitted.json").exists()


def test_dry_run_forwards_runtime_flags_without_changing_configs_or_signatures(planning):
    _, invoke, _, _, generated = planning
    initial = invoke()
    original_configs = {path.name: path.read_bytes() for path in (generated / "configs").glob("*.yaml")}
    original_jobs = {job["name"]: job for job in initial["jobs"]}
    for job in initial["jobs"]:
        for command in job["commands"]:
            assert "--log-every" not in command and "--progress-every-seconds" not in command
            assert "--attention-every" not in command
    changed = invoke("--log-every", "1", "--progress-every-seconds", "2.5", "--attention-every", "10")
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
                    assert "--attention-every" not in command
                else:
                    assert option(command, "--progress-every-seconds") == "2.5"
                    assert option(command, "--attention-every") == "10"
                script = Path(job["script"]).read_text()
                assert "--log-every 1" in script
                assert ("--progress-every-seconds 2.5" in script) == (job["stage"] != "router")
                assert ("--attention-every 10" in script) == (job["stage"] != "router")
                subprocess.run(["bash", "-n", job["script"]], check=True)
            else:
                assert "--log-every" not in command and "--progress-every-seconds" not in command
                assert "--attention-every" not in command
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
        "--attention-every",
        "0",
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
        assert ("--attention-every" in command) == (stage == "teacher")
        if stage == "teacher":
            assert option(command, "--attention-every") == "0"
        assert job["signature"] == original_jobs[job["name"]]["signature"]
        script = Path(job["script"]).read_text()
        assert f"--signature {job['signature']}" in script
        assert "--log-every 3" in script and f"--resume {checkpoint}" in script
        assert ("--progress-every-seconds 0.25" in script) == (stage == "teacher")
        assert ("--attention-every 0" in script) == (stage == "teacher")
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
        ("--attention-every", "-1"),
        ("--attention-every", "1.5"),
        ("--attention-every", "nan"),
        ("--attention-every", "inf"),
        ("--attention-every", "nope"),
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
    runtime = ["--log-every", "5", "--progress-every-seconds", "1.5", "--attention-every", "10"]
    assert planner.scientific_training_command(command + runtime) == command
    assert planner.scientific_training_command(command + ["--attention-every", "0"]) == command
    assert planner.scientific_training_command(command + ["--attention-every", "-1"]) == (
        command + ["--attention-every", "-1"]
    )
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
    assert cfg["training"]["effective_batch"] == 256
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
