"""Fresh teacher launches keep active runs isolated and honor the full update budget."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import yaml


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def teacher_launcher(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith(("STMEM_", "WANDB_", "SLURM_")):
            monkeypatch.delenv(key)
    spec = importlib.util.spec_from_file_location("isolated_teacher_launcher", REPO / "hpc/train_teacher.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    repo = tmp_path / "repository with spaces"
    shutil.copytree(REPO / "configs", repo / "configs")
    (repo / "hpc").mkdir()
    for script in ("run_job.sh", "job_status.py"):
        shutil.copy2(REPO / "hpc" / script, repo / "hpc" / script)
    experiment = repo / "configs/teacher_test.yaml"
    experiment.write_text(
        "extends: teacher.yaml\n"
        "training:\n"
        "  steps: 100000\n"
        "  effective_batch: 128\n"
        "  microbatch: 64\n"
        "  encoder_block_batch_size: 128\n"
        "  checkpoint_steps: [50000, 100000]\n"
    )
    run_root = tmp_path / "new training run"
    old_root = tmp_path / "currently running"
    old_output = old_root / "seed0/teacher"
    old_output.mkdir(parents=True)
    (old_output / "last.pt").write_bytes(b"Existing run must remain untouched")
    site = tmp_path / "site_4gpu.env"
    values = {
        "STMEM_REPO_ROOT": str(repo),
        "STMEM_DATA_ROOT": str(tmp_path / "prepared data"),
        "STMEM_CACHE_ROOT": str(tmp_path / "shared cache"),
        "STMEM_RUN_ROOT": str(old_root),
        "STMEM_PYTHON": sys.executable,
        "STMEM_PROFILE": "h200_4gpu",
        "STMEM_ACCOUNT": "my-account",
        "STMEM_PARTITION": "gpuq",
        "STMEM_GPU_TYPE": "customgpu",
        "STMEM_CPUS": "16",
        "STMEM_MEM": "128G",
        "STMEM_WALLTIME": "48:00:00",
        "WANDB_MODE": "offline",
        "WANDB_PROJECT": "teacher-tests",
        "WANDB_ENTITY": "",
    }
    site.write_text("\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n")
    generated = repo / "hpc/generated/training/fresh-teacher"
    baseline = {path: path.read_bytes() for path in (site, experiment, repo / "configs/base.yaml")}

    def invoke(*arguments):
        launcher.main(
            [
                "--site",
                str(site),
                "--config",
                str(experiment),
                "--name",
                "fresh-teacher",
                "--run-root",
                str(run_root),
                *arguments,
            ]
        )
        return json.loads((generated / "plan.json").read_text())

    return SimpleNamespace(
        launcher=launcher,
        invoke=invoke,
        repo=repo,
        run_root=run_root,
        old_root=old_root,
        site=site,
        experiment=experiment,
        generated=generated,
        baseline=baseline,
    )


def test_requested_full_training_budget_and_isolated_outputs(teacher_launcher, monkeypatch):
    fixture = teacher_launcher
    with monkeypatch.context() as scoped:
        scoped.setattr(
            fixture.launcher.subprocess,
            "run",
            lambda *args, **kwargs: pytest.fail("Dry-run must not call subprocesses"),
        )
        plan = fixture.invoke("--dry-run")
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    assert cfg["training"]["steps"] == plan["steps"] == 100000
    assert cfg["training"]["effective_batch"] == plan["effective_batch"] == 128
    assert cfg["training"]["microbatch"] == plan["microbatch"] == 64
    assert cfg["training"]["encoder_block_batch_size"] == plan["encoder_block_batch_size"] == 128
    assert cfg["training"]["checkpoint_steps"] == [50000, 100000]
    assert cfg["paths"]["run_root"] == str(fixture.run_root)
    assert plan["actual_microbatch_per_gpu"] == plan["examples_per_gpu"] == 32
    assert plan["gradient_accumulation_steps"] == 1 and plan["gpus"] == 4
    assert plan["output"] == str(fixture.run_root / "seed0/teacher")
    assert not fixture.run_root.exists()
    job = plan["jobs"][0]
    assert job["deps"] == [] and job["inputs"] == []
    assert job["walltime"] == "48:00:00"
    assert str(fixture.run_root / "seed0/teacher/step_00100000.pt") in job["outputs"]
    assert str(fixture.run_root / "seed0/teacher/complete.json") in job["outputs"]
    assert Path(job["marker"]).is_relative_to(fixture.run_root)
    command = job["commands"][0]
    assert command[command.index("--stage") + 1] == "teacher"
    assert "--resume" not in command and "--steps" not in command
    script = Path(job["script"]).read_text()
    assert "#SBATCH --signal=B:USR1@180" in script
    assert "#SBATCH --ntasks=1" in script and "#SBATCH --nodes=1" in script
    assert "#SBATCH --gres=gpu:customgpu:4" in script
    assert "--nproc-per-node=4" in script
    assert "run_job.sh" in script and ' --stop-file "$STMEM_STOP_FILE"' in script
    assert "--log-every 1" in script and "--progress-every-seconds 30" in script
    assert "unset WANDB_RUN_ID WANDB_RESUME" in script
    subprocess.run(["bash", "-n", job["script"]], check=True)
    for path, before in fixture.baseline.items():
        assert path.read_bytes() == before
    assert (fixture.old_root / "seed0/teacher/last.pt").read_bytes() == b"Existing run must remain untouched"


@pytest.mark.parametrize(
    "profile, examples, microbatch, accumulation",
    [("h200_1gpu", 128, 64, 2), ("h200_4gpu", 32, 32, 1), ("b200_8gpu", 16, 16, 1)],
)
def test_reports_effective_per_device_batch(teacher_launcher, profile, examples, microbatch, accumulation):
    plan = teacher_launcher.invoke("--profile", profile)
    assert plan["examples_per_gpu"] == examples
    assert plan["actual_microbatch_per_gpu"] == microbatch
    assert plan["gradient_accumulation_steps"] == accumulation
    assert plan["effective_batch"] == 128 and plan["microbatch"] == 64


def test_explicit_submission_records_id_and_cannot_mutate_queued_job(teacher_launcher, monkeypatch):
    fixture = teacher_launcher
    plan = fixture.invoke()
    calls = []

    def fake_sbatch(command, **kwargs):
        calls.append(command)
        assert command == ["sbatch", "--parsable", plan["jobs"][0]["script"]]
        assert kwargs == {"check": True, "capture_output": True, "text": True}
        assert (fixture.repo / "hpc/logs").is_dir()
        assert not fixture.run_root.exists()
        return SimpleNamespace(stdout="12345;fixture-cluster\n")

    monkeypatch.setattr(fixture.launcher.subprocess, "run", fake_sbatch)
    fixture.invoke("--submit")
    receipt = json.loads((fixture.generated / "submitted.json").read_text())
    assert receipt["job_id"] == "12345" and len(calls) == 1
    saved = {path: path.read_bytes() for path in fixture.generated.rglob("*") if path.is_file()}
    fixture.experiment.write_text(fixture.experiment.read_text().replace("100000", "200000"))
    for action in ("--submit", "--dry-run"):
        with pytest.raises(SystemExit):
            fixture.invoke(action)
        assert len(calls) == 1
        for path, before in saved.items():
            assert path.read_bytes() == before


def test_requires_fresh_root_and_preserves_existing_generated_files(teacher_launcher):
    fixture = teacher_launcher
    fixture.run_root.mkdir()
    sentinel = fixture.run_root / "already-here"
    sentinel.write_text("Do not overwrite")
    with pytest.raises(SystemExit):
        fixture.invoke()
    assert sentinel.read_text() == "Do not overwrite"
    assert not fixture.generated.exists()
    with pytest.raises(SystemExit):
        fixture.launcher.main(["--site", str(fixture.site), "--name", "unsafe-default-root", "--dry-run"])
    assert not (fixture.repo / "hpc/generated").exists()


def test_queued_job_refuses_root_created_after_rendering(teacher_launcher):
    fixture = teacher_launcher
    plan = fixture.invoke()
    fixture.run_root.mkdir()
    sentinel = fixture.run_root / "other-job-checkpoint"
    sentinel.write_bytes(b"Another allocation claimed this root")
    result = subprocess.run(["bash", plan["jobs"][0]["script"]], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert sentinel.read_bytes() == b"Another allocation claimed this root"
    assert list(fixture.run_root.iterdir()) == [sentinel]
    assert not Path(plan["jobs"][0]["marker"]).exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "compact"},
        {"data": {"synthetic": True}},
        {"data": {"split": "test"}},
        {"data": {"split": "val"}},
        {"tracking": {"enabled": False}},
        {"training": {"effective_batch": 127}},
        {"training": {"steps": 0}},
    ],
)
def test_invalid_training_config_fails_before_writing(teacher_launcher, changes):
    fixture = teacher_launcher
    cfg = yaml.safe_load(fixture.experiment.read_text())
    for key, value in changes.items():
        if isinstance(value, dict):
            cfg.setdefault(key, {}).update(value)
        else:
            cfg[key] = value
    fixture.experiment.write_text(yaml.safe_dump(cfg))
    with pytest.raises(SystemExit):
        fixture.invoke()
    assert not fixture.generated.exists()
    assert not fixture.run_root.exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ("--name", "../existing"),
        ("--log-every", "0"),
        ("--progress-every-seconds", "0"),
        ("--profile", "a5000_debug"),
    ],
)
def test_invalid_launch_options_fail_before_writing(teacher_launcher, arguments):
    with pytest.raises(SystemExit):
        teacher_launcher.invoke(*arguments)
    assert not (teacher_launcher.repo / "hpc/generated").exists()
    assert not teacher_launcher.run_root.exists()


@pytest.mark.parametrize("complete", [False, True])
def test_generated_job_requires_full_completion_and_preserves_quoted_paths(
    teacher_launcher, monkeypatch, tmp_path, complete
):
    fixture = teacher_launcher
    plan = fixture.invoke()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    output = Path(plan["output"])
    arguments_capture = tmp_path / "launch.json"
    outputs = plan["jobs"][0]["outputs"] if complete else [str(output / "step_00050000.pt")]
    fake_srun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"Path({str(arguments_capture)!r}).write_text(json.dumps({{"
        "'args': sys.argv[1:], 'cwd': os.getcwd(),"
        "'run_root': os.environ.get('STMEM_RUN_ROOT'),"
        "'old_id': os.environ.get('WANDB_RUN_ID')}))\n"
        f"for path in {outputs!r}:\n"
        "    target = Path(path)\n"
        "    target.parent.mkdir(parents=True, exist_ok=True)\n"
        "    target.write_text('fake artifact for launcher test')\n"
    )
    fake_srun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("WANDB_RUN_ID", "active-teacher")
    monkeypatch.setenv("WANDB_RESUME", "must")
    monkeypatch.setenv("SLURM_JOB_ID", "launcher-fixture-only")
    result = subprocess.run(["bash", plan["jobs"][0]["script"]], capture_output=True, text=True, timeout=20)
    assert result.returncode == (0 if complete else 66), result.stdout + result.stderr
    launch = json.loads(arguments_capture.read_text())
    assert launch["cwd"] == str(fixture.repo)
    assert launch["run_root"] == str(fixture.run_root) and launch["old_id"] is None
    args = launch["args"]
    assert args[:4] == ["--ntasks=1", sys.executable, "-m", "torch.distributed.run"]
    assert args.count("--nproc-per-node=4") == 1
    assert args[args.index("--config") + 1] == plan["config"]
    assert args[args.index("--stage") + 1] == "teacher"
    assert "launcher-fixture-only.stop" in args[args.index("--stop-file") + 1]
    marker = Path(plan["jobs"][0]["marker"])
    assert marker.exists() is complete
    record = json.loads((marker if complete else marker.with_suffix(".incomplete.json")).read_text())
    assert record["status"] == ("complete" if complete else "incomplete")
    assert record["allocated_gpus"] == 4
    if not complete:
        assert str(output / "step_00100000.pt") in record["missing_outputs"]
        assert str(output / "complete.json") in record["missing_outputs"]


def test_warning_signal_requests_stop_and_keeps_partial_job_incomplete(
    teacher_launcher, monkeypatch, tmp_path
):
    fixture = teacher_launcher
    plan = fixture.invoke()
    fake_bin = tmp_path / "signal-bin"
    fake_bin.mkdir()
    ready = tmp_path / "child-ready"
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "from pathlib import Path\n"
        "stop = Path(sys.argv[sys.argv.index('--stop-file') + 1])\n"
        f"Path({str(ready)!r}).write_text(str(stop))\n"
        "deadline = time.monotonic() + 10\n"
        "while time.monotonic() < deadline:\n"
        "    if stop.exists():\n"
        "        raise SystemExit(75)\n"
        "    time.sleep(0.01)\n"
        "raise SystemExit(99)\n"
    )
    fake_srun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("SLURM_JOB_ID", "signal-fixture-only")
    process = subprocess.Popen(
        ["bash", plan["jobs"][0]["script"]], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "Fixture command must start before sending the warning signal"
        process.send_signal(signal.SIGUSR1)
        stdout, stderr = process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=15)
    assert process.returncode == 75, stdout + stderr
    assert Path(ready.read_text()).is_file()
    marker = Path(plan["jobs"][0]["marker"])
    assert not marker.exists()
    record = json.loads(marker.with_suffix(".incomplete.json").read_text())
    assert record["status"] == "incomplete" and record["exit_code"] == 75
