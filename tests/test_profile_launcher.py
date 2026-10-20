"""The separate profiler launcher must never touch or submit normal training."""

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

from stmem.config import load_config


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def profile_launcher(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith(("STMEM_", "WANDB_", "SLURM_")):
            monkeypatch.delenv(key)
    spec = importlib.util.spec_from_file_location("profile_hpc_launcher", REPO / "hpc/profile_memory.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    repo = tmp_path / "repository with spaces"
    shutil.copytree(REPO / "configs", repo / "configs")
    (repo / "hpc/jobs").mkdir(parents=True)
    shutil.copyfile(REPO / "hpc/jobs/profile_teacher.sbatch", repo / "hpc/jobs/profile_teacher.sbatch")
    site = tmp_path / "site.env"
    run_root = tmp_path / "run outputs"
    values = {
        "STMEM_REPO_ROOT": str(repo),
        "STMEM_DATA_ROOT": str(tmp_path / "prepared data"),
        "STMEM_CACHE_ROOT": str(tmp_path / "cache"),
        "STMEM_RUN_ROOT": str(run_root),
        "STMEM_PYTHON": sys.executable,
        "STMEM_PROFILE": "h200_4gpu",
        "STMEM_ACCOUNT": "my-account",
        "STMEM_PARTITION": "gpuq",
        "STMEM_GPU_TYPE": "customgpu",
        "STMEM_CPUS": "16",
        "STMEM_MEM": "128G",
        "WANDB_MODE": "offline",
        "WANDB_PROJECT": "profile-test",
        "WANDB_ENTITY": "",
    }
    site.write_text("\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n")

    def invoke(*arguments):
        launcher.main(["--site", str(site), "--name", "memory-first", *arguments])
        return json.loads((repo / "hpc/generated/profiles/memory-first/plan.json").read_text())

    return launcher, invoke, repo, run_root, site


@pytest.mark.parametrize("profile, gpus", [("h200_1gpu", 1), ("h200_4gpu", 4), ("b200_8gpu", 8)])
def test_dry_run_isolated_paths_config_and_single_agent(profile_launcher, monkeypatch, profile, gpus):
    launcher, invoke, repo, run_root, _ = profile_launcher
    with monkeypatch.context() as scoped:
        scoped.setattr(
            launcher.subprocess,
            "run",
            lambda *args, **kwargs: pytest.fail("Dry-run must never execute a command"),
        )
        plan = invoke("--profile", profile)
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    source = load_config(repo / "configs/teacher.yaml")
    assert plan["warmup_steps"] == 10 and plan["measured_steps"] == 5 and plan["trace_steps"] == 1
    assert plan["gpus"] == gpus and plan["slurm_tasks"] == 1 and plan["job_count"] == 1
    assert cfg["training"] == source["training"]
    assert cfg["profile"] == profile and cfg["paths"]["run_root"] == str(run_root)
    assert "extends" not in cfg and cfg["data"]["synthetic"] is False
    assert Path(plan["output"]) == run_root / "profiles/memory-first"
    assert not run_root.exists()  # Rendering cannot create/overwrite any training or profiling result.
    assert (repo / "hpc/logs").is_dir()
    script = Path(plan["script"]).read_text()
    assert "#SBATCH --nodes=1" in script and "#SBATCH --ntasks=1" in script
    assert f"#SBATCH --gres=gpu:customgpu:{gpus}" in script
    assert "#SBATCH --account=my-account" in script
    assert "unset WANDB_RUN_ID WANDB_RESUME" in script
    assert "stmem.train" not in script and "run_job.sh" not in script
    assert plan["command"].count("--nnodes=1") == 1
    assert f"--nproc-per-node={gpus}" in plan["command"]
    assert "--warmup-steps" in plan["command"]
    assert not (Path(plan["script"]).parent / "submitted.json").exists()
    subprocess.run(["bash", "-n", plan["script"]], check=True)


def test_generated_script_preserves_arguments_with_spaces_and_launches_one_agent(
    profile_launcher,
    monkeypatch,
    tmp_path,
):
    _, invoke, repo, _, _ = profile_launcher
    plan = invoke("--warmup-steps", "5", "--steps", "7", "--trace-steps", "0", "--scenario", "long-history")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), "
        "'old_id': os.environ.get('WANDB_RUN_ID')}))\n"
    )
    fake_srun.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("SLURM_JOB_ID", "fixture-only")
    monkeypatch.setenv("WANDB_RUN_ID", "existing-teacher")
    result = subprocess.run(["bash", plan["script"]], capture_output=True, text=True, check=True)
    launch = json.loads(result.stdout)
    assert launch["cwd"] == str(repo) and launch["old_id"] is None
    args = launch["args"]
    assert args[:5] == ["--ntasks=1", sys.executable, "-u", "-m", "torch.distributed.run"]
    assert args.count("--nproc-per-node=4") == 1 and args.count("--nnodes=1") == 1
    assert args[args.index("--config") + 1] == plan["config"]
    assert args[args.index("--output") + 1] == plan["output"]
    assert args[args.index("--warmup-steps") + 1] == "5"
    assert args[args.index("--steps") + 1] == "7"
    assert args[args.index("--trace-steps") + 1] == "0"
    assert args[args.index("--scenario") + 1] == "long-history"


def test_submit_is_explicit_once_and_never_reuses_outputs(profile_launcher, monkeypatch):
    launcher, invoke, repo, run_root, _ = profile_launcher
    plan = invoke("--dry-run", "--microbatch", "8", "--effective-batch", "32")
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    assert cfg["training"]["microbatch"] == 8 and cfg["training"]["effective_batch"] == 32
    calls = []

    def sbatch(command, **kwargs):
        calls.append(command)
        assert command == ["sbatch", "--parsable", plan["script"]]
        assert (repo / "hpc/logs").is_dir()
        assert not run_root.exists()
        return SimpleNamespace(stdout="12345;fixture-cluster\n")

    monkeypatch.setattr(launcher.subprocess, "run", sbatch)
    invoke("--submit", "--microbatch", "8", "--effective-batch", "32")
    receipt = json.loads((Path(plan["script"]).parent / "submitted.json").read_text())
    assert receipt["job_id"] == "12345" and len(calls) == 1
    with pytest.raises(SystemExit):
        invoke("--submit")
    assert len(calls) == 1
    (Path(plan["script"]).parent / "submitted.json").unlink()
    Path(plan["output"]).mkdir(parents=True)
    with pytest.raises(SystemExit):
        invoke("--dry-run")


@pytest.mark.parametrize(
    "arguments",
    [
        ("--warmup-steps", "4"),
        ("--warmup-steps", "11"),
        ("--steps", "0"),
        ("--steps", "101"),
        ("--trace-steps", "2"),
        ("--name", "../teacher"),
        ("--microbatch", "0"),
        ("--walltime", "00:00:00"),
        ("--effective-batch", "63"),
    ],
)
def test_invalid_budgets_and_output_names_fail_before_writes(profile_launcher, arguments):
    _, invoke, repo, run_root, _ = profile_launcher
    with pytest.raises(SystemExit) as error:
        invoke(*arguments)
    assert error.value.code == 2
    assert not (repo / "hpc/generated").exists() and not run_root.exists()


def test_hpc_refuses_synthetic_and_unlogged_profiling(profile_launcher):
    _, invoke, repo, _, site = profile_launcher
    with pytest.raises(SystemExit):
        invoke("--config", "configs/debug/teacher.yaml")
    site.write_text(site.read_text().replace("WANDB_MODE=offline", "WANDB_MODE=disabled"))
    with pytest.raises(SystemExit):
        invoke()
    assert not (repo / "hpc/generated").exists()
