"""Exercise the CPU preparation launcher without Slurm or dataset access."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def preparation_job(tmp_path):
    repository = tmp_path / "repository with spaces"
    jobs = repository / "hpc/jobs"
    jobs.mkdir(parents=True)
    for name in ("common.sh", "prepare.sbatch"):
        shutil.copyfile(REPO / "hpc/jobs" / name, jobs / name)
    python = repository / "python executable"
    python.symlink_to(sys.executable)
    (repository / "hpc/site.env").write_text(
        f"STMEM_REPO_ROOT={shlex.quote(str(repository))}\n"
        f"STMEM_PYTHON={shlex.quote(str(python))}\n"
        "STMEM_CPUS=16\nSTMEM_PROFILE=b200_8gpu\n"
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_srun = fake_bin / "srun"
    fake_srun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), 'threads': "
        "{k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', "
        "'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}))\n"
    )
    fake_srun.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("STMEM_", "SLURM_"))}
    env.update(
        PATH=f"{fake_bin}{os.pathsep}{env.get('PATH', '')}", SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="16"
    )
    return repository, jobs / "prepare.sbatch", python, env


@pytest.mark.parametrize(
    "cpus, requested, expected", [("16", None, "8"), ("4", None, "4"), ("64", "16", "16")]
)
def test_prepare_launch_bounds_workers_and_forwards_literal_arguments(
    preparation_job, cpus, requested, expected
):
    repository, script, python, env = preparation_job
    env["SLURM_CPUS_PER_TASK"] = cpus
    if requested is not None:
        env["STMEM_PREP_WORKERS"] = requested
    env.update(
        OMP_NUM_THREADS="32", MKL_NUM_THREADS="32", OPENBLAS_NUM_THREADS="32", NUMEXPR_NUM_THREADS="32"
    )
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(
        ["bash", str(script), "--tasks", "VideoUnmask", "--config", "custom config.yaml", "--workers", "99"],
        cwd=repository,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    launch = json.loads(result.stdout)
    args = launch["args"]
    assert args[:5] == ["--ntasks=1", str(python), "-u", "-m", "stmem.data.prepare"]
    assert args[-5:] == ["--workers", expected, "--threads-per-worker", "1", "--skip-existing"]
    assert ["--config", "custom config.yaml"] == args[
        args.index("custom config.yaml") - 1 : args.index("custom config.yaml") + 1
    ]
    assert launch["cwd"] == str(repository)
    assert set(launch["threads"].values()) == {"1"}
    assert "torch.distributed.run" not in args and "run_job.sh" not in args


@pytest.mark.parametrize("requested", ["0", "-1", "2x", "17", "999999999999999999999999999999", "1+1"])
def test_prepare_rejects_invalid_workers_before_srun(preparation_job, requested):
    repository, script, _, env = preparation_job
    env["STMEM_PREP_WORKERS"] = requested
    result = subprocess.run(["bash", str(script)], cwd=repository, env=env, capture_output=True, text=True)
    assert result.returncode == 64 and "STMEM_PREP_WORKERS must" in result.stderr
    assert not result.stdout


def test_prepare_requires_allocation_and_rejects_cpu_oversubscription(preparation_job):
    repository, script, _, env = preparation_job
    env.pop("SLURM_JOB_ID")
    result = subprocess.run(["bash", str(script)], cwd=repository, env=env, capture_output=True, text=True)
    assert result.returncode == 64 and "inside an allocation" in result.stderr
    env.update(SLURM_JOB_ID="123", SLURM_CPUS_PER_TASK="2", STMEM_PREP_WORKERS="4")
    result = subprocess.run(["bash", str(script)], cwd=repository, env=env, capture_output=True, text=True)
    assert result.returncode == 64 and "from 1 to 2" in result.stderr
    assert not result.stdout
