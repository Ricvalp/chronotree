"""CPU-only Slurm rendering checks; synthetic serialized fixtures, no training."""

from copy import deepcopy
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
import torch
import yaml

from stmem.config import load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith(("STMEM_", "WANDB_", "SLURM_")):
            monkeypatch.delenv(key)
    spec = importlib.util.spec_from_file_location("parallel_evaluation_launcher", REPO / "hpc/evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "repository with spaces"
    (repo / "hpc").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "src/policy_fixture.py").write_text("# Explicit synthetic source-identity fixture\n")
    for name in ("plan.py", "run_job.sh"):
        shutil.copyfile(REPO / "hpc" / name, repo / "hpc" / name)
    (repo / "hpc/run_job.sh").chmod(0o755)
    cfg = load_config(REPO / "configs/teacher.yaml")
    cfg["model"].update(horizon=16, execute=4)
    cfg["paths"] = {"run_root": str(tmp_path / "original training run"),
                    "data_root": str(tmp_path / "data"), "cache_root": str(tmp_path / "cache")}
    cfg["evaluation"]["episode_count"] = 2
    source = tmp_path / "last.pt"
    torch.save({"config": cfg, "step": 17, "model": {"synthetic_fixture": torch.zeros(1)}}, source)
    site = tmp_path / "site.env"
    values = {
        "STMEM_REPO_ROOT": str(repo), "STMEM_PYTHON": sys.executable,
        "STMEM_DATA_ROOT": str(tmp_path / "data"), "STMEM_CACHE_ROOT": str(tmp_path / "cache"),
        "STMEM_RUN_ROOT": str(tmp_path / "unrelated site run root"),
        "STMEM_PARTITION": "aiq", "STMEM_GPU_TYPE": "h200", "STMEM_CPUS": "16",
        "STMEM_ACCOUNT": "fixture-account", "WANDB_MODE": "offline", "WANDB_PROJECT": "fixture",
    }
    site.write_text("\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + "\n")

    def invoke(*args):
        module.main(["--site", str(site), "--checkpoint", str(source), "--name", "example", *args])
        return json.loads((repo / "hpc/generated/example/plan.json").read_text())

    return SimpleNamespace(module=module, repo=repo, cfg=cfg, source=source, site=site, invoke=invoke)


def test_default_dry_run_is_one_allocation_and_global_episode_video_budget(launcher, monkeypatch):
    with monkeypatch.context() as guard:
        guard.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm during dry-run"))
        plan = launcher.invoke("--execute-actions", "8")
    assert plan["job_count"] == plan["nodes"] == plan["slurm_tasks"] == 1
    assert plan["gpus"] == 4 and plan["requested_workers"] == plan["num_shards"] == 16
    assert plan["cpus"] == 64 and plan["partition"] == "gpuq" and plan["gres"] == "gpu:4"
    assert plan["planned_episodes"] == 64 and plan["episode_count_per_task"] == 4
    assert plan["video_cap_total"] == 10 and plan["execution"]["execute_actions"] == 8
    assert "torch.distributed" not in " ".join(plan["command"])
    assert plan["command"].count("--ntasks=1") == 1
    assert "--cpus-per-task=64" in plan["command"] and "--cpu-bind=cores" in plan["command"]
    assert "--gpu-bind=none" in plan["command"]
    assert plan["command"][8] == "stmem.evaluation.parallel"
    config = yaml.safe_load(Path(plan["config"]).read_text())
    assert config["model"] == launcher.cfg["model"]
    assert config["training"] == launcher.cfg["training"]
    assert config["paths"]["run_root"] == launcher.cfg["paths"]["run_root"]
    assert config["evaluation"]["episode_count"] == 4 and config["evaluation"]["video_cap"] == 10
    assert Path(plan["checkpoint"]) != launcher.source
    assert Path(plan["checkpoint"]).read_bytes() == launcher.source.read_bytes()
    assert not Path(plan["output"]).exists()
    script = Path(plan["script"]).read_text()
    assert "#SBATCH --gres=gpu:4" in script and "#SBATCH --cpus-per-task=64" in script
    assert "#SBATCH --account=fixture-account" in script and "STMEM_STOP_FILE" in script
    assert Path(plan["logs"]).is_dir()
    subprocess.run(["bash", "-n", plan["script"]], check=True)


def test_source_last_pt_can_advance_without_changing_reviewed_snapshot(launcher, monkeypatch):
    original = launcher.invoke("--execute-actions", "8")
    torch.save({"config": launcher.cfg, "step": 2000, "model": {"changed": torch.ones(1)}}, launcher.source)
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        assert command == ["sbatch", "--parsable", original["script"]]
        state = torch.load(original["checkpoint"], weights_only=False)
        assert state["step"] == 17
        return SimpleNamespace(stdout="70001;fixture-cluster\n")

    monkeypatch.setattr(launcher.module.subprocess, "run", submit)
    submitted = launcher.invoke("--execute-actions", "8", "--submit")
    assert submitted["checkpoint_sha256"] == original["checkpoint_sha256"]
    assert submitted["checkpoint_step"] == 17 and len(calls) == 1
    receipt = json.loads((Path(original["script"]).parent / "submitted.json").read_text())
    assert receipt["job_id"] == "70001"
    with pytest.raises(SystemExit):
        launcher.invoke("--execute-actions", "8", "--submit")
    assert len(calls) == 1


def test_resume_refuses_active_allocation_and_then_reuses_same_snapshot(launcher, monkeypatch):
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=str(70000 + len(calls)) + "\n")

    monkeypatch.setattr(launcher.module.subprocess, "run", submit)
    original = launcher.invoke("--execute-actions", "8", "--submit")
    monkeypatch.setattr(launcher.module, "active_user_job_ids", lambda: {"70001"})
    with pytest.raises(SystemExit):
        launcher.invoke("--execute-actions", "8", "--submit", "--resume")
    assert len(calls) == 1
    monkeypatch.setattr(launcher.module, "active_user_job_ids", set)
    resumed = launcher.invoke("--execute-actions", "8", "--submit", "--resume", "--walltime", "08:00:00")
    assert len(calls) == 2 and resumed["checkpoint"] == original["checkpoint"]
    assert resumed["signature"] == original["signature"] and "--resume" in resumed["command"]


@pytest.mark.parametrize("change", [
    ("--execute-actions", "7"), ("--workers-per-gpu", "8"), ("--episode-count", "5"),
])
def test_named_plan_cannot_mix_execution_shards_or_episode_sets(launcher, change):
    original = launcher.invoke()
    with pytest.raises(SystemExit):
        launcher.invoke(*change)
    assert json.loads(Path(original["script"]).with_name("plan.json").read_text()) == original


@pytest.mark.parametrize("arguments", [
    ("--workers-per-gpu", "16"), ("--episode-count", "51"), ("--gpus", "0"),
    ("--video-cap", "-1"), ("--mem", "0"), ("--walltime", "00:00:00"),
    ("--partition", ""),
])
def test_invalid_resource_or_episode_budgets_do_not_copy_weights(launcher, arguments):
    with pytest.raises(SystemExit):
        launcher.invoke(*arguments)
    assert not (launcher.repo / "runs/evaluation_inputs").exists()


def test_generated_script_launches_one_supervisor_with_quoted_paths(launcher, monkeypatch, tmp_path):
    plan = launcher.invoke("--execute-actions", "8", "--workers-per-gpu", "8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    srun = fake_bin / "srun"
    srun.write_text(
        f"#!{sys.executable}\nimport os,sys,json\n"
        "print(json.dumps({'args':sys.argv[1:],'cwd':os.getcwd(),"
        "'cuda':os.environ.get('CUDA_VISIBLE_DEVICES'),'rank':os.environ.get('RANK'),"
        "'stop':os.environ.get('STMEM_STOP_FILE'),'wandb_id':os.environ.get('WANDB_RUN_ID')}))\n"
    )
    srun.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-one,GPU-two,GPU-three,GPU-four")
    monkeypatch.setenv("SLURM_JOB_ID", "99999")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WANDB_RUN_ID", "old-training-run")
    run = subprocess.run(["bash", plan["script"]], check=True, capture_output=True, text=True)
    result = json.loads(run.stdout)
    assert result["cwd"] == str(launcher.repo)
    assert result["cuda"] == "GPU-one,GPU-two,GPU-three,GPU-four"
    assert result["rank"] is None and result["wandb_id"] is None
    assert result["args"][:8] == ["--ntasks=1", "--cpus-per-task=64", "--cpu-bind=cores", "--gpu-bind=none",
                                  sys.executable, "-u", "-m", "stmem.evaluation.parallel"]
    assert result["args"][result["args"].index("--workers-per-gpu") + 1] == "8"
    assert result["args"][result["args"].index("--execute-actions") + 1] == "8"
    assert result["stop"].endswith("/stops/99999.stop")
    assert Path(result["stop"]).parent.parent == Path(plan["script"]).parent
    assert not Path(plan["output"]).exists()  # Supervisor requires an empty fresh output.


def test_corrupted_snapshot_and_changed_model_config_are_rejected(launcher, tmp_path):
    original = launcher.invoke()
    Path(original["checkpoint"]).write_bytes(b"incomplete transfer")
    with pytest.raises(SystemExit):
        launcher.invoke()
    changed = deepcopy(launcher.cfg)
    changed["geometry"]["crop"][0][0] += 0.01
    config = tmp_path / "different.yaml"
    config.write_text(yaml.safe_dump(changed))
    with pytest.raises(SystemExit):
        launcher.module.main(["--site", str(launcher.site), "--checkpoint", str(launcher.source),
                              "--name", "different", "--config", str(config)])
    # Fixing a bad optional YAML does not silently resnapshot an advancing last.pt.
    launcher.module.main(["--site", str(launcher.site), "--checkpoint", str(launcher.source),
                          "--name", "different"])


def test_queued_source_change_fails_before_starting_workers(launcher):
    plan = launcher.invoke()
    (launcher.repo / "src/policy_fixture.py").write_text("# Different scientific implementation\n")
    result = subprocess.run(["bash", plan["script"]], capture_output=True, text=True)
    assert result.returncode != 0 and "Source changed since rendering" in result.stderr
    assert not (Path(plan["output"]) / "stops").exists()


def test_completed_result_cannot_be_resubmitted(launcher, monkeypatch):
    plan = launcher.invoke()
    output = Path(plan["output"])
    output.mkdir(parents=True)
    (output / "complete.json").write_text('{"status":"complete"}')
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Duplicate submission"))
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--submit")
