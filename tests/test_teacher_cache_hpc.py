"""CPU-only launch tests with explicitly synthetic serialized checkpoint fixtures."""

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
def launcher(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("STMEM_", "SLURM_", "WANDB_")):
            monkeypatch.delenv(name)
    spec = importlib.util.spec_from_file_location("teacher_cache_launcher", REPO / "hpc/cache_teacher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "repo with spaces"
    (repo / "hpc").mkdir(parents=True)
    (repo / "src").mkdir()
    (repo / "src/fixture.py").write_text("# synthetic launch fixture\n")
    for name in ("plan.py", "run_job.sh"):
        shutil.copyfile(REPO / "hpc" / name, repo / "hpc" / name)
    (repo / "hpc/run_job.sh").chmod(0o755)
    cfg = load_config(REPO / "configs/teacher_fv_no_prop_10hz.yaml")
    original_root = tmp_path / "original training"
    cfg["paths"] = {"run_root": str(original_root), "data_root": str(tmp_path / "old data"),
                    "cache_root": str(tmp_path / "old cache")}
    source = repo / "evaluated checkpoint.pt"
    torch.save({"config": cfg, "stage": "teacher", "step": 104110,
                "model": {"synthetic_fixture": torch.zeros(1)}}, source)
    evaluation = repo / "completed evaluation"
    evaluation.mkdir()
    (evaluation / "manifest.json").write_text(json.dumps({
        "checkpoints": {str(source): module.sha256(source)}, "policy_checkpoint_sha256": module.sha256(source),
    }))
    site = tmp_path / "site.env"
    site.write_text("\n".join(f"{key}={shlex.quote(str(value))}" for key, value in {
        "STMEM_REPO_ROOT": repo, "STMEM_PYTHON": sys.executable,
        "STMEM_DATA_ROOT": tmp_path / "data", "STMEM_CACHE_ROOT": tmp_path / "cache",
        "STMEM_RUN_ROOT": original_root, "STMEM_PARTITION": "aiq", "STMEM_GPU_TYPE": "b200",
        "STMEM_PROFILE": "b200_4gpu", "STMEM_CPUS": 64,
        "STMEM_ACCOUNT": "fixture-account",
    }.items()) + "\n")

    # Adoption's actual scientific evidence checks are covered in test_teacher_adoption.
    # This fixture isolates planner behavior, not acceptance of synthetic evaluations.
    def adopt(checkpoint, _evaluation, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            assert module.sha256(output) == module.sha256(checkpoint)
        else:
            shutil.copyfile(checkpoint, output)
        record = {"sha256": module.sha256(output), "evaluation_config": {"model": {"execute": 8}},
                  "execution_override": {"execute_actions": 8}, "synthetic_launcher_fixture": True}
        output.with_suffix(".selection.json").write_text(json.dumps(record))
        return output

    monkeypatch.setattr(module, "adopt_teacher", adopt)

    def invoke(*args):
        module.main(["--site", str(site), "--evaluation", str(evaluation), "--name", "memory_fv", *args])
        return json.loads((repo / "hpc/generated/memory_fv/plan.json").read_text())

    return SimpleNamespace(module=module, repo=repo, cfg=cfg, source=source,
                           evaluation=evaluation, site=site, invoke=invoke)


def test_dry_run_inherits_full_teacher_contract_and_creates_only_cache_array(launcher, monkeypatch):
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm in dry run"))
    plan = launcher.invoke()
    assert plan["array"] == "0-15%4" and plan["task_count"] == 16
    assert plan["gres"] == "gpu:1" and plan["gpus_per_task"] == 1 and plan["partition"] == "gpuq"
    assert plan["cpus"] == 8 and plan["threads"] == 4 and plan["mem"] == "64G"
    assert plan["walltime"] == "24:00:00"
    assert plan["run_root"] == str(launcher.repo / "runs/memory_fv")
    assert Path(plan["checkpoint"]).read_bytes() == launcher.source.read_bytes()
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    assert cfg["model"] == launcher.cfg["model"]
    assert cfg["training"] == launcher.cfg["training"]
    assert cfg["model"]["proprioception_mode"] == "none" and cfg["model"]["execute"] == 2
    assert cfg["data"]["observation_stride"] == cfg["model"]["action_stride"] == 2
    assert cfg["data"]["alignment_path"] == str(Path(cfg["paths"]["data_root"]) / "alignment.json")
    assert cfg["checkpoints"]["teacher"] == plan["checkpoint"]
    script = Path(plan["script"]).read_text()
    assert "--exclusive" not in script and "torch.distributed" not in script
    assert "#SBATCH --gres=gpu:1" in script and "#SBATCH --array=0-15%4" in script
    assert "#SBATCH --account=fixture-account" in script
    assert "--task-index" in script and "--device cuda" in script
    assert "--gpu-bind=none" in script and "--cpu-bind=cores" in script
    assert Path(plan["logs"]).is_dir()
    assert not Path(plan["completion_record"]).exists()


def test_array_script_quotes_paths_and_has_unique_stop_files(launcher, monkeypatch, tmp_path):
    plan = launcher.invoke()
    subprocess.run(["bash", "-n", plan["script"]], check=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    srun = bin_dir / "srun"
    srun.write_text(f"#!{sys.executable}\nimport json,os,sys\n"
                    "print(json.dumps({'args':sys.argv[1:], 'stop':os.environ['STMEM_STOP_FILE'],"
                    "'rank':os.environ.get('RANK'),'scratch':os.environ.get('STMEM_SCRATCH_ROOT')}))\n")
    srun.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "123")
    monkeypatch.setenv("SLURM_JOB_ID", "130")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("STMEM_SCRATCH_ROOT", "/unrelated/scratch")
    result = subprocess.run(["bash", plan["script"]], capture_output=True, text=True, check=True)
    d = json.loads(result.stdout)
    assert d["args"][-2:] == ["--task-index", "7"]
    assert d["stop"].endswith("/stops/123_7_130.stop")
    assert d["rank"] is None and d["scratch"] is None


def test_submission_resume_guards_active_array_and_preserves_config(launcher, monkeypatch):
    calls = []
    def submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"{70000+len(calls)};cluster\n")
    monkeypatch.setattr(launcher.module.subprocess, "run", submit)
    first = launcher.invoke("--submit")
    assert calls == [["sbatch", "--parsable", first["script"]]]
    with pytest.raises(SystemExit):
        launcher.invoke("--submit")
    monkeypatch.setattr(launcher.module, "active_array_ids", lambda: {"70001"})
    with pytest.raises(SystemExit):
        launcher.invoke("--submit", "--resume")
    assert len(calls) == 1
    monkeypatch.setattr(launcher.module, "active_array_ids", set)
    second = launcher.invoke("--submit", "--resume", "--walltime", "48:00:00", "--max-concurrent", "2")
    assert len(calls) == 2 and second["signature"] == first["signature"]
    assert second["array"] == "0-15%2"
    assert second["config_sha256"] == first["config_sha256"]


def test_active_query_tracks_array_base_ids_and_unknown_states(launcher, monkeypatch):
    def query(command, **kwargs):
        assert "--format=%F|%T" in command
        assert "SQUEUE_STATES" not in kwargs["env"]
        return SimpleNamespace(stdout="70001|COMPLETED\n70001|RUNNING\n70002|SUSPENDED\n70003|FAILED\n")
    monkeypatch.setenv("SQUEUE_STATES", "PENDING")
    monkeypatch.setattr(launcher.module.subprocess, "run", query)
    assert launcher.module.active_array_ids() == {"70001", "70002"}


@pytest.mark.parametrize("change", ["checkpoint", "config", "source"])
def test_rejects_changed_named_inputs(launcher, change):
    plan = launcher.invoke()
    if change == "checkpoint":
        state = torch.load(launcher.source, weights_only=False)
        state["step"] += 1
        torch.save(state, launcher.source)
    elif change == "config":
        Path(plan["config"]).write_text("changed: true\n")
    else:
        (launcher.repo / "src/fixture.py").write_text("# changed scientific source\n")
    with pytest.raises(SystemExit):
        launcher.invoke()


@pytest.mark.parametrize("arguments", [("--threads", "9"), ("--max-concurrent", "0"),
                                       ("--mem", "0"), ("--partition", ""), ("--walltime", "00:00:00")])
def test_bad_resource_settings_never_adopt_teacher(launcher, arguments):
    with pytest.raises(SystemExit):
        launcher.invoke(*arguments)
    assert not (launcher.repo / "runs/memory_fv").exists()


def test_explicit_paths_rebased_without_changing_visual_identity(launcher):
    state = torch.load(launcher.source, weights_only=False)
    state["config"]["data"]["alignment_path"] = str(Path(launcher.cfg["paths"]["data_root"]) / "alignment.json")
    spec = state["config"]["model"]["visual_features"]
    spec["checkpoint"] = str(Path(launcher.cfg["paths"]["cache_root"]) / "backbones/clip_RN50.pt")
    spec["cache_dir"] = str(Path(launcher.cfg["paths"]["data_root"]) / "features_clip_rn50_layer1_v1")
    site = launcher.module.read_site(launcher.site)
    cfg = launcher.module.cache_config(state, launcher.repo / "selected.pt", site, launcher.repo, launcher.repo / "runs/new")
    assert cfg["data"]["alignment_path"] == str(Path(site["STMEM_DATA_ROOT"]) / "alignment.json")
    assert cfg["model"]["visual_features"]["checkpoint"].startswith(site["STMEM_CACHE_ROOT"])
    assert cfg["model"]["visual_features"]["cache_dir"].startswith(site["STMEM_DATA_ROOT"])
    assert state["config"]["paths"] == launcher.cfg["paths"]


def test_render_reuses_config_without_rewriting_it(launcher):
    plan = launcher.invoke()
    before = Path(plan["config"]).stat().st_mtime_ns
    launcher.invoke("--resume", "--dry-run", "--walltime", "48:00:00")
    assert Path(plan["config"]).stat().st_mtime_ns == before


def test_startup_integrity_guards_work_under_python_optimization(launcher, monkeypatch):
    plan = launcher.invoke()
    Path(plan["config"]).write_text("tampered: true\n")
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    result = subprocess.run(["bash", plan["script"]], text=True, capture_output=True)
    assert result.returncode != 0 and "Cache configuration changed" in result.stderr


def test_status_uses_saved_paths_despite_inherited_training_environment(launcher, monkeypatch, capsys):
    plan = launcher.invoke()
    monkeypatch.setenv("STMEM_RUN_ROOT", "/unrelated/training")
    original_run = subprocess.run
    def no_slurm(command, **kwargs):
        assert command[0] not in {"squeue", "sbatch"}, "Slurm in status"
        return original_run(command, **kwargs)
    monkeypatch.setattr(launcher.module.subprocess, "run", no_slurm)
    capsys.readouterr()
    launcher.module.main(["--site", str(launcher.site), "--name", "memory_fv", "--status"])
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "not_started"
    assert value["location_record"].startswith(plan["run_root"])
