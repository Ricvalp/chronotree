"""CPU-only router continuation launcher checks with synthetic serialized fixtures."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import yaml

from stmem.config import load_config, run_dir
from stmem.training.checkpoints import capture_rng


REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("STMEM_", "SLURM_", "WANDB_")):
            monkeypatch.delenv(name)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec = importlib.util.spec_from_file_location("router_overfit_launcher", REPO / "hpc/overfit_router.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "repository with spaces"
    (repo / "src").mkdir(parents=True)
    (repo / "src/fixture.py").write_text("# Synthetic diagnostic fixture only.\n")
    generated = repo / "hpc/generated/memory_fv_seed0_routers"
    generated.mkdir(parents=True)
    original_root = repo / "runs/memory_fv_seed0_routers"
    cfg = load_config(REPO / "configs/router_tree.yaml")
    cfg["stage"] = "router"
    cfg["reader"] = "tree"
    cfg["paths"] = {"run_root": str(original_root / "velocity"), "data_root": str(tmp_path / "data"),
                    "cache_root": str(tmp_path / "cache")}
    cfg["training"].update(steps=5000, schedule_steps=5000, lr=0.0003, effective_batch=256, microbatch=256,
                           checkpoint_every=500, checkpoint_steps=[5000, 10000], log_every=100)
    cfg["labels"].update(objective="velocity", integration_steps=8, use_refresh=False, contexts=2)
    label_path = run_dir(cfg, "router") / "labels_initial/shard_00000.h5"
    label_path.parent.mkdir(parents=True)
    width = cfg["model"]["width"]
    ident = {"schema_version": 2, "teacher": "teacher_hash", "compact": "compact_hash", "cache": "cache_hash",
             "data": "synthetic_fixture_hash", "reader": "tree", "objective": "velocity", "integration_steps": 8,
             "partition": {"fixture/train": "train", "fixture/validation": "validation"},
             "source_sha256": "old_label_generation_source", "sample_plan": "fixture_plan",
             "contexts": 2, "shard_index": 0, "shard_count": 1, "refresh": False, "router": None}
    with h5py.File(label_path, "w") as f:
        f.attrs["complete"] = True
        f.attrs["identity"] = json.dumps(ident)
        f["query"] = np.zeros((2, 2 * width + 27), dtype="f4")
        f["candidate"] = np.ones((2, 2 * width + 23), dtype="f4")
        f["gain"] = np.array([0.1, -0.2], dtype="f4")
        f["partition"] = np.array([0, 1], dtype="i1")
        f["context_id"] = np.array([0, 1], dtype="i8")
        f["attempted_context_ids"] = np.array([0, 1], dtype="i8")
        f["budget"] = np.array([128, 128], dtype="i8")
        f.create_dataset("episode", data=["fixture/train", "fixture/validation"], dtype=h5py.string_dtype())
    cfg["labels"]["initial_shards"] = [str(label_path)]
    cfg["labels"]["refresh_shards"] = [str(label_path.with_name("must_not_be_loaded.h5"))]
    _, _, _, _, identity = module._load_labels(cfg)
    state = {"stage": "router", "step": 5000, "reader": "tree", "objective": "velocity", "integration_steps": 8,
             "config": deepcopy(cfg), "model": {"synthetic_fixture": torch.ones(1)},
             "optimizer": {"state": {}, "param_groups": [{"lr": 0.0003, "weight_decay": cfg["training"]["weight_decay"]}]},
             "rng_by_rank": [capture_rng()], "label_hashes": [module.sha256(label_path)],
             "label_identity": identity, "target_scale": 0.1}
    checkpoint = run_dir(cfg, "router") / "step_00005000.pt"
    torch.save(state, checkpoint)
    config_path = generated / "initial_velocity_tree.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    plan = {"run_root": str(original_root), "teacher_sha256": "teacher_hash", "compact_sha256": "compact_hash",
            "cache_manifest_digest": "cache_hash", "configs": {"initial_velocity_tree": {
                "path": str(config_path), "sha256": module.sha256(config_path)}}}
    plan_path = generated / "plan.json"
    plan_path.write_text(json.dumps(plan))
    site = repo / "hpc/site_b200_4gpu.env"
    site.write_text(f"STMEM_REPO_ROOT='{repo}'\nSTMEM_PYTHON='/example/python'\nSTMEM_RUN_ROOT='/ignored/site/root'\n")

    def invoke(*arguments):
        return module.main(["--site", str(site), *arguments])

    return SimpleNamespace(module=module, repo=repo, site=site, original_plan=plan_path, original_config=config_path,
                           source=checkpoint, cfg=cfg, labels=label_path, invoke=invoke)


def test_dry_run_preserves_inputs_and_only_extends_initial_fit(launcher, monkeypatch):
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm in dry run"))
    before = {p: p.read_bytes() for p in (launcher.source, launcher.labels, launcher.original_plan, launcher.original_config)}
    plan = launcher.invoke()
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    assert plan["name"] == "memory_fv_seed0_routers_overfit_tree_velocity_100k"
    assert (plan["start_step"], plan["steps"], plan["additional_steps"]) == (5000, 100000, 95000)
    assert (plan["gres"], plan["gpus"], plan["cpus"], plan["mem"], plan["walltime"]) == ("gpu:1", 1, 4, "16G", "01:00:00")
    assert plan["partition"] == "gpuq" and plan["initial_labels_only"]
    assert plan["wandb_project"] == cfg["tracking"]["project"] == "chronotree-routers"
    assert cfg["training"]["steps"] == cfg["training"]["schedule_steps"] == 100000
    assert cfg["training"]["router_diagnostics"] is True
    assert cfg["training"]["log_every"] == 1000
    assert cfg["training"]["checkpoint_steps"] == [10000, 25000, 50000, 100000]
    for key in ("effective_batch", "lr", "weight_decay", "grad_clip"):
        assert cfg["training"][key] == launcher.cfg["training"][key]
    for key in ("model", "data", "labels", "geometry"):
        assert cfg[key] == launcher.cfg[key]
    assert cfg["labels"]["use_refresh"] is False
    assert cfg["paths"]["run_root"] != launcher.cfg["paths"]["run_root"]
    assert Path(plan["checkpoint"]).read_bytes() == before[launcher.source]
    assert all(p.read_bytes() == content for p, content in before.items())
    assert launcher.module.ready_plan(plan["plan_path"]) == plan
    assert launcher.invoke() == plan


def test_script_syntax_and_single_gpu_resume(launcher):
    plan = launcher.invoke()
    subprocess.run(["bash", "-n", plan["script"]], check=True)
    script = Path(plan["script"]).read_text()
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --partition=gpuq" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "--exclusive" not in script and "--array" not in script
    assert "-m stmem.train --stage router" in script
    assert '--resume "$resume_checkpoint"' in script
    assert "run_job.sh" in script and '--stop-file "$STMEM_STOP_FILE"' in script
    assert "WANDB_RUN_ID" in script and "STMEM_RUN_ROOT=" in script
    assert "export WANDB_PROJECT=chronotree-routers" in script
    assert "ready_plan" in script
    assert "stmem.training.label_experiment" not in script


@pytest.mark.parametrize("change", ["stage", "step", "reader", "objective", "labels", "identity", "refresh", "optimizer", "lr"])
def test_incompatible_input_rejected(launcher, change):
    state = torch.load(launcher.source, weights_only=False)
    if change in ("stage", "reader", "objective"):
        state[change] = "wrong"
    elif change == "step":
        state["step"] = 10000
    elif change == "labels":
        state["label_hashes"] = ["wrong"]
    elif change == "identity":
        state["label_identity"]["teacher"] = "wrong"
    elif change == "refresh":
        state["config"]["labels"]["use_refresh"] = True
    elif change == "lr":
        state["optimizer"]["param_groups"][0]["lr"] = 0.1
    else:
        state["optimizer"] = None
    torch.save(state, launcher.source)
    with pytest.raises(SystemExit):
        launcher.invoke()


@pytest.mark.parametrize("change", ["labels", "snapshot", "config", "source", "plan"])
def test_startup_guard_rejects_changed_bytes(launcher, change):
    plan = launcher.invoke()
    path = {"labels": launcher.labels, "snapshot": Path(plan["checkpoint"]), "config": Path(plan["config"]),
            "source": launcher.repo / "src/fixture.py", "plan": Path(plan["plan_path"])}[change]
    if change == "plan":
        changed = json.loads(path.read_text())
        changed["steps"] += 1
        path.write_text(json.dumps(changed))
    else:
        path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError):
        launcher.module.ready_plan(plan["plan_path"])


def test_named_run_cannot_overwrite_original_or_change_total(launcher):
    with pytest.raises(SystemExit):
        launcher.invoke("--name", "memory_fv_seed0_routers")
    with pytest.raises(SystemExit):
        launcher.invoke("--steps", "5000")
    plan = launcher.invoke()
    with pytest.raises(SystemExit):
        launcher.invoke("--name", plan["name"], "--steps", "50000")
    assert launcher.module.read_json(plan["plan_path"]) == plan


def test_submit_guard_and_safe_restart(launcher, monkeypatch):
    calls = []

    def fake_submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"{8000 + len(calls)}\n")

    monkeypatch.setattr(launcher.module.subprocess, "run", fake_submit)
    plan = launcher.invoke("--submit")
    assert calls == [["sbatch", "--parsable", plan["script"]]]
    with pytest.raises(SystemExit):
        launcher.invoke("--submit")
    monkeypatch.setattr(launcher.module, "active_user_job_ids", lambda: {"8001"})
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--submit")
    assert len(calls) == 1
    monkeypatch.setattr(launcher.module, "active_user_job_ids", lambda: set())
    launcher.invoke("--resume", "--submit")
    assert len(calls) == 2


def test_uncertain_submission_refuses_retry(launcher, monkeypatch):
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(SystemExit):
        launcher.invoke("--submit")
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Duplicate submit"))
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--submit")


def test_status_is_read_only_and_does_not_contact_slurm(launcher, monkeypatch):
    plan = launcher.invoke()
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm in status"))
    before = {p: p.read_bytes() for p in Path(plan["generated"]).rglob("*") if p.is_file()}
    result = launcher.invoke("--status")
    assert result["completed_steps"] == 5000 and result["status"] == "incomplete"
    assert before == {p: p.read_bytes() for p in Path(plan["generated"]).rglob("*") if p.is_file()}


def test_diagnostic_resume_rejects_foreign_last_checkpoint(launcher):
    plan = launcher.invoke()
    directory = Path(plan["run_directory"])
    directory.mkdir(parents=True)
    torch.save(torch.load(launcher.source, weights_only=False), directory / "last.pt")
    with pytest.raises(ValueError, match="different work"):
        launcher.module.ready_plan(plan["plan_path"])


def test_completed_diagnostic_checks_checkpoint_before_skipping_submit(launcher, monkeypatch):
    plan = launcher.invoke()
    directory = Path(plan["run_directory"])
    directory.mkdir(parents=True)
    state = torch.load(launcher.source, weights_only=False)
    state.update(step=plan["steps"], config=yaml.safe_load(Path(plan["config"]).read_text()))
    checkpoint = directory / "last.pt"
    torch.save(state, checkpoint)
    complete = directory / "complete.json"
    complete.write_text(json.dumps({"status": "complete", "completed_steps": plan["steps"],
                                    "checkpoint_sha256": "wrong"}))
    assert launcher.module.status(plan)["status"] == "incomplete"
    complete.write_text(json.dumps({"status": "complete", "completed_steps": plan["steps"],
                                    "checkpoint_sha256": launcher.module.sha256(checkpoint)}))
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm after completion"))
    launcher.invoke("--resume", "--submit")
    assert launcher.invoke("--status")["status"] == "complete"


@pytest.mark.parametrize("project", [None, "custom-router-diagnostic"])
def test_project_is_separate_and_overrides_site_and_environment(launcher, monkeypatch, project):
    monkeypatch.setenv("WANDB_PROJECT", "ambient-pretraining")
    launcher.site.write_text(launcher.site.read_text() + "WANDB_PROJECT=chronotree\n")
    before = launcher.source.read_bytes(), launcher.original_config.read_bytes()
    plan = launcher.invoke(*(["--wandb-project", project] if project else []))
    expected = project or "chronotree-routers"
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    assert plan["wandb_project"] == cfg["tracking"]["project"] == expected
    script = Path(plan["script"]).read_text()
    assert f"export WANDB_PROJECT={expected}\n" in script
    assert "ambient-pretraining" not in script and "export WANDB_PROJECT=chronotree\n" not in script
    assert (launcher.source.read_bytes(), launcher.original_config.read_bytes()) == before
