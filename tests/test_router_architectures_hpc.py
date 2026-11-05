"""CPU-only orchestration checks; serialized fixtures are not RoboMME results."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
import yaml

from stmem.config import load_config

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith(("STMEM_", "SLURM_", "WANDB_")):
            monkeypatch.delenv(key)
    spec = importlib.util.spec_from_file_location("router_architecture_launcher", REPO / "hpc/router_architectures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "repository with spaces"
    (repo / "src").mkdir(parents=True)
    (repo / "src/fixture.py").write_text("# Synthetic launcher fixture.\n")
    (repo / "hpc").mkdir()
    source_root = repo / "runs/original"
    source_root.mkdir(parents=True)
    original_plan = source_root / "plan.json"
    original_plan.write_text(json.dumps({"run_root": str(source_root)}))
    originals = {}
    for reader in ("tree", "flat"):
        cfg = load_config(REPO / "configs/router_tree.yaml")
        cfg["reader"] = reader
        cfg["paths"] = {"run_root": str(source_root), "cache_root": str(tmp_path / "cache"),
                        "data_root": str(tmp_path / "data")}
        cfg["labels"].update(objective="final_action", integration_steps=8, use_refresh=False, contexts=16)
        cfg["training"]["steps"] = 5000
        shards = []
        for index in range(4):
            p = source_root / f"{reader}_labels_{index}.h5"
            p.write_bytes(f"synthetic launcher bytes {reader} {index}".encode())
            shards.append({"path": str(p), "sha256": module.sha256(p)})
        cfg["labels"]["initial_shards"] = [row["path"] for row in shards]
        checkpoint = source_root / f"{reader}.pt"
        torch.save({"stage": "router", "step": 5000, "synthetic_launcher_fixture": True}, checkpoint)
        identity = {"reader": reader, "teacher": "teacher", "compact": "compact", "cache": "cache"}
        originals[reader] = (cfg, {}, checkpoint, shards, identity, original_plan)

    def fixture_source(_repo, _experiment, reader, objective):
        assert objective in ("final_action", "velocity")
        result = deepcopy(originals[reader])
        result[0]["labels"]["objective"] = objective
        return result

    monkeypatch.setattr(module, "initial_checkpoint", fixture_source)
    site = repo / "hpc/site_b200_4gpu.env"
    site.write_text(f"STMEM_REPO_ROOT='{repo}'\nSTMEM_PYTHON='/example/python'\n"
                    "STMEM_RUN_ROOT='/ignored/old/run'\nWANDB_PROJECT=chronotree\n")
    return SimpleNamespace(module=module, repo=repo, originals=originals, site=site,
                           invoke=lambda *args: module.main(["--site", str(site), *args]))


def test_default_suite_fresh_fits_and_fixed_inputs(launcher, monkeypatch):
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm in dry run"))
    before = {p: p.read_bytes() for p in (launcher.repo / "runs/original").iterdir()}
    plan = launcher.invoke()
    assert plan["training_job_count"] == 8
    assert plan["job_count"] == 12
    assert plan["maximum_concurrent_allocations"] == 8
    assert plan["fresh_initialization"] and plan["initial_labels_only"]
    assert plan["wandb_project"] == "chronotree-routers"
    assert plan["gres"] == "gpu:1" and plan["partition"] == "gpuq"
    for job in plan["jobs"]:
        if job["kind"] != "train":
            continue
        cfg = yaml.safe_load(Path(job["config"]).read_text())
        assert cfg["model"]["router_architecture"] == job["architecture"]
        assert cfg["training"]["steps"] == 100000 and cfg["training"]["router_fit_contexts"] == 0
        assert cfg["training"]["router_diagnostics"]
        assert cfg["tracking"]["project"] == "chronotree-routers"
        assert cfg["labels"]["use_refresh"] is False
        assert cfg["labels"]["initial_shards"] == launcher.originals[job["reader"]][0]["labels"]["initial_shards"]
        assert ("router_inputs" in cfg["labels"]) == (job["architecture"] in launcher.module.RICH)
        assert cfg["checkpoints"]["teacher"].startswith(str(launcher.repo / "runs/original"))
        assert cfg["paths"]["run_root"] != launcher.originals[job["reader"]][0]["paths"]["run_root"]
        assert job["deps"] == (["inputs"] if job["architecture"] in launcher.module.RICH else [])
    assert all(p.read_bytes() == content for p, content in before.items())
    assert launcher.module.ready_plan(plan["plan_path"]) == plan
    assert launcher.invoke() == plan


def test_all_generated_scripts_parse_and_request_untyped_gpu(launcher):
    plan = launcher.invoke()
    for job in plan["jobs"]:
        subprocess.run(["bash", "-n", job["script"]], check=True)
        script = Path(job["script"]).read_text()
        assert "#SBATCH --gres=gpu:1" in script
        assert "--exclusive" not in script and "gpu:h200" not in script and "gpu:b200" not in script
        assert "export WANDB_PROJECT=chronotree-routers" in script
        assert "ready_plan" in script
        assert "stmem.training.label_experiment" not in script
        if job["kind"] == "inputs":
            assert "#SBATCH --array=0-3%4" in script
            subprocess.run(["bash", "-n", job["runner"]], check=True)
            runner = Path(job["runner"]).read_text()
            assert runner.count("-m stmem.training.router_inputs") == 2
            assert "--stop-file" in runner and "run_job.sh" in script
        else:
            assert "resume_args=()" in script
            assert '"${resume_args[@]}"' in script
            assert "initial_tree.pt" not in script and "initial_flat.pt" not in script
            assert "--stop-file" in script and "run_job.sh" in script
            assert ("#SBATCH --kill-on-invalid-dep=yes" in script) == bool(job["deps"])


def test_only_lightweight_variants_need_no_preparation(launcher):
    plan = launcher.invoke("--variants", "dot", "mlp", "--readers", "flat")
    assert plan["training_job_count"] == plan["job_count"] == 2
    assert plan["input_readers"] == []
    assert all(job["kind"] == "train" and not job["deps"] for job in plan["jobs"])
    assert launcher.module.input_shards_complete(plan)


def test_small_fit_is_explicit_and_immutable(launcher):
    plan = launcher.invoke("--name", "router_tiny_fit", "--fit-contexts", "256", "--steps", "10000")
    assert plan["fit_contexts"] == 256
    for job in plan["jobs"]:
        if job["kind"] == "train":
            cfg = yaml.safe_load(Path(job["config"]).read_text())
            assert cfg["training"]["router_fit_contexts"] == 256 and cfg["training"]["steps"] == 10000
    with pytest.raises(SystemExit):
        launcher.invoke("--name", "router_tiny_fit", "--fit-contexts", "128", "--steps", "10000")


@pytest.mark.parametrize("change", ["source", "label", "config", "snapshot", "plan"])
def test_startup_rejects_changed_work(launcher, change):
    plan = launcher.invoke()
    p = {"source": launcher.repo / "src/fixture.py",
         "label": Path(plan["readers"]["tree"]["labels"][0]["path"]),
         "config": Path(plan["configs"]["tree_mlp"]["path"]),
         "snapshot": Path(plan["frozen_inputs"][0]["path"]),
         "plan": Path(plan["plan_path"])}[change]
    if change == "plan":
        content = json.loads(p.read_text())
        content["steps"] += 1
        p.write_text(json.dumps(content))
    else:
        p.write_bytes(p.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        launcher.module.ready_plan(plan["plan_path"])


def test_submit_dependencies_and_resume_without_duplicates(launcher, monkeypatch):
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"{9000 + len(calls)}\n")

    monkeypatch.setattr(launcher.module.subprocess, "run", submit)
    plan = launcher.invoke("--submit")
    assert len(calls) == 9
    assert calls[0] == ["sbatch", "--parsable", plan["jobs"][0]["script"]]
    for job, command in zip(plan["jobs"][1:], calls[1:]):
        assert ("--dependency=afterok:9001" in command) == bool(job["deps"])
    with pytest.raises(SystemExit):
        launcher.invoke("--submit")
    monkeypatch.setattr(launcher.module, "active_user_job_ids", lambda: {str(i) for i in range(9001, 9010)})
    launcher.invoke("--resume", "--submit")
    assert len(calls) == 9
    monkeypatch.setattr(launcher.module, "active_user_job_ids", lambda: set())
    launcher.invoke("--resume", "--submit")
    assert len(calls) == 18
    assert "--dependency=afterok:9010" in next(command for command in calls[10:] if any("dependency" in s for s in command))


def test_uncertain_submit_is_not_retried(launcher, monkeypatch):
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(SystemExit):
        launcher.invoke("--submit")
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Duplicate submit"))
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--submit")


def test_failed_array_waits_for_dependent_cancellation(launcher, monkeypatch):
    calls = []
    def fake_submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"{7000 + len(calls)}\n")
    monkeypatch.setattr(launcher.module.subprocess, "run", fake_submit)
    plan = launcher.invoke("--submit")
    rich_id = str(7000 + next(i + 1 for i, job in enumerate(plan["jobs"]) if job["deps"]))
    monkeypatch.setattr(launcher.module, "active_user_job_ids", lambda: {rich_id})
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--submit")
    assert len(calls) == 9


def test_status_readonly(launcher, monkeypatch):
    plan = launcher.invoke()
    root = Path(plan["generated"])
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm in status"))
    result = launcher.invoke("--status")
    assert result["status"] == "incomplete" and not result["inputs_complete"]
    assert len(result["jobs"]) == 8
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_foreign_resume_checkpoint_rejected(launcher):
    plan = launcher.invoke("--variants", "mlp", "--readers", "tree")
    job = plan["jobs"][0]
    p = Path(job["run_directory"]) / "last.pt"
    p.parent.mkdir(parents=True)
    torch.save({"stage": "teacher", "step": 100}, p)
    with pytest.raises(ValueError, match="different work"):
        launcher.module.ready_plan(plan["plan_path"], job["name"])


def test_unsupported_flat_children_only_rejected(launcher):
    with pytest.raises(SystemExit):
        launcher.invoke("--readers", "flat", "--variants", "children")
