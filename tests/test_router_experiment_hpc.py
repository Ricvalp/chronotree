"""CPU-only orchestration invariants; all serialized inputs are synthetic fixtures."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py
import pytest
import torch
import yaml

from stmem.config import digest
from test_compact_hpc import launcher as cache_launcher  # noqa: F401

REPO = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


@pytest.fixture
def launcher(cache_launcher, monkeypatch):  # noqa: F811
    fixture = cache_launcher
    spec = importlib.util.spec_from_file_location("router_launcher", REPO / "hpc/train_routers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The shared fixture validates actual cache provenance. Add tiny frozen state
    # tensors to exercise compact/teacher identity without allocating a policy.
    original = torch.load(fixture.teacher, weights_only=False)
    original["model"] = {"encoder.weight": torch.ones(2), "text_encoder.weight": torch.ones(3),
                         "normalizer.mean": torch.zeros(7), "decoder.weight": torch.ones(1)}
    torch.save(original, fixture.teacher)
    teacher_hash = module.sha256(fixture.teacher)
    selection = json.loads(fixture.selection.read_text())
    selection["sha256"] = teacher_hash
    write_json(fixture.selection, selection)
    from stmem.data.cache import cache_identity
    old_root = fixture.cache_root
    identity = cache_identity(fixture.cfg, fixture.teacher, {"synthetic_launcher_fixture": True})
    cache_root = Path(fixture.cfg["paths"]["cache_root"]) / digest(identity)[:24]
    old_root.rename(cache_root)
    manifest = json.loads((cache_root / "manifest.json").read_text())
    manifest["identity"] = identity
    write_json(cache_root / "manifest.json", manifest)
    for receipt in cache_root.glob("*.complete.json"):
        obj = json.loads(receipt.read_text())
        obj["identity_sha256"] = digest(identity)
        write_json(receipt, obj)
    complete = json.loads(fixture.complete.read_text())
    complete.update(cache_path=str(cache_root), teacher_sha256=teacher_hash,
                    identity_sha256=digest(identity), cache_manifest_sha256=module.sha256(cache_root / "manifest.json"))
    write_json(fixture.complete, complete)
    cache_plan = json.loads(fixture.cache_plan.read_text())
    cache_plan.update(checkpoint_sha256=teacher_hash, selection_sha256=module.sha256(fixture.selection))
    write_json(fixture.cache_plan, cache_plan)
    hierarchy = cache_root.with_name(cache_root.name + "-hierarchy")
    hierarchy.mkdir()
    hierarchy_manifest = {"synthetic_launcher_fixture": True, "leaf_cache": digest(manifest)}
    write_json(hierarchy / "manifest.json", hierarchy_manifest)
    monkeypatch.setattr(module, "inspect_hierarchy", lambda _cache, root: json.loads((Path(root) / "manifest.json").read_text()))
    compact = fixture.repo / "runs/memory_fv_seed0_compact_hierarchy/seed0/compact/last.pt"
    compact.parent.mkdir(parents=True)
    compact_cfg = deepcopy(fixture.cfg)
    compact_cfg["stage"] = "compact"
    compact_cfg["paths"]["run_root"] = str(compact.parents[2])
    compact_cfg["training"]["hierarchy_cache_root"] = str(hierarchy)
    compact_state = {**original, "stage": "compact", "step": 60000, "config": compact_cfg,
                     "provenance": {"dependencies": {"teacher": teacher_hash, "cache": digest(manifest),
                                                     "hierarchy": digest(hierarchy_manifest)}}}
    torch.save(compact_state, compact)

    def invoke(*arguments):
        return module.main(["--site", str(fixture.site), *arguments])

    return SimpleNamespace(module=module, repo=fixture.repo, site=fixture.site, teacher=fixture.teacher,
                           compact=compact, compact_cfg=compact_cfg, cache_plan=fixture.cache_plan,
                           hierarchy=hierarchy, invoke=invoke)


def test_dry_run_freezes_checkpoint_and_builds_matched_four_router_dag(launcher, monkeypatch):
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm in dry-run"))
    before = launcher.compact.read_bytes(), launcher.teacher.read_bytes()
    plan = launcher.invoke()
    assert plan["job_count"] == 10 and plan["allocation_count"] == 16
    assert plan["maximum_concurrent_allocations"] == 4
    assert plan["gres"] == "gpu:1" and plan["partition"] == "gpuq"
    assert plan["shards"] == 4 and plan["cpus"] == 16 and plan["mem"] == "32G"
    assert plan["compact_step"] == 60000
    assert Path(plan["compact"]).read_bytes() == before[0]
    selection = json.loads(Path(plan["compact_selection"]).read_text())
    assert selection["selection_method"] == "user_designated_checkpoint"
    assert selection["matched_validation_selection"] is False
    labels = yaml.safe_load(Path(plan["configs"]["labels"]["path"]).read_text())
    for field in ("data", "geometry", "model"):
        assert labels[field] == launcher.compact_cfg[field]
    assert labels["paths"]["data_root"] == launcher.compact_cfg["paths"]["data_root"]
    assert labels["paths"]["cache_root"] == launcher.compact_cfg["paths"]["cache_root"]
    assert labels["labels"]["contexts"] == 16000 and labels["labels"]["refresh_contexts"] == 8000
    assert labels["labels"]["objectives"] == ["velocity", "final_action"]
    assert labels["labels"]["integration_steps"] == 8
    assert len(labels["labels"]["refresh_router_checkpoints"]) == 4
    assert all(path.endswith("step_00005000.pt") for path in labels["labels"]["refresh_router_checkpoints"].values())
    for objective in ("velocity", "final_action"):
        for reader in ("tree", "flat"):
            initial = yaml.safe_load(Path(plan["configs"][f"initial_{objective}_{reader}"]["path"]).read_text())
            final = yaml.safe_load(Path(plan["configs"][f"final_{objective}_{reader}"]["path"]).read_text())
            assert initial["training"]["steps"] == 5000
            assert initial["training"]["router_device"] == final["training"]["router_device"] == "cuda"
            assert final["training"]["steps"] == 10000
            assert initial["training"]["effective_batch"] == final["training"]["effective_batch"] == 256
            assert initial["training"]["lr"] == final["training"]["lr"] == pytest.approx(0.0003)
            assert initial["labels"]["objective"] == objective and final["labels"]["use_refresh"]
            assert len(initial["labels"]["initial_shards"]) == len(final["labels"]["refresh_shards"]) == 4
            assert initial["paths"]["run_root"].endswith("velocity" if objective == "velocity" else "final_action_8")
    assert (launcher.compact.read_bytes(), launcher.teacher.read_bytes()) == before
    launcher.module.ready_plan(plan["startup_plan"])


def test_scripts_are_valid_shell_and_use_bounded_independent_allocations(launcher):
    plan = launcher.invoke("--contexts", "32", "--refresh-contexts", "16", "--integration-steps", "4")
    assert plan["evaluation_flow_steps"] == 8
    for job in plan["jobs"]:
        subprocess.run(["bash", "-n", job["script"]], check=True)
        script = Path(job["script"]).read_text()
        assert "#SBATCH --gres=gpu:1" in script and "--exclusive" not in script
        assert "run_job.sh" in script and "--stop-file" in script
        assert "STMEM_RUN_ROOT=" in script and "WANDB_RUN_ID" in script
        if job["kind"] == "labels":
            assert "#SBATCH --array=0-3%4" in script
            assert "--workers 4" in script and "--batch-size 4" in script
            assert "--teacher-token-budget 32768" in script and "--compact-batch-size 64" in script
            assert "--prefetch-mb 512" in script
            assert "--nproc-per-node" not in script
        else:
            assert "#SBATCH --cpus-per-task=4" in script and "#SBATCH --mem=16G" in script
            assert 'resume_args=(--resume ' in script
    repeated = launcher.invoke("--contexts", "32", "--refresh-contexts", "16", "--integration-steps", "4")
    assert repeated == plan


def test_only_submit_contacts_slurm_and_records_entire_dependency_chain(launcher, monkeypatch):
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=str(1000 + len(calls)) + "\n")
    monkeypatch.setattr(launcher.module.subprocess, "run", fake_run)
    plan = launcher.invoke("--submit")
    assert len(calls) == 10
    assert not any("--dependency" in part for part in calls[0])
    assert all("--dependency=afterok:1001" in command for command in calls[1:5])
    assert "--dependency=afterok:1002:1003:1004:1005" in calls[5]
    assert all("--dependency=afterok:1006" in command for command in calls[6:])
    ledger = json.loads((Path(plan["generated"]) / "submitted.json").read_text())
    assert len(ledger["jobs"]) == 10
    monkeypatch.setattr(launcher.module, "active_user_job_ids", lambda: {"1002"})
    with pytest.raises(SystemExit, match="2"):
        launcher.invoke("--resume", "--submit")
    assert len(calls) == 10


def test_uncertain_submission_stops_retry_instead_of_duplicate(launcher, monkeypatch):
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(OSError("network interrupted")))
    with pytest.raises(SystemExit):
        launcher.invoke("--submit")
    monkeypatch.setattr(launcher.module, "active_user_job_ids", lambda: set())
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--submit")


def test_resume_uses_frozen_compact_even_if_source_last_advances(launcher):
    plan = launcher.invoke()
    original = Path(plan["compact"]).read_bytes()
    launcher.compact.write_bytes(b"A new mutable last checkpoint, deliberately not loaded on resume")
    resumed = launcher.invoke("--resume")
    assert resumed == plan and Path(plan["compact"]).read_bytes() == original


@pytest.mark.parametrize("target", ["compact", "teacher", "config", "source", "hierarchy", "selection"])
def test_allocation_guard_rejects_changed_frozen_inputs(launcher, target):
    plan = launcher.invoke()
    path = {"compact": Path(plan["compact"]), "teacher": launcher.teacher,
            "config": Path(plan["configs"]["labels"]["path"]), "source": launcher.repo / "src/fixture.py",
            "hierarchy": launcher.hierarchy / "manifest.json", "selection": Path(plan["compact_selection"])}[target]
    path.write_bytes(path.read_bytes() + b"\nchanged")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        launcher.module.ready_plan(plan["startup_plan"])


@pytest.mark.parametrize("change", ["encoder", "normalizer", "teacher", "action_contract", "data_path"])
def test_incompatible_compact_rejected_before_render_or_submit(launcher, monkeypatch, change):
    state = torch.load(launcher.compact, weights_only=False)
    if change in ("encoder", "normalizer"):
        key = "encoder.weight" if change == "encoder" else "normalizer.mean"
        state["model"][key] += 1
    elif change == "teacher":
        state["provenance"]["dependencies"]["teacher"] = "wrong"
    elif change == "action_contract":
        state["config"]["model"]["horizon"] = 8
    else:
        state["config"]["paths"]["data_root"] = "/unrelated"
    torch.save(state, launcher.compact)
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm with invalid compact"))
    with pytest.raises(SystemExit):
        launcher.invoke("--submit")
    assert not (launcher.repo / "hpc/generated/memory_fv_seed0_routers/plan.json").exists()


def test_predecessor_completion_guard_refuses_empty_labels(launcher):
    plan = launcher.invoke()
    with pytest.raises(ValueError, match="predecessor"):
        launcher.module.ready_job(plan["startup_plan"], "initial_velocity_tree")


def test_immutable_config_and_scientific_settings(launcher):
    plan = launcher.invoke()
    with pytest.raises(SystemExit):
        launcher.invoke("--contexts", "32")
    assert json.loads(Path(plan["plan_path"]).read_text()) == plan


def complete_initial_labels(launcher, plan):
    contexts = plan["contexts_per_reader_objective"]
    for objective in launcher.module.OBJECTIVES:
        for reader in launcher.module.READERS:
            paths = launcher.module.label_paths(Path(plan["run_root"]), plan["seed"], objective, reader,
                                                 "initial", plan["shards"], plan["integration_steps"])
            for shard, name in enumerate(paths):
                path = Path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                ids = list(range(shard, contexts, plan["shards"]))
                identity = {
                    "schema_version": 2, "teacher": plan["teacher_sha256"], "compact": plan["compact_sha256"],
                    "source_sha256": plan["source_sha256"],
                    "cache": plan["cache_manifest_digest"], "reader": reader, "objective": objective,
                    "integration_steps": plan["integration_steps"], "refresh": False, "router": None,
                    "contexts": contexts, "shard_index": shard, "shard_count": plan["shards"],
                    "partition": {"fixture_train": 0, "fixture_validation": 1}, "sample_plan": "synthetic_fixture",
                    "data": {"synthetic_launcher_fixture": True}, "label_sampling": {"seed": 2026},
                }
                with h5py.File(path, "w") as handle:
                    handle.attrs["complete"] = True
                    handle.attrs["identity"] = json.dumps(identity)
                    handle.attrs["commit"] = json.dumps([len(ids), len(ids)])
                    for key in ("query", "candidate", "gain", "partition", "budget", "episode"):
                        handle.create_dataset(key, data=[0] * len(ids))
                    handle.create_dataset("context_id", data=ids)
                    handle.create_dataset("attempted_context_ids", data=ids)


def test_status_and_resume_skip_completed_initial_stages(launcher, monkeypatch):
    plan = launcher.invoke("--contexts", "16", "--refresh-contexts", "8")
    complete_initial_labels(launcher, plan)
    for job in plan["jobs"]:
        if job["kind"] != "router" or job["phase"] != "initial":
            continue
        cfg = yaml.safe_load(Path(plan["configs"][job["config"]]["path"]).read_text())
        directory = launcher.module.run_dir(cfg, "router")
        torch.save({"stage": "router", "step": 5000, "config": cfg}, directory / "step_00005000.pt")
    launcher.module.ready_job(plan["startup_plan"], "labels_refresh")
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Scheduler queried by status"))
    status = launcher.invoke("--status")
    assert status["labels_initial"] == "complete" and status["labels_refresh"] == "incomplete"
    calls = []
    def fake_submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=str(2000 + len(calls)) + "\n")
    monkeypatch.setattr(launcher.module.subprocess, "run", fake_submit)
    launcher.invoke("--contexts", "16", "--refresh-contexts", "8", "--resume", "--submit")
    assert len(calls) == 5
    assert calls[0][-1].endswith("labels_refresh.sbatch")
    assert not any(part.startswith("--dependency") for part in calls[0])
    assert all("--dependency=afterok:2001" in command for command in calls[1:])


@pytest.mark.parametrize("corruption", ["objective", "integration_steps", "sample_plan", "attempted", "rows"])
def test_completed_label_integrity_is_checked_before_stage_release(launcher, corruption):
    plan = launcher.invoke("--contexts", "16", "--refresh-contexts", "8")
    complete_initial_labels(launcher, plan)
    path = launcher.module.label_paths(Path(plan["run_root"]), plan["seed"], "velocity", "tree", "initial", 4, 8)[0]
    with h5py.File(path, "r+") as handle:
        if corruption in ("objective", "integration_steps", "sample_plan"):
            identity = json.loads(handle.attrs["identity"])
            identity[corruption] = "wrong"
            handle.attrs["identity"] = json.dumps(identity)
        elif corruption == "attempted":
            handle["attempted_context_ids"][0] = 9999
        else:
            handle.attrs["commit"] = json.dumps([999, 4])
    with pytest.raises(ValueError):
        launcher.module.ready_job(plan["startup_plan"], "initial_velocity_tree")



def test_startup_plan_rejects_configuration_hash_rewrite(launcher):
    plan = launcher.invoke()
    config_path = Path(plan["configs"]["labels"]["path"])
    cfg = yaml.safe_load(config_path.read_text())
    cfg["labels"]["contexts"] += 1
    config_path.write_text(yaml.safe_dump(cfg))
    plan["configs"]["labels"]["sha256"] = launcher.module.sha256(config_path)
    write_json(Path(plan["startup_plan"]), plan)
    with pytest.raises(ValueError, match="startup plan"):
        launcher.module.ready_plan(plan["startup_plan"])
