"""CPU-only launch/integrity checks; serialized fixtures are synthetic diagnostics."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from stmem.config import digest, load_config
from stmem.data.cache import cache_identity

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith(("STMEM_", "SLURM_", "WANDB_")):
            monkeypatch.delenv(name)
    spec = importlib.util.spec_from_file_location("compact_launcher", REPO / "hpc/train_compact.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "repository with spaces"
    (repo / "src").mkdir(parents=True)
    (repo / "src/fixture.py").write_text("# Synthetic launcher fixture, no training.\n")
    generated = repo / "hpc/generated/memory_fv_seed0"
    generated.mkdir(parents=True)
    run_root = repo / "runs/memory_fv_seed0"
    teacher = run_root / "seed0/teacher/selected.pt"
    teacher.parent.mkdir(parents=True)
    cfg = load_config(REPO / "configs/teacher_fv_no_prop_10hz.yaml")
    cfg["paths"] = {
        "run_root": str(run_root), "data_root": str(tmp_path / "data"),
        "cache_root": str(tmp_path / "cache"),
    }
    cfg["training"].update(steps=100000, schedule_steps=100000, lr=0.001)
    cfg["checkpoints"]["teacher"] = str(teacher)
    torch.save({"stage": "teacher", "step": 104110, "config": cfg,
                "model": {"synthetic_fixture": torch.zeros(1)}}, teacher)
    selection = teacher.with_suffix(".selection.json")
    write_json(selection, {
        "schema": "accepted_teacher_v1", "sha256": module.sha256(teacher), "step": 104110,
        "macro_success": 0.3125,
        "evaluation_protocol": {"execute_actions": 8, "action_stride": 2,
                                "observation_stride": 2, "replan_primitive_steps": 16},
        "synthetic_launcher_fixture": True,
    })
    config_path = generated / "cache.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    identity = cache_identity(cfg, teacher, {"synthetic_launcher_fixture": True})
    cache_root = Path(cfg["paths"]["cache_root"]) / digest(identity)[:24]
    cache_root.mkdir(parents=True)
    shards = {}
    for task in cfg["data"]["tasks"]:
        shard = cache_root / f"{task}.h5"
        shard.write_bytes(f"Synthetic bytes for {task}; not a training cache".encode())
        shards[shard.name] = {"sha256": module.sha256(shard), "bytes": shard.stat().st_size,
                              "episodes": cfg["data"]["episodes_per_task"]}
        write_json(cache_root / f"{task}.complete.json", {
            "kind": "encoder_cache_task", "status": "complete", "task": task,
            "identity_sha256": digest(identity), **shards[shard.name],
        })
    manifest_path = cache_root / "manifest.json"
    write_json(manifest_path, {"identity": identity, "status": "complete",
                               "tasks": cfg["data"]["tasks"], "shards": shards})
    complete = run_root / "seed0/cache/complete.json"
    write_json(complete, {"status": "complete", "cache_path": str(cache_root),
                          "cache_manifest_sha256": module.sha256(manifest_path),
                          "teacher_sha256": module.sha256(teacher), "identity_sha256": digest(identity)})
    write_json(complete.with_name("cache_location.json"), {
        "cache_path": str(cache_root), "teacher_sha256": module.sha256(teacher),
        "tasks": cfg["data"]["tasks"],
    })
    cache_plan = {
        "kind": "selected-teacher-encoder-cache", "name": "memory_fv_seed0", "repo": str(repo),
        "python": sys.executable, "run_root": str(run_root),
        "checkpoint": str(teacher), "checkpoint_sha256": module.sha256(teacher), "checkpoint_step": 104110,
        "selection": str(selection), "selection_sha256": module.sha256(selection),
        "config": str(config_path), "config_sha256": module.sha256(config_path),
        "source_sha256": module.source_hash(repo), "completion_record": str(complete),
        "tasks": cfg["data"]["tasks"], "task_count": 16,
        "evaluation_execution": {"execute_actions": 8, "action_stride": 2},
    }
    write_json(generated / "plan.json", cache_plan)
    site = tmp_path / "site.env"
    site.write_text("\n".join(f"{key}={shlex.quote(str(value))}" for key, value in {
        "STMEM_REPO_ROOT": repo, "STMEM_PYTHON": sys.executable,
        "STMEM_RUN_ROOT": repo / "runs/unrelated_teacher", "STMEM_DATA_ROOT": tmp_path / "wrong data",
        "STMEM_CACHE_ROOT": tmp_path / "wrong cache", "STMEM_PARTITION": "aiq", "STMEM_GPU_TYPE": "b200",
        "STMEM_CPUS": 64, "STMEM_ACCOUNT": "fixture-account", "WANDB_PROJECT": "chronotree",
        "WANDB_MODE": "online",
    }.items()) + "\n")

    def invoke(*args):
        module.main(["--site", str(site), "--cache-name", "memory_fv_seed0", "--legacy-cache", *args])
        return json.loads((repo / "hpc/generated/memory_fv_seed0_compact/plan.json").read_text())

    return SimpleNamespace(module=module, repo=repo, cfg=cfg, teacher=teacher, selection=selection,
                           cache_root=cache_root, complete=complete, cache_plan=generated / "plan.json",
                           site=site, invoke=invoke)


def test_dry_run_preserves_teacher_contract_and_cached_inputs(launcher, monkeypatch):
    inputs = [launcher.teacher, launcher.selection, launcher.complete,
              *launcher.cache_root.iterdir(), launcher.cache_plan]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in inputs}
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm in dry-run"))
    plan = launcher.invoke()
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    assert cfg["stage"] == "compact"
    for key in ("data", "geometry", "model", "paths"):
        assert cfg[key] == launcher.cfg[key]
    assert cfg["model"]["proprioception_mode"] == "none"
    assert cfg["model"]["condition_on_previous_commands"] is False
    assert cfg["model"]["visual_features"]["mode"] == "clip_rn50_layer1"
    assert cfg["model"]["horizon"] == 16 and cfg["model"]["execute"] == 2
    assert cfg["model"]["action_stride"] == cfg["data"]["observation_stride"] == 2
    assert cfg["training"]["steps"] == cfg["training"]["schedule_steps"] == 30000
    assert cfg["training"]["checkpoint_steps"] == [15000, 30000]
    assert cfg["training"]["lr"] == pytest.approx(0.0001)
    assert cfg["training"]["effective_batch"] == 128 and cfg["training"]["microbatch"] == 32
    assert cfg["training"]["context_mix"] == {"tree": 0.45, "flat": 0.45, "uniform": 0.05, "current": 0.05}
    assert cfg["training"]["workers"] == 16
    assert plan["gres"] == "gpu:2" and plan["gpus"] == 2 and plan["partition"] == "gpuq"
    assert plan["cpus"] == 64 and plan["mem"] == "128G" and plan["walltime"] == "48:00:00"
    assert plan["run_root"] == launcher.cfg["paths"]["run_root"]
    script = Path(plan["script"]).read_text()
    assert "#SBATCH --gres=gpu:2" in script and "--exclusive" not in script
    assert "#SBATCH --ntasks=1" in script and "--nproc-per-node=2" in script
    assert "--stage compact" in script and "--resume" not in script
    assert "WANDB_MODE=online" in script and "WANDB_PROJECT=chronotree" in script
    assert "WANDB_RUN_ID" in script and "WANDB_RESUME" in script
    assert not (Path(plan["run_root"]) / "seed0/compact/last.pt").exists()
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())


def test_rendered_script_has_valid_shell_and_idempotent_config(launcher):
    plan = launcher.invoke()
    subprocess.run(["bash", "-n", plan["script"]], check=True)
    mtime = Path(plan["config"]).stat().st_mtime_ns
    launcher.invoke("--walltime", "03:00:00")
    assert Path(plan["config"]).stat().st_mtime_ns == mtime


def test_cache_rejects_selection_step_mismatch(launcher):
    plan = json.loads(launcher.cache_plan.read_text())
    plan["checkpoint_step"] = 140000
    write_json(launcher.cache_plan, plan)
    with pytest.raises(ValueError, match="selection step"):
        launcher.module.inspect_cache(launcher.cache_plan)


@pytest.mark.parametrize("field", ["teacher_sha256", "teacher_step", "encoded_cache_id", "hierarchy_cache_id"])
@pytest.mark.parametrize("changed", ["settings", "actual"])
def test_study_launch_binds_manifest_settings_and_actual_inputs(launcher, monkeypatch, field, changed):
    from stmem.inspection import refinement_data

    pins = {"teacher_sha256": "a" * 64, "teacher_step": 140000,
            "encoded_cache_id": "b" * 24, "hierarchy_cache_id": "c" * 24}
    # The separate manifest tests establish its seal/provenance. Exercise the
    # final launch boundary here: individually valid settings/caches must still
    # refer to the same teacher and cache as that manifest.
    monkeypatch.setattr(refinement_data, "study_dependencies", lambda _panel: dict(pins))
    settings, actual = dict(pins), dict(pins)
    launcher.module.validate_study_inputs(settings, {}, **actual)
    target = settings if changed == "settings" else actual
    target[field] = 104110 if field == "teacher_step" else "d" * len(pins[field])
    with pytest.raises(ValueError, match=field):
        launcher.module.validate_study_inputs(settings, {}, **actual)


@pytest.mark.parametrize("target", ["teacher", "selection", "manifest", "completion", "shard", "receipt", "cache_config"])
def test_bad_cache_inputs_fail_before_submission(launcher, monkeypatch, target):
    if target == "shard":
        (launcher.cache_root / "BinFill.h5").unlink()
    elif target == "receipt":
        (launcher.cache_root / "BinFill.complete.json").unlink()
    else:
        path = {
            "teacher": launcher.teacher, "selection": launcher.selection,
            "manifest": launcher.cache_root / "manifest.json", "completion": launcher.complete,
            "cache_config": Path(json.loads(launcher.cache_plan.read_text())["config"]),
        }[target]
        path.write_bytes(path.read_bytes() + b"corruption")
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm with bad cache"))
    with pytest.raises(SystemExit):
        launcher.invoke("--submit")


def test_reject_changed_generated_config_and_hyperparameters(launcher):
    plan = launcher.invoke()
    with pytest.raises(SystemExit):
        launcher.invoke("--effective-batch", "256")
    path = Path(plan["config"])
    path.write_text(path.read_text() + "# changed after render\n")
    with pytest.raises(SystemExit):
        launcher.invoke()


def test_ready_plan_rejects_changes_after_render(launcher):
    plan = launcher.invoke()
    plan_path = Path(plan["script"]).with_name("plan.json")
    launcher.module.ready_plan(plan_path)
    Path(plan["config"]).write_text("stage: teacher\n")
    with pytest.raises((ValueError, RuntimeError)):
        launcher.module.ready_plan(plan_path)


def test_missing_resume_checkpoint_and_existing_run_are_rejected(launcher):
    launcher.invoke()
    with pytest.raises(SystemExit):
        launcher.invoke("--resume")
    run = Path(launcher.cfg["paths"]["run_root"]) / "seed0/compact"
    run.mkdir(parents=True)
    (run / "last.pt").write_bytes(b"synthetic incomplete checkpoint")
    with pytest.raises(SystemExit):
        launcher.invoke()


@pytest.mark.parametrize("arguments", [("--gpus", "0"), ("--cpus", "0"), ("--workers", "-1"),
                                       ("--effective-batch", "127"), ("--microbatch", "0"),
                                       ("--mem", "0"), ("--partition", ""), ("--walltime", "00:00:00")])
def test_invalid_resource_or_batch_choices(launcher, arguments):
    with pytest.raises(SystemExit):
        launcher.invoke(*arguments)


def save_resume_fixture(launcher, plan, *, step=1000, stage="compact", continuation=None):
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    checkpoint = Path(plan["output"]) / "last.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "stage": stage, "step": step, "config": cfg,
        "model": {"synthetic_launcher_fixture": torch.zeros(1)},
        "provenance": {"world_size": plan["gpus"], "dependencies": {
            "teacher": plan["checkpoint_sha256"], "cache": plan["cache_manifest_digest"],
        }},
    }
    if plan.get("hierarchy_manifest_digest"):
        state["provenance"]["dependencies"]["hierarchy"] = plan["hierarchy_manifest_digest"]
    if continuation is not None:
        state["continuation"] = continuation
    torch.save(state, checkpoint)
    return checkpoint


def test_submit_then_resume_uses_only_own_checkpoint_and_blocks_active_job(launcher, monkeypatch):
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"{71000 + len(calls)};cluster\n")

    monkeypatch.setattr(launcher.module.subprocess, "run", submit)
    first = launcher.invoke("--submit")
    assert calls == [["sbatch", "--parsable", first["script"]]]
    assert first["resume"] is None and first["start_step"] == 0
    with pytest.raises(SystemExit):
        launcher.invoke("--submit")
    checkpoint = save_resume_fixture(launcher, first)
    monkeypatch.setattr(launcher.module, "active_user_job_ids", lambda: {"71001"})
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--submit")
    assert len(calls) == 1
    monkeypatch.setattr(launcher.module, "active_user_job_ids", set)
    second = launcher.invoke("--resume", "--submit", "--walltime", "03:00:00")
    assert len(calls) == 2
    assert second["signature"] == first["signature"]
    assert second["config_sha256"] == first["config_sha256"]
    assert second["resume"] == str(checkpoint) and second["start_step"] == 1000
    assert second["remaining_updates"] == 29000
    assert "--resume " + shlex.quote(str(checkpoint)) in Path(second["script"]).read_text()
    receipt = json.loads(Path(second["script"]).with_name("submitted.json").read_text())
    assert receipt["job_id"] == "71002" and receipt["previous"]["job_id"] == "71001"
    launcher.module.ready_plan(Path(second["script"]).with_name("plan.json"))
    checkpoint.write_bytes(checkpoint.read_bytes() + b"changed after resume plan")
    with pytest.raises(ValueError, match="Resume checkpoint"):
        launcher.module.ready_plan(Path(second["script"]).with_name("plan.json"))


@pytest.mark.parametrize("change", ["teacher_stage", "wrong_teacher", "completed", "world_size"])
def test_resume_rejects_wrong_stage_dependencies_or_completed_checkpoint(launcher, change):
    plan = launcher.invoke()
    checkpoint = save_resume_fixture(launcher, plan)
    state = torch.load(checkpoint, weights_only=False)
    if change == "teacher_stage":
        state["stage"] = "teacher"
    elif change == "wrong_teacher":
        state["provenance"]["dependencies"]["teacher"] = "different teacher"
    elif change == "completed":
        state["step"] = 30000
    else:
        state["provenance"]["world_size"] = 4
    torch.save(state, checkpoint)
    with pytest.raises(SystemExit):
        launcher.invoke("--resume")


def test_different_named_plan_cannot_share_same_training_output(launcher):
    plan = launcher.invoke()
    config = Path(plan["config"]).read_bytes()
    with pytest.raises(SystemExit):
        launcher.invoke("--name", "another_compact_plan")
    assert Path(plan["config"]).read_bytes() == config


def test_allocation_guard_rejects_source_change(launcher):
    plan = launcher.invoke()
    (launcher.repo / "src/fixture.py").write_text("# changed after rendering\n")
    with pytest.raises(ValueError, match="Source changed"):
        launcher.module.ready_plan(Path(plan["script"]).with_name("plan.json"))


def test_explicit_loader_upgrade_preserves_config_checkpoint_and_schedule(launcher):
    first = launcher.invoke("--prefetch-factor", "1")
    checkpoint = save_resume_fixture(launcher, first, step=5000)
    original_checkpoint = checkpoint.read_bytes()
    original_config = Path(first["config"]).read_bytes()
    (launcher.repo / "src/fixture.py").write_text("# Reviewed synthetic loader-only source change.\n")
    with pytest.raises(SystemExit):
        launcher.invoke("--resume")
    second = launcher.invoke("--resume", "--upgrade-loader", "--prefetch-factor", "2")
    assert Path(second["config"]).read_bytes() == original_config
    assert checkpoint.read_bytes() == original_checkpoint
    assert second["start_step"] == 5000 and second["remaining_updates"] == 25000
    assert second["continue_to_step"] is None and second["continuation"] is None
    assert second["config_sha256"] == first["config_sha256"]
    assert second["runtime_loader"] == {"workers": 16, "prefetch_factor": 2, "prefetch_capacity_rank": 32}
    assert "--prefetch-factor 2" in Path(second["script"]).read_text()
    upgrade = second["loader_upgrade"]
    assert upgrade["from_signature"] == first["signature"]
    assert upgrade["to_signature"] == second["signature"]
    assert upgrade["checkpoint_sha256"] == launcher.module.sha256(checkpoint)
    assert upgrade["optimizer_step"] == 5000
    launcher.module.ready_plan(second["startup_plan"])
    # Rendering then submitting uses the same audit record without requiring an
    # artificial LR continuation or rewriting any checkpoint.
    third = launcher.invoke("--resume", "--upgrade-loader", "--prefetch-factor", "2")
    assert third["loader_upgrade"] == upgrade
    assert third["signature"] == second["signature"]
    launcher.module.ready_plan(third["startup_plan"])


@pytest.mark.parametrize("override", [("--microbatch", "64"), ("--workers", "8"),
                                     ("--gpus", "4"), ("--effective-batch", "256")])
def test_loader_upgrade_never_allows_scientific_or_nonallowlisted_config_changes(launcher, override):
    plan = launcher.invoke()
    save_resume_fixture(launcher, plan)
    (launcher.repo / "src/fixture.py").write_text("# Reviewed synthetic loader-only source change.\n")
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--upgrade-loader", *override)
    assert json.loads(Path(plan["plan_path"]).read_text()) == plan


def test_loader_upgrade_is_explicit_resume_only_and_cannot_extend_endpoint(launcher):
    with pytest.raises(SystemExit):
        launcher.invoke("--upgrade-loader")
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--upgrade-loader", "--continue-to-step", "100000")


def test_loader_upgrade_guard_rejects_tampered_runtime_and_upgrade_audit(launcher):
    first = launcher.invoke()
    save_resume_fixture(launcher, first)
    (launcher.repo / "src/fixture.py").write_text("# Reviewed synthetic loader-only source change.\n")
    plan = launcher.invoke("--resume", "--upgrade-loader")
    path = Path(plan["plan_path"])
    plan["runtime_loader"]["prefetch_capacity_rank"] += 1
    write_json(path, plan)
    with pytest.raises(ValueError, match="runtime loader"):
        launcher.module.ready_plan(path)
    plan["runtime_loader"]["prefetch_capacity_rank"] -= 1
    plan["loader_upgrade"]["from_signature"] = "not the same configuration"
    write_json(path, plan)
    with pytest.raises(ValueError, match="Loader upgrade"):
        launcher.module.ready_plan(path)


def test_new_launch_rejects_teacher_contract_drift_even_with_refreshed_config_hash(launcher):
    cache_plan = json.loads(launcher.cache_plan.read_text())
    config_path = Path(cache_plan["config"])
    cfg = yaml.safe_load(config_path.read_text())
    # Horizon is a decoder contract, so changing it does not alter encoded-cache identity.
    cfg["model"]["horizon"] = 32
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cache_plan["config_sha256"] = launcher.module.sha256(config_path)
    write_json(launcher.cache_plan, cache_plan)
    with pytest.raises(SystemExit):
        launcher.invoke()


def test_resume_dry_run_preserves_earlier_attempt_snapshot(launcher):
    initial = launcher.invoke()
    save_resume_fixture(launcher, initial)
    first_resume = launcher.invoke("--resume", "--dry-run")
    snapshot = Path(first_resume["startup_plan"])
    before = snapshot.read_bytes(), snapshot.stat().st_mtime_ns
    first_script = Path(first_resume["script"]).read_text()
    assert str(snapshot) in first_script
    assert snapshot != Path(first_resume["plan_path"])
    later_resume = launcher.invoke("--resume", "--dry-run", "--walltime", "03:00:00")
    assert later_resume["startup_plan"] != first_resume["startup_plan"]
    assert (snapshot.read_bytes(), snapshot.stat().st_mtime_ns) == before
    assert json.loads(snapshot.read_text())["walltime"] == "48:00:00"
    assert json.loads(Path(later_resume["plan_path"]).read_text())["walltime"] == "03:00:00"
    # Even changing the latest-plan pointer does not change the queued script's guard.
    Path(later_resume["plan_path"]).write_text("{}\n")
    launcher.module.ready_plan(snapshot)


def test_restart_unstarted_requires_existing_plan(launcher, monkeypatch):
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm without plan"))
    with pytest.raises(SystemExit):
        launcher.invoke("--restart-unstarted", "--submit")


def test_restart_unstarted_blocks_active_job_and_can_retry_inactive_attempt(launcher, monkeypatch):
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"{72000 + len(calls)}\n")

    monkeypatch.setattr(launcher.module.subprocess, "run", submit)
    first = launcher.invoke("--submit")
    monkeypatch.setattr(launcher.module, "active_user_job_ids", lambda: {"72001"})
    with pytest.raises(SystemExit):
        launcher.invoke("--restart-unstarted", "--submit")
    assert len(calls) == 1
    monkeypatch.setattr(launcher.module, "active_user_job_ids", set)
    retry = launcher.invoke("--restart-unstarted", "--submit")
    assert len(calls) == 2 and retry["signature"] == first["signature"]
    assert retry["start_step"] == 0 and retry["resume"] is None
    assert retry["restart_unstarted"] is True and retry["remaining_updates"] == 30000
    assert "--resume" not in Path(retry["script"]).read_text()
    receipt = json.loads(Path(retry["script"]).with_name("submitted.json").read_text())
    assert receipt["job_id"] == "72002" and receipt["previous"]["job_id"] == "72001"


@pytest.mark.parametrize("checkpoint_name", ["last.pt", "last.prev.pt", "step_00015000.pt"])
def test_restart_unstarted_never_discards_existing_checkpoints(launcher, checkpoint_name):
    plan = launcher.invoke()
    output = Path(plan["output"])
    output.mkdir(parents=True)
    (output / checkpoint_name).write_bytes(b"synthetic fixture checkpoint")
    with pytest.raises(SystemExit):
        launcher.invoke("--restart-unstarted")
    assert (output / checkpoint_name).read_bytes() == b"synthetic fixture checkpoint"


@pytest.mark.parametrize("field,value", [("identity_sha256", "unrelated cache"),
                                         ("kind", "encoder_cache_episode"), ("status", "incomplete")])
def test_inconsistent_shard_receipt_contract_is_rejected(launcher, field, value):
    path = launcher.cache_root / "BinFill.complete.json"
    record = json.loads(path.read_text())
    record[field] = value
    write_json(path, record)
    with pytest.raises(SystemExit):
        launcher.invoke()


def test_fresh_run_root_isolates_training_and_reuses_selected_teacher_and_cache(launcher, monkeypatch):
    old_plan = launcher.invoke()
    old_output = Path(old_plan["output"])
    old_output.mkdir(parents=True)
    (old_output / "tracking.json").write_text('{"id":"old compact run"}\n')
    (old_output / "last.pt").write_bytes(b"old compact checkpoint retained")
    immutable_paths = [launcher.teacher, launcher.selection, launcher.complete, launcher.cache_plan,
                       *launcher.cache_root.iterdir(), *old_output.iterdir(),
                       Path(old_plan["config"]), Path(old_plan["plan_path"])]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable_paths}
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm in dry-run"))
    name = "memory_fv_seed0_compact_prefetch"
    new_root = launcher.repo / "runs" / name
    launcher.module.main(["--site", str(launcher.site), "--cache-name", "memory_fv_seed0", "--legacy-cache",
                          "--name", name, "--run-root", "runs/" + name, "--dry-run"])
    plan_path = launcher.repo / "hpc/generated" / name / "plan.json"
    plan = json.loads(plan_path.read_text())
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    assert plan["run_root"] == str(new_root)
    assert plan["output"] == str(new_root / "seed0/compact")
    assert cfg["paths"] == {**launcher.cfg["paths"], "run_root": str(new_root)}
    assert cfg["checkpoints"]["teacher"] == str(launcher.teacher)
    assert cfg["checkpoints"]["compact"] == str(new_root / "seed0/compact/selected.pt")
    assert plan["cache_path"] == str(launcher.cache_root)
    assert plan["resume"] is None and plan["start_step"] == 0
    assert (new_root / ".compact-launch.lock").exists()
    assert (new_root / "seed0/.compact-plan.json").exists()
    script = Path(plan["script"]).read_text()
    assert "export STMEM_RUN_ROOT=" + shlex.quote(str(new_root)) in script
    launcher.module.ready_plan(plan_path)
    assert not (new_root / "seed0/compact/last.pt").exists()
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())


@pytest.mark.parametrize("change", ["data_root", "cache_root", "teacher"])
def test_fresh_run_runtime_guard_only_allows_output_root_change(launcher, change):
    name = "fresh_compact"
    launcher.module.main(["--site", str(launcher.site), "--cache-name", "memory_fv_seed0", "--legacy-cache",
                          "--name", name, "--run-root", "runs/fresh_compact"])
    plan_path = launcher.repo / "hpc/generated" / name / "plan.json"
    plan = json.loads(plan_path.read_text())
    config_path = Path(plan["config"])
    cfg = yaml.safe_load(config_path.read_text())
    if change == "teacher":
        cfg["checkpoints"]["teacher"] = str(launcher.repo / "unrelated_teacher.pt")
    else:
        cfg["paths"][change] = str(launcher.repo / "unrelated_data")
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    plan["config_sha256"] = launcher.module.sha256(config_path)
    write_json(plan_path, plan)
    with pytest.raises(ValueError):
        launcher.module.ready_plan(plan_path)


def test_fresh_plan_refuses_nonempty_output_even_without_checkpoint(launcher):
    output = launcher.repo / "runs/used/seed0/compact"
    output.mkdir(parents=True)
    (output / "tracking.json").write_text('{"id":"existing logical run"}\n')
    with pytest.raises(SystemExit):
        launcher.module.main(["--site", str(launcher.site), "--cache-name", "memory_fv_seed0", "--legacy-cache",
                              "--name", "new_plan", "--run-root", "runs/used"])
    assert json.loads((output / "tracking.json").read_text())["id"] == "existing logical run"


@pytest.mark.parametrize("arguments", [
    ("--continue-to-step", "60000"),
    ("--restart-unstarted", "--continue-to-step", "60000"),
    ("--resume", "--continue-to-step", "0"),
    ("--resume", "--continue-to-step", "-1"),
])
def test_invalid_continuation_arguments_fail_before_submission(launcher, monkeypatch, arguments):
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Unexpected Slurm"))
    with pytest.raises(SystemExit):
        launcher.invoke(*arguments, "--submit")


def test_completed_continuation_preserves_config_dependencies_and_prior_startup_plan(launcher, monkeypatch):
    initial = launcher.invoke()
    checkpoint = save_resume_fixture(launcher, initial, step=30000)
    complete = checkpoint.parent / "complete.json"
    write_json(complete, {"status": "complete", "step": 30000})
    snapshot = Path(initial["startup_plan"])
    immutable = [checkpoint, complete, snapshot, Path(initial["config"]), launcher.cache_plan,
                 launcher.teacher, launcher.selection, launcher.complete, *launcher.cache_root.iterdir()]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable}
    # Existing real runs use older code. Updating only the implementation must
    # permit the explicit extension without changing the saved scientific run.
    (launcher.repo / "src/fixture.py").write_text("# Updated synthetic runtime for bounded continuation.\n")
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Slurm in dry-run"))
    continued = launcher.invoke("--resume", "--continue-to-step", "60000", "--dry-run")
    assert continued["start_step"] == continued["remaining_updates"] == 30000
    assert continued["continue_to_step"] == 60000
    assert continued["continuation"] == {
        "mode": "bounded_constant_lr_tail_v1", "original_steps": 30000,
        "schedule_steps": 30000, "final_lr": 0.1 * 0.0001,
        "enabled_at_step": 30000, "intended_final_step": 60000,
    }
    for key in ("training", "config_sha256", "cache_manifest_digest", "checkpoint_sha256",
                "candidate_checkpoints", "run_root"):
        assert continued[key] == initial[key]
    assert continued["training"]["steps"] == continued["training"]["schedule_steps"] == 30000
    assert continued["training"]["checkpoint_steps"] == [15000, 30000]
    assert continued["source_sha256"] != initial["source_sha256"]
    assert continued["previous_source_sha256"] == initial["source_sha256"]
    assert continued["signature"] != initial["signature"]
    owner = json.loads((checkpoint.parent.parent / ".compact-plan.json").read_text())
    assert owner == {"plan": continued["plan_path"], "signature": continued["signature"]}
    assert continued["startup_plan"] != str(snapshot)
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())
    script = Path(continued["script"]).read_text()
    assert "--resume " + shlex.quote(str(checkpoint)) in script
    assert "--continue-to-step 60000" in script
    launcher.module.ready_plan(continued["startup_plan"])


@pytest.mark.parametrize("step,endpoint", [(29999, 60000), (30000, 30000), (30000, 20000), (60000, 60000)])
def test_continuation_requires_completed_original_run_and_remaining_steps(launcher, step, endpoint):
    initial = launcher.invoke()
    save_resume_fixture(launcher, initial, step=step)
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--continue-to-step", str(endpoint))
    assert json.loads(Path(initial["plan_path"]).read_text()) == initial


def test_planned_continuation_requires_flag_and_cannot_shorten_even_before_new_checkpoint(launcher):
    initial = launcher.invoke()
    save_resume_fixture(launcher, initial, step=30000)
    continued = launcher.invoke("--resume", "--continue-to-step", "60000")
    snapshot = Path(continued["startup_plan"])
    before = snapshot.read_bytes(), snapshot.stat().st_mtime_ns
    for arguments in [("--resume",), ("--resume", "--continue-to-step", "50000")]:
        with pytest.raises(SystemExit):
            launcher.invoke(*arguments)
    assert json.loads(Path(continued["plan_path"]).read_text()) == continued
    extended = launcher.invoke("--resume", "--continue-to-step", "70000")
    assert extended["continue_to_step"] == 70000 and extended["remaining_updates"] == 40000
    assert extended["signature"] == continued["signature"]
    assert extended["startup_plan"] != continued["startup_plan"]
    assert (snapshot.read_bytes(), snapshot.stat().st_mtime_ns) == before
    launcher.module.ready_plan(snapshot)


def test_interrupted_continuation_resumes_existing_checkpoint_at_same_endpoint(launcher):
    initial = launcher.invoke()
    checkpoint = save_resume_fixture(launcher, initial, step=30000)
    continued = launcher.invoke("--resume", "--continue-to-step", "60000")
    save_resume_fixture(launcher, continued, step=41000, continuation=continued["continuation"])
    resumed = launcher.invoke("--resume", "--continue-to-step", "60000")
    assert resumed["start_step"] == 41000 and resumed["remaining_updates"] == 19000
    assert resumed["continuation"] == continued["continuation"]
    assert resumed["config_sha256"] == initial["config_sha256"]
    launcher.module.ready_plan(resumed["startup_plan"])
    # An earlier allocation cannot read the newer checkpoint as its original input.
    with pytest.raises(ValueError, match="Resume checkpoint"):
        launcher.module.ready_plan(continued["startup_plan"])
    assert resumed["resume_sha256"] == launcher.module.sha256(checkpoint)


def test_continuation_source_upgrade_never_allows_training_config_change(launcher):
    initial = launcher.invoke()
    save_resume_fixture(launcher, initial, step=30000)
    ownership = Path(initial["output"]).parent / ".compact-plan.json"
    before = ownership.read_bytes()
    (launcher.repo / "src/fixture.py").write_text("# Updated synthetic runtime.\n")
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--continue-to-step", "60000", "--effective-batch", "256")
    assert ownership.read_bytes() == before
    assert json.loads(Path(initial["plan_path"]).read_text()) == initial


def test_ordinary_resume_does_not_silently_allow_source_upgrade(launcher):
    initial = launcher.invoke()
    save_resume_fixture(launcher, initial)
    (launcher.repo / "src/fixture.py").write_text("# Updated synthetic runtime.\n")
    with pytest.raises(SystemExit):
        launcher.invoke("--resume")
    assert json.loads(Path(initial["plan_path"]).read_text()) == initial


def test_continuation_blocks_active_job_before_replacing_plan_or_ownership(launcher, monkeypatch):
    calls = []

    def submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="73000\n")

    monkeypatch.setattr(launcher.module.subprocess, "run", submit)
    initial = launcher.invoke("--submit")
    checkpoint = save_resume_fixture(launcher, initial, step=30000)
    write_json(checkpoint.parent / "complete.json", {"status": "complete", "step": 30000})
    monkeypatch.setattr(launcher.module, "active_user_job_ids", lambda: {"73000"})
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--continue-to-step", "60000", "--submit")
    assert len(calls) == 1
    assert json.loads(Path(initial["plan_path"]).read_text()) == initial


def test_startup_guard_checks_continuation_metadata(launcher):
    initial = launcher.invoke()
    save_resume_fixture(launcher, initial, step=30000)
    continued = launcher.invoke("--resume", "--continue-to-step", "60000")
    continued["continuation"]["final_lr"] = 0.5
    write_json(Path(continued["plan_path"]), continued)
    with pytest.raises(ValueError, match="continuation differs"):
        launcher.module.ready_plan(continued["plan_path"])


def isolated_code_checkout(launcher):
    root = launcher.repo.parent / "isolated runtime checkout"
    for relative, content in {
        "src/stmem/train.py": "# Synthetic isolated runtime fixture; no training.\n",
        "hpc/train_compact.py": "# Synthetic isolated launcher fixture.\n",
        "hpc/run_job.sh": "#!/usr/bin/env bash\nexit 0\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return root


def test_isolated_code_root_preserves_main_checkout_and_existing_run_paths(launcher):
    initial = launcher.invoke()
    checkpoint = save_resume_fixture(launcher, initial, step=30000)
    code_root = isolated_code_checkout(launcher)
    immutable = [launcher.repo / "src/fixture.py", Path(initial["config"]),
                 Path(initial["startup_plan"]), launcher.cache_plan, checkpoint]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in immutable}
    continued = launcher.invoke("--resume", "--continue-to-step", "60000", "--code-root", str(code_root))
    assert continued["repo"] == str(launcher.repo)
    assert continued["code_root"] == str(code_root)
    assert continued["python"] == sys.executable
    assert continued["source_sha256"] == launcher.module.source_hash(code_root)
    assert continued["previous_source_sha256"] == initial["source_sha256"]
    for key in ("run_root", "output", "config", "config_sha256", "cache_plan", "cache_path", "logs", "script"):
        assert continued[key] == initial[key]
    assert not (code_root / "runs").exists() and not (code_root / "hpc/generated").exists()
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in before.items())
    script = Path(continued["script"]).read_text()
    assert "#SBATCH --chdir=" + shlex.quote(str(code_root)) in script
    assert "\ncd " + shlex.quote(str(code_root)) + "\n" in script
    assert "export STMEM_REPO_ROOT=" + shlex.quote(str(code_root)) + "\n" in script
    assert "export PYTHONPATH=" + shlex.quote(str(code_root / "src")) + "\n" in script
    assert "exec " + shlex.quote(str(code_root / "hpc/run_job.sh")) in script
    assert "export STMEM_RUN_ROOT=" + shlex.quote(initial["run_root"]) in script
    assert "--config " + shlex.quote(initial["config"]) in script
    assert "--resume " + shlex.quote(str(checkpoint)) in script
    assert "--continue-to-step 60000" in script
    subprocess.run(["bash", "-n", continued["script"]], check=True)
    launcher.module.ready_plan(continued["startup_plan"])
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--continue-to-step", "60000")
    assert json.loads(Path(continued["plan_path"]).read_text()) == continued
    repeated = launcher.invoke("--resume", "--continue-to-step", "60000", "--code-root", str(code_root))
    assert repeated["code_root"] == str(code_root) and repeated["signature"] == continued["signature"]
    (code_root / "src/stmem/train.py").write_text("# Changed isolated source after rendering.\n")
    with pytest.raises(ValueError, match="Source changed"):
        launcher.module.ready_plan(continued["startup_plan"])
    assert (launcher.repo / "src/fixture.py").read_bytes() == before[launcher.repo / "src/fixture.py"][0]


@pytest.mark.parametrize("missing", ["src/stmem/train.py", "hpc/train_compact.py", "hpc/run_job.sh"])
def test_incomplete_isolated_code_root_rejected_before_submission(launcher, monkeypatch, missing):
    initial = launcher.invoke()
    save_resume_fixture(launcher, initial, step=30000)
    root = isolated_code_checkout(launcher)
    (root / missing).unlink()
    monkeypatch.setattr(launcher.module.subprocess, "run", lambda *_a, **_k: pytest.fail("Unexpected Slurm"))
    with pytest.raises(SystemExit):
        launcher.invoke("--resume", "--continue-to-step", "60000", "--code-root", str(root), "--submit")
    assert json.loads(Path(initial["plan_path"]).read_text()) == initial
