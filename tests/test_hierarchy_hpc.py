"""CPU-only hierarchy launch checks; cache contents are synthetic launcher fixtures."""

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import h5py
import pytest
import torch
import yaml

from test_compact_hpc import REPO, launcher as compact_launcher, save_resume_fixture, write_json  # noqa: F401
from stmem.config import digest


@pytest.fixture
def hierarchy_launcher(compact_launcher, monkeypatch):  # noqa: F811 -- imported pytest fixture
    from stmem.data import hierarchy_cache

    launcher = compact_launcher
    output = launcher.cache_root / "hierarchy/synthetic-launcher-fixture"
    monkeypatch.setattr(hierarchy_cache, "default_hierarchy_root", lambda *_args, **_kw: output)

    def status(_leaf_root, output_root=None):
        path = Path(output_root or output)
        complete = (path / "manifest.json").exists()
        return {"status": "complete" if complete else "incomplete", "hierarchy_cache_root": str(path),
                "completed_tasks": launcher.cfg["data"]["tasks"] if complete else [],
                "missing_tasks": [] if complete else launcher.cfg["data"]["tasks"]}

    def validate(leaf_root, output_root=None):
        assert Path(leaf_root) == launcher.cache_root
        path = Path(output_root or output) / "manifest.json"
        manifest = json.loads(path.read_text())
        if manifest.get("status") != "complete" or manifest.get("synthetic_launcher_fixture") is not True:
            raise ValueError("Invalid synthetic fixture hierarchy")
        return manifest

    monkeypatch.setattr(hierarchy_cache, "hierarchy_status", status)
    monkeypatch.setattr(hierarchy_cache, "validate_hierarchy_cache", validate)
    spec = importlib.util.spec_from_file_location("hierarchy_hpc_launcher", REPO / "hpc/cache_hierarchy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def invoke(*args):
        module.main(["--site", str(launcher.site), "--cache-name", "memory_fv_seed0", *args])
        return json.loads((launcher.repo / "hpc/generated/memory_fv_seed0_hierarchy/plan.json").read_text())

    def complete():
        manifest = {"status": "complete", "synthetic_launcher_fixture": True,
                    "source_manifest_sha256": module.sha256(launcher.cache_root / "manifest.json")}
        write_json(output / "manifest.json", manifest)
        return manifest

    return SimpleNamespace(module=module, invoke=invoke, output=output, complete=complete, source=launcher)


def test_hierarchy_dry_run_resources_and_no_data_mutation(hierarchy_launcher, monkeypatch):
    fixture = hierarchy_launcher
    original = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in fixture.source.cache_root.iterdir()}
    monkeypatch.setattr(fixture.module.subprocess, "run", lambda *_a, **_kw: pytest.fail("Slurm in dry-run"))
    plan = fixture.invoke()
    assert plan["array"] == "0-15%4" and plan["maximum_concurrent_allocations"] == 4
    assert plan["gres"] == "gpu:1" and plan["partition"] == "gpuq"
    assert plan["cpus"] == 4 and plan["threads"] == 1 and plan["mem"] == "32G"
    assert plan["hierarchy_cache_root"] == str(fixture.output)
    assert not fixture.output.exists()
    script = Path(plan["script"]).read_text()
    assert "#SBATCH --gres=gpu:1" in script and "--exclusive" not in script
    assert "stmem.data.hierarchy_cache" in script and "stmem.data.cache " not in script
    assert "--task-index" in script and "--stop-file" in script and "torch.distributed.run" not in script
    assert plan["startup_plan"] in script
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in original.items())
    fixture.module.ready_plan(plan["startup_plan"])


def test_hierarchy_script_syntax_and_startup_snapshot(hierarchy_launcher):
    first = hierarchy_launcher.invoke()
    subprocess.run(["bash", "-n", first["script"]], check=True)
    snapshot = Path(first["startup_plan"])
    before = snapshot.read_bytes(), snapshot.stat().st_mtime_ns
    later = hierarchy_launcher.invoke("--max-concurrent", "8", "--walltime", "03:00:00")
    assert later["array"] == "0-15%8" and later["signature"] == first["signature"]
    assert (snapshot.read_bytes(), snapshot.stat().st_mtime_ns) == before
    hierarchy_launcher.module.ready_plan(snapshot)


def test_status_is_read_only_even_before_output_exists(hierarchy_launcher, monkeypatch, capsys):
    fixture = hierarchy_launcher
    fixture.invoke()
    files = {path: (path.read_bytes(), path.stat().st_mtime_ns)
             for path in fixture.source.repo.rglob("*") if path.is_file()}
    monkeypatch.setattr(fixture.module.subprocess, "run", lambda *_a, **_kw: pytest.fail("Slurm in status"))
    capsys.readouterr()
    fixture.invoke("--status")
    assert json.loads(capsys.readouterr().out)["status"] == "incomplete"
    assert not fixture.output.exists()
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == value for path, value in files.items())


def test_status_missing_plan_creates_nothing(hierarchy_launcher):
    source = hierarchy_launcher.source
    target = source.repo / "hpc/generated/memory_fv_seed0_hierarchy"
    with pytest.raises(SystemExit):
        hierarchy_launcher.invoke("--status")
    assert not target.exists()


@pytest.mark.parametrize("change", ["config", "cache_plan", "manifest", "source"])
def test_hierarchy_startup_rejects_changed_inputs(hierarchy_launcher, change):
    fixture = hierarchy_launcher
    plan = fixture.invoke()
    target = {"config": Path(plan["config"]), "cache_plan": Path(plan["cache_plan"]),
              "manifest": fixture.source.cache_root / "manifest.json",
              "source": fixture.source.repo / "src/fixture.py"}[change]
    target.write_bytes(target.read_bytes() + b"\nchanged")
    with pytest.raises(ValueError):
        fixture.module.ready_plan(plan["startup_plan"])


def test_hierarchy_submission_requires_explicit_resume_and_inactive_array(hierarchy_launcher, monkeypatch):
    fixture = hierarchy_launcher
    calls = []

    def submit(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=f"{73000 + len(calls)};cluster\n")

    monkeypatch.setattr(fixture.module.subprocess, "run", submit)
    first = fixture.invoke("--submit")
    assert calls == [["sbatch", "--parsable", first["script"]]]
    with pytest.raises(SystemExit):
        fixture.invoke("--submit")
    monkeypatch.setattr(fixture.module, "active_array_ids", lambda: {"73001"})
    with pytest.raises(SystemExit):
        fixture.invoke("--resume", "--submit")
    assert len(calls) == 1
    monkeypatch.setattr(fixture.module, "active_array_ids", set)
    second = fixture.invoke("--resume", "--submit")
    assert len(calls) == 2 and second["signature"] == first["signature"]
    receipt = json.loads(Path(second["plan_path"]).with_name("submitted.json").read_text())
    assert receipt["job_id"] == "73002" and receipt["previous"]["job_id"] == "73001"


def test_completed_hierarchy_refuses_redundant_submission(hierarchy_launcher, monkeypatch):
    fixture = hierarchy_launcher
    fixture.complete()
    monkeypatch.setattr(fixture.module.subprocess, "run", lambda *_a, **_kw: pytest.fail("Slurm after completion"))
    with pytest.raises(SystemExit):
        fixture.invoke("--submit")


def test_compact_requires_hierarchy_and_defaults_to_fresh_root(hierarchy_launcher):
    fixture = hierarchy_launcher
    source = fixture.source
    args = ["--site", str(source.site), "--cache-name", "memory_fv_seed0"]
    with pytest.raises(SystemExit):
        source.module.main(args)
    assert not (source.repo / "runs/memory_fv_seed0/seed0/compact").exists()
    manifest = fixture.complete()
    source.module.main(args)
    plan_path = source.repo / "hpc/generated/memory_fv_seed0_compact_hierarchy/plan.json"
    plan = json.loads(plan_path.read_text())
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    assert plan["hierarchy_manifest_digest"] == digest(manifest)
    assert plan["hierarchy_manifest_sha256"] == source.module.sha256(fixture.output / "manifest.json")
    assert cfg["training"]["hierarchy_cache_root"] == str(fixture.output)
    assert cfg["training"]["workers"] == 16 and plan["cpus"] == 64
    assert plan["run_root"] == str(source.repo / "runs/memory_fv_seed0_compact_hierarchy")
    assert cfg["training"]["effective_batch"] == 128 and cfg["training"]["microbatch"] == 32
    source.module.ready_plan(plan["startup_plan"])
    save_resume_fixture(source, plan)
    source.module.main([*args, "--resume"])
    changed = {**manifest, "changed": True}
    write_json(fixture.output / "manifest.json", changed)
    with pytest.raises(ValueError, match="Hierarchy manifest"):
        source.module.ready_plan(plan["startup_plan"])
    with pytest.raises(SystemExit):
        source.module.main([*args, "--resume"])


@pytest.mark.parametrize("args", [("--cpus", "0"), ("--threads", "5"), ("--max-concurrent", "0"),
                                  ("--mem", "0"), ("--partition", "bad partition"),
                                  ("--name", "memory_fv_seed0")])
def test_invalid_hierarchy_resources_are_rejected(hierarchy_launcher, args):
    with pytest.raises(SystemExit):
        hierarchy_launcher.invoke(*args)


def test_actual_hierarchy_builder_connects_to_compact_launcher(compact_launcher):  # noqa: F811
    """Build tiny synthetic HDF5 shards, with no models, GPU or Slurm calls."""
    from stmem.data.cache import episode_key, write_records
    from stmem.data.hierarchy_cache import build_hierarchy_cache
    from stmem.memory.records import leaf_record

    source = compact_launcher
    manifest_path = source.cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    width = source.cfg["model"]["width"]
    for task in source.cfg["data"]["tasks"]:
        path = source.cache_root / f"{task}.h5"
        with h5py.File(path, "w") as file:
            group = file.create_group(episode_key(task, "episode_0"))
            group.attrs.update(task=task, episode_id="episode_0")
            group["text"] = torch.zeros(width).numpy()
            for name in ("conditioning", "execution", "prefix", "current"):
                group.create_group(name)
            write_records(group["current"].create_group("0"), [
                leaf_record(f"{task}:current", torch.zeros(width), torch.zeros(3), 0.0,
                            "execution", "visual"),
            ])
        shard = {"sha256": source.module.sha256(path), "bytes": path.stat().st_size, "episodes": 1}
        manifest["shards"][path.name] = shard
        write_json(path.with_suffix(".complete.json"), {
            "kind": "encoder_cache_task", "status": "complete", "task": task,
            "identity_sha256": digest(manifest["identity"]), **shard,
        })
    write_json(manifest_path, manifest)
    completion = json.loads(source.complete.read_text())
    completion["cache_manifest_sha256"] = source.module.sha256(manifest_path)
    write_json(source.complete, completion)
    root = build_hierarchy_cache(source.cache_root)
    source.module.main(["--site", str(source.site), "--cache-name", "memory_fv_seed0"])
    plan_path = source.repo / "hpc/generated/memory_fv_seed0_compact_hierarchy/plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["hierarchy_cache_root"] == str(root)
    source.module.ready_plan(plan["startup_plan"])
    save_resume_fixture(source, plan)
    source.module.main(["--site", str(source.site), "--cache-name", "memory_fv_seed0", "--resume"])
