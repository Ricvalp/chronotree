"""CPU-only launch checks; all checkpoints and manifests below are synthetic."""
import json
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc import utility_router_routes as routes
from stmem.config import digest
from stmem.inspection import utility_router_routes as runtime


@pytest.fixture
def launch(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    pilot_root = repo / "runs/pilot"
    pilot_root.mkdir(parents=True)
    frozen = pilot_root / "frozen/student.pt"
    frozen.parent.mkdir()
    frozen.write_bytes(b"synthetic frozen student")
    parent = repo / "original.json"
    parent.write_text("{}")
    sources = []
    for name in ("old", "new"):
        root = repo / name
        (root / "src").mkdir(parents=True)
        (root / "src/example.py").write_text(f"# synthetic {name}\n")
        (root / "hpc").mkdir()
        (root / "hpc/plan.py").write_text("# administrative fixture\n")
        (root / ".utility-source.json").write_text(json.dumps({"git": "a" * 40, "source_sha256": "b" * 64}))
        sources.append(root)
    study = {"experiment_id": "synthetic-child", "source": {"source_sha256": "b" * 64, "git": "a" * 40},
             "student": {"checkpoint": str(frozen), "sha256": routes.sha256(frozen), "step": 95000, "name": "S_STD"},
             "parent_manifest": str(parent), "parent_manifest_sha256": routes.sha256(parent),
             "protocol": {"budget": 128, "execute_targets": 8, "flow_steps": 8, "seed": 1,
                          "controller": "packet", "fit": {"steps": 5000}},
             "configuration": {}, "dependencies": {}, "data_identity": "synthetic-data",
             "states": [{"state_id": str(i), "partition": "development", "cache_eligible": True}
                        for i in range(128)] + [{"state_id": "fit", "partition": "fit"}]}
    study_path = pilot_root / "manifest.json"
    study_path.write_text(json.dumps(study))
    fits = {}
    for reader in ("TREE", "FLAT"):
        for target in ("immediate", "packet"):
            folder = pilot_root / "fits" / f"{reader.lower()}_{target}"
            (folder / "evaluation").mkdir(parents=True)
            path = folder / "last.pt"
            path.write_bytes(f"synthetic router {reader} {target}".encode())
            (folder / "complete.json").write_text("{}")
            (folder / "evaluation/summary.json").write_text("{}")
            fits[reader, target] = path
    pilot = {"name": "pilot", "study": str(study_path), "code_root": str(sources[0])}
    pilot_path = repo / "hpc/generated/pilot/plan.json"
    pilot_path.parent.mkdir(parents=True)
    pilot_path.write_text(json.dumps(pilot))
    checked = []
    monkeypatch.setattr(routes, "read_site", lambda _: {"STMEM_REPO_ROOT": str(repo), "STMEM_PYTHON": sys.executable})
    monkeypatch.setattr(routes, "evaluation_study", lambda value: checked.append("old-runtime") or study)
    monkeypatch.setattr(routes, "evaluation_inputs", lambda *_: checked.append("four-completed-fits") or (frozen, None, fits))
    monkeypatch.setattr(routes, "current_source_snapshot", lambda *_: sources[1])
    monkeypatch.setattr(routes, "verify_study_source", lambda *_: {})

    def build_manifest(path, child, parent_runtime, routers, source, *, workers, batch_size):
        value = {"schema": routes.SCHEMA, "source": source, "parent_study": {"path": str(path)},
                 "parent_runtime": parent_runtime, "routers": routers, "student": child["student"],
                 "states": [s for s in child["states"] if s["partition"] == "development"],
                 "protocol": {**child["protocol"], "seed": 2, "workers": workers, "batch_size": batch_size,
                              "discovery_draws": 4, "confirmation_draws": 4}}
        value["experiment_id"] = digest(value)
        return value

    real_builder = runtime.build_manifest
    monkeypatch.setattr(runtime, "build_manifest", build_manifest)
    calls, actual_run = [], subprocess.run

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["bash", "-n"]:
            return actual_run(command, **kwargs)
        if command[0] == "sbatch":
            return SimpleNamespace(stdout="321\n")
        if "stmem.inspection.utility_router_routes" in command:
            assert "report" in command
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
            return SimpleNamespace(returncode=0)
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "run", run)
    return SimpleNamespace(repo=repo, pilot=pilot, study=study, frozen=frozen, fits=fits, checked=checked,
                           sources=sources, calls=calls, real_builder=real_builder,
                           args=["--name", "routes", "--pilot-name", "pilot"])


def test_dry_run_validates_original_inputs_and_renders_typed_array(launch):
    routes.main([*launch.args, "--dry-run"])
    assert launch.checked == ["old-runtime", "four-completed-fits"]
    root = launch.repo / "hpc/generated/routes"
    plan = routes.read_json(root / "plan.json")
    manifest = routes.read_json(plan["manifest"])
    assert manifest["student"]["checkpoint"] == str(launch.frozen)
    assert len(manifest["states"]) == 128
    assert all(state["partition"] == "development" for state in manifest["states"])
    assert manifest["protocol"]["seed"] != launch.study["protocol"]["seed"]
    receipt = routes.read_json(root / "original_runtime_validation.json")
    assert receipt["status"] == "complete"
    assert receipt["study_sha256"] == routes.sha256(Path(launch.pilot["study"]))
    assert manifest["parent_runtime"]["sha256"] == routes.sha256(root / "original_runtime_validation.json")
    job = routes.read_json(plan["job"])
    script = Path(job["script"]).read_text()
    assert "#SBATCH --gres=gpu:b200:1" in script and "#SBATCH --partition=aiq" in script
    assert "#SBATCH --array=0-3%4" in script and "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=64G" in script and "--exclusive" not in script
    assert "WANDB_PROJECT=chronotree-routers" in script
    assert '"${SLURM_ARRAY_TASK_ID:-0}"' in script
    assert job["commands"][0][-1] == "3.8"
    assert len(manifest["routers"]) == 4
    assert not any(command[0] == "sbatch" for command, _ in launch.calls)


def test_submit_and_resume_keep_one_active_array_and_pinned_source(launch, monkeypatch):
    routes.main([*launch.args, "--submit"])
    assert len([c for c, _ in launch.calls if c[0] == "sbatch"]) == 1
    monkeypatch.setattr(routes, "active_user_job_ids", lambda: {"321"})
    monkeypatch.setattr(routes, "current_source_snapshot", lambda *_: pytest.fail("Resume exported new source"))
    routes.main([*launch.args, "--submit", "--resume"])
    assert len([c for c, _ in launch.calls if c[0] == "sbatch"]) == 1
    with pytest.raises(ValueError, match="Immutable --workers"):
        routes.main([*launch.args, "--workers", "8", "--dry-run"])


def test_changed_router_stops_before_submission(launch):
    routes.main([*launch.args, "--dry-run"])
    launch.fits["TREE", "packet"].write_bytes(b"different router")
    with pytest.raises(ValueError, match="Frozen full-route diagnostic input changed"):
        routes.main([*launch.args, "--submit"])
    assert not any(command[0] == "sbatch" for command, _ in launch.calls)


def test_status_is_read_only_and_report_waits_for_all_workers(launch, monkeypatch):
    routes.main([*launch.args, "--stage", "status"])
    assert not (launch.repo / "hpc/generated/routes").exists()
    routes.main([*launch.args, "--dry-run"])
    with pytest.raises(ValueError, match="Every full-route worker"):
        routes.main([*launch.args, "--stage", "report"])
    plan = routes.read_json(launch.repo / "hpc/generated/routes/plan.json")
    output = Path(plan["output"])
    for worker in range(4):
        (output / f"worker_{worker}.json").write_text(json.dumps({
            "experiment_id": plan["experiment_id"], "status": "complete", "completed": 32, "planned": 32}))
    routes.main([*launch.args, "--stage", "report"])
    report = [command for command, _ in launch.calls if "stmem.inspection.utility_router_routes" in command]
    assert len(report) == 1 and report[0][-2:] == ["--output", str(output)]
    monkeypatch.setattr(routes, "validate_plan", lambda *_: pytest.fail("Status must not validate inputs or source"))
    routes.main([*launch.args, "--stage", "status"])


def test_incomplete_or_changed_protocol_is_rejected_before_rendering(launch):
    launch.study["states"] = launch.study["states"][:127]
    with pytest.raises(ValueError, match="exact 128"):
        routes.main([*launch.args, "--dry-run"])
    assert not any(command[0] == "sbatch" for command, _ in launch.calls)


def test_actual_runtime_adopts_and_verifies_launcher_receipt(launch, monkeypatch):
    monkeypatch.setattr(runtime, "build_manifest", launch.real_builder)
    routes.main([*launch.args, "--dry-run"])
    plan = routes.read_json(launch.repo / "hpc/generated/routes/plan.json")
    manifest = routes.read_json(plan["manifest"])
    # The actual diagnostic guard verifies adoption/source/manifest hashes;
    # only the older study's unavailable synthetic data dependencies are mocked.
    parent_checks = []

    def verify_old(path, **kwargs):
        parent_checks.append((str(path), kwargs))
        return launch.study

    old_study = importlib.import_module("stmem.training.utility_router_study")
    monkeypatch.setattr(old_study, "verify_study", verify_old)
    monkeypatch.setattr(runtime, "source_revision", lambda: manifest["source"])
    assert runtime.verify_manifest(plan["manifest"]) == manifest
    assert parent_checks == [(launch.pilot["study"], {"verify_source": False, "verify_dependencies": True})]
    assert manifest["protocol"]["methods"] == list(runtime.METHODS)
    assert manifest["study"]["experiment_id"] == launch.study["experiment_id"]
    receipt = Path(manifest["parent_runtime"]["path"])
    receipt.write_text(receipt.read_text() + "\n")
    with pytest.raises(ValueError, match="verification receipt changed"):
        runtime.verify_manifest(plan["manifest"])
