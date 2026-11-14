"""CPU-only launch contracts for typed GPUs and immutable utility runtimes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "utility_runtime_launcher", Path(__file__).parents[1] / "hpc/refinement_utility.py")
launcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(launcher)


@pytest.fixture
def launch_fixture(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src/current.py").write_text("# current administrative checkout\n")
    checkpoint = repo / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic dry-run checkpoint, not model weights")
    manifest_path = repo / "manifest.json"
    manifest_path.write_text("{}")
    manifest = {"dependencies": {"compact": str(checkpoint)},
                "protocol": {"student_training": {"steps": 100000}}}
    monkeypatch.setattr(launcher, "read_site", lambda _: {
        "STMEM_REPO_ROOT": str(repo), "STMEM_PYTHON": sys.executable})
    monkeypatch.setattr(launcher, "verify_manifest", lambda *_a, **_kw: manifest)
    calls = []
    shell_run = subprocess.run

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["bash", "-n"]:
            return shell_run(command, **kwargs)
        if command[:2] == ["sbatch", "--parsable"]:
            return SimpleNamespace(stdout="12345\n", returncode=0)
        raise AssertionError(f"Unexpected external command: {command}")

    monkeypatch.setattr(launcher.subprocess, "run", run)
    args = ["--name", "fixture", "--manifest", str(manifest_path), "--stage", "measure",
            "--partition", "aiq", "--gpu-type", "b200"]
    return SimpleNamespace(repo=repo, checkpoint=checkpoint, manifest_path=manifest_path,
                           manifest=manifest, calls=calls, args=args)


def plan_of(fixture):
    return json.loads(next((fixture.repo / "hpc/generated/fixture").glob("measure_*.json")).read_text())


def test_typed_b200_dry_run_preserves_science_and_has_no_submission(launch_fixture):
    fixture = launch_fixture
    launcher.main([*fixture.args, "--dry-run", "--shards", "4"])
    plan = plan_of(fixture)
    script = Path(plan["script"]).read_text()
    assert "#SBATCH --gres=gpu:b200:1" in script
    assert "#SBATCH --partition=aiq" in script
    assert "#SBATCH --array=0-3%4" in script
    assert "--exclusive" not in script
    assert plan["gres"] == "gpu:b200:1"
    assert plan["inputs"][str(fixture.checkpoint)] == launcher.sha256(fixture.checkpoint)
    assert plan["commands"][0][-2:] == ["--max-hours", "3.8"]
    assert all(call[:2] == ["bash", "-n"] for call in fixture.calls)


def test_resume_accepts_purged_job_but_rejects_active_array(launch_fixture, monkeypatch):
    fixture = launch_fixture
    launcher.main([*fixture.args, "--submit"])
    monkeypatch.setattr(launcher, "active_user_job_ids", lambda: {"12345"})
    with pytest.raises(ValueError, match="still active"):
        launcher.main([*fixture.args, "--resume", "--submit"])
    monkeypatch.setattr(launcher, "active_user_job_ids", lambda: set())
    launcher.main([*fixture.args, "--resume", "--submit"])
    assert sum(call[0] == "sbatch" for call in fixture.calls) == 2


def test_isolated_runtime_retains_original_panel_and_ignores_main_src_edits(launch_fixture, monkeypatch):
    fixture = launch_fixture
    runtime = fixture.repo / "hpc/generated/original_source"
    (runtime / "src").mkdir(parents=True)
    (runtime / "src/original.py").write_text("# unchanged scientific runtime\n")
    (runtime / "hpc").mkdir()
    for name in ("plan.py", "profile_memory.py"):
        (runtime / "hpc" / name).write_text("# synthetic runtime\n")
    checked = []

    def manifest(*args, **kwargs):
        assert args[1] == runtime and args[2] == fixture.manifest_path
        assert kwargs["verify_source"] is True
        return fixture.manifest

    monkeypatch.setattr(launcher, "runtime_manifest", manifest)
    monkeypatch.setattr(launcher, "validate_runtime_checkpoint", lambda *args: checked.append(args))
    args = [*fixture.args, "--code-root", str(runtime), "--dry-run"]
    launcher.main(args)
    before = plan_of(fixture)
    script = Path(before["script"]).read_text()
    assert before["source_sha256"] == launcher.source_hash(runtime)
    assert before["manifest"] == str(fixture.manifest_path)
    assert str(runtime / "src") in script
    assert f"cd {runtime}" in script
    assert "Runtime launcher changed" in script
    assert before["runtime_hpc"][str(runtime / "hpc/plan.py")]
    assert checked[0][1:4] == (runtime, fixture.manifest_path, "S0")
    (fixture.repo / "src/current.py").write_text("# newer unrelated main checkout\n")
    launcher.main(args)
    assert plan_of(fixture) == before
    (runtime / "src/original.py").write_text("# changed runtime must fail\n")
    with pytest.raises(ValueError, match="Saved plan identity differs"):
        launcher.main(args)


@pytest.mark.parametrize("flag,value", [("--gpu-type", "b200\n#SBATCH --exclusive"),
                                        ("--partition", "aiq;bad")])
def test_resource_arguments_reject_injection(flag, value):
    with pytest.raises(SystemExit):
        launcher.main([flag, value])


@pytest.mark.parametrize("value,expected", [("04:00:00", 3.8), ("01:00:00", .8),
                                           ("1-00:00:00", 23.8)])
def test_worker_bound_tracks_allocation(value, expected):
    assert float(launcher.measurement_hours(value)) == pytest.approx(expected)


def test_too_short_measurement_rejected():
    with pytest.raises(ValueError, match="exceed 12 minutes"):
        launcher.measurement_hours("00:10:00")


def test_runtime_preflight_imports_isolated_source_on_cpu(tmp_path):
    package = tmp_path / "src/stmem/inspection"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "refinement_data.py").write_text(
        "import os\ndef verify_manifest(path, verify_source=True):\n"
        "    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
        "    assert verify_source\n"
        "    return {'runtime': 'original', 'path': path}\n")
    result = launcher.runtime_manifest(sys.executable, tmp_path, tmp_path / "manifest.json")
    assert result == {"runtime": "original", "path": str(tmp_path / "manifest.json")}
