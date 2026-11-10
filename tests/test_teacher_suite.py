"""CPU-only lifecycle checks for the teacher suite; no model inference."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from stmem.inspection import teacher_suite as suite


def example_plan(tmp_path):
    output = tmp_path / "output"
    return {
        "schema_version": 1, "scope": "Synthetic launcher fixture, no scientific result",
        "output": str(output), "bundle": str(tmp_path / "bundle"), "model": "frozen_10hz",
        "checkpoint": {"path": str(tmp_path / "checkpoint.pt"), "sha256": "a" * 64},
        "assets_config": str(tmp_path / "assets.yaml"), "clip_checkpoint": str(tmp_path / "clip.pt"),
        "reference": {key: str(tmp_path / "reference" / key) for key in suite.PATH_NAMES},
        "paths": {key: str(output / key) for key in suite.PATH_NAMES},
    }


def configured(tmp_path, monkeypatch):
    plan = example_plan(tmp_path)
    monkeypatch.setattr(suite, "create_plan", lambda *_args: copy.deepcopy(plan))
    monkeypatch.setattr(suite, "verify_plan", lambda _plan: None)
    return plan, ["--bundle", plan["bundle"], "--output", plan["output"]]


def test_dry_run_does_not_create_results_or_launch_children(tmp_path, monkeypatch, capsys):
    plan, argv = configured(tmp_path, monkeypatch)
    monkeypatch.setattr(suite, "run_command", lambda *_args: pytest.fail("dry-run launched a child"))
    suite.main([*argv, "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert list(result["commands"]) == ["usual", "history", "recent"]
    assert not Path(plan["output"]).exists()
    assert result["commands"]["history"][-4:] == ["--execute-actions", "10", "--clip-checkpoint", plan["clip_checkpoint"]]


def test_resume_uses_manifest_boundaries_and_completed_usual_stays_cpu_only(tmp_path):
    plan = example_plan(tmp_path)
    usual = Path(plan["paths"]["usual"])
    history = Path(plan["paths"]["history"])
    usual.mkdir(parents=True)
    history.mkdir()
    (usual / "inspection_manifest.json").write_text("{}")
    (usual / "summary.json").write_text('{"status":"complete"}')
    (history / "manifest.json").write_text("{}")
    commands = suite.stage_commands(plan, 3, resume=True)
    assert "--resume" in commands["usual"] and "--report-only" in commands["usual"]
    assert "--resume" in commands["history"]
    assert "--resume" not in commands["recent"]
    assert commands["history"][commands["history"].index("--workers") + 1] == "3"
    assert commands["recent"][commands["recent"].index("--reference-output") + 1] == str(history)


def test_cpu_audit_stops_before_simulation_and_can_resume_without_replanning(tmp_path, monkeypatch):
    plan, argv = configured(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(suite, "run_command", lambda name, command, *_args: calls.append((name, command)))
    monkeypatch.setattr(suite, "build_reports", lambda *_args: pytest.fail("audit called reports"))
    suite.main([*argv, "--audit-only"])
    path = Path(plan["output"]) / "teacher_inspection.json"
    original = path.read_bytes()
    assert [name for name, _ in calls] == ["usual"]
    assert "--audit-only" in calls[0][1]
    assert json.loads((path.parent / "status.json").read_text())["status"] == "audit_complete"
    suite.main([*argv, "--audit-only", "--resume", "--workers", "4"])
    assert path.read_bytes() == original
    plan["checkpoint"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="inputs/source/reference changed"):
        suite.main([*argv, "--audit-only", "--resume"])
    assert path.read_bytes() == original


def test_stage_failure_keeps_completed_outputs_and_never_runs_dependents(tmp_path, monkeypatch):
    plan, argv = configured(tmp_path, monkeypatch)
    calls = []
    def run(name, *_args):
        calls.append(name)
        if name == "history":
            raise RuntimeError("diagnosed worker interruption")
        (Path(plan["output"]) / "preserved.json").write_text("completed evidence")
    monkeypatch.setattr(suite, "run_command", run)
    monkeypatch.setattr(suite, "build_reports", lambda *_args: pytest.fail("reported incomplete simulation"))
    with pytest.raises(RuntimeError, match="interruption"):
        suite.main(argv)
    assert calls == ["usual", "history"]
    root = Path(plan["output"])
    assert (root / "preserved.json").read_text() == "completed evidence"
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "failed" and "interruption" in status["error"]


def test_report_only_never_starts_history_workers(tmp_path, monkeypatch):
    plan, argv = configured(tmp_path, monkeypatch)
    output = Path(plan["output"])
    output.mkdir()
    (output / "teacher_inspection.json").write_text(json.dumps(plan))
    calls = []
    monkeypatch.setattr(suite, "run_command", lambda name, command, *_args: calls.append((name, command)))
    monkeypatch.setattr(suite, "build_reports", lambda _plan: {})
    suite.main([*argv, "--report-only"])
    assert len(calls) == 1 and calls[0][0] == "usual" and "--report-only" in calls[0][1]
    assert json.loads((output / "status.json").read_text())["status"] == "complete"


def test_deadline_before_launch_never_starts_a_process(tmp_path, monkeypatch):
    monkeypatch.setattr(suite.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("late process launch"))
    with pytest.raises(TimeoutError, match="max-hours"):
        suite.run_command("fixture", [sys.executable, "-c", "pass"], tmp_path, time.monotonic() - 1)


def test_deadline_during_stage_terminates_real_cpu_child(tmp_path, monkeypatch):
    """Short-lived process test: bounded execution must not leave a worker."""
    actual_popen, children = subprocess.Popen, []
    def start(*args, **kwargs):
        child = actual_popen(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(suite.subprocess, "Popen", start)
    with pytest.raises(TimeoutError, match="max-hours"):
        suite.run_command("fixture", [sys.executable, "-c", "import time; time.sleep(120)"],
                          tmp_path, time.monotonic() + 0.2)
    assert len(children) == 1 and children[0].poll() is not None


@pytest.mark.parametrize("hours", ["0", "-1", "nan", "inf"])
def test_unbounded_or_invalid_time_limit_rejected(hours):
    with pytest.raises(SystemExit):
        suite.main(["--bundle", "/unused", "--max-hours", hours])


def test_integrity_check_detects_reference_changed_after_planning(tmp_path, monkeypatch):
    files = []
    for name in ("reference.json", "checkpoint.pt", "assets.yaml", "clip.pt"):
        path = tmp_path / name
        path.write_text(name)
        files.append(path)
    from stmem.training.checkpoints import file_sha256
    plan = {"reference_identities": {str(files[0]): file_sha256(files[0])},
            "checkpoint": {"path": str(files[1]), "sha256": file_sha256(files[1])},
            "assets_config": str(files[2]), "assets_config_sha256": file_sha256(files[2]),
            "clip_checkpoint": str(files[3]), "clip_sha256": file_sha256(files[3]),
            "source_sha256": "fixed"}
    monkeypatch.setattr(suite, "source_revision", lambda: {"source_sha256": "fixed"})
    suite.verify_plan(plan)
    files[0].write_text("replaced baseline")
    with pytest.raises(ValueError, match="input changed"):
        suite.verify_plan(plan)


def test_detached_conflict_fails_before_spawning(tmp_path, monkeypatch):
    plan, argv = configured(tmp_path, monkeypatch)
    output = Path(plan["output"])
    output.mkdir()
    (output / "teacher_inspection.json").write_text(json.dumps(plan))
    monkeypatch.setattr(suite, "detach", lambda *_args: pytest.fail("spawned conflicting detached run"))
    with pytest.raises(FileExistsError, match="resume"):
        suite.main([*argv, "--detach"])


def test_compact_report_uses_visual_precision_probe_not_command_control():
    def row(kind, value):
        return {"kind": kind, "variant": "baseline", "state_subset": "all_states",
                "metrics": {f"native{n}_tcp_error_mm": {"reference": value, "current": value + 1}
                            for n in (4, 16)}}
    usual = {"offline_summary_pairs": [row("commands", 999), row("visual", 5)],
             "grounding": {side: {"first_close_observed_xy_mm": None, "max_cube_lift_mm": 0}
                           for side in ("reference", "current")}, "handoff": {"paired_runs": []}}
    result = suite.usual_key_metrics(usual)
    assert [r["reference"] for r in result[:2]] == [5, 5]
    assert result[2]["reference"] is None
