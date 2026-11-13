"""CPU-only inspection orchestration/identity tests; fixtures are synthetic."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from stmem.config import digest
from stmem.inspection import student_suite as suite
from stmem.training.checkpoints import atomic_json, file_sha256


def audit(**updates):
    return {"config": {}, "teacher_sha256": "a" * 64, "frozen_encoder_sha256": "b" * 64,
            "normalizer_sha256": "c" * 64, "step": 100, "variant": "S_STD", **updates}


@pytest.mark.parametrize("field", ["teacher_sha256", "frozen_encoder_sha256", "normalizer_sha256"])
def test_shared_evidence_rejects_different_frozen_dependencies(field, monkeypatch):
    monkeypatch.setattr(suite, "validate_policy_config", lambda *_: None)
    with pytest.raises(ValueError, match=field):
        suite.validate_shared_models({"s_std": audit(), "s_packet": audit(variant="S_PACKET", **{field: "d" * 64})})
    suite.validate_shared_models({"s_std": audit(), "s_packet": audit(step=1000, variant="S_PACKET")})


def test_student_labels_cannot_hide_wrong_variant_or_identical_weights(monkeypatch):
    monkeypatch.setattr(suite, "validate_policy_config", lambda *_: None)
    with pytest.raises(ValueError, match="variant"):
        suite.validate_shared_models({"s_packet": audit()})
    with pytest.raises(ValueError, match="Duplicate"):
        suite.validate_shared_models({"first": audit(sha256="a"), "second": audit(sha256="a")})


def test_optional_teacher_requires_exact_parent_and_same_normalization(monkeypatch):
    monkeypatch.setattr(suite, "validate_policy_config", lambda *_: None)
    teacher = audit(sha256="a" * 64)
    suite.validate_shared_models({"s_std": audit()}, teacher)
    with pytest.raises(ValueError, match="exact checkpoint"):
        suite.validate_shared_models({"s_std": audit()}, {**teacher, "sha256": "d" * 64})
    with pytest.raises(ValueError, match="normalizer"):
        suite.validate_shared_models({"s_std": audit()}, {**teacher, "normalizer_sha256": "d" * 64})


def sealed_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "policy_source_hash", lambda: "policy")
    monkeypatch.setattr(suite, "diagnostic_source_hash", lambda: "inspection")
    monkeypatch.setattr(suite, "resolve_assets", lambda _: ({"prepared": "fixed"}, {"prepared": "hash"}))
    checkpoint, config = tmp_path / "checkpoint.pt", tmp_path / "assets.yaml"
    checkpoint.write_bytes(b"synthetic-checkpoint-integrity-only")
    config.write_bytes(b"synthetic-assets")
    plan = {"schema": suite.SCHEMA, "output": str(tmp_path), "source_sha256": "policy",
            "diagnostic_source_sha256": "inspection", "teacher": None,
            "checkpoints": {"s_std": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)}},
            "assets_config": str(config), "assets_config_sha256": file_sha256(config),
            "assets": {"prepared": "fixed"}, "asset_hashes": {"prepared": "hash"}}
    plan["identity"] = digest(plan)
    return plan


def test_plan_seals_protocol_inputs_and_diagnostic_implementation(tmp_path, monkeypatch):
    plan = sealed_plan(tmp_path, monkeypatch)
    suite.verify_plan(plan)
    altered = deepcopy(plan)
    altered["teacher"] = {"sha256": "other"}
    with pytest.raises(ValueError, match="seal"):
        suite.verify_plan(altered)
    monkeypatch.setattr(suite, "diagnostic_source_hash", lambda: "new-code")
    with pytest.raises(ValueError, match="implementation"):
        suite.verify_plan(plan)


def test_plan_catches_checkpoint_replacement_and_changed_assets(tmp_path, monkeypatch):
    plan = sealed_plan(tmp_path, monkeypatch)
    Path(plan["checkpoints"]["s_std"]["path"]).write_bytes(b"replacement")
    with pytest.raises(ValueError, match="checkpoint changed"):
        suite.verify_plan(plan)
    plan = sealed_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(suite, "resolve_assets", lambda _: ({"prepared": "fixed"}, {"prepared": "changed"}))
    with pytest.raises(ValueError, match="asset contents"):
        suite.verify_plan(plan)


def test_artifact_publication_and_resume_reject_incomplete_or_changed_bytes(tmp_path):
    plan = {"identity": "fixed"}
    path = tmp_path / "case.json"
    assert not suite.verified_artifact(path, plan)
    path.write_text('{"synthetic": true}')
    assert not suite.verified_artifact(path, plan)
    suite.seal_artifact(path, plan)
    assert suite.verified_artifact(path, plan)
    with pytest.raises(ValueError, match="artifact changed"):
        suite.verified_artifact(path, {"identity": "another"})
    path.write_text('{"synthetic": false}')
    with pytest.raises(ValueError, match="artifact changed"):
        suite.verified_artifact(path, plan)


def test_published_stage_may_not_silently_rerun_missing_results(tmp_path):
    path = tmp_path / "case.json"
    path.write_text("synthetic")
    plan = {"identity": "fixed", "output": str(tmp_path)}
    job = {"name": "measure_s_std"}
    assert not suite.job_complete(plan, job)
    atomic_json({"identity": "fixed", "files": {str(path): file_sha256(path)}},
                tmp_path / "completed/measure_s_std.json")
    assert suite.job_complete(plan, job)
    path.unlink()
    with pytest.raises(ValueError, match="artifact changed"):
        suite.job_complete(plan, job)


def test_report_and_resume_reject_changed_generated_arrays(tmp_path):
    path = tmp_path / "chunk.npz"
    path.write_bytes(b"synthetic-array-integrity-fixture")
    result = {"generated": [{"hashes": {str(path): file_sha256(path)}}]}
    suite.validate_generated_artifacts(result)
    path.write_bytes(b"altered")
    with pytest.raises(ValueError, match="generated-action artifact changed"):
        suite.validate_generated_artifacts(result)


def test_deadline_does_not_launch_and_terminates_real_child(tmp_path, monkeypatch):
    with pytest.raises(TimeoutError, match="max-hours"):
        suite.run_command("expired", ["must-not-launch"], tmp_path, time.monotonic() - 1)
    original, children = subprocess.Popen, []
    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(suite.subprocess, "Popen", spawn)
    with pytest.raises(TimeoutError, match="max-hours"):
        suite.run_command("bounded", [sys.executable, "-c", "import time; time.sleep(30)"],
                          tmp_path, time.monotonic() + 0.1)
    assert len(children) == 1 and children[0].poll() is not None


@pytest.mark.parametrize("args", [["--max-hours", "nan"], ["--states", "0"], ["--draws", "1"],
                                  ["--generated-cases", "30"], ["--budgets", "1"],
                                  ["--stages", "utility", "utility"]])
def test_invalid_bounds_rejected_before_loading(args, monkeypatch):
    monkeypatch.setattr(suite, "create_plan", lambda *_: pytest.fail("unbounded plan built"))
    with pytest.raises(SystemExit):
        suite.main(["--bundle", "/unused", *args])


def test_report_keeps_episode_unit_and_marks_unmatched_steps(tmp_path, monkeypatch):
    plan = {"identity": "fixed", "output": str(tmp_path), "matched_training_steps": False, "teacher": None,
            "checkpoints": {"s_std": {"step": 83000}}, "protocol": {"budgets": [128]}}
    states = [{"state_id": str(i), "task": "T", "episode_id": episode, "source_index": i}
              for i, episode in enumerate(["episode_0", "episode_0", "episode_1"])]
    atomic_json({"states": states}, tmp_path / "states.json")
    suite.seal_artifact(tmp_path / "states.json", plan)
    for i, state in enumerate(states):
        value = [1, 3, 10][i]
        metrics = {k: value for k in ("packet_loss", "random_loss", "packet_vs_random", "packet_vs_immediate",
                                     "packet_vs_dispersed", "packet_vs_low")}
        row = {"reader": "TREE", "family": "packet", "budget": 128, "ranking_eligible": True,
               "winner_reversal": None, **metrics}
        path = tmp_path / "measurements/s_std" / f"{i}.json"
        atomic_json({"label": "s_std", "state": state, "summary": {"selection": [row]}}, path)
        suite.seal_artifact(path, plan)
    monkeypatch.setattr(suite, "stage_commands", lambda *_: [])
    from stmem.inspection import student_simulation
    monkeypatch.setattr(student_simulation, "simulation_report_links", lambda _: [])
    from stmem.inspection import student_report
    monkeypatch.setattr(student_report, "summarize_local_measurements", lambda _: {})
    monkeypatch.setattr(student_report, "write_diagnostic_figures", lambda *_: [])
    result = suite.build_report(plan)
    row = next(row for row in result["comparisons"] if row["reader"] == "TREE")
    assert row["episodes"] == 2 and row["states"] == 3
    assert row["confirmation_episode_means"]["packet_vs_random"] == 6  # ((1+3)/2 + 10)/2.
    assert any("steps differ" in note for note in result["limitations"])
    assert any("Teacher agreement is not measured" in note for note in result["limitations"])
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "complete"
