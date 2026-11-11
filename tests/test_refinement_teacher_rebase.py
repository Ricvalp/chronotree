"""Synthetic CPU guards for a new adopted teacher and three matched arms."""
from copy import deepcopy
import json

import pytest
import torch

from stmem.config import load_config
from stmem.inspection.refinement_data import TEACHER_REFERENCE
from stmem.inspection.refinement_utility import validate_measurement_models, validate_student_endpoint
from stmem.training.checkpoints import file_sha256


@pytest.fixture
def new_study(tmp_path):
    teacher_hash = "a" * 64
    deps = {
        "teacher_sha256": teacher_hash, "compact_sha256": teacher_hash,
        "teacher_step": 140000, "compact_step": 140000,
        "encoded_cache_id": "b" * 24, "hierarchy_cache_id": "c" * 24,
        "accepted_cache_plan": "synthetic-plan.json", "accepted_cache_plan_sha256": "d" * 64,
        "teacher_selection": "synthetic-selection.json", "teacher_selection_sha256": "e" * 64,
        "reference": dict(TEACHER_REFERENCE),
    }
    manifest = {
        "experiment_id": "synthetic-rebased-study", "dependencies": deps,
        "reference": dict(TEACHER_REFERENCE), "primary_endpoints": ["S_STD", "S_STAGE", "S_PACKET"],
        "protocol": {"student_training": {"steps": 100000}, "execute_targets": 8,
                     "gain_epsilon": 1e-6, "seed": 17, "precision": "fp32"},
        **{key: deps[key] for key in ("teacher_sha256", "compact_sha256", "encoded_cache_id", "hierarchy_cache_id")},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return manifest, path


def student_state(manifest, path):
    deps = manifest["dependencies"]
    study = {
        "variant": "S_STAGE", "experiment_id": manifest["experiment_id"],
        "manifest_sha256": file_sha256(path), "teacher_sha256": deps["teacher_sha256"],
        "encoded_cache_id": deps["encoded_cache_id"], "hierarchy_cache_id": deps["hierarchy_cache_id"],
        "initialization": "accepted_teacher_weights", "initialization_sha256": deps["teacher_sha256"],
        "optimizer_initialization": "fresh", "primary_updates": 100000,
        "development_steps": list(range(5000, 100000, 5000)),
    }
    return {"stage": "compact", "step": 100000,
            "provenance": {"dependencies": {"teacher": deps["teacher_sha256"]}},
            "config": {"training": {"utility_study": study, "steps": 100000, "schedule_steps": 100000}}}


def test_new_teacher_endpoint_uses_its_own_provenance_and_excludes_old_s0(new_study):
    manifest, path = new_study
    state = student_state(manifest, path)
    assert validate_student_endpoint(manifest, path, "S_STAGE", state, "validation")["endpoint"] == 100000
    wrong = deepcopy(state)
    wrong["provenance"]["dependencies"]["teacher"] = "f" * 64
    with pytest.raises(ValueError, match="different accepted-teacher"):
        validate_student_endpoint(manifest, path, "S_STAGE", wrong, "validation")
    wrong = deepcopy(state)
    wrong["config"]["training"]["utility_study"]["encoded_cache_id"] = "0" * 24
    with pytest.raises(ValueError, match="encoded_cache_id"):
        validate_student_endpoint(manifest, path, "S_STAGE", wrong, "validation")
    with pytest.raises(ValueError, match="outside"):
        validate_student_endpoint(manifest, path, "S0", {"stage": "compact", "step": 90000}, "train")
    state["config"]["training"]["utility_study"].update(
        initialization="S0_weights", initialization_sha256=manifest["teacher_sha256"], continuation_updates=100000)
    with pytest.raises(ValueError, match="predeclared common S0"):
        validate_student_endpoint(manifest, path, "S_STAGE", state, "train")


def test_teacher_compact_context_smoke_is_exact_and_never_a_primary_student(new_study):
    manifest, path = new_study
    cfg = load_config("configs/debug/teacher.yaml")
    manifest["configuration"] = cfg
    model = torch.nn.Linear(2, 2)
    other = deepcopy(model)
    state = {"stage": "teacher", "step": 140000, "config": cfg}
    identity = validate_student_endpoint(manifest, path, "S_INIT", state, "train")
    assert identity["compact_updates"] == 0 and identity["preflight_only"]
    args = dict(checkpoint_hash=manifest["teacher_sha256"], mode="smoke", partition="train")
    validate_measurement_models(manifest, "S_INIT", model, state, other, state, **args)
    for change in ({"mode": "main"}, {"partition": "validation"}, {"checkpoint_hash": "0" * 64}):
        with pytest.raises(ValueError, match="smoke/train only"):
            validate_measurement_models(manifest, "S_INIT", model, state, other, state, **{**args, **change})
    with torch.no_grad():
        other.weight.add_(1)
    with pytest.raises(ValueError, match="identical model tensors"):
        validate_measurement_models(manifest, "S_INIT", model, state, other, state, **args)
    with pytest.raises(ValueError, match="preflight"):
        validate_student_endpoint(manifest, path, "S_INIT", state, "validation")
    # The special smoke does not relax the normal teacher/compact stage check.
    with pytest.raises(ValueError, match="shared compact checkpoint"):
        validate_measurement_models(manifest, "S_STAGE", model, state, model, state, **args)


def test_generated_plan_defaults_to_three_new_students_and_eight_targets(new_study, tmp_path, monkeypatch):
    from stmem.inspection import refinement_generated

    manifest, path = new_study
    manifest["states"] = []
    monkeypatch.setattr(refinement_generated, "verify_manifest", lambda *args: manifest)
    monkeypatch.setattr(refinement_generated, "source_revision", lambda: {"source_sha256": "synthetic"})
    checkpoints = {}
    for student in manifest["primary_endpoints"]:
        checkpoint = tmp_path / f"{student}.pt"
        checkpoint.write_bytes(f"synthetic checkpoint {student}".encode())
        checkpoints[student] = checkpoint

    def folder(root, student, mode, checkpoint_hash, experiment_id, states, **kwargs):
        return tmp_path, {"experiment_id": experiment_id, "checkpoint_sha256": checkpoint_hash,
                          "step": 100000, "endpoint": 100000, "initialization": "accepted_teacher_weights"}

    monkeypatch.setattr(refinement_generated, "_measurement_folder", folder)
    plan = refinement_generated.create_plan(path, tmp_path / "generated.json", checkpoints=checkpoints)
    assert set(plan["students"]) == set(manifest["primary_endpoints"])
    assert plan["execute_targets"] == 8
    for historical in ("S0", "S_INIT"):
        with pytest.raises(ValueError, match="primary students"):
            refinement_generated.create_plan(path, tmp_path / "bad.json", students=[historical],
                                             checkpoints=checkpoints, limited_check=True)


def test_three_arm_confirmation_does_not_require_s0_or_accept_s_init_as_missing_arm(new_study, tmp_path):
    from stmem.inspection.refinement_report import assess_hypotheses

    manifest, _ = new_study
    manifest["states"] = [{"state_id": "state"}]
    effect = {"mean": 1.0, "interval": [0.5, 1.5]}
    result = {"inventory": [], "analyses": [], "population": {
        "selection": {}, "backgrounds": {"S_STD/validation/TREE/complete": effect}}}
    for student in manifest["primary_endpoints"]:
        result["inventory"].append({"student": student, "primary_endpoint": True, "mode": "main",
                                    "initialization": "accepted_teacher_weights", "declared_endpoint": 100000,
                                    "step": 100000})
        result["analyses"].append({"state": {"state_id": "state", "student": student}})
    assert assess_hypotheses(manifest, result, tmp_path)["H2"]["status"] == "supported"
    result["inventory"][-1].update(student="S_INIT", primary_endpoint=False)
    result["analyses"][-1]["state"]["student"] = "S_INIT"
    assert assess_hypotheses(manifest, result, tmp_path)["H2"]["status"] == "inconclusive"
