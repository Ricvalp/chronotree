from copy import deepcopy
import json

import pytest

from stmem.inspection.refinement_report import assess_hypotheses, student_training_artifacts, write_report
from stmem.training.checkpoints import file_sha256


def report_fixture():
    manifest = {"experiment_id": "test", "states": [{"state_id": "s0"}], "kind": "synthetic_test",
                "source": {"git": "test", "source_sha256": "test"}}
    result = {"inventory": [], "analyses": [], "population": {"selection": {}, "backgrounds": {}}}
    for student in ("S0", "S_STD", "S_STAGE", "S_PACKET"):
        result["inventory"].append({"student": student, "primary_endpoint": True, "mode": "main",
                                    "measured_states": 1, "status": "complete", "step": 30000,
                                    "declared_endpoint": 30000, "initialization": "accepted_teacher_weights"})
        result["analyses"].append({"state": {"state_id": "s0", "student": student}})
    return manifest, result


def test_hypothesis_assessments_depend_on_complete_endpoints_and_qualified_partition(tmp_path):
    manifest, result = report_fixture()
    effect = {"mean": 2., "interval": [1., 3.], "episodes": 8}
    result["population"]["backgrounds"] = {"S_STD/validation/TREE/complete": effect}
    result["population"]["selection"] = {"S_STD/validation/TREE/packet/128": {"packet_vs_immediate": effect}}
    assessed = assess_hypotheses(manifest, result, tmp_path)
    assert assessed["H2"]["status"] == assessed["H3"]["status"] == "supported"
    assert assessed["H5"]["status"] == "inconclusive"
    partial = deepcopy(result)
    partial["analyses"].pop()
    assert assess_hypotheses(manifest, partial, tmp_path)["H2"]["status"] == "inconclusive"
    mismatched = deepcopy(result)
    mismatched["inventory"][-1]["initialization"] = "S0_weights"
    assert assess_hypotheses(manifest, mismatched, tmp_path)["H2"]["status"] == "inconclusive"
    result["population"]["backgrounds"] = {"S_STD/train/TREE/complete": effect}
    assert assess_hypotheses(manifest, result, tmp_path)["H2"]["status"] == "inconclusive"
    (tmp_path / "hypothesis_assessments.json").write_text(json.dumps({"experiment_id": "other"}))
    with pytest.raises(ValueError, match="another experiment"):
        assess_hypotheses(manifest, result, tmp_path)


def test_training_curves_check_identity_stop_at_endpoint_and_deduplicate_retries(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic_identity_only")
    identity = {"checkpoint_path": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint), "step": 2}
    (tmp_path / "metrics.jsonl").write_text('\n'.join(json.dumps(row) for row in [
        {"step": 1, "training_action_loss": 3.}, {"step": 2, "training_action_loss": 4.},
        {"step": 2, "training_action_loss": 2.}, {"step": 3, "training_action_loss": 1.}]))
    (tmp_path / "study_strata_rank0.jsonl").write_text('\n'.join(json.dumps(row) for row in [
        {"step": 1, "dimension": "family", "value": "TREE", "count": 4, "mean_expert_fm_loss": 3.},
        {"step": 1, "dimension": "family", "value": "TREE", "count": 4, "mean_expert_fm_loss": 2.}]))
    analyses = [{"state": {"student": "S_STAGE"}, "identity": identity}]
    result = student_training_artifacts(analyses)["S_STAGE"]
    assert [r["training_action_loss"] for r in result["learning_curve"]] == [3., 2.]
    assert result["realized_strata"][0]["examples"] == 4
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        student_training_artifacts(analyses)


def test_empty_report_keeps_missing_matrix_explicit_and_renders_new_sections(tmp_path):
    manifest, result = report_fixture()
    manifest['protocol'] = {'student_training': {'steps': 100000}}
    result["analyses"] = []
    for row in result["inventory"]:
        row.update(measured_states=0, status="not_run")
    write_report(manifest, result, tmp_path)
    text = (tmp_path / "report.md").read_text()
    assert "First-ten-target secondary control" in text
    assert "Prevalence, null effects and all-negative pools" in text
    assert "Matched student comparisons" in text
    assert "declares 100,000 updates" in text and "recorded primary endpoint" in text
    assert "not_run" in text


def test_explicitly_ineligible_states_are_accounted_not_required_as_measurements(tmp_path):
    manifest, result = report_fixture()
    effect = {"mean": 2., "interval": [1., 3.], "episodes": 8}
    result["population"]["backgrounds"] = {"S_STD/validation/TREE/complete": effect}
    manifest["states"].append({"state_id": "excluded", "cache_eligible": False,
                               "cache_ineligible_reason": "no_complete_causal_horizon"})
    assessed = assess_hypotheses(manifest, result, tmp_path)
    assert assessed["H2"]["status"] == "supported"
    assert assessed["H2"]["predeclared_excluded_states"] == 1
    del manifest["states"][-1]["cache_ineligible_reason"]
    assert assess_hypotheses(manifest, result, tmp_path)["H2"]["status"] == "inconclusive"
    manifest["states"] = [{"state_id": "excluded", "cache_eligible": False,
                           "cache_ineligible_reason": "no_complete_causal_horizon"}]
    assert assess_hypotheses(manifest, result, tmp_path)["H2"]["status"] == "inconclusive"


def test_cost_projection_requires_multitask_smoke_and_accounts_shared_teacher_once():
    from stmem.inspection.refinement_report import runtime_projection

    planned = {"context_counts": {str(i): 10 for i in range(256)},
               "compact_model_examples_four_students": 4096,
               "teacher_model_examples_shared": 2048}
    rows = [{"state": {"state_id": str(i), "task": str(i % 3), "episode_id": str(i)},
             "identity": {"mode": "smoke", "limited_check": False},
             "timings": {"cache_context_assembly_seconds": 1, "compact_model_examples": 8,
                         "compact_context_seconds": 2, "compact_forward_seconds": 6,
                         "teacher_model_examples": 8, "teacher_context_seconds": 4,
                         "teacher_forward_seconds": 12}} for i in range(8)]
    result = runtime_projection({"kind": "matched_student_refinement_utility"}, rows, planned)
    assert result["total_serial_seconds_min_median_max"] == [1024 + 4096 + 4096] * 3
    assert runtime_projection({"kind": "limited_real_data_diagnostic"}, rows, planned)["status"] == "not_estimated"
    for row in rows:
        row["identity"]["limited_check"] = True
    assert runtime_projection({"kind": "matched_student_refinement_utility"}, rows, planned)["status"] == "not_estimated"
