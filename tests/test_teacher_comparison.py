"""Synthetic CPU fixtures for comparison integrity, never policy-performance tests."""
from copy import deepcopy
import hashlib
import json

import pytest

from stmem.inspection.teacher_comparison import METRICS, compare_behavior, compare_usual, summarize_behavior


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def usual(root, *, step, error=2.0, source="same scientific implementation"):
    """Minimal saved reports with toy serialized bytes; no checkpoint is executed."""
    root.mkdir()
    checkpoint, asset = root / "fixture.pt", root / "observations.bin"
    checkpoint.write_bytes(f"synthetic checkpoint {step}".encode())
    asset.write_bytes(b"identical observations")
    identity = {"path": str(checkpoint), "step": step, "sha256": sha(checkpoint)}
    config = {name: {"fixture": True} for name in ("model", "data", "geometry", "evaluation")}
    manifest = {"checkpoints": {"fixture_teacher": identity}, "protocol": {"executed_targets": 2},
                "assets": {"prepared": str(asset)}, "asset_hashes": {"prepared": sha(asset)},
                "policy_source_sha256": source}
    write(root / "inspection_manifest.json", manifest)
    manifest_sha = sha(root / "inspection_manifest.json")
    write(root / "checkpoint_audit.json", {"inspection_manifest_sha256": manifest_sha,
          "checkpoints": {"fixture_teacher": {**identity, "status": "passed", "config": config}}})
    write(root / "pipeline_status.json", {"status": "complete"})
    write(root / "runtime_identity.json", {"identity": {"python": "fixture"}, "scope": "Synthetic fixture"})
    for name in ("offline/validation.json", "grounding/validation.json", "handoff/validation.json",
                 "handoff/media_validation.json", "report_validation.json"):
        write(root / name, {"status": "passed", "inspection_manifest_sha256": manifest_sha})
    rows = []
    for index in (0, 32):
        row = {"episode": "episode_9", "index": index, "noise_seed": 17,
               "kind": "visual", "variant": "baseline", "target_actions": [[1, 2]],
               "target_valid": [True], "native_target_actions": [[1, 2]], "native_target_valid": [True],
               **{metric: error + index / 32 for metric in METRICS}}
        rows.append(row)
    write(root / "offline/fixture_teacher/sensitivity.json", {"rows": rows})
    write(root / "offline/protocol.json", {"visual_probe_precision": "bf16", "native_action_metrics": "20Hz"})
    write(root / "grounding/validation.json", {"status": "passed", "inspection_manifest_sha256": manifest_sha,
          "models": {"fixture_teacher": {"grasp_flag_commands": [2, 4], "grasped": True,
                     "state80_response_mm": {}, "state80_first0p8s_response_mm": {},
                     "state80_full_horizon_response_mm": {}, "state80_endpoint_response_mm": {}}}})
    write(root / "reset_cube_chunks/summary.json", {"models": {"fixture_teacher": {"scenes": []}}})
    run = {"task": "VideoUnmask", "episode": 0, "prefix": 4, "variant": "fixture_teacher",
           "expert_control": False, "checkpoint_step": step, "status": "success"}
    write(root / "handoff/summary.json", {"runs": [run], "by_model_scene": []})
    return root


def test_usual_dynamic_steps_recomputed_metrics_and_honest_source_review(tmp_path):
    old = usual(tmp_path / "reference", step=17, error=2)
    new = usual(tmp_path / "current", step=39, error=1, source="changed source")
    output = tmp_path / "comparison.json"
    report = compare_usual(old, new, model="fixture_teacher", output=output)
    assert report["checkpoints"]["reference"]["step"] == 17
    assert report["checkpoints"]["current"]["step"] == 39
    baseline = next(row for row in report["offline_summary_pairs"] if row["state_subset"] == "all_states")
    assert baseline["metrics"]["native16_tcp_error_mm"] == {
        "reference": 2.5, "current": 1.5, "delta_current_minus_reference": -1.0}
    assert report["compatibility"]["source_review_required"]
    assert not report["compatibility"]["runtime_review_required"]
    assert report["compatibility"]["checkpoint_hashes_verified_now"] == {"reference": True, "current": True}
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("mutation", ["expert_targets", "missing_row", "duplicate", "precision"])
def test_usual_rejects_unmatched_records_or_precision(tmp_path, mutation):
    old = usual(tmp_path / "reference", step=17)
    new = usual(tmp_path / "current", step=39)
    path = new / "offline/fixture_teacher/sensitivity.json"
    record = json.loads(path.read_text())
    if mutation == "expert_targets":
        record["rows"][0]["native_target_actions"] = [[99, 99]]
    elif mutation == "missing_row":
        record["rows"].pop()
    elif mutation == "duplicate":
        record["rows"].append(record["rows"][0])
    else:
        write(new / "offline/protocol.json", {"visual_probe_precision": "fp32", "native_action_metrics": "20Hz"})
    write(path, record)
    with pytest.raises(ValueError, match="Mismatched|Duplicate"):
        compare_usual(old, new, model="fixture_teacher", verify_files=False)


@pytest.mark.parametrize("mutation", ["checkpoint", "asset", "stale_validation"])
def test_usual_rechecks_bytes_and_validation_identity(tmp_path, mutation):
    old = usual(tmp_path / "reference", step=17)
    new = usual(tmp_path / "current", step=39)
    if mutation == "checkpoint":
        (new / "fixture.pt").write_bytes(b"incomplete rsync")
    elif mutation == "asset":
        (new / "observations.bin").write_bytes(b"different observations")
    else:
        write(new / "handoff/validation.json", {"status": "passed", "inspection_manifest_sha256": "stale"})
    with pytest.raises(ValueError, match="changed|Mismatched"):
        compare_usual(old, new, model="fixture_teacher")


def placement(*, lifted=True):
    return {"task": "VideoPlaceOrder", "episode_idx": 7, "arm": "full", "status": "fail",
            "primitive_commands": 20, "noise_seed": 73, "checkpoint_sha256": "checkpoint",
            "task_diagnostics": {
                "status": "complete", "errors": [], "goal": "fixture goal",
                "target": {"kind": "target_placement", "first_index": 1, "candidate_count": 2},
                "input_fingerprints": {"reset_sha256": "same reset"},
                "summary": {"first_grasp_command": None, "completed_subgoal_index_increase": 0},
                "reset": {"current_task_index": 0},
                "timeline": [{"command_index": 2, "tcp_xyz_m": [0, 0, 0.5], "current_task_index": 0,
                              "controlled_object": {"xyz_m": [.01, 0, .1], "grasped": lifted, "lifted": lifted},
                              "candidates": [{"index": 0, "xyz_m": [0, 0, 0]},
                                             {"index": 1, "xyz_m": [.3, 0, 0]}]},
                             {"command_index": 20, "excluded_reason": "terminal_auto_step"}]}}


def test_placement_uses_controlled_cube_and_active_samples_not_pad_contact():
    summary = summarize_behavior(placement())
    assert summary["controlled_cube_first_grasp_command"] == 2
    assert summary["controlled_cube_first_lift_command"] == 2
    assert summary["last_observed"]["cube_pad_xy_distances_mm"] == pytest.approx([10, 290])
    assert summary["last_observed"]["nearest_pad"] == 0
    assert summary["last_observed"]["nearest_is_target"] is False
    assert summary["status"] == "fail"
    no_lift = summarize_behavior(placement(lifted=False))
    assert no_lift["controlled_cube_first_lift_command"] is None
    assert "not an executed placement" in no_lift["destination_observation"]


def test_repick_cycles_are_observed_lower_bound_not_invented_success():
    record = placement()
    record["task"] = "VideoRepick"
    diagnostic = record["task_diagnostics"]
    diagnostic["target"] = {"kind": "cube_pickup", "first_index": 1, "candidate_count": 2, "num_repeats": 3}
    diagnostic["summary"]["completed_subgoal_index_increase"] = 4
    for index, row in enumerate(diagnostic["timeline"][:1]):
        row["current_task_index"] = 4
        for candidate in row["candidates"]:
            candidate.update(grasped=candidate["index"] == 0, lifted=candidate["index"] == 0)
    summary = summarize_behavior(record)
    assert summary["observed_completed_pick_drop_cycles"] == 2
    assert summary["observed_reached_button_stage"] is False
    assert summary["first_observed_button_stage_command"] is None
    assert summary["other_indices_ever_grasped"] == [0]
    assert not summary["target_ever_grasped"]
    assert summary["status"] == "fail"


def history(root, *, step=17, row=None):
    row = placement() if row is None else row
    manifest = {"checkpoint": {"step": step, "sha256": "checkpoint", "path": "fixture.pt"},
                "arms": {"full": {"method": "FULL", "intervention": "none"}},
                "execution_override": {"execute_actions": 10},
                "planned_cases": [{"task": "VideoPlaceOrder", "episode_idx": 7, "arm": "full"}]}
    write(root / "manifest.json", manifest)
    write(root / "results/full/episodes/VideoPlaceOrder_007.json", row)
    return root


def test_behavior_dynamic_plan_and_missing_cases_remain_pending(tmp_path):
    old = history(tmp_path / "reference", step=17)
    new = history(tmp_path / "current", step=39)
    report = compare_behavior(old, new)
    assert report["completed_pairs"] == 1
    assert report["checkpoints"]["current"]["step"] == 39
    assert report["cases"][0]["input_parity"]["reset_fingerprint"] == "matched"
    assert report["outcomes_by_task_arm"][0]["planned"] == 1
    assert report["suggested_video_review"] == []
    (new / "results/full/episodes/VideoPlaceOrder_007.json").unlink()
    report = compare_behavior(old, new, require_complete=False)
    assert report["completed_pairs"] == 0
    assert report["cases"][0]["current"] is None
    with pytest.raises(ValueError, match="missing planned"):
        compare_behavior(old, new)


@pytest.mark.parametrize("mutation", ["reset", "noise", "checkpoint", "target"])
def test_behavior_rejects_nonmatching_scene_inputs_and_checkpoint(tmp_path, mutation):
    old = history(tmp_path / "reference")
    row = placement()
    if mutation == "reset":
        row["task_diagnostics"]["input_fingerprints"]["reset_sha256"] = "different"
    elif mutation == "noise":
        row["noise_seed"] += 1
    elif mutation == "checkpoint":
        row["checkpoint_sha256"] = "stale checkpoint"
    else:
        row["task_diagnostics"]["target"]["first_index"] = 0
    new = history(tmp_path / "current", row=row)
    with pytest.raises(ValueError, match="Mismatched|checkpoint mismatch"):
        compare_behavior(old, new)


def test_missing_observer_does_not_invent_zero_success_or_contact(tmp_path):
    row = placement()
    row["task_diagnostics"] = {"status": "diagnostic_error", "input_fingerprints": None}
    old = history(tmp_path / "reference", row=deepcopy(row))
    new = history(tmp_path / "current", row=row)
    report = compare_behavior(old, new)
    result = report["cases"][0]
    assert result["input_parity"]["reset_fingerprint"] == "unknown"
    assert result["current"]["available"] is False
    assert "controlled_cube_first_lift_command" not in result["current"]


def test_video_review_flags_changed_outcomes_but_not_error_as_regression(tmp_path):
    row = placement()
    row["status"] = "success"
    old = history(tmp_path / "reference", row=row)
    new = history(tmp_path / "current")
    report = compare_behavior(old, new)
    assert report["outcomes_by_task_arm"][0]["lost_success_episodes"] == [7]
    assert report["suggested_video_review"][0]["reasons"] == ["lost official success"]
    row["status"] = "infrastructure_error"
    write(new / "results/full/episodes/VideoPlaceOrder_007.json", row)
    report = compare_behavior(old, new)
    assert report["outcomes_by_task_arm"][0]["error_pairs"] == 1
    assert report["outcomes_by_task_arm"][0]["lost_success_episodes"] == []
    assert report["suggested_video_review"] == []
