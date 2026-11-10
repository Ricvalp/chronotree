"""CPU-only synthetic fixtures for saved-result audit integrity and paired effects."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stmem.inspection.history_comparison import compare_history_runs


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_run(root, step, *, missing=False, source="same"):
    root.mkdir()
    clip = root.parent / "clip.pt"
    if not clip.exists():
        clip.write_bytes(b"Synthetic frozen-feature identity fixture")
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(f"Synthetic checksum fixture {step}".encode())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    arms = {
        "full": {"method": "FULL", "intervention": "none"},
        "current_only": {"method": "CURRENT", "intervention": "none"},
    }
    override = {"execute_actions": 2, "predicted_actions_per_chunk": 3, "action_stride": 2}
    manifest = {
        "checkpoint": {"path": str(checkpoint), "sha256": digest, "step": step, "stage": "teacher"},
        "source": {"source_sha256": source},
        "split": "val",
        "arms": arms,
        "probe_decisions": [0],
        "planned_cases": [{"task": "VideoRepick", "episode_idx": 0, "arm": arm} for arm in arms],
        "execution_override": override,
    }
    cfg = {
        "data": {"tasks": ["VideoRepick"], "nominal_control_hz": 20, "observation_stride": 2},
        "model": {
            "horizon": 3,
            "action_stride": 2,
            "flow_steps": 5,
            "visual_features": {"checkpoint": str(clip)},
        },
        "geometry": {},
        "training": {},
        "seed": 0,
        "evaluation": {"seed": 123, "max_steps": 3, "timeout_seconds": 30, "episode_count": 1},
    }
    dump(root / "manifest.json", manifest)
    dump(root / "config.json", cfg)
    physical = [[i / 10] * 8 for i in range(3)]
    payloads = {}
    for arm in ("FULL", "erase_conditioning", "erase_execution", "CURRENT"):
        delta = 0.01 if arm in ("CURRENT", "erase_conditioning") else 0.0
        counts = {} if delta else {"conditioning": 2}
        payloads[arm] = {
            "physical_actions": physical,
            "tcp_targets_m": [[delta, 0, 0]] * 3,
            "tcp_change_executed_targets_mean_mm": delta * 1000,
            "history_leaf_count": sum(counts.values()),
            "history_source_counts": counts,
        }
    probe = {
        "status": "passed",
        "full_replay_passed": True,
        "full_replay_max_abs_action_error": 0.0,
        "decision_index": 0,
        "executed_targets_per_chunk": 2,
        "execution_source_index": 0,
        "noise_sha256": "shared_noise",
        "arms": payloads,
    }
    for arm in arms:
        if missing and arm == "current_only":
            continue
        video = root / "videos" / arm / "VideoRepick_000.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"Synthetic video fixture; media decoding tested separately")
        row = {
            "task": "VideoRepick",
            "episode_idx": 0,
            "arm": arm,
            "split": "val",
            "method": arms[arm]["method"],
            "checkpoint_sha256": digest,
            "noise_seed": 123,
            "execution_override": override,
            "status": "success" if arm == "full" else "fail",
            "error": None,
            "runtime_seconds": 1.0,
            "primitive_commands": 4,
            "actions": [a for a in physical[:2] for _ in range(2)],
            "decisions": [{"decision_index": 0}],
            "video_path": str(video),
            "environment": {
                "requested_max_steps": 3,
                "effective_max_steps": 5,
                "conditioning_frames_recorded": 3,
            },
            "task_diagnostics": {
                "status": "complete",
                "errors": [],
                "input_fingerprints": {
                    "reset_sha256": "reset",
                    "goal_sha256": "goal",
                    "current": {"sha256": "current"},
                    "conditioning": {"sha256": "demo"},
                },
                "timeline": [],
                "summary": {"completed_subgoal_index_increase": None},
                "goal": "<script>unsafe label</script>",
            },
        }
        if arm == "full":
            row["fixed_current_probes"] = [deepcopy(probe)]
        dump(root / "results" / arm / "episodes" / "VideoRepick_000.json", row)
    return root


def result_path(run, arm):
    return run / "results" / arm / "episodes" / "VideoRepick_000.json"


def edit_row(run, arm, edit):
    path = result_path(run, arm)
    row = json.loads(path.read_text())
    edit(row)
    dump(path, row)


def test_generic_comparison_and_independent_calls(tmp_path):
    a, b = make_run(tmp_path / "first", 33), make_run(tmp_path / "second", 71, source="changed")
    before = {p: p.read_bytes() for root in (a, b) for p in root.rglob("*") if p.is_file()}
    first = compare_history_runs(a, b, output=tmp_path / "report", require_complete=True)
    second = compare_history_runs(a, b, output=tmp_path / "report2", require_complete=True)
    assert first["valid"] and first["complete"] and first["requires_source_review"]
    assert first["total_checks"] == second["total_checks"]  # No global accumulator leakage.
    assert first["protocol"]["predicted_targets"] == 3 and first["protocol"]["flow_steps"] == 5
    benefit = first["comparison"]["history_benefit"][0]
    assert benefit["old"]["success_benefit_percentage_points"] == 100
    assert benefit["new"]["success_benefit_percentage_points"] == 100
    html = Path(first["artifacts"]["html"]).read_text()
    assert "33 → 71" in html and "120k" not in html and "104k" not in html
    assert "<script>unsafe label</script>" not in html
    assert first["html_links"]["missing"] == []
    assert before == {p: p.read_bytes() for p in before}


def test_incomplete_is_pending_not_a_lost_success(tmp_path):
    a, b = make_run(tmp_path / "first", 33), make_run(tmp_path / "second", 71, missing=True)
    result = compare_history_runs(a, b, output=tmp_path / "report", require_complete=True)
    assert not result["complete"] and not result["valid"] and result["audit_passed"]
    assert result["comparison"]["history_benefit"][0]["new"]["success_benefit_percentage_points"] is None
    assert "pending pair" in Path(result["artifacts"]["html"]).read_text()
    with pytest.raises(ValueError, match="both"):
        compare_history_runs(a, b, output=tmp_path / "other", recent=b)


@pytest.mark.parametrize(
    "tamper,expected",
    [
        (lambda r: r["actions"][0].__setitem__(0, 0.5), "executed_targets_match"),
        (lambda r: r.__setitem__("fixed_current_probes", []), "probe_coverage"),
        (
            lambda r: r["task_diagnostics"]["input_fingerprints"].__setitem__("reset_sha256", "wrong"),
            "cross_checkpoint_reset",
        ),
        (
            lambda r: r["fixed_current_probes"][0]["arms"]["CURRENT"].__setitem__(
                "tcp_change_executed_targets_mean_mm", 999
            ),
            "tcp_recalculation",
        ),
    ],
)
def test_corrupted_issued_actions_probes_or_reset_invalidate(tmp_path, tamper, expected):
    a, b = make_run(tmp_path / "first", 33), make_run(tmp_path / "second", 71)
    edit_row(b, "full", tamper)
    result = compare_history_runs(a, b, output=tmp_path / "report", require_complete=True)
    assert not result["valid"]
    assert any(expected in row["check"] for row in result["failed_checks"])


def test_errors_and_input_mismatches_cannot_create_history_benefit(tmp_path):
    a, b = make_run(tmp_path / "first", 33), make_run(tmp_path / "second", 71)
    edit_row(b, "current_only", lambda r: r.update(status="infrastructure_error", error="synthetic"))
    result = compare_history_runs(a, b, output=tmp_path / "report", require_complete=True)
    assert not result["valid"]
    assert result["comparison"]["history_benefit"][0]["new"]["valid_pairs"] == 0
    assert result["comparison"]["history_benefit"][0]["new"]["success_benefit_percentage_points"] is None


def test_timeout_uses_declared_limits_not_fixed_1301(tmp_path):
    a, b = make_run(tmp_path / "first", 33), make_run(tmp_path / "second", 71)
    for root in (a, b):
        edit_row(root, "current_only", lambda r: r.update(status="timeout", terminal_status="timeout"))
    result = compare_history_runs(a, b, output=tmp_path / "report", require_complete=True)
    assert result["valid"]
    edit_row(b, "current_only", lambda r: r.update(runtime_seconds=31.0))
    invalid = compare_history_runs(a, b, output=tmp_path / "bad", require_complete=True)
    assert any("within_wall_clock_guard" in row["check"] for row in invalid["failed_checks"])


def test_different_recent_window_or_duplicate_plan_is_rejected(tmp_path):
    a, b = make_run(tmp_path / "first", 33), make_run(tmp_path / "second", 71)
    path = b / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["arms"]["current_only"]["completed_blocks"] = 3
    dump(path, manifest)
    result = compare_history_runs(a, b, output=tmp_path / "report")
    assert not result["valid"]
    assert any(row["check"] == "matched_manifest:arms" for row in result["failed_checks"])
    manifest["planned_cases"].append(deepcopy(manifest["planned_cases"][0]))
    dump(path, manifest)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_history_runs(a, b, output=tmp_path / "duplicate")


def test_media_decode_failure_is_an_audit_failure(tmp_path, monkeypatch):
    from stmem.inspection import history_comparison

    a, b = make_run(tmp_path / "first", 33), make_run(tmp_path / "second", 71)
    monkeypatch.setattr(history_comparison.shutil, "which", lambda name: "synthetic_ffmpeg")
    monkeypatch.setattr(
        history_comparison.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="frame=0\nout_time_us=0\nprogress=end\n", stderr="synthetic invalid video"
        ),
    )
    result = compare_history_runs(a, b, output=tmp_path / "report", validate_media=True)
    assert not result["valid"]
    assert any(row["check"].startswith("video_decode:") for row in result["failed_checks"])
    assert all(not row["full_decode_passed"] for row in result["media"]["videos"])


def test_recent_followup_cannot_belong_to_another_checkpoint(tmp_path):
    a, b = make_run(tmp_path / "first", 33), make_run(tmp_path / "second", 71)
    ra, rb = make_run(tmp_path / "recent_first", 34), make_run(tmp_path / "recent_second", 72)
    result = compare_history_runs(a, b, output=tmp_path / "report", reference_recent=ra, recent=rb)
    assert not result["valid"]
    assert any(
        row["check"].startswith("recent_matches_primary_checkpoint:") for row in result["failed_checks"]
    )


def seal_run(root):
    from stmem.inspection.history_resume import seal_plan, seal_result

    for path in root.glob("results/*/episodes/*.json"):
        dump(path, seal_result(json.loads(path.read_text())))
    seal_plan(
        root, json.loads((root / "manifest.json").read_text()), json.loads((root / "config.json").read_text())
    )


def make_recent(root, primary):
    manifest = json.loads((primary / "manifest.json").read_text())
    make_run(root, manifest["checkpoint"]["step"])
    value = json.loads((root / "manifest.json").read_text())
    value["arms"] = {
        "full": value["arms"]["full"],
        "recent_execution": {"method": "FULL", "intervention": "none", "completed_blocks": 2},
    }
    for case in value["planned_cases"]:
        if case["arm"] == "current_only":
            case["arm"] = "recent_execution"
    original = result_path(primary, "full")
    full = json.loads(original.read_text())
    original_video = full["video_path"]
    copied_video = json.loads(result_path(root, "full").read_text())["video_path"]
    Path(copied_video).write_bytes(Path(original_video).read_bytes())
    full.update(video_path=copied_video, reused_reference_result=str(original))
    dump(result_path(root, "full"), full)
    row = json.loads(result_path(root, "current_only").read_text())
    row.update(arm="recent_execution", method="FULL")
    dump(result_path(root, "recent_execution"), row)
    result_path(root, "current_only").unlink()
    value["reference_results"] = {str(original): hashlib.sha256(original.read_bytes()).hexdigest()}
    value["reference_videos"] = {
        original_video: hashlib.sha256(Path(original_video).read_bytes()).hexdigest()
    }
    dump(root / "manifest.json", value)
    seal_run(root)
    return root


def test_sealed_references_compare_after_copy_reseals_paths(tmp_path):
    a, b = make_run(tmp_path / "first", 33), make_run(tmp_path / "second", 71)
    seal_run(a)
    seal_run(b)
    ra, rb = make_recent(tmp_path / "recent_first", a), make_recent(tmp_path / "recent_second", b)
    result = compare_history_runs(
        a, b, output=tmp_path / "report", reference_recent=ra, recent=rb, require_complete=True
    )
    assert result["valid"], result["failed_checks"]
    assert result["input_integrity"] == {"reference": "sealed", "candidate": "sealed"}
    old_seal = json.loads(result_path(b, "full").read_text())["resume_integrity"]
    copied_seal = json.loads(result_path(rb, "full").read_text())["resume_integrity"]
    assert old_seal["payload_sha256"] != copied_seal["payload_sha256"]
    assert old_seal["video_sha256"] == copied_seal["video_sha256"]


@pytest.mark.parametrize("change", ["record", "video", "config", "missing_identity"])
def test_seals_detect_tampering_even_in_report_only_mode(tmp_path, change):
    a, b = make_run(tmp_path / "first", 33), make_run(tmp_path / "second", 71)
    seal_run(a)
    seal_run(b)
    if change == "record":
        edit_row(b, "full", lambda row: row.update(runtime_seconds=2.0))
    elif change == "video":
        row = json.loads(result_path(b, "full").read_text())
        Path(row["video_path"]).write_bytes(b"Different synthetic bytes")
    elif change == "config":
        path = b / "config.json"
        path.write_text(path.read_text() + "\n")
    else:
        (b / "identity.json").unlink()
    result = compare_history_runs(a, b, output=tmp_path / "report", require_complete=True)
    assert not result["valid"]
    assert any(
        "integrity" in row["check"] or "identity" in row["check"] or "sealed_plan" in row["check"]
        for row in result["failed_checks"]
    )
