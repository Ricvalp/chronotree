"""Matched outcomes, incomplete runs and safe portable diagnostic reports."""

from copy import deepcopy
import ast
import json
from pathlib import Path
import re

import pytest

from stmem.inspection.history_use_report import load_report, summarize, write_report
from stmem.inspection import history_use_report


def fixture(tmp_path):
    arms = ["full", "erase_conditioning", "erase_execution", "current_only"]
    planned = [{"task": "SwingXtimes", "episode_idx": index, "arm": arm}
               for index in range(4) for arm in arms]
    (tmp_path / "manifest.json").write_text(json.dumps({
        "planned_cases": planned, "scope": "Synthetic report fixture </script><script>alert(1)</script>",
    }))
    fingerprints = {"reset_sha256": "reset", "goal_sha256": "goal",
                    "current": {"sha256": "current"}, "conditioning": {"sha256": "conditioning", "count": 0}}
    records = {}
    statuses = {"full": ["success", "fail", "success", "success"],
                "erase_conditioning": ["timeout", "success", "error", None],
                "erase_execution": ["success", "fail", "timeout", "timeout"]}
    for arm, values in statuses.items():
        for index, status in enumerate(values):
            if status is None:
                continue
            row = {"task": "SwingXtimes", "episode_idx": index, "arm": arm,
                   "status": status, "noise_seed": index + 100,
                   "task_diagnostics": {"goal": "Swing 3 times </script><script>alert(2)</script>",
                                        "input_fingerprints": deepcopy(fingerprints)}}
            if arm == "erase_execution" and index == 2:
                row["task_diagnostics"]["input_fingerprints"]["current"]["sha256"] = "different_current"
            if arm == "erase_execution" and index == 3:
                row.pop("noise_seed")
            if status == "error":
                row["error"] = "Synthetic error </script><script>alert(3)</script>"
            records[(arm, index)] = row
            path = tmp_path / "results" / arm / "episodes" / f"SwingXtimes_{index:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(row))
    return records


def test_planned_denominators_and_completed_pairs(tmp_path):
    fixture(tmp_path)
    report = load_report(tmp_path)
    s = report["summary"]
    assert s["planned_cases"] == 16
    assert s["recorded_cases"] == 11
    assert s["complete"] is False
    assert s["per_arm"]["full"]["overall"]["success_rate"] == .75
    assert s["per_arm"]["erase_conditioning"]["overall"]["success_rate"] == .25
    assert s["per_arm"]["current_only"]["overall"]["missing"] == 4
    pair = s["paired"]["erase_conditioning"]
    assert pair["planned"] == 4
    assert pair["completed_pairs"] == 2
    assert pair["full_only_success"] == pair["control_only_success"] == 1
    assert pair["error_pairs"] == pair["pending_pairs"] == 1
    assert pair["verified_input_pairs"] == 2
    assert summarize(tmp_path) == s


def test_mismatches_and_unknown_parity_are_not_equivalent(tmp_path):
    fixture(tmp_path)
    pair = summarize(tmp_path)["paired"]["erase_execution"]
    assert pair["input_mismatch_pairs"] == 1
    assert pair["completed_pairs"] == 3
    assert pair["verified_input_pairs"] == 2
    assert pair["unknown_input_pairs"] == 1
    assert pair["full_only_success"] == 1  # Only episode 3, not mismatched episode 2.
    assert pair["cases"][2]["input_parity"]["current"] == "mismatch"
    assert pair["cases"][3]["input_parity"]["noise_seed"] == "unknown"


def test_relative_videos_and_absent_probes(tmp_path):
    fixture(tmp_path)
    video = tmp_path / "videos" / "full" / "SwingXtimes_000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"Synthetic fixture, not a real video")
    scene = load_report(tmp_path)["scenes"][0]
    assert scene["arms"]["full"]["video"] == "videos/full/SwingXtimes_000.mp4"
    assert scene["arms"]["erase_conditioning"]["video"] is None
    assert scene["arms"]["full"]["fixed_current_probes"] == []


@pytest.mark.parametrize("mutation", [
    "duplicate_plan", "unmatched_plan", "unplanned_result", "duplicate_result", "wrong_arm",
    "nonfinite", "outside_video",
])
def test_reject_invalid_or_ambiguous_measurements(tmp_path, mutation):
    rows = fixture(tmp_path)
    if mutation in {"duplicate_plan", "unmatched_plan"}:
        path = tmp_path / "manifest.json"
        manifest = json.loads(path.read_text())
        if mutation == "duplicate_plan":
            manifest["planned_cases"].append(manifest["planned_cases"][0])
        else:
            manifest["planned_cases"].pop()
        path.write_text(json.dumps(manifest))
    else:
        row = rows[("full", 0)]
        path = tmp_path / "results/full/episodes/SwingXtimes_000.json"
        if mutation == "unplanned_result":
            row["episode_idx"] = 90
        elif mutation == "duplicate_result":
            path = path.with_name("duplicate.json")
        elif mutation == "wrong_arm":
            row["arm"] = "erase_conditioning"
        elif mutation == "nonfinite":
            row["fixed_current_probes"] = [{"tcp_change_mean_mm": float("inf")}]
        else:
            row["video_path"] = "../../outside.mp4"
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        load_report(tmp_path)


def test_render_escapes_untrusted_goal_and_keeps_pending_cases(tmp_path):
    fixture(tmp_path)
    page = write_report(tmp_path).read_text()
    assert "<script>alert(" not in page
    match = re.search(r'<script type="application/json" id="payload">(.*?)</script>', page, re.S)
    payload = json.loads(match.group(1))
    assert len(payload["scenes"]) == 4
    assert payload["scenes"][0]["arms"]["current_only"]["pending"] is True
    assert payload["scenes"][0]["arms"]["full"]["goal"].endswith("<script>alert(2)</script>")
    assert "out-of-distribution intervention" in page
    assert "Fixed-current action probes" in page


def test_html_compacts_traces_without_changing_raw_evidence(tmp_path):
    rows = fixture(tmp_path)
    row = rows[("full", 0)]
    diagnostic = row["task_diagnostics"]
    diagnostic["timeline"] = [{"command_index": 2, "large_trace_marker": "raw_trace_only"}]
    diagnostic["input_fingerprints"]["conditioning"]["frames"] = [{"sha256": "raw_frame_only"}]
    diagnostic["summary"] = {"first_grasp_correct": True}
    row["fixed_current_probes"] = [{"status": "passed", "arms": {
        "FULL": {"tcp_change_mean_mm": 0.0, "raw_actions": [[0.0] * 8]},
    }}]
    path = tmp_path / "results/full/episodes/SwingXtimes_000.json"
    raw_bytes = json.dumps(row).encode()
    path.write_bytes(raw_bytes)
    full = load_report(tmp_path)
    assert full["scenes"][0]["arms"]["full"]["task_diagnostics"]["timeline"]
    page = write_report(tmp_path).read_text()
    payload = json.loads(re.search(r'<script type="application/json" id="payload">(.*?)</script>', page, re.S).group(1))
    compact = payload["scenes"][0]["arms"]["full"]
    assert "timeline" not in compact["task_diagnostics"]
    assert compact["task_diagnostics"]["summary"] == {"first_grasp_correct": True}
    assert compact["task_diagnostics"]["input_fingerprints"]["conditioning"] == {"sha256": "conditioning", "count": 0}
    assert compact["fixed_current_probes"] == row["fixed_current_probes"]
    assert compact["raw_result"] == "results/full/episodes/SwingXtimes_000.json"
    assert payload["summary"] == full["summary"]
    assert path.read_bytes() == raw_bytes
    assert "raw_trace_only" not in page and "raw_frame_only" not in page


def test_policy_start_uses_recorded_frames_and_actual_writer_convention(tmp_path):
    # Verify the cross-module video convention rather than assuming the model's
    # 10 Hz observation cadence or the environment control rate is the FPS.
    runner = Path(history_use_report.__file__).parents[1] / "evaluation" / "runner.py"
    tree = ast.parse(runner.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "get_writer"]
    writer_fps = [ast.literal_eval(keyword.value) for call in calls
                  for keyword in call.keywords if keyword.arg == "fps"]
    assert writer_fps == [history_use_report._RECORDED_VIDEO_FPS]
    rows = fixture(tmp_path)
    row = rows[("full", 0)]
    row["environment"] = {"conditioning_frames_recorded": 265, "conditioning_frames": 133,
                          "observation_hz": 10, "control_hz": 20}
    path = tmp_path / "results/full/episodes/SwingXtimes_000.json"
    path.write_text(json.dumps(row))
    full = load_report(tmp_path)["scenes"][0]["arms"]["full"]
    assert full["policy_video_start_seconds"] == 265 / writer_fps[0]
    assert full["policy_video_start_seconds"] != 133 / writer_fps[0]
    assert load_report(tmp_path)["scenes"][0]["arms"]["erase_execution"]["policy_video_start_seconds"] is None
    assert 'id="playPolicy"' in write_report(tmp_path).read_text()
