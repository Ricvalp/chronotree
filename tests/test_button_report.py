"""CPU-only report checks with explicitly synthetic trajectory measurements."""

from copy import deepcopy
import json
import re

import pytest

from stmem.inspection.button_report import validate_result, write_report


def sample_result():
    return {
        "protocol": {"scope": "Synthetic viewer fixture, not measured results", "execute_actions": 8,
                     "action_stride": 2, "native_control_hz": 20},
        "trials": [{
            "label": "student", "task": "VideoRepick", "episode_idx": 0,
            "method": "UNIFORM", "checkpoint_step": 80000, "status": "timeout",
            "decisions": [{
                "native_step": 32,
                "diagnostics": {"tcp_xyz": [0, 0, .15], "button_xyz": [0, 0, .02],
                                "button_depth_m": 0, "task_index": 4, "task_name": "press button"},
                "points": [[0, 0, 0, .5, .5, .5]],
                "target_tcp_xyz": [[0, 0, .15 - i * .005] for i in range(16)],
                "physical_actions": [[0.] * 8 for _ in range(16)],
                "metrics": {"tail_descent_mm": 40.},
            }],
            "timeline": [{"native_step": i, "tcp_xyz": [0, 0, .15], "button_xyz": [0, 0, .02],
                          "button_depth_m": 0, "task_index": 4, "task_name": "press button"}
                         for i in range(32, 49)],
        }],
    }


def test_report_preserves_exact_recorded_paths_and_distinguishes_actual_motion(tmp_path):
    result = sample_result()
    result["trials"][0]["decisions"][0]["diagnostics"]["task_name"] = '</script><img src=x>'
    before = deepcopy(result)
    page = write_report(result, tmp_path).read_text()
    assert before == result
    assert "not a benchmark success rate" in page
    assert "not executed robot motion" in page
    assert "Actual ensuing TCP motion" in page
    assert "discarded tail" in page
    assert '<script src="http' not in page
    payload = re.search(r'<script type="application/json" id="payload">(.*?)</script>', page, re.S)
    assert payload and json.loads(payload.group(1)) == result
    assert '</script><img' not in payload.group(1)


@pytest.mark.parametrize("change, message", [
    (lambda r: r["trials"][0]["decisions"][0]["target_tcp_xyz"].pop(), "shape"),
    (lambda r: r["trials"][0].update(execute_actions=17), "Executed prefix"),
    (lambda r: r["trials"][0]["timeline"][1].update(native_step=32), "strictly increase"),
    (lambda r: r["trials"][0]["decisions"][0].update(points=[[float("nan")] * 6]), "finite"),
])
def test_report_rejects_misaligned_or_nonfinite_measurements(change, message):
    result = sample_result()
    change(result)
    with pytest.raises(ValueError, match=message):
        validate_result(result)


def test_failed_trial_is_visible_without_fabricating_measurements(tmp_path):
    result = sample_result()
    result["trials"].append({"label": "teacher", "task": "VideoRepick", "episode_idx": 0,
                             "method": "FULL", "checkpoint_step": 100000,
                             "status": "infrastructure_error", "decisions": [], "timeline": []})
    assert "infrastructure_error" in write_report(result, tmp_path).read_text()


def test_branch_videos_are_relative_and_preserve_measured_outcomes(tmp_path):
    result = sample_result()
    video = tmp_path / "branches" / "step0032_execute16.mp4"
    result["trials"][0]["branches"] = [{"native_step": 32, "mode": "execute16", "status": "success",
                                        "actual_descent_mm": 80., "actual_button_progress_mm": 78.,
                                        "maximum_button_depth_mm": 6., "replay_simulator_max_error": 0.,
                                        "video": str(video), "prefix_video_frames": 80}]
    before = deepcopy(result)
    page = write_report(result, tmp_path).read_text()
    assert before == result
    payload = re.search(r'<script type="application/json" id="payload">(.*?)</script>', page, re.S)
    row = json.loads(payload.group(1))["trials"][0]["branches"][0]
    assert row["video"] == "branches/step0032_execute16.mp4"
    assert row["status"] == "success" and row["replay_simulator_max_error"] == 0.
    assert "Jump to branch" in page and "!r.terminal" in page


def test_display_compaction_does_not_change_source_or_target_measurements(tmp_path):
    result = sample_result()
    row = result["trials"][0]["decisions"][0]
    row["points"][0][0] = .123456789
    row["target_tcp_xyz"][0][0] = .123456789
    row["timing"] = {"represented_leaf_counts": [1] * 1000}
    row["raw_actions"] = [[.123456789] * 8 for _ in range(16)]
    before = deepcopy(result)
    page = write_report(result, tmp_path).read_text()
    assert result == before
    payload = re.search(r'<script type="application/json" id="payload">(.*?)</script>', page, re.S)
    rendered = json.loads(payload.group(1))["trials"][0]["decisions"][0]
    assert rendered["points"][0][0] == .12346
    assert rendered["target_tcp_xyz"][0][0] == .123456789
    assert "timing" not in rendered and "raw_actions" not in rendered
