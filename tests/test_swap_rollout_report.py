"""Planned-denominator and presentation checks for ordinary swap rollouts."""

from copy import deepcopy
import json
import re

import pytest

from stmem.inspection.swap_rollout_report import load_report, write_report


def fixture(tmp_path):
    (tmp_path / "episodes").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "planned_episodes": list(range(10)), "task": "VideoUnmaskSwap",
        "scope": "Synthetic rendering fixture, not a measured success result.",
    }))
    reset = {"tcp_xyz_m": [0, 0, .4], "bins": [
        {"index": 0, "xyz_m": [.2, -.1, .04]}, {"index": 1, "xyz_m": [-.1, .2, .04]},
    ]}
    diagnostic = {"status": "complete", "target": {"first_bin_index": 0, "color": "blue"},
                  "reset": reset, "timeline": [{"command_index": 1, "tcp_xyz_m": [0, 0, .35]}],
                  "summary": {"first_approach_bin_index": None, "first_grasp_bin_indices": None,
                              "first_lift_bin_indices": None, "first_lift_correct": None}}
    records = [
        {"task": "VideoUnmaskSwap", "episode_idx": 0, "status": "timeout",
         "choice_diagnostics": diagnostic, "video_path": "videos/VideoUnmaskSwap_000.mp4"},
        {"task": "VideoUnmaskSwap", "episode_idx": 1, "status": "success",
         "choice_diagnostics": deepcopy(diagnostic)},
        {"task": "VideoUnmaskSwap", "episode_idx": 2, "status": "error",
         "error": "Failure </script><script>alert(1)</script>",
         "choice_diagnostics": {"status": "not_observed", "reset": None, "target": None,
                                "summary": {"first_approach_bin_index": None}}},
    ]
    records[1]["choice_diagnostics"]["summary"].update(
        first_approach_bin_index=0, first_grasp_bin_indices=[0], first_lift_bin_indices=[0],
        first_lift_correct=True, first_grasp_correct=True,
    )
    for record in records:
        (tmp_path / "episodes" / f"VideoUnmaskSwap_{record['episode_idx']:03d}.json").write_text(json.dumps(record))
    return records


def test_preserves_planned_denominator_and_distinguishes_missing_diagnostics(tmp_path):
    fixture(tmp_path)
    result = load_report(tmp_path)
    assert len(result["episodes"]) == 10
    assert result["summary"]["overall"]["success_rate"] == .1
    assert result["summary"]["overall"]["missing"] == 7
    assert result["summary"]["overall"]["errors"] == 1
    assert result["summary"]["overall"]["timeouts"] == 1
    assert result["summary"]["rates_are_lower_bounds"] is True
    assert result["choice_summary"]["first_lift_observed"] == 1
    assert result["choice_summary"]["correct_first_lift"] == 1
    assert result["choice_summary"]["no_approach_observed"] == 1  # Error/pending are not no-approach observations.
    assert result["episodes"][2]["observer_available"] is False
    assert result["episodes"][0]["layout_ranks"]["0"] == {
        "x_rank": 2, "y_rank": 1, "count": 2, "xyz_m": [.2, -.1, .04],
    }
    assert result["episodes"][0]["video"] is None


def test_videos_are_relative_and_optional(tmp_path):
    fixture(tmp_path)
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos/VideoUnmaskSwap_000.mp4").write_bytes(b"Explicit test fixture, not a real video")
    result = load_report(tmp_path)
    assert result["episodes"][0]["video"] == "videos/VideoUnmaskSwap_000.mp4"
    assert result["episodes"][1]["video"] is None


@pytest.mark.parametrize("mutation", [
    "duplicate", "unplanned", "duplicate_plan", "nonfinite", "outside_video",
])
def test_reject_ambiguous_or_invalid_results(tmp_path, mutation):
    records = fixture(tmp_path)
    if mutation == "duplicate":
        (tmp_path / "episodes/duplicate.json").write_text(json.dumps(records[0]))
    elif mutation == "unplanned":
        (tmp_path / "episodes/extra.json").write_text(json.dumps({**records[0], "episode_idx": 10}))
    elif mutation == "duplicate_plan":
        (tmp_path / "manifest.json").write_text(json.dumps({"planned_episodes": [0, 0]}))
    else:
        if mutation == "nonfinite":
            records[0]["choice_diagnostics"]["reset"]["bins"][0]["xyz_m"][0] = float("nan")
        else:
            records[0]["video_path"] = "../../elsewhere.mp4"
        (tmp_path / "episodes/VideoUnmaskSwap_000.json").write_text(json.dumps(records[0]))
    with pytest.raises(ValueError):
        load_report(tmp_path)


def test_html_retains_all_episodes_and_escapes_error_text(tmp_path):
    fixture(tmp_path)
    page = write_report(tmp_path).read_text()
    assert "<script>alert(1)</script>" not in page
    payload = re.search(r'<script type="application/json" id="payload">(.*?)</script>', page, re.S).group(1)
    loaded = json.loads(payload)
    assert len(loaded["episodes"]) == 10
    assert loaded["episodes"][2]["error"] == "Failure </script><script>alert(1)</script>"
    assert "World X/Y ranks" in page
    assert "not policy inputs" in page
