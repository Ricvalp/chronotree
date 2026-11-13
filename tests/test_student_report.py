"""CPU tests for descriptive, episode-aware local diagnostic reporting."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from stmem.inspection.student_report import summarize_local_measurements, write_diagnostic_figures


def record(episode, state, gains, loss, *, tied=False, reversal=False):
    return {"label": "s_std", "state": {"task": "VideoUnmask", "episode_id": episode,
                                        "state_id": state, "decision_index": 4, "phase": "early"},
            "effective_tie_epsilon": 1e-6,
            "summary": {
                "gains": [{"reader": "TREE", "family": "packet", "background": 120,
                           "candidate_id": str(i), "kind": "complete", "delta_tokens": 8,
                           "discovery_gain": value, "confirmation_gain": value,
                           "confirmation_noise_sd": .2, "confirmation_positive_draw_fraction": .75}
                          for i, value in enumerate(gains)],
                "selection": [{"reader": "TREE", "family": "packet", "budget": 128,
                               "selection": {"packet": ["a", "b"] if tied else ["a"], "immediate": ["b"]},
                               "ranking_eligible": True, "packet_loss": loss, "random_loss": loss + 1,
                               "packet_vs_random": 1, "packet_vs_immediate": .5, "packet_vs_dispersed": .2,
                               "packet_vs_low": 2, "winner_reversal": reversal}],
                "capability": {"tree_ordinary128": {"expert_full": loss, "tokens": 128},
                               "all_leaves": {"omitted": "Synthetic fixture unavailable"}},
                "interventions": {"erase_conditioning": {"expert_full": loss + 2,
                                                          "baseline_expert_full": loss,
                                                          "loss_change": 2, "tokens": 64}},
                "teacher_reference": {"available": False}},
            "numerical": {"noop_prediction_max_abs": 0, "serial_packed_loss_max_abs": 1e-8}}


def test_summary_averages_within_state_then_episode_without_candidate_pseudoreplication():
    records = [record("a", "a0", [1, 3], 2), record("a", "a1", [5], 4), record("b", "b0", [9], 10)]
    result = summarize_local_measurements(records)
    gain = result["gain_summaries"][0]["metrics"]["confirmation_gain"]
    assert gain == {"mean": 6.25, "states": 3, "episodes": 2, "tasks": 1}
    assert result["capability"][0]["metrics"]["expert_full"]["mean"] == 6.5
    assert len(result["omissions"]) == 3
    assert not any(row["context"] == "teacher_full" for row in result["capability"])
    assert result["numerical"][0]["maximum"]["noop_prediction_max_abs"] == 0
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_local_measurements(records + [records[0]])


def test_tied_winners_are_not_reversal_observations_and_single_moves_are_not_duplicated():
    records = [record("a", "a0", [1], 2, tied=True, reversal=None),
               record("b", "b0", [-1], 3, reversal=True)]
    duplicate = deepcopy(records[0]["summary"]["gains"][0])
    duplicate.update(family="immediate", kind="first", delta_tokens=1)
    records[0]["summary"]["gains"].extend([duplicate, {**duplicate, "kind": "complete"}])
    result = summarize_local_measurements(records)
    metric = result["selection_summaries"][0]["metrics"]
    assert metric["discovery_packet_tie"]["mean"] == .5
    assert metric["winner_reversal"] == {"mean": 1.0, "states": 1, "episodes": 1, "tasks": 1}
    assert len(result["plot_rows"]["gains"]) == 3
    assert result["plot_rows"]["gains"][-1]["confirmation_gain"] == -1
    records[0]["summary"]["gains"][0]["confirmation_gain"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        summarize_local_measurements(records)


def test_plots_and_evidence_page_rebuild_from_results_without_model_or_teacher(tmp_path):
    records = [record("a", "a0", [1, -.5], 2), record("b", "b0", [1.2, -.2], 3)]
    summary = summarize_local_measurements(records)
    links = write_diagnostic_figures(tmp_path, summary)
    assert len(links) == 5
    for link in links:
        path = tmp_path / link["path"]
        assert path.is_file()
        if path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    saved = json.loads((tmp_path / "diagnostics/summary.json").read_text())
    assert saved == summary
    html = (tmp_path / "diagnostics/index.html").read_text()
    assert "not independent episodes" in html
    assert "No population confidence intervals" in html
    assert "Synthetic fixture unavailable" in html
    assert "teacher_full" not in html
    assert all(Path(link["path"]).name in html for link in links[1:])


def test_empty_measurements_are_explicit_not_zero_success_or_fake_effects(tmp_path):
    result = summarize_local_measurements([])
    assert result["gain_summaries"] == result["capability"] == []
    assert result["measured_student_state_cells"] == 0
    links = write_diagnostic_figures(tmp_path, result)
    assert all((tmp_path / link["path"]).is_file() for link in links)
