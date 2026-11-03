"""CPU-only count-probe report checks with explicitly synthetic predictions."""

from copy import deepcopy
import json
import re

import pytest

from stmem.inspection.count_report import aggregate_rows, validate_result, write_report


def fixture():
    scene = {
        "points": [[0.0, 0.0, 0.0, 0.2, 0.3, 0.4], [0.1, 0.0, 0.0, 0.4, 0.3, 0.2]],
        "valid": [True, False], "tcp": [0.0, 0.0, 0.1],
        "button": [0.2, 0.0, 0.0], "cube": [0.1, 0.1, 0.0],
    }
    rows = []
    for x in (1, 2):
        for seed in (11, 12):
            rows.append({
                "label": "teacher", "method": "FULL", "checkpoint_step": 100,
                "history_mode": "fixed_current", "scene_id": "scene1", "k": 1, "x": x,
                "noise_seed": seed, "expected_next": "press_button" if x == 1 else "pick_cube",
                "tcp_positions_m": [[0.01 * i, 0.0, 0.1] for i in range(16)],
                "physical_actions": [[0.0] * 8 for _ in range(16)],
                "metrics": {"button_distance_end_mm": float(seed),
                            "button_distance_min_first8_mm": float(seed + 1),
                            "tcp_displacement_first8_mm": float(seed + 2)},
            })
    return {
        "protocol": {"task": "PickXtimes", "scope": "Synthetic report fixture, not measured results",
                     "execute_actions": 8, "action_stride": 2, "noise_seeds": [11, 12], "notes": []},
        "source": {"goals": {"1": "Pick once", "2": "Pick twice"}},
        "current_scenes": {"scene1": scene}, "predictions": rows,
    }


def test_means_keep_model_reader_and_count_conditions_separate():
    result = fixture()
    student = deepcopy(result["predictions"][0])
    student.update(label="student", method="UNIFORM", checkpoint_step=15000)
    student["metrics"]["button_distance_end_mm"] = 90.0
    result["predictions"].append(student)
    assert validate_result(result) == 16
    rows = aggregate_rows(result)
    assert len(rows) == 3
    teacher = [r for r in rows if r["label"] == "teacher"]
    assert [r["x"] for r in teacher] == [1, 2]
    assert all(r["samples"] == 2 and r["scenes"] == 1 and r["noise_seeds"] == 2 for r in teacher)
    assert all(r["button_distance_end_mm"] == 11.5 for r in teacher)
    assert next(r for r in rows if r["label"] == "student")["button_distance_end_mm"] == 90.0


def test_missing_metric_is_not_silently_averaged_over_fewer_seeds():
    result = fixture()
    result["predictions"][0]["metrics"].pop("button_distance_end_mm")
    assert aggregate_rows(result)[0]["button_distance_end_mm"] is None


@pytest.mark.parametrize("change, message", [
    (lambda r: r["predictions"][0].update(expected_next="pick_cube"), "Expected next phase"),
    (lambda r: r["predictions"][0].update(k=3), "completed k"),
    (lambda r: r["predictions"].append(deepcopy(r["predictions"][0])), "Duplicate"),
    (lambda r: r["predictions"][0]["tcp_positions_m"].pop(), "shape"),
    (lambda r: r["predictions"][0]["physical_actions"][0].__setitem__(0, float("nan")), "finite"),
    (lambda r: r["current_scenes"]["scene1"].update(valid=[True]), "boolean mask"),
])
def test_invalid_scientific_artifacts_fail_before_render(change, message):
    result = fixture()
    change(result)
    with pytest.raises(ValueError, match=message):
        validate_result(result)


def test_standalone_report_retains_exact_predictions_and_escapes_payload(tmp_path):
    result = fixture()
    result["source"]["goals"]["1"] = '</script><img src=x onerror="bad()">'
    before = deepcopy(result)
    path = write_report(result, tmp_path)
    document = path.read_text()
    assert result == before
    assert "plotly.js" in document and '<script src="http' not in document
    assert "not a benchmark success rate" in document
    assert "not executed robot motion" in document
    assert "First 8 targets: would execute" in document
    assert "Targets 9–16: prediction only" in document
    assert '<option value="action">Action area</option>' in document
    assert '<option value="full">Full scene</option>' in document
    assert "This changes only the view, never the policy input or predictions" in document
    payload = re.search(r'<script type="application/json" id="payload">(.*?)</script>', document, re.S)
    assert payload is not None and json.loads(payload.group(1)) == result
    assert '</script><img' not in payload.group(1)
    assert "11.50" in document
