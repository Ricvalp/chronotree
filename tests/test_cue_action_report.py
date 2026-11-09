"""Presentation checks for controlled history-cue action diagnostics."""

from copy import deepcopy
from html.parser import HTMLParser
import json
import re

import pytest

from stmem.inspection.cue_action_report import validate_result, write_report


def fixture():
    variant = {"name": "uniform", "tokens": 4, "description": "Synthetic diagnostic fixture",
               "actions": [[[0.] * 8, [1.] * 8]] * 4,
               "tcp_paths": [[[0., 0., 0.], [.1, .2, .3]]] * 4,
               "selected_nodes": [{"anchor": [0., 0., .1], "time_anchor": 0, "count": 3}]}
    return {"protocol": {"scope": "Synthetic report fixture; no policy measurement.",
                         "noise_seeds": [11, 12, 13, 14], "notes": ["<script>alert(1)</script>"]},
            "scenes": [{"id": "example", "decision_index": 0,
                        "goal": "Find </script><script>alert(2)</script>",
                        "current": {"points": [[0, 0, .1, .2, .3, .4]], "tcp": [0, 0, .2]},
                        "variants": [variant, {**deepcopy(variant), "name": "cue_target"}],
                        "comparisons": [{"a": "uniform", "b": "cue_target", "mean_tcp_delta_cm": [0.] * 4}]}]}


@pytest.mark.parametrize("change", [
    lambda x: x["protocol"].update(noise_seeds=[11] * 4),
    lambda x: x["scenes"][0]["variants"][0].update(actions=[[[0.] * 8]] * 4),
    lambda x: x["scenes"][0]["variants"][0].update(tcp_paths=[[[0., 0., 0.], [0., 0., 0.]]] * 3),
    lambda x: x["scenes"][0]["variants"][0]["actions"][0][0].__setitem__(0, float("nan")),
    lambda x: x["scenes"][0]["current"].update(points=[[0., 0., 0.]]),
    lambda x: x["scenes"][0]["current"].update(images=[{"data_url": "javascript:alert(1)"}]),
    lambda x: x["scenes"][0]["comparisons"][0].update(b="missing"),
])
def test_reject_mismatched_or_nonfinite_measurements(change):
    result = fixture()
    change(result)
    with pytest.raises(ValueError):
        validate_result(result)


def test_offline_report_preserves_measured_data_without_script_injection(tmp_path):
    result = fixture()
    original = deepcopy(result)
    path = write_report(result, tmp_path)
    html = path.read_text()
    payload = re.search(r'<script type="application/json" id="payload">(.*?)</script>', html, re.S).group(1)
    assert json.loads(payload) == original and result == original
    assert "<script>alert(1)</script>" not in html
    assert "<script>alert(2)</script>" not in html
    assert "coarse summaries still contain their descendants" in html
    assert "plotly.js" in html

    class Sources(HTMLParser):
        def __init__(self):
            super().__init__()
            self.sources = []

        def handle_starttag(self, tag, attrs):
            self.sources.extend(value for name, value in attrs if name == "src")

    parsed = Sources()
    parsed.feed(html)
    assert not parsed.sources


def test_counterfactual_title_description_and_geometry_remain_explicit(tmp_path):
    result = fixture()
    result["protocol"].update(
        title="Swap <blue> and red?",
        intervention_description="Historical RGB colors are exchanged. <script>bad()</script>",
        expert_label="Expert: original history only",
        default_comparison={"a": "uniform", "b": "cue_target"},
    )
    result["scenes"][0]["target_regions"] = [
        {"label": "Original blue location", "center": [.1, .2, .03]},
        {"label": "Swapped blue location", "center": [.2, .1, .03]},
    ]
    result["scenes"][0]["cue_frames"] = [
        {"source": "conditioning", "source_index": 0, "label": "Original", "points": []},
        {"source": "conditioning", "source_index": 0, "label": "Swapped", "points": []},
    ]
    html = write_report(result, tmp_path).read_text()
    assert "Swap &lt;blue&gt; and red?" in html
    assert "Historical RGB colors are exchanged. &lt;script&gt;bad()&lt;/script&gt;" in html
    assert "<script>bad()</script>" not in html
    assert "coarse summaries still contain their descendants" not in html
    encoded = re.search(r'<script type="application/json" id="payload">(.*?)</script>', html, re.S).group(1)
    assert json.loads(encoded) == result
    result["scenes"][0]["target_regions"][0]["center"] = [0, 1]
    with pytest.raises(ValueError, match="Annotated region center"):
        validate_result(result)
