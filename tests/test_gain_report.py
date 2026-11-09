"""Presentation invariants for measured, paired memory-utility diagnostics."""

from copy import deepcopy
from html.parser import HTMLParser
import json
import re

import pytest

from stmem.inspection.gain_report import summarize_context, validate_result, write_report
from stmem.inspection.gain_statistics import summarize_gains


def fixture():
    return {
        "protocol": {
            "scope": "Synthetic report fixture, not a policy measurement.",
            "noise_seeds": list(range(16)), "integration_steps": 8,
        },
        "contexts": [{
            "id": "scene_0", "task": "VideoUnmask", "episode": "episode_0",
            "decision_index": 12, "reader": "TREE", "budget": 128,
            "goal": "Inspect </script><script>alert(1)</script>",
            "current": {"points": [[0, 0, .1, .2, .3, .4]], "tcp": [0, 0, .2]},
            "candidates": [
                {"id": "parent_0", "gains": {"final_action": [3.] * 8 + [-1.] * 8}},
                {"id": "parent_1", "gains": {"final_action": [1.] * 16}},
                {"id": "parent_2", "gains": {"final_action": [-2.] * 8 + [5.] * 8}},
            ],
        }],
    }


def test_extremes_selected_on_discovery_draws_only():
    result = summarize_context(fixture()["contexts"][0])
    assert result["high_id"] == "parent_0"
    assert result["low_id"] == "parent_2"
    assert result["confirmation_difference_mean"] == -6
    assert result["confirmation_high_beats_low_fraction"] == 0
    assert result["discovery_draws"] == result["confirmation_draws"] == 8
    assert result["candidates"][0]["positive_fraction"] == .5
    assert result["candidates"][2]["best_fraction"] == .5


def test_exact_ties_share_best_credit_and_keep_distinct_default_choices():
    payload = fixture()
    for candidate in payload["contexts"][0]["candidates"]:
        candidate["gains"]["final_action"] = [-2.] * 16
    summary = summarize_context(payload["contexts"][0])
    assert summary["high_id"] != summary["low_id"]
    assert summary["confirmation_difference_mean"] == 0
    assert all(r["best_fraction"] == pytest.approx(1 / 3) for r in summary["candidates"])
    assert all(r["positive_fraction"] == 0 for r in summary["candidates"])


@pytest.mark.parametrize("mutation", [
    lambda p: p["protocol"].update(noise_seeds=[1] * 16),
    lambda p: p["contexts"][0]["candidates"][0]["gains"].update(final_action=[1.] * 15),
    lambda p: p["contexts"][0]["candidates"][0]["gains"].update(final_action=[float("nan")] * 16),
    lambda p: p["contexts"][0]["candidates"][0]["gains"].update(velocity=[1.] * 16),
    lambda p: p["contexts"][0]["current"].update(points=[[0, 0, 0]]),
    lambda p: p["contexts"][0]["current"].update(images=[{"data_url": "https://example.com/image.png"}]),
    lambda p: p["contexts"][0]["candidates"][0].update(tcp_paths={"after": [[[0, 0, 0]]] * 15}),
    lambda p: p["contexts"][0].update(expert_valid=[1, 1, 0]),
])
def test_reject_incomplete_or_unpaired_payload(mutation):
    payload = fixture()
    mutation(payload)
    with pytest.raises(ValueError):
        validate_result(payload)


def test_offline_export_preserves_every_candidate_and_escapes_script_data(tmp_path):
    payload = fixture()
    original = deepcopy(payload)
    index = write_report(payload, tmp_path)
    page = next(tmp_path.glob("000_*.html")).read_text()
    encoded = re.search(r'<script type="application/json" id="payload">(.*?)</script>', page, re.S).group(1)
    decoded = json.loads(encoded)
    assert decoded["context"] == original["contexts"][0]
    assert payload == original
    assert "<script>alert(1)</script>" not in page
    assert "plotly.js" in page
    class Sources(HTMLParser):
        def __init__(self):
            super().__init__()
            self.urls = []

        def handle_starttag(self, tag, attributes):
            self.urls.extend(v for k, v in attributes if k == "src")

    parser = Sources()
    parser.feed(page)
    assert not parser.urls
    assert "Held-out" in page
    assert "not task success rates" in index.read_text()
    assert "-6" in index.read_text()


def test_checkpoint_mismatch_is_prominent_in_index_and_each_example(tmp_path):
    payload = fixture()
    payload["protocol"].update(exact_hpc_compact_file=False, compact={"step": 83000})
    path = write_report(payload, tmp_path)
    for page in [path, *tmp_path.glob("000_*.html")]:
        text = page.read_text()
        assert '<p class="warning">Checkpoint scope:' in text
        assert '83,000' in text
        assert 'not an exact reproduction' in text


def test_report_index_uses_same_near_tie_convention_as_measured_statistics():
    import numpy as np

    payload = fixture()
    context = payload["contexts"][0]
    gains = np.array([[1e-5, 1.05e-5, 0.], [1e-5, 1.05e-5, 0.],
                      [1e-5, -1e-5, 0.], [-1e-5, 1e-5, 0.]])
    for index, candidate in enumerate(context["candidates"]):
        candidate["gains"]["final_action"] = gains[:, index].tolist()
    expected = summarize_gains(gains, epsilon=1e-6)
    context["statistics"] = {"final_action": expected}
    measured = summarize_context(context)
    assert measured["confirmation_difference_mean"] == expected["validation"]["high_minus_low_mean_gain"]
    assert measured["confirmation_high_beats_low_fraction"] == expected["validation"]["high_beats_low_probability"]


def test_featured_examples_link_to_every_measured_context(tmp_path):
    payload = fixture()
    payload["featured"] = [{"id": "scene_0", "title": "A useful counterexample", "note": "Both gains are negative."}]
    index = write_report(payload, tmp_path).read_text()
    assert "A useful counterexample" in index
    assert "All 1 measured contexts" in index
    assert "not a representative subset" in index
    assert index.count('href="000_VideoUnmask_TREE.html"') == 2
