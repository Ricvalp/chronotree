"""Synthetic CPU invariants for compact-student action witnesses."""

# ruff: noqa: F811
from copy import deepcopy
from html.parser import HTMLParser
import json

import numpy as np
import pytest

from test_cached_context_loader import fixture  # noqa: F401
from test_packed_context import indexed as indexed
from test_refinement_measurements import payload, STUDENT, TEACHER
from stmem.inspection import student_behavior as behavior
from stmem.inspection.refinement_measurements import generated_state


def selection_fixture():
    family = {"reader": "TREE", "panel": "packet", "background": 120, "final_budget": 128,
              "base": "base", "controls": {"dispersed": "dispersed", "uniform": "uniform"},
              "candidates": [{"candidate_id": name, "prefixes": {"1": name + "1", "8": name + "8"}}
                             for name in ("a", "b", "c")]}
    losses = {"base": 1.5, "dispersed": 1.0, "uniform": 1.1,
              "a1": .3, "a8": .8, "b1": .5, "b8": .4, "c1": .9, "c8": 1.2}
    rows = [{"context_id": key, "bank": bank, "draw_index": draw, "expert_full": loss + draw * .01}
            for bank in ("discovery", "confirmation") for draw in range(2) for key, loss in losses.items()]
    return {"families": [family]}, rows


def test_selection_uses_only_discovery_and_compares_completed_equal_cost_packets():
    panel, rows = selection_fixture()
    first = behavior.select_behavior_contexts(panel, rows, reader="TREE", random_seed=19)
    assert first["aliases"]["immediate_completed"] == "a8"
    assert first["aliases"]["packet_best"] == "b8"
    assert first["aliases"]["packet_worst"] == "c8"
    assert first["packet_cost"] == 8 and first["final_budget"] == 128
    changed = deepcopy(rows)
    for row in changed:
        if row["bank"] == "confirmation":
            row["expert_full"] = -1000 if row["context_id"] == "c8" else 1000
    assert behavior.select_behavior_contexts(panel, list(reversed(changed)), reader="TREE", random_seed=19) == first


def test_selection_keeps_ties_and_rejects_incomplete_or_duplicate_discovery():
    panel, rows = selection_fixture()
    for row in rows:
        if row["context_id"] in ("a8", "b8", "c8"):
            row["expert_full"] = .5
    selection = behavior.select_behavior_contexts(panel, rows, reader="TREE")
    assert selection["tie_sets"]["packet_best"] == ["a", "b", "c"]
    assert selection["tie_sets"]["packet_worst"] == ["a", "b", "c"]
    assert selection["aliases"]["packet_best"] == selection["aliases"]["packet_worst"]
    with pytest.raises(ValueError, match="same discovery draws"):
        behavior.select_behavior_contexts(panel, rows[1:], reader="TREE")
    with pytest.raises(ValueError, match="Duplicate"):
        behavior.select_behavior_contexts(panel, rows + [rows[0]], reader="TREE")
    with pytest.raises(ValueError, match="predeclared"):
        behavior.select_behavior_contexts(panel, rows, reader="FLAT")


def test_generation_is_compact_without_teacher_and_independent_of_expert_targets(indexed):
    _, student, item, contexts, draws = payload(indexed)
    banks = {"base": contexts["tree_base"], "packet_best": contexts["flat_base"]}
    options = {"checkpoint_hash": STUDENT, "task": "PickXtimes", "prefix_targets": 1}
    seeds = [d["seed"] for d in draws[:2]]
    original = behavior.generate_compact_actions(student, item, banks, seeds, **options)
    changed = deepcopy(item)
    changed["target"] += 100
    changed["valid"][:, -1] = False
    other = behavior.generate_compact_actions(student, changed, dict(reversed(list(banks.items()))),
                                               seeds, batch_size=1, **options)
    assert original["teacher_status"] == "not_supplied" and "teacher_full" not in original["contexts"]
    assert original["initial_noise_sha256"] == other["initial_noise_sha256"]
    for name in banks:
        np.testing.assert_allclose(original["contexts"][name]["normalized_actions"],
                                   other["contexts"][name]["normalized_actions"], atol=2e-6, rtol=3e-5)
    assert original["contexts"]["base"]["expert_full"] != other["contexts"]["base"]["expert_full"]
    assert original["history_token_counts"]["base"] == int(banks["base"].valid.sum())
    assert original["expert_tcp_valid"] == item["valid"][:, :7].all(-1).tolist()
    assert np.isfinite(original["contexts"]["base"]["tcp_paths"]).all()


def test_optional_teacher_agrees_with_existing_generated_measurement(indexed):
    teacher, student, item, contexts, draws = payload(indexed)
    banks = {"base": contexts["tree_base"], "dispersed": contexts["flat_base"]}
    seeds = [draws[0]["seed"]]
    expected = generated_state(teacher, student, item, banks, seeds, teacher_hash=TEACHER,
                               student_hash=STUDENT, task="PickXtimes", prefix_targets=1)
    got = behavior.generate_compact_actions(student, item, banks, seeds, checkpoint_hash=STUDENT,
                                             teacher=teacher, teacher_hash=TEACHER,
                                             task="PickXtimes", prefix_targets=1)
    assert got["teacher_status"] == "measured"
    for name in expected["contexts"]:
        np.testing.assert_allclose(got["contexts"][name]["normalized_actions"],
                                   expected["contexts"][name]["normalized_actions"], atol=2e-6, rtol=3e-5)
        np.testing.assert_allclose(got["contexts"][name]["physical_actions"],
                                   expected["contexts"][name]["physical_actions"], atol=2e-6, rtol=3e-5)


def test_generation_rejects_unfrozen_policy_unknown_robot_and_fake_teacher(indexed):
    _, student, item, contexts, _ = payload(indexed)
    banks = {"base": contexts["tree_base"]}
    options = {"checkpoint_hash": STUDENT, "task": "PickXtimes", "prefix_targets": 1}
    with pytest.raises(ValueError, match="teacher hash"):
        behavior.generate_compact_actions(student, item, banks, [10], teacher_hash=TEACHER, **options)
    with pytest.raises(ValueError, match="distinct"):
        behavior.generate_compact_actions(student, item, banks, [10, 10], **options)
    with pytest.raises(ValueError, match="eight Euler"):
        behavior.generate_compact_actions(student, item, banks, [10], integration_steps=4, **options)
    student.decoder.train()
    with pytest.raises(ValueError, match="frozen eval"):
        behavior.generate_compact_actions(student, item, banks, [10], **options)


def test_report_rebuild_checks_arrays_and_explains_contexts(indexed, tmp_path):
    pytest.importorskip("plotly")
    _, student, item, contexts, _ = payload(indexed)
    banks = {"base": contexts["tree_base"], "packet_best": contexts["flat_base"]}
    result = behavior.generate_compact_actions(student, item, banks, [101], checkpoint_hash=STUDENT,
                                               task="PickXtimes", prefix_targets=1)
    selection = {"aliases": {"base": "context_a", "packet_best": "context_b"},
                 "reader": "TREE", "background": int(banks["base"].valid.sum()),
                 "final_budget": int(banks["packet_best"].valid.sum()),
                 "selection_bank": "discovery"}
    with pytest.raises(ValueError, match="token count"):
        behavior.write_behavior_case(tmp_path / "invalid.npz", result, state={}, student="S_STD",
                                     selection={**selection, "final_budget": selection["final_budget"] + 1})
    paths = behavior.write_behavior_case(tmp_path / "witness.npz", result,
                                         state={"state_id": "synthetic", "task": "PickXtimes",
                                                "episode_id": "synthetic_fixture", "decision_index": 2},
                                         student="S_STD</script>", selection=selection,
                                         scene_points=np.ones((3, 6)), identity={"synthetic": True})
    html = (tmp_path / "witness.html").read_text()
    assert "Recorded expert" in html and "Selected history" in html
    assert "source-local time" in html
    assert "S_STD</script>" not in html and "S_STD\\u003c/script>" in html
    class Scripts(HTMLParser):
        external = []

        def handle_starttag(self, tag, attrs):
            if tag == "script" and "src" in dict(attrs):
                self.external.append(dict(attrs)["src"])

    parser = Scripts()
    parser.feed(html)
    assert not parser.external
    metadata = json.loads((tmp_path / "witness.json").read_text())
    assert metadata["simulation_status"] == "not_run"
    assert metadata["identity"]["synthetic"]
    assert metadata["metrics"]["base"]["expert_prefix"] == result["contexts"]["base"]["expert_prefix"]
    assert paths["html"].endswith("witness.html")
    (tmp_path / "witness.npz").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        behavior.report_behavior_case(tmp_path / "witness.npz")
