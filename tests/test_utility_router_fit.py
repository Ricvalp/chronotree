"""Synthetic CPU diagnostics for expert-label fitting, selection and resumption."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from stmem.memory.utility_routing import feature_spec
from stmem.training import utility_routers as fitting
from stmem.training.checkpoints import file_sha256
from test_tracking_routers import RecordingTracker


def study():
    return {
        "experiment_id": "synthetic-utility-study", "source": {"source_sha256": "synthetic-source"},
        "student": {"name": "S_STD", "sha256": "student-hash", "step": 95000, "primary_endpoint": 100000},
        "dependencies": {"teacher_sha256": "teacher-hash"},
        "configuration": {"seed": 9, "data": {"synthetic": True}, "model": {"width": 8},
                          "tracking": {"mode": "disabled"}},
        "protocol": {"seed": 9, "key_width": 4, "budget": 16, "packet_max_cost": 8,
                     "candidate_limit": 8, "fit": {"steps": 4, "batch": 6, "lr": .003,
                                                   "log_every": 1, "eval_every": 2, "checkpoint_every": 2}},
        "states": [{"state_id": f"s{i}", "task": "SyntheticTask", "episode_id": f"episode_{i}",
                    "partition": "fit" if i < 2 else "development"} for i in range(4)],
    }


def contexts(tmp_path, specification):
    rng = np.random.default_rng(29)
    spec = feature_spec(8)
    result = []
    for index, state in enumerate(specification["states"]):
        path = tmp_path / f"synthetic_labels_{index}.npz"
        query = rng.normal(size=spec["query_dim"]).astype("f4")
        candidates = rng.normal(size=(3, spec["candidate_dim"])).astype("f4")
        first = np.array([-.1, .3, -.05], dtype="f4")
        packet = np.array([-.3, -.1, .4], dtype="f4")
        np.savez(path, query=query, candidates=candidates, first=first, packet=packet)
        result.append({**state, "context_id": f"context_{index}", "reader": "TREE",
                       "experiment_id": specification["experiment_id"], "student_sha256": "student-hash",
                       "query": query, "candidates": candidates, "packet_cost": 2,
                       "discovery_first_gain": first, "discovery_packet_gain": packet,
                       "confirmation_first_gain": first * .8,
                       "confirmation_packet_gain": packet * .5,
                       "confirmation_background_loss": 1., "data_path": str(path),
                       "data_sha256": file_sha256(path)})
    return result


@pytest.fixture
def setup_fit(tmp_path, monkeypatch):
    specification = study()
    rows = contexts(tmp_path, specification)
    table = fitting.build_table(rows, specification, "TREE")
    monkeypatch.setattr(fitting, "load_table", lambda *_: (deepcopy(specification), deepcopy(table)))
    monkeypatch.setattr(fitting, "environment_manifest", lambda: {"device": "CPU synthetic test"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    RecordingTracker.instances = []
    monkeypatch.setattr(fitting, "RunTracker", RecordingTracker)
    return specification, rows, table


def test_selection_uses_completed_packet_confirmation_for_both_targets(setup_fit):
    _, _, table = setup_fit
    immediate = fitting.context_choice_rows(table["discovery_first_gain"], table, "immediate", "development")
    packet = fitting.context_choice_rows(table["discovery_packet_gain"], table, "packet", "development")
    assert all(row["selected_indices"] == [1] for row in immediate)
    assert all(row["selected_confirmation_packet_gain"] == pytest.approx(-.05) for row in immediate)
    assert all(row["selected_indices"] == [2] for row in packet)
    assert all(row["selected_confirmation_packet_gain"] == pytest.approx(.2) for row in packet)
    assert all(row["selected_confirmation_packet_loss"] == pytest.approx(.8) for row in packet)


def test_prediction_ties_receive_random_choice_credit(setup_fit):
    _, _, table = setup_fit
    rows = fitting.context_choice_rows(np.zeros(len(table["query"])), table, "packet", "development")
    assert all(row["selected_indices"] == [0, 1, 2] for row in rows)
    assert all(row["selected_minus_random_packet_gain"] == 0 for row in rows)
    assert all(row["confirmation_rank_agreement"] == .5 for row in rows)


def test_episode_aggregation_does_not_count_duplicate_contexts_as_episodes(setup_fit):
    _, _, table = setup_fit
    rows = fitting.context_choice_rows(table["discovery_packet_gain"], table, "packet", "development")
    original = fitting.choice_summary(rows)
    duplicated = fitting.choice_summary(rows + [deepcopy(rows[0])] * 100)
    assert original["episodes"] == duplicated["episodes"] == 2
    assert original["selected_minus_random_packet_gain"] == duplicated["selected_minus_random_packet_gain"]
    first = fitting.paired_episode_interval(rows, "selected_minus_random_packet_gain", draws=100)
    second = fitting.paired_episode_interval(rows + [deepcopy(rows[0])] * 100,
                                            "selected_minus_random_packet_gain", draws=100)
    assert first == second


@pytest.mark.parametrize("target", ["immediate", "packet"])
def test_resume_is_identical_to_uninterrupted_fitting(tmp_path, setup_fit, target):
    stop = tmp_path / "stop"
    stop.touch()
    resumed = tmp_path / "resumed"
    assert fitting.train(tmp_path / "study.json", resumed, reader="TREE", target=target,
                         device="cpu", stop_file=stop) == 75
    saved = torch.load(resumed / "last.pt", weights_only=False)
    assert saved["step"] == 1
    stop.unlink()
    assert fitting.train(tmp_path / "study.json", resumed, reader="TREE", target=target,
                         device="cpu", resume=True) == 0
    uninterrupted = tmp_path / "uninterrupted"
    assert fitting.train(tmp_path / "study.json", uninterrupted, reader="TREE", target=target,
                         device="cpu") == 0
    actual = torch.load(resumed / "last.pt", weights_only=False)
    expected = torch.load(uninterrupted / "last.pt", weights_only=False)
    assert actual["step"] == expected["step"] == 4
    for key in actual["model"]:
        torch.testing.assert_close(actual["model"][key], expected["model"][key], rtol=0, atol=0)
    torch.testing.assert_close(actual["rng_by_rank"][0]["torch"], expected["rng_by_rank"][0]["torch"], rtol=0, atol=0)
    assert actual["provenance"]["compact"] == "student-hash"
    assert actual["utility_protocol"] == actual["protocol"]
    assert (resumed / "step_00000004.pt").is_file()
    receipt = json.loads((resumed / "complete.json").read_text())
    before = len(RecordingTracker.instances)
    assert fitting.train(tmp_path / "study.json", resumed, reader="TREE", target=target,
                         device="cpu", resume=True) == 0
    assert len(RecordingTracker.instances) == before
    reused = json.loads((resumed / "complete.json").read_text())
    for key in ("experiment_id", "reader", "target", "checkpoint_sha256", "completed_steps", "intended_steps"):
        assert reused[key] == receipt[key]


def test_development_targets_cannot_change_training_or_scale(tmp_path, setup_fit):
    _, _, table = setup_fit
    original_dir = tmp_path / "original"
    fitting.train(tmp_path / "study.json", original_dir, reader="TREE", target="packet", device="cpu")
    original = torch.load(original_dir / "last.pt", weights_only=False)
    for name in ("discovery_packet_gain", "discovery_first_gain", "confirmation_packet_gain", "confirmation_first_gain"):
        table[name][table["development_ids"]] *= 1000
    changed_dir = tmp_path / "changed"
    fitting.train(tmp_path / "study.json", changed_dir, reader="TREE", target="packet", device="cpu")
    changed = torch.load(changed_dir / "last.pt", weights_only=False)
    assert original["target_scale"] == changed["target_scale"]
    for key in original["model"]:
        torch.testing.assert_close(original["model"][key], changed["model"][key], rtol=0, atol=0)


def test_loader_rejects_policy_reader_and_schema_mismatches(tmp_path, setup_fit):
    fitting.train(tmp_path / "study.json", tmp_path / "fit", reader="TREE", target="packet", device="cpu")
    path = tmp_path / "fit/last.pt"
    router, state = fitting.load_utility_router(path, expected_student_sha256="student-hash",
                                               expected_teacher_sha256="teacher-hash", expected_reader="tree")
    assert not router.training and not any(p.requires_grad for p in router.parameters())
    with pytest.raises(ValueError, match="student_sha256"):
        fitting.load_utility_router(path, expected_student_sha256="different-student")
    with pytest.raises(ValueError, match="reader"):
        fitting.load_utility_router(path, expected_reader="FLAT")
    state["schema"] = "old_teacher_agreement_schema"
    torch.save(state, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="expert-FM"):
        fitting.load_utility_router(tmp_path / "old.pt")


def test_offline_evaluation_records_independent_choices_and_logs(tmp_path, setup_fit):
    fitting.train(tmp_path / "study.json", tmp_path / "fit", reader="TREE", target="packet", device="cpu")
    result = fitting.evaluate(tmp_path / "study.json", tmp_path / "fit", reader="TREE", target="packet", device="cpu")
    assert result["not_closed_loop_success"] is True
    assert result["metrics"]["states"] == 2
    assert result["selected_minus_random_packet_gain_interval"]["unit"] == "episode_within_task"
    assert json.loads((tmp_path / "fit/evaluation/summary.json").read_text())["status"] == "complete"
    tracker = RecordingTracker.instances[-1]
    assert tracker.options["job_type"] == "evaluation"
    assert "development/selected_minus_random_packet_gain" in tracker.logs[0][1]


def test_resume_rejects_changed_labels_or_target(tmp_path, setup_fit):
    _, _, table = setup_fit
    output = tmp_path / "fit"
    fitting.train(tmp_path / "study.json", output, reader="TREE", target="packet", device="cpu")
    with pytest.raises(ValueError, match="identity"):
        fitting.train(tmp_path / "study.json", output, reader="TREE", target="immediate", device="cpu", resume=True)
    table["files"][0]["sha256"] = "changed"
    with pytest.raises(ValueError, match="identity"):
        fitting.train(tmp_path / "study.json", output, reader="TREE", target="packet", device="cpu", resume=True)


def test_resume_without_checkpoint_can_retry_unstarted_allocation(tmp_path, setup_fit):
    assert fitting.train(tmp_path / "study.json", tmp_path / "fit", reader="TREE", target="packet",
                         device="cpu", resume=True) == 0


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows[0].update(partition="development"), "partition"),
    (lambda rows: rows[0].update(student_sha256="other"), "identity"),
    (lambda rows: rows[0].update(packet_cost=0), "positive-cost"),
    (lambda rows: rows.append(deepcopy(rows[0])), "Duplicate"),
    (lambda rows: rows[0].update(discovery_packet_gain=[np.nan] * 3), "Nonfinite|nonfinite"),
])
def test_corrupt_label_rows_are_rejected(tmp_path, mutation, match):
    specification = study()
    rows = contexts(tmp_path, specification)
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        fitting.build_table(rows, specification, "TREE")


def test_same_episode_cannot_cross_fit_development(tmp_path):
    specification = study()
    specification["states"][2]["episode_id"] = specification["states"][0]["episode_id"]
    rows = contexts(tmp_path, specification)
    with pytest.raises(ValueError, match="Episode leakage"):
        fitting.build_table(rows, specification, "TREE")
