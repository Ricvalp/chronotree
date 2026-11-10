from stmem.inspection.refinement_statistics import episode_bootstrap, select_and_confirm, tie_set


def test_confirmation_never_reselects_and_ties_average():
    discovery = {"a": 1.0, "b": 1.0 + 1e-7, "c": 2.0}
    first = select_and_confirm(discovery, {"a": 7., "b": 9., "c": 0.})
    second = select_and_confirm(discovery, {"a": 0., "b": 1., "c": -100.})
    assert first["selected_ids"] == second["selected_ids"] == ["a", "b"]
    assert first["confirmation_loss"] == 8.
    assert first["tie_count"] == 2
    assert tie_set({"all_negative_a": -.001, "all_negative_b": -.01}, minimize=False) == ["all_negative_a"]


def test_bootstrap_unit_is_episode_not_draw_or_candidate():
    rows = [{"task": "T", "episode_id": "a", "loss": 2.},
            {"task": "T", "episode_id": "b", "loss": 4.}]
    once = episode_bootstrap(rows, "loss")
    repeated = episode_bootstrap([rows[0]] * 500 + [rows[1]], "loss")
    assert once == repeated
    assert once["episodes"] == 2 and once["mean"] == 3.
    assert episode_bootstrap(rows[:1], "loss")["interval"] is None


def panel_fixture():
    from stmem.inspection.refinement_statistics import analyze_panel

    state = {"student": "S_STD", "task": "T", "episode_id": "episode_0", "state_id": "s0",
             "partition": "validation", "phase": "late"}
    panel = {"groups": [{"reader": "TREE", "family": "packet", "background": 120, "cost": 8,
              "base": "base", "paired": True, "controls": {"uniform": "u", "dispersed": "d"},
              "candidates": [{"id": "a", "first": "a1", "complete": "a8", "depth": 3,
                              "source": "conditioning", "support_count": 64},
                             {"id": "b", "first": "b1", "complete": "b8", "depth": 5,
                              "source": "execution", "support_count": 512}]}]}
    values = {"base": (10., 10.), "a1": (7., 9.), "b1": (8., 2.), "a8": (6., 9.),
              "b8": (8., 1.), "u": (8., 3.), "d": (9., 4.)}
    rows = [{"context_id": key, "bank": bank, "draw_index": draw, "expert_full": value,
             "expert_prefix": prefix, "teacher_full": 12. - value, "teacher_prefix": prefix}
            for bank in ("discovery", "confirmation") for draw in (0, 1)
            for key, (value, prefix) in values.items()]
    return state, panel, rows, analyze_panel(state, panel, rows)


def test_prefix_is_secondary_and_teacher_and_low_selection_remain_distinct():
    _, _, _, analysis = panel_fixture()
    row = analysis["selection"][0]
    # Full-chunk selection chooses a although b is much better on prefix loss.
    assert row["selection"]["packet"]["selected_ids"] == ["a"]
    assert row["packet_prefix_loss"] == 9.
    assert row["selection"]["teacher"]["selected_ids"] == ["b"]
    assert row["selection"]["low"]["selected_ids"] == ["b"]
    assert row["teacher_selected_prefix_loss"] == row["low_selected_prefix_loss"] == 1.
    assert row["uniform_prefix_loss"] == 3. and row["dispersed_prefix_loss"] == 4.


def test_all_negative_pools_retained_and_candidate_decision_denominators_separate():
    from stmem.inspection.refinement_statistics import analyze_panel, population_summary

    state, panel, rows, _ = panel_fixture()
    for row in rows:
        if row["context_id"] == "base":
            row["expert_full"] = 1.
    analysis = analyze_panel(state, panel, rows)
    selection = analysis["selection"][0]
    assert selection["all_negative_discovery"] == selection["all_negative_confirmation"] == 1.
    population = population_summary([analysis])
    assert population["prevalence"]["S_STD/validation/TREE/packet/complete"]["candidate_measurements"] == 2
    decision = population["decision_prevalence"]["S_STD/validation/TREE/packet/complete/120"]
    assert decision["decision_contexts"] == 1 and decision["all_stable_harmful"]["mean"] == 1.
    strata = population["gain_strata"]
    assert strata["S_STD/validation/TREE/packet/complete/source/conditioning"]["candidate_measurements"] == 1
    assert strata["S_STD/validation/TREE/packet/complete/phase/late"]["confirmation_gain"]["episodes"] == 1


def test_matched_student_summary_requires_same_state_and_excludes_s0():
    from copy import deepcopy
    from stmem.inspection.refinement_statistics import paired_student_summary

    _, _, _, a = panel_fixture()
    a["identity"] = {"student": "S_STD", "experiment_id": "fixed", "initialization": "accepted_teacher_weights",
                     "initialization_sha256": "teacher", "training_manifest_sha256": "panel", "step": 30000,
                     "endpoint": 30000, "precision": "fp32", "mode": "main"}
    b = deepcopy(a)
    b["identity"]["student"] = "S_STAGE"
    for row in b["selection"]:
        row.update(student="S_STAGE", packet_loss=row["packet_loss"] - 2,
                   packet_vs_random=row["packet_vs_random"] + 1)
    c = deepcopy(a)
    c["selection"][0].update(student="S0", packet_loss=-100)
    summary = paired_student_summary([a, b, c])
    assert len(summary) == 1
    result = next(iter(summary.values()))
    assert result["absolute_packet_improvement"]["mean"] == 2.
    assert result["allocation_benefit_improvement"]["mean"] == 1.
    b["selection"][0]["state_id"] = "different_state"
    assert not paired_student_summary([a, b])


def test_student_pairs_require_matched_step_initialization_and_declared_endpoint():
    from copy import deepcopy
    from stmem.inspection.refinement_statistics import paired_student_summary, population_summary

    _, _, _, a = panel_fixture()
    a["identity"] = {"student": "S_STD", "experiment_id": "fixed", "initialization": "accepted_teacher_weights",
                     "initialization_sha256": "teacher", "training_manifest_sha256": "panel", "step": 30000,
                     "endpoint": 30000, "precision": "fp32", "mode": "main"}
    b = deepcopy(a)
    b["identity"]["student"] = "S_STAGE"
    b["selection"][0]["student"] = "S_STAGE"
    for change in ({"step": 15000}, {"initialization": "S0_weights", "initialization_sha256": "S0"},
                   {"endpoint": 40000}, {"training_manifest_sha256": "different_pool"}):
        unmatched = deepcopy(b)
        unmatched["identity"].update(change)
        result = population_summary([a, unmatched])
        assert result["student_pairs"] == {}
        omitted = result["student_pair_omissions"]
        assert omitted and set(change) <= set(omitted[0]["differing_fields"])
    absent = deepcopy(b)
    absent.pop("identity")
    assert not paired_student_summary([a, absent])
    for analysis in (a, b):
        analysis["identity"].update(initialization="S0_weights", initialization_sha256="S0",
                                    step=10000, endpoint=10000)
    result = paired_student_summary([a, b])
    assert len(result) == 1
    matched = next(iter(result.values()))["matched_identity"]
    assert matched["initialization"] == "S0_weights" and matched["step"] == matched["endpoint"] == 10000


def test_population_honors_configured_gain_deadband_and_relative_thresholds():
    import pytest
    from stmem.inspection.refinement_statistics import population_summary

    _, _, _, analysis = panel_fixture()
    summary = population_summary([analysis], epsilon=3., relative_thresholds=[.1, .5])
    complete = summary["prevalence"]["S_STD/validation/TREE/packet/complete"]
    assert complete["stable_positive"]["mean"] == .5  # gains 4 and 2
    assert complete["small_relative_0.1"]["mean"] == 0.
    assert complete["small_relative_0.5"]["mean"] == 1.
    assert "small_relative_0.005" not in complete
    assert summary["gain_epsilon"] == 3. and summary["relative_thresholds"] == [.1, .5]
    for invalid in ([], [.1, .1], [float("nan")], [-1.]):
        with pytest.raises(ValueError, match="Relative thresholds"):
            population_summary([], relative_thresholds=invalid)
