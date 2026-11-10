"""Synthetic checks: common cases are selected before confirmation is inspected."""
from copy import deepcopy
import json

import numpy as np
import pytest

from stmem.inspection.refinement_generated import (
    _measurement_folder, _seal, _write_result, diagnostic_state, select_common_cases, select_diagnostic_case,
    validate_generated_endpoints,
)


def record(state_id="state0", task="VideoUnmask", episode="episode_0", *, flip=False):
    state = {"state_id": state_id, "task": task, "episode_id": episode,
             "decision_index": 40, "partition": "train", "cache_eligible": True}
    rows, families = [], []
    for background in (56, 120):
        base, dispersed = f"{background}/base", f"{background}/dispersed"
        candidates = [{"candidate_id": name, "support_hash": name + "_support",
                       "prefixes": {"1": f"{background}/{name}/first", "8": f"{background}/{name}/full"}}
                      for name in ("left", "right")]
        families.append({"reader": "TREE", "panel": "packet", "background": background,
                         "final_budget": background + 8, "base": base, "paired": True,
                         "controls": {"dispersed": dispersed}, "candidates": candidates})
        values = {base: 10.0, dispersed: 9.0,
                  candidates[0]["prefixes"]["1"]: 5.0 if not flip else 9.0,
                  candidates[1]["prefixes"]["1"]: 9.0 if not flip else 5.0,
                  candidates[0]["prefixes"]["8"]: 8.0 if not flip else 2.0,
                  candidates[1]["prefixes"]["8"]: 2.0 if not flip else 8.0}
        for context, value in values.items():
            for bank in ("discovery", "confirmation"):
                for draw in range(4):
                    rows.append({"context_id": context, "bank": bank, "draw_index": draw,
                                 "expert_full": value + (0 if background == 56 else .1),
                                 "expert_prefix": value, "teacher_full": 10 - value,
                                 "teacher_prefix": 10 - value})
    return {"state": state, "panel": {"families": families}, "rows": rows}


def test_confirmation_values_and_validation_states_cannot_change_case_choices():
    records = {student: [record(f"state{i}", episode=f"episode_{i}", flip=student == "S_PACKET") for i in range(5)]
               for student in ("S0", "S_STD", "S_STAGE", "S_PACKET")}
    before = select_common_cases(records, seed=5, max_cases=4)
    changed = deepcopy(records)
    for student, values in changed.items():
        for value in values:
            for row in value["rows"]:
                if row["bank"] == "confirmation":
                    row["expert_full"] = -10000 * (row["draw_index"] + 1)
                    row["teacher_full"] = 99999999
        validation = record("validation", task="RouteStick", episode="episode_9")
        validation["state"]["partition"] = "validation"
        values.insert(0, validation)
    after = select_common_cases(changed, seed=5, max_cases=4)
    assert before == after
    assert len(before["cases"]) == 4
    assert all(set(case["students"]) == set(records) for case in before["cases"])
    for case in before["cases"]:
        assert case["students"]["S0"]["representatives"]["immediate"] == "left"
        assert case["students"]["S0"]["representatives"]["packet"] == "right"
        assert case["students"]["S_PACKET"]["representatives"]["packet"] == "left"


def test_selection_is_symmetric_under_student_order_and_obeys_diversity_caps():
    records = {student: [record(f"state{task}_{episode}_{decision}", task=task, episode=f"episode_{episode}")
                         for task in ("VideoUnmask", "RouteStick", "PickXtimes")
                         for episode in range(3) for decision in range(2)]
               for student in ("S0", "S_STD", "S_STAGE", "S_PACKET")}
    forward = select_common_cases(records, seed=10, max_cases=16, max_per_task=4, max_per_episode=1)
    reverse = select_common_cases(dict(reversed(list(records.items()))), seed=10, max_cases=16,
                                  max_per_task=4, max_per_episode=1)
    assert forward == reverse
    cases = forward["cases"]
    assert len(cases) == 9
    assert len({(c["state"]["task"], c["state"]["episode_id"]) for c in cases}) == len(cases)
    assert len({c["state"]["task"] for c in cases}) == 3


def test_case_matching_rejects_different_candidate_manifests():
    records = {"S_STD": [record()], "S_PACKET": [record()]}
    records["S_PACKET"][0]["panel"]["families"][0]["candidates"][0]["support_hash"] = "changed"
    with pytest.raises(ValueError, match="exact gain-blind"):
        select_common_cases(records, seed=0)


def test_near_ties_have_fixed_representative_and_preserved_tie_set():
    item = record()
    for row in item["rows"]:
        if row["bank"] == "discovery" and row["context_id"].endswith("/full"):
            row["expert_full"] = 1.0
    result = select_common_cases({"S0": [item]}, seed=0)
    assert result["cases"][0]["students"]["S0"]["tie_sets"]["packet"] == ["left", "right"]
    assert result["cases"][0]["students"]["S0"]["representatives"]["packet"] == "left"


def test_zero_effects_are_only_nominated_as_null_controls():
    item = record()
    for row in item["rows"]:
        row["expert_full"] = 2.0
        row["teacher_full"] = 3.0
    result = select_common_cases({"S0": [item]}, seed=0)
    assert len(result["cases"]) == 1
    assert {reason["stratum"] for reason in result["cases"][0]["selection_reasons"]} == {"null"}


def test_measurement_folder_does_not_mix_endpoints_or_final_only_data(tmp_path):
    states = [record()["state"], record("state1")["state"]]
    root = tmp_path / "measurements" / "S_STD"
    identities = [("step_00015000/main/train", "wrong", "train", ["state0", "state1"]),
                  ("step_00030000/main/train", "wanted", "train", ["state0"]),
                  ("step_00030000/main/both", "wanted", "both", ["state0", "state1"]),
                  ("step_00030000/main/validation", "wanted", "validation", ["state0", "state1"])]
    for location, checkpoint, partition, present in identities:
        folder = root / location
        folder.mkdir(parents=True)
        (folder / "identity.json").write_text(json.dumps({"mode": "main", "experiment_id": "e",
                                                         "checkpoint_sha256": checkpoint, "partition": partition}))
        for state_id in present:
            (folder / f"{state_id}.json").write_text("{}")
    folder, identity = _measurement_folder(tmp_path, "S_STD", "main", "wanted", "e", states)
    assert folder.name == "both" and identity["checkpoint_sha256"] == "wanted"
    with pytest.raises(ValueError, match="No matching"):
        _measurement_folder(tmp_path, "S_PACKET", "main", "wanted", "e", states)


def test_generated_arrays_are_compact_checksum_bound_and_plan_seals_inputs(tmp_path):
    plan = _seal({"cases": ["a"], "source": "source", "student_hash": "a" * 64})
    assert _seal(plan) == plan
    assert _seal({**plan, "student_hash": "b" * 64})["plan_id"] != plan["plan_id"]
    result = {"contexts": {"base": {"normalized_actions": np.zeros((2, 16, 8)),
                                     "tcp_paths": np.zeros((2, 16, 3)), "expert_full": [1.0, 2.0]}},
              "precision": "fp32", "seeds": [1, 2]}
    path = tmp_path / "state.npz"
    receipt = _write_result(path, result, {"plan_id": plan["plan_id"]})
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"base/normalized_actions", "base/tcp_paths"}
    assert receipt["metrics"]["base"]["expert_full"] == [1.0, 2.0]
    assert len(receipt["arrays_sha256"]) == 64


def test_explicit_asset_state_retains_partition_and_cannot_choose_later_favorable_state():
    first, later = record(), record("later")
    first["state"].update(partition="validation", episode_index=0, decision_index=36)
    later["state"].update(partition="validation", episode_index=0, decision_index=72)
    manifest = {"kind": "limited_real_data_diagnostic", "states": [later["state"], first["state"]]}
    selected = diagnostic_state(manifest, "state0", limited_check=True, max_cases=1)
    assert selected == first["state"] and selected["partition"] == "validation"
    with pytest.raises(ValueError, match="gain-blind first"):
        diagnostic_state(manifest, "later", limited_check=True, max_cases=1)
    for altered, limited, maximum in (({**manifest, "kind": "primary_study"}, True, 1),
                                      (manifest, False, 1), (manifest, True, 2)):
        with pytest.raises(ValueError, match="asset-only"):
            diagnostic_state(altered, "state0", limited_check=limited, max_cases=maximum)
    assert not select_common_cases({"S0": [first]}, seed=0)["cases"]
    result = select_diagnostic_case({"S0": [first]}, selected, epsilon=1e-6)
    assert result["cases"][0]["reader"] == "TREE" and result["cases"][0]["background"] == 120
    assert result["selection_audit"]["original_partition"] == "validation"
    for row in first["rows"]:
        if row["bank"] == "confirmation":
            row["expert_full"] = -1e10
    assert result == select_diagnostic_case({"S0": [first]}, selected, epsilon=1e-6)
    with pytest.raises(ValueError, match="exactly"):
        select_diagnostic_case({"S0": [first, later]}, selected, epsilon=1e-6)


def test_generated_endpoint_supports_matched_continuations_without_mixing_arms():
    identities = {name: {"initialization": "S0_weights", "endpoint": 10000, "step": 10000}
                  for name in ("S_STD", "S_STAGE", "S_PACKET")}
    identities["S0"] = {"initialization": "production_reference", "endpoint": 90000, "step": 90000}
    validate_generated_endpoints(identities, limited_check=False)
    changed = deepcopy(identities)
    changed["S_STAGE"].update(initialization="accepted_teacher_weights", endpoint=30000, step=30000)
    with pytest.raises(ValueError, match="initializations"):
        validate_generated_endpoints(changed, limited_check=False)
    changed = deepcopy(identities)
    changed["S_STAGE"]["step"] = 5000
    with pytest.raises(ValueError, match="predeclared"):
        validate_generated_endpoints(changed, limited_check=False)
    with pytest.raises(ValueError, match="observed"):
        validate_generated_endpoints(changed, limited_check=True)


def test_generated_endpoint_accepts_declared_100k_but_never_mixes_30k_and_100k():
    identities = {name: {"initialization": "accepted_teacher_weights", "endpoint": 100000, "step": 100000}
                  for name in ("S_STD", "S_STAGE", "S_PACKET")}
    validate_generated_endpoints(identities, limited_check=False)
    changed = deepcopy(identities)
    changed["S_STAGE"]["step"] = 30000
    with pytest.raises(ValueError, match="predeclared"):
        validate_generated_endpoints(changed, limited_check=False)
    with pytest.raises(ValueError, match="observed"):
        validate_generated_endpoints(changed, limited_check=True)
    changed["S_STAGE"]["endpoint"] = 30000
    with pytest.raises(ValueError, match="declared endpoints"):
        validate_generated_endpoints(changed, limited_check=False)
