"""CPU-only synthetic identity/split checks; no production training claims."""
from copy import deepcopy

import pytest

from stmem.config import digest
from stmem.training.utility_router_study import validate_pool, validate_states


def state(task, episode, index, partition):
    identity = {"task": task, "episode_id": episode, "decision_index": index}
    return {**identity, "state_id": digest(identity)[:20], "episode_index": 0, "partition": partition}


def parent():
    return {
        "configuration": {"data": {"tasks": ["A", "B"]}},
        "training_pool": [{"task": t, "episode_id": "episode_0"} for t in ("A", "B")],
        "diagnostic_episodes": [{"task": t, "episode_id": "episode_1"} for t in ("A", "B")],
        "reserved_validation_episodes": [{"task": t, "episode_id": "episode_2"} for t in ("A", "B")],
        "episode_partitions": [{"task": t, "episode_id": f"episode_{i}", "partition": "validation" if i == 2 else "train"}
                               for t in ("A", "B") for i in range(3)],
        "states": [state(t, "episode_1", 10, "train") for t in ("A", "B")]
                  + [state(t, "episode_2", 10, "validation") for t in ("A", "B")],
    }


def valid_states():
    return [state(t, "episode_0", i, "fit") for t in ("A", "B") for i in (10, 20)] + [
        {**s, "partition": "development"} for s in parent()["states"] if s["partition"] == "train"]


def test_fit_episode_boundaries_and_development_panel_are_independent():
    validate_states(parent(), valid_states(), fit_states=4)
    for forbidden in ("episode_1", "episode_2"):
        changed = valid_states()
        changed[0] = state("A", forbidden, 20, "fit")
        with pytest.raises(ValueError, match="leaked"):
            validate_states(parent(), changed, fit_states=4)
    changed = valid_states()
    changed[-1] = state("B", "episode_2", 10, "development")
    with pytest.raises(ValueError, match="fixed development"):
        validate_states(parent(), changed, fit_states=4)


def test_duplicate_missing_and_unbalanced_states_cannot_complete_a_pilot():
    changed = valid_states()
    changed[1] = deepcopy(changed[0])
    with pytest.raises(ValueError, match="duplicated"):
        validate_states(parent(), changed, fit_states=4)
    with pytest.raises(ValueError, match="count"):
        validate_states(parent(), valid_states()[:-1], fit_states=4)
    changed = valid_states()
    changed[0] = state("B", "episode_0", 30, "fit")
    with pytest.raises(ValueError, match="balanced"):
        validate_states(parent(), changed, fit_states=4)
    changed = parent()
    changed["training_pool"].append({"task": "A", "episode_id": "episode_2"})
    with pytest.raises(ValueError, match="excluded"):
        validate_pool(changed)


def test_relabeling_confirmation_or_changing_state_identity_is_rejected():
    changed = valid_states()
    changed[0]["decision_index"] += 2
    with pytest.raises(ValueError, match="identity"):
        validate_states(parent(), changed, fit_states=4)
    changed = valid_states()
    changed[-1]["partition"] = "validation"
    with pytest.raises(ValueError, match="Final confirmation"):
        validate_states(parent(), changed, fit_states=4)
