from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stmem.inspection.history_observer import HistoryObserver, reset_fingerprints


def frame(source="execution", index=0):
    proprio = torch.zeros(15)
    proprio[7:10] = torch.tensor([0., 0., .5])
    return SimpleNamespace(points=torch.zeros(4, 6), valid=torch.ones(4, dtype=torch.bool),
                           proprio=proprio, previous_action=torch.zeros(8), action_available=False,
                           source=source, source_index=index, time=index / 20,
                           event_id="fresh-uuid", center_indices={2: torch.tensor([0, 1])},
                           visual_features=torch.ones(4, 256, dtype=torch.bfloat16),
                           point_camera=torch.zeros(4, dtype=torch.long), point_uv=torch.zeros(4, 2, dtype=torch.long),
                           visual_feature_signature="features", center_sampling_signature="centers")


def example(task="VideoRepick"):
    actors = [SimpleNamespace(name=f"object_{i}", pose=SimpleNamespace(p=np.array([i * .1, 0., .03])),
                              grasped=False) for i in range(3)]
    env = SimpleNamespace(seed=12, difficulty="easy", current_task_index=2, spawned_cubes=actors,
                          spawned_bins=actors, all_cubes=actors, targets=actors, bin_0=actors[0],
                          target_cube_1=actors[1], target_cube=actors[1], target_target=actors[2],
                          num_repeats=2, target_color_name="blue", which_in_subset=2,
                          configs={"easy": {"pick": 1}}, color_names=["blue", "green", "red"],
                          agent=SimpleNamespace(is_grasping=lambda actor: actor.grasped),
                          evaluate=lambda: pytest.fail("No privileged task evaluation allowed"))
    reset = SimpleNamespace(goal="watch and repick", current=frame(), conditioning=[frame("conditioning")])
    step = SimpleNamespace(done=False, frames=[frame(index=2)], command=np.zeros(8))
    adapter = SimpleNamespace(task=task, env=env, reset=lambda: reset, step=lambda action: step)
    return adapter, env, actors, reset, step


def test_fingerprints_preserve_values_and_ignore_only_event_uuid():
    *_, reset, _ = example()
    copy = deepcopy(reset)
    copy.current.event_id = "another-reset"
    copy.conditioning[0].event_id = "another-video"
    original = reset_fingerprints(reset)
    assert reset_fingerprints(copy) == original
    copy.goal += " twice"
    assert reset_fingerprints(copy)["reset_sha256"] != original["reset_sha256"]
    copy.goal = reset.goal
    copy.current.center_indices[2][0] = 2
    assert reset_fingerprints(copy)["current"]["sha256"] != original["current"]["sha256"]
    copy = deepcopy(reset)
    copy.conditioning[0].visual_features[0, 0] = 2
    assert reset_fingerprints(copy)["conditioning"]["sha256"] != original["conditioning"]["sha256"]


@pytest.mark.parametrize("task,index", [("VideoRepick", 1), ("VideoUnmask", 0),
                                      ("VideoPlaceOrder", 2), ("SwingXtimes", 1)])
def test_identity_and_exact_task_targets(task, index):
    adapter, env, actors, reset, step = example(task)
    observer = HistoryObserver(adapter)
    original = reset_fingerprints(reset)
    before = [set(vars(item)) for item in (reset, reset.current, step)]
    assert observer.reset() is reset
    assert observer.step(np.zeros(8)) is step
    assert [set(vars(item)) for item in (reset, reset.current, step)] == before
    assert reset_fingerprints(reset) == original
    result = observer.report("timeout")
    assert result["status"] == "complete" and result["errors"] == []
    assert result["target"]["first_index"] == index
    assert result["input_fingerprints"] == original
    assert result["goal"] == reset.goal
    if task == "VideoPlaceOrder":
        assert all("grasped" not in a for a in result["reset"]["candidates"])
        assert result["reset"]["controlled_object"]["name"] == actors[1].name


def test_contact_lift_and_terminal_exclusion():
    adapter, env, actors, reset, step = example()
    observer = HistoryObserver(adapter)
    observer.reset()
    actors[1].grasped = True
    observer.step(np.zeros(8))
    first = observer.report("ongoing")["summary"]
    assert first["first_grasp_indices"] == [1] and first["first_grasp_correct"]
    assert first["first_lift_indices"] is None
    actors[1].pose.p[2] = .1
    step.done = True
    observer.step(np.zeros(8))
    result = observer.report("success")
    assert result["summary"]["first_lift_indices"] is None
    assert result["summary"]["official_task_success"]
    assert result["timeline"][-1]["excluded_reason"] == "terminal_auto_step"


def test_diagnostic_error_does_not_change_outcome_or_input():
    adapter, env, actors, reset, step = example()
    env.target_cube_1 = object()
    observer = HistoryObserver(adapter)
    assert observer.reset() is reset
    assert observer.step(np.zeros(8)) is step
    result = observer.report("success")
    assert result["status"] == "diagnostic_error"
    assert result["target"] is None
    assert result["input_fingerprints"] is not None
    assert result["summary"]["official_task_success"]
