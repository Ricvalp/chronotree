"""Reporting-only VideoUnmaskSwap choice observations must not affect rollouts."""
from types import SimpleNamespace

import numpy as np
import pytest

from stmem.inspection.swap_targeting import SwapTargetingAdapter


def frame(tcp, index=0):
    proprio = np.zeros(15, dtype=np.float32)
    proprio[7:10] = tcp
    return SimpleNamespace(proprio=proprio, source_index=index)


@pytest.fixture
def example():
    bins = [SimpleNamespace(name=f"bin_{i}", pose=SimpleNamespace(p=np.array([x, 0, .05])), grasped=False)
            for i, x in enumerate([-.2, 0., .2])]
    env = SimpleNamespace(spawned_bins=bins, selected_bins=[bins[2], bins[0]], pick_times=2,
                          color_names=["blue", "red", "green"], target_bin=bins[0],
                          seed=6000, difficulty="hard", current_task_index=1,
                          agent=SimpleNamespace(is_grasping=lambda actor: actor.grasped),
                          evaluate=lambda: pytest.fail("Observer must not call environment.evaluate"))
    reset = SimpleNamespace(current=frame([-.5, 0, .5]), metadata={"control_hz": 20})
    step = SimpleNamespace(done=False, frames=[frame([.2, 0, .25], 2)], command=np.zeros(8), observation_skipped=False)
    adapter = SimpleNamespace(task="VideoUnmaskSwap", env=env, reset=lambda: reset, step=lambda _action: step)
    return adapter, env, bins, reset, step


def test_true_requested_actor_and_transparent_inputs(example):
    adapter, env, bins, reset, step = example
    observer = SwapTargetingAdapter(adapter)
    reset_keys, frame_keys = set(vars(reset)), set(vars(reset.current))
    assert observer.reset() is reset
    assert set(vars(reset)) == reset_keys and set(vars(reset.current)) == frame_keys
    assert observer.target == {"first_bin_index": 2, "color": "blue", "ordered_bin_indices": [2, 0],
                               "ordered_colors": ["blue", "red"]}
    assert observer.report("running")["summary"]["first_approach_bin_index"] is None
    step_keys = set(vars(step))
    assert observer.step(np.zeros(8)) is step
    assert set(vars(step)) == step_keys
    report = observer.report("timeout")
    assert report["status"] == "complete"
    assert report["summary"]["first_approach_bin_index"] == 2
    assert report["summary"]["first_approach_command"] == 1
    assert report["summary"]["first_grasp_bin_indices"] is None
    assert report["summary"]["first_approach_margin_m"] == pytest.approx(.2)
    assert env.target_bin is bins[0]  # Wrong alternate attribute was neither used nor changed.


def test_contact_and_lift_distinguish_wrong_choice_from_proximity(example):
    adapter, env, bins, _, step = example
    observer = SwapTargetingAdapter(adapter)
    observer.reset()
    bins[1].grasped = True
    step.frames = [frame([0, 0, .1], 2)]
    observer.step(np.zeros(8))
    bins[1].pose.p[2] = .16
    step.frames = [frame([0, 0, .2], 4)]
    observer.step(np.zeros(8))
    report = observer.report("fail")
    summary = report["summary"]
    assert summary["first_grasp_bin_indices"] == [1]
    assert summary["first_lift_bin_indices"] == [1]
    assert summary["first_grasp_correct"] is False and summary["first_lift_correct"] is False
    assert summary["any_correct_grasp"] is False and summary["any_wrong_grasp"] is True


def test_height_without_contact_is_not_lift_and_terminal_is_excluded(example):
    adapter, _, bins, _, step = example
    observer = SwapTargetingAdapter(adapter)
    observer.reset()
    bins[2].pose.p[2] = .2
    observer.step(np.zeros(8))
    assert observer.report("running")["summary"]["first_lift_bin_indices"] is None
    bins[2].grasped = True
    step.done = True
    observer.step(np.zeros(8))
    report = observer.report("success")
    assert report["summary"]["official_task_success"] is True
    assert report["summary"]["first_lift_bin_indices"] is None
    assert report["timeline"][-1]["excluded_reason"] == "terminal_auto_step"


def test_reset_proximity_and_skipped_frames_are_not_choice_events(example):
    adapter, _, _, reset, step = example
    reset.current = frame([.2, 0, .2])
    observer = SwapTargetingAdapter(adapter)
    observer.reset()
    summary = observer.report("running")["summary"]
    assert summary["initial_near_bin_index"] == 2
    assert summary["first_approach_bin_index"] is None
    step.frames = []
    step.observation_skipped = True
    observer.step(np.zeros(8))
    assert observer.report("running")["summary"]["first_approach_bin_index"] is None
    step.frames = [frame([.2, 0, .2], 2)]
    observer.step(np.zeros(8))
    summary = observer.report("running")["summary"]
    assert summary["first_approach_command"] == 2
    assert summary["first_approach_already_near_at_reset"] is True


def test_observer_errors_never_replace_policy_result(example):
    adapter, env, _, reset, step = example
    env.selected_bins = []
    observer = SwapTargetingAdapter(adapter)
    assert observer.reset() is reset
    assert observer.step(np.zeros(8)) is step
    report = observer.report("success")
    assert report["status"] == "diagnostic_error"
    assert report["summary"]["official_task_success"] is True
    assert report["target"] is None
