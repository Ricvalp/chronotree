"""Synthetic read-only simulator instrumentation checks, not policy outcomes."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stmem.data.dataset import synthetic_episode
from stmem.evaluation.robomme_adapter import ResetResult, StepResult
from stmem.evaluation.targeting import TargetingAdapter, summarize_targeting
from stmem.visualization.kinematics import panda_tcp_positions


COMMAND = np.array([0, 0, 0, -1.5, 0, 1.5, 0.7, -1], dtype=np.float32)


class FixtureAdapter:
    def __init__(self, task="PickXtimes", *, initial_grasp=False, error=False):
        self.task = task
        self.frames = synthetic_episode(5, 0, points=4).execution
        self.target = SimpleNamespace(pose=SimpleNamespace(p=torch.tensor([[0.0, 0.0, 0.02]])))
        self.grasped = initial_grasp
        self.error = error
        self.index = 0
        self.closed = False
        self.contact_queries = 0
        self.env = SimpleNamespace(
            target_cube=self.target,
            bin_0=self.target,
            agent=SimpleNamespace(is_grasping=self.is_grasping),
            current_task_index=0,
            evaluate=lambda: pytest.fail("Diagnostics must not invoke task evaluation"),
        )
        self.env.unwrapped = self.env
        self.reset_result = ResetResult([], self.frames[0], "fixture")
        self.step_results = []

    def is_grasping(self, target):
        assert target is self.target
        self.contact_queries += 1
        if self.error:
            raise RuntimeError("contact API unavailable")
        return torch.tensor([self.grasped])

    def reset(self):
        return self.reset_result

    def step(self, action):
        assert np.array_equal(action, COMMAND)
        self.index += 1
        frame = self.frames[self.index]
        commanded = panda_tcp_positions(COMMAND[:7], task=self.task)
        # A known post-command 5mm tracking difference; unrelated to target pose.
        frame.proprio[7:10] = torch.tensor(commanded + [0.005, 0, 0], dtype=torch.float32)
        height = 0.08 if self.task == "PickXtimes" else 0.20
        self.target.pose.p = torch.tensor([[0.0, 0.0, 0.02 if self.index == 1 else height]])
        self.grasped = self.index >= 2
        self.env.current_task_index = 1 if self.index >= 2 else 0
        result = StepResult(
            [frame], "success" if self.index == 3 else "ongoing", self.index == 3, False, command=action
        )
        self.step_results.append(result)
        return result

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "task,target,threshold",
    [
        ("PickXtimes", "target_cube", 0.05),
        ("VideoUnmask", "bin_0", 0.15),
    ],
)
def test_targeting_metrics_and_observation_command_transparency(task, target, threshold):
    original = FixtureAdapter(task)
    adapter = TargetingAdapter(original)
    reset = adapter.reset()
    assert reset is original.reset_result
    fields = set(vars(reset.current))
    for index in range(3):
        outcome = adapter.step(COMMAND)
        assert outcome is original.step_results[index]
        assert outcome.command is COMMAND
        assert set(vars(outcome.frames[0])) == fields
    adapter.close()
    assert original.closed
    result = adapter.report("success")
    assert result["status"] == "complete"
    assert result["target_actor"] == target
    assert result["target_lift_height_threshold_m"] == threshold
    summary = result["summary"]
    assert summary["official_task_success"] is True
    assert summary["first_close_command"] == 1
    assert summary["first_grasp_command"] == summary["first_new_grasp_command"] == 2
    assert summary["first_grasped_lift_command"] == 2
    assert summary["autonomous_first_grasp_observed"] is True
    assert summary["any_grasped_target_lift_after_reset"] is True
    assert summary["measured_active_commands"] == 2
    assert summary["issued_commands"] == 3
    assert summary["completed_subgoal_index_increase"] == 1
    assert summary["command_tracking_error_mean_m"] == pytest.approx(0.005, abs=1e-7)
    assert result["timeline"][-1]["excluded_reason"] == "terminal_auto_step"
    assert "target_xyz_m" not in result["timeline"][-1]
    # Reset + two nonterminal observations: terminal auto-step is not queried.
    assert original.contact_queries == 3


def test_no_grasp_inference_from_terminal_outcome_or_existing_reset_grasp():
    original = FixtureAdapter(initial_grasp=True)
    adapter = TargetingAdapter(original)
    adapter.reset()
    # One closing command without a grasp. Terminal success below cannot be
    # silently interpreted as an observed contact or lift in this diagnostic.
    adapter.step(COMMAND)
    result = adapter.report("success")
    assert result["summary"]["target_grasped_at_reset"] is True
    assert result["summary"]["official_task_success"] is True
    assert result["summary"]["autonomous_first_grasp_observed"] is False
    assert result["summary"]["any_grasped_target_lift_after_reset"] is False


def test_diagnostic_error_does_not_replace_official_outcome():
    original = FixtureAdapter(error=True)
    adapter = TargetingAdapter(original)
    assert adapter.reset() is original.reset_result
    outcome = adapter.step(COMMAND)
    assert outcome.status == "ongoing"
    result = adapter.report("timeout")
    assert result["status"] == "diagnostic_error"
    assert result["errors"][0]["error"] == "RuntimeError: contact API unavailable"
    assert result["summary"]["official_status"] == "timeout"
    assert result["summary"]["any_target_grasp_after_reset"] is None


def test_unsupported_tasks_have_no_oracle_reads_and_summary_keeps_denominators():
    original = FixtureAdapter()
    original.task = "PatternLock"
    adapter = TargetingAdapter(original)
    adapter.reset()
    assert original.contact_queries == 0
    report = adapter.report("success")
    assert report["status"] == "not_applicable"
    records = [
        {"task": "PatternLock", "episode_idx": 0, "status": "success", "targeting_diagnostics": report},
        {
            "task": "PickXtimes",
            "episode_idx": 0,
            "status": "timeout",
            "targeting_diagnostics": {"status": "diagnostic_error", "errors": ["fixture"], "summary": {}},
        },
    ]
    summary = summarize_targeting(records)
    assert summary["tasks"]["PatternLock"]["not_applicable"] == 1
    assert summary["tasks"]["PickXtimes"]["diagnostics_errors"] == 1
    assert summary["tasks"]["PickXtimes"]["autonomous_first_grasp_observed"] == {"count": 0, "measured": 0}
