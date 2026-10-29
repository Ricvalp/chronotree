"""Evaluation-only target/contact measurements, kept outside policy inputs.

These read-only diagnostics never call the environment's evaluate/task functions
or alter observations, commands, rewards, success flags or subgoal progression.
They use privileged object poses exclusively for reporting, not for the policy.
"""

from __future__ import annotations

from collections import defaultdict
import time

import numpy as np

from stmem.evaluation.robomme_adapter import numpy_value
from stmem.visualization.kinematics import panda_tcp_positions


TARGETS = {"PickXtimes": ("target_cube", 0.05), "VideoUnmask": ("bin_0", 0.15)}
VERSION = "autonomous_targeting_v1"
DEFINITIONS = {
    "scope": "Autonomous episode from reset; privileged metrics never enter Frame or policy context.",
    "target": "PickXtimes: target_cube. VideoUnmask: first requested container bin_0, fixed throughout episode.",
    "grasp": "Pinned ManiSkill agent.is_grasping: both finger contact forces >=0.5N and angles <=85 degrees.",
    "first_close": "First issued physical gripper command <0; distances use its returned nonterminal observation.",
    "grasp_distance": "Observed world TCP to target actor origin, XY and 3D Euclidean distances; not surface/contact gap.",
    "height_threshold": "Target actor origin world z >0.05m for PickXtimes, >0.15m for VideoUnmask, matching pinned task checks.",
    "lift": "Height threshold AND contact grasp observed together after reset; height-only crossing is reported separately.",
    "tracking": "FK of the clipped issued absolute joint target versus observed post-command TCP; includes controller lag/dynamics.",
    "partial_progress": "Read-only change in current_task_index since reset; no subgoal evaluation function is called.",
    "terminal": "Terminal wrapper may execute an extra automatic step: its state is recorded as excluded and never used for targeting aggregates.",
    "limitations": "First container only for multi-container VideoUnmask. Brief contact and final terminal-only events may be missed at observation cadence. These are diagnostics, not replacement official task success.",
}


def _scalar(value):
    array = numpy_value(value).reshape(-1)
    if len(array) != 1:
        raise ValueError("Target diagnostics require one simulator environment")
    return array[0].item()


class TargetingAdapter:
    """Transparent adapter observer; the exact ResetResult/StepResult is returned."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.task = adapter.task
        self.rows = []
        self.reset_row = None
        self.errors = []
        self.seconds = 0.0
        self.commands = 0
        self.applicable = self.task in TARGETS

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def _observe(self, frame, command, *, reset=False):
        started = time.perf_counter()
        try:
            env = getattr(self.adapter.env, "unwrapped", self.adapter.env)
            attribute, threshold = TARGETS[self.task]
            target = getattr(env, attribute)
            xyz = numpy_value(target.pose.p).astype(np.float64).reshape(-1, 3)
            if xyz.shape != (1, 3):
                raise ValueError("Target diagnostics require exactly one target instance")
            xyz = xyz[0]
            tcp = numpy_value(frame.proprio)[7:10].astype(np.float64)
            if not np.isfinite(xyz).all() or not np.isfinite(tcp).all():
                raise ValueError("Nonfinite simulator target/TCP position")
            grasped = bool(_scalar(env.agent.is_grasping(target)))
            row = {
                "command_index": self.commands,
                "source_index": frame.source_index,
                "observation_available": True,
                "terminal": False,
                "target_xyz_m": xyz.tolist(),
                "observed_tcp_xyz_m": tcp.tolist(),
                "tcp_target_xy_error_m": float(np.linalg.norm(tcp[:2] - xyz[:2])),
                "tcp_target_error_m": float(np.linalg.norm(tcp - xyz)),
                "target_grasped": grasped,
                "target_height_m": float(xyz[2]),
                "target_above_threshold": bool(xyz[2] > threshold),
                "target_grasped_and_lifted": bool(grasped and xyz[2] > threshold),
            }
            index = getattr(env, "current_task_index", None)
            if index is not None:
                row["current_task_index"] = int(_scalar(index))
            if command is not None:
                command = np.asarray(command, dtype=np.float64)
                row["command"] = command.tolist()
                commanded = panda_tcp_positions(command[:7], task=self.task)
                row["commanded_tcp_xyz_m"] = commanded.tolist()
                row["commanded_tcp_target_xy_error_m"] = float(np.linalg.norm(commanded[:2] - xyz[:2]))
                row["commanded_tcp_target_error_m"] = float(np.linalg.norm(commanded - xyz))
                row["command_tracking_error_m"] = float(np.linalg.norm(commanded - tcp))
            if reset:
                self.reset_row = row
            else:
                self.rows.append(row)
        except Exception as exc:
            # A missing diagnostic actor/contact API must not turn an otherwise
            # valid official rollout into a policy or simulator failure.
            self.errors.append({"command_index": self.commands, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self.seconds += time.perf_counter() - started

    def reset(self):
        result = self.adapter.reset()
        self.rows, self.errors = [], []
        self.reset_row, self.commands, self.seconds = None, 0, 0.0
        if self.applicable:
            self._observe(result.current, None, reset=True)
        return result

    def step(self, action):
        result = self.adapter.step(action)
        self.commands += 1
        if self.applicable:
            if result.done or not result.frames:
                self.rows.append(
                    {
                        "command_index": self.commands,
                        "observation_available": bool(result.frames),
                        "terminal": bool(result.done),
                        "command": None if result.command is None else np.asarray(result.command).tolist(),
                        "excluded_reason": "terminal_auto_step"
                        if result.done
                        else "temporal_subsampling"
                        if getattr(result, "observation_skipped", False)
                        else "missing_observation",
                    }
                )
            else:
                self._observe(result.frames[-1], result.command)
        return result

    def report(self, official_status):
        if not self.applicable:
            return {
                "version": VERSION,
                "status": "not_applicable",
                "task": self.task,
                "reason": "Targeting diagnostics are defined only for PickXtimes and VideoUnmask",
            }
        attribute, threshold = TARGETS[self.task]
        active = [row for row in self.rows if not row.get("excluded_reason")]
        first_close = next((row for row in active if row.get("command", [0] * 8)[7] < 0), None)
        first_grasp = next((row for row in active if row["target_grasped"]), None)
        first_height = next((row for row in active if row["target_above_threshold"]), None)
        first_lift = next((row for row in active if row["target_grasped_and_lifted"]), None)
        reset = self.reset_row
        previous_grasp = None if reset is None else reset["target_grasped"]
        first_new_grasp = None
        for row in active:
            if previous_grasp is False and row["target_grasped"]:
                first_new_grasp = row
                break
            previous_grasp = row["target_grasped"]
        summary = {
            "official_task_success": official_status == "success",
            "official_status": official_status,
            "measured_active_commands": len(active),
            "issued_commands": self.commands,
            "target_grasped_at_reset": None if reset is None else reset["target_grasped"],
            "target_lifted_at_reset": None if reset is None else reset["target_grasped_and_lifted"],
            "any_target_grasp_after_reset": None if not active else first_grasp is not None,
            "autonomous_first_grasp_observed": None
            if not active or reset is None
            else first_new_grasp is not None,
            "any_target_above_threshold_after_reset": None if not active else first_height is not None,
            "any_grasped_target_lift_after_reset": None if not active else first_lift is not None,
            "first_close_command": None if first_close is None else first_close["command_index"],
            "first_grasp_command": None if first_grasp is None else first_grasp["command_index"],
            "first_new_grasp_command": None if first_new_grasp is None else first_new_grasp["command_index"],
            "first_height_threshold_command": None if first_height is None else first_height["command_index"],
            "first_grasped_lift_command": None if first_lift is None else first_lift["command_index"],
        }
        for name, row in (("first_close", first_close), ("first_grasp", first_grasp)):
            for metric in (
                "tcp_target_xy_error_m",
                "tcp_target_error_m",
                "commanded_tcp_target_xy_error_m",
                "command_tracking_error_m",
            ):
                summary[f"{name}_{metric}"] = None if row is None else row.get(metric)
        if active:
            errors = np.asarray(
                [row["command_tracking_error_m"] for row in active if "command_tracking_error_m" in row]
            )
            summary.update(
                min_tcp_target_xy_error_m=min(row["tcp_target_xy_error_m"] for row in active),
                min_tcp_target_error_m=min(row["tcp_target_error_m"] for row in active),
                max_target_height_m=max(row["target_height_m"] for row in active),
                command_tracking_error_mean_m=None if not len(errors) else float(errors.mean()),
                command_tracking_error_max_m=None if not len(errors) else float(errors.max()),
            )
            if reset is not None and "current_task_index" in reset:
                indices = [row["current_task_index"] for row in active if "current_task_index" in row]
                summary["completed_subgoal_index_increase"] = (
                    max(0, max(indices) - reset["current_task_index"]) if indices else None
                )
        return {
            "version": VERSION,
            "status": "diagnostic_error" if self.errors else "not_observed" if reset is None else "complete",
            "task": self.task,
            "target_actor": attribute,
            "target_lift_height_threshold_m": threshold,
            "definitions": DEFINITIONS,
            "summary": summary,
            "reset": reset,
            "timeline": self.rows,
            "errors": self.errors,
            "observer_seconds": self.seconds,
        }


def summarize_targeting(records):
    """Compact artifact, without additional live W&B scalar panels."""
    episodes, tasks = [], defaultdict(list)
    for result in records:
        targeting = result.get("targeting_diagnostics")
        if targeting is None:
            continue
        item = {
            "task": result["task"],
            "episode_idx": result["episode_idx"],
            "status": targeting["status"],
            "summary": targeting.get("summary"),
            "target_actor": targeting.get("target_actor"),
            "errors": targeting.get("errors", []),
        }
        episodes.append(item)
        tasks[item["task"]].append(item)
    grouped = {}
    for task, rows in sorted(tasks.items()):
        usable = [row["summary"] for row in rows if row["status"] == "complete" and row["summary"]]
        grouped[task] = {
            "recorded_episodes": len(rows),
            "diagnostics_complete": len(usable),
            "diagnostics_errors": sum(row["status"] == "diagnostic_error" for row in rows),
            "not_applicable": sum(row["status"] == "not_applicable" for row in rows),
        }
        for metric in ("autonomous_first_grasp_observed", "any_grasped_target_lift_after_reset"):
            measured = [row[metric] for row in usable if row[metric] is not None]
            grouped[task][metric] = {
                "count": sum(bool(value) for value in measured),
                "measured": len(measured),
            }
    return {"version": VERSION, "definitions": DEFINITIONS, "tasks": grouped, "episodes": episodes}
