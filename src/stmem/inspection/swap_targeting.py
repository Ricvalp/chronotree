"""Read-only container-choice observations for normal VideoUnmaskSwap rollouts.

Privileged actor poses/contact information are recorded after the policy action,
never attached to Frame, ResetResult, StepResult, or any policy input. No task
predicate or environment evaluation method is called by this observer.
"""
from __future__ import annotations

import time

import numpy as np

from stmem.evaluation.robomme_adapter import numpy_value


VERSION = "videounmaskswap_container_choice_v1"
DEFINITIONS = {
    "scope": "Ordinary autonomous VideoUnmaskSwap; privileged observations are reporting only.",
    "target": "First requested actor is selected_bins[0], not the unrelated env.target_bin attribute. Optional second pickup is selected_bins[1].",
    "bin_index": "Zero-based index in spawned_bins, hence creation identity, not current left-to-right rank.",
    "approach": "First nonterminal returned observation with TCP z <=0.30m and nearest container XY distance <=0.08m. This is a proximity proxy, not a grasp or successful choice.",
    "approach_margin": "Second-nearest minus nearest container XY distance; small margin means ambiguous proximity.",
    "grasp": "Pinned agent.is_grasping contact test on each container; simultaneous contacts are retained as a list.",
    "lift": "Container origin z >0.15m AND contact grasp in the same active observation, matching the pinned pickup height threshold.",
    "reset": "Container poses are measured after the unchanged conditioning video and its swaps. Reset proximity/contact is reported separately, never counted as an autonomous step event.",
    "terminal": "Terminal observations are excluded because the wrapper may execute an extra automatic terminal step. Brief or terminal-only contact events can be missed.",
    "correctness": "First-contact/lift correctness refers to the first requested container. Contacting another requested container later may be legitimate; any_wrong_grasp is a descriptive first-target comparison.",
    "success": "Official terminal status remains authoritative. Choice diagnostics never replace it.",
}


def _scalar(value):
    values = numpy_value(value).reshape(-1)
    if len(values) != 1:
        raise ValueError("Container-choice diagnostics require one simulator environment")
    return values[0].item()


def _xyz(value):
    values = numpy_value(value).astype(np.float64).reshape(-1, 3)
    if values.shape != (1, 3) or not np.isfinite(values).all():
        raise ValueError("Expected one finite world XYZ value")
    return values[0]


class SwapTargetingAdapter:
    """Transparent adapter wrapper with independent diagnostic bookkeeping."""

    def __init__(self, adapter):
        self.adapter = adapter
        if adapter.task != "VideoUnmaskSwap":
            raise ValueError("This observer is specific to VideoUnmaskSwap")
        self.rows, self.errors = [], []
        self.reset_row, self.target = None, None
        self.episode, self.commands, self.seconds = {}, 0, 0.0

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def _environment(self):
        return getattr(self.adapter.env, "unwrapped", self.adapter.env)

    def _identify(self):
        env = self._environment()
        bins = list(env.spawned_bins)
        intended = list(env.selected_bins)
        count = int(getattr(env, "pick_times", 1))
        if not bins or not intended or not 1 <= count <= len(intended):
            raise ValueError("Missing actual requested container sequence")
        indices = []
        for actor in intended[:count]:
            matches = [i for i, candidate in enumerate(bins) if candidate is actor]
            if len(matches) != 1:
                raise ValueError("Requested container is not uniquely present in spawned_bins")
            indices.append(matches[0])
        colors = [str(value) for value in env.color_names[:count]]
        if len(colors) != count:
            raise ValueError("Requested color sequence does not match requested containers")
        self.target = {"first_bin_index": indices[0], "color": colors[0],
                       "ordered_bin_indices": indices, "ordered_colors": colors}
        self.episode = {"seed": int(_scalar(env.seed)), "difficulty": str(env.difficulty)}

    def _observe(self, frame, command, *, reset=False):
        started = time.perf_counter()
        try:
            env = self._environment()
            if reset:
                self._identify()
            tcp = _xyz(numpy_value(frame.proprio)[7:10])
            bins = []
            for index, actor in enumerate(env.spawned_bins):
                xyz = _xyz(actor.pose.p)
                grasped = bool(_scalar(env.agent.is_grasping(actor)))
                bins.append({"index": index, "name": str(getattr(actor, "name", f"bin_{index}")),
                             "xyz_m": xyz.tolist(), "grasped": grasped,
                             "above_threshold": bool(xyz[2] > .15),
                             "grasped_and_lifted": bool(grasped and xyz[2] > .15)})
            distances = np.asarray([np.linalg.norm(tcp[:2] - np.asarray(b["xyz_m"][:2])) for b in bins])
            order = np.argsort(distances, kind="stable")
            nearest = int(order[0])
            row = {
                "command_index": self.commands, "source_index": frame.source_index,
                "tcp_xyz_m": tcp.tolist(), "bins": bins,
                "nearest_bin_index": nearest, "nearest_xy_m": float(distances[nearest]),
                "nearest_margin_m": None if len(order) < 2 else float(distances[order[1]] - distances[nearest]),
                "near_container": bool(tcp[2] <= .30 and distances[nearest] <= .08),
            }
            if command is not None:
                row["command"] = np.asarray(command, dtype=np.float64).tolist()
            if hasattr(env, "current_task_index"):
                row["current_task_index"] = int(_scalar(env.current_task_index))
            if reset:
                self.reset_row = row
            else:
                self.rows.append(row)
        except Exception as exc:
            self.errors.append({"command_index": self.commands, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self.seconds += time.perf_counter() - started

    def reset(self):
        result = self.adapter.reset()
        self.rows, self.errors = [], []
        self.reset_row, self.target = None, None
        self.episode, self.commands, self.seconds = {}, 0, 0.0
        self._observe(result.current, None, reset=True)
        return result

    def step(self, action):
        result = self.adapter.step(action)
        self.commands += 1
        if result.done or not result.frames:
            self.rows.append({"command_index": self.commands,
                              "excluded_reason": "terminal_auto_step" if result.done else
                                  "temporal_subsampling" if getattr(result, "observation_skipped", False) else
                                  "missing_observation"})
        else:
            self._observe(result.frames[-1], result.command)
        return result

    def report(self, official_status):
        active = [r for r in self.rows if "excluded_reason" not in r]
        first_near = next((r for r in active if r["near_container"]), None)
        first_grasp = next((r for r in active if any(b["grasped"] for b in r["bins"])), None)
        first_lift = next((r for r in active if any(b["grasped_and_lifted"] for b in r["bins"])), None)
        target = None if self.target is None else self.target["first_bin_index"]
        grasp_ids = None if first_grasp is None else [b["index"] for b in first_grasp["bins"] if b["grasped"]]
        lift_ids = None if first_lift is None else [b["index"] for b in first_lift["bins"] if b["grasped_and_lifted"]]
        reset_near = None if self.reset_row is None or not self.reset_row["near_container"] else self.reset_row["nearest_bin_index"]
        summary = {
            "official_status": official_status, "official_task_success": official_status == "success",
            "issued_commands": self.commands, "measured_active_commands": len(active),
            "initial_near_bin_index": reset_near,
            "first_approach_bin_index": None if first_near is None else first_near["nearest_bin_index"],
            "first_approach_command": None if first_near is None else first_near["command_index"],
            "first_approach_xy_m": None if first_near is None else first_near["nearest_xy_m"],
            "first_approach_margin_m": None if first_near is None else first_near["nearest_margin_m"],
            "first_approach_already_near_at_reset": None if first_near is None else reset_near == first_near["nearest_bin_index"],
            "first_grasp_bin_indices": grasp_ids, "first_grasp_command": None if first_grasp is None else first_grasp["command_index"],
            "first_lift_bin_indices": lift_ids, "first_lift_command": None if first_lift is None else first_lift["command_index"],
            "first_grasp_correct": None if target is None or grasp_ids is None else grasp_ids == [target],
            "first_lift_correct": None if target is None or lift_ids is None else lift_ids == [target],
            "any_correct_grasp": None if target is None or not active else any(b["grasped"] and b["index"] == target for r in active for b in r["bins"]),
            "any_wrong_grasp": None if target is None or not active else any(b["grasped"] and b["index"] != target for r in active for b in r["bins"]),
            "any_correct_lift": None if target is None or not active else any(b["grasped_and_lifted"] and b["index"] == target for r in active for b in r["bins"]),
        }
        if self.reset_row is not None and "current_task_index" in self.reset_row:
            indices = [r["current_task_index"] for r in active if "current_task_index" in r]
            summary["completed_subgoal_index_increase"] = None if not indices else max(0, max(indices) - self.reset_row["current_task_index"])
        return {"version": VERSION, "status": "diagnostic_error" if self.errors else "not_observed" if self.reset_row is None else "complete",
                "definitions": DEFINITIONS, "episode": self.episode, "target": self.target,
                "reset": self.reset_row, "timeline": self.rows, "summary": summary,
                "errors": self.errors, "observer_seconds": self.seconds}
