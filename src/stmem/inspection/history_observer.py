"""Reporting-only reset fingerprints and task observations for history ablations.

Privileged simulator state is never added to the original adapter's return values.
Event UUIDs are excluded from fingerprints because equivalent simulator resets
receive different UUIDs; values, tensor layouts, source clocks and text are kept.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import torch

from stmem.evaluation.robomme_adapter import numpy_value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _array_fingerprint(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous()
        metadata = {"shape": list(array.shape), "dtype": str(array.dtype)}
        payload = array.reshape(-1).view(torch.uint8).numpy().tobytes()
    else:
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError("Object arrays cannot fingerprint policy tensors")
        metadata = {"shape": list(array.shape), "dtype": str(array.dtype)}
        payload = array.tobytes()
    metadata["sha256"] = hashlib.sha256(payload).hexdigest()
    return metadata


def frame_fingerprint(frame):
    fields = {name: _array_fingerprint(getattr(frame, name, None)) for name in (
        "points", "valid", "proprio", "previous_action", "visual_features", "point_camera", "point_uv")}
    fields["center_indices"] = {
        str(count): _array_fingerprint(indices)
        for count, indices in sorted(getattr(frame, "center_indices", {}).items())}
    for name in ("source", "source_index", "time", "action_available",
                 "center_sampling_signature", "visual_feature_signature"):
        value = getattr(frame, name, None)
        fields[name] = value.item() if isinstance(value, np.generic) else value
    return {"sha256": _digest(fields), "fields": fields}


def reset_fingerprints(result):
    current = frame_fingerprint(result.current)
    frames = [frame_fingerprint(frame) for frame in result.conditioning]
    goal = hashlib.sha256(result.goal.encode("utf-8")).hexdigest()
    conditioning = {"count": len(frames), "sha256": _digest([f["sha256"] for f in frames]),
                    "frames": frames}
    return {"goal_sha256": goal, "current": current, "conditioning": conditioning,
            "reset_sha256": _digest([goal, current["sha256"], conditioning["sha256"]])}


def _scalar(value):
    array = numpy_value(value).reshape(-1)
    if array.size != 1:
        raise ValueError("Expected one simulator environment")
    return array[0].item()


def _xyz(value):
    array = numpy_value(value).astype(float).reshape(-1, 3)
    if array.shape != (1, 3) or not np.isfinite(array).all():
        raise ValueError("Expected a finite world XYZ value")
    return array[0].tolist()


class HistoryObserver:
    """Transparent wrapper; diagnostic failures never replace simulator results."""

    TASKS = ("VideoUnmask", "VideoRepick", "VideoPlaceOrder", "SwingXtimes")

    def __init__(self, adapter):
        if adapter.task not in self.TASKS:
            raise ValueError(f"Unsupported history-observation task: {adapter.task}")
        self.adapter = adapter
        self._clear()

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def _clear(self):
        self.goal, self.inputs, self.target, self.reset_row = None, None, None, None
        self.rows, self.errors, self.episode = [], [], {}
        self.commands, self.seconds = 0, 0.0
        self.candidates, self.controlled = [], None
        self.contact, self.lift_height = True, .05

    def _env(self):
        return getattr(self.adapter.env, "unwrapped", self.adapter.env)

    def _error(self, phase, exc):
        self.errors.append({"phase": phase, "command_index": self.commands,
                            "error": f"{type(exc).__name__}: {exc}"})

    def _identify(self):
        env, task = self._env(), self.adapter.task
        metadata = {}
        if task == "VideoUnmask":
            self.candidates = list(env.spawned_bins)
            count = int(env.configs[env.difficulty]["pick"])
            intended = [getattr(env, f"bin_{i}") for i in range(count)]
            self.lift_height = .15
            metadata = {"kind": "container_pickup", "colors": list(env.color_names[:count])}
        elif task == "VideoRepick":
            self.candidates, intended = list(env.spawned_cubes), [env.target_cube_1]
            metadata = {"kind": "cube_pickup", "num_repeats": int(env.num_repeats)}
        elif task == "SwingXtimes":
            self.candidates, intended = list(env.all_cubes), [env.target_cube]
            metadata = {"kind": "swing_counting", "num_repeats": int(env.num_repeats),
                        "color": str(env.target_color_name)}
        else:
            self.candidates, intended = list(env.targets), [env.target_target]
            self.controlled, self.contact = env.target_cube, False
            metadata = {"kind": "target_placement", "ordinal": int(env.which_in_subset),
                        "color": str(env.target_color_name)}
        indices = []
        for actor in intended:
            matches = [i for i, candidate in enumerate(self.candidates) if actor is candidate]
            if len(matches) != 1:
                raise ValueError("Target actor is absent or duplicated in candidate actors")
            indices.append(matches[0])
        if not indices:
            raise ValueError("No target actor recorded")
        self.target = {**metadata, "first_index": indices[0], "ordered_indices": indices,
                       "candidate_count": len(self.candidates)}
        self.episode = {"seed": int(_scalar(env.seed)), "difficulty": str(env.difficulty)}

    def _actor(self, actor, index, *, contact):
        xyz = _xyz(actor.pose.p)
        row = {"index": index, "name": str(getattr(actor, "name", index)), "xyz_m": xyz}
        if contact:
            grasp = bool(_scalar(self._env().agent.is_grasping(actor)))
            row.update(grasped=grasp, lifted=bool(grasp and xyz[2] > self.lift_height))
        return row

    def _observe(self, frame):
        env = self._env()
        row = {"command_index": self.commands, "source_index": int(frame.source_index),
               "tcp_xyz_m": _xyz(numpy_value(frame.proprio)[7:10]),
               "candidates": [self._actor(actor, i, contact=self.contact)
                              for i, actor in enumerate(self.candidates)]}
        if self.controlled is not None:
            row["controlled_object"] = self._actor(self.controlled, None, contact=True)
        if hasattr(env, "current_task_index"):
            row["current_task_index"] = int(_scalar(env.current_task_index))
        return row

    def reset(self):
        result = self.adapter.reset()
        self._clear()
        started = time.perf_counter()
        self.goal = result.goal
        try:
            self.inputs = reset_fingerprints(result)
        except Exception as exc:
            self._error("reset_fingerprint", exc)
        try:
            self._identify()
            self.reset_row = self._observe(result.current)
        except Exception as exc:
            self._error("reset_geometry", exc)
        self.seconds += time.perf_counter() - started
        return result

    def step(self, action):
        result = self.adapter.step(action)
        self.commands += 1
        started = time.perf_counter()
        if result.done or not result.frames:
            self.rows.append({"command_index": self.commands, "excluded_reason":
                              "terminal_auto_step" if result.done else "no_retained_observation"})
        else:
            try:
                self.rows.append(self._observe(result.frames[-1]))
            except Exception as exc:
                self._error("step_geometry", exc)
        self.seconds += time.perf_counter() - started
        return result

    def report(self, status):
        active = [r for r in self.rows if "tcp_xyz_m" in r]
        summary = {"official_status": status, "official_task_success": status == "success",
                   "issued_commands": self.commands, "measured_active_commands": len(active)}
        target = None if self.target is None else self.target["first_index"]
        for label, field in (("grasp", "grasped"), ("lift", "lifted")):
            first = next((r for r in active if any(a.get(field) for a in r["candidates"])), None)
            ids = None if first is None else [a["index"] for a in first["candidates"] if a.get(field)]
            summary[f"first_{label}_indices"] = ids
            summary[f"first_{label}_command"] = None if first is None else first["command_index"]
            summary[f"first_{label}_correct"] = None if target is None or ids is None else ids == [target]
        baseline = (self.reset_row or {}).get("current_task_index")
        progression = [r["current_task_index"] for r in active if "current_task_index" in r]
        summary["completed_subgoal_index_increase"] = (None if baseline is None or not progression else
                                                       max(0, max(progression) - baseline))
        return {"version": "history_observer_v1", "task": self.adapter.task,
                "status": "diagnostic_error" if self.errors else "complete" if self.inputs else "not_observed",
                "goal": self.goal, "input_fingerprints": self.inputs, "episode": self.episode,
                "target": self.target, "reset": self.reset_row, "timeline": self.rows,
                "summary": summary, "errors": self.errors, "observer_seconds": self.seconds,
                "definitions": {"privileged_state": "Report only; never added to policy inputs",
                    "fingerprints": "Exact reset policy input fields; random event UUIDs excluded",
                    "lift": "Grasp contact AND actor origin z>.15m for containers or >.05m for cubes",
                    "terminal": "Terminal automatic-step observations are excluded; terminal-only events can be missed",
                    "first_grasp": "First active sampled contact; transient contact is not completed task choice"}}
