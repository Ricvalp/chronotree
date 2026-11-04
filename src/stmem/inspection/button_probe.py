"""Record whole action chunks during genuine button-approach rollouts.

The production episode executor, cadence and causal memory are unchanged.
Privileged state is read only for annotations, never supplied to the policy.
Additional noise draws use independent generators and cannot change execution.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from stmem.config import source_revision
from stmem.data.center_sampling import center_counts, center_sampler_settings
from stmem.data.pointcloud import point_sampler_settings
from stmem.evaluation.robomme_adapter import RoboMMEAdapter, numpy_value, physical_action
from stmem.evaluation.runner import OnlinePolicy, run_episode
from stmem.evaluation.execution import evaluation_execution_config
from stmem.inspection.count_expert import native_proprio
from stmem.inspection.count_probe import audit_checkpoint, tensor_state_hash
from stmem.models.policy import Evidence, choose_records
from stmem.training.checkpoints import atomic_json, atomic_save
from stmem.training.runtime import load_checkpoint, load_policy, validate_policy_config
from stmem.visualization.kinematics import panda_observation_errors, panda_tcp_positions
from stmem.vision.frozen import FrozenCLIPFeatures


def diagnostic_state(adapter):
    """Observe task bookkeeping without calling mutating task predicates."""
    env = adapter.env.unwrapped
    proprio = native_proprio(adapter)
    cap = env.cap_link
    if isinstance(cap, (tuple, list)):
        if len(cap) != 1:
            raise ValueError("Expected one button cap")
        cap = cap[0]
    button = numpy_value(cap.pose.p).reshape(-1, 3)[0]
    index = int(env.timestep)
    task = env.task_list[index] if index < len(env.task_list) else {"name": "completed"}
    result = {
        "native_step": adapter._step,
        "tcp_xyz": proprio[7:10].astype(float).tolist(),
        "proprio": proprio.astype(float).tolist(),
        "button_xyz": button.astype(float).tolist(),
        "button_depth_m": -float(numpy_value(env.button.get_qpos()).reshape(-1)[0]),
        "task_index": index, "task_count": len(env.task_list), "task_name": task["name"],
        "final_button_phase": index == len(env.task_list) - 1 and "button" in task["name"].lower(),
    }
    result["button_distance_mm"] = float(np.linalg.norm(proprio[7:10] - button) * 1000)
    result["button_xy_mm"] = float(np.linalg.norm(proprio[7:9] - button[:2]) * 1000)
    result["height_above_button_mm"] = float((proprio[9] - button[2]) * 1000)
    return result


def chunk_metrics(xyz, current, button, execute=8):
    xyz, current, button = (np.asarray(v, dtype=float) for v in (xyz, current, button))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not 0 < execute < len(xyz):
        raise ValueError("Expected a chunk with both executed and unused targets")
    if not all(np.isfinite(v).all() for v in (xyz, current, button)):
        raise ValueError("Nonfinite geometry")
    distance = np.linalg.norm(xyz - button, axis=1)
    initial = np.linalg.norm(current - button)
    return {
        "progress_at_target8_mm": float((initial - distance[execute - 1]) * 1000),
        "progress_at_target16_mm": float((initial - distance[-1]) * 1000),
        "extra_tail_progress_mm": float((distance[execute - 1] - distance[-1]) * 1000),
        "minimum_first8_distance_mm": float(distance[:execute].min() * 1000),
        "minimum_tail_distance_mm": float(distance[execute:].min() * 1000),
        "descent_at_target8_mm": float((current[2] - xyz[execute - 1, 2]) * 1000),
        "descent_at_target16_mm": float((current[2] - xyz[-1, 2]) * 1000),
        "first8_displacement_mm": float(np.linalg.norm(xyz[:execute] - current, axis=1).max() * 1000),
        "whole_chunk_displacement_mm": float(np.linalg.norm(xyz - current, axis=1).max() * 1000),
    }


def cpu_state(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_state(v) for v in value]
    return deepcopy(value)


class TracedAdapter(RoboMMEAdapter):
    def __init__(self, *args, stable_observation_ids=False, **kwargs):
        self.stable_observation_ids = stable_observation_ids
        super().__init__(*args, **kwargs)

    def ingest(self, obs, info, *, event_id, reset=False, command=None):
        if self.stable_observation_ids:
            # Diagnostic paired-rollout control: transport UUIDs affect hashed
            # node IDs and thus UNIFORM tie breaks. Use the same unique-per-reset
            # sequence for both execution horizons; no sensor values change.
            event_id = "button-diagnostic/reset" if reset else f"button-diagnostic/step/{self._step}"
        return super().ingest(obs, info, event_id=event_id, reset=reset, command=command)

    def reset(self):
        result = super().reset()
        self.timeline = [diagnostic_state(self)]
        self.commands = []
        self.video_start_frames = len(self.last_rgb)
        return result

    def step(self, action):
        result = super().step(action)
        self.commands.append(result.command.astype(float).tolist())
        if self.last_raw_observation and self.last_raw_observation.get("front_rgb_list"):
            row = diagnostic_state(self)
            row["terminal"] = result.done
            self.timeline.append(row)
        return result


class TracedOnline(OnlinePolicy):
    def __init__(self, *args, adapter, output, comparison=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_adapter = adapter
        self.trace_output = output
        self.comparison = comparison
        self.trace = []
        self.probed = {}
        self.last_notice = 0

    @torch.no_grad()
    def decide(self):
        # This is exactly the production draw; diagnostics run afterward.
        actions, timing = super().decide()
        state = diagnostic_state(self.trace_adapter)
        fk_error = panda_observation_errors(state["proprio"], task=self.trace_adapter.task)
        if fk_error["position_error_m"] > 1e-4:
            raise ValueError(f"Observed/FK TCP mismatch: {fk_error}")
        physical = np.stack([physical_action(a, self.trace_adapter.task) for a in actions])
        xyz = panda_tcp_positions(physical[:, :7], task=self.trace_adapter.task)
        frame = self.memory.current
        row = {
            "native_step": self.trace_adapter._step, "diagnostics": state,
            "raw_actions": actions.float().cpu().tolist(), "physical_actions": physical.tolist(),
            "target_tcp_xyz": xyz.tolist(), "fk_error": fk_error,
            "metrics": chunk_metrics(xyz, state["tcp_xyz"], state["button_xyz"]),
            "points": frame.points[frame.valid].float().cpu().tolist(),
            "timing": timing,
        }
        reason = None
        if state["final_button_phase"]:
            if "enter_button" not in self.probed:
                reason = "enter_button"
            elif ("hover" not in self.probed and state["button_xy_mm"] < 120
                  and state["height_above_button_mm"] > 30 and len(self.trace) >= 3
                  and np.linalg.norm(np.asarray(state["tcp_xyz"])
                                     - self.trace[-3]["diagnostics"]["tcp_xyz"]) < .02):
                reason = "hover"
            elif ("late_hover" not in self.probed and "hover" in self.probed
                  and row["native_step"] >= self.probed["hover"] + 160):
                reason = "late_hover"
        if reason:
            self.probed[reason] = row["native_step"]
            row["probe_reason"] = reason
            row["probes"] = self.additional_predictions(state)
            path = self.trace_output / f"probe_{row['native_step']:04d}.pt"
            atomic_save({"diagnostics": state, "commands": self.trace_adapter.commands,
                         "sim_state": cpu_state(self.trace_adapter.env.unwrapped.get_state_dict()),
                         "chunk": physical, "probes": row["probes"],
                         "goal": self.goal}, path, keep_previous=False)
            row["probe_snapshot"] = str(path.resolve())
        self.trace.append(row)
        with (self.trace_output / "decisions.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if time.monotonic() - self.last_notice > 30 or reason:
            print(json.dumps({"event": "button_trace", "native_step": row["native_step"],
                              "phase": state["task_name"], "button_xy_mm": state["button_xy_mm"],
                              "height_mm": state["height_above_button_mm"],
                              "probe": reason, **row["metrics"]}), flush=True)
            self.last_notice = time.monotonic()
        return actions, timing

    @torch.no_grad()
    def additional_predictions(self, state):
        evidence = Evidence(self.memory.archive, self.policy.encoder.current(self.memory.current),
                            self.text, self.memory.current, self.clocks.copy())
        models = [("rollout_policy", self.policy, self.method)]
        if self.comparison:
            models.append(("accepted_teacher_on_same_history", self.comparison, "FULL"))
        result = []
        for label, policy, method in models:
            # Same observed evidence and clocks; only the trained reader/head differ.
            selected = choose_records(evidence, method, self.budget, rng=random.Random(2701))
            with torch.autocast(policy.device.type, dtype=self.autocast_dtype,
                                enabled=self.precision != "fp32"):
                context, bias = policy.context(evidence, selected.records)
                for seed in (2701, 2702, 2703, 2704):
                    normalized = policy.decoder.integrate(
                        context, bias, steps=self.cfg["model"]["flow_steps"],
                        generator=torch.Generator(device=policy.device).manual_seed(seed))
                    actions = policy.normalizer.denormalize(normalized)
                    physical = np.stack([physical_action(a, self.trace_adapter.task) for a in actions])
                    xyz = panda_tcp_positions(physical[:, :7], task=self.trace_adapter.task)
                    result.append({"label": label, "method": method, "noise_seed": seed,
                                   "physical_actions": physical.tolist(), "target_tcp_xyz": xyz.tolist(),
                                   "metrics": chunk_metrics(xyz, state["tcp_xyz"], state["button_xyz"])})
        return result


def make_adapter(cfg, task, episode, extractor, device, *, replay=False, stable_observation_ids=False):
    return TracedAdapter(
        task, episode, "val", points=cfg["model"]["points"], max_steps=cfg["evaluation"]["max_steps"],
        crop=cfg["geometry"]["crop"], voxel_size=cfg["geometry"]["voxel_size"],
        max_depth_m=cfg["geometry"]["max_depth"], point_sampling=point_sampler_settings(cfg),
        center_sampling=center_sampler_settings(cfg), center_counts=center_counts(cfg),
        feature_extractor=extractor, render_device=device,
        observation_stride=100000 if replay else cfg["data"]["observation_stride"],
        stable_observation_ids=stable_observation_ids,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=("VideoRepick", "SwingXtimes", "PickXtimes"), default="VideoRepick")
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--method", choices=("FULL", "UNIFORM", "TREE", "FLAT"), default="UNIFORM")
    parser.add_argument("--execute-actions", choices=(8, 16), type=int, default=8)
    parser.add_argument("--stable-observation-ids", action="store_true",
                        help="Use deterministic transport identities for paired UNIFORM rollouts")
    parser.add_argument("--clip-checkpoint", type=Path, default=Path("cache/backbones/clip_RN50.pt"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new/empty diagnostic output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    source = load_checkpoint(args.checkpoint)
    identity = audit_checkpoint(args.checkpoint, source["stage"], source["config"])
    policy, state = load_policy(args.checkpoint, args.device)
    policy.eval().requires_grad_(False)
    cfg = deepcopy(state["config"])
    cfg["stage"] = state["stage"]
    cfg["model"]["visual_features"]["checkpoint"] = str(args.clip_checkpoint.resolve())
    cfg["evaluation"]["split"] = "val"
    validate_policy_config(state["config"], cfg)
    cfg, execution_override = evaluation_execution_config(
        state["config"], cfg, execute_actions=args.execute_actions)
    comparison = None
    if args.teacher:
        comparison, teacher_state = load_policy(args.teacher, args.device)
        comparison.eval().requires_grad_(False)
        teacher_identity = audit_checkpoint(args.teacher, "teacher", state["config"])
        if tensor_state_hash(teacher_state["model"], ("encoder.", "text_encoder.")) != identity["frozen_encoder_sha256"]:
            raise ValueError("Shared-evidence comparison requires identical frozen encoders")
    else:
        teacher_identity = None
    extractor = FrozenCLIPFeatures(cfg, device=policy.device)
    output = {"protocol": {"scope": "Fixed validation-scene button diagnostic, not a benchmark score",
                           "execute_actions": args.execute_actions, "action_stride": 2, "horizon": 16,
                           "episodes": args.episodes, "task": args.task,
                           "labels_only": "button pose and task phase never enter policy inputs",
                           "noise_protocol": "production sequential generator; extra probes use independent fixed seeds",
                           "source": source_revision(), "execution_override": execution_override},
              "checkpoint": identity, "comparison_teacher": teacher_identity, "trials": []}
    output["protocol"]["stable_observation_ids"] = args.stable_observation_ids
    atomic_json(cfg, args.output / "config.json")
    atomic_json(output, args.output / "results.json")
    print(json.dumps({"event": "button_probe_start", "checkpoint": identity}), flush=True)
    for episode in args.episodes:
        directory = args.output / f"{args.task}_{episode:02d}"
        directory.mkdir()
        adapter = make_adapter(cfg, args.task, episode, extractor, args.device,
                               stable_observation_ids=args.stable_observation_ids)
        online = TracedOnline(policy, cfg, args.method, 128, adapter=adapter, output=directory,
                              comparison=comparison,
                              selection_family=args.method.lower() if args.method in ("TREE", "FLAT") else None)
        seed = cfg["evaluation"]["seed"] + cfg["data"]["tasks"].index(args.task) * 1000 + episode
        video = directory / "rollout.mp4"
        try:
            result = run_episode(adapter, online, task=args.task, episode_idx=episode, split="val", seed=seed,
                                 timeout_seconds=1800, video_path=str(video), checkpoint_hash=identity["sha256"])
            result.update(label=f"{state['stage']}_{state['step']}", checkpoint_step=state["step"],
                          decisions=online.trace, timeline=getattr(adapter, "timeline", []),
                          video=str(video.resolve()), video_start_frames=getattr(adapter, "video_start_frames", 0),
                          goal=online.goal, execute_actions=args.execute_actions, action_stride=2)
            atomic_json(result, directory / "result.json")
            output["trials"].append(result)
            atomic_json(output, args.output / "results.json")
            print(json.dumps({"event": "trial_finished", "episode": episode, "status": result["status"],
                              "commands": result["primitive_commands"], "error": result["error"],
                              "runtime_seconds": result["runtime_seconds"]}), flush=True)
        finally:
            adapter.close()
    from stmem.inspection.button_report import write_report
    print(write_report(output, args.output))


if __name__ == "__main__":
    main()
