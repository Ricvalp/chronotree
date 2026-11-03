"""Generate one fixed real expert trajectory for count-history diagnostics.

PickXtimes training episode 1 (seed 1101, easy) naturally requires three
pick/place cycles. Its pinned official planner supplies all commands, followed
by an expert button-press control. This is regenerated expert data, not a claim
of byte identity with the released demonstrations. Privileged labels stay in
annotations and never become canonical Frame fields or model inputs.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from stmem.config import digest, source_revision
from stmem.data.center_sampling import center_counts, center_sampler_settings
from stmem.data.pointcloud import point_sampler_settings, proprioception
from stmem.evaluation.robomme_adapter import OBSERVATION_FIELDS, RoboMMEAdapter, numpy_value
from stmem.training.checkpoints import atomic_json, atomic_save, file_sha256
from stmem.training.runtime import load_checkpoint, validate_policy_config
from stmem.vision.frozen import FrozenCLIPFeatures, resolve_visual_features

TASK = "PickXtimes"
EPISODE = 1
SEED = 1101
CYCLES = 3
MAX_NATIVE_COMMANDS = 1300
REPLAY_TOLERANCE = 1e-5
KIND = "regenerated_official_planner_training_scene_count_diagnostic"


class ExpertSuccess(BaseException):
    """Stop the planner before its retreat calls step on a terminal environment."""


def native_proprio(adapter):
    observation = adapter.last_raw_observation
    if not observation or not observation.get("front_rgb_list"):
        raise ValueError("Expert step has no observable robot state")
    allowed = {
        field: numpy_value(observation[field + "_list"][-1]) for field in OBSERVATION_FIELDS
    }
    return proprioception(allowed).numpy()


def diagnostics(adapter):
    """Read annotations without calling evaluate or task predicates."""
    env = adapter.env.unwrapped

    def position(actor):
        if isinstance(actor, (list, tuple)):
            if len(actor) != 1:
                raise ValueError("Expected one actor/link for the fixed counting scene")
            actor = actor[0]
        return numpy_value(actor.pose.p).reshape(-1, 3)[0].astype(float).tolist()

    proprio = native_proprio(adapter)
    result = {
        "proprio": proprio.astype(float).tolist(),
        "tcp_xyz": proprio[7:10].astype(float).tolist(),
        "cube_xyz": position(env.target_cube),
        "cube_quaternion": numpy_value(env.target_cube.pose.q).reshape(-1, 4)[0].tolist(),
        "target_xyz": position(env.target),
        "button_xyz": position(env.cap_link),
        "button_base_xyz": position(env.button),
        "button_depth_m": -float(numpy_value(env.button.get_qpos()).reshape(-1)[0]),
        "cube_grasped": bool(numpy_value(env.agent.is_grasping(env.target_cube)).reshape(-1)[0]),
        "active_task_index": int(env.timestep),
        "source_index": adapter._clock["execution"] - 1,
        "camera_extrinsics": {
            camera: numpy_value(adapter.last_raw_observation[camera + "_camera_extrinsic_list"][-1]).tolist()
            for camera in ("front", "wrist")
        },
    }
    result["cube_target_xy_distance_m"] = float(
        np.linalg.norm(np.asarray(result["cube_xyz"])[:2] - np.asarray(result["target_xyz"])[:2])
    )
    return result


def official_goals(color):
    """Use the benchmark's first wording; change only its declared target X."""
    from robomme.robomme_env.utils.task_goal import get_language_goal

    result = {}
    for count in range(1, CYCLES + 1):
        state = SimpleNamespace(num_repeats=count, target_color_name=color)
        owner = SimpleNamespace(env=SimpleNamespace(unwrapped=state))
        result[str(count)] = get_language_goal(owner, TASK)[0]
    return result


def make_adapter(cfg, extractor, device, *, replay=False):
    geometry = cfg["geometry"]
    return RoboMMEAdapter(
        TASK, EPISODE, "train", points=cfg["model"]["points"],
        max_steps=MAX_NATIVE_COMMANDS,
        crop=geometry["crop"], voxel_size=geometry["voxel_size"],
        max_depth_m=geometry["max_depth"], point_sampling=point_sampler_settings(cfg),
        center_sampling=center_sampler_settings(cfg), center_counts=center_counts(cfg),
        feature_extractor=extractor,
        # Replay checks native observations directly and needs no extra point
        # preparation. This only skips processing, never simulator commands.
        observation_stride=MAX_NATIVE_COMMANDS + 10 if replay else cfg["data"]["observation_stride"],
        render_device=device,
    )


def validate_scene(adapter, reset):
    env = adapter.env.unwrapped
    if (env.seed, env.difficulty, env.num_repeats) != (SEED, "easy", CYCLES):
        raise ValueError("Pinned training scene no longer has its declared three-cycle task")
    if reset.conditioning or reset.metadata["conditioning_frames_recorded"]:
        raise ValueError("This count test expects no conditioning demonstration")
    if len(env.task_list) != 2 * CYCLES + 1 or any(
        task.get("demonstration", False) for task in env.task_list
    ):
        raise ValueError("Unexpected counting task sequence")
    if env.robomme_failure_recovery:
        raise ValueError("This fixed expert source does not permit injected failure recovery")


def cpu_frame(frame):
    frame = frame.to("cpu")
    for value in (frame.points, frame.proprio, frame.previous_action, frame.visual_features):
        if value is not None and not torch.isfinite(value).all():
            raise ValueError("Nonfinite expert Frame")
    if frame.source != "execution":
        raise ValueError("Unexpected conditioning source")
    return frame


def annotated(rgb, phase, native_step):
    image = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + 48), "#111827")
    image.paste(Image.fromarray(rgb), (0, 48))
    draw = ImageDraw.Draw(image)
    draw.text((8, 5), "PickXtimes TRAIN episode 1 | regenerated official expert | target X=3", fill="white")
    draw.text((8, 24), f"{phase} | native command {native_step} | diagnostic, not benchmark score", fill="#fbbf24")
    return np.asarray(image)


def replay_check(cfg, device, commands, expected, boundaries, expected_goal, output):
    adapter = make_adapter(cfg, None, device, replay=True)
    fields = ("proprio", "cube_xyz", "cube_quaternion", "target_xyz", "button_xyz", "button_depth_m")
    maximum = dict.fromkeys((*fields, "front_extrinsics", "wrist_extrinsics"), 0.0)
    try:
        reset = adapter.reset()
        validate_scene(adapter, reset)
        if reset.goal != expected_goal:
            raise ValueError("Independent replay changed the language goal")

        def compare(actual, reference, step):
            for field in fields:
                error = float(np.max(np.abs(np.asarray(actual[field]) - np.asarray(reference[field]))))
                maximum[field] = max(maximum[field], error)
            for camera in ("front", "wrist"):
                error = float(np.max(np.abs(
                    np.asarray(actual["camera_extrinsics"][camera])
                    - np.asarray(reference["camera_extrinsics"][camera])
                )))
                maximum[camera + "_extrinsics"] = max(maximum[camera + "_extrinsics"], error)
            if max(maximum.values()) >= REPLAY_TOLERANCE:
                raise ValueError(f"Independent expert replay diverged at native command {step}: {maximum}")
            for field in ("active_task_index", "cube_grasped", "source_index"):
                if actual[field] != reference[field]:
                    raise ValueError(f"Independent expert replay changed {field} at command {step}")

        compare(diagnostics(adapter), expected[0], 0)
        terminal_status = None
        for index, command in enumerate(commands, 1):
            result = adapter.step(command)
            if result.done and (index != len(commands) or result.status != "success"):
                raise ValueError(f"Independent expert replay ended early: {index}: {result.status}: {result.error}")
            compare(diagnostics(adapter), expected[index], index)
            terminal_status = result.status
            if index % 50 == 0:
                progress = {"event": "count_expert_replay", "native_commands": index,
                            "planned_commands": len(commands), "maximum_errors": maximum}
                atomic_json(progress, output / "progress.json")
                print(json.dumps(progress), flush=True)
        if terminal_status != "success":
            raise ValueError("Independent expert control did not finish successfully")
        for boundary in boundaries:
            if expected[boundary["native_step"]]["active_task_index"] != 2 * boundary["completed_cycles"]:
                raise ValueError("A recorded boundary does not finish the declared number of cycles")
    finally:
        adapter.close()
    return {"status": "passed", "native_commands": len(commands), "observations_compared": len(expected),
            "maximum_absolute_errors": maximum, "tolerance": REPLAY_TOLERANCE,
            "terminal_status": terminal_status,
            "note": "Native RGB-D camera geometry and robot/object states compared in an independently reset simulator; replay omitted CLIP/point processing."}


def generate(args):
    checkpoint = args.checkpoint.resolve()
    state = load_checkpoint(checkpoint)
    original_cfg = deepcopy(state["config"])
    if state["stage"] != "teacher" or original_cfg.get("debug") or original_cfg["data"].get("synthetic"):
        raise ValueError("A real trained teacher checkpoint is required for source geometry")
    cfg = deepcopy(original_cfg)
    if cfg["data"].get("observation_stride", 1) != 2:
        raise ValueError("This fixed diagnostic expects the accepted teacher's 10 Hz observation cadence")
    if resolve_visual_features(cfg)["mode"] != "clip_rn50_layer1":
        raise ValueError("This fixed diagnostic expects frozen CLIP visual features")
    clip = args.clip_checkpoint.resolve()
    if not clip.is_file():
        raise FileNotFoundError(f"Missing existing CLIP checkpoint; no download is performed: {clip}")
    cfg["model"]["visual_features"]["checkpoint"] = str(clip)
    validate_policy_config(original_cfg, cfg)
    output = args.output.resolve()
    if any((output / name).exists() for name in ("source.pt", "source.json", "progress.json", "expert.mp4")):
        raise FileExistsError("Expert output already contains a generation attempt; use a new output directory")
    started = time.perf_counter()
    payload = {
        "schema_version": 1, "kind": KIND, "config": cfg, "checkpoint_config": original_cfg,
        "task": TASK, "split": "train", "episode_idx": EPISODE, "seed": SEED, "difficulty": "easy",
        "target_cycles": CYCLES, "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_step": int(state["step"]), "checkpoint_config_sha256": digest(original_cfg),
        "runtime_config_sha256": digest(cfg), "clip_checkpoint_sha256": file_sha256(clip),
        "source_revision": source_revision(), "generator_sha256": file_sha256(__file__),
        "conditioning": [], "execution": [], "commands": [], "boundaries": [], "native_diagnostics": [],
        "notes": ["Fixed training-scene diagnostic; not official validation or test.",
                  "Commands are regenerated by the pinned official planner; not released HDF5 action identity.",
                  "Counterfactual probes must truncate execution at their own boundary; later expert frames and commands are references only.",
                  "All privileged object, count and phase fields are diagnostic annotations only.",
                  "One physical held command may align a completed cycle to the retained observation cadence; no observation is copied."],
    }
    del state
    extractor = FrozenCLIPFeatures(cfg, device=args.device)
    adapter = make_adapter(cfg, extractor, args.device)
    writer = imageio.get_writer(output / "expert.mp4", fps=20)
    phase = "reset"

    def progress(event):
        item = {"event": event, "phase": phase, "native_commands": len(payload["commands"]),
                "retained_frames": len(payload["execution"]), "completed_cycles": len(payload["boundaries"]),
                "elapsed_seconds": time.perf_counter() - started}
        atomic_json(item, output / "progress.json")
        print(json.dumps(item), flush=True)

    try:
        reset = adapter.reset()
        validate_scene(adapter, reset)
        base = adapter.env.unwrapped
        payload.update(goal=reset.goal, goals=official_goals(base.target_color_name),
                       target_color=base.target_color_name, environment=reset.metadata,
                       robot_base_pose=numpy_value(base.agent.robot.pose.raw_pose).reshape(-1, 7)[0].tolist())
        if reset.goal != payload["goals"][str(CYCLES)]:
            raise ValueError("Reset goal differs from the declared official three-cycle wording")
        payload["execution"].append(cpu_frame(reset.current))
        payload["native_diagnostics"].append(diagnostics(adapter))
        for rgb in adapter.last_rgb:
            writer.append_data(annotated(rgb, phase, 0))

        class PlannerTransport:
            unwrapped = base

            def step(self, action):
                if len(payload["commands"]) >= MAX_NATIVE_COMMANDS:
                    raise RuntimeError("Expert exceeded the declared native-command cap")
                action = np.asarray(action, dtype=np.float32).reshape(-1)
                if action.shape != (8,) or not np.isfinite(action).all():
                    raise ValueError("Expert command must be finite native joint targets of shape [8]")
                result = adapter.step(action)
                if result.status in ("fail", "error", "timeout") or result.error:
                    raise RuntimeError(f"Expert failed during {phase}: {result.status}: {result.error}")
                if result.done and (phase != "button" or result.status != "success"):
                    raise RuntimeError(f"Unexpected expert termination during {phase}: {result.status}")
                payload["commands"].append(np.asarray(result.command, dtype=np.float32).copy())
                payload["execution"].extend(cpu_frame(frame) for frame in result.frames)
                row = diagnostics(adapter)
                row["phase"] = phase
                row["native_step"] = len(payload["commands"])
                payload["native_diagnostics"].append(row)
                if phase != "button" and row["button_depth_m"] > 0.005:
                    raise ValueError("Expert pressed button before finishing the three cycles")
                for rgb in adapter.last_rgb:
                    writer.append_data(annotated(rgb, phase, len(payload["commands"])))
                if len(payload["commands"]) % 25 == 0:
                    progress("count_expert_progress")
                if result.done:
                    payload["terminal_status"] = result.status
                    raise ExpertSuccess()
                return adapter.last_raw_observation, 0.0, result.terminated, result.truncated, {"status": result.status}

        from robomme.robomme_env.utils.planner_fail_safe import FailAwarePandaArmMotionPlanningSolver

        transport = PlannerTransport()
        planner = FailAwarePandaArmMotionPlanningSolver(
            transport, debug=False, vis=False, base_pose=base.agent.robot.pose,
            visualize_target_grasp_pose=False, print_env_info=False,
        )
        for cycle in range(1, CYCLES + 1):
            for stage in ("pickup", "placement"):
                index = 2 * (cycle - 1) + (stage == "placement")
                if int(base.timestep) != index:
                    raise ValueError(f"Expected expert subtask {index}, got {base.timestep}")
                phase = f"{stage}_{cycle}"
                base.task_list[index]["solve"](base, planner)
                if int(base.timestep) != index + 1:
                    raise ValueError(f"Expert did not complete {phase}: active subtask {base.timestep}")
            expert_end = len(payload["commands"])
            while len(payload["commands"]) % cfg["data"]["observation_stride"]:
                phase = f"cadence_hold_after_cycle_{cycle}"
                transport.step(payload["commands"][-1])
            row = diagnostics(adapter)
            if row["cube_grasped"] or row["cube_target_xy_distance_m"] > 0.05:
                raise ValueError(f"Cycle {cycle} did not leave the cube released on the target")
            frame_index = len(payload["execution"]) - 1
            if payload["execution"][frame_index].source_index != len(payload["commands"]):
                raise ValueError("Cycle boundary does not have an exactly aligned retained observation")
            boundary = {"completed_cycles": cycle, "frame_index": frame_index,
                        "native_step": len(payload["commands"]), "expert_completed_native_step": expert_end,
                        "cadence_alignment_commands": len(payload["commands"]) - expert_end,
                        "diagnostics": row}
            payload["boundaries"].append(boundary)
            imageio.imwrite(output / f"cycle_{cycle}_current.png", adapter.last_rgb[-1])
            progress("count_expert_boundary")
        phase = "button"
        try:
            base.task_list[-1]["solve"](base, planner)
        except ExpertSuccess:
            pass
        if payload.get("terminal_status") != "success":
            raise ValueError("Expert button control did not finish the official task successfully")
        payload["terminal_protocol_note"] = (
            "The continuous wrapper may execute an automatic extra terminal step. "
            "Terminal observations are future reference data and never part of a cycle-boundary policy input."
        )
    finally:
        writer.close()
        adapter.close()
    del extractor
    payload["replay_validation"] = replay_check(
        cfg, args.device, payload["commands"], payload["native_diagnostics"], payload["boundaries"],
        payload["goal"], output,
    )
    payload["commands"] = torch.from_numpy(np.stack(payload["commands"]))
    payload["elapsed_seconds"] = time.perf_counter() - started
    atomic_save(payload, output / "source.pt", keep_previous=False)
    metadata = {key: value for key, value in payload.items()
                if key not in ("execution", "conditioning", "commands", "native_diagnostics")}
    metadata.update(status="complete", source_sha256=file_sha256(output / "source.pt"),
                    native_commands=len(payload["commands"]), retained_frames=len(payload["execution"]),
                    native_observations=len(payload["native_diagnostics"]),
                    expert_video=str(output / "expert.mp4"), source=str(output / "source.pt"))
    atomic_json(metadata, output / "source.json")
    progress("count_expert_complete")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-checkpoint", type=Path, default=Path("cache/backbones/clip_RN50.pt"))
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = generate(args)
        except BaseException as error:
            atomic_json({"status": "failed", "error": f"{type(error).__name__}: {error}"},
                        args.output / "generation_failure.json")
            raise
    print(json.dumps({"status": result["status"], "source": result["source"],
                      "native_commands": result["native_commands"],
                      "retained_frames": result["retained_frames"]}), flush=True)


if __name__ == "__main__":
    main()
