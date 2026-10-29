"""Render first-policy-decision cube-position diagnostics, with real simulator RGB-D.

This is a bounded, artificial scene intervention, not a benchmark evaluation.
No predicted command is executed. The official reset's automatic initial step is
retained. The production OnlinePolicy path generates each chunk independently.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
import torch

from stmem.config import source_revision
from stmem.data.center_sampling import center_counts, center_sampler_settings
from stmem.data.pointcloud import point_sampler_settings
from stmem.evaluation.robomme_adapter import RoboMMEAdapter, numpy_value, physical_action
from stmem.evaluation.runner import OnlinePolicy, episode_timeout
from stmem.training.checkpoints import file_sha256
from stmem.training.runtime import environment_manifest, load_policy
from stmem.visualization.kinematics import panda_observation_errors, panda_tcp_pose
from stmem.vision.frozen import FrozenCLIPFeatures, resolve_visual_features, visual_feature_identity


TASK = "PickXtimes"
PLACEMENTS = (
    ("baseline", "Original cube position", [0.0, 0.0, 0.0]),
    ("x_plus_10cm", "Cube X +10 cm", [0.1, 0.0, 0.0]),
    ("x_minus_10cm", "Cube X −10 cm", [-0.1, 0.0, 0.0]),
    ("y_plus_10cm", "Cube Y +10 cm", [0.0, 0.1, 0.0]),
    ("y_minus_10cm", "Cube Y −10 cm", [0.0, -0.1, 0.0]),
)
MODEL_LABELS = {
    "control": "Control continuation",
    "continuation_no_commands": "Continuation without previous joint commands",
    "fresh_no_commands": "From scratch without previous joint commands",
    "h16": "H16 without previous joint commands",
    "h32": "H32 without previous joint commands",
}


def plain(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return numpy_value(value).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_json(path, value):
    path.write_text(json.dumps(value, default=plain, indent=2, allow_nan=False) + "\n")


def rotation(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def pose_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation(numpy_value(pose.q).reshape(-1, 4)[0])
    matrix[:3, 3] = numpy_value(pose.p).reshape(-1, 3)[0]
    return matrix


def relocate_before_reset(adapter, shift):
    from mani_skill.utils.structs.pose import Pose

    base = adapter.env.unwrapped
    initialize = base._initialize_episode

    def initialize_with_translation(env_idx, options):
        initialize(env_idx, options)
        cube = base.target_cube
        cube.set_pose(Pose.create_from_pq(
            cube.pose.p + torch.as_tensor(shift, device=cube.pose.p.device),
            cube.pose.q.clone(),
        ))

    base._initialize_episode = initialize_with_translation


def make_adapter(cfg, episode, split, feature_extractor=None):
    return RoboMMEAdapter(
        TASK, episode, split, points=cfg["model"]["points"],
        max_steps=cfg["evaluation"]["max_steps"], crop=cfg["geometry"]["crop"],
        voxel_size=cfg["geometry"]["voxel_size"], max_depth_m=cfg["geometry"]["max_depth"],
        point_sampling=point_sampler_settings(cfg), center_sampling=center_sampler_settings(cfg),
        center_counts=center_counts(cfg),
        feature_extractor=feature_extractor,
    )


def observation_signature(frame):
    return {
        "proprio": frame.proprio.tolist(), "previous_action": frame.previous_action.tolist(),
        "action_available": frame.action_available, "time": frame.time,
        "source": frame.source, "source_index": frame.source_index,
    }


def frame_capture(adapter, reset, name, label, shift, directory):
    frame = reset.current
    base = adapter.env.unwrapped
    cube = base.target_cube
    cube_data = {
        "p": numpy_value(cube.pose.p).reshape(-1, 3)[0].copy(),
        "q": numpy_value(cube.pose.q).reshape(-1, 4)[0].copy(),
        "half_size": np.broadcast_to(numpy_value(base.cube_half_size), (3,)).copy(),
    }
    robot_base = pose_matrix(base.agent.robot.pose)
    fk_error = panda_observation_errors(frame.proprio.numpy(), task=TASK, base_pose=robot_base)
    assert fk_error["position_error_m"] < 1e-5, fk_error
    assert fk_error["rotation_error_rad"] < 1e-3, fk_error
    points = frame.points[frame.valid].numpy()
    in_box = np.all(np.abs((points[:, :3] - cube_data["p"]) @ rotation(cube_data["q"]))
                    <= cube_data["half_size"] + 0.003, axis=-1)
    # Diagnostic label only, never used for point selection or policy conditioning.
    channel = {"red": 0, "green": 1, "blue": 2}[base.target_color_name]
    cube_color = np.all(points[:, 3 + channel, None] >
                        points[:, [3 + c for c in range(3) if c != channel]] + .15, axis=-1)
    assert np.any(in_box & cube_color), "Relocated cube must be visible in policy input"
    arrays = {"points": frame.points.numpy(), "valid": frame.valid.numpy(),
              "proprio": frame.proprio.numpy(), "previous_action": frame.previous_action.numpy()}
    arrays.update({f"center_indices_{k}": numpy_value(v) for k, v in frame.center_indices.items()})
    np.savez_compressed(directory / f"{name}_input.npz", **arrays)
    image = Image.fromarray(adapter.last_rgb[-1])
    image.save(directory / f"{name}_cameras.png")
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    calibration = {key: numpy_value(value).tolist() for key, value in adapter.calibration.items()}
    for camera in ("front", "wrist"):
        calibration[f"{camera}_camera_extrinsic"] = numpy_value(
            adapter.last_raw_observation[f"{camera}_camera_extrinsic_list"][-1]).tolist()
    other_objects = {"button": pose_matrix(base.button.pose),
                     "placement_target": pose_matrix(base.target.pose)}
    other_objects.update({f"non_target_cube_{i}": pose_matrix(actor.pose)
                          for i, actor in enumerate(base.non_target_cubes)})
    center_bounds = np.array([[-.3, .1], [-.2, .2]]) + np.array([1, -1]) * float(base.cube_half_size)
    return {
        "id": name, "label": label, "shift_xyz_m": shift, "cube": cube_data,
        "points": points, "cube_point_count": int((in_box & cube_color).sum()),
        "cube_point_count_definition": "Oriented box +3mm and target-color RGB diagnostic label, not segmentation",
        "cube_color": base.target_color_name,
        "original_cube_center_sampling_xy_bounds_m": center_bounds,
        "inside_original_cube_center_sampling_xy_bounds": bool(np.all(
            (cube_data["p"][:2] >= center_bounds[:, 0]) & (cube_data["p"][:2] <= center_bounds[:, 1]))),
        "other_objects_diagnostic_only": other_objects,
        "current_tcp_pose": panda_tcp_pose(frame.proprio.numpy()[:7], task=TASK, base_pose=robot_base),
        "robot_base_pose": robot_base, **observation_signature(frame),
        "conditioning_frames": len(reset.conditioning), "execution_history_frames": 0,
        "goal": reset.goal, "environment": reset.metadata, "calibration": calibration,
        "fk_observation_error": fk_error,
        "rgb_data_uri": "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode(),
        "input_npz": f"{name}_input.npz", "predictions": {},
    }


def generate(args):
    torch.set_num_threads(2)
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested_placements = getattr(args, "placements", None)
    placements = (
        [next(p for p in PLACEMENTS if p[0] == name) for name in requested_placements]
        if requested_placements else PLACEMENTS
    )
    if placements[0][0] != "baseline" or len({p[0] for p in placements}) != len(placements):
        raise ValueError("Placements must be unique and start with baseline")
    if (output / "payload.json").exists():
        raise FileExistsError("Use a new output directory, or --render-only to rebuild HTML")
    models, online_models, reference_cfg = [], {}, None
    for specification in args.checkpoint:
        identifier, separator, raw_path = specification.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]*", identifier):
            raise ValueError("--checkpoint must be a unique lowercase_id=/absolute/checkpoint.pt")
        if identifier in online_models:
            raise ValueError(f"Duplicate model ID: {identifier}")
        path = Path(raw_path).resolve()
        policy, state = load_policy(path, args.device)
        cfg = state["config"]
        if args.visual_checkpoint:
            cfg["model"].setdefault("visual_features", {})["checkpoint"] = str(args.visual_checkpoint.resolve())
        online = OnlinePolicy(policy, cfg, "FULL", cfg["model"]["budget"])
        assert policy.config == cfg
        if reference_cfg is None:
            reference_cfg = deepcopy(cfg)
        else:
            assert cfg["geometry"] == reference_cfg["geometry"], "Different geometric preprocessing"
            for key in ("points", "horizon", "execute", "flow_steps"):
                assert cfg["model"][key] == reference_cfg["model"][key], key
            assert center_counts(cfg) == center_counts(reference_cfg)
            assert visual_feature_identity(cfg) == visual_feature_identity(reference_cfg), "Different frozen visual preprocessing"
        models.append({"id": identifier, "label": MODEL_LABELS.get(identifier, identifier),
                       "step": state["step"], "checkpoint": str(path),
                       "checkpoint_sha256": file_sha256(path),
                       "condition_on_previous_joint_commands": cfg["model"].get(
                           "condition_on_previous_joint_commands", True),
                       "precision": online.precision})
        models[-1].update(
            proprioception_mode=cfg["model"].get("proprioception_mode", "all"),
            condition_on_previous_commands=cfg["model"].get("condition_on_previous_commands", True),
            visual_features=visual_feature_identity(cfg),
        )
        online_models[identifier] = online
        print(json.dumps({"event": "loaded", "model": identifier, "step": state["step"],
                          "precision": online.precision}), flush=True)
    assert len({model["precision"] for model in models}) == 1, "Mixed numerical protocols"
    cfg = reference_cfg
    feature_extractor = FrozenCLIPFeatures(cfg, device=args.device) if resolve_visual_features(cfg)["mode"] != "xyzrgb" else None
    payload = {
        "schema_version": 1, "kind": "initial_cube_relocation_diagnostic_not_benchmark_score",
        "task": TASK, "split": args.split, "episode_idx": args.episode, "decision_step": 0,
        "horizon": cfg["model"]["horizon"], "execute_steps": cfg["model"]["execute"],
        "flow_steps": cfg["model"]["flow_steps"], "noise_seeds": args.seeds,
        "precision": {"encoder": "fp32", "context_decoder": models[0]["precision"]},
        "models": models, "scenes": [], "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_revision(), "environment": environment_manifest(),
        "generator_sha256": file_sha256(__file__), "geometry": cfg["geometry"],
        "notes": [
            "First policy decision after official reset (which includes its automatic initialization step).",
            "PickXtimes has no conditioning video; execution history is empty and previous commands unavailable.",
            "Only cube translation changes. Both camera RGB-D images and production point sampling are regenerated.",
            "Cube poses label diagnostics and place the cube; no privileged object state enters the policy.",
            "Each noise seed starts at its first draw for every scene and checkpoint. No predicted action is executed.",
            "Shown points are the exact valid, sampled front+wrist cloud supplied to the current encoder.",
            "Chunk markers are TCP positions of physical joint targets via FK, not simulated motion or Cartesian actions.",
            "Connecting lines show target order only; there are exactly horizon discrete action targets.",
            "Only the first execute_steps commands would normally run before replanning; later targets are predictions.",
            "Shifted scenes are artificial diagnostics, not official episode configurations; original XY spawn-range membership is reported per scene.",
            "At reset, previous commands are unavailable in every model; differences between checkpoints reflect learned weights.",
            "Each checkpoint's training step and input settings are listed separately; only compare matched training budgets as a controlled ablation.",
        ],
    }
    basestate = None
    baseline_reset = None
    with torch.no_grad(), episode_timeout(args.timeout_seconds):
        for name, label, shift in placements:
            adapter = make_adapter(cfg, args.episode, args.split, feature_extractor)
            try:
                if any(shift):
                    relocate_before_reset(adapter, shift)
                reset = adapter.reset()
                frame = reset.current
                assert not reset.conditioning
                assert frame.time == 0 and frame.source_index == 0 and frame.source == "execution"
                assert not frame.action_available and torch.count_nonzero(frame.previous_action) == 0
                signature = observation_signature(frame)
                if basestate is None:
                    basestate, baseline_reset = signature, reset
                else:
                    assert signature == basestate, "Cube intervention changed robot/command/clock inputs"
                    assert reset.goal == baseline_reset.goal, "Cube intervention changed goal text"
                scene = frame_capture(adapter, reset, name, label, shift, output)
                if payload["scenes"]:
                    baseline = payload["scenes"][0]
                    assert scene["calibration"] == baseline["calibration"]
                    observed_shift = scene["cube"]["p"] - baseline["cube"]["p"]
                    assert np.max(np.abs(observed_shift - shift)) < 1e-4, observed_shift
                    assert abs(float(scene["cube"]["p"][2]) - float(baseline["cube"]["p"][2])) < 1e-4
                    for key, matrix in scene["other_objects_diagnostic_only"].items():
                        assert np.allclose(matrix, baseline["other_objects_diagnostic_only"][key],
                                           rtol=0, atol=1e-6), f"Other object changed: {key}"
                    scene["actual_shift_xyz_m"] = observed_shift
                else:
                    scene["actual_shift_xyz_m"] = np.zeros(3)
                for identifier, online in online_models.items():
                    predictions = scene["predictions"][identifier] = {}
                    for seed in args.seeds:
                        online.reset(reset, seed)
                        assert not online.memory.archive.leaves
                        assert online.decision_index == 0
                        raw, timing = online.decide()
                        assert timing["history_tokens"] == timing["archive_fine_leaves"] == 0
                        assert raw.shape == (payload["horizon"], 8) and torch.isfinite(raw).all()
                        physical = np.stack([physical_action(a, TASK) for a in raw])
                        poses = panda_tcp_pose(physical[:, :7], task=TASK, base_pose=scene["robot_base_pose"])
                        predictions[str(seed)] = {
                            "raw_actions": raw.cpu().numpy(), "physical_actions": physical,
                            "tcp_xyz": poses[:, :3, 3], "tcp_pose": poses,
                            "clipped_joint_components": int(np.sum(physical[:, :7] != raw.cpu().numpy()[:, :7])),
                            "decision": timing,
                        }
                payload["scenes"].append(scene)
                print(json.dumps({"event": "scene_complete", "scene": name,
                                  "points": len(scene["points"]), "cube_points": scene["cube_point_count"],
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
            finally:
                adapter.close()
        # Repeat baseline without interventions: verify renderer/sampling and full predictions.
        adapter = make_adapter(cfg, args.episode, args.split, feature_extractor)
        try:
            repeat = adapter.reset()
            for field in ("points", "valid", "proprio", "previous_action"):
                assert torch.equal(getattr(repeat.current, field), getattr(baseline_reset.current, field)), field
            if feature_extractor is not None:
                assert torch.equal(repeat.current.visual_features, baseline_reset.current.visual_features)
            assert repeat.goal == baseline_reset.goal
            for identifier, online in online_models.items():
                online.reset(repeat, args.seeds[0])
                raw, _ = online.decide()
                assert np.array_equal(raw.cpu().numpy(), payload["scenes"][0]["predictions"][identifier][str(args.seeds[0])]["raw_actions"])
        finally:
            adapter.close()
    for scene in payload["scenes"]:
        for identifier, predictions in scene["predictions"].items():
            for seed, prediction in predictions.items():
                reference = payload["scenes"][0]["predictions"][identifier][seed]
                delta = prediction["tcp_xyz"] - reference["tcp_xyz"]
                displacement = np.linalg.norm(delta, axis=-1) * 1000
                prediction["metrics"] = {
                    "first4_tcp_change_mean_mm": float(displacement[:payload["execute_steps"]].mean()),
                    "full16_tcp_change_mean_mm": float(displacement[:16].mean()),
                    "first16_tcp_change_mean_mm": float(displacement[:16].mean()),
                    "full_chunk_tcp_change_mean_mm": float(displacement.mean()),
                    "max_tcp_change_mm": float(displacement.max()),
                    "tcp_change_by_target_mm": displacement,
                    "tcp_signed_delta_xyz_mm": delta * 1000,
                    "gripper_flips": int(np.sum(prediction["physical_actions"][:, 7] != reference["physical_actions"][:, 7])),
                }
    payload["validation"] = {
        "reset_robot_goal_clock_calibration_identical": True,
        "no_history_or_available_previous_commands": True,
        "repeat_baseline_points_and_first_seed_predictions_bit_identical": True,
        "actual_cube_translation_matches_requested_within_m": 1e-4,
        "fk_matches_observed_tcp_within_m": 1e-5,
        "finite_action_chunks": True,
        "predicted_chunks": len(models) * len(placements) * len(args.seeds),
        "predicted_commands_executed": 0,
    }
    payload["elapsed_seconds"] = time.perf_counter() - started
    save_json(output / "payload.json", payload)
    print(json.dumps({"event": "data_complete", "path": str(output / "payload.json"),
                      "elapsed_seconds": payload["elapsed_seconds"]}), flush=True)
    return json.loads((output / "payload.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", help="Repeat: model_id=/path/to/checkpoint.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visual-checkpoint", type=Path, help="Local CLIP RN50 path when inspecting checkpoints copied from another machine")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2701, 2702, 2703, 2704])
    parser.add_argument("--placements", nargs="+", choices=[p[0] for p in PLACEMENTS],
                        help="Ordered cube placements; must start with baseline (default: all five)")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only and not args.checkpoint:
        parser.error("Supply at least one --checkpoint model_id=path")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Noise seeds must be unique")
    payload = json.loads((args.output / "payload.json").read_text()) if args.render_only else generate(args)
    if not args.data_only:
        from stmem.visualization.reset_chunks_html import write_reset_chunks_html

        print(write_reset_chunks_html(payload, args.output))


if __name__ == "__main__":
    main()
