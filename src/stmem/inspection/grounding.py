"""Bounded real-scene diagnostics, preserving each checkpoint's native cadence.

Object poses label measurements only. Physically relocated scenes replay the same
old commands; their counterfactual predictions are never executed. Fresh rollouts
use the production OnlinePolicy and zero-order hold executor at native 20 Hz.
"""

from __future__ import annotations
import argparse
from copy import deepcopy
import gc
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import torch

from stmem.config import source_revision
from stmem.data.center_sampling import center_counts, center_sampler_settings
from stmem.data.pointcloud import point_sampler_settings, proprioception
from stmem.evaluation.robomme_adapter import RoboMMEAdapter, OBSERVATION_FIELDS, numpy_value, physical_action
from stmem.evaluation.runner import OnlinePolicy, episode_timeout
from stmem.models.policy import Evidence
from stmem.models.attention_diagnostics import CrossAttentionDiagnostics
from stmem.training.attention import context_groups
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import load_policy, validate_policy_config
from stmem.vision.frozen import FrozenCLIPFeatures, resolve_visual_features
from stmem.visualization.kinematics import panda_tcp_positions
from stmem.inspection import grounding_helpers as helpers
from stmem.inspection.common import read_manifest

SEEDS = (2701, 2702, 2703, 2704)
TASK = "PickXtimes"


def save(path, value):
    plain = json.loads(json.dumps(value, default=helpers.json_default, allow_nan=False))
    atomic_json(plain, path)


def arr(value):
    return numpy_value(value).copy()


def cube_mask(points, cube):
    # The fixed validation scene's target is blue. Privileged box/color labels
    # are never used for sampling, feature extraction or model conditioning.
    return (
        helpers.cube_mask(points, cube)
        & (points[:, 5] > points[:, 3] + 0.15)
        & (points[:, 5] > points[:, 4] + 0.15)
    )


def native_proprio(adapter):
    obs = adapter.last_raw_observation
    if not obs or not obs.get("front_rgb_list"):
        return None
    allowed = {field: numpy_value(obs[field + "_list"][-1]) for field in OBSERVATION_FIELDS}
    return proprioception(allowed).numpy()


def make_adapter(cfg, extractor):
    return RoboMMEAdapter(
        TASK,
        0,
        "val",
        points=cfg["model"]["points"],
        max_steps=cfg["evaluation"]["max_steps"],
        crop=cfg["geometry"]["crop"],
        voxel_size=cfg["geometry"]["voxel_size"],
        max_depth_m=cfg["geometry"]["max_depth"],
        point_sampling=point_sampler_settings(cfg),
        center_sampling=center_sampler_settings(cfg),
        center_counts=center_counts(cfg),
        feature_extractor=extractor,
        observation_stride=cfg["data"].get("observation_stride", 1),
        include_conditioning=cfg["model"].get("history_mode", "full") != "none",
    )


def relocate(adapter, shift):
    from mani_skill.utils.structs.pose import Pose

    base = adapter.env.unwrapped
    initialize = base._initialize_episode

    def initialize_shifted(env_idx, options):
        initialize(env_idx, options)
        cube = base.target_cube
        cube.set_pose(
            Pose.create_from_pq(
                cube.pose.p + torch.as_tensor(shift, device=cube.pose.p.device), cube.pose.q.clone()
            )
        )

    base._initialize_episode = initialize_shifted


def fixed_noise(policy, step, seed):
    generator = torch.Generator(device=policy.device).manual_seed(seed)
    for _ in range(step // 4 + 1):
        value = torch.randn((policy.config["model"]["horizon"], 8), device=policy.device, generator=generator)
    return value


def physical_prediction(raw, cfg):
    physical = np.stack([physical_action(row, TASK) for row in raw])
    targets = panda_tcp_positions(physical[:, :7], task=TASK)
    expanded = np.repeat(physical, cfg["model"].get("action_stride", 1), axis=0)
    native_tcp = panda_tcp_positions(expanded[:, :7], task=TASK)
    return dict(
        raw_actions=raw,
        physical_actions=physical,
        tcp_targets=targets,
        native_commands=expanded,
        native_tcp=native_tcp,
        first4_native_tcp=native_tcp[:4],
    )


@torch.no_grad()
def predictions(online, step):
    policy = online.policy
    frame = online.memory.current
    archive = online.memory.archive
    clocks = dict(
        conditioning=online.clocks["conditioning"], execution=frame.time if online.history_enabled else 0.0
    )
    evidence = Evidence(archive, policy.encoder.current(frame), online.text, frame, clocks)
    records = archive.leaves
    groups = context_groups(
        evidence,
        records,
        policy.device,
        current_state_enabled=policy.current_state_enabled,
        clock_enabled=policy.clock_enabled,
    )
    result = dict(
        history_tokens=len(records),
        current_tokens=len(evidence.current_records),
        current_source_index=frame.source_index,
        predictions={},
    )
    for precision in dict.fromkeys(("fp32", online.precision)):
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        with torch.autocast(policy.device.type, dtype=dtype, enabled=precision != "fp32"):
            context, bias = policy.context(evidence, records)
        rows = []
        for seed in SEEDS:
            noise = fixed_noise(policy, step, seed)
            with torch.autocast(policy.device.type, dtype=dtype, enabled=precision != "fp32"):
                normalized = policy.decoder.integrate(
                    context, bias, steps=online.cfg["model"]["flow_steps"], noise=noise
                )
            row = physical_prediction(policy.normalizer.denormalize(normalized).cpu().numpy(), online.cfg)
            row["seed"] = seed
            if seed == SEEDS[0]:
                with (
                    torch.autocast(policy.device.type, dtype=dtype, enabled=precision != "fp32"),
                    CrossAttentionDiagnostics(
                        policy.decoder, groups, capture_on_call=online.cfg["model"]["flow_steps"]
                    ) as capture,
                ):
                    observed = policy.decoder.integrate(
                        context, bias, steps=online.cfg["model"]["flow_steps"], noise=noise
                    )
                assert torch.equal(observed, normalized), "Attention hook must not modify predictions"
                row["attention"] = dict(
                    metrics=capture.metrics(),
                    layers=capture.layer_metrics(),
                    hook_control="bit_identical",
                    scope="last_flow_evaluation_all_action_queries",
                )
            rows.append(row)
        result["predictions"][precision] = rows
    return result


def pooling_summary(trace, cubes):
    points, frame_ids = arr(trace["points"]), arr(trace["point_frame_indices"])
    membership = np.zeros(len(points), dtype=bool)
    for i, cube in enumerate(cubes):
        membership[frame_ids == i] = cube_mask(points[frame_ids == i], cube)
    neighbors, allowed, weights = [
        arr(trace[key]) for key in ("neighbor_indices", "neighbor_mask", "weights")
    ]
    mass = (weights * (membership[neighbors] & allowed)).sum(-1)
    valid = arr(trace["valid"])
    return dict(
        total_points=len(points),
        cube_points=int(membership.sum()),
        valid_tokens=int(valid.sum()),
        cube_majority_tokens=int(((mass > 0.5) & valid).sum()),
        max_cube_mass=float(mass.max(initial=0)),
        cube_mass_per_token=mass,
        anchors=arr(trace["anchor"]),
    )


@torch.no_grad()
def coverage(policy, frames, cubes, output, step):
    result = dict(step=step, banks={})
    banks = [("current", frames[-1:], cubes[-1:])]
    if policy.history_enabled and len(frames) > 1:
        preceding = len(frames) - 1
        start = ((preceding - 1) // policy.config["model"]["block_size"]) * policy.config["model"][
            "block_size"
        ]
        banks.append(("latest_history_block", frames[start:preceding], cubes[start:preceding]))
    for name, bank, positions in banks:
        trace = {}
        if name == "current":
            records = policy.encoder.current(bank[-1], trace=lambda value: trace.update(value))
        else:
            records = [
                r
                for r in policy.encoder.encode_frames(
                    bank,
                    "execution",
                    "diagnostic-coverage",
                    completed=len(bank) == policy.config["model"]["block_size"],
                    trace=lambda value: trace.update(value),
                )
                if r.modality == "visual"
            ]
        summary = pooling_summary(trace, positions)
        summary["source_indices"] = [f.source_index for f in bank]
        if records:
            content = torch.stack([r.content.float() for r in records])
            summary["mean_content_norm"] = float(content.norm(dim=-1).mean())
            majority = torch.as_tensor(
                summary["cube_mass_per_token"][arr(trace["valid"])] > 0.5, device=policy.device
            )
            anchors = torch.stack([r.anchor for r in records]).to(policy.device)
            table = anchors[:, 2].abs() < 0.003
            if majority.any() and table.any():
                a, b = content[majority].mean(0), content[table].mean(0)
                summary["cube_table_content_centroid_cosine"] = float(
                    torch.nn.functional.cosine_similarity(a, b, dim=0)
                )
                summary["cube_table_content_centroid_l2"] = float((a - b).norm())
        arrays = {key: arr(value) for key, value in trace.items() if isinstance(value, torch.Tensor)}
        arrays.update(
            cube_positions=np.array([c["p"] for c in positions]),
            cube_quaternions=np.array([c["q"] for c in positions]),
            cube_half_sizes=np.array([c["half_size"] for c in positions]),
        )
        np.savez_compressed(output / f"step_{step:03d}_{name}_trace.npz", **arrays)
        result["banks"][name] = summary
    return result


@torch.no_grad()
def rendered(policy, cfg, extractor, base, output, replay_path):
    previous = json.loads(replay_path.read_text())
    result = dict(
        base,
        mode="rendered_cube",
        scenes={},
        noise_seeds=SEEDS,
        replay_source=str(replay_path),
        replay_source_sha256=file_sha256(replay_path),
    )
    baseline_inputs = {}
    for name, shift in (
        ("baseline", [0.0, 0.0, 0.0]),
        ("x_plus_10cm", [0.1, 0.0, 0.0]),
        ("y_plus_10cm", [0.0, 0.1, 0.0]),
    ):
        adapter = make_adapter(cfg, extractor)
        with torch.enable_grad():
            online = OnlinePolicy(policy, cfg, "FULL", cfg["model"]["budget"])
        scene = dict(shift_xyz_m=shift, states={}, replay_errors=[])
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        try:
            if any(shift):
                relocate(adapter, shift)
            reset = adapter.reset()
            online.reset(reset, SEEDS[0])
            assert not reset.conditioning
            assert adapter.env.unwrapped.target_color_name == "blue"
            scene["goal"] = reset.goal
            scene["environment"] = reset.metadata
            frames, cubes = [reset.current], [helpers.cube_state(adapter.env.unwrapped)]
            for step in range(81):
                if step:
                    old_row = previous["timeline"][step - 1]
                    observed = adapter.step(np.asarray(old_row["command"], dtype=np.float32))
                    assert not observed.done
                    current_proprio = native_proprio(adapter)
                    error = float(np.max(np.abs(current_proprio - old_row["observed_proprio"])))
                    assert error < 1e-5, (step, error)
                    scene["replay_errors"].append(error)
                    for frame in observed.frames:
                        online.observe(frame)
                        frames.append(frame)
                        cubes.append(helpers.cube_state(adapter.env.unwrapped))
                if step in (0, 40, 80):
                    frame = online.memory.current
                    signature = dict(
                        proprio=frame.proprio.tolist(),
                        previous_action=frame.previous_action.tolist(),
                        source_index=frame.source_index,
                        time=frame.time,
                        action_available=frame.action_available,
                        goal=reset.goal,
                    )
                    if name == "baseline":
                        baseline_inputs[step] = signature
                    else:
                        assert signature == baseline_inputs[step], (
                            "Cube shift changed robot/action/time/task inputs"
                        )
                    data = predictions(online, step)
                    data.update(
                        cube=cubes[-1],
                        input_signature=signature,
                        coverage=coverage(policy, frames, cubes, directory, step),
                    )
                    scene["states"][str(step)] = data
                    imageio.imwrite(directory / f"step_{step:03d}.png", adapter.last_rgb[-1])
                    print(
                        json.dumps(
                            dict(event="rendered_probe", variant=base["variant"], scene=name, step=step)
                        ),
                        flush=True,
                    )
            scene["max_replay_proprio_error"] = max(scene.pop("replay_errors"))
            result["scenes"][name] = scene
            save(output / "partial.json", result)
        finally:
            adapter.close()
    result["summary"] = {}
    reference = result["scenes"]["baseline"]
    for name, scene in result["scenes"].items():
        result["summary"][name] = {}
        for step, data in scene["states"].items():
            result["summary"][name][step] = {}
            for precision, rows in data["predictions"].items():
                delta = [
                    row["native_tcp"] - np.asarray(ref["native_tcp"])
                    for row, ref in zip(rows, reference["states"][step]["predictions"][precision])
                ]
                result["summary"][name][step][precision] = {
                    window: dict(
                        mean_tcp_change_mm=float(
                            np.mean([np.linalg.norm(d[:n], axis=-1).mean() for d in delta]) * 1000
                        ),
                        mean_xyz_change_mm=np.mean([d[:n].mean(0) for d in delta], axis=0) * 1000,
                    )
                    for window, n in (
                        ("first4_native_0p2s", 4),
                        ("first16_native_0p8s", 16),
                        ("full", len(delta[0])),
                    )
                }
    result["status"] = "complete"
    save(output / "result.json", result)
    return result


@torch.no_grad()
def rollout(policy, cfg, extractor, base, output, cap):
    adapter = make_adapter(cfg, extractor)
    with torch.enable_grad():
        online = OnlinePolicy(policy, cfg, "FULL", cfg["model"]["budget"])
    result = dict(
        base,
        mode="rollout",
        cap_native_commands=cap,
        noise_seed=SEEDS[0],
        decisions=[],
        timeline=[],
        coverage=[],
    )
    video = None
    try:
        reset = adapter.reset()
        online.reset(reset, SEEDS[0])
        result.update(goal=reset.goal, environment=reset.metadata, precision=online.precision)
        frames, cubes = [reset.current], [helpers.cube_state(adapter.env.unwrapped)]
        result["initial_cube"] = cubes[-1]
        video_path = output / f"PickXtimes_val0_first{cap}.mp4"
        video = imageio.get_writer(video_path, fps=20)
        for rgb in adapter.last_rgb:
            video.append_data(rgb)
        count, done = 0, False
        while count < cap and not done:
            chunk, timing = online.decide()
            result["decisions"].append(dict(step=count, raw_actions=chunk.cpu().numpy(), timing=timing))
            commands = chunk[: cfg["model"]["execute"]].repeat_interleave(
                cfg["model"].get("action_stride", 1), dim=0
            )
            for action in commands:
                cube_before = helpers.cube_state(adapter.env.unwrapped)
                observed = adapter.step(action)
                count += 1
                cube = helpers.cube_state(adapter.env.unwrapped)
                native = native_proprio(adapter)
                for frame in observed.frames:
                    online.observe(frame)
                    frames.append(frame)
                    cubes.append(cube)
                command_tcp = panda_tcp_positions(observed.command[:7], task=TASK)
                grasped = bool(
                    arr(adapter.env.unwrapped.agent.is_grasping(adapter.env.unwrapped.target_cube)).reshape(
                        -1
                    )[0]
                )
                row = dict(
                    step=count,
                    command=observed.command,
                    observed_proprio=native,
                    cube=cube,
                    command_tcp=command_tcp,
                    status=observed.status,
                    done=observed.done,
                    observed_source_indices=[f.source_index for f in observed.frames],
                    memory_current_source_index=online.memory.current.source_index,
                    command_tcp_cube_xy_distance_m=float(
                        np.linalg.norm(command_tcp[:2] - cube_before["p"][:2])
                    ),
                    grasped=grasped,
                    cube_lift_from_reset_m=float(cube["p"][2] - result["initial_cube"]["p"][2]),
                )
                if native is not None:
                    row.update(
                        actual_tcp_cube_xy_distance_m=float(np.linalg.norm(native[7:9] - cube["p"][:2])),
                        command_tracking_error_m=float(np.linalg.norm(command_tcp - native[7:10])),
                    )
                result["timeline"].append(row)
                for rgb in adapter.last_rgb:
                    video.append_data(rgb)
                if count % 40 == 0 and observed.frames and not observed.done:
                    result["coverage"].append(coverage(policy, frames, cubes, output, count))
                    imageio.imwrite(output / f"step_{count:03d}.png", adapter.last_rgb[-1])
                    save(output / "partial.json", result)
                    print(
                        json.dumps(
                            dict(event="rollout", variant=base["variant"], step=count, grasped=grasped)
                        ),
                        flush=True,
                    )
                if observed.done:
                    done = True
                    result["status"] = observed.status
                    break
                if count >= cap:
                    break
        result.setdefault("status", "stopped_at_declared_diagnostic_cap")
        result["video"] = str(video_path)
    finally:
        if video is not None:
            video.close()
        adapter.close()
    save(output / "result.json", result)
    return result


def reset_generator():
    """Reuse the repository's tracked, independently callable reset viewer."""
    path = Path(__file__).resolve().parents[3] / "scripts/visualize_reset_cube_chunks.py"
    spec = importlib.util.spec_from_file_location("stmem_reset_inspection_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_reset(manifest, label, manifest_sha256, *, resume):
    info = manifest["checkpoints"][label]
    checkpoint = Path(info["path"])
    if file_sha256(checkpoint) != info["sha256"]:
        raise ValueError(f"Checkpoint changed: {checkpoint}")
    output = Path(manifest["output"]) / "reset_cube_chunks" / label
    identity = {
        "inspection_manifest_sha256": manifest_sha256,
        "checkpoint_sha256": info["sha256"],
    }
    result_path = output / "payload.json"
    job_path = output / "job.json"
    if result_path.exists():
        partial = json.loads(result_path.read_text())
        if "inspection_manifest_sha256" not in partial:
            # An interruption may occur after the tracked generator's atomic
            # completion but before this wrapper adds its manifest identity.
            # Only the separately recorded job identity permits a fresh retry.
            if not resume or not job_path.exists() or json.loads(job_path.read_text()) != identity:
                raise ValueError(f"Unidentified reset output: {result_path}")
            result_path.unlink()
    previous = helpers.verified_existing(result_path, identity, resume=resume)
    if previous:
        return previous
    # The tracked generator writes payload.json only after its full reset-repeat
    # controls have passed. A process killed before attaching our identity may
    # leave that file; never assume it belongs to this run without its job record.
    output.mkdir(parents=True, exist_ok=True)
    if job_path.exists() and json.loads(job_path.read_text()) != identity:
        raise ValueError(f"Reset job identity mismatch: {job_path}")
    save(job_path, identity)
    if result_path.exists():
        result_path.unlink()
    cfg = info.get("config")
    if cfg is None:
        from stmem.training.runtime import load_checkpoint

        cfg = load_checkpoint(checkpoint)["config"]
    visual_checkpoint = (
        Path(manifest["assets"]["clip_checkpoint"])
        if resolve_visual_features(cfg)["mode"] != "xyzrgb"
        else None
    )
    args = SimpleNamespace(
        output=output,
        checkpoint=[f"{label}={checkpoint}"],
        device=manifest["device"],
        visual_checkpoint=visual_checkpoint,
        split="val",
        episode=0,
        seeds=list(SEEDS),
        placements=["baseline", "x_plus_10cm", "y_plus_10cm", "y_minus_10cm"],
        timeout_seconds=1800,
    )
    payload = reset_generator().generate(args)
    if payload["models"][0]["checkpoint_sha256"] != info["sha256"]:
        raise ValueError(f"Checkpoint changed during reset generation: {checkpoint}")
    payload.update(identity, diagnostic_complete=True)
    payload["action_stride"] = cfg["model"].get("action_stride", 1)
    payload["observation_stride"] = cfg["data"].get("observation_stride", 1)
    payload["native_control_hz"] = 20
    payload["models"][0]["history_mode"] = cfg["model"].get("history_mode", "full")
    save(result_path, payload)
    return payload


def run_model(manifest, label, mode, manifest_sha256, *, resume):
    started = time.perf_counter()
    info = manifest["checkpoints"][label]
    checkpoint = Path(info["path"])
    if file_sha256(checkpoint) != info["sha256"]:
        raise ValueError(f"Checkpoint changed: {checkpoint}")
    output = (
        Path(manifest["output"]) / ("grounding" if mode == "rendered_cube" else "rollouts") / label / mode
    )
    identity = {
        "inspection_manifest_sha256": manifest_sha256,
        "checkpoint_sha256": info["sha256"],
        "checkpoint_step": info["step"],
    }
    previous = helpers.verified_existing(output / "result.json", identity, resume=resume)
    if previous:
        return previous
    output.mkdir(parents=True, exist_ok=True)
    policy, state = load_policy(checkpoint, manifest["device"])
    cfg = deepcopy(state["config"])
    if resolve_visual_features(cfg)["mode"] != "xyzrgb":
        cfg["model"]["visual_features"]["checkpoint"] = manifest["assets"]["clip_checkpoint"]
    validate_policy_config(state["config"], cfg)
    policy.config = cfg
    policy.eval().requires_grad_(False)
    extractor = (
        FrozenCLIPFeatures(cfg, device=manifest["device"])
        if resolve_visual_features(cfg)["mode"] != "xyzrgb"
        else None
    )
    base = dict(
        identity,
        kind="bounded_real_validation_diagnostic_not_benchmark_score",
        variant=label,
        checkpoint=str(checkpoint),
        local_source=source_revision(),
        harness_sha256=file_sha256(__file__),
        config=cfg,
        task=TASK,
        split="val",
        episode_idx=0,
        temporal_sampling=dict(
            native_hz=20,
            action_stride=cfg["model"].get("action_stride", 1),
            observation_stride=cfg["data"].get("observation_stride", 1),
            horizon=cfg["model"]["horizon"],
            execute_targets=cfg["model"]["execute"],
            executor="zero_order_hold",
        ),
        notes=[
            "Object poses and grasp flags label diagnostics only, never policy inputs.",
            "Training exposure and architecture/input choices differ; this is not an isolated architecture ablation.",
            "First4-native metrics span 0.2 seconds; full predicted horizon durations may differ.",
            "Counterfactual cube-shift predictions are never executed; fresh rollouts use the production executor.",
        ],
    )
    with episode_timeout(1800):
        result = (
            rendered(policy, cfg, extractor, base, output, Path(manifest["assets"]["replay_commands"]))
            if mode == "rendered_cube"
            else rollout(policy, cfg, extractor, base, output, 160)
        )
    result.update(diagnostic_complete=True, elapsed_seconds=time.perf_counter() - started)
    if policy.device.type == "cuda":
        result["gpu_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(policy.device)
    save(output / "result.json", result)
    print(
        json.dumps(
            dict(
                event="finished",
                variant=label,
                mode=mode,
                status=result["status"],
                elapsed_seconds=result["elapsed_seconds"],
            )
        ),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label", help="Inspect just this manifest label")
    parser.add_argument(
        "--mode", choices=("all", "reset", "rendered_cube", "rollout", "report"), default="all"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    manifest_sha256 = file_sha256(args.manifest)
    labels = [args.label] if args.label else manifest["labels"]
    if any(label not in manifest["labels"] for label in labels):
        parser.error("--label must be declared in the inspection manifest")
    torch.set_num_threads(2)
    if args.mode != "report":
        for label in labels:
            modes = ("reset", "rendered_cube", "rollout") if args.mode == "all" else (args.mode,)
            for mode in modes:
                if mode == "reset":
                    run_reset(manifest, label, manifest_sha256, resume=args.resume)
                else:
                    run_model(manifest, label, mode, manifest_sha256, resume=args.resume)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if args.mode in ("all", "report") and not args.label:
        from stmem.inspection.grounding_report import build_reports

        build_reports(manifest, manifest_sha256)


if __name__ == "__main__":
    main()
