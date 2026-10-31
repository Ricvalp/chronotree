"""Fixed training-scene handoffs for an immutable checkpoint bundle; no training.

Uses production action/observation cadence and native RGB-D/frozen features.
The only optimized path is CPU metadata transfer, checked bit-identical in situ.
"""

from __future__ import annotations
import argparse
import fcntl
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import imageio.v2 as imageio
import numpy as np
import torch

from stmem.config import source_revision, digest
from stmem.data.center_sampling import center_counts, center_sampler_settings
from stmem.data.pointcloud import point_sampler_settings, proprioception
from stmem.evaluation.robomme_adapter import RoboMMEAdapter, numpy_value, OBSERVATION_FIELDS
from stmem.evaluation.runner import OnlinePolicy, episode_timeout
from stmem.models.metadata_adapter import metadata
from stmem.models.policy import Evidence
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import load_policy, validate_policy_config
from stmem.vision.frozen import FrozenCLIPFeatures, resolve_visual_features

from stmem.inspection import handoff_helpers as base

ROOT = None
ASSETS = {}
ASSET_HASHES = {}
VARIANTS = ()
CHECKPOINTS = {}
MANIFEST_DIGEST = None
SCENES = (("VideoUnmask", 0), ("VideoUnmask", 1), ("PickXtimes", 0))
PREFIXES = (0, 4, 16, 32, 64)
NOISE_SEED = 2701


def configure(manifest):
    global ROOT, ASSETS, ASSET_HASHES, VARIANTS, CHECKPOINTS, MANIFEST_DIGEST
    protocol = manifest["protocol"]
    if (
        protocol["handoff_native_commands"] != 320
        or tuple(protocol["handoff_prefixes"]) != PREFIXES
        or tuple(map(tuple, protocol["handoff_scenes"])) != SCENES
    ):
        raise ValueError("Handoff manifest differs from the fixed diagnostic protocol")
    ROOT = Path(manifest["output"]) / "handoff"
    ROOT.mkdir(parents=True, exist_ok=True)
    ASSETS = manifest["assets"]
    ASSET_HASHES = manifest["asset_hashes"]
    if file_sha256(ASSETS["alignment"]) != ASSET_HASHES["alignment"]:
        raise ValueError("Alignment record changed after inspection planning")
    VARIANTS = tuple(manifest["labels"])
    CHECKPOINTS = manifest["checkpoints"]
    MANIFEST_DIGEST = file_sha256(Path(manifest["output"]) / "inspection_manifest.json")


def source_for(task, episode):
    source = base.source_for(task, episode, ASSETS)
    key = "prepared" if task == "VideoUnmask" else "pick_prefix"
    if source["sha256"] != ASSET_HASHES[key]:
        raise ValueError(f"Expert source changed after inspection planning: {key}")
    return source


def runner_sha256():
    return digest({p.name: file_sha256(p) for p in (Path(__file__), Path(base.__file__))})


def valid_result(existing, identity, *, task, episode, prefix, control, cap):
    expected = (task, episode, prefix, control, cap)
    recorded = tuple(
        existing[key]
        for key in ("task", "episode_idx", "prefix_commands", "expert_control", "cap_total_commands")
    )
    if recorded != expected or existing.get("inspection_manifest_sha256") != MANIFEST_DIGEST:
        raise ValueError(
            "Handoff output belongs to a different manifest/protocol; use another output directory"
        )
    if existing["checkpoint_identity"] != identity or existing["runner_sha256"] != runner_sha256():
        raise ValueError("Handoff checkpoint/runner changed; use another output directory")
    return existing["status"] not in ("diagnostic_error", "expert_record_exhausted_without_terminal")


def fast_transfer(policy):
    original = policy.adapter.forward

    def fast(records, clocks, device, *, timeless=False):
        if not records or any(r.content_sum.device.type != "cpu" for r in records):
            return original(records, clocks, device, timeless=timeless)
        latent = torch.stack([r.content.float() for r in records]).to(device)
        meta = torch.stack([metadata(r, clocks, device="cpu") for r in records]).to(device)
        if timeless:
            meta[:, 9:13] = 0
        return policy.adapter.norm(policy.adapter.content(latent) + policy.adapter.meta(meta))

    policy.adapter.forward = fast
    return original, fast


@torch.no_grad()
def transport_parity(online, implementations):
    policy, memory = online.policy, online.memory
    archive = memory.archive
    clocks = dict(online.clocks, execution=memory.current.time if online.history_enabled else 0.0)
    evidence = Evidence(archive, policy.encoder.current(memory.current), online.text, memory.current, clocks)
    noise = torch.randn(
        (online.cfg["model"]["horizon"], 8),
        device=policy.device,
        generator=torch.Generator(device=policy.device).manual_seed(NOISE_SEED),
    )
    rows = []
    for forward in implementations:
        policy.adapter.forward = forward
        with torch.autocast("cuda", dtype=online.autocast_dtype, enabled=online.precision != "fp32"):
            context, bias = policy.context(evidence, archive.leaves)
            prediction = policy.decoder.integrate(
                context, bias, steps=online.cfg["model"]["flow_steps"], noise=noise
            )
        rows.append((context, bias, prediction))
    for i, key in enumerate(("context", "bias", "prediction")):
        assert torch.equal(rows[0][i], rows[1][i]), key
    return {
        "source_index": memory.current.source_index,
        "history_leaves": len(archive.leaves),
        "context_bit_identical": True,
        "bias_bit_identical": True,
        "actions_bit_identical": True,
    }


def raw_frame(adapter, fallback, step):
    obs = adapter.last_raw_observation
    if not obs or not obs.get("front_rgb_list"):
        return fallback, False
    allowed = {field: numpy_value(obs[field + "_list"][-1]) for field in OBSERVATION_FIELDS}
    return SimpleNamespace(proprio=proprioception(allowed), source_index=step), True


@torch.no_grad()
def run(
    policy,
    cfg,
    state,
    identity,
    extractor,
    implementations,
    source,
    *,
    task,
    episode,
    prefix,
    variant,
    cap=320,
    control=False,
):
    directory = (
        ROOT
        / task
        / f"episode_{episode}"
        / variant
        / ("expert_control" if control else f"prefix_{prefix:03d}")
    )
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if valid_result(
            existing, identity, task=task, episode=episode, prefix=prefix, control=control, cap=cap
        ):
            from stmem.inspection.handoff_validation import validate_trial

            validate_trial(result_path, source, cfg)
            print(json.dumps({"event": "reuse", "path": str(result_path)}), flush=True)
            return existing
    started = time.perf_counter()
    astride, ostride = cfg["model"].get("action_stride", 1), cfg["data"].get("observation_stride", 1)
    execute = cfg["model"]["execute"]
    replan = execute * astride
    assert replan == 4 and prefix % replan == 0
    history = cfg["model"].get("history_mode", "full") != "none"
    result = {
        "kind": "real_simulator_training_scene_expert_prefix_handoff_not_benchmark_score",
        "task": task,
        "split": "train",
        "episode_idx": episode,
        "variant": variant,
        "checkpoint_step": state["step"],
        "checkpoint_identity": identity,
        "training_lr": cfg["training"]["lr"],
        "prefix_commands": prefix,
        "expert_control": control,
        "cap_total_commands": cap,
        "noise_seed": NOISE_SEED,
        "noise_protocol": "native-horizon sequential draws, skip prefix/replan draws, matched by absolute primitive time within each model",
        "cross_model_noise_note": "H16 and H32 consume different-length draws; their sequential noise differs after reset. Same-horizon models share draws.",
        "expert_source": source["kind"],
        "expert_source_path": source["path"],
        "expert_source_sha256": source["sha256"],
        "inspection_manifest_sha256": MANIFEST_DIGEST,
        "source": source_revision(),
        "precision": cfg["training"]["precision"],
        "temporal_sampling": {
            "action_stride": astride,
            "observation_stride": ostride,
            "execute": execute,
            "replan_primitive_steps": replan,
            "action_hold": "zero_order_hold",
            "native_hz": 20,
        },
        "history_enabled": history,
        "proprioception_mode": cfg["model"].get("proprioception_mode", "all"),
        "visual_mode": resolve_visual_features(cfg)["mode"],
        "leaves_per_history_frame": cfg["model"]["visual_tokens"] // cfg["model"]["block_size"]
        + int(policy.encoder.history_events_enabled)
        if history
        else 0,
        "observations": "only production sampled frames/allowed modalities enter policy; full native observations recorded separately for diagnostic labels",
        "diagnostic_transfer": "batched CPU record/metadata transfer; in-situ bit-identical parity before first VideoUnmask prefix32 prediction",
        "runner_sha256": runner_sha256(),
        "timeline": [],
        "decisions": [],
        "prefix_replay_checks": [],
        "video": str(directory / "rollout.mp4"),
        "notes": [
            "Hidden object poses/grasp labels never enter policy inputs.",
            "Diagnostic cap is not an official failure.",
            "Same-time expert error after handoff is divergence, not a corrective-action label.",
            "Video includes supplied conditioning even for the model that does not receive it.",
        ],
    }
    adapter, video = None, None
    try:
        with episode_timeout(900):
            adapter = RoboMMEAdapter(
                task,
                episode,
                "train",
                points=cfg["model"]["points"],
                max_steps=cfg["evaluation"]["max_steps"],
                crop=cfg["geometry"]["crop"],
                voxel_size=cfg["geometry"]["voxel_size"],
                max_depth_m=cfg["geometry"]["max_depth"],
                point_sampling=point_sampler_settings(cfg),
                center_sampling=center_sampler_settings(cfg),
                center_counts=center_counts(cfg),
                feature_extractor=extractor,
                observation_stride=ostride,
                include_conditioning=history,
            )
            reset = adapter.reset()
            reset_error = float(
                np.max(np.abs(reset.current.proprio[:7].numpy() - source["reset_proprio"][:7]))
            )
            assert reset_error < 1e-4 and reset.goal == source["goal"]
            raw_conditioning = len(adapter.last_rgb) - 1
            if "conditioning_proprio" in source:
                expected = source["conditioning_proprio"]
                assert raw_conditioning == len(expected)
                errors = []
                for i, q in enumerate(expected):
                    allowed = {
                        field: numpy_value(adapter.last_raw_observation[field + "_list"][i])
                        for field in OBSERVATION_FIELDS
                    }
                    errors.append(np.max(np.abs(proprioception(allowed)[:7].numpy() - q[:7])))
                assert max(errors, default=0.0) < 1e-4
                result["conditioning_max_joint_difference_rad"] = float(max(errors, default=0.0))
            online = OnlinePolicy(policy, cfg, "FULL", cfg["model"]["budget"])
            online.reset(reset, NOISE_SEED)
            for _ in range(prefix // replan):
                torch.randn((cfg["model"]["horizon"], 8), device=policy.device, generator=online.generator)
            online.decision_index = prefix // replan
            result.update(
                reset_joint_max_difference_rad=reset_error,
                goal=reset.goal,
                environment=reset.metadata,
                conditioning_frames=len(reset.conditioning),
                video_conditioning_frames=raw_conditioning,
                reset_diagnostics=base.diagnostics(adapter, reset.current),
            )
            video = imageio.get_writer(
                directory / "rollout.mp4", fps=20, macro_block_size=16, ffmpeg_params=["-threads", "2"]
            )
            for i, rgb in enumerate(adapter.last_rgb):
                phase = (
                    "conditioning" if i < raw_conditioning else ("expert" if prefix or control else "policy")
                )
                if phase == "conditioning" and not history:
                    phase = "conditioning (excluded)"
                video.append_data(base.annotated(rgb, task, episode, variant, prefix, 0, phase, "ongoing"))
            step_index, done = 0, False
            last_raw_frame = reset.current
            while step_index < cap and not done:
                expert = control or step_index < prefix
                if expert:
                    if step_index >= len(source["commands"]):
                        result["status"] = "expert_record_exhausted_without_terminal"
                        break
                    commands = [torch.as_tensor(source["commands"][step_index])]
                else:
                    if step_index == prefix:
                        result["handoff"] = {
                            "step": step_index,
                            "current_source_index": online.memory.current.source_index,
                            "execution_observations": online.memory.execution_observations,
                            "history_leaf_counts": {
                                s: sum(r.source == s for r in online.memory.archive.leaves)
                                for s in ("conditioning", "execution")
                            },
                            "diagnostics": base.diagnostics(adapter, last_raw_frame),
                        }
                        assert online.memory.execution_observations == prefix // ostride + 1
                        if task == "VideoUnmask" and episode == 0 and prefix == 32:
                            result["transport_parity"] = transport_parity(online, implementations)
                    rng_hash = hashlib.sha256(
                        online.generator.get_state().cpu().numpy().tobytes()
                    ).hexdigest()
                    chunk, timing = online.decide()
                    assert chunk.shape == (cfg["model"]["horizon"], 8) and torch.isfinite(chunk).all()
                    result["decisions"].append(
                        {
                            "step": step_index,
                            "chunk_raw": chunk.cpu().numpy(),
                            "timing": timing,
                            "rng_state_before_sha256": rng_hash,
                        }
                    )
                    commands = chunk[:execute].repeat_interleave(astride, dim=0)
                for action in commands:
                    if step_index >= cap:
                        break
                    outcome = adapter.step(action)
                    step_index += 1
                    if not outcome.frames and not outcome.done and not outcome.observation_skipped:
                        raise RuntimeError("Ongoing step without allowed or deliberately skipped observation")
                    for frame in outcome.frames:
                        online.observe(frame, adapter.last_preprocess_seconds / len(outcome.frames))
                    if not outcome.frames:
                        online.update_seconds += adapter.last_preprocess_seconds
                    last_raw_frame, raw_available = raw_frame(adapter, last_raw_frame, step_index)
                    entry = {
                        "step": step_index,
                        "phase": "expert" if expert else "policy",
                        "command": outcome.command,
                        "status": outcome.status,
                        "terminal": outcome.done,
                        "observation_available": raw_available,
                        "policy_observation_ingested": bool(outcome.frames),
                        "policy_current_source_index": online.memory.current.source_index,
                        "policy_execution_observations": online.memory.execution_observations,
                        **base.diagnostics(adapter, last_raw_frame),
                    }
                    if not raw_available:
                        entry["diagnostic_note"] = (
                            "No terminal observation; previous observed proprio excluded from aggregate positions."
                        )
                    if step_index <= len(source["step_proprio"]) and not outcome.done:
                        ref = source["step_proprio"][step_index - 1]
                        q = last_raw_frame.proprio.numpy()
                        entry["same_time_expert_joint_rmse_rad"] = float(
                            np.sqrt(np.mean((q[:7] - ref[:7]) ** 2))
                        )
                        entry["same_time_expert_tcp_distance_mm"] = float(
                            np.linalg.norm(q[7:10] - ref[7:10]) * 1000
                        )
                        if expert:
                            error = float(np.max(np.abs(q[:7] - ref[:7])))
                            assert error < 1e-4, ("expert replay mismatch", step_index, error)
                            result["prefix_replay_checks"].append(
                                {"step": step_index, "joint_max_abs_difference_rad": error}
                            )
                    result["timeline"].append(entry)
                    for rgb in adapter.last_rgb:
                        video.append_data(
                            base.annotated(
                                rgb,
                                task,
                                episode,
                                variant,
                                prefix,
                                step_index,
                                entry["phase"],
                                outcome.status,
                            )
                        )
                    if (
                        step_index
                        in (prefix, prefix + 4, prefix + 16, prefix + 32, prefix + 64, 128, 192, 256)
                        and adapter.last_rgb
                    ):
                        imageio.imwrite(
                            directory / f"frame_{step_index:04d}.png",
                            base.annotated(
                                adapter.last_rgb[-1],
                                task,
                                episode,
                                variant,
                                prefix,
                                step_index,
                                entry["phase"],
                                outcome.status,
                            ),
                        )
                    if step_index % 64 == 0 or outcome.done:
                        base.save(directory / "progress.json", result)
                        print(
                            json.dumps(
                                {
                                    "event": "progress",
                                    "task": task,
                                    "episode": episode,
                                    "variant": variant,
                                    "prefix": prefix,
                                    "step": step_index,
                                    "height_m": entry["target_height_m"],
                                    "elapsed_seconds": time.perf_counter() - started,
                                }
                            ),
                            flush=True,
                        )
                    if outcome.done:
                        result.update(
                            status=outcome.status,
                            terminal_error=outcome.error,
                            policy_handoff_occurred="handoff" in result,
                            terminal_during_expert_prefix=expert and not control,
                        )
                        done = True
                        break
            result.setdefault("status", "stopped_at_declared_diagnostic_cap")
    except Exception as exc:
        result.update(
            status="diagnostic_error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()
        )
        print(result["traceback"], flush=True)
    finally:
        if video is not None:
            video.close()
        if adapter is not None:
            adapter.close()
        result["elapsed_seconds"] = time.perf_counter() - started
        result["summary"] = base.summarize(result)
        base.save(result_path, result)
        print(
            json.dumps(
                {
                    "event": "finished",
                    "task": task,
                    "episode": episode,
                    "variant": variant,
                    "prefix": prefix,
                    "control": control,
                    "status": result["status"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    **result["summary"],
                }
            ),
            flush=True,
        )
    if result["status"] in ("diagnostic_error", "expert_record_exhausted_without_terminal"):
        raise RuntimeError(result.get("error", result["status"]))
    return result


def run_locked(*args, **kwargs):
    # Both launcher lanes may reach completed cases; one writer per trial.
    key = f"{kwargs['variant']}_{kwargs['task']}_{kwargs['episode']}_{kwargs.get('prefix', 0)}_{int(kwargs.get('control', False))}"
    locks = ROOT / "locks"
    locks.mkdir(exist_ok=True)
    with (locks / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return run(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--task", choices=("VideoUnmask", "PickXtimes"))
    parser.add_argument("--episodes", nargs="+", type=int)
    parser.add_argument("--prefixes", nargs="+", type=int, default=list(PREFIXES))
    parser.add_argument("--skip-expert-controls", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    configure(manifest)
    variants = args.labels or list(VARIANTS)
    if not variants or not set(variants) <= set(VARIANTS):
        parser.error("Requested labels must be in the checkpoint manifest")
    if not set(args.prefixes) <= set(PREFIXES):
        parser.error("Prefixes must belong to the fixed0/4/16/32/64 protocol")
    if args.report_only:
        from stmem.inspection.handoff_validation import validate_suite
        from stmem.inspection.handoff_report import build_report

        validate_suite(manifest)
        build_report(manifest)
        return
    torch.set_num_threads(2)
    started = time.perf_counter()
    for variant in variants:
        checkpoint = Path(CHECKPOINTS[variant]["path"])
        if file_sha256(checkpoint) != CHECKPOINTS[variant]["sha256"]:
            raise ValueError(f"Changed checkpoint{checkpoint}")
        policy, state = load_policy(checkpoint, manifest.get("device", "cuda:0"))
        policy.eval().requires_grad_(False)
        cfg = deepcopy(state["config"])
        if resolve_visual_features(cfg)["mode"] != "xyzrgb":
            cfg["model"]["visual_features"]["checkpoint"] = str(ASSETS["clip_checkpoint"])
        validate_policy_config(state["config"], cfg)
        from stmem.doctor import choose_precision

        if cfg["training"]["precision"] == "auto":
            cfg["training"]["precision"] = choose_precision(policy.device)
        identity = {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "step": state["step"],
            "horizon": cfg["model"]["horizon"],
            "history_mode": cfg["model"].get("history_mode", "full"),
            "action_stride": cfg["model"].get("action_stride", 1),
            "observation_stride": cfg["data"].get("observation_stride", 1),
        }
        atomic_json(identity, ROOT / f"checkpoint_{variant}.json")
        atomic_json(cfg, ROOT / f"config_{variant}.json")
        extractor = (
            FrozenCLIPFeatures(cfg, device=policy.device)
            if resolve_visual_features(cfg)["mode"] != "xyzrgb"
            else None
        )
        implementations = fast_transfer(policy)
        for task, episode in SCENES:
            if (args.task and args.task != task) or (args.episodes and episode not in args.episodes):
                continue
            source = source_for(task, episode)
            if not args.skip_expert_controls and variant == VARIANTS[0] and task == "VideoUnmask":
                result = run_locked(
                    policy,
                    cfg,
                    state,
                    identity,
                    extractor,
                    implementations,
                    source,
                    task=task,
                    episode=episode,
                    prefix=0,
                    variant="expert",
                    control=True,
                )
                if result["status"] != "success":
                    raise RuntimeError(f"Expert replay control failed:{task}/{episode}:{result['status']}")
            for prefix in args.prefixes:
                run_locked(
                    policy,
                    cfg,
                    state,
                    identity,
                    extractor,
                    implementations,
                    source,
                    task=task,
                    episode=episode,
                    prefix=prefix,
                    variant=variant,
                )
        del policy, state, extractor, implementations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "event": "requested_handoff_cases_finished",
                "variants": variants,
                "elapsed_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
    if not args.labels and not args.task and args.episodes is None and tuple(args.prefixes) == PREFIXES:
        from stmem.inspection.handoff_validation import validate_suite
        from stmem.inspection.handoff_report import build_report

        validate_suite(manifest)
        build_report(manifest)


if __name__ == "__main__":
    main()
