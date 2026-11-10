"""One-case exact-state continuation of a frozen, discovery-selected first chunk.

This is an exploratory expert-prefix handoff, never a benchmark-from-reset
score. Only the first chunk differs. Every subsequent chunk uses the same
ordinary UNIFORM reader over each branch's genuinely updated causal memory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from stmem.config import digest, source_revision
from stmem.data.center_sampling import center_counts, center_sampler_settings
from stmem.data.pointcloud import point_sampler_settings, proprioception
from stmem.evaluation.robomme_adapter import OBSERVATION_FIELDS, RoboMMEAdapter, numpy_value
from stmem.evaluation.runner import OnlinePolicy, episode_timeout
from stmem.inspection import handoff_helpers as helpers
from stmem.inspection.handoff import raw_frame
from stmem.inspection.refinement_data import COMPACT_SHA, read_json, stable_seed
from stmem.inspection.refinement_generated import verify_plan
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import load_policy
from stmem.vision.frozen import FrozenCLIPFeatures

BRANCHES = ("base", "immediate_completed", "packet_completed", "dispersed")
SCHEMA = "refinement_first_chunk_continuation_v1"


def predeclared_state(manifest):
    eligible = [state for state in manifest["states"] if state["task"] == "VideoUnmask"
                and state["episode_id"] == "episode_0" and state.get("cache_eligible", True)
                and state["decision_index"] >= 32]
    if not eligible:
        raise ValueError("No eligible local VideoUnmask episode-0 decision at or after native step 32")
    return min(eligible, key=lambda state: (state["decision_index"], state["state_id"]))


def held_commands(chunk):
    """H16 at 10 Hz; execute exactly ten targets, each held for two native steps."""
    chunk = np.asarray(chunk)
    if chunk.shape != (16, 8) or not np.isfinite(chunk).all():
        raise ValueError("Expected one finite H16 eight-coordinate chunk")
    return np.repeat(chunk[:10], 2, axis=0)


def verify_joint_state(observed, expected, label):
    error = float(np.max(np.abs(np.asarray(observed)[:7] - np.asarray(expected)[:7])))
    if not np.isfinite(error) or error >= 1e-4:
        raise ValueError(f"{label}: expert replay joint mismatch {error}")
    return error


def ingest_step(adapter, online, command):
    outcome = adapter.step(torch.as_tensor(command))
    if not outcome.frames and not outcome.done and not outcome.observation_skipped:
        raise RuntimeError("Ongoing simulator step omitted an observation without cadence justification")
    for frame in outcome.frames:
        online.observe(frame, adapter.last_preprocess_seconds / len(outcome.frames))
    if not outcome.frames:
        online.update_seconds += adapter.last_preprocess_seconds
    return outcome


def observed_choice(adapter, frame):
    """Privileged outcome labels only; these never enter any model or reader."""
    result = helpers.diagnostics(adapter, frame)
    env = adapter.env.unwrapped
    positions = np.asarray([numpy_value(actor.pose.p).reshape(-1, 3)[0] for actor in env.spawned_bins])
    result.update(container_heights_m=positions[:, 2].tolist(),
                  grasped_container_indices=[i for i, actor in enumerate(env.spawned_bins)
                      if bool(numpy_value(env.agent.is_grasping(actor)).reshape(-1)[0])],
                  tcp_nearest_container=int(np.linalg.norm(positions[:, :2] - frame.proprio.numpy()[7:9], axis=-1).argmin()))
    return result


def validate_inputs(plan_path, assets_config, *, draw=0, cap=320, seconds=300):
    plan_path = Path(plan_path).resolve()
    plan, manifest = verify_plan(plan_path)
    if (manifest["kind"] != "limited_real_data_diagnostic" or set(plan["students"]) != {"S0"}
            or plan.get("scope") != "limited_asset_exploratory_single_state"):
        raise ValueError("This bounded wrapper requires an explicitly limited real-data S0 diagnostic")
    chosen = predeclared_state(manifest)
    if len(plan["cases"]) != 1 or plan["cases"][0]["state_id"] != chosen["state_id"]:
        raise ValueError("Case must be the gain-blind first eligible VideoUnmask decision >=32")
    case = plan["cases"][0]
    if case["state"]["partition"] != chosen["partition"] or (case["reader"], case["background"]) != ("TREE", 120):
        raise ValueError("Do not relabel the original partition or change the fixed TREE/B120 case")
    if chosen["decision_index"] % 2 or not chosen["decision_index"] + 20 <= cap <= 640 or not 1 <= seconds <= 900:
        raise ValueError("Invalid bounded native-command/wall-time limit or observation phase")
    generated_path = plan_path.parent / (plan_path.stem + "_results") / "S0" / f"{chosen['state_id']}.npz"
    generated = read_json(generated_path.with_suffix(".json"))
    if generated["plan_id"] != plan["plan_id"] or generated["arrays_sha256"] != file_sha256(generated_path):
        raise ValueError("Generated chunk identity or arrays changed")
    if generated["contexts"] != case["students"]["S0"]["contexts"]:
        raise ValueError("Generated contexts differ from the frozen discovery selections")
    if (generated["seeds"] != case["generation_seeds"] or generated["integration_steps"] != 8
            or generated["prefix_targets"] != 10 or generated["precision"] != plan["precision"]):
        raise ValueError("Generated noise or action/precision protocol differs from the frozen plan")
    if not 0 <= draw < len(case["generation_seeds"]):
        raise ValueError("Requested generation draw is unavailable")
    with np.load(generated_path, allow_pickle=False) as data:
        chunks = {branch: np.array(data[f"{branch}/physical_actions"][draw], copy=True) for branch in BRANCHES}
    for chunk in chunks.values():
        held_commands(chunk)
    checkpoint = Path(plan["students"]["S0"]["checkpoint"])
    if plan["students"]["S0"]["sha256"] != COMPACT_SHA:
        raise ValueError("Continuation requires unchanged accepted S0")
    repo = Path(__file__).resolve().parents[3]
    assets = {key: str((repo / value).resolve()) for key, value in yaml.safe_load(Path(assets_config).read_text()).items()}
    cfg = deepcopy(manifest["configuration"])
    expected_prepared = Path(cfg["paths"]["data_root"]) / cfg["data"]["prepared_subdir"] / "VideoUnmask.h5"
    if Path(assets["prepared"]).resolve() != expected_prepared.resolve():
        raise ValueError("Replay expert source differs from the frozen cached observation source")
    cfg["model"]["visual_features"]["checkpoint"] = assets["clip_checkpoint"]
    cfg["model"]["execute"] = 10
    cfg["model"]["flow_steps"] = 8
    cfg["training"]["precision"] = plan["precision"]
    if cfg["model"].get("action_stride") != 2 or cfg["data"].get("observation_stride") != 2:
        raise ValueError("Required action and observation cadence is 10 Hz over native 20 Hz")
    record = {"schema": SCHEMA, "plan_id": plan["plan_id"], "source": source_revision(),
              "state": chosen, "scope": "limited_exploratory_expert_prefix_handoff_not_benchmark_score",
              "state_selection": "first eligible VideoUnmask episode_0 native decision >=32, before reading any losses",
              "checkpoint": str(checkpoint), "checkpoint_sha256": COMPACT_SHA,
              "generated_arrays": str(generated_path), "generated_arrays_sha256": generated["arrays_sha256"],
              "generated_metadata_sha256": file_sha256(generated_path.with_suffix(".json")),
              "contexts": case["students"]["S0"]["contexts"], "tie_sets": case["students"]["S0"]["tie_sets"],
              "first_chunk_noise_seed": case["generation_seeds"][draw], "generation_draw": draw,
              "continuation_noise_seed": stable_seed(plan["plan_id"], chosen["state_id"], draw, "common_uniform_continuation"),
              "continuation_reader": "UNIFORM", "continuation_budget": 128,
              "execute_targets": 10, "native_hold_steps": 2, "integration_steps": 8,
              "cap_total_native_commands": cap, "wall_seconds_per_branch": seconds, "precision": plan["precision"],
              "assets": {key: {"path": assets[key], "sha256": file_sha256(assets[key])}
                         for key in ("prepared", "alignment", "clip_checkpoint")}}
    record["identity"] = digest(record)
    return record, cfg, assets, chunks


@torch.inference_mode()
def run_branch(policy, cfg, extractor, source, record, branch, chunk, output):
    directory = Path(output) / branch
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        previous = read_json(result_path)
        if previous["identity"] != record["identity"]:
            raise ValueError("Existing continuation result belongs to a changed experiment")
        return previous
    prefix = record["state"]["decision_index"]
    started, adapter, video = time.monotonic(), None, None
    result = {"identity": record["identity"], "branch": branch, "timeline": [], "decisions": [],
              "prefix_replay_checks": [], "video": str(directory / "rollout.mp4")}
    try:
        with episode_timeout(record["wall_seconds_per_branch"]):
            adapter = RoboMMEAdapter("VideoUnmask", 0, "train", points=cfg["model"]["points"],
                max_steps=cfg["evaluation"]["max_steps"], crop=cfg["geometry"]["crop"],
                voxel_size=cfg["geometry"]["voxel_size"], max_depth_m=cfg["geometry"]["max_depth"],
                point_sampling=point_sampler_settings(cfg), center_sampling=center_sampler_settings(cfg),
                center_counts=center_counts(cfg), feature_extractor=extractor, observation_stride=2,
                include_conditioning=True)
            reset = adapter.reset()
            if reset.goal != source["goal"]:
                raise ValueError("Reset task goal differs from the genuine expert episode")
            result["reset_joint_max_error"] = verify_joint_state(reset.current.proprio.numpy(), source["reset_proprio"], "reset")
            raw_conditioning = len(adapter.last_rgb) - 1
            expected = source["conditioning_proprio"]
            if raw_conditioning != len(expected):
                raise ValueError("Reset conditioning length differs from expert record")
            errors = []
            for i, q in enumerate(expected):
                allowed = {field: numpy_value(adapter.last_raw_observation[field + "_list"][i]) for field in OBSERVATION_FIELDS}
                errors.append(verify_joint_state(proprioception(allowed).numpy(), q, f"conditioning {i}"))
            result["conditioning_joint_max_error"] = max(errors, default=0.)
            online = OnlinePolicy(policy, cfg, "UNIFORM", 128)
            online.reset(reset, record["continuation_noise_seed"])
            video = imageio.get_writer(directory / "rollout.mp4", fps=20, macro_block_size=16,
                                       ffmpeg_params=["-threads", "2"])
            for i, rgb in enumerate(adapter.last_rgb):
                video.append_data(helpers.annotated(rgb, "VideoUnmask", 0, branch, prefix, 0,
                                                   "conditioning" if i < raw_conditioning else "expert", "ongoing"))
            step, done, first_sent = 0, False, False
            latest = reset.current
            while step < record["cap_total_native_commands"] and not done:
                if step < prefix:
                    phase, commands = "expert", [source["commands"][step]]
                elif not first_sent:
                    if online.memory.current.source_index != prefix or online.memory.execution_observations != prefix // 2 + 1:
                        raise ValueError("Handoff current/history does not match the genuine causal prefix")
                    result["handoff"] = {"native_step": step, "source_index": online.memory.current.source_index,
                        "history_leaf_count": len(online.memory.archive.leaves),
                        "joint_max_error": verify_joint_state(latest.proprio.numpy(), source["step_proprio"][prefix-1], "handoff"),
                        "diagnostics": observed_choice(adapter, latest)}
                    phase, commands, first_sent = "intervention", held_commands(chunk), True
                    result["decisions"].append({"native_step": step, "reader": branch,
                        "context_id": record["contexts"][branch], "generated_chunk": np.asarray(chunk).tolist(),
                        "noise_seed": record["first_chunk_noise_seed"]})
                else:
                    phase = "uniform"
                    generated, timing = online.decide()
                    commands = held_commands(generated.cpu().numpy())
                    result["decisions"].append({"native_step": step, "reader": "UNIFORM",
                        "timing": timing, "generated_chunk": generated.cpu().tolist()})
                for command in commands:
                    if step >= record["cap_total_native_commands"]:
                        break
                    outcome = ingest_step(adapter, online, command)
                    step += 1
                    latest, available = raw_frame(adapter, latest, step)
                    entry = {"native_step": step, "phase": phase, "status": outcome.status,
                        "command": numpy_value(outcome.command).tolist(), "observation_available": available,
                        "current_source_index": online.memory.current.source_index,
                        "execution_observations": online.memory.execution_observations,
                        **observed_choice(adapter, latest)}
                    if phase == "expert" and not outcome.done:
                        error = verify_joint_state(latest.proprio.numpy(), source["step_proprio"][step-1], f"prefix {step}")
                        result["prefix_replay_checks"].append({"native_step": step, "joint_max_error": error})
                    result["timeline"].append(entry)
                    for rgb in adapter.last_rgb:
                        video.append_data(helpers.annotated(rgb, "VideoUnmask", 0, branch, prefix, step, phase, outcome.status))
                    if outcome.done:
                        result["status"] = outcome.status
                        result["terminal_during_expert_prefix"] = phase == "expert"
                        done = True
                        break
            result.setdefault("status", "stopped_at_declared_diagnostic_cap")
            result["final_native_step"] = step
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if video is not None:
            video.close()
        if adapter is not None:
            adapter.close()
        result["elapsed_seconds"] = time.monotonic() - started
        observed = [row for row in result["timeline"] if row["phase"] != "expert" and row["observation_available"]]
        result["choice_summary"] = {"first_grasped_container_indices": next((row["grasped_container_indices"]
            for row in observed if row["grasped_container_indices"]), []),
            "max_target_height_m": max((row["target_height_m"] for row in observed), default=None),
            "last_nearest_container": observed[-1]["tcp_nearest_container"] if observed else None}
        helpers.save(result_path, result)
    return result


def run(plan_path, output, *, assets_config="configs/inspection.yaml", draw=0, cap=320, seconds=300, device="cuda:0"):
    record, cfg, assets, chunks = validate_inputs(plan_path, assets_config, draw=draw, cap=cap, seconds=seconds)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and read_json(path) != record:
        raise ValueError("Continuation inputs changed; use a new output directory")
    atomic_json(record, path)
    policy, _ = load_policy(record["checkpoint"], device)
    policy.freeze_encoders()
    policy.eval().requires_grad_(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    extractor = FrozenCLIPFeatures(cfg, device=policy.device)
    source = helpers.source_for("VideoUnmask", 0, assets)
    if source["sha256"] != record["assets"]["prepared"]["sha256"] or len(source["commands"]) < record["state"]["decision_index"]:
        raise ValueError("Expert prefix source changed or is too short")
    rows = [run_branch(policy, cfg, extractor, source, record, branch, chunks[branch], output) for branch in BRANCHES]
    summary = {"identity": record["identity"], "scope": record["scope"], "original_partition": record["state"]["partition"],
        "state": record["state"], "branches": [{k: row.get(k) for k in
        ("branch", "status", "error", "choice_summary", "elapsed_seconds", "video")} for row in rows],
        "note": "One expert-prefix handoff and one intervened chunk; later UNIFORM history updates are causal. This is not autonomous benchmark success or evidence of long-term learned routing."}
    atomic_json(summary, output / "summary.json")
    lines = ["# One-chunk resolution continuation diagnostic", "", summary["note"], "",
        f"Original panel partition: **{summary['original_partition']}**. Task: VideoUnmask episode0; native expert prefix: {record['state']['decision_index']}.", "",
        "| First context | Outcome | First grasped container indices | Maximum target height (m) | Video |", "|---|---|---|---:|---|"]
    for row in rows:
        choice = row["choice_summary"]
        lines.append(f"| {row['branch']} | {row['status']} | {choice['first_grasped_container_indices']} | "
                     f"{choice['max_target_height_m']} | [video]({row['branch']}/rollout.mp4) |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assets-config", default="configs/inspection.yaml")
    parser.add_argument("--draw", type=int, default=0)
    parser.add_argument("--cap-native-commands", type=int, default=320)
    parser.add_argument("--seconds-per-branch", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(run(args.plan, args.output, assets_config=args.assets_config, draw=args.draw,
        cap=args.cap_native_commands, seconds=args.seconds_per_branch, device=args.device), indent=2))


if __name__ == "__main__":
    main()
