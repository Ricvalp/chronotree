"""Replay an observed hover and test saved commands, without oracle control.

Replanning control replays the two original eight-target predictions. The
extension executes all sixteen targets from the first prediction. Their first
sixteen native commands must be identical. Simulator state is verified before
branching; state annotations never generate commands.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from stmem.evaluation.runner import episode_timeout
from stmem.inspection.button_probe import diagnostic_state, make_adapter
from stmem.training.checkpoints import atomic_json, file_sha256


def state_error(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("Replay simulator state keys differ")
        return max((state_error(actual[k], v) for k, v in expected.items()), default=0.)
    if isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError("Replay simulator state lengths differ")
        return max((state_error(a, b) for a, b in zip(actual, expected)), default=0.)
    actual = actual.detach().cpu().numpy() if isinstance(actual, torch.Tensor) else np.asarray(actual)
    expected = expected.detach().cpu().numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError("Replay simulator state shapes differ")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite simulator state")
    return float(np.max(np.abs(actual.astype(float) - expected.astype(float)), initial=0.))


def branch_commands(trial, row, mode):
    start = row["native_step"]
    stride = trial["action_stride"]
    horizon = len(row["physical_actions"])
    if mode == "replan8":
        commands = np.asarray(trial["actions"][start:start + horizon * stride])
    elif mode == "execute16":
        commands = np.repeat(np.asarray(row["physical_actions"]), stride, axis=0)
    elif mode == "teacher16":
        candidates = [p for p in row.get("probes", [])
                      if p["label"] == "accepted_teacher_on_same_history" and p["noise_seed"] == 2701]
        if len(candidates) != 1:
            raise ValueError("No fixed-seed teacher prediction at this state")
        commands = np.repeat(np.asarray(candidates[0]["physical_actions"]), stride, axis=0)
    else:
        raise ValueError("Unknown branch")
    prefix = trial["execute_actions"] * stride
    if mode in ("replan8", "execute16"):
        baseline = np.asarray(trial["actions"][start:start + prefix])
        if not np.array_equal(commands[:len(baseline)], baseline):
            raise ValueError("The compared branches do not share identical initial commands")
    if commands.ndim != 2 or commands.shape[1] != 8:
        raise ValueError("Missing branch commands")
    return commands


def annotated(rgb, text):
    image = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + 48), "#112131")
    image.paste(Image.fromarray(rgb), (0, 48))
    draw = ImageDraw.Draw(image)
    draw.text((8, 5), text[:95], fill="white")
    draw.text((8, 24), "Diagnostic replay | no button controller or oracle actions", fill="#f8ca60")
    return np.asarray(image)


def run_branch(cfg, trial, row, mode, directory, device):
    snapshot_path = Path(row["probe_snapshot"])
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    if len(snapshot["commands"]) != row["native_step"]:
        raise ValueError("Probe prefix length differs")
    if not np.array_equal(snapshot["commands"], trial["actions"][:row["native_step"]]):
        raise ValueError("Probe prefix differs from recorded rollout")
    commands = branch_commands(trial, row, mode)
    adapter = make_adapter(cfg, trial["task"], trial["episode_idx"], None, device, replay=True)
    video_path = directory / f"step{row['native_step']:04d}_{mode}.mp4"
    result = {"task": trial["task"], "episode_idx": trial["episode_idx"], "mode": mode,
              "native_step": row["native_step"], "snapshot_sha256": file_sha256(snapshot_path),
              "scope": "One saved chunk from a replay-verified state; not a rollout success-rate estimate",
              "video": str(video_path.resolve()), "timeline": [], "commands": [], "status": "error"}
    try:
        with episode_timeout(600):
            adapter.reset()
            if adapter.goal != snapshot["goal"]:
                raise ValueError("Replay goal changed")
            recent = deque(maxlen=80)
            for command in snapshot["commands"]:
                step = adapter.step(command)
                if step.done:
                    raise ValueError("Replay terminated before recorded probe")
                recent.extend(adapter.last_rgb)
            actual = diagnostic_state(adapter)
            error = state_error(adapter.env.unwrapped.get_state_dict(), snapshot["sim_state"])
            proprio_error = state_error(actual["proprio"], snapshot["diagnostics"]["proprio"])
            if max(error, proprio_error) > 1e-5 or actual["task_index"] != snapshot["diagnostics"]["task_index"]:
                raise ValueError(f"Replay mismatch: simulator={error}, proprio={proprio_error}")
            result.update(replay_simulator_max_error=error, replay_proprio_max_error=proprio_error,
                          start=actual, prefix_video_frames=len(recent))
            with imageio.get_writer(video_path, fps=20) as video:
                for rgb in recent:
                    video.append_data(annotated(rgb, f"{trial['task']} {trial['episode_idx']} | identical replay prefix"))
                for index, command in enumerate(commands):
                    step = adapter.step(command)
                    result["commands"].append(step.command.tolist())
                    if adapter.last_raw_observation and adapter.last_raw_observation.get("front_rgb_list"):
                        state = diagnostic_state(adapter)
                        state["terminal"] = step.done
                        result["timeline"].append(state)
                    for rgb in adapter.last_rgb:
                        video.append_data(annotated(rgb, f"{mode} | native {row['native_step']+index+1} | {step.status}"))
                    if step.done:
                        result["status"] = step.status
                        break
                else:
                    result["status"] = "ongoing_after_saved_chunk"
            if result["timeline"]:
                result["end"] = result["timeline"][-1]
                result["actual_descent_mm"] = 1000 * (actual["tcp_xyz"][2] - result["end"]["tcp_xyz"][2])
                result["actual_button_progress_mm"] = actual["button_distance_mm"] - result["end"]["button_distance_mm"]
                result["maximum_button_depth_mm"] = 1000 * max(r["button_depth_m"] for r in result["timeline"])
    except Exception as exc:
        import traceback
        result.update(error=str(exc), traceback=traceback.format_exc())
    finally:
        adapter.close()
        atomic_json(result, directory / f"step{row['native_step']:04d}_{mode}.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--modes", nargs="+", choices=("replan8", "execute16", "teacher16"),
                        default=["replan8", "execute16", "teacher16"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new branch output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    trial = json.loads(args.trial.read_text())
    cfg = json.loads(args.config.read_text())
    result = {"trial_sha256": file_sha256(args.trial), "branches": []}
    for step in args.steps:
        rows = [r for r in trial["decisions"] if r["native_step"] == step and r.get("probe_snapshot")]
        if len(rows) != 1:
            raise ValueError(f"No saved probe at native step {step}")
        for mode in args.modes:
            branch = run_branch(cfg, trial, rows[0], mode, args.output, args.device)
            result["branches"].append(branch)
            atomic_json(result, args.output / "results.json")
            print(json.dumps({k: v for k, v in branch.items() if k not in ("timeline", "commands", "start", "end")}), flush=True)


if __name__ == "__main__":
    main()
