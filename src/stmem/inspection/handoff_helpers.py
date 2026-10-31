"""Diagnostic-only handoff labels, video overlays, and expert source readers.

Privileged object/grasp labels stay in the saved report, never policy inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from stmem.data.canonical import action_alignment_offset
from stmem.data.dataset import EpisodeDataset
from stmem.evaluation.robomme_adapter import numpy_value
from stmem.training.checkpoints import file_sha256


def converted(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial.json")
    temp.write_text(json.dumps(data, indent=2, default=converted) + "\n")
    temp.replace(path)


def diagnostics(adapter, frame):
    """Read-only labels: never call evaluate/task functions to advance progress."""
    env = adapter.env.unwrapped
    target = env.bin_0 if adapter.task == "VideoUnmask" else env.target_cube
    xyz = numpy_value(target.pose.p).reshape(-1, 3)[0]
    tcp = frame.proprio.numpy()[7:10]
    record = {
        "target_actor": "bin_0" if adapter.task == "VideoUnmask" else "target_cube",
        "target_xyz": xyz.copy(),
        "tcp_xyz": tcp.copy(),
        "tcp_target_xy_distance_mm": float(np.linalg.norm(tcp[:2] - xyz[:2]) * 1000),
        "tcp_target_distance_mm": float(np.linalg.norm(tcp - xyz) * 1000),
        "target_height_m": float(xyz[2]),
        "target_grasped": bool(numpy_value(env.agent.is_grasping(target)).reshape(-1)[0]),
        "observed_proprio": frame.proprio.numpy().copy(),
        "source_index": frame.source_index,
    }
    if adapter.task == "VideoUnmask":
        record["other_bin_heights_m"] = [
            float(numpy_value(b.pose.p).reshape(-1, 3)[0, 2]) for b in env.spawned_bins[1:]
        ]
    for key in ("current_task_index", "current_task_index_online", "current_task_name_online"):
        value = getattr(env, key, None)
        if isinstance(value, (str, int, float, bool)):
            record[key] = value
    return record


def annotated(rgb, task, episode, variant, prefix, step, phase, status):
    # Add a label strip; never paint labels into the policy's RGB-D input.
    image = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + 48), "#111827")
    image.paste(Image.fromarray(rgb), (0, 48))
    draw = ImageDraw.Draw(image)
    draw.text((8, 5), f"{task} train {episode} | {variant} | expert prefix {prefix}", fill="white")
    color = "#fbbf24" if phase in ("conditioning", "expert") else "#5eead4"
    draw.text((8, 22), f"{phase.upper()} | execution step {step} | {status}", fill=color)
    return np.asarray(image)


def summarize(result):
    rows = [r for r in result["timeline"] if r["phase"] == "policy" and r["observation_available"]]
    if not rows:
        rows = [r for r in result["timeline"] if r["observation_available"]]
    if not rows:
        return {}
    closing = [r for r in rows if r["command"][7] < 0]
    threshold = 0.15 if result["task"] == "VideoUnmask" else 0.05
    return {
        "policy_commands": sum(r["phase"] == "policy" for r in result["timeline"]),
        "expert_commands": sum(r["phase"] == "expert" for r in result["timeline"]),
        "first_target_grasp_step": next((r["step"] for r in rows if r["target_grasped"]), None),
        "any_target_grasp": any(r["target_grasped"] for r in rows),
        "first_target_above_threshold_step": next(
            (r["step"] for r in rows if r["target_height_m"] > threshold), None
        ),
        "target_lift_threshold_m": threshold,
        "max_target_height_m": max(r["target_height_m"] for r in rows),
        "min_tcp_target_xy_distance_mm": min(r["tcp_target_xy_distance_mm"] for r in rows),
        "first_close_step": closing[0]["step"] if closing else None,
        "first_close_xy_distance_mm": closing[0]["tcp_target_xy_distance_mm"] if closing else None,
        "final_tcp_target_xy_distance_mm": rows[-1]["tcp_target_xy_distance_mm"],
    }


def source_for(task, episode, assets):
    """The exact fixed expert commands, excluding reset's automatic command."""
    if task == "VideoUnmask":
        path = Path(assets["prepared"])
        dataset = EpisodeDataset(path, tasks=[task], alignment_path=assets["alignment"])
        ep = dataset.load_episode(episode)
        if action_alignment_offset(ep.alignment) != 1:
            raise ValueError("This diagnostic protocol requires verified post-action expert records")
        valid = ep.action_valid[1:, :7].all(-1).numpy()
        length = int(np.argmax(~valid)) if (~valid).any() else len(valid)
        source = {
            "kind": "released_recorded_expert",
            "path": str(path),
            "goal": ep.goal,
            "commands": ep.actions[1 : 1 + length].numpy(),
            "reset_proprio": ep.execution[0].proprio.numpy(),
            "step_proprio": np.stack([f.proprio.numpy() for f in ep.execution[1 : 1 + length]]),
            "conditioning_proprio": np.stack([f.proprio.numpy() for f in ep.conditioning]),
            "alignment": dataset.alignment,
        }
    else:
        path = Path(assets["pick_prefix"])
        source = json.loads(path.read_text())
        if (source["task"], source["split"], source["episode_idx"]) != (task, "train", episode):
            raise ValueError("Pick expert prefix belongs to another scene")
        source["path"] = str(path)
        source["kind"] = "regenerated_official_planner_expert_not_released_record_identity"
        for key in ("commands", "reset_proprio", "step_proprio"):
            source[key] = np.asarray(source[key], dtype=np.float32)
    source["sha256"] = file_sha256(path)
    if len(source["commands"]) < 64 or len(source["commands"]) != len(source["step_proprio"]):
        raise ValueError("Expert source needs at least64 aligned command/observation pairs")
    if source["commands"].ndim != 2 or source["commands"].shape[1] != 8:
        raise ValueError("Expert command shape must be[N,8]")
    for key in ("commands", "reset_proprio", "step_proprio"):
        if not np.isfinite(source[key]).all():
            raise ValueError(f"Nonfinite expert source field{key}")
    return source
