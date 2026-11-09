"""Normal VideoUnmaskSwap rollouts with read-only container-choice diagnostics.

No observations, histories, target colors or simulator configurations are edited.
Each worker uses the standard OnlinePolicy and run_episode evaluation path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from stmem.training.checkpoints import atomic_json, file_sha256


TASK = "VideoUnmaskSwap"


def worker(output: Path, worker_index: int):
    import torch

    from stmem.data.center_sampling import center_counts, center_sampler_settings
    from stmem.data.pointcloud import point_sampler_settings
    from stmem.evaluation.robomme_adapter import RoboMMEAdapter
    from stmem.evaluation.runner import OnlinePolicy, run_episode
    from stmem.inspection.swap_targeting import SwapTargetingAdapter
    from stmem.training.runtime import load_policy
    from stmem.vision.frozen import FrozenCLIPFeatures

    torch.set_num_threads(4)
    manifest = json.loads((output / "manifest.json").read_text())
    cfg = json.loads((output / "config.json").read_text())
    checkpoint = manifest["checkpoint"]["path"]
    if file_sha256(checkpoint) != manifest["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint changed after planning")
    device = "cuda:0"
    torch.cuda.set_device(device)
    policy, _ = load_policy(checkpoint, device)
    policy.eval().requires_grad_(False)
    features = FrozenCLIPFeatures(cfg, device=device)
    online = OnlinePolicy(policy, cfg, "FULL", cfg["model"]["budget"])
    for episode_idx in manifest["planned_episodes"][worker_index::manifest["workers"]]:
        print(json.dumps({"event": "episode_started", "episode_idx": episode_idx}), flush=True)
        video = output / "videos" / f"{TASK}_{episode_idx:03d}.mp4"
        seed = cfg["evaluation"]["seed"] + cfg["data"]["tasks"].index(TASK) * 1000 + episode_idx
        adapter = None
        result = None
        try:
            adapter = SwapTargetingAdapter(RoboMMEAdapter(
                TASK, episode_idx, "val", points=cfg["model"]["points"],
                max_steps=cfg["evaluation"]["max_steps"], crop=cfg["geometry"]["crop"],
                voxel_size=cfg["geometry"]["voxel_size"], max_depth_m=cfg["geometry"]["max_depth"],
                point_sampling=point_sampler_settings(cfg),
                center_sampling=center_sampler_settings(cfg), center_counts=center_counts(cfg),
                feature_extractor=features, observation_stride=cfg["data"].get("observation_stride", 1),
                include_conditioning=True, render_device=device,
            ))
            result = run_episode(
                adapter, online, task=TASK, episode_idx=episode_idx, split="val", seed=seed,
                timeout_seconds=cfg["evaluation"]["timeout_seconds"], video_path=video,
                checkpoint_hash=manifest["checkpoint"]["sha256"],
            )
        except Exception as exc:
            result = {
                "task": TASK, "episode_idx": episode_idx, "split": "val", "method": "FULL",
                "status": "infrastructure_error", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(), "actions": [], "noise_seed": seed,
            }
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception as exc:
                    if result is not None:
                        result["close_error"] = f"{type(exc).__name__}: {exc}"
                        result["status"] = "infrastructure_error"
        if adapter is not None:
            # report() consumes only the observer's cached measurements, so
            # cleanup errors and the durable outcome retain the same status.
            result["choice_diagnostics"] = adapter.report(result["status"])
        result["video_path"] = str(video) if video.exists() else None
        result["execution_override"] = manifest["execution_override"]
        atomic_json(result, output / "episodes" / f"{TASK}_{episode_idx:03d}.json")
        print(json.dumps({"event": "episode_finished", "episode_idx": episode_idx,
                          "status": result["status"], "error": result.get("error"),
                          "runtime_seconds": result.get("runtime_seconds")}), flush=True)


def main():
    from stmem.config import source_revision
    from stmem.evaluation.execution import evaluation_execution_config
    from stmem.evaluation.metrics import summarize_episodes
    from stmem.tracking import RunTracker
    from stmem.training.runtime import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--execute-actions", type=int, default=10)
    parser.add_argument("--clip-checkpoint", type=Path, default=Path("cache/backbones/clip_RN50.pt"))
    parser.add_argument("--worker-index", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker_index is not None:
        worker(output, args.worker_index)
        return
    if args.checkpoint is None or not 1 <= args.episodes <= 50 or not 1 <= args.workers <= args.episodes:
        parser.error("Supply --checkpoint, 1–50 episodes and 1–episodes workers")
    if output.exists():
        parser.error("Use a fresh output directory; existing diagnostics are never overwritten")
    checkpoint = args.checkpoint.resolve()
    state = load_checkpoint(checkpoint)
    if state["stage"] != "teacher":
        parser.error("This diagnostic uses the full-history teacher")
    cfg, override = evaluation_execution_config(
        state["config"], state["config"], execute_actions=args.execute_actions, split="val",
    )
    cfg["model"]["visual_features"]["checkpoint"] = str(args.clip_checkpoint.resolve())
    cfg["paths"]["run_root"] = str(output)
    cfg["evaluation"].update(episode_count=args.episodes, video_cap=args.episodes)
    cfg["tracking"].update(project="chronotree-routers", entity="ricvalp", mode="online", dir=None)
    manifest = {
        "scope": "Normal closed-loop validation diagnostic; original conditioning and observations, no interventions",
        "task": TASK, "split": "val", "method": "FULL", "planned_episodes": list(range(args.episodes)),
        "workers": args.workers, "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint),
                                                 "step": state["step"], "stage": state["stage"]},
        "execution_override": override, "source": source_revision(),
        "choice_observation": "Read-only actor poses and contact flags; never passed into policy inputs",
    }
    del state
    output.mkdir(parents=True)
    for directory in ("episodes", "logs", "videos"):
        (output / directory).mkdir()
    atomic_json(manifest, output / "manifest.json")
    atomic_json(cfg, output / "config.json")
    planned = [(TASK, index) for index in manifest["planned_episodes"]]
    processes, logs = [], []
    with RunTracker(cfg, output, job_type="evaluation", name="VideoUnmaskSwap-normal-10-teacher",
                    metadata=manifest) as tracker:
        try:
            for index in range(args.workers):
                log = (output / "logs" / f"worker_{index}.log").open("w")
                logs.append(log)
                env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                       "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
                processes.append(subprocess.Popen(
                    [sys.executable, "-u", "-m", "stmem.inspection.swap_rollouts", "--output", str(output),
                     "--worker-index", str(index)], stdout=log, stderr=subprocess.STDOUT, env=env,
                ))
            count = -1
            while True:
                results = [json.loads(path.read_text()) for path in sorted((output / "episodes").glob("*.json"))]
                if len(results) != count:
                    count = len(results)
                    summary = summarize_episodes(results, planned)
                    atomic_json(summary, output / "summary.json")
                    tracker.log({"evaluation/completed_episodes": count,
                                 "evaluation/progress_percent": 100 * count / args.episodes,
                                 "evaluation/success_rate_percent": 100 * (summary["macro_success"] or 0)}, step=count)
                    print(json.dumps({"event": "evaluation_progress", "completed": count,
                                      "planned": args.episodes, "summary": summary["overall"]}), flush=True)
                if all(process.poll() is not None for process in processes):
                    # Read after process exit so the final atomic result cannot be missed.
                    results = [json.loads(path.read_text()) for path in sorted((output / "episodes").glob("*.json"))]
                    break
                time.sleep(3)
            summary = summarize_episodes(results, planned)
            summary["worker_exit_codes"] = [process.returncode for process in processes]
            atomic_json(summary, output / "summary.json")
            tracker.log({"evaluation/completed_episodes": len(results),
                         "evaluation/progress_percent": 100 * len(results) / args.episodes,
                         "evaluation/success_rate_percent": 100 * (summary["macro_success"] or 0)},
                        step=len(results))
            tracker.summary({"evaluation/summary": summary, "evaluation/output_directory": str(output)})
            if tracker.run is not None:
                import wandb

                table = wandb.Table(columns=["episode", "status", "target", "first_grasp", "first_lift", "video"])
                for row in results:
                    diag = row.get("choice_diagnostics", {})
                    choices = diag.get("summary", {})
                    video = row.get("video_path")
                    table.add_data(row["episode_idx"], row["status"], (diag.get("target") or {}).get("first_bin_index"),
                                   str(choices.get("first_grasp_bin_indices")), str(choices.get("first_lift_bin_indices")),
                                   wandb.Video(video, format="mp4") if video and Path(video).stat().st_size else None)
                tracker.run.log({"evaluation/episodes": table})
            from stmem.inspection.swap_rollout_report import write_report

            print(write_report(output), flush=True)
            if len(results) != args.episodes or any(code != 0 for code in summary["worker_exit_codes"]):
                raise RuntimeError("Incomplete diagnostic; inspect logs and planned denominator in summary.json")
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
