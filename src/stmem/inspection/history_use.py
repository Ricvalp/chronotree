"""Bounded paired validation rollouts for history use by a frozen teacher.

All arms use ordinary simulator resets and observations. Controls change only
the history records read by the action head; current input, text and clocks are
preserved at each decision. Their subsequent closed-loop states can diverge.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.inspection.history_resume import (
    case_path, interrupt_signals, result_seed, run_lock, seal_plan, seal_result,
    validate_existing, validate_plan, validate_result, verify_worker_lock, worker_lifetime,
)


ARMS = {
    "full": {"method": "FULL", "intervention": "none"},
    "erase_conditioning": {"method": "FULL", "intervention": "erase_conditioning"},
    "erase_execution": {"method": "FULL", "intervention": "erase_execution"},
    "current_only": {"method": "CURRENT", "intervention": "none"},
}
TASKS = ("SwingXtimes", "VideoRepick", "VideoPlaceOrder")


def run_worker(output, worker_index, worker_count=None):
    from stmem.config import source_revision

    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    cfg = json.loads((output / "config.json").read_text())
    validate_plan(output, {**manifest, "source": source_revision()}, cfg)
    worker_count = manifest["workers"] if worker_count is None else worker_count
    if not 0 <= worker_index < worker_count <= 4:
        raise ValueError("Invalid diagnostic worker index/count")
    cases = []
    for task, episode_idx in manifest["planned_scenes"][worker_index::worker_count]:
        for arm in manifest["arms"]:
            case = {"task": task, "episode_idx": episode_idx, "arm": arm}
            path = case_path(output, case)
            if path.exists():
                validate_result(path, case, manifest, cfg, video_root=output)
            else:
                cases.append(case)
    if not cases:
        return

    import torch

    from stmem.data.center_sampling import center_counts, center_sampler_settings
    from stmem.data.pointcloud import point_sampler_settings
    from stmem.evaluation.robomme_adapter import RoboMMEAdapter
    from stmem.evaluation.runner import OnlinePolicy, run_episode
    from stmem.inspection.history_observer import HistoryObserver
    from stmem.inspection.history_probe import ProbedOnlinePolicy
    from stmem.training.runtime import load_policy
    from stmem.vision.frozen import FrozenCLIPFeatures

    torch.set_num_threads(4)
    checkpoint = manifest["checkpoint"]["path"]
    if file_sha256(checkpoint) != manifest["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint changed after planning")
    device = "cuda:0"
    torch.cuda.set_device(device)
    policy, _ = load_policy(checkpoint, device)
    policy.eval().requires_grad_(False)
    features = FrozenCLIPFeatures(cfg, device=device)
    for case in cases:
        task, episode_idx, arm = case["task"], case["episode_idx"], case["arm"]
        options = manifest["arms"][arm]
        result_path = case_path(output, case)
        print(json.dumps({"event": "episode_started", **case}), flush=True)
        video = output / "videos" / arm / f"{task}_{episode_idx:03d}.mp4"
        seed = result_seed(cfg, case)
        leaf_permutation = options.get("implementation") == "full_leaf_context_v1"
        online = OnlinePolicy(policy, cfg, options["method"], cfg["model"]["budget"],
                              intervention="none" if leaf_permutation else options["intervention"])
        if arm == "full":
            online = ProbedOnlinePolicy(online, task=task, decision_indices=manifest["probe_decisions"])
        elif arm == "recent_execution":
            from stmem.inspection.history_window import RecentExecutionOnlinePolicy

            online = RecentExecutionOnlinePolicy(online, completed_blocks=options["completed_blocks"])
        elif leaf_permutation:
            from stmem.inspection.history_permutation import PermutedExecutionOnlinePolicy

            online = PermutedExecutionOnlinePolicy(online, seed=cfg["evaluation"]["seed"])
        adapter, result = None, None
        try:
            adapter = HistoryObserver(RoboMMEAdapter(
                task, episode_idx, "val", points=cfg["model"]["points"],
                max_steps=cfg["evaluation"]["max_steps"], crop=cfg["geometry"]["crop"],
                voxel_size=cfg["geometry"]["voxel_size"], max_depth_m=cfg["geometry"]["max_depth"],
                point_sampling=point_sampler_settings(cfg), center_sampling=center_sampler_settings(cfg),
                center_counts=center_counts(cfg), feature_extractor=features,
                observation_stride=cfg["data"].get("observation_stride", 1),
                include_conditioning=True, render_device=device,
            ))
            result = run_episode(
                adapter, online, task=task, episode_idx=episode_idx, split="val", seed=seed,
                timeout_seconds=cfg["evaluation"]["timeout_seconds"], video_path=video,
                checkpoint_hash=manifest["checkpoint"]["sha256"],
            )
        except Exception as exc:
            result = {**case, "split": "val", "method": options["method"],
                      "status": "infrastructure_error", "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc(), "actions": [], "noise_seed": seed}
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception as exc:
                    if result is not None:
                        result["close_error"] = f"{type(exc).__name__}: {exc}"
                        result["status"] = "infrastructure_error"
        if adapter is not None:
            result["task_diagnostics"] = adapter.report(result["status"])
        if arm == "full":
            result["fixed_current_probes"] = online.probes
        result.update(case)
        result["checkpoint_sha256"] = manifest["checkpoint"]["sha256"]
        result["video_path"] = str(video) if video.is_file() else None
        result["execution_override"] = manifest["execution_override"]
        if result_path.exists():
            raise FileExistsError(f"Refusing to overwrite a published result: {result_path}")
        atomic_json(seal_result(result), result_path)
        print(json.dumps({"event": "episode_finished", **case, "status": result["status"],
                          "error": result.get("error"), "runtime_seconds": result.get("runtime_seconds")}),
              flush=True)


def collect(output, manifest):
    from stmem.evaluation.metrics import summarize_episodes

    results, summaries = [], {}
    for arm in manifest["arms"]:
        rows = [json.loads(path.read_text()) for path in sorted((output / "results" / arm / "episodes").glob("*.json"))]
        results.extend(rows)
        summaries[arm] = summarize_episodes(rows, [tuple(scene) for scene in manifest["planned_scenes"]])
    return results, {"per_arm": summaries, "recorded": len(results), "planned": len(manifest["planned_cases"])}


def main():
    from stmem.config import source_revision
    from stmem.evaluation.execution import evaluation_execution_config
    from stmem.tracking import RunTracker
    from stmem.training.runtime import load_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--execute-actions", type=int, default=10)
    parser.add_argument("--clip-checkpoint", type=Path, default=Path("cache/backbones/clip_RN50.pt"))
    parser.add_argument("--recent-swing", action="store_true",
                        help="Four-scene follow-up keeping two completed execution blocks plus pending history")
    parser.add_argument("--permuted-swing", action="store_true",
                        help="Four-scene follow-up preserving token counts while permuting execution payloads across time slots")
    parser.add_argument("--reference-output", type=Path,
                        help="Reuse completed matching FULL SwingXtimes results for --recent-swing")
    parser.add_argument("--resume", action="store_true",
                        help="Fill missing trials under the identical saved protocol; never retry error records")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--lock-fd", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker_index is not None:
        verify_worker_lock(output, args.lock_fd)
        run_worker(output, args.worker_index, args.worker_count)
        return
    if args.checkpoint is None or not 1 <= args.episodes <= 10 or not 1 <= args.workers <= 4:
        parser.error("Supply checkpoint, 1–10 validation episodes per task, and 1–4 workers")
    if output.exists() and not args.resume:
        parser.error("Use --resume for matching saved work, or a fresh output directory")
    if args.resume and not (output / "manifest.json").is_file():
        parser.error("--resume requires an existing planned diagnostic")
    followup = args.recent_swing or args.permuted_swing
    if args.recent_swing and args.permuted_swing:
        parser.error("Choose exactly one follow-up")
    if followup != (args.reference_output is not None):
        parser.error("A Swing follow-up requires --reference-output")
    checkpoint = args.checkpoint.resolve()
    state = load_checkpoint(checkpoint)
    if state["stage"] != "teacher":
        parser.error("This diagnostic uses the selected full-history teacher")
    cfg, override = evaluation_execution_config(state["config"], state["config"],
                                                 execute_actions=args.execute_actions, split="val")
    cfg["model"]["visual_features"]["checkpoint"] = str(args.clip_checkpoint.resolve())
    cfg["paths"]["run_root"] = str(output)
    cfg["evaluation"].update(episode_count=args.episodes, video_cap=3 * args.episodes)
    cfg["tracking"].update(project="chronotree-routers", entity="ricvalp", mode="online", dir=None)
    tasks = ("SwingXtimes",) if followup else TASKS
    arms = ({"full": ARMS["full"], "recent_execution": {
        "method": "FULL", "intervention": "none", "completed_blocks": 2,
    }} if args.recent_swing else ARMS)
    if args.permuted_swing:
        arms = {"full": ARMS["full"], "permute_execution": {
            "method": "FULL", "intervention": "permute_execution",
            "implementation": "full_leaf_context_v1",
        }}
    scenes = [[task, index] for task in tasks for index in range(args.episodes)]
    manifest = {
        "scope": "Paired frozen-teacher history-use diagnostic on fixed official validation configurations",
        "tasks": list(tasks), "split": "val", "arms": arms,
        "planned_scenes": scenes,
        "planned_cases": [{"task": task, "episode_idx": index, "arm": arm}
                          for task, index in scenes for arm in arms],
        "probe_decisions": [0, 2, 5, 10, 20], "workers": args.workers,
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint),
                       "step": state["step"], "stage": state["stage"]},
        "execution_override": override, "source": source_revision(),
        "selection_reason": "Tasks with previously observed teacher competence, spanning execution counting, demonstrated identity and order; validation indices fixed before this comparison",
        "limitations": ["History erasure can be out of distribution for the teacher",
                        "Closed-loop trajectories may diverge; fixed-current probes separately hold observations and decoder noise constant",
                        "Small diagnostic subset; no test-seed tuning and no architecture optimization",
                        "Walltime includes read-only probes and observation logging; not a latency benchmark"],
    }
    references = []
    if args.reference_output is not None:
        reference = args.reference_output.resolve()
        prior = json.loads((reference / "manifest.json").read_text())
        if (prior["checkpoint"]["sha256"] != manifest["checkpoint"]["sha256"]
                or prior["execution_override"] != override
                or prior["source"]["source_sha256"] != manifest["source"]["source_sha256"]):
            raise ValueError("Reference policy/execution protocol differs")
        prior_cfg = json.loads((reference / "config.json").read_text())
        if any(prior_cfg[key] != cfg[key] for key in ("model", "data", "geometry", "training", "seed")):
            raise ValueError("Reference input or policy configuration differs")
        if prior_cfg["evaluation"]["seed"] != cfg["evaluation"]["seed"]:
            raise ValueError("Reference action-noise seed differs")
        for task, index in scenes:
            path = reference / "results" / "full" / "episodes" / f"{task}_{index:03d}.json"
            record = validate_result(path, {"task": task, "episode_idx": index, "arm": "full"},
                                     manifest, cfg, require_seal=False, video_root=reference)
            references.append((path, record))
        manifest["reference_output"] = str(reference)
        manifest["reference_results"] = {str(path): file_sha256(path) for path, _ in references}
        manifest["reference_videos"] = {record["video_path"]: file_sha256(record["video_path"])
                                         for _, record in references}
        manifest["selection_reason"] = (
            "Follow-up on the same four predeclared SwingXtimes scenes: preserve recent motor history while removing older execution evidence"
            if args.recent_swing else
            "Follow-up on the same four predeclared SwingXtimes scenes: retain all historical content/geometry, source/modality counts and time slots, while shuffling the association between execution payloads and recorded time"
        )
    del state
    with run_lock(output) as lock_fd, interrupt_signals():
        if args.resume:
            manifest, cfg = validate_plan(output, manifest, cfg)
            validate_existing(output, manifest, cfg)
        else:
            if (output / "manifest.json").exists():
                raise FileExistsError("Another coordinator already planned this output; use --resume")
            atomic_json(manifest, output / "manifest.json")
            atomic_json(cfg, output / "config.json")
            seal_plan(output, manifest, cfg)
        for arm in arms:
            (output / "results" / arm / "episodes").mkdir(parents=True, exist_ok=True)
            (output / "videos" / arm).mkdir(parents=True, exist_ok=True)
        copy_references(output, references, manifest, cfg)
        run_coordinator(output, args, manifest, cfg, lock_fd, RunTracker)


def copy_references(output, references, manifest, cfg):
    """Copy immutable FULL controls, preserving any already-published copies."""
    for path, record in references:
        destination_result = output / "results" / "full" / "episodes" / path.name
        if destination_result.exists():
            case = {"task": record["task"], "episode_idx": record["episode_idx"], "arm": "full"}
            existing = validate_result(destination_result, case, manifest, cfg, video_root=output)
            if existing.get("reused_reference_result") != str(path):
                raise ValueError("Existing FULL result is not the declared copied reference")
            continue
        if file_sha256(path) != manifest["reference_results"][str(path)]:
            raise ValueError("Reference result changed during planning")
        source_video = Path(record["video_path"])
        expected_video_hash = manifest["reference_videos"][str(source_video)]
        if file_sha256(source_video) != expected_video_hash:
            raise ValueError("Reference video changed during planning")
        destination = output / "videos" / "full" / source_video.name
        temporary = destination.with_suffix(".copying.mp4")
        shutil.copy2(source_video, temporary)
        if file_sha256(temporary) != expected_video_hash:
            raise ValueError("Copied reference video does not match")
        os.replace(temporary, destination)
        record["reused_reference_result"] = str(path)
        record["video_path"] = str(destination)
        atomic_json(seal_result(record), destination_result)


def run_coordinator(output, args, manifest, cfg, lock_fd, tracker_class):
    """One bounded attempt. A subsequent --resume only fills missing results."""
    from stmem.inspection.history_use_report import write_report

    attempt = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{time.time_ns()}_{os.getpid()}"
    logs_dir = output / "logs" / attempt
    logs_dir.mkdir(parents=True)
    receipt = {
        "attempt": attempt, "pid": os.getpid(), "resume": args.resume,
        "workers": args.workers, "status": "running",
        "manifest_sha256": file_sha256(output / "manifest.json"),
        "config_sha256": file_sha256(output / "config.json"),
        "preserved_results": {str(path.relative_to(output)): file_sha256(path)
                              for path in sorted((output / "results").glob("*/episodes/*.json"))},
    }
    def save_status():
        atomic_json(receipt, logs_dir / "attempt.json")
        atomic_json(receipt, output / "run_status.json")

    save_status()
    processes, logs = [], []
    try:
        with tracker_class(cfg, output, job_type="evaluation", resume=args.resume,
                           name=("teacher-recent-history-SwingXtimes" if args.recent_swing else
                                 "teacher-permuted-history-SwingXtimes" if args.permuted_swing else
                                 "teacher-history-use-three-tasks"), metadata=manifest) as tracker, worker_lifetime(processes):
            initial, _ = collect(output, manifest)
            if len(initial) < len(manifest["planned_cases"]):
                for index in range(args.workers):
                    log = (logs_dir / f"worker_{index}.log").open("w")
                    logs.append(log)
                    env = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
                    processes.append(subprocess.Popen(
                        [sys.executable, "-u", "-m", "stmem.inspection.history_use", "--output", str(output),
                         "--worker-index", str(index), "--worker-count", str(args.workers),
                         "--lock-fd", str(lock_fd)], stdout=log, stderr=subprocess.STDOUT, env=env,
                        start_new_session=True, pass_fds=(lock_fd,)))
            count = -1
            while True:
                results, summary = collect(output, manifest)
                if len(results) != count:
                    count = len(results)
                    atomic_json(summary, output / "summary.json")
                    tracker.log({"evaluation/completed_rollouts": count,
                                 "evaluation/progress_percent": 100 * count / len(manifest["planned_cases"]),
                                 **{f"evaluation/{arm}_success_rate_percent": 100 * (stats["macro_success"] or 0)
                                    for arm, stats in summary["per_arm"].items()}}, step=count)
                    print(json.dumps({"event": "evaluation_progress", "completed": count,
                                      "planned": len(manifest["planned_cases"]),
                                      "successes": {arm: stats["overall"]["successes"]
                                                    for arm, stats in summary["per_arm"].items()}}), flush=True)
                if all(process.poll() is not None for process in processes):
                    results, summary = collect(output, manifest)
                    break
                if any(process.poll() not in (None, 0) for process in processes):
                    raise RuntimeError("A history worker failed; inspect its attempt log before resuming")
                time.sleep(3)
            summary["worker_exit_codes"] = [p.returncode for p in processes]
            atomic_json(summary, output / "summary.json")
            print(write_report(output), flush=True)
            if len(results) != len(manifest["planned_cases"]) or any(p.returncode for p in processes):
                raise RuntimeError("Incomplete paired diagnostic; inspect worker logs and planned counts")
            validate_existing(output, manifest, cfg)
            for relative, expected_hash in receipt["preserved_results"].items():
                if file_sha256(output / relative) != expected_hash:
                    raise ValueError("A previously published result changed during resume")
            tracker.summary({"evaluation/summary": summary, "evaluation/output_directory": str(output)})
            if tracker.run is not None:
                import wandb

                table = wandb.Table(columns=["task", "episode", "arm", "status", "video"])
                for row in results:
                    path = row.get("video_path")
                    table.add_data(row["task"], row["episode_idx"], row["arm"], row["status"],
                                   wandb.Video(path, format="mp4") if path and Path(path).stat().st_size else None)
                tracker.run.log({"evaluation/paired_episodes": table})
        receipt.update(status="complete", recorded=len(results), planned=len(manifest["planned_cases"]))
    except BaseException as exc:
        receipt.update(status="interrupted" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for log in logs:
            log.close()
        receipt["worker_exit_codes"] = [process.returncode for process in processes]
        save_status()


if __name__ == "__main__":
    main()
