"""Official seeded RoboMME evaluation; shards never duplicate padded episodes."""

from __future__ import annotations
import argparse
import json
import math
from collections import defaultdict
from numbers import Real
from pathlib import Path
import time
import numpy as np
import torch
from stmem.config import load_config, dependency, run_dir, source_revision
from stmem.evaluation.runner import OnlinePolicy, run_episode
from stmem.evaluation.metrics import summarize_episodes
from stmem.evaluation.robomme_adapter import RoboMMEAdapter
from stmem.data.pointcloud import point_sampler_settings
from stmem.data.center_sampling import center_sampler_settings, center_counts
from stmem.training.runtime import load_policy, load_router, validate_policy_config, environment_manifest
from stmem.training.checkpoints import file_sha256, atomic_json
from stmem.tracking import RunTracker


def numeric_scalars(value: dict, prefix: str) -> dict[str, int | float]:
    """Flatten only existing finite scalar measurements; never convert tensors."""
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(item, dict):
            result.update(numeric_scalars(item, name))
        elif isinstance(item, Real) and math.isfinite(item):
            result[name] = int(item) if isinstance(item, (bool, int, np.integer)) else float(item)
    return result


def _distribution(values, prefix, *, total=False):
    values = [float(v) for v in values if isinstance(v, Real) and math.isfinite(v)]
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    result = {
        f"{prefix}/count": len(values),
        f"{prefix}/mean": float(array.mean()),
        f"{prefix}/max": float(array.max()),
        f"{prefix}/p50": float(np.percentile(array, 50)),
        f"{prefix}/p95": float(np.percentile(array, 95)),
    }
    if total:
        result[f"{prefix}/total"] = float(array.sum())
    return result


def evaluation_tracking_scalars(records, planned=None, *, summary=None, prefix="evaluation"):
    """Detailed diagnostics for local artifacts and aggregate research reports.

    Missing measurements remain absent. Byte occupancy is summarized as occupancy,
    while per-decision transfer/scoring costs may be summed. Startup and the first
    decision of each episode remain distinguishable from steady-state decisions.
    These are deliberately not published to the live evaluation dashboard.
    """
    summary = summarize_episodes(records, planned) if summary is None else summary
    scalars = numeric_scalars(summary, prefix)
    decision_values = defaultdict(list)
    component_values = defaultdict(list)
    first, warm = [], []
    occupancy = (
        "history_tokens",
        "active_tokens",
        "archive_fine_leaves",
        "archive_payload_bytes",
        "pending_raw_bytes",
        "event_identity_bytes",
        "routing_key_bytes",
        "cpu_process_rss_bytes",
        "cpu_process_peak_rss_bytes",
        "selected_payload_bytes",
    )
    costs = (
        "nodes_scored",
        "leaves_scored",
        "records_fetched",
        "splits",
        "selected_transfer_payload_bytes",
        "routing_transferred_bytes",
        "new_keys_encoded",
    )
    timing = ("complete_online_seconds", "decision_seconds", "observation_update_seconds")
    for record in records:
        decisions = record.get("decisions", [])
        for index, decision in enumerate(decisions):
            for key in (*occupancy, *costs, *timing):
                if key in decision:
                    decision_values[key].append(decision[key])
            latency = decision.get("complete_online_seconds", decision.get("decision_seconds"))
            if latency is not None:
                (first if index == 0 else warm).append(latency)
            for key, value in decision.get("component_seconds", {}).items():
                component_values[key].append(value)
            for key in ("resolution_depths", "represented_leaf_counts"):
                decision_values[key].extend(decision.get(key, []))
    for key, values in decision_values.items():
        scalars.update(_distribution(values, f"{prefix}/decisions/{key}", total=key in (*costs, *timing)))
    for name, values in (("cold_first_decision", first), ("warm_after_first_decision", warm)):
        scalars.update(_distribution(values, f"{prefix}/latency/{name}_seconds", total=True))
    for key, values in component_values.items():
        scalars.update(_distribution(values, f"{prefix}/components/{key}_seconds", total=True))
    for key in ("runtime_seconds", "startup_seconds", "reset_encoding_seconds", "primitive_commands"):
        scalars.update(
            _distribution([r[key] for r in records if key in r], f"{prefix}/episodes/{key}", total=True)
        )
    # These encoder counters are cumulative within each episode, not per-decision deltas.
    for key in ("encoding_calls", "encoded_frames_including_prefix_reencodes"):
        last = [next((d[key] for d in reversed(r.get("decisions", [])) if key in d), None) for r in records]
        scalars.update(_distribution(last, f"{prefix}/episodes/{key}_at_last_decision", total=True))
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        values = [r.get("accelerator_memory", {}).get(key) for r in records]
        scalars.update(_distribution(values, f"{prefix}/accelerator_memory/{key}"))
    return scalars


def _log_evaluation_progress(tracker, summary, *, new_episodes, reused_episodes, last_episode=None):
    """Two fixed charts; completed failures/errors stay in the live denominator.

    The running success rate is a monitoring statistic over recorded episodes,
    not the planned-denominator task macro used for scientific reporting. A run
    with no recorded outcomes has no success estimate, rather than zero success.
    """
    overall = summary["overall"]
    recorded, planned = overall["recorded"], overall["planned"]
    tracker.log(
        {
            "evaluation/success_rate_so_far_percent": 100 * overall["successes"] / recorded
            if recorded
            else None,
            "evaluation/progress_percent": 100 * recorded / planned if planned else 100.0,
        },
        step=recorded,
    )
    tracker.summary(
        {
            "evaluation/status": summary["status"],
            "evaluation/progress": {
                "episodes_recorded": recorded,
                "episodes_planned": planned,
                "successes": overall["successes"],
                "errors": overall["errors"],
                "timeouts": overall["timeouts"],
                "new_episodes": new_episodes,
                "reused_episodes": reused_episodes,
                "scope": "this evaluation shard",
                "success_rate_denominator": "recorded episodes, including failures, errors and timeouts",
            },
        }
    )
    if last_episode is not None:
        tracker.summary(
            {
                "evaluation/last_episode": {
                    key: last_episode.get(key)
                    for key in ("task", "episode_idx", "status", "error", "close_error")
                }
            }
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument(
        "--method", choices=["FULL", "TREE", "FLAT", "UNIFORM", "PERCEIVER", "CURRENT"], default="FULL"
    )
    p.add_argument("--checkpoint")
    p.add_argument("--router-checkpoint")
    p.add_argument("--output")
    p.add_argument("--episode-count", type=int)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--budget", type=int)
    p.add_argument("--selection-family", choices=["tree", "flat"])
    p.add_argument(
        "--targeting-diagnostics",
        action="store_true",
        help="Save evaluation-only target/contact/trajectory diagnostics for PickXtimes and VideoUnmask",
    )
    p.add_argument(
        "--intervention",
        choices=["none", "erase_execution", "permute_execution", "erase_conditioning"],
        default="none",
    )
    args = p.parse_args()
    cfg = load_config(args.config)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    count = args.episode_count or cfg["evaluation"]["episode_count"]
    if not 1 <= count <= 50:
        raise ValueError("Official split provides 50 episode configurations per task")
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir(cfg, "perceiver") / "last.pt"
        if args.method == "PERCEIVER"
        else dependency(cfg, "teacher" if args.method == "FULL" else "compact")
    )
    router_path = args.router_checkpoint
    if args.method in ("TREE", "FLAT") and not router_path and not args.selection_family:
        router_path = str(run_dir({**cfg, "reader": args.method.lower()}, "router") / "last.pt")
    hashes = {str(checkpoint.resolve()): file_sha256(checkpoint)}
    if router_path:
        hashes[str(Path(router_path).resolve())] = file_sha256(router_path)
    if args.split == "test":
        frozen = cfg["evaluation"].get("test_frozen_manifest")
        if not frozen:
            raise ValueError(
                "Official test requires the frozen protocol manifest; use hpc/plan.py --phase test --freeze-manifest PATH"
            )
        protocol = json.loads(Path(frozen).read_text())
        from stmem.evaluation.protocol import validate_frozen

        validate_frozen(
            cfg,
            protocol,
            args.method,
            args.budget or cfg["model"]["budget"],
            count,
            hashes,
            args.selection_family,
            args.intervention,
        )
    output = (
        Path(args.output)
        if args.output
        else run_dir(cfg, "evaluation") / args.split / args.method / f"shard{args.shard_index}"
    )
    output.mkdir(parents=True, exist_ok=True)
    all_planned = [(task, i) for task in cfg["data"]["tasks"] for i in range(count)]
    planned = all_planned[args.shard_index :: args.num_shards]
    manifest = {
        "config": cfg,
        "environment": environment_manifest(),
        "source": source_revision(),
        "checkpoints": hashes,
        "method": args.method,
        "split": args.split,
        "planned": planned,
        "all_planned": all_planned,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selection_family": args.selection_family,
        "intervention": args.intervention,
        "budget": args.budget or cfg["model"]["budget"],
        "targeting_diagnostics": args.targeting_diagnostics,
    }
    existing = output / "manifest.json"
    resuming = existing.exists()
    if resuming:
        previous = json.loads(existing.read_text())
        comparable = json.loads(json.dumps(manifest))
        for value in (previous, comparable):
            value["source"] = value["source"]["source_sha256"]
            # Tracking transport has no effect on the scientific evaluation protocol.
            value["config"].pop("tracking", None)
        if previous != comparable:
            raise ValueError("Evaluation output exists with different inputs; choose a new output")
    else:
        atomic_json(manifest, existing)
    with RunTracker(
        cfg,
        output,
        job_type="evaluation",
        name=f"{args.method}-{args.split}-seed{cfg['seed']}-shard{args.shard_index}",
        metadata={
            "evaluation_manifest": json.loads(existing.read_text()),
            "manifest_sha256": file_sha256(existing),
        },
        resume=resuming,
    ) as tracker:
        _evaluate(cfg, args, checkpoint, router_path, output, planned, hashes, existing, tracker)


def _evaluate(cfg, args, checkpoint, router_path, output, planned, hashes, existing, tracker):
    # Include all durable outcomes in the initial progress, even when a resumed
    # shard has gaps. Loading them once also avoids replaying completed episodes.
    saved = {}
    for task, episode_idx in planned:
        path = output / "episodes" / f"{task}_{episode_idx:03d}.json"
        if path.exists():
            saved[task, episode_idx] = json.loads(path.read_text())
    results = list(saved.values())
    reused = len(results)
    new_episodes = 0
    _log_evaluation_progress(
        tracker,
        summarize_episodes(results, planned),
        new_episodes=new_episodes,
        reused_episodes=reused,
        last_episode=results[-1] if results else None,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    policy, checkpoint_state = load_policy(checkpoint, device)
    validate_policy_config(checkpoint_state["config"], cfg)
    from stmem.vision.frozen import FrozenCLIPFeatures, resolve_visual_features

    visual_spec = resolve_visual_features(cfg)
    feature_extractor = FrozenCLIPFeatures(cfg, device=device) if visual_spec["mode"] != "xyzrgb" else None
    tracker.summary(
        {
            "evaluation/policy_parameters": sum(p.numel() for p in policy.parameters()),
            "evaluation/frozen_visual_parameters": 0
            if feature_extractor is None
            else sum(p.numel() for p in feature_extractor.parameters()),
        }
    )
    router = None
    if router_path:
        router, router_state = load_router(router_path, device)
        if router_state["provenance"]["compact"] != file_sha256(checkpoint):
            raise ValueError("Router was supervised for a different frozen compact policy")
    online = OnlinePolicy(
        policy,
        cfg,
        args.method,
        args.budget or cfg["model"]["budget"],
        router=router,
        router_hash=file_sha256(router_path) if router_path else "",
        selection_family=args.selection_family,
        intervention=args.intervention,
    )
    load_seconds = time.perf_counter() - started
    tracker.summary({"evaluation/model_load_seconds_this_invocation": load_seconds})
    for episode_position, (task, episode_idx) in enumerate(planned):
        path = output / "episodes" / f"{task}_{episode_idx:03d}.json"
        if (task, episode_idx) in saved:
            continue
        adapter = None
        result = None
        try:
            adapter = RoboMMEAdapter(
                task,
                episode_idx,
                args.split,
                points=cfg["model"]["points"],
                max_steps=cfg["evaluation"]["max_steps"],
                crop=cfg["geometry"]["crop"],
                voxel_size=cfg["geometry"]["voxel_size"],
                max_depth_m=cfg["geometry"]["max_depth"],
                point_sampling=point_sampler_settings(cfg),
                center_sampling=center_sampler_settings(cfg),
                center_counts=center_counts(cfg),
                feature_extractor=feature_extractor,
            )
            if args.targeting_diagnostics:
                from stmem.evaluation.targeting import TargetingAdapter

                adapter = TargetingAdapter(adapter)
            seed = cfg["evaluation"]["seed"] + cfg["data"]["tasks"].index(task) * 1000 + episode_idx
            video = (
                output / "videos" / f"{task}_{episode_idx:03d}.mp4"
                if episode_position < cfg["evaluation"]["video_cap"]
                else None
            )
            result = run_episode(
                adapter,
                online,
                task=task,
                episode_idx=episode_idx,
                split=args.split,
                seed=seed,
                timeout_seconds=cfg["evaluation"]["timeout_seconds"],
                video_path=video,
                checkpoint_hash=hashes[str(checkpoint.resolve())],
                router_hash=file_sha256(router_path) if router_path else "",
            )
        except Exception as exc:
            result = {
                "task": task,
                "episode_idx": episode_idx,
                "split": args.split,
                "method": args.method,
                "status": "infrastructure_error",
                "error": f"{type(exc).__name__}: {exc}",
                "actions": [],
            }
        finally:
            if adapter:
                try:
                    adapter.close()
                except Exception as exc:
                    close_error = f"{type(exc).__name__}: {exc}"
                    if result is None:
                        # A fatal interruption is already propagating. Preserve it
                        # while recording the secondary cleanup failure.
                        tracker.summary({"evaluation/interrupted_close_error": close_error[:2048]})
                    else:
                        result["close_error"] = close_error
                        result["status"] = "infrastructure_error"
                if args.targeting_diagnostics and result is not None:
                    result["targeting_diagnostics"] = adapter.report(result["status"])
        result["model_load_seconds"] = load_seconds
        atomic_json(result, path)
        results.append(result)
        new_episodes += 1
        # Logging follows both synchronized episode timing and durable result writes.
        _log_evaluation_progress(
            tracker,
            summarize_episodes(results, planned),
            new_episodes=new_episodes,
            reused_episodes=reused,
            last_episode=result,
        )
        print(
            json.dumps(
                {k: result.get(k) for k in ("task", "episode_idx", "status", "error", "runtime_seconds")}
            ),
            flush=True,
        )
    summary = summarize_episodes(results, planned)
    atomic_json(summary, output / "summary.json")
    # Retain full latency/storage/cost measurements without hundreds of W&B panels.
    atomic_json(evaluation_tracking_scalars(results, planned, summary=summary), output / "diagnostics.json")
    if args.targeting_diagnostics:
        from stmem.evaluation.targeting import summarize_targeting

        atomic_json(summarize_targeting(results), output / "targeting.json")
        tracker.summary({"evaluation/targeting_report": str(output / "targeting.json")})
    if len(results) == len(planned):
        atomic_json(
            {
                "status": "complete",
                "planned": len(planned),
                "recorded": len(results),
                "manifest_sha256": file_sha256(existing),
            },
            output / "complete.json",
        )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
