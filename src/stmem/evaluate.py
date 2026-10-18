"""Official seeded RoboMME evaluation; shards never duplicate padded episodes."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from stmem.config import load_config, dependency, run_dir, source_revision
from stmem.evaluation.runner import OnlinePolicy, run_episode
from stmem.evaluation.metrics import summarize_episodes
from stmem.evaluation.robomme_adapter import RoboMMEAdapter
from stmem.training.runtime import load_policy, load_router, validate_policy_config, environment_manifest
from stmem.training.checkpoints import file_sha256, atomic_json


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
    }
    existing = output / "manifest.json"
    if existing.exists():
        previous = json.loads(existing.read_text())
        comparable = json.loads(json.dumps(manifest))
        for value in (previous, comparable):
            value["source"] = value["source"]["source_sha256"]
        if previous != comparable:
            raise ValueError("Evaluation output exists with different inputs; choose a new output")
    atomic_json(manifest, existing)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    policy, checkpoint_state = load_policy(checkpoint, device)
    validate_policy_config(checkpoint_state["config"], cfg)
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
    results = []
    for task, episode_idx in planned:
        path = output / "episodes" / f"{task}_{episode_idx:03d}.json"
        if path.exists():
            results.append(json.loads(path.read_text()))
            continue
        adapter = None
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
            )
            seed = cfg["evaluation"]["seed"] + cfg["data"]["tasks"].index(task) * 1000 + episode_idx
            video = (
                output / "videos" / f"{task}_{episode_idx:03d}.mp4"
                if len(results) < cfg["evaluation"]["video_cap"]
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
                    result["close_error"] = f"{type(exc).__name__}: {exc}"
                    result["status"] = "infrastructure_error"
        result["model_load_seconds"] = load_seconds
        atomic_json(result, path)
        results.append(result)
        print(
            json.dumps(
                {k: result.get(k) for k in ("task", "episode_idx", "status", "error", "runtime_seconds")}
            ),
            flush=True,
        )
    summary = summarize_episodes(results, planned)
    atomic_json(summary, output / "summary.json")
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
