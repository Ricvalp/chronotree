#!/usr/bin/env python3
"""Prepare the matched S_STD/S_STAGE/S_PACKET study; only --submit calls Slurm."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import nonnegative_integer, positive_integer, read_site, save_json, sha256  # noqa: E402
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from hpc.train_compact import prepare  # noqa: E402
from stmem.training.utility_study import (  # noqa: E402
    COMPACT_SHA256, ENCODED_CACHE_ID, HIERARCHY_CACHE_ID, TEACHER_SHA256,
    VARIANTS, primary_schedule, read_study_manifest,
)


def manifest_schedule(manifest, args):
    # Previously sealed studies keep their declared 30k/15k protocol. New
    # studies derive their 100k schedule from the versioned protocol YAML.
    declared = manifest.get("protocol", {}).get("student_training")
    values = {"steps": 30000, "checkpoint_every": 1000, "snapshot_every": 15000} if declared is None else declared
    if not isinstance(values, dict) or "steps" not in values:
        raise ValueError("Manifest student_training requires its declared steps")
    schedule = primary_schedule(values["steps"], values.get("checkpoint_every", 1000),
                                values.get("snapshot_every", 5000))
    for argument, field in (("steps", "primary_updates"), ("checkpoint_every", "checkpoint_every"),
                            ("snapshot_every", "snapshot_every")):
        override = getattr(args, argument, None)
        if override is not None and override != schedule[field]:
            raise ValueError(f"--{argument.replace('_', '-')} differs from the sealed study protocol; "
                             "prepare a new study with the desired student_training schedule")
    return schedule


def make_settings(manifest, manifest_path, variant, args, initialization=None):
    schedule = manifest_schedule(manifest, args) if initialization is None else {}
    return {
        "schema_version": 1, "variant": variant, "experiment_id": manifest["experiment_id"],
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "teacher_sha256": TEACHER_SHA256, "encoded_cache_id": ENCODED_CACHE_ID,
        "hierarchy_cache_id": HIERARCHY_CACHE_ID, "family_proportions": [0.45, 0.45, 0.05, 0.05],
        "router_template_fraction": 0.5, "target_budgets": {"128": 0.75, "64": 0.25},
        "completed_prefix_fraction": 0.5, "packet_cost": 8,
        "action_noise_stream": "per_global_position_sha256_flow_v1",
        "context_stream": "per_global_position_sha256_exposure_and_route_v1",
        "checkpoint_rule": f"fixed_{schedule['primary_updates']}_endpoint" if initialization is None else "fixed_continuation_endpoint",
        **schedule,
        "optimizer_initialization": "fresh", "initialization_checkpoint": initialization,
        "initialization_sha256": COMPACT_SHA256 if initialization else TEACHER_SHA256,
        "continuation_updates": args.continuation_updates if initialization else None,
        "continuation_lr": args.continuation_lr if initialization else None,
        "initialization": "S0_weights" if initialization else "accepted_teacher_weights",
        "parent_step": 90000 if initialization else 104110,
        "conditional_branches": "not_activated",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site_b200_4gpu.env")
    parser.add_argument("--cache-name", type=profile_name, default="memory_fv_seed0")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--name", type=profile_name, default="utility_students_v1")
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--initialization", choices=("teacher", "continuation"), default="teacher")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--steps", type=positive_integer,
                        help="Teacher-initialized updates; must match the sealed protocol (new studies: 100000)")
    parser.add_argument("--checkpoint-every", type=positive_integer,
                        help="Rolling last.pt interval; must match the sealed protocol (new studies: 1000)")
    parser.add_argument("--snapshot-every", type=positive_integer,
                        help="Numbered checkpoint interval; must match the sealed protocol (new studies: 5000)")
    parser.add_argument("--continuation-updates", type=positive_integer, default=10000)
    parser.add_argument("--continuation-lr", type=float, default=1e-5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partition", default="gpuq")
    parser.add_argument("--gpus", type=positive_integer, default=2)
    parser.add_argument("--cpus", type=positive_integer, default=64)
    parser.add_argument("--mem", default="128G")
    parser.add_argument("--walltime", type=walltime, default="48:00:00")
    parser.add_argument("--microbatch", type=positive_integer, default=32)
    parser.add_argument("--workers", type=nonnegative_integer, default=16)
    args = parser.parse_args(argv)
    if len(set(args.variants)) != len(args.variants):
        parser.error("Student variants must be distinct")
    if 128 % args.gpus or args.microbatch > 128 // args.gpus:
        parser.error("Effective batch 128 must divide across GPUs and fit the per-rank microbatch")
    if args.cpus < args.gpus * (1 + args.workers):
        parser.error("Allocate at least one CPU per trainer and loader worker")
    if args.initialization == "continuation" and args.initial_checkpoint is None:
        parser.error("--initialization continuation requires the immutable S0 --initial-checkpoint")
    if args.initialization == "continuation" and any(value is not None for value in (
            args.steps, args.checkpoint_every, args.snapshot_every)):
        parser.error("Primary schedule options apply to teacher initialization; use --continuation-updates instead")
    if args.initialization == "teacher" and args.initial_checkpoint is not None:
        parser.error("Teacher initialization takes its checkpoint only from the accepted cache plan")
    if not 0 < args.continuation_lr < 1:
        parser.error("Continuation LR must lie in (0, 1)")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.partition):
        parser.error("Invalid --partition")
    if not re.fullmatch(r"[1-9][0-9]*(?:[KMGT]i?B?|[kmgt])?", args.mem):
        parser.error("Invalid --mem")
    try:
        site = read_site(args.site)
        # The study explicitly belongs to the existing router project.
        site["WANDB_PROJECT"] = "chronotree-routers"
        repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
        manifest_path = absolute_path(args.manifest, repo)
        manifest = read_study_manifest(manifest_path)
        schedule = manifest_schedule(manifest, args) if args.initialization == "teacher" else None
        updates = schedule["primary_updates"] if schedule else args.continuation_updates
        generated = repo / "hpc/generated" / args.name
        generated.mkdir(parents=True, exist_ok=True)
        with (generated / ".launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pinned = generated / "panel_manifest.json"
            if pinned.exists():
                if sha256(pinned) != sha256(manifest_path):
                    raise ValueError("This study name already pins a different panel manifest")
            else:
                shutil.copyfile(manifest_path, pinned)
            initialization = None
            if args.initial_checkpoint:
                source = absolute_path(args.initial_checkpoint, repo)
                if sha256(source) != COMPACT_SHA256:
                    raise ValueError("Continuation initialization must be the exact immutable 90k S0")
                target = generated / "initialization.pt"
                if not target.exists():
                    shutil.copyfile(source, target)
                if sha256(target) != COMPACT_SHA256:
                    raise ValueError("Pinned continuation initialization changed")
                initialization = str(target)
            matrix = {"schema_version": 1, "kind": "matched_student_utility_study",
                      "experiment_id": manifest["experiment_id"], "manifest_sha256": sha256(pinned),
                      "initialization": args.initialization, "students": {},
                      "reference": {"variant": "S0", "sha256": COMPACT_SHA256, "updates": 0},
                      "conditional_branches": "configured, not activated", "job_count": len(args.variants),
                      "total_updates": len(args.variants) * updates,
                      "total_example_exposures": len(args.variants) * updates * 128,
                      "primary_schedule": schedule,
                      "effective_batch": 128, "gpu_hours": "unknown; measure each sampler on real caches"}
            configurations = []
            for variant in args.variants:
                name = args.name + "_" + variant.lower()
                run_root = repo / "runs" / name
                run_root.mkdir(parents=True, exist_ok=True)
                output = repo / "hpc/generated" / name
                output.mkdir(parents=True, exist_ok=True)
                child = argparse.Namespace(**vars(args))
                child.name, child.run_root = name, run_root
                child.code_root = None
                child.hierarchy_root = None
                child.legacy_cache = False
                child.effective_batch = 128
                child.restart_unstarted = False
                child.continue_to_step = None
                child.utility_study = make_settings(manifest, pinned, variant, args, initialization)
                child.submit = False
                configurations.append((child, output))
                with (run_root / ".compact-launch.lock").open("a") as child_lock:
                    fcntl.flock(child_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    prepare(child, site, repo, repo / "hpc/generated" / args.cache_name / "plan.json", output)
                matrix["students"][variant] = {
                    "plan": str(output / "plan.json"), "config": str(output / "compact.yaml"),
                    "run_root": str(run_root), "primary_checkpoint": str(run_root / "seed0/compact" /
                        f"step_{updates:08d}.pt"),
                    "depends_on": ["fixed_panel", "accepted_teacher", "encoded_cache", "hierarchy_cache"],
                }
            save_json(generated / "matrix.json", matrix)
            if args.submit:
                # All three arms must render successfully before any allocation is submitted.
                for child, output in configurations:
                    child.submit = True
                    with (Path(child.run_root) / ".compact-launch.lock").open("a") as child_lock:
                        fcntl.flock(child_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        prepare(child, site, repo, repo / "hpc/generated" / args.cache_name / "plan.json", output)
            print(json.dumps(matrix, indent=2))
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
