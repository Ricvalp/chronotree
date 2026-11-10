#!/usr/bin/env python3
"""Conditional paired 10k rehearsal continuation; explicit evidence is required."""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import re
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import positive_integer, read_site, save_json, sha256  # noqa: E402
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from hpc.train_compact import prepare  # noqa: E402
from stmem.training.runtime import load_checkpoint  # noqa: E402
from stmem.training.utility_study import (  # noqa: E402
    ENCODED_CACHE_ID, HIERARCHY_CACHE_ID, TEACHER_SHA256, read_study_manifest,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site_b200_4gpu.env")
    parser.add_argument("--cache-name", type=profile_name, default="memory_fv_seed0")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True,
                        help="Development finding justifying richer-history rehearsal; pinned verbatim")
    parser.add_argument("--name", type=profile_name, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partition", default="gpuq")
    parser.add_argument("--gpus", type=positive_integer, default=2)
    parser.add_argument("--cpus", type=positive_integer, default=64)
    parser.add_argument("--mem", default="128G")
    parser.add_argument("--walltime", type=walltime, default="48:00:00")
    args = parser.parse_args(argv)
    if 128 % args.gpus:
        parser.error("Effective batch 128 must divide across GPUs")
    if args.cpus < args.gpus * 5:
        parser.error("Allocate at least five CPUs per GPU (trainer plus four workers)")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.partition):
        parser.error("Invalid --partition")
    if not re.fullmatch(r"[1-9][0-9]*(?:[KMGT]i?B?|[kmgt])?", args.mem):
        parser.error("Invalid --mem")
    try:
        site = read_site(args.site)
        site["WANDB_PROJECT"] = "chronotree-routers"
        repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
        source_manifest = absolute_path(args.manifest, repo)
        panel = read_study_manifest(source_manifest)
        source = absolute_path(args.parent, repo)
        parent = load_checkpoint(source)
        if parent["stage"] != "compact" or parent["provenance"]["dependencies"].get("teacher") != TEACHER_SHA256:
            raise ValueError("Rehearsal requires a compact policy using the accepted teacher")
        evidence = absolute_path(args.evidence, repo)
        if not evidence.read_text().strip():
            raise ValueError("Rehearsal activation requires a nonempty development finding")
        generated = repo / "hpc/generated" / args.name
        generated.mkdir(parents=True, exist_ok=True)
        with (generated / ".launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for incoming, name in ((source, "parent.pt"), (source_manifest, "panel_manifest.json"),
                                   (evidence, "activation_evidence.md")):
                target = generated / name
                if target.exists() and sha256(target) != sha256(incoming):
                    raise ValueError(f"Conditional study dependency changed: {name}")
                if not target.exists():
                    shutil.copyfile(incoming, target)
            arms = []
            for arm in ("rehearsal_control", "rehearsal_full"):
                child = argparse.Namespace(**vars(args))
                child.name = args.name + "_" + arm
                child.run_root = repo / "runs" / child.name
                child.run_root.mkdir(parents=True, exist_ok=True)
                child.code_root = None
                child.hierarchy_root = None
                child.legacy_cache = False
                child.effective_batch = 128
                # Singleton history batches avoid padding every short context to
                # the long FULL example. Accumulation keeps exposure at 128.
                child.microbatch = 1
                child.workers = 4
                child.restart_unstarted = False
                child.continue_to_step = None
                child.utility_study = {
                    "schema_version": 1,
                    "variant": parent["config"]["training"].get("utility_study", {}).get("variant", "S_STD"),
                    "experiment_id": panel["experiment_id"], "followup": arm,
                    "manifest": str(generated / "panel_manifest.json"),
                    "manifest_sha256": sha256(generated / "panel_manifest.json"),
                    "teacher_sha256": TEACHER_SHA256, "encoded_cache_id": ENCODED_CACHE_ID,
                    "hierarchy_cache_id": HIERARCHY_CACHE_ID,
                    "initialization_checkpoint": str(generated / "parent.pt"),
                    "initialization_sha256": sha256(generated / "parent.pt"),
                    "continuation_updates": 10000, "continuation_lr": 1e-5,
                    "optimizer_initialization": "fresh_identical_reset", "parent_step": parent["step"],
                    "activation_evidence": str(generated / "activation_evidence.md"),
                    "activation_evidence_sha256": sha256(generated / "activation_evidence.md"),
                    "full_replacement_fraction": 0.1 if arm == "rehearsal_full" else 0.0,
                    "batching": "singleton_history_batches_with_effective_batch_128_accumulation",
                    "truncate_history": False,
                }
                destination = repo / "hpc/generated" / child.name
                destination.mkdir(parents=True, exist_ok=True)
                child.submit = False
                with (child.run_root / ".compact-launch.lock").open("a") as child_lock:
                    fcntl.flock(child_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    prepare(child, site, repo, repo / "hpc/generated" / args.cache_name / "plan.json", destination)
                arms.append((child, destination))
            save_json(generated / "matrix.json", {
                "kind": "conditional_paired_rehearsal", "status": "prepared",
                "parent_sha256": sha256(generated / "parent.pt"), "updates_per_arm": 10000,
                "activation_evidence_sha256": sha256(generated / "activation_evidence.md"),
                "plans": [str(destination / "plan.json") for _, destination in arms],
                "remaining_evidence": "Measure singleton accumulation throughput and longest-history memory before submission",
            })
            if args.submit:
                for child, destination in arms:
                    child.submit = True
                    with (child.run_root / ".compact-launch.lock").open("a") as child_lock:
                        fcntl.flock(child_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        prepare(child, site, repo, repo / "hpc/generated" / args.cache_name / "plan.json", destination)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
