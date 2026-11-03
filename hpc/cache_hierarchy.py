#!/usr/bin/env python3
"""Render CPU hierarchy preparation from an existing frozen teacher leaf cache.

Only --submit contacts Slurm. No neural encoders, raw observations or model
weights are evaluated; each array element processes one existing task shard.
"""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.cache_teacher import active_array_ids  # noqa: E402
from hpc.plan import positive_integer, read_site, save_json, sha256, source_hash  # noqa: E402
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from hpc.train_compact import inspect_cache, read_json, require_hash  # noqa: E402
from stmem.config import digest  # noqa: E402


def ready_plan(path):
    """Verify the immutable launch inputs inside each allocation."""
    plan = read_json(path)
    if plan.get("kind") != "derived-teacher-hierarchy-cache":
        raise ValueError("Expected a hierarchy cache plan")
    require_hash(Path(__file__), plan["launcher_sha256"], "Hierarchy launcher")
    require_hash(Path(__file__).with_name("train_compact.py"), plan["guard_sha256"], "Cache input guard")
    require_hash(plan["cache_plan"], plan["cache_plan_sha256"], "Source cache plan")
    require_hash(plan["config"], plan["config_sha256"], "Source cache configuration")
    if source_hash(Path(plan["repo"])) != plan["source_sha256"]:
        raise ValueError("Source changed since hierarchy job rendering")
    _, original, manifest = inspect_cache(plan["cache_plan"])
    if (digest(manifest) != plan["cache_manifest_digest"]
            or original["checkpoint_sha256"] != plan["checkpoint_sha256"]):
        raise ValueError("Frozen leaf cache dependencies changed")
    require_hash(Path(plan["cache_path"]) / "manifest.json", plan["cache_manifest_sha256"], "Leaf manifest")
    return plan


def render_script(plan, site):
    directives = {
        "job-name": "stmem_hierarchy_" + plan["name"], "partition": plan["partition"],
        "gres": "gpu:1", "nodes": "1", "ntasks": "1", "cpus-per-task": str(plan["cpus"]),
        "mem": plan["mem"], "time": plan["walltime"], "array": plan["array"],
        "output": str(Path(plan["logs"]) / "%x_%A_%a.out"),
        "error": str(Path(plan["logs"]) / "%x_%A_%a.err"),
        "chdir": plan["repo"], "signal": "B:USR1@180",
    }
    for key in ("ACCOUNT", "QOS", "CONSTRAINT"):
        if site.get("STMEM_" + key):
            directives[key.lower()] = site["STMEM_" + key]
    if any("\n" in value or "\r" in value for value in directives.values()):
        raise ValueError("Newline in Slurm directive")
    lines = ["#!/usr/bin/env bash", *(f"#SBATCH --{key}={shlex.quote(value)}"
                                      for key, value in directives.items()),
             "set -euo pipefail", "cd " + shlex.quote(plan["repo"])]
    if site.get("STMEM_ACTIVATE"):
        lines.append("source " + shlex.quote(site["STMEM_ACTIVATE"]))
    exported = {
        "STMEM_SITE_FILE": "/dev/null", "STMEM_REPO_ROOT": plan["repo"],
        "STMEM_PYTHON": plan["python"], "PYTHONPATH": str(Path(plan["repo"]) / "src"),
        "PYTHONUNBUFFERED": "1", "STMEM_ALLOCATED_GPUS": "1",
        "OMP_NUM_THREADS": str(plan["threads"]), "MKL_NUM_THREADS": str(plan["threads"]),
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }
    lines.append("unset RANK LOCAL_RANK WORLD_SIZE STMEM_SCRATCH_ROOT WANDB_RUN_ID WANDB_RESUME")
    for key, value in sorted(exported.items()):
        lines.append(f"export {key}={shlex.quote(value)}")
    guard = "from hpc.cache_hierarchy import ready_plan; ready_plan(" + repr(plan["startup_plan"]) + ")"
    lines.append(shlex.join([plan["python"], "-c", guard]))
    lines.append('STMEM_HIERARCHY_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this as the generated array job}"')
    lines.append("export STMEM_STOP_FILE=" + shlex.quote(str(Path(plan["script"]).parent / "stops"))
                 + '/"${SLURM_ARRAY_JOB_ID:-local}_${STMEM_HIERARCHY_INDEX}_${SLURM_JOB_ID:-local}.stop"')
    command = ["srun", "--ntasks=1", f"--cpus-per-task={plan['cpus']}", "--cpu-bind=cores",
               plan["python"], "-u", "-m", "stmem.data.hierarchy_cache", "--cache-root", plan["cache_path"],
               "--output-root", plan["hierarchy_cache_root"], "--threads", str(plan["threads"])]
    lines.append("exec " + shlex.quote(str(Path(plan["repo"]) / "hpc/run_job.sh"))
                 + ' "$STMEM_STOP_FILE" ' + shlex.join(command)
                 + ' --task-index "$STMEM_HIERARCHY_INDEX" --stop-file "$STMEM_STOP_FILE"')
    return "\n".join(lines) + "\n"


def status(plan_path):
    """Read recorded progress; deliberately no directory creation or Slurm query."""
    from stmem.data.hierarchy_cache import hierarchy_status

    plan = read_json(plan_path)
    if plan.get("kind") != "derived-teacher-hierarchy-cache":
        raise ValueError("Expected a hierarchy cache plan")
    require_hash(plan["cache_plan"], plan["cache_plan_sha256"], "Source cache plan")
    require_hash(plan["config"], plan["config_sha256"], "Source cache configuration")
    require_hash(Path(plan["cache_path"]) / "manifest.json", plan["cache_manifest_sha256"], "Leaf manifest")
    return hierarchy_status(plan["cache_path"], plan["hierarchy_cache_root"])


def prepare(args, site, repo, generated):
    from stmem.data.hierarchy_cache import default_hierarchy_root, hierarchy_status

    cache_plan_path = repo / "hpc/generated" / args.cache_name / "plan.json"
    cfg, cache_plan, manifest = inspect_cache(cache_plan_path)
    cache_root = Path(cfg["paths"]["cache_root"]) / digest(manifest["identity"])[:24]
    output = absolute_path(args.output_root, repo) if args.output_root else default_hierarchy_root(cache_root)
    if output == cache_root:
        raise ValueError("Hierarchy output must be separate from the immutable leaf cache")
    plan_path, receipt_path = generated / "plan.json", generated / "submitted.json"
    previous = read_json(plan_path) if plan_path.exists() else None
    if args.resume and previous is None:
        raise ValueError("No existing hierarchy plan to resume")
    if receipt_path.exists() and not args.resume:
        raise ValueError("Already submitted; use --resume once all array elements are inactive")
    if args.submit and receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["job_id"] in active_array_ids():
            raise ValueError(f"Hierarchy array {receipt['job_id']} still has active tasks")
    tasks = list(cfg["data"]["tasks"])
    if not tasks or len(set(tasks)) != len(tasks) or any(not re.fullmatch(r"[A-Za-z0-9_]+", task) for task in tasks):
        raise ValueError("Invalid or duplicate tasks")
    progress = hierarchy_status(cache_root, output)
    if args.submit and progress["status"] == "complete":
        raise ValueError("Hierarchy cache is already complete; proceed to compact training")
    signature = digest({"cache_plan_sha256": sha256(cache_plan_path), "cache": digest(manifest),
                        "output": str(output), "source": source_hash(repo)})
    if previous and previous["signature"] != signature:
        raise ValueError("Hierarchy inputs or source changed; this named plan is immutable")
    logs = generated / "logs"
    logs.mkdir(exist_ok=True)
    plan = {
        "kind": "derived-teacher-hierarchy-cache", "name": args.name, "repo": str(repo),
        "python": site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python"),
        "cache_plan": str(cache_plan_path), "cache_plan_sha256": sha256(cache_plan_path),
        "config": cache_plan["config"], "config_sha256": cache_plan["config_sha256"],
        "checkpoint": cache_plan["checkpoint"], "checkpoint_sha256": cache_plan["checkpoint_sha256"],
        "cache_path": str(cache_root), "cache_manifest_digest": digest(manifest),
        "cache_manifest_sha256": sha256(cache_root / "manifest.json"),
        "hierarchy_cache_root": str(output), "signature": signature, "source_sha256": source_hash(repo),
        "launcher_sha256": sha256(Path(__file__)),
        "guard_sha256": sha256(Path(__file__).with_name("train_compact.py")),
        "plan_path": str(plan_path), "script": str(generated / "hierarchy.sbatch"), "logs": str(logs),
        "tasks": tasks, "task_count": len(tasks), "array": f"0-{len(tasks)-1}%{min(args.max_concurrent,len(tasks))}",
        "gpus_per_task": 1, "gres": "gpu:1", "partition": args.partition, "cpus": args.cpus,
        "threads": args.threads, "mem": args.mem, "walltime": args.walltime,
        "maximum_concurrent_allocations": min(args.max_concurrent, len(tasks)),
        "resume": args.resume, "progress_at_render": progress,
        "estimated_gpu_hours": "unknown; CPU preparation reserves one GPU per allocation on this cluster",
        "computation": "CPU-only deterministic hierarchy construction from existing frozen leaves; no encoding",
    }
    attempt = generated / "attempts" / (digest(plan)[:24] + ".json")
    plan["startup_plan"] = str(attempt)
    if attempt.exists():
        if read_json(attempt) != plan:
            raise ValueError("Immutable hierarchy startup plan changed")
    else:
        save_json(attempt, plan)
    Path(plan["script"]).write_text(render_script(plan, site))
    save_json(plan_path, plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.submit:
        result = subprocess.run(["sbatch", "--parsable", plan["script"]], check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdecimal():
            raise ValueError(f"Unexpected sbatch receipt: {result.stdout!r}")
        save_json(receipt_path, {"job_id": job_id, "signature": signature,
                                "previous": read_json(receipt_path) if receipt_path.exists() else None})
        print(f"Submitted hierarchy array {job_id}: {len(tasks)} tasks, at most {plan['maximum_concurrent_allocations']} simultaneous")
    else:
        print("Dry-run only: hierarchy plan saved; no data processed or Slurm jobs submitted.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Default: validate and render only")
    action.add_argument("--submit", action="store_true")
    action.add_argument("--status", action="store_true", help="Read saved progress without writing or contacting Slurm")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site.env")
    parser.add_argument("--cache-name", required=True, type=profile_name)
    parser.add_argument("--name", type=profile_name, help="Default: CACHE_NAME_hierarchy")
    parser.add_argument("--output-root", type=Path, help="Default: content-addressed hierarchy directory inside leaf cache")
    parser.add_argument("--resume", action="store_true", help="Resubmit inactive array; verified completed work is reused")
    parser.add_argument("--partition", default="gpuq")
    parser.add_argument("--cpus", type=positive_integer, default=4)
    parser.add_argument("--threads", type=positive_integer, default=1)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--walltime", type=walltime, default="24:00:00")
    parser.add_argument("--max-concurrent", type=positive_integer, default=4)
    args = parser.parse_args(argv)
    args.name = args.name or args.cache_name + "_hierarchy"
    if args.name == args.cache_name:
        parser.error("Hierarchy --name must differ from --cache-name")
    if args.threads > args.cpus:
        parser.error("--threads cannot exceed --cpus")
    if not re.fullmatch(r"[1-9][0-9]*(?:[KMGT]i?B?|[kmgt])?", args.mem):
        parser.error("Invalid --mem")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.partition):
        parser.error("Invalid --partition")
    try:
        site = read_site(args.site)
        repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
        generated = repo / "hpc/generated" / args.name
        if args.status:
            print(json.dumps(status(generated / "plan.json"), indent=2), flush=True)
            return
        generated.mkdir(parents=True, exist_ok=True)
        with (generated / ".launcher.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare(args, site, repo, generated)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
