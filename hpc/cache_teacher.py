#!/usr/bin/env python3
"""Adopt an evaluated teacher and render a resumable task-parallel encoder cache."""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import (  # noqa: E402
    positive_integer, read_site, save_json, sha256, source_hash,
)
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from stmem.config import digest, validate_config  # noqa: E402
from stmem.training.adopt_teacher import adopt_teacher  # noqa: E402
from stmem.training.runtime import load_checkpoint, validate_policy_config  # noqa: E402


def active_array_ids():
    # %A identifies individual array elements; %F retains their common base ID.
    result = subprocess.run(
        ["squeue", "--noheader", "--all", "--states=all", "--user", str(os.getuid()), "--format=%F|%T"],
        check=True, capture_output=True, text=True,
        env={key: value for key, value in os.environ.items() if not key.startswith("SQUEUE_")},
    )
    terminal = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
                "OUT_OF_MEMORY", "PREEMPTED", "TIMEOUT"}
    active = set()
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split("|")]
        if len(fields) != 2 or not fields[0].isdecimal() or not re.fullmatch(r"[A-Z_]+", fields[1]):
            raise ValueError("Unrecognized squeue array-ID/state output; cannot verify active cache jobs")
        if fields[1] not in terminal:
            active.add(fields[0])
    return active


def evaluated_checkpoint(evaluation):
    """Use the evaluated immutable snapshot, never a potentially advancing last.pt."""
    manifest = json.loads((evaluation / "manifest.json").read_text())
    entries = manifest.get("checkpoints", {})
    expected = manifest.get("policy_checkpoint_sha256")
    matches = [Path(path) for path, value in entries.items() if expected is None or value == expected]
    if len(matches) != 1:
        raise ValueError("Evaluation does not identify one policy checkpoint; supply --checkpoint")
    if not matches[0].is_absolute():
        raise ValueError("Evaluation checkpoint path must be absolute; supply --checkpoint")
    return matches[0]


def cache_config(state, selected, site, repo, run_root):
    """Inherit the entire learned input contract rather than generic teacher defaults."""
    cfg = deepcopy(state["config"])
    if state.get("stage") != "teacher" or cfg.get("stage", "teacher") != "teacher":
        raise ValueError("Cache adoption requires a teacher-stage checkpoint")
    if cfg.get("debug") or cfg["data"].get("synthetic"):
        raise ValueError("HPC cache preparation requires a real-data teacher")
    if cfg["model"].get("history_mode", "full") != "full":
        raise ValueError("The compact-reader experiment requires a full-history teacher")
    old_paths = deepcopy(cfg["paths"])
    for key in ("data_root", "cache_root"):
        cfg["paths"][key] = str(absolute_path(site.get("STMEM_" + key.upper()) or old_paths[key], repo))
    cfg["paths"]["run_root"] = str(run_root)

    def rebase(value, key):
        if not value or "{" in str(value):
            return value
        value = Path(value)
        if value.is_absolute():
            try:
                return str(Path(cfg["paths"][key]) / value.relative_to(Path(old_paths[key])))
            except ValueError:
                return str(value)
        # Resolve data/alignment.json with the same semantics as alignment_path.
        if key == "data_root" and value.parts and value.parts[0] == "data":
            return str(Path(cfg["paths"][key]).joinpath(*value.parts[1:]))
        return str(absolute_path(value, repo))

    cfg["data"]["alignment_path"] = rebase(cfg["data"].get("alignment_path", "data/alignment.json"), "data_root")
    spec = cfg["model"].get("visual_features")
    if isinstance(spec, dict):
        for key, root in (("checkpoint", "cache_root"), ("cache_dir", "data_root")):
            if key in spec:
                spec[key] = rebase(spec[key], root)
    cfg.setdefault("checkpoints", {})["teacher"] = str(selected)
    validate_config(cfg)
    validate_policy_config(state["config"], cfg)
    return cfg


def render_script(plan, site, cfg):
    directives = {
        "job-name": "stmem_cache_" + plan["name"], "partition": plan["partition"],
        "gres": plan["gres"], "nodes": "1", "ntasks": "1", "cpus-per-task": str(plan["cpus"]),
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
    lines = ["#!/usr/bin/env bash", *(f"#SBATCH --{k}={shlex.quote(v)}" for k, v in directives.items()),
             "set -euo pipefail", f"cd {shlex.quote(plan['repo'])}"]
    exported = {
        "STMEM_REPO_ROOT": plan["repo"], "STMEM_PYTHON": plan["python"],
        "STMEM_SITE_FILE": "/dev/null", "PYTHONPATH": str(Path(plan["repo"]) / "src"),
        "OMP_NUM_THREADS": str(plan["threads"]), "MKL_NUM_THREADS": str(plan["threads"]),
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        **{f"STMEM_{key.upper()}": cfg["paths"][key] for key in ("data_root", "cache_root", "run_root")},
    }
    for key, value in sorted(exported.items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append("unset RANK LOCAL_RANK WORLD_SIZE STMEM_FEATURE_CACHE_ROOT STMEM_SCRATCH_ROOT WANDB_RUN_ID WANDB_RESUME")
    if site.get("STMEM_ACTIVATE"):
        lines.append("source " + shlex.quote(site["STMEM_ACTIVATE"]))
    guard = (
        "from pathlib import Path\nfrom hpc.plan import source_hash,sha256\n"
        "checks = [\n"
        f"(source_hash(Path({plan['repo']!r})), {plan['source_sha256']!r}, 'Source changed since cache rendering'),\n"
        f"(sha256(Path({plan['config']!r})), {plan['config_sha256']!r}, 'Cache configuration changed'),\n"
        f"(sha256(Path({plan['checkpoint']!r})), {plan['checkpoint_sha256']!r}, 'Selected teacher changed'),\n"
        f"(sha256(Path({plan['selection']!r})), {plan['selection_sha256']!r}, 'Teacher selection record changed')]\n"
        "for observed, expected, label in checks:\n"
        "    if observed != expected:\n        raise RuntimeError(label)\n"
    )
    lines.append(shlex.join([plan["python"], "-c", guard]))
    lines.append('STMEM_CACHE_INDEX="${SLURM_ARRAY_TASK_ID:?Submit this as the generated array job}"')
    lines.append("export STMEM_STOP_FILE=" + shlex.quote(str(Path(plan["script"]).parent / "stops"))
                 + '/"${SLURM_ARRAY_JOB_ID:-local}_${STMEM_CACHE_INDEX}_${SLURM_JOB_ID:-local}.stop"')
    command = ["srun", "--ntasks=1", f"--cpus-per-task={plan['cpus']}", "--cpu-bind=cores", "--gpu-bind=none",
               plan["python"], "-u", "-m", "stmem.data.cache", "--config", plan["config"],
               "--checkpoint", plan["checkpoint"], "--device", "cuda", "--threads", str(plan["threads"])]
    lines.append("exec " + shlex.quote(str(Path(plan["repo"]) / "hpc/run_job.sh"))
                 + ' "$STMEM_STOP_FILE" ' + shlex.join(command) + ' --task-index "$STMEM_CACHE_INDEX"')
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Default: freeze checkpoint and render, no Slurm submission")
    action.add_argument("--submit", action="store_true")
    action.add_argument("--status", action="store_true", help="Read this named cache's progress; no data/model loading")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site.env")
    parser.add_argument("--evaluation", type=Path, help="Completed accepted FULL validation output; required except --status")
    parser.add_argument("--checkpoint", type=Path, help="Default: exact snapshot recorded by evaluation")
    parser.add_argument("--name", required=True, type=profile_name)
    parser.add_argument("--run-root", type=Path, help="Default: REPO/runs/NAME; ignores the site's training RUN_ROOT")
    parser.add_argument("--resume", action="store_true", help="Resubmit an inactive array; completed cache work is reused")
    parser.add_argument("--partition", default="gpuq")
    parser.add_argument("--gpu-type", default="", help="Empty default requests gpu:1")
    parser.add_argument("--cpus", type=positive_integer, default=8)
    parser.add_argument("--threads", type=positive_integer, default=4)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--walltime", type=walltime, default="24:00:00")
    parser.add_argument("--max-concurrent", type=positive_integer, default=4)
    args = parser.parse_args(argv)
    if not args.status and args.evaluation is None:
        parser.error("--evaluation is required except with --status")
    if args.threads > args.cpus:
        parser.error("--threads cannot exceed --cpus")
    if not re.fullmatch(r"[1-9][0-9]*(?:[KMGT]i?B?|[kmgt])?", args.mem):
        parser.error("Invalid --mem")
    if not args.partition or any(not re.fullmatch(r"[A-Za-z0-9_.-]*", v) for v in (args.partition, args.gpu_type)):
        parser.error("Invalid partition or GPU type")
    site = read_site(args.site)
    repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
    generated = repo / "hpc/generated" / args.name
    if args.status:
        try:
            from stmem.data.cache_builder import cache_status

            plan = json.loads((generated / "plan.json").read_text())
            path = Path(plan["config"])
            if sha256(path) != plan["config_sha256"]:
                raise ValueError("Saved cache configuration changed")
            cfg = yaml.safe_load(path.read_text())
            print(json.dumps(cache_status(cfg, Path(plan["checkpoint"])), indent=2))
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        return
    generated.mkdir(parents=True, exist_ok=True)
    with (generated / ".launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            prepare(args, site, repo, generated)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))


def prepare(args, site, repo, generated):
    evaluation = absolute_path(args.evaluation, repo)
    source = absolute_path(args.checkpoint, repo) if args.checkpoint else evaluated_checkpoint(evaluation)
    run_root = absolute_path(args.run_root or repo / "runs" / args.name, repo)
    plan_path, receipt_path = generated / "plan.json", generated / "submitted.json"
    previous = json.loads(plan_path.read_text()) if plan_path.exists() else None
    if args.resume and previous is None:
        raise ValueError("No existing cache plan to resume")
    if receipt_path.exists() and not args.resume:
        raise ValueError("Already submitted; use --resume after the array finishes or fails")
    if previous and (previous["evaluation"] != str(evaluation) or previous["run_root"] != str(run_root)):
        raise ValueError("Named plan refers to different inputs; choose a new --name")
    if args.submit and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        active = active_array_ids()
        if receipt["job_id"] in active:
            raise ValueError(f"Cache array {receipt['job_id']} still has active tasks; refusing duplicate submission")
    state = load_checkpoint(source)
    seed = state["config"]["seed"]
    selected = run_root / f"seed{seed}/teacher/selected.pt"
    cfg = cache_config(state, selected, site, repo, run_root)
    tasks = list(cfg["data"]["tasks"])
    if not tasks or len(set(tasks)) != len(tasks) or any(not re.fullmatch(r"[A-Za-z0-9_]+", task) for task in tasks):
        raise ValueError("Invalid or duplicate task names")
    complete = run_root / f"seed{seed}/cache/complete.json"
    if complete.exists() and args.submit:
        raise ValueError(f"Cache is already complete; inspect {complete}")
    signature = digest({"checkpoint": sha256(source), "config": cfg, "source": source_hash(repo),
                        "evaluation": str(evaluation)})
    if previous and previous["signature"] != signature:
        raise ValueError("Named cache scientific inputs or source changed; use a new --name")
    adopt_teacher(source, evaluation, selected)
    config_path = generated / "cache.yaml"
    payload = yaml.safe_dump(cfg, sort_keys=False)
    if previous and (not config_path.exists() or sha256(config_path) != previous["config_sha256"]):
        raise ValueError("Existing cache configuration was modified")
    if previous:
        if config_path.read_text() != payload:
            raise ValueError("Resolved cache configuration differs from the saved plan")
    else:
        temporary = config_path.with_suffix(".yaml.tmp")
        temporary.write_text(payload)
        temporary.replace(config_path)
    selection_path = selected.with_suffix(".selection.json")
    selection = json.loads(selection_path.read_text())
    logs = generated / "logs"
    logs.mkdir(exist_ok=True)
    plan = {
        "kind": "selected-teacher-encoder-cache", "name": args.name, "repo": str(repo),
        "python": site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python"),
        "evaluation": str(evaluation), "source_checkpoint": str(source), "run_root": str(run_root),
        "checkpoint": str(selected), "checkpoint_sha256": sha256(selected), "checkpoint_step": state["step"],
        "selection": str(selection_path), "selection_sha256": sha256(selection_path),
        "config": str(config_path), "config_sha256": sha256(config_path), "source_sha256": source_hash(repo),
        "signature": signature, "script": str(generated / "cache.sbatch"), "logs": str(logs),
        "tasks": tasks, "task_count": len(tasks), "array": f"0-{len(tasks)-1}%{min(args.max_concurrent,len(tasks))}",
        "gpus_per_task": 1, "cpus": args.cpus, "threads": args.threads, "mem": args.mem,
        "partition": args.partition, "gres": f"gpu:{args.gpu_type + ':' if args.gpu_type else ''}1",
        "walltime": args.walltime, "maximum_concurrent_allocations": min(args.max_concurrent,len(tasks)),
        "completion_record": str(complete), "cache_parent": cfg["paths"]["cache_root"],
        "evaluation_record": str(selected.parent / (selected.stem + ".evaluation")),
        "accepted_evaluation": selection.get("evaluation", selection.get("evaluations")),
        "accepted_macro_success": selection.get("macro_success"),
        "evaluation_execution": {key: value for key, value in selection.get("evaluation_protocol", {}).items()
                                 if key not in {"planned", "method", "split"}},
        "estimated_gpu_hours": "unknown; per-task progress and measured completion times are recorded",
    }
    Path(plan["script"]).write_text(render_script(plan, site, cfg))
    save_json(plan_path, plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.submit:
        result = subprocess.run(["sbatch", "--parsable", plan["script"]], check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise ValueError(f"Unexpected sbatch receipt: {result.stdout!r}")
        previous_receipts = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        save_json(receipt_path, {"job_id": job_id, "signature": signature, "previous": previous_receipts})
        print(f"Submitted cache array {job_id}: {len(tasks)} tasks, at most {plan['maximum_concurrent_allocations']} simultaneous GPUs")
    else:
        print("Dry-run only: selected teacher and cache plan saved; no Slurm job submitted.")


if __name__ == "__main__":
    main()
