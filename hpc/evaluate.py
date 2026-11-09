#!/usr/bin/env python3
"""Render one parallel validation allocation; only --submit contacts Slurm."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import secrets
import shlex
import shutil
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import (  # noqa: E402
    WANDB_SITE_VARIABLES, active_user_job_ids, config_hashes, positive_integer,
    read_site, resolved_config, save_json, sha256, source_hash,
)
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from stmem.config import digest, validate_config  # noqa: E402
from stmem.evaluation.execution import evaluation_execution_config  # noqa: E402
from stmem.evaluation.selection import validate_selection_protocol  # noqa: E402
from stmem.training.runtime import load_checkpoint  # noqa: E402


def nonnegative(value):
    count = int(value)
    if count < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return count


def snapshot_checkpoint(source, destination):
    """Copy a stable checkpoint once; a queued job never follows last.pt."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    before = sha256(source)
    try:
        shutil.copyfile(source, temporary)
        if sha256(temporary) != before or sha256(source) != before:
            raise ValueError("Source checkpoint changed while copying; rerun to take a coherent snapshot")
        state = load_checkpoint(temporary)
        temporary.replace(destination)
        return state, before
    finally:
        temporary.unlink(missing_ok=True)


def prepare_router_checkpoint(source, snapshot, previous, checkpoint_hash, method):
    """Pin a learned reader to its exact compact policy, including on resume."""
    record_path = snapshot.with_suffix(".json")
    if previous:
        if previous.get("source_router_checkpoint") != str(source):
            raise ValueError("Named plan has a different router source; choose a new --name")
        router_hash = previous.get("router_checkpoint_sha256")
        if not snapshot.is_file() or sha256(snapshot) != router_hash:
            raise ValueError("Immutable router checkpoint snapshot is missing or changed")
        state = load_checkpoint(snapshot)
    elif snapshot.exists():
        if not record_path.is_file():
            raise ValueError("Router snapshot has no identity record; choose a new --name")
        record = json.loads(record_path.read_text())
        router_hash = record.get("sha256")
        if record.get("source_checkpoint") != str(source) or sha256(snapshot) != router_hash:
            raise ValueError("Existing router snapshot identity differs; choose a new --name")
        state = load_checkpoint(snapshot)
    else:
        state, router_hash = snapshot_checkpoint(source, snapshot)
        save_json(record_path, {"source_checkpoint": str(source), "sha256": router_hash,
                                "step": state["step"]})
    if state.get("stage") != "router":
        raise ValueError("--router-checkpoint must be a trained router checkpoint")
    if state.get("reader") != method.lower() or state["config"].get("reader") != method.lower():
        raise ValueError("Router reader must match --method TREE or FLAT")
    if state.get("provenance", {}).get("compact") != checkpoint_hash:
        raise ValueError("Router was trained against a different compact checkpoint (SHA256 mismatch)")
    return state, router_hash


def render_script(plan, site, cfg):
    directives = {
        "job-name": f"stmem_eval_{plan['name']}", "nodes": "1", "ntasks": "1",
        "cpus-per-task": str(plan["cpus"]), "mem": plan["mem"], "time": plan["walltime"],
        "partition": plan["partition"], "gres": plan["gres"],
        "output": str(Path(plan["logs"]) / "%x_%j.out"),
        "error": str(Path(plan["logs"]) / "%x_%j.err"),
        "chdir": plan["repo"], "signal": "B:USR1@180",
    }
    for key in ("ACCOUNT", "QOS", "CONSTRAINT"):
        if site.get("STMEM_" + key):
            directives[key.lower()] = site["STMEM_" + key]
    if any("\n" in value or "\r" in value for value in directives.values()):
        raise ValueError("Newline in Slurm directive")
    lines = ["#!/usr/bin/env bash", *(f"#SBATCH --{k}={shlex.quote(v)}" for k, v in directives.items())]
    if plan["exclusive"]:
        lines.append("#SBATCH --exclusive")
    lines.extend(["set -euo pipefail", f"cd {shlex.quote(plan['repo'])}"])
    exported = {
        "STMEM_REPO_ROOT": plan["repo"], "STMEM_PYTHON": plan["python"],
        "STMEM_SITE_FILE": "/dev/null", "PYTHONPATH": str(Path(plan["repo"]) / "src"),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        **{f"STMEM_{name.upper()}": cfg["paths"][name] for name in ("data_root", "cache_root", "run_root")},
    }
    for name in WANDB_SITE_VARIABLES:
        if name in site:
            exported[name] = site[name]
    for name, value in sorted(exported.items()):
        lines.append(f"export {name}={shlex.quote(str(value))}" if value else f"unset {name}")
    lines.append("unset WANDB_RUN_ID WANDB_RESUME RANK LOCAL_RANK WORLD_SIZE")
    if site.get("STMEM_ACTIVATE"):
        lines.append(f"source {shlex.quote(site['STMEM_ACTIVATE'])}")
    # Fail before inference if a queued job's implementation was replaced.
    code = (
        "from pathlib import Path; from hpc.plan import source_hash; "
        f"assert source_hash(Path({plan['repo']!r})) == {plan['source_sha256']!r}, "
        "'Source changed since rendering; render a new named evaluation'"
    )
    lines.append(shlex.join([plan["python"], "-c", code]))
    lines.append(
        f"export STMEM_STOP_FILE={shlex.quote(str(Path(plan['script']).parent / 'stops'))}"
        '/"${SLURM_JOB_ID:-local_$$}.stop"'
    )
    lines.append(
        "exec " + shlex.quote(str(Path(plan["repo"]) / "hpc/run_job.sh"))
        + ' "$STMEM_STOP_FILE" ' + shlex.join(plan["command"])
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Default: snapshot and render only")
    action.add_argument("--submit", action="store_true", help="Submit exactly one parallel allocation")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site.env")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Default: configuration saved inside the checkpoint")
    parser.add_argument("--name", type=profile_name, help="Stable generated plan name; default UTC plus suffix")
    parser.add_argument("--output", type=Path, help="Default: REPO/runs/evaluation/NAME")
    parser.add_argument("--resume", action="store_true", help="Resume this existing named plan and output")
    parser.add_argument("--method", choices=("FULL", "CURRENT", "UNIFORM", "PERCEIVER", "TREE", "FLAT"), default="FULL")
    parser.add_argument("--selection-family", choices=("tree", "flat"),
                        help="Random compact checkpoint selection; must match TREE/FLAT method")
    parser.add_argument("--router-checkpoint", type=Path,
                        help="Learned TREE/FLAT router trained against this exact compact checkpoint")
    parser.add_argument("--wandb-project", help="Override the checkpoint and site W&B project")
    parser.add_argument("--budget", type=positive_integer, help="History token budget; default from checkpoint")
    parser.add_argument("--gpus", type=positive_integer, default=4)
    parser.add_argument("--workers-per-gpu", type=positive_integer, default=4)
    parser.add_argument("--threads-per-worker", type=positive_integer, default=2)
    parser.add_argument("--cpus", type=positive_integer, default=64)
    parser.add_argument("--mem", default="128G")
    parser.add_argument("--partition", default="gpuq")
    parser.add_argument("--gpu-type", default="", help="Empty default requests untyped gpu:N, as required here")
    parser.add_argument("--walltime", type=walltime, default="06:00:00")
    parser.add_argument("--exclusive", action="store_true", help="Reserve the full node from other jobs")
    parser.add_argument("--episode-count", type=positive_integer, default=4, help="Official validation episodes per task")
    parser.add_argument("--video-cap", type=nonnegative, default=10, help="Total cap across all workers")
    parser.add_argument("--execute-actions", type=positive_integer)
    args = parser.parse_args(argv)
    if args.selection_family is not None and args.method != args.selection_family.upper():
        parser.error("--selection-family must match --method TREE or FLAT")
    if args.router_checkpoint is not None and args.selection_family is not None:
        parser.error("--router-checkpoint and --selection-family are mutually exclusive")
    if args.router_checkpoint is not None and args.method not in ("TREE", "FLAT"):
        parser.error("--router-checkpoint requires --method TREE or FLAT")
    if args.method in ("TREE", "FLAT") and args.selection_family is None and args.router_checkpoint is None:
        parser.error("TREE/FLAT requires --selection-family for random selection or --router-checkpoint")
    if args.wandb_project is not None and not args.wandb_project.strip():
        parser.error("--wandb-project must be nonempty")
    if args.resume and not args.name:
        parser.error("--resume requires the original --name")
    if args.episode_count > 50:
        parser.error("Official validation supplies at most 50 episodes per task")
    if args.gpus * args.workers_per_gpu * args.threads_per_worker > args.cpus:
        parser.error("Requested worker threads exceed allocated --cpus")
    if not re.fullmatch(r"[1-9][0-9]*(?:[KMGT]i?B?|[kmgt])?", args.mem):
        parser.error("--mem must be a positive Slurm memory quantity, for example 128G")
    for field in ("partition", "gpu_type"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]*", getattr(args, field)):
            parser.error(f"Invalid --{field.replace('_', '-')}")
    if not args.partition:
        parser.error("--partition must be nonempty")
    site = read_site(args.site)
    if args.wandb_project is not None:
        site["WANDB_PROJECT"] = args.wandb_project
    repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
    name = args.name or ("evaluation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                         + "_" + secrets.token_hex(3))
    generated = repo / "hpc/generated" / name
    generated.mkdir(parents=True, exist_ok=True)
    # Protect render+submit as a transaction, including two concurrent login shells.
    with (generated / ".launcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another launcher is using this --name")
        try:
            _prepare_and_submit(args, site, repo, name, generated)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))


def _prepare_and_submit(args, site, repo, name, generated):
    plan_path = generated / "plan.json"
    previous = json.loads(plan_path.read_text()) if plan_path.exists() else None
    if args.resume and previous is None:
        raise ValueError("No existing named plan to resume")
    source = absolute_path(args.checkpoint, repo)
    output = absolute_path(args.output or repo / "runs/evaluation" / name, repo)
    snapshot = repo / "runs/evaluation_inputs" / name / "checkpoint.pt"
    snapshot_record = snapshot.with_suffix(".json")
    receipt_path = generated / "submitted.json"
    if receipt_path.exists() and not args.resume:
        raise ValueError("This plan was already submitted; use --resume or a new --name")
    if (output / "complete.json").exists():
        raise ValueError("Evaluation is already complete; choose a new --name for a new experiment")
    if (output / "manifest.json").exists() and not args.resume:
        raise ValueError("Evaluation output exists; use --resume")
    if previous:
        if previous["source_checkpoint"] != str(source) or previous["output"] != str(output):
            raise ValueError("Named plan has a different source/output; choose a new --name")
        if not snapshot.is_file() or sha256(snapshot) != previous["checkpoint_sha256"]:
            raise ValueError("Immutable checkpoint snapshot is missing or changed")
        state = load_checkpoint(snapshot)
        checkpoint_hash = previous["checkpoint_sha256"]
    elif snapshot.exists():
        # A preflight error after the atomic copy can leave an input snapshot
        # without a rendered plan. Its separate identity permits a safe retry.
        if not snapshot_record.is_file():
            raise ValueError("Checkpoint snapshot has no identity record; choose a new --name")
        record = json.loads(snapshot_record.read_text())
        if record.get("source_checkpoint") != str(source) or sha256(snapshot) != record.get("sha256"):
            raise ValueError("Existing checkpoint snapshot identity differs; choose a new --name")
        state = load_checkpoint(snapshot)
        checkpoint_hash = record["sha256"]
    else:
        state, checkpoint_hash = snapshot_checkpoint(source, snapshot)
        save_json(snapshot_record, {"source_checkpoint": str(source), "sha256": checkpoint_hash,
                                    "step": state["step"]})
    router_snapshot = None
    router_fields = {}
    if args.router_checkpoint is not None:
        if state.get("stage") != "compact":
            raise ValueError("Learned router evaluation requires a compact policy checkpoint")
        router_source = absolute_path(args.router_checkpoint, repo)
        router_snapshot = snapshot.parent / "router.pt"
        router_state, router_hash = prepare_router_checkpoint(
            router_source, router_snapshot, previous, checkpoint_hash, args.method,
        )
        router_fields = {
            "source_router_checkpoint": str(router_source), "router_checkpoint": str(router_snapshot),
            "router_checkpoint_sha256": router_hash, "router_checkpoint_step": router_state["step"],
        }
    elif previous and previous.get("router_checkpoint") is not None:
        raise ValueError("Named plan requires its original --router-checkpoint")
    validate_selection_protocol(
        args.method, args.selection_family, router_snapshot, checkpoint_stage=state.get("stage"), split="val",
    )
    cfg = deepcopy(state["config"])
    if args.config:
        path = absolute_path(args.config, repo)
        config_hashes(path)
        cfg = resolved_config(path)
    if any(c.get("debug") or c.get("data", {}).get("synthetic") for c in (cfg, state["config"])):
        raise ValueError("HPC evaluation requires real-data checkpoint/configuration")
    for env, key in (("STMEM_DATA_ROOT", "data_root"), ("STMEM_CACHE_ROOT", "cache_root")):
        cfg["paths"][key] = str(absolute_path(site.get(env) or cfg["paths"][key], repo))
    cfg["paths"]["run_root"] = str(absolute_path(cfg["paths"]["run_root"], repo))
    cfg["evaluation"].update(episode_count=args.episode_count, video_cap=args.video_cap, split="val")
    if args.wandb_project is not None:
        cfg.setdefault("tracking", {})["project"] = args.wandb_project
    validate_config(cfg)
    effective, override = evaluation_execution_config(
        state["config"], cfg, execute_actions=args.execute_actions, split="val",
    )
    budget = args.budget if args.budget is not None else cfg["model"]["budget"]
    if not cfg.get("tracking", {}).get("enabled", True) or (
        site.get("WANDB_MODE") or cfg.get("tracking", {}).get("mode", "offline")
    ) == "disabled":
        raise ValueError("The parent evaluation requires W&B tracking (online or offline)")
    requested_workers = args.gpus * args.workers_per_gpu
    episode_total = len(cfg["data"]["tasks"]) * args.episode_count
    config = generated / "evaluation.yaml"
    python = site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python")
    command = ["srun", "--ntasks=1", f"--cpus-per-task={args.cpus}", "--cpu-bind=cores", "--gpu-bind=none",
               python, "-u", "-m", "stmem.evaluation.parallel",
               "--config", str(config), "--checkpoint", str(snapshot), "--output", str(output),
               "--method", args.method, "--split", "val", "--episode-count", str(args.episode_count),
               "--num-gpus", str(args.gpus), "--workers-per-gpu", str(args.workers_per_gpu),
               "--threads-per-worker", str(args.threads_per_worker), "--video-cap", str(args.video_cap)]
    if args.execute_actions is not None:
        command.extend(["--execute-actions", str(args.execute_actions)])
    if args.budget is not None:
        command.extend(["--budget", str(budget)])
    if args.selection_family is not None:
        command.extend(["--selection-family", args.selection_family])
    if router_snapshot is not None:
        command.extend(["--router-checkpoint", str(router_snapshot)])
    if args.resume:
        command.append("--resume")
    signature_inputs = {
        "checkpoint_sha256": checkpoint_hash, "config": cfg, "method": args.method,
        "gpus": args.gpus, "workers_per_gpu": args.workers_per_gpu,
        "threads_per_worker": args.threads_per_worker, "execution_override": override,
        "source_sha256": source_hash(repo),
    }
    if args.selection_family is not None or args.budget is not None:
        signature_inputs.update(selection_family=args.selection_family, budget=budget)
    if router_snapshot is not None:
        signature_inputs["router_checkpoint_sha256"] = router_fields["router_checkpoint_sha256"]
    signature = digest(signature_inputs)
    if previous and previous["signature"] != signature:
        raise ValueError("Named evaluation inputs, code or worker assignment changed; choose a new --name")
    logs = generated / "logs"
    plan = {
        "kind": "parallel-validation", "name": name, "repo": str(repo), "python": python,
        "source_checkpoint": str(source), "checkpoint": str(snapshot), "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": state["step"], "config": str(config), "output": str(output),
        "script": str(generated / "evaluate.sbatch"), "logs": str(logs), "command": command,
        "source_sha256": signature_inputs["source_sha256"], "signature": signature,
        "method": args.method, "selection_family": args.selection_family, "budget": budget,
        "split": "val", "episode_count_per_task": args.episode_count,
        "tasks": cfg["data"]["tasks"], "planned_episodes": episode_total,
        "video_cap_total": args.video_cap, "gpus": args.gpus,
        "workers_per_gpu": args.workers_per_gpu, "requested_workers": requested_workers,
        "num_shards": min(requested_workers, episode_total), "threads_per_worker": args.threads_per_worker,
        "cpus": args.cpus, "mem": args.mem, "partition": args.partition,
        "gres": f"gpu:{args.gpu_type + ':' if args.gpu_type else ''}{args.gpus}",
        "walltime": args.walltime, "exclusive": args.exclusive,
        "job_count": 1, "nodes": 1, "slurm_tasks": 1, "resume": args.resume,
        "execution": override or {"execute_actions": effective["model"]["execute"],
                                   "action_stride": effective["model"].get("action_stride", 1)},
        "estimated_gpu_hours": "unknown; throughput must be measured on the allocated hardware",
        **router_fields,
    }
    script = render_script(plan, site, cfg)
    if args.submit and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["job_id"] in active_user_job_ids():
            raise ValueError(f"Evaluation job {receipt['job_id']} is still active; no duplicate submitted")
    logs.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    Path(plan["script"]).write_text(script)
    save_json(plan_path, plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.submit:
        print("Dry-run only: checkpoint snapshot, configuration, sbatch file and plan are ready.")
        print(f"Use --name {shlex.quote(name)} and --submit with the same options to launch.")
        return
    result = subprocess.run(["sbatch", "--parsable", plan["script"]], check=True,
                            capture_output=True, text=True)
    raw = result.stdout.strip()
    if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", raw):
        raise RuntimeError(f"sbatch returned an unexpected job identifier: {raw!r}")
    receipt = {"job_id": raw.split(";", 1)[0], "script": plan["script"], "output": str(output),
               "signature": signature, "resume": args.resume,
               "submitted_utc": datetime.now(timezone.utc).isoformat()}
    save_json(receipt_path, receipt)
    save_json(generated / f"submitted_{receipt['job_id']}.json", receipt)
    print(f"Submitted evaluation job {receipt['job_id']}; results: {output}; logs: {logs}")


if __name__ == "__main__":
    main()
