#!/usr/bin/env python3
"""Continue one frozen initial router fit as a separate overfitting diagnostic.

Only --submit contacts Slurm. The original 5k checkpoint and initial labels are
immutable inputs; this job neither regenerates labels nor joins the refresh DAG.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import active_user_job_ids, positive_integer, read_site, sha256, source_hash  # noqa: E402
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from stmem.config import digest, run_dir, validate_config  # noqa: E402
from stmem.training.checkpoints import atomic_copy, atomic_json  # noqa: E402
from stmem.training.routers import _load_labels  # noqa: E402
from stmem.training.runtime import load_checkpoint  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text())


def require_hash(path, expected, label):
    if not Path(path).is_file() or sha256(Path(path)) != expected:
        raise ValueError(f"{label} is missing or changed: {path}")


def initial_checkpoint(repo, experiment, reader, objective):
    plan_path = repo / "hpc/generated" / experiment / "plan.json"
    plan = read_json(plan_path)
    entry = plan["configs"][f"initial_{objective}_{reader}"]
    config_path = Path(entry["path"])
    require_hash(config_path, entry["sha256"], "Original initial configuration")
    cfg = yaml.safe_load(config_path.read_text())
    validate_config(cfg)
    if cfg.get("debug") or cfg["data"].get("synthetic"):
        raise ValueError("HPC router diagnostics require real-data labels/checkpoints")
    if (cfg.get("stage") != "router" or cfg.get("reader") != reader
            or cfg["labels"].get("objective") != objective
            or cfg["labels"].get("use_refresh", False)
            or cfg["training"]["steps"] != 5000):
        raise ValueError("Expected the requested initial-only 5k router configuration")
    source = run_dir(cfg, "router") / "step_00005000.pt"
    state = load_checkpoint(source)
    if (state.get("stage") != "router" or state.get("step") != 5000
            or state.get("reader") != reader or state.get("config") != cfg
            or state.get("objective") != objective
            or state.get("integration_steps") != cfg["labels"]["integration_steps"]):
        raise ValueError("Initial router checkpoint differs from its immutable configuration")
    if not state.get("optimizer") or not state.get("rng_by_rank"):
        raise ValueError("Initial checkpoint lacks optimizer/RNG state for continuation")
    groups = state["optimizer"].get("param_groups", [])
    if not groups or any(group.get("lr") != cfg["training"]["lr"]
                         or group.get("weight_decay") != cfg["training"]["weight_decay"] for group in groups):
        raise ValueError("Initial optimizer learning rate/weight decay differs from its configuration")
    # Full schema, episode partition, candidate and attempted-ID checks. This
    # consumes existing label arrays; no teacher, raw data or GPU is needed.
    _, paths, _, _, identity = _load_labels(cfg)
    labels = [{"path": str(path.resolve()), "sha256": sha256(path)} for path in paths]
    if state.get("label_hashes") != [entry["sha256"] for entry in labels]:
        raise ValueError("Initial label bytes differ from those used by the source router")
    if state.get("label_identity") != identity:
        raise ValueError("Initial router label provenance differs from its dataset")
    if (identity["teacher"] != plan["teacher_sha256"]
            or identity["compact"] != plan["compact_sha256"]
            or identity["cache"] != plan["cache_manifest_digest"]):
        raise ValueError("Initial router labels differ from the frozen experiment inputs")
    return cfg, state, source, labels, identity, plan_path


def signature(plan):
    return digest({key: value for key, value in plan.items() if key != "signature"})


def ready_plan(path):
    """Recheck every frozen input at allocation startup, before CUDA work."""
    plan = read_json(path)
    if signature(plan) != plan["signature"]:
        raise ValueError("Router diagnostic plan changed")
    if source_hash(Path(plan["repo"])) != plan["source_sha256"]:
        raise ValueError("Source changed since rendering; use a new diagnostic name")
    for source, expected in plan["launcher_hashes"].items():
        require_hash(source, expected, "Launcher source")
    require_hash(plan["checkpoint"], plan["checkpoint_sha256"], "Frozen initial router")
    require_hash(plan["config"], plan["config_sha256"], "Diagnostic configuration")
    for label in plan["labels"]:
        require_hash(label["path"], label["sha256"], "Initial label shard")
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    last = Path(plan["run_directory"]) / "last.pt"
    if last.exists():
        state = load_checkpoint(last)
        if (state.get("stage") != "router" or state.get("config") != cfg
                or state.get("label_hashes") != [item["sha256"] for item in plan["labels"]]
                or state.get("label_identity") != plan["label_identity"]
                or not plan["start_step"] <= state.get("step", -1) <= plan["steps"]):
            raise ValueError("Existing diagnostic checkpoint belongs to different work")
    return plan


def render_script(plan, site, cfg):
    directives = {
        "job-name": "stmem_" + plan["name"], "partition": plan["partition"],
        "gres": "gpu:1", "nodes": "1", "ntasks": "1", "cpus-per-task": str(plan["cpus"]),
        "mem": plan["mem"], "time": plan["walltime"], "signal": "B:USR1@180",
        "chdir": plan["repo"], "output": str(Path(plan["logs"]) / "%x_%j.out"),
        "error": str(Path(plan["logs"]) / "%x_%j.err"),
    }
    for key in ("ACCOUNT", "QOS", "CONSTRAINT"):
        if site.get("STMEM_" + key):
            directives[key.lower()] = site["STMEM_" + key]
    if any("\n" in value or "\r" in value for value in directives.values()):
        raise ValueError("Newline in Slurm directive")
    lines = ["#!/usr/bin/env bash", *(f"#SBATCH --{key}={shlex.quote(value)}" for key, value in directives.items()),
             "set -euo pipefail", "cd " + shlex.quote(plan["repo"])]
    if site.get("STMEM_ACTIVATE"):
        lines.append("source " + shlex.quote(site["STMEM_ACTIVATE"]))
    exports = {
        "STMEM_REPO_ROOT": plan["repo"], "STMEM_PYTHON": plan["python"], "STMEM_SITE_FILE": "/dev/null",
        "PYTHONPATH": str(Path(plan["repo"]) / "src"), "PYTHONUNBUFFERED": "1", "STMEM_ALLOCATED_GPUS": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        **{f"STMEM_{key.upper()}": cfg["paths"][key] for key in ("data_root", "cache_root", "run_root")},
        "WANDB_PROJECT": plan["wandb_project"],
        "WANDB_MODE": site.get("WANDB_MODE") or "online", "WANDB_ENTITY": site.get("WANDB_ENTITY") or "",
        "WANDB_DIR": site.get("WANDB_DIR") or "",
    }
    lines.append("unset RANK LOCAL_RANK WORLD_SIZE STMEM_FEATURE_CACHE_ROOT STMEM_SCRATCH_ROOT WANDB_RUN_ID WANDB_RESUME")
    lines.extend(f"export {key}={shlex.quote(str(value))}" for key, value in sorted(exports.items()))
    guard = f"from hpc.overfit_router import ready_plan; ready_plan({plan['plan_path']!r})"
    lines.append(shlex.join([plan["python"], "-c", guard]))
    last = Path(plan["run_directory"]) / "last.pt"
    lines.append("resume_checkpoint=" + shlex.quote(plan["checkpoint"]))
    lines.append("if [[ -f " + shlex.quote(str(last)) + " ]]; then resume_checkpoint=" + shlex.quote(str(last)) + "; fi")
    lines.append("export STMEM_STOP_FILE=" + shlex.quote(str(Path(plan["generated"]) / "stops"))
                 + '/"${SLURM_JOB_ID:-local_$$}.stop"')
    command = ["srun", "--ntasks=1", f"--cpus-per-task={plan['cpus']}", "--cpu-bind=cores", "--gpu-bind=none",
               plan["python"], "-u", "-m", "stmem.train", "--stage", "router", "--config", plan["config"]]
    lines.append("exec " + shlex.quote(str(Path(plan["repo"]) / "hpc/run_job.sh"))
                 + ' "$STMEM_STOP_FILE" ' + shlex.join(command)
                 + ' --resume "$resume_checkpoint" --stop-file "$STMEM_STOP_FILE"')
    return "\n".join(lines) + "\n"


def status(plan):
    last = Path(plan["run_directory"]) / "last.pt"
    step = load_checkpoint(last)["step"] if last.exists() else plan["start_step"]
    complete = Path(plan["run_directory"]) / "complete.json"
    record = read_json(complete) if complete.is_file() else {}
    done = (last.is_file() and step == plan["steps"] and record.get("status") == "complete"
            and record.get("completed_steps") == plan["steps"]
            and record.get("checkpoint_sha256") == sha256(last))
    return {"name": plan["name"], "status": "complete" if done else "incomplete",
            "completed_steps": step, "planned_steps": plan["steps"],
            "run_directory": plan["run_directory"], "logs": plan["logs"]}


def submit(plan, resume):
    ready_plan(plan["plan_path"])
    if status(plan)["status"] == "complete":
        print("Diagnostic already complete; no job submitted.")
        return
    receipt = Path(plan["generated"]) / "submitted.json"
    old = read_json(receipt) if receipt.exists() else None
    if old:
        if old.get("signature") != plan["signature"]:
            raise ValueError("Submission receipt belongs to another diagnostic")
        if old.get("state") == "submitting":
            raise ValueError("Submission outcome is uncertain; reconcile the receipt with Slurm before retrying")
        if not resume:
            raise ValueError("Diagnostic already submitted; use --resume after it stops")
        if old["job_id"] in active_user_job_ids():
            raise ValueError("Diagnostic job is still active; refusing a duplicate")
    command = ["sbatch", "--parsable", plan["script"]]
    atomic_json({"signature": plan["signature"], "state": "submitting", "command": command,
                 "previous": old}, receipt)
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdecimal():
        raise ValueError("Unexpected sbatch receipt; reconcile submission before retrying")
    atomic_json({"signature": plan["signature"], "state": "submitted", "job_id": job_id,
                 "command": command, "previous": old}, receipt)
    print(f"Submitted router overfitting diagnostic: {job_id}")


def prepare(args, site, repo, generated):
    path = generated / "plan.json"
    previous = read_json(path) if path.exists() else None
    if args.resume and previous is None:
        raise ValueError("No existing diagnostic to resume")
    cfg, state, source, labels, identity, original_plan = initial_checkpoint(
        repo, args.experiment, args.reader, args.objective,
    )
    if args.steps <= state["step"]:
        raise ValueError("Requested total steps must exceed the initial checkpoint step")
    original_root = Path(read_json(original_plan)["run_root"]).resolve()
    run_root = repo / "runs" / args.name
    if run_root.resolve().is_relative_to(original_root):
        raise ValueError("Diagnostic output must be separate from the original experiment")
    if not previous and run_root.exists() and any(run_root.iterdir()):
        raise ValueError("A fresh diagnostic requires an empty output directory")
    snapshot = generated / "frozen/initial_router.pt"
    expected = sha256(source)
    if previous and (str(source) != previous["source_checkpoint"] or expected != previous["checkpoint_sha256"]):
        raise ValueError("Original immutable router checkpoint changed")
    atomic_copy(source, snapshot, immutable=True)
    if sha256(snapshot) != expected or sha256(source) != expected:
        raise ValueError("Initial router changed during snapshotting")
    cfg = deepcopy(cfg)
    cfg["paths"]["run_root"] = str(run_root)
    cfg.setdefault("tracking", {})["project"] = args.wandb_project
    cfg["training"].update(
        steps=args.steps, schedule_steps=max(cfg["training"].get("schedule_steps", 0), args.steps),
        router_diagnostics=True, router_device="cuda", workers=0, log_every=1000,
        checkpoint_every=10000,
        checkpoint_steps=sorted({step for step in (10000, 25000, 50000, args.steps) if state["step"] < step <= args.steps}),
    )
    validate_config(cfg)
    if not cfg.get("tracking", {}).get("enabled", True) or site.get("WANDB_MODE") == "disabled":
        raise ValueError("Router overfitting diagnostic requires W&B logging")
    config = generated / "router.yaml"
    payload = yaml.safe_dump(cfg, sort_keys=False)
    if config.exists() and config.read_text() != payload:
        raise ValueError("Named diagnostic configuration changed; use a new --name")
    config.write_text(payload)
    plan = {
        "kind": "router-overfit-diagnostic", "name": args.name, "experiment": args.experiment,
        "repo": str(repo), "python": site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python"),
        "generated": str(generated), "plan_path": str(path), "script": str(generated / "overfit.sbatch"),
        "source_checkpoint": str(source), "checkpoint": str(snapshot), "checkpoint_sha256": expected,
        "original_plan": str(original_plan), "original_plan_sha256": sha256(original_plan),
        "config": str(config), "config_sha256": sha256(config), "labels": labels, "label_identity": identity,
        "run_root": str(run_root), "run_directory": str(run_dir(cfg, "router")), "logs": str(generated / "logs"),
        "reader": args.reader, "objective": args.objective, "integration_steps": cfg["labels"]["integration_steps"],
        "start_step": state["step"], "steps": args.steps, "additional_steps": args.steps - state["step"],
        "source_sha256": source_hash(repo),
        "launcher_hashes": {str(p): sha256(p) for p in (Path(__file__), Path(__file__).with_name("run_job.sh"),
                                                         Path(__file__).with_name("plan.py"),
                                                         Path(__file__).with_name("profile_memory.py"))},
        "partition": args.partition, "gres": "gpu:1", "gpus": 1, "cpus": args.cpus, "mem": args.mem,
        "walltime": args.walltime, "initial_labels_only": True, "wandb_project": args.wandb_project,
        "diagnostic": "longer fitting on unchanged initial labels; not a matched four-router benchmark",
    }
    plan["signature"] = signature(plan)
    if previous is not None and previous != plan:
        raise ValueError("Named diagnostic inputs/resources/source changed; use a new --name")
    Path(plan["logs"]).mkdir(parents=True, exist_ok=True)
    Path(plan["script"]).write_text(render_script(plan, site, cfg))
    atomic_json(plan, path)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.submit:
        submit(plan, args.resume)
    else:
        print("Dry-run only: frozen initial router, diagnostic configuration and one job script; no Slurm contact.")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    action.add_argument("--status", action="store_true", help="Read recorded filesystem progress only")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site_b200_4gpu.env")
    parser.add_argument("--experiment", type=profile_name, default="memory_fv_seed0_routers")
    parser.add_argument("--name", type=profile_name)
    parser.add_argument("--reader", choices=("tree", "flat"), default="tree")
    parser.add_argument("--objective", choices=("velocity", "final_action"), default="velocity")
    parser.add_argument("--steps", type=positive_integer, default=100000, help="Total updates, including the initial 5000")
    parser.add_argument("--wandb-project", type=profile_name, default="chronotree-routers",
                        help="Dedicated router diagnostics project; overrides WANDB_PROJECT from the site/environment")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partition", default="gpuq")
    parser.add_argument("--cpus", type=positive_integer, default=4)
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--walltime", type=walltime, default="01:00:00")
    args = parser.parse_args(argv)
    steps_name = f"{args.steps // 1000}k" if args.steps % 1000 == 0 else str(args.steps)
    args.name = args.name or f"{args.experiment}_overfit_{args.reader}_{args.objective}_{steps_name}"
    if args.name == args.experiment:
        parser.error("Diagnostic --name must differ from --experiment")
    if not re.fullmatch(r"[1-9][0-9]*(?:[KMGT]i?B?|[kmgt])?", args.mem):
        parser.error("Invalid --mem")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.partition):
        parser.error("Invalid --partition")
    try:
        site = read_site(args.site)
        repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
        generated = repo / "hpc/generated" / args.name
        if args.status:
            result = status(read_json(generated / "plan.json"))
            print(json.dumps(result, indent=2))
            return result
        generated.mkdir(parents=True, exist_ok=True)
        with (generated / ".launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return prepare(args, site, repo, generated)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
