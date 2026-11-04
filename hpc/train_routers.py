#!/usr/bin/env python3
"""Render or submit a matched velocity/final-action router experiment.

The selected teacher/cache and a user-designated immutable compact snapshot own
all scientific inputs. Only --submit contacts Slurm. Four independent one-GPU
label workers share teacher work across both readers and both objectives.
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

import h5py
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import active_user_job_ids, nonnegative_integer, positive_integer, read_site, sha256, source_hash  # noqa: E402
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from hpc.train_compact import inspect_cache, inspect_hierarchy, read_json, require_hash  # noqa: E402
from stmem.config import dependency, digest, run_dir, validate_config  # noqa: E402
from stmem.training.checkpoints import atomic_copy, atomic_json  # noqa: E402
from stmem.training.runtime import load_checkpoint, validate_policy_config  # noqa: E402

OBJECTIVES = ("velocity", "final_action")
READERS = ("tree", "flat")


def variant(objective, steps):
    return "velocity" if objective == "velocity" else f"final_action_{steps}"


def immutable_yaml(path, cfg):
    payload = yaml.safe_dump(cfg, sort_keys=False)
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Immutable configuration changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def inspect_compact(path, cache_cfg, cache_plan, manifest):
    """Prove compact inputs and frozen encoder/normalizer weights match the cache."""
    state = load_checkpoint(path)
    if state.get("stage") != "compact":
        raise ValueError("--compact must identify a compact policy checkpoint")
    cfg = state["config"]
    validate_config(cfg)
    validate_policy_config(cache_cfg, cfg)
    if cfg.get("debug") or cfg["data"].get("synthetic"):
        raise ValueError("This launcher requires a real-data compact checkpoint")
    if cfg["seed"] != cache_cfg["seed"] or cfg["data"] != cache_cfg["data"]:
        raise ValueError("Compact checkpoint changed cache task/data/seed identity")
    for key in ("data_root", "cache_root"):
        if cfg["paths"][key] != cache_cfg["paths"][key]:
            raise ValueError(f"Compact checkpoint changed cached {key}")
    if dependency(cfg, "teacher") != Path(cache_plan["checkpoint"]):
        raise ValueError("Compact checkpoint used a different selected teacher")
    dependencies = state.get("provenance", {}).get("dependencies", {})
    if (dependencies.get("teacher") != cache_plan["checkpoint_sha256"]
            or dependencies.get("cache") != digest(manifest)):
        raise ValueError("Compact checkpoint teacher/cache provenance mismatch")
    teacher = load_checkpoint(cache_plan["checkpoint"])
    if teacher.get("stage") != "teacher":
        raise ValueError("Selected teacher checkpoint has the wrong stage")
    validate_policy_config(teacher["config"], cfg)
    for prefix in ("encoder.", "text_encoder.", "normalizer."):
        expected = {key for key in teacher["model"] if key.startswith(prefix)}
        actual = {key for key in state["model"] if key.startswith(prefix)}
        if not expected or expected != actual:
            raise ValueError(f"Missing or changed frozen compact state: {prefix}")
        if any(not torch.equal(teacher["model"][key], state["model"][key]) for key in expected):
            raise ValueError(f"Compact frozen encoder/action normalizer differs: {prefix}")
    return state


def snapshot_compact(source, destination, previous=None):
    """Pin the supplied last.pt once; later resumes use these immutable bytes."""
    selection = destination.with_suffix(".selection.json")
    if previous:
        if str(source) != previous["compact_source"]:
            raise ValueError("This experiment already froze a different compact source")
        require_hash(destination, previous["compact_sha256"], "Frozen compact snapshot")
        require_hash(selection, previous["compact_selection_sha256"], "Compact selection record")
        return
    before = sha256(source)
    atomic_copy(source, destination, immutable=True)
    if sha256(destination) != before or sha256(source) != before:
        raise ValueError("Compact source changed while copying; stop training or choose an immutable step checkpoint")
    state = load_checkpoint(destination)
    record = {
        "schema": "user_designated_frozen_compact_v1", "sha256": before,
        "source": str(source), "step": state["step"], "stage": state["stage"],
        "selection_method": "user_designated_checkpoint",
        "matched_validation_selection": False,
        "note": "Frozen for the requested router-objective ablation; no matched validation selection is claimed.",
    }
    if selection.exists() and read_json(selection) != record:
        raise ValueError("Compact snapshot already has a different selection record")
    if not selection.exists():
        atomic_json(record, selection)


def label_paths(root, seed, objective, reader, phase, shards, integration_steps):
    directory = root / variant(objective, integration_steps) / f"seed{seed}" / f"router_{reader}"
    return [str(directory / f"labels_{phase}" / f"shard_{i:05d}.h5") for i in range(shards)]


def make_configs(cfg, args, generated, compact, teacher, hierarchy_root):
    base = deepcopy(cfg)
    base["stage"] = "router"
    base["paths"]["run_root"] = str(args.run_root)
    base["checkpoints"].update(teacher=str(teacher), compact=str(compact))
    base["training"]["hierarchy_cache_root"] = str(hierarchy_root)
    base["model"]["budgets"] = [64, 128, 256]
    base["model"]["budget"] = 128
    base["labels"].update(
        contexts=args.contexts, refresh_contexts=args.refresh_contexts, candidates=4,
        budgets=[64, 128, 256], split_fraction=0.8,
        objectives=list(OBJECTIVES), integration_steps=args.integration_steps,
        experiment_root=str(args.run_root),
    )
    router_checkpoints = {}
    for objective in OBJECTIVES:
        for reader in READERS:
            directory = args.run_root / variant(objective, args.integration_steps) / f"seed{cfg['seed']}" / f"router_{reader}"
            router_checkpoints[f"{objective}/{reader}"] = str(directory / "step_00005000.pt")
    base["labels"]["refresh_router_checkpoints"] = router_checkpoints
    configs = {}
    labels = generated / "labels.yaml"
    immutable_yaml(labels, base)
    configs["labels"] = {"path": str(labels), "sha256": sha256(labels)}
    for phase in ("initial", "final"):
        for objective in OBJECTIVES:
            for reader in READERS:
                result = deepcopy(base)
                result["reader"] = reader
                result["paths"]["run_root"] = str(args.run_root / variant(objective, args.integration_steps))
                result["labels"].update(objective=objective, use_refresh=phase == "final")
                result["labels"]["initial_shards"] = label_paths(
                    args.run_root, cfg["seed"], objective, reader, "initial", args.shards, args.integration_steps,
                )
                result["labels"]["refresh_shards"] = label_paths(
                    args.run_root, cfg["seed"], objective, reader, "refresh", args.shards, args.integration_steps,
                )
                result["training"].update(
                    steps=5000 if phase == "initial" else 10000,
                    effective_batch=256, microbatch=256, lr=0.0003,
                    checkpoint_steps=[5000, 10000], checkpoint_every=500,
                    workers=0, log_every=100, router_device="cuda",
                )
                validate_config(result)
                key = f"{phase}_{objective}_{reader}"
                path = generated / f"{key}.yaml"
                immutable_yaml(path, result)
                configs[key] = {"path": str(path), "sha256": sha256(path)}
    return configs


def jobs_for(plan):
    jobs = []
    for phase in ("initial", "refresh"):
        key = f"labels_{phase}"
        jobs.append({
            "key": key, "kind": "labels", "phase": phase, "config": "labels",
            "parents": [] if phase == "initial" else [f"initial_{o}_{r}" for o in OBJECTIVES for r in READERS],
        })
        fitting = "initial" if phase == "initial" else "final"
        for objective in OBJECTIVES:
            for reader in READERS:
                key_router = f"{fitting}_{objective}_{reader}"
                jobs.append({"key": key_router, "kind": "router", "phase": fitting,
                             "objective": objective, "reader": reader, "config": key_router, "parents": [key]})
    for job in jobs:
        job["script"] = str(Path(plan["generated"]) / f"{job['key']}.sbatch")
    return jobs


def labels_complete(plan, phase):
    contexts = plan["contexts_per_reader_objective" if phase == "initial" else "refresh_contexts_per_reader_objective"]
    shared = None
    for objective in OBJECTIVES:
        for reader in READERS:
            paths = label_paths(Path(plan["run_root"]), plan["seed"], objective, reader, phase,
                                plan["shards"], plan["integration_steps"])
            for shard, path in enumerate(paths):
                if not Path(path).is_file():
                    return False
                with h5py.File(path, "r") as handle:
                    if not handle.attrs.get("complete"):
                        return False
                    identity = json.loads(handle.attrs["identity"])
                    expected = {
                        "schema_version": 2, "teacher": plan["teacher_sha256"], "compact": plan["compact_sha256"],
                        "source_sha256": plan["source_sha256"],
                        "cache": plan["cache_manifest_digest"], "reader": reader, "objective": objective,
                        "integration_steps": plan["integration_steps"], "refresh": phase == "refresh",
                        "contexts": contexts, "shard_index": shard, "shard_count": plan["shards"],
                    }
                    if any(identity.get(key) != value for key, value in expected.items()):
                        raise ValueError(f"Completed label shard has incompatible identity: {path}")
                    matched = {key: identity[key] for key in ("partition", "sample_plan", "data", "label_sampling")}
                    if shared is not None and shared != matched:
                        raise ValueError("Label shards changed episode partitions or the matched context plan")
                    shared = matched
                    if phase == "refresh":
                        checkpoint = Path(path).parent.parent / "step_00005000.pt"
                        if identity.get("router") != sha256(checkpoint):
                            raise ValueError("Refresh labels changed their frozen initial scorer")
                    elif identity.get("router") is not None:
                        raise ValueError("Initial labels unexpectedly depend on a trained scorer")
                    expected_ids = set(range(shard, contexts, plan["shards"]))
                    attempted = handle["attempted_context_ids"][:].tolist()
                    if len(attempted) != len(expected_ids) or set(attempted) != expected_ids:
                        raise ValueError(f"Completed labels do not cover their planned contexts: {path}")
                    rows, attempted_count = json.loads(handle.attrs["commit"])
                    if (attempted_count != len(expected_ids) or rows > 4 * attempted_count
                            or any(len(handle[key]) != rows for key in
                                   ("query", "candidate", "gain", "context_id", "partition", "budget", "episode"))
                            or not set(handle["context_id"][:].tolist()) <= expected_ids):
                        raise ValueError(f"Completed labels have inconsistent committed rows: {path}")
    return True


def job_complete(plan, job):
    if job["kind"] == "labels":
        return labels_complete(plan, job["phase"])
    cfg = yaml.safe_load(Path(plan["configs"][job["config"]]["path"]).read_text())
    directory = run_dir(cfg, "router")
    checkpoint = directory / ("step_00005000.pt" if job["phase"] == "initial" else "step_00010000.pt")
    if not checkpoint.exists():
        return False
    state = load_checkpoint(checkpoint)
    if (state.get("stage") != "router" or state.get("config") != cfg
            or state.get("step") != cfg["training"]["steps"]):
        raise ValueError(f"Router completion checkpoint differs from the planned run: {checkpoint}")
    return True


def ready_plan(path):
    """Allocation startup guard: never silently inherit new data or model bytes."""
    plan = read_json(path)
    signed = {key: value for key, value in plan.items()
              if key not in ("walltime", "cpus", "mem", "partition", "signature", "jobs", "startup_plan")}
    if digest(signed) != plan["signature"] or jobs_for(plan) != plan["jobs"]:
        raise ValueError("Immutable router startup plan changed")
    if source_hash(Path(plan["repo"])) != plan["source_sha256"]:
        raise ValueError("Source changed since router experiment rendering")
    for path, expected in plan["launcher_hashes"].items():
        require_hash(path, expected, "Launcher source")
    for key, label in (("cache_plan", "Teacher cache plan"), ("teacher", "Selected teacher"),
                       ("compact", "Frozen compact"), ("compact_selection", "Compact selection")):
        require_hash(plan[key], plan[key + "_sha256"], label)
    cfg, cache_plan, manifest = inspect_cache(plan["cache_plan"])
    if cache_plan["checkpoint_sha256"] != plan["teacher_sha256"] or digest(manifest) != plan["cache_manifest_digest"]:
        raise ValueError("Frozen teacher/cache identity changed")
    hierarchy = inspect_hierarchy(plan["cache_path"], plan["hierarchy_cache_root"])
    if digest(hierarchy) != plan["hierarchy_manifest_digest"]:
        raise ValueError("Frozen hierarchy changed")
    require_hash(Path(plan["hierarchy_cache_root"]) / "manifest.json",
                 plan["hierarchy_manifest_sha256"], "Hierarchy manifest")
    for config in plan["configs"].values():
        require_hash(config["path"], config["sha256"], "Router experiment configuration")
        requested = yaml.safe_load(Path(config["path"]).read_text())
        validate_policy_config(cfg, requested)
        if (requested["data"] != cfg["data"]
                or any(requested["paths"][key] != cfg["paths"][key] for key in ("data_root", "cache_root"))
                or dependency(requested, "teacher") != Path(plan["teacher"])
                or dependency(requested, "compact") != Path(plan["compact"])):
            raise ValueError("Router configuration changed frozen inputs")
    return plan


def ready_job(path, key):
    plan = ready_plan(path)
    job = next(item for item in plan["jobs"] if item["key"] == key)
    for parent in job["parents"]:
        dependency_job = next(item for item in plan["jobs"] if item["key"] == parent)
        if not job_complete(plan, dependency_job):
            raise ValueError(f"Required predecessor is incomplete: {parent}")
    return plan


def render_script(plan, job, site):
    labels = job["kind"] == "labels"
    cpus, memory, duration = (plan["cpus"], plan["mem"], plan["walltime"]) if labels else (4, "16G", "01:00:00")
    directives = {
        "job-name": "stmem_" + plan["name"] + "_" + job["key"], "partition": plan["partition"],
        "gres": "gpu:1", "nodes": "1", "ntasks": "1", "cpus-per-task": str(cpus),
        "mem": memory, "time": duration, "signal": "B:USR1@180", "chdir": plan["repo"],
        "output": str(Path(plan["logs"]) / "%x_%A_%a.out"),
        "error": str(Path(plan["logs"]) / "%x_%A_%a.err"),
    }
    if labels:
        directives["array"] = f"0-{plan['shards'] - 1}%{min(4, plan['shards'])}"
    for setting in ("ACCOUNT", "QOS", "CONSTRAINT"):
        if site.get("STMEM_" + setting):
            directives[setting.lower()] = site["STMEM_" + setting]
    if any("\n" in value or "\r" in value for value in directives.values()):
        raise ValueError("Newline in Slurm directive")
    lines = ["#!/usr/bin/env bash", *(f"#SBATCH --{key}={shlex.quote(value)}" for key, value in directives.items()),
             "set -euo pipefail", "cd " + shlex.quote(plan["repo"])]
    if site.get("STMEM_ACTIVATE"):
        lines.append("source " + shlex.quote(site["STMEM_ACTIVATE"]))
    cfg = yaml.safe_load(Path(plan["configs"][job["config"]]["path"]).read_text())
    exports = {
        "STMEM_REPO_ROOT": plan["repo"], "STMEM_PYTHON": plan["python"], "STMEM_SITE_FILE": "/dev/null",
        "PYTHONPATH": str(Path(plan["repo"]) / "src"), "PYTHONUNBUFFERED": "1", "STMEM_ALLOCATED_GPUS": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        **{f"STMEM_{key.upper()}": cfg["paths"][key] for key in ("data_root", "cache_root", "run_root")},
        "WANDB_PROJECT": site.get("WANDB_PROJECT") or cfg.get("tracking", {}).get("project") or "chronotree",
        "WANDB_MODE": site.get("WANDB_MODE") or "online", "WANDB_ENTITY": site.get("WANDB_ENTITY") or "",
        "WANDB_DIR": site.get("WANDB_DIR") or "",
    }
    lines.append("unset RANK LOCAL_RANK WORLD_SIZE STMEM_FEATURE_CACHE_ROOT STMEM_SCRATCH_ROOT WANDB_RUN_ID WANDB_RESUME")
    lines.extend(f"export {key}={shlex.quote(str(value))}" for key, value in sorted(exports.items()))
    guard = f"from hpc.train_routers import ready_job; ready_job({plan['startup_plan']!r}, {job['key']!r})"
    lines.append(shlex.join([plan["python"], "-c", guard]))
    lines.append("export STMEM_STOP_FILE=" + shlex.quote(str(Path(plan["generated"]) / "stops"))
                 + '/"${SLURM_JOB_ID:-local_$$}_${SLURM_ARRAY_TASK_ID:-0}.stop"')
    command = ["srun", "--ntasks=1", f"--cpus-per-task={cpus}", "--cpu-bind=cores", "--gpu-bind=none",
               plan["python"], "-u", "-m"]
    if labels:
        command += ["stmem.training.label_experiment", "--config", plan["configs"]["labels"]["path"],
                    "--phase", job["phase"], "--shards", str(plan["shards"]), "--device", "cuda",
                    "--workers", str(plan["workers"]), "--batch-size", str(plan["batch_size"]),
                    "--teacher-token-budget", str(plan["teacher_token_budget"]),
                    "--compact-batch-size", str(plan["compact_batch_size"]),
                    "--prefetch-mb", str(plan["prefetch_mb"])]
        rendered = shlex.join(command) + ' --shard-index "${SLURM_ARRAY_TASK_ID:?}"'
    else:
        command += ["stmem.train", "--stage", "router", "--config", plan["configs"][job["config"]]["path"]]
        last = run_dir(cfg, "router") / "last.pt"
        lines.append("resume_args=()")
        lines.append("if [[ -f " + shlex.quote(str(last)) + " ]]; then resume_args=(--resume " + shlex.quote(str(last)) + "); fi")
        if job["phase"] == "final":
            lines.append('if [[ ${#resume_args[@]} -eq 0 ]]; then echo "Missing initial router checkpoint" >&2; exit 1; fi')
        rendered = shlex.join(command) + ' "${resume_args[@]}"'
    lines.append("exec " + shlex.quote(str(Path(plan["repo"]) / "hpc/run_job.sh"))
                 + ' "$STMEM_STOP_FILE" ' + rendered + ' --stop-file "$STMEM_STOP_FILE"')
    return "\n".join(lines) + "\n"


def submit_plan(plan, resume=False):
    ledger_path = Path(plan["generated"]) / "submitted.json"
    ledger = read_json(ledger_path) if ledger_path.exists() else {"signature": plan["signature"], "jobs": {}}
    if ledger["signature"] != plan["signature"]:
        raise ValueError("Submission ledger belongs to another experiment")
    if ledger["jobs"]:
        active = active_user_job_ids()
        running = [entry["job_id"] for entry in ledger["jobs"].values() if entry.get("job_id") in active]
        if running:
            raise ValueError("Router experiment jobs are still active; refusing duplicates: " + ", ".join(running))
        if any(entry.get("state") == "submitting" for entry in ledger["jobs"].values()):
            raise ValueError("Submission outcome is uncertain; reconcile the submitting ledger entry with Slurm before retrying")
        if not resume:
            raise ValueError("Experiment already submitted; use --resume after all jobs are inactive")
    ready_plan(plan["startup_plan"])
    submitted = {}
    for job in plan["jobs"]:
        if job_complete(plan, job):
            print(f"Already complete: {job['key']}")
            continue
        command = ["sbatch", "--parsable"]
        parents = [submitted[parent] for parent in job["parents"] if parent in submitted]
        if parents:
            command.append("--dependency=afterok:" + ":".join(parents))
        command.append(job["script"])
        previous = ledger["jobs"].get(job["key"])
        # Persist intent first: interruption after sbatch cannot silently duplicate work.
        ledger["jobs"][job["key"]] = {"state": "submitting", "command": command, "previous": previous}
        atomic_json(ledger, ledger_path)
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdecimal():
            raise ValueError(f"Unexpected sbatch receipt: {result.stdout!r}; reconcile ledger before retrying")
        ledger["jobs"][job["key"]] = {"state": "submitted", "job_id": job_id, "command": command, "previous": previous}
        atomic_json(ledger, ledger_path)
        submitted[job["key"]] = job_id
        print(f"Submitted {job['key']}: {job_id}")


def prepare(args, site, repo, generated):
    plan_path = generated / "plan.json"
    previous = read_json(plan_path) if plan_path.exists() else None
    if args.resume and previous is None:
        raise ValueError("No existing router experiment to resume")
    cache_plan_path = repo / "hpc/generated" / args.cache_name / "plan.json"
    cache_cfg, cache_plan, manifest = inspect_cache(cache_plan_path)
    cache_root = Path(cache_cfg["paths"]["cache_root"]) / digest(manifest["identity"])[:24]
    compact_source = absolute_path(args.compact, repo) if args.compact else (
        Path(previous["compact_source"]) if previous else repo / "runs" / f"{args.cache_name}_compact_hierarchy"
        / f"seed{cache_cfg['seed']}" / "compact/last.pt"
    )
    compact = generated / "frozen/compact.pt"
    if previous is None:
        if args.run_root.exists() and any(path.name != ".router-launch.lock" for path in args.run_root.iterdir()):
            raise ValueError("A fresh router experiment requires an empty --run-root")
        inspect_compact(compact_source, cache_cfg, cache_plan, manifest)
    snapshot_compact(compact_source, compact, previous)
    state = inspect_compact(compact, cache_cfg, cache_plan, manifest)
    from stmem.data.hierarchy_cache import default_hierarchy_root
    hierarchy_root = absolute_path(args.hierarchy_root, repo) if args.hierarchy_root else Path(
        state["config"]["training"].get("hierarchy_cache_root") or default_hierarchy_root(cache_root)
    )
    hierarchy = inspect_hierarchy(cache_root, hierarchy_root)
    if state["provenance"]["dependencies"].get("hierarchy", digest(hierarchy)) != digest(hierarchy):
        raise ValueError("Compact checkpoint used a different hierarchy cache")
    configs = make_configs(state["config"], args, generated, compact, cache_plan["checkpoint"], hierarchy_root)
    selection = compact.with_suffix(".selection.json")
    plan = {
        "kind": "matched-router-objective-experiment", "name": args.name, "repo": str(repo),
        "python": site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python"),
        "generated": str(generated), "plan_path": str(plan_path), "run_root": str(args.run_root),
        "cache_plan": str(cache_plan_path), "cache_plan_sha256": sha256(cache_plan_path),
        "cache_path": str(cache_root), "cache_manifest_digest": digest(manifest),
        "teacher": cache_plan["checkpoint"], "teacher_sha256": cache_plan["checkpoint_sha256"],
        "compact": str(compact), "compact_sha256": sha256(compact), "compact_source": str(compact_source),
        "compact_step": state["step"], "compact_selection": str(selection), "compact_selection_sha256": sha256(selection),
        "hierarchy_cache_root": str(hierarchy_root), "hierarchy_manifest_digest": digest(hierarchy),
        "hierarchy_manifest_sha256": sha256(hierarchy_root / "manifest.json"),
        "source_sha256": source_hash(repo), "launcher_hashes": {
            str(path): sha256(path) for path in (Path(__file__), Path(__file__).with_name("run_job.sh"),
                                               Path(__file__).with_name("train_compact.py"),
                                               Path(__file__).with_name("plan.py"),
                                               Path(__file__).with_name("profile_memory.py"))
        },
        "configs": configs, "tasks": state["config"]["data"]["tasks"], "seed": state["config"]["seed"],
        "objectives": list(OBJECTIVES), "integration_steps": args.integration_steps,
        "evaluation_flow_steps": state["config"]["model"]["flow_steps"],
        "contexts_per_reader_objective": args.contexts, "refresh_contexts_per_reader_objective": args.refresh_contexts,
        "candidates_per_context": 4, "initial_router_steps": 5000, "total_router_steps": 10000,
        "partition": args.partition, "gres": "gpu:1", "shards": args.shards, "workers": args.workers,
        "batch_size": args.batch_size, "teacher_token_budget": args.teacher_token_budget,
        "compact_batch_size": args.compact_batch_size, "prefetch_mb": args.prefetch_mb,
        "cpus": args.cpus, "mem": args.mem, "walltime": args.walltime,
        "logs": str(generated / "logs"), "job_count": 10,
        "allocation_count": 2 * args.shards + 8, "maximum_concurrent_allocations": 4,
        "estimated_gpu_hours": "unknown; measure the bounded pilot before estimating full-label cost",
        "scientific_status": "planned; no benchmark or training outcome is claimed",
    }
    signature_fields = {key: value for key, value in plan.items() if key not in ("walltime", "cpus", "mem", "partition")}
    plan["signature"] = digest(signature_fields)
    if previous and previous["signature"] != plan["signature"]:
        raise ValueError("Router experiment inputs/configuration/source changed; choose a new --name")
    ownership = args.run_root / ".router-plan.json"
    owner = {"plan": str(plan_path), "signature": plan["signature"]}
    if ownership.exists() and read_json(ownership) != owner:
        raise ValueError("Another router experiment owns this output root")
    plan["jobs"] = jobs_for(plan)
    attempt = generated / "attempts" / (digest(plan)[:24] + ".json")
    plan["startup_plan"] = str(attempt)
    if attempt.exists() and read_json(attempt) != plan:
        raise ValueError("Immutable startup plan changed")
    if not attempt.exists():
        atomic_json(plan, attempt)
    Path(plan["logs"]).mkdir(exist_ok=True)
    for job in plan["jobs"]:
        Path(job["script"]).write_text(render_script(plan, job, site))
    atomic_json(owner, ownership)
    atomic_json(plan, plan_path)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.submit:
        submit_plan(plan, args.resume)
    else:
        print("Dry-run only: immutable inputs, configurations and 10 job scripts rendered; no Slurm contact.")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    action.add_argument("--status", action="store_true", help="Report saved artifact completion without contacting Slurm")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site_b200_4gpu.env")
    parser.add_argument("--cache-name", type=profile_name, default="memory_fv_seed0")
    parser.add_argument("--name", type=profile_name)
    parser.add_argument("--compact", type=Path, help="Checkpoint to freeze; default runs/CACHE_NAME_compact_hierarchy/seed0/compact/last.pt")
    parser.add_argument("--hierarchy-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume incomplete stages once every prior allocation is inactive")
    parser.add_argument("--partition", default="gpuq")
    parser.add_argument("--shards", type=positive_integer, default=4)
    parser.add_argument("--contexts", type=positive_integer, default=16000)
    parser.add_argument("--refresh-contexts", type=positive_integer, default=8000)
    parser.add_argument("--integration-steps", type=int, choices=(4, 8), default=8)
    parser.add_argument("--workers", type=nonnegative_integer, default=4)
    parser.add_argument("--batch-size", type=positive_integer, default=4)
    parser.add_argument("--teacher-token-budget", type=positive_integer, default=32768)
    parser.add_argument("--compact-batch-size", type=positive_integer, default=64)
    parser.add_argument("--prefetch-mb", type=positive_integer, default=512)
    parser.add_argument("--cpus", type=positive_integer, default=16)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--walltime", type=walltime, default="12:00:00")
    args = parser.parse_args(argv)
    args.name = args.name or args.cache_name + "_routers"
    if args.name == args.cache_name:
        parser.error("Router experiment --name must differ from --cache-name")
    if args.cpus <= args.workers:
        parser.error("Reserve at least one CPU beyond the preparation workers")
    if args.contexts < args.shards or args.refresh_contexts < args.shards:
        parser.error("Both context totals must be at least the number of shards")
    if not re.fullmatch(r"[1-9][0-9]*(?:[KMGT]i?B?|[kmgt])?", args.mem):
        parser.error("Invalid --mem")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.partition):
        parser.error("Invalid --partition")
    try:
        site = read_site(args.site)
        repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
        generated = repo / "hpc/generated" / args.name
        if args.status:
            plan = read_json(generated / "plan.json")
            result = {job["key"]: "complete" if job_complete(plan, job) else "incomplete" for job in plan["jobs"]}
            print(json.dumps({"name": plan["name"], "stages": result, "logs": plan["logs"]}, indent=2))
            return result
        args.run_root = absolute_path(args.run_root or repo / "runs" / args.name, repo)
        args.run_root.mkdir(parents=True, exist_ok=True)
        with (args.run_root / ".router-launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            generated.mkdir(parents=True, exist_ok=True)
            return prepare(args, site, repo, generated)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
