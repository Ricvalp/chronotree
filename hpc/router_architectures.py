#!/usr/bin/env python3
"""Fit controlled router architecture ablations on unchanged initial labels.

Only --submit contacts Slurm. Input reconstruction reads the existing encoded
cache; it never regenerates utility targets or trains the teacher/compact policy.
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

from hpc.overfit_router import initial_checkpoint, read_json, require_hash  # noqa: E402
from hpc.plan import active_user_job_ids, nonnegative_integer, positive_integer, read_site, sha256, source_hash  # noqa: E402
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from hpc.train_routers import immutable_yaml  # noqa: E402
from stmem.config import dependency, digest, run_dir, validate_config  # noqa: E402
from stmem.training.checkpoints import atomic_copy, atomic_json  # noqa: E402
from stmem.training.runtime import load_checkpoint  # noqa: E402

VARIANTS = ("dot", "mlp", "children", "current_attention", "children_current_attention")
RICH = set(VARIANTS) - {"dot", "mlp"}


def signature(plan):
    return digest({key: value for key, value in plan.items() if key != "signature"})


def ready_plan(path, job=None):
    """Check frozen inputs before startup; richer arrays are checked by their loader."""
    plan = read_json(path)
    if signature(plan) != plan["signature"]:
        raise ValueError("Router architecture plan changed")
    if source_hash(Path(plan["repo"])) != plan["source_sha256"]:
        raise ValueError("Source changed since rendering; use a new experiment name")
    for source, expected in plan["launcher_hashes"].items():
        require_hash(source, expected, "Launcher source")
    for entry in plan["frozen_inputs"]:
        require_hash(entry["path"], entry["sha256"], "Frozen architecture experiment input")
    for entry in plan["configs"].values():
        require_hash(entry["path"], entry["sha256"], "Architecture configuration")
    if job is not None:
        entry = next((entry for entry in plan["jobs"] if entry["name"] == job), None)
        if entry is None:
            raise ValueError(f"Unknown architecture job: {job}")
        if entry["kind"] == "train":
            last = Path(entry["run_directory"]) / "last.pt"
            if last.exists():
                state = load_checkpoint(last)
                cfg = yaml.safe_load(Path(entry["config"]).read_text())
                expected = plan["readers"][entry["reader"]]
                if (state.get("stage") != "router" or state.get("config") != cfg
                        or state.get("reader") != entry["reader"]
                        or state.get("label_hashes") != expected["label_hashes"]
                        or state.get("label_identity") != expected["label_identity"]
                        or not 0 <= state.get("step", -1) <= plan["steps"]):
                    raise ValueError("Existing architecture checkpoint belongs to different work")
    return plan


def input_shards_complete(plan):
    """Cheap filesystem status; content hashes are validated before reuse/training."""
    for reader in plan["input_readers"]:
        info = plan["readers"][reader]
        for path, expected in zip(info["input_shards"], info["label_hashes"]):
            p = Path(path)
            record = p.with_suffix(".json")
            if not p.is_file() or not record.is_file():
                return False
            meta = read_json(record)
            if meta.get("status") != "complete" or meta.get("source_label_sha256") != expected:
                return False
    return True


def training_status(job, steps):
    directory = Path(job["run_directory"])
    last = directory / "last.pt"
    record_path = directory / "complete.json"
    record = read_json(record_path) if record_path.exists() else {}
    state = load_checkpoint(last) if last.exists() else None
    if state is not None:
        cfg = yaml.safe_load(Path(job["config"]).read_text())
        if (state.get("stage") != "router" or state.get("reader") != job["reader"]
                or state.get("config") != cfg or not 0 <= state.get("step", -1) <= steps):
            raise ValueError("Existing architecture checkpoint belongs to different work")
    step = state["step"] if state is not None else 0
    done = (last.exists() and step == steps and record.get("status") == "complete"
            and record.get("completed_steps") == steps and record.get("checkpoint_sha256") == sha256(last))
    return {"name": job["name"], "status": "complete" if done else "incomplete",
            "completed_steps": step, "planned_steps": steps, "run_directory": str(directory)}


def status(plan):
    rows = [training_status(job, plan["steps"]) for job in plan["jobs"] if job["kind"] == "train"]
    inputs_done = input_shards_complete(plan)
    return {"name": plan["name"], "status": "complete" if inputs_done and all(
        row["status"] == "complete" for row in rows) else "incomplete",
        "inputs_complete": inputs_done, "jobs": rows, "logs": plan["logs"],
        "note": "Recorded filesystem progress; no Slurm query or submission."}


def render_script(plan, job, site):
    prep = job["kind"] == "inputs"
    cpus = plan["input_cpus"] if prep else plan["cpus"]
    directives = {
        "job-name": "stmem_router_" + job["name"], "partition": plan["partition"], "gres": "gpu:1",
        "nodes": "1", "ntasks": "1", "cpus-per-task": str(cpus),
        "mem": plan["input_mem"] if prep else plan["mem"],
        "time": plan["input_walltime"] if prep else plan["walltime"], "signal": "B:USR1@180",
        "chdir": plan["repo"], "output": str(Path(plan["logs"]) / ("%x_%A_%a.out" if prep else "%x_%j.out")),
        "error": str(Path(plan["logs"]) / ("%x_%A_%a.err" if prep else "%x_%j.err")),
    }
    if prep:
        directives["array"] = f"0-{plan['input_shard_count'] - 1}%{plan['input_shard_count']}"
    else:
        if job["deps"]:
            # Failed preparation must not leave dependent fits pending forever.
            directives["kill-on-invalid-dep"] = "yes"
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
        "WANDB_PROJECT": plan["wandb_project"], "WANDB_MODE": site.get("WANDB_MODE") or "online",
        "WANDB_ENTITY": site.get("WANDB_ENTITY") or "", "WANDB_DIR": site.get("WANDB_DIR") or "",
    }
    # Configurations own data/cache/run paths. Do not inherit the previous teacher
    # job's paths, checkpoint or W&B identity from the submitting shell.
    lines.append("unset RANK LOCAL_RANK WORLD_SIZE STMEM_RUN_ROOT STMEM_DATA_ROOT STMEM_CACHE_ROOT "
                 "STMEM_FEATURE_CACHE_ROOT STMEM_SCRATCH_ROOT STMEM_STOP_FILE WANDB_RUN_ID WANDB_RESUME")
    lines.extend(f"export {key}={shlex.quote(str(value))}" for key, value in sorted(exports.items()))
    guard = f"from hpc.router_architectures import ready_plan; ready_plan({plan['plan_path']!r}, {job['name']!r})"
    lines.append(shlex.join([plan["python"], "-c", guard]))
    prefix = ["srun", "--ntasks=1", f"--cpus-per-task={cpus}", "--cpu-bind=cores", "--gpu-bind=none",
              plan["python"], "-u"]
    if prep:
        lines.append("export STMEM_STOP_FILE=" + shlex.quote(str(Path(plan["generated"]) / "stops" / "inputs"))
                     + '/"${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local_$$}}_${SLURM_ARRAY_TASK_ID:-0}.stop"')
        lines.append("exec " + shlex.quote(str(Path(plan["repo"]) / "hpc/run_job.sh"))
                     + ' "$STMEM_STOP_FILE" bash ' + shlex.quote(job["runner"]))
    else:
        last = Path(job["run_directory"]) / "last.pt"
        lines.extend(["resume_args=()", "if [[ -f " + shlex.quote(str(last))
                      + " ]]; then resume_args=(--resume " + shlex.quote(str(last)) + "); fi"])
        lines.append("export STMEM_STOP_FILE=" + shlex.quote(str(Path(plan["generated"]) / "stops" / job["name"]))
                     + '/"${SLURM_JOB_ID:-local_$$}.stop"')
        lines.append("exec " + shlex.quote(str(Path(plan["repo"]) / "hpc/run_job.sh"))
                     + ' "$STMEM_STOP_FILE" ' + shlex.join([*prefix, "-m", "stmem.train", "--stage", "router",
                                                           "--config", job["config"]])
                     + ' "${resume_args[@]}" --stop-file "$STMEM_STOP_FILE"')
    return "\n".join(lines) + "\n"


def render_input_runner(plan):
    """The signal wrapper owns the allocation; this child closes each HDF5 safely."""
    prefix = ["srun", "--ntasks=1", f"--cpus-per-task={plan['input_cpus']}", "--cpu-bind=cores",
              "--gpu-bind=none", plan["python"], "-u", "-m", "stmem.training.router_inputs"]
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for reader in plan["input_readers"]:
        lines.append(shlex.join([*prefix, "--config", plan["configs"]["inputs_" + reader]["path"], "--shard-index"])
                     + ' "${SLURM_ARRAY_TASK_ID:?}" --stop-file "$STMEM_STOP_FILE"')
    return "\n".join(lines) + "\n"


def submit(plan, resume):
    ready_plan(plan["plan_path"])
    receipt_path = Path(plan["generated"]) / "submitted.json"
    receipt = read_json(receipt_path) if receipt_path.exists() else {"signature": plan["signature"], "jobs": {}}
    if receipt.get("signature") != plan["signature"]:
        raise ValueError("Submission receipt belongs to another architecture experiment")
    if any(row.get("state") == "submitting" for row in receipt["jobs"].values()):
        raise ValueError("Submission outcome is uncertain; reconcile the receipt with Slurm before retrying")
    if receipt["jobs"] and not resume:
        raise ValueError("Experiment already submitted; use --resume to submit remaining work")
    active = active_user_job_ids() if receipt["jobs"] else set()
    prep_id = None
    inputs_done = input_shards_complete(plan)
    old_prep = receipt["jobs"].get("inputs")
    if not inputs_done and old_prep and old_prep["job_id"] not in active:
        pending_dependents = [job["name"] for job in plan["jobs"] if job["deps"]
                              and receipt["jobs"].get(job["name"], {}).get("job_id") in active]
        if pending_dependents:
            raise ValueError("Previous input array ended, but dependent allocations are still active; "
                             "wait for their cancellation before resuming: " + ", ".join(pending_dependents))
    for job in plan["jobs"]:
        name = job["name"]
        if job["kind"] == "inputs" and inputs_done:
            continue
        if job["kind"] == "train" and training_status(job, plan["steps"])["status"] == "complete":
            continue
        old = receipt["jobs"].get(name)
        if old and old["job_id"] in active:
            if job["kind"] == "inputs":
                prep_id = old["job_id"]
            print(f"Still active: {name} ({old['job_id']}); not duplicated.")
            continue
        command = ["sbatch", "--parsable"]
        if job["deps"] and not inputs_done:
            if prep_id is None:
                raise ValueError("Missing input-preparation dependency")
            command.append("--dependency=afterok:" + prep_id)
        command.append(job["script"])
        receipt["jobs"][name] = {"state": "submitting", "command": command, "previous": old}
        atomic_json(receipt, receipt_path)
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdecimal():
            raise ValueError("Unexpected sbatch receipt; reconcile submission before retrying")
        receipt["jobs"][name] = {"state": "submitted", "job_id": job_id, "command": command, "previous": old}
        atomic_json(receipt, receipt_path)
        if job["kind"] == "inputs":
            prep_id = job_id
        print(f"Submitted {name}: {job_id}")


def prepare(args, site, repo, generated):
    path = generated / "plan.json"
    previous = read_json(path) if path.exists() else None
    if args.resume and previous is None:
        raise ValueError("No existing architecture experiment to resume")
    root = repo / "runs" / args.name
    if previous is None and root.exists() and any(root.iterdir()):
        raise ValueError("A fresh architecture experiment requires an empty output directory")
    combinations = [(reader, variant) for reader in args.readers for variant in args.variants
                    if reader == "tree" or "children" not in variant]
    if not combinations:
        raise ValueError("No supported reader/architecture combination requested")
    input_readers = [reader for reader in args.readers if any(r == reader and v in RICH for r, v in combinations)]
    readers, configs, frozen, counts = {}, {}, [], set()
    for reader in args.readers:
        if not any(r == reader for r, _ in combinations):
            continue
        cfg, _state, checkpoint, labels, identity, original_plan = initial_checkpoint(
            repo, args.experiment, reader, args.objective)
        original_root = Path(read_json(original_plan)["run_root"]).resolve()
        if root.resolve().is_relative_to(original_root):
            raise ValueError("Architecture output must be separate from the original experiment")
        # Pin dependency paths before changing run_root; some old configurations
        # still use format strings rooted at the original experiment.
        cfg = deepcopy(cfg)
        for name in ("teacher", "compact"):
            cfg.setdefault("checkpoints", {})[name] = str(dependency(cfg, name).resolve())
        snapshot = generated / "frozen" / f"initial_{reader}.pt"
        atomic_copy(checkpoint, snapshot, immutable=True)
        if sha256(snapshot) != sha256(checkpoint):
            raise ValueError("Original router changed during snapshotting")
        frozen.append({"path": str(snapshot), "sha256": sha256(snapshot)})
        frozen.extend(labels)
        counts.add(len(labels))
        input_paths = [str(root / "inputs" / reader / f"shard_{i:05d}.h5") for i in range(len(labels))]
        cfg["labels"].update(use_refresh=False)
        cfg["labels"].pop("router_inputs", None)
        cfg["tracking"].update(project=args.wandb_project, enabled=True, system_stats=True)
        if site.get("WANDB_MODE") == "disabled":
            raise ValueError("Architecture training requires W&B logging")
        cfg["training"].update(steps=args.steps, schedule_steps=max(args.steps, cfg["training"].get("schedule_steps", 0)),
                               router_diagnostics=True, router_device="cuda", router_fit_contexts=args.fit_contexts,
                               workers=0, log_every=1000, checkpoint_every=10000,
                               checkpoint_steps=sorted({step for step in (10000, 25000, 50000, args.steps)
                                                        if step <= args.steps}))
        readers[reader] = {"label_hashes": [entry["sha256"] for entry in labels], "label_identity": identity,
                           "input_shards": input_paths, "labels": labels, "source_checkpoint": str(checkpoint)}
        if reader in input_readers:
            inputs_cfg = deepcopy(cfg)
            inputs_cfg["labels"]["router_inputs"] = input_paths
            inputs_cfg["model"]["router_architecture"] = "children_current_attention" if reader == "tree" else "current_attention"
            inputs_cfg["paths"]["run_root"] = str(root / "inputs" / reader)
            p = generated / "configs" / f"inputs_{reader}.yaml"
            immutable_yaml(p, inputs_cfg)
            configs["inputs_" + reader] = {"path": str(p), "sha256": sha256(p)}
        for r, variant in combinations:
            if r != reader:
                continue
            fit = deepcopy(cfg)
            fit["paths"]["run_root"] = str(root / variant / reader)
            fit["model"]["router_architecture"] = variant
            if variant in RICH:
                fit["labels"]["router_inputs"] = input_paths
            fit["tracking"]["name"] = f"{args.name}-{reader}-{variant}"
            validate_config(fit)
            p = generated / "configs" / f"{reader}_{variant}.yaml"
            immutable_yaml(p, fit)
            configs[f"{reader}_{variant}"] = {"path": str(p), "sha256": sha256(p),
                                                "run_directory": str(run_dir(fit, "router"))}
    if len(counts) != 1:
        raise ValueError("Reader label shard counts differ; prepare separate suites")
    shard_count = counts.pop()
    jobs = []
    if input_readers:
        jobs.append({"name": "inputs", "kind": "inputs", "deps": [],
                     "runner": str(generated / "jobs" / "inputs_commands.sh"),
                     "script": str(generated / "jobs" / "inputs.sbatch")})
    for reader, variant in combinations:
        key = f"{reader}_{variant}"
        jobs.append({"name": key, "kind": "train", "reader": reader, "architecture": variant,
                     "deps": ["inputs"] if variant in RICH else [], "config": configs[key]["path"],
                     "run_directory": configs[key]["run_directory"],
                     "script": str(generated / "jobs" / f"{key}.sbatch")})
    plan = {
        "kind": "router-architecture-diagnostic", "name": args.name, "experiment": args.experiment,
        "repo": str(repo), "python": site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python"),
        "generated": str(generated), "plan_path": str(path), "run_root": str(root), "logs": str(generated / "logs"),
        "source_sha256": source_hash(repo), "launcher_hashes": {
            str(p): sha256(p) for p in (Path(__file__), Path(__file__).with_name("overfit_router.py"),
                                      Path(__file__).with_name("train_routers.py"), Path(__file__).with_name("run_job.sh"),
                                      Path(__file__).with_name("plan.py"), Path(__file__).with_name("profile_memory.py"))},
        "frozen_inputs": frozen, "readers": readers, "configs": configs, "jobs": jobs,
        "input_readers": input_readers, "input_shard_count": shard_count,
        "objective": args.objective, "initial_labels_only": True, "fresh_initialization": True,
        "steps": args.steps, "fit_contexts": args.fit_contexts, "wandb_project": args.wandb_project,
        "site_runtime": {key: site.get(key) or "" for key in (
            "STMEM_ACCOUNT", "STMEM_QOS", "STMEM_CONSTRAINT", "STMEM_ACTIVATE",
            "WANDB_MODE", "WANDB_ENTITY", "WANDB_DIR")},
        "partition": args.partition, "gres": "gpu:1", "gpus_per_allocation": 1,
        "cpus": args.cpus, "mem": args.mem, "walltime": args.walltime,
        "input_cpus": args.input_cpus, "input_mem": args.input_mem, "input_walltime": args.input_walltime,
        "training_job_count": len(combinations), "job_count": len(combinations) + (shard_count if input_readers else 0),
        "maximum_concurrent_allocations": max(len(combinations), shard_count + sum(v not in RICH for _, v in combinations))
        if input_readers else len(combinations),
        "estimated_gpu_hours": "unknown; architecture and input-reconstruction timings require measurement",
        "diagnostic": "Fixed-label architecture ablation. Children variants are TREE-only, not matched TREE/FLAT claims.",
    }
    plan["signature"] = signature(plan)
    if previous is not None and previous != plan:
        raise ValueError("Named experiment settings/inputs/source changed; use a new --name")
    Path(plan["logs"]).mkdir(parents=True, exist_ok=True)
    (generated / "jobs").mkdir(parents=True, exist_ok=True)
    for job in jobs:
        if job["kind"] == "inputs":
            Path(job["runner"]).write_text(render_input_runner(plan))
        Path(job["script"]).write_text(render_script(plan, job, site))
    atomic_json(plan, path)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.submit:
        submit(plan, args.resume)
    else:
        print("Dry-run only: frozen labels/provenance, input-reconstruction array and fresh training jobs; no Slurm contact.")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    action.add_argument("--status", action="store_true", help="Read recorded filesystem progress without Slurm")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site_b200_4gpu.env")
    parser.add_argument("--experiment", type=profile_name, default="memory_fv_seed0_routers")
    parser.add_argument("--name", type=profile_name, default="memory_fv_seed0_router_arch_final8")
    parser.add_argument("--objective", choices=("velocity", "final_action"), default="final_action")
    parser.add_argument("--readers", nargs="+", choices=("tree", "flat"), default=["tree", "flat"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--steps", type=positive_integer, default=100000)
    parser.add_argument("--fit-contexts", type=nonnegative_integer, default=0,
                        help="Whole training contexts for a small memorization diagnostic; 0 uses all")
    parser.add_argument("--wandb-project", type=profile_name, default="chronotree-routers")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--partition", default="gpuq")
    parser.add_argument("--cpus", type=positive_integer, default=4)
    parser.add_argument("--mem", default="16G")
    parser.add_argument("--walltime", type=walltime, default="02:00:00")
    parser.add_argument("--input-cpus", type=positive_integer, default=8)
    parser.add_argument("--input-mem", default="32G")
    parser.add_argument("--input-walltime", type=walltime, default="02:00:00")
    args = parser.parse_args(argv)
    args.readers = list(dict.fromkeys(args.readers))
    args.variants = list(dict.fromkeys(args.variants))
    if args.name == args.experiment:
        parser.error("Architecture --name must differ from the source --experiment")
    if any(not re.fullmatch(r"[1-9][0-9]*(?:[KMGT]i?B?|[kmgt])?", value) for value in (args.mem, args.input_mem)):
        parser.error("Invalid memory request")
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
