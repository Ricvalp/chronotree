#!/usr/bin/env python3
"""Train one shared compact policy from completed leaf and hierarchy caches.

Only --submit contacts Slurm. The saved cache configuration owns the scientific
inputs and cache paths. New runs use an isolated output root and require the
derived hierarchy; --legacy-cache explicitly retains on-demand construction.
The site file supplies the interpreter and account settings.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import math
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
    active_user_job_ids, nonnegative_integer, positive_integer, read_site,
    save_json, sha256, source_hash,
)
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from stmem.config import dependency, digest, run_dir, validate_config  # noqa: E402
from stmem.data.cache import cache_identity  # noqa: E402
from stmem.train import continuation_record  # noqa: E402
from stmem.training.runtime import load_checkpoint, validate_policy_config  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text())


def require_hash(path, expected, label):
    if not expected or sha256(Path(path)) != expected:
        raise ValueError(f"{label} changed or has an invalid hash: {path}")


def inspect_cache(cache_plan_path):
    """Metadata/size validation; never rebuild or read every shard's data bytes."""
    plan = read_json(cache_plan_path)
    if plan.get("kind") != "selected-teacher-encoder-cache":
        raise ValueError("Expected a plan made by hpc/cache_teacher.py")
    for key, label in (("config", "Cache configuration"), ("checkpoint", "Selected teacher"),
                       ("selection", "Teacher selection")):
        require_hash(plan[key], plan[key + "_sha256"], label)
    # Do not use load_config: inherited STMEM_* variables may point at an old run.
    cfg = yaml.safe_load(Path(plan["config"]).read_text())
    validate_config(cfg)
    if cfg.get("debug") or cfg["data"].get("synthetic") or cfg.get("stage") != "teacher":
        raise ValueError("Expected a real-data teacher cache configuration")
    if cfg["model"].get("history_mode", "full") != "full":
        raise ValueError("Compact memory training requires a full-history teacher")
    if Path(cfg["paths"]["run_root"]) != Path(plan["run_root"]):
        raise ValueError("Cache plan and configuration run roots differ")
    if dependency(cfg, "teacher") != Path(plan["checkpoint"]):
        raise ValueError("Cache configuration points to a different teacher")
    selection = read_json(plan["selection"])
    if selection.get("sha256") != plan["checkpoint_sha256"]:
        raise ValueError("Teacher selection does not match the cached checkpoint")
    if selection.get("step") != plan.get("checkpoint_step"):
        raise ValueError("Teacher selection step does not match the cache plan")
    complete_path = run_dir(cfg, "cache") / "complete.json"
    if complete_path != Path(plan["completion_record"]):
        raise ValueError("Cache completion record path differs from the configuration")
    complete = read_json(complete_path)
    if complete.get("status") != "complete":
        raise ValueError("Encoder cache is not complete")
    root = Path(complete["cache_path"])
    require_hash(root / "manifest.json", complete["cache_manifest_sha256"], "Cache manifest")
    manifest = read_json(root / "manifest.json")
    identity = cache_identity(cfg, Path(plan["checkpoint"]), manifest["identity"]["data_identity"])
    if (manifest.get("status") != "complete" or manifest["identity"] != identity
            or complete.get("identity_sha256") != digest(identity)
            or complete.get("teacher_sha256") != plan["checkpoint_sha256"]):
        raise ValueError("Cache identity, encoder source or teacher changed; incompatible cache")
    if root != Path(cfg["paths"]["cache_root"]) / digest(identity)[:24]:
        raise ValueError("Cache location differs from its content-addressed configuration")
    tasks = cfg["data"]["tasks"]
    if (plan["tasks"] != tasks or manifest.get("tasks") != tasks
            or set(manifest["shards"]) != {task + ".h5" for task in tasks}):
        raise ValueError("Cache task coverage differs from the training configuration")
    for task in tasks:
        path = root / f"{task}.h5"
        shard = manifest["shards"][path.name]
        receipt = read_json(root / f"{task}.complete.json")
        if (not path.is_file() or path.stat().st_size != shard["bytes"]
                or receipt.get("task") != task
                or receipt.get("kind") != "encoder_cache_task" or receipt.get("status") != "complete"
                or receipt.get("identity_sha256") != digest(identity)
                or any(receipt.get(key) != shard[key] for key in ("sha256", "bytes", "episodes"))):
            raise ValueError(f"Missing or inconsistent cache shard/receipt: {task}")
    return cfg, plan, manifest


def inspect_hierarchy(cache_root, hierarchy_root):
    """Require the completed derived cache without rescanning its large payloads."""
    from stmem.data.hierarchy_cache import validate_hierarchy_cache

    try:
        return validate_hierarchy_cache(cache_root, hierarchy_root)
    except FileNotFoundError as error:
        raise ValueError(
            f"Missing or incomplete hierarchy cache: {hierarchy_root}. "
            "Run hpc/cache_hierarchy.py and wait for --status to report complete."
        ) from error


def compact_config(cfg, args):
    result = deepcopy(cfg)
    result["stage"] = "compact"
    result["profile"] = f"{args.partition}_{args.gpus}gpu"
    result["model"]["budget"] = 128
    result["model"]["budgets"] = [64, 128, 256]
    result["training"].update({
        "steps": 30000, "schedule_steps": 30000, "checkpoint_steps": [15000, 30000],
        "checkpoint_every": 1000, "effective_batch": args.effective_batch,
        "microbatch": args.microbatch, "workers": args.workers, "prefetch_factor": 1,
        "lr": 0.0001, "warmup_steps": 1000, "log_every": 1,
        "attention_every": 10, "progress_every_seconds": 30,
        "context_mix": {"tree": 0.45, "flat": 0.45, "uniform": 0.05, "current": 0.05},
        "intermediate_probability": 0.5,
    })
    if args.legacy_cache:
        result["training"].pop("hierarchy_cache_root", None)
    else:
        result["training"]["hierarchy_cache_root"] = str(args.hierarchy_root)
    result.setdefault("tracking", {}).update(enabled=True, system_stats=True, mode="online")
    # Evaluation will use the adopted execution override, separately from the
    # checkpoint's unchanged action contract. This is the fixed selection subset.
    result["evaluation"]["selection_episode_count"] = 4
    result.setdefault("checkpoints", {})["compact"] = str(run_dir(result, "compact") / "selected.pt")
    study = getattr(args, "utility_study", None)
    if study:
        result["training"]["utility_study"] = deepcopy(study)
        result["tracking"]["project"] = "chronotree-routers"
        result["tracking"]["group"] = study["experiment_id"]
        if study.get("initialization_checkpoint"):
            endpoint = study["continuation_updates"]
            result["training"].update(steps=endpoint, schedule_steps=endpoint, warmup_steps=0,
                                      lr=study["continuation_lr"], checkpoint_steps=[endpoint])
        elif "primary_updates" in study:
            from stmem.training.utility_study import primary_schedule

            schedule = primary_schedule(study["primary_updates"], study["checkpoint_every"],
                                        study["snapshot_every"])
            if any(study.get(key) != value for key, value in schedule.items()):
                raise ValueError("Student checkpoint declarations differ from its primary schedule")
            result["training"].update(steps=schedule["primary_updates"], schedule_steps=schedule["primary_updates"],
                                      checkpoint_steps=schedule["checkpoint_steps"],
                                      checkpoint_every=schedule["checkpoint_every"])
    validate_config(result)
    validate_policy_config(cfg, result)
    return result


def validate_study_inputs(study, panel, *, teacher_sha256, teacher_step,
                          encoded_cache_id, hierarchy_cache_id):
    """Bind optimization settings to the sealed panel and the actual cache."""
    from stmem.inspection.refinement_data import study_dependencies

    pins = study_dependencies(panel)
    actual = {"teacher_sha256": teacher_sha256, "teacher_step": teacher_step,
              "encoded_cache_id": encoded_cache_id, "hierarchy_cache_id": hierarchy_cache_id}
    for key, value in actual.items():
        # Legacy settings did not duplicate the teacher step; the sealed
        # manifest still fixes it. New study settings include it explicitly.
        configured = study.get(key, pins[key] if key == "teacher_step" else None)
        if configured != pins[key] or value != pins[key]:
            raise ValueError(f"Student study {key} differs from its manifest or actual cache")


def ready_plan(plan_path):
    """Run inside the allocation before torchrun, without loading a GPU model."""
    plan = read_json(plan_path)
    require_hash(plan["config"], plan["config_sha256"], "Compact configuration")
    require_hash(plan["cache_plan"], plan["cache_plan_sha256"], "Source cache plan")
    if source_hash(Path(plan.get("code_root", plan["repo"]))) != plan["source_sha256"]:
        raise ValueError("Source changed since compact job rendering")
    cfg, cache_plan, manifest = inspect_cache(plan["cache_plan"])
    if (digest(manifest) != plan["cache_manifest_digest"]
            or cache_plan["checkpoint_sha256"] != plan["checkpoint_sha256"]):
        raise ValueError("Compact job's frozen dependencies changed")
    requested = yaml.safe_load(Path(plan["config"]).read_text())
    if requested["training"].get("utility_study"):
        from stmem.training.utility_study import read_study_manifest

        study = requested["training"]["utility_study"]
        panel = read_study_manifest(study["manifest"], study["manifest_sha256"])
        validate_study_inputs(
            study, panel, teacher_sha256=cache_plan["checkpoint_sha256"],
            teacher_step=cache_plan["checkpoint_step"],
            encoded_cache_id=digest(manifest["identity"])[:24],
            hierarchy_cache_id=Path(plan["hierarchy_cache_root"]).name,
        )
        if study.get("initialization_checkpoint"):
            require_hash(study["initialization_checkpoint"], study["initialization_sha256"],
                         "Study initialization checkpoint")
        if study.get("activation_evidence"):
            require_hash(study["activation_evidence"], study["activation_evidence_sha256"],
                         "Conditional-study activation evidence")
    validate_policy_config(cfg, requested)
    expected_paths = {**cfg["paths"], "run_root": plan["run_root"]}
    if requested["data"] != cfg["data"] or requested["paths"] != expected_paths:
        raise ValueError("Compact job changed cached data or paths")
    if dependency(requested, "teacher") != Path(cache_plan["checkpoint"]):
        raise ValueError("Compact job changed the selected teacher dependency")
    if plan.get("hierarchy_cache_root"):
        require_hash(Path(plan["hierarchy_cache_root"]) / "manifest.json",
                     plan["hierarchy_manifest_sha256"], "Hierarchy manifest")
        hierarchy = inspect_hierarchy(plan["cache_path"], plan["hierarchy_cache_root"])
        if (digest(hierarchy) != plan["hierarchy_manifest_digest"]
                or requested["training"].get("hierarchy_cache_root") != plan["hierarchy_cache_root"]):
            raise ValueError("Compact hierarchy dependency changed")
    elif requested["training"].get("hierarchy_cache_root"):
        raise ValueError("Unrecorded hierarchy dependency in compact configuration")
    endpoint = plan.get("continue_to_step")
    if endpoint is not None and not plan["resume"]:
        raise ValueError("Compact continuation requires a resume checkpoint")
    if plan["resume"]:
        require_hash(plan["resume"], plan["resume_sha256"], "Resume checkpoint")
        continuation = continuation_record(
            requested, load_checkpoint(plan["resume"]), requested=False, continue_to_step=endpoint,
        )
        if continuation != plan.get("continuation"):
            raise ValueError("Compact continuation differs from the immutable startup plan")
    elif (run_dir(requested, "compact") / "last.pt").exists():
        raise ValueError("Compact checkpoint already exists; render with --resume")


def render_script(plan, site, cfg):
    code_root = plan.get("code_root", plan["repo"])
    directives = {
        "job-name": "stmem_compact_" + plan["name"], "partition": plan["partition"],
        "gres": f"gpu:{plan['gpus']}", "nodes": "1", "ntasks": "1",
        "cpus-per-task": str(plan["cpus"]), "mem": plan["mem"], "time": plan["walltime"],
        "output": str(Path(plan["logs"]) / "%x_%j.out"),
        "error": str(Path(plan["logs"]) / "%x_%j.err"),
        "chdir": code_root, "signal": "B:USR1@180",
    }
    for key in ("ACCOUNT", "QOS", "CONSTRAINT"):
        if site.get("STMEM_" + key):
            directives[key.lower()] = site["STMEM_" + key]
    if any("\n" in value or "\r" in value for value in directives.values()):
        raise ValueError("Newline in Slurm directive")
    lines = ["#!/usr/bin/env bash", *(f"#SBATCH --{k}={shlex.quote(v)}" for k, v in directives.items()),
             "set -euo pipefail", f"cd {shlex.quote(code_root)}"]
    if site.get("STMEM_ACTIVATE"):
        lines.append("source " + shlex.quote(site["STMEM_ACTIVATE"]))
    exported = {
        "STMEM_REPO_ROOT": code_root, "STMEM_PYTHON": plan["python"],
        "STMEM_SITE_FILE": "/dev/null", "PYTHONPATH": str(Path(code_root) / "src"),
        "STMEM_ALLOCATED_GPUS": str(plan["gpus"]), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "online", "WANDB_PROJECT": cfg["tracking"].get("project") or "chronotree",
        "WANDB_ENTITY": cfg["tracking"].get("entity") or "", "WANDB_DIR": cfg["tracking"].get("dir") or "",
        **{f"STMEM_{key.upper()}": cfg["paths"][key] for key in ("data_root", "cache_root", "run_root")},
    }
    for key in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_DIR"):
        if site.get(key):
            exported[key] = site[key]
    lines.append("unset RANK LOCAL_RANK WORLD_SIZE STMEM_FEATURE_CACHE_ROOT STMEM_SCRATCH_ROOT WANDB_RUN_ID WANDB_RESUME")
    for key, value in sorted(exported.items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    guard = "from hpc.train_compact import ready_plan; ready_plan(" + repr(plan["startup_plan"]) + ")"
    lines.append(shlex.join([plan["python"], "-c", guard]))
    lines.append("export STMEM_STOP_FILE=" + shlex.quote(str(Path(plan["script"]).parent / "stops"))
                 + '/"${SLURM_JOB_ID:-local_$$}.stop"')
    command = ["srun", "--ntasks=1", f"--cpus-per-task={plan['cpus']}", "--cpu-bind=cores", "--gpu-bind=none",
               plan["python"], "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc-per-node={plan['gpus']}", "-m", "stmem.train", "--stage", "compact",
               "--config", plan["config"], "--log-every", "1", "--attention-every", "10",
               "--progress-every-seconds", "30"]
    if plan["resume"]:
        command += ["--resume", plan["resume"]]
    if plan.get("continue_to_step") is not None:
        command += ["--continue-to-step", str(plan["continue_to_step"])]
    lines.append("exec " + shlex.quote(str(Path(code_root) / "hpc/run_job.sh"))
                 + ' "$STMEM_STOP_FILE" ' + shlex.join(command) + ' --stop-file "$STMEM_STOP_FILE"')
    return "\n".join(lines) + "\n"


def prepare(args, site, repo, cache_plan_path, generated):
    code_root = absolute_path(args.code_root, repo) if args.code_root is not None else repo
    if args.code_root is not None:
        for relative in ("src/stmem/train.py", "hpc/train_compact.py", "hpc/run_job.sh"):
            if not (code_root / relative).is_file():
                raise ValueError(f"Incomplete --code-root checkout: missing {code_root / relative}")
    cfg, cache_plan, manifest = inspect_cache(cache_plan_path)
    cache_root = Path(cfg["paths"]["cache_root"]) / digest(manifest["identity"])[:24]
    hierarchy = None
    if not args.legacy_cache:
        from stmem.data.hierarchy_cache import default_hierarchy_root

        args.hierarchy_root = (absolute_path(args.hierarchy_root, repo) if args.hierarchy_root
                               else default_hierarchy_root(cache_root))
        hierarchy = inspect_hierarchy(cache_root, args.hierarchy_root)
    if args.run_root is not None:
        cfg["paths"]["run_root"] = str(absolute_path(args.run_root, repo))
    cfg = compact_config(cfg, args)
    initial = load_checkpoint(cache_plan["checkpoint"])
    if cfg["training"].get("utility_study"):
        from stmem.training.utility_study import read_study_manifest

        study = cfg["training"]["utility_study"]
        panel = read_study_manifest(study["manifest"], study["manifest_sha256"])
        validate_study_inputs(
            study, panel, teacher_sha256=cache_plan["checkpoint_sha256"],
            teacher_step=initial["step"], encoded_cache_id=cache_root.name,
            hierarchy_cache_id=Path(args.hierarchy_root).name,
        )
        if set(cfg["data"]["tasks"]) != {row["task"] for row in panel["episode_partitions"]}:
            raise ValueError("Student study panel must partition every configured training task")
    if initial["stage"] != "teacher":
        raise ValueError("Selected input is not a teacher checkpoint")
    validate_policy_config(initial["config"], cfg)
    output = run_dir(cfg, "compact")
    if (output / "complete.json").exists() and args.continue_to_step is None:
        raise ValueError(f"Compact training already complete: {output}")
    plan_path, receipt_path = generated / "plan.json", generated / "submitted.json"
    previous = read_json(plan_path) if plan_path.exists() else None
    if (previous and args.code_root is None
            and Path(previous.get("code_root", str(repo))) != repo):
        raise ValueError(f"This compact plan uses an isolated checkout; repeat --code-root {previous['code_root']}")
    if previous is None and output.exists() and any(output.iterdir()):
        raise ValueError("Fresh compact training needs an empty output directory; choose a new --run-root")
    retry = args.resume or args.restart_unstarted
    if receipt_path.exists() and not retry:
        raise ValueError("Already submitted; use --resume after the job finishes")
    if retry and previous is None:
        raise ValueError("No existing compact plan to resume")
    if args.submit and receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["job_id"] in active_user_job_ids():
            raise ValueError(f"Compact job {receipt['job_id']} is still active; refusing duplicate submission")
    previous_endpoint = previous.get("continue_to_step") if previous else None
    if previous_endpoint is not None:
        if args.continue_to_step is None:
            raise ValueError("This compact plan uses continuation; repeat --continue-to-step when resuming")
        if args.continue_to_step < previous_endpoint:
            raise ValueError("Cannot shorten an already planned compact continuation endpoint")
    current_source = source_hash(code_root)
    signature_inputs = {"config": cfg, "teacher": cache_plan["checkpoint_sha256"],
                        "cache": digest(manifest), "gpus": args.gpus, "source": current_source}
    if hierarchy is not None:
        signature_inputs["hierarchy"] = digest(hierarchy)
    signature = digest(signature_inputs)
    if previous and previous["signature"] != signature:
        # A completed run predates this continuation feature. Permit its code
        # upgrade explicitly, while proving every scientific input is unchanged.
        previous_signature = digest({**signature_inputs, "source": previous["source_sha256"]})
        if args.continue_to_step is None or previous["signature"] != previous_signature:
            raise ValueError("Compact configuration/dependencies/world size/source changed; this plan is immutable")
    ownership = output.parent / ".compact-plan.json"
    owner_signature = previous["signature"] if previous else signature
    if ownership.exists() and read_json(ownership) != {"plan": str(plan_path), "signature": owner_signature}:
        raise ValueError("A different compact plan owns this output directory")
    checkpoint = output / "last.pt"
    if args.restart_unstarted and (checkpoint.exists() or any(output.glob("*.pt"))):
        raise ValueError("Run already has checkpoints; use --resume, not --restart-unstarted")
    if checkpoint.exists() and not args.resume:
        raise ValueError("Compact checkpoint already exists; use --resume")
    start = 0
    continuation = None
    endpoint = args.continue_to_step if args.continue_to_step is not None else cfg["training"]["steps"]
    if args.resume:
        saved = load_checkpoint(checkpoint)
        expected = {"teacher": cache_plan["checkpoint_sha256"], "cache": digest(manifest)}
        if hierarchy is not None:
            expected["hierarchy"] = digest(hierarchy)
        study = cfg["training"].get("utility_study")
        if study and study.get("initialization_checkpoint"):
            expected["study_initialization"] = study["initialization_sha256"]
        if (saved["stage"] != "compact" or saved["config"] != cfg
                or saved["provenance"]["world_size"] != args.gpus
                or saved["provenance"]["dependencies"] != expected):
            raise ValueError("Resume checkpoint configuration or dependencies differ")
        start = saved["step"]
        continuation = continuation_record(
            cfg, saved, requested=False, continue_to_step=args.continue_to_step,
        )
        if not 0 <= start < endpoint:
            raise ValueError("Resume checkpoint has no remaining compact training steps")
    config_path = generated / "compact.yaml"
    payload = yaml.safe_dump(cfg, sort_keys=False)
    if previous:
        require_hash(config_path, previous["config_sha256"], "Compact configuration")
        if config_path.read_text() != payload:
            raise ValueError("Saved compact configuration differs")
    else:
        temporary = config_path.with_suffix(".yaml.tmp")
        temporary.write_text(payload)
        temporary.replace(config_path)
    logs = generated / "logs"
    logs.mkdir(exist_ok=True)
    plan = {
        "kind": "shared-compact-training", "name": args.name, "repo": str(repo),
        "code_root": str(code_root),
        "python": site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python"),
        "plan_path": str(plan_path), "cache_plan": str(cache_plan_path),
        "cache_plan_sha256": sha256(cache_plan_path), "cache_manifest_digest": digest(manifest),
        "cache_path": str(Path(cfg["paths"]["cache_root"]) / digest(manifest["identity"])[:24]),
        "hierarchy_cache_root": str(args.hierarchy_root) if hierarchy is not None else None,
        "hierarchy_manifest_digest": digest(hierarchy) if hierarchy is not None else None,
        "hierarchy_manifest_sha256": sha256(args.hierarchy_root / "manifest.json") if hierarchy is not None else None,
        "checkpoint": cache_plan["checkpoint"], "checkpoint_sha256": cache_plan["checkpoint_sha256"],
        "teacher_step": initial["step"], "config": str(config_path), "config_sha256": sha256(config_path),
        "source_sha256": current_source, "signature": signature, "run_root": cfg["paths"]["run_root"],
        "previous_source_sha256": previous["source_sha256"] if previous else None,
        "output": str(output), "script": str(generated / "compact.sbatch"), "logs": str(logs),
        "gpus": args.gpus, "gres": f"gpu:{args.gpus}", "partition": args.partition,
        "cpus": args.cpus, "mem": args.mem, "walltime": args.walltime,
        "job_count": 1, "maximum_concurrent_allocations": 1,
        "seed": cfg["seed"], "tasks": cfg["data"]["tasks"], "training": cfg["training"],
        "microbatches_per_update_per_gpu": math.ceil(args.effective_batch / args.gpus / args.microbatch),
        "resume": str(checkpoint) if args.resume else None,
        "resume_sha256": sha256(checkpoint) if args.resume else None,
        "restart_unstarted": args.restart_unstarted,
        "start_step": start, "remaining_updates": endpoint - start,
        "continue_to_step": args.continue_to_step, "continuation": continuation,
        "candidate_checkpoints": [str(output / f"step_{step:08d}.pt")
                                  for step in cfg["training"]["checkpoint_steps"]],
        "accepted_evaluation_execution": cache_plan.get("evaluation_execution"),
        "estimated_gpu_hours": "unknown; cached compact training has not been measured on this allocation",
        "cache_verification": "manifest hash, identity, receipt consistency and shard sizes; no full-shard byte scan",
    }
    # Slurm copies the submitted shell script, but that script reads a plan at
    # startup. A dry-run for a later attempt must not change a queued job's guard.
    attempt = generated / "attempts" / (digest(plan)[:24] + ".json")
    plan["startup_plan"] = str(attempt)
    if attempt.exists():
        if read_json(attempt) != plan:
            raise ValueError("Immutable compact startup plan changed")
    else:
        save_json(attempt, plan)
    Path(plan["script"]).write_text(render_script(plan, site, cfg))
    save_json(plan_path, plan)
    save_json(ownership, {"plan": str(plan_path), "signature": signature})
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.submit:
        result = subprocess.run(["sbatch", "--parsable", plan["script"]], check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdecimal():
            raise ValueError(f"Unexpected sbatch receipt: {result.stdout!r}")
        save_json(receipt_path, {"job_id": job_id, "signature": signature,
                                "previous": read_json(receipt_path) if receipt_path.exists() else None})
        print(f"Submitted compact training job {job_id}")
    else:
        print("Dry-run only: compact configuration and job saved; no Slurm job submitted.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Default: validate dependencies and render only")
    action.add_argument("--submit", action="store_true")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site.env")
    parser.add_argument("--code-root", type=Path,
                        help="Runtime checkout; default site repository. Run/cache/generated paths stay at the site.")
    parser.add_argument("--cache-name", type=profile_name, required=True)
    parser.add_argument("--name", type=profile_name, help="Default: CACHE_NAME_compact_hierarchy")
    parser.add_argument("--run-root", type=Path,
                        help="Fresh compact output root; default REPO/runs/NAME")
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument("--hierarchy-root", type=Path,
                            help="Completed compatible hierarchy directory; default derived from leaf cache")
    cache_mode.add_argument("--legacy-cache", action="store_true",
                            help="Explicit old on-demand tree path; no hierarchy sidecar (slower)")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument("--resume", action="store_true", help="Resume this compact run's last.pt, after it stops")
    restart.add_argument("--restart-unstarted", action="store_true",
                         help="Retry an inactive attempt from step zero only if it has no checkpoints")
    parser.add_argument("--continue-to-step", type=positive_integer,
                        help="With --resume, extend a completed run to this optimizer step at its final LR")
    parser.add_argument("--partition", default="gpuq")
    parser.add_argument("--gpus", type=positive_integer, default=2)
    parser.add_argument("--cpus", type=positive_integer, default=64)
    parser.add_argument("--mem", default="128G")
    parser.add_argument("--walltime", type=walltime, default="48:00:00")
    parser.add_argument("--effective-batch", type=positive_integer, default=128)
    parser.add_argument("--microbatch", type=positive_integer, default=32)
    parser.add_argument("--workers", type=nonnegative_integer, default=16)
    args = parser.parse_args(argv)
    if args.continue_to_step is not None and not args.resume:
        parser.error("--continue-to-step requires --resume")
    args.name = args.name or args.cache_name + ("_compact" if args.legacy_cache else "_compact_hierarchy")
    if args.name == args.cache_name:
        parser.error("Compact --name must differ from --cache-name")
    if args.effective_batch % args.gpus or args.microbatch > args.effective_batch // args.gpus:
        parser.error("Batch must divide across GPUs and microbatch must fit within each rank's batch")
    if args.cpus < args.gpus * (1 + args.workers):
        parser.error("Request at least one CPU per training process and loading worker")
    if not re.fullmatch(r"[1-9][0-9]*(?:[KMGT]i?B?|[kmgt])?", args.mem):
        parser.error("Invalid --mem")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.partition):
        parser.error("Invalid --partition")
    try:
        site = read_site(args.site)
        repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
        cache_plan_path = repo / "hpc/generated" / args.cache_name / "plan.json"
        # All named launchers targeting this experiment share an ownership lock.
        cache_plan = read_json(cache_plan_path)
        root = absolute_path(args.run_root or (cache_plan["run_root"] if args.legacy_cache
                                              else repo / "runs" / args.name), repo)
        args.run_root = root
        root.mkdir(parents=True, exist_ok=True)
        with (root / ".compact-launch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            generated = repo / "hpc/generated" / args.name
            generated.mkdir(parents=True, exist_ok=True)
            prepare(args, site, repo, cache_plan_path, generated)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
