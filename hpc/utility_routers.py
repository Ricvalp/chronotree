#!/usr/bin/env python3
"""Prepare and submit the distinct expert-loss, immediate/packet router pilot.

Only --submit contacts Slurm. The all stage schedules preparation, one label
array and four independent matched fits, each followed by offline evaluation.
Closed-loop screening is a separate explicit stage after those fits complete.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import active_user_job_ids, read_site, sha256  # noqa: E402
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from hpc.refinement_utility import measurement_hours  # noqa: E402
from hpc.utility_jobs import (  # noqa: E402
    current_source_snapshot, immutable_json, read_json, render_job, runtime_environment, submit_job,
    validate_resources,
)
from hpc.utility_source import create_study_source, verify_study_source  # noqa: E402
from stmem.config import digest  # noqa: E402
from stmem.training.runtime import load_checkpoint  # noqa: E402


VARIANTS = tuple((reader, target) for reader in ("TREE", "FLAT") for target in ("immediate", "packet"))


def fit_directory(output, reader, target):
    return Path(output) / "fits" / f"{reader.lower()}_{target}"


def completed(plan, key):
    output = Path(plan["output"])
    if key == "prepare":
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            return False
        manifest = read_json(manifest_path)
        if manifest.get("schema") != "utility_router_study_v1":
            raise ValueError("Pilot output contains a different experiment schema")
        body = {key: value for key, value in manifest.items() if key != "experiment_id"}
        if digest(body) != manifest.get("experiment_id"):
            raise ValueError("Pilot manifest changed after preparation")
        if manifest.get("student", {}).get("sha256") != plan["checkpoint_sha256"]:
            raise ValueError("Pilot manifest contains another frozen student")
        return True
    if key == "labels":
        for worker in range(plan["label_shards"]):
            path = output / "labels" / f"worker_{worker}.json"
            if not path.exists() or read_json(path).get("status") != "complete":
                return False
        return True
    reader, target = key.removeprefix("fit_").removeprefix("offline_").split("_", 1)
    directory = fit_directory(output, reader, target)
    if key.startswith("fit_"):
        path = directory / "complete.json"
        evaluation = directory / "evaluation/summary.json"
        if not (path.is_file() and (directory / "last.pt").is_file() and evaluation.is_file()):
            return False
        receipt, result = read_json(path), read_json(evaluation)
        if receipt.get("status") != "complete" or result.get("status") != "complete":
            return False
        checkpoint_hash = sha256(directory / "last.pt")
        if (receipt.get("checkpoint_sha256") != checkpoint_hash
                or result.get("router_sha256") != checkpoint_hash):
            raise ValueError("Completed router checkpoint changed after fitting/evaluation")
        return True
    summary = directory / "evaluation/summary.json"
    return summary.is_file() and read_json(summary).get("status") == "complete"


def render_plan(args, repo, site, generated):
    if args.parent_manifest is None or args.checkpoint is None:
        raise ValueError("The first invocation needs --parent-manifest and --checkpoint")
    parent = absolute_path(args.parent_manifest, repo)
    checkpoint = absolute_path(args.checkpoint, repo)
    original_source = create_study_source(repo, parent)
    code_root = current_source_snapshot(repo, generated)
    python = site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python")
    output = repo / "runs" / args.name
    study = output / "manifest.json"
    inputs = {str(parent): sha256(parent), str(checkpoint): sha256(checkpoint),
              str(original_source / ".utility-source.json"): sha256(original_source / ".utility-source.json")}
    commands = {
        "prepare": [[python, "-u", "-m", "stmem.training.utility_router_study",
                     "--parent-manifest", str(parent), "--checkpoint", str(checkpoint),
                     "--student", args.student, "--output", str(output),
                     "--fit-states", str(args.fit_states), "--steps", str(args.steps),
                     "--workers", str(args.label_shards), "--study-source-root", str(original_source)]],
        "labels": [[python, "-u", "-m", "stmem.training.utility_router_data",
                    "--study", str(study), "--output", str(output / "labels"),
                    "--worker-index", "@WORKER@", "--workers", str(args.label_shards),
                    "--max-hours", measurement_hours(args.label_walltime), "--device", "cuda:0"]],
    }
    if args.development_step is not None:
        commands["prepare"][0] += ["--development-step", str(args.development_step)]
    dependencies = {"prepare": [], "labels": ["prepare"]}
    for reader, target in VARIANTS:
        variant = f"{reader.lower()}_{target}"
        base = [python, "-u", "-m", "stmem.training.utility_routers",
                "--study", str(study), "--reader", reader, "--target", target,
                "--output", str(fit_directory(output, reader, target)), "--device", "cuda"]
        commands["fit_" + variant] = [base + ["--mode", "train", "--resume"], base + ["--mode", "evaluate"]]
        commands["offline_" + variant] = [base + ["--mode", "evaluate"]]
        dependencies["fit_" + variant] = ["labels"]
        dependencies["offline_" + variant] = ["fit_" + variant]
    jobs = {}
    for key, run_commands in commands.items():
        label = key == "labels"
        fit = key.startswith("fit_") or key.startswith("offline_")
        path, job = render_job(
            generated, key, repo=repo, code_root=code_root, python=python, site=site,
            commands=run_commands, inputs=inputs, partition=args.partition, gpu_type=args.gpu_type,
            cpus=4 if fit else 8, mem="16G" if fit else "64G",
            walltime=args.label_walltime if label else args.fit_walltime if fit else args.prepare_walltime,
            workers=args.label_shards if label else 1,
            max_concurrent=args.max_concurrent if label else None,
            exclude=args.exclude, dependencies=dependencies[key],
            outputs=[study] if key == "prepare" else [output / "labels"] if label else
                    [fit_directory(output, *key.removeprefix("fit_").removeprefix("offline_").split("_", 1))],
        )
        jobs[key] = {"plan": str(path), "dependencies": dependencies[key], "resources": job["resources"]}
    return {"schema": "utility_router_hpc_v1", "name": args.name, "repo": str(repo),
            "site": str(args.site.resolve()), "python": python, "code_root": str(code_root),
            "parent_source_root": str(original_source), "parent_manifest": str(parent),
            "checkpoint": str(checkpoint), "checkpoint_sha256": inputs[str(checkpoint)],
            "student": args.student, "development_step": args.development_step,
            "output": str(output), "study": str(study), "fit_states": args.fit_states,
            "steps": args.steps, "label_shards": args.label_shards, "jobs": jobs,
            "launch_options": {key: getattr(args, key) for key in (
                "student", "development_step", "fit_states", "steps", "label_shards", "max_concurrent",
                "partition", "gpu_type", "exclude", "prepare_walltime", "label_walltime", "fit_walltime")},
            "resources": {"partition": args.partition, "gpu_type": args.gpu_type, "exclude": args.exclude},
            "scientific_scope": "Separate expert-FM immediate/completed-packet pilot; parent study is unchanged",
            "maximum_concurrent_gpu_allocations": max(args.max_concurrent or args.label_shards, 4),
            "allocated_gpu_hours": "unknown; no throughput estimate from old router fits"}


def selected_jobs(stage):
    fits = [f"fit_{reader.lower()}_{target}" for reader, target in VARIANTS]
    if stage == "all":
        return ["prepare", "labels", *fits]
    if stage in ("prepare", "labels"):
        return [stage]
    if stage == "train":
        return fits
    if stage == "offline":
        return [key.replace("fit_", "offline_", 1) for key in fits]
    return []


def launch_jobs(plan, stage, *, submit, resume):
    selected = selected_jobs(stage)
    statuses = {key: completed(plan, key) for key in plan["jobs"]}
    result = {"selected_jobs": selected, "completed_skipped": [key for key in selected if statuses[key]],
              "remaining": [key for key in selected if not statuses[key]]}
    if not submit:
        return result
    active = active_user_job_ids() if resume else set()
    ids = {}
    for key in selected:
        if statuses[key]:
            continue
        after = []
        for dependency in plan["jobs"][key]["dependencies"]:
            if statuses[dependency]:
                continue
            if dependency in ids:
                after.append(ids[dependency])
                continue
            receipt_path = Path(plan["jobs"][dependency]["plan"]).with_suffix(".submitted.json")
            if receipt_path.is_file():
                receipt = read_json(receipt_path)
                if receipt["job_id"] in active:
                    after.append(receipt["job_id"])
                    continue
            raise ValueError(f"{key} requires incomplete {dependency}; submit --stage all --resume, or complete that stage first")
        ids[key] = submit_job(plan["jobs"][key]["plan"], resume=resume, after=after, active=active)
    result["job_ids"] = ids
    return result


def evaluation_study(plan):
    """Verify the child under its own runtime, even after a main-checkout pull."""
    path, code_root = Path(plan["study"]), Path(plan["code_root"])
    if not path.is_file():
        raise ValueError("Prepare the pilot before rendering closed-loop screening")
    verify_study_source(code_root, path)
    command = ("import json,sys; from stmem.training.utility_router_study import verify_study; "
               "print(json.dumps(verify_study(sys.argv[1])))")
    result = subprocess.run([plan["python"], "-c", command, str(path)], cwd=code_root,
                            env=runtime_environment(code_root, cpu=True), check=True,
                            capture_output=True, text=True)
    study = json.loads(result.stdout)
    if (study["student"]["sha256"] != plan["checkpoint_sha256"]
            or study["parent_manifest"] != plan["parent_manifest"]):
        raise ValueError("Child study no longer matches the rendered router pilot")
    return study


def evaluation_inputs(plan, study):
    """Require all four completed fits and immutable policy/solver identities.

    This runs before the first evaluation dry-run, so an invalid fourth fit
    cannot leave three apparently ready evaluation plans behind.
    """
    protocol = study["protocol"]
    budget, execute, flow_steps = (protocol[key] for key in ("budget", "execute_targets", "flow_steps"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
           for value in (budget, execute, flow_steps)):
        raise ValueError("Child study contains invalid allocation or action settings")
    student = Path(study["student"]["checkpoint"])
    student_hash = study["student"]["sha256"]
    if not student.is_file() or sha256(student) != student_hash:
        raise ValueError("Frozen pilot student checkpoint is missing or changed")
    state = load_checkpoint(student)
    cfg = deepcopy(state["config"])
    if state.get("stage") != "compact" or state.get("step") != study["student"]["step"]:
        raise ValueError("Frozen pilot student stage/step differs from the child study")
    if cfg["model"]["flow_steps"] != flow_steps:
        raise ValueError("Frozen student flow integration steps differ from the immutable pilot protocol")
    if not 1 <= execute <= cfg["model"]["horizon"]:
        raise ValueError("Executed target count exceeds the frozen action horizon")
    for key in ("horizon", "solver", "action_stride"):
        if cfg["model"].get(key) != study["configuration"]["model"].get(key):
            raise ValueError(f"Frozen policy {key} differs from the pilot action convention")
    # --config makes the checkpoint's exact solver explicit to hpc/evaluate;
    # --execute-actions is the existing validation-only cadence override.
    cfg["model"]["budget"] = budget
    routers = {}
    intended = protocol["fit"]["steps"]
    for reader, target in VARIANTS:
        key = f"fit_{reader.lower()}_{target}"
        if not completed(plan, key):
            raise ValueError(f"Router fit and offline evaluation must complete before screening: {key}")
        directory = fit_directory(plan["output"], reader, target)
        path = directory / "last.pt"
        receipt, summary = read_json(directory / "complete.json"), read_json(directory / "evaluation/summary.json")
        router = load_checkpoint(path)
        identity = router.get("identity", {})
        if (router.get("schema") != "utility_router_checkpoint_v1" or router.get("stage") != "router"
                or router.get("step") != intended or receipt.get("completed_steps") != intended
                or receipt.get("intended_steps") != intended
                or summary.get("router_step") != intended):
            raise ValueError(f"Router {key} did not complete the declared fitting schedule")
        if (identity.get("experiment_id") != study["experiment_id"]
                or identity.get("student_sha256") != student_hash
                or identity.get("teacher_sha256") != study["dependencies"]["teacher_sha256"]
                or identity.get("reader") != reader or identity.get("target") != target
                or router.get("utility_protocol") != protocol or router.get("protocol") != protocol
                or identity.get("protocol") != protocol):
            raise ValueError(f"Router {key} belongs to another student, protocol or supervision target")
        if (receipt.get("experiment_id") != study["experiment_id"]
                or receipt.get("reader") != reader or receipt.get("target") != target
                or summary.get("experiment_id") != study["experiment_id"]
                or summary.get("reader") != reader or summary.get("target") != target
                or summary.get("student", {}).get("sha256") != student_hash):
            raise ValueError(f"Router {key} completion/evaluation receipts belong to another fit")
        routers[reader, target] = path
    config = Path(plan["output"]) / "screening/policy_config.json"
    immutable_json(config, cfg)
    return student, config, routers


def evaluate_jobs(plan, args):
    """Matched packet-credit screen, with gain-blind packet and global controls."""
    repo = Path(plan["repo"])
    evaluation_name = args.eval_name or plan["name"]
    study = evaluation_study(plan)
    student, config, routers = evaluation_inputs(plan, study)
    execute, budget = study["protocol"]["execute_targets"], study["protocol"]["budget"]
    variants = [(f"{reader.lower()}_{target}", reader, target, "learned") for reader, target in VARIANTS]
    variants += [(f"{reader.lower()}_random", reader, "packet", "random") for reader in ("TREE", "FLAT")]
    variants += [("uniform", "UNIFORM", None, None)]
    commands = []
    for suffix, reader, target, selection in variants:
        name = f"{evaluation_name}_{suffix}_{args.controller}_eval"
        command = [plan["python"], str(repo / "hpc/evaluate.py"), "--site", str(args.site.resolve()),
                   "--checkpoint", str(student), "--config", str(config), "--name", name,
                   "--method", reader, "--budget", str(budget), "--execute-actions", str(execute),
                   "--partition", plan["resources"]["partition"], "--gpu-type", plan["resources"]["gpu_type"],
                   "--gpus", str(args.eval_gpus), "--workers-per-gpu", str(args.workers_per_gpu),
                   "--cpus", str(args.eval_cpus), "--mem", "128G", "--walltime", args.eval_walltime,
                   "--episode-count", str(args.episode_count), "--video-cap", str(args.video_cap),
                   "--wandb-project", "chronotree-routers"]
        if target:
            router = routers[reader, target]
            command += ["--router-checkpoint", str(router), "--router-controller", args.controller,
                        "--router-selection", selection]
        if args.resume:
            command.append("--resume")
        command.append("--submit" if args.submit else "--dry-run")
        commands.append(command)
    # Validate/render the complete screen before any allocation is submitted.
    for command in commands:
        dry_command = ["--dry-run" if value == "--submit" else value for value in command]
        subprocess.run(dry_command, cwd=repo, check=True)
    if args.submit:
        for command in commands:
            subprocess.run(command, cwd=repo, check=True)
    return {"evaluation_name": evaluation_name, "evaluation_commands": commands,
            "episodes_per_task": args.episode_count,
            "controller_comparison": args.controller, "same_episode_seeds": True}


def main(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--submit", action="store_true")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site_b200_4gpu.env")
    parser.add_argument("--name", type=profile_name, default="utility_router_std95k")
    parser.add_argument("--stage", choices=("all", "prepare", "labels", "train", "offline", "evaluate", "status"), default="all")
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--student", choices=("S_STD", "S_STAGE", "S_PACKET"), default="S_STD")
    parser.add_argument("--development-step", type=int, default=95000)
    parser.add_argument("--fit-states", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--label-shards", type=int, default=4)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--partition", default="aiq")
    parser.add_argument("--gpu-type", default="b200")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--prepare-walltime", type=walltime, default="02:00:00")
    parser.add_argument("--label-walltime", type=walltime, default="04:00:00")
    parser.add_argument("--fit-walltime", type=walltime, default="02:00:00")
    parser.add_argument("--eval-walltime", type=walltime, default="06:00:00")
    parser.add_argument("--eval-name", type=profile_name,
                        help="Separate output prefix for this evaluation campaign; keeps the fitted --name pilot fixed")
    parser.add_argument("--eval-gpus", type=int, default=4)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--eval-cpus", type=int, default=64)
    parser.add_argument("--episode-count", type=int, default=2)
    parser.add_argument("--video-cap", type=int, default=10)
    parser.add_argument("--controller", choices=("packet", "local"), default="packet")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(supplied)
    if args.eval_name is not None and args.stage != "evaluate":
        parser.error("--eval-name is only used with --stage evaluate")
    validate_resources(args.partition, args.gpu_type, args.exclude)
    if (not 1 <= args.label_shards <= 32 or min(args.fit_states, args.steps, args.eval_gpus,
                                             args.workers_per_gpu, args.eval_cpus, args.episode_count) < 1
            or args.video_cap < 0 or (args.development_step is not None and args.development_step < 1)):
        parser.error("Use positive bounded work/resource counts")
    if args.max_concurrent is not None and not 1 <= args.max_concurrent <= args.label_shards:
        parser.error("--max-concurrent must be between 1 and --label-shards")
    site = read_site(args.site)
    repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
    generated = repo / "hpc/generated" / args.name
    path = generated / "plan.json"
    if args.stage == "status":
        if not path.is_file():
            print(json.dumps({"status": "not_prepared", "plan": str(path)}, indent=2))
            return
        plan = read_json(path)
        print(json.dumps({"output": plan["output"], "stages": {key: completed(plan, key) for key in plan["jobs"]}}, indent=2))
        return
    generated.mkdir(parents=True, exist_ok=True)
    with (generated / ".launcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.is_file():
            plan = read_json(path)
            explicit = {value.split("=", 1)[0] for value in supplied if value.startswith("--")}
            for key, value in plan["launch_options"].items():
                if "--" + key.replace("_", "-") in explicit and getattr(args, key) != value:
                    raise ValueError(f"Immutable --{key.replace('_', '-')} differs; choose a new --name")
            for key in ("parent_manifest", "checkpoint"):
                supplied_path = getattr(args, key)
                if supplied_path is not None and str(absolute_path(supplied_path, repo)) != plan[key]:
                    raise ValueError(f"Immutable {key} differs; choose a new --name")
        else:
            plan = render_plan(args, repo, site, generated)
            immutable_json(path, plan)
        result = (evaluate_jobs(plan, args) if args.stage == "evaluate" else
                  launch_jobs(plan, args.stage, submit=args.submit, resume=args.resume))
        print(json.dumps({**plan, **result}, indent=2))
        if not args.submit and args.stage != "evaluate":
            print("Dry-run: immutable jobs are ready; only --submit contacts Slurm.")


if __name__ == "__main__":
    main()
