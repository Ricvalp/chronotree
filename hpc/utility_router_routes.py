#!/usr/bin/env python3
"""Submit a separate full-route offline check of the four completed routers.

The previous pilot is verified under its own source, then adopted without
editing it. This diagnostic owns a new source snapshot and manifest. Only
--submit contacts Slurm; status only reads existing filesystem progress.
"""
from __future__ import annotations

import argparse
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
    current_source_snapshot, immutable_json, read_json, render_job, runtime_environment,
    submit_job, validate_resources,
)
from hpc.utility_routers import evaluation_inputs, evaluation_study  # noqa: E402
from hpc.utility_source import verify_study_source  # noqa: E402


SCHEMA = "utility_router_routes_v1"
LAUNCH_SCHEMA = "utility_router_routes_hpc_v1"


def diagnostic_manifest(pilot_path, pilot, study, routers, code_root, generated, *, workers, batch_size):
    """Adopt validated fixed inputs; no selection by observed routing scores."""
    from stmem.inspection.utility_router_routes import build_manifest

    states = [state for state in study["states"] if state["partition"] == "development"]
    if len(states) != 128 or len({state["state_id"] for state in states}) != len(states):
        raise ValueError("The full-route check requires the pilot's exact 128 development states")
    if any(state.get("cache_eligible") is False for state in states):
        raise ValueError("The fixed development panel contains an ineligible state; do not silently replace it")
    protocol = study["protocol"]
    if (protocol["budget"], protocol["execute_targets"], protocol["flow_steps"]) != (128, 8, 8):
        raise ValueError("This matched full-route diagnostic requires budget128, execute8, flow8")
    code_root = Path(code_root)
    source_receipt = read_json(code_root / ".utility-source.json")
    parent_source_receipt = Path(pilot["code_root"]) / ".utility-source.json"
    validation = {
        "status": "complete", "study_path": str(Path(pilot["study"]).resolve()),
        "study_sha256": sha256(Path(pilot["study"])), "experiment_id": study["experiment_id"],
        "source_sha256": study["source"]["source_sha256"],
        "source_receipt": str(parent_source_receipt), "source_receipt_sha256": sha256(parent_source_receipt),
        "code_root": pilot["code_root"], "pilot_plan": str(Path(pilot_path).resolve()),
        "pilot_plan_sha256": sha256(pilot_path),
    }
    validation_path = Path(generated) / "original_runtime_validation.json"
    immutable_json(validation_path, validation)
    parent_runtime = {"path": str(validation_path), "sha256": sha256(validation_path),
                      "source_sha256": study["source"]["source_sha256"], "code_root": pilot["code_root"]}
    router_inputs = {f"{reader.lower()}_{target}": {
        "checkpoint": str(Path(path).resolve()), "sha256": sha256(path), "reader": reader,
        "target": target, "step": study["protocol"]["fit"]["steps"],
    } for (reader, target), path in sorted(routers.items())}
    source = {"git": source_receipt["git"], "source_sha256": source_receipt["source_sha256"], "dirty": False}
    return build_manifest(pilot["study"], study, parent_runtime, router_inputs, source,
                          workers=workers, batch_size=batch_size)


def render_plan(args, repo, site, generated):
    pilot_path = repo / "hpc/generated" / args.pilot_name / "plan.json"
    if not pilot_path.is_file():
        raise ValueError(f"Completed pilot plan is missing: {pilot_path}")
    pilot = read_json(pilot_path)
    study = evaluation_study(pilot)
    student, _, routers = evaluation_inputs(pilot, study)
    code_root = current_source_snapshot(repo, generated)
    output = repo / "runs" / args.name
    manifest_path = output / "manifest.json"
    manifest = diagnostic_manifest(pilot_path, pilot, study, routers, code_root, generated,
                                   workers=args.workers, batch_size=args.batch_size)
    immutable_json(manifest_path, manifest)
    python = site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python")
    inputs = {str(manifest_path): sha256(manifest_path), str(pilot_path): sha256(pilot_path),
              str(Path(pilot["study"])): sha256(Path(pilot["study"])), str(student): sha256(student),
              str(generated / "original_runtime_validation.json"): sha256(generated / "original_runtime_validation.json"),
              str(Path(pilot["code_root"]) / ".utility-source.json"): sha256(Path(pilot["code_root"]) / ".utility-source.json"),
              study["parent_manifest"]: study["parent_manifest_sha256"]}
    for item in manifest["routers"].values():
        inputs[item["checkpoint"]] = item["sha256"]
        for receipt in ("complete.json", "evaluation/summary.json"):
            path = Path(item["checkpoint"]).parent / receipt
            inputs[str(path)] = sha256(path)
    command = [python, "-u", "-m", "stmem.inspection.utility_router_routes", "--mode", "measure",
               "--manifest", str(manifest_path), "--output", str(output),
               "--worker-index", "@WORKER@", "--workers", str(args.workers),
               "--batch-size", str(args.batch_size), "--device", "cuda:0",
               "--max-hours", measurement_hours(args.walltime)]
    job_path, job = render_job(generated, "full_routes", repo=repo, code_root=code_root, python=python,
                              site=site, commands=[command], inputs=inputs, partition=args.partition,
                              gpu_type=args.gpu_type, cpus=8, mem="64G", walltime=args.walltime,
                              workers=args.workers, max_concurrent=args.max_concurrent, exclude=args.exclude,
                              outputs=[output / f"worker_{worker}.json" for worker in range(args.workers)])
    return {"schema": LAUNCH_SCHEMA, "name": args.name, "pilot_name": args.pilot_name,
            "pilot_plan": str(pilot_path), "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path), "experiment_id": manifest["experiment_id"],
            "repo": str(repo), "python": python, "code_root": str(code_root), "output": str(output),
            "job": str(job_path), "resources": job["resources"],
            "launch_options": {key: getattr(args, key) for key in (
                "pilot_name", "workers", "max_concurrent", "batch_size", "partition", "gpu_type", "walltime", "exclude")},
            "scope": "Frozen development states and full memory routes; no simulator success or final-partition claim"}


def status(plan):
    output = Path(plan["output"])
    states = {}
    for worker in range(plan["launch_options"]["workers"]):
        path = output / f"worker_{worker}.json"
        if not path.is_file():
            states[str(worker)] = {"status": "not_started"}
            continue
        receipt = read_json(path)
        if receipt.get("experiment_id") != plan["experiment_id"]:
            raise ValueError(f"Worker {worker} receipt belongs to another diagnostic")
        states[str(worker)] = {key: receipt.get(key) for key in ("status", "completed", "planned")}
    return {"status": "complete" if all(item["status"] == "complete" for item in states.values()) else "incomplete",
            "workers": states, "output": str(output), "summary": str(output / "summary.json"),
            "report": str(output / "report.md")}


def validate_plan(plan):
    path = Path(plan["manifest"])
    if not path.is_file() or sha256(path) != plan["manifest_sha256"]:
        raise ValueError("Full-route diagnostic manifest changed")
    verify_study_source(plan["code_root"], path)
    job = read_json(plan["job"])
    if any(not Path(path).is_file() or sha256(Path(path)) != expected for path, expected in job["inputs"].items()):
        raise ValueError("Frozen full-route diagnostic input changed")


def verified_results(plan):
    """Check actual state hashes and full coverage before skipping an array."""
    code = ("import json,sys; from stmem.inspection.utility_router_routes import verify_manifest,verify_results; "
            "m=verify_manifest(sys.argv[1]); print(json.dumps(verify_results(m,sys.argv[2])))")
    result = subprocess.run([plan["python"], "-c", code, plan["manifest"], plan["output"]],
                            cwd=plan["code_root"], env=runtime_environment(plan["code_root"], cpu=True),
                            check=True, capture_output=True, text=True)
    verified = json.loads(result.stdout)
    if verified.get("status") != "complete":
        raise ValueError("Full-route worker receipts claim completion without verified complete results")
    return verified


def main(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--submit", action="store_true")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site_b200_4gpu.env")
    parser.add_argument("--name", type=profile_name, default="utility_router_std95k_full_routes")
    parser.add_argument("--pilot-name", type=profile_name, default="utility_router_std95k")
    parser.add_argument("--stage", choices=("run", "status", "report"), default="run")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--partition", default="aiq")
    parser.add_argument("--gpu-type", default="b200")
    parser.add_argument("--walltime", type=walltime, default="04:00:00")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(supplied)
    if args.workers < 1 or not 1 <= args.max_concurrent <= args.workers or args.batch_size < 1:
        parser.error("Use positive worker/batch counts and max-concurrent <= workers")
    validate_resources(args.partition, args.gpu_type, args.exclude)
    site = read_site(args.site)
    repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
    generated = repo / "hpc/generated" / args.name
    plan_path = generated / "plan.json"
    if args.stage == "status":
        result = status(read_json(plan_path)) if plan_path.is_file() else {"status": "not_prepared", "plan": str(plan_path)}
        print(json.dumps(result, indent=2))
        return
    generated.mkdir(parents=True, exist_ok=True)
    with (generated / ".launcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if plan_path.is_file():
            plan = read_json(plan_path)
            explicit = {value.split("=", 1)[0] for value in supplied if value.startswith("--")}
            for key, value in plan["launch_options"].items():
                if "--" + key.replace("_", "-") in explicit and getattr(args, key) != value:
                    raise ValueError(f"Immutable --{key.replace('_', '-')} differs; choose a new --name")
        else:
            if args.stage == "report":
                raise ValueError("Prepare and complete the diagnostic before requesting its report")
            plan = render_plan(args, repo, site, generated)
            immutable_json(plan_path, plan)
        validate_plan(plan)
        progress = status(plan)
        if args.stage == "run" and progress["status"] == "complete":
            progress["verified_results"] = verified_results(plan)
        if args.stage == "report":
            if progress["status"] != "complete":
                raise ValueError("Every full-route worker must complete before the aggregate report")
            command = [plan["python"], "-u", "-m", "stmem.inspection.utility_router_routes", "--mode", "report",
                       "--manifest", plan["manifest"], "--output", plan["output"]]
            subprocess.run(command, cwd=plan["code_root"], env=runtime_environment(plan["code_root"], cpu=True), check=True)
        elif args.submit and progress["status"] != "complete":
            progress["job_id"] = submit_job(plan["job"], resume=args.resume,
                                            active=active_user_job_ids() if args.resume else set())
        print(json.dumps({**plan, **progress}, indent=2))
        if args.stage == "run" and not args.submit:
            print("Dry-run: separate full-route diagnostic and typed-B200 array ready; add --submit to launch.")


if __name__ == "__main__":
    main()
