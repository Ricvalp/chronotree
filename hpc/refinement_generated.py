#!/usr/bin/env python3
"""Plan and submit matched generated-action diagnostics using the original study code."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import active_user_job_ids, read_site, sha256  # noqa: E402
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from hpc.refinement_utility import measurement_hours, runtime_manifest  # noqa: E402
from hpc.utility_jobs import (  # noqa: E402
    immutable_json, read_json, render_job, runtime_environment, submit_job, validate_resources,
)
from hpc.utility_source import create_study_source  # noqa: E402


STUDENTS = ("S_STD", "S_STAGE", "S_PACKET")


def planning_adapter(generated):
    """Freeze the administrative read optimization outside the original source."""
    source = Path(__file__).with_name("refinement_generated_planning.py")
    fingerprint = sha256(source)
    destination = Path(generated) / "planning" / fingerprint / "planner.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(".partial.py")
        temporary.write_bytes(source.read_bytes())
        if sha256(temporary) != fingerprint:
            raise ValueError("Planning adapter changed while snapshotting it")
        temporary.replace(destination)
    if sha256(destination) != fingerprint:
        raise ValueError("Frozen planning adapter changed")
    return destination, fingerprint


# The archived scientific function writes the aggregate HTML after each worker.
# Serialize that presentation-only part; model generation remains independent.
GENERATE = """import fcntl,json,sys
from pathlib import Path
from stmem.inspection import refinement_generated as module
plan=Path(sys.argv[1])
original=module.report
def serial_report(path):
    with plan.with_suffix('.report.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        return original(path)
module.report=serial_report
result=module.generate(plan,student=sys.argv[2],device='cuda:0',batch_size=int(sys.argv[3]),max_hours=float(sys.argv[4]))
print(json.dumps(result,indent=2))
raise SystemExit(75 if result.get('status')=='partial' else 0)
"""


def complete_student(case_plan, student):
    plan = read_json(case_plan)
    output = case_plan.parent / (case_plan.stem + "_results")
    receipt_path = output / f"receipt_{student}.json"
    if not receipt_path.is_file():
        return False
    receipt = read_json(receipt_path)
    expected = [[student, case["state_id"]] for case in plan["cases"]]
    if (receipt.get("plan_id") != plan["plan_id"] or receipt.get("status") != "complete"
            or sorted(receipt.get("completed", [])) != sorted(expected)
            or sorted(receipt.get("planned", [])) != sorted(expected)):
        return False
    for case in plan["cases"]:
        data = output / student / f"{case['state_id']}.npz"
        metadata = data.with_suffix(".json")
        if not data.is_file() or not metadata.is_file():
            return False
        saved = read_json(metadata)
        if saved.get("plan_id") != plan["plan_id"] or saved.get("arrays_sha256") != sha256(data):
            raise ValueError("Completed generated-action artifact changed")
    return True


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--submit", action="store_true")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site_b200_4gpu.env")
    parser.add_argument("--name", type=profile_name, default="utility140k_generated95k")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stage", choices=("run", "plan", "generate", "report", "status"), default="run")
    parser.add_argument("--checkpoints", nargs="*", default=[], metavar="STUDENT=PATH")
    parser.add_argument("--limited-check", action="store_true",
                        help="Explicitly label matching earlier snapshots as development, not primary confirmation")
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--partition", default="aiq")
    parser.add_argument("--gpu-type", default="b200")
    parser.add_argument("--walltime", type=walltime, default="02:00:00")
    parser.add_argument("--exclude", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_cases <= 16 or args.batch_size < 1:
        parser.error("Use 1..16 cases and a positive batch size")
    validate_resources(args.partition, args.gpu_type, args.exclude)
    site = read_site(args.site)
    repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
    generated = repo / "hpc/generated" / args.name
    case_plan = repo / "runs" / args.name / "generated_plan.json"
    plan_path = generated / "plan.json"
    if args.stage == "status":
        if not case_plan.is_file():
            print(json.dumps({"status": "not_prepared", "plan": str(case_plan)}, indent=2))
            return
        plan = read_json(case_plan)
        result = {student: complete_student(case_plan, student) for student in plan["students"]}
        print(json.dumps({"status": "complete" if all(result.values()) else "incomplete", "students": result,
                          "plan": str(case_plan), "results": str(case_plan.parent / "generated_plan_results")}, indent=2))
        return
    generated.mkdir(parents=True, exist_ok=True)
    progress_path = generated / "preparation_progress.jsonl"

    def progress(phase, **details):
        row = {"event": "generated_action_preparation", "phase": phase,
               "pid": os.getpid(), "elapsed_seconds": time.perf_counter() - started, **details}
        line = json.dumps(row)
        with progress_path.open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    with (generated / ".launcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read_json(plan_path) if plan_path.is_file() else None
        manifest_path = absolute_path(args.manifest, repo) if args.manifest else (
            Path(previous["manifest"]) if previous else None)
        if manifest_path is None:
            parser.error("The first invocation requires --manifest")
        if previous and previous["manifest"] != str(manifest_path):
            raise ValueError("This launch name already belongs to a different parent study")
        progress("validating_source_snapshot")
        code_root = create_study_source(repo, manifest_path)
        python = site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python")
        progress("checking_cache_identity")
        runtime_manifest(python, code_root, manifest_path)
        if not case_plan.is_file() or args.stage in ("run", "plan"):
            checkpoints = []
            for entry in args.checkpoints:
                if "=" not in entry:
                    parser.error("Use --checkpoints STUDENT=PATH")
                student, path = entry.split("=", 1)
                if student not in STUDENTS or not path:
                    parser.error("Checkpoints must name S_STD, S_STAGE or S_PACKET")
                checkpoints.append(f"{student}={absolute_path(path, repo)}")
            if not checkpoints and previous:
                checkpoints = previous["checkpoints"]
            if not checkpoints:
                parser.error("Supply all three matching --checkpoints for the first case plan")
            names = [entry.split("=", 1)[0] for entry in checkpoints]
            if sorted(names) != sorted(STUDENTS):
                parser.error("This matched launcher requires exactly S_STD, S_STAGE and S_PACKET")
            adapter, adapter_hash = planning_adapter(generated)
            command = [python, "-u", str(adapter), "--progress-file", str(progress_path), "--mode", "plan",
                       "--manifest", str(manifest_path), "--plan", str(case_plan),
                       "--max-cases", str(args.max_cases), "--checkpoints", *sorted(checkpoints)]
            limited = args.limited_check or (previous or {}).get("limited_check", False)
            if limited:
                command.append("--limited-check")
            progress("selecting_common_cases", adapter_sha256=adapter_hash)
            subprocess.run(command, cwd=code_root, env=runtime_environment(code_root, cpu=True), check=True)
            if sha256(adapter) != adapter_hash:
                raise ValueError("Planning adapter changed during execution")
            plan_hash = sha256(case_plan)
            immutable_json(adapter.parent / f"{plan_hash}.receipt.json", {
                "schema": "generated_case_planning_io_v1", "adapter": str(adapter),
                "adapter_sha256": adapter_hash, "original_code_root": str(code_root),
                "manifest_sha256": sha256(manifest_path), "case_plan": str(case_plan),
                "case_plan_sha256": plan_hash,
                "scope": "Memoize NPZ columns per file; original checks and case selection unchanged",
            })
            progress("case_plan_ready", plan=str(case_plan))
        cases = read_json(case_plan)
        if not cases["cases"]:
            raise ValueError("No common development cases available; do not submit empty GPU jobs")
        checkpoint_inputs = {value["checkpoint"]: value["sha256"] for value in cases["students"].values()}
        inputs = {str(manifest_path): sha256(manifest_path), str(case_plan): sha256(case_plan), **checkpoint_inputs}
        jobs = {}
        progress("rendering_jobs", selected_cases=len(cases["cases"]))
        for student in sorted(cases["students"]):
            key = "generated_" + student.lower()
            command = [python, "-c", GENERATE, str(case_plan), student, str(args.batch_size),
                       measurement_hours(args.walltime)]
            path, job = render_job(generated, key, repo=repo, code_root=code_root, python=python, site=site,
                                  commands=[command], inputs=inputs, partition=args.partition, gpu_type=args.gpu_type,
                                  walltime=args.walltime, exclude=args.exclude, outputs=[
                                      case_plan.parent / "generated_plan_results" / f"receipt_{student}.json"])
            jobs[student] = {"plan": str(path), "resources": job["resources"]}
        launch = {"schema": "refinement_generated_hpc_v1", "manifest": str(manifest_path),
                  "code_root": str(code_root), "case_plan": str(case_plan), "case_plan_id": cases["plan_id"],
                  "checkpoints": sorted(f"{key}={value['checkpoint']}" for key, value in cases["students"].items()),
                  "limited_check": cases["limited_check"], "jobs": jobs,
                  "report": str(case_plan.parent / "generated_plan_results/report.md"),
                  "html": str(case_plan.parent / "generated_plan_results/index.html"),
                  "scope": "Generated joint-target/FK paths; no simulator success is measured"}
        immutable_json(plan_path, launch)
        if args.stage == "report":
            # Presentation is CPU-only and the same advisory lock is used by workers.
            with case_plan.with_suffix(".report.lock").open("a+") as report_lock:
                fcntl.flock(report_lock, fcntl.LOCK_EX)
                subprocess.run([python, "-m", "stmem.inspection.refinement_generated", "--mode", "report",
                                "--plan", str(case_plan)], cwd=code_root,
                               env=runtime_environment(code_root, cpu=True), check=True)
        elif args.submit and args.stage in ("run", "generate"):
            active = active_user_job_ids() if args.resume else set()
            launch["submission"] = {}
            for student, job in jobs.items():
                if complete_student(case_plan, student):
                    launch["submission"][student] = "complete_skipped"
                    progress("complete_skipped", student=student)
                else:
                    progress("submitting_job", student=student)
                    job_id = submit_job(job["plan"], resume=args.resume, active=active)
                    launch["submission"][student] = job_id
                    progress("job_submitted_or_reused", student=student, job_id=job_id)
        print(json.dumps(launch, indent=2))
        if not args.submit and args.stage in ("run", "generate"):
            print("Dry-run: matched case plan and three independent typed-B200 jobs are ready; add --submit to launch.")


if __name__ == "__main__":
    main()
