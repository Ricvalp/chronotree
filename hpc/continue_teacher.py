#!/usr/bin/env python3
"""Render two matched teacher continuations; only --submit calls Slurm."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import (  # noqa: E402
    PROFILES,
    Job,
    active_user_job_ids,
    positive_integer,
    read_site,
    render_job,
    save_json,
)
from stmem.config import digest, validate_config  # noqa: E402
from stmem.training.checkpoints import atomic_copy, file_sha256  # noqa: E402
from stmem.training.forks import teacher_fork_config  # noqa: E402
from stmem.training.runtime import load_checkpoint  # noqa: E402


BRANCHES = {"control": True, "no_previous_joint_commands": False}


def absolute_path(value: str | Path, repo: Path) -> Path:
    return (repo / Path(value).expanduser()).resolve()


def overlaps(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _frozen_parent(generated: Path, source: Path):
    """Reuse the reviewed snapshot, never a moving last.pt, after planning."""
    plan_path = generated / "plan.json"
    snapshot = generated / "inputs" / "teacher.pt"
    if not plan_path.exists():
        if snapshot.exists():
            raise FileExistsError(
                f"Checkpoint snapshot exists without a completed plan: {snapshot}. "
                "Use a new --output-dir; an unreviewed snapshot will not be replaced."
            )
        return None
    previous = json.loads(plan_path.read_text())
    if previous.get("kind") != "matched_teacher_continuations":
        raise ValueError("Existing generated directory does not contain a teacher continuation plan")
    if previous.get("source_checkpoint") != str(source) or previous.get("snapshot") != str(snapshot):
        raise ValueError("The checkpoint source path differs from the existing frozen continuation plan")
    if file_sha256(snapshot) != previous.get("source_sha256"):
        raise ValueError("Frozen teacher snapshot hash differs from the reviewed plan")
    return load_checkpoint(snapshot)


def _fresh_destinations(run_root: Path, generated: Path, seed: int) -> None:
    for path in (run_root / "continuation_submission.json", generated / "submitted.json"):
        if path.exists():
            raise FileExistsError(
                f"This continuation pair already has a submission reservation: {path}. "
                "Inspect its job IDs; resume existing branches instead of submitting fresh copies."
            )
    for branch in BRANCHES:
        output = run_root / branch / f"seed{seed}" / "teacher"
        if output.exists():
            raise FileExistsError(f"Continuation output already exists; choose a fresh --run-root: {output}")


def _prepare(args, site, repo, run_root, generated, source, saved, profile):
    """Called with both output-directory locks held; never invokes Slurm."""
    seed = saved["config"]["seed"]
    _fresh_destinations(run_root, generated, seed)
    snapshot = generated / "inputs" / "teacher.pt"
    # Reuse a reviewed plan's pinned input even if the original last.pt advances
    # between dry-run and submission. The first plan copies one immutable input.
    pinned = _frozen_parent(generated, source)
    if pinned is None:
        atomic_copy(source, snapshot, immutable=True)
        pinned = load_checkpoint(snapshot)
    source_sha256 = file_sha256(snapshot)
    if pinned["step"] != saved["step"] or pinned["config"] != saved["config"]:
        raise ValueError("Source checkpoint changed while planning; rerun with a fixed checkpoint file")
    identity = {
        "source_sha256": source_sha256,
        "source_step": pinned["step"],
        "run_root": str(run_root),
        "additional_steps": args.additional_steps,
        "profile": profile,
    }
    plan_path = generated / "plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError(
                "Existing generated plan describes a different experiment; use a new --output-dir"
            )
    python = site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python")
    logs = repo / "hpc/logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs = []
    configs = {}
    for branch, enabled in BRANCHES.items():
        branch_root = run_root / branch
        cfg = teacher_fork_config(
            pinned,
            run_root=branch_root,
            additional_steps=args.additional_steps,
            condition_on_previous_joint_commands=enabled,
        )
        cfg["profile"] = profile
        branch_site = {**site, "STMEM_RUN_ROOT": str(branch_root), "STMEM_REPO_ROOT": str(repo)}
        for env, field in (("STMEM_DATA_ROOT", "data_root"), ("STMEM_CACHE_ROOT", "cache_root")):
            cfg["paths"][field] = str(absolute_path(site.get(env) or cfg["paths"][field], repo))
            branch_site[env] = cfg["paths"][field]
        validate_config(cfg)
        path = generated / "configs" / f"{branch}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs[branch] = str(path)
        final_step = cfg["training"]["steps"]
        output = branch_root / f"seed{seed}" / "teacher"
        job = Job(
            name=f"seed{seed}_teacher_{branch}",
            stage="teacher",
            commands=[
                [
                    python,
                    "-m",
                    "stmem.train",
                    "--stage",
                    "teacher",
                    "--config",
                    str(path),
                    "--fork-from",
                    str(snapshot),
                    "--fork-sha256",
                    source_sha256,
                    "--log-every",
                    "1",
                    "--progress-every-seconds",
                    "30",
                ]
            ],
            inputs=[str(snapshot)],
            outputs=[str(output / f"step_{final_step:08d}.pt")],
            gpus=PROFILES[profile],
            walltime=site.get("STMEM_WALLTIME", "48:00:00"),
            seed=seed,
        )
        render_job(
            job,
            site=branch_site,
            repo=repo,
            generated=generated,
            logs=logs,
            python=python,
            profile=profile,
        )
        script = Path(job.script)
        script.write_text(
            script.read_text().replace(
                "set -euo pipefail\n", "set -euo pipefail\nunset WANDB_RUN_ID WANDB_RESUME\n", 1
            )
        )
        jobs.append(job)
    details = {
        "kind": "matched_teacher_continuations",
        **identity,
        "source_checkpoint": str(source),
        "snapshot": str(snapshot),
        "parent_tracking": pinned.get("tracking", {}),
        "end_step": final_step,
        "schedule_steps": cfg["training"]["schedule_steps"],
        "configs": configs,
        "jobs": [asdict(job) for job in jobs],
        "job_count": len(jobs),
        "maximum_concurrent_allocations": len(jobs),
        "gpus_per_allocation": PROFILES[profile],
        "maximum_total_gpus": len(jobs) * PROFILES[profile],
        "estimated_gpu_hours": "unknown; measured timings required",
        "resume_existing_runs": False,
    }
    save_json(plan_path, details)
    return details, jobs


def _resume_inputs(args, run_root, generated, repo, profile):
    """Validate both saved branches before writing any resume artifacts."""
    plan = json.loads((generated / "plan.json").read_text())
    if plan.get("kind") != "matched_teacher_continuations":
        raise ValueError("--resume requires the original matched continuation plan")
    if plan.get("run_root") != str(run_root) or plan.get("profile") != profile:
        raise ValueError("Resume run root and GPU profile must match the original continuation plan")
    if args.checkpoint and str(absolute_path(args.checkpoint, repo)) != plan["source_checkpoint"]:
        raise ValueError("--checkpoint differs from the original continuation source")
    if args.additional_steps is not None:
        raise ValueError("--resume preserves the original endpoint; do not pass --additional-steps")
    end = plan["end_step"]
    if end != plan["source_step"] + plan["additional_steps"] or end > plan["schedule_steps"]:
        raise ValueError("The original continuation plan has an inconsistent training budget")
    if set(plan["configs"]) != set(BRANCHES):
        raise ValueError("The original continuation plan must contain both matched branches")
    branches = {}
    for branch, enabled in BRANCHES.items():
        config_path = generated / "configs" / f"{branch}.yaml"
        if plan["configs"][branch] != str(config_path):
            raise ValueError(f"Unexpected original config path for {branch}")
        expected = yaml.safe_load(config_path.read_text())
        output = run_root / branch / f"seed{expected['seed']}" / "teacher"
        checkpoint = output / "last.pt"
        saved = load_checkpoint(checkpoint)
        cfg = saved["config"]
        if cfg != expected:
            raise ValueError(f"{branch} checkpoint config differs from its original continuation config")
        if saved.get("stage") != "teacher" or cfg.get("stage") != "teacher":
            raise ValueError(f"{branch} requires a teacher checkpoint")
        if cfg.get("debug", False) or cfg["data"].get("synthetic", False):
            raise ValueError("HPC resumes require real-data teacher checkpoints")
        if cfg["paths"]["run_root"] != str(run_root / branch):
            raise ValueError(f"{branch} checkpoint belongs to a different run directory")
        if cfg["model"].get("condition_on_previous_joint_commands", True) is not enabled:
            raise ValueError(f"{branch} checkpoint has the wrong previous-joint-command conditioning")
        if cfg["training"]["steps"] != end or cfg["training"].get("schedule_steps") != plan["schedule_steps"]:
            raise ValueError(f"{branch} checkpoint changed the original endpoint or LR schedule")
        if not plan["source_step"] <= saved["step"] < end:
            raise ValueError(f"{branch} checkpoint is outside the unfinished continuation budget")
        provenance = saved["provenance"]
        if provenance["world_size"] != PROFILES[profile]:
            raise ValueError(f"{branch} resume GPU count must equal the checkpoint's world size")
        for field in ("model", "training"):
            if provenance.get(field) != digest(cfg[field]):
                raise ValueError(f"{branch} checkpoint {field} disagrees with its provenance")
        if (
            provenance["seed"] != cfg["seed"]
            or provenance["effective_batch"] != cfg["training"]["effective_batch"]
        ):
            raise ValueError(f"{branch} checkpoint batch/seed provenance mismatch")
        expected_sampler = {
            "next_global_position": saved["step"] * provenance["effective_batch"],
            "boundary": "optimizer_step",
            "version": provenance["sampler"],
            "rank_assignment": provenance["rank_assignment"],
        }
        if saved.get("sampler") != expected_sampler or len(saved["rng_by_rank"]) != PROFILES[profile]:
            raise ValueError(f"{branch} checkpoint lacks a complete optimizer-step resume boundary")
        lineage = saved.get("fork") or {}
        expected_lineage = {
            "source_sha256": plan["source_sha256"],
            "source_step": plan["source_step"],
            "schedule_steps": plan["schedule_steps"],
            "intended_final_step": end,
            "source_tracking_id": plan["parent_tracking"].get("id"),
            "changes": {"condition_on_previous_joint_commands": {"before": True, "after": enabled}},
        }
        if any(lineage.get(key) != value for key, value in expected_lineage.items()):
            raise ValueError(f"{branch} checkpoint fork lineage differs from the original continuation plan")
        tracking = json.loads((output / "tracking.json").read_text())
        if not saved.get("tracking", {}).get("id") or tracking.get("id") != saved["tracking"]["id"]:
            raise ValueError(f"{branch} tracking.json does not preserve the checkpoint's W&B run identity")
        for key in ("project", "entity", "job_type"):
            if key in saved["tracking"] and tracking.get(key) != saved["tracking"][key]:
                raise ValueError(f"{branch} tracking.json changed the checkpoint's W&B {key}")
        if tracking.get("job_type") != "teacher" or not tracking.get("project"):
            raise ValueError(f"{branch} tracking.json must identify a teacher run and W&B project")
        if tracking["id"] == plan["parent_tracking"].get("id"):
            raise ValueError(f"{branch} must retain its own W&B identity, separate from the parent")
        validate_config(cfg)
        branches[branch] = {"checkpoint": checkpoint, "saved": saved, "tracking": tracking}
    if len({item["tracking"]["id"] for item in branches.values()}) != len(BRANCHES):
        raise ValueError("The matched branches must have distinct W&B run identities")
    return plan, branches


def _resume_receipt(run_root, generated, expected_names):
    paths = (run_root / "continuation_submission.json", generated / "submitted.json")
    receipts = [json.loads(path.read_text()) for path in paths]
    previous = receipts[0]
    if receipts[1] != previous or previous.get("generated") != str(generated):
        raise ValueError("Continuation submission receipts disagree; inspect them before resubmitting")
    if previous.get("status") != "submitted" or set(previous.get("jobs", {})) != set(expected_names):
        raise ValueError(
            "Previous submission is ambiguous or incomplete; inspect its Slurm jobs before resuming"
        )
    ids = {item.get("job_id", "") for item in previous["jobs"].values()}
    if len(ids) != len(BRANCHES) or any(not value.isdecimal() for value in ids):
        raise ValueError("The prior submission does not identify both Slurm jobs")
    # Query the user's current queue, not --jobs=<expired ids>, which can fail
    # once Slurm has purged a completed allocation. Query failures fail closed.
    active = ids.intersection(active_user_job_ids())
    if active:
        raise ValueError(f"Continuation jobs are still queued or running: {', '.join(sorted(active))}")
    return previous


def _prepare_resume(plan, branches, site, repo, generated, profile):
    # Preserve the original fork recipe. Resume artifacts never replace its
    # source snapshot, config files or original sbatch scripts.
    destination = generated / "resume"
    python = site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python")
    logs = repo / "hpc/logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs, configs, progress = [], {}, {}
    for branch, item in branches.items():
        cfg = deepcopy(item["saved"]["config"])
        tracking = item["tracking"]
        branch_site = {**site, "STMEM_REPO_ROOT": str(repo)}
        for env, field in (
            ("STMEM_RUN_ROOT", "run_root"),
            ("STMEM_DATA_ROOT", "data_root"),
            ("STMEM_CACHE_ROOT", "cache_root"),
        ):
            branch_site[env] = cfg["paths"][field]
        # Existing tracking.json remains untouched; keep its project/entity even
        # if the current shell/site file was configured for another experiment.
        branch_site["WANDB_PROJECT"] = tracking["project"]
        branch_site["WANDB_ENTITY"] = tracking.get("entity") or ""
        path = destination / "configs" / f"{branch}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs[branch] = str(path)
        checkpoint = item["checkpoint"]
        job = Job(
            name=f"seed{cfg['seed']}_teacher_{branch}",
            stage="teacher",
            seed=cfg["seed"],
            commands=[
                [
                    python,
                    "-m",
                    "stmem.train",
                    "--stage",
                    "teacher",
                    "--config",
                    str(path),
                    "--resume",
                    str(checkpoint),
                    "--log-every",
                    "1",
                    "--progress-every-seconds",
                    "30",
                ]
            ],
            inputs=[str(checkpoint)],
            outputs=[str(checkpoint.parent / f"step_{plan['end_step']:08d}.pt")],
            gpus=PROFILES[profile],
            walltime=site.get("STMEM_WALLTIME", "48:00:00"),
        )
        render_job(
            job, site=branch_site, repo=repo, generated=destination, logs=logs, python=python, profile=profile
        )
        script = Path(job.script)
        script.write_text(
            script.read_text().replace(
                "set -euo pipefail\n",
                "set -euo pipefail\nunset WANDB_RUN_ID WANDB_RESUME\n",
                1,
            )
        )
        jobs.append(job)
        progress[branch] = {
            "checkpoint": str(checkpoint),
            "checkpoint_step": item["saved"]["step"],
            "remaining_steps": plan["end_step"] - item["saved"]["step"],
            "wandb_run_id": tracking["id"],
        }
    details = {
        **plan,
        "configs": configs,
        "jobs": [asdict(job) for job in jobs],
        "resume_existing_runs": True,
        "resume_progress": progress,
        "original_plan": str(generated / "plan.json"),
    }
    save_json(destination / "plan.json", details)
    return details, jobs


def _submit(jobs, run_root, generated, previous=None):
    # Reserve before calling sbatch: even an ambiguous submission failure must
    # not allow an automatic duplicate on a second invocation.
    receipt = {
        "status": "submitting",
        "generated": str(generated),
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "jobs": {},
    }
    if previous is not None:
        receipt["history"] = [
            *previous.get("history", []),
            {key: value for key, value in previous.items() if key != "history"},
        ]
        receipt["resume_existing_runs"] = True

    def record():
        save_json(run_root / "continuation_submission.json", receipt)
        save_json(generated / "submitted.json", receipt)

    record()
    try:
        for job in jobs:
            result = subprocess.run(
                ["sbatch", "--parsable", job.script], capture_output=True, text=True, check=True
            )
            job_id = result.stdout.strip().split(";")[0]
            if not job_id.isdecimal():
                raise ValueError(f"Unrecognized sbatch response for {job.name}: {result.stdout.strip()!r}")
            receipt["jobs"][job.name] = {"job_id": job_id, "script": job.script, "signature": job.signature}
            record()
            print(f"Submitted {job.name}: {job_id}", flush=True)
        receipt["status"] = "submitted"
        record()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        receipt["status"] = "submission_incomplete"
        receipt["error"] = str(error)
        record()
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry-run", action="store_true", help="Default: snapshot and render without submission"
    )
    action.add_argument("--submit", action="store_true", help="Submit both independent continuation jobs")
    parser.add_argument("--checkpoint", type=Path, help="Parent checkpoint; required for a new pair")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each existing branch from its own last.pt; preserve the original endpoint",
    )
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site_b200_4gpu.env")
    parser.add_argument("--profile", choices=[name for name in PROFILES if name != "a5000_debug"])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--additional-steps", type=positive_integer, help="New pairs only; defaults to 10000")
    args = parser.parse_args(argv)
    try:
        if not args.site.is_file():
            raise FileNotFoundError(f"Configure the site file before planning: {args.site}")
        site = read_site(args.site)
        repo = Path(site.get("STMEM_REPO_ROOT", str(REPO))).expanduser().resolve()
        run_root = absolute_path(args.run_root, repo)
        generated = absolute_path(args.output_dir or Path("hpc/generated") / run_root.name, repo)
        profile = args.profile or site.get("STMEM_PROFILE") or "b200_4gpu"
        if profile not in PROFILES or profile == "a5000_debug":
            raise ValueError("Matched HPC continuations require a supported non-debug GPU profile")
        if args.resume:
            # Unlike fresh planning, resume never creates missing run roots.
            # Load the checkpoints once, under both locks, before rendering.
            with ExitStack() as stack:
                for directory in sorted({run_root, generated}):
                    stream = stack.enter_context((directory / ".continue_teacher.lock").open("a"))
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                plan, branches = _resume_inputs(args, run_root, generated, repo, profile)
                previous = (
                    _resume_receipt(run_root, generated, [job["name"] for job in plan["jobs"]])
                    if args.submit
                    else None
                )
                details, jobs = _prepare_resume(plan, branches, site, repo, generated, profile)
                print(json.dumps(details, indent=2, sort_keys=True), flush=True)
                if args.submit:
                    _submit(jobs, run_root, generated, previous=previous)
                else:
                    print(f"DRY RUN: no jobs submitted. Inspect {generated / 'resume/plan.json'}", flush=True)
            return
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when creating a new continuation pair")
        args.additional_steps = 10000 if args.additional_steps is None else args.additional_steps
        source = absolute_path(args.checkpoint, repo)
        saved = _frozen_parent(generated, source)
        if saved is None:
            saved = load_checkpoint(source)
        if saved.get("stage") != "teacher" or saved["config"].get("stage") != "teacher":
            raise ValueError("Matched continuations require a teacher checkpoint")
        if saved["config"].get("debug", False) or saved["config"]["data"].get("synthetic", False):
            raise ValueError("HPC continuations require a real-data teacher checkpoint")
        if saved["config"]["model"].get("condition_on_previous_joint_commands", True) is not True:
            raise ValueError(
                "Both continuations must start from a teacher trained with previous joint commands"
            )
        if saved["provenance"]["world_size"] != PROFILES[profile]:
            raise ValueError("The continuation GPU count must equal the checkpoint's world size")
        original_root = absolute_path(saved["config"]["paths"]["run_root"], repo)
        if overlaps(run_root, original_root) or overlaps(run_root, source.parent):
            raise ValueError(
                "The new run root must be separate from the source checkpoint and original run root"
            )
        if overlaps(generated, run_root) or overlaps(generated, source.parent):
            raise ValueError("Generated files must be separate from source and destination run directories")
        # Validate the bounded training recipe before creating any directories.
        teacher_fork_config(
            saved,
            run_root=run_root / "control",
            additional_steps=args.additional_steps,
            condition_on_previous_joint_commands=True,
        )
        # Protect both keys: alternate output directories cannot submit the same
        # run root concurrently, and alternate run roots cannot overwrite a plan.
        with ExitStack() as stack:
            for directory in sorted({run_root, generated}):
                directory.mkdir(parents=True, exist_ok=True)
                stream = stack.enter_context((directory / ".continue_teacher.lock").open("a"))
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            details, jobs = _prepare(args, site, repo, run_root, generated, source, saved, profile)
            print(json.dumps(details, indent=2, sort_keys=True), flush=True)
            if args.submit:
                _submit(jobs, run_root, generated)
            else:
                print(f"DRY RUN: no jobs submitted. Inspect {generated / 'plan.json'}", flush=True)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
