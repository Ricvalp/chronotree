#!/usr/bin/env python3
"""Render one fresh, isolated teacher training job; only --submit calls Slurm."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
    PROFILES,
    Job,
    config_hashes,
    positive_integer,
    positive_seconds,
    read_site,
    render_job,
    resolved_config,
    save_json,
)
from stmem.config import validate_config  # noqa: E402


def experiment_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise argparse.ArgumentTypeError("name must use 1–96 letters, digits, hyphens or underscores")
    return value


def absolute_path(value: str | Path, repo: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Default: render without submission")
    action.add_argument("--submit", action="store_true", help="Render, then submit exactly one job")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site.env")
    parser.add_argument("--profile", choices=[name for name in PROFILES if name != "a5000_debug"])
    parser.add_argument("--config", type=Path, help="Default: configs/teacher.yaml in the site repository")
    parser.add_argument("--name", type=experiment_name, required=True, help="Unique generated job name")
    parser.add_argument(
        "--run-root", type=Path, required=True, help="Fresh experiment root; must not already exist"
    )
    parser.add_argument("--log-every", type=positive_integer, default=1)
    parser.add_argument("--progress-every-seconds", type=positive_seconds, default=30)
    args = parser.parse_args(argv)
    try:
        if not args.site.is_file():
            raise FileNotFoundError(f"Missing site configuration: {args.site}")
        site = read_site(args.site)
        if any("\n" in value or "\r" in value for value in site.values()):
            raise ValueError("Newline in site configuration")
        repo = absolute_path(site.get("STMEM_REPO_ROOT") or REPO, REPO)
        profile = args.profile or site.get("STMEM_PROFILE") or "h200_1gpu"
        if profile not in PROFILES or profile == "a5000_debug":
            raise ValueError("Use an H200/B200 profile here; run local diagnostics directly")
        source_config = absolute_path(args.config or "configs/teacher.yaml", repo)
        sources = config_hashes(source_config)
        cfg = resolved_config(source_config)
        validate_config(cfg)
        if cfg.get("stage") != "teacher" or cfg.get("data", {}).get("synthetic", False):
            raise ValueError("This launcher requires a real-data teacher configuration")
        if cfg.get("data", {}).get("split", "train") != "train":
            raise ValueError("Train on released training episodes only, never validation/test seeds")
        tracking = cfg.get("tracking", {})
        if not tracking.get("enabled", True) or (
            site.get("WANDB_MODE") or tracking.get("mode", "offline")
        ) not in ("online", "offline"):
            raise ValueError("Teacher training requires W&B tracking in online or offline mode")
        budget = cfg["training"]["effective_batch"]
        steps = cfg["training"]["steps"]
        if (
            not isinstance(budget, int)
            or isinstance(budget, bool)
            or not isinstance(steps, int)
            or isinstance(steps, bool)
        ):
            raise ValueError("training.effective_batch and training.steps must be positive integers")
        gpus = PROFILES[profile]
        if budget % gpus:
            raise ValueError("training.effective_batch must be divisible by the GPU count")
        seed = cfg["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        candidates = cfg["training"].get("checkpoint_steps", [])
        if not isinstance(candidates, list) or any(
            not isinstance(step, int) or isinstance(step, bool) or not 1 <= step <= steps
            for step in candidates
        ):
            raise ValueError("training.checkpoint_steps must contain integers between 1 and training.steps")
        root = absolute_path(args.run_root, repo)
        generated = repo / "hpc/generated/training" / args.name
        if root.exists():
            raise FileExistsError(f"Run root already exists; choose a fresh --run-root: {root}")
        if (generated / "submitted.json").exists():
            raise FileExistsError(f"This named job was already submitted; choose a new --name: {args.name}")
        paths = cfg.setdefault("paths", {})
        for env, key, fallback in (
            ("STMEM_DATA_ROOT", "data_root", "data"),
            ("STMEM_CACHE_ROOT", "cache_root", "cache"),
        ):
            paths[key] = str(absolute_path(site.get(env) or paths.get(key, fallback), repo))
            site[env] = paths[key]
        paths["run_root"] = str(root)
        cfg["profile"] = profile
        python = site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python")
        site.update(
            STMEM_REPO_ROOT=str(repo),
            STMEM_RUN_ROOT=str(root),
            STMEM_PYTHON=python,
        )
        output = root / f"seed{seed}" / "teacher"
        resolved = generated / "config.yaml"
        command = [
            python,
            "-m",
            "stmem.train",
            "--stage",
            "teacher",
            "--config",
            str(resolved),
            "--log-every",
            str(args.log_every),
            "--progress-every-seconds",
            str(args.progress_every_seconds),
        ]
        job = Job(
            name=f"seed{seed}_teacher",
            stage="teacher",
            commands=[command],
            outputs=[
                *(str(output / f"step_{step:08d}.pt") for step in sorted(set(candidates))),
                str(output / "last.pt"),
                str(output / "complete.json"),
            ],
            gpus=gpus,
            walltime=site.get("STMEM_WALLTIME") or "48:00:00",
            seed=seed,
        )
        if not re.fullmatch(r"(?:\d+-)?\d{1,3}:[0-5]\d:[0-5]\d", job.walltime) or not any(
            char in "123456789" for char in job.walltime
        ):
            raise ValueError("STMEM_WALLTIME must be a positive [days-]hours:minutes:seconds")
        configured_microbatch = cfg["training"].get("microbatch", 16)
        local_examples = budget // gpus
        actual_microbatch = min(configured_microbatch, local_examples)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))

    # The resolved snapshot is the input to the existing stage-signature renderer.
    # Neither rendering nor submission creates the actual training output root.
    generated.mkdir(parents=True, exist_ok=True)
    logs = repo / "hpc/logs"
    logs.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    render_job(
        job,
        site=site,
        repo=repo,
        generated=generated,
        logs=logs,
        python=python,
        profile=profile,
    )
    # Claim the fresh root atomically before job_status creates any stage files.
    # This also protects against differently named queued jobs targeting one root.
    # Clear inherited W&B identity settings for this explicitly fresh-only launch.
    script = Path(job.script)
    fresh_launch = "\n".join(
        [
            "unset WANDB_RUN_ID WANDB_RESUME",
            shlex.join(["mkdir", "-p", "--", str(root.parent)]),
            shlex.join(["mkdir", "--", str(root)]),
        ]
    )
    body = script.read_text()
    anchor = "\nSTMEM_STOP_FILE="
    if body.count(anchor) != 1:
        raise RuntimeError("Cannot add the fresh-output guard to the rendered training script")
    script.write_text(body.replace(anchor, "\n" + fresh_launch + anchor, 1))
    plan = {
        "kind": "teacher-training",
        "name": args.name,
        "profile": profile,
        "gpus": gpus,
        "job_count": 1,
        "jobs": [asdict(job)],
        "source_config": str(source_config),
        "source_config_sha256": sources,
        "config": str(resolved),
        "run_root": str(root),
        "output": str(output),
        "steps": steps,
        "effective_batch": budget,
        "microbatch": configured_microbatch,
        "actual_microbatch_per_gpu": actual_microbatch,
        "examples_per_gpu": local_examples,
        "gradient_accumulation_steps": math.ceil(local_examples / actual_microbatch),
        "encoder_block_batch_size": cfg["training"].get("encoder_block_batch_size", 32),
        "log_every": args.log_every,
        "progress_every_seconds": args.progress_every_seconds,
        "resumes_existing_training": False,
        "estimated_gpu_hours": "unknown; measured training timings required",
    }
    save_json(generated / "plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    print(
        f"Batch: {budget} global / {gpus} GPUs = {local_examples} examples per GPU; "
        f"actual microbatch {actual_microbatch} (configured cap {configured_microbatch}); "
        f"{plan['gradient_accumulation_steps']} forward/backward pass(es) per update.",
        flush=True,
    )
    if not args.submit:
        print("Dry-run only. Inspect config.yaml, jobs/seed*_teacher.sbatch and plan.json.")
        print(f"Repeat the same options with --name {shlex.quote(args.name)} --submit to submit.")
        return
    result = subprocess.run(["sbatch", "--parsable", job.script], check=True, capture_output=True, text=True)
    raw = result.stdout.strip()
    if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", raw):
        raise RuntimeError(f"sbatch returned an unexpected job identifier: {raw!r}")
    job_id = raw.split(";", 1)[0]
    save_json(
        generated / "submitted.json",
        {"job_id": job_id, "script": job.script, "run_root": str(root), "output": str(output)},
    )
    print(f"Submitted teacher training job {job_id}; output: {output}")


if __name__ == "__main__":
    main()
