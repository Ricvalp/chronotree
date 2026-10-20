#!/usr/bin/env python3
"""Render one isolated teacher-memory profiling job; only --submit calls Slurm."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import (  # noqa: E402
    PROFILES,
    WANDB_SITE_VARIABLES,
    config_hashes,
    positive_integer,
    read_site,
    resolved_config,
    save_json,
)


def warmup_steps(value: str) -> int:
    count = positive_integer(value)
    if not 5 <= count <= 10:
        raise argparse.ArgumentTypeError("warmup must be between 5 and 10 optimizer steps")
    return count


def measured_steps(value: str) -> int:
    count = positive_integer(value)
    if count > 100:
        raise argparse.ArgumentTypeError("profiling is bounded to at most 100 measured optimizer steps")
    return count


def profile_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise argparse.ArgumentTypeError("name must use 1–96 letters, digits, hyphens or underscores")
    return value


def walltime(value: str) -> str:
    if not re.fullmatch(r"(?:\d+-)?\d{1,3}:[0-5]\d:[0-5]\d", value) or not any(
        char in "123456789" for char in value
    ):
        raise argparse.ArgumentTypeError("walltime must be a positive [days-]hours:minutes:seconds")
    return value


def absolute_path(value: str | Path, repo: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


def render_script(
    *,
    site: dict[str, str],
    repo: Path,
    logs: Path,
    profile: str,
    gpus: int,
    python: str,
    job_name: str,
    time_limit: str,
    arguments: list[str],
) -> str:
    """Render literal resources and invoke the isolated, single-agent job body."""
    directives = {
        "job-name": f"stmem_profile_{job_name}",
        "nodes": "1",
        "ntasks": "1",
        "cpus-per-task": site.get("STMEM_CPUS", "16"),
        "mem": site.get("STMEM_MEM", "128G"),
        "time": time_limit,
        "output": str(logs / "%x_%j.out"),
        "error": str(logs / "%x_%j.err"),
        "chdir": str(repo),
    }
    for setting, directive in (
        ("PARTITION", "partition"),
        ("ACCOUNT", "account"),
        ("QOS", "qos"),
        ("CONSTRAINT", "constraint"),
    ):
        value = site.get(f"STMEM_{setting}", "gpuq" if setting == "PARTITION" else "")
        if value:
            directives[directive] = value
    gpu_type = site.get("STMEM_GPU_TYPE", "")
    directives["gres"] = f"gpu:{gpu_type + ':' if gpu_type else ''}{gpus}"
    if any("\n" in value or "\r" in value for value in directives.values()):
        raise ValueError("Newline in Slurm directive")
    lines = [
        "#!/usr/bin/env bash",
        *(f"#SBATCH --{key}={shlex.quote(value)}" for key, value in directives.items()),
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo))}",
    ]
    exported = {
        **site,
        "STMEM_REPO_ROOT": str(repo),
        "STMEM_PYTHON": python,
        "STMEM_PROFILE": profile,
        "STMEM_ALLOCATED_GPUS": str(gpus),
    }
    for key, value in sorted(exported.items()):
        if key in WANDB_SITE_VARIABLES and not value:
            lines.append(f"unset {key}")
        else:
            lines.append(f"export {key}={shlex.quote(value)}")
    if site.get("STMEM_ACTIVATE"):
        lines.append(f"source {shlex.quote(site['STMEM_ACTIVATE'])}")
    lines.append("unset WANDB_RUN_ID WANDB_RESUME")
    lines.append(
        shlex.join(
            [
                "exec",
                "bash",
                str(repo / "hpc/jobs/profile_teacher.sbatch"),
                *arguments,
            ]
        )
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Default: render without submission")
    action.add_argument("--submit", action="store_true", help="Render, then submit exactly one job")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site.env")
    parser.add_argument("--profile", choices=[name for name in PROFILES if name != "a5000_debug"])
    parser.add_argument("--config", type=Path, help="Default: configs/teacher.yaml in the site repository")
    parser.add_argument("--name", type=profile_name, help="Unique output name; default: UTC time plus suffix")
    parser.add_argument("--warmup-steps", type=warmup_steps, default=10)
    parser.add_argument("--steps", type=measured_steps, default=5, help="Measured optimizer steps, 1–100")
    parser.add_argument("--trace-steps", type=int, choices=(0, 1), default=1)
    parser.add_argument("--scenario", choices=("representative", "long-history"), default="representative")
    parser.add_argument("--microbatch", type=positive_integer, help="Override per-GPU examples per pass")
    parser.add_argument(
        "--effective-batch", type=positive_integer, help="Override global examples per update"
    )
    parser.add_argument("--walltime", type=walltime, default="02:00:00")
    args = parser.parse_args(argv)
    try:
        site = read_site(args.site)
        repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
        profile = args.profile or site.get("STMEM_PROFILE") or "h200_1gpu"
        if profile not in PROFILES or profile == "a5000_debug":
            raise ValueError("Use an H200/B200 profile here; run local diagnostics directly")
        config = absolute_path(args.config or "configs/teacher.yaml", repo)
        config_hashes(config)  # Validate all inheritance before writing artifacts.
        cfg = resolved_config(config)
        if cfg.get("stage") != "teacher" or cfg.get("data", {}).get("synthetic", False):
            raise ValueError("HPC memory profiling requires a real-data teacher configuration")
        if cfg.get("data", {}).get("split", "train") != "train":
            raise ValueError("Profile released training episodes only, never validation/test seeds")
        if (
            not cfg.get("tracking", {}).get("enabled", True)
            or (site.get("WANDB_MODE") or cfg.get("tracking", {}).get("mode", "offline")) == "disabled"
        ):
            raise ValueError("Profiling requires W&B tracking in online or offline mode")
        for key in ("microbatch", "effective_batch"):
            if getattr(args, key) is not None:
                cfg["training"][key] = getattr(args, key)
        gpus = PROFILES[profile]
        if cfg["training"]["effective_batch"] % gpus:
            raise ValueError("training.effective_batch must be divisible by the GPU count")
        paths = cfg.setdefault("paths", {})
        for env, key, fallback in (
            ("STMEM_DATA_ROOT", "data_root", "data"),
            ("STMEM_CACHE_ROOT", "cache_root", "cache"),
            ("STMEM_RUN_ROOT", "run_root", "runs"),
        ):
            paths[key] = str(absolute_path(site.get(env) or paths.get(key, fallback), repo))
            site[env] = paths[key]
        cfg["profile"] = profile
        name = args.name or (
            f"teacher_{profile}_{args.scenario}_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + secrets.token_hex(3)
        )
        generated = repo / "hpc/generated/profiles" / name
        output = Path(paths["run_root"]) / "profiles" / name
        if output.exists():
            raise FileExistsError(f"Profiling output already exists; choose a new --name: {output}")
        if (generated / "submitted.json").exists():
            raise FileExistsError(f"This named profile was already submitted; choose a new --name: {name}")
        python = site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python")
        logs = repo / "hpc/logs"
        resolved = generated / "config.yaml"
        script = generated / "profile_teacher.sbatch"
        arguments = [
            "--config",
            str(resolved),
            "--output",
            str(output),
            "--warmup-steps",
            str(args.warmup_steps),
            "--steps",
            str(args.steps),
            "--trace-steps",
            str(args.trace_steps),
            "--scenario",
            args.scenario,
        ]
        body = render_script(
            site=site,
            repo=repo,
            logs=logs,
            profile=profile,
            gpus=gpus,
            python=python,
            job_name=name,
            time_limit=args.walltime,
            arguments=arguments,
        )
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    # Slurm opens logs before executing the body; create the directory before sbatch.
    logs.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    script.write_text(body)
    command = [
        "srun",
        "--ntasks=1",
        python,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={gpus}",
        str(repo / "scripts/profile_teacher_memory.py"),
        *arguments,
    ]
    plan = {
        "kind": "teacher-memory-profile",
        "name": name,
        "profile": profile,
        "gpus": gpus,
        "nodes": 1,
        "slurm_tasks": 1,
        "job_count": 1,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "trace_steps": args.trace_steps,
        "scenario": args.scenario,
        "walltime": args.walltime,
        "effective_batch": cfg["training"]["effective_batch"],
        "microbatch": cfg["training"].get("microbatch", 16),
        "precision": cfg["training"]["precision"],
        "tasks": cfg["data"]["tasks"],
        "source_config": str(config),
        "config": str(resolved),
        "script": str(script),
        "output": str(output),
        "command": command,
        "estimated_gpu_hours": "unknown; this job measures timings",
        "max_allocated_gpu_hours": (
            "Requested walltime × allocated GPUs; actual usage depends on measured completion time"
        ),
        "does_not_resume_or_write_teacher_checkpoints": True,
    }
    save_json(generated / "plan.json", plan)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.submit:
        print("Dry-run only. Inspect config.yaml, profile_teacher.sbatch and plan.json.")
        print(f"Use --name {shlex.quote(name)} --submit with the same options to submit this profile.")
        return
    result = subprocess.run(
        ["sbatch", "--parsable", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = result.stdout.strip()
    if not re.fullmatch(r"\d+(?:;[A-Za-z0-9_.-]+)?", raw):
        raise RuntimeError(f"sbatch returned an unexpected job identifier: {raw!r}")
    job_id = raw.split(";", 1)[0]
    save_json(generated / "submitted.json", {"job_id": job_id, "script": str(script), "output": str(output)})
    print(f"Submitted profiling job {job_id}; output: {output}")


if __name__ == "__main__":
    main()
