"""Small shared launcher contracts for the expert-utility follow-up jobs.

Scientific manifests belong to the runtime modules. This file handles only
immutable launch plans, source snapshots, resource rendering and Slurm receipts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from hpc.plan import WANDB_SITE_VARIABLES, active_user_job_ids, save_json, sha256, source_hash
from hpc.utility_source import create_study_source
from stmem.config import digest


def read_json(path):
    return json.loads(Path(path).read_text())


def immutable_json(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Immutable launch identity changed: {path}; choose a new --name")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, value)


def validate_resources(partition, gpu_type, exclude=""):
    for name, value in (("partition", partition), ("GPU type", gpu_type)):
        if not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError(f"Invalid {name}; this launcher requires a literal typed GPU request")
    if exclude and not re.fullmatch(r"[A-Za-z0-9_.-]+(?:,[A-Za-z0-9_.-]+)*", exclude):
        raise ValueError("--exclude requires comma-separated literal node names")


def current_source_snapshot(repo, generated):
    """Freeze committed current code; never replace an older study's source.

    A normal push/pull includes the new implementation in HEAD. Uncommitted
    Python edits cannot silently become another revision's scientific runtime.
    Documentation-only dirtiness is harmless because the complete source hash
    still has to match the exported commit.
    """
    repo, generated = Path(repo), Path(generated)
    revision = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
    body = {"schema": "utility_launcher_source_request_v1",
            "source": {"git": revision, "source_sha256": source_hash(repo)}}
    body["experiment_id"] = digest(body)
    request = generated / "runtime_source_request.json"
    immutable_json(request, body)
    return create_study_source(repo, request)


def runtime_environment(code_root, *, cpu=False):
    root = Path(code_root)
    env = {**os.environ, "PYTHONPATH": str(root / "src"),
           "GIT_CEILING_DIRECTORIES": str(root.parent), "STMEM_SITE_FILE": "/dev/null"}
    for key in ("STMEM_DATA_ROOT", "STMEM_CACHE_ROOT", "STMEM_RUN_ROOT", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
        env.pop(key, None)
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def source_inputs(code_root):
    root = Path(code_root)
    # Guard launch-time administrative code as well as the scientific src hash.
    paths = [root / ".utility-source.json", *sorted((root / "hpc").glob("*.py")),
             *sorted((root / "hpc").glob("*.sh"))]
    return {str(path): sha256(path) for path in paths if path.is_file()}


def render_job(generated, key, *, repo, code_root, python, site, commands, inputs,
               partition="aiq", gpu_type="b200", cpus=8, mem="64G", walltime="04:00:00",
               workers=1, max_concurrent=None, exclude="", dependencies=(), outputs=()):
    """Render literal Slurm directives and guard both code and fixed inputs."""
    validate_resources(partition, gpu_type, exclude)
    if cpus < 1 or workers < 1 or not re.fullmatch(r"[1-9][0-9]*[KMGTP]?", mem):
        raise ValueError("Positive CPU/worker counts and a valid memory quantity are required")
    maximum = workers if max_concurrent is None else max_concurrent
    if not 1 <= maximum <= workers:
        raise ValueError("Concurrency must be between one and the number of workers")
    root, generated = Path(code_root).resolve(), Path(generated).resolve()
    logs = generated / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    script, plan_path = generated / f"{key}.sbatch", generated / f"{key}.json"
    plan = {"schema": "utility_followup_job_v1", "key": key, "repo": str(repo),
            "code_root": str(root), "source_sha256": source_hash(root), "python": python,
            "commands": commands, "inputs": {**source_inputs(root), **inputs},
            "resources": {"partition": partition, "gpu_type": gpu_type, "gpus": 1,
                          "cpus": cpus, "mem": mem, "walltime": walltime, "exclude": exclude,
                          "workers": workers, "max_concurrent": maximum},
            "script": str(script), "logs": str(logs), "dependencies": list(dependencies),
            "outputs": list(map(str, outputs)), "estimated_gpu_hours": "unknown; measure these jobs"}
    immutable_json(plan_path, plan)
    directives = {"job-name": f"stmem_{key}", "nodes": "1", "ntasks": "1",
                  "cpus-per-task": str(cpus), "mem": mem, "partition": partition,
                  "gres": f"gpu:{gpu_type}:1", "time": walltime,
                  "output": str(logs / f"{key}_%A_%a.out"),
                  "error": str(logs / f"{key}_%A_%a.err")}
    if workers > 1:
        directives["array"] = f"0-{workers - 1}%{maximum}"
    if exclude:
        directives["exclude"] = exclude
    for key_name in ("ACCOUNT", "QOS", "CONSTRAINT"):
        if site.get("STMEM_" + key_name):
            directives[key_name.lower()] = site["STMEM_" + key_name]
    if any("\n" in value or "\r" in value for value in directives.values()):
        raise ValueError("Newline in Slurm directive")
    lines = ["#!/usr/bin/env bash", *(f"#SBATCH --{name}={shlex.quote(value)}" for name, value in directives.items()),
             "set -euo pipefail", f"cd {shlex.quote(str(root))}"]
    if site.get("STMEM_ACTIVATE"):
        lines.append(f"source {shlex.quote(site['STMEM_ACTIVATE'])}")
    exports = {"PYTHONPATH": str(root / "src"), "GIT_CEILING_DIRECTORIES": str(root.parent),
               "STMEM_SITE_FILE": "/dev/null", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
               **{name: site[name] for name in WANDB_SITE_VARIABLES if name in site},
               "WANDB_PROJECT": "chronotree-routers"}
    for name, value in sorted(exports.items()):
        lines.append(f"export {name}={shlex.quote(str(value))}" if value else f"unset {name}")
    lines.append("unset STMEM_DATA_ROOT STMEM_CACHE_ROOT STMEM_RUN_ROOT RANK LOCAL_RANK WORLD_SIZE WANDB_RUN_ID WANDB_RESUME")
    guard = ("import json; from pathlib import Path; from hpc.plan import source_hash,sha256; "
             f"p=json.load(open({str(plan_path)!r})); "
             "assert source_hash(Path(p['code_root']))==p['source_sha256'],'Source changed'; "
             "assert all(sha256(Path(k))==v for k,v in p['inputs'].items()),'Frozen inputs changed'")
    lines.append(shlex.join([python, "-c", guard]))
    for index, command in enumerate(commands):
        # The one deliberately dynamic argument is a decimal Slurm array index.
        rendered = " ".join('"${SLURM_ARRAY_TASK_ID:-0}"' if str(value) == "@WORKER@"
                            else shlex.quote(str(value)) for value in command)
        lines.append(("exec " if index == len(commands) - 1 else "") + rendered)
    text = "\n".join(lines) + "\n"
    if script.exists() and script.read_text() != text:
        raise ValueError("Rendered batch script changed; choose a new launch name")
    script.write_text(text)
    subprocess.run(["bash", "-n", str(script)], check=True)
    return plan_path, plan


def submit_job(plan_path, *, resume=False, after=(), active=None):
    """One explicit submission; never launch a duplicate active allocation."""
    plan_path = Path(plan_path)
    plan = read_json(plan_path)
    receipt_path = plan_path.with_suffix(".submitted.json")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt.get("plan_sha256") != sha256(plan_path):
            raise ValueError("Submission receipt refers to another plan")
        if not resume:
            raise ValueError(f"{plan['key']} was already submitted; use --resume after checking its status")
        live = active_user_job_ids() if active is None else active
        if receipt["job_id"] in live:
            return receipt["job_id"]
    command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
    if after:
        if any(not re.fullmatch(r"[0-9]+", str(job)) for job in after):
            raise ValueError("Dependencies must be Slurm allocation IDs")
        command += ["--dependency=afterok:" + ":".join(map(str, after))]
    command.append(plan["script"])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    if not re.fullmatch(r"[0-9]+", job_id):
        raise RuntimeError("Slurm returned an unrecognized job ID; reconcile submission before retrying")
    save_json(receipt_path, {"job_id": job_id, "plan_sha256": sha256(plan_path), "dependency_ids": list(after)})
    return job_id
