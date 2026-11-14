#!/usr/bin/env python3
"""Prepare utility manifests, bounded diagnostics and independent GPU shards."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import active_user_job_ids, read_site, save_json, sha256, source_hash  # noqa: E402
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from hpc.train_compact import inspect_cache, inspect_hierarchy  # noqa: E402
from stmem.config import digest  # noqa: E402
from stmem.data.hierarchy_cache import default_hierarchy_root  # noqa: E402
from stmem.inspection.refinement_data import (  # noqa: E402
    COMPACT_SHA, ENCODED_ID, HIERARCHY_ID, TEACHER_SHA, TEACHER_REFERENCE,
    primary_students, reference_name, validate_protocol_teacher, verify_manifest,
)


def runtime_manifest(python, code_root, manifest_path, *, verify_source=True):
    """Validate with the pinned runtime, never reinterpret an older study here."""
    command = (
        "import json,sys; from stmem.inspection.refinement_data import verify_manifest; "
        "print(json.dumps(verify_manifest(sys.argv[1], verify_source=sys.argv[2]=='1')))"
    )
    result = subprocess.run(
        [python, "-c", command, str(manifest_path), "1" if verify_source else "0"],
        cwd=code_root, env={**os.environ, "PYTHONPATH": str(code_root / "src"), "CUDA_VISIBLE_DEVICES": "",
                            "GIT_CEILING_DIRECTORIES": str(code_root.parent)},
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def validate_runtime_checkpoint(python, code_root, manifest_path, student, checkpoint, partition):
    """Reject missing/mismatched endpoints on the login CPU before allocating GPUs."""
    command = (
        "import sys; from pathlib import Path; "
        "from stmem.inspection.refinement_data import verify_manifest; "
        "from stmem.inspection.refinement_utility import validate_student_endpoint; "
        "from stmem.training.runtime import load_checkpoint; "
        "p=Path(sys.argv[1]); validate_student_endpoint(verify_manifest(p),p,sys.argv[2],"
        "load_checkpoint(sys.argv[3]),sys.argv[4])"
    )
    subprocess.run(
        [python, "-c", command, str(manifest_path), student, str(checkpoint), partition],
        cwd=code_root, env={**os.environ, "PYTHONPATH": str(code_root / "src"), "CUDA_VISIBLE_DEVICES": "",
                            "GIT_CEILING_DIRECTORIES": str(code_root.parent)},
        check=True,
    )


def measurement_hours(value):
    """Leave 12 minutes for startup/shutdown; workers stop between complete states."""
    days, clock = value.split("-", 1) if "-" in value else ("0", value)
    hours, minutes, seconds = map(int, clock.split(":"))
    remaining = 24 * int(days) + hours + minutes / 60 + seconds / 3600 - .2
    if remaining <= 0:
        raise ValueError("Utility measurement walltime must exceed 12 minutes")
    return format(remaining, ".8g")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site_b200_4gpu.env")
    parser.add_argument("--cache-name", type=profile_name, default="memory_fv_seed0")
    parser.add_argument("--name", type=profile_name, default="router_utility_v1")
    parser.add_argument("--stage", choices=("prepare", "audit", "smoke", "measure", "extension", "report"), default="prepare")
    parser.add_argument("--student-study-name", type=profile_name, default="utility_students_v1")
    parser.add_argument("--student", choices=("S0", "S_INIT", "S_STD", "S_STAGE", "S_PACKET"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compact", type=Path)
    parser.add_argument("--teacher-reference", action="store_true",
                        help="New adopted-teacher study: S_INIT smoke, three fresh primary students; no old S0")
    parser.add_argument("--manifest", type=Path)
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--study-source", action="store_true",
                         help="Restore and verify the manifest's exact committed code in an ignored snapshot")
    runtime.add_argument("--code-root", type=Path,
                         help="Existing runtime checkout matching the unchanged study manifest")
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/inspection/refinement_utility_pilot.yaml")
    parser.add_argument("--partition", default="gpuq")
    parser.add_argument("--gpu-type", default="", help="GPU GRES type, e.g. b200; empty retains gpu:1")
    parser.add_argument("--panel-partition", choices=("train", "validation", "both"), default="train")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--walltime", type=walltime, default="04:00:00")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extension-base", type=Path)
    parser.add_argument("--additional-draws", type=int)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.partition):
        parser.error("Invalid --partition")
    if args.gpu_type and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.gpu_type):
        parser.error("Invalid --gpu-type")
    if args.stage == "prepare" and (args.study_source or args.code_root):
        parser.error("Runtime source options reuse an existing study; they cannot prepare a new panel")
    if args.teacher_reference and args.stage != "prepare":
        parser.error("--teacher-reference is recorded by --stage prepare; later stages read the manifest")
    if args.teacher_reference and args.compact is not None:
        parser.error("--teacher-reference uses the adopted teacher itself; do not supply --compact")
    if not 1 <= args.shards <= 16:
        parser.error("Use 1..16 bounded independent shards")
    if args.stage == "extension":
        if args.extension_base is None or args.additional_draws is None or not 1 <= args.additional_draws <= 16:
            parser.error("Extension needs --extension-base and --additional-draws in 1..16")
    elif args.extension_base is not None or args.additional_draws is not None:
        parser.error("Extension options belong only to --stage extension")
    site = read_site(args.site)
    repo = absolute_path(site.get("STMEM_REPO_ROOT", REPO), REPO)
    generated = repo / "hpc/generated" / args.name
    output = repo / "runs" / args.name
    manifest_path = absolute_path(args.manifest or output / "manifest.json", repo)
    code_root = absolute_path(args.code_root, repo) if args.code_root else repo
    if args.study_source:
        from hpc.utility_source import create_study_source

        code_root = create_study_source(repo, manifest_path)
    pinned_runtime = args.study_source or args.code_root is not None
    generated.mkdir(parents=True, exist_ok=True)
    with (generated / ".launcher.lock").open("a+") as launcher_lock:
        fcntl.flock(launcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logs = generated / "logs"
        logs.mkdir(exist_ok=True)
        python = site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python")
        def read_manifest(*, verify_source=True):
            if pinned_runtime:
                return runtime_manifest(python, code_root, manifest_path, verify_source=verify_source)
            return (verify_manifest(manifest_path) if verify_source
                    else verify_manifest(manifest_path, verify_source=False))
        base = [python, "-u", "-m", "stmem.inspection.refinement_utility", "--manifest", str(manifest_path)]
        count = args.shards if args.stage in ("measure", "extension") else 1
        commands, inputs = [], {}
        if args.stage == "prepare":
            cache_plan_path = repo / "hpc/generated" / args.cache_name / "plan.json"
            cfg, cache_plan, leaf = inspect_cache(cache_plan_path)
            encoded = Path(cfg["paths"]["cache_root"]) / digest(leaf["identity"])[:24]
            hierarchy = default_hierarchy_root(encoded, leaf) if args.teacher_reference else encoded / "hierarchy" / HIERARCHY_ID
            parents = inspect_hierarchy(encoded, hierarchy)
            if not args.teacher_reference and (cache_plan["checkpoint_sha256"] != TEACHER_SHA or digest(leaf["identity"])[:24] != ENCODED_ID
                    or digest(parents["identity"])[:24] != HIERARCHY_ID):
                raise ValueError("Accepted immutable cache dependencies differ")
            compact = (Path(cache_plan["checkpoint"]) if args.teacher_reference else absolute_path(
                args.compact or repo / "hpc/generated/memory_fv_seed0_routers/frozen/compact.pt", repo))
            if not args.teacher_reference and sha256(compact) != COMPACT_SHA:
                raise ValueError("S0 must be the exact preserved production router-study compact checkpoint")
            config = generated / "data.yaml"
            if config.exists() and yaml.safe_load(config.read_text()) != cfg:
                raise ValueError("Study data configuration changed")
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            protocol_source = absolute_path(args.protocol, repo)
            if args.teacher_reference:
                validate_protocol_teacher(yaml.safe_load(protocol_source.read_text()), {
                    "teacher_sha256": cache_plan["checkpoint_sha256"],
                    "teacher_step": cache_plan["checkpoint_step"], "reference": TEACHER_REFERENCE})
            protocol = generated / "protocol.yaml"
            if protocol.exists() and protocol.read_bytes() != protocol_source.read_bytes():
                raise ValueError("Pinned utility protocol changed; use a new study name")
            if not protocol.exists():
                protocol.write_bytes(protocol_source.read_bytes())
            inputs = {str(protocol): sha256(protocol), str(config): sha256(config), str(compact): sha256(compact),
                      str(cache_plan["checkpoint"]): cache_plan["checkpoint_sha256"]}
            commands = [base + ["--mode", "manifest", "--config", str(protocol), "--data-config", str(config),
                               "--teacher", cache_plan["checkpoint"], "--compact", str(compact),
                               "--encoded", str(encoded), "--hierarchy", str(hierarchy)],
                        base + ["--mode", "contexts"]]
            if args.teacher_reference:
                commands[0] += ["--teacher-reference", "--cache-plan", str(cache_plan_path)]
                inputs[str(cache_plan_path)] = sha256(cache_plan_path)
                inputs[str(cache_plan["selection"])] = cache_plan["selection_sha256"]
            args.student = args.student or ("S_INIT" if args.teacher_reference else "S0")
        elif args.stage == "audit":
            manifest = read_manifest()
            args.student = args.student or reference_name(manifest)
            config = repo / "hpc/generated" / f"{args.student_study_name}_s_std" / "compact.yaml"
            if not config.is_file():
                raise ValueError("Render hpc/student_utility.py --dry-run before the sampler audit")
            inputs = {str(config): sha256(config), str(manifest_path): sha256(manifest_path)}
            commands = [[python, "-u", "-m", "stmem.training.utility_sampler_audit", "--config", str(config),
                         "--count", "4096", "--output", str(output / "sampler_audit")]]
        elif args.stage == "report":
            manifest = read_manifest(verify_source=False)
            args.student = args.student or reference_name(manifest)
            commands = [base + ["--mode", "report"]]
        else:
            manifest = read_manifest()
            args.student = args.student or reference_name(manifest)
            if args.student == "S_INIT" and (reference_name(manifest) != "S_INIT"
                                             or args.stage != "smoke" or args.panel_partition != "train"):
                parser.error("S_INIT is only a training-partition smoke diagnostic for its new teacher study")
            if args.student not in (*primary_students(manifest), reference_name(manifest)):
                parser.error("Student is not part of this study; old S0 must remain in its original study")
            inputs[str(manifest_path)] = sha256(manifest_path)
            checkpoint = absolute_path(args.checkpoint or manifest["dependencies"]["compact"], repo)
            if args.student != reference_name(manifest) and args.checkpoint is None:
                parser.error("A new student measurement requires its fixed endpoint --checkpoint")
            if pinned_runtime:
                validate_runtime_checkpoint(python, code_root, manifest_path, args.student,
                                            checkpoint, args.panel_partition)
            inputs[str(checkpoint)] = sha256(checkpoint)
            mode = {"smoke": "smoke", "measure": "main", "extension": "extension"}[args.stage]
            commands = [base + ["--mode", mode,
                               "--student", args.student, "--checkpoint", str(checkpoint),
                               "--partition", args.panel_partition, "--workers", str(count),
                               "--max-hours", measurement_hours(args.walltime)]]
            if args.stage == "extension":
                extension_base = absolute_path(args.extension_base, repo)
                commands[0] += ["--extension-base", str(extension_base),
                                "--additional-draws", str(args.additional_draws)]
                if args.additional_draws > int(manifest["protocol"].get("additional_draw_cap", 16)):
                    parser.error("Additional draws exceed the fixed study protocol cap")
                inputs[str(extension_base / "identity.json")] = sha256(extension_base / "identity.json")
                for state in manifest["states"]:
                    metadata = extension_base / f"{state['state_id']}.json"
                    if metadata.is_file():
                        inputs[str(metadata)] = sha256(metadata)
        suffix = f"{args.stage}_{args.student}_{args.panel_partition}"
        if args.stage in ("smoke", "measure", "extension"):
            suffix += "_" + inputs[str(checkpoint)][:12]
        if args.stage == "extension":
            suffix += f"_draws{args.additional_draws:02d}"
        plan_path = generated / f"{suffix}.json"
        script = generated / f"{suffix}.sbatch"
        study_protocol = (yaml.safe_load(protocol.read_text()) if args.stage == "prepare"
                          else manifest.get("protocol", {}))
        student_steps = study_protocol.get("student_training", {}).get("steps", 30000)
        plan = {"commands": commands, "inputs": inputs, "source_sha256": source_hash(code_root),
                "repo": str(repo), "manifest": str(manifest_path), "stage": args.stage,
                "student": args.student, "shards": count, "gpus_per_shard": 1, "cpus_per_shard": 8,
                "mem": "64G", "walltime": args.walltime, "partition": args.partition,
                "script": str(script), "logs": str(logs), "estimated_gpu_hours": "unknown until smoke",
                "dependency_graph": {"prepare": [],
                    ("S_INIT" if (args.teacher_reference if args.stage == "prepare"
                                  else reference_name(manifest) == "S_INIT") else "S0"): ["prepare", "smoke"],
                    "S_STD": ["prepare", "sampler_audit", f"training_{student_steps}"],
                    "S_STAGE": ["prepare", "sampler_audit", f"training_{student_steps}"],
                    "S_PACKET": ["prepare", "sampler_audit", f"training_{student_steps}"],
                    "report": ["all_available_frozen_measurements"],
                    "conditional_router_fit": ["development_findings", "separate_training_pool_labels"]}}
        if args.gpu_type:
            plan["gpu_type"] = args.gpu_type
            plan["gres"] = f"gpu:{args.gpu_type}:1"
        if pinned_runtime:
            plan["code_root"] = str(code_root)
            plan["planner"] = str(Path(__file__).resolve())
            plan["planner_sha256"] = sha256(Path(__file__))
            plan["runtime_hpc"] = {str(code_root / relative): sha256(code_root / relative)
                                   for relative in ("hpc/plan.py", "hpc/profile_memory.py")}
            receipt = code_root / ".utility-source.json"
            if receipt.exists():
                plan["runtime_hpc"][str(receipt)] = sha256(receipt)
            if args.study_source:
                plan["source_snapshot_helper_sha256"] = sha256(REPO / "hpc/utility_source.py")
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Saved plan identity differs; choose a new name or restore its exact inputs")
        save_json(plan_path, plan)
        lines = ["#!/usr/bin/env bash", f"#SBATCH --job-name=stmem_utility_{suffix}",
                 "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=64G",
                 f"#SBATCH --gres={plan.get('gres', 'gpu:1')}", f"#SBATCH --partition={shlex.quote(args.partition)}",
                 f"#SBATCH --time={args.walltime}", f"#SBATCH --output={shlex.quote(str(logs / (suffix + '_%A_%a.out')))}",
                 f"#SBATCH --error={shlex.quote(str(logs / (suffix + '_%A_%a.err')))}", "#SBATCH --signal=B:TERM@180"]
        if count > 1:
            lines.append(f"#SBATCH --array=0-{count-1}%{count}")
        for key in ("ACCOUNT", "QOS", "CONSTRAINT"):
            if site.get("STMEM_" + key):
                lines.append(f"#SBATCH --{key.lower()}={shlex.quote(site['STMEM_' + key])}")
        lines += ["set -euo pipefail", f"cd {shlex.quote(str(code_root))}"]
        if site.get("STMEM_ACTIVATE"):
            lines.append(f"source {shlex.quote(site['STMEM_ACTIVATE'])}")
        lines += ["export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                  "export WANDB_PROJECT=chronotree-routers", f"export PYTHONPATH={shlex.quote(str(code_root / 'src'))}",
                  "export STMEM_SITE_FILE=/dev/null",
                  "unset STMEM_DATA_ROOT STMEM_CACHE_ROOT STMEM_RUN_ROOT RANK LOCAL_RANK WORLD_SIZE"]
        if pinned_runtime:
            lines.append("export GIT_CEILING_DIRECTORIES=" + shlex.quote(str(code_root.parent)))
        guard = ("import json; from pathlib import Path; from hpc.plan import source_hash,sha256; "
                 f"p=json.load(open({str(plan_path)!r})); "
                 "assert source_hash(Path(p.get('code_root',p['repo'])))==p['source_sha256'],'Source changed'; "
                 "assert all(sha256(Path(k))==v for k,v in p.get('runtime_hpc',{}).items()),'Runtime launcher changed'; "
                 "assert all(sha256(Path(k))==v for k,v in p['inputs'].items()),'Inputs changed'")
        lines.append(shlex.join([python, "-c", guard]))
        for i, command in enumerate(commands):
            line = shlex.join(command)
            if args.stage in ("measure", "extension"):
                line += ' --worker-index "${SLURM_ARRAY_TASK_ID:-0}"'
            if i == len(commands) - 1:
                line = "exec " + line
            lines.append(line)
        script.write_text("\n".join(lines) + "\n")
        subprocess.run(["bash", "-n", str(script)], check=True)
        receipt = plan_path.with_suffix(".submitted.json")
        if args.submit:
            if receipt.exists() and not args.resume:
                raise ValueError("Plan already submitted; --resume is required after its allocations are inactive")
            if receipt.exists():
                previous = json.loads(receipt.read_text())["job_id"]
                if previous in active_user_job_ids():
                    raise ValueError("Previous utility array is still active")
            submitted = subprocess.run(["sbatch", "--parsable", str(script)], check=True, capture_output=True, text=True)
            save_json(receipt, {"job_id": submitted.stdout.strip().split(";")[0], "plan_sha256": sha256(plan_path)})
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
