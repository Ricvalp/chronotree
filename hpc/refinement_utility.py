#!/usr/bin/env python3
"""Prepare utility manifests, bounded diagnostics and independent GPU shards."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hpc.plan import read_site, save_json, sha256, source_hash  # noqa: E402
from hpc.profile_memory import absolute_path, profile_name, walltime  # noqa: E402
from hpc.train_compact import inspect_cache, inspect_hierarchy  # noqa: E402
from stmem.config import digest  # noqa: E402
from stmem.data.hierarchy_cache import default_hierarchy_root  # noqa: E402
from stmem.inspection.refinement_data import (  # noqa: E402
    COMPACT_SHA, ENCODED_ID, HIERARCHY_ID, TEACHER_SHA, TEACHER_REFERENCE,
    primary_students, reference_name, validate_protocol_teacher, verify_manifest,
)


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
    parser.add_argument("--protocol", type=Path, default=REPO / "configs/inspection/refinement_utility_pilot.yaml")
    parser.add_argument("--partition", default="gpuq")
    parser.add_argument("--panel-partition", choices=("train", "validation", "both"), default="train")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--walltime", type=walltime, default="04:00:00")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extension-base", type=Path)
    parser.add_argument("--additional-draws", type=int)
    args = parser.parse_args(argv)
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
    generated.mkdir(parents=True, exist_ok=True)
    with (generated / ".launcher.lock").open("a+") as launcher_lock:
        fcntl.flock(launcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logs = generated / "logs"
        logs.mkdir(exist_ok=True)
        python = site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python")
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
            manifest = verify_manifest(manifest_path)
            args.student = args.student or reference_name(manifest)
            config = repo / "hpc/generated" / f"{args.student_study_name}_s_std" / "compact.yaml"
            if not config.is_file():
                raise ValueError("Render hpc/student_utility.py --dry-run before the sampler audit")
            inputs = {str(config): sha256(config), str(manifest_path): sha256(manifest_path)}
            commands = [[python, "-u", "-m", "stmem.training.utility_sampler_audit", "--config", str(config),
                         "--count", "4096", "--output", str(output / "sampler_audit")]]
        elif args.stage == "report":
            manifest = verify_manifest(manifest_path, verify_source=False)
            args.student = args.student or reference_name(manifest)
            commands = [base + ["--mode", "report"]]
        else:
            manifest = verify_manifest(manifest_path)
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
            inputs[str(checkpoint)] = sha256(checkpoint)
            mode = {"smoke": "smoke", "measure": "main", "extension": "extension"}[args.stage]
            commands = [base + ["--mode", mode,
                               "--student", args.student, "--checkpoint", str(checkpoint),
                               "--partition", args.panel_partition, "--workers", str(count),
                               "--max-hours", "3.8"]]
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
        plan = {"commands": commands, "inputs": inputs, "source_sha256": source_hash(repo),
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
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError("Saved plan identity differs; choose a new name or restore its exact inputs")
        save_json(plan_path, plan)
        lines = ["#!/usr/bin/env bash", f"#SBATCH --job-name=stmem_utility_{suffix}",
                 "#SBATCH --nodes=1", "#SBATCH --ntasks=1", "#SBATCH --cpus-per-task=8", "#SBATCH --mem=64G",
                 "#SBATCH --gres=gpu:1", f"#SBATCH --partition={shlex.quote(args.partition)}",
                 f"#SBATCH --time={args.walltime}", f"#SBATCH --output={shlex.quote(str(logs / (suffix + '_%A_%a.out')))}",
                 f"#SBATCH --error={shlex.quote(str(logs / (suffix + '_%A_%a.err')))}", "#SBATCH --signal=B:TERM@180"]
        if count > 1:
            lines.append(f"#SBATCH --array=0-{count-1}%{count}")
        for key in ("ACCOUNT", "QOS", "CONSTRAINT"):
            if site.get("STMEM_" + key):
                lines.append(f"#SBATCH --{key.lower()}={shlex.quote(site['STMEM_' + key])}")
        lines += ["set -euo pipefail", f"cd {shlex.quote(str(repo))}",
                  "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
                  "export WANDB_PROJECT=chronotree-routers", f"export PYTHONPATH={shlex.quote(str(repo / 'src'))}",
                  "export STMEM_SITE_FILE=/dev/null",
                  "unset STMEM_DATA_ROOT STMEM_CACHE_ROOT STMEM_RUN_ROOT RANK LOCAL_RANK WORLD_SIZE"]
        if site.get("STMEM_ACTIVATE"):
            lines.append(f"source {shlex.quote(site['STMEM_ACTIVATE'])}")
        guard = ("import json; from pathlib import Path; from hpc.plan import source_hash,sha256; "
                 f"p=json.load(open({str(plan_path)!r})); "
                 "assert source_hash(Path(p['repo']))==p['source_sha256'],'Source changed'; "
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
                active = subprocess.run(["squeue", "-h", "-j", previous, "-o", "%A"], capture_output=True, text=True)
                if active.returncode or active.stdout.strip():
                    raise ValueError("Cannot establish that the previous array is inactive")
            submitted = subprocess.run(["sbatch", "--parsable", str(script)], check=True, capture_output=True, text=True)
            save_json(receipt, {"job_id": submitted.stdout.strip().split(";")[0], "plan_sha256": sha256(plan_path)})
        print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
