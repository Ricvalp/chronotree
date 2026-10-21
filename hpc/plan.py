#!/usr/bin/env python3
"""Render an explicit bounded Slurm DAG. Submission requires --submit."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
PROFILES = {"a5000_debug": 1, "h200_1gpu": 1, "h200_4gpu": 4, "b200_1gpu": 1, "b200_8gpu": 8}
# Only nonsecret W&B settings belong in rendered scripts or site.env. Credentials
# remain in the account's W&B login store or its external runtime environment.
WANDB_SITE_VARIABLES = frozenset({"WANDB_MODE", "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_DIR"})


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_seconds(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number of seconds")
    return parsed


def scientific_training_command(command: list[str]) -> list[str]:
    """Remove only validated runtime logging flag/value pairs from identity."""
    if command[1:3] != ["-m", "stmem.train"]:
        return command
    validators = {"--log-every": positive_integer, "--progress-every-seconds": positive_seconds}
    result = []
    index = 0
    while index < len(command):
        validator = validators.get(command[index])
        if validator is not None and index + 1 < len(command):
            try:
                validator(command[index + 1])
            except argparse.ArgumentTypeError:
                pass
            else:
                index += 2
                continue
        result.append(command[index])
        index += 1
    return result


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def config_hashes(path: Path, seen: set[Path] | None = None) -> dict[str, str]:
    seen = set() if seen is None else seen
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    seen.add(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration {path}")
    result = {str(path): sha256(path)}
    parent = yaml.safe_load(path.read_text()).get("extends")
    if parent:
        result.update(config_hashes(path.parent / parent, seen))
    return result


def resolved_config(path: Path) -> dict:
    """Resolve inheritance for portable stage identity, independent of plan folder."""
    current = yaml.safe_load(path.read_text())
    parent = current.pop("extends", None)
    base = resolved_config((path.parent / parent).resolve()) if parent else {}

    def merge(left: dict, right: dict) -> dict:
        result = dict(left)
        for key, value in right.items():
            result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
        return result

    return merge(base, current)


def source_hash(repo: Path) -> str:
    value = hashlib.sha256()
    for source in sorted((repo / "src").rglob("*.py")):
        value.update(str(source.relative_to(repo)).encode())
        value.update(source.read_bytes())
    return value.hexdigest()


def read_site(path: Path) -> dict[str, str]:
    """Parse literal site assignments, never execute an env file in the planner."""
    values = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("STMEM_") or key in WANDB_SITE_VARIABLES
    }
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            words = shlex.split(line, comments=True)
            if not words:
                continue
            if words[0] == "export":
                words.pop(0)
            if len(words) != 1 or "=" not in words[0]:
                raise ValueError(f"{path}:{number}: expected one literal supported NAME=value assignment")
            key, value = words[0].split("=", 1)
            if not re.fullmatch(r"STMEM_[A-Z0-9_]+", key) and key not in WANDB_SITE_VARIABLES:
                raise ValueError(f"{path}:{number}: unsupported setting {key}")
            if "$" in value or "`" in value:
                raise ValueError("site.env uses literal values; shell substitutions are not evaluated")
            values[key] = value
    if values.get("WANDB_MODE") not in (None, "", "offline", "online", "disabled"):
        raise ValueError("WANDB_MODE must be offline, online, or disabled")
    return values


@dataclass
class Job:
    name: str
    stage: str
    commands: list[list[str]]
    deps: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    gpus: int = 1
    walltime: str = "30:00:00"
    seed: int = 0
    marker: str = ""
    script: str = ""
    signature: str = ""


def build_jobs(
    manifest: dict[str, Any],
    *,
    repo: Path,
    run_root: Path,
    generated: Path,
    profile: str,
    python: str,
    phase: str,
    walltime: str,
    eval_walltime: str,
    log_every: int | None = None,
    progress_every_seconds: float | None = None,
    teacher_config: Path | None = None,
) -> list[Job]:
    jobs: list[Job] = []
    gpu_count = PROFILES[profile]
    stage_configs: dict[tuple[int, str], str] = {}
    base_prefix = [python, "-m"]

    def config(seed: int, name: str, changes: dict | None = None) -> str:
        key = seed, name
        if key not in stage_configs:
            base = name.replace("_initial", "").replace("_final", "")
            target = generated / "configs" / f"seed{seed}_{name}.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            obj = {
                "extends": str(
                    teacher_config
                    if base == "teacher" and teacher_config
                    else repo / "configs" / f"{base}.yaml"
                ),
                "seed": seed,
                "profile": profile,
                "paths": {"run_root": str(run_root)},
            }
            if changes:
                obj.update(changes)
            target.write_text(yaml.safe_dump(obj, sort_keys=False))
            stage_configs[key] = str(target)
        return stage_configs[key]

    def add(
        seed: int,
        short: str,
        stage: str,
        command: list[str] | list[list[str]],
        deps=(),
        inputs=(),
        outputs=(),
        gpus: int | None = None,
    ) -> str:
        name = f"seed{seed}_{short}"
        commands = command if command and isinstance(command[0], list) else [command]
        if stage in ("teacher", "compact", "perceiver", "router"):
            commands = [list(item) for item in commands]
            for item in commands:
                if log_every is not None:
                    item += ["--log-every", str(log_every)]
                if progress_every_seconds is not None and stage != "router":
                    item += ["--progress-every-seconds", str(progress_every_seconds)]
        jobs.append(
            Job(
                name=name,
                stage=stage,
                commands=commands,
                deps=list(deps),
                inputs=list(map(str, inputs)),
                outputs=list(map(str, outputs)),
                gpus=(gpu_count if stage in ("teacher", "compact", "perceiver") else 1)
                if gpus is None
                else gpus,
                walltime=eval_walltime if stage == "evaluate" else walltime,
                seed=seed,
            )
        )
        return name

    def evaluate(
        seed: int,
        name: str,
        method: str,
        checkpoint: Path,
        config_path: str,
        dependencies: list[str],
        output: Path,
        *,
        episode_count: int,
        split: str = "val",
        family: str | None = None,
        router: Path | None = None,
        shard: int = 0,
        shards: int = 1,
    ) -> str:
        command = base_prefix + [
            "stmem.evaluate",
            "--config",
            config_path,
            "--split",
            split,
            "--method",
            method,
            "--checkpoint",
            str(checkpoint),
            "--episode-count",
            str(episode_count),
            "--output",
            str(output),
            "--budget",
            str(manifest["budget"]),
            "--shard-index",
            str(shard),
            "--num-shards",
            str(shards),
        ]
        inputs = [checkpoint]
        if router:
            command += ["--router-checkpoint", str(router)]
            inputs.append(router)
        if family:
            command += ["--selection-family", family]
        return add(seed, name, "evaluate", command, dependencies, inputs, [output / "complete.json"], gpus=1)

    for seed in manifest["seeds"]:
        root = run_root / f"seed{seed}"
        teacher, compact = root / "teacher", root / "compact"
        t_cfg, c_cfg = config(seed, "teacher"), config(seed, "compact")
        teacher_selected, compact_selected = teacher / "selected.pt", compact / "selected.pt"
        selected_deps: dict[str, list[str]] = {}
        if phase == "validation":
            t_candidates = [teacher / f"step_{step:08d}.pt" for step in manifest["teacher_candidates"]]
            teacher_job = add(
                seed,
                "teacher",
                "teacher",
                base_prefix + ["stmem.train", "--stage", "teacher", "--config", t_cfg],
                outputs=t_candidates,
            )
            t_evals = []
            t_paths = []
            selection_count = len(manifest["checkpoint_selection"]["episode_indices"])
            for step, checkpoint in zip(manifest["teacher_candidates"], t_candidates):
                output = teacher / "selection" / str(step)
                t_paths.append(output)
                t_evals.append(
                    evaluate(
                        seed,
                        f"teacher_val_{step}",
                        "FULL",
                        checkpoint,
                        t_cfg,
                        [teacher_job],
                        output,
                        episode_count=selection_count,
                    )
                )
            t_select = add(
                seed,
                "select_teacher",
                "select",
                base_prefix
                + [
                    "stmem.training.select_checkpoint",
                    "--candidates",
                    *map(str, t_candidates),
                    "--evaluations",
                    *map(str, t_paths),
                    "--output",
                    str(teacher_selected),
                ],
                t_evals,
                t_candidates + t_paths,
                [teacher_selected],
                gpus=0,
            )
            cache_job = add(
                seed,
                "cache",
                "cache",
                base_prefix + ["stmem.data.cache", "--config", t_cfg, "--checkpoint", str(teacher_selected)],
                [t_select],
                [teacher_selected],
                [root / "cache" / "complete.json"],
            )
            c_candidates = [compact / f"step_{step:08d}.pt" for step in manifest["compact_candidates"]]
            c_job = add(
                seed,
                "compact",
                "compact",
                base_prefix + ["stmem.train", "--stage", "compact", "--config", c_cfg],
                [cache_job, t_select],
                [teacher_selected],
                c_candidates,
            )
            c_evals, c_paths = [], []
            # Selection utility averages both families at both candidate steps.
            # Each candidate directory contains the tree and flat subdirectories.
            for step, checkpoint in zip(manifest["compact_candidates"], c_candidates):
                candidate_path = compact / "selection" / str(step)
                for family in manifest["checkpoint_selection"]["compact_families"]:
                    c_paths.append(candidate_path / family)
                    c_evals.append(
                        evaluate(
                            seed,
                            f"compact_val_{step}_{family}",
                            family.upper(),
                            checkpoint,
                            c_cfg,
                            [c_job],
                            candidate_path / family,
                            episode_count=selection_count,
                            family=family,
                        )
                    )
            c_select = add(
                seed,
                "select_compact",
                "select",
                base_prefix
                + [
                    "stmem.training.select_checkpoint",
                    "--candidates",
                    *map(str, c_candidates),
                    "--evaluations",
                    *map(str, c_paths),
                    "--output",
                    str(compact_selected),
                ],
                c_evals,
                c_candidates + c_paths,
                [compact_selected],
                gpus=0,
            )
            selected_deps = {"FULL": [t_select], "UNIFORM": [c_select], "CURRENT": [c_select]}
            for reader in ("tree", "flat"):
                r_cfg = config(seed, f"router_{reader}")
                r_initial = config(
                    seed,
                    f"router_{reader}_initial",
                    {
                        "training": {
                            "steps": manifest["router"]["initial_steps"],
                            "checkpoint_steps": [manifest["router"]["initial_steps"]],
                        }
                    },
                )
                r_final = config(
                    seed,
                    f"router_{reader}_final",
                    {
                        "training": {"steps": manifest["router"]["total_steps"]},
                        "labels": {"use_refresh": True},
                    },
                )
                initial_checkpoint = (
                    root / f"router_{reader}" / f"step_{manifest['router']['initial_steps']:08d}.pt"
                )
                final_checkpoint = root / f"router_{reader}" / "last.pt"
                labels = add(
                    seed,
                    f"labels_{reader}",
                    "labels",
                    base_prefix + ["stmem.training.gain_labels", "--reader", reader, "--config", r_cfg],
                    [c_select, cache_job],
                    [teacher_selected, compact_selected],
                    [root / f"router_{reader}" / "labels_initial.h5"],
                )
                initial = add(
                    seed,
                    f"router_{reader}_initial",
                    "router",
                    base_prefix + ["stmem.train", "--stage", "router", "--config", r_initial],
                    [labels],
                    [compact_selected, root / f"router_{reader}" / "labels_initial.h5"],
                    [initial_checkpoint],
                )
                refresh = add(
                    seed,
                    f"labels_{reader}_refresh",
                    "labels",
                    base_prefix
                    + [
                        "stmem.training.gain_labels",
                        "--reader",
                        reader,
                        "--config",
                        r_cfg,
                        "--refresh",
                        "--router-checkpoint",
                        str(initial_checkpoint),
                    ],
                    [initial],
                    [teacher_selected, compact_selected, initial_checkpoint],
                    [root / f"router_{reader}" / "labels_refresh.h5"],
                )
                final = add(
                    seed,
                    f"router_{reader}_final",
                    "router",
                    base_prefix
                    + [
                        "stmem.train",
                        "--stage",
                        "router",
                        "--config",
                        r_final,
                        "--resume",
                        str(initial_checkpoint),
                    ],
                    [refresh],
                    [
                        initial_checkpoint,
                        root / f"router_{reader}" / "labels_initial.h5",
                        root / f"router_{reader}" / "labels_refresh.h5",
                    ],
                    [final_checkpoint],
                )
                selected_deps[reader.upper()] = [final, c_select]
            if "PERCEIVER" in manifest["methods"]:
                p_job = add(
                    seed,
                    "perceiver",
                    "perceiver",
                    base_prefix
                    + ["stmem.train", "--stage", "perceiver", "--config", config(seed, "perceiver")],
                    [cache_job, t_select],
                    [teacher_selected],
                    [root / "perceiver" / "last.pt"],
                )
                selected_deps["PERCEIVER"] = [p_job]
        evaluation_jobs = []
        split = "val" if phase == "validation" else "test"
        count = manifest["evaluation"][
            "episodes_per_task" if split == "val" else "final_test_episodes_per_task"
        ]
        for method in manifest["methods"]:
            checkpoint = (
                teacher_selected
                if method == "FULL"
                else root / "perceiver" / "last.pt"
                if method == "PERCEIVER"
                else compact_selected
            )
            policy_config = (
                config(seed, "perceiver") if method == "PERCEIVER" else t_cfg if method == "FULL" else c_cfg
            )
            router = root / f"router_{method.lower()}" / "last.pt" if method in ("TREE", "FLAT") else None
            for shard in range(manifest["evaluation"]["shards"]):
                output = root / "evaluation" / split / method / f"shard{shard}"
                evaluation_jobs.append(
                    evaluate(
                        seed,
                        f"{split}_{method.lower()}_{shard}",
                        method,
                        checkpoint,
                        policy_config,
                        selected_deps.get(method, []),
                        output,
                        episode_count=count,
                        split=split,
                        router=router,
                        shard=shard,
                        shards=manifest["evaluation"]["shards"],
                    )
                )
        add(
            seed,
            f"report_{split}",
            "report",
            base_prefix + ["stmem.report", "--run-root", str(root), "--split", split],
            evaluation_jobs,
            outputs=[root / "reports" / split / "summary.json"],
            gpus=0,
        )
    return jobs


def render_job(
    job: Job, *, site: dict[str, str], repo: Path, generated: Path, logs: Path, python: str, profile: str
) -> None:
    # Stable across core/replication plan directories: seed0 stages are reused.
    stage_root = Path(site.get("STMEM_RUN_ROOT", str(repo / "runs"))).resolve()
    job.marker = str(stage_root / f"seed{job.seed}" / ".stages" / (job.name + ".json"))
    job.script = str(generated / "jobs" / (job.name + ".sbatch"))
    signature = {
        key: value
        for key, value in asdict(job).items()
        if key not in ("signature", "marker", "script", "commands", "walltime", "gpus")
    }
    signature["commands"] = []
    for command in job.commands:
        normalized = list(command)
        if "--config" in command:
            index = command.index("--config") + 1
            # Validate inheritance for cycles, then hash the actual merged content.
            config_hashes(Path(command[index]))
            normalized[index] = json.dumps(resolved_config(Path(command[index])), sort_keys=True)
        signature["commands"].append(scientific_training_command(normalized))
    signature["data_roots"] = {key: site.get(key) for key in ("STMEM_DATA_ROOT", "STMEM_CACHE_ROOT")}
    signature["source_sha256"] = source_hash(repo)
    job.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    directives = {
        "job-name": f"stmem_{job.name}",
        "nodes": "1",
        "ntasks": "1",
        "cpus-per-task": site.get("STMEM_CPUS", "16"),
        "mem": site.get("STMEM_MEM", "128G"),
        "time": job.walltime,
        "output": str(logs / "%x_%j.out"),
        "error": str(logs / "%x_%j.err"),
        "signal": "B:USR1@180",
        "chdir": str(repo),
    }
    for setting, directive in (
        ("PARTITION", "partition"),
        ("ACCOUNT", "account"),
        ("QOS", "qos"),
        ("CONSTRAINT", "constraint"),
    ):
        if site.get(f"STMEM_{setting}", "gpuq" if setting == "PARTITION" else ""):
            directives[directive] = site.get(f"STMEM_{setting}", "gpuq")
    if job.gpus:
        gpu_type = site.get("STMEM_GPU_TYPE", "")
        directives["gres"] = f"gpu:{gpu_type + ':' if gpu_type else ''}{job.gpus}"
    # Slurm reads literal directives before shell expansion. Validate newlines.
    if any("\n" in value or "\r" in value for value in directives.values()):
        raise ValueError("Newline in Slurm directive")
    lines = ["#!/usr/bin/env bash"] + [
        f"#SBATCH --{key}={shlex.quote(value)}" for key, value in directives.items()
    ]
    lines += ["set -euo pipefail", f"cd {shlex.quote(str(repo))}"]
    for key, value in site.items():
        # Empty optional overrides explicitly clear a submit-shell setting, so
        # an inherited entity/directory cannot defeat a blank site assignment.
        if key in WANDB_SITE_VARIABLES and not value:
            lines.append(f"unset {key}")
        else:
            lines.append(f"export {key}={shlex.quote(value)}")
    lines += [f"export STMEM_PROFILE={shlex.quote(profile)}", f"export STMEM_ALLOCATED_GPUS={job.gpus}"]
    if site.get("STMEM_ACTIVATE"):
        lines.append(f"source {shlex.quote(site['STMEM_ACTIVATE'])}")
    lines += [
        f'STMEM_STOP_FILE={shlex.quote(str(generated / "stops"))}/"{job.name}_${{SLURM_JOB_ID:-local_$$}}.stop"',
        "export STMEM_STOP_FILE",
    ]
    for command in job.commands:
        launch = list(command)
        if job.stage in ("teacher", "compact", "router", "perceiver"):
            launch = [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                f"--nproc-per-node={job.gpus}",
                *command[1:],
            ]
            # Stop filename is passed explicitly; no signals are forwarded to torchrun.
            text_command = shlex.join(["srun", "--ntasks=1", *launch]) + ' --stop-file "$STMEM_STOP_FILE"'
        else:
            text_command = shlex.join(["srun", "--ntasks=1", *launch])
        lines.append(
            f'exec {shlex.quote(str(repo / "hpc/run_job.sh"))} "$STMEM_STOP_FILE" '
            + shlex.join(
                [
                    python,
                    str(repo / "hpc/job_status.py"),
                    "--marker",
                    job.marker,
                    "--signature",
                    job.signature,
                    "--profile",
                    profile,
                    "--gpus",
                    str(job.gpus),
                    "--outputs",
                    json.dumps(job.outputs),
                    "--",
                ]
            )
            + " "
            + text_command
        )
    Path(job.script).parent.mkdir(parents=True, exist_ok=True)
    Path(job.script).write_text("\n".join(lines) + "\n")


def completed(job: Job) -> bool:
    marker = Path(job.marker)
    if not marker.exists():
        return False
    try:
        state = json.loads(marker.read_text())
        return (
            state.get("status") == "complete"
            and state.get("signature") == job.signature
            and all(
                Path(path).is_file() and state.get("output_sha256", {}).get(path) == sha256(Path(path))
                for path in job.outputs
            )
        )
    except (OSError, ValueError):
        return False


def freeze_protocol(
    manifest: dict,
    run_root: Path,
    destination: Path,
    repo: Path = REPO,
    teacher_config: Path | None = None,
) -> None:
    from stmem.evaluation.protocol import inference_contract

    checkpoints = {}
    checkpoint_roles = {}
    for seed in manifest["seeds"]:
        roles = {}
        for stage, filename in (
            ("teacher", "selected.pt"),
            ("compact", "selected.pt"),
            ("router_tree", "last.pt"),
            ("router_flat", "last.pt"),
        ):
            path = run_root / f"seed{seed}" / stage / filename
            checkpoints[str(path)] = sha256(path)
            roles[stage] = str(path)
        if "PERCEIVER" in manifest["methods"]:
            path = run_root / f"seed{seed}" / "perceiver" / "last.pt"
            checkpoints[str(path)] = sha256(path)
            roles["perceiver"] = str(path)
        checkpoint_roles[str(seed)] = roles
    configs = {}
    inference_contracts = {}
    teacher_config = (teacher_config or repo / "configs/teacher.yaml").resolve()
    for name in ("teacher", "compact", "router_tree", "router_flat", "perceiver"):
        path = teacher_config if name == "teacher" else repo / "configs" / f"{name}.yaml"
        configs.update(config_hashes(path))
        if name in ("teacher", "compact", "perceiver"):
            cfg = resolved_config(path)
            inference_contracts[name] = inference_contract(cfg)
    save_json(
        destination,
        {
            "protocol": manifest,
            "teacher_config": str(teacher_config),
            "checkpoints": checkpoints,
            "configuration_hashes": configs,
            "inference_contracts": inference_contracts,
            "checkpoint_roles": checkpoint_roles,
            "source_sha256": source_hash(repo),
            "status": "frozen before official test evaluation",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", choices=("core_seed0", "central_3seed"), default="core_seed0")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--submit", action="store_true")
    parser.add_argument("--site", type=Path, default=REPO / "hpc/site.env")
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument(
        "--teacher-config",
        type=Path,
        help="Teacher source YAML, relative to the site repository; default: configs/teacher.yaml",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        help="Override STMEM_RUN_ROOT for this plan; relative paths use the site repository",
    )
    parser.add_argument("--phase", choices=("validation", "test"), default="validation")
    parser.add_argument("--freeze-protocol", type=Path)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--log-every",
        type=positive_integer,
        help="Runtime training metric interval in optimizer updates; leaves scientific config unchanged",
    )
    parser.add_argument(
        "--progress-every-seconds",
        type=positive_seconds,
        help="Runtime policy accumulation progress interval; not passed to router training",
    )
    parser.add_argument(
        "--only", nargs="+", help="Stage categories or exact job names, without rerunning ancestors"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    site = read_site(args.site)
    repo = Path(site.get("STMEM_REPO_ROOT", str(REPO))).resolve()
    run_root = Path(site.get("STMEM_RUN_ROOT", str(repo / "runs"))).resolve()
    manifest = yaml.safe_load((repo / "configs" / f"experiment_{args.plan}.yaml").read_text())
    if args.run_root is not None:
        run_root = (repo / args.run_root.expanduser()).resolve()
        # Runtime configuration and stage markers also read the exported site root.
        site["STMEM_RUN_ROOT"] = str(run_root)
    teacher_config = (repo / (args.teacher_config or Path("configs/teacher.yaml")).expanduser()).resolve()
    if args.teacher_config is not None:
        from stmem.config import validate_config

        try:
            config_hashes(teacher_config)  # Check the complete inheritance chain before writing anything.
            teacher = resolved_config(teacher_config)
            validate_config(teacher)
            if (
                teacher.get("stage") != "teacher"
                or teacher.get("debug", False)
                or teacher["data"].get("synthetic", False)
                or teacher["data"].get("split") != "train"
            ):
                raise ValueError("--teacher-config must describe real-data teacher training")
            if teacher["data"]["tasks"] != manifest["tasks"]:
                raise ValueError("--teacher-config tasks must match the experiment manifest")
            training = teacher["training"]
            if any(
                step not in training["checkpoint_steps"] or not 1 <= step <= training["steps"]
                for step in manifest["teacher_candidates"]
            ):
                raise ValueError("--teacher-config must produce every planned teacher candidate checkpoint")
        except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
            parser.error(str(error))
    profile = args.profile or site.get("STMEM_PROFILE", manifest["profile"])
    if profile == "a5000_debug":
        parser.error(
            "a5000_debug is a local shape profile; use configs/debug/*.yaml directly, not core submission"
        )
    if args.freeze_protocol:
        freeze_protocol(manifest, run_root, args.freeze_protocol, repo, teacher_config)
        print(f"Frozen checkpoint hashes and protocol: {args.freeze_protocol}")
        return
    if args.phase == "test":
        if not args.freeze_manifest:
            parser.error(
                "--phase test requires --freeze-manifest, created with --freeze-protocol after validation"
            )
        frozen = json.loads(args.freeze_manifest.read_text())
        if frozen.get("teacher_config", str(repo / "configs/teacher.yaml")) != str(teacher_config):
            parser.error("Teacher configuration differs from the frozen test protocol")
        if frozen["protocol"] != manifest:
            parser.error("Frozen protocol differs from current manifest")
        if frozen.get("source_sha256") != source_hash(repo):
            parser.error("Source code differs from the frozen test protocol")
        for path, digest in frozen["checkpoints"].items():
            if sha256(Path(path)) != digest:
                parser.error(f"Frozen checkpoint changed: {path}")
        for path, digest in frozen["configuration_hashes"].items():
            if sha256(Path(path)) != digest:
                parser.error(f"Frozen configuration changed: {path}")
    generated = (args.output_dir or repo / "hpc/generated" / f"{args.plan}_{args.phase}_{profile}").resolve()
    logs = repo / "hpc/logs"
    # Must precede sbatch: Slurm opens stdout/stderr before the shell runs.
    logs.mkdir(parents=True, exist_ok=True)
    generated.mkdir(parents=True, exist_ok=True)
    python = site.get("STMEM_PYTHON") or str(repo / ".venv/bin/python")
    jobs = build_jobs(
        manifest,
        repo=repo,
        run_root=run_root,
        generated=generated,
        profile=profile,
        python=python,
        phase=args.phase,
        walltime=site.get("STMEM_WALLTIME", "30:00:00"),
        eval_walltime=site.get("STMEM_EVAL_WALLTIME", "06:00:00"),
        log_every=args.log_every,
        progress_every_seconds=args.progress_every_seconds,
        teacher_config=teacher_config,
    )
    # Configuration provenance contains bounded label totals explicitly.
    for seed in manifest["seeds"]:
        for reader in ("tree", "flat"):
            path = generated / "configs" / f"seed{seed}_router_{reader}.yaml"
            if path.exists():
                cfg = yaml.safe_load(path.read_text())
                cfg["labels"] = {
                    "contexts": manifest["router"]["initial_contexts"],
                    "refresh_contexts": manifest["router"]["refresh_contexts"],
                    "candidates": manifest["router"]["candidate_moves"],
                    "budgets": manifest["training_budgets"],
                }
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if args.phase == "test":
        for path in (generated / "configs").glob("*.yaml"):
            cfg = yaml.safe_load(path.read_text())
            cfg["evaluation"] = {"test_frozen_manifest": str(args.freeze_manifest.resolve())}
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    limit = int(site.get("STMEM_MAX_CONCURRENT", manifest["evaluation"]["max_concurrent_allocations"]))
    if limit < 1:
        parser.error("Maximum concurrent allocations must be positive")
    for job in jobs:
        render_job(job, site=site, repo=repo, generated=generated, logs=logs, python=python, profile=profile)
    all_jobs = {job.name: job for job in jobs}
    selected = [job for job in jobs if not args.only or job.name in args.only or job.stage in args.only]
    if args.only and not selected:
        parser.error("--only did not match any stage or job")
    skipped = [job.name for job in selected if args.resume and completed(job)]
    selected = [job for job in selected if job.name not in skipped]
    if args.resume:
        for job in selected:
            if job.stage in ("teacher", "compact", "router", "perceiver"):
                stage = (
                    job.name.removeprefix(f"seed{job.seed}_").removesuffix("_initial").removesuffix("_final")
                )
                checkpoint = run_root / f"seed{job.seed}" / stage / "last.pt"
                if checkpoint.exists():
                    if "--resume" in job.commands[0]:
                        index = job.commands[0].index("--resume") + 1
                        job.commands[0][index] = str(checkpoint)
                    else:
                        job.commands[0] += ["--resume", str(checkpoint)]
                    # Save status under the original stage signature: resume is execution state,
                    # not a scientific configuration change.
                    original_signature = job.signature
                    render_job(
                        job,
                        site=site,
                        repo=repo,
                        generated=generated,
                        logs=logs,
                        python=python,
                        profile=profile,
                    )
                    text = Path(job.script).read_text().replace(job.signature, original_signature)
                    job.signature = original_signature
                    Path(job.script).write_text(text)
    # Serialize N scheduling lanes with afterany; true scientific deps remain afterok.
    # This supplies a hard bound across GPU and CPU allocations without site QoS assumptions.
    lanes = {
        job.name: selected[index - limit].name if index >= limit else None
        for index, job in enumerate(selected)
    }
    state_file = generated / "submitted.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    selected_names = {job.name for job in selected}
    details = {
        "plan": args.plan,
        "phase": args.phase,
        "profile": profile,
        "teacher_config": str(teacher_config),
        "run_root": str(run_root),
        "seeds": manifest["seeds"],
        "tasks": manifest["tasks"],
        "jobs": [asdict(job) for job in selected],
        "job_count": len(selected),
        "maximum_concurrent_allocations": min(limit, len(selected)),
        "estimated_gpu_hours": "unknown; measured timings required",
        "gpu_hour_accounting": "elapsed allocation walltime * allocated GPUs; H200 and B200 separate",
        "completed_skipped": skipped,
        "remaining_stage_work": [job.name for job in selected],
        "concurrency_predecessors": lanes,
    }
    save_json(generated / "plan.json", details)
    print(json.dumps(details, indent=2))
    if not args.submit:
        print(f"DRY RUN: no jobs submitted. Rendered plan: {generated / 'plan.json'}")
        return
    if not args.site.exists():
        parser.error(
            "Submission needs a configured --site file; copy hpc/site.env.example and set site values"
        )
    for job in selected:
        # Existing active IDs are reported, never resubmitted blindly.
        if args.resume and job.name in state:
            check = subprocess.run(
                ["squeue", "-h", "-j", state[job.name]["job_id"], "-o", "%T"],
                capture_output=True,
                text=True,
                check=True,
            )
            if check.stdout.strip():
                print(f"Already active {job.name}: {state[job.name]['job_id']}")
                continue
        dependency_ids = []
        for name in job.deps:
            dependency = all_jobs[name]
            if completed(dependency):
                continue
            if name in state and (name in selected_names or args.resume):
                dependency_ids.append(state[name]["job_id"])
            elif (
                name not in selected_names
                and all(Path(path).exists() for path in dependency.outputs)
                and dependency.outputs
            ):
                # Selective stage use requires actual upstream outputs; input hashes are recorded by the stage.
                continue
            else:
                parser.error(f"Missing dependency for {job.name}: {name}; resume/submit it first")
        for path in job.inputs:
            ancestors = set(job.deps)
            frontier = list(job.deps)
            while frontier:
                for name in all_jobs[frontier.pop()].deps:
                    if name not in ancestors:
                        ancestors.add(name)
                        frontier.append(name)
            promised = [Path(output) for name in ancestors for output in all_jobs[name].outputs]
            if not Path(path).exists() and not any(
                Path(path) == output or Path(path) in output.parents for output in promised
            ):
                parser.error(f"Unresolved external input for {job.name}: {path}")
        args_sbatch = ["sbatch", "--parsable"]
        clauses = []
        if dependency_ids:
            clauses.append("afterok:" + ":".join(dict.fromkeys(dependency_ids)))
        predecessor = lanes[job.name]
        if predecessor and predecessor in state:
            clauses.append("afterany:" + state[predecessor]["job_id"])
        if clauses:
            args_sbatch.append("--dependency=" + ",".join(clauses))
        args_sbatch.append(job.script)
        result = subprocess.run(args_sbatch, text=True, capture_output=True, check=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdecimal():
            raise RuntimeError(f"Unrecognized sbatch --parsable response: {result.stdout!r}")
        state[job.name] = {"job_id": job_id, "signature": job.signature, "command": args_sbatch}
        save_json(state_file, state)
        print(f"SUBMITTED {job.name}: {job_id}")


if __name__ == "__main__":
    main()
