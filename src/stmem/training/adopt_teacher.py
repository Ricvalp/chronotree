"""Adopt a user-accepted, already evaluated teacher without rerunning selection.

This records an explicit acceptance decision, not a claim that this checkpoint
won the original two-candidate selection protocol. The checkpoint bytes and
training configuration remain unchanged; rollout overrides are separate evidence.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
import yaml

from stmem.config import digest, validate_config
from stmem.evaluation.execution import evaluation_execution_config
from stmem.evaluation.metrics import TASKS, episode_key, summarize_episodes
from stmem.evaluation.workload import planned_episodes
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import load_checkpoint

SCHEMA = "accepted_teacher_v1"


def _read(path):
    return json.loads(Path(path).read_text())


def _same(actual, expected, name):
    if json.loads(json.dumps(actual)) != json.loads(json.dumps(expected)):
        raise ValueError(f"Accepted teacher evaluation {name} mismatch")


def evaluation_files(directory):
    """Only scientific evidence, excluding videos, telemetry and live log files."""
    directory = Path(directory)
    paths = {directory / name for name in ("manifest.json", "summary.json", "complete.json")}
    manifest = _read(directory / "manifest.json")
    if manifest.get("kind") == "parallel_validation_evaluation":
        count = manifest.get("num_shards")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Invalid accepted evaluation shard count")
        expected = {f"shard{index:03d}" for index in range(count)}
        actual = {path.name for path in (directory / "shards").iterdir() if path.is_dir()}
        if actual != expected:
            raise ValueError("Accepted evaluation contains missing or unexpected shard directories")
        for index in range(count):
            shard = directory / "shards" / f"shard{index:03d}"
            paths.update(shard / name for name in ("manifest.json", "complete.json"))
        paths.update((directory / "shards").glob("*/manifest.json"))
        paths.update((directory / "shards").glob("*/complete.json"))
        paths.update((directory / "shards").glob("*/episodes/*.json"))
    else:
        paths.update((directory / "episodes").glob("*.json"))
    if (directory / "evaluation.yaml").exists():
        paths.add(directory / "evaluation.yaml")
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("Evaluation evidence contains missing files or symbolic links")
    return {str(path.relative_to(directory)): file_sha256(path) for path in sorted(paths)}


def _completion(directory, manifest, count):
    marker = _read(directory / "complete.json")
    for key, expected in (
        ("status", "complete"), ("manifest_sha256", file_sha256(directory / "manifest.json")),
        ("planned", count), ("recorded", count),
    ):
        _same(marker.get(key), expected, f"completion {key}")
    if manifest.get("kind") == "parallel_validation_evaluation":
        _same(marker.get("num_shards"), manifest["num_shards"], "completion shard count")


def _serial_records(directory, manifest, plan, checkpoint_hash):
    cfg = manifest["config"]
    _same(manifest.get("num_shards"), 1, "serial shard count")
    _same(manifest.get("shard_index"), 0, "serial shard index")
    rows = []
    manifest_hash = file_sha256(directory / "manifest.json")
    for path in sorted((directory / "episodes").glob("*.json")):
        row = _read(path)
        key = episode_key(row)
        if key not in plan or path.name != f"{key[0]}_{key[1]:03d}.json":
            raise ValueError("Unplanned or incorrectly named episode in accepted evaluation")
        expected = {
            "evaluation_manifest_sha256": manifest_hash,
            "checkpoint_sha256": checkpoint_hash,
            "method": "FULL", "split": "val", "training_seed": cfg["seed"],
            "noise_seed": cfg["evaluation"]["seed"] + cfg["data"]["tasks"].index(key[0])*1000 + key[1],
            "execution_override": manifest.get("execution_override"),
        }
        for field, value in expected.items():
            _same(row.get(field), value, f"episode {field}")
        rows.append(row)
    return rows


def validate_evaluation(directory, state, checkpoint_hash):
    """Validate a private evidence copy; aggregation writes only into that copy."""
    directory = Path(directory)
    manifest = _read(directory / "manifest.json")
    saved_summary = _read(directory / "summary.json")
    cfg = manifest["config"]
    _same(manifest.get("split"), "val", "split")
    _same(manifest.get("method"), "FULL", "method")
    if manifest.get("selection_family") is not None or manifest.get("intervention", "none") != "none":
        raise ValueError("Teacher adoption requires FULL validation without a history intervention")
    checkpoints = manifest.get("checkpoints", {})
    if len(checkpoints) != 1 or set(checkpoints.values()) != {checkpoint_hash}:
        raise ValueError("Evaluation must identify exactly the accepted teacher checkpoint")
    if "policy_checkpoint_sha256" in manifest:
        _same(manifest["policy_checkpoint_sha256"], checkpoint_hash, "policy checkpoint")
    tasks = cfg["data"]["tasks"]
    if len(tasks) != len(TASKS) or set(tasks) != set(TASKS):
        raise ValueError("Teacher adoption requires all 16 official tasks")
    if cfg.get("debug") or cfg["data"].get("synthetic"):
        raise ValueError("Synthetic/debug evaluation cannot select a scientific teacher")
    _same(cfg["seed"], state["config"]["seed"], "training seed")
    override = manifest.get("execution_override")
    effective, expected_override = evaluation_execution_config(
        state["config"], cfg,
        execute_actions=override.get("execute_actions") if override else None, split="val",
    )
    _same(effective, cfg, "effective configuration")
    _same(override, expected_override, "execution override")
    if (directory / "evaluation.yaml").exists():
        _same(yaml.safe_load((directory / "evaluation.yaml").read_text()), cfg, "saved configuration")
    plan = planned_episodes(cfg)
    _same(manifest["planned"], plan, "planned episodes")
    _same(manifest["all_planned"], plan, "full episode plan")
    _completion(directory, manifest, len(plan))
    if manifest.get("kind") == "parallel_validation_evaluation":
        from stmem.evaluation.aggregation import aggregate_shards

        if manifest.get("policy_checkpoint_sha256") != checkpoint_hash:
            raise ValueError("Parallel evaluation lacks the accepted policy checkpoint identity")
        rows, summary = aggregate_shards(directory, manifest)
        workers = summary["worker_completion"]
        if workers["complete"] != workers["planned"]:
            raise ValueError("All evaluation workers must be complete before adoption")
    else:
        rows = _serial_records(directory, manifest, plan, checkpoint_hash)
        summary = summarize_episodes(rows, plan)
    _same(summary, saved_summary, "recomputed summary")
    if (summary["status"] != "complete" or summary["overall"]["missing"]
            or summary["overall"]["errors"] or summary["rates_are_lower_bounds"]):
        raise ValueError("Teacher adoption requires complete evaluation without missing/error episodes")
    if any(row.get("status") not in {"success", "fail", "timeout", "truncated"} for row in rows):
        raise ValueError("Evaluation contains an unrecognized outcome")
    return {
        "macro_success": summary["macro_success"], "overall": summary["overall"],
        "evaluation_config": deepcopy(cfg), "execution_override": override,
        "evaluation_protocol": {
            "method": "FULL", "split": "val", "planned": plan,
            "noise_seed": cfg["evaluation"]["seed"], "training_seed": cfg["seed"],
            "execute_actions": cfg["model"]["execute"],
            "action_stride": cfg["model"].get("action_stride", 1),
            "observation_stride": cfg["data"].get("observation_stride", 1),
            "replan_primitive_steps": cfg["model"]["execute"] * cfg["model"].get("action_stride", 1),
            "predicted_actions_per_chunk": cfg["model"]["horizon"],
            "max_steps": cfg["evaluation"]["max_steps"],
        },
    }


def _teacher_state(checkpoint):
    state = load_checkpoint(checkpoint)
    cfg = state["config"]
    validate_config(cfg)
    if (state.get("stage") != "teacher" or state.get("synthetic")
            or cfg.get("debug") or cfg["data"].get("synthetic")
            or cfg["model"].get("history_mode", "full") != "full"):
        raise ValueError("Adoption requires a real-data full-history teacher checkpoint")
    if (not isinstance(state.get("step"), int) or isinstance(state["step"], bool)
            or state["step"] <= 0):
        raise ValueError("Teacher checkpoint must record a positive training step")
    if len(cfg["data"]["tasks"]) != len(TASKS) or set(cfg["data"]["tasks"]) != set(TASKS):
        raise ValueError("Teacher must have the all-task training configuration")
    from stmem.models.policy import MemoryPolicy

    policy = MemoryPolicy(cfg)
    policy.load_state_dict(state["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in state["model"].values()):
        raise ValueError("Teacher model contains nonfinite tensors")
    return state


def _snapshot_hashes(directory):
    paths = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Immutable evaluation snapshot contains symbolic links")
    return {str(path.relative_to(directory)): file_sha256(path) for path in paths if path.is_file()}


@contextmanager
def _lock(output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(f".{output.name}.adoption.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is adopting this teacher") from error
        yield


def adopt_teacher(checkpoint, evaluation, output):
    """Freeze an explicitly accepted teacher; output is a selected.pt filepath."""
    checkpoint, evaluation, output = (Path(value).resolve() for value in (checkpoint, evaluation, output))
    if output == checkpoint or output.is_relative_to(evaluation):
        raise ValueError("Adopted output must be separate from source checkpoint/evaluation")
    record_path = output.with_suffix(".selection.json")
    evidence_path = output.with_suffix(".evaluation")
    with _lock(output):
        checkpoint_hash = file_sha256(checkpoint)
        source_hashes = evaluation_files(evaluation)
        if record_path.exists():
            record = _read(record_path)
            _same(record.get("schema"), SCHEMA, "adoption schema")
            _same(record.get("sha256"), checkpoint_hash, "previously adopted checkpoint")
            _same(record.get("source_evidence_hashes"), source_hashes, "previously adopted evidence")
            if (not output.is_file() or file_sha256(output) != checkpoint_hash
                    or not evidence_path.is_dir()
                    or _snapshot_hashes(evidence_path) != record.get("evidence_hashes")):
                raise ValueError("Immutable adopted checkpoint/evaluation snapshot is missing or changed")
            return output
        if output.exists() and file_sha256(output) != checkpoint_hash:
            raise FileExistsError("Adopted output already contains a different checkpoint")
        state = _teacher_state(checkpoint)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.stem}.adoption-", dir=output.parent))
        try:
            evidence = temporary / "evaluation"
            evidence.mkdir()
            for relative, expected_hash in source_hashes.items():
                destination = evidence / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(evaluation / relative, destination)
                if file_sha256(destination) != expected_hash:
                    raise ValueError("Evaluation evidence changed while copying")
            result = validate_evaluation(evidence, state, checkpoint_hash)
            snapshot_hashes = _snapshot_hashes(evidence)
            saved_checkpoint = temporary / "checkpoint.pt"
            shutil.copyfile(checkpoint, saved_checkpoint)
            if file_sha256(saved_checkpoint) != checkpoint_hash or file_sha256(checkpoint) != checkpoint_hash:
                raise ValueError("Teacher checkpoint changed while adopting; use the evaluated snapshot")
            if evaluation_files(evaluation) != source_hashes:
                raise ValueError("Evaluation evidence changed while adopting")
            if evidence_path.exists():
                if _snapshot_hashes(evidence_path) != snapshot_hashes:
                    raise FileExistsError("Existing adopted evaluation snapshot differs")
            else:
                evidence.rename(evidence_path)
            if not output.exists():
                os.replace(saved_checkpoint, output)
            atomic_json({
                "schema": SCHEMA, "selected": str(checkpoint), "sha256": checkpoint_hash,
                "step": state["step"], "checkpoint_config_sha256": digest(state["config"]),
                "rule": "explicit user acceptance of a completed official FULL validation evaluation",
                "not_candidate_selection": True, "accepted_utc": datetime.now(timezone.utc).isoformat(),
                "evaluations": [str(evaluation)], "evidence_directory": str(evidence_path),
                "source_evidence_hashes": source_hashes, "evidence_hashes": snapshot_hashes,
                **result,
            }, record_path)
            return output
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(adopt_teacher(args.checkpoint, args.evaluation, args.output))


if __name__ == "__main__":
    main()
