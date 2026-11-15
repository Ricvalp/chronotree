"""A separately sealed expert-utility router pilot, derived from a frozen study.

The parent measurement experiment is never rewritten or reinterpreted as a
completed primary-endpoint experiment. Its original code validates the parent
and checkpoint; the new code owns a new manifest and new training labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import torch

from stmem.config import digest, source_revision
from stmem.data.canonical import aligned_action_chunk
from stmem.inspection.refinement_data import (
    episode_name, read_json, seal_manifest, stable_seed, verify_manifest,
)
from stmem.training.cached_loader import LightweightDecisions
from stmem.training.checkpoints import atomic_copy, atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.label_data import _ReferenceData
from stmem.training.runtime import load_checkpoint, validate_policy_config

SCHEMA = "utility_router_study_v1"


def episode_key(row):
    return row["task"], row["episode_id"]


def validate_pool(parent):
    """Every fit episode must belong to the parent's designated fitting pool."""
    pool = list(map(episode_key, parent["training_pool"]))
    excluded = set(map(episode_key, parent["diagnostic_episodes"]))
    reserved = set(map(episode_key, parent["reserved_validation_episodes"]))
    tags = {episode_key(row): row["partition"] for row in parent["episode_partitions"]}
    tasks = set(parent["configuration"]["data"]["tasks"])
    if (not pool or len(set(pool)) != len(pool) or set(pool) & (excluded | reserved)
            or any(tags.get(key) != "train" for key in pool)
            or {task for task, _ in pool} != tasks):
        raise ValueError("Router fitting pool is incomplete, duplicated, or includes excluded episodes")
    return set(pool)


def validate_states(parent, states, *, fit_states):
    pool = validate_pool(parent)
    development = {row["state_id"]: row for row in parent["states"] if row["partition"] == "train"}
    seen, counts = set(), Counter()
    for state in states:
        key = (state["task"], state["episode_id"], state["decision_index"])
        if key in seen or state["state_id"] != digest(dict(zip(("task", "episode_id", "decision_index"), key)))[:20]:
            raise ValueError("Router state is duplicated or its identity changed")
        seen.add(key)
        if state["partition"] == "fit":
            if episode_key(state) not in pool:
                raise ValueError("Diagnostic or reserved episode leaked into router fitting")
        elif state["partition"] == "development":
            original = development.get(state["state_id"])
            if original is None or any(original[k] != state[k] for k in ("task", "episode_id", "decision_index", "episode_index")):
                raise ValueError("Router development states must be the parent's fixed development panel")
        else:
            raise ValueError("Final confirmation is not a router pilot fitting/development partition")
        counts[state["partition"]] += 1
    if counts["fit"] != fit_states or counts["development"] != len(development):
        raise ValueError("Router state count differs from the declared fitting/development panels")
    task_counts = Counter(row["task"] for row in states if row["partition"] == "fit")
    if set(task_counts) != set(parent["configuration"]["data"]["tasks"]) or max(task_counts.values()) - min(task_counts.values()) > 1:
        raise ValueError("Router fitting states must remain task balanced")


def select_fitting_states(data, parent, *, count=4096, seed=20261115, progress=None):
    """Gain-blind, unique, task/episode/phase balanced full-horizon decisions."""
    allowed = validate_pool(parent)
    if count < len(data.tasks):
        raise ValueError("Fit state count must cover every task")
    small_reader = LightweightDecisions(_ReferenceData(data))
    selected = []
    try:
        for task_slot, task in enumerate(data.tasks):
            requested = count // len(data.tasks) + int(task_slot < count % len(data.tasks))
            by_episode = defaultdict(list)
            for index, t in data.candidates[task]:
                if episode_name(data, index) in allowed:
                    by_episode[index].append(t)
            pools = {}
            for index, times in sorted(by_episode.items()):
                small = small_reader._episode(index)
                structural = small.action_valid.any(0).bool()
                if not structural.any():
                    continue
                # Evaluate masks only, never sample or evaluate policy outputs.
                eligible = []
                for t in times:
                    _, valid = aligned_action_chunk(small.actions, small.action_valid, t, data.horizon,
                                                    small.alignment, data.action_stride)
                    if bool(valid[:, structural].all()):
                        eligible.append(t)
                buckets = [[], [], []]
                for offset, t in enumerate(eligible):
                    phase = min(2, 3 * offset // len(eligible))
                    buckets[phase].append((t, ("early", "middle", "late")[phase]))
                rng = random.Random(stable_seed(seed, task, episode_name(data, index)[1], "fitting"))
                for bucket in buckets:
                    rng.shuffle(bucket)
                order = [bucket[offset] for offset in range(max(map(len, buckets), default=0))
                         for bucket in buckets if offset < len(bucket)]
                if order:
                    pools[index] = order
            indices = sorted(pools)
            random.Random(stable_seed(seed, task, "episode_order")).shuffle(indices)
            pairs = [(index, pools[index][offset]) for offset in range(max(map(len, pools.values()), default=0))
                     for index in indices if offset < len(pools[index])]
            if len(pairs) < requested:
                raise ValueError(f"{task}: only {len(pairs)} unique complete-horizon fitting states; need {requested}")
            for index, (t, phase) in pairs[:requested]:
                _, episode = episode_name(data, index)
                small = small_reader._episode(index)
                key = {"task": task, "episode_id": episode, "decision_index": t}
                selected.append({**key, "state_id": digest(key)[:20], "episode_index": index,
                                 "source_local_index": int(small.execution_indices[t]),
                                 "partition": "fit", "phase": phase})
            if progress is not None:
                progress("task_sampled", task=task, eligible_states=len(pairs),
                         selected_states=requested, tasks_completed=task_slot + 1,
                         tasks_total=len(data.tasks))
    finally:
        small_reader.close()
    # Interleave tasks so every shard is representative; no rank owns one task.
    by_task = {task: [row for row in selected if row["task"] == task] for task in data.tasks}
    return [rows[offset] for offset in range(max(map(len, by_task.values())))
            for rows in by_task.values() if offset < len(rows)]


def verified_parent(parent_path, checkpoint, student, code_root):
    """Run the unchanged parent's validation under its recorded source code."""
    parent_path, checkpoint, code_root = map(lambda p: Path(p).resolve(), (parent_path, checkpoint, code_root))
    from hpc.utility_source import verify_study_source

    receipt = verify_study_source(code_root, parent_path)
    command = (
        "import json,sys; from pathlib import Path; "
        "from stmem.inspection.refinement_data import verify_manifest; "
        "from stmem.inspection.refinement_utility import validate_student_endpoint; "
        "from stmem.training.runtime import load_checkpoint; "
        "p=Path(sys.argv[1]); m=verify_manifest(p); s=load_checkpoint(sys.argv[2]); "
        "v=validate_student_endpoint(m,p,sys.argv[3],s,'train'); "
        "print(json.dumps({'manifest':m,'training_identity':v}))"
    )
    result = subprocess.run([sys.executable, "-c", command, str(parent_path), str(checkpoint), student],
                            cwd=code_root, env={**os.environ, "PYTHONPATH": str(code_root / "src"),
                                               "CUDA_VISIBLE_DEVICES": "", "GIT_CEILING_DIRECTORIES": str(code_root.parent)},
                            check=True, text=True, capture_output=True)
    return json.loads(result.stdout), {"code_root": str(code_root), "receipt_sha256": receipt["receipt_sha256"],
                                      "git": receipt["git"], "source_sha256": receipt["source_sha256"]}


def prepare(parent_manifest, checkpoint, output, *, student="S_STD", fit_states=4096, steps=5000,
            workers=4, development_step=95000, study_source_root=None, seed=20261115):
    started = time.perf_counter()

    def progress(phase, **details):
        print(json.dumps({"event": "utility_router_preparation", "phase": phase,
                          "elapsed_seconds": time.perf_counter() - started, **details}), flush=True)

    parent_manifest, checkpoint, output = map(lambda p: Path(p).resolve(), (parent_manifest, checkpoint, output))
    if student not in ("S_STD", "S_STAGE", "S_PACKET") or min(fit_states, steps, workers, development_step) < 1:
        raise ValueError("Positive pilot limits and a matched primary student are required")
    if study_source_root is None:
        raise ValueError("Supply --study-source-root for original-parent validation; never rewrite the old manifest")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = {"parent_manifest": str(parent_manifest), "parent_manifest_sha256": file_sha256(parent_manifest),
                   "source_checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
                   "student": student, "fit_states": fit_states, "steps": steps, "workers": workers,
                   "development_step": development_step, "seed": seed}
        path = output / "manifest.json"
        if path.exists():
            existing = verify_study(path)
            if existing["request"] != request:
                raise ValueError("Pilot request changed; choose a new output directory")
            return existing
        progress("validating_parent")
        verified, parent_runtime = verified_parent(parent_manifest, checkpoint, student, study_source_root)
        parent, training_identity = verified["manifest"], verified["training_identity"]
        if file_sha256(parent_manifest) != request["parent_manifest_sha256"]:
            raise ValueError("Parent manifest changed while validating it")
        if parent.get("kind") != "matched_student_refinement_utility" or parent["dependencies"].get("diagnostic_assets"):
            raise ValueError("Router training requires the full real-data study, not a synthetic/limited diagnostic cache")
        snapshot = output / "frozen" / "student.pt"
        progress("snapshotting_student")
        atomic_copy(checkpoint, snapshot)
        if file_sha256(snapshot) != request["checkpoint_sha256"]:
            raise ValueError("Student checkpoint changed while snapshotting it")
        state = load_checkpoint(snapshot)
        if state.get("stage") != "compact" or state["step"] != development_step:
            raise ValueError("Checkpoint must match the explicitly declared pilot development step")
        validate_policy_config(state["config"], parent["configuration"])
        cfg = deepcopy(parent["configuration"])
        cfg.setdefault("tracking", {}).update(enabled=True, project="chronotree-routers")
        if cfg["data"].get("synthetic") or cfg.get("debug"):
            raise ValueError("Production router pilot cannot fit synthetic/debug inputs")
        progress("catalog_started")
        data = TrainingData(cfg)
        progress("catalog_ready")
        encoded = read_json(Path(parent["dependencies"]["encoded"]) / "manifest.json")
        if data.identity != encoded["identity"]["data_identity"]:
            raise ValueError("Router decisions differ from the accepted encoded cache dataset")
        states = select_fitting_states(data, parent, count=fit_states, seed=seed, progress=progress)
        states += [{**row, "partition": "development"} for row in parent["states"] if row["partition"] == "train"]
        validate_states(parent, states, fit_states=fit_states)
        source = source_revision()
        runtime_root = Path(__file__).resolve().parents[3]
        runtime_receipt = runtime_root / ".utility-source.json"
        if runtime_receipt.is_file():
            source["git"] = read_json(runtime_receipt)["git"]
        primary_endpoint = training_identity["endpoint"]
        scope = "development_snapshot_pilot" if development_step != primary_endpoint else "primary_endpoint_router_pilot"
        manifest = seal_manifest({
            "schema": SCHEMA, "source": source, "request": request,
            "parent_manifest": str(parent_manifest), "parent_manifest_sha256": request["parent_manifest_sha256"],
            "parent_experiment_id": parent["experiment_id"], "parent_runtime": parent_runtime,
            "configuration": cfg, "dependencies": deepcopy(parent["dependencies"]),
            "student": {"name": student, "checkpoint": str(snapshot), "sha256": request["checkpoint_sha256"],
                        "step": development_step, "primary_endpoint": primary_endpoint, **training_identity},
            "protocol": {"scope": scope, "seed": seed, "fit_states": fit_states, "workers": workers,
                         "readers": ["TREE", "FLAT"], "targets": ["immediate", "packet"], "budget": 128,
                         "packet_max_cost": 8, "candidate_limit": 8, "route_prefixes_per_state": 2,
                         "discovery_draws": 4, "confirmation_draws": 4, "key_width": 128,
                         "precision": parent["protocol"].get("precision", "fp32"),
                         "fit": {"steps": steps, "batch": 256, "lr": .0003, "checkpoint_every": 1000,
                                 "log_every": 10, "eval_every": 250, "weight_decay": .01},
                         "execute_targets": parent["protocol"].get("execute_targets", 8),
                         "flow_steps": cfg["model"]["flow_steps"], "wandb_project": "chronotree-routers",
                         "controller": "packet", "route_distribution": "gain_blind_reachable_prefixes",
                         "confirmation_status": "parent_final_partition_not_consumed",
                         "scope_note": "User-requested router follow-up from declared development snapshots; parent100k endpoint unchanged."},
            "states": states, "training_pool": deepcopy(parent["training_pool"]),
            "diagnostic_episodes": deepcopy(parent["diagnostic_episodes"]),
            "reserved_validation_episodes": deepcopy(parent["reserved_validation_episodes"]),
            "data_identity": data.identity,
        })
        atomic_json(manifest, path)
        progress("complete", states=len(states), manifest=str(path))
        return manifest


def verify_study(path, *, verify_source=True, verify_dependencies=True):
    """Validate new runtime ownership without changing the parent's guards."""
    manifest = read_json(path)
    if manifest.get("schema") != SCHEMA or seal_manifest(manifest)["experiment_id"] != manifest.get("experiment_id"):
        raise ValueError("Router study schema or content hash changed")
    if verify_source and manifest["source"]["source_sha256"] != source_revision()["source_sha256"]:
        raise ValueError("Router pilot source changed; use its frozen runtime or a new study")
    parent_path = Path(manifest["parent_manifest"])
    if file_sha256(parent_path) != manifest["parent_manifest_sha256"]:
        raise ValueError("Parent study changed")
    # Only the NEW stage executes here. The original source was checked by its
    # own runtime at adoption; the receipt and immutable parent hash bind it.
    parent = verify_manifest(parent_path, verify_source=False, verify_dependencies=verify_dependencies)
    if parent["experiment_id"] != manifest["parent_experiment_id"] or parent["dependencies"] != manifest["dependencies"]:
        raise ValueError("Pilot and parent dependencies differ")
    if (manifest["parent_runtime"]["source_sha256"] != parent["source"]["source_sha256"]
            or manifest["parent_runtime"]["git"] != parent["source"]["git"]):
        raise ValueError("Original parent runtime identity changed")
    validate_states(parent, manifest["states"], fit_states=manifest["protocol"]["fit_states"])
    if verify_dependencies and file_sha256(manifest["student"]["checkpoint"]) != manifest["student"]["sha256"]:
        raise ValueError("Frozen router-label student checkpoint changed")
    if verify_dependencies and file_sha256(manifest["dependencies"]["teacher"]) != manifest["dependencies"]["teacher_sha256"]:
        raise ValueError("Frozen teacher reference checkpoint changed")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-manifest", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--student", choices=("S_STD", "S_STAGE", "S_PACKET"), default="S_STD")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--fit-states", type=int, default=4096)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--development-step", type=int, default=95000)
    p.add_argument("--study-source-root", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261115)
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    result = prepare(**vars(args))
    print(json.dumps({"status": "complete", "manifest": str(args.output / "manifest.json"),
                      "experiment_id": result["experiment_id"], "scope": result["protocol"]["scope"],
                      "states": len(result["states"])}, indent=2))


if __name__ == "__main__":
    main()
