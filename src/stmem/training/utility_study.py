"""Matched student-study boundaries, random streams and context exposure.

This opt-in consumer of the existing immutable caches changes no encoder, action
mask or cache identity. Episode exclusions affect optimization sampling only;
the canonical episode catalogue and its original router partition stay intact.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from stmem.config import digest
from stmem.training.checkpoints import file_sha256


TEACHER_SHA256 = "eb57ccd4c131888a5f5d6225a3a0cbe5cd14b3afc5c6ff0a1a17ee319cd70918"
COMPACT_SHA256 = "4fe22fd00fbc64dc740ba046d6ce1e58ade2a70580f8792e152477caad316427"
ENCODED_CACHE_ID = "3db7b7ce80926d21659863c3"
HIERARCHY_CACHE_ID = "3567142ed84d1b30b4a744ca"
VARIANTS = ("S_STD", "S_STAGE", "S_PACKET")


def primary_schedule(steps, checkpoint_every=1000, snapshot_every=5000):
    """Declare the common endpoint before any arm is trained or evaluated."""
    for name, value in (("steps", steps), ("checkpoint_every", checkpoint_every),
                        ("snapshot_every", snapshot_every)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Student {name} must be a positive integer")
    development = list(range(snapshot_every, steps, snapshot_every))
    return {"primary_updates": steps, "development_steps": development,
            "checkpoint_steps": [*development, steps], "checkpoint_every": checkpoint_every,
            "snapshot_every": snapshot_every}


def stream_seed(seed, position, stream):
    payload = json.dumps([int(seed), int(position), str(stream)], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little") % (2**63 - 1)


def study_settings(cfg):
    value = cfg["training"].get("utility_study")
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("variant") not in VARIANTS:
        raise ValueError("Utility student study requires S_STD, S_STAGE or S_PACKET")
    if cfg.get("stage") != "compact" or not cfg["training"].get("hierarchy_cache_root"):
        raise ValueError("Utility student study requires indexed compact training")
    if not value.get("manifest") or not value.get("manifest_sha256"):
        raise ValueError("Utility student study requires an immutable panel manifest")
    return value


def episode_keys(rows, label):
    keys = [(str(row["task"]), str(row["episode_id"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Duplicate {label} episode identities")
    return set(keys)


def read_study_manifest(path, expected_hash=None):
    path = Path(path)
    if expected_hash is not None and file_sha256(path) != expected_hash:
        raise ValueError("Student study manifest changed")
    manifest = json.loads(path.read_text())
    if manifest.get("experiment_id") != digest({key: value for key, value in manifest.items()
                                                if key != "experiment_id"}):
        raise ValueError("Student study manifest seal is missing or invalid")
    from stmem.inspection.refinement_data import reference_name, study_dependencies, validate_protocol_teacher

    if any(field not in manifest for field in
           ("teacher_sha256", "compact_sha256", "encoded_cache_id", "hierarchy_cache_id")):
        raise ValueError("Student study manifest is missing immutable dependency identities")
    dependencies = study_dependencies(manifest)
    if reference_name(manifest) == "S_INIT":
        validate_protocol_teacher(manifest["protocol"], dependencies)
    pool = episode_keys(manifest["training_pool"], "training")
    diagnostic = episode_keys(manifest["diagnostic_episodes"], "diagnostic")
    reserved = episode_keys(manifest["reserved_validation_episodes"], "reserved-validation")
    partitions = {(str(row["task"]), str(row["episode_id"])): row["partition"]
                  for row in manifest["episode_partitions"]}
    if len(partitions) != len(manifest["episode_partitions"]):
        raise ValueError("Duplicate episode partition identities")
    if not pool or pool & (diagnostic | reserved):
        raise ValueError("Student training pool is empty or overlaps diagnostic/reserved episodes")
    if any(partitions.get(key) != "train" for key in pool):
        raise ValueError("Student training contains an episode outside the original router-fitting partition")
    expected_reserved = {key for key, value in partitions.items() if value == "validation"}
    if reserved != expected_reserved:
        raise ValueError("Reserved validation episodes do not cover the complete original partition")
    if pool != {key for key, value in partitions.items() if value == "train"} - diagnostic:
        raise ValueError("Student training pool must contain every remaining router-fitting episode")
    return manifest


def restrict_training_pool(data, cfg):
    settings = study_settings(cfg)
    if settings is None:
        return None
    manifest = read_study_manifest(settings["manifest"], settings["manifest_sha256"])
    pool = episode_keys(manifest["training_pool"], "training")
    partitions = {(row["task"], row["episode_id"]): row["partition"] for row in manifest["episode_partitions"]}
    allowed, found = set(), set()
    for task in data.tasks:
        for index in data.by_task[task]:
            if data.synthetic:
                key = (task, f"episode_{data.refs[index][1]}")
            else:
                ref = data.dataset.episodes[data.refs[index]]
                key = (ref.task, ref.key)
            if partitions.get(key) != data.label_partition(index):
                raise ValueError(f"Original router partition changed for {key}")
            found.add(key)
            if key in pool:
                allowed.add(index)
    if not pool <= found or found != set(partitions):
        raise ValueError("Student episode catalogue differs from the fixed study manifest")
    # Do not change refs/by_task: doing so would silently re-partition episodes.
    data.candidates = {task: [(episode, index) for episode, index in rows if episode in allowed]
                       for task, rows in data.candidates.items()}
    if any(not values for values in data.candidates.values()):
        raise ValueError("A student-study task has no remaining training decisions")
    data._permutation.cache_clear()
    data.sampler_version = data.sampler_version + "+utility_pool_v1"
    return {"manifest_sha256": settings["manifest_sha256"], "training_episodes": len(allowed),
            "training_episodes_by_task": dict(Counter(task for task, _ in pool)),
            "decisions": {task: len(values) for task, values in data.candidates.items()},
            "selection": "predeclared_{}_endpoint".format(
                settings.get("primary_updates", 30000) if not settings.get("initialization_checkpoint")
                else settings["continuation_updates"]),
            "inherited_models_may_have_seen_excluded_episodes": True}


def paired_flow_randomness(action, positions, seed):
    """Same example gets the same FM draw across samplers, workers and batching."""
    noise, times = [], []
    for position in positions:
        generator = torch.Generator(device=action.device).manual_seed(stream_seed(seed, position, "flow_v1"))
        noise.append(torch.randn(action.shape[1:], device=action.device, dtype=action.dtype, generator=generator))
        times.append(torch.rand((), device=action.device, dtype=action.dtype, generator=generator))
    return torch.stack(noise), torch.stack(times)


def summarize_training_rows(rows):
    """One-way strata keep useful losses without duplicating millions of rows."""
    sums, counts = Counter(), Counter()
    for row in rows:
        dimensions = {
            "task": row["task"], "phase": row["phase"], "family": row["family"],
            "history_length": "short" if row["history_frames"] < 64 else (
                "medium" if row["history_frames"] < 256 else "long"),
            "budget": str(row["actual_count"]),
            "flow_time": str(min(3, int(row["flow_time"] * 4))),
            "allocation": "template" if row["template"] else "ordinary",
            "fallback": str(row.get("fallback_reason") or "none"),
            "prefix": "within_packet" if row.get("within_packet") else "boundary_or_ordinary",
        }
        for dimension, value in dimensions.items():
            key = (dimension, value)
            counts[key] += 1
            sums[key] += float(row["loss"])
    return [{"dimension": key[0], "value": key[1], "count": count,
             "mean_expert_fm_loss": sums[key] / count} for key, count in sorted(counts.items())]


def ordinary_family(cfg, position, root_count):
    from stmem.training.packed_context import context_family

    try:
        method, budget, rng = context_family(cfg, position, root_count)
        return method, budget, rng, None
    except ValueError:
        feasible = [int(value) for value in cfg["model"]["budgets"] if int(value) >= root_count]
        if not feasible:
            raise ValueError(f"Study archive has {root_count} roots, exceeding every configured budget") from None
        alternate = deepcopy(cfg)
        alternate["model"]["budgets"] = [min(feasible)]
        method, budget, rng = context_family(alternate, position, root_count)
        return method, budget, rng, "legacy_nominal_below_root_count"


def consumed_packet_costs(route, count):
    """Lengths actually traversed to the sampled prefix, including a partial packet."""
    if count not in route["contexts"]:
        raise ValueError("Sampled prefix is absent from the completed route")
    costs = [min(int(packet["cost"]), count - int(packet["start_count"]))
             for packet in route["packets"] if int(packet["start_count"]) < count]
    if sum(costs) != count - route["initial_count"]:
        raise AssertionError("Consumed packets do not cover the sampled route prefix")
    return costs


def study_context(index, roots, leaves, cfg, position):
    context, metadata = _ordinary_study_context(index, roots, leaves, cfg, position)
    settings = study_settings(cfg)
    followup = settings.get("followup")
    if followup not in (None, "rehearsal_control", "rehearsal_full"):
        raise ValueError("Unsupported conditional student loader; no silent route substitution")
    if followup == "rehearsal_full" and context["reader"] != "CURRENT":
        rng = random.Random(stream_seed(cfg["seed"], position, "rehearsal_v1"))
        if rng.random() < 0.1:
            from stmem.inspection.refinement_candidates import IndexedArchive

            context = IndexedArchive(index, roots, leaves).context("TREE", leaves)
            metadata.update(original_family=metadata["family"], family="FULL", rehearsal=True,
                            actual_count=context["token_count"], budget=context["token_count"])
    return context, metadata


def _ordinary_study_context(index, roots, leaves, cfg, position):
    """Return selected indexed rows and an auditable, gain-blind exposure record."""
    from stmem.inspection.refinement_candidates import IndexedArchive, training_routes
    from stmem.training.packed_context import _flat_selection, _tree_indices

    settings = study_settings(cfg)
    if settings is None:
        raise ValueError("study_context is opt-in")
    method, ordinary_budget, ordinary_rng, fallback = ordinary_family(cfg, position, len(roots))
    common = random.Random(stream_seed(cfg["seed"], position, "exposure_v1"))
    use_template = method in ("TREE", "FLAT") and common.random() < 0.5
    requested = 128 if common.random() < 0.75 else 64
    metadata = {"variant": settings["variant"], "family": method, "template": False,
                "requested_target": requested, "fallback_reason": fallback,
                "position": int(position), "seed": stream_seed(cfg["seed"], position, "route_v1")}
    if settings["variant"] != "S_STD" and use_template:
        routes = training_routes(index, roots, leaves, method, requested, metadata["seed"])
        metadata.update(initial_count=routes["initial_count"], target_count=routes["target_count"],
                        saturated=routes["saturated"], fallback_reason=routes["reason"])
        if routes["eligible"]:
            start, end = routes["initial_count"], routes["target_count"]
            count = end if common.random() < 0.5 or end == start else common.randrange(start, end)
            route = routes["stage" if settings["variant"] == "S_STAGE" else "packet"]
            selected = route["contexts"][count]
            packet_boundaries = {item["start_count"] + item["cost"] for item in routes["packet"]["packets"]}
            metadata.update(template=True, budget=count, prefix_count=count,
                            within_packet=count not in packet_boundaries and count != start,
                            packet_costs=[item["cost"] for item in route["packets"]],
                            consumed_packet_costs=consumed_packet_costs(route, count),
                            region_groups=[item["group"] for item in route["packets"]],
                            eligible_pool_sizes=[item["eligible_pool_size"] for item in route["packets"]])
            return selected, metadata
    metadata.update(budget=ordinary_budget, prefix_count=None)
    if method in ("TREE", "UNIFORM"):
        rows = _tree_indices(index, roots, ordinary_budget, ordinary_rng, uniform=method == "UNIFORM")
    elif method == "FLAT":
        mask, _, _ = _flat_selection(index, leaves, ordinary_budget, ordinary_rng)
        rows = np.asarray(leaves)[mask]
    elif method == "CURRENT":
        rows = []
    else:
        raise ValueError(f"Unknown study family {method}")
    context = {"reader": "TREE" if method == "UNIFORM" else method, "rows": list(map(int, rows))}
    if method == "CURRENT":
        context["token_count"] = 0
    elif method == "FLAT":
        archive = IndexedArchive(index, roots, leaves)
        selected, codes, groups = archive.flat_arrays(context)
        context["token_count"] = int(selected.sum()) + len(set(codes[~selected]))
    else:
        context["token_count"] = len(rows)
    return context, metadata
