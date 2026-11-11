"""Immutable state/episode boundaries for the student × allocation study.

This module reads the existing action index and indexed causal caches. It never
encodes observations, chooses states from measured gains, or changes a cache.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import torch

from stmem.config import digest, source_revision
from stmem.data.cache import cache_identity, episode_key
from stmem.data.canonical import aligned_action_chunk
from stmem.data.hierarchy_cache import validate_hierarchy_cache
from stmem.training.cached_loader import LightweightDecisions
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.label_data import _ReferenceData
from stmem.training.packed_context import PackedContextReader, columns_bank, flat_residual_columns
from stmem.training.runtime import load_checkpoint, validate_policy_config

TEACHER_SHA = "eb57ccd4c131888a5f5d6225a3a0cbe5cd14b3afc5c6ff0a1a17ee319cd70918"
COMPACT_SHA = "4fe22fd00fbc64dc740ba046d6ce1e58ade2a70580f8792e152477caad316427"
ENCODED_ID = "3db7b7ce80926d21659863c3"
HIERARCHY_ID = "3567142ed84d1b30b4a744ca"
NEW_STUDENTS = ("S_STD", "S_STAGE", "S_PACKET")
TEACHER_REFERENCE = {"name": "S_INIT", "kind": "frozen_teacher_compact_smoke", "compact_updates": 0}


def reference_name(manifest):
    reference = manifest.get("reference", manifest.get("dependencies", {}).get("reference"))
    if reference is None:
        return "S0"
    if reference != TEACHER_REFERENCE:
        raise ValueError("Unknown utility-study reference contract")
    return "S_INIT"


def primary_students(manifest):
    expected = NEW_STUDENTS if reference_name(manifest) == "S_INIT" else ("S0", *NEW_STUDENTS)
    declared = manifest.get("primary_endpoints", manifest.get("protocol", {}).get("primary_endpoints", expected))
    if tuple(declared) != expected:
        raise ValueError("Primary students differ from the declared reference contract")
    protocol = manifest.get("protocol", {}).get("primary_endpoints")
    if protocol is not None and tuple(protocol) != expected:
        raise ValueError("Protocol primary students differ from the study manifest")
    return expected


def study_dependencies(manifest):
    """Resolve a sealed study's identities without mixing old and new encoders.

    Old manifests retain their original hard pins. A new teacher is accepted only
    through the explicit teacher-reference contract and its adopted-cache receipt.
    This metadata check complements the file validation at manifest construction.
    """
    deps = dict(manifest.get("dependencies", {}))
    fields = ("teacher_sha256", "compact_sha256", "encoded_cache_id", "hierarchy_cache_id")
    if reference_name(manifest) == "S0":
        defaults = dict(zip(fields, (TEACHER_SHA, COMPACT_SHA, ENCODED_ID, HIERARCHY_ID)))
        defaults.update(teacher_step=104110, compact_step=90000)
        for field, expected in defaults.items():
            if manifest.get(field, expected) != expected or deps.get(field, expected) != expected:
                raise ValueError(f"Legacy student study requires the recorded {field}")
        deps.update(defaults)
    else:
        for field in (*fields, "teacher_step", "compact_step", "accepted_cache_plan",
                      "accepted_cache_plan_sha256", "teacher_selection", "teacher_selection_sha256"):
            if field not in deps:
                raise ValueError(f"Teacher-reference study is missing dependency {field}")
        for field in fields:
            if manifest.get(field) != deps[field]:
                raise ValueError(f"Teacher-reference study has inconsistent {field}")
        for field in ("teacher_sha256", "compact_sha256", "accepted_cache_plan_sha256", "teacher_selection_sha256"):
            value = deps[field]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Teacher-reference study has invalid {field}")
        for field in ("encoded_cache_id", "hierarchy_cache_id"):
            value = deps[field]
            if not isinstance(value, str) or len(value) != 24 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"Teacher-reference study has invalid {field}")
        step = deps["teacher_step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ValueError("Teacher-reference study requires a positive teacher step")
        if deps["compact_sha256"] != deps["teacher_sha256"] or deps["compact_step"] != step:
            raise ValueError("S_INIT must contain exactly the accepted teacher weights, not an older compact policy")
        if deps.get("reference", TEACHER_REFERENCE) != TEACHER_REFERENCE:
            raise ValueError("Reference dependency differs from the manifest")
    primary_students(manifest)
    return deps


def teacher_identity(manifest):
    deps = study_dependencies(manifest)
    return deps["teacher_sha256"], deps["teacher_step"]


def reference_identity(manifest):
    deps = study_dependencies(manifest)
    return deps["compact_sha256"], deps["compact_step"]


def read_json(path):
    return json.loads(Path(path).read_text())


def stable_seed(*values):
    return int(digest(values)[:15], 16)


def episode_name(data, index):
    if data.synthetic:
        task, number, _ = data.refs[index]
        return task, f"episode_{number}"
    ref = data.dataset.episodes[data.refs[index]]
    return ref.task, ref.key


def verify_dependencies(cfg, teacher, compact, encoded, hierarchy, *, diagnostic_assets=False,
                        teacher_reference=False, cache_plan=None):
    """Strict production identities; explicit limited-data diagnostics are separate."""
    teacher, compact, encoded, hierarchy = map(Path, (teacher, compact, encoded, hierarchy))
    adopted = {}
    teacher_hash, compact_hash = file_sha256(teacher), file_sha256(compact)
    if teacher_reference:
        if diagnostic_assets or cache_plan is None:
            raise ValueError("A new teacher study requires a verified production adopted-cache plan")
        from hpc.train_compact import inspect_cache

        plan_path = Path(cache_plan).resolve()
        saved_cfg, plan, accepted_leaf = inspect_cache(plan_path)
        if (saved_cfg != cfg or teacher.resolve() != Path(plan["checkpoint"]).resolve()
                or teacher_hash != plan["checkpoint_sha256"] or compact_hash != teacher_hash):
            raise ValueError("Teacher-reference inputs differ from the accepted teacher/cache plan")
        accepted_root = Path(cfg["paths"]["cache_root"]) / digest(accepted_leaf["identity"])[:24]
        if encoded.resolve() != accepted_root.resolve():
            raise ValueError("Encoded cache is not the one verified by the adopted teacher plan")
        selection = read_json(plan["selection"])
        if (selection.get("schema") != "accepted_teacher_v1"
                or selection.get("sha256") != teacher_hash
                or selection.get("step") != plan.get("checkpoint_step")):
            raise ValueError("New teacher study requires an explicit validated teacher adoption")
        adopted = {"accepted_cache_plan": str(plan_path), "accepted_cache_plan_sha256": file_sha256(plan_path),
                   "teacher_selection": str(Path(plan["selection"]).resolve()),
                   "teacher_selection_sha256": plan["selection_sha256"],
                   "accepted_evaluation": selection.get("evaluations"),
                   "accepted_macro_success": selection.get("macro_success"),
                   "accepted_evaluation_protocol": selection.get("evaluation_protocol"),
                   "reference": deepcopy(TEACHER_REFERENCE)}
    else:
        if cache_plan is not None:
            raise ValueError("--cache-plan requires the explicit --teacher-reference contract")
        if teacher_hash != TEACHER_SHA or compact_hash != COMPACT_SHA:
            raise ValueError("Required immutable legacy teacher or compact checkpoint differs")
    t, s = load_checkpoint(teacher), load_checkpoint(compact)
    if teacher_reference:
        if t.get("stage") != "teacher" or s.get("stage") != "teacher" or t["step"] != plan["checkpoint_step"]:
            raise ValueError("Teacher-reference checkpoint stage/step differs from the adoption")
    else:
        if t["step"] != 104110 or s["step"] != 90000:
            raise ValueError("Wrong reference checkpoint steps")
        if s.get("provenance", {}).get("dependencies", {}).get("teacher") != TEACHER_SHA:
            raise ValueError("S0 does not depend on the accepted teacher")
    for state in (t, s):
        validate_policy_config(state["config"], cfg)
    m = cfg["model"]
    if (m["horizon"] != 16 or m.get("action_stride", 1) != 2
            or cfg["data"].get("observation_stride", 1) != 2
            or m.get("proprioception_mode") != "none" or m.get("condition_on_previous_commands", True)):
        raise ValueError("Study requires the accepted H16/10Hz/no-proprioception/no-command contract")
    leaf = read_json(encoded / "manifest.json")
    parents = validate_hierarchy_cache(encoded, hierarchy)
    if not diagnostic_assets:
        if not teacher_reference and (digest(leaf["identity"])[:24] != ENCODED_ID
                                      or digest(parents["identity"])[:24] != HIERARCHY_ID):
            raise ValueError("The production encoded/hierarchy identities differ")
        if leaf.get("kind", "").startswith("bounded_real"):
            raise ValueError("A diagnostic cache cannot replace the production cache")
    expected = cache_identity(cfg, teacher, leaf["identity"]["data_identity"])
    keys = ("format", "teacher_sha256", "geometry", "model_encoder", "normalization_version",
            "frozen_encoder_precision", "prefix_convention", "code")
    if any(expected[key] != leaf["identity"][key] for key in keys):
        raise ValueError("Frozen cache representation differs from the requested checkpoint/configuration")
    if not diagnostic_assets and expected != leaf["identity"]:
        raise ValueError("Production data configuration must exactly match the accepted cache")
    if diagnostic_assets:
        receipt = read_json(encoded.parent / "benchmark.json")
        recorded = set(receipt["provenance"]["inputs"].values())
        prepared = Path(cfg["paths"]["data_root"]) / cfg["data"]["prepared_subdir"]
        for task in leaf["tasks"]:
            if file_sha256(prepared / f"{task}.h5") not in recorded:
                raise ValueError("Diagnostic prepared data differs from its original cache construction")
    # Hash once at manifest construction. Later runs verify these same receipts
    # and manifests plus selected shard bytes, never derive a new cache silently.
    shard_guards = {}
    for root, record in ((encoded, leaf), (hierarchy, parents)):
        for name, shard in record["shards"].items():
            path = root / name
            before = path.stat()
            if file_sha256(path) != shard["sha256"]:
                raise ValueError(f"Corrupt immutable cache shard: {path}")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Immutable cache shard changed during verification: {path}")
            shard_guards[str(path.resolve())] = {"sha256": shard["sha256"],
                                                "bytes": after.st_size, "mtime_ns": after.st_mtime_ns}
    return {
        "teacher_sha256": teacher_hash, "compact_sha256": compact_hash,
        "teacher_step": t["step"], "compact_step": s["step"],
        "encoded_cache_id": digest(leaf["identity"])[:24],
        "hierarchy_cache_id": digest(parents["identity"])[:24],
        "encoded_manifest_sha256": file_sha256(encoded / "manifest.json"),
        "hierarchy_manifest_sha256": file_sha256(hierarchy / "manifest.json"),
        "teacher": str(teacher.resolve()), "compact": str(compact.resolve()),
        "encoded": str(encoded.resolve()), "hierarchy": str(hierarchy.resolve()),
        "diagnostic_assets": diagnostic_assets, "verified_shards": shard_guards, **adopted,
    }


def select_states(data, *, seed=20261110, episodes_per_partition=4):
    """Freeze episodes first, then two separated full-H states, before packets."""
    refs = _ReferenceData(data)
    decisions = LightweightDecisions(refs)
    partitions, diagnostics, states, exclusions = [], [], [], []
    try:
        for task in data.tasks:
            pools = {"train": [], "validation": []}
            for index in data.by_task[task]:
                _, episode = episode_name(data, index)
                partition = data.label_partition(index)
                partitions.append({"task": task, "episode_id": episode, "partition": partition})
                pools[partition].append(index)
            for partition, pool in pools.items():
                pool = sorted(pool, key=lambda i: episode_name(data, i)[1])
                random.Random(stable_seed(seed, task, partition, "episode")).shuffle(pool)
                if len(pool) < episodes_per_partition:
                    exclusions.append({"task": task, "partition": partition,
                                       "reason": "insufficient_episodes", "available": len(pool),
                                       "requested": episodes_per_partition})
                for slot, episode_index in enumerate(pool[:episodes_per_partition]):
                    _, episode = episode_name(data, episode_index)
                    diagnostics.append({"task": task, "episode_id": episode})
                    small = decisions._episode(episode_index)
                    structural = small.action_valid.any(0).bool()
                    eligible, excluded = [], 0
                    candidates = [t for ep, t in data.candidates[task] if ep == episode_index]
                    for t in candidates:
                        _, mask = aligned_action_chunk(small.actions, small.action_valid, t, data.horizon,
                                                       small.alignment, data.action_stride)
                        if structural.any() and bool(mask[:, structural].all()):
                            eligible.append(t)
                        else:
                            excluded += 1
                    exclusions.append({"task": task, "episode_id": episode,
                                       "reason": "incomplete_available_action_horizon", "count": excluded,
                                       "structural_action_mask": structural.tolist()})
                    if len(eligible) < 2:
                        exclusions.append({"task": task, "episode_id": episode,
                                           "reason": "fewer_than_two_full_horizon_decisions"})
                        continue
                    used = []
                    for visit, phase in enumerate(((2 * slot) % 3, (2 * slot + 1) % 3)):
                        lo, hi = len(eligible) * phase // 3, len(eligible) * (phase + 1) // 3
                        choices = eligible[lo:hi]
                        minimum_separation = max(data.stride, (eligible[-1] - eligible[0]) // 6)
                        choices = [t for t in choices if all(abs(t - old) >= minimum_separation for old in used)]
                        if not choices:
                            exclusions.append({"task": task, "episode_id": episode, "phase": phase,
                                               "reason": "no_separated_state_in_predeclared_phase"})
                            continue
                        t = random.Random(stable_seed(seed, task, episode, visit, "decision")).choice(choices)
                        used.append(t)
                        refs.reference = episode_index, t
                        decision = decisions.decision(0)
                        state = {"task": task, "episode_id": episode, "episode_index": episode_index,
                                 "decision_index": t, "source_local_index": int(small.execution_indices[t]),
                                 "partition": partition, "phase": ("early", "middle", "late")[phase],
                                 "structural_action_mask": structural.tolist(),
                                 "valid_coordinates": int(decision.target_valid.sum()),
                                 "conditioning_frames": decision.info.conditioning_frames,
                                 "execution_history_frames": decision.info.execution_history_frames}
                        state["state_id"] = digest({k: state[k] for k in (
                            "task", "episode_id", "decision_index")})[:20]
                        states.append(state)
    finally:
        decisions.close()
    held = {(r["task"], r["episode_id"]) for r in diagnostics}
    training_pool = [{"task": r["task"], "episode_id": r["episode_id"]} for r in partitions
                     if r["partition"] == "train" and (r["task"], r["episode_id"]) not in held]
    reserved = [{"task": r["task"], "episode_id": r["episode_id"]} for r in partitions
                if r["partition"] == "validation"]
    return dict(states=states, diagnostic_episodes=diagnostics, training_pool=training_pool,
                reserved_validation_episodes=reserved, episode_partitions=partitions,
                sampling_exclusions=exclusions)


def select_cached_diagnostic_states(data, reader, *, seed, maximum=8):
    """Asset-limited correctness panel, never a replacement primary sample.

    Select only pre-existing cached decisions before evaluating any model. This
    explicit diagnostic branch has no student training pool and cannot be used
    by the production study launcher. It may lack the required three-task smoke.
    """
    candidates, episodes, exclusions = [], [], []
    for task in data.tasks:
        for episode_index in data.by_task[task]:
            _, episode_id = episode_name(data, episode_index)
            path = reader.packed.hierarchy_root / f"{task}.h5"
            if not path.is_file():
                continue
            file = reader.packed.hierarchy_files.get(path)
            key = episode_key(task, episode_id)
            if key not in file:
                continue
            small = reader.decisions._episode(episode_index)
            structural = small.action_valid.any(0).bool()
            episodes.append({"task": task, "episode_id": episode_id})
            group = file[key]
            eligible = []
            for value in sorted(group["decisions"], key=int):
                index = int(value)
                _, mask = aligned_action_chunk(small.actions, small.action_valid, index, data.horizon,
                                               small.alignment, data.action_stride)
                if structural.any() and bool(mask[:, structural].all()):
                    eligible.append(index)
            if not eligible:
                exclusions.append({"task": task, "episode_id": episode_id,
                                   "reason": "cached_episode_has_no_complete_target_horizon"})
                continue
            # Cover the cached timeline deterministically rather than replacing
            # unavailable episodes from the production diagnostic panel.
            selected = np.linspace(0, len(eligible) - 1, min(maximum, len(eligible)), dtype=int)
            for offset in selected:
                index = eligible[int(offset)]
                reader.reference.reference = episode_index, index
                decision = reader.decisions.decision(0)
                state = {"task": task, "episode_id": episode_id, "episode_index": episode_index,
                         "decision_index": index, "source_local_index": int(small.execution_indices[index]),
                         "partition": data.label_partition(episode_index),
                         "phase": ("early", "middle", "late")[min(2, 3 * int(offset) // len(eligible))],
                         "structural_action_mask": structural.tolist(),
                         "valid_coordinates": int(decision.target_valid.sum()),
                         "conditioning_frames": decision.info.conditioning_frames,
                         "execution_history_frames": decision.info.execution_history_frames,
                         "diagnostic_asset_selection": True}
                state["state_id"] = digest({key: state[key] for key in ("task", "episode_id", "decision_index")})[:20]
                candidates.append(state)
    # Task/episode round-robin ensures a larger local cache is not monopolized
    # by the first episode. There is no policy-dependent selection.
    grouped = {}
    for state in candidates:
        grouped.setdefault((state["task"], state["episode_id"]), []).append(state)
    keys = sorted(grouped)
    random.Random(stable_seed(seed, "limited_cached_smoke")).shuffle(keys)
    states = [grouped[key][offset] for offset in range(maximum) for key in keys
              if offset < len(grouped[key])][:maximum]
    partitions = [{**episode, "partition": data.label_partition(next(index for index in data.by_task[episode["task"]]
                   if episode_name(data, index)[1] == episode["episode_id"]))} for episode in episodes]
    return {"states": states, "diagnostic_episodes": episodes, "training_pool": [],
            "reserved_validation_episodes": [{key: row[key] for key in ("task", "episode_id")}
                                              for row in partitions if row["partition"] == "validation"],
            "episode_partitions": partitions, "sampling_exclusions": exclusions,
            "sampling_scope": "existing_cached_decisions_only_asset_limited_correctness_check",
            "required_multitask_panel_satisfied": len(states) >= 8 and len({row["task"] for row in states}) >= 3}


class UtilityDataReader:
    """Canonical small action reader plus the existing indexed cache assembler."""
    def __init__(self, data, cfg, encoded, hierarchy):
        self.data, self.cfg = data, cfg
        self.reference = _ReferenceData(data)
        self.decisions = LightweightDecisions(self.reference)
        self.packed = PackedContextReader(cfg, encoded, hierarchy)

    def close(self):
        self.decisions.close()
        self.packed.close()

    def archive(self, state):
        from stmem.inspection.refinement_candidates import IndexedArchive
        self.reference.reference = state["episode_index"], state["decision_index"]
        decision = self.decisions.decision(0)
        if (decision.info.task, decision.info.episode_id) != (state["task"], state["episode_id"]):
            raise ValueError("State index resolves to a different episode")
        packed = self.packed
        group = packed.hierarchy_files.get(packed.hierarchy_root / f"{state['task']}.h5")[
            episode_key(state["task"], state["episode_id"])]
        entry = group["decisions"][str(state["decision_index"])]
        nfull, pending = divmod(decision.info.execution_history_frames, self.cfg["model"]["block_size"])
        expected = f"{nfull * self.cfg['model']['block_size']}_{pending}" if pending else ""
        actual = entry.attrs["prefix_key"]
        if isinstance(actual, bytes):
            actual = actual.decode()
        if int(entry.attrs["nfull"]) != nfull or actual != expected:
            raise ValueError("Causal partial-block cache does not match the selected decision")
        index = dict(packed._index(group))
        for key in ("anchor", "time_anchor", "depth", "time_max"):
            index[key] = packed._read(group["nodes"][key])
        leaves = packed._active_leaves(group, entry)
        if len(leaves) != int(entry.attrs["history_leaf_count"]):
            raise ValueError("Causal cache leaf count differs")
        execution = leaves[index["source"][leaves] == "execution"]
        if len(execution) and np.max(index["time_max"][execution]) >= decision.clocks[1]:
            raise ValueError("Current or future execution observation leaked into history")
        return decision, group, IndexedArchive(index, packed._read(entry["roots"]), leaves)

    def prepare(self, state, context_specs):
        decision, group, archive = self.archive(state)
        packed, width = self.packed, self.cfg["model"]["width"]
        columns = packed._columns(group["nodes"], archive.leaves, index=archive.index)
        base = packed.base_files.get(packed.cache_root / f"{state['task']}.h5")[
            episode_key(state["task"], state["episode_id"])]
        item = {"task": state["task"], "episode": f"{state['task']}/{state['episode_id']}",
                "decision_index": state["decision_index"], "target": decision.target.float(),
                "valid": decision.target_valid, "current_state": decision.state.float(),
                "clocks": decision.clocks, "history_leaves": len(archive.leaves),
                "text": torch.from_numpy(packed._read(base["text"])).float(),
                "current": columns_bank(packed._columns(base["current"][str(state["decision_index"])]),
                                        decision.clocks, width),
                "full": columns_bank(columns, decision.clocks, width)}
        contexts = {}
        for key, spec in context_specs.items():
            archive.validate(spec)
            if spec["reader"].upper() == "TREE":
                selected = packed._columns(group["nodes"], spec["rows"], index=archive.index)
            else:
                selected = flat_residual_columns(columns, *archive.flat_arrays(spec))
            contexts[key] = columns_bank(selected, decision.clocks, width)
            if contexts[key].content.shape[1] != archive.records_count(spec):
                raise ValueError("Actual assembled history budget differs from the manifest")
        return item, contexts


def seal_manifest(manifest):
    manifest = deepcopy(manifest)
    manifest.pop("experiment_id", None)
    manifest["experiment_id"] = digest(manifest)
    return manifest


def verify_manifest(path, *, verify_source=True, verify_dependencies=True):
    manifest = read_json(path)
    if seal_manifest(manifest)["experiment_id"] != manifest.get("experiment_id"):
        raise ValueError("Study manifest content hash differs")
    if verify_source and manifest["source"]["source_sha256"] != source_revision()["source_sha256"]:
        raise ValueError("Study source changed; create a separately named experiment")
    if reference_name(manifest) == "S_INIT":
        deps = study_dependencies(manifest)
        validate_protocol_teacher(manifest["protocol"], deps)
    if not verify_dependencies:
        return manifest
    if reference_name(manifest) == "S_INIT":
        for key in ("accepted_cache_plan", "teacher_selection"):
            if file_sha256(deps[key]) != deps[key + "_sha256"]:
                raise ValueError(f"Accepted teacher dependency changed: {key}")
    for key in ("encoded", "hierarchy"):
        root = Path(manifest["dependencies"][key])
        if file_sha256(root / "manifest.json") != manifest["dependencies"][f"{key}_manifest_sha256"]:
            raise ValueError(f"Study {key} manifest changed")
    for name, guard in manifest["dependencies"].get("verified_shards", {}).items():
        path = Path(name)
        stat = path.stat()
        if stat.st_size != guard["bytes"]:
            raise ValueError(f"Immutable cache shard size changed: {path}")
        if stat.st_mtime_ns != guard["mtime_ns"] and file_sha256(path) != guard["sha256"]:
            raise ValueError(f"Immutable cache shard content changed: {path}")
    return manifest


def validate_protocol_teacher(protocol, dependencies):
    """Bind optional accepted-result pins before allocating or selecting states."""
    accepted = protocol.get("accepted_teacher")
    if accepted is not None:
        if (accepted.get("sha256") != dependencies["teacher_sha256"]
                or accepted.get("step") != dependencies["teacher_step"]):
            raise ValueError("Protocol accepted_teacher differs from the adopted checkpoint")
    if dependencies.get("reference") == TEACHER_REFERENCE:
        if tuple(protocol.get("primary_endpoints", ())) != NEW_STUDENTS:
            raise ValueError("A new teacher study requires exactly S_STD, S_STAGE and S_PACKET primary endpoints")


def build_manifest(cfg, protocol, dependencies, output):
    validate_protocol_teacher(protocol, dependencies)
    data = TrainingData(cfg)
    if not dependencies["diagnostic_assets"]:
        actual = read_json(Path(dependencies["encoded"]) / "manifest.json")["identity"]["data_identity"]
        if data.identity != actual:
            raise ValueError("Action-index dataset identity differs from the accepted encoded cache")
    reader = UtilityDataReader(data, cfg, dependencies["encoded"], dependencies["hierarchy"])
    states = []
    try:
        panel = (select_cached_diagnostic_states(data, reader, seed=protocol["seed"])
                 if dependencies["diagnostic_assets"] else
                 select_states(data, seed=protocol["seed"],
                               episodes_per_partition=protocol.get("episodes_per_partition", 4)))
        for state in panel["states"]:
            try:
                _, _, archive = reader.archive(state)
                state.update(available_leaf_count=len(archive.leaves), root_count=len(archive.roots),
                             source_counts=dict(Counter(map(str, archive.index["source"][archive.leaves]))),
                             cache_eligible=True)
            except KeyError as exc:
                state.update(cache_eligible=False, cache_ineligible_reason=str(exc))
            states.append(state)
    finally:
        reader.close()
    panel["states"] = states
    # Two gain-blind cases/task, preferring distinct episodes. Fixed temporal
    # strata ensure stop/count and destination phases are represented rather
    # than always choosing the first episode's early/middle states. These labels
    # declare the question, never assert that a semantic cue is present.
    capability, capability_roles = [], []
    counting = {"BinFill", "PickXtimes", "SwingXtimes", "StopCube"}
    destination = {"VideoPlaceButton", "VideoPlaceOrder", "MoveCube", "InsertPeg", "RouteStick"}
    for task in data.tasks:
        used_episodes = set()
        phases = ("late", "middle") if task in counting else (("middle", "late") if task in destination else ("early", "middle"))
        for phase in phases:
            candidates = [state for state in states if state["task"] == task and state["cache_eligible"]
                          and state["state_id"] not in capability]
            candidates.sort(key=lambda state: (state["episode_id"] in used_episodes,
                                               state["phase"] != phase, state["partition"] != "train",
                                               state["episode_id"], state["decision_index"]))
            if not candidates:
                continue
            selected = candidates[0]
            capability.append(selected["state_id"])
            used_episodes.add(selected["episode_id"])
            capability_roles.append({"state_id": selected["state_id"], "task": task,
                                     "requested_phase": phase, "actual_phase": selected["phase"],
                                     "question": "stop_or_count" if task in counting else (
                                         "destination" if task in destination else "target_or_sequence"),
                                     "selection": "gain_blind_task_and_phase_stratum"})
    capability = capability[:32]
    capability_roles = capability_roles[:32]
    reference_metadata = ({"reference": deepcopy(dependencies["reference"]),
                           "primary_endpoints": list(NEW_STUDENTS)}
                          if dependencies.get("reference") == TEACHER_REFERENCE else {})
    manifest = seal_manifest({"schema_version": 1, "source": source_revision(),
                             **reference_metadata,
                             **{k: dependencies[k] for k in ("teacher_sha256", "compact_sha256",
                                "encoded_cache_id", "hierarchy_cache_id")},
                             "dependencies": dependencies, "configuration": cfg, "protocol": protocol,
                             **panel, "capability_states": capability, "capability_roles": capability_roles,
                             "kind": "limited_real_data_diagnostic" if dependencies["diagnostic_assets"]
                             else "matched_student_refinement_utility",
                             "phase_status": "states_fixed_before_policy_losses",
                             "confirmation_status": "reserved_not_inspected",
                             "training_pool_counts": dict(Counter(r["task"] for r in panel["training_pool"])),
                             "note": "Held out from NEW student training; inherited teacher/reference may have seen episodes."})
    if reference_metadata:
        study_dependencies(manifest)
    output = Path(output)
    if output.exists() and read_json(output) != manifest:
        raise FileExistsError("Existing manifest differs; choose a new study directory")
    atomic_json(manifest, output)
    return manifest
