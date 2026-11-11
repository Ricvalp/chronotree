"""Bounded fixed-manifest student × allocation measurements; no optimization.

Manifest and context construction are separate from model measurement. Each
student has independent immutable rows, sharing teacher references and draws.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import yaml

from stmem.config import digest, load_config
from stmem.inspection.refinement_data import (
    UtilityDataReader, build_manifest, primary_students, read_json, reference_identity,
    reference_name, stable_seed, teacher_identity, verify_dependencies, verify_manifest,
)
from stmem.inspection.refinement_statistics import analyze_panel, population_summary
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.runtime import load_policy, validate_policy_config


@contextmanager
def lock(path, *, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def normalized_panel(panel):
    groups = []
    for family in panel["families"]:
        cost = family["final_budget"] - family["background"]
        groups.append({"reader": family["reader"], "family": family["panel"],
                       "background": family["background"], "cost": cost,
                       "paired": family.get("paired", False),
                       "base": family["base"], "controls": family["controls"],
                       "candidates": [{"id": c["candidate_id"], "first": c["prefixes"]["1"],
                                       "complete": c["prefixes"][str(cost)],
                                       **{key: c.get(key) for key in ("source", "depth", "support_count")}}
                                      for c in family["candidates"]]})
    return {"groups": groups}


def build_contexts(manifest_path):
    manifest = verify_manifest(manifest_path)
    root = Path(manifest_path).parent
    dep, protocol = manifest["dependencies"], manifest["protocol"]
    data = TrainingData(manifest["configuration"])
    reader = UtilityDataReader(data, manifest["configuration"], dep["encoded"], dep["hierarchy"])
    counts, eligibility = {}, []
    try:
        for state in manifest["states"]:
            if not state["cache_eligible"]:
                eligibility.append({"state_id": state["state_id"], "eligible": False,
                                    "reason": state["cache_ineligible_reason"]})
                continue
            _, _, archive = reader.archive(state)
            panel = archive.diagnostic_panel(
                stable_seed(protocol["seed"], state["state_id"], "candidates"),
                budgets=tuple(protocol["budgets"]), packet_size=protocol["packet_cost"],
                max_candidates=protocol["max_candidates"], audit_candidates=protocol["audit_candidates"])
            capability = {}
            if state["state_id"] in manifest["capability_states"]:
                for family, budget, mode in (("ordinary128", 128, "ordinary"),
                                              ("detailed128", 128, "packet"),
                                              ("ordinary256", 256, "ordinary"),
                                              ("all_leaves", len(archive.leaves), "full")):
                    try:
                        seed = stable_seed(protocol["seed"], state["state_id"], family)
                        if mode == "ordinary":
                            context = archive.ordinary("TREE", budget, seed)
                        elif mode == "packet":
                            context = archive.route("TREE", budget, seed, mode="packet")["contexts"][
                                min(budget, len(archive.leaves))]
                        else:
                            context = archive.context("TREE", archive.leaves)
                        panel["contexts"][context["context_id"]] = context
                        capability[family] = context["context_id"]
                    except ValueError as exc:
                        capability[family] = {"omitted": str(exc)}
            panel.update(state_id=state["state_id"], experiment_id=manifest["experiment_id"],
                         capability=capability, detailed_prefixes=[1, 2, 4, 8])
            panel["panel_sha256"] = digest(panel)
            path = root / "contexts" / f"{state['state_id']}.json"
            if path.exists() and read_json(path) != panel:
                raise ValueError("Immutable candidate construction changed")
            atomic_json(panel, path)
            counts[state["state_id"]] = len(panel["contexts"])
            eligibility.extend({"state_id": state["state_id"], **row} for row in panel["eligibility"])
    finally:
        reader.close()
    draws = protocol["discovery_draws"] + protocol["confirmation_draws"]
    student_steps = protocol.get("student_training", {}).get("steps", 30000)
    if not isinstance(student_steps, int) or isinstance(student_steps, bool) or student_steps < 1:
        raise ValueError("Student training steps must be a positive declared integer")
    students = primary_students(manifest)
    result = {"experiment_id": manifest["experiment_id"], "status": "complete",
              "context_counts": counts, "eligibility": eligibility,
              "primary_students": list(students), "primary_student_count": len(students),
              "compact_model_examples_per_student": sum(counts.values()) * draws,
              "compact_model_examples_primary_students": len(students) * sum(counts.values()) * draws,
              "teacher_model_examples_shared": sum(bool(count) for count in counts.values()) * draws,
              "primary_training_updates": 3 * student_steps,
              "primary_training_example_exposures": 3 * student_steps * 128,
              "prefixes_2_4": "Included for all eligible states in the exact context counts above",
              "capability": "Included in exact context counts; teacher FULL reused",
              "numerical_control_extra": "At most 2 states/student, 4 contexts, serial+packed+FP32; counted in smoke receipt",
              "generated_extra_bound": protocol["max_generated_cases"] * 2 * 8 * 6 * len(students),
              "walltime_estimate": "Unmeasured until real-data smoke; context count is not a GPU launch count"}
    if len(students) == 4:
        result["compact_model_examples_four_students"] = result["compact_model_examples_primary_students"]
    atomic_json(result, root / "contexts_complete.json")
    return result


def read_panel(root, state_id, experiment_id):
    panel = read_json(root / "contexts" / f"{state_id}.json")
    body = {k: v for k, v in panel.items() if k != "panel_sha256"}
    if panel["panel_sha256"] != digest(body) or panel["experiment_id"] != experiment_id:
        raise ValueError("Candidate manifest changed")
    return panel


def save_rows(path, rows, metadata):
    """One compact immutable transaction per state; completion marker is last."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    fields = list(rows[0]) if rows else []
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **{field: np.asarray([r[field] for r in rows]) for field in fields})
    os.replace(temporary, path)
    atomic_json({**metadata, "rows": len(rows), "rows_sha256": file_sha256(path)}, path.with_suffix(".json"))


def load_rows(path):
    metadata = read_json(Path(path).with_suffix(".json"))
    if file_sha256(path) != metadata["rows_sha256"]:
        raise ValueError("Stored utility row checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        if not arrays.files:
            return [], metadata
        rows = [{key: arrays[key][i].item() for key in arrays.files} for i in range(metadata["rows"])]
    return rows, metadata


def choose_smoke(states, maximum):
    """Gain-blind round-robin over tasks, prefer distinct episodes first."""
    result, seen = [], set()
    requirements = (
        lambda row: int(row.get("execution_history_frames", 0)) % 8 != 0,
        lambda row: int(row.get("conditioning_frames", 0)) > 0,
        lambda row: int(row.get("execution_history_frames", 0)) > 0,
        lambda row: any(not value for value in row.get("structural_action_mask", [])),
    )
    for requirement in requirements:
        if any(requirement(row) for row in result):
            continue
        candidate = next((row for row in states if requirement(row) and row not in result), None)
        if candidate is not None and len(result) < maximum:
            result.append(candidate)
            seen.add((candidate["task"], candidate["episode_id"]))
    for unique_episodes in (True, False):
        for offset in range(maximum):
            for task in dict.fromkeys(s["task"] for s in states):
                candidates = [s for s in states if s["task"] == task and s not in result
                              and (not unique_episodes or (s["task"], s["episode_id"]) not in seen)]
                if candidates and len(result) < maximum:
                    result.append(candidates[0])
                    seen.add((task, candidates[0]["episode_id"]))
    return result


def validate_student_endpoint(manifest, manifest_path, student_name, state, partition):
    """Do not relabel a different sampler, panel, or initialization as an arm."""
    teacher_hash, teacher_step = teacher_identity(manifest)
    reference_hash, reference_step = reference_identity(manifest)
    if student_name == "S_INIT":
        if reference_name(manifest) != "S_INIT" or partition != "train":
            raise ValueError("S_INIT is only the declared fitting-partition preflight diagnostic")
        if state.get("stage") != "teacher" or state.get("step") != teacher_step:
            raise ValueError("S_INIT must use the exact adopted teacher, with zero compact updates")
        return {"initialization": "frozen_teacher_compact_smoke", "endpoint": teacher_step,
                "initialization_sha256": teacher_hash, "compact_updates": 0, "preflight_only": True}
    if student_name not in primary_students(manifest):
        raise ValueError("Requested student is outside this study's primary arms")
    if student_name == "S0":
        if state.get("step") != reference_step:
            raise ValueError("S0 must remain the unchanged production reference checkpoint")
        return {"initialization": "production_reference", "endpoint": reference_step}
    study = state["config"].get("training", {}).get("utility_study", {})
    expected = {"variant": student_name, "experiment_id": manifest["experiment_id"],
                "manifest_sha256": file_sha256(manifest_path), "teacher_sha256": teacher_hash,
                "encoded_cache_id": manifest["encoded_cache_id"],
                "hierarchy_cache_id": manifest["hierarchy_cache_id"]}
    for key, value in expected.items():
        if study.get(key) != value:
            raise ValueError(f"Student {key} does not match the fixed study arm")
    if state.get("provenance", {}).get("dependencies", {}).get("teacher") != teacher_hash:
        raise ValueError("Student has a different accepted-teacher dependency")
    initialization = study.get("initialization")
    if initialization == "accepted_teacher_weights":
        explicit = "primary_updates" in study or "development_steps" in study
        if explicit and not all(key in study for key in ("primary_updates", "development_steps")):
            raise ValueError("Student schedule must declare both primary_updates and development_steps")
        endpoint = study["primary_updates"] if explicit else 30000
        development = study["development_steps"] if explicit else [15000]
        if (not isinstance(endpoint, int) or isinstance(endpoint, bool) or endpoint < 1
                or not isinstance(development, list)
                or any(not isinstance(step, int) or isinstance(step, bool) or not 0 < step < endpoint
                       for step in development)
                or development != sorted(set(development))):
            raise ValueError("Student primary/development checkpoint schedule is invalid")
        declared_steps = manifest.get("protocol", {}).get("student_training", {}).get("steps")
        if declared_steps is not None and endpoint != declared_steps:
            raise ValueError("Student endpoint differs from the study's declared training duration")
        if explicit:
            training = state["config"]["training"]
            if training.get("steps") != endpoint or training.get("schedule_steps") != endpoint:
                raise ValueError("Student optimizer and cosine schedule must use the declared primary endpoint")
        if study.get("initialization_sha256") != teacher_hash or study.get("optimizer_initialization") != "fresh":
            raise ValueError("Primary students require matched fresh teacher initialization")
        if state["step"] not in (*development, endpoint):
            raise ValueError("Primary students require a predeclared development or primary checkpoint")
        if state["step"] != endpoint and partition != "train":
            raise ValueError("Earlier checkpoints are development-only; final confirmation uses the declared primary endpoint")
    elif initialization == "S0_weights":
        endpoint = int(study.get("continuation_updates") or 0)
        if (reference_name(manifest) != "S0" or endpoint < 1 or state["step"] != endpoint
                or study.get("initialization_sha256") != reference_hash):
            raise ValueError("Continuation must use its predeclared common S0 endpoint")
    else:
        raise ValueError("Unknown student initialization; do not mix unmatched arms")
    return {"initialization": initialization, "endpoint": endpoint,
            "initialization_sha256": study["initialization_sha256"],
            "training_manifest_sha256": study["manifest_sha256"]}


def validate_measurement_models(manifest, student_name, teacher, teacher_state, student, student_state,
                                *, checkpoint_hash, mode, partition):
    """Permit an exact teacher-on-compact-context smoke without relabeling it S0.

    Production compact checkpoints retain the existing strict stage/frozen-tensor
    checks. S_INIT is a separate zero-update diagnostic, never a primary arm.
    """
    from stmem.training.label_experiment import validate_models

    if student_name != "S_INIT":
        validate_models(manifest["configuration"], teacher, teacher_state, student, student_state)
        return
    teacher_hash, teacher_step = teacher_identity(manifest)
    if (reference_name(manifest) != "S_INIT" or mode != "smoke" or partition != "train"
            or checkpoint_hash != teacher_hash):
        raise ValueError("S_INIT requires the exact adopted teacher and smoke/train only")
    for state in (teacher_state, student_state):
        if state.get("stage") != "teacher" or state.get("step") != teacher_step:
            raise ValueError("S_INIT must be the unchanged adopted teacher with zero compact updates")
        validate_policy_config(state["config"], manifest["configuration"])
    cfg = manifest["configuration"]
    if cfg["model"].get("decoder", "flow") != "flow" or cfg["model"].get("history_mode", "full") != "full":
        raise ValueError("S_INIT requires the adopted full-history flow teacher")
    teacher_weights, student_weights = teacher.state_dict(), student.state_dict()
    if teacher_weights.keys() != student_weights.keys() or any(
            not torch.equal(value.cpu(), student_weights[key].cpu()) for key, value in teacher_weights.items()):
        raise ValueError("S_INIT teacher and compact-context probe must have identical model tensors")


def combine_reference_timings(compact_timings, reference_timings):
    """Account once for separately cached teacher computation and peak memory."""
    result = dict(compact_timings)
    for key in ("teacher_context_seconds", "teacher_forward_seconds", "teacher_model_examples",
                "identity_seconds", "wall_seconds"):
        result[key] = result.get(key, 0) + reference_timings.get(key, 0)
    for key in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
        result[key] = max(result.get(key, 0), reference_timings.get(key, 0))
    result["reference_compute_wall_seconds"] = reference_timings.get("wall_seconds", 0.)
    return result


def empty_measurement_timings():
    return {key: 0 for key in ("teacher_context_seconds", "teacher_forward_seconds",
                              "compact_context_seconds", "compact_forward_seconds",
                              "wall_seconds", "identity_seconds", "compact_model_examples",
                              "teacher_model_examples", "contexts_adapted",
                              "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes")}


def measurement_directory(root, student, step, mode, partition):
    return Path(root) / "measurements" / student / f"step_{int(step):08d}" / mode / partition


def confirmation_extension(manifest, root, *, student, step, checkpoint_hash, partition,
                           base_directory, additional_draws):
    """Bind one bounded fresh bank to immutable, already completed primary rows.

    This extension never replaces discovery/confirmation or supplies fresh
    choices to the primary report. Keeping the whole base artifact inventory
    sealed also prevents an opportunistically changing subset on resume.
    """
    cap = min(int(manifest["protocol"].get("additional_draw_cap", 16)), 16)
    if not 1 <= additional_draws <= cap:
        raise ValueError(f"Additional confirmation draws must be in 1..{cap}")
    base = Path(base_directory).resolve()
    identity_path = base / "identity.json"
    identity = read_json(identity_path)
    if identity.get("mode") not in ("main", "smoke"):
        raise ValueError("Confirmation extensions must bind original main/smoke measurements")
    expected = {"experiment_id": manifest["experiment_id"], "student": student,
                "checkpoint_sha256": checkpoint_hash, "step": step, "partition": partition,
                "precision": manifest["protocol"]["precision"]}
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Confirmation extension base differs from the frozen student/panel/partition/precision")
    if base != measurement_directory(root, student, step, identity["mode"], partition).resolve():
        raise ValueError("Confirmation extension base must belong to this study's primary measurement directory")
    artifacts = {}
    states = {s["state_id"]: s for s in manifest["states"] if s["cache_eligible"] and
              (partition == "both" or s["partition"] == partition)}
    for state_id in sorted(states):
        path = base / f"{state_id}.npz"
        if not path.with_suffix(".json").exists():
            continue
        rows, metadata = load_rows(path)
        if metadata["identity"] != identity or metadata["state"] != states[state_id]:
            raise ValueError("Confirmation extension base contains mixed measurement identities")
        if rows and {row["bank"] for row in rows} != {"discovery", "confirmation"}:
            raise ValueError("Confirmation extension requires both original banks and never another extension")
        panel = read_panel(Path(root), state_id, manifest["experiment_id"])
        if metadata["panel_sha256"] != panel["panel_sha256"]:
            raise ValueError("Confirmation extension candidate contexts differ from the primary panel")
        artifacts[state_id] = {"rows_sha256": file_sha256(path),
                               "metadata_sha256": file_sha256(path.with_suffix(".json")),
                               "panel_sha256": panel["panel_sha256"]}
    if not artifacts:
        raise ValueError("Confirmation extension needs at least one committed primary state")
    return {"bank": "confirmation_extension", "draws": additional_draws, "cap": cap,
            "base_directory": str(base), "base_identity_sha256": file_sha256(identity_path),
            "base_state_artifacts": artifacts,
            "analysis_policy": "raw_only_separate_bank; original discovery choices remain fixed; never pooled into the primary report"}


def measurement_draws(protocol, state, extension=None):
    from stmem.inspection.refinement_measurements import draw_plan
    extra = {} if extension is None else {extension["bank"]: extension["draws"]}
    draws = draw_plan(protocol["seed"], state["task"], state["episode_id"], state["decision_index"],
                      discovery=protocol["discovery_draws"], confirmation=protocol["confirmation_draws"],
                      extra_banks=extra)
    return draws if extension is None else [draw for draw in draws if draw["bank"] == extension["bank"]]


def measure(args):
    from stmem.inspection.refinement_measurements import (
        action_mask_summary, evaluate_state, numerical_controls,
    )
    manifest = verify_manifest(args.manifest)
    root = args.manifest.parent
    complete = read_json(root / "contexts_complete.json")
    if complete["experiment_id"] != manifest["experiment_id"]:
        raise ValueError("Contexts were not fixed for this experiment")
    dep, protocol = manifest["dependencies"], manifest["protocol"]
    teacher_hash, _ = teacher_identity(manifest)
    reference_hash, _ = reference_identity(manifest)
    args.student = args.student or reference_name(manifest)
    if args.student == "S_INIT" and (args.mode != "smoke" or args.partition != "train"):
        raise ValueError("S_INIT is a smoke/train-only preflight, not a primary measurement")
    checkpoint = args.checkpoint or Path(dep["compact"])
    checkpoint_hash = file_sha256(checkpoint)
    if args.student == reference_name(manifest) and checkpoint_hash != reference_hash:
        raise ValueError("Reference must be the unchanged manifest-pinned checkpoint")
    if file_sha256(dep["teacher"]) != teacher_hash:
        raise ValueError("Accepted teacher changed")
    teacher, tstate = load_policy(dep["teacher"], args.device)
    student, sstate = load_policy(checkpoint, args.device)
    endpoint = validate_student_endpoint(manifest, args.manifest, args.student, sstate, args.partition)
    validate_measurement_models(manifest, args.student, teacher, tstate, student, sstate,
                                checkpoint_hash=checkpoint_hash, mode=args.mode, partition=args.partition)
    for model in (teacher, student):
        model.freeze_encoders()
        model.eval().requires_grad_(False)
    extension = None
    if args.mode == "extension":
        extension = confirmation_extension(
            manifest, root, student=args.student, step=sstate["step"], checkpoint_hash=checkpoint_hash,
            partition=args.partition, base_directory=args.extension_base, additional_draws=args.additional_draws)
    output = measurement_directory(root, args.student, sstate["step"], args.mode, args.partition)
    identity = {"experiment_id": manifest["experiment_id"], "student": args.student,
                "checkpoint_sha256": checkpoint_hash, "checkpoint_path": str(Path(checkpoint).resolve()),
                "step": sstate["step"],
                "precision": protocol["precision"], "partition": args.partition,
                "max_states": args.max_states, "mode": args.mode, "workers": args.workers,
                "limited_check": args.limited_check, **endpoint}
    if extension is not None:
        identity["extension"] = extension
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    with lock(output / "identity.lock"):
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("Measurement identity changed; use a new named study/student stage")
        atomic_json(identity, identity_path)
    states = [s for s in manifest["states"] if s["cache_eligible"] and
              (args.partition == "both" or s["partition"] == args.partition)]
    if extension is not None:
        states = [s for s in states if s["state_id"] in extension["base_state_artifacts"]]
    if args.mode == "smoke":
        states = choose_smoke(states, args.max_states or 8)
        block = manifest["configuration"]["model"]["block_size"]
        complete_smoke = (8 <= len(states) <= 16 and len({s["task"] for s in states}) >= 3
                          and any(s["execution_history_frames"] % block for s in states)
                          and any(s["conditioning_frames"] for s in states)
                          and any(s["execution_history_frames"] for s in states))
        if not args.limited_check and not complete_smoke:
            raise ValueError("Required smoke needs 8–16 states across ≥3 tasks, both source banks and a causal partial block; use --limited-check only for an explicitly incomplete asset check")
    elif args.max_states is not None:
        states = states[:args.max_states]
    selected = [s for i, s in enumerate(states) if i % args.workers == args.worker_index]
    data = TrainingData(manifest["configuration"])
    reader = UtilityDataReader(data, manifest["configuration"], dep["encoded"], dep["hierarchy"])
    stopped = False
    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
    original = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    started, timings, done = time.monotonic(), [], []
    try:
        with lock(output / f"worker_{args.worker_index}.lock"):
            for state in selected:
                if stopped or time.monotonic() - started >= args.max_hours * 3600:
                    break
                path = output / f"{state['state_id']}.npz"
                panel = read_panel(root, state["state_id"], manifest["experiment_id"])
                if path.with_suffix(".json").exists():
                    _, saved = load_rows(path)
                    if saved["identity"] != identity or saved["panel_sha256"] != panel["panel_sha256"]:
                        raise ValueError("Committed state belongs to different measurement/candidate identity")
                    done.append(state["state_id"])
                    continue
                begin = time.monotonic()
                item, contexts = reader.prepare(state, panel["contexts"])
                preparation_seconds = time.monotonic() - begin
                draws = measurement_draws(protocol, state, extension)
                reference_inputs = {key: dep[key] for key in (
                    "encoded_manifest_sha256", "hierarchy_manifest_sha256")}
                if contexts:
                    reference_root = root / "teacher_references"
                    if extension is not None:
                        reference_root = reference_root / extension["bank"] / f"draws_{extension['draws']:02d}"
                    reference_path = reference_root / f"{state['state_id']}.pt"
                    reference_timings = {}
                    with lock(reference_path.with_suffix(".lock"), blocking=True):
                        reference = (torch.load(reference_path, weights_only=False, map_location="cpu")
                                     if reference_path.exists() else None)
                        if reference is None:
                            reference_result = evaluate_state(
                                teacher, student, item, {}, draws, teacher_hash=teacher_hash,
                                student_hash=checkpoint_hash, precision=protocol["precision"],
                                prefix_targets=protocol.get("execute_targets", 10),
                                reference_identity=reference_inputs, reference_only=True)
                            reference, reference_timings = reference_result["teacher_reference"], reference_result["timings"]
                            temporary = reference_path.with_suffix(".partial.pt")
                            torch.save(reference, temporary)
                            os.replace(temporary, reference_path)
                    result = evaluate_state(teacher, student, item, contexts, draws, teacher_hash=teacher_hash,
                                            student_hash=checkpoint_hash, precision=protocol["precision"],
                                            prefix_targets=protocol.get("execute_targets", 10),
                                            compact_batch_size=args.batch_size, reference=reference,
                                            reference_identity=reference_inputs)
                    result["timings"] = combine_reference_timings(result["timings"], reference_timings)
                else:
                    result = {"rows": [], "timings": empty_measurement_timings(),
                              "mask": action_mask_summary(item)}
                controls = None
                if contexts and args.mode == "smoke" and len(done) < 2:
                    controls = numerical_controls(teacher, student, item, dict(list(contexts.items())[:4]), draws[:1],
                                                  teacher_hash=teacher_hash, student_hash=checkpoint_hash,
                                                  prefix_targets=protocol.get("execute_targets", 10),
                                                  precision=protocol["precision"])
                result["timings"]["cache_context_assembly_seconds"] = preparation_seconds
                metadata = {"identity": identity, "state": state, "panel_sha256": panel["panel_sha256"],
                            "timings": result["timings"], "numerical_controls": controls, "mask": result["mask"]}
                save_rows(path, result["rows"], metadata)
                timings.append(result["timings"])
                done.append(state["state_id"])
                print(json.dumps({"event": "utility_state_complete", "student": args.student,
                                  "state_id": state["state_id"], "rows": len(result["rows"]),
                                  "timings": result["timings"]}), flush=True)
    except BaseException as error:
        atomic_json({"identity": identity, "status": "failed", "completed_states": done,
                     "planned_states": [state["state_id"] for state in selected],
                     "error_type": type(error).__name__, "error": str(error),
                     "wall_seconds": time.monotonic() - started},
                    output / f"worker_{args.worker_index}.json")
        raise
    finally:
        reader.close()
        for sig, handler in original.items():
            signal.signal(sig, handler)
    receipt = {"identity": identity, "completed_states": done, "planned_states": [s["state_id"] for s in selected],
               "status": "complete" if len(done) == len(selected) else "partial",
               "wall_seconds": time.monotonic() - started, "timings": timings,
               "required_multitask_smoke": args.mode == "smoke" and not args.limited_check}
    atomic_json(receipt, output / f"worker_{args.worker_index}.json")
    return receipt


def report(manifest_path):
    from stmem.inspection.refinement_report import write_report
    manifest = verify_manifest(manifest_path, verify_source=False, verify_dependencies=False)
    root, analyses, inventory = Path(manifest_path).parent, [], []
    expected_states = {s["state_id"] for s in manifest["states"] if s.get("cache_eligible", True)}
    exclusions = [{"state_id": s["state_id"], "reason": s.get("cache_ineligible_reason")}
                  for s in manifest["states"] if s.get("cache_eligible") is False]
    primary = primary_students(manifest)
    reported = tuple(dict.fromkeys((*primary, reference_name(manifest))))
    for student in reported:
        student_root = root / "measurements" / student
        all_paths = sorted(path for mode in ("main", "smoke")
                           for path in student_root.glob(f"step_*/{mode}/*/*.npz"))
        uncommitted = [str(path) for path in all_paths if not path.with_suffix(".json").is_file()]
        all_paths = [path for path in all_paths if path.with_suffix(".json").is_file()]
        declarations = set()
        for path in all_paths:
            identity = read_json(path.with_suffix(".json"))["identity"]
            if (identity.get("experiment_id") != manifest["experiment_id"] or identity.get("student") != student
                    or identity.get("step") != int(path.parents[2].name.removeprefix("step_"))
                    or identity.get("mode") != path.parents[1].name):
                raise ValueError("Report measurement path/study/student identity differs")
            endpoint, initialization = identity.get("endpoint"), identity.get("initialization")
            if (not isinstance(endpoint, int) or isinstance(endpoint, bool) or endpoint < 1
                    or initialization not in ("production_reference", "accepted_teacher_weights", "S0_weights",
                                               "frozen_teacher_compact_smoke")):
                raise ValueError("Report requires a recorded initialization and declared endpoint")
            if student == "S_INIT" and (identity.get("mode") != "smoke" or not identity.get("preflight_only")
                                         or identity.get("partition") != "train"):
                raise ValueError("S_INIT rows must remain a fitting-only smoke diagnostic")
            declarations.add((initialization, identity.get("initialization_sha256"), endpoint,
                              identity.get("training_manifest_sha256")))
        if len(declarations) > 1:
            raise ValueError(f"Mixed initializations or declared training endpoints for {student}; use separate study roots")
        declaration = next(iter(declarations), None)
        initialization = declaration[0] if declaration else None
        primary_step = declaration[2] if declaration else None
        paths = [path for path in all_paths if path.parents[1].name == "main"]
        mode = "main" if paths else "smoke"
        if not paths:
            paths = [path for path in all_paths if path.parents[1].name == "smoke"]
        # Do not pool a development checkpoint with the frozen final endpoint.
        steps = {int(path.parents[2].name.removeprefix("step_")) for path in paths}
        chosen_step = primary_step if primary_step in steps else max(steps, default=None)
        paths = [path for path in paths if int(path.parents[2].name.removeprefix("step_")) == chosen_step]
        inventory.append({"student": student, "mode": mode, "measured_states": 0,
                          "step": chosen_step, "initialization": initialization, "declared_endpoint": primary_step,
                          "primary_endpoint": student in primary and chosen_step is not None and chosen_step == primary_step,
                          "preflight_only": student not in primary,
                          "uncommitted_rows": uncommitted,
                          "expected_states": len(expected_states), "excluded_states": exclusions,
                          "status": ("no_eligible_states" if not expected_states else
                                     "not_run" if not paths else "partial_or_diagnostic")})
        seen = set()
        # A both-partitions rerun is a repeat, never extra independent evidence.
        paths.sort(key=lambda path: (path.parent.name == "both", str(path)))
        for path in paths:
            rows, metadata = load_rows(path)
            identity = metadata["identity"]
            if identity["experiment_id"] != manifest["experiment_id"] or identity["student"] != student:
                raise ValueError("Report encountered rows belonging to a different frozen study/student")
            state = {**metadata["state"], "student": student}
            if state["state_id"] in seen:
                continue
            seen.add(state["state_id"])
            panel = read_panel(root, state["state_id"], manifest["experiment_id"])
            if metadata["panel_sha256"] != panel["panel_sha256"]:
                raise ValueError("Report candidate identities differ from committed measurements")
            inventory[-1]["measured_states"] += 1
            result = analyze_panel(state, normalized_panel(panel), rows,
                                   epsilon=manifest["protocol"]["gain_epsilon"])
            result.update(state=state, identity=identity, timings=metadata["timings"], capability=panel["capability"],
                          raw_rows=rows, numerical_controls=metadata["numerical_controls"], mask=metadata["mask"])
            analyses.append(result)
        if (student in primary and mode == "main" and chosen_step == primary_step and expected_states and
                expected_states <= seen and all(entry["reason"] for entry in exclusions)):
            inventory[-1]["status"] = "complete"
    extensions = []
    for path in sorted((root / "measurements").glob("*/step_*/extension/*/identity.json")):
        identity = read_json(path)
        if identity.get("experiment_id") != manifest["experiment_id"] or identity.get("mode") != "extension":
            raise ValueError("Report encountered a mismatched confirmation extension")
        extension = identity["extension"]
        state_count, row_count = 0, 0
        for rows_path in sorted(path.parent.glob("*.npz")):
            if not rows_path.with_suffix(".json").is_file():
                continue
            rows, metadata = load_rows(rows_path)
            if metadata["identity"] != identity or any(row["bank"] != extension["bank"] for row in rows):
                raise ValueError("Report encountered mixed confirmation extension banks")
            state_count += 1
            row_count += len(rows)
        extensions.append({"student": identity["student"], "step": identity["step"],
                           "partition": identity["partition"], "bank": extension["bank"],
                           "draws": extension["draws"], "measured_states": state_count,
                           "raw_rows": row_count, "directory": str(path.parent),
                           "status": "raw_only", "note": extension["analysis_policy"]})
    result = {"experiment_id": manifest["experiment_id"], "kind": manifest["kind"],
              "inventory": inventory,
              "population": population_summary([a for a in analyses if a["state"]["student"] in primary],
                                               epsilon=manifest["protocol"]["gain_epsilon"],
                                               relative_thresholds=manifest["protocol"].get(
                                                   "relative_thresholds", (.005, .01, .05))),
              "analyses": analyses, "confirmation_extensions": extensions}
    atomic_json(result, root / "analysis.json")
    write_report(manifest, result, root)
    return {"report": str(root / "report.md"), "inventory": inventory}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("manifest", "contexts", "smoke", "main", "extension", "report"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/inspection/refinement_utility_pilot.yaml"))
    parser.add_argument("--data-config", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--compact", type=Path)
    parser.add_argument("--encoded", type=Path)
    parser.add_argument("--hierarchy", type=Path)
    parser.add_argument("--diagnostic-assets", action="store_true")
    parser.add_argument("--teacher-reference", action="store_true",
                        help="Use the adopted teacher as a zero-update compact-context smoke, keeping old S0 separate")
    parser.add_argument("--cache-plan", type=Path, help="Verified adopted teacher cache plan for --teacher-reference")
    parser.add_argument("--student", choices=("S0", "S_INIT", "S_STD", "S_STAGE", "S_PACKET"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--partition", choices=("train", "validation", "both"), default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--limited-check", action="store_true")
    parser.add_argument("--extension-base", type=Path,
                        help="Committed primary main/smoke measurement directory to keep fixed")
    parser.add_argument("--additional-draws", type=int,
                        help="Independent confirmation-extension draws (1..16, protocol cap applies)")
    args = parser.parse_args(argv)
    if not 0 <= args.worker_index < args.workers or args.batch_size < 1 or args.max_hours <= 0:
        parser.error("Invalid worker, batch or wall-time bound")
    if args.max_states is not None and not 1 <= args.max_states <= 256:
        parser.error("--max-states must be in 1..256")
    if args.mode == "smoke" and args.max_states is not None and args.max_states > 16:
        parser.error("The bounded smoke accepts at most 16 states")
    if args.mode == "extension":
        if args.extension_base is None or args.additional_draws is None:
            parser.error("An extension needs --extension-base and --additional-draws")
        if not 1 <= args.additional_draws <= 16:
            parser.error("--additional-draws must be in 1..16")
    elif args.extension_base is not None or args.additional_draws is not None:
        parser.error("Extension options belong only to --mode extension")
    if args.mode == "manifest":
        if any(getattr(args, key) is None for key in ("data_config", "teacher", "compact", "encoded", "hierarchy")):
            parser.error("Manifest construction needs --data-config --teacher --compact --encoded --hierarchy")
        cfg, protocol = load_config(args.data_config), yaml.safe_load(args.config.read_text())
        deps = verify_dependencies(cfg, args.teacher, args.compact, args.encoded, args.hierarchy,
                                   diagnostic_assets=args.diagnostic_assets,
                                   teacher_reference=args.teacher_reference, cache_plan=args.cache_plan)
        result = build_manifest(cfg, protocol, deps, args.manifest)
        result = {"manifest": str(args.manifest), "experiment_id": result["experiment_id"],
                  "states": len(result["states"]), "training_pool_counts": result["training_pool_counts"]}
    elif args.mode == "contexts":
        result = build_contexts(args.manifest)
    elif args.mode == "report":
        result = report(args.manifest)
    else:
        result = measure(args)
    print(json.dumps(result, indent=2, default=str))
    return 75 if result.get("status") == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
