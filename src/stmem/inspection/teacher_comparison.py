"""CPU-only comparisons of saved teacher inspection and observer records.

These summaries never run a policy or infer visual conclusions from filenames.
The original per-stage validators remain authoritative for detailed numerical
invariants. Different source hashes are reported as requiring review, never
silently certified as equivalent implementations.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


METRICS = (
    "native4_tcp_error_mm", "native16_tcp_error_mm",
    "native4_joint_error_rmse", "native16_joint_error_rmse",
    "native4_gripper_sign_errors", "native16_gripper_sign_errors",
    "native4_tcp_change_mm", "native16_tcp_change_mm",
)
TARGET_FIELDS = ("target_actions", "target_valid", "native_target_actions", "native_target_valid")
BEHAVIOR_LIMITATIONS = [
    "Privileged observer labels are reporting-only and were not policy inputs.",
    "Closed-loop trajectories diverge after matched resets; later behavior is not a fixed-current counterfactual.",
    "Only retained active observer samples are measured; terminal automatic-step observations are excluded.",
    "Observed pick/drop cycles are lower bounds if a subgoal completes only in an excluded sample.",
    "Nearest-pad identity alone does not establish valid placement, intent, or semantic target selection.",
    "No manual video inspection is implied by this automatically generated report.",
]


def _read(path):
    return json.loads(Path(path).read_text())


def _write(output, result):
    if output is None:
        return
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_hash(path):
    path = Path(path)
    if path.is_dir():
        return {str(item.relative_to(path)): _sha(item) for item in sorted(path.rglob("*"))
                if item.is_file() and not item.name.endswith(".lock")}
    return _sha(path)


def _scalar(record):
    return {key: value for key, value in record.items() if not isinstance(value, (dict, list))}


def _pair(reference, current):
    return {"reference": reference, "current": current,
            "delta_current_minus_reference": current - reference}


def _keyed(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"Duplicate record key: {key}")
        result[key] = row
    return result


def _require_equal(reference, current, description):
    if reference != current:
        raise ValueError(f"Mismatched {description}; cannot claim a matched comparison")


def _identity(entry):
    return {name: entry[name] for name in ("path", "sha256", "step")}


def _usual_side(root, model):
    manifest = _read(root / "inspection_manifest.json")
    manifest_hash = _sha(root / "inspection_manifest.json")
    audit = _read(root / "checkpoint_audit.json")
    _require_equal(audit["inspection_manifest_sha256"], manifest_hash, "checkpoint audit manifest")
    checkpoint = audit["checkpoints"][model]
    if checkpoint.get("status") != "passed":
        raise ValueError(f"Checkpoint audit has not passed: {root} ({model})")
    _require_equal(_identity(manifest["checkpoints"][model]), _identity(checkpoint), "checkpoint identities")
    pipeline = _read(root / "pipeline_status.json")
    if pipeline["status"] != "complete":
        raise ValueError(f"Usual inspection is not complete: {root}")
    validation = {}
    for name in ("offline/validation.json", "grounding/validation.json", "handoff/validation.json",
                 "handoff/media_validation.json", "report_validation.json"):
        record = _read(root / name)
        if record.get("status") != "passed":
            raise ValueError(f"Validation has not passed: {root / name}")
        if name != "report_validation.json":
            _require_equal(record.get("inspection_manifest_sha256"), manifest_hash, f"{name} manifest")
        validation[name] = _scalar(record)
    return manifest, checkpoint, pipeline, validation


def compare_usual(reference_analysis, analysis, *, model="frozen_10hz", output=None, verify_files=True):
    """Compare complete usual inspections and optionally write one JSON report.

    Protocol/config/asset/target mismatches fail rather than producing a paired
    effect. Source and runtime identities are reported separately: a changed
    source requires human compatibility review, not an assumed implementation
    match. ``verify_files=False`` uses saved checksums without rehashing assets
    or weights and is explicitly recorded (useful for archived reports).
    """
    reference, current = Path(reference_analysis).resolve(), Path(analysis).resolve()
    manifests, checkpoints, pipelines, validations = {}, {}, {}, {}
    for label, root in (("reference", reference), ("current", current)):
        manifests[label], checkpoints[label], pipelines[label], validations[label] = _usual_side(root, model)
    left, right = manifests["reference"], manifests["current"]
    _require_equal(left["protocol"], right["protocol"], "usual inspection protocols")
    _require_equal(left["asset_hashes"], right["asset_hashes"], "asset manifests")
    for key in ("model", "data", "geometry", "evaluation"):
        _require_equal(checkpoints["reference"]["config"][key], checkpoints["current"]["config"][key],
                       f"checkpoint {key} settings")
    offline_protocols = {label: _read(root / "offline/protocol.json")
                         for label, root in (("reference", reference), ("current", current))}
    nonprotocol = {"inspection_manifest_sha256", "prepared_data", "alignment", "frozen_cache"}
    _require_equal({key: value for key, value in offline_protocols["reference"].items() if key not in nonprotocol},
                   {key: value for key, value in offline_protocols["current"].items() if key not in nonprotocol},
                   "offline precision, action timing, and intervention protocols")
    compatibility = {
        "protocol_equal": True, "offline_protocol_equal": True, "asset_manifest_hashes_equal": True,
        "scientific_input_and_model_configs_equal": True,
        "global_policy_source_equal": left["policy_source_sha256"] == right["policy_source_sha256"],
        "policy_source_sha256": {label: value["policy_source_sha256"] for label, value in manifests.items()},
        "checkpoint_hashes_verified_now": None, "asset_hashes_verified_now": None,
        "runtime_identity_equal": _read(reference / "runtime_identity.json")["identity"] ==
                                  _read(current / "runtime_identity.json")["identity"],
        "runtime_record_scopes": {label: _read(root / "runtime_identity.json").get("scope")
                                  for label, root in (("reference", reference), ("current", current))},
    }
    compatibility["source_review_required"] = not compatibility["global_policy_source_equal"]
    compatibility["runtime_review_required"] = not compatibility["runtime_identity_equal"]
    if verify_files:
        checkpoint_checks, asset_checks = {}, {}
        hash_cache = {}
        for label, manifest in manifests.items():
            entry = checkpoints[label]
            checkpoint_checks[label] = _sha(entry["path"]) == entry["sha256"]
            if not checkpoint_checks[label]:
                raise ValueError(f"Checkpoint bytes changed: {entry['path']}")
            asset_checks[label] = {}
            for name, raw_path in manifest["assets"].items():
                path = str(Path(raw_path).resolve())
                if path not in hash_cache:
                    hash_cache[path] = _asset_hash(path)
                asset_checks[label][name] = hash_cache[path] == manifest["asset_hashes"][name]
                if not asset_checks[label][name]:
                    raise ValueError(f"Inspection asset bytes changed: {path}")
        compatibility["checkpoint_hashes_verified_now"] = checkpoint_checks
        compatibility["asset_hashes_verified_now"] = asset_checks

    rows = {}
    fields = ("kind", "variant", "episode", "index", "noise_seed")
    for label, root in (("reference", reference), ("current", current)):
        rows[label] = _keyed(_read(root / "offline" / model / "sensitivity.json")["rows"], fields)
        if not any(key[:2] == ("visual", "baseline") for key in rows[label]):
            raise ValueError(f"Offline inspection lacks visual baseline records: {root}")
    _require_equal(rows["reference"].keys(), rows["current"].keys(), "offline scene/noise/intervention keys")
    for key, row in rows["reference"].items():
        for name in TARGET_FIELDS:
            _require_equal(row[name], rows["current"][key][name], f"expert targets at {key}: {name}")
    compatibility.update(offline_row_keys_equal=True, offline_targets_equal=True)
    groups = defaultdict(list)
    for key in rows["reference"]:
        kind, variant, episode, index, _noise = key
        if variant == "baseline" or (kind == "visual" and variant != "baseline_repeat"):
            groups[(kind, variant, "all_states")].append(key)
            if variant == "baseline":
                groups[(kind, variant, "reset" if index == 0 else "after_reset")].append(key)
    summaries = []
    for key, members in sorted(groups.items()):
        metrics = {}
        for metric in METRICS:
            means = {}
            for label in rows:
                values = [rows[label][item][metric] for item in members]
                if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in values):
                    raise ValueError(f"Nonfinite/non-numeric offline metric: {metric}")
                means[label] = sum(values) / len(values)
            metrics[metric] = _pair(means["reference"], means["current"])
        summaries.append(dict(zip(("kind", "variant", "state_subset"), key)) |
                         {"samples": len(members), "metrics": metrics})
    baseline_by_episode = []
    episodes = sorted({key[2] for key in rows["reference"]})
    for episode in episodes:
        keys = [key for key in rows["reference"] if key[:3] == ("visual", "baseline", episode)]
        if keys:
            baseline_by_episode.append({"episode": episode, "samples": len(keys), "metrics": {
                metric: _pair(*(sum(rows[label][key][metric] for key in keys) / len(keys)
                                for label in ("reference", "current")))
                for metric in ("native4_tcp_error_mm", "native16_tcp_error_mm")}})
    grounding = {}
    for label, root in (("reference", reference), ("current", current)):
        record = _read(root / "grounding/validation.json")["models"][model]
        grounding[label] = {**_scalar(record), "grasp_flag_commands": len(record["grasp_flag_commands"]),
                           **{key: record[key] for key in ("state80_response_mm", "state80_first0p8s_response_mm",
                                                         "state80_full_horizon_response_mm", "state80_endpoint_response_mm")},
                           "reset_cube_shifts": [
                               {**_scalar(scene), "mean_over_noise": scene["mean_over_noise"]}
                               for scene in _read(root / "reset_cube_chunks/summary.json")["models"][model]["scenes"]]}
    handoffs = {label: _read(root / "handoff/summary.json")
                for label, root in (("reference", reference), ("current", current))}
    selected = {label: _keyed([row for row in value["runs"] if row["variant"] == model and not row["expert_control"]],
                             ("task", "episode", "prefix")) for label, value in handoffs.items()}
    if not selected["reference"] or not selected["current"]:
        raise ValueError("Handoff inspection has no policy trials for the requested teacher")
    _require_equal(selected["reference"].keys(), selected["current"].keys(), "handoff scenes/prefixes")
    handoff = {
        "counts": {label: {"successes": sum(row["status"] == "success" for row in value.values()),
                            "trials": len(value)} for label, value in selected.items()},
        "paired_runs": [{"key": list(key), "reference": selected["reference"][key],
                         "current": selected["current"][key]} for key in sorted(selected["reference"])],
        "by_model_scene": {label: [row for row in value["by_model_scene"] if row["variant"] == model]
                           for label, value in handoffs.items()},
    }
    command_changes = {}
    for label, records in rows.items():
        values = [row["max_abs_raw_action_change"] for row in records.values() if row["kind"] == "commands"]
        command_changes[label] = max(values) if values else None
    result = {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved fixed-scene diagnostics, not benchmark success rates or new policy execution.",
        "reference_analysis": str(reference), "analysis": str(current), "model": model,
        "checkpoints": {label: _identity(entry) for label, entry in checkpoints.items()},
        "compatibility": compatibility, "offline_protocol": offline_protocols["current"],
        "offline_summary_pairs": summaries, "offline_baseline_by_episode": baseline_by_episode,
        "previous_command_max_raw_change": command_changes, "grounding": grounding,
        "handoff": handoff, "validations": validations,
        "limitations": [
            "Correlated flow-noise draws and training-state predictions are not independent task trials.",
            "Forward-kinematics target changes are predictions, not observed simulated robot movement.",
            "A different source hash needs explicit compatibility review; matching settings is not source equivalence.",
            "Runtime identity may have been recorded after preexisting outputs; consult each recorded scope.",
            "Automatic summaries do not claim that a person viewed the videos.",
        ],
    }
    _write(output, result)
    return result


def _first(rows, predicate):
    return next((row["command_index"] for row in rows if predicate(row)), None)


def _xy(left, right):
    return 1000 * math.dist(left[:2], right[:2])


def summarize_behavior(record, *, source_result=None):
    """Extract observer-supported behavior; never invent a terminal failure cause."""
    diagnostic = record.get("task_diagnostics") or {}
    result = {"status": record["status"], "commands": record.get("primitive_commands"),
              "task": record["task"], "episode": record["episode_idx"], "arm": record["arm"],
              "video_path": record.get("video_path"), "source_result": str(source_result) if source_result else None,
              "conditioning_video_frames": (record.get("environment") or {}).get("conditioning_frames_recorded"),
              "goal": diagnostic.get("goal"), "target": diagnostic.get("target"),
              "observer_status": diagnostic.get("status"), "observer_errors": diagnostic.get("errors"),
              "summary_as_saved": diagnostic.get("summary"), "available": False}
    rows = [row for row in diagnostic.get("timeline", []) if "tcp_xyz_m" in row]
    target = diagnostic.get("target")
    if not rows or not target or diagnostic.get("status") != "complete":
        result["unavailable_reason"] = "Missing, failed, or empty observer record; no zero-valued behavior inferred"
        return result
    index = target["first_index"]
    reset_index = (diagnostic.get("reset") or {}).get("current_task_index")
    result.update(available=True, subgoal_transitions=[])
    previous = reset_index
    for row in rows:
        value = row.get("current_task_index")
        if value is not None and value != previous:
            result["subgoal_transitions"].append({"command": row["command_index"], "task_index": value,
                                                  "relative_to_reset": value - reset_index if reset_index is not None else None})
            previous = value
    progress = (diagnostic.get("summary") or {}).get("completed_subgoal_index_increase")
    if record["task"] == "VideoRepick" and progress is not None and reset_index is not None:
        repeats = target["num_repeats"]
        result["observed_completed_pick_drop_cycles"] = min(repeats, max(0, progress) // 2)
        result["observed_reached_button_stage"] = progress >= 2 * repeats
        result["first_observed_button_stage_command"] = _first(
            rows, lambda row: row.get("current_task_index", reset_index) - reset_index >= 2 * repeats)
    if target.get("kind") == "target_placement":
        # Candidate objects are destination pads, not the cube being manipulated.
        # Generic observer first_grasp fields cannot be used for placement tasks.
        first_grasp = _first(rows, lambda row: row["controlled_object"]["grasped"])
        first_lift = _first(rows, lambda row: row["controlled_object"]["lifted"])
        result.update(controlled_cube_first_grasp_command=first_grasp,
                      controlled_cube_first_lift_command=first_lift,
                      controlled_cube_max_height_m=max(row["controlled_object"]["xyz_m"][2] for row in rows))
        last = rows[-1]
        distances = [_xy(last["controlled_object"]["xyz_m"], pad["xyz_m"]) for pad in last["candidates"]]
        nearest = min(range(len(distances)), key=distances.__getitem__)
        result["last_observed"] = {"command": last["command_index"], "cube": last["controlled_object"],
                                  "tcp_xyz_m": last["tcp_xyz_m"], "cube_pad_xy_distances_mm": distances,
                                  "nearest_pad": nearest, "nearest_is_target": nearest == index,
                                  "target_distance_mm": distances[index]}
        after_lift = [row for row in rows if first_lift is not None and row["command_index"] >= first_lift]
        result["after_first_lift_min_cube_pad_xy_mm"] = [
            min(_xy(row["controlled_object"]["xyz_m"], row["candidates"][i]["xyz_m"]) for row in after_lift)
            for i in range(target["candidate_count"])] if after_lift else None
        low = next((row for row in after_lift if row["controlled_object"]["xyz_m"][2] < .03), None)
        result["first_low_cube_after_lift"] = None
        if low is not None:
            low_distances = [_xy(low["controlled_object"]["xyz_m"], pad["xyz_m"]) for pad in low["candidates"]]
            low_nearest = min(range(len(low_distances)), key=low_distances.__getitem__)
            result["first_low_cube_after_lift"] = {
                "command": low["command_index"], "cube_xyz_m": low["controlled_object"]["xyz_m"],
                "grasped": low["controlled_object"]["grasped"], "nearest_pad": low_nearest,
                "nearest_pad_distance_mm": low_distances[low_nearest], "target_distance_mm": low_distances[index],
            }
        result["destination_observation"] = (
            "No controlled-cube lift observed; nearest-pad identity is not an executed placement."
            if first_lift is None else
            "Final saved cube is nearest the intended pad; distance and official status determine completion."
            if nearest == index else
            "Final saved cube is nearest another pad; inspect distances for ambiguity before inferring a wrong destination.")
    else:
        result.update(
            min_tcp_target_xy_mm=min(_xy(row["tcp_xyz_m"], row["candidates"][index]["xyz_m"]) for row in rows),
            target_max_height_m=max(row["candidates"][index]["xyz_m"][2] for row in rows),
            target_ever_grasped=any(row["candidates"][index].get("grasped", False) for row in rows),
            target_ever_lifted=any(row["candidates"][index].get("lifted", False) for row in rows),
            other_indices_ever_grasped=sorted({candidate["index"] for row in rows for candidate in row["candidates"]
                                             if candidate["index"] != index and candidate.get("grasped")}),
            other_indices_ever_lifted=sorted({candidate["index"] for row in rows for candidate in row["candidates"]
                                            if candidate["index"] != index and candidate.get("lifted")}),
            last_observed_tcp_xyz_m=rows[-1]["tcp_xyz_m"],
            last_observed_target_xyz_m=rows[-1]["candidates"][index]["xyz_m"],
        )
    return result


def _history_side(root):
    manifest = _read(root / "manifest.json")
    planned = _keyed(manifest["planned_cases"], ("task", "episode_idx", "arm"))
    if not planned:
        raise ValueError("History comparison requires a nonempty planned case set")
    records = {}
    for path in sorted((root / "results").glob("*/episodes/*.json")):
        record = _read(path)
        key = (record["task"], record["episode_idx"], record["arm"])
        if (key in records or key not in planned or path.parent.parent.name != key[2]
                or path.name != f"{key[0]}_{key[1]:03d}.json"):
            raise ValueError(f"Duplicate, unplanned or mislabeled history result: {path}")
        if record.get("checkpoint_sha256") != manifest["checkpoint"]["sha256"]:
            raise ValueError(f"Result checkpoint mismatch: {path}")
        records[key] = (record, path)
    return manifest, planned, records


def compare_behavior(reference_history, history, *, output=None, require_complete=True):
    """Summarize all planned task/arm pairs, including missing cases explicitly.

    Reset fingerprints, seeds and privileged target labels must match when both
    results have them. Unknown parity is recorded, never certified as matching.
    The separate history-comparison audit checks detailed online action parity.
    """
    reference, current = Path(reference_history).resolve(), Path(history).resolve()
    left_manifest, left_plan, left = _history_side(reference)
    right_manifest, right_plan, right = _history_side(current)
    _require_equal(left_plan.keys(), right_plan.keys(), "planned history cases")
    _require_equal(left_manifest["arms"], right_manifest["arms"], "history interventions")
    _require_equal(left_manifest["execution_override"], right_manifest["execution_override"], "execution override")
    cases = []
    for key in left_plan:
        a, a_path = left.get(key, (None, None))
        b, b_path = right.get(key, (None, None))
        complete = a is not None and b is not None
        parity = {}
        if complete:
            for name, av, bv in (
                ("noise_seed", a.get("noise_seed"), b.get("noise_seed")),
                ("reset_fingerprint", ((a.get("task_diagnostics") or {}).get("input_fingerprints") or {}).get("reset_sha256"),
                 ((b.get("task_diagnostics") or {}).get("input_fingerprints") or {}).get("reset_sha256")),
                ("target_labels", (a.get("task_diagnostics") or {}).get("target"),
                 (b.get("task_diagnostics") or {}).get("target")),
            ):
                parity[name] = "unknown" if av is None or bv is None else "matched" if av == bv else "mismatch"
                if parity[name] == "mismatch":
                    raise ValueError(f"Mismatched {name} at {key}")
        cases.append({"task": key[0], "episode": key[1], "arm": key[2],
                      "pair_status": "complete" if complete else "pending", "input_parity": parity,
                      "reference": summarize_behavior(a, source_result=a_path) if a else None,
                      "current": summarize_behavior(b, source_result=b_path) if b else None})
    complete = all(row["pair_status"] == "complete" for row in cases)
    if require_complete and not complete:
        raise ValueError("History behavior comparison has missing planned cases")
    outcomes, review = {}, []
    ordinary = {"success", "fail", "timeout"}
    for case in cases:
        group = outcomes.setdefault((case["task"], case["arm"]), {
            "task": case["task"], "arm": case["arm"], "planned": 0,
            "reference_successes": 0, "current_successes": 0, "complete_pairs": 0,
            "error_pairs": 0, "gained_success_episodes": [], "lost_success_episodes": [],
        })
        group["planned"] += 1
        a, b = case["reference"], case["current"]
        group["reference_successes"] += int(a is not None and a["status"] == "success")
        group["current_successes"] += int(b is not None and b["status"] == "success")
        if case["pair_status"] != "complete":
            continue
        group["complete_pairs"] += 1
        if a["status"] not in ordinary or b["status"] not in ordinary:
            group["error_pairs"] += 1
            continue
        reasons = []
        if (a["status"] == "success") != (b["status"] == "success"):
            direction = "gained" if b["status"] == "success" else "lost"
            group[f"{direction}_success_episodes"].append(case["episode"])
            reasons.append(f"{direction} official success")
        for name in ("observed_completed_pick_drop_cycles", "target_ever_grasped", "target_ever_lifted"):
            if name in a and name in b and a[name] != b[name]:
                reasons.append(f"changed {name}")
        if (a.get("controlled_cube_first_lift_command") is not None
                and b.get("controlled_cube_first_lift_command") is not None
                and a["last_observed"]["nearest_pad"] != b["last_observed"]["nearest_pad"]):
            reasons.append("changed nearest pad after observed lift; inspect distances for ambiguous placement")
        if reasons:
            review.append({"task": case["task"], "episode": case["episode"], "arm": case["arm"],
                           "reasons": reasons,
                           "reference_video": a["video_path"], "current_video": b["video_path"]})
    result = {
        "schema_version": 1, "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Saved observer evidence only; no simulator execution or inferred manual visual review.",
        "reference_history": str(reference), "history": str(current),
        "checkpoints": {"reference": left_manifest["checkpoint"], "current": right_manifest["checkpoint"]},
        "complete": complete, "completed_pairs": sum(row["pair_status"] == "complete" for row in cases),
        "planned_pairs": len(cases), "cases": cases, "limitations": BEHAVIOR_LIMITATIONS,
        "outcomes_by_task_arm": list(outcomes.values()), "suggested_video_review": review,
        "derived_metric_definitions": {
            "pad_distance": "Euclidean XY distance between actual cube and pad origins in mm, at retained active samples.",
            "last_observed": "Last active observer sample; terminal automatic-step observation is excluded.",
            "pick_drop_cycles": "VideoRepick observed subgoal advance // 2, capped at the requested repeat count.",
            "first_low_cube_after_lift": "First retained sample after lift with cube origin z<0.03m; not an official success predicate.",
            "status": "Saved benchmark status is authoritative; these summaries do not recompute success.",
        },
    }
    _write(output, result)
    return result
