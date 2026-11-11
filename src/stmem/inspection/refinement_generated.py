"""Bounded common-case generated chunks selected only from development discovery.

Plan and generation are separate transactions. All requested students receive
the same state/reader/background cases, with their own discovery-selected
completed packets. These are diagnostic loss selections, not deployed routers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from stmem.config import digest, source_revision
from stmem.inspection.refinement_data import (
    UtilityDataReader, primary_students, read_json, reference_identity, stable_seed,
    teacher_identity, verify_manifest,
)
from stmem.inspection.refinement_measurements import draw_seed, generated_state
from stmem.inspection.refinement_statistics import context_means, tie_set
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.runtime import load_policy


STUDENTS = ("S0", "S_STD", "S_STAGE", "S_PACKET")
STRATA = ("packet_delayed", "background_contrast", "objective_conflict", "null")
SCHEMA = "refinement_generated_v1"


def _packet_groups(state, panel, rows, epsilon, *, allow_diagnostic_partition=False):
    """Ignore every confirmation row, including for illustrative case choice."""
    if state["partition"] != "train" and not allow_diagnostic_partition:
        return {}
    discovery_rows = [row for row in rows if row["bank"] == "discovery"]
    means = context_means(discovery_rows)
    groups = {}
    for family in panel["families"]:
        if family["panel"] != "packet" or "dispersed" not in family["controls"]:
            continue
        cost = int(family["final_budget"] - family["background"])
        if cost != 8:
            continue
        required = [family["base"], family["controls"]["dispersed"]]
        candidates = family["candidates"]
        if len(candidates) < 2 or not all((key, "discovery") in means for key in required):
            continue
        if any(not all((c["prefixes"].get(str(n)), "discovery") in means for n in (1, 8))
               for c in candidates):
            continue
        first = {c["candidate_id"]: means[c["prefixes"]["1"], "discovery"]["expert_full"] for c in candidates}
        complete = {c["candidate_id"]: means[c["prefixes"]["8"], "discovery"]["expert_full"] for c in candidates}
        agreement = {c["candidate_id"]: means[c["prefixes"]["8"], "discovery"]["teacher_full"] for c in candidates}
        choices = {"immediate": tie_set(first, epsilon=epsilon), "packet": tie_set(complete, epsilon=epsilon),
                   "teacher": tie_set(agreement, epsilon=epsilon)}
        # A deterministic representative is explicitly declared; ties are not
        # counted as directional evidence. Full tie sets remain in the plan.
        representatives = {key: values[0] for key, values in choices.items()}
        by_id = {c["candidate_id"]: c for c in candidates}
        base_loss = means[family["base"], "discovery"]["expert_full"]
        gains = {key: base_loss - value for key, value in complete.items()}
        key = state["state_id"], family["reader"], int(family["background"])
        groups[key] = {"family": family, "tie_sets": choices, "representatives": representatives,
                       "contexts": {"base": family["base"],
                                    "immediate_completed": by_id[representatives["immediate"]]["prefixes"]["8"],
                                    "packet_completed": by_id[representatives["packet"]]["prefixes"]["8"],
                                    "dispersed": family["controls"]["dispersed"]},
                       "gains": gains,
                       "scores": {"packet_delayed": complete[representatives["immediate"]] -
                                                       complete[representatives["packet"]],
                                  "objective_conflict": complete[representatives["teacher"]] -
                                                        complete[representatives["packet"]],
                                  "null": -float(np.mean(np.abs(list(gains.values()))))},
                       "discovery_base_loss": base_loss,
                       "discovery_completed_losses": complete}
    for key, group in groups.items():
        other = groups.get((key[0], key[1], 120 if key[2] == 56 else 56))
        if other is None or not group["family"].get("paired") or not other["family"].get("paired"):
            continue
        candidate_support = {c["candidate_id"]: c.get("support_hash") for c in group["family"]["candidates"]}
        other_support = {c["candidate_id"]: c.get("support_hash") for c in other["family"]["candidates"]}
        common = [key for key in group["gains"] if key in other["gains"] and candidate_support[key] == other_support[key]]
        if common:
            group["scores"]["background_contrast"] = max(abs(group["gains"][k] - other["gains"][k]) for k in common)
    return groups


def select_common_cases(records, *, seed, max_cases=16, max_per_task=4, max_per_episode=1, epsilon=1e-6):
    """Symmetric capped union of per-student/stratum discovery rankings.

    Each round takes one nomination from every student and every stratum.
    Their union is ranked by the number of distinct nominating students, then
    total nominations, then a fixed hash tie-break. Student dictionary order
    and labels never determine priority. One state appears at most once.
    """
    if not 1 <= max_cases <= 16 or max_per_task < 1 or max_per_episode < 1:
        raise ValueError("Generated cases must be bounded by 16 with positive diversity caps")
    students = sorted(records)
    if not students:
        raise ValueError("No evaluated students supplied")
    tables, states, row_counts = {}, {}, {}
    for student in students:
        table = {}
        for record in records[student]:
            state = record["state"]
            if state["partition"] != "train":
                continue
            states[state["state_id"]] = state
            table.update(_packet_groups(state, record["panel"], record["rows"], epsilon))
        tables[student] = table
        row_counts[student] = len(table)
    common = set.intersection(*(set(tables[s]) for s in students))
    for key in common:
        families = {digest(tables[student][key]["family"]) for student in students}
        if len(families) != 1:
            raise ValueError("Matched students must share the exact gain-blind candidate/context manifest")
    rankings = {}
    for student in students:
        for stratum in STRATA:
            keys = [key for key in common if stratum in tables[student][key]["scores"]
                    and (stratum == "null" or tables[student][key]["scores"][stratum] > epsilon)]
            keys.sort(key=lambda key: (-tables[student][key]["scores"][stratum], key))
            # One family's most extreme example per state is enough to nominate.
            seen, unique = set(), []
            for key in keys:
                if key[0] not in seen:
                    seen.add(key[0])
                    unique.append(key)
            rankings[student, stratum] = unique
    selected, selected_ids = [], set()
    task_counts, episode_counts = Counter(), Counter()
    nominations = defaultdict(list)
    maximum_round = max(map(len, rankings.values()), default=0)
    for rank in range(maximum_round):
        for (student, stratum), keys in sorted(rankings.items()):
            if rank < len(keys):
                key = keys[rank]
                nominations[key[0]].append({"student": student, "stratum": stratum,
                                             "rank": rank + 1, "reader": key[1], "background": key[2]})
        available = [state_id for state_id in nominations if state_id not in selected_ids]
        available.sort(key=lambda state_id: (-len({n["student"] for n in nominations[state_id]}),
                                              -len(nominations[state_id]), stable_seed(seed, state_id)))
        for state_id in available:
            state = states[state_id]
            episode = state["task"], state["episode_id"]
            if task_counts[state["task"]] >= max_per_task or episode_counts[episode] >= max_per_episode:
                continue
            eligible = [key for key in common if key[0] == state_id]
            def family_priority(key):
                matching = [n for n in nominations[state_id] if (n["reader"], n["background"]) == key[1:]]
                return -len({n["student"] for n in matching}), -len(matching), key[1], key[2]
            key = min(eligible, key=family_priority)
            selected.append({"state_id": state_id, "state": state, "reader": key[1], "background": key[2],
                             "final_budget": key[2] + 8,
                             "selection_reasons": sorted(nominations[state_id], key=lambda n: (n["stratum"], n["student"])),
                             "students": {s: {field: tables[s][key][field] for field in (
                                 "contexts", "tie_sets", "representatives", "scores", "discovery_base_loss",
                                 "discovery_completed_losses")} for s in students}})
            selected_ids.add(state_id)
            task_counts[state["task"]] += 1
            episode_counts[episode] += 1
            if len(selected) == max_cases:
                break
        if len(selected) == max_cases:
            break
    return {"cases": selected, "selection_audit": {
        "students": students, "eligible_state_reader_backgrounds_by_student": row_counts,
        "common_state_reader_backgrounds": len(common), "selected_states": len(selected),
        "requested_maximum": max_cases, "tasks": dict(task_counts), "episodes": len(episode_counts),
        "max_per_task": max_per_task, "max_per_episode": max_per_episode,
        "strata": list(STRATA), "criterion": "development partition, discovery bank only",
        "positive_strata_minimum_score": epsilon,
        "positive_strata_note": "Non-null nominations must exceed the numerical deadband; this is not statistical significance or a practical effect threshold.",
        "algorithm": "roundwise symmetric per-student/stratum nomination union with diversity caps",
        "tie_policy": "lexicographically first discovery near-tie representative; full tie sets retained",
        "scope": "Selected diagnostic strata, not an unbiased task-success sample"}}


def _seal(plan):
    plan = dict(plan)
    plan.pop("plan_id", None)
    plan["plan_id"] = digest(plan)
    return plan


def _measurement_folder(root, student, mode, checkpoint_hash, experiment_id, states, *, partition="train"):
    """Resolve endpoint/partition directories without mixing student revisions."""
    wanted = {state["state_id"] for state in states if state["partition"] == partition and state["cache_eligible"]}
    candidates = []
    for path in (root / "measurements" / student).rglob("identity.json"):
        identity = read_json(path)
        if (identity.get("mode") != mode or identity.get("experiment_id") != experiment_id or
                identity.get("checkpoint_sha256") != checkpoint_hash or
                identity.get("partition") not in (partition, "both")):
            continue
        present = {state_id for state_id in wanted if (path.parent / f"{state_id}.json").exists()}
        candidates.append((-len(present), identity["partition"] != partition, str(path.parent), identity))
    if not candidates:
        raise ValueError(f"No matching development measurements for {student}, {mode}, {checkpoint_hash}")
    _, _, folder, identity = min(candidates, key=lambda value: value[:3])
    return Path(folder), identity


def diagnostic_state(manifest, state_id, *, limited_check, max_cases):
    """Explicit one-state asset check; never relabel held-out episodes as training."""
    if manifest["kind"] != "limited_real_data_diagnostic" or not limited_check or max_cases != 1:
        raise ValueError("An asset-only state requires a limited_real_data_diagnostic manifest, --limited-check and --max-cases 1")
    candidates = [state for state in manifest["states"] if state["cache_eligible"] and
                  state["task"] == "VideoUnmask" and state["decision_index"] >= 32]
    if not candidates:
        raise ValueError("No cached VideoUnmask state at native decision >=32 for the bounded asset check")
    selected = min(candidates, key=lambda state: (state["episode_index"], state["decision_index"], state["state_id"]))
    if state_id != selected["state_id"]:
        raise ValueError("The limited asset check requires the gain-blind first cached VideoUnmask decision >=32")
    return selected


def select_diagnostic_case(records, state, *, epsilon):
    """Fixed state and TREE/B120 precede discovery-based alternative selection."""
    groups = {}
    for student, values in records.items():
        if len(values) != 1 or values[0]["state"] != state:
            raise ValueError("The limited asset diagnostic must contain exactly the predeclared state")
        record = values[0]
        groups[student] = _packet_groups(state, record["panel"], record["rows"], epsilon,
                                         allow_diagnostic_partition=True).get((state["state_id"], "TREE", 120))
        if groups[student] is None:
            raise ValueError("Predeclared TREE/background120 packet family is unavailable; no gain-selected fallback")
    if not groups or len({digest(group["family"]) for group in groups.values()}) != 1:
        raise ValueError("The asset diagnostic requires a shared immutable candidate family")
    case = {"state_id": state["state_id"], "state": state, "reader": "TREE", "background": 120,
            "final_budget": 128,
            "selection_reasons": [{"stratum": "gain_blind_asset_check", "state_rule": "first cached VideoUnmask native decision>=32",
                                   "reader": "TREE", "background": 120}],
            "students": {name: {field: group[field] for field in (
                "contexts", "tie_sets", "representatives", "scores", "discovery_base_loss",
                "discovery_completed_losses")} for name, group in sorted(groups.items())}}
    return {"cases": [case], "selection_audit": {
        "students": sorted(groups), "selected_states": 1, "requested_maximum": 1,
        "tasks": {state["task"]: 1}, "episodes": 1, "max_per_task": 1, "max_per_episode": 1,
        "original_partition": state["partition"], "diagnostic_state_id": state["state_id"],
        "criterion": "State and TREE/background120 fixed before reading gains; alternatives use discovery only",
        "scope": "Exploratory local asset check; no production held-out-confirmation claim and no episode relabeling"}}


def validate_generated_endpoints(identities, *, limited_check):
    """Compare the same declared training endpoint and initialization across arms."""
    new_students = [identity for name, identity in identities.items() if name != "S0"]
    for identity in new_students:
        if (identity.get("initialization") not in ("accepted_teacher_weights", "S0_weights") or
                int(identity.get("endpoint") or 0) < 1):
            raise ValueError("Generated-action students need their declared initialization and endpoint identity")
        if not limited_check and identity["step"] != identity["endpoint"]:
            raise ValueError("The primary common-case matrix requires each student's predeclared endpoint")
    if len({(value["initialization"], value["endpoint"]) for value in new_students}) > 1:
        raise ValueError("Generated-action comparisons cannot mix student initializations or declared endpoints")
    if len({value["step"] for value in new_students}) > 1:
        raise ValueError("Generated-action comparisons cannot mix observed checkpoint steps")


def create_plan(manifest_path, output, *, students=None, checkpoints=None, max_cases=16,
                measurement_mode="main", limited_check=False, diagnostic_state_id=None):
    from stmem.inspection.refinement_utility import load_rows, read_panel
    manifest = verify_manifest(manifest_path)
    root = Path(manifest_path).parent
    primary = primary_students(manifest)
    students = tuple(sorted(set(primary if students is None else students)))
    if not students or set(students) - set(primary):
        raise ValueError("Specify this manifest's primary students; preflight/historical references are excluded")
    if (set(students) != set(primary) or measurement_mode != "main") and not limited_check:
        raise ValueError("A partial-student/smoke probe needs explicit --limited-check")
    checkpoints = dict(checkpoints or {})
    if "S0" in primary:
        checkpoints.setdefault("S0", manifest["dependencies"]["compact"])
    selected_state = (None if diagnostic_state_id is None else diagnostic_state(
        manifest, diagnostic_state_id, limited_check=limited_check, max_cases=max_cases))
    selected_states = manifest["states"] if selected_state is None else [selected_state]
    records, inputs, identities = {}, [], {}
    for student in students:
        if student not in checkpoints:
            raise ValueError(f"Missing immutable --checkpoints {student}=PATH")
        checkpoint = Path(checkpoints[student]).resolve()
        checkpoint_hash = file_sha256(checkpoint)
        if student == "S0" and checkpoint_hash != reference_identity(manifest)[0]:
            raise ValueError("S0 must remain the exact production 90k compact checkpoint")
        folder, identity = _measurement_folder(root, student, measurement_mode, checkpoint_hash,
                                                manifest["experiment_id"], selected_states,
                                                partition="train" if selected_state is None else selected_state["partition"])
        if identity["experiment_id"] != manifest["experiment_id"] or identity["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Generated-action student differs from the measured checkpoint")
        identities[student] = {"checkpoint": str(checkpoint), "sha256": checkpoint_hash, "step": identity["step"],
                               "initialization": identity.get("initialization"), "endpoint": identity.get("endpoint")}
        records[student] = []
        for state in selected_states:
            if (state["partition"] != "train" and selected_state is None) or not state["cache_eligible"]:
                continue
            path = folder / f"{state['state_id']}.npz"
            if not path.exists():
                if limited_check:
                    continue
                raise ValueError(f"Development measurements incomplete: {path}")
            rows, metadata = load_rows(path)
            if metadata["identity"] != identity:
                raise ValueError("Mixed student measurement identities")
            panel = read_panel(root, state["state_id"], manifest["experiment_id"])
            records[student].append({"state": state, "rows": rows, "panel": panel})
            inputs.append({"path": str(path.resolve()), "sha256": file_sha256(path),
                           "metadata_sha256": file_sha256(path.with_suffix(".json")),
                           "panel_sha256": panel["panel_sha256"]})
    validate_generated_endpoints(identities, limited_check=limited_check)
    selection = (select_common_cases(records, seed=manifest["protocol"]["seed"], max_cases=max_cases,
                                     epsilon=manifest["protocol"]["gain_epsilon"]) if selected_state is None else
                 select_diagnostic_case(records, selected_state, epsilon=manifest["protocol"]["gain_epsilon"]))
    for case in selection["cases"]:
        state = case["state"]
        case["generation_seeds"] = [draw_seed(manifest["protocol"]["seed"], state["task"],
                                              state["episode_id"], state["decision_index"],
                                              "generated_action_validation", i) for i in range(2)]
    plan = _seal({"schema": SCHEMA, "experiment_id": manifest["experiment_id"],
                  "manifest": str(Path(manifest_path).resolve()), "source": source_revision(),
                  "students": identities, "measurement_inputs": inputs,
                  "measurement_mode": measurement_mode, "limited_check": limited_check,
                  "precision": manifest["protocol"]["precision"], "integration_steps": 8,
                  "execute_targets": manifest["protocol"].get("execute_targets", 10), "native_commands_per_target": 2,
                  "generated_draws": 2, **selection,
                  "model_example_bound": len(selection["cases"]) * len(students) * 5 * 2 * 8,
                  "simulation_status": "not_run",
                  "simulation_note": "Generated FK paths are not executed trajectories. An exact-state intervention runner is not silently substituted with ordinary from-reset evaluation.",
                  "scope": ("limited_asset_exploratory_single_state" if selected_state is not None else
                            "partial_student_development_probe" if limited_check else "shared_primary_student_generated_diagnostic")})
    output = Path(output)
    if output.exists() and read_json(output) != plan:
        raise FileExistsError("Immutable generated case plan differs; choose a new --plan path")
    atomic_json(plan, output)
    return plan


def verify_plan(path, *, verify_source=True):
    plan = read_json(path)
    if _seal(plan)["plan_id"] != plan.get("plan_id"):
        raise ValueError("Generated case plan changed")
    manifest = verify_manifest(plan["manifest"], verify_source=verify_source)
    if manifest["experiment_id"] != plan["experiment_id"]:
        raise ValueError("Parent experiment changed")
    if not plan["students"] or set(plan["students"]) - set(primary_students(manifest)):
        raise ValueError("Generated-action plan contains a preflight or historical non-primary student")
    if verify_source and plan["source"]["source_sha256"] != source_revision()["source_sha256"]:
        raise ValueError("Generated-case implementation changed")
    for dependency in plan["measurement_inputs"]:
        path = Path(dependency["path"])
        if file_sha256(path) != dependency["sha256"] or file_sha256(path.with_suffix(".json")) != dependency["metadata_sha256"]:
            raise ValueError("Frozen discovery measurement artifact changed")
    for student in plan["students"].values():
        if file_sha256(student["checkpoint"]) != student["sha256"]:
            raise ValueError("Generated-action checkpoint changed")
    return plan, manifest


def _write_result(path, result, metadata):
    arrays, metrics = {}, {}
    for name, context in result["contexts"].items():
        metrics[name] = {}
        for field, value in context.items():
            if isinstance(value, np.ndarray):
                arrays[f"{name}/{field}"] = value
            else:
                metrics[name][field] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    summary = {**metadata, **{k: v for k, v in result.items() if k != "contexts"},
               "metrics": metrics, "arrays_sha256": file_sha256(path)}
    atomic_json(summary, path.with_suffix(".json"))
    return summary


def load_frozen_policy(path, device):
    policy, state = load_policy(path, device)
    policy.freeze_encoders()
    policy.eval().requires_grad_(False)
    return policy, state


def generate(plan_path, *, student=None, device="cuda:0", batch_size=16, max_hours=1):
    from stmem.inspection.refinement_utility import lock, read_panel
    from stmem.training.label_experiment import validate_models
    plan, manifest = verify_plan(plan_path)
    root = Path(plan["manifest"]).parent
    output = Path(plan_path).parent / (Path(plan_path).stem + "_results")
    selected_students = [student] if student is not None else sorted(plan["students"])
    if any(s not in plan["students"] for s in selected_students):
        raise ValueError("Requested student is outside the frozen common-case matrix")
    if max_hours <= 0 or batch_size < 1:
        raise ValueError("Positive generation time and batch bounds required")
    dep = manifest["dependencies"]
    teacher_hash, _ = teacher_identity(manifest)
    teacher, tstate = load_frozen_policy(dep["teacher"], device)
    if file_sha256(dep["teacher"]) != teacher_hash:
        raise ValueError("Accepted teacher changed")
    data = TrainingData(manifest["configuration"])
    reader = UtilityDataReader(data, manifest["configuration"], dep["encoded"], dep["hierarchy"])
    stopped = False
    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
    original = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    started, done = time.monotonic(), []
    try:
        for name in selected_students:
            identity = plan["students"][name]
            policy, pstate = load_frozen_policy(identity["checkpoint"], device)
            validate_models(manifest["configuration"], teacher, tstate, policy, pstate)
            with lock(output / name / "worker.lock"):
                for case in plan["cases"]:
                    if stopped or time.monotonic() - started >= max_hours * 3600:
                        break
                    path = output / name / f"{case['state_id']}.npz"
                    if path.with_suffix(".json").exists():
                        saved = read_json(path.with_suffix(".json"))
                        if saved["plan_id"] != plan["plan_id"] or saved["arrays_sha256"] != file_sha256(path):
                            raise ValueError("Generated output identity/checksum changed")
                        done.append([name, case["state_id"]])
                        continue
                    panel = read_panel(root, case["state_id"], manifest["experiment_id"])
                    aliases = case["students"][name]["contexts"]
                    specs = {key: panel["contexts"][value] for key, value in aliases.items()}
                    begin = time.monotonic()
                    item, contexts = reader.prepare(case["state"], specs)
                    load_seconds = time.monotonic() - begin
                    result = generated_state(teacher, policy, item, contexts, case["generation_seeds"],
                                              teacher_hash=teacher_hash, student_hash=identity["sha256"],
                                              task=case["state"]["task"], precision=plan["precision"],
                                              prefix_targets=plan["execute_targets"],
                                              compact_batch_size=batch_size)
                    result["timings"]["cache_context_assembly_seconds"] = load_seconds
                    result["current_visual_anchors"] = item["current"].metadata[
                        0, item["current"].valid[0], :3].cpu().tolist()
                    result["current_tcp"] = item["current_state"][7:10].cpu().tolist()
                    _write_result(path, result, {"plan_id": plan["plan_id"], "student": name,
                                                "state": case["state"], "contexts": aliases,
                                                "selection_reasons": case["selection_reasons"]})
                    done.append([name, case["state_id"]])
                    print(json.dumps({"event": "generated_utility_case_complete", "student": name,
                                      "state_id": case["state_id"], "timings": result["timings"]}), flush=True)
            del policy
            if stopped or time.monotonic() - started >= max_hours * 3600:
                break
    finally:
        reader.close()
        for sig, handler in original.items():
            signal.signal(sig, handler)
    planned = [[name, case["state_id"]] for name in selected_students for case in plan["cases"]]
    receipt = {"plan_id": plan["plan_id"], "completed": done, "planned": planned,
               "status": "complete" if len(done) == len(planned) else "partial",
               "wall_seconds": time.monotonic() - started, "simulation_status": "not_run"}
    atomic_json(receipt, output / ("receipt_" + (student or "all") + ".json"))
    report(plan_path)
    return receipt


def report(plan_path):
    from plotly.offline import get_plotlyjs
    plan, _ = verify_plan(plan_path, verify_source=False)
    output = Path(plan_path).parent / (Path(plan_path).stem + "_results")
    rows, missing = [], []
    for case in plan["cases"]:
        for student in sorted(plan["students"]):
            path = output / student / f"{case['state_id']}.npz"
            if not path.with_suffix(".json").exists():
                missing.append([student, case["state_id"]])
                continue
            row = read_json(path.with_suffix(".json"))
            if row["plan_id"] != plan["plan_id"] or file_sha256(path) != row["arrays_sha256"]:
                raise ValueError("Generated output checksum or plan identity mismatch")
            with np.load(path, allow_pickle=False) as data:
                row["paths"] = {name.split("/")[0]: data[name].tolist() for name in data.files if name.endswith("/tcp_paths")}
            row.update(reader=case["reader"], background=case["background"], final_budget=case["final_budget"])
            rows.append(row)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"rows": rows, "missing": missing, "selection": plan["selection_audit"], "scope": plan["scope"],
               "execute_targets": plan["execute_targets"], "native_commands_per_target": plan["native_commands_per_target"]}
    atomic_json(payload, output / "summary.json")
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    page = _HTML.replace("__PLOTLY__", get_plotlyjs().replace("</script", "<\\/script")).replace("__DATA__", encoded)
    (output / "index.html").write_text(page)
    lines = ["# Generated-action utility diagnostic", "", f"Scope: `{plan['scope']}`.",
             f"Completed student × case cells: **{len(rows)} / {len(rows) + len(missing)}**.",
             "", ("This is a gain-blind, one-state local asset check retaining its original episode partition; alternative contexts use discovery only. No production held-out-confirmation claim is made."
                    if plan["scope"] == "limited_asset_exploratory_single_state" else
                    "Cases use development episodes and discovery losses only. Every requested student is evaluated on every retained case."),
             "A fixed representative of each discovery near-tie set is used; tie sets remain in the plan. This is a selected diagnostic, not a success rate.",
             "", "| Student | Task / episode / decision | Reader / final B | Immediate expert MSE | Packet expert MSE | Dispersed expert MSE |", "|---|---|---|---:|---:|---:|"]
    for row in rows:
        state, metrics = row["state"], row["metrics"]
        values = [float(np.mean(metrics[k]["expert_full"])) for k in ("immediate_completed", "packet_completed", "dispersed")]
        lines.append(f"| {row['student']} | {state['task']} / {state['episode_id']} / {state['decision_index']} | {row['reader']} / {row['final_budget']} | " + " | ".join(f"{v:.6g}" for v in values) + " |")
    lines.extend(["", "[Interactive predicted FK paths](index.html). Current-scene dots are cached visual token anchors, not object masks or the complete point cloud.",
                  "", "No simulator continuation was run by this command. The action paths show clipped joint-target FK, not realized motion, contact, or task success.",
                  "Exact-state continuation requires the recorded causal prefix, a verified simulator handoff, one selected-chunk intervention and a newly rebuilt common reader afterward. Ordinary from-reset scripts do not implement that intervention and are not substituted here."])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return {"report": str(output / "report.md"), "html": str(output / "index.html"), "completed": len(rows), "missing": len(missing)}


_HTML = """<!doctype html><meta charset="utf-8"><title>Generated utility diagnostic</title>
<style>body{font:16px/1.5 system-ui;margin:24px;color:#1b293d}select{font:inherit;padding:6px;margin:4px}#plot{height:620px}pre{white-space:pre-wrap}.note{background:#edf3fb;padding:14px;border-left:4px solid #416b9f}</style>
<h1>Generated chunks under selected history allocations</h1>
<p class="note">These are predicted TCP paths of absolute joint targets, not executed simulation trajectories. The same current observation, expert causal history, task and fresh Gaussian draw are used for each comparison. Cases follow the declared plan scope; alternative context choices use discovery losses only. No measured task success is implied.</p>
<select id="case"></select><label>Generation draw <select id="draw"><option value="0">1</option><option value="1">2</option></select></label>
<div id="status"></div><div id="plot"></div><pre id="metrics"></pre>
<script>__PLOTLY__</script><script>const data=__DATA__;const c=document.getElementById('case'),d=document.getElementById('draw');
data.rows.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${r.student} · ${r.state.task} ${r.state.episode_id} t=${r.state.decision_index} · ${r.reader} B${r.final_budget}`;c.appendChild(o)});
const colors={base:'#7e8894',immediate_completed:'#3266bd',packet_completed:'#c04b3a',dispersed:'#a48226',teacher_full:'#16836b'};
function trace(points,name,color,mode='lines+markers',dash='solid'){return {type:'scatter3d',mode,x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),name,line:{color,width:4,dash},marker:{color,size:3}}}
function render(){if(!data.rows.length){document.getElementById('status').textContent='No completed generation cases.';return;}const r=data.rows[Number(c.value)],draw=Number(d.value),traces=[],executed=data.execute_targets;
traces.push(trace(r.current_visual_anchors,'Current visual-token anchors','#abb1b7','markers'));
if(r.current_tcp)traces.push(trace([r.current_tcp],'Observed TCP','#111827','markers'));
for(const [name,paths] of Object.entries(r.paths)){const p=paths[draw];traces.push(trace(p.slice(0,executed),name+` · first ${executed} targets`,colors[name]));if(p.length>executed)traces.push(trace(p.slice(executed-1),name+' · discarded tail',colors[name],'lines+markers','dash'));}
Plotly.react('plot',traces,{margin:{l:0,r:0,t:10,b:0},scene:{aspectmode:'data',xaxis:{title:'World x (m)'},yaxis:{title:'World y (m)'},zaxis:{title:'World z (m)'}},legend:{orientation:'h'},uirevision:r.state.state_id},{responsive:true,displaylogo:false});
document.getElementById('status').textContent=`Scope: ${data.scope}; original episode partition: ${r.state.partition}. Shared background ${r.background} tokens; completed alternatives ${r.final_budget} tokens; ${executed} targets = ${(executed/10).toFixed(1)} seconds at nominal 10 Hz. Anchors are not the full point cloud. ${data.missing.length} matrix cells remain unrun.`;
document.getElementById('metrics').textContent=JSON.stringify({selection_reasons:r.selection_reasons,metrics:r.metrics,comparisons:r.comparisons,note:r.note},null,2)}c.onchange=d.onchange=render;render();</script>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "generate", "report"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--students", nargs="+", choices=STUDENTS,
                        help="Defaults to the manifest's declared primary arms")
    parser.add_argument("--checkpoints", nargs="*", default=[], metavar="STUDENT=PATH")
    parser.add_argument("--student", choices=STUDENTS)
    parser.add_argument("--measurement-mode", choices=("main", "smoke"), default="main")
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--limited-check", action="store_true")
    parser.add_argument("--diagnostic-state-id", help="Explicit gain-blind one-state asset check; requires limited diagnostic manifest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-hours", type=float, default=1)
    args = parser.parse_args(argv)
    if args.mode == "plan":
        if args.manifest is None:
            parser.error("Plan construction needs --manifest")
        checkpoints = {}
        for value in args.checkpoints:
            if "=" not in value:
                parser.error("Use --checkpoints STUDENT=PATH")
            name, path = value.split("=", 1)
            if name not in STUDENTS or name in checkpoints or not path:
                parser.error("Each checkpoint must name one distinct study student")
            checkpoints[name] = path
        plan = create_plan(args.manifest, args.plan, students=args.students, checkpoints=checkpoints,
                           max_cases=args.max_cases, measurement_mode=args.measurement_mode,
                           limited_check=args.limited_check, diagnostic_state_id=args.diagnostic_state_id)
        result = {"plan": str(args.plan), "plan_id": plan["plan_id"], "selection_audit": plan["selection_audit"],
                  "model_example_bound": plan["model_example_bound"]}
    elif args.mode == "generate":
        result = generate(args.plan, student=args.student, device=args.device,
                          batch_size=args.batch_size, max_hours=args.max_hours)
    else:
        result = report(args.plan)
    print(json.dumps(result, indent=2))
    return 75 if result.get("status") == "partial" else 0


if __name__ == "__main__":
    raise SystemExit(main())
