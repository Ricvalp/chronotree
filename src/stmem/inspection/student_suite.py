"""Bounded, resumable local compact-student diagnostics; never optimizer updates.

This is an exploratory local-data screen, separate from the sealed 256-state
HPC study. Shared causal evidence is freshly encoded from the students' identical
frozen encoders, not read from another teacher's learned cache.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from stmem.config import digest, validate_config
from stmem.inspection.common import REPO, discover_checkpoints, policy_source_hash, read_json, resolve_assets
from stmem.inspection.count_probe import tensor_state_hash
from stmem.inspection.pipeline import inspection_lock, inspect_runtime, terminate_stage, termination_signals
from stmem.training.checkpoints import atomic_json, atomic_save, file_sha256
from stmem.training.runtime import load_checkpoint, load_policy, validate_policy_config

SCHEMA = "local_compact_student_inspection_v1"
SCOPE = ("Exploratory local released-demonstration diagnostics, not the reserved HPC utility panel, "
         "router fitting data, or an official benchmark success estimate.")


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def diagnostic_source_hash():
    value = hashlib.sha256()
    paths = list((REPO / "src/stmem/inspection").glob("*.py"))
    paths.extend([REPO / "scripts/inspect_students.py", REPO / "scripts/visualize_reset_cube_chunks.py"])
    for path in sorted(paths):
        value.update(str(path.relative_to(REPO)).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError("Checkpoint has nonfinite model/optimizer tensors")
        return 1
    if isinstance(value, dict):
        return sum(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(finite_tensors(v) for v in value)
    return 0


def audit_student(entry, *, expected_stage="compact"):
    path = Path(entry["path"])
    if file_sha256(path) != entry["sha256"]:
        raise ValueError("Checkpoint changed during transfer/audit")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Checkpoint ZIP CRC failed")
    state = load_checkpoint(path)
    cfg = state["config"]
    validate_config(cfg)
    if state["stage"] != expected_stage or cfg.get("debug") or cfg["data"].get("synthetic"):
        raise ValueError(f"Expected a real-data {expected_stage} checkpoint")
    m = cfg["model"]
    if (m.get("decoder", "flow") != "flow" or m["horizon"] != 16
            or m.get("action_stride", 1) != 2 or cfg["data"].get("observation_stride", 1) != 2
            or m.get("proprioception_mode") != "none" or m.get("condition_on_previous_commands", True)
            or m.get("history_mode", "full") == "none"):
        raise ValueError("This inspection requires H16/10Hz history-enabled no-proprio/no-command students")
    for field in ("model", "training"):
        if state["provenance"][field] != digest(cfg[field]):
            raise ValueError(f"Checkpoint {field} provenance differs from its configuration")
    tensors = finite_tensors(state["model"]) + finite_tensors(state.get("optimizer", {}))
    policy, _ = load_policy(path, "cpu")  # Strict parameter names and shapes.
    if "optimizer" in state:
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                                     lr=cfg["training"]["lr"])
        optimizer.load_state_dict(state["optimizer"])
        for parameter, values in optimizer.state.items():
            for field in ("exp_avg", "exp_avg_sq"):
                if field in values and values[field].shape != parameter.shape:
                    raise ValueError("Optimizer tensor has a different parameter shape")
            if "step" in values and int(values["step"]) != state["step"]:
                raise ValueError("Optimizer step differs from checkpoint")
    if state.get("scheduler", {}).get("last_epoch", state["step"]) != state["step"]:
        raise ValueError("Scheduler boundary differs from checkpoint")
    provenance = state["provenance"]
    if "sampler" in state:
        if (state["sampler"].get("next_global_position") != state["step"] * provenance["effective_batch"]
                or state["sampler"].get("boundary") != "optimizer_step"):
            raise ValueError("Checkpoint is not at the saved optimizer/sample boundary")
    if "rng_by_rank" in state and len(state["rng_by_rank"]) != provenance["world_size"]:
        raise ValueError("Checkpoint is missing per-rank RNG records")
    study = cfg["training"].get("utility_study", {})
    result = {**entry, "step": state["step"], "stage": expected_stage, "config": cfg,
              "variant": study.get("variant"), "training_study": study,
              "teacher_sha256": provenance.get("dependencies", {}).get("teacher"),
              "frozen_encoder_sha256": tensor_state_hash(state["model"], ("encoder.", "text_encoder.")),
              "normalizer_sha256": tensor_state_hash(state["model"], ("normalizer.",)),
              "checked_tensors": tensors, "parameter_count": sum(p.numel() for p in policy.parameters()),
              "bytes": path.stat().st_size, "status": "passed"}
    if file_sha256(path) != entry["sha256"]:
        raise ValueError("Checkpoint changed while auditing")
    return result


def validate_shared_models(audits, teacher=None):
    first = next(iter(audits.values()))
    identities = [audit["sha256"] for audit in audits.values() if "sha256" in audit]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate checkpoint bytes cannot represent different students")
    if not first["teacher_sha256"]:
        raise ValueError("Compact checkpoint must identify its frozen teacher dependency")
    for label, audit in audits.items():
        expected = {"s_std": "S_STD", "s_stage": "S_STAGE", "s_packet": "S_PACKET"}.get(label)
        if expected is not None and audit.get("variant") != expected:
            raise ValueError(f"{label}: checkpoint variant must be {expected}")
        validate_policy_config(first["config"], audit["config"])
        for field in ("teacher_sha256", "frozen_encoder_sha256", "normalizer_sha256"):
            if audit[field] != first[field]:
                raise ValueError(f"{label}: shared evidence requires identical {field}")
    if teacher is not None:
        validate_policy_config(first["config"], teacher["config"])
        if teacher["sha256"] != first["teacher_sha256"]:
            raise ValueError("Optional teacher must be the exact checkpoint that produced these students")
        for field in ("frozen_encoder_sha256", "normalizer_sha256"):
            if teacher[field] != first[field]:
                raise ValueError(f"Optional teacher has different {field}")


def create_plan(args):
    entries = discover_checkpoints(args.bundle)
    audits = {}
    for label, entry in entries.items():
        audits[label] = audit_student(entry)
        print(json.dumps({"event": "student_verified", "label": label,
                          "step": audits[label]["step"]}), flush=True)
    teacher = None
    if args.teacher:
        teacher = audit_student({"path": str(args.teacher.resolve()), "sha256": file_sha256(args.teacher)},
                                expected_stage="teacher")
    validate_shared_models(audits, teacher)
    assets, hashes = resolve_assets(args.assets_config)
    output = (args.output or Path(str(args.bundle.resolve()) + "_student_inspection")).resolve()
    from stmem.inspection.student_handoff import CAP, PREFIXES, SCENES
    protocol = {
        "seed": 20261112, "state_limit": args.states, "budgets": sorted(set(args.budgets)),
        "packet_cost": 8, "max_candidates": args.candidates, "audit_candidates": args.candidates,
        "discovery_draws": args.draws, "confirmation_draws": args.draws,
        "generated_cases": args.generated_cases, "generated_draws": 2,
        "execute_targets": 8, "integration_steps": 8, "budget": 128,
        "precision": "fp32", "gain_epsilon": 1e-6, "batch_size": args.batch_size,
        "stages": args.stages, "simulation_tasks": ["VideoRepick"], "simulation_episodes": [0],
        "handoff_scenes": [list(scene) for scene in SCENES], "handoff_prefixes": list(PREFIXES),
        "handoff_cap_native_commands": CAP, "handoff_noise": "paired_by_absolute_primitive_time",
        "selection": "gain_blind_local_full_horizon_states_before_losses",
        "generated_selection": "evenly_spaced_common_state_indices_before_losses",
        "interpretation": "Episode membership in student training is not established; do not call held-out.",
        "teacher_agreement": "exact_matching_teacher" if teacher else "not_measured_teacher_not_supplied",
    }
    result = {"schema": SCHEMA, "scope": SCOPE, "bundle": str(args.bundle.resolve()), "output": str(output),
              "device": args.device, "checkpoints": audits, "teacher": teacher,
              "assets_config": str(args.assets_config.resolve()), "assets_config_sha256": file_sha256(args.assets_config),
              "assets": assets, "asset_hashes": hashes, "protocol": protocol,
              "source_sha256": policy_source_hash(), "diagnostic_source_sha256": diagnostic_source_hash(),
              "matched_training_steps": len({a["step"] for a in audits.values()}) == 1,
              "runtime": {k: v for k, v in inspect_runtime().items() if k != "initialized_cuda_device"}}
    result["identity"] = digest(result)
    return result


def verify_plan(plan):
    body = {k: v for k, v in plan.items() if k != "identity"}
    if plan.get("schema") != SCHEMA or plan.get("identity") != digest(body):
        raise ValueError("Student inspection plan seal differs")
    if plan["source_sha256"] != policy_source_hash() or plan["diagnostic_source_sha256"] != diagnostic_source_hash():
        raise ValueError("Inspection implementation changed; use a new output directory")
    for record in [*plan["checkpoints"].values(), *([plan["teacher"]] if plan["teacher"] else [])]:
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError("Inspection checkpoint changed")
    if file_sha256(plan["assets_config"]) != plan["assets_config_sha256"]:
        raise ValueError("Inspection assets configuration changed")
    assets, hashes = resolve_assets(plan["assets_config"])
    if assets != plan["assets"] or hashes != plan["asset_hashes"]:
        raise ValueError("Inspection asset contents changed")


def seal_artifact(path, plan, **details):
    path = Path(path)
    record = {"identity": plan["identity"], "sha256": file_sha256(path), **details}
    atomic_json(record, path.with_name(path.name + ".receipt.json"))


def verified_artifact(path, plan):
    path = Path(path)
    receipt = path.with_name(path.name + ".receipt.json")
    if not receipt.exists():
        return False
    record = read_json(receipt)
    if record["identity"] != plan["identity"] or not path.is_file() or file_sha256(path) != record["sha256"]:
        raise ValueError(f"Inspection artifact changed: {path}")
    return True


def validate_generated_artifacts(result):
    for record in result.get("generated", []):
        for filename, expected in record.get("hashes", {}).items():
            if not Path(filename).is_file() or file_sha256(filename) != expected:
                raise ValueError("Stored generated-action artifact changed")


def status(output, value, **details):
    record = {"status": value, "updated_utc": datetime.now(timezone.utc).isoformat(), **details}
    atomic_json(record, Path(output) / "status.json")


def run_command(name, command, output, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Student inspection reached --max-hours; use --resume")
    logs = Path(output) / "logs"
    logs.mkdir(exist_ok=True)
    logfile = logs / f"{name}.log"
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("STMEM_") or key in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}:
            env.pop(key)
    env.update(PYTHONPATH=str(REPO / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", WANDB_MODE="disabled")
    process = None
    try:
        with logfile.open("a") as stream:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            notice = 0
            while process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError("Student inspection reached --max-hours; use --resume")
                if now - notice >= 30:
                    status(output, "running", stage=name, log=str(logfile), child_pid=process.pid)
                    print(json.dumps({"event": "student_inspection_progress", "stage": name,
                                      "log": str(logfile)}), flush=True)
                    notice = now
                time.sleep(0.5)
        if process.returncode:
            raise RuntimeError(f"{name} exited {process.returncode}; inspect {logfile}")
    except BaseException:
        terminate_stage(process)
        raise


def frozen_policy(path, device):
    policy, _ = load_policy(path, device)
    policy.freeze_encoders()
    policy.eval().requires_grad_(False)
    return policy


def stage_commands(plan, plan_path):
    base = [sys.executable, "-u", "-m", "stmem.inspection.student_suite", "--plan", str(plan_path)]
    jobs = []
    if "utility" in plan["protocol"]["stages"]:
        jobs.append({"name": "prepare", "stage": "utility", "command": [*base, "--worker", "prepare"]})
        jobs.extend({"name": f"measure_{label}", "stage": "utility", "label": label,
                     "command": [*base, "--worker", "measure", "--label", label]}
                    for label in plan["checkpoints"])
    from stmem.inspection.student_simulation import simulation_jobs
    jobs.extend(job for job in simulation_jobs(plan, plan_path)
                if job["stage"] in plan["protocol"]["stages"])
    return jobs


def job_complete(plan, job):
    receipt = Path(plan["output"]) / "completed" / f"{job['name']}.json"
    if not receipt.exists():
        return False
    record = read_json(receipt)
    if record.get("identity") != plan["identity"]:
        raise ValueError("Completed stage belongs to another inspection")
    for path, expected in record["files"].items():
        if not Path(path).is_file() or file_sha256(path) != expected:
            raise ValueError(f"Completed stage artifact changed: {path}")
    return True


def finish_job(plan, job):
    if job_complete(plan, job):
        return
    root = Path(plan["output"])
    if job["stage"] == "utility":
        selection_path = root / "states.json"
        if not verified_artifact(selection_path, plan):
            raise ValueError("Shared state selection is missing")
        selected = read_json(selection_path)
        files = [selection_path]
        for state in selected["states"]:
            key = state["state_id"]
            path = (root / "prepared" / f"{key}.pt" if job["name"] == "prepare" else
                    root / "measurements" / job["label"] / f"{key}.json")
            if not verified_artifact(path, plan):
                raise ValueError(f"Incomplete state artifact: {path}")
            files.extend([path, path.with_name(path.name + ".receipt.json")])
            if job["name"] != "prepare":
                for record in read_json(path).get("generated", []):
                    if record.get("status") == "omitted":
                        continue
                    files.extend(Path(value) for value in record["files"].values())
            else:
                context_path = root / "contexts" / f"{key}.json"
                if not verified_artifact(context_path, plan):
                    raise ValueError(f"Incomplete context manifest: {context_path}")
                files.extend([context_path, context_path.with_name(context_path.name + ".receipt.json")])
    else:
        from stmem.inspection.student_simulation import validate_simulation_job
        checked = validate_simulation_job(plan, job)
        files = [Path(path) for path in checked["files"]]
    receipt = {"identity": plan["identity"], "name": job["name"], "status": "complete",
               "files": {str(path): file_sha256(path) for path in files}}
    atomic_json(receipt, root / "completed" / f"{job['name']}.json")


def worker(plan_path, mode, label=None):
    from stmem.inspection.refinement_measurements import draw_plan, draw_seed
    from stmem.inspection.student_utility import (
        evaluate_student_state, open_local_dataset, prepare_local_state, select_local_states, summarize_utility,
    )
    plan = read_json(plan_path)
    verify_plan(plan)
    root, protocol = Path(plan["output"]), plan["protocol"]
    states_path = root / "states.json"
    if mode == "prepare":
        first = next(iter(plan["checkpoints"].values()))
        dataset = open_local_dataset(first["config"], plan["assets"])
        selected = select_local_states(dataset, first["config"], maximum=protocol["state_limit"])
        generated_positions = (np.linspace(0, len(selected) - 1,
                                          min(protocol["generated_cases"], len(selected)), dtype=int)
                               if protocol["generated_cases"] else [])
        selection = {"identity": plan["identity"], "scope": SCOPE,
                     "states": [row["state"] for row in selected],
                     "generated_state_ids": [selected[int(i)]["state"]["state_id"] for i in generated_positions],
                     "selection_before_policy_losses": True}
        if verified_artifact(states_path, plan):
            if read_json(states_path) != selection:
                raise ValueError("Gain-blind state selection changed")
        else:
            atomic_json(selection, states_path)
            seal_artifact(states_path, plan)
        policy = frozen_policy(first["path"], plan["device"])
        for row in selected:
            state = row["state"]
            path = root / "prepared" / f"{state['state_id']}.pt"
            context_path = root / "contexts" / f"{state['state_id']}.json"
            if verified_artifact(path, plan):
                if not verified_artifact(context_path, plan):
                    prepared = torch.load(path, map_location="cpu", weights_only=False)
                    atomic_json(plain(prepared["panel"]), context_path)
                    seal_artifact(context_path, plan)
                continue
            seed = draw_seed(protocol["seed"], state["task"], state["episode_id"],
                             state["decision_index"], "candidates", 0)
            prepared = prepare_local_state(policy, row["decision"], seed=seed,
                                           budgets=tuple(protocol["budgets"]), packet_cost=protocol["packet_cost"],
                                           max_candidates=protocol["max_candidates"],
                                           audit_candidates=protocol["audit_candidates"])
            prepared["state"] = state
            atomic_save(prepared, path, keep_previous=False)
            seal_artifact(path, plan, state=state, fingerprint=prepared["state_fingerprint"])
            atomic_json(plain(prepared["panel"]), context_path)
            seal_artifact(context_path, plan)
            print(json.dumps({"event": "state_prepared", **state,
                              "contexts": len(prepared["contexts"])}), flush=True)
    else:
        if label not in plan["checkpoints"]:
            raise ValueError("Unknown student label")
        if not verified_artifact(states_path, plan):
            raise ValueError("Run shared preparation before measuring students")
        selection = read_json(states_path)
        audit = plan["checkpoints"][label]
        policy = frozen_policy(audit["path"], plan["device"])
        teacher = frozen_policy(plan["teacher"]["path"], plan["device"]) if plan["teacher"] else None
        teacher_hash = plan["teacher"]["sha256"] if plan["teacher"] else None
        for state in selection["states"]:
            key = state["state_id"]
            path = root / "measurements" / label / f"{key}.json"
            if verified_artifact(path, plan):
                # Validate every associated generated artifact before reusing a state.
                validate_generated_artifacts(read_json(path))
                continue
            prepared_path = root / "prepared" / f"{key}.pt"
            if not verified_artifact(prepared_path, plan):
                raise ValueError("Shared encoded evidence is incomplete")
            prepared = torch.load(prepared_path, map_location="cpu", weights_only=False)
            draws = draw_plan(protocol["seed"], state["task"], state["episode_id"], state["decision_index"],
                              discovery=protocol["discovery_draws"], confirmation=protocol["confirmation_draws"])
            result = evaluate_student_state(policy, prepared, draws, checkpoint_hash=audit["sha256"],
                                            teacher=teacher, teacher_hash=teacher_hash,
                                            prefix_targets=protocol["execute_targets"],
                                            batch_size=protocol["batch_size"], precision=protocol["precision"])
            epsilon = max(protocol["gain_epsilon"], 10 * result["numerical"]["serial_packed_loss_max_abs"])
            summary = summarize_utility(prepared["panel"], result["rows"], epsilon=epsilon)
            result.update(identity=plan["identity"], label=label, state=state,
                          summary=summary, effective_tie_epsilon=epsilon, generated=[])
            if key in selection["generated_state_ids"]:
                from stmem.inspection.student_behavior import (
                    generate_compact_actions, select_behavior_contexts, write_behavior_case,
                )
                seeds = [draw_seed(protocol["seed"], state["task"], state["episode_id"],
                                   state["decision_index"], "generated", i)
                         for i in range(protocol["generated_draws"])]
                for reader in ("TREE", "FLAT"):
                    for budget in protocol["budgets"]:
                        try:
                            selected = select_behavior_contexts(prepared["panel"], result["rows"], reader=reader,
                                                                final_budget=budget, epsilon=epsilon,
                                                                random_seed=seeds[0])
                        except ValueError as error:
                            # Only structural eligibility may omit a preselected behavior cell.
                            families = [f for f in prepared["panel"]["families"] if f["reader"] == reader
                                        and f["panel"] == "packet" and f["final_budget"] == budget]
                            if len(families) == 1 and len(families[0]["candidates"]) >= 2:
                                raise
                            result["generated"].append({"reader": reader, "budget": budget,
                                                        "status": "omitted", "reason": str(error)})
                            continue
                        contexts = {name: prepared["contexts"][context]
                                    for name, context in selected["aliases"].items()}
                        behavior = generate_compact_actions(
                            policy, prepared["item"], contexts, seeds, checkpoint_hash=audit["sha256"],
                            teacher=teacher, teacher_hash=teacher_hash, prefix_targets=protocol["execute_targets"],
                            precision=protocol["precision"], batch_size=protocol["batch_size"])
                        case_path = root / "generated" / label / f"{key}_{reader.lower()}_b{budget}.npz"
                        files = write_behavior_case(case_path, behavior, state=state, student=label,
                                                    selection=selected, identity=plan["identity"],
                                                    scene_points=prepared["visual"]["scene_points"])
                        result["generated"].append({"reader": reader, "budget": budget, "status": "complete",
                                                    "files": files, "hashes": {p: file_sha256(p) for p in files.values()}})
            atomic_json(plain(result), path)
            seal_artifact(path, plan, label=label, state_id=key)
            print(json.dumps({"event": "student_state_measured", "label": label, **state,
                              "rows": len(result["rows"]), "numerical": result["numerical"]}), flush=True)
    verify_plan(plan)


def build_report(plan):
    root = Path(plan["output"])
    state_path = root / "states.json"
    states = read_json(state_path)["states"] if verified_artifact(state_path, plan) else []
    rows, links, missing, all_values = [], [], [], []
    for label, audit in plan["checkpoints"].items():
        values = []
        for state in states:
            path = root / "measurements" / label / f"{state['state_id']}.json"
            if not verified_artifact(path, plan):
                missing.append(str(path.relative_to(root)))
                continue
            result = read_json(path)
            validate_generated_artifacts(result)
            values.append(result)
            links.append({"label": f"{label}: raw losses/capabilities {state['episode_id']} t={state['source_index']}",
                          "path": str(path.relative_to(root))})
            for item in result.get("generated", []):
                if item.get("status") == "complete":
                    links.append({"label": f"{label}: {state['episode_id']} t={state['source_index']} "
                                           f"{item['reader']} B{item['budget']}",
                                  "path": str(Path(item["files"]["html"]).relative_to(root))})
        all_values.extend(values)
        for reader in ("TREE", "FLAT"):
            for budget in plan["protocol"]["budgets"]:
                selected = [(value["state"], item) for value in values
                            for item in value["summary"]["selection"]
                            if item["reader"] == reader and item["family"] == "packet" and item["budget"] == budget
                            and item["ranking_eligible"]]
                # First average states within each episode; do not treat candidates/draws as episodes.
                grouped = {}
                metrics = ("packet_loss", "random_loss", "packet_vs_random", "packet_vs_immediate",
                           "packet_vs_dispersed", "packet_vs_low", "winner_reversal")
                for state, item in selected:
                    episode = state["task"], state["episode_id"]
                    for metric in metrics:
                        if item[metric] is not None:
                            grouped.setdefault(metric, {}).setdefault(episode, []).append(float(item[metric]))
                averages = {metric: float(np.mean([np.mean(v) for v in episodes.values()]))
                            for metric, episodes in grouped.items()}
                rows.append({"student": label, "step": audit["step"], "reader": reader, "budget": budget,
                             "states": len(selected), "episodes": len({(s["task"], s["episode_id"]) for s, _ in selected}),
                             "confirmation_episode_means": averages})
    jobs = stage_commands(plan, root / "student_inspection.json")
    unfinished = [job["name"] for job in jobs if not job_complete(plan, job)]
    from stmem.inspection.student_simulation import simulation_report_links
    links.extend(simulation_report_links(plan))
    from stmem.inspection.student_report import summarize_local_measurements, write_diagnostic_figures
    detailed = summarize_local_measurements(all_values)
    links.extend(write_diagnostic_figures(root, detailed))
    limitations = [SCOPE, "Only VideoUnmask episodes 0 and 1 support local offline utility measurements.",
                   "Two episodes cannot establish population label quality or unseen-episode generalization.",
                   "Independent noise confirmation is not an independent episode holdout.",
                   "Loss-selected contexts are diagnostics, not learned router outputs.",
                   "Full-leaf compact inputs may be outside training support; they are not a guaranteed upper bound.",
                   "Generated paths are joint-target forward kinematics, not executed motion or task success.",
                   "Known invalid-coordinate FM/generation conventions are retained, not silently repaired."]
    if not plan["matched_training_steps"]:
        limitations.append("Student steps differ; results do not isolate training-recipe effects at equal exposure.")
    if not plan["teacher"]:
        limitations.append("Teacher agreement is not measured: no exact matching teacher was supplied.")
    summary = {"schema": SCHEMA, "identity": plan["identity"], "scope": SCOPE,
               "status": "partial" if unfinished or missing else "complete", "comparisons": rows,
               "unfinished_jobs": unfinished, "missing": missing, "links": links, "limitations": limitations,
               "label_diagnostics": detailed,
               "interpretation": "Positive packet_vs_* means lower independent-confirmation expert FM loss. "
                                 "Check absolute loss, raw signed gains, numerical floor and generated paths together."}
    atomic_json(summary, root / "summary.json")
    def fmt(value):
        return "—" if value is None else f"{value:.5g}"
    table = ["| Student | Step | Reader | Budget | States | Packet vs random | Packet vs immediate | Packet vs dispersed |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    html_rows = []
    for row in rows:
        m = row["confirmation_episode_means"]
        cells = [row["student"], str(row["step"]), row["reader"], str(row["budget"]), str(row["states"]),
                 *(fmt(m.get(k)) for k in ("packet_vs_random", "packet_vs_immediate", "packet_vs_dispersed"))]
        table.append("| " + " | ".join(cells) + " |")
        html_rows.append("<tr>" + "".join(f"<td>{escape(c)}</td>" for c in cells) + "</tr>")
    (root / "summary.md").write_text("\n".join(["# Compact student inspection", "", summary["interpretation"], "",
                                              *table, "", *[f"- {note}" for note in limitations], "",
                                              *[f"- [{link['label']}]({link['path']})" for link in links]]) + "\n")
    headers = ("Student", "Step", "Reader", "Budget", "States", "Packet vs random", "Packet vs immediate", "Packet vs dispersed")
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>Compact student inspection</title>"
            "<style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:0 20px}"
            "td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}li{margin:8px 0}</style></head><body>"
            f"<h1>Compact student inspection</h1><p>Status: {summary['status']}</p>"
            f"<p>{escape(summary['interpretation'])}</p><table><tr>" +
            "".join(f"<th>{escape(h)}</th>" for h in headers) + "</tr>" + "".join(html_rows) + "</table>"
            "<h2>Generated actions and simulations</h2><ul>" +
            "".join(f"<li><a href='{escape(link['path'], quote=True)}'>{escape(link['label'])}</a></li>" for link in links) +
            "</ul><h2>Scope and interpretation</h2><ul>" + "".join(f"<li>{escape(n)}</li>" for n in limitations) +
            "</ul><p><a href='summary.json'>Numerical summary</a> · <a href='student_inspection.json'>Protocol and identities</a>"
            " · <a href='states.json'>Fixed decision panel</a></p></body></html>")
    (root / "index.html").write_text(html)
    return summary


def detach(argv, output):
    logs = Path(output) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    logfile = logs / f"launcher_{stamp}.log"
    command = [sys.executable, "-u", "-m", "stmem.inspection.student_suite",
               *[value for value in argv if value != "--detach"]]
    with logfile.open("x") as stream:
        process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    result = {"pid": process.pid, "log": str(logfile), "status_path": str(Path(output) / "status.json"),
              "command": command, "note": "Bounded coordinator launched; inspect status.json for progress"}
    atomic_json(result, logs / f"launch_{stamp}.json")
    print(json.dumps(result, indent=2))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--teacher", type=Path, help="Optional exact matching teacher; never an older substitute")
    parser.add_argument("--assets-config", type=Path, default=REPO / "configs/inspection.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--budgets", type=int, nargs="+", default=[128])
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--draws", type=int, default=8, help="Draws in EACH independent discovery/confirmation bank")
    parser.add_argument("--generated-cases", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--stages", choices=("utility", "reset", "rollouts", "handoff"), nargs="+",
                        default=["utility", "reset"])
    parser.add_argument("--max-hours", type=float, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    modes.add_argument("--report-only", action="store_true")
    parser.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", choices=("prepare", "measure"), help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.worker:
        if not args.plan:
            parser.error("Worker requires --plan")
        worker(args.plan, args.worker, args.label)
        return 0
    if not args.bundle:
        parser.error("--bundle is required")
    if (not 1 <= args.states <= 32 or not 1 <= args.candidates <= 8 or not 2 <= args.draws <= 16
            or not 0 <= args.generated_cases <= min(args.states, 16) or not 1 <= args.batch_size <= 64
            or any(value not in (64, 128, 256) for value in args.budgets)):
        parser.error("Use states1–32,candidates1–8,draws2–16,generated-cases0–min(states,16),batch1–64,B64/128/256")
    if len(set(args.stages)) != len(args.stages):
        parser.error("Stages must be unique")
    if not math.isfinite(args.max_hours) or args.max_hours <= 0:
        parser.error("--max-hours must be finite and positive")
    if args.detach and (args.audit_only or args.report_only or args.dry_run):
        parser.error("--detach applies only to diagnostic execution")
    plan = create_plan(args)
    output = Path(plan["output"])
    plan_path = output / "student_inspection.json"
    if plan_path.exists():
        if not args.resume:
            raise FileExistsError("Inspection exists; repeat with --resume")
        if read_json(plan_path) != plan:
            raise ValueError("Inspection inputs/protocol/runtime changed; use a new output directory")
    if args.dry_run:
        print(json.dumps({"plan": plan, "commands": stage_commands(plan, plan_path)}, indent=2))
        return 0
    if args.detach:
        detach(argv, output)
        return 0
    with inspection_lock(output), termination_signals():
        if plan_path.exists():
            if not args.resume or read_json(plan_path) != plan:
                raise ValueError("Inspection ownership changed while acquiring lock")
        else:
            extras = [p for p in output.iterdir() if p.name not in (".inspection.lock", "logs")]
            if extras:
                raise FileExistsError("Nonempty output has no matching inspection plan")
            atomic_json(plan, plan_path)
        if args.audit_only:
            status(output, "audit_complete", plan=str(plan_path))
            return 0
        try:
            if not args.report_only:
                deadline = time.monotonic() + args.max_hours * 3600
                for job in stage_commands(plan, plan_path):
                    if not job_complete(plan, job):
                        reuse = False
                        if job["stage"] != "utility":
                            from stmem.inspection.student_simulation import prepare_simulation_job
                            reuse = prepare_simulation_job(plan, job)["reuse"]
                        if not reuse:
                            run_command(job["name"], job["command"], output, deadline)
                    finish_job(plan, job)
            verify_plan(plan)
            summary = build_report(plan)
            status(output, summary["status"], report=str(output / "index.html"))
        except BaseException as error:
            # Completed measurements remain useful even if a later simulator
            # stage is interrupted. Preserve the original error if reporting
            # itself finds an invalid artifact.
            report_error = None
            try:
                build_report(plan)
            except Exception as failure:
                report_error = f"{type(failure).__name__}: {failure}"
            status(output, "interrupted" if isinstance(error, (KeyboardInterrupt, TimeoutError)) else "failed",
                   error=f"{type(error).__name__}: {error}", report_error=report_error)
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
