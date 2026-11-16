"""Run the fixed router diagnostics from a checksummed portable evidence bundle.

No training data, HPC path remapping or old-manifest rewriting is performed.
The local adoption binds the exported evidence, exact frozen weights, source and
batch setting. Simulator success remains a separate evaluation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import time

import torch

from stmem.config import digest, source_revision
from stmem.inspection.refinement_data import read_json, stable_seed
from stmem.inspection.refinement_generated import load_frozen_policy, _write_result
from stmem.inspection.refinement_measurements import action_mask_summary, draw_plan, evaluate_state, generated_state
from stmem.inspection.refinement_utility import lock
from stmem.inspection import utility_router_routes as routes
from stmem.tracking import RunTracker
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.label_experiment import validate_models
from stmem.training.utility_routers import load_utility_router

SCHEMA = "portable_utility_local_adoption_v1"


def _immutable_json(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError("Local diagnostic identity changed; choose a new output directory")
    if not path.exists():
        atomic_json(value, path)


def local_manifest(bundle, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive integer batch size required")
    protocol = deepcopy(bundle["protocol"])
    protocol.update(workers=1, batch_size=batch_size, scope="portable_fixed_development_full_routes")
    value = {"schema": SCHEMA, "bundle_id": bundle["bundle_id"], "source": source_revision(),
             "configuration": deepcopy(bundle["configuration"]), "protocol": protocol,
             "student": deepcopy(bundle["student"]), "states": deepcopy(bundle["states"]),
             "models": deepcopy(bundle["models"]),
             "generated_plan_id": bundle["generated_plan"]["plan_id"],
             "generated_cases": deepcopy(bundle["generated_cases"]),
             "note": "Local execution of the original fixed development evidence; no replacement cases or training"}
    # Dirty status / Git ID of a copied export are not scientific semantics; the
    # complete source fingerprint remains the required identity.
    value["source"] = {"source_sha256": value["source"]["source_sha256"]}
    value["experiment_id"] = digest(value)
    return value


def adopt(bundle, output, batch_size):
    manifest = local_manifest(bundle, batch_size)
    output.mkdir(parents=True, exist_ok=True)
    with lock(output / "adoption.lock", blocking=True):
        _immutable_json(output / "manifest.json", manifest)
    return manifest


def _model_path(root, manifest, name):
    entry = manifest["models"][name]
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or file_sha256(path) != entry["sha256"]:
        raise ValueError(f"Portable model changed or escaped the bundle: {name}")
    return path


def _policies(root, manifest, student_name, device):
    teacher, teacher_state = load_frozen_policy(_model_path(root, manifest, "teacher"), device)
    student, student_state = load_frozen_policy(_model_path(root, manifest, student_name), device)
    validate_models(manifest["configuration"], teacher, teacher_state, student, student_state)
    return teacher, student


def _routers(root, manifest, device):
    values = {}
    for name in routes.ROUTERS:
        entry = manifest["models"][name]
        reader, target = name.split("_")
        model, checkpoint = load_utility_router(
            _model_path(root, manifest, name), device=device,
            expected_student_sha256=manifest["models"]["S_STD"]["sha256"],
            expected_teacher_sha256=manifest["models"]["teacher"]["sha256"], expected_reader=reader)
        if checkpoint["target"] != target or checkpoint["step"] != entry["step"]:
            raise ValueError("Portable router target or endpoint changed")
        for key in ("budget", "execute_targets", "flow_steps", "candidate_limit", "packet_max_cost", "controller"):
            if checkpoint["utility_protocol"].get(key) != manifest["protocol"].get(key):
                raise ValueError(f"Portable router protocol differs: {key}")
        values[name] = model
    return values


@contextmanager
def _runtime():
    stopped = []
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    previous_project = os.environ.get("WANDB_PROJECT")
    try:
        for sig in handlers:
            signal.signal(sig, lambda number, _frame: stopped.append(number))
        os.environ["WANDB_PROJECT"] = "chronotree-routers"
        yield stopped
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if previous_project is None:
            os.environ.pop("WANDB_PROJECT", None)
        else:
            os.environ["WANDB_PROJECT"] = previous_project


def _tracking(manifest):
    cfg = deepcopy(manifest["configuration"])
    cfg.setdefault("tracking", {}).update(project="chronotree-routers", system_stats=True)
    return cfg


def run_routes(bundle_root, bundle, manifest, output, *, device, max_hours, resume):
    from stmem.inspection.utility_router_bundle import load_state

    root = output / "routes"
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "worker_0.json"
    planned = manifest["states"]
    started = time.monotonic()
    receipt = {"schema": routes.SCHEMA, "experiment_id": manifest["experiment_id"], "status": "running",
               "worker_index": 0, "planned": len(planned), "completed": 0,
               "planned_state_ids": [s["state_id"] for s in planned], "completed_state_ids": [], "states": {}}
    with lock(root / "worker_0.lock"):
        prior = routes.verify_results(manifest, root)
        if prior["status"] == "complete":
            return prior
        if receipt_path.exists() and not resume:
            raise ValueError("Partial local routes exist; use --resume")
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            receipt.update(status="running")
            receipt.pop("error", None)
        with _runtime() as stopped, RunTracker(
                _tracking(manifest), root / "tracking", job_type="utility_complete_routes",
                name="utility-portable-complete-routes", metadata={"experiment_id": manifest["experiment_id"],
                "bundle_id": bundle["bundle_id"]}, resume=receipt_path.exists()) as tracker:
            try:
                teacher, student = _policies(bundle_root, manifest, "S_STD", device)
                models = _routers(bundle_root, manifest, device)
                protocol = manifest["protocol"]
                for state in planned:
                    if state["state_id"] in receipt["states"]:
                        continue
                    if stopped or time.monotonic() - started >= max_hours * 3600:
                        break
                    if not routes._state_path(root, state).exists():
                        state_started = time.monotonic()
                        item, archive, _ = load_state(bundle_root, bundle, state["state_id"])
                        if not action_mask_summary(item)["full_horizon"]:
                            raise ValueError("Fixed development state has incomplete expert horizon")
                        route_seed = stable_seed(protocol["seed"], state["state_id"], "selection")
                        contexts, methods = routes.construct_routes(archive, item["current"], item["text"],
                            item["clocks"], models, protocol, route_seed)
                        banks = {key: archive.bank(spec) for key, spec in contexts.items()}
                        draws = draw_plan(protocol["seed"], state["task"], state["episode_id"], state["decision_index"],
                                          discovery=protocol["discovery_draws"], confirmation=protocol["confirmation_draws"])
                        result = evaluate_state(teacher, student, item, banks, draws,
                            teacher_hash=manifest["models"]["teacher"]["sha256"],
                            student_hash=manifest["models"]["S_STD"]["sha256"],
                            precision=protocol.get("precision", "fp32"), compact_batch_size=protocol["batch_size"],
                            prefix_targets=protocol["execute_targets"])
                        routes._write_state(root, manifest, state, {
                            "contexts": contexts, "methods": methods, "draws": draws, "rows": result["rows"],
                            "route_seed": route_seed, "available_leaves": len(archive.archive.leaves),
                            "available_roots": len(archive.archive.roots), "mask": result["mask"],
                            "timings": {**result["timings"], "complete_state_wall_seconds": time.monotonic() - state_started}})
                    record = routes.read_state(root, manifest, state)
                    receipt["states"][state["state_id"]] = {
                        "sha256": file_sha256(routes._state_path(root, state)), "state_sha256": record["state_sha256"]}
                    receipt["completed_state_ids"].append(state["state_id"])
                    receipt.update(completed=len(receipt["completed_state_ids"]), elapsed_seconds=time.monotonic() - started)
                    atomic_json(receipt, receipt_path)
                    tracker.log({"routes/states_completed": receipt["completed"], "routes/states_planned": len(planned)},
                                step=receipt["completed"])
                    print(json.dumps({"event": "portable_routes_progress", "completed": receipt["completed"],
                                      "planned": len(planned)}), flush=True)
                receipt["status"] = "complete" if receipt["completed"] == len(planned) else "partial"
                atomic_json(receipt, receipt_path)
                tracker.summary({"status": receipt["status"], "states": receipt["completed"]})
            except Exception as error:
                receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
                atomic_json(receipt, receipt_path)
                raise
    return receipt


def _generation_cells(bundle):
    return [(name, case) for name in sorted(bundle["generated_plan"]["students"]) for case in bundle["generated_cases"]]


def _validate_generated(path, bundle, manifest, name, case):
    saved, plan = read_json(path.with_suffix(".json")), bundle["generated_plan"]
    expected = {"plan_id": plan["plan_id"], "local_experiment_id": manifest["experiment_id"],
                "student": name, "state": case["state"],
                "student_hash": manifest["models"][name]["sha256"],
                "teacher_hash": manifest["models"]["teacher"]["sha256"],
                "seeds": case["generation_seeds"], "integration_steps": plan["integration_steps"],
                "prefix_targets": plan["execute_targets"], "precision": plan["precision"],
                "contexts": case["students"][name]["contexts"]}
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Generated local output belongs to another case, model or protocol")
    if file_sha256(path) != saved["arrays_sha256"]:
        raise ValueError("Generated local output checksum changed")
    return saved


def generated_status(bundle, manifest, output):
    root = output / "generated"
    receipt_path = root / "receipt.json"
    planned = [[name, case["state_id"]] for name, case in _generation_cells(bundle)]
    if not receipt_path.exists():
        return {"status": "missing", "completed": 0, "planned": len(planned)}
    receipt = read_json(receipt_path)
    done = receipt["completed_cells"]
    if (receipt.get("experiment_id") != manifest["experiment_id"] or receipt.get("planned_cells") != planned
            or done != planned[:len(done)] or len(done) != receipt["completed"]
            or set(receipt.get("files", {})) != {f"{name}/{state_id}" for name, state_id in done}
            or (receipt["status"] == "complete" and done != planned)):
        raise ValueError("Generated local receipt identity or coverage changed")
    cases = {case["state_id"]: case for case in bundle["generated_cases"]}
    for name, state_id in done:
        path = root / name / f"{state_id}.npz"
        guard = receipt["files"][f"{name}/{state_id}"]
        if file_sha256(path) != guard["arrays_sha256"] or file_sha256(path.with_suffix(".json")) != guard["metadata_sha256"]:
            raise ValueError("Committed generated local output checksum changed")
        _validate_generated(path, bundle, manifest, name, cases[state_id])
    return {"status": receipt["status"], "completed": len(done), "planned": len(planned)}


def run_generated(bundle_root, bundle, manifest, output, *, device, max_hours, resume):
    from stmem.inspection.utility_router_bundle import load_state

    root = output / "generated"
    root.mkdir(parents=True, exist_ok=True)
    receipt_path, plan = root / "receipt.json", bundle["generated_plan"]
    cells = _generation_cells(bundle)
    receipt = {"experiment_id": manifest["experiment_id"], "status": "running", "completed": 0,
               "planned_cells": [[name, case["state_id"]] for name, case in cells], "completed_cells": [], "files": {}}
    started = time.monotonic()
    with lock(root / "worker.lock"):
        prior = generated_status(bundle, manifest, output)
        if prior["status"] == "complete":
            return prior
        if receipt_path.exists() and not resume:
            raise ValueError("Partial local generated results exist; use --resume")
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            receipt.update(status="running")
            receipt.pop("error", None)
        with _runtime() as stopped, RunTracker(
                _tracking(manifest), root / "tracking", job_type="utility_generated_actions",
                name="utility-portable-generated-actions", metadata={"experiment_id": manifest["experiment_id"],
                "bundle_id": bundle["bundle_id"]}, resume=receipt_path.exists()) as tracker:
            try:
                teacher, teacher_state = load_frozen_policy(_model_path(bundle_root, manifest, "teacher"), device)
                active_name, student = None, None
                for name, case in cells:
                    if f"{name}/{case['state_id']}" in receipt["files"]:
                        continue
                    if stopped or time.monotonic() - started >= max_hours * 3600:
                        break
                    path = root / name / f"{case['state_id']}.npz"
                    if not path.with_suffix(".json").exists():
                        if active_name != name:
                            if student is not None:
                                del student
                            student, student_state = load_frozen_policy(_model_path(bundle_root, manifest, name), device)
                            validate_models(manifest["configuration"], teacher, teacher_state, student, student_state)
                            active_name = name
                        item, _, contexts = load_state(bundle_root, bundle, case["state_id"])
                        result = generated_state(teacher, student, item, contexts[name], case["generation_seeds"],
                            teacher_hash=manifest["models"]["teacher"]["sha256"],
                            student_hash=manifest["models"][name]["sha256"], task=case["state"]["task"],
                            precision=plan["precision"], integration_steps=plan["integration_steps"],
                            prefix_targets=plan["execute_targets"], compact_batch_size=manifest["protocol"]["batch_size"])
                        result["current_visual_anchors"] = item["current"].metadata[
                            0, item["current"].valid[0], :3].cpu().tolist()
                        result["current_tcp"] = item["current_state"][7:10].cpu().tolist()
                        _write_result(path, result, {"plan_id": plan["plan_id"], "local_experiment_id": manifest["experiment_id"],
                            "student": name, "state": case["state"], "contexts": case["students"][name]["contexts"],
                            "selection_reasons": case["selection_reasons"]})
                    _validate_generated(path, bundle, manifest, name, case)
                    receipt["completed_cells"].append([name, case["state_id"]])
                    receipt["files"][f"{name}/{case['state_id']}"] = {
                        "arrays_sha256": file_sha256(path), "metadata_sha256": file_sha256(path.with_suffix(".json"))}
                    receipt["completed"] = len(receipt["completed_cells"])
                    atomic_json(receipt, receipt_path)
                    tracker.log({"generated/completed_cells": receipt["completed"], "generated/planned_cells": len(cells)},
                                step=receipt["completed"])
                    print(json.dumps({"event": "portable_generated_progress", "completed": receipt["completed"],
                                      "planned": len(cells)}), flush=True)
                receipt["status"] = "complete" if receipt["completed"] == len(cells) else "partial"
                atomic_json(receipt, receipt_path)
                tracker.summary({"status": receipt["status"], "cells": receipt["completed"]})
            except Exception as error:
                receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
                atomic_json(receipt, receipt_path)
                raise
    return receipt


def report(bundle, manifest, output):
    from stmem.inspection.refinement_generated import report_verified

    route_status = routes.report_verified(manifest, output / "routes",
                                         manifest_sha256=file_sha256(output / "manifest.json"))
    generation = generated_status(bundle, manifest, output)
    if generation["status"] == "complete":
        report_verified(bundle["generated_plan"], output / "generated")
    result = {"bundle_id": bundle["bundle_id"], "experiment_id": manifest["experiment_id"],
              "routes": route_status["status"], "generated": generation["status"],
              "status": "complete" if route_status["status"] == generation["status"] == "complete" else "partial",
              "reports": {"routes": str(output / "routes/report.md"), "generated": str(output / "generated/report.md"),
                          "generated_html": str(output / "generated/index.html")}}
    atomic_json(result, output / "summary.json")
    return result


def main(argv=None):
    from stmem.inspection.utility_router_bundle import verify_bundle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("all", "routes", "generated", "report", "status"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-hours", type=float, default=4.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0 or args.batch_size < 1:
        parser.error("Positive finite time and batch bounds are required")
    torch.set_num_threads(1)
    root = args.bundle.resolve()
    output = args.output.resolve() if args.output else root.parent / (root.name + "_analysis")
    bundle = verify_bundle(root)
    if args.mode in ("status", "report") and not (output / "manifest.json").exists():
        print(json.dumps({"status": "not_started", "output": str(output)}))
        return 0
    manifest = local_manifest(bundle, args.batch_size)
    if args.mode == "status":
        if read_json(output / "manifest.json") != manifest:
            raise ValueError("Local diagnostic adoption changed")
        result = {"routes": routes.verify_results(manifest, output / "routes"),
                  "generated": generated_status(bundle, manifest, output)}
    else:
        manifest = adopt(bundle, output, args.batch_size)
        started = time.monotonic()
        routes_complete = True
        if args.mode in ("routes", "all"):
            measured = run_routes(root, bundle, manifest, output, device=args.device, max_hours=args.max_hours, resume=args.resume)
            routes_complete = measured["status"] == "complete"
        remaining = args.max_hours - (time.monotonic() - started) / 3600
        if args.mode in ("generated", "all") and routes_complete and remaining > 0:
            run_generated(root, bundle, manifest, output, device=args.device, max_hours=remaining, resume=args.resume)
        result = report(bundle, manifest, output)
    print(json.dumps(result, indent=2))
    if args.mode in ("status", "report"):
        return 0
    stage = result.get("status") if args.mode == "all" else result.get(args.mode)
    return 0 if stage == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
