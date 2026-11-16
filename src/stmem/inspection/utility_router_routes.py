"""Fixed-development diagnostic of complete, deployable router-built contexts.

Seven selectors start at the causal coarse archive and fill the same budget.
Expert actions and FM draws are revealed only after all routes have finished.
This measures final-context prediction losses, not closed-loop task success.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from stmem.config import digest, source_revision
from stmem.inspection.refinement_data import UtilityDataReader, read_json, stable_seed
from stmem.inspection.refinement_measurements import action_mask_summary, draw_plan, evaluate_state
from stmem.inspection.refinement_statistics import episode_bootstrap
from stmem.inspection.refinement_utility import lock, publish_measurement_identity
from stmem.memory.utility_routing import candidate_features, query_features, route
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.packed_context import _take, _tree_indices, columns_bank, flat_residual_columns
from stmem.training.runtime import load_policy
from stmem.tracking import RunTracker

SCHEMA = "utility_router_routes_v1"
ROUTERS = ("tree_immediate", "tree_packet", "flat_immediate", "flat_packet")
METHODS = (*ROUTERS, "tree_random", "flat_random", "uniform")
METRICS = ("expert_full", "expert_prefix", "teacher_full", "teacher_prefix")


def seal_manifest(value):
    body = {key: item for key, item in value.items() if key != "experiment_id"}
    return {**body, "experiment_id": digest(body)}


def build_manifest(study_path, study, parent_runtime, routers, source, workers=4, batch_size=32):
    """Adopt an already validated older pilot into a separately owned diagnostic."""
    study_path = Path(study_path).resolve()
    if read_json(study_path) != study:
        raise ValueError("Prepared study differs from the pinned pilot file")
    if set(routers) != set(ROUTERS):
        raise ValueError("Complete-route diagnostic requires all four frozen routers")
    if not isinstance(workers, int) or not isinstance(batch_size, int) or workers < 1 or batch_size < 1:
        raise ValueError("Worker and batch counts must be positive integers")
    protocol = deepcopy(study["protocol"])
    if (protocol["budget"] != 128 or protocol["execute_targets"] != 8
            or protocol["flow_steps"] != 8 or protocol.get("controller") != "packet"):
        raise ValueError("Complete-route diagnostic requires the fixed B128/execute8/flow8 packet protocol")
    if study["student"]["name"] != "S_STD" or study["student"]["step"] != 95000:
        raise ValueError("This development diagnostic requires the declared S_STD95k student")
    states = [deepcopy(state) for state in study["states"] if state["partition"] == "development"]
    if len(states) != 128 or len({s["state_id"] for s in states}) != 128:
        raise ValueError("Complete-route diagnostic must use all 128 fixed development states")
    normalized = {}
    for name, entry in routers.items():
        reader, target = name.split("_")
        path = Path(entry.get("checkpoint", entry.get("path", ""))).resolve()
        if (entry["reader"].upper() != reader.upper() or entry["target"] != target
                or file_sha256(path) != entry["sha256"]):
            raise ValueError("Router checkpoint identity or declared reader/target changed")
        normalized[name] = {**entry, "checkpoint": str(path), "reader": reader.upper()}
    protocol.update(seed=stable_seed(protocol["seed"], "complete_routes_v1"),
                    discovery_draws=4, confirmation_draws=4, workers=workers, batch_size=batch_size,
                    scope="fixed_development_full_route_diagnostic",
                    route_distribution="complete_learned_or_gain_blind_routes",
                    methods=list(METHODS), fresh_draw_namespace="complete_routes_v1",
                    confirmation_status="fresh_draws_on_development_states_not_reserved_final_episodes")
    manifest = seal_manifest({
        "schema": SCHEMA, "source": deepcopy(source),
        "study": {"path": str(study_path), "sha256": file_sha256(study_path),
                  "experiment_id": study["experiment_id"], "source": deepcopy(study["source"])},
        "parent_runtime": deepcopy(parent_runtime),
        "configuration": deepcopy(study["configuration"]), "dependencies": deepcopy(study["dependencies"]),
        "student": deepcopy(study["student"]), "data_identity": deepcopy(study["data_identity"]),
        "states": states, "routers": normalized, "protocol": protocol,
        "not_closed_loop_success": True,
        "note": "All complete routes are fixed before fresh expert FM draws; reserved final episodes remain unused.",
    })
    _validate_parent_receipt(manifest)
    return manifest


def _validate_parent_receipt(manifest):
    adopted, parent = manifest["parent_runtime"], manifest["study"]
    if file_sha256(adopted["path"]) != adopted["sha256"]:
        raise ValueError("Original-runtime verification receipt changed")
    receipt = read_json(adopted["path"])
    expected = {"status": "complete", "study_path": parent["path"],
                "study_sha256": parent["sha256"], "experiment_id": parent["experiment_id"],
                "source_sha256": parent["source"]["source_sha256"]}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Original-runtime verification receipt belongs to another pilot or failed")
    if adopted["source_sha256"] != expected["source_sha256"]:
        raise ValueError("Adopted source identity differs from the verified pilot")
    if file_sha256(receipt["source_receipt"]) != receipt["source_receipt_sha256"]:
        raise ValueError("Original exported source receipt changed")
    source = read_json(receipt["source_receipt"])
    if (source.get("source_sha256") != expected["source_sha256"]
            or source.get("git") != parent["source"]["git"]):
        raise ValueError("Original exported source receipt belongs to another revision")


def verify_manifest(path, *, verify_source=True, verify_dependencies=True):
    """Validate NEW runtime ownership after the exact old-runtime preflight."""
    from stmem.training.utility_router_study import verify_study

    manifest = read_json(path)
    if manifest.get("schema") != SCHEMA or seal_manifest(manifest) != manifest:
        raise ValueError("Complete-route manifest schema or content hash changed")
    if verify_source and manifest["source"]["source_sha256"] != source_revision()["source_sha256"]:
        raise ValueError("Complete-route diagnostic source changed; use its frozen source export")
    _validate_parent_receipt(manifest)
    parent = manifest["study"]
    if file_sha256(parent["path"]) != parent["sha256"]:
        raise ValueError("Frozen pilot study changed")
    # This separately owned stage explicitly adopts the exact original-runtime
    # preflight above. The old study, old checks, and old source stay untouched.
    study = verify_study(parent["path"], verify_source=False, verify_dependencies=verify_dependencies)
    if study["experiment_id"] != parent["experiment_id"] or study["source"] != parent["source"]:
        raise ValueError("Adopted pilot identity differs")
    expected = build_manifest(parent["path"], study, manifest["parent_runtime"], manifest["routers"],
                              manifest["source"], manifest["protocol"]["workers"],
                              manifest["protocol"]["batch_size"])
    if expected != manifest:
        raise ValueError("Diagnostic inputs or protocol differ from the adopted pilot")
    return manifest


class CachedRouteArchive:
    """Same column/scoring semantics as LiveIndexedArchive, from causal cache rows.

    Payload is read once for all selectors. Cost reporting distinguishes this
    common cached-input read from each controller's actual work; these timings
    are not claimed to be streaming simulator latency.
    """
    def __init__(self, reader, group, archive, width, clocks):
        started = time.perf_counter()
        self.archive, self.width, self.clocks = archive, width, clocks
        active = np.asarray(sorted(archive.active), dtype=np.int64)
        self.positions = {int(row): i for i, row in enumerate(active)}
        self.columns = reader.packed._columns(group["nodes"], active, include_max=True, index=archive.index)
        self.leaf_columns = self.take(archive.leaves)
        self.payload_bytes = sum(value.numel() * value.element_size() for value in self.columns.values()
                                 if isinstance(value, torch.Tensor))
        self.input_seconds = time.perf_counter() - started

    def take(self, rows):
        return _take(self.columns, np.asarray([self.positions[int(row)] for row in rows], dtype=np.int64))

    def bank(self, context):
        context = self.archive.validate(context)
        columns = (self.take(context["rows"]) if context["reader"] == "TREE" else
                   flat_residual_columns(self.leaf_columns, *self.archive.flat_arrays(context)))
        return columns_bank(columns, self.clocks, self.width)


@torch.no_grad()
def construct_routes(live, current, text, clocks, routers, protocol, seed):
    """Exact online bounded-candidate packet controller, including random draws."""
    contexts, methods = {}, {}
    for method in METHODS:
        # A live replan starts a fresh IndexedArchive. Sharing immutable payload
        # must not give later selectors another controller's lazy ordering work.
        live.archive._order_cache.clear()
        started = time.perf_counter()
        if method == "uniform":
            # This is the global UNIFORM baseline used by choose_records, not
            # 'largest support in a subsampled packet pool'. Same heap/tie rules.
            rows = _tree_indices(live.archive.index, live.archive.roots, protocol["budget"],
                                 random.Random(seed), uniform=True)
            context = live.archive.context("TREE", rows)
            work = {"controller": "global_uniform_heap", "actual_count": context["token_count"],
                    "coverage_hash": context["coverage_hash"], "steps": [], "scorer_calls": 0,
                    "candidates_scored": 0, "feature_records_fetched": 0,
                    "scoring_transfer_bytes": 0, "selection_seconds": time.perf_counter() - started,
                    "budget": protocol["budget"], "saturated": len(live.archive.leaves) <= protocol["budget"]}
        else:
            reader, selection = method.split("_")
            rng = random.Random(seed)
            model = None if selection == "random" else routers[method]
            fetched, transferred = 0, 0

            def score(context, packets):
                nonlocal fetched, transferred
                if selection == "random":
                    return [rng.random() for _ in packets]
                cost = packets[0]["cost"]
                query = query_features(current, text, live.bank(context), clocks, width=live.width,
                                       budget=protocol["budget"], packet_cost=cost)
                candidates = candidate_features(live.take([p["region"] for p in packets]),
                                                width=live.width, cost=cost, budget=protocol["budget"])
                fetched += context["token_count"] + len(packets)
                device = next(model.parameters()).device
                if device.type != "cpu":
                    transferred += 4 * (query.numel() + candidates.numel() + len(packets))
                return model(query.to(device), candidates.to(device)).float().cpu().numpy()

            context, work = route(live.archive, reader.upper(), protocol["budget"], score, seed=seed,
                                  candidate_limit=protocol["candidate_limit"],
                                  packet_max_cost=protocol["packet_max_cost"], mode="packet")
            work.update(controller="packet", selection="random" if model is None else "learned",
                        feature_records_fetched=fetched, scoring_transfer_bytes=transferred,
                        selection_seconds=time.perf_counter() - started)
        live.archive.validate(context)
        if context["token_count"] != min(protocol["budget"], len(live.archive.leaves)):
            raise AssertionError("Complete controller stopped before filling its attainable budget")
        contexts[context["context_id"]] = context
        methods[method] = {"context_id": context["context_id"], "route": work,
                           "description": live.archive.describe(context)}
    if len({c["coverage_hash"] for c in contexts.values()}) != 1:
        raise AssertionError("Compared routes do not cover identical causal evidence")
    return contexts, methods


def _load_routers(manifest, device):
    from stmem.training.utility_routers import load_utility_router

    result = {}
    for name, entry in manifest["routers"].items():
        model, state = load_utility_router(entry["checkpoint"], device=device,
                                          expected_student_sha256=manifest["student"]["sha256"],
                                          expected_teacher_sha256=manifest["dependencies"]["teacher_sha256"],
                                          expected_reader=entry["reader"])
        if (state["target"] != entry["target"] or state["step"] != entry["step"]
                or state["study"]["experiment_id"] != manifest["study"]["experiment_id"]):
            raise ValueError("Frozen router target, endpoint or pilot identity differs")
        for key in ("budget", "execute_targets", "flow_steps", "candidate_limit", "packet_max_cost", "controller"):
            if state["utility_protocol"].get(key) != manifest["protocol"].get(key):
                raise ValueError(f"Router online protocol differs at {key}")
        result[name] = model
    return result


def _state_path(root, state):
    return Path(root) / "states" / f"{state['state_id']}.json"


def _write_state(root, manifest, state, payload):
    record = {"schema": SCHEMA, "experiment_id": manifest["experiment_id"], "state": state, **payload}
    record["state_sha256"] = digest(record)
    path = _state_path(root, state)
    atomic_json(record, path)
    return {"sha256": file_sha256(path), "state_sha256": record["state_sha256"]}


def read_state(root, manifest, state):
    path = _state_path(root, state)
    record = read_json(path)
    if (record.get("schema") != SCHEMA or record.get("experiment_id") != manifest["experiment_id"]
            or record.get("state") != state
            or digest({k: v for k, v in record.items() if k != "state_sha256"}) != record.get("state_sha256")):
        raise ValueError("Route state identity or sealed content changed")
    if set(record["methods"]) != set(METHODS):
        raise ValueError("Route state is missing matched selectors")
    protocol = manifest["protocol"]
    draws = draw_plan(protocol["seed"], state["task"], state["episode_id"], state["decision_index"],
                      discovery=protocol["discovery_draws"], confirmation=protocol["confirmation_draws"])
    if record["draws"] != draws:
        raise ValueError("Complete routes do not use the predeclared fresh paired draw banks")
    keys = {(row["context_id"], row["bank"], row["draw_index"], row["seed"]) for row in record["rows"]}
    expected = {(context, draw["bank"], draw["draw_index"], draw["seed"])
                for context in record["contexts"] for draw in draws}
    if len(keys) != len(record["rows"]) or keys != expected:
        raise ValueError("Complete-route measurements lack exact matched context/draw coverage")
    for row in record["rows"]:
        if not all(math.isfinite(row[key]) and row[key] >= 0 for key in METRICS):
            raise ValueError("Final-context losses must be finite and nonnegative")
    coverage = {c["coverage_hash"] for c in record["contexts"].values()}
    for method in record["methods"].values():
        context = record["contexts"][method["context_id"]]
        if (context["token_count"] != min(protocol["budget"], record["available_leaves"])
                or method["route"]["coverage_hash"] != context["coverage_hash"]):
            raise ValueError("Complete-route context does not fill the exact causal budget")
    if len(coverage) != 1:
        raise ValueError("Complete-route contexts differ in available evidence")
    return record


def measure(args):
    from stmem.training.label_experiment import validate_models

    manifest = verify_manifest(args.manifest)
    protocol, cfg = manifest["protocol"], manifest["configuration"]
    if (args.workers != protocol["workers"] or args.batch_size != protocol["batch_size"]
            or not 0 <= args.worker_index < args.workers
            or not math.isfinite(args.max_hours) or args.max_hours <= 0):
        raise ValueError("Worker/batch settings must match the sealed diagnostic; time limit must be positive")
    root = Path(args.output)
    publish_measurement_identity(root, {"schema": SCHEMA, "experiment_id": manifest["experiment_id"],
                                        "manifest_sha256": file_sha256(args.manifest)})
    selected = manifest["states"][args.worker_index::args.workers]
    receipt_path = root / f"worker_{args.worker_index}.json"
    receipt = {"schema": SCHEMA, "experiment_id": manifest["experiment_id"], "status": "running",
               "worker_index": args.worker_index, "planned": len(selected), "completed": 0,
               "planned_state_ids": [state["state_id"] for state in selected],
               "completed_state_ids": [], "states": {}}
    started, stopped, owned, reader = time.monotonic(), False, False, None

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    previous_project = os.environ.get("WANDB_PROJECT")
    os.environ["WANDB_PROJECT"] = "chronotree-routers"
    tracker_cfg = deepcopy(cfg)
    tracker_cfg.setdefault("tracking", {}).update(project="chronotree-routers", system_stats=True)
    try:
        with lock(root / f"worker_{args.worker_index}.lock"):
            if receipt_path.exists():
                # Bind resumed states to their previously committed bytes,
                # not just a self-seal that could have been recomputed.
                verify_results(manifest, root)
            owned = True
            with RunTracker(tracker_cfg, root / "tracking" / f"worker_{args.worker_index}",
                            job_type="utility_complete_routes", name=f"utility-complete-routes-{args.worker_index}",
                            metadata={"experiment_id": manifest["experiment_id"], "schema": SCHEMA,
                                      "student_sha256": manifest["student"]["sha256"]},
                            resume=receipt_path.exists()) as tracker:
                teacher, ts = load_policy(manifest["dependencies"]["teacher"], args.device)
                student, ss = load_policy(manifest["student"]["checkpoint"], args.device)
                validate_models(cfg, teacher, ts, student, ss)
                for model in (teacher, student):
                    model.freeze_encoders()
                    model.eval().requires_grad_(False)
                routers = _load_routers(manifest, args.device)
                data = TrainingData(cfg)
                if data.identity != manifest["data_identity"]:
                    raise ValueError("Training action-index identity changed")
                reader = UtilityDataReader(data, cfg, manifest["dependencies"]["encoded"],
                                           manifest["dependencies"]["hierarchy"])
                for state in selected:
                    if stopped or time.monotonic() - started >= args.max_hours * 3600:
                        break
                    if not _state_path(root, state).exists():
                        state_started = time.monotonic()
                        _, group, archive = reader.archive(state)
                        item, _ = reader.prepare(state, {})
                        if not action_mask_summary(item)["full_horizon"]:
                            raise ValueError("Fixed development state has incomplete expert horizon")
                        live = CachedRouteArchive(reader, group, archive, cfg["model"]["width"], item["clocks"])
                        route_seed = stable_seed(protocol["seed"], state["state_id"], "selection")
                        contexts, methods = construct_routes(live, item["current"], item["text"], item["clocks"],
                                                             routers, protocol, route_seed)
                        # Routing has now finished. Only here may expert/noise
                        # inputs enter measurement of the frozen final contexts.
                        banks = {key: live.bank(context) for key, context in contexts.items()}
                        draws = draw_plan(protocol["seed"], state["task"], state["episode_id"], state["decision_index"],
                                          discovery=protocol["discovery_draws"],
                                          confirmation=protocol["confirmation_draws"])
                        result = evaluate_state(teacher, student, item, banks, draws,
                                                teacher_hash=manifest["dependencies"]["teacher_sha256"],
                                                student_hash=manifest["student"]["sha256"],
                                                precision=protocol.get("precision", "fp32"),
                                                compact_batch_size=args.batch_size,
                                                prefix_targets=protocol["execute_targets"])
                        _write_state(root, manifest, state, {
                            "contexts": contexts, "methods": methods, "draws": draws, "rows": result["rows"],
                            "route_seed": route_seed, "available_leaves": len(archive.leaves),
                            "available_roots": len(archive.roots), "mask": result["mask"],
                            "timings": {**result["timings"], "shared_cache_read_seconds": live.input_seconds,
                                        "shared_cache_payload_bytes": live.payload_bytes,
                                        "complete_state_wall_seconds": time.monotonic() - state_started}})
                    record = read_state(root, manifest, state)
                    receipt["states"][state["state_id"]] = {"sha256": file_sha256(_state_path(root, state)),
                                                          "state_sha256": record["state_sha256"]}
                    receipt["completed_state_ids"].append(state["state_id"])
                    receipt.update(completed=len(receipt["completed_state_ids"]),
                                   elapsed_seconds=time.monotonic() - started)
                    atomic_json(receipt, receipt_path)
                    tracker.log({"routes/states_completed": receipt["completed"], "routes/states_planned": len(selected),
                                 "routes/elapsed_seconds": receipt["elapsed_seconds"]}, step=receipt["completed"])
                    print(json.dumps({"event": "complete_routes_progress", "worker": args.worker_index,
                                      "completed": receipt["completed"], "planned": len(selected),
                                      "state_id": state["state_id"]}), flush=True)
                receipt["status"] = "complete" if receipt["completed"] == len(selected) else "partial"
                atomic_json(receipt, receipt_path)
                tracker.summary({"status": receipt["status"], "states": receipt["completed"]})
    except Exception as error:
        if owned:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
            atomic_json(receipt, receipt_path)
        raise
    finally:
        if reader is not None:
            reader.close()
        for sig, previous in handlers.items():
            signal.signal(sig, previous)
        if previous_project is None:
            os.environ.pop("WANDB_PROJECT", None)
        else:
            os.environ["WANDB_PROJECT"] = previous_project
    return receipt


def status(manifest, output):
    """Read-only receipt inventory; a partial experiment is never summarized as complete."""
    workers, total = [], 0
    for worker in range(manifest["protocol"]["workers"]):
        path = Path(output) / f"worker_{worker}.json"
        if not path.exists():
            workers.append({"worker_index": worker, "status": "missing", "completed": 0})
            continue
        receipt = read_json(path)
        planned = [state["state_id"] for state in manifest["states"][worker::manifest["protocol"]["workers"]]]
        completed = receipt.get("completed_state_ids", [])
        if (receipt.get("schema") != SCHEMA or receipt.get("experiment_id") != manifest["experiment_id"]
                or receipt.get("worker_index") != worker or receipt.get("planned_state_ids") != planned
                or len(completed) != len(set(completed)) or completed != planned[:len(completed)]
                or receipt.get("completed") != len(completed) or receipt.get("planned") != len(planned)
                or set(receipt.get("states", {})) != set(completed)
                or (receipt["status"] == "complete" and completed != planned)):
            raise ValueError("Complete-route worker receipt identity or coverage differs")
        workers.append({"worker_index": worker, "status": receipt["status"], "completed": len(completed)})
        total += len(completed)
    return {"schema": SCHEMA, "experiment_id": manifest["experiment_id"], "planned": len(manifest["states"]),
            "completed": total, "workers": workers,
            "status": "complete" if all(w["status"] == "complete" for w in workers) else "partial"}


def verify_results(manifest, output):
    """Read-only validation of every committed worker artifact, including partials."""
    inventory = status(manifest, output)
    root = Path(output)
    for worker in inventory["workers"]:
        if worker["status"] == "missing":
            continue
        receipt = read_json(root / f"worker_{worker['worker_index']}.json")
        planned = manifest["states"][worker["worker_index"]::manifest["protocol"]["workers"]]
        for state in planned[:worker["completed"]]:
            record = read_state(root, manifest, state)
            if receipt["states"][state["state_id"]] != {
                    "sha256": file_sha256(_state_path(root, state)), "state_sha256": record["state_sha256"]}:
                raise ValueError("Completed worker artifact checksum changed")
    return inventory


def report(manifest_path, output):
    manifest = verify_manifest(manifest_path)
    return report_verified(manifest, output, manifest_sha256=file_sha256(manifest_path))


def report_verified(manifest, output, *, manifest_sha256):
    """Report sealed results after the caller validates its execution manifest.

    The HPC entry point verifies its adopted study; a portable local entry point
    separately verifies its bundle and adoption. Neither rewrites the original
    study or skips the committed result checks below.
    """
    output = Path(output)
    inventory = verify_results(manifest, output)
    if inventory["status"] != "complete":
        return inventory
    states, costs = [], []
    for state in manifest["states"]:
        record = read_state(output, manifest, state)
        for method, details in record["methods"].items():
            costs.append({**state, "method": method, **details["route"],
                          **{k: v for k, v in record["timings"].items() if k.startswith("shared_cache_")}})
            for bank in ("discovery", "confirmation"):
                rows = [row for row in record["rows"] if row["context_id"] == details["context_id"] and row["bank"] == bank]
                states.append({**state, "method": method, "bank": bank,
                               **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS}})
    pairs = [(method, reader + "_random") for reader in ("tree", "flat")
             for method in (reader + "_immediate", reader + "_packet")]
    pairs += [(reader + "_packet", reader + "_immediate") for reader in ("tree", "flat")]
    pairs += [(method, "uniform") for method in METHODS if method != "uniform"]
    summaries, comparisons = {}, {}
    for bank in ("discovery", "confirmation"):
        for method in METHODS:
            rows = [row for row in states if row["method"] == method and row["bank"] == bank]
            summaries[f"{bank}/{method}"] = {metric: episode_bootstrap(rows, metric) for metric in METRICS}
        lookup = {(row["state_id"], row["method"]): row for row in states if row["bank"] == bank}
        for learned, baseline in pairs:
            rows = [{**state, **{metric: lookup[state["state_id"], baseline][metric]
                               - lookup[state["state_id"], learned][metric] for metric in METRICS}}
                    for state in manifest["states"]]
            comparisons[f"{bank}/{learned}_vs_{baseline}"] = {
                "positive_means": "lower loss for first named selector", "baseline": baseline, "selector": learned,
                **{metric: episode_bootstrap(rows, metric) for metric in METRICS}}
    cost_fields = ("selection_seconds", "candidates_scored", "scorer_calls", "index_rows_scanned",
                   "descendant_supports_examined", "feature_records_fetched", "scoring_transfer_bytes")
    costs_summary = {method: {field: episode_bootstrap([r for r in costs if r["method"] == method], field)
                             for field in cost_fields} for method in METHODS}
    summary = {**inventory, "manifest_sha256": manifest_sha256, "methods": summaries,
               "paired_comparisons": comparisons, "routing_costs": costs_summary,
               "state_metrics": states, "routing_cost_rows": costs,
               "scope": manifest["protocol"]["scope"], "not_closed_loop_success": True,
               "coverage": "All final contexts exactly cover the identical causal leaves at min(B128, available leaves)",
               "aggregation": "Draw mean within state; states within episode; episodes within task; macro across tasks",
               "interval": "95% paired episode bootstrap stratified by task; one training seed",
               "timing_note": "Cached payload is loaded once and shared across selectors; route timings exclude that common read. Not end-to-end simulator latency.",
               "confirmation_note": "Fresh confirmation noise on fixed development episodes, not reserved final evaluation episodes"}
    atomic_json(summary, output / "summary.json")
    lines = ["# Complete router route diagnostic", "", "This evaluates complete budget-filling routes, not individual refinements. "
             "All seven contexts were constructed before expert actions or fresh FM noise entered loss measurement. "
             "These are prediction diagnostics, not closed-loop success rates.", "",
             f"Student: {manifest['student']['name']} at {manifest['student']['step']:,} steps. "
             f"States: {len(manifest['states'])}; budget: {manifest['protocol']['budget']}; "
             f"executed-target prefix: {manifest['protocol']['execute_targets']}.", "",
             "Four learned packet controllers (TREE/FLAT × immediate/packet credit) are compared with two "
             "matched random packet controllers and the global UNIFORM heap. Random uses the same bounded "
             "candidate proposal protocol. UNIFORM considers all eligible frontier nodes.", "",
             "## Fresh confirmation draws", "", "Positive paired gains mean lower final-context expert FM loss. "
             "Intervals resample paired episodes within each task; noise draws and candidates are not independent episodes.", "",
             "| Selector | Baseline | Full-chunk gain [95% CI] | First executed-target gain [95% CI] |", "|---|---|---:|---:|"]
    def formatted(value):
        interval = value["interval"]
        return f"{value['mean']:.6f}" + (f" [{interval[0]:.6f}, {interval[1]:.6f}]" if interval else " [insufficient episodes]")
    for first, second in pairs:
        row = comparisons[f"confirmation/{first}_vs_{second}"]
        lines.append(f"| {first} | {second} | {formatted(row['expert_full'])} | {formatted(row['expert_prefix'])} |")
    lines += ["", "Teacher-discrepancy results, absolute losses, per-state metrics, route traces and controller costs "
              "are in `summary.json` and `states/*.json`. Teacher discrepancy is a secondary diagnostic, not the router target.",
              "", summary["timing_note"], "", summary["confirmation_note"] + ". "
              "No cases or routes were selected using confirmation losses; all declared states and selectors are included.", ""]
    (output / "report.md").write_text("\n".join(lines))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("measure", "report", "status"), default="measure")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-hours", type=float, default=3.8)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.mode == "measure":
        result = measure(args)
    elif args.mode == "report":
        result = report(args.manifest, args.output)
    else:
        result = status(verify_manifest(args.manifest), args.output)
    print(json.dumps({key: result[key] for key in ("status", "experiment_id", "completed", "planned")}, indent=2))
    return 0 if result["status"] == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
