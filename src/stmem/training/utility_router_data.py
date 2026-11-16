"""Fresh expert-FM first-move/packet labels for a frozen developmental student.

One transaction stores all readers, reachable frontiers and shared noise draws
for a state.  This schema cannot be mistaken for the older teacher-agreement
gain files.  Preparation fixes fitting episodes and sampled states; this worker
never samples replacement states after seeing losses or eligibility.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np

from stmem.config import digest
from stmem.inspection.refinement_candidates import RECIPE
from stmem.inspection.refinement_data import UtilityDataReader, stable_seed
from stmem.inspection.refinement_measurements import action_mask_summary, draw_plan, evaluate_state
from stmem.memory.utility_routing import (
    CONTROLLER_SCHEMA, FEATURE_SCHEMA, candidate_features, feature_spec,
    propose_packets, query_features,
)
from stmem.tracking import RunTracker
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.runtime import load_policy


SCHEMA = "expert_fm_packet_labels_v1"
READERS = ("TREE", "FLAT")


@contextmanager
def _lock(path, *, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def _read(path):
    return json.loads(Path(path).read_text())


def _seal(body, field="content_sha256"):
    body = deepcopy(body)
    body.pop(field, None)
    body[field] = digest(body)
    return body


def label_identity(study, workers):
    if workers < 1:
        raise ValueError("Label workers must be positive")
    states = study["states"]
    if len({row["state_id"] for row in states}) != len(states):
        raise ValueError("Duplicate state IDs in router study")
    return {"schema": SCHEMA, "study_id": study["experiment_id"],
            "student_sha256": study["student"]["sha256"],
            "student_step": study["student"]["step"], "source": study["source"],
            "features": feature_spec(study["configuration"]["model"]["width"]),
            "protocol": study["protocol"], "readers": list(READERS),
            "controller_schema": CONTROLLER_SCHEMA, "packet_recipe": RECIPE,
            "workers": workers, "states": states}


def publish_identity(root, identity):
    root = Path(root)
    with _lock(root / "identity.lock", blocking=True):
        path = root / "identity.json"
        if path.exists() and _read(path) != identity:
            raise ValueError("Router label identity changed; use a fresh output directory")
        if not path.exists():
            atomic_json(identity, path)


def reachable_panels(archive, state, protocol):
    """Pre-loss candidates at early and late prefixes of full gain-blind routes.

    A fixed 50/50 mixture of local and coherent construction paths covers both
    intended controllers. Prefix strata are sampled without model outputs and
    cannot consume expert actions. Every objective uses precisely these panels.
    """
    budget = int(protocol["budget"])
    count = int(protocol.get("route_prefixes_per_state", 2))
    if count < 1:
        raise ValueError("At least one reachable prefix must be declared")
    panels, contexts, exclusions = [], {}, []
    seed = stable_seed(protocol["seed"], state["task"], state["episode_id"],
                       state["decision_index"], "router-route-v1")
    route_rng = random.Random(seed)
    route_mode = "local" if route_rng.random() < .5 else "packet"
    for reader in READERS:
        start = archive.coarse(reader)["token_count"]
        target = min(budget, len(archive.leaves))
        if start >= target:
            exclusions.append({"reader": reader, "reason": "minimum_exceeds_budget" if start > target
                               else "saturated_no_positive_refinements", "initial_count": start,
                               "target_count": target})
            continue
        route = archive.route(reader, budget, seed, mode=route_mode,
                              packet_size=int(protocol["packet_max_cost"]))
        available = target - start
        chosen_counts = []
        # Partition all reachable preterminal counts, so late labels are never
        # incorrectly advertised as roots-to-budget coverage.
        for stratum in range(min(count, available)):
            divisions = min(count, available)
            lo, hi = start + available * stratum // divisions, start + available * (stratum + 1) // divisions
            rng = random.Random(stable_seed(seed, reader, "prefix", stratum))
            chosen_counts.append(rng.randrange(lo, hi))
        for position, actual_count in enumerate(chosen_counts):
            base = archive.validate(route["contexts"][actual_count])
            packets, costs = propose_packets(
                archive, base, budget, seed=seed,
                candidate_limit=int(protocol["candidate_limit"]),
                packet_max_cost=int(protocol["packet_max_cost"]), mode="packet")
            q = packets[0]["cost"]
            context_id = digest([state["state_id"], reader, base["context_id"], CONTROLLER_SCHEMA])[:24]
            panel = {"context_id": context_id, "reader": reader, "base": base["context_id"],
                     "route_mode": route_mode, "route_seed": seed, "prefix_stratum": position,
                     "initial_count": start, "background_count": actual_count,
                     "target_budget": budget, "attainable_target": target,
                     "final_count": actual_count + q, "packet_cost": q,
                     "coverage_hash": archive.coverage, "candidates": [], "controls": {},
                     "construction_costs": costs,
                     "route_prefix": [archive.names[row] for row in route["moves"][:actual_count - start]]}
            contexts[base["context_id"]] = base
            for packet in packets:
                entry = {key: packet[key] for key in ("candidate_id", "region", "region_id", "cost",
                                                       "support_count", "balancing")}
                entry["moves"] = [archive.names[row] for row in packet["moves"]]
                entry["first"] = packet["prefixes"][1]["context_id"]
                entry["packet"] = packet["final"]["context_id"]
                panel["candidates"].append(entry)
                contexts[entry["first"]] = archive.validate(packet["prefixes"][1])
                contexts[entry["packet"]] = archive.validate(packet["final"])
            for mode in ("uniform", "dispersed"):
                final, detail = archive._extend(base, actual_count + q, mode=mode, seed=seed)
                contexts[final["context_id"]] = archive.validate(final)
                panel["controls"][mode] = final["context_id"]
                panel["construction_costs"][mode + "_moves"] = len(detail["moves"])
            panels.append(panel)
    return {"panels": panels, "contexts": contexts, "exclusions": exclusions,
            "route_recipe": "half_local_half_packet_stratified_reachable_prefixes_v1"}


def measured_arrays(rows, panel, draws):
    """Save raw losses and derive signed targets independently in each draw bank."""
    lookup = {(row["context_id"], row["bank"], row["draw_index"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Repeated raw context/draw measurements")
    def losses(context, field):
        result = np.asarray([lookup[context, draw["bank"], draw["draw_index"]][field]
                             for draw in draws], dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite expert/teacher losses")
        return result
    result = {}
    for suffix, field in (("loss", "expert_full"), ("prefix_loss", "expert_prefix"),
                          ("teacher_loss", "teacher_full")):
        result["background_" + suffix] = losses(panel["base"], field)
        for operation in ("first", "packet"):
            result[operation + "_" + suffix] = np.stack(
                [losses(candidate[operation], field) for candidate in panel["candidates"]])
        for control, context in panel["controls"].items():
            result[control + "_" + suffix] = losses(context, field)
    for bank in ("discovery", "confirmation"):
        mask = np.asarray([draw["bank"] == bank for draw in draws], dtype=bool)
        if not mask.any():
            raise ValueError("Both independent noise banks are required")
        for operation in ("first", "packet"):
            result[f"{bank}_{operation}_gain"] = (
                result["background_loss"][None, mask] - result[operation + "_loss"][:, mask]).mean(1)
    result["draw_seeds"] = np.asarray([draw["seed"] for draw in draws], dtype=np.int64)
    result["draw_bank"] = np.asarray([draw["bank"] for draw in draws])
    return result


def write_state(root, state, identity, contexts, arrays, *, exclusions, timings, draws):
    """Metadata is the atomic commit marker; a half-written NPZ is never reused."""
    root = Path(root) / "states"
    root.mkdir(parents=True, exist_ok=True)
    path = root / (state["state_id"] + ".npz")
    temp = path.with_suffix(".partial.npz")
    with temp.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    metadata = _seal({"schema": SCHEMA, "study_id": identity["study_id"],
                      "student_sha256": identity["student_sha256"], "state": state,
                      "identity_sha256": digest(identity), "contexts": contexts,
                      "exclusions": exclusions, "timings": timings, "draws": draws,
                      "npz_sha256": file_sha256(path)})
    atomic_json(metadata, path.with_suffix(".json"))
    return {"npz_sha256": metadata["npz_sha256"],
            "json_sha256": file_sha256(path.with_suffix(".json")), "contexts": len(contexts)}


def read_state(root, state, identity):
    path = Path(root) / "states" / (state["state_id"] + ".npz")
    metadata = _read(path.with_suffix(".json"))
    if (metadata != _seal(metadata) or metadata.get("schema") != SCHEMA
            or metadata.get("study_id") != identity["study_id"]
            or metadata.get("identity_sha256") != digest(identity)
            or metadata.get("student_sha256") != identity["student_sha256"]
            or metadata.get("state") != state):
        raise ValueError("Label state provenance changed")
    if file_sha256(path) != metadata["npz_sha256"]:
        raise ValueError("Label state payload checksum differs")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    return metadata, arrays


def _validate_arrays(meta, arrays, identity):
    width = identity["features"]["width"]
    spec = feature_spec(width)
    ids = set()
    for context in meta["contexts"]:
        if context["context_id"] in ids:
            raise ValueError("Repeated context identity within state")
        ids.add(context["context_id"])
        prefix = context["array_prefix"]
        values = {key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)}
        count = len(context["candidates"])
        if (context["reader"] not in READERS or count < 1
                or values.get("query", np.empty(0)).shape != (spec["query_dim"],)
                or values.get("candidates", np.empty(0)).shape != (count, spec["candidate_dim"])):
            raise ValueError("Invalid observable router feature shapes")
        ndraws = len(meta["draws"])
        if values["background_loss"].shape != (ndraws,):
            raise ValueError("Raw baseline losses do not cover every draw")
        for operation in ("first", "packet"):
            if values[operation + "_loss"].shape != (count, ndraws):
                raise ValueError("Raw candidate losses do not cover every draw")
            for bank in ("discovery", "confirmation"):
                mask = np.asarray([draw["bank"] == bank for draw in meta["draws"]])
                expected = (values["background_loss"][None, mask] - values[operation + "_loss"][:, mask]).mean(1)
                if not np.array_equal(expected, values[f"{bank}_{operation}_gain"]):
                    raise ValueError("Signed expert utility is inconsistent with raw losses")
        for key, value in values.items():
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError(f"Nonfinite label array: {key}")


def labels_identity(root, *, progress=None):
    """Require complete exact worker/state coverage before consuming any labels."""
    root = Path(root)
    identity = _read(root / "identity.json")
    if identity.get("schema") != SCHEMA or identity.get("features", {}).get("schema") != FEATURE_SCHEMA:
        raise ValueError("Expected new expert-FM packet labels, not teacher-agreement labels")
    if identity["features"] != feature_spec(identity["features"]["width"]):
        raise ValueError("Router feature contract changed")
    workers = identity["workers"]
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("Label worker inventory is invalid")
    inventory = {}
    for worker in range(workers):
        receipt = _read(root / f"worker_{worker}.json")
        expected = identity["states"][worker::workers]
        planned = [state["state_id"] for state in expected]
        if (receipt.get("status") != "complete" or receipt.get("identity_sha256") != digest(identity)
                or receipt.get("worker_index") != worker or receipt.get("planned_state_ids") != planned
                or receipt.get("completed_state_ids") != planned
                or set(receipt.get("states", {})) != set(planned)):
            raise ValueError("Incomplete or inconsistent expert-label worker receipt")
        for state in expected:
            name = state["state_id"]
            if name in inventory:
                raise ValueError("A label state was assigned to multiple workers")
            metadata, arrays = read_state(root, state, identity)
            _validate_arrays(metadata, arrays, identity)
            path = root / "states" / (name + ".npz")
            item = {"npz_sha256": metadata["npz_sha256"],
                    "json_sha256": file_sha256(path.with_suffix(".json")), "contexts": len(metadata["contexts"])}
            if receipt["states"][name] != item:
                raise ValueError("Committed worker inventory differs from state files")
            inventory[name] = item
            if progress is not None:
                progress(len(inventory), len(identity["states"]))
    if set(inventory) != {state["state_id"] for state in identity["states"]}:
        raise ValueError("Label inventory does not cover the planned states exactly")
    return {"identity": identity, "identity_sha256": digest(identity), "states": inventory,
            "dataset_sha256": digest([identity, inventory])}


def load_label_rows(root, reader, partition=None):
    """Validated per-context arrays for fitting and independent development checks."""
    reader = reader.upper()
    if reader not in READERS or partition not in (None, "fit", "development"):
        raise ValueError("Unknown reader or data partition")
    bound = labels_identity(root)
    identity = bound["identity"]
    result = []
    for state in identity["states"]:
        if partition is not None and state["partition"] != partition:
            continue
        metadata, arrays = read_state(root, state, identity)
        for context in metadata["contexts"]:
            if context["reader"] != reader:
                continue
            prefix = context["array_prefix"]
            values = {key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)}
            path = Path(root) / "states" / (state["state_id"] + ".npz")
            result.append({**{key: value for key, value in context.items() if key != "candidates"},
                           **values, "candidate_records": context["candidates"],
                           "task": state["task"], "episode_id": state["episode_id"],
                           "state_id": state["state_id"], "decision_index": state["decision_index"],
                           "partition": state["partition"], "study_id": identity["study_id"],
                           "student_sha256": identity["student_sha256"], "data_path": str(path),
                           "data_sha256": metadata["npz_sha256"],
                           "data_json_sha256": bound["states"][state["state_id"]]["json_sha256"],
                           "dataset_sha256": bound["dataset_sha256"],
                           "label_identity_sha256": bound["identity_sha256"]})
    return result


def generate(args):
    from stmem.training.utility_router_study import verify_study
    from stmem.training.label_experiment import validate_models

    study = verify_study(args.study)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0 or args.batch_size < 1:
        raise ValueError("Label time and batch limits must be finite and positive")
    if not 0 <= args.worker_index < args.workers:
        raise ValueError("Worker index must lie in the declared worker range")
    if args.workers != study["protocol"]["workers"]:
        raise ValueError("Label worker count differs from the sealed study")
    root = Path(args.output)
    identity = label_identity(study, args.workers)
    publish_identity(root, identity)
    cfg, protocol = study["configuration"], study["protocol"]
    selected = study["states"][args.worker_index::args.workers]
    receipt = {"schema": SCHEMA, "identity_sha256": digest(identity),
               "worker_index": args.worker_index, "planned_state_ids": [s["state_id"] for s in selected],
               "completed_state_ids": [], "states": {}, "status": "running"}
    stopped = False
    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
    signals = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    tracker_cfg = deepcopy(cfg)
    tracker_cfg.setdefault("tracking", {}).update(project="chronotree-routers", system_stats=True)
    tracker_cfg.setdefault("paths", {})["run_root"] = str(Path(args.study).parent)
    started = time.monotonic()
    data_reader = None
    owns_worker = False
    previous_project = os.environ.get("WANDB_PROJECT")
    os.environ["WANDB_PROJECT"] = "chronotree-routers"
    receipt_path = root / f"worker_{args.worker_index}.json"
    try:
        with _lock(root / f"worker_{args.worker_index}.lock"):
            owns_worker = True
            with RunTracker(tracker_cfg, root / "tracking" / f"worker_{args.worker_index}",
                            job_type="utility_expert_labels",
                            name=f"utility-labels-{study['student']['name']}-{args.worker_index}",
                            metadata={"study_id": study["experiment_id"], "schema": SCHEMA,
                                      "student_sha256": study["student"]["sha256"]},
                            resume=receipt_path.exists()) as tracker:
                teacher, teacher_state = load_policy(study["dependencies"]["teacher"], args.device)
                student, student_state = load_policy(study["student"]["checkpoint"], args.device)
                validate_models(cfg, teacher, teacher_state, student, student_state)
                for model in (teacher, student):
                    model.freeze_encoders()
                    model.eval().requires_grad_(False)
                data = TrainingData(cfg)
                if data.identity != study["data_identity"]:
                    raise ValueError("Training data identity differs from the sealed router study")
                data_reader = UtilityDataReader(data, cfg, study["dependencies"]["encoded"],
                                               study["dependencies"]["hierarchy"])
                for state in selected:
                    if stopped or time.monotonic() - started >= args.max_hours * 3600:
                        break
                    path = root / "states" / (state["state_id"] + ".json")
                    if path.exists():
                        metadata, arrays = read_state(root, state, identity)
                        _validate_arrays(metadata, arrays, identity)
                        item_receipt = {"npz_sha256": metadata["npz_sha256"],
                                        "json_sha256": file_sha256(path), "contexts": len(metadata["contexts"])}
                    else:
                        state_started = time.monotonic()
                        _, group, archive = data_reader.archive(state)
                        allocation = reachable_panels(archive, state, protocol)
                        item, contexts = data_reader.prepare(state, allocation["contexts"])
                        if not action_mask_summary(item)["full_horizon"]:
                            raise ValueError("Preselected fitting state has an incomplete expert horizon")
                        draws = draw_plan(protocol["seed"], state["task"], state["episode_id"],
                                          state["decision_index"], discovery=protocol["discovery_draws"],
                                          confirmation=protocol["confirmation_draws"])
                        arrays, metadata_rows = {}, []
                        result = None
                        if contexts:
                            result = evaluate_state(
                                teacher, student, item, contexts, draws,
                                teacher_hash=study["dependencies"]["teacher_sha256"],
                                student_hash=study["student"]["sha256"],
                                precision=protocol.get("precision", "fp32"),
                                compact_batch_size=args.batch_size,
                                prefix_targets=int(protocol.get("execute_targets", 8)))
                        for i, panel in enumerate(allocation["panels"]):
                            columns = data_reader.packed._columns(
                                group["nodes"], np.asarray([c["region"] for c in panel["candidates"]]),
                                include_max=True, index=archive.index)
                            values = measured_arrays(result["rows"], panel, draws)
                            width = cfg["model"]["width"]
                            values["query"] = query_features(
                                item["current"], item["text"], contexts[panel["base"]], item["clocks"],
                                width=width, budget=protocol["budget"], packet_cost=panel["packet_cost"]).numpy()
                            values["candidates"] = candidate_features(
                                columns, width=width, budget=protocol["budget"], cost=panel["packet_cost"]).numpy()
                            prefix = f"c{i}_"
                            arrays.update({prefix + key: value for key, value in values.items()})
                            metadata_rows.append({**panel, "array_prefix": prefix})
                        timings = dict(result["timings"] if result is not None else {})
                        timings["complete_state_wall_seconds"] = time.monotonic() - state_started
                        item_receipt = write_state(root, state, identity, metadata_rows, arrays,
                                                   exclusions=allocation["exclusions"], timings=timings, draws=draws)
                    receipt["states"][state["state_id"]] = item_receipt
                    receipt["completed_state_ids"].append(state["state_id"])
                    receipt.update(elapsed_seconds=time.monotonic() - started,
                                   completed=len(receipt["completed_state_ids"]), planned=len(selected))
                    atomic_json(receipt, receipt_path)
                    tracker.log({"labels/states_completed": receipt["completed"],
                                 "labels/states_planned": receipt["planned"],
                                 "labels/contexts_written": sum(v["contexts"] for v in receipt["states"].values()),
                                 "labels/elapsed_seconds": receipt["elapsed_seconds"]}, step=receipt["completed"])
                    print(json.dumps({"event": "expert_labels_progress", "worker": args.worker_index,
                                      "completed": receipt["completed"], "planned": len(selected),
                                      "state_id": state["state_id"]}), flush=True)
                receipt["status"] = "complete" if len(receipt["completed_state_ids"]) == len(selected) else "partial"
                atomic_json(receipt, receipt_path)
                tracker.summary({"status": receipt["status"], "states": len(receipt["completed_state_ids"])})
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        if owns_worker:
            atomic_json(receipt, receipt_path)
        raise
    finally:
        if data_reader is not None:
            data_reader.close()
        for sig, handler in signals.items():
            signal.signal(sig, handler)
        if previous_project is None:
            os.environ.pop("WANDB_PROJECT", None)
        else:
            os.environ["WANDB_PROJECT"] = previous_project
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", "--manifest", dest="study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-index", "--shard-index", dest="worker_index", type=int, default=0)
    parser.add_argument("--workers", "--shards", dest="workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-hours", type=float, default=3.8)
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours <= 0 or args.batch_size < 1:
        parser.error("Time and batch limits must be positive")
    receipt = generate(args)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "complete" else 75


if __name__ == "__main__":
    raise SystemExit(main())
