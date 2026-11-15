"""Matched expert-FM utility scorers and independent development selection checks.

This schema is deliberately separate from the older teacher-agreement routers.
Both targets see identical observable features and the same candidate packets.
The immediate target measures the first move, but its selection is evaluated by
the completed packet, exactly like the completed-packet target. Nothing in this
module changes the frozen action policy or fits on diagnostic episodes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from stmem.config import digest
from stmem.tracking import RunTracker
from stmem.training.checkpoints import (
    atomic_copy, atomic_json, atomic_save, capture_rng, file_sha256,
    mark_complete, restore_rng,
)
from stmem.training.runtime import environment_manifest, load_checkpoint, seed_all


CHECKPOINT_SCHEMA = "utility_router_checkpoint_v1"
TARGETS = ("immediate", "packet")


def normalize_reader(value):
    value = str(value).upper()
    if value not in ("TREE", "FLAT"):
        raise ValueError("Utility reader must be TREE or FLAT")
    return value


def _target_name(target):
    if target not in TARGETS:
        raise ValueError("Utility target must be immediate or packet")
    return "first" if target == "immediate" else "packet"


def _finite(array, shape, name):
    array = np.asarray(array, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid/nonfinite {name}; expected shape {shape}")
    return array


def build_table(contexts, study, reader):
    """Flatten complete contexts without losing their statistical/split identity."""
    from stmem.memory.utility_routing import feature_spec

    reader = normalize_reader(reader)
    width = int(study["configuration"]["model"]["width"])
    spec = feature_spec(width)
    qdim, kdim = spec["query_dim"], spec["candidate_dim"]
    expected = {row["state_id"]: row for row in study["states"]}
    fields = ("query", "candidate", "discovery_first_gain", "discovery_packet_gain",
              "confirmation_first_gain", "confirmation_packet_gain", "confirmation_packet_loss")
    arrays = {name: [] for name in fields}
    groups, seen, partitions, paths = [], set(), {}, {}
    inventories = set()
    offset = 0
    for row in sorted(contexts, key=lambda item: str(item["context_id"])):
        context_id = str(row["context_id"])
        if context_id in seen:
            raise ValueError("Duplicate utility context ID")
        seen.add(context_id)
        state_id = row["state_id"]
        if state_id not in expected:
            raise ValueError("Utility label state is absent from the fixed study")
        state = expected[state_id]
        partition = row["partition"]
        if partition not in ("fit", "development") or partition != state["partition"]:
            raise ValueError("Utility label contradicts the study episode partition")
        if row["task"] != state["task"] or row["episode_id"] != state["episode_id"]:
            raise ValueError("Utility label contradicts the study episode identity")
        episode = (row["task"], row["episode_id"])
        if episode in partitions and partitions[episode] != partition:
            raise ValueError("Episode leakage between utility fitting and development")
        partitions[episode] = partition
        if (row.get("experiment_id", row.get("study_id")) != study["experiment_id"]
                or row["student_sha256"] != study["student"]["sha256"]
                or normalize_reader(row["reader"]) != reader):
            raise ValueError("Utility labels changed experiment/student/reader identity")
        candidates = np.asarray(row["candidates"], dtype=np.float32)
        if candidates.ndim != 2 or candidates.shape[1] != kdim:
            raise ValueError("Utility candidate feature dimensions changed")
        count = len(candidates)
        if not 1 <= count <= int(study["protocol"]["candidate_limit"]):
            raise ValueError("Invalid number of utility packet candidates")
        query = _finite(row["query"], (qdim,), "query")
        arrays["query"].append(np.broadcast_to(query, (count, qdim)).copy())
        arrays["candidate"].append(_finite(candidates, (count, kdim), "candidates"))
        for key in fields[2:6]:
            arrays[key].append(_finite(row[key], (count,), key))
        if "confirmation_packet_loss" in row:
            losses = row["confirmation_packet_loss"]
        elif "packet_loss" in row and "draw_bank" in row:
            confirmation = np.asarray(row["draw_bank"]) == "confirmation"
            if not confirmation.any():
                raise ValueError("No independent confirmation draws in router labels")
            losses = np.asarray(row["packet_loss"])[:, confirmation].mean(axis=1)
        else:
            losses = float(row["confirmation_background_loss"]) - np.asarray(row["confirmation_packet_gain"])
        arrays["confirmation_packet_loss"].append(_finite(losses, (count,), "confirmation packet loss"))
        cost = int(row["packet_cost"])
        if not 1 <= cost <= int(study["protocol"]["packet_max_cost"]):
            raise ValueError("Packet cost is outside the fixed positive-cost protocol")
        candidate_ids = row.get("candidate_ids", [entry["candidate_id"] for entry in row["candidate_records"]]
                                if "candidate_records" in row else list(range(count)))
        if len(candidate_ids) != count or len(set(map(str, candidate_ids))) != count:
            raise ValueError("Duplicate utility candidate identity")
        group = {"context_id": context_id, "state_id": state_id,
                       "task": row["task"], "episode_id": row["episode_id"],
                       "partition": partition, "start": offset, "count": count,
                       "packet_cost": cost, "candidate_ids": candidate_ids}
        for control in ("uniform", "dispersed"):
            if control + "_loss" in row and "draw_bank" in row:
                confirmation = np.asarray(row["draw_bank"]) == "confirmation"
                value = float(np.asarray(row[control + "_loss"])[confirmation].mean())
                if not math.isfinite(value):
                    raise ValueError("Nonfinite independent control loss")
                group["confirmation_" + control + "_loss"] = value
        groups.append(group)
        offset += count
        path = str(Path(row["data_path"]).resolve())
        sha = row["data_sha256"]
        if path in paths and paths[path] != sha:
            raise ValueError("Utility label file changed while loading")
        paths[path] = sha
        if "data_json_sha256" in row:
            paths[str(Path(path).with_suffix(".json"))] = row["data_json_sha256"]
        if "dataset_sha256" in row:
            inventories.add((row["dataset_sha256"], row["label_identity_sha256"]))
    if not groups:
        raise ValueError("No feasible utility label contexts")
    merged = {name: np.concatenate(values, axis=0) for name, values in arrays.items()}
    merged["groups"] = groups
    merged["files"] = [{"path": path, "sha256": sha} for path, sha in sorted(paths.items())]
    merged["feature_spec"] = spec
    if len(inventories) > 1:
        raise ValueError("Utility rows disagree on the complete label inventory")
    merged["label_inventory"] = list(next(iter(inventories))) if inventories else None
    for partition in ("fit", "development"):
        merged[partition + "_ids"] = np.concatenate([
            np.arange(g["start"], g["start"] + g["count"], dtype=np.int64)
            for g in groups if g["partition"] == partition
        ]) if any(g["partition"] == partition for g in groups) else np.empty(0, dtype=np.int64)
        if not len(merged[partition + "_ids"]):
            raise ValueError(f"Utility labels have no feasible {partition} contexts")
    return merged


def load_table(study_path, reader):
    from stmem.training.utility_router_study import verify_study
    from stmem.training.utility_router_data import load_label_rows

    study_path = Path(study_path).resolve()
    study = verify_study(study_path)
    rows = load_label_rows(study_path.parent / "labels", normalize_reader(reader))
    return study, build_table(rows, study, reader)


def fit_options(study):
    options = {"steps": 5000, "batch": 256, "lr": 3e-4,
               "weight_decay": .01, "grad_clip": 1., "checkpoint_every": 1000,
               "log_every": 10, "eval_every": 250, "evaluation_batch": 4096}
    options.update(study["protocol"].get("fit", {}))
    for key in ("steps", "batch", "checkpoint_every", "log_every", "eval_every", "evaluation_batch"):
        if isinstance(options[key], bool) or not isinstance(options[key], int) or options[key] < 1:
            raise ValueError(f"Router fit {key} must be a positive integer")
    for key in ("lr", "grad_clip"):
        if not math.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"Router fit {key} must be finite and positive")
    if not math.isfinite(options["weight_decay"]) or options["weight_decay"] < 0:
        raise ValueError("Router weight decay must be finite and nonnegative")
    return options


def _device(requested):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Router device must be auto, cpu or cuda")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for utility router but unavailable")
    return requested


def _identity(study, table, reader, target):
    return {"schema": CHECKPOINT_SCHEMA, "experiment_id": study["experiment_id"],
            "student_sha256": study["student"]["sha256"],
            "teacher_sha256": study["dependencies"]["teacher_sha256"],
            "reader": normalize_reader(reader), "target": target,
            "source": study["source"], "protocol": study["protocol"],
            "feature_spec": table["feature_spec"], "label_files": table["files"],
            "label_inventory": table["label_inventory"],
            "contexts_sha256": digest(table["groups"])}


def _config(study, reader, output):
    cfg = deepcopy(study["configuration"])
    cfg["reader"] = normalize_reader(reader).lower()
    cfg["seed"] = int(study["protocol"]["seed"])
    cfg.setdefault("model", {}).update(router_architecture="utility_mlp",
                                        router_width=int(study["protocol"].get("key_width", 128)))
    cfg.setdefault("memory", {})["budget"] = int(study["protocol"]["budget"])
    cfg.setdefault("paths", {})["run_root"] = str(Path(output).resolve().parent)
    cfg.setdefault("tracking", {}).update(project="chronotree-routers", system_stats=True)
    return cfg


@contextmanager
def _router_project():
    """An inherited pretraining project must not mix these explicitly separate fits."""
    previous = os.environ.get("WANDB_PROJECT")
    os.environ["WANDB_PROJECT"] = "chronotree-routers"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("WANDB_PROJECT", None)
        else:
            os.environ["WANDB_PROJECT"] = previous


@contextmanager
def _stop_signals():
    requested = []
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    try:
        for sig in old:
            signal.signal(sig, lambda signum, _frame: requested.append(signum))
        yield requested
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def _macro(rows, key):
    """Average contexts within state, states within episode, then tasks equally."""
    states = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            states[row["task"], row["episode_id"], row["state_id"]].append(float(row[key]))
    episodes = defaultdict(list)
    for (task, episode, _), values in states.items():
        episodes[task, episode].append(float(np.mean(values)))
    tasks = defaultdict(list)
    for (task, _), values in episodes.items():
        tasks[task].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in tasks.values()])) if tasks else None


def context_choice_rows(predictions, table, target, partition, *, epsilon=1e-6):
    """Select on predicted scores; report fresh-noise completed-packet outcomes."""
    target_name = _target_name(target)
    predictions = np.asarray(predictions)
    if predictions.shape != (len(table["query"]),) or not np.isfinite(predictions).all():
        raise ValueError("Invalid utility router predictions")
    rows = []
    for group in table["groups"]:
        if group["partition"] != partition:
            continue
        sl = slice(group["start"], group["start"] + group["count"])
        scores = predictions[sl]
        chosen = np.flatnonzero(scores == scores.max())
        first = table["discovery_first_gain"][sl]
        packet = table["discovery_packet_gain"][sl]
        # A numerical tie receives average credit; no arbitrary row-order oracle.
        immediate_oracle = np.flatnonzero(first >= first.max() - epsilon)
        packet_oracle = np.flatnonzero(packet >= packet.max() - epsilon)
        confirmed = table["confirmation_packet_gain"][sl]
        target_confirmed = table[f"confirmation_{target_name}_gain"][sl]
        selected_gain = float(confirmed[chosen].mean())
        random_gain = float(confirmed.mean())
        left, right = np.triu_indices(group["count"], 1)
        differences = target_confirmed[left] - target_confirmed[right]
        distinct = np.abs(differences) > epsilon
        predicted_diff = scores[left[distinct]] - scores[right[distinct]]
        correct = (np.sign(predicted_diff) == np.sign(differences[distinct])).astype(np.float64)
        correct[predicted_diff == 0] = .5
        result = {**group,
                     "selected_indices": chosen.tolist(), "scores": scores.astype(float).tolist(),
                     "selected_confirmation_packet_gain": selected_gain,
                     "random_confirmation_packet_gain": random_gain,
                     "selected_minus_random_packet_gain": selected_gain - random_gain,
                     "selected_confirmation_packet_loss": float(table["confirmation_packet_loss"][sl][chosen].mean()),
                     "random_confirmation_packet_loss": float(table["confirmation_packet_loss"][sl].mean()),
                     "discovery_immediate_selected_confirmation_packet_gain": float(confirmed[immediate_oracle].mean()),
                     "discovery_packet_selected_confirmation_packet_gain": float(confirmed[packet_oracle].mean()),
                     "discovery_packet_choice_minus_router_gain": float(confirmed[packet_oracle].mean()) - selected_gain,
                     "confirmation_rank_agreement": float(correct.mean()) if len(correct) else None,
                     "informative_rank_pairs": len(correct)}
        for control in ("uniform", "dispersed"):
            key = "confirmation_" + control + "_loss"
            if key in group:
                result["selected_improvement_vs_" + control] = group[key] - result["selected_confirmation_packet_loss"]
        rows.append(result)
    return rows


def choice_summary(rows):
    keys = ("selected_confirmation_packet_gain", "random_confirmation_packet_gain",
            "selected_minus_random_packet_gain", "selected_confirmation_packet_loss",
            "random_confirmation_packet_loss", "discovery_immediate_selected_confirmation_packet_gain",
            "discovery_packet_selected_confirmation_packet_gain", "discovery_packet_choice_minus_router_gain",
            "confirmation_rank_agreement", "selected_improvement_vs_uniform", "selected_improvement_vs_dispersed")
    result = {key: _macro(rows, key) for key in keys}
    result.update(contexts=len(rows), states=len({r["state_id"] for r in rows}),
                  episodes=len({(r["task"], r["episode_id"]) for r in rows}),
                  tasks=len({r["task"] for r in rows}),
                  informative_rank_pairs=sum(row["informative_rank_pairs"] for row in rows))
    return result


def paired_episode_interval(rows, key, *, seed=0, draws=2000):
    """Paired episode bootstrap within tasks; no draw/candidate pseudo-replication."""
    states = defaultdict(list)
    for row in rows:
        if row.get(key) is not None:
            states[row["task"], row["episode_id"], row["state_id"]].append(float(row[key]))
    episodes = defaultdict(list)
    for (task, episode, _), values in states.items():
        episodes[task, episode].append(float(np.mean(values)))
    by_task = defaultdict(list)
    for (task, _), values in episodes.items():
        by_task[task].append(float(np.mean(values)))
    if not by_task:
        return None
    rng = np.random.default_rng(seed)
    means = []
    for values in by_task.values():
        values = np.asarray(values)
        means.append(values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1))
    samples = np.stack(means).mean(axis=0)
    low, high = np.quantile(samples, [.025, .975])
    return {"mean": _macro(rows, key), "low": float(low), "high": float(high),
            "draws": draws, "unit": "episode_within_task", "training_seed_replication": False}


@torch.inference_mode()
def predict(router, query, candidate, batch_size):
    router.eval()
    return torch.cat([router(q, c) for q, c in zip(query.split(batch_size), candidate.split(batch_size))])


@torch.inference_mode()
def fit_metrics(router, tensors, table, target, scale, batch_size):
    predictions = predict(router, tensors["query"], tensors["candidate"], batch_size)
    target_name = _target_name(target)
    raw_predictions = predictions.cpu().numpy() * scale
    result = {}
    for split in ("fit", "development"):
        ids = torch.as_tensor(table[split + "_ids"], device=predictions.device)
        expected = tensors[f"discovery_{target_name}_gain"][ids] / scale
        result[f"{split}/gain_loss"] = float(F.smooth_l1_loss(predictions[ids], expected).cpu())
        prefix = "training" if split == "fit" else "development"
        rows = context_choice_rows(raw_predictions, table, target, split)
        for name, value in choice_summary(rows).items():
            if value is not None:
                result[f"{prefix}/{name}"] = value
    return result


def _checkpoint_state(router, optimizer, study, table, identity, cfg, scale, step, tracker, environment):
    return {"schema": CHECKPOINT_SCHEMA, "stage": "router", "step": step,
            "model": router.state_dict(), "optimizer": optimizer.state_dict(),
            "config": cfg, "reader": identity["reader"].lower(), "target": identity["target"],
            "target_scale": scale, "rng_by_rank": [capture_rng()], "identity": identity,
            "feature_schema": table["feature_spec"], "protocol": study["protocol"],
            "utility_protocol": deepcopy(study["protocol"]),
            "study": {"experiment_id": study["experiment_id"], "student": study["student"],
                      "scope": study["protocol"].get("scope", "development"), "source": study["source"]},
            "environment": environment, "tracking": dict(getattr(tracker, "identity", {})),
            "provenance": {"compact": study["student"]["sha256"],
                           "teacher": study["dependencies"]["teacher_sha256"],
                           "dependencies": {"compact": study["student"]["sha256"],
                                              "teacher": study["dependencies"]["teacher_sha256"]},
                           "objective": "expert_full_h16_fm", "credit_target": identity["target"],
                           "experiment_id": study["experiment_id"]}}


def load_utility_router(path, *, device="cpu", expected_student_sha256=None,
                        expected_teacher_sha256=None, expected_reader=None):
    from stmem.memory.utility_routing import build_scorer, feature_spec

    state = load_checkpoint(path)
    if state.get("schema") != CHECKPOINT_SCHEMA or state.get("stage") != "router":
        raise ValueError("Not an expert-FM utility router checkpoint")
    identity = state["identity"]
    for expected, key in ((expected_student_sha256, "student_sha256"),
                          (expected_teacher_sha256, "teacher_sha256")):
        if expected is not None and identity[key] != expected:
            raise ValueError(f"Utility router {key} differs from the frozen deployment policy")
    reader = normalize_reader(identity["reader"])
    if expected_reader is not None and reader != normalize_reader(expected_reader):
        raise ValueError("Utility router reader differs from deployment")
    _target_name(identity["target"])
    width = int(state["config"]["model"]["width"])
    spec = feature_spec(width)
    if spec != state["feature_schema"] or spec != identity["feature_spec"]:
        raise ValueError("Utility router observable feature schema changed")
    if state["protocol"] != identity["protocol"] or state.get("utility_protocol") != state["protocol"]:
        raise ValueError("Utility router allocation protocol changed")
    if (state["provenance"].get("compact") != identity["student_sha256"]
            or state["provenance"].get("teacher") != identity["teacher_sha256"]
            or normalize_reader(state["reader"]) != reader
            or state["target"] != identity["target"]):
        raise ValueError("Utility router checkpoint has inconsistent provenance")
    if not math.isfinite(state["target_scale"]) or state["target_scale"] <= 0:
        raise ValueError("Invalid utility target scale")
    router = build_scorer(width, key_width=int(state["protocol"].get("key_width", 128))).to(device)
    router.load_state_dict(state["model"], strict=True)
    router.eval().requires_grad_(False)
    return router, state


def train(study_path, output, *, reader, target, resume=False, device="auto", stop_file=None, max_hours=None):
    reader = normalize_reader(reader)
    _target_name(target)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Utility scorer fitting uses one process and one device")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".fit.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train_locked(study_path, output, reader=reader, target=target, resume=resume,
                             device=device, stop_file=stop_file, max_hours=max_hours)


def _train_locked(study_path, output, *, reader, target, resume, device, stop_file, max_hours):
    from stmem.memory.utility_routing import build_scorer

    study, table = load_table(study_path, reader)
    options = fit_options(study)
    identity = _identity(study, table, reader, target)
    cfg = _config(study, reader, output)
    if max_hours is not None and (not math.isfinite(max_hours) or max_hours <= 0):
        raise ValueError("max_hours must be finite and positive")
    device = _device(device)
    seed_all(int(study["protocol"]["seed"]))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    router = build_scorer(cfg["model"]["width"], key_width=cfg["model"]["router_width"]).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    target_name = _target_name(target)
    train_ids = table["fit_ids"]
    raw_gain = table[f"discovery_{target_name}_gain"]
    scale = max(float(torch.from_numpy(raw_gain[train_ids]).abs().median()), 1e-6)
    start = 0
    last = output / "last.pt"
    if (output / "identity.json").exists() and json.loads((output / "identity.json").read_text()) != identity:
        raise ValueError("Utility fit directory belongs to a different label/policy/target identity")
    if resume and last.is_file():
        old = load_checkpoint(last)
        if old.get("schema") != CHECKPOINT_SCHEMA or old.get("identity") != identity:
            raise ValueError("Utility router resume changed labels, policy, source, target or protocol")
        if old["target_scale"] != scale:
            raise ValueError("Training-only utility scale changed on resume")
        router.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        start = int(old["step"])
        restore_rng(old["rng_by_rank"][0])
    elif not resume and (last.exists() or (output / "identity.json").exists()):
        raise FileExistsError("Utility fit already exists; use --resume with unchanged study/options")
    if start > options["steps"]:
        raise ValueError("Checkpoint exceeds the predeclared router fit budget")
    atomic_json(identity, output / "identity.json")
    atomic_json(cfg, output / "config.json")
    if start == options["steps"]:
        mark_complete(output / "complete.json", completed_steps=start, intended_steps=start, checkpoint=last,
                      experiment_id=study["experiment_id"], target=target, reader=reader)
        return 0
    tensors = {key: torch.as_tensor(table[key], device=device, dtype=torch.float32)
               for key in ("query", "candidate", "discovery_first_gain", "discovery_packet_gain")}
    target_gain = tensors[f"discovery_{target_name}_gain"]
    (output / "complete.json").unlink(missing_ok=True)
    environment = environment_manifest()
    metadata = {"identity": identity, "student": study["student"], "planned_steps": options["steps"],
                "resume_step": start, "target_scale_training_only": scale,
                "fit_rows": len(train_ids), "development_rows": len(table["development_ids"]),
                "router_parameters": sum(p.numel() for p in router.parameters()), "device": device,
                "diagnostic": "95k development pilot" if study["student"]["step"] != study["student"].get("primary_endpoint")
                else "fixed endpoint router development; no reserved confirmation selection"}
    started = time.perf_counter()
    with _stop_signals() as signals, _router_project(), RunTracker(
        cfg, output, job_type="utility_router", name=f"{study['student']['name']}-{reader.lower()}-{target}",
        metadata=metadata, resume=resume,
    ) as tracker:
        def publish(row, step):
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
            tracker.log(row, step=step)

        initial = fit_metrics(router, tensors, table, target, scale, options["evaluation_batch"])
        publish({"step": start, "target_scale_training_only": scale, **initial}, start)
        for step in range(start, options["steps"]):
            iteration_start = time.perf_counter()
            router.train()
            ids = torch.as_tensor(train_ids[torch.randint(len(train_ids), (options["batch"],)).numpy()], device=device)
            prediction = router(tensors["query"][ids], tensors["candidate"][ids])
            loss = F.smooth_l1_loss(prediction, target_gain[ids] / scale)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite utility router regression loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(router.parameters(), options["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
            current = step + 1
            stop = bool(signals or (stop_file and Path(stop_file).exists())
                        or (max_hours is not None and time.perf_counter() - started >= max_hours * 3600))
            evaluation_due = current % options["eval_every"] == 0 or current == options["steps"] or stop
            if current == start + 1 or current % options["log_every"] == 0 or evaluation_due:
                row = {"step": current, "training/gain_loss": float(loss.detach().cpu()),
                       "training/gradient_norm": float(grad_norm.detach().cpu()),
                       "training/step_seconds": time.perf_counter() - iteration_start,
                       "training/learning_rate": optimizer.param_groups[0]["lr"],
                       "training/examples_this_allocation": (current - start) * options["batch"]}
                if evaluation_due:
                    row.update(fit_metrics(router, tensors, table, target, scale, options["evaluation_batch"]))
                publish(row, current)
            if current % options["checkpoint_every"] == 0 or current == options["steps"] or stop:
                state = _checkpoint_state(router, optimizer, study, table, identity, cfg, scale, current, tracker, environment)
                atomic_save(state, last)
                if current % options["checkpoint_every"] == 0 or current == options["steps"]:
                    atomic_copy(last, output / f"step_{current:08d}.pt", immutable=True)
                tracker.checkpoint(last, step=current)
            if stop and current < options["steps"]:
                atomic_json({"status": "checkpointed_partial", "step": current, "planned_steps": options["steps"]},
                            output / "partial.json")
                tracker.finish(status="checkpointed_partial", exit_code=75)
                return 75
        mark_complete(output / "complete.json", completed_steps=options["steps"],
                      intended_steps=options["steps"], checkpoint=last,
                      experiment_id=study["experiment_id"], target=target, reader=reader)
        (output / "partial.json").unlink(missing_ok=True)
        tracker.summary({"completed_steps": options["steps"], "checkpoint/sha256": file_sha256(last)})
    return 0


def evaluate(study_path, output, *, reader, target, checkpoint=None, device="auto"):
    reader = normalize_reader(reader)
    _target_name(target)
    output = Path(output).resolve()
    checkpoint = Path(checkpoint or output / "last.pt").resolve()
    study, table = load_table(study_path, reader)
    identity = _identity(study, table, reader, target)
    device = _device(device)
    router, state = load_utility_router(checkpoint, device=device,
                                        expected_student_sha256=study["student"]["sha256"],
                                        expected_teacher_sha256=study["dependencies"]["teacher_sha256"],
                                        expected_reader=reader)
    if state["identity"] != identity:
        raise ValueError("Evaluation labels/protocol differ from this utility router fit")
    destination = output / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    evaluation_identity = {"router_sha256": file_sha256(checkpoint), "fit_identity": identity}
    identity_path = destination / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != evaluation_identity:
        raise ValueError("Utility evaluation directory belongs to a different router; use a new output directory")
    atomic_json(evaluation_identity, identity_path)
    cfg = _config(study, reader, output)
    with _router_project(), RunTracker(cfg, destination, job_type="evaluation",
                                      name=f"utility-{study['student']['name']}-{reader.lower()}-{target}",
                                      metadata={**evaluation_identity, "scope": "offline development, not robot success"},
                                      resume=(destination / "tracking.json").exists()) as tracker:
        query = torch.as_tensor(table["query"], device=device, dtype=torch.float32)
        candidate = torch.as_tensor(table["candidate"], device=device, dtype=torch.float32)
        scores = predict(router, query, candidate, fit_options(study)["evaluation_batch"]).cpu().numpy() * state["target_scale"]
        rows = context_choice_rows(scores, table, target, "development")
        summary = {"status": "complete", "scope": "offline_development_selection",
                   "not_closed_loop_success": True, "experiment_id": study["experiment_id"],
                   "student": study["student"], "reader": reader, "target": target,
                   "router_sha256": evaluation_identity["router_sha256"], "router_step": state["step"],
                   "metrics": choice_summary(rows),
                   "selected_minus_random_packet_gain_interval": paired_episode_interval(
                       rows, "selected_minus_random_packet_gain", seed=int(study["protocol"]["seed"]))}
        atomic_json(rows, destination / "choices.json")
        atomic_json(summary, destination / "summary.json")
        tracker.log({f"development/{key}": value for key, value in summary["metrics"].items() if value is not None}, step=state["step"])
        tracker.summary({"status": "complete", "router_step": state["step"], **summary["metrics"]})
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--reader", choices=("TREE", "FLAT", "tree", "flat"), required=True)
    parser.add_argument("--target", choices=TARGETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args(argv)
    if args.mode == "train":
        if args.checkpoint is not None:
            parser.error("Training resumes its own last.pt with --resume; --checkpoint is evaluation-only")
        return train(args.study, args.output, reader=args.reader, target=args.target, resume=args.resume,
                     device=args.device, stop_file=args.stop_file, max_hours=args.max_hours)
    if args.stop_file is not None or args.max_hours is not None:
        parser.error("Stop options apply only to training")
    evaluate(args.study, args.output, reader=args.reader, target=args.target, checkpoint=args.checkpoint, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
