"""Paired flow-velocity utility labels with episode-disjoint scorer diagnostics."""

from __future__ import annotations
import argparse
import json
import os
import random
import time
import h5py
import numpy as np
import torch
from stmem.config import load_config, dependency, run_dir, digest
from stmem.data.cache import FrozenCache
from stmem.memory.frontier import split_frontier
from stmem.memory.flat import FlatState
from stmem.models.action_flow import flow_sample, masked_mse
from stmem.models.policy import choose_records
from stmem.models.router import LocalRouter, CachedScorer
from stmem.tracking import RunTracker
from stmem.training.data import TrainingData
from stmem.training.runtime import load_policy, load_router, seed_all, validate_selection
from stmem.training.checkpoints import file_sha256, atomic_json


def paired_gains(teacher, compact, evidence, before, after, action, valid, *, noise=None, time=None):
    """One shared x_u/u for teacher, before and all candidate after-contexts."""
    x, u, _, noise = flow_sample(action, valid, teacher.normalizer, noise=noise, time=time)
    reference = teacher(None, x, u, evidence=evidence)
    initial = compact(None, x, u, evidence=evidence, records=before)
    base = masked_mse(initial, reference, valid)
    contexts = [compact.context(evidence, records) for records in after]
    if not contexts:
        return [], {"noise": noise, "time": u, "noisy_action": x}
    max_tokens = max(len(c) for c, _ in contexts)
    padded = torch.stack([torch.nn.functional.pad(c, (0, 0, 0, max_tokens - len(c))) for c, _ in contexts])
    biases = torch.stack(
        [torch.nn.functional.pad(b, (0, max_tokens - len(b)), value=-torch.inf) for _, b in contexts]
    )
    predictions = compact.decoder(
        x[None].expand(len(contexts), -1, -1), u.expand(len(contexts)), padded, biases
    )
    gains = []
    for prediction, records in zip(predictions, after):
        cost = len(records) - len(before)
        if cost <= 0:
            raise ValueError("Utility labels require a strictly positive token increment")
        gains.append((base - masked_mse(prediction, reference, valid)) / cost)
    return gains, {"noise": noise, "time": u, "noisy_action": x}


def label_path(cfg, refresh=False):
    return run_dir(cfg, "router") / ("labels_refresh.h5" if refresh else "labels_initial.h5")


@torch.no_grad()
def generate(cfg, reader, *, refresh=False, router_checkpoint=None):
    cfg["reader"] = reader
    if cfg["model"].get("decoder", "flow") != "flow":
        raise ValueError(
            "Paired velocity labels require the flow decoder; regression is a standalone debugging control"
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(4)
    seed_all(cfg["labels"]["seed"])
    data = TrainingData(cfg)
    teacher_path = dependency(cfg, "teacher")
    compact_path = dependency(cfg, "compact")
    for checkpoint in (teacher_path, compact_path):
        validate_selection(checkpoint, debug=cfg.get("debug", False) and data.synthetic)
    teacher, _ = load_policy(teacher_path, device)
    compact, _ = load_policy(compact_path, device)
    teacher.eval().requires_grad_(False)
    compact.eval().requires_grad_(False)
    cache = FrozenCache(cfg, teacher_path, data)
    prototype = LocalRouter(cfg["model"]["width"], cfg["model"]["router_width"]).to(device)
    scorer = None
    if refresh:
        if not router_checkpoint:
            raise ValueError("A label refresh requires the initial frozen router checkpoint")
        router, _ = load_router(router_checkpoint, device)
        scorer = CachedScorer(router, file_sha256(router_checkpoint))
    contexts = cfg["labels"]["refresh_contexts" if refresh else "contexts"]
    identity = {
        "teacher": file_sha256(teacher_path),
        "compact": file_sha256(compact_path),
        "cache": digest(cache.manifest),
        "data": data.identity,
        "reader": reader,
        "labels": cfg["labels"],
        "refresh": refresh,
        "router": file_sha256(router_checkpoint) if refresh else None,
        "partition": {
            f"{data.episode(i).task}/{data.episode(i).episode_id}": data.label_partition(i)
            for i in range(len(data.refs))
        },
    }
    destination = label_path(cfg, refresh)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with h5py.File(destination, "r") as f:
            if json.loads(f.attrs["identity"]) != identity or not f.attrs.get("complete"):
                raise ValueError("Existing labels have different inputs; use a new run directory")
        print(f"Labels already complete: {destination}")
        return destination
    temporary = destination.with_suffix(".partial.h5")
    phase = "refresh" if refresh else "initial"
    tracking_metadata = {
        "reader": reader,
        "phase": phase,
        "planned_contexts": contexts,
        "teacher_sha256": identity["teacher"],
        "compact_sha256": identity["compact"],
        "router_sha256": identity["router"],
        "cache_identity": identity["cache"],
        "data_identity": identity["data"],
        "labels_identity": digest(identity),
        "partition_sha256": digest(identity["partition"]),
        "label_path": str(destination),
        "diagnostic": "conditional router-label validation; policy/encoders may have seen these episodes",
    }
    with RunTracker(
        cfg,
        destination.parent / f"tracking_labels_{phase}",
        job_type=f"label_{reader}_{phase}",
        metadata=tracking_metadata,
        resume=temporary.exists(),
    ) as tracker:
        started = time.perf_counter()
        with h5py.File(temporary, "a") as f:
            if "identity" not in f.attrs:
                f.attrs["identity"] = json.dumps(identity, sort_keys=True)
                f.attrs["next_context"] = 0
                f.attrs["committed_rows"] = 0
                for name, shape, dtype in [
                    ("query", (prototype.query_dim,), "f4"),
                    ("candidate", (2 * cfg["model"]["width"] + 23,), "f4"),
                    ("gain", (), "f4"),
                    ("context_id", (), "i8"),
                    ("partition", (), "i1"),
                    ("budget", (), "i8"),
                    ("episode", (), h5py.string_dtype()),
                ]:
                    f.create_dataset(
                        name, shape=(0, *shape), maxshape=(None, *shape), chunks=True, dtype=dtype
                    )
            elif json.loads(f.attrs["identity"]) != identity:
                raise ValueError("Partial label inputs changed; cannot resume")
            committed = int(f.attrs.get("committed_rows", len(f["gain"])))
            for dataset in f.values():
                if len(dataset) < committed:
                    raise ValueError(
                        "Interrupted label shard is damaged; preserve it for diagnosis and regenerate"
                    )
                dataset.resize(committed, axis=0)
            start_context = int(f.attrs["next_context"])

            def log_progress(completed_contexts):
                if completed_contexts % max(1, contexts // 10) == 0 or completed_contexts == contexts:
                    elapsed = time.perf_counter() - started
                    row = {
                        "label_context": completed_contexts,
                        "planned": contexts,
                        "moves": len(f["gain"]),
                        "seconds": elapsed,
                        "contexts_per_second": (completed_contexts - start_context) / max(elapsed, 1e-9),
                    }
                    print(json.dumps(row), flush=True)
                    tracker.log(row, step=completed_contexts)

            for context_id in range(start_context, contexts):
                seed = cfg["labels"]["seed"] + context_id + (1000000 if refresh else 0)
                rng = random.Random(seed)
                task = data.tasks[context_id % len(data.tasks)]
                # Cycle every episode (including scorer-validation episodes) before repeating.
                epidx = data.by_task[task][(context_id // len(data.tasks)) % len(data.by_task[task])]
                ep = data.episode(epidx)
                times = [
                    i
                    for i in range(0, len(ep), data.stride)
                    if ep.decision(i, data.horizon).target_valid.any()
                ]
                # Explicit early/middle/late strata; never oracle cue boundaries.
                stratum = (context_id // len(cfg["labels"]["budgets"])) % 3
                lo, hi = len(times) * stratum // 3, len(times) * (stratum + 1) // 3
                time_index = rng.choice(times[lo:hi] or times)
                decision = ep.decision(time_index, data.horizon)
                evidence = cache.evidence(decision)
                budget = cfg["labels"]["budgets"][context_id % len(cfg["labels"]["budgets"])]
                root_count = len(evidence.archive.roots)
                if root_count > budget:
                    raise ValueError("Label budget cannot cover roots")
                query = compact.routing_query(evidence, budget)
                use_learned = scorer is not None and rng.random() >= cfg["labels"]["exploration_fraction"]
                if use_learned:
                    candidates_for_keys = (
                        evidence.archive.records if reader == "tree" else evidence.archive.leaves
                    )
                    scorer.retain(candidates_for_keys)
                    scorer.prepare(candidates_for_keys)
                    scorer.set_query(query)
                cap = rng.randint(max(root_count, 1), max(root_count, budget - 1))
                selection = choose_records(
                    evidence, reader, cap, scorer=scorer if use_learned else None, rng=rng
                )
                before = selection.records
                if len(before) >= budget:
                    f.attrs["next_context"] = context_id + 1
                    f.flush()
                    log_progress(context_id + 1)
                    continue
                if reader == "tree":
                    candidates = [r for r in before if r.children]
                    rng.shuffle(candidates)
                    candidates = candidates[: cfg["labels"]["candidates"]]
                    after = [split_frontier(before, node) for node in candidates]
                else:
                    selected = {r.node_id for r in before if r.is_leaf}
                    state = FlatState(tuple(evidence.archive.leaves), selected).normalize_singletons()
                    before = state.records
                    candidates = [r for r in state.unselected if state.move_cost(r) > 0]
                    rng.shuffle(candidates)
                    candidates = candidates[: cfg["labels"]["candidates"]]
                    after = [state.move(leaf).normalize_singletons().records for leaf in candidates]
                if candidates:
                    generator = torch.Generator(device=device).manual_seed(seed)
                    noise = torch.randn(decision.target_chunk.shape, device=device, generator=generator)
                    u = torch.rand((), device=device, generator=generator)
                    gains, _ = paired_gains(
                        teacher,
                        compact,
                        evidence,
                        before,
                        after,
                        decision.target_chunk.to(device),
                        decision.target_valid.to(device),
                        noise=noise,
                        time=u,
                    )
                    n = len(candidates)
                    start = len(f["gain"])
                    values = {
                        "query": query.cpu().numpy()[None].repeat(n, 0),
                        "candidate": prototype.candidate_input(candidates).cpu().numpy(),
                        "gain": torch.stack(gains).cpu().numpy(),
                        "context_id": [context_id] * n,
                        "partition": [int(data.label_partition(epidx) == "validation")] * n,
                        "budget": [budget] * n,
                        "episode": [f"{task}/{ep.episode_id}"] * n,
                    }
                    for key, value in values.items():
                        f[key].resize(start + n, axis=0)
                        f[key][start:] = value
                f.attrs["next_context"] = context_id + 1
                f.attrs["committed_rows"] = len(f["gain"])
                f.flush()
                log_progress(context_id + 1)
            gain = np.asarray(f["gain"])
            if not len(gain):
                raise ValueError("No valid positive-cost candidate moves; label stage is not complete")
            eps = cfg["labels"]["gain_epsilon"]
            summary = {
                "contexts": contexts,
                "labels": len(gain),
                "positive": int((gain > eps).sum()),
                "negative": int((gain < -eps).sum()),
                "near_zero": int((abs(gain) <= eps).sum()),
                "mean": float(gain.mean()),
                "std": float(gain.std()),
                "seconds": time.perf_counter() - started,
                "diagnostic": "conditional router-label validation; policy/encoders may have seen these episodes",
            }
            f.attrs["complete"] = True
            f.attrs["summary"] = json.dumps(summary)
            f.flush()
        os.replace(temporary, destination)
        atomic_json(summary, destination.with_suffix(".json"))
        tracker.log(
            {
                "label_context": contexts,
                "moves": summary["labels"],
                "gain_positive": summary["positive"],
                "gain_negative": summary["negative"],
                "gain_near_zero": summary["near_zero"],
                "gain_mean": summary["mean"],
                "gain_std": summary["std"],
                "generation_seconds": summary["seconds"],
            },
            step=contexts,
        )
        tracker.summary(summary)
        print(json.dumps(summary))
        return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--reader", choices=["tree", "flat"], required=True)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--router-checkpoint")
    args = p.parse_args()
    generate(
        load_config(args.config), args.reader, refresh=args.refresh, router_checkpoint=args.router_checkpoint
    )


if __name__ == "__main__":
    main()
