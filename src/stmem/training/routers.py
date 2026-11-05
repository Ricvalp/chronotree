"""Equal-capacity robust signed-gain regression for tree and flat scorers."""

from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from stmem.config import run_dir, digest
from stmem.evaluation.timing import process_memory
from stmem.models.router import build_router
from stmem.tracking import RunTracker
from stmem.training.gain_labels import label_path
from stmem.training.logging_options import positive_log_every, resolve_logging
from stmem.training.runtime import seed_all, load_checkpoint, environment_manifest
from stmem.training.checkpoints import (
    atomic_save,
    atomic_json,
    atomic_copy,
    capture_rng,
    restore_rng,
    mark_complete,
    file_sha256,
)


def _validation_ranking_pairs(context_arrays, validation_ids, validation_gain, epsilon):
    """Prepare bounded within-context comparisons, keeping refresh draws separate."""
    groups = defaultdict(list)
    offset = 0
    positions = {int(label_id): i for i, label_id in enumerate(validation_ids)}
    for shard, contexts in enumerate(context_arrays):
        for row, context in enumerate(contexts):
            position = positions.get(offset + row)
            if position is not None:
                groups[(shard, int(context))].append(position)
        offset += len(contexts)
    pairs = [pair for members in groups.values() for pair in combinations(members, 2)]
    if not pairs:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty
    indices = torch.tensor(pairs, dtype=torch.long)
    distinct = (validation_gain[indices[:, 0]] - validation_gain[indices[:, 1]]).abs() > epsilon
    return indices[distinct, 0], indices[distinct, 1]


def _context_groups(context_arrays, ids):
    """Padded positions within a split, never joining initial and refresh draws."""
    positions = {int(row): i for i, row in enumerate(ids)}
    groups = defaultdict(list)
    offset = 0
    for phase, contexts in enumerate(context_arrays):
        for row, context in enumerate(contexts):
            if offset + row in positions:
                groups[phase, int(context)].append(positions[offset + row])
        offset += len(contexts)
    members = [rows for rows in groups.values() if len(rows) > 1]
    indices = torch.zeros((len(members), max(map(len, members), default=0)), dtype=torch.long)
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for i, rows in enumerate(members):
        indices[i, :len(rows)] = torch.tensor(rows)
        valid[i, :len(rows)] = True
    return indices, valid


def _fit_training_ids(cfg, context_arrays, train_ids):
    """Optional fixed *training-context* subset, independent of global RNG/validation."""
    count = cfg["training"].get("router_fit_contexts", 0)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("training.router_fit_contexts must be a nonnegative integer")
    if not count:
        return train_ids
    allowed = set(map(int, train_ids))
    groups, offset = defaultdict(list), 0
    for phase, contexts in enumerate(context_arrays):
        for row, context in enumerate(contexts):
            if offset + row in allowed:
                groups[phase, int(context)].append(offset + row)
        offset += len(contexts)
    keys = sorted(groups)
    if count > len(keys):
        raise ValueError(f"router_fit_contexts={count} exceeds {len(keys)} available training contexts")
    chosen = np.random.default_rng(cfg["seed"]).choice(len(keys), count, replace=False)
    return np.asarray(sorted(row for index in chosen for row in groups[keys[int(index)]]), dtype=np.int64)


def _router_prediction(router, query, candidate, ids, inputs=None):
    if not inputs:
        return router(query[ids], candidate[ids])
    children = inputs["children"][ids] if router.requires_children else None
    if not router.requires_current:
        return router(query[ids], candidate[ids], children=children)
    # Several candidate moves share exactly the same current observation/task.
    # Compute attention once per distinct context in this minibatch.
    contexts, inverse = torch.unique(inputs["row_to_context"][ids], sorted=True, return_inverse=True)
    encoded_query = router.encode_query(
        inputs["context_query"][contexts], current=inputs["current"][contexts],
        current_valid=inputs["current_valid"][contexts],
    )
    keys, bias = router.encode_candidates(candidate[ids], children=children)
    return router.score_encoded(encoded_query[inverse], keys, bias)


def _choice_metrics(predictions, gain, groups, scale, prefix):
    indices, valid = groups
    if not len(indices):
        return {}
    scores = predictions[indices].masked_fill(~valid, -torch.inf)
    actual = gain[indices]
    best = actual.masked_fill(~valid, -torch.inf).amax(dim=1)
    # Give exact prediction ties their expected gain, not arbitrary row-order credit.
    selected = valid & (scores == scores.amax(dim=1, keepdim=True))
    chosen = (actual * selected).sum(dim=1) / selected.sum(dim=1)
    random = (actual * valid).sum(dim=1) / valid.sum(dim=1)
    values = [chosen.mean(), best.mean(), random.mean(), (best - chosen).mean(),
              ((best - chosen) / scale).mean()]
    names = [f"{prefix}_{name}" for name in (
        "selected_label_gain", "best_candidate_label_gain", "random_candidate_label_gain",
        "top_choice_label_regret", "top_choice_label_regret_scaled",
    )]
    return dict(zip(names, values))


def _objective(cfg):
    labels = cfg.get("labels", {})
    objective = labels.get("objective", "velocity")
    steps = labels.get("integration_steps", 8)
    if objective not in ("velocity", "final_action"):
        raise ValueError("Router label objective must be velocity or final_action")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Router label integration_steps must be a positive integer")
    return objective, steps


def _router_device(cfg):
    requested = cfg["training"].get("router_device", "auto")
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("training.router_device must be auto, cpu, or cuda")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Router training requested CUDA but CUDA is unavailable")
    return requested


def _load_labels(cfg):
    """Validate complete immutable phases, then merge in context order.

    Context IDs may repeat for distinct candidate moves within one context. They
    may never occur in two shards of the same phase. Initial and refresh phases
    deliberately have separate context namespaces and separate ranking groups.
    """
    objective, integration_steps = _objective(cfg)
    labels = cfg.get("labels", {})
    names = ("query", "candidate", "gain", "partition", "context_id", "episode")
    required = {
        "teacher", "compact", "cache", "data", "reader", "partition", "objective",
        "integration_steps", "sample_plan", "shard_index", "shard_count", "refresh", "router",
        "schema_version", "contexts", "source_sha256",
    }
    paths, identities, phase_arrays, context_arrays = [], [], [], []
    seen_paths = set()
    frozen = None
    for refresh in ([False, True] if labels.get("use_refresh", False) else [False]):
        setting = "refresh_shards" if refresh else "initial_shards"
        supplied = labels.get(setting)
        if supplied is not None and (not isinstance(supplied, (list, tuple)) or not supplied):
            raise ValueError(f"labels.{setting} must be a nonempty list of shard paths")
        phase_paths = [Path(p) for p in supplied] if supplied is not None else [label_path(cfg, refresh)]
        shards = []
        for path in phase_paths:
            resolved = path.resolve()
            if resolved in seen_paths:
                raise ValueError("Duplicate label shard path")
            seen_paths.add(resolved)
            with h5py.File(path, "r") as f:
                if not f.attrs.get("complete"):
                    raise ValueError(f"Incomplete label shard: {path}")
                ident = json.loads(f.attrs["identity"])
                modern = "objective" in ident or supplied is not None
                if modern and not required <= ident.keys():
                    raise ValueError(f"Label shard lacks objective/shard provenance: {sorted(required - ident.keys())}")
                if modern and ident["schema_version"] != 2:
                    raise ValueError("Unsupported router label schema_version")
                if modern and any(
                    not isinstance(ident[key], str) or not ident[key]
                    for key in ("teacher", "compact", "cache", "data", "reader", "sample_plan")
                ):
                    raise ValueError("Label shard has empty or invalid identity fields")
                if not modern and (objective != "velocity" or integration_steps != 8):
                    raise ValueError("Legacy labels are supported only for the default velocity objective")
                actual_objective = ident.get("objective", "velocity")
                actual_steps = ident.get("integration_steps", 8)
                if actual_objective != objective or actual_steps != integration_steps:
                    raise ValueError("Label objective/integration_steps mismatch")
                if ident.get("reader", cfg["reader"]) != cfg["reader"]:
                    raise ValueError("Label reader mismatch")
                if ident.get("refresh", refresh) != refresh:
                    raise ValueError("Label initial/refresh phase mismatch")
                if modern and (
                    (refresh and (not isinstance(ident["router"], str) or not ident["router"]))
                    or (not refresh and ident["router"] is not None)
                ):
                    raise ValueError("Initial labels must omit a router hash; refresh labels require one")
                signature = {
                    key: ident.get(key) for key in ("teacher", "compact", "cache", "data", "partition", "source_sha256")
                }
                signature.update(reader=cfg["reader"], objective=objective, integration_steps=integration_steps)
                if not all(ident.get(key) is not None for key in ("teacher", "compact", "data", "partition")):
                    raise ValueError("Label shard lacks frozen-policy/data provenance")
                if frozen is not None and signature != frozen:
                    raise ValueError("Label shards changed frozen teacher/compact/cache/data/partition provenance")
                frozen = signature
                index, count = ident.get("shard_index", 0), ident.get("shard_count", 1)
                if (
                    isinstance(index, bool) or not isinstance(index, int)
                    or isinstance(count, bool) or not isinstance(count, int)
                    or count < 1 or not 0 <= index < count
                ):
                    raise ValueError("Invalid label shard index/count")
                data = {name: np.asarray(f[name]) for name in names}
                rows = len(data["gain"])
                width = cfg["model"]["width"]
                shapes = {"query": (rows, 2 * width + 27), "candidate": (rows, 2 * width + 23)}
                shapes.update({name: (rows,) for name in names[2:]})
                if any(data[name].shape != shape for name, shape in shapes.items()):
                    raise ValueError("Label rows or feature dimensions disagree")
                if any(not np.isfinite(data[name]).all() for name in ("query", "candidate", "gain")):
                    raise ValueError("Nonfinite label features or gains")
                contexts = data["context_id"]
                if not np.issubdtype(contexts.dtype, np.integer) or np.any(contexts < 0):
                    raise ValueError("Label context IDs must be nonnegative integers")
                if not np.isin(data["partition"], [0, 1]).all():
                    raise ValueError("Label partition must identify train or validation")
                if modern and np.any(contexts % count != index):
                    raise ValueError("Label context belongs to a different shard")
                if modern:
                    total = ident["contexts"]
                    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
                        raise ValueError("Label planned contexts must be a positive integer")
                    if "attempted_context_ids" not in f:
                        raise ValueError("Label shard lacks attempted-context coverage")
                    attempted = np.asarray(f["attempted_context_ids"])
                    expected_ids = np.arange(index, total, count, dtype=np.int64)
                    if (
                        attempted.ndim != 1 or not np.issubdtype(attempted.dtype, np.integer)
                        or not np.array_equal(np.sort(attempted), expected_ids)
                    ):
                        raise ValueError("Label shard has incomplete or duplicate attempted-context coverage")
                    if np.any(contexts >= total):
                        raise ValueError("Label row exceeds the planned context range")
                    budget = np.asarray(f["budget"])
                    if (
                        budget.shape != (rows,) or not np.issubdtype(budget.dtype, np.integer)
                        or np.any(budget < 1)
                    ):
                        raise ValueError("Label budgets must be positive integers, one per row")
                    order = np.argsort(contexts, kind="stable")
                    same_context = contexts[order][1:] == contexts[order][:-1]
                    for name in ("query", "partition", "episode"):
                        values = data[name][order]
                        if np.any(values[1:][same_context] != values[:-1][same_context]):
                            raise ValueError("Rows from one label context disagree on query/episode/partition")
                    if np.any(budget[order][1:][same_context] != budget[order][:-1][same_context]):
                        raise ValueError("Rows from one label context disagree on budget")
                    _, context_counts = np.unique(contexts, return_counts=True)
                    if np.any(context_counts > labels.get("candidates", 4)):
                        raise ValueError("Label context has more candidate rows than configured")
                    if "move_index" in f:
                        moves = np.asarray(f["move_index"])
                        if moves.shape != (rows,) or not np.issubdtype(moves.dtype, np.integer):
                            raise ValueError("Label move indices must identify each candidate row")
                        pairs = np.stack([contexts, moves], axis=1)
                        if np.any(moves < 0) or len(np.unique(pairs, axis=0)) != rows:
                            raise ValueError("Duplicate or invalid candidate moves within a label context")
                    for episode, partition in zip(data["episode"], data["partition"]):
                        episode = episode.decode() if isinstance(episode, bytes) else str(episode)
                        expected = "validation" if partition else "train"
                        if ident["partition"].get(episode) != expected:
                            raise ValueError("Label row contradicts the episode partition manifest")
                shards.append((index, count, path, ident, data))
        shards.sort(key=lambda shard: shard[0])
        counts = {shard[1] for shard in shards}
        if len(counts) != 1 or [shard[0] for shard in shards] != list(range(shards[0][1])):
            raise ValueError("Missing or duplicate label shards in phase")
        phase_identity = None
        seen_contexts = set()
        for index, count, path, ident, data in shards:
            phase_signature = {key: ident.get(key) for key in ("sample_plan", "router", "contexts")}
            if phase_identity is not None and phase_signature != phase_identity:
                raise ValueError("Label phase mixes sample plans or refresh router checkpoints")
            phase_identity = phase_signature
            contexts = set(map(int, data["context_id"]))
            if seen_contexts & contexts:
                raise ValueError("Duplicate context IDs across label shards")
            seen_contexts.update(contexts)
            paths.append(path)
            identities.append(ident)
        phase = {name: np.concatenate([shard[4][name] for shard in shards]) for name in names}
        # This gives one reproducible row order regardless of input shard-list order.
        order = np.argsort(phase["context_id"], kind="stable")
        phase = {name: values[order] for name, values in phase.items()}
        context_arrays.append(phase["context_id"])
        phase_arrays.append(phase)
    merged = {name: np.concatenate([phase[name] for phase in phase_arrays]) for name in names}
    return merged, paths, identities, context_arrays, frozen


@torch.no_grad()
def _split_metrics(router, query, candidate, gain, ids, left, right, scale, batch_size, *,
                   training=False, inputs=None, groups=None):
    predictions = torch.cat([
        _router_prediction(router, query, candidate, chunk, inputs) for chunk in ids.split(batch_size)
    ])
    expected_gain = gain[ids]
    metrics = [
        F.smooth_l1_loss(predictions, expected_gain / scale),
        ((predictions > 0) == (expected_gain > 0)).float().mean(),
    ]
    prefix = "training" if training else "validation"
    names = ["training_full_gain_loss" if training else "conditional_label_validation_loss",
             f"{prefix}_sign_agreement"]
    if len(left):
        predicted_order = (predictions[left] - predictions[right]).sign()
        expected_order = (expected_gain[left] - expected_gain[right]).sign()
        metrics.append((predicted_order == expected_order).float().mean())
        names.append(f"{prefix}_within_context_rank_agreement")
    if groups is not None:
        choice = _choice_metrics(predictions, expected_gain, groups, scale, prefix)
        names.extend(choice)
        metrics.extend(choice.values())
    # A single small transfer; no per-validation-batch GPU synchronization.
    return dict(zip(names, torch.stack(metrics).cpu().tolist()))


def _validation_metrics(router, query, candidate, gain, ids, left, right, scale, batch_size, **kwargs):
    return _split_metrics(router, query, candidate, gain, ids, left, right, scale, batch_size, **kwargs)


@torch.no_grad()
def _constant_baselines(gain, train_ids, val_ids, scale):
    """Fixed reference losses; the signed-median predictor uses training labels only."""
    median = gain[train_ids].median() / scale
    values, names = [], []
    for split, ids in (("training", train_ids), ("validation", val_ids)):
        targets = gain[ids] / scale
        values.extend((F.smooth_l1_loss(torch.zeros_like(targets), targets),
                       F.smooth_l1_loss(median.expand_as(targets), targets)))
        names.extend((f"baseline/{split}_zero_gain_loss", f"baseline/{split}_median_gain_loss"))
    return dict(zip(names, torch.stack(values).cpu().tolist()))


def train_router(cfg, *, resume=None, stop_file=None, log_every=None):
    logging_options = resolve_logging(cfg, log_every=log_every, attention_every=0)
    validation_every = positive_log_every(cfg["training"]["log_every"])
    diagnostics = cfg["training"].get("router_diagnostics", False)
    if not isinstance(diagnostics, bool):
        raise ValueError("training.router_diagnostics must be a boolean")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Small router training uses one worker; select the one-GPU router resource profile")
    seed_all(cfg["seed"])
    device = _router_device(cfg)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    merged, paths, identities, context_arrays, label_identity = _load_labels(cfg)
    objective, integration_steps = _objective(cfg)
    train_ids = np.flatnonzero(merged["partition"] == 0)
    val_ids = np.flatnonzero(merged["partition"] == 1)
    if not len(train_ids) or not len(val_ids):
        raise ValueError("Router labels must contain nonempty disjoint train and validation episode sets")
    if set(merged["episode"][train_ids]) & set(merged["episode"][val_ids]):
        raise ValueError("Episode leakage in router-label diagnostic split")
    available_train_labels = len(train_ids)
    train_ids = _fit_training_ids(cfg, context_arrays, train_ids)
    fit_ids_sha256 = digest(train_ids.tolist())
    gain_cpu = torch.from_numpy(merged["gain"]).float()
    scale = max(float(gain_cpu[train_ids].abs().median()), 1e-6)
    ranking_left, ranking_right = _validation_ranking_pairs(
        context_arrays, val_ids, gain_cpu[val_ids], cfg.get("labels", {}).get("gain_epsilon", 1e-6)
    )
    if diagnostics:
        train_left, train_right = _validation_ranking_pairs(
            context_arrays, train_ids, gain_cpu[train_ids], cfg.get("labels", {}).get("gain_epsilon", 1e-6)
        )
        train_left, train_right = train_left.to(device), train_right.to(device)
    validation_groups = tuple(value.to(device) for value in _context_groups(context_arrays, val_ids))
    training_groups = tuple(value.to(device) for value in _context_groups(context_arrays, train_ids)) if diagnostics else None
    # The full FP32 dataset is small (about 263 MiB for 64k default-width rows).
    # Transfer once; minibatches thereafter transfer only their sampled indices.
    query = torch.from_numpy(merged["query"]).to(device=device, dtype=torch.float32)
    candidate = torch.from_numpy(merged["candidate"]).to(device=device, dtype=torch.float32)
    gain = gain_cpu.to(device)
    val_indices = torch.from_numpy(val_ids).to(device)
    train_indices = torch.from_numpy(train_ids).to(device) if diagnostics else None
    ranking_left, ranking_right = ranking_left.to(device), ranking_right.to(device)
    feature_bytes = sum(value.numel() * value.element_size() for value in (query, candidate, gain))
    router = build_router(cfg).to(device)
    inputs, input_identity = {}, None
    if router.requires_children or router.requires_current:
        from stmem.training.router_inputs import load_router_inputs

        arrays, input_identity = load_router_inputs(cfg, paths, merged)
        if router.requires_children:
            if arrays.get("children") is None:
                raise ValueError("Child-aware router requires TREE child descriptors")
            inputs["children"] = torch.as_tensor(arrays["children"], dtype=torch.float32, device=device)
        if router.requires_current:
            for key, dtype in (("current", torch.float32), ("current_valid", torch.bool),
                               ("row_to_context", torch.long)):
                inputs[key] = torch.as_tensor(arrays[key], dtype=dtype, device=device)
            unique, first = np.unique(arrays["row_to_context"], return_index=True)
            inputs["context_query"] = query.new_zeros((len(arrays["current"]), query.shape[-1]))
            inputs["context_query"][torch.as_tensor(unique, device=device)] = query[torch.as_tensor(first, device=device)]
        feature_bytes += sum(value.numel() * value.element_size() for value in inputs.values())
        del arrays
    del merged, context_arrays, gain_cpu
    validation_batch = cfg["training"].get("router_validation_batch", 4096)
    if isinstance(validation_batch, bool) or not isinstance(validation_batch, int) or validation_batch < 1:
        raise ValueError("training.router_validation_batch must be a positive integer")
    label_hashes = [file_sha256(p) for p in paths]
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    start = 0
    if resume:
        old = load_checkpoint(resume)
        if old["config"]["reader"] != cfg["reader"] or old["config"]["model"] != cfg["model"]:
            raise ValueError("Router resume reader/model mismatch")
        if _objective(old["config"]) != (objective, integration_steps):
            raise ValueError("Router resume objective/integration_steps mismatch")
        if old.get("label_identity", label_identity) != label_identity:
            raise ValueError("Router resume frozen label identity changed")
        if old.get("router_input_identity") != input_identity:
            raise ValueError("Router resume richer input identity changed")
        if old["config"]["training"].get("router_fit_contexts", 0) != cfg["training"].get("router_fit_contexts", 0):
            raise ValueError("Router resume training-context subset changed")
        if cfg["training"].get("router_fit_contexts", 0) and old.get("fit_ids_sha256") != fit_ids_sha256:
            raise ValueError("Router resume training-context IDs changed")
        if old["label_hashes"] != label_hashes[:len(old["label_hashes"])]:
            raise ValueError("Router resume labels changed; refresh may only append a new fixed shard")
        if len(old["label_hashes"]) < len(label_hashes):
            if old["config"].get("labels", {}).get("use_refresh", False):
                raise ValueError("Router resume cannot append labels after the one allowed refresh")
            for ident in identities[len(old["label_hashes"]):]:
                if "objective" in ident and (not ident["refresh"] or ident["router"] != file_sha256(resume)):
                    raise ValueError("Refresh labels were not generated by the resumed router checkpoint")
        router.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        start = old["step"]
        restore_rng(old["rng_by_rank"][0])
        scale = old["target_scale"]
    directory = run_dir(cfg, "router")
    directory.mkdir(parents=True, exist_ok=True)
    if not resume and (directory / "last.pt").exists():
        raise FileExistsError("Router already trained; use --resume")
    steps = cfg["training"]["steps"]
    if start > steps:
        raise ValueError("Router checkpoint is beyond the intended optimizer-step budget")
    if start < steps:
        (directory / "complete.json").unlink(missing_ok=True)
    else:
        # An unchanged completed replay has no new optimization work to track.
        mark_complete(
            directory / "complete.json",
            completed_steps=steps,
            intended_steps=steps,
            checkpoint=directory / "last.pt",
        )
        (directory / "partial.json").unlink(missing_ok=True)
        return 0
    atomic_json(cfg, directory / "config.json")
    batch = cfg["training"]["effective_batch"]
    environment = environment_manifest()
    tracking_metadata = {
        "reader": cfg["reader"],
        "phase": "refresh" if cfg.get("labels", {}).get("use_refresh", False) else "initial",
        "label_hashes": label_hashes,
        "label_identity": label_identity,
        "objective": objective,
        "integration_steps": integration_steps,
        "router_device": device,
        "router_architecture": cfg["model"].get("router_architecture", "dot"),
        "router_parameters": sum(p.numel() for p in router.parameters()),
        "router_input_identity": input_identity,
        "router_fit_contexts": cfg["training"].get("router_fit_contexts", 0),
        "fit_ids_sha256": fit_ids_sha256,
        "available_train_labels": available_train_labels,
        "feature_bytes": feature_bytes,
        "teacher_sha256": identities[0]["teacher"],
        "compact_sha256": identities[0]["compact"],
        "data_identity": identities[0]["data"],
        "partition_sha256": digest(identities[0]["partition"]),
        "diagnostic": "conditional router-label validation; policy/encoders may have seen these episodes",
        "train_labels": len(train_ids),
        "validation_labels": len(val_ids),
        "planned_steps": steps,
        "resume_step": start,
        "router_diagnostics": diagnostics,
        "environment": environment,
        "logging": {**logging_options, "validation_every": validation_every},
    }
    with RunTracker(
        cfg,
        directory,
        job_type=f"router_{cfg['reader']}",
        name=cfg.get("tracking", {}).get("name") or ((f"seed{cfg.get('seed', 0)}-router-{'overfit-' if diagnostics else ''}{cfg['reader']}-{objective}"
              + (f"-{integration_steps}steps" if objective == "final_action" else ""))
        if diagnostics or "objective" in cfg.get("labels", {}) else None),
        metadata=tracking_metadata,
        resume=bool(resume),
    ) as tracker:
        started = time.perf_counter()
        baselines = _constant_baselines(gain, train_indices, val_indices, scale) if diagnostics else {}

        def full_fit_metrics():
            router.eval()
            result = _validation_metrics(
                router, query, candidate, gain, val_indices, ranking_left, ranking_right,
                scale, validation_batch,
                inputs=inputs, groups=validation_groups,
            )
            if diagnostics:
                result.update(_split_metrics(
                    router, query, candidate, gain, train_indices, train_left, train_right,
                    scale, validation_batch, training=True,
                    inputs=inputs, groups=training_groups,
                ))
                result.update(baselines, training_rank_pairs=len(train_left))
            return result

        def publish(row, current):
            with (directory / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            tracker.log(row, step=current)

        if diagnostics:
            # Evaluate the unchanged source checkpoint before the first update.
            # LocalRouter contains no stochastic eval layers; these read-only
            # passes consume no RNG and never enter the optimization graph.
            publish({
                "step": start, "target_scale_training_only": scale,
                "train_labels": len(train_ids), "validation_labels": len(val_ids),
                "validation_rank_pairs": len(ranking_left),
                "training_examples_this_allocation": 0, **full_fit_metrics(),
                "seconds": time.perf_counter() - started,
            }, start)
            tracker.summary(baselines)
        for step in range(start, steps):
            router.train()
            # Retain the original CPU RNG draw and row order for exact resume.
            indices = torch.from_numpy(train_ids[torch.randint(len(train_ids), (batch,)).numpy()]).to(device)
            prediction = _router_prediction(router, query, candidate, indices, inputs)
            loss = F.smooth_l1_loss(prediction, gain[indices] / scale)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite router loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                router.parameters(), cfg["training"]["grad_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            current = step + 1
            stop = bool(stop_file and Path(stop_file).exists())
            # More frequent scalar diagnostics must not run the full held-out
            # label evaluation more often or alter its established cadence.
            validation_due = current % validation_every == 0 or current == steps or stop
            if current == start + 1 or current % logging_options["log_every"] == 0 or validation_due:
                row = {
                    "step": current,
                    "training_gain_loss": float(loss.detach()),
                    "target_scale_training_only": scale,
                    "train_labels": len(train_ids),
                    "validation_labels": len(val_ids),
                    "validation_rank_pairs": len(ranking_left),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "effective_batch": batch,
                    "training_examples_this_allocation": (current - start) * batch,
                    "gradient_norm_before_clip": float(gradient_norm.detach()),
                    "gpu_peak_allocated_bytes_rank": torch.cuda.max_memory_allocated(device)
                    if device == "cuda"
                    else 0,
                    "gpu_peak_reserved_bytes_rank": torch.cuda.max_memory_reserved(device)
                    if device == "cuda"
                    else 0,
                    **process_memory(),
                }
                if validation_due:
                    row.update(full_fit_metrics())
                row["seconds"] = time.perf_counter() - started
                row["training_examples_per_second"] = row["training_examples_this_allocation"] / max(
                    row["seconds"], 1e-9
                )
                publish(row, current)
            if (
                current % cfg["training"]["checkpoint_every"] == 0
                or current == steps
                or stop
                or current in cfg["training"]["checkpoint_steps"]
            ):
                state = {
                    "stage": "router",
                    "environment": environment,
                    "tracking": dict(getattr(tracker, "identity", {})),
                    "reader": cfg["reader"],
                    "model": router.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": None,
                    "scaler": None,
                    "step": current,
                    "config": cfg,
                    "rng_by_rank": [capture_rng()],
                    "target_scale": scale,
                    "label_hashes": label_hashes,
                    "label_identity": label_identity,
                    "router_input_identity": input_identity,
                    "fit_ids_sha256": fit_ids_sha256,
                    "objective": objective,
                    "integration_steps": integration_steps,
                    "provenance": {
                        "compact": identities[0]["compact"],
                        "teacher": identities[0]["teacher"],
                        "data": identities[0]["data"],
                        "partition": digest(identities[0]["partition"]),
                        "cache": identities[0].get("cache"),
                        "objective": objective,
                        "integration_steps": integration_steps,
                    },
                }
                atomic_save(state, directory / "last.pt")
                checkpoint_path = directory / "last.pt"
                if current in cfg["training"]["checkpoint_steps"]:
                    candidate_path = directory / f"step_{current:08d}.pt"
                    atomic_copy(directory / "last.pt", candidate_path, immutable=True)
                    checkpoint_path = candidate_path
                tracker.checkpoint(checkpoint_path, step=current)
            if stop and current < steps:
                atomic_json({"status": "checkpointed_partial", "step": current}, directory / "partial.json")
                tracker.summary({"completed_steps": current, "planned_steps": steps})
                tracker.finish(status="checkpointed_partial", exit_code=75)
                return 75
        mark_complete(
            directory / "complete.json",
            completed_steps=steps,
            intended_steps=steps,
            checkpoint=directory / "last.pt",
        )
        (directory / "partial.json").unlink(missing_ok=True)
        tracker.summary(
            {
                "completed_steps": steps,
                "planned_steps": steps,
                "target_scale_training_only": scale,
                "train_labels": len(train_ids),
                "validation_labels": len(val_ids),
                "checkpoint/sha256": json.loads((directory / "complete.json").read_text())[
                    "checkpoint_sha256"
                ],
            }
        )
        return 0
