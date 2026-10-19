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
from stmem.models.router import LocalRouter
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


def train_router(cfg, *, resume=None, stop_file=None, log_every=None):
    logging_options = resolve_logging(cfg, log_every=log_every)
    validation_every = positive_log_every(cfg["training"]["log_every"])
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Small router training uses one worker; select the one-GPU router resource profile")
    seed_all(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    paths = [label_path(cfg)]
    if cfg.get("labels", {}).get("use_refresh", False):
        paths.append(label_path(cfg, True))
    arrays = {k: [] for k in ("query", "candidate", "gain", "partition", "context_id", "episode")}
    identities = []
    for p in paths:
        with h5py.File(p, "r") as f:
            if not f.attrs.get("complete"):
                raise ValueError("Incomplete label shard")
            identities.append(json.loads(f.attrs["identity"]))
            for k in arrays:
                arrays[k].append(np.asarray(f[k]))
    merged = {k: np.concatenate(v) for k, v in arrays.items()}
    train_ids = np.flatnonzero(merged["partition"] == 0)
    val_ids = np.flatnonzero(merged["partition"] == 1)
    if not len(train_ids) or not len(val_ids):
        raise ValueError("Router labels must contain nonempty disjoint train and validation episode sets")
    if set(merged["episode"][train_ids]) & set(merged["episode"][val_ids]):
        raise ValueError("Episode leakage in router-label diagnostic split")
    for ident in identities[1:]:
        if ident["partition"] != identities[0]["partition"] or ident["compact"] != identities[0]["compact"]:
            raise ValueError("Refresh changed episode partitions or frozen compact policy")
    query = torch.from_numpy(merged["query"]).float()
    candidate = torch.from_numpy(merged["candidate"]).float()
    gain = torch.from_numpy(merged["gain"]).float()
    scale = max(float(gain[train_ids].abs().median()), 1e-6)
    router = LocalRouter(cfg["model"]["width"], cfg["model"]["router_width"]).to(device)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"]
    )
    start = 0
    if resume:
        old = load_checkpoint(resume)
        if old["config"]["reader"] != cfg["reader"] or old["config"]["model"] != cfg["model"]:
            raise ValueError("Router resume reader/model mismatch")
        if not set(old["label_hashes"]).issubset({file_sha256(p) for p in paths}):
            raise ValueError("Router resume labels changed; refresh may only append a new fixed shard")
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
    label_hashes = [file_sha256(p) for p in paths]
    batch = cfg["training"]["effective_batch"]
    ranking_left, ranking_right = _validation_ranking_pairs(
        arrays["context_id"], val_ids, gain[val_ids], cfg.get("labels", {}).get("gain_epsilon", 1e-6)
    )
    environment = environment_manifest()
    tracking_metadata = {
        "reader": cfg["reader"],
        "phase": "refresh" if cfg.get("labels", {}).get("use_refresh", False) else "initial",
        "label_hashes": label_hashes,
        "teacher_sha256": identities[0]["teacher"],
        "compact_sha256": identities[0]["compact"],
        "data_identity": identities[0]["data"],
        "partition_sha256": digest(identities[0]["partition"]),
        "diagnostic": "conditional router-label validation; policy/encoders may have seen these episodes",
        "train_labels": len(train_ids),
        "validation_labels": len(val_ids),
        "planned_steps": steps,
        "resume_step": start,
        "environment": environment,
        "logging": {**logging_options, "validation_every": validation_every},
    }
    with RunTracker(
        cfg,
        directory,
        job_type=f"router_{cfg['reader']}",
        metadata=tracking_metadata,
        resume=bool(resume),
    ) as tracker:
        started = time.perf_counter()
        for step in range(start, steps):
            router.train()
            indices = train_ids[torch.randint(len(train_ids), (batch,)).numpy()]
            prediction = router(query[indices].to(device), candidate[indices].to(device))
            loss = F.smooth_l1_loss(prediction, gain[indices].to(device) / scale)
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
                    router.eval()
                    with torch.no_grad():
                        val_pred = torch.cat(
                            [
                                router(query[ids].to(device), candidate[ids].to(device)).cpu()
                                for ids in np.array_split(val_ids, max(1, len(val_ids) // 1024 + 1))
                            ]
                        )
                        val_loss = F.smooth_l1_loss(val_pred, gain[val_ids] / scale)
                        sign_agreement = ((val_pred > 0) == (gain[val_ids] > 0)).float().mean()
                    row.update(
                        conditional_label_validation_loss=float(val_loss),
                        validation_sign_agreement=float(sign_agreement),
                    )
                    if len(ranking_left):
                        predicted_order = (val_pred[ranking_left] - val_pred[ranking_right]).sign()
                        expected_order = (gain[val_ids][ranking_left] - gain[val_ids][ranking_right]).sign()
                        row["validation_within_context_rank_agreement"] = float(
                            (predicted_order == expected_order).float().mean()
                        )
                row["seconds"] = time.perf_counter() - started
                row["training_examples_per_second"] = row["training_examples_this_allocation"] / max(
                    row["seconds"], 1e-9
                )
                with (directory / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                tracker.log(row, step=current)
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
                    "provenance": {
                        "compact": identities[0]["compact"],
                        "teacher": identities[0]["teacher"],
                        "data": identities[0]["data"],
                        "partition": digest(identities[0]["partition"]),
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
