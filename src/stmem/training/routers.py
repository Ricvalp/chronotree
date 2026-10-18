"""Equal-capacity robust signed-gain regression for tree and flat scorers."""

from __future__ import annotations
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from stmem.config import run_dir, digest
from stmem.models.router import LocalRouter
from stmem.training.gain_labels import label_path
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


def train_router(cfg, *, resume=None, stop_file=None):
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
    atomic_json(cfg, directory / "config.json")
    label_hashes = [file_sha256(p) for p in paths]
    batch = cfg["training"]["effective_batch"]
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
        torch.nn.utils.clip_grad_norm_(
            router.parameters(), cfg["training"]["grad_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        current = step + 1
        stop = bool(stop_file and Path(stop_file).exists())
        if current % cfg["training"]["log_every"] == 0 or current == steps or stop:
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
            row = {
                "step": current,
                "training_gain_loss": float(loss.detach()),
                "conditional_label_validation_loss": float(val_loss),
                "validation_sign_agreement": float(sign_agreement),
                "target_scale_training_only": scale,
                "seconds": time.perf_counter() - started,
                "train_labels": len(train_ids),
                "validation_labels": len(val_ids),
            }
            with (directory / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if (
            current % cfg["training"]["checkpoint_every"] == 0
            or current == steps
            or stop
            or current in cfg["training"]["checkpoint_steps"]
        ):
            state = {
                "stage": "router",
                "environment": environment_manifest(),
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
            if current in cfg["training"]["checkpoint_steps"]:
                candidate_path = directory / f"step_{current:08d}.pt"
                atomic_copy(directory / "last.pt", candidate_path, immutable=True)
        if stop and current < steps:
            atomic_json({"status": "checkpointed_partial", "step": current}, directory / "partial.json")
            return 75
    mark_complete(
        directory / "complete.json",
        completed_steps=steps,
        intended_steps=steps,
        checkpoint=directory / "last.pt",
    )
    (directory / "partial.json").unlink(missing_ok=True)
    return 0
