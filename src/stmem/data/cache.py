"""Immutable per-task HDF5 shards for completed blocks and queried causal prefixes."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from stmem.config import load_config, digest, dependency, source_revision, run_dir
from stmem.training.checkpoints import file_sha256, atomic_json
from stmem.training.runtime import environment_manifest

FORMAT = 1


def cache_identity(cfg, teacher_path, data_identity):
    identity = {
        "format": FORMAT,
        "teacher_sha256": file_sha256(teacher_path),
        "geometry": cfg["geometry"],
        "data": cfg["data"],
        "model_encoder": {
            **{
                k: cfg["model"][k]
                for k in (
                    "encoder",
                    "points",
                    "block_size",
                    "visual_tokens",
                    "current_tokens",
                    "width",
                    "encoder_options",
                )
            },
            "condition_on_previous_joint_commands": cfg["model"].get(
                "condition_on_previous_joint_commands", True
            ),
        },
        "normalization_version": "masked_training_commands_v1",
        "frozen_encoder_precision": "fp32",
        "data_identity": data_identity,
        "prefix_convention": "history_strictly_before_current_v1",
        "code": source_revision()["source_sha256"],
    }
    # Preserve the legacy omission semantics; new input contracts must never
    # share a cache with the old state/command or XYZRGB recipe.
    from stmem.vision.frozen import visual_feature_identity, resolve_visual_features

    for key, default in (("proprioception_mode", "all"), ("condition_on_previous_commands", True)):
        if key in cfg["model"]:
            identity["model_encoder"][key] = cfg["model"].get(key, default)
    if resolve_visual_features(cfg)["mode"] != "xyzrgb":
        identity["model_encoder"]["visual_features"] = visual_feature_identity(cfg)
        identity["model_encoder"]["feature_projection_dim"] = resolve_visual_features(cfg)["projection_dim"]
    return identity


def write_records(group, records):
    group.attrs["length"] = len(records)
    if not records:
        return
    for field in ("content_sum", "anchor", "bounds_min", "bounds_max", "max_content"):
        group.create_dataset(
            field,
            data=torch.stack([getattr(r, field).detach().cpu() for r in records]).numpy(),
            compression="lzf",
        )
    string = h5py.string_dtype("utf-8")
    for field in ("node_id", "source", "modality"):
        group.create_dataset(field, data=[getattr(r, field) for r in records], dtype=string)
    for field in (
        "count",
        "time_min",
        "time_max",
        "depth",
        "scale",
        "completed",
        "disconnected",
        "time_anchor",
        "occupancy",
    ):
        group.create_dataset(field, data=[getattr(r, field) for r in records])


def read_records(group):
    from stmem.memory.records import MemoryRecord

    n = int(group.attrs["length"])
    if not n:
        return []
    values = {k: np.asarray(v) for k, v in group.items()}
    result = []
    tensor_names = {"content_sum", "anchor", "bounds_min", "bounds_max", "max_content"}
    for i in range(n):
        fields = {}
        for key, arr in values.items():
            v = arr[i]
            if key in tensor_names:
                v = torch.from_numpy(v.copy()).float()
            elif isinstance(v, bytes):
                v = v.decode()
            elif hasattr(v, "item"):
                v = v.item()
            fields[key] = v
        result.append(MemoryRecord(**fields))
    return result


def episode_key(task, episode_id):
    return digest([task, episode_id])[:24]


class FrozenCache:
    def __init__(self, cfg, teacher_path, data):
        self.cfg = cfg
        self.identity = cache_identity(cfg, teacher_path, data.identity)
        self.root = Path(cfg["paths"]["cache_root"]) / digest(self.identity)[:24]
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"No compatible frozen cache: {self.root}. Run stmem.data.cache after checkpoint selection."
            )
        actual = json.loads(manifest.read_text())
        if actual["identity"] != self.identity or actual["status"] != "complete":
            raise ValueError("Stale or incomplete encoder cache")
        self.manifest = actual

    def evidence(self, decision):
        from stmem.memory.hierarchy import Archive
        from stmem.models.policy import Evidence

        fsize = self.cfg["model"]["block_size"]
        archive = Archive()
        with h5py.File(self.root / f"{decision.task}.h5", "r") as f:
            group = f[episode_key(decision.task, decision.episode_id)]
            for key in sorted(group["conditioning"], key=int):
                archive.add_block(read_records(group["conditioning"][key]), f"conditioning/{key}")
            nfull, pending = divmod(decision.index, fsize)
            for block in range(nfull):
                archive.add_block(read_records(group["execution"][str(block * fsize)]), f"execution/{block}")
            if pending:
                archive.set_pending("execution", read_records(group["prefix"][f"{nfull * fsize}_{pending}"]))
            current = read_records(group["current"][str(decision.index)])
            text = torch.from_numpy(np.asarray(group["text"]).copy())
        clocks = {
            "conditioning": decision.conditioning[-1].time if decision.conditioning else 0.0,
            "execution": decision.current.time,
        }
        return Evidence(archive, current, text, decision.current, clocks)


def write_episode_cache(group, episode, policy, stride, horizon):
    """Cache exactly the same completed blocks and queried prefixes as streaming."""
    group.attrs["task"] = episode.task
    group.attrs["episode_id"] = episode.episode_id
    group.create_dataset("text", data=policy.text_encoder(episode.goal).detach().cpu().numpy())
    for name in ("conditioning", "execution", "prefix", "current"):
        group.create_group(name)
    fsize = policy.encoder.block_size
    # BlockEncoder centralizes source boundaries, completion status, and IDs.
    for source, frames in (("conditioning", episode.conditioning), ("execution", episode.execution)):
        for offset in range(0, len(frames), fsize):
            block = frames[offset : offset + fsize]
            if source == "execution" and len(block) < fsize:
                continue
            leaves = policy.encoder.encode_frames(block, source=source, completed=True)
            write_records(group[source].create_group(str(offset)), leaves)
    for time_index in range(0, len(episode), stride):
        if not episode.decision(time_index, horizon).target_valid.any():
            continue
        write_records(
            group["current"].create_group(str(time_index)),
            policy.encoder.current(episode.execution[time_index]),
        )
        offset = time_index // fsize * fsize
        n = time_index % fsize
        if n:
            # completed=False determines identity as well as metadata. Changing
            # flags after encoding would incorrectly reuse completed-block IDs.
            leaves = policy.encoder.encode_frames(
                episode.execution[offset:time_index], source="execution", completed=False
            )
            write_records(group["prefix"].create_group(f"{offset}_{n}"), leaves)


@torch.no_grad()
def build_cache(cfg, checkpoint):
    from stmem.training.data import TrainingData
    from stmem.training.runtime import (
        load_policy,
        load_checkpoint,
        validate_selection,
        validate_policy_config,
    )

    validate_selection(checkpoint, debug=cfg.get("debug", False) and cfg["data"].get("synthetic", False))
    checkpoint_state = load_checkpoint(checkpoint)
    validate_policy_config(checkpoint_state["config"], cfg)
    del checkpoint_state
    data = TrainingData(cfg)
    identity = cache_identity(cfg, checkpoint, data.identity)
    root = Path(cfg["paths"]["cache_root"]) / digest(identity)[:24]
    root.mkdir(parents=True, exist_ok=True)

    def record_completion():
        manifest_path = root / "manifest.json"
        atomic_json(
            {
                "status": "complete",
                "cache_path": str(root.resolve()),
                "cache_manifest_sha256": file_sha256(manifest_path),
                "teacher_sha256": file_sha256(checkpoint),
                "identity_sha256": digest(identity),
            },
            run_dir(cfg, "cache") / "complete.json",
        )

    if (root / "manifest.json").exists():
        FrozenCache(cfg, checkpoint, data)
        record_completion()
        print(f"Compatible cache already complete: {root}")
        return root
    policy, _ = load_policy(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    policy.requires_grad_(False)
    start_time = time.perf_counter()
    for task in data.tasks:
        target = root / f"{task}.h5"
        temporary = target.with_suffix(".h5.tmp")
        with h5py.File(temporary, "w") as out:
            for epidx in data.by_task[task]:
                ep = data.episode(epidx)
                group = out.create_group(episode_key(task, ep.episode_id))
                write_episode_cache(group, ep, policy, data.stride, data.horizon)
            out.flush()
        os.replace(temporary, target)
    manifest = {
        "identity": identity,
        "status": "complete",
        "environment": environment_manifest(),
        "seconds": time.perf_counter() - start_time,
        "shards": {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size} for p in root.glob("*.h5")},
    }
    atomic_json(manifest, root / "manifest.json")
    record_completion()
    print(json.dumps({"cache": str(root), "seconds": manifest["seconds"]}))
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint")
    args = p.parse_args()
    cfg = load_config(args.config)
    build_cache(cfg, args.checkpoint or dependency(cfg, "teacher"))


if __name__ == "__main__":
    main()
