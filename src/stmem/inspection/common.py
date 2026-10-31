"""Checkpoint integrity, immutable inspection manifests and local data assets."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

import torch
import yaml

from stmem.config import digest, validate_config
from stmem.models.policy import MemoryPolicy
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.continuation import continuation_record

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = {
    "version": "usual_checkpoint_inspection_v1",
    "scope": "Fixed-scene diagnostics, not official benchmark success rates",
    "offline_task": "VideoUnmask",
    "offline_episodes": [0, 1],
    "offline_decisions": [0, 32, 64, 96, 128],
    "offline_noise_seeds": [1701, 1702, 1703, 1704],
    "cube_task": "PickXtimes",
    "cube_split": "val",
    "cube_episode": 0,
    "cube_noise_seeds": [2701, 2702, 2703, 2704],
    "cube_shifts_m": [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, -0.1, 0]],
    "replay_probe_steps": [0, 40, 80],
    "rollout_native_commands": 160,
    "handoff_scenes": [["VideoUnmask", 0], ["VideoUnmask", 1], ["PickXtimes", 0]],
    "handoff_prefixes": [0, 4, 16, 32, 64],
    "handoff_native_commands": 320,
}


def read_json(path):
    return json.loads(Path(path).read_text())


def read_manifest(path):
    result = read_json(path)
    if result.get("schema_version") != 1 or result.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported inspection manifest/protocol")
    return result


def discover_checkpoints(bundle):
    """Accept the rsync bundle format; reject incomplete or escaping paths."""
    bundle = Path(bundle).resolve()
    checksums = bundle / "SHA256SUMS"
    if not checksums.is_file():
        raise FileNotFoundError(f"Missing transfer integrity manifest: {checksums}")
    entries = {}
    for line in checksums.read_text().splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([a-fA-F0-9]{64})\s+\*?([a-z][a-z0-9_]*)/checkpoint\.pt", line)
        if match is None:
            raise ValueError(f"Unsupported SHA256SUMS entry: {line!r}")
        sha, label = match.groups()
        if label in entries:
            raise ValueError(f"Duplicate checkpoint label: {label}")
        path = (bundle / label / "checkpoint.pt").resolve()
        if not path.is_relative_to(bundle):
            raise ValueError(f"Checkpoint must be inside the bundle: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint has not been transferred: {path}")
        actual = file_sha256(path)
        if actual != sha.lower():
            raise ValueError(f"Incomplete or changed checkpoint transfer: {path} (SHA256 mismatch)")
        entries[label] = {"path": str(path), "sha256": actual}
    if not entries:
        raise ValueError("No checkpoints in SHA256SUMS")
    return entries


def _finite_tensors(value):
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError("Checkpoint contains a nonfinite tensor")
        return 1
    if isinstance(value, dict):
        return sum(_finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_finite_tensors(v) for v in value)
    return 0


def audit_checkpoint(entry):
    path = Path(entry["path"])
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Checkpoint ZIP CRC failed: {path}: {bad}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    validate_config(cfg)
    if state["stage"] != "teacher" or cfg.get("debug") or cfg["data"].get("synthetic"):
        raise ValueError("This inspection requires real-data teacher checkpoints")
    for field in ("model", "training"):
        if state["provenance"][field] != digest(cfg[field]):
            raise ValueError(f"Checkpoint {field} configuration/provenance mismatch")
    policy = MemoryPolicy(cfg).cpu().eval()
    policy.load_state_dict(state["model"], strict=True)
    tensors = _finite_tensors(state["model"]) + _finite_tensors(state["optimizer"])
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["training"]["lr"])
    optimizer.load_state_dict(state["optimizer"])
    for parameter, values in optimizer.state.items():
        for name in ("exp_avg", "exp_avg_sq"):
            if name in values and values[name].shape != parameter.shape:
                raise ValueError(f"Invalid optimizer {name} tensor shape")
        if "step" in values and int(values["step"]) != state["step"]:
            raise ValueError("Optimizer step disagrees with checkpoint step")
    provenance = state["provenance"]
    expected_sampler = {
        "next_global_position": state["step"] * provenance["effective_batch"],
        "boundary": "optimizer_step", "version": provenance["sampler"],
        "rank_assignment": provenance["rank_assignment"],
    }
    if state["sampler"] != expected_sampler:
        raise ValueError("Invalid checkpoint sampler boundary")
    if len(state["rng_by_rank"]) != provenance["world_size"]:
        raise ValueError("Missing per-rank random states")
    if state["scheduler"]["last_epoch"] != state["step"]:
        raise ValueError("Scheduler step disagrees with checkpoint step")
    t = cfg["training"]
    horizon = t.get("schedule_steps", t["steps"])
    warmup, step = t["warmup_steps"], state["step"]
    factor = min((step + 1) / max(warmup, 1), 1.0) * (
        0.1 + 0.9 * (1 + math.cos(math.pi * min(max(step-warmup, 0) / max(horizon-warmup, 1), 1))) / 2
    )
    lrs = [group["lr"] for group in optimizer.param_groups]
    if any(not math.isclose(lr, t["lr"] * factor, rel_tol=1e-12) for lr in lrs):
        raise ValueError("Saved learning rate disagrees with the original schedule")
    if state.get("continuation") is not None:
        continuation_record(cfg, state, requested=True)
    sidecar = path.parent / "source.json"
    source = read_json(sidecar) if sidecar.exists() else {}
    if source and (source.get("step") != step or source.get("sha256") != entry["sha256"]):
        raise ValueError(f"source.json disagrees with checkpoint: {path}")
    return {
        **entry, "step": step, "bytes": path.stat().st_size, "status": "passed",
        "parameter_count": sum(p.numel() for p in policy.parameters()),
        "checked_tensors": tensors, "config": cfg, "provenance": provenance,
        "tracking": state.get("tracking"), "continuation": state.get("continuation"),
        "source": source, "learning_rates": lrs,
        "checks": {key: True for key in (
            "sha256", "zip_crc", "finite_tensors", "strict_model_load", "optimizer_load",
            "optimizer_shapes", "optimizer_step", "sampler", "scheduler", "rng_ranks", "learning_rate",
        )},
    }


def resolve_assets(config):
    config = Path(config).resolve()
    values = yaml.safe_load(config.read_text())
    names = {"prepared", "alignment", "features", "clip_checkpoint", "pick_prefix", "replay_commands"}
    if not isinstance(values, dict) or set(values) != names:
        raise ValueError(f"Asset config must define exactly {sorted(names)}")
    assets, hashes = {}, {}
    for name, value in values.items():
        path = (REPO / Path(value).expanduser()).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing inspection asset {name}: {path}; update {config}")
        assets[name] = str(path)
        if path.is_file():
            hashes[name] = file_sha256(path)
        else:
            files = sorted(p for p in path.rglob("*") if p.is_file() and not p.name.endswith(".lock"))
            if not files:
                raise ValueError(f"Empty inspection asset directory: {path}")
            hashes[name] = {str(p.relative_to(path)): file_sha256(p) for p in files}
    return assets, hashes


def policy_source_hash():
    """Scientific implementation identity, separate from diagnostic report code."""
    value = hashlib.sha256()
    for path in sorted((REPO / "src/stmem").rglob("*.py")):
        if "inspection" in path.relative_to(REPO / "src/stmem").parts:
            continue
        value.update(str(path.relative_to(REPO)).encode())
        value.update(path.read_bytes())
    return value.hexdigest()


def prepare_manifest(bundle, output, assets_config, *, device="cuda:0", resume=False):
    output = Path(output).resolve()
    manifest_path = output / "inspection_manifest.json"
    if manifest_path.exists() and not resume:
        raise FileExistsError(f"Inspection exists; use --resume: {output}")
    entries = discover_checkpoints(bundle)
    assets, asset_hashes = resolve_assets(assets_config)
    audits = {}
    for label, entry in entries.items():
        audits[label] = audit_checkpoint(entry)
        print(json.dumps({"event": "checkpoint_verified", "label": label, "step": audits[label]["step"]}), flush=True)
    candidate = {
        "schema_version": 1, "bundle": str(Path(bundle).resolve()), "output": str(output),
        "device": device, "labels": list(entries),
        "checkpoints": {name: {**entry, "step": audits[name]["step"]} for name, entry in entries.items()},
        "assets": assets, "asset_hashes": asset_hashes, "protocol": PROTOCOL,
        "policy_source_sha256": policy_source_hash(),
    }
    if manifest_path.exists():
        previous = read_manifest(manifest_path)
        if previous != candidate:
            raise ValueError("Inspection inputs/protocol/source changed; use a new output directory")
    else:
        if output.exists() and any(p.name != ".inspection.lock" for p in output.iterdir()):
            raise FileExistsError(f"Nonempty output has no inspection manifest: {output}")
        atomic_json(candidate, manifest_path)
    atomic_json({
        "status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": audits, "inspection_manifest_sha256": file_sha256(manifest_path),
        "scope": "CPU checkpoint integrity only; no model optimization",
    }, output / "checkpoint_audit.json")
    return candidate, manifest_path
