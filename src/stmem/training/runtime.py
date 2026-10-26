"""Model loading and deterministic distributed runtime helpers."""

from __future__ import annotations
import os
import random
import json
from pathlib import Path
from functools import lru_cache
import importlib.metadata
import platform
import subprocess
import sys
import numpy as np
import torch
from stmem.models.policy import MemoryPolicy


@lru_cache(maxsize=1)
def environment_manifest():
    """Actual process environment, distinct from configured target GPU names."""
    from stmem.config import digest

    packages = {}
    for package in ("torch", "torchvision", "numpy", "h5py", "PyYAML", "robomme", "mani-skill", "sapien"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    driver = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            driver = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_wheel_cuda": torch.version.cuda,
        "driver": driver,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
    }
    return {"details": info, "sha256": digest(info)}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_setup():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        torch.distributed.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device, rank, world


def load_checkpoint(path):
    # Only load checkpoints produced by this trusted local project.
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_policy_config(checkpoint_cfg: dict, cfg: dict) -> None:
    """Reject silent changes to the learned policy's observation/action contract.

    Reader budgets, declared episode bounds, checkpointing and archive placement
    may change without altering encoded evidence. Sensor geometry, tokenization,
    decoder semantics and primitive control cadence must match, including debug
    runs. Shorter diagnostic environment limits are deliberately allowed.
    """
    from stmem.config import require_sampling_contract

    require_sampling_contract(checkpoint_cfg)
    require_sampling_contract(cfg)
    model_fields = (
        "encoder",
        "points",
        "block_size",
        "visual_tokens",
        "current_tokens",
        "width",
        "layers",
        "heads",
        "text_layers",
        "max_text_bytes",
        "horizon",
        "execute",
        "flow_steps",
        "solver",
        "decoder",
        "encoder_options",
        "action_space",
        "mass_bias",
    )
    mismatches = []
    # Older checkpoints used all command channels; omission means the original
    # contract, rather than an unspecified setting that can change at evaluation.
    key = "condition_on_previous_joint_commands"
    recorded = checkpoint_cfg.get("model", {}).get(key, True)
    requested = cfg.get("model", {}).get(key, True)
    if not isinstance(recorded, bool) or not isinstance(requested, bool):
        raise ValueError(f"model.{key} must be boolean")
    if recorded != requested:
        mismatches.append(f"model.{key}: checkpoint={recorded!r}, requested={requested!r}")
    for key in model_fields:
        recorded = checkpoint_cfg.get("model", {}).get(key)
        requested = cfg.get("model", {}).get(key)
        if recorded != requested:
            mismatches.append(f"model.{key}: checkpoint={recorded!r}, requested={requested!r}")
    if checkpoint_cfg.get("geometry") != cfg.get("geometry"):
        mismatches.append("geometry differs from checkpoint calibration/sampling settings")
    for key in (
        "clock",
        "nominal_control_hz",
        "process_every_observation",
        "sensors",
        "benchmark_revision",
        "simulator_revision",
    ):
        if checkpoint_cfg.get("data", {}).get(key) != cfg.get("data", {}).get(key):
            mismatches.append(f"data.{key} differs from checkpoint")
    if mismatches:
        raise ValueError("Policy input/action contract mismatch: " + "; ".join(mismatches))


def validate_selection(path, *, debug=False):
    from stmem.training.checkpoints import file_sha256

    path = Path(path)
    manifest = path.with_suffix(".selection.json")
    if not manifest.exists():
        raise ValueError(f"Immutable checkpoint selection record missing: {manifest}")
    record = json.loads(manifest.read_text())
    if debug and record.get("synthetic_debug_only"):
        return
    if record.get("sha256") != file_sha256(path):
        raise ValueError("Selected checkpoint hash differs from its validation selection record")


def load_policy(path, device="cpu"):
    state = load_checkpoint(path)
    policy = MemoryPolicy(state["config"])
    if state["stage"] == "perceiver":
        policy.add_perceiver()
    policy.load_state_dict(state["model"])
    if state["stage"] != "teacher":
        policy.freeze_encoders()
    policy.to(device)
    return policy, state


def load_router(path, device="cpu"):
    from stmem.models.router import LocalRouter

    state = load_checkpoint(path)
    m = state["config"]["model"]
    router = LocalRouter(m["width"], m["router_width"])
    router.load_state_dict(state["model"])
    router.to(device).eval().requires_grad_(False)
    return router, state
