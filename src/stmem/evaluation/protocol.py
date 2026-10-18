"""Enforce the frozen official-test protocol even through direct CLI calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stmem.config import source_revision
from stmem.training.checkpoints import file_sha256


def inference_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    """Paths may move; observations, architecture, cadence and evaluation cannot."""
    evaluation = {
        key: value for key, value in cfg["evaluation"].items() if key not in ("test_frozen_manifest", "split")
    }
    return {
        "model": cfg["model"],
        "geometry": cfg["geometry"],
        "data": {key: value for key, value in cfg["data"].items() if key != "alignment_path"},
        "evaluation": evaluation,
        "precision": cfg["training"]["precision"],
        "debug": cfg.get("debug", False),
    }


def validate_frozen(
    cfg: dict[str, Any],
    frozen: dict[str, Any],
    method: str,
    budget: int,
    count: int,
    hashes: dict[str, str],
    selection_family: str | None = None,
    intervention: str | None = None,
) -> None:
    manifest = frozen.get("protocol", {})
    if method not in manifest.get("methods", []):
        raise ValueError("Test method was not declared in the frozen protocol")
    if cfg["seed"] not in manifest.get("seeds", []):
        raise ValueError("Training seed was not declared in the frozen protocol")
    if cfg["data"]["tasks"] != manifest.get("tasks"):
        raise ValueError("Official test task set/order differs from the frozen protocol")
    if budget != manifest.get("budget"):
        raise ValueError("Test budget differs from the frozen protocol")
    if count != manifest.get("evaluation", {}).get("final_test_episodes_per_task"):
        raise ValueError("Official test episode count differs from the frozen protocol")
    if selection_family is not None or intervention not in (None, "none"):
        raise ValueError("Selection contexts and history interventions are not frozen primary test methods")
    if frozen.get("source_sha256") != source_revision()["source_sha256"]:
        raise ValueError("Source code changed after official test protocol freeze")
    expected_configs = frozen.get("configuration_hashes", {})
    if not expected_configs:
        raise ValueError("Frozen protocol lacks configuration provenance")
    for path, expected in expected_configs.items():
        if not Path(path).is_file() or file_sha256(path) != expected:
            raise ValueError(f"Frozen configuration changed or is unavailable: {path}")
    expected_checkpoints = frozen.get("checkpoints", {})
    if not hashes or any(
        expected_checkpoints.get(str(Path(path).resolve())) != value for path, value in hashes.items()
    ):
        raise ValueError("Test checkpoint differs from the frozen protocol")
    for path, expected in expected_checkpoints.items():
        if not Path(path).is_file() or file_sha256(path) != expected:
            raise ValueError(f"Frozen checkpoint changed or is unavailable: {path}")
    family = "teacher" if method == "FULL" else "perceiver" if method == "PERCEIVER" else "compact"
    roles = frozen.get("checkpoint_roles", {}).get(str(cfg["seed"]), {})
    expected_roles = [family] + (["router_" + method.lower()] if method in ("TREE", "FLAT") else [])
    if any(role not in roles for role in expected_roles) or set(hashes) != {
        roles[role] for role in expected_roles
    }:
        raise ValueError("Checkpoint roles/training seed do not match the frozen method")
    if frozen.get("inference_contracts", {}).get(family) != inference_contract(cfg):
        raise ValueError(
            "Inference configuration changed after freeze (sensors/model/geometry/cadence/evaluation)"
        )
