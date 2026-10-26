"""Explicit YAML configuration, inheritance and stable provenance."""

from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any
import yaml


def merge(base: dict, changes: dict) -> dict:
    result = dict(base)
    for key, value in changes.items():
        result[key] = merge(result.get(key, {}), value) if isinstance(value, dict) else value
    return result


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()

    def read(p: Path, seen: set) -> dict:
        if p in seen:
            raise ValueError(f"Configuration inheritance cycle: {p}")
        obj = yaml.safe_load(os.path.expandvars(p.read_text()))
        if not isinstance(obj, dict):
            raise ValueError(f"Expected mapping in {p}")
        parent = obj.pop("extends", None)
        return merge(read((p.parent / parent).resolve(), seen | {p}), obj) if parent else obj

    cfg = read(path, set())
    # Site roots are deliberately explicit environment overrides.
    for env, key in [
        ("STMEM_DATA_ROOT", "data_root"),
        ("STMEM_CACHE_ROOT", "cache_root"),
        ("STMEM_RUN_ROOT", "run_root"),
    ]:
        if os.environ.get(env):
            cfg.setdefault("paths", {})[key] = os.environ[env]
    return cfg


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source_revision() -> dict:
    def git(*args):
        p = subprocess.run(["git", *args], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else "uncommitted"

    # Content hash is useful before the initial repository commit too.
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    for p in sorted((root / "src").rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return {
        "git": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "source_sha256": h.hexdigest(),
    }


def run_dir(cfg: dict, stage: str | None = None) -> Path:
    name = stage or cfg.get("stage", "teacher")
    if name == "router":
        name += "_" + cfg["reader"]
    return Path(cfg["paths"]["run_root"]) / f"seed{cfg['seed']}" / name


def dependency(cfg: dict, name: str) -> Path:
    value = cfg.get("checkpoints", {}).get(name)
    if value:
        return Path(value.format(run_root=cfg["paths"]["run_root"], seed=cfg["seed"]))
    return run_dir(cfg, name) / "selected.pt"


def require_sampling_contract(cfg: dict) -> None:
    """Old resolved snapshots must not acquire different preprocessing defaults.

    Direct low-level encoders retain an explicit FPS reference. Policy configs
    must record both selectors so preparation and learned inference cannot
    resolve missing settings to different algorithms.
    """
    if not isinstance(cfg.get("geometry", {}).get("point_sampling"), dict):
        raise ValueError(
            "Missing explicit geometry.point_sampling; regenerate the configuration from configs/base.yaml"
        )
    if not isinstance(cfg.get("model", {}).get("encoder_options", {}).get("center_sampling"), dict):
        raise ValueError(
            "Missing explicit model.encoder_options.center_sampling; regenerate the configuration from configs/base.yaml"
        )


def validate_config(cfg: dict) -> None:
    require_sampling_contract(cfg)
    if not isinstance(cfg.get("tracking", {}).get("system_stats", True), bool):
        raise ValueError("tracking.system_stats must be boolean")
    scenes = cfg.get("tracking", {}).get("scenes", {})
    if not isinstance(scenes, dict) or set(scenes) - {"enabled", "count", "max_points"}:
        raise ValueError("tracking.scenes accepts only enabled, count, max_points")
    if not isinstance(scenes.get("enabled", True), bool):
        raise ValueError("tracking.scenes.enabled must be boolean")
    for key, low, high in (("count", 1, 2), ("max_points", 1, 4096)):
        value = scenes.get(key, 2 if key == "count" else 1024)
        if not isinstance(value, int) or isinstance(value, bool) or not low <= value <= high:
            raise ValueError(f"tracking.scenes.{key} must be an integer in [{low}, {high}]")
    m, t = cfg["model"], cfg["training"]
    if not isinstance(m.get("condition_on_previous_joint_commands", True), bool):
        raise ValueError("model.condition_on_previous_joint_commands must be boolean")
    from stmem.data.pointcloud import point_sampler_settings
    from stmem.data.center_sampling import center_sampler_settings, center_counts

    point_sampler_settings(cfg)
    center_sampler_settings(cfg)
    center_counts(cfg)
    points = m["points"]
    if not isinstance(points, int) or isinstance(points, bool) or points < 1:
        raise ValueError("model.points must be a positive integer")
    subdir = Path(cfg["data"].get("prepared_subdir", "prepared_balanced_v2"))
    if subdir.is_absolute() or ".." in subdir.parts or subdir == Path("."):
        raise ValueError("data.prepared_subdir must be a nonempty relative directory below data-root")
    if m["width"] % m["heads"]:
        raise ValueError("Model width must be divisible by attention heads")
    if not 0 < m["execute"] <= m["horizon"]:
        raise ValueError("Execute horizon must be in [1, action horizon]")
    if t["effective_batch"] < 1 or t["steps"] < 1:
        raise ValueError("Invalid training budget")
    schedule_steps = t.get("schedule_steps", t["steps"])
    if not isinstance(schedule_steps, int) or isinstance(schedule_steps, bool) or schedule_steps < t["steps"]:
        raise ValueError("training.schedule_steps must be an integer at least training.steps")
    for key, default in (
        ("microbatch", 16),
        ("encoder_block_batch_size", 32),
        ("prefetch_factor", 1),
    ):
        value = t.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    workers = t.get("workers", 0)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise ValueError("training.workers must be a nonnegative integer")
    if t.get("length_bucket_size", 1) != 1:
        raise ValueError(
            "training.length_bucket_size must be 1; ordered length buckets are no longer supported"
        )
    if cfg["data"].get("synthetic") and not cfg.get("debug", False):
        raise ValueError("Synthetic data is restricted to explicitly marked debug configurations")
    from stmem.memory.hierarchy import maximum_root_count

    # Checked again against actual archives at every read.
    required = maximum_root_count(m["max_execution_frames"], m["max_conditioning_frames"], m["block_size"])
    if min(m["budgets"]) < required:
        raise ValueError(f"Budget {min(m['budgets'])} cannot cover worst-case {required} roots")
