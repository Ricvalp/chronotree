"""Exclusive preparation writers and conservative metadata/layout reuse checks.

Reuse verification intentionally does not checksum prepared array contents or
read point-cloud chunks. The caller must freshly hash the raw input and verify
against its download manifest when an extracted-file digest is available. This
check detects stale preprocessing, missing episodes, incomplete metadata and
malformed layout, not arbitrary data bit rot.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import re

import h5py
import numpy as np


IDENTITY_FIELDS = frozenset(
    {
        "format",
        "task",
        "dataset_revision",
        "raw_sha256",
        "points",
        "crop",
        "voxel_size",
        "max_depth_m",
        "calibration",
        "clock",
        "physical_continuity_verified",
        "nominal_control_hz",
        "alignment",
    }
)
_REPORT_FIELDS = frozenset({"raw_path", "episodes", "elapsed_seconds", "prepared_bytes", "output"})


@contextmanager
def preparation_lock(output: Path):
    """Hold a nonblocking exclusive writer lock for a canonical output path.

    The sibling ``<output.name>.lock`` remains after release. Removing it would
    let a second process lock a different inode while another writer still held
    the original lock. A crashed process releases its OS lock automatically.
    """
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.with_name(output.name + ".lock")
    with lock_path.open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another preparation writer holds the output lock: {lock_path}") from error
        try:
            yield lock_path
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value if isinstance(value, str) else None


def _json_value(value):
    # Tuple/list equivalence is serialization, not approximate numeric matching.
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _layout_matches(group, name, shape, dtype):
    if not isinstance(group.get(name, getlink=True), h5py.HardLink):
        return False
    dataset = group.get(name)
    return isinstance(dataset, h5py.Dataset) and dataset.shape == shape and dataset.dtype == np.dtype(dtype)


def _episode_matches(group, key, frames, points, task, report):
    if not isinstance(group, h5py.Group):
        return False
    if (
        _text(group.attrs.get("task")) != task
        or _text(group.attrs.get("episode_id")) != key
        or _text(group.attrs.get("goal")) is None
    ):
        return False
    layouts = {
        "points": ((frames, points, 6), "f4"),
        "valid": ((frames, points), "?"),
        "proprio": ((frames, 15), "f4"),
        "actions": ((frames, 8), "f4"),
        "action_valid": ((frames, 8), "?"),
        "source": ((frames,), "?"),
        "completed": ((frames,), "?"),
        "raw_index": ((frames,), "i8"),
        "source_index": ((frames,), "i8"),
    }
    if not all(_layout_matches(group, name, *layout) for name, layout in layouts.items()):
        return False
    if not isinstance(report, dict) or report.get("episode_id") != key:
        return False
    for name in ("conditioning_frames", "execution_frames", "terminal_suffix_records", "seed"):
        if not _integer(report.get(name)):
            return False
    if not isinstance(report.get("difficulty"), str) or not isinstance(
        report.get("raw_indices_contiguous"), bool
    ):
        return False
    if "first_completed_raw_index" not in report:
        return False
    first_complete = report["first_completed_raw_index"]
    if first_complete is not None and not _integer(first_complete):
        return False
    # These are small per-frame index/boolean arrays, never RGB-D or robot payloads.
    source = group["source"][()]
    completed = group["completed"][()]
    raw_index = group["raw_index"][()]
    source_index = group["source_index"][()]
    conditioning = int(source.sum())
    if report["conditioning_frames"] != conditioning or report["execution_frames"] != frames - conditioning:
        return False
    if np.any(source[1:] & ~source[:-1]) or np.any(raw_index < 0) or np.any(np.diff(raw_index) <= 0):
        return False
    if not np.array_equal(source_index[source], np.arange(conditioning, dtype=np.int64)):
        return False
    if not np.array_equal(source_index[~source], np.arange(frames - conditioning, dtype=np.int64)):
        return False
    completed_rows = np.flatnonzero(completed)
    if first_complete != (int(raw_index[completed_rows[0]]) if len(completed_rows) else None):
        return False
    if report["terminal_suffix_records"] != max(0, len(completed_rows) - 1):
        return False
    return report["raw_indices_contiguous"] == bool(np.all(np.diff(raw_index) == 1))


def verify_prepared(output: Path, expected_manifest: dict, episode_layout: dict[str, int]) -> dict | None:
    """Return the embedded report only for the exact requested preparation.

    Missing, stale, malformed or unreadable output returns None without modifying
    it. The requested episode keys must match exactly; a small diagnostic cannot
    satisfy a full-data request, nor can a superset silently widen a subset run.
    ``raw_path`` is excluded so a verified dataset can be relocated. Sidecars are
    not consulted: a caller may recreate one from the returned embedded report.
    """
    if not isinstance(expected_manifest, dict) or not IDENTITY_FIELDS <= expected_manifest.keys():
        raise ValueError("Expected manifest must provide every preparation identity field")
    if (
        not isinstance(episode_layout, dict)
        or not episode_layout
        or any(
            not isinstance(key, str)
            or not re.fullmatch(r"episode_\d+", key)
            or not _integer(count)
            or count < 1
            for key, count in episode_layout.items()
        )
    ):
        raise ValueError("episode_layout must map requested episode_N keys to positive frame counts")
    points = expected_manifest["points"]
    if not _integer(points) or points < 1:
        raise ValueError("Expected point count must be a positive integer")
    try:
        with h5py.File(Path(output), "r") as prepared:
            report = json.loads(prepared.attrs["manifest"])
            if not isinstance(report, dict):
                return None
            for key, value in expected_manifest.items():
                if key not in _REPORT_FIELDS and (
                    key not in report or _json_value(report[key]) != _json_value(value)
                ):
                    return None
            if set(prepared.keys()) != set(episode_layout):
                return None
            episodes = report.get("episodes")
            if not isinstance(episodes, list) or len(episodes) != len(episode_layout):
                return None
            if not all(
                isinstance(item, dict) and isinstance(item.get("episode_id"), str) for item in episodes
            ):
                return None
            episode_reports = {item["episode_id"]: item for item in episodes}
            if len(episode_reports) != len(episodes) or set(episode_reports) != set(episode_layout):
                return None
            for key, frames in episode_layout.items():
                if not isinstance(prepared.get(key, getlink=True), h5py.HardLink):
                    return None
                if not _episode_matches(
                    prepared[key], key, frames, points, report["task"], episode_reports[key]
                ):
                    return None
            return report
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, UnicodeError):
        return None
