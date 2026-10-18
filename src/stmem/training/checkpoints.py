"""Durable, single-writer checkpoints and exact random-state capture.

Call atomic_save only on rank zero. Distributed callers gather rank RNG and
sampler states before saving and restore their own rank on resume.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

import numpy as np
import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint has CUDA RNG states but CUDA is unavailable")
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; exact RNG resume is not possible")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_save(payload: Any, path: str | Path, keep_previous: bool = True) -> None:
    """Write on the destination filesystem; retain ``<name>.previous``.

    An interrupted serialization never replaces the existing checkpoint. The
    previous copy is replaced atomically before the new checkpoint is promoted.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    backup_temporary: Path | None = None
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if keep_previous and path.exists():
            fd, backup_name = tempfile.mkstemp(prefix=f".{path.name}.previous.", dir=path.parent)
            backup_temporary = Path(backup_name)
            with os.fdopen(fd, "wb") as destination, path.open("rb") as source:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(backup_temporary, path.with_name(path.name + ".previous"))
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        if backup_temporary is not None:
            backup_temporary.unlink(missing_ok=True)


def atomic_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        _sync_directory(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_copy(source: str | Path, destination: str | Path, *, immutable: bool = True) -> None:
    """Durably copy checkpoint bytes, retaining a stable content identity."""
    source, destination = Path(source), Path(destination)
    if destination.exists() and immutable:
        if file_sha256(source) != file_sha256(destination):
            raise FileExistsError(f"Immutable checkpoint already exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as original:
            shutil.copyfileobj(original, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, destination)
        _sync_directory(destination.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def mark_complete(
    path: str | Path,
    *,
    completed_steps: int,
    intended_steps: int,
    checkpoint: str | Path | None = None,
    **metadata: Any,
) -> None:
    """Completion is an explicit stage invariant, never a successful save."""
    if completed_steps != intended_steps:
        raise ValueError("Partial or overrun stage cannot be marked complete")
    record = dict(metadata, status="complete", completed_steps=completed_steps, intended_steps=intended_steps)
    if checkpoint is not None:
        record.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint))
    atomic_json(record, path)
