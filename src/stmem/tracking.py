"""Low-volume W&B telemetry, isolated from model RNG and inference timing.

W&B's service handles persistence/networking asynchronously. Only rank zero
creates a run; callers submit existing scalar metrics at their normal cadence.
The small append-only journal reserves monotonic W&B history indices across
offline restarts. Optimizer/episode progress is an independent, rewindable axis.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib.metadata
import json
import math
import numbers
import os
from pathlib import Path
import random
import secrets
import time

import numpy as np
import torch

from stmem.config import digest, source_revision
from stmem.training.checkpoints import atomic_json


_SECRET_KEYS = {"api_key", "wandb_api_key", "password", "secret", "access_token", "authorization"}


def _plain(value, key=""):
    """No tensor conversion, GPU synchronization, arbitrary object repr or secrets."""
    if key.lower() in _SECRET_KEYS:
        return "[redacted]"
    if isinstance(value, dict):
        return {str(k): _plain(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(value) else None
    raise TypeError(f"Tracking requires CPU scalars/JSON, received {type(value).__name__}")


@contextmanager
def _preserve_rng():
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    # Never initialize CUDA or touch an unassigned GPU just to initialize logging.
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


class RunTracker:
    """One logical run per output directory, with restartable local identity.

    Explicit online initialization failures are errors, never silently unlogged
    training. Offline mode (the default) creates real .wandb files for later sync.
    No model watch hooks, media, model uploads or system-polling thread is enabled.
    """

    def __init__(self, cfg, directory, *, job_type, name=None, metadata=None, rank=0, resume=False):
        self.cfg = cfg
        self.directory = Path(directory)
        self.job_type = job_type
        self.name = name or f"seed{cfg.get('seed', 0)}-{job_type}"
        self.metadata = metadata or {}
        self.resume = bool(resume)
        options = cfg.get("tracking", {})
        self.mode = os.environ.get("WANDB_MODE") or options.get("mode", "offline")
        if self.mode not in ("online", "offline", "disabled"):
            raise ValueError("WANDB_MODE must be online, offline or disabled")
        self.enabled = rank == 0 and options.get("enabled", True) and self.mode != "disabled"
        self.project = os.environ.get("WANDB_PROJECT") or options.get("project") or "chronotree"
        self.entity = os.environ.get("WANDB_ENTITY") or options.get("entity") or None
        self.storage = Path(os.environ.get("WANDB_DIR") or options.get("dir") or self.directory)
        self.identity = {}
        self.run = None
        self._closed = False
        self._history_step = 0
        self._journal = None
        self._summary = {}
        self._log_calls = 0
        self._log_seconds = 0.0

    def __enter__(self):
        if not self.enabled:
            return self
        self.directory.mkdir(parents=True, exist_ok=True)
        self.storage.mkdir(parents=True, exist_ok=True)
        path = self.directory / "tracking.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        if previous:
            for key, requested in (
                ("project", self.project),
                ("entity", self.entity),
                ("job_type", self.job_type),
            ):
                if previous[key] != requested:
                    raise ValueError(f"W&B {key} changed for an existing run; use a new run directory")
        self.identity = {
            "id": previous.get("id") or secrets.token_hex(8),
            "project": self.project,
            "entity": self.entity,
            "job_type": self.job_type,
            "name": self.name,
            "attempt": previous.get("attempt", 0) + 1,
            "mode": self.mode,
            "status": "initializing",
            "resume_requested": self.resume,
        }
        self._history_step = previous.get("next_history_step", 0)
        journal_path = self.directory / "tracking_history.jsonl"
        needs_newline = False
        if journal_path.exists():
            # Small scalar-only journal; tolerate a final interrupted write.
            with journal_path.open() as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._history_step = max(self._history_step, row["history_step"] + 1)
            if journal_path.stat().st_size:
                with journal_path.open("rb") as stream:
                    stream.seek(-1, os.SEEK_END)
                    needs_newline = stream.read(1) != b"\n"
        atomic_json(self.identity, path)
        started = time.perf_counter()
        try:
            with _preserve_rng():
                import wandb

                run_root = str(Path(self.cfg.get("paths", {}).get("run_root", self.directory)).resolve())
                provenance = {
                    "source": source_revision(),
                    "metadata": _plain(self.metadata),
                    "run_directory": str(self.directory.resolve()),
                    "logical_run_id": self.identity["id"],
                    "attempt": self.identity["attempt"],
                    "resume_requested": self.resume,
                    "synthetic_debug_only": bool(self.cfg.get("data", {}).get("synthetic", False)),
                    "debug": bool(self.cfg.get("debug", False)),
                    "wandb_version": importlib.metadata.version("wandb"),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                }
                self.run = wandb.init(
                    id=self.identity["id"],
                    project=self.project,
                    entity=self.entity,
                    name=self.name,
                    group=f"{Path(run_root).name}-{digest(run_root)[:8]}-seed{self.cfg.get('seed', 0)}",
                    job_type=self.job_type,
                    tags=[
                        "synthetic-debug"
                        if provenance["synthetic_debug_only"]
                        else "real-data-debug"
                        if provenance["debug"]
                        else "real-data",
                        self.job_type,
                    ],
                    config={**_plain(self.cfg), "chronotree": provenance},
                    allow_val_change=True,
                    dir=str(self.storage.resolve()),
                    mode=self.mode,
                    resume="allow" if self.mode == "online" else None,
                    settings=wandb.Settings(
                        console="off",
                        disable_git=True,
                        disable_code=True,
                        save_code=False,
                        x_disable_stats=True,
                        x_disable_meta=True,
                        x_save_requirements=False,
                        init_timeout=60,
                        login_timeout=30,
                        quiet=True,
                        reinit="finish_previous",
                        x_sync_dir_suffix=f"attempt{self.identity['attempt']}",
                    ),
                )
                self.run.define_metric("progress/step")
                self.run.define_metric("*", step_metric="progress/step", step_sync=False)
                self._history_step = max(self._history_step, self.run.step)
            self.identity.update(
                status="running",
                init_seconds=time.perf_counter() - started,
                wandb_directory=str(Path(self.run.dir).parent),
                url=self.run.url if self.mode == "online" else None,
            )
            self._journal = journal_path.open("a", encoding="utf-8")
            if needs_newline:
                # Separate an interrupted fragment (or complete unterminated row)
                # from the next record so a second restart can recover its index.
                self._journal.write("\n")
                self._journal.flush()
            atomic_json(self.identity, path)
            self.summary({"tracking/attempt": self.identity["attempt"], "tracking/status": "running"})
            return self
        except BaseException:
            self.identity["status"] = "initialization_failed"
            try:
                if self.run is not None:
                    with _preserve_rng():
                        self.run.finish(exit_code=1)
            except Exception as cleanup_error:
                self.identity["cleanup_error_type"] = type(cleanup_error).__name__
            finally:
                self._closed = True
                if self._journal is not None:
                    self._journal.close()
                atomic_json(self.identity, path)
            raise

    def log(self, values, step=None):
        if not self.enabled or self._closed:
            return
        started = time.perf_counter()
        row = _plain(values)
        row["progress/step"] = int(step) if step is not None else self._history_step
        row["tracking/attempt"] = self.identity["attempt"]
        # Reserve before publishing; network/disk work continues in W&B's service.
        self._journal.write(json.dumps({"history_step": self._history_step, "metrics": row}) + "\n")
        self._journal.flush()  # OS buffered; no per-record fsync on shared storage.
        self.run.log(row, step=self._history_step, commit=True)
        self._history_step += 1
        self._log_calls += 1
        self._log_seconds += time.perf_counter() - started

    def summary(self, values):
        if not self.enabled or self._closed:
            return
        plain = _plain(values)
        self._summary.update(plain)
        self.run.summary.update(plain)

    def checkpoint(self, path, step):
        if not self.enabled or self._closed:
            return
        path = Path(path)
        self.summary(
            {
                "checkpoint/path": str(path.resolve()),
                "checkpoint/step": int(step),
                "checkpoint/bytes": path.stat().st_size,
            }
        )

    def finish(self, status="completed", exit_code=0):
        if not self.enabled or self._closed:
            return
        self.summary(
            {
                "tracking/status": status,
                "tracking/log_calls": self._log_calls,
                "tracking/log_enqueue_seconds": self._log_seconds,
                "tracking/init_seconds": self.identity["init_seconds"],
            }
        )
        self.identity.update(status=status, next_history_step=self._history_step, summary=self._summary)
        try:
            with _preserve_rng():
                self.run.finish(exit_code=exit_code)
        finally:
            self._closed = True
            self._journal.close()
            atomic_json(self.identity, self.directory / "tracking.json")

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.summary({"tracking/error_type": exc_type.__name__})
        self.finish(status="failed" if exc_type is not None else "completed", exit_code=1 if exc_type else 0)
        return False
