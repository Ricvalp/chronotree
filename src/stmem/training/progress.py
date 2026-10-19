"""Optional wall-clock heartbeats; the worker never accesses Torch or W&B."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time

from .logging_options import positive_progress_seconds


_STDOUT_LOCK = threading.Lock()


def print_json(record):
    # A single serialized write keeps heartbeat and main-thread JSON lines apart.
    line = json.dumps(record) + "\n"
    with _STDOUT_LOCK:
        sys.stdout.write(line)
        sys.stdout.flush()


class TrainingProgress:
    """Snapshot host-side progress while an example or optimizer update is slow.

    Only the main thread changes phases/counters. The reporting thread reads
    Python scalars and a monotonic clock; it performs no CUDA queries, tensor
    reads, collectives, RNG operations or calls to the W&B client. Phase times
    describe where the host is working/waiting, not CUDA kernel profiling.
    """

    def __init__(self, directory, interval_seconds, *, rank=0):
        self.directory = Path(directory)
        self.interval = None if interval_seconds is None else positive_progress_seconds(interval_seconds)
        self.enabled = self.interval is not None and rank == 0
        self.rank = rank
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._stream = None
        self._error = None
        self._state = {
            "phase": "initializing",
            "optimizer_steps_completed": None,
            "optimizer_step_in_progress": None,
            "examples_completed_rank": 0,
        }
        self._started = self._phase_started = time.perf_counter()
        self._update_started = None

    def __enter__(self):
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._stream = (self.directory / "progress.jsonl").open("a")
            self._thread = threading.Thread(target=self._run, name="stmem-progress", daemon=True)
            self._thread.start()
        return self

    def _check_error(self):
        if self._error is not None:
            raise RuntimeError("Training progress reporting failed") from self._error

    def update(self, phase, **fields):
        if not self.enabled:
            return
        self._check_error()
        with self._lock:
            self._state.update(phase=phase, **fields)
            self._phase_started = time.perf_counter()

    def begin_update(self, completed_step, examples_per_update_rank):
        if not self.enabled:
            return
        self._check_error()
        with self._lock:
            self._update_started = self._phase_started = time.perf_counter()
            self._state.update(
                phase="update_start",
                optimizer_steps_completed=completed_step,
                optimizer_step_in_progress=completed_step + 1,
                examples_completed_rank=0,
                examples_per_update_rank=examples_per_update_rank,
                example_in_update_rank=None,
                global_sample_position=None,
                task=None,
                episode_id=None,
                decision_index=None,
                conditioning_frames=None,
                execution_history_frames=None,
            )

    def finish_update(self, completed_step):
        if not self.enabled:
            return
        self._check_error()
        with self._lock:
            self._update_started = None
            self._phase_started = time.perf_counter()
            self._state.update(
                phase="update_complete",
                optimizer_steps_completed=completed_step,
                optimizer_step_in_progress=None,
            )

    def _emit(self):
        with self._lock:
            now = time.perf_counter()
            record = {
                "event": "training_progress",
                "rank": self.rank,
                **self._state,
                "elapsed_seconds": now - self._started,
                "phase_seconds": now - self._phase_started,
            }
            if self._update_started is not None:
                record["update_seconds"] = now - self._update_started
        self._stream.write(json.dumps(record) + "\n")
        self._stream.flush()
        print_json(record)

    def _run(self):
        try:
            self._emit()
            while not self._stop.wait(self.interval):
                self._emit()
        except Exception as error:
            self._error = error

    def __exit__(self, error_type, error, traceback):
        if self.enabled:
            self._stop.set()
            self._thread.join()
            try:
                try:
                    if error_type is not None:
                        self._state.update(phase="failed", error_type=error_type.__name__)
                    if self._error is None:
                        self._emit()
                    if error_type is None:
                        self._check_error()
                finally:
                    self._stream.close()
            except Exception:
                # A final flush/close failure must not hide the training error.
                if error_type is None:
                    raise
