"""Synchronized end-to-end and component timings, including first decisions."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
import os
from pathlib import Path
import resource
from typing import Iterator

import numpy as np
import torch


def process_memory() -> dict[str, int | None]:
    """Measured Linux RSS includes Python objects/allocators beyond logical payload."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        current = resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        current = None
    return {
        "cpu_process_rss_bytes": current,
        "cpu_process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }


def synchronize(device: torch.device | str | None = None) -> None:
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@dataclass
class DecisionTimer:
    device: torch.device | str | None = None
    components: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0

    def __enter__(self) -> "DecisionTimer":
        synchronize(self.device)
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        synchronize(self.device)
        self.seconds = time.perf_counter() - self._started

    @contextmanager
    def component(self, name: str) -> Iterator[None]:
        synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            self.components[name] = self.components.get(name, 0.0) + time.perf_counter() - started

    def as_dict(self) -> dict[str, object]:
        return {"decision_seconds": self.seconds, "component_seconds": dict(self.components)}


def latency_summary(seconds: list[float], warmup_decisions: int = 1) -> dict[str, object]:
    values = np.asarray(seconds, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Latency values must be finite nonnegative seconds")

    def describe(array: np.ndarray) -> dict[str, float | int | None]:
        return {
            "count": len(array),
            "p50_seconds": float(np.percentile(array, 50)) if len(array) else None,
            "p95_seconds": float(np.percentile(array, 95)) if len(array) else None,
            "total_seconds": float(array.sum()),
        }

    return {
        "all": describe(values),
        "cold": describe(values[:warmup_decisions]),
        "warm": describe(values[warmup_decisions:]),
        "warmup_decisions": warmup_decisions,
    }


def accelerator_memory(device: torch.device | str | None = None) -> dict[str, int]:
    if device is None or torch.device(device).type != "cuda":
        return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def allocated_gpu_hours(elapsed_seconds: float, allocated_gpu_count: int) -> float:
    if elapsed_seconds < 0 or allocated_gpu_count < 0:
        raise ValueError("Elapsed time and GPU allocation must be nonnegative")
    return elapsed_seconds * allocated_gpu_count / 3600
