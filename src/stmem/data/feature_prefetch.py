"""Bounded rank-local descriptor reads and overlapped CUDA transfers.

Only immutable pre-projection descriptors are prefetched. Their order, values,
padding and gradients through the learned encoder are unchanged. CPU slots are
reused only after their copy event completes; CUDA outputs are never reused
because autograd may retain them until backward.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import queue
import threading
import time

import torch

from .feature_cache import FEATURE_DIM, FeatureRowReader


DEFAULT_FEATURE_PREFETCH_CHUNKS = 2
_LOCK = threading.Lock()
_METRICS = {}
_TRANSFERS = []
_COLLECT_METRICS = False


def runtime_feature_io_settings():
    value = os.environ.get("STMEM_FEATURE_PREFETCH_CHUNKS", str(DEFAULT_FEATURE_PREFETCH_CHUNKS))
    try:
        depth = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("STMEM_FEATURE_PREFETCH_CHUNKS must be a nonnegative integer") from exc
    if depth < 0:
        raise ValueError("STMEM_FEATURE_PREFETCH_CHUNKS must be a nonnegative integer")
    return {
        "prefetch_chunks": depth,
        "pinned_host_buffers": depth > 0,
        "asynchronous_transfer": depth > 0,
    }


def reset_feature_io_metrics():
    """Enable/start update accounting after prior pipelines close.

    Callers which never enable metrics (ordinary inference) retain no timing
    events. Training resets and snapshots once per optimizer update.
    """
    global _COLLECT_METRICS
    with _LOCK:
        _COLLECT_METRICS = True
        _METRICS.clear()
        _METRICS.update(
            read_seconds=0.0, wait_seconds=0.0, transfer_seconds=0.0, bytes=0, transfer_bytes=0, chunks=0
        )
        _TRANSFERS.clear()


def _add(**values):
    if not _COLLECT_METRICS:
        return
    with _LOCK:
        for key, value in values.items():
            _METRICS[key] = _METRICS.get(key, 0) + value


def feature_io_metrics(synchronize=True):
    """Report host reads/waits and actual CUDA copy-event durations.

    These phases overlap and must not be added as an update wall-time estimate.
    With ``synchronize=False``, unfinished transfer events remain pending.
    """
    with _LOCK:
        remaining = []
        for start, end in _TRANSFERS:
            if synchronize:
                end.synchronize()
            elif not end.query():
                remaining.append((start, end))
                continue
            _METRICS["transfer_seconds"] = _METRICS.get("transfer_seconds", 0.0) + (
                start.elapsed_time(end) / 1000
            )
        _TRANSFERS[:] = remaining
        return {
            key: _METRICS.get(key, 0)
            for key in (
                "read_seconds",
                "wait_seconds",
                "transfer_seconds",
                "bytes",
                "transfer_bytes",
                "chunks",
            )
        }


@dataclass
class _Ready:
    tensor: torch.Tensor
    event: object = None


@dataclass
class _Failure:
    error: BaseException


_END = object()


class DescriptorPrefetch:
    """Ordered chunks with at most ``depth`` queued/producing future chunks.

    For CUDA, pinned storage is ``min(depth, chunks) * max_rows * points * 512``
    bytes. The producer uses one dedicated copy stream. GPU results have their
    own lifetime and can remain live for backward, just as in the synchronous
    implementation. No CUDA is used in DataLoader worker processes.

    CPU input tensors are immutable for the context's lifetime. CUDA-resident
    source descriptors use the synchronous path to preserve their originating
    stream dependencies. Closing drains only in-flight work, discards queued
    results, and joins the producer, including when the model raises an error.
    """

    def __init__(self, groups, *, point_count, device="cpu", depth=None):
        self.groups = groups
        self.point_count = point_count
        if point_count < 0:
            raise ValueError("point_count cannot be negative")
        self.device = torch.device(device)
        self.depth = runtime_feature_io_settings()["prefetch_chunks"] if depth is None else depth
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("Descriptor prefetch depth must be a nonnegative integer")
        if any(
            isinstance(row, torch.Tensor) and row.device.type != "cpu" for group in groups for row in group
        ):
            self.depth = 0
        self._reader = None
        self._thread = None
        self._closed = False
        self._started = False
        self._index = 0
        self._stop = threading.Event()
        self._buffers = []
        self._slot_events = []
        self._stream = None
        self._capacity = min(self.depth, len(groups))
        self._queue = queue.Queue(maxsize=max(1, self._capacity))
        self._credits = threading.Semaphore(self._capacity)

    def __enter__(self):
        if self._started or self._closed:
            raise RuntimeError("Descriptor prefetch can only be entered once")
        self._started = True
        if self.device.type == "cuda" and torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("Descriptor CUDA prefetch must run in the training rank, not a loader worker")
        if not self._capacity:
            self._reader = FeatureRowReader()
            return self
        if self.device.type == "cuda":
            self._stream = torch.cuda.Stream(device=self.device)
            shape = (max(map(len, self.groups)), self.point_count, FEATURE_DIM)
            self._buffers = [
                torch.empty(shape, dtype=torch.float16, pin_memory=True) for _ in range(self._capacity)
            ]
            self._slot_events = [None] * self._capacity
        self._thread = threading.Thread(target=self._produce, name="stmem-feature-prefetch", daemon=True)
        self._thread.start()
        return self

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def _acquire(self):
        while not self._stop.is_set():
            if self._credits.acquire(timeout=0.05):
                return not self._stop.is_set()
        return False

    def _read(self, reader, rows, out=None):
        started = time.perf_counter()
        tensor = reader.read(rows, point_count=self.point_count, out=out)
        _add(
            read_seconds=time.perf_counter() - started,
            bytes=sum(row.shape[0] * FEATURE_DIM * 2 for row in rows if row is not None),
            chunks=1,
        )
        return tensor

    def _produce(self):
        try:
            with FeatureRowReader() as reader:
                for index, rows in enumerate(self.groups):
                    if not self._acquire():
                        return
                    if self._stream is None:
                        item = _Ready(self._read(reader, rows))
                    else:
                        slot = index % self._capacity
                        event = self._slot_events[slot]
                        if event is not None:
                            event.synchronize()  # CPU source must remain unchanged until H2D completes.
                        host = self._read(reader, rows, self._buffers[slot][: len(rows)])
                        if self._stop.is_set():
                            return
                        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
                            begin = torch.cuda.Event(enable_timing=True) if _COLLECT_METRICS else None
                            end = torch.cuda.Event(enable_timing=begin is not None)
                            if begin is not None:
                                begin.record(self._stream)
                            result = host.to(self.device, non_blocking=True)
                            end.record(self._stream)
                        self._slot_events[slot] = end
                        if begin is not None:
                            with _LOCK:
                                _TRANSFERS.append((begin, end))
                        _add(transfer_bytes=host.numel() * host.element_size())
                        item = _Ready(result, end)
                    if not self._put(item):
                        return
                self._put(_END)
        except BaseException as error:
            self._put(_Failure(error))

    def __iter__(self):
        return self

    def __next__(self):
        if not self._started or self._closed:
            raise RuntimeError("Descriptor prefetch is not active")
        if self._index == len(self.groups):
            raise StopIteration
        if not self._capacity:
            rows = self.groups[self._index]
            self._index += 1
            host = self._read(self._reader, rows)
            if self.device.type != "cpu":
                begin = torch.cuda.Event(enable_timing=True) if _COLLECT_METRICS else None
                end = torch.cuda.Event(enable_timing=True) if begin is not None else None
                if begin is not None:
                    begin.record(torch.cuda.current_stream(self.device))
                result = host.to(self.device)
                if begin is not None:
                    end.record(torch.cuda.current_stream(self.device))
                    with _LOCK:
                        _TRANSFERS.append((begin, end))
                _add(transfer_bytes=host.numel() * host.element_size())
                return result
            return host
        started = time.perf_counter()
        item = self._queue.get()
        _add(wait_seconds=time.perf_counter() - started)
        self._credits.release()
        if isinstance(item, _Failure):
            self.close()
            raise item.error
        if item is _END:
            raise RuntimeError("Descriptor producer stopped before all chunks were read")
        self._index += 1
        if item.event is not None:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(item.event)
            item.tensor.record_stream(stream)
        return item.tensor

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        # A producer may have scheduled a transfer just before cancellation.
        if self._stream is not None:
            self._stream.synchronize()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._buffers.clear()
        self._slot_events.clear()
        if self._reader is not None:
            self._reader.close()

    def __exit__(self, *_exc):
        self.close()
