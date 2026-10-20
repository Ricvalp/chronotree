"""Ordered, bounded CPU prefetch for policy training.

Only raw observations and their packed CPU tensors enter workers. Sampling uses
the already validated action-only index, so workers never rebuild that index,
initialize CUDA, or access the model/tracker. The dedicated loader RNG does not
consume training RNG state when workers start or a checkpoint is resumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import time

import torch
from torch.utils.data import DataLoader, Dataset

from stmem.models.batched_policy import PackedPolicyBatch, pack_policy_batch
from stmem.training.data import rank_positions


@dataclass
class LoadedBatch:
    positions: tuple[int, ...]
    packed: PackedPolicyBatch
    load_seconds: float
    pack_seconds: float
    worker_pid: int


class _PolicyDataset(Dataset):
    def __init__(self, data, block_size):
        # TrainingData contains paths and index metadata, never live HDF5
        # handles. Its method LRU caches are not serialized into spawn workers.
        self.data = data
        self.block_size = block_size

    def __getitem__(self, positions):
        started = time.perf_counter()
        decisions = self.data.sample_positions(positions)
        loaded = time.perf_counter()
        packed = pack_policy_batch(decisions, self.block_size)
        return LoadedBatch(
            tuple(positions), packed, loaded - started, time.perf_counter() - loaded, os.getpid()
        )


class _Microbatches:
    def __init__(self, start, steps, batch, rank, world, microbatch):
        self.start, self.steps = start, steps
        self.batch, self.rank, self.world = batch, rank, world
        self.microbatch = microbatch

    def __iter__(self):
        for step in range(self.start, self.steps):
            positions = rank_positions(step, self.batch, self.rank, self.world)
            for offset in range(0, len(positions), self.microbatch):
                yield tuple(positions[offset : offset + self.microbatch])

    def __len__(self):
        local_n = self.batch // self.world
        return (self.steps - self.start) * ((local_n + self.microbatch - 1) // self.microbatch)


def _identity(value):
    return value


def _initialize_worker(_worker_id):
    # DataLoader also sets Torch threads to one. Be explicit: each rank has its
    # own workers, and the Slurm CPU allocation is shared across all ranks.
    torch.set_num_threads(1)


class PolicyBatchLoader:
    """One persistent loader per rank, yielding the original sampler positions.

    At most ``workers * prefetch_factor`` microbatches are queued or being
    prepared, in addition to the consumer's batch. FP16 retains the current
    optimizer update's CPU batches for overflow retries; other precisions do not.
    Prefetched future updates are disposable, never checkpointed sampler state.
    """

    def __init__(self, data, *, start, steps, batch, rank, world, microbatch, retain_for_retry=False):
        self.data = data
        self.start, self.steps = start, steps
        self.batch, self.rank, self.world = batch, rank, world
        self.microbatch = microbatch
        self.workers = data.cfg["training"].get("workers", 0)
        self.prefetch_factor = data.cfg["training"].get("prefetch_factor", 1)
        self.retain_for_retry = retain_for_retry
        self._loader = None
        self._iterator = None
        self._next_step = start
        self._last_step = None
        self._retry_batches = []
        self._update_complete = True
        self._closed = False

    @property
    def prefetch_capacity(self):
        return self.workers * self.prefetch_factor

    def __enter__(self):
        if self._closed or self._loader is not None:
            raise RuntimeError("A policy loader can only be started once")
        sampler = _Microbatches(self.start, self.steps, self.batch, self.rank, self.world, self.microbatch)
        kwargs = {}
        if self.workers:
            kwargs.update(
                multiprocessing_context="spawn",
                persistent_workers=True,
                prefetch_factor=self.prefetch_factor,
                worker_init_fn=_initialize_worker,
            )
        generator = torch.Generator().manual_seed(self.data.cfg["seed"] + self.rank)
        self._loader = DataLoader(
            _PolicyDataset(self.data, self.data.cfg["model"]["block_size"]),
            batch_size=None,
            sampler=sampler,
            collate_fn=_identity,
            num_workers=self.workers,
            generator=generator,
            in_order=True,
            **kwargs,
        )
        # Avoid starting processes for a checkpoint already at the final step.
        if len(sampler):
            try:
                self._iterator = iter(self._loader)
            except BaseException:
                self.close()
                raise
        return self

    def read_update(self, step):
        """Yield one update in order, or replay its CPU batches after overflow."""
        if self._closed or self._iterator is None:
            raise RuntimeError("Policy loader is not active")
        if step == self._last_step and self._update_complete and self.retain_for_retry:
            yield from self._retry_batches
            return
        if step != self._next_step or not self._update_complete:
            raise ValueError("Policy batches must be consumed at consecutive optimizer boundaries")
        self._last_step = step
        self._update_complete = False
        self._retry_batches = []
        positions = rank_positions(step, self.batch, self.rank, self.world)
        for offset in range(0, len(positions), self.microbatch):
            prepared = next(self._iterator)
            if prepared.positions != tuple(positions[offset : offset + self.microbatch]):
                raise RuntimeError("Prefetch changed the deterministic training sample order")
            if self.retain_for_retry:
                self._retry_batches.append(prepared)
            if offset + self.microbatch >= len(positions):
                # Set this before yielding the last batch: callers may consume
                # exactly N batches without requesting the StopIteration.
                self._next_step = step + 1
                self._update_complete = True
            yield prepared

    def close(self):
        if self._closed:
            return
        self._closed = True
        iterator, self._iterator = self._iterator, None
        try:
            if self.workers and iterator is not None:
                # Torch 2.9.1 has no public explicit close for persistent
                # iterators. Use its shutdown protocol (also used by __del__)
                # now, including on safe-stop/errors, rather than waiting for GC.
                iterator._shutdown_workers()
                for worker in iterator._workers:
                    worker.join(timeout=5)
        finally:
            self._loader = None
            self._retry_batches.clear()

    def __exit__(self, *_exc):
        self.close()
