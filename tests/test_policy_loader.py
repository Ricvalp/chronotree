"""CPU invariants for bounded multiprocessing, ordering, retries and failures."""

from dataclasses import fields, is_dataclass
from itertools import islice
import os
from pathlib import Path
import time

import pytest
import torch

from stmem.training.data import TrainingData, rank_positions
from stmem.training.loader import PolicyBatchLoader, _Microbatches
from test_batched_training import _config
from test_training_startup import _fixture_config


class ObservedTrainingData(TrainingData):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.parent_pid = os.getpid()

    def sample_positions(self, positions):
        marker = Path(self.cfg["paths"]["run_root"]) / str(positions[0])
        marker.write_text(str(os.getpid()))
        if self.cfg.get("fail_worker") and os.getpid() != self.parent_pid:
            raise ValueError("diagnostic worker read failure")
        return super().sample_positions(positions)


def _loader(data, *, steps=4, retry=False, start=0):
    return PolicyBatchLoader(
        data,
        start=start,
        steps=steps,
        batch=5,
        rank=0,
        world=1,
        microbatch=2,
        retain_for_retry=retry,
    )


def _assert_packed_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif is_dataclass(left):
        assert type(left) is type(right)
        for field in fields(left):
            _assert_packed_equal(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_packed_equal(a, b)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_packed_equal(left[key], right[key])
    else:
        assert left == right


@pytest.mark.parametrize("steps", [4, None])
def test_workers_persist_prefetch_is_bounded_and_overflow_reuses_same_batches(tmp_path, steps):
    cfg = _config(tmp_path)
    cfg["training"].update(workers=2, prefetch_factor=1)
    data = ObservedTrainingData(cfg)
    with _loader(data, steps=steps, retry=True) as loader:
        workers = list(loader._iterator._workers)
        deadline = time.monotonic() + 20
        while len(list(tmp_path.iterdir())) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        # Without a consumer, only two microbatches can start. Finished work
        # cannot refill an unbounded ready queue.
        assert {p.name for p in tmp_path.iterdir()} == {"0", "2"}
        assert loader.prefetch_capacity == 2
        first = list(loader.read_update(0))
        retried = list(loader.read_update(0))
        assert all(a is b for a, b in zip(first, retried))
        second = list(loader.read_update(1))
        assert [p for item in second for p in item.positions] == rank_positions(1, 5, 0, 1)
        pids = {item.worker_pid for item in first + second}
        assert pids == {worker.pid for worker in workers}
        assert os.getpid() not in pids
        assert all(worker.is_alive() for worker in workers)
    assert all(not worker.is_alive() for worker in workers)
    assert not loader._retry_batches
    # Six consumed microbatches plus the two bounded prefetched ones; cleanup
    # must not ask an open-ended sampler for further work while draining.
    assert len(list(tmp_path.iterdir())) <= 8


@pytest.mark.parametrize("steps", [4, None])
def test_worker_exception_propagates_and_other_workers_are_joined(tmp_path, steps):
    cfg = _config(tmp_path)
    cfg["training"].update(workers=2, prefetch_factor=1)
    cfg["fail_worker"] = True
    data = ObservedTrainingData(cfg)
    with pytest.raises(ValueError, match="diagnostic worker read failure"):
        with _loader(data, steps=steps) as loader:
            workers = list(loader._iterator._workers)
            next(loader.read_update(0))
    assert all(not worker.is_alive() for worker in workers)


@pytest.mark.parametrize("alignment", ["pre_action_v1", "post_action_v1"])
def test_hdf5_workers_match_synchronous_packing_and_resume_position(tmp_path, alignment):
    # This fixture includes interleaved sources, command gaps, unavailable
    # gripper commands and a terminal suffix. It is not benchmark data.
    cfg = _fixture_config(tmp_path, alignment=alignment)
    data = TrainingData(cfg)
    with _loader(data, start=1, steps=3) as loader:
        expected = [item for step in (1, 2) for item in loader.read_update(step)]
    cfg["training"].update(workers=2, prefetch_factor=1)
    with _loader(data, start=1, steps=3) as loader:
        actual = [item for step in (1, 2) for item in loader.read_update(step)]
    for a, b in zip(expected, actual):
        assert a.positions == b.positions
        _assert_packed_equal(a.packed, b.packed)
        assert b.worker_pid != os.getpid()
        assert b.load_seconds >= 0 and b.pack_seconds >= 0


def test_finished_checkpoint_starts_no_workers_and_invalid_order_is_rejected(tmp_path):
    cfg = _config(tmp_path)
    cfg["training"].update(workers=2, prefetch_factor=1)
    data = TrainingData(cfg)
    with _loader(data, start=3, steps=3) as loader:
        assert loader._iterator is None
    cfg["training"]["workers"] = 0
    with _loader(data) as loader:
        with pytest.raises(ValueError, match="consecutive"):
            list(loader.read_update(1))
        list(loader.read_update(0))
        with pytest.raises(ValueError, match="consecutive"):
            list(loader.read_update(0))


def test_open_ended_sampler_has_no_finite_length_and_preserves_rotating_rank_coverage():
    # Cross the production 100k boundary without loading data. No large finite
    # sentinel may cap the stream or change rank ownership after this boundary.
    by_rank = []
    for rank in range(4):
        sampler = _Microbatches(99999, None, 12, rank, 4, 2)
        with pytest.raises(TypeError, match="no finite length"):
            len(sampler)
        batches = list(islice(sampler, 6))
        expected = []
        for step in range(99999, 100002):
            positions = rank_positions(step, 12, rank, 4)
            expected.extend([tuple(positions[:2]), tuple(positions[2:])])
        assert batches == expected
        by_rank.append(batches)
    for offset, step in enumerate(range(99999, 100002)):
        combined = [
            position
            for batches in by_rank
            for batch in batches[offset * 2 : (offset + 1) * 2]
            for position in batch
        ]
        assert sorted(combined) == list(range(step * 12, (step + 1) * 12))


@pytest.mark.parametrize("workers", [0, 2])
def test_open_ended_loader_matches_finite_data_and_resumes_beyond_configured_budget(tmp_path, workers):
    # The declared schedule ends at three, but an open-ended sampler must
    # preserve the same data after that point, including after discarding its
    # prefetched future updates on shutdown and reconstructing the loader.
    cfg = _config(tmp_path, steps=3)
    # Use the same center quota in current/history frames, so optional cached
    # center counts do not differ with worker-local episode-cache warmth.
    cfg["model"]["current_tokens"] = 1
    cfg["training"].update(workers=0, prefetch_factor=1)
    data = TrainingData(cfg)
    with _loader(data, start=1, steps=5) as loader:
        expected = {step: list(loader.read_update(step)) for step in range(1, 5)}
    cfg["training"]["workers"] = workers
    rng = torch.get_rng_state().clone()
    with _loader(data, start=1, steps=None, retry=True) as loader:
        processes = list(loader._iterator._workers) if workers else []
        for step in range(1, 5):
            actual = list(loader.read_update(step))
            for a, b in zip(expected[step], actual):
                assert a.positions == b.positions
                _assert_packed_equal(a.packed, b.packed)
            retried = list(loader.read_update(step))
            assert all(a is b for a, b in zip(actual, retried))
    assert all(not process.is_alive() for process in processes)
    assert torch.equal(torch.get_rng_state(), rng)
    with _loader(data, start=4, steps=None) as resumed:
        actual = list(resumed.read_update(4))
        for a, b in zip(expected[4], actual):
            assert a.positions == b.positions
            _assert_packed_equal(a.packed, b.packed)
        assert resumed._retry_batches == []
    assert torch.equal(torch.get_rng_state(), rng)
