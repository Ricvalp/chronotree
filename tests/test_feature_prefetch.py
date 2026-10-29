"""Synthetic diagnostics for ordered, bounded descriptor I/O and gradient parity."""

from copy import deepcopy
from dataclasses import replace
import threading

import numpy as np
import pytest
import torch

from stmem.data.feature_cache import FeatureRow, FeatureRowReader, load_feature_rows
from stmem.data.feature_prefetch import (
    DEFAULT_FEATURE_PREFETCH_CHUNKS,
    DescriptorPrefetch,
    feature_io_metrics,
    reset_feature_io_metrics,
    runtime_feature_io_settings,
)
from stmem.models.batched_policy import pack_policy_batch
from stmem.models.policy import MemoryPolicy
from test_frozen_feature_policy import configuration, episode


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)
    reset_feature_io_metrics()


def rows_on_disk(tmp_path, count=9, points=7, signature="synthetic"):
    values = np.arange(count * points * 256, dtype=np.float32).reshape(count, points, 256)
    values = (values / 256).astype("<f2")
    path = tmp_path / "synthetic_descriptors.bin"
    path.write_bytes(values.tobytes())
    rows = [FeatureRow(str(path), "episode_0", i, points, signature, i * points * 512) for i in range(count)]
    return rows, torch.from_numpy(values.copy())


@pytest.mark.parametrize("depth", [0, 1, 2, 4])
def test_order_padding_owned_outputs_metrics_and_rng(tmp_path, depth):
    rows, values = rows_on_disk(tmp_path)
    groups = [[rows[5], None, rows[1]], [rows[0]], [rows[8], torch.ones(3, 256)], [rows[3]]]
    expected = [load_feature_rows(group, point_count=10) for group in groups]
    rng = torch.get_rng_state().clone()
    with DescriptorPrefetch(groups, point_count=10, depth=depth) as prefetch:
        outputs = list(prefetch)
    assert torch.equal(torch.get_rng_state(), rng)
    for output, reference in zip(outputs, expected):
        torch.testing.assert_close(output, reference, atol=0, rtol=0)
    assert len({output.data_ptr() for output in outputs}) == len(outputs)
    outputs[0].zero_()
    torch.testing.assert_close(outputs[1][0, :7], values[0], atol=0, rtol=0)
    metrics = feature_io_metrics()
    assert metrics["chunks"] == len(groups)
    assert metrics["bytes"] == (5 * 7 + 3) * 512
    assert metrics["read_seconds"] > 0
    assert metrics["transfer_bytes"] == metrics["transfer_seconds"] == 0
    assert metrics["wait_seconds"] >= 0
    reset_feature_io_metrics()
    assert all(value == 0 for value in feature_io_metrics().values())


def test_reader_reuses_mapping_without_exporting_cache_storage(tmp_path):
    rows, expected = rows_on_disk(tmp_path)
    out = torch.full((2, 10, 256), float("nan"), dtype=torch.float16)
    with FeatureRowReader() as reader:
        result = reader.read([rows[0], None], point_count=10, out=out)
        assert result.data_ptr() == out.data_ptr()
        mapping = reader.maps[rows[0].path]
        torch.testing.assert_close(result[0, :7], expected[0], atol=0, rtol=0)
        assert not result[0, 7:].any() and not result[1].any()
        result.zero_()
        again = reader.read([rows[0]])
        assert reader.maps[rows[0].path] is mapping
        torch.testing.assert_close(again[0], expected[0], atol=0, rtol=0)
    assert mapping._mmap.closed
    with pytest.raises(ValueError, match="exact shape"):
        FeatureRowReader().read([rows[0]], point_count=10, out=torch.zeros(1, 10, 256))


def test_queue_is_bounded_and_early_close_joins_producer(tmp_path, monkeypatch):
    rows, _ = rows_on_disk(tmp_path)
    original_read = FeatureRowReader.read
    read_count = 0
    condition = threading.Condition()
    mappings = []

    def monitored_read(reader, *args, **kwargs):
        nonlocal read_count
        value = original_read(reader, *args, **kwargs)
        with condition:
            read_count += 1
            mappings.extend(reader.maps.values())
            condition.notify_all()
        return value

    monkeypatch.setattr(FeatureRowReader, "read", monitored_read)
    prefetch = DescriptorPrefetch([[row] for row in rows], point_count=7, depth=2)
    with prefetch:
        with condition:
            assert condition.wait_for(lambda: read_count == 2, timeout=3)
            # Producer has exhausted its credits without consuming a chunk.
            assert not condition.wait_for(lambda: read_count > 2, timeout=0.05)
        next(prefetch)
        with condition:
            assert condition.wait_for(lambda: read_count == 3, timeout=3)
    assert not prefetch._thread.is_alive()
    assert read_count == 3
    assert mappings and all(mapping._mmap.closed for mapping in mappings)
    prefetch.close()  # Idempotent cleanup.


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_read_failure_propagates_and_closes_thread(tmp_path, depth):
    rows, _ = rows_on_disk(tmp_path)
    invalid = replace(rows[0], offset=10**9)
    prefetch = DescriptorPrefetch([[rows[0]], [invalid], [rows[1]]], point_count=7, depth=depth)
    with pytest.raises(ValueError, match="Truncated descriptor cache"):
        with prefetch:
            next(prefetch)
            next(prefetch)
    assert prefetch._thread is None or not prefetch._thread.is_alive()


def test_consumer_failure_cancels_bounded_producer(tmp_path):
    rows, _ = rows_on_disk(tmp_path)
    prefetch = DescriptorPrefetch([[row] for row in rows], point_count=7, depth=2)
    with pytest.raises(RuntimeError, match="consumer"):
        with prefetch:
            next(prefetch)
            raise RuntimeError("consumer failed during encoding")
    assert not prefetch._thread.is_alive()


def test_runtime_controls_and_cpu_path_never_initialize_cuda(monkeypatch):
    monkeypatch.delenv("STMEM_FEATURE_PREFETCH_CHUNKS", raising=False)
    assert runtime_feature_io_settings()["prefetch_chunks"] == DEFAULT_FEATURE_PREFETCH_CHUNKS == 2
    for invalid in ("-1", "1.5", "garbage"):
        monkeypatch.setenv("STMEM_FEATURE_PREFETCH_CHUNKS", invalid)
        with pytest.raises(ValueError, match="nonnegative integer"):
            runtime_feature_io_settings()
    monkeypatch.setenv("STMEM_FEATURE_PREFETCH_CHUNKS", "0")
    assert not runtime_feature_io_settings()["pinned_host_buffers"]

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU prefetch must not use CUDA")

    for name in ("is_available", "Stream", "Event", "current_stream", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    with DescriptorPrefetch([[torch.ones(3, 256)], [None]], point_count=3, depth=2) as prefetch:
        assert next(prefetch).sum() == 768
        assert not next(prefetch).any()
    assert feature_io_metrics()["transfer_seconds"] == 0


def test_uninstrumented_callers_retain_no_metrics(tmp_path, monkeypatch):
    from stmem.data import feature_prefetch

    monkeypatch.setattr(feature_prefetch, "_COLLECT_METRICS", False)
    rows, _ = rows_on_disk(tmp_path)
    with DescriptorPrefetch([[row] for row in rows], point_count=7, depth=2) as prefetch:
        list(prefetch)
    assert not feature_prefetch._TRANSFERS
    assert all(value == 0 for value in feature_io_metrics().values())


@pytest.mark.parametrize("depth", [0, 2])
def test_cuda_is_rejected_in_loader_worker_before_any_allocation(monkeypatch, depth):
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    with pytest.raises(RuntimeError, match="training rank"):
        with DescriptorPrefetch([[None]], point_count=2, depth=depth, device="cuda"):
            pass


def policy_inputs_on_disk(tmp_path, checkpoint=False):
    cfg = configuration("none")
    cfg["model"]["checkpoint_blocks"] = checkpoint
    episodes = [episode(cfg, seed=seed) for seed in (12, 43)]
    path = tmp_path / "policy_descriptors.bin"
    offset = 0
    with path.open("wb") as handle:
        for ep in episodes:
            for index, frame in enumerate(ep.conditioning + ep.execution):
                values = frame.visual_features
                handle.write(values.numpy().tobytes())
                frame.visual_features = FeatureRow(
                    str(path), "synthetic", index, len(frame.points), frame.visual_feature_signature, offset
                )
                offset += values.numel() * values.element_size()
    decisions = [episodes[0].decision(1, 4), episodes[1].decision(6, 4)]
    return cfg, pack_policy_batch(decisions, cfg["model"]["block_size"])


@pytest.mark.parametrize("checkpoint", [False, True])
def test_packed_history_current_outputs_and_gradients_are_identical(tmp_path, monkeypatch, checkpoint):
    cfg, packed = policy_inputs_on_disk(tmp_path, checkpoint)
    torch.manual_seed(874)
    first = MemoryPolicy(cfg)
    second = deepcopy(first)
    noisy = torch.randn(2, 4, 8)
    time = torch.tensor([0.3, 0.7])
    results = []
    for depth, policy in ((0, first), (2, second)):
        monkeypatch.setenv("STMEM_FEATURE_PREFETCH_CHUNKS", str(depth))
        output = policy(packed.metadata, noisy, time, packed_batch=packed)
        output.square().sum().backward()
        results.append(output.detach())
    torch.testing.assert_close(results[1], results[0], atol=0, rtol=0)
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0, msg=name)
    assert feature_io_metrics()["chunks"] > 2  # Includes both history and current banks.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA device")
def test_cuda_reuses_pinned_slots_without_overwriting_gpu_outputs(tmp_path, monkeypatch):
    rows, expected = rows_on_disk(tmp_path, count=12, points=256)
    original_read = FeatureRowReader.read
    pointers = set()

    def record_slots(reader, *args, **kwargs):
        out = kwargs.get("out")
        assert out.is_pinned()
        pointers.add(out.data_ptr())
        return original_read(reader, *args, **kwargs)

    monkeypatch.setattr(FeatureRowReader, "read", record_slots)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with DescriptorPrefetch([[row, None] for row in rows], point_count=257, device="cuda", depth=2) as pf:
            outputs = list(pf)
            # These CUDA inputs all survive until backward; none can share a slot.
            weight = torch.ones(256, device="cuda", requires_grad=True)
            loss = sum((output.float() * weight).sum() for output in outputs)
            loss.backward()
    stream.synchronize()
    assert len(pointers) == 2
    assert len({output.data_ptr() for output in outputs}) == len(rows)
    for index, output in enumerate(outputs):
        torch.testing.assert_close(output[0, :256].cpu(), expected[index], atol=0, rtol=0)
        assert not output[1].any() and not output[0, 256:].any()
    torch.testing.assert_close(weight.grad.cpu(), expected.float().sum((0, 1)), atol=0.1, rtol=1e-6)
    metrics = feature_io_metrics()
    assert metrics["transfer_seconds"] > 0 and metrics["transfer_bytes"] == 12 * 2 * 257 * 512
    assert feature_io_metrics() == metrics  # Events are resolved once.


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA device")
@pytest.mark.parametrize("checkpoint", [False, True])
def test_cuda_packed_output_and_gradient_parity(tmp_path, monkeypatch, checkpoint):
    cfg, packed = policy_inputs_on_disk(tmp_path, checkpoint)
    torch.manual_seed(243)
    first = MemoryPolicy(cfg).cuda()
    second = deepcopy(first)
    noisy, time = torch.randn(2, 4, 8, device="cuda"), torch.tensor([0.3, 0.7], device="cuda")
    results = []
    for depth, policy in ((0, first), (2, second)):
        monkeypatch.setenv("STMEM_FEATURE_PREFETCH_CHUNKS", str(depth))
        output = policy(packed.metadata, noisy, time, packed_batch=packed)
        output.square().sum().backward()
        results.append(output.detach())
    torch.cuda.synchronize()
    torch.testing.assert_close(results[1], results[0], atol=0, rtol=0)
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0, msg=name)
