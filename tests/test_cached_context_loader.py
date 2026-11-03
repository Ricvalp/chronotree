"""Synthetic CPU parity/IPC checks for cached compact-policy preparation."""

from copy import deepcopy
from dataclasses import fields, is_dataclass
import pickle
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from stmem.config import load_config
from stmem.data.cache import FrozenCache, episode_key, write_episode_cache
from stmem.memory.frontier import validate_frontier
from stmem.memory.records import MemoryRecord
from stmem.models.batched_policy import records_bank
from stmem.models.policy import MemoryPolicy
from stmem.train import context_sample
from stmem.training.cached_loader import (
    CachedArchiveReader, CachedPolicyBatchLoader, CachedPolicyDataset,
    LightweightDecisions, selected_context,
)
from stmem.training.data import TrainingData


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(31)
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["paths"].update(cache_root=str(tmp_path / "cache"), run_root=str(tmp_path / "run"))
    cfg["data"].update(tasks=["TaskA", "TaskB"], episodes_per_task=2,
                       observation_stride=2, process_every_observation=False,
                       synthetic_execution_frames=12, synthetic_conditioning_frames=5,
                       decision_stride=2)
    cfg["model"].update(width=16, heads=4, layers=1, text_layers=1, max_text_bytes=32,
                        points=8, block_size=2, visual_tokens=4, current_tokens=3,
                        horizon=2, execute=2, action_stride=2, max_execution_frames=12,
                        max_conditioning_frames=5, budgets=[16, 24, 32])
    cfg["model"]["encoder_options"].update(neighbor_cap=4, center_chunk=2)
    cfg["training"].update(workers=0, prefetch_factor=1)
    data = TrainingData(cfg)
    policy = MemoryPolicy(cfg).eval().requires_grad_(False)
    root = tmp_path / "encoded"
    root.mkdir()
    with torch.inference_mode():
        for task in data.tasks:
            with h5py.File(root / f"{task}.h5", "w") as out:
                for index in data.by_task[task]:
                    episode = data.episode(index)
                    group = out.create_group(episode_key(episode.task, episode.episode_id))
                    write_episode_cache(group, episode, policy, data.stride, data.horizon)
    cache = FrozenCache.__new__(FrozenCache)
    cache.cfg, cache.root = cfg, root
    yield cfg, data, policy, cache


def test_lightweight_decisions_and_cached_trees_match_reference(fixture):
    cfg, data, policy, cache = fixture
    light = LightweightDecisions(data)
    reader = CachedArchiveReader(cfg, cache.root)
    methods, prefixes = set(), set()
    try:
        for position in range(60):
            reference = data.sample_position(position)
            decision = light.decision(position)
            torch.testing.assert_close(decision.target, reference.target_chunk, rtol=0, atol=0)
            torch.testing.assert_close(decision.target_valid, reference.target_valid, rtol=0, atol=0)
            torch.testing.assert_close(decision.state[:15], reference.current.proprio, rtol=0, atol=0)
            torch.testing.assert_close(decision.state[15:23], reference.current.previous_action, rtol=0, atol=0)
            assert bool(decision.state[23]) == reference.current.action_available
            assert decision.info.conditioning_frames == len(reference.conditioning)
            assert decision.info.execution_history_frames == len(reference.execution_prefix)
            prefixes.add(len(reference.execution_prefix) % cfg["model"]["block_size"])
            old, new = cache.evidence(reference), reader.evidence(decision)
            assert old.clocks == new.clocks
            assert [r.node_id for r in old.archive.records] == [r.node_id for r in new.archive.records]
            for left, right in zip(old.archive.records, new.archive.records):
                assert left.count == right.count and left.completed == right.completed
                assert left.leaf_ids == right.leaf_ids
                torch.testing.assert_close(left.content_sum, right.content_sum, rtol=0, atol=0)
                torch.testing.assert_close(left.anchor, right.anchor, rtol=0, atol=0)
            expected = context_sample(policy, old, cfg, position)
            actual = selected_context(new, cfg, position)
            assert [r.node_id for r in actual] == [r.node_id for r in expected]
            if actual:
                validate_frontier(actual, new.archive.leaves, max(cfg["model"]["budgets"]))
            methods.add(bool(actual))
        assert prefixes == {0, 1}
        assert methods == {False, True}  # Includes the current-only mixture control.
        assert reader.hits > 0 and reader.bytes <= reader.maximum_bytes
    finally:
        reader.close()
        light.close()


def _contains_records(value):
    if isinstance(value, MemoryRecord):
        return True
    if is_dataclass(value):
        return any(_contains_records(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, (tuple, list)):
        return any(_contains_records(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_records(item) for item in value.values())
    return False


def test_worker_payload_contains_only_selected_tensors_with_reference_metadata(fixture):
    cfg, data, policy, cache = fixture
    dataset = CachedPolicyDataset(data, cache.root)
    positions = tuple(range(20))
    try:
        actual = dataset[positions]
        decisions = [data.sample_position(position) for position in positions]
        evidence = [cache.evidence(decision) for decision in decisions]
        records = [context_sample(policy, e, cfg, position) for e, position in zip(evidence, positions)]
        expected_history = records_bank(policy, records, actual.clocks)
        expected_current = records_bank(policy, [e.current_records for e in evidence], actual.clocks)
        for name, expected in (("current", expected_current), ("history", expected_history)):
            for field in fields(expected):
                torch.testing.assert_close(
                    getattr(getattr(actual, name), field.name), getattr(expected, field.name), rtol=0, atol=0
                )
        assert actual.history.content.shape[1] <= max(cfg["model"]["budgets"])
        assert not _contains_records(actual)
        assert actual.worker_pid > 0 and actual.cache_seconds > 0 and actual.context_seconds > 0
        assert actual.load_seconds > 0 and actual.pack_seconds > 0
        # Recursive trees and observations cannot accidentally appear in the IPC payload.
        restored = pickle.loads(pickle.dumps(actual))
        torch.testing.assert_close(restored.history.content, actual.history.content, rtol=0, atol=0)
        assert len(pickle.dumps(actual)) < 500_000
    finally:
        dataset.close()


def test_block_lru_is_bounded_and_eviction_preserves_evidence(fixture):
    cfg, data, _, cache = fixture
    light = LightweightDecisions(data)
    reader = CachedArchiveReader(cfg, cache.root, maximum_blocks=1, maximum_bytes=1024 * 1024)
    try:
        for position in range(12):
            decision = light.decision(position)
            actual = reader.evidence(decision)
            expected = cache.evidence(data.sample_position(position))
            assert [r.node_id for r in actual.archive.records] == [r.node_id for r in expected.archive.records]
            assert len(reader.blocks) <= 1 and reader.bytes <= reader.maximum_bytes
    finally:
        reader.close()
        light.close()
    assert not reader.blocks and not reader.files.handles


def test_spawn_workers_preserve_order_and_selected_tensors(fixture):
    cfg, data, _, cache = fixture
    batches = {}
    for workers in (0, 2):
        data.cfg = deepcopy(cfg)
        data.cfg["training"]["workers"] = workers
        with CachedPolicyBatchLoader(
            data, cache, start=1, steps=3, batch=8, rank=1, world=2, microbatch=2,
        ) as loader:
            batches[workers] = [batch for step in (1, 2) for batch in loader.read_update(step)]
        assert loader._iterator is None
    for single, parallel in zip(batches[0], batches[2]):
        assert single.positions == parallel.positions
        assert single.metadata == parallel.metadata
        assert single.worker_pid != parallel.worker_pid
        for field in ("target_chunk", "target_valid", "clocks", "current_state", "text", "history_leaf_counts"):
            torch.testing.assert_close(getattr(single, field), getattr(parallel, field), rtol=0, atol=0)
        for field in fields(single.history):
            torch.testing.assert_close(
                getattr(single.history, field.name), getattr(parallel.history, field.name), rtol=0, atol=0
            )


@pytest.mark.parametrize("alignment", ["post_action_v1", "pre_action_v1"])
def test_real_reader_never_reads_point_clouds_or_descriptors(fixture, tmp_path, alignment):
    cfg, synthetic, _, _ = fixture
    episode = deepcopy(synthetic.episode(0))
    episode.alignment = alignment
    native = episode.conditioning + episode.execution
    path = tmp_path / "prepared.h5"
    with h5py.File(path, "w") as file:
        group = file.create_group("episode_0")
        group["source"] = np.asarray([frame.source == "conditioning" for frame in native])
        group["source_index"] = np.asarray([frame.source_index for frame in native])
        group["actions"] = torch.cat([torch.zeros(len(episode.conditioning), 8), episode.actions]).numpy()
        group["action_valid"] = np.ones((len(native), 8), dtype=bool)
        group["proprio"] = torch.stack([frame.proprio for frame in native]).numpy()
        # There are intentionally no points, visual features, depths or RGB datasets.
    ref = SimpleNamespace(path=path, key="episode_0", task="TaskA")
    dataset = SimpleNamespace(
        episodes=[ref], _require_feature_episode=lambda _ref: None,
        _alignment_for=lambda _ref: alignment, _validate_episode_layout=lambda _group: None,
    )
    data = SimpleNamespace(synthetic=False, dataset=dataset, refs=[0], horizon=2,
                           observation_stride=2, action_stride=2, sample_reference=lambda _pos: (0, 6))
    reader = LightweightDecisions(data)
    try:
        actual = reader.decision(0)
        expected = episode.decision(6, 2)
        torch.testing.assert_close(actual.target, expected.target_chunk, rtol=0, atol=0)
        torch.testing.assert_close(actual.state[:15], expected.current.proprio, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.state[15:23], episode.actions[6 if alignment == "post_action_v1" else 5], rtol=0, atol=0
        )
        assert actual.info.conditioning_frames == 3 and actual.info.execution_history_frames == 3
        assert actual.clocks == (4.0, 6.0)
    finally:
        reader.close()


def test_prefetched_cpu_batches_can_be_replayed_without_resampling(fixture):
    _, data, _, cache = fixture
    with CachedPolicyBatchLoader(
        data, cache, start=0, steps=2, batch=4, rank=0, world=1, microbatch=2, retain_for_retry=True,
    ) as loader:
        first = list(loader.read_update(0))
        replay = list(loader.read_update(0))
        assert all(left is right for left, right in zip(first, replay))
        next_update = list(loader.read_update(1))
        assert next_update[0].positions == (4, 5)
