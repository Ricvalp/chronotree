"""Synthetic scientific parity checks for indexed compact context selection."""

from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from test_cached_context_loader import _contains_records, fixture  # noqa: F401 -- shared synthetic fixture
from stmem.data.cache import episode_key
from stmem.data.hierarchy_cache import write_episode_hierarchy
from stmem.models.batched_policy import records_bank
from stmem.training.cached_loader import (
    CachedPolicyBatchLoader, CachedPolicyDataset, LightweightDecisions, selected_context,
)
from stmem.training.packed_context import PackedContextReader, SCALAR_FIELDS, STRING_FIELDS, TENSOR_FIELDS


@pytest.fixture
def indexed(fixture, tmp_path):  # noqa: F811 -- pytest injects the imported fixture
    cfg, data, policy, cache = fixture
    root = tmp_path / "hierarchy"
    root.mkdir()
    for task in data.tasks:
        with h5py.File(cache.root / f"{task}.h5", "r") as base, h5py.File(root / f"{task}.h5", "w") as out:
            for episode in data.by_task[task]:
                canonical = data.episode(episode)
                key = episode_key(task, canonical.episode_id)
                manifest = {"identity": {"model_encoder": cfg["model"], "data": cfg["data"]}}
                write_episode_hierarchy(base[key], out.create_group(key),
                                        {"synthetic_diagnostic": True}, manifest)
    return cfg, data, policy, cache, root


@pytest.mark.parametrize("family", ["tree", "flat", "uniform", "current"])
def test_indexed_families_match_reference_context_and_all_summary_fields(indexed, family):
    cfg, data, policy, cache, root = indexed
    cfg = deepcopy(cfg)
    cfg["training"]["context_mix"] = {family: 1.0}
    light = LightweightDecisions(data)
    reader = PackedContextReader(cfg, cache.root, root)
    try:
        for position in range(24):
            decision = light.decision(position)
            original = cache.evidence(data.sample_position(position))
            expected = selected_context(original, cfg, position)
            actual = reader.prepare(decision, position, debug=True)
            assert tuple(actual.debug["columns"]["node_id"]) == tuple(record.node_id for record in expected)
            assert actual.leaf_count == len(original.archive.leaves)
            for field in (*TENSOR_FIELDS, *SCALAR_FIELDS, *STRING_FIELDS):
                values = actual.debug["columns"][field]
                for row, record in enumerate(expected):
                    if field in TENSOR_FIELDS:
                        torch.testing.assert_close(values[row], getattr(record, field), rtol=0, atol=0)
                    else:
                        assert values[row] == getattr(record, field), field
            clocks = torch.tensor([decision.clocks], dtype=torch.float32)
            banks = {
                "current": records_bank(policy, [original.current_records], clocks),
                "history": records_bank(policy, [expected], clocks),
            }
            for name, bank in banks.items():
                for field in fields(bank):
                    torch.testing.assert_close(getattr(getattr(actual, name), field.name),
                                               getattr(bank, field.name), rtol=0, atol=0)
            assert actual.history.content.shape[1] <= actual.debug["budget"]
    finally:
        reader.close()
        light.close()


def test_current_does_not_open_historical_node_payload(indexed, monkeypatch):
    cfg, data, _, cache, root = indexed
    cfg = deepcopy(cfg)
    cfg["training"]["context_mix"] = {"current": 1.0}
    reader = PackedContextReader(cfg, cache.root, root)
    light = LightweightDecisions(data)
    original = reader._read

    def read(dataset, rows=None):
        assert "/nodes/" not in dataset.name, dataset.name
        return original(dataset, rows)

    monkeypatch.setattr(reader, "_read", read)
    try:
        actual = reader.prepare(light.decision(0), 0)
        assert actual.history.valid.numel() == 0
        assert actual.index_hits == actual.index_misses == 0
    finally:
        reader.close()
        light.close()


def test_tree_reads_only_selected_historical_payload_and_avoids_maximum_columns(indexed, monkeypatch):
    cfg, data, _, cache, root = indexed
    cfg = deepcopy(cfg)
    cfg["training"]["context_mix"] = {"tree": 1.0}
    reader = PackedContextReader(cfg, cache.root, root)
    light = LightweightDecisions(data)
    original = reader._read
    payload_calls = []

    def read(dataset, rows=None):
        if "/nodes/" in dataset.name and dataset.name.rsplit("/", 1)[1] in TENSOR_FIELDS:
            assert rows is not None
            assert len(rows) <= max(cfg["model"]["budgets"])
            assert not dataset.name.endswith("/max_content")
            payload_calls.append(dataset.name)
        return original(dataset, rows)

    monkeypatch.setattr(reader, "_read", read)
    try:
        reader.prepare(light.decision(0), 0)
        assert payload_calls
    finally:
        reader.close()
        light.close()


def test_partial_prefix_key_mismatch_fails_before_selection(indexed):
    cfg, data, _, cache, root = indexed
    light = LightweightDecisions(data)
    decision = light.decision(0)
    key = episode_key(decision.info.task, decision.info.episode_id)
    with h5py.File(root / f"{decision.info.task}.h5", "a") as file:
        file[key]["decisions"][str(decision.info.index)].attrs["prefix_key"] = "wrong_prefix"
    reader = PackedContextReader(cfg, cache.root, root)
    try:
        with pytest.raises(ValueError, match="cadence"):
            reader.prepare(decision, 0)
    finally:
        reader.close()
        light.close()


def test_indexed_worker_payload_matches_legacy_and_reports_phase_times(indexed):
    cfg, data, _, cache, root = indexed
    legacy = CachedPolicyDataset(data, cache.root)
    expected = legacy[tuple(range(20))]
    legacy.close()
    data.cfg = deepcopy(cfg)
    data.cfg["training"]["hierarchy_cache_root"] = str(root)
    packed = CachedPolicyDataset(data, cache.root)
    try:
        actual = packed[tuple(range(20))]
        assert actual.hierarchy_enabled
        assert actual.hierarchy_read_seconds > 0 and actual.hierarchy_read_bytes > 0
        assert actual.hierarchy_select_seconds > 0
        for name in ("current", "history"):
            for field in fields(expected.current):
                torch.testing.assert_close(getattr(getattr(actual, name), field.name),
                                           getattr(getattr(expected, name), field.name), rtol=0, atol=0)
    finally:
        packed.close()


def test_indexed_spawn_workers_preserve_order_and_only_transfer_selected_tensors(indexed):
    cfg, data, _, cache, root = indexed
    batches = {}
    for workers in (0, 2):
        data.cfg = deepcopy(cfg)
        data.cfg["training"].update(workers=workers, hierarchy_cache_root=str(root))
        with CachedPolicyBatchLoader(
            data, cache, start=1, steps=3, batch=8, rank=1, world=2, microbatch=2,
        ) as loader:
            batches[workers] = [batch for step in (1, 2) for batch in loader.read_update(step)]
        assert loader._iterator is None
        assert loader._dataset._reader is None
    for single, parallel in zip(batches[0], batches[2]):
        assert single.positions == parallel.positions and single.metadata == parallel.metadata
        assert single.worker_pid != parallel.worker_pid
        assert parallel.hierarchy_enabled and not _contains_records(parallel)
        for name in ("target_chunk", "target_valid", "clocks", "current_state", "text", "history_leaf_counts"):
            torch.testing.assert_close(getattr(single, name), getattr(parallel, name), rtol=0, atol=0)
        for name in ("current", "history"):
            for field in fields(single.history):
                torch.testing.assert_close(getattr(getattr(single, name), field.name),
                                           getattr(getattr(parallel, name), field.name), rtol=0, atol=0)


def test_indexed_topology_lru_eviction_and_handle_shutdown(indexed):
    cfg, data, _, cache, root = indexed
    light = LightweightDecisions(data)
    reader = PackedContextReader(cfg, cache.root, root, maximum_indices=1, maximum_index_bytes=1_000_000)
    try:
        for position in range(12):
            actual = reader.prepare(light.decision(position), position, debug=True)
            reference = selected_context(cache.evidence(data.sample_position(position)), cfg, position)
            assert list(actual.debug["columns"]["node_id"]) == [record.node_id for record in reference]
            assert len(reader.indices) <= 1 and reader.index_bytes <= reader.maximum_index_bytes
        assert reader.misses > 1
    finally:
        reader.close()
        light.close()
    assert not reader.indices and not reader.index_bytes
    assert not reader.base_files.values and not reader.hierarchy_files.values


def test_lazy_geometry_growth_respects_index_byte_limit(indexed):
    """An index can fit before geometry is loaded but exceed the cap afterwards."""
    cfg, data, _, cache, root = indexed
    decision = LightweightDecisions(data)
    reader = PackedContextReader(cfg, cache.root, root)
    try:
        info = decision.decision(0).info
        group = reader.hierarchy_files.get(root / f"{info.task}.h5")[episode_key(info.task, info.episode_id)]
        index = reader._index(group, geometry=False)
        assert "anchor" not in index
        reader.maximum_index_bytes = reader.index_bytes
        reader._ensure_geometry(group, index)
        assert all(field in index for field in ("anchor", "time_anchor", "depth"))
        assert not reader.indices and reader.index_bytes == 0
        for field in ("anchor", "time_anchor", "depth"):
            np.testing.assert_array_equal(index[field], np.asarray(group["nodes"][field]))
        # This same decision remains usable even though its index is too big
        # to retain; a second lazy fill must not reread or replace its arrays.
        reads = reader.read_bytes
        reader._ensure_geometry(group, index)
        assert reader.read_bytes == reads
    finally:
        reader.close()
        decision.close()


def test_empty_history_keeps_current_tokens_for_every_family(indexed, tmp_path):
    cfg, data, _, cache, _ = indexed
    episode = data.episode(data.by_task[data.tasks[0]][0])
    key = episode_key(episode.task, episode.episode_id)
    base_root, hierarchy_root = tmp_path / "empty_base", tmp_path / "empty_hierarchy"
    base_root.mkdir()
    hierarchy_root.mkdir()
    with h5py.File(cache.root / f"{episode.task}.h5", "r") as original:
        with h5py.File(base_root / f"{episode.task}.h5", "w") as base:
            source = base.create_group(key)
            source.attrs.update(task=episode.task, episode_id=episode.episode_id)
            for name in ("conditioning", "execution", "prefix", "current"):
                source.create_group(name)
            original.copy(original[key]["current"]["0"], source["current"], name="0")
            original.copy(original[key]["text"], source, name="text")
            with h5py.File(hierarchy_root / f"{episode.task}.h5", "w") as out:
                write_episode_hierarchy(source, out.create_group(key), {"synthetic_diagnostic": True},
                                        {"identity": {"model_encoder": cfg["model"], "data": cfg["data"]}})
    decision = SimpleNamespace(info=SimpleNamespace(task=episode.task, episode_id=episode.episode_id,
                                                   index=0, execution_history_frames=0), clocks=(0., 0.))
    for family in ("tree", "flat", "uniform", "current"):
        config = deepcopy(cfg)
        config["training"]["context_mix"] = {family: 1.0}
        reader = PackedContextReader(config, base_root, hierarchy_root)
        try:
            actual = reader.prepare(decision, 0)
            assert actual.leaf_count == 0 and actual.history.valid.numel() == 0
            assert actual.current.valid.sum() == cfg["model"]["current_tokens"]
        finally:
            reader.close()
