"""Synthetic diagnostics for immutable architecture inputs, not policy performance."""

# Shared synthetic cached-evidence fixture.
# ruff: noqa: F811
from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from test_cached_context_loader import fixture  # noqa: F401
from test_packed_context import indexed  # noqa: F401
from stmem.config import digest
from stmem.models.router import ArchitectureRouter, current_token_features
from stmem.training.checkpoints import file_sha256
from stmem.training.label_data import LabelDataReader, LabelPlan
from stmem.training.label_experiment import LabelShard, _episode_name
from stmem.training.router_inputs import load_router_inputs, validate_input_shard, write_input_shard
from stmem.training.routers import _load_labels


def _labels(indexed, tmp_path, family="tree", shards=2):
    cfg, data, policy, cache, root = indexed
    cfg = deepcopy(cfg)
    cfg["stage"], cfg["reader"] = "router", family
    cfg["labels"].update(contexts=24, refresh_contexts=12, budgets=[8, 12, 16], candidates=4,
                         objective="final_action", integration_steps=8, use_refresh=False)
    plan = LabelPlan(data, cfg)
    items = plan.items()
    partition = {_episode_name(data, i): data.label_partition(i) for i in range(len(data.refs))}
    sources, sidecars = [], []
    for shard in range(shards):
        source, sidecar = tmp_path / f"labels_{shard}.h5", tmp_path / f"inputs_{shard}.h5"
        identity = {"schema_version": 2, "teacher": "synthetic_teacher", "compact": "synthetic_compact",
                    "cache": "synthetic_cache", "data": data.identity, "partition": partition,
                    "source_sha256": "synthetic_source", "contexts": len(items), "sample_plan": digest(items),
                    "refresh": False, "shard_index": shard, "shard_count": shards, "reader": family,
                    "objective": "final_action", "integration_steps": 8, "router": None,
                    "label_sampling": {key: cfg["labels"][key] for key in
                                       ("seed", "budgets", "candidates", "exploration_fraction", "split_fraction")}}
        writer = LabelShard(source, identity, cfg["model"]["width"])
        try:
            with LabelDataReader(data, cfg, cache.root, root) as reader:
                # Reverse context order exercises preservation of physical row order.
                for item in reversed(plan.items(shard_index=shard, shard_count=shards)):
                    payload = reader.prepare(item)
                    n = len(payload["readers"][family]["candidate"])
                    gains = torch.arange(n, dtype=torch.float32) + item["context_id"] / 100
                    writer.append([payload], [gains], family)
            writer.finish()
        finally:
            writer.close()
        sources.append(str(source))
        sidecars.append(str(sidecar))
    cfg["labels"].update(initial_shards=sources, router_inputs=sidecars)
    return cfg, data, policy, cache, root


@pytest.mark.parametrize("family", ["tree", "flat"])
def test_input_replay_preserves_labels_and_aligns_unsorted_shards(indexed, tmp_path, family):
    cfg, data, policy, cache, root = _labels(indexed, tmp_path, family)
    source_hashes = [file_sha256(path) for path in cfg["labels"]["initial_shards"]]
    for source, sidecar in zip(cfg["labels"]["initial_shards"], cfg["labels"]["router_inputs"]):
        assert write_input_shard(cfg, data, cache.root, root, source, sidecar) == 0
        before = file_sha256(sidecar)
        assert write_input_shard(cfg, data, cache.root, root, source, sidecar) == 0
        assert file_sha256(sidecar) == before
    assert source_hashes == [file_sha256(path) for path in cfg["labels"]["initial_shards"]]
    # Input settings may be supplied in arbitrary shard order, like label settings.
    cfg["labels"]["initial_shards"].reverse()
    cfg["labels"]["router_inputs"].reverse()
    merged, paths, _, _, _ = _load_labels(cfg)
    arrays, identity = load_router_inputs(cfg, paths, merged)
    assert len(identity["shards"]) == 2
    assert len(arrays["row_to_context"]) == len(merged["gain"])
    assert arrays["current"].shape == (24, cfg["model"]["current_tokens"], cfg["model"]["width"] + 23)
    assert (arrays["children"] is not None) == (family == "tree")
    prototype = ArchitectureRouter(cfg["model"]["width"], cfg["model"]["router_width"], architecture="children")
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        for item in LabelPlan(data, cfg).items():
            payload = reader.prepare(item, router_inputs=True)
            row_ids = np.flatnonzero(merged["context_id"] == item["context_id"])
            if not len(row_ids):
                continue
            current = arrays["current"][arrays["row_to_context"][row_ids[0]]]
            expected = torch.cat((payload["current"].content[0], payload["current"].metadata[0]), -1)
            np.testing.assert_array_equal(current[:len(expected)], expected.numpy())
            evidence = cache.evidence(data.episode(item["episode_index"]).decision(item["index"], data.horizon))
            online_current = current_token_features(evidence.current_records, evidence.clocks,
                                                    width=cfg["model"]["width"], device="cpu")
            np.testing.assert_array_equal(current[:len(expected)], online_current.numpy())
            if family == "tree":
                records = {record.node_id: record for record in evidence.archive.records}
                names = payload["readers"][family]["candidate_ids"]
                expected = prototype.child_input([records[name] for name in names]).detach().numpy()
                np.testing.assert_array_equal(arrays["children"][row_ids], expected)


@pytest.mark.parametrize("family", ["tree", "flat"])
def test_input_only_path_reads_no_full_historical_payload(indexed, family, monkeypatch):
    cfg, data, _, cache, root = indexed
    cfg["labels"].update(contexts=24, budgets=[8, 12, 16], candidates=4)
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        item = LabelPlan(data, cfg).items()[10]
        expected = reader.prepare(item, router_inputs=True, reader_family=family)
        original = reader.packed._columns

        def guarded(group, rows=None, **kwargs):
            if group.name.endswith("/nodes"):
                assert rows is not None and len(rows) <= cfg["labels"]["candidates"] * 3
            return original(group, rows, **kwargs)

        monkeypatch.setattr(reader.packed, "_columns", guarded)
        actual = reader.prepare(item, router_inputs=True, inputs_only=True, reader_family=family)
    assert actual["full"] is None
    torch.testing.assert_close(actual["query"], expected["query"], rtol=0, atol=0)
    for key in ("candidate", "costs"):
        torch.testing.assert_close(actual["readers"][family][key], expected["readers"][family][key], rtol=0, atol=0)
    assert actual["readers"][family]["candidate_ids"] == expected["readers"][family]["candidate_ids"]


def test_replay_rejects_changed_original_candidate_before_publication(indexed, tmp_path):
    cfg, data, _, cache, root = _labels(indexed, tmp_path, shards=1)
    source, output = cfg["labels"]["initial_shards"][0], cfg["labels"]["router_inputs"][0]
    with h5py.File(source, "a") as file:
        file["candidate"][0, 0] += 1
    with pytest.raises(ValueError, match="features differ"):
        write_input_shard(cfg, data, cache.root, root, source, output)
    assert not Path(output).exists()


def test_partial_commit_resume_and_tamper_detection(indexed, tmp_path, monkeypatch):
    cfg, data, _, cache, root = _labels(indexed, tmp_path, shards=1)
    source, output = cfg["labels"]["initial_shards"][0], cfg["labels"]["router_inputs"][0]
    original = LabelDataReader.prepare
    calls = 0

    def interrupted(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("diagnostic interruption")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LabelDataReader, "prepare", interrupted)
    with pytest.raises(RuntimeError, match="diagnostic interruption"):
        write_input_shard(cfg, data, cache.root, root, source, output)
    with h5py.File(Path(output).with_suffix(".partial.h5"), "r") as file:
        assert json.loads(file.attrs["commit"])[1] == 3
    monkeypatch.setattr(LabelDataReader, "prepare", original)
    assert write_input_shard(cfg, data, cache.root, root, source, output) == 0
    validate_input_shard(output, source, cfg["model"]["width"], cfg["model"]["current_tokens"])
    with h5py.File(output, "a") as file:
        file["children"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="receipt/hash mismatch"):
        validate_input_shard(output, source, cfg["model"]["width"], cfg["model"]["current_tokens"])


def test_refresh_inputs_fail_explicitly(indexed, tmp_path):
    cfg, data, _, cache, root = _labels(indexed, tmp_path, shards=1)
    source, output = cfg["labels"]["initial_shards"][0], cfg["labels"]["router_inputs"][0]
    with h5py.File(source, "a") as file:
        identity = json.loads(file.attrs["identity"])
        identity["refresh"], identity["router"] = True, "some_router"
        file.attrs["identity"] = json.dumps(identity)
    with pytest.raises(ValueError, match="INITIAL"):
        write_input_shard(cfg, data, cache.root, root, source, output)


def test_richer_inputs_ignore_future_expert_actions(indexed):
    cfg, data, _, cache, root = indexed
    cfg["labels"].update(contexts=24, budgets=[8, 12, 16], candidates=4)
    item = LabelPlan(data, cfg).items()[10]
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        before = reader.prepare(item, router_inputs=True, inputs_only=True, reader_family="tree")
        episode = reader.decisions._episode(item["episode_index"])
        episode.actions = episode.actions.clone()
        episode.actions[item["index"] + 1:] += 99
        after = reader.prepare(item, router_inputs=True, inputs_only=True, reader_family="tree")
    assert not torch.equal(before["target"], after["target"])
    for field in ("query", "text", "clocks"):
        torch.testing.assert_close(before[field], after[field], rtol=0, atol=0)
    for field in ("content", "metadata", "valid"):
        torch.testing.assert_close(getattr(before["current"], field), getattr(after["current"], field), rtol=0, atol=0)
    for field in ("candidate", "children"):
        torch.testing.assert_close(before["readers"]["tree"][field], after["readers"]["tree"][field], rtol=0, atol=0)


def test_input_loader_rejects_feature_code_drift(indexed, tmp_path, monkeypatch):
    cfg, data, _, cache, root = _labels(indexed, tmp_path, shards=1)
    source, output = cfg["labels"]["initial_shards"][0], cfg["labels"]["router_inputs"][0]
    write_input_shard(cfg, data, cache.root, root, source, output)
    monkeypatch.setattr("stmem.training.router_inputs._feature_source", lambda: "different_feature_code")
    with pytest.raises(ValueError, match="identity differs"):
        validate_input_shard(output, source, cfg["model"]["width"], cfg["model"]["current_tokens"])
