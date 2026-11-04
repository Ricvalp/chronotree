"""Router objectives and sharded label provenance must survive fitting/resume."""

from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from stmem.config import run_dir
from stmem.training import routers
from stmem.training.checkpoints import file_sha256
from test_tracking_routers import RecordingTracker, router_config, write_labels


@pytest.fixture
def trackers(monkeypatch):
    RecordingTracker.instances = []
    monkeypatch.setattr(routers, "RunTracker", RecordingTracker)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(routers, "environment_manifest", lambda: {"device": "cpu-test"})
    return RecordingTracker.instances


def _config(path, objective="velocity"):
    cfg = router_config(path)
    cfg["labels"].update(
        objective=objective, integration_steps=8, contexts=8, refresh_contexts=8, candidates=4
    )
    cfg["training"].update(router_device="cpu", router_validation_batch=3)
    return cfg


def _write_shards(cfg, *, refresh=False, count=2, router_hash="initial-router"):
    root = Path(cfg["paths"]["run_root"]) / "labels"
    root.mkdir(parents=True, exist_ok=True)
    phase = "refresh" if refresh else "initial"
    paths = []
    total = cfg["labels"]["refresh_contexts" if refresh else "contexts"]
    for index in range(count):
        ids = np.arange(index, total, count, dtype=np.int64)
        contexts = np.repeat(ids, 2)
        rows = len(contexts)
        generator = np.random.default_rng(23 + index + int(refresh) * 100)
        width = cfg["model"]["width"]
        query = np.repeat(generator.normal(size=(len(ids), 2 * width + 27)), 2, axis=0).astype("f4")
        partition = (contexts >= total // 2).astype("i1")
        episodes = ["task/validation" if p else "task/training" for p in partition]
        ident = {
            "schema_version": 2,
            "source_sha256": "synthetic-fixed-source",
            "teacher": "teacher-hash",
            "compact": "compact-hash",
            "cache": "cache-hash",
            "data": "data-hash",
            "reader": cfg["reader"],
            "partition": {"task/training": "train", "task/validation": "validation"},
            "objective": cfg["labels"]["objective"],
            "integration_steps": cfg["labels"]["integration_steps"],
            "contexts": total,
            "sample_plan": phase + "-plan",
            "shard_index": index,
            "shard_count": count,
            "refresh": refresh,
            "router": router_hash if refresh else None,
        }
        path = root / f"{phase}-{index}.h5"
        with h5py.File(path, "w") as f:
            f.attrs["complete"] = True
            f.attrs["identity"] = json.dumps(ident)
            f["query"] = query
            f["candidate"] = generator.normal(size=(rows, 2 * width + 23)).astype("f4")
            f["gain"] = np.tile(np.array([-.3, .7], dtype="f4"), len(ids))
            f["context_id"] = contexts
            f["partition"] = partition
            f["budget"] = np.full(rows, 64, dtype="i8")
            f["move_index"] = np.tile([0, 1], len(ids))
            f["attempted_context_ids"] = ids
            f.create_dataset("episode", data=episodes, dtype=h5py.string_dtype())
        paths.append(str(path))
    cfg["labels"]["refresh_shards" if refresh else "initial_shards"] = paths
    return paths


def _alter_identity(path, **updates):
    with h5py.File(path, "r+") as f:
        identity = json.loads(f.attrs["identity"])
        identity.update(updates)
        f.attrs["identity"] = json.dumps(identity)


@pytest.mark.parametrize("objective,integration_steps", [("velocity", 8), ("final_action", 8), ("final_action", 4)])
def test_sharded_objective_trains_and_preserves_exact_resume(tmp_path, trackers, objective, integration_steps):
    cfg = _config(tmp_path / "resumed", objective)
    cfg["labels"]["integration_steps"] = integration_steps
    cfg["training"].update(steps=3, checkpoint_steps=[3])
    _write_shards(cfg)
    marker = tmp_path / "stop"
    marker.touch()
    assert routers.train_router(cfg, stop_file=marker) == 75
    marker.unlink()
    checkpoint = run_dir(cfg, "router") / "last.pt"
    # Path ordering is irrelevant; semantic shard order determines row order.
    cfg["labels"]["initial_shards"].reverse()
    assert routers.train_router(cfg, resume=checkpoint) == 0
    actual = torch.load(checkpoint, weights_only=False)
    assert actual["objective"] == actual["provenance"]["objective"] == objective
    assert actual["integration_steps"] == integration_steps
    assert actual["label_identity"]["cache"] == "cache-hash"
    assert trackers[-1].options["metadata"]["objective"] == objective
    assert trackers[-1].options["metadata"]["router_device"] == "cpu"
    assert trackers[-1].options["metadata"]["feature_bytes"] == 16 * (43 + 39 + 1) * 4
    control = _config(tmp_path / "control", objective)
    control["labels"]["integration_steps"] = integration_steps
    control["training"].update(steps=3, checkpoint_steps=[3])
    _write_shards(control)
    assert routers.train_router(control) == 0
    expected = torch.load(run_dir(control, "router") / "last.pt", weights_only=False)
    for name in expected["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)


def test_refresh_matches_frozen_identity_and_resumed_router(tmp_path, trackers):
    cfg = _config(tmp_path)
    _write_shards(cfg)
    assert routers.train_router(cfg) == 0
    checkpoint = run_dir(cfg, "router") / "last.pt"
    original_scale = torch.load(checkpoint, weights_only=False)["target_scale"]
    refreshed = deepcopy(cfg)
    refreshed["labels"]["use_refresh"] = True
    refreshed["training"].update(steps=3, checkpoint_steps=[3])
    paths = _write_shards(refreshed, refresh=True, router_hash="wrong-router")
    with pytest.raises(ValueError, match="resumed router checkpoint"):
        routers.train_router(refreshed, resume=checkpoint)
    for path in paths:
        _alter_identity(path, router=file_sha256(checkpoint))
    assert routers.train_router(refreshed, resume=checkpoint) == 0
    state = torch.load(checkpoint, weights_only=False)
    assert len(state["label_hashes"]) == 4
    assert state["target_scale"] == original_scale
    assert trackers[-1].logs[-1][1]["validation_rank_pairs"] == 8
    assert routers.train_router(refreshed, resume=checkpoint) == 0


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"objective": "final_action"}, "objective"),
        ({"integration_steps": 4}, "integration_steps"),
        ({"reader": "flat"}, "reader"),
        ({"teacher": "other"}, "provenance"),
        ({"compact": "other"}, "provenance"),
        ({"cache": "other"}, "provenance"),
        ({"data": "other"}, "provenance"),
        ({"sample_plan": "other"}, "sample plans"),
        ({"refresh": True}, "phase"),
    ],
)
def test_rejects_mixed_label_contracts(tmp_path, updates, match):
    cfg = _config(tmp_path)
    paths = _write_shards(cfg)
    _alter_identity(paths[1], **updates)
    with pytest.raises(ValueError, match=match):
        routers._load_labels(cfg)


def test_rejects_missing_or_duplicate_shards(tmp_path):
    cfg = _config(tmp_path)
    paths = _write_shards(cfg)
    cfg["labels"]["initial_shards"] = paths[:1]
    with pytest.raises(ValueError, match="Missing or duplicate"):
        routers._load_labels(cfg)
    cfg["labels"]["initial_shards"] = [paths[0], paths[0]]
    with pytest.raises(ValueError, match="Duplicate label shard path"):
        routers._load_labels(cfg)


def test_refresh_cannot_change_teacher_or_data_identity(tmp_path):
    cfg = _config(tmp_path)
    _write_shards(cfg)
    cfg["labels"]["use_refresh"] = True
    paths = _write_shards(cfg, refresh=True)
    for path in paths:
        _alter_identity(path, teacher="different-teacher")
    with pytest.raises(ValueError, match="provenance"):
        routers._load_labels(cfg)


@pytest.mark.parametrize("damage", ["attempted", "ownership", "moves", "query", "partition"])
def test_rejects_corrupt_context_coverage_or_rows(tmp_path, damage):
    cfg = _config(tmp_path)
    paths = _write_shards(cfg)
    with h5py.File(paths[0], "r+") as f:
        if damage == "attempted":
            f["attempted_context_ids"][0] = 2
        elif damage == "ownership":
            f["context_id"][0] = 1
        elif damage == "moves":
            f["move_index"][1] = 0
        elif damage == "query":
            f["query"][1, 0] += 1
        else:
            f["partition"][0] = 1
    with pytest.raises(ValueError):
        routers._load_labels(cfg)


def test_attempted_contexts_may_have_no_candidate_rows(tmp_path):
    cfg = _config(tmp_path)
    paths = _write_shards(cfg)
    # A legitimate context with no eligible moves is recorded as attempted only.
    with h5py.File(paths[0], "r+") as f:
        for name in ("query", "candidate", "gain", "context_id", "partition", "episode", "move_index", "budget"):
            remaining = np.asarray(f[name])[2:]
            del f[name]
            f[name] = remaining
    merged, *_ = routers._load_labels(cfg)
    assert len(merged["gain"]) == 14


def test_legacy_labels_cannot_be_reinterpreted_as_final_action(tmp_path):
    cfg = router_config(tmp_path)
    write_labels(cfg)
    cfg["labels"]["objective"] = "final_action"
    with pytest.raises(ValueError, match="Legacy"):
        routers._load_labels(cfg)


def test_resume_cannot_switch_objective_even_with_new_valid_labels(tmp_path, trackers):
    cfg = _config(tmp_path / "velocity")
    _write_shards(cfg)
    assert routers.train_router(cfg) == 0
    checkpoint = run_dir(cfg, "router") / "last.pt"
    final = _config(tmp_path / "final", "final_action")
    _write_shards(final)
    with pytest.raises(ValueError, match="resume objective"):
        routers.train_router(final, resume=checkpoint)


def test_device_validation_is_explicit(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert routers._router_device(cfg) == "cpu"
    cfg["training"]["router_device"] = "auto"
    assert routers._router_device(cfg) == "cpu"
    cfg["training"]["router_device"] = "cuda"
    with pytest.raises(RuntimeError, match="unavailable"):
        routers._router_device(cfg)
    cfg["training"]["router_device"] = "wrong"
    with pytest.raises(ValueError, match="router_device"):
        routers._router_device(cfg)
