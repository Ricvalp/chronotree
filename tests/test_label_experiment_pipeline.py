"""Small synthetic diagnostic of the complete four-router label/refresh path."""

# Shared fixtures contain actual encoded leaves and indexed causal hierarchies.
# ruff: noqa: F811

from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from test_cached_context_loader import fixture  # noqa: F401
from test_packed_context import indexed as indexed
from test_tracking_routers import RecordingTracker
from stmem.config import run_dir
from stmem.training import label_experiment, routers
from stmem.training.checkpoints import file_sha256
from stmem.training.label_data import LabelPlan
from stmem.training.label_objectives import variant_name


class _Tracker(RecordingTracker):
    def __init__(self, cfg, directory, **kwargs):
        Path(directory).mkdir(parents=True, exist_ok=True)
        super().__init__(cfg, directory, **kwargs)


def _configure_pipeline(indexed, tmp_path, monkeypatch):
    cfg, _, policy, cache, hierarchy = indexed
    cfg = deepcopy(cfg)
    cfg["labels"].update(
        contexts=12, refresh_contexts=12, budgets=[8, 12, 16], candidates=2,
        objectives=["velocity", "final_action"], integration_steps=8,
        experiment_root=str(tmp_path / "experiment"),
    )
    cfg["training"].update(
        hierarchy_cache_root=str(hierarchy), steps=1, checkpoint_steps=[1],
        checkpoint_every=1, log_every=1, effective_batch=4,
        router_device="cpu", router_validation_batch=4,
    )
    cfg["checkpoints"] = {}
    for stage in ("teacher", "compact"):
        path = tmp_path / f"{stage}.pt"
        checkpoint_config = deepcopy(cfg)
        checkpoint_config["stage"] = stage
        torch.save({"stage": stage, "config": checkpoint_config, "model": policy.state_dict()}, path)
        path.with_suffix(".selection.json").write_text(json.dumps({"synthetic_debug_only": True}))
        cfg["checkpoints"][stage] = str(path)
    # This fixture intentionally has no production receipts. Bypass only cache
    # packaging validation; inference, causal reads, checkpoints and labels are real.
    cache.manifest = {"status": "complete", "identity": {"synthetic_diagnostic": True}}
    monkeypatch.setattr(label_experiment, "FrozenCache", lambda *_: cache)
    monkeypatch.setattr(label_experiment, "validate_hierarchy_cache", lambda *_: None)
    monkeypatch.setattr(label_experiment, "RunTracker", _Tracker)
    monkeypatch.setattr(routers, "RunTracker", _Tracker)
    monkeypatch.setattr(routers, "environment_manifest", lambda: {"device": "synthetic-cpu-test"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return cfg


def _router_config(cfg, objective, reader, *, refresh=False):
    result = deepcopy(cfg)
    result["stage"], result["reader"] = "router", reader
    result["paths"]["run_root"] = str(Path(cfg["labels"]["experiment_root"]) / variant_name(objective, 8))
    result["labels"].update(
        objective=objective,
        initial_shards=[str(label_experiment.output_path(cfg, objective, reader, "initial", shard))
                        for shard in range(2)],
        use_refresh=refresh,
    )
    if refresh:
        result["labels"]["refresh_shards"] = [
            str(label_experiment.output_path(cfg, objective, reader, "refresh", shard)) for shard in range(2)
        ]
        result["training"].update(steps=2, checkpoint_steps=[2])
    return result


def test_synthetic_four_router_pipeline_recovers_cross_output_interruption(indexed, tmp_path, monkeypatch):
    cfg = _configure_pipeline(indexed, tmp_path, monkeypatch)
    original_append = label_experiment.LabelShard.append
    interrupted = False

    def interrupt_after_first_output(self, *args, **kwargs):
        nonlocal interrupted
        original_append(self, *args, **kwargs)
        if not interrupted:
            interrupted = True
            raise RuntimeError("synthetic interruption between independent output commits")

    monkeypatch.setattr(label_experiment.LabelShard, "append", interrupt_after_first_output)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        label_experiment.generate(cfg, shards=2, shard_index=0, device="cpu", workers=0,
                                  batch_size=2, compact_batch_size=8, teacher_token_budget=128)
    monkeypatch.setattr(label_experiment.LabelShard, "append", original_append)
    for shard in range(2):
        assert label_experiment.generate(cfg, shards=2, shard_index=shard, device="cpu", workers=0,
                                         batch_size=2, compact_batch_size=8, teacher_token_budget=128) == 0

    checkpoints = {}
    for objective in cfg["labels"]["objectives"]:
        for reader in ("tree", "flat"):
            router_cfg = _router_config(cfg, objective, reader)
            arrays, paths, identities, _, _ = routers._load_labels(router_cfg)
            assert set(arrays["partition"]) == {0, 1}
            assert len(arrays["gain"]) > 0
            assert np.isfinite(arrays["gain"]).all()
            assert identities[0]["objective"] == objective
            for shard, path in enumerate(paths):
                with h5py.File(path, "r") as f:
                    np.testing.assert_array_equal(f["attempted_context_ids"][:], np.arange(shard, 12, 2))
            assert routers.train_router(router_cfg) == 0
            checkpoint = run_dir(router_cfg, "router") / "step_00000001.pt"
            checkpoints[f"{objective}/{reader}"] = str(checkpoint)

    # A repeated completed shard is an exact reuse, with no payload rewrites.
    hashes = {str(label_experiment.output_path(cfg, obj, reader, "initial", 0)):
              file_sha256(label_experiment.output_path(cfg, obj, reader, "initial", 0))
              for obj in cfg["labels"]["objectives"] for reader in ("tree", "flat")}
    assert label_experiment.generate(cfg, shards=2, shard_index=0, device="cpu", workers=0) == 0
    assert hashes == {path: file_sha256(path) for path in hashes}

    cfg["labels"]["refresh_router_checkpoints"] = checkpoints
    for shard in range(2):
        assert label_experiment.generate(cfg, phase="refresh", shards=2, shard_index=shard, device="cpu",
                                         workers=0, batch_size=2, compact_batch_size=8,
                                         teacher_token_budget=128) == 0
    for objective in cfg["labels"]["objectives"]:
        for reader in ("tree", "flat"):
            router_cfg = _router_config(cfg, objective, reader, refresh=True)
            checkpoint = checkpoints[f"{objective}/{reader}"]
            assert routers.train_router(router_cfg, resume=checkpoint) == 0
            final = torch.load(run_dir(router_cfg, "router") / "last.pt", weights_only=False)
            assert final["step"] == 2
            assert final["objective"] == objective
            assert final["integration_steps"] == 8
            assert len(final["label_hashes"]) == 4


def test_spawned_label_worker_preserves_payload_and_plan_order(indexed):
    cfg, data, _, cache, hierarchy = indexed
    cfg = deepcopy(cfg)
    cfg["labels"].update(contexts=4, budgets=[8, 12, 16], candidates=2)
    items = LabelPlan(data, cfg).items()[:2]
    payloads = {}
    for workers in (0, 1):
        payloads[workers] = list(label_experiment.prepared_items(
            data, cfg, cache.root, hierarchy, items, refresh=False, workers=workers,
            prefetch_bytes=1024 * 1024,
        ))
    for serial, spawned in zip(payloads[0], payloads[1]):
        assert serial["context_id"] == spawned["context_id"]
        assert serial["partition"] == spawned["partition"]
        for name in ("query", "target", "valid", "text", "clocks", "current_state"):
            torch.testing.assert_close(serial[name], spawned[name], rtol=0, atol=0)
        for reader in ("tree", "flat"):
            for name in ("candidate", "costs"):
                torch.testing.assert_close(serial["readers"][reader][name], spawned["readers"][reader][name],
                                           rtol=0, atol=0)
            assert len(serial["readers"][reader]["after"]) == len(spawned["readers"][reader]["after"])
