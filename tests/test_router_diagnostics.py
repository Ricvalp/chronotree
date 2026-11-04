"""Router overfit diagnostics measure fitting without changing optimization."""

from copy import deepcopy

import h5py
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from stmem.config import run_dir
from stmem.models.router import LocalRouter
from stmem.training import routers
from stmem.training.checkpoints import file_sha256
from test_router_objectives import _config, _write_shards
from test_tracking_routers import RecordingTracker, _assert_identical_optimization


@pytest.fixture
def trackers(monkeypatch):
    RecordingTracker.instances = []
    monkeypatch.setattr(routers, "RunTracker", RecordingTracker)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(routers, "environment_manifest", lambda: {"device": "cpu-test"})
    return RecordingTracker.instances


def _checkpoint(cfg):
    return torch.load(run_dir(cfg, "router") / "last.pt", weights_only=False, map_location="cpu")


def _assert_rng_equal(actual, expected):
    for left, right in zip(actual["rng_by_rank"], expected["rng_by_rank"], strict=True):
        assert left["python"] == right["python"]
        assert left["numpy"][0] == right["numpy"][0]
        np.testing.assert_array_equal(left["numpy"][1], right["numpy"][1])
        assert left["numpy"][2:] == right["numpy"][2:]
        torch.testing.assert_close(left["torch"], right["torch"], rtol=0, atol=0)
        assert len(left["cuda"]) == len(right["cuda"])
        for a, b in zip(left["cuda"], right["cuda"], strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_full_dataset_diagnostics_preserve_weights_rng_and_validation(tmp_path, trackers):
    control = _config(tmp_path / "control")
    control["training"].update(steps=5, checkpoint_steps=[5], log_every=3)
    _write_shards(control)
    assert routers.train_router(control) == 0
    expected = _checkpoint(control)
    expected_rows = dict(trackers[-1].logs)

    diagnostic = deepcopy(control)
    diagnostic["paths"]["run_root"] = str(tmp_path / "diagnostic")
    diagnostic["training"]["router_diagnostics"] = True
    assert routers.train_router(diagnostic) == 0
    actual = _checkpoint(diagnostic)
    _assert_identical_optimization(actual, expected)
    _assert_rng_equal(actual, expected)
    tracker = trackers[-1]
    assert tracker.logs[0][0] == 0
    assert tracker.options["metadata"]["router_diagnostics"] is True
    assert tracker.options["name"] == "seed4-router-overfit-tree-velocity"
    rows = dict(tracker.logs)
    for step, expected_row in expected_rows.items():
        for key, value in expected_row.items():
            if key.startswith("validation_") or key in (
                "conditional_label_validation_loss", "training_gain_loss", "gradient_norm_before_clip"
            ):
                assert rows[step][key] == value
    assert {step for step, row in tracker.logs if "training_full_gain_loss" in row} == {0, 3, 5}


def test_full_training_metrics_match_all_training_rows(tmp_path, trackers):
    cfg = _config(tmp_path)
    cfg["training"].update(router_diagnostics=True, steps=2, checkpoint_steps=[2])
    _write_shards(cfg)
    assert routers.train_router(cfg) == 0
    final = trackers[-1].logs[-1][1]
    state = _checkpoint(cfg)
    model = LocalRouter(cfg["model"]["width"], cfg["model"]["router_width"]).eval()
    model.load_state_dict(state["model"])
    data, *_ = routers._load_labels(cfg)
    train = data["partition"] == 0
    with torch.inference_mode():
        prediction = model(torch.from_numpy(data["query"][train]), torch.from_numpy(data["candidate"][train]))
    gain = torch.from_numpy(data["gain"][train])
    expected_loss = F.smooth_l1_loss(prediction, gain / state["target_scale"]).item()
    expected_sign = ((prediction > 0) == (gain > 0)).float().mean().item()
    # This fixture has two candidates in each of four distinct training contexts.
    contexts = data["context_id"][train]
    agreements = []
    for context in np.unique(contexts):
        left, right = np.flatnonzero(contexts == context)
        agreements.append(float((prediction[left] - prediction[right]).sign() == (gain[left] - gain[right]).sign()))
    assert final["training_full_gain_loss"] == pytest.approx(expected_loss, abs=1e-6)
    assert final["training_sign_agreement"] == expected_sign
    assert final["training_within_context_rank_agreement"] == np.mean(agreements)
    assert final["training_rank_pairs"] == 4


def test_baselines_fit_training_labels_only(tmp_path, trackers):
    cfg = _config(tmp_path / "original")
    cfg["training"].update(router_diagnostics=True)
    _write_shards(cfg)
    assert routers.train_router(cfg) == 0
    original_baseline = trackers[-1].logs[0][1]
    expected = _checkpoint(cfg)

    changed = _config(tmp_path / "changed_validation")
    changed["training"]["router_diagnostics"] = True
    for path in _write_shards(changed):
        with h5py.File(path, "r+") as handle:
            values = handle["gain"][:]
            valid = handle["partition"][:] == 1
            values[valid] = values[valid] * 100 + 50
            handle["gain"][:] = values
    assert routers.train_router(changed) == 0
    changed_baseline = trackers[-1].logs[0][1]
    for key in ("baseline/training_zero_gain_loss", "baseline/training_median_gain_loss"):
        assert changed_baseline[key] == original_baseline[key]
    assert changed_baseline["baseline/validation_zero_gain_loss"] > original_baseline["baseline/validation_zero_gain_loss"]
    assert changed_baseline["baseline/validation_median_gain_loss"] > original_baseline["baseline/validation_median_gain_loss"]
    _assert_identical_optimization(_checkpoint(changed), expected)
    _assert_rng_equal(_checkpoint(changed), expected)


def test_separate_output_continuation_logs_source_step_and_preserves_original(tmp_path, trackers):
    source = _config(tmp_path / "source")
    source["training"].update(steps=2, checkpoint_steps=[2], log_every=3)
    _write_shards(source)
    assert routers.train_router(source) == 0
    initial = run_dir(source, "router") / "last.pt"
    original_hash = file_sha256(initial)
    source_metrics = (run_dir(source, "router") / "metrics.jsonl").read_bytes()

    continuation = deepcopy(source)
    continuation["paths"]["run_root"] = str(tmp_path / "continuation")
    continuation["training"].update(router_diagnostics=True, steps=5, checkpoint_steps=[5])
    assert routers.train_router(continuation, resume=initial) == 0
    assert trackers[-1].logs[0][0] == 2
    assert trackers[-1].options["metadata"]["resume_step"] == 2
    assert trackers[-1].directory == run_dir(continuation, "router")
    assert file_sha256(initial) == original_hash
    assert (run_dir(source, "router") / "metrics.jsonl").read_bytes() == source_metrics

    control = deepcopy(source)
    control["paths"]["run_root"] = str(tmp_path / "uninterrupted")
    control["training"].update(steps=5, checkpoint_steps=[5])
    assert routers.train_router(control) == 0
    _assert_identical_optimization(_checkpoint(continuation), _checkpoint(control))
    _assert_rng_equal(_checkpoint(continuation), _checkpoint(control))


def test_resume_baselines_use_checkpoint_target_scale(tmp_path, trackers):
    source = _config(tmp_path / "source")
    _write_shards(source)
    assert routers.train_router(source) == 0
    state = _checkpoint(source)
    # A retained scale can differ from the current median after a label refresh.
    # Exercise that condition without changing any frozen labels or partition.
    state["target_scale"] = 2.0
    initial = tmp_path / "retained_scale.pt"
    torch.save(state, initial)
    cfg = deepcopy(source)
    cfg["paths"]["run_root"] = str(tmp_path / "continued")
    cfg["training"].update(router_diagnostics=True, steps=3, checkpoint_steps=[3])
    assert routers.train_router(cfg, resume=initial) == 0
    row = trackers[-1].logs[0][1]
    data, *_ = routers._load_labels(cfg)
    target = torch.from_numpy(data["gain"][data["partition"] == 0]) / 2.0
    assert row["target_scale_training_only"] == 2.0
    assert row["baseline/training_zero_gain_loss"] == pytest.approx(
        F.smooth_l1_loss(torch.zeros_like(target), target).item()
    )
    assert row["baseline/training_median_gain_loss"] == pytest.approx(
        F.smooth_l1_loss(target.median().expand_as(target), target).item()
    )


def test_training_ranking_with_only_tied_targets_is_omitted(tmp_path, trackers):
    cfg = _config(tmp_path)
    cfg["training"]["router_diagnostics"] = True
    for path in _write_shards(cfg):
        with h5py.File(path, "r+") as handle:
            gain = handle["gain"][:]
            gain[handle["partition"][:] == 0] = 0.3
            handle["gain"][:] = gain
    assert routers.train_router(cfg) == 0
    for _, row in trackers[-1].logs:
        if "training_full_gain_loss" not in row:
            continue
        assert row["training_rank_pairs"] == 0
        assert "training_within_context_rank_agreement" not in row
        assert np.isfinite(row["training_full_gain_loss"])
        assert row["validation_rank_pairs"] == 4
        assert "validation_within_context_rank_agreement" in row


@pytest.mark.parametrize("value", [1, 0, "true", "false", None, {}, []])
def test_router_diagnostics_requires_boolean(tmp_path, value, trackers):
    cfg = _config(tmp_path)
    cfg["training"]["router_diagnostics"] = value
    _write_shards(cfg)
    with pytest.raises(ValueError, match="router_diagnostics.*boolean"):
        routers.train_router(cfg)
