"""Tracking must preserve staged router work, resume steps, and label provenance."""

from copy import deepcopy
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from stmem.config import run_dir
from stmem.memory.hierarchy import Archive
from stmem.memory.records import leaf_record
from stmem.models.router import LocalRouter
from stmem.training import gain_labels, routers
from stmem.training.logging_options import resolve_logging


class RecordingTracker:
    instances = []

    def __init__(self, cfg, directory, **kwargs):
        self.directory = Path(directory)
        self.options = kwargs
        self.logs = []
        self.summaries = []
        self.checkpoints = []
        self.finished = None
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if self.finished is None:
            self.finish("failed" if kind else "completed", 1 if kind else 0)

    def log(self, row, step=None):
        assert not any(isinstance(value, torch.Tensor) for value in row.values())
        self.logs.append((step, deepcopy(row)))

    def summary(self, row):
        self.summaries.append(deepcopy(row))

    def checkpoint(self, path, step):
        assert Path(path).is_file()
        self.checkpoints.append((Path(path), step))

    def finish(self, status="completed", exit_code=0):
        self.finished = status, exit_code


@pytest.fixture
def trackers(monkeypatch):
    RecordingTracker.instances = []
    monkeypatch.setattr(routers, "RunTracker", RecordingTracker)
    monkeypatch.setattr(gain_labels, "RunTracker", RecordingTracker)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(routers, "environment_manifest", lambda: {"device": "cpu-test"})
    return RecordingTracker.instances


def router_config(tmp_path):
    return {
        "seed": 4,
        "reader": "tree",
        "debug": True,
        "paths": {"run_root": str(tmp_path)},
        "model": {"width": 8, "router_width": 4},
        "training": {
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "steps": 2,
            "effective_batch": 4,
            "log_every": 1,
            "checkpoint_every": 1,
            "checkpoint_steps": [2],
        },
        "labels": {"gain_epsilon": 1e-6},
    }


def write_labels(cfg, *, refresh=False):
    path = gain_labels.label_path(cfg, refresh)
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(7 + refresh)
    width = cfg["model"]["width"]
    identity = {
        "teacher": "teacher-hash",
        "compact": "compact-hash",
        "data": "data-hash",
        "partition": {"training": "train", "validation": "validation"},
    }
    with h5py.File(path, "w") as f:
        f.attrs["complete"] = True
        f.attrs["identity"] = json.dumps(identity)
        f["query"] = generator.normal(size=(8, 2 * width + 27)).astype("f4")
        f["candidate"] = generator.normal(size=(8, 2 * width + 23)).astype("f4")
        f["gain"] = np.array([0.1, -0.2, 0.3, -0.4, 0.2, -0.5, 0.7, -0.1], dtype="f4")
        f["partition"] = np.array([0] * 4 + [1] * 4)
        f["context_id"] = np.repeat(np.arange(4), 2)
        f.create_dataset("episode", data=["train"] * 4 + ["val"] * 4, dtype=h5py.string_dtype())
    return path


def test_router_tracks_initial_refresh_and_skips_unchanged_replay(tmp_path, trackers):
    cfg = router_config(tmp_path)
    write_labels(cfg)
    assert routers.train_router(cfg) == 0
    first = trackers[-1]
    assert [step for step, _ in first.logs] == [1, 2]
    row = first.logs[-1][1]
    assert row["validation_rank_pairs"] == 2
    assert 0 <= row["validation_within_context_rank_agreement"] <= 1
    assert row["learning_rate"] == cfg["training"]["lr"]
    assert row["gradient_norm_before_clip"] >= 0
    assert row["gpu_peak_allocated_bytes_rank"] == row["gpu_peak_reserved_bytes_rank"] == 0
    assert row["cpu_process_rss_bytes"] > 0
    assert row["cpu_process_peak_rss_bytes"] > 0
    assert first.options["metadata"]["environment"] == {"device": "cpu-test"}
    assert len(first.checkpoints) == 2
    assert first.finished == ("completed", 0)
    checkpoint = run_dir(cfg, "router") / "last.pt"
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    assert state["environment"] == first.options["metadata"]["environment"]
    assert state["tracking"] == {}  # lightweight tracker doubles need not expose an identity
    assert (
        first.summaries[-1]["checkpoint/sha256"]
        == json.loads((checkpoint.parent / "complete.json").read_text())["checkpoint_sha256"]
    )
    refreshed = deepcopy(cfg)
    refreshed["labels"]["use_refresh"] = True
    refreshed["training"].update(steps=3, checkpoint_steps=[3])
    write_labels(refreshed, refresh=True)
    assert routers.train_router(refreshed, resume=checkpoint) == 0
    second = trackers[-1]
    assert second.directory == first.directory
    assert second.options["job_type"] == first.options["job_type"] == "router_tree"
    assert second.options["resume"]
    assert [step for step, _ in second.logs] == [3]
    assert second.logs[-1][1]["validation_rank_pairs"] == 4  # never cross initial/refresh draws
    assert len(second.options["metadata"]["label_hashes"]) == 2
    count = len(trackers)
    assert routers.train_router(refreshed, resume=checkpoint) == 0
    assert len(trackers) == count


def test_partial_router_tracks_stop_and_preserves_exact_resume(tmp_path, trackers):
    cfg = router_config(tmp_path / "stopped")
    cfg["training"].update(steps=3, checkpoint_steps=[3])
    write_labels(cfg)
    marker = tmp_path / "stop"
    marker.touch()
    assert routers.train_router(cfg, stop_file=marker) == 75
    assert trackers[-1].finished == ("checkpointed_partial", 75)
    assert trackers[-1].summaries[-1] == {"completed_steps": 1, "planned_steps": 3}
    marker.unlink()
    checkpoint = run_dir(cfg, "router") / "last.pt"
    assert routers.train_router(cfg, resume=checkpoint) == 0
    assert [step for step, _ in trackers[-1].logs] == [2, 3]
    control = deepcopy(cfg)
    control["paths"]["run_root"] = str(tmp_path / "control")
    write_labels(control)
    assert routers.train_router(control) == 0
    expected = torch.load(run_dir(control, "router") / "last.pt", weights_only=False, map_location="cpu")
    actual = torch.load(checkpoint, weights_only=False, map_location="cpu")
    for name in expected["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)


def _assert_identical_optimization(actual, expected):
    for name in expected["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], rtol=0, atol=0)
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    for parameter, state in expected["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(actual["optimizer"]["state"][parameter][key], value, rtol=0, atol=0)
    assert actual["step"] == expected["step"]


def test_router_frequent_scalars_preserve_validation_cadence_and_weights(tmp_path, monkeypatch, trackers):
    cfg = router_config(tmp_path / "control")
    cfg["training"].update(steps=5, checkpoint_steps=[5], log_every=3)
    write_labels(cfg)
    validation_calls = []
    original_forward = LocalRouter.forward

    def observed_forward(self, *args, **kwargs):
        if not self.training:
            validation_calls.append(True)
        return original_forward(self, *args, **kwargs)

    monkeypatch.setattr(LocalRouter, "forward", observed_forward)
    assert routers.train_router(cfg) == 0
    assert len(validation_calls) == 2
    control_logs = trackers[-1].logs
    assert [step for step, _ in control_logs] == [1, 3, 5]
    expected = torch.load(run_dir(cfg, "router") / "last.pt", weights_only=False, map_location="cpu")

    frequent = deepcopy(cfg)
    frequent["paths"]["run_root"] = str(tmp_path / "frequent")
    write_labels(frequent)
    before = deepcopy(frequent)
    validation_calls.clear()
    assert routers.train_router(frequent, log_every=1) == 0
    assert frequent == before
    assert len(validation_calls) == 2  # Frequent diagnostics do not rerun the held-out label evaluation.
    tracker = trackers[-1]
    assert [step for step, _ in tracker.logs] == [1, 2, 3, 4, 5]
    assert tracker.options["metadata"]["logging"] == {
        "log_every": 1,
        "progress_every_seconds": None,
        "attention_every": 0,
        "validation_every": 3,
    }
    validation_logs = {step: row for step, row in tracker.logs if "conditional_label_validation_loss" in row}
    assert set(validation_logs) == {3, 5}
    for step, row in control_logs:
        if step in validation_logs:
            for key in row:
                if key.startswith("validation_") or key == "conditional_label_validation_loss":
                    assert row[key] == validation_logs[step][key]
    actual = torch.load(run_dir(frequent, "router") / "last.pt", weights_only=False, map_location="cpu")
    _assert_identical_optimization(actual, expected)
    assert actual["config"] == before


def test_router_runtime_logging_override_can_change_on_exact_resume(tmp_path, trackers):
    cfg = router_config(tmp_path / "stopped")
    cfg["training"].update(steps=5, checkpoint_steps=[5], log_every=3)
    write_labels(cfg)
    marker = tmp_path / "stop"
    marker.touch()
    assert routers.train_router(cfg, stop_file=marker) == 75
    checkpoint = run_dir(cfg, "router") / "last.pt"
    marker.unlink()
    assert routers.train_router(cfg, resume=checkpoint, log_every=1) == 0
    assert [step for step, _ in trackers[-1].logs] == [2, 3, 4, 5]
    assert trackers[-1].options["metadata"]["logging"]["log_every"] == 1

    control = deepcopy(cfg)
    control["paths"]["run_root"] = str(tmp_path / "control")
    write_labels(control)
    assert routers.train_router(control) == 0
    expected = torch.load(run_dir(control, "router") / "last.pt", weights_only=False, map_location="cpu")
    actual = torch.load(checkpoint, weights_only=False, map_location="cpu")
    _assert_identical_optimization(actual, expected)
    assert actual["config"]["training"] == cfg["training"]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1.5", "no"])
def test_runtime_logging_rejects_invalid_update_interval(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        resolve_logging({"training": {"log_every": 50}}, log_every=value)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), "no"])
def test_runtime_logging_rejects_invalid_progress_interval(value):
    with pytest.raises(argparse.ArgumentTypeError, match="finite positive"):
        resolve_logging({"training": {"log_every": 50}}, progress_every_seconds=value)


@pytest.mark.parametrize("value", [-1, True, 1.5, "1.5", "nan", "inf", "no"])
def test_runtime_logging_rejects_invalid_attention_interval(value):
    with pytest.raises(argparse.ArgumentTypeError, match="nonnegative integer"):
        resolve_logging({"training": {"log_every": 50}}, attention_every=value)


def test_runtime_logging_resolution_preserves_scientific_configuration():
    cfg = {"training": {"log_every": 50}}
    before = deepcopy(cfg)
    assert resolve_logging(cfg) == {"log_every": 50, "progress_every_seconds": None, "attention_every": 10}
    assert resolve_logging(cfg, "1", "0.5") == {
        "log_every": 1, "progress_every_seconds": 0.5, "attention_every": 10,
    }
    assert resolve_logging(cfg, attention_every="0") == {
        "log_every": 50, "progress_every_seconds": None, "attention_every": 0,
    }
    assert resolve_logging(cfg, attention_every="25")["attention_every"] == 25
    assert cfg == before


def test_ranking_ignores_ties_and_never_compares_different_label_shards():
    left, right = routers._validation_ranking_pairs(
        [np.array([1, 1, 2, 2]), np.array([1, 1])],
        np.arange(6),
        torch.tensor([0.1, 0.8, 0.5, 0.5, 0.9, 0.2]),
        1e-6,
    )
    assert list(zip(left.tolist(), right.tolist())) == [(0, 1), (4, 5)]


def label_fixture(tmp_path, monkeypatch):
    cfg = router_config(tmp_path)
    cfg["model"]["decoder"] = "flow"
    cfg["labels"].update(
        seed=11, contexts=3, refresh_contexts=2, budgets=[4], candidates=1, exploration_fraction=1.0
    )
    teacher_path, compact_path = tmp_path / "teacher.pt", tmp_path / "compact.pt"
    teacher_path.write_bytes(b"teacher")
    compact_path.write_bytes(b"compact")
    monkeypatch.setattr(gain_labels, "dependency", lambda cfg, name: tmp_path / f"{name}.pt")
    monkeypatch.setattr(gain_labels, "validate_selection", lambda *args, **kwargs: None)
    decision = SimpleNamespace(
        target_chunk=torch.zeros(2, 8), target_valid=torch.ones(2, 8, dtype=torch.bool)
    )

    class Episode:
        task = "task"

        def __init__(self, i):
            self.episode_id = str(i)

        def __len__(self):
            return 3

        def decision(self, index, horizon):
            return decision

    class Data:
        synthetic = True
        tasks = ["task"]
        refs = [0, 1]
        by_task = {"task": [0, 1]}
        stride = 1
        horizon = 2
        identity = "tiny-data"

        def __init__(self, cfg):
            pass

        def episode(self, i):
            return Episode(i)

        def label_partition(self, i):
            return "train" if i == 0 else "validation"

    archive = Archive()
    archive.add_block(
        [
            leaf_record(
                str(i),
                torch.full((8,), float(i)),
                torch.tensor([float(i), 0.0, 0.0]),
                float(i),
                "execution",
                "visual",
            )
            for i in range(4)
        ]
    )
    evidence = SimpleNamespace(archive=archive)

    class Policy:
        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

        def routing_query(self, evidence, budget):
            return torch.zeros(43)

    monkeypatch.setattr(gain_labels, "TrainingData", Data)
    monkeypatch.setattr(gain_labels, "load_policy", lambda *args: (Policy(), {}))
    monkeypatch.setattr(
        gain_labels,
        "FrozenCache",
        lambda *args: SimpleNamespace(
            manifest={"identity": "tiny-cache"}, evidence=lambda decision: evidence
        ),
    )
    monkeypatch.setattr(
        gain_labels, "choose_records", lambda *args, **kwargs: SimpleNamespace(records=archive.roots)
    )
    monkeypatch.setattr(gain_labels, "paired_gains", lambda *args, **kwargs: ([torch.tensor(0.25)], {}))
    monkeypatch.setattr(
        gain_labels, "load_router", lambda *args: (LocalRouter(8, 4).eval().requires_grad_(False), {})
    )
    return cfg


def test_labels_track_separate_initial_refresh_and_skip_completed(tmp_path, monkeypatch, trackers):
    cfg = label_fixture(tmp_path, monkeypatch)
    destination = gain_labels.generate(cfg, "tree")
    initial = trackers[-1]
    assert initial.options["job_type"] == "label_tree_initial"
    assert initial.logs[-1][1]["gain_positive"] == 3
    assert initial.summaries[-1]["labels"] == 3
    assert initial.finished == ("completed", 0)
    count = len(trackers)
    assert gain_labels.generate(cfg, "tree") == destination
    assert len(trackers) == count
    checkpoint = tmp_path / "router.pt"
    checkpoint.write_bytes(b"frozen router")
    gain_labels.generate(cfg, "tree", refresh=True, router_checkpoint=checkpoint)
    refreshed = trackers[-1]
    assert refreshed.options["job_type"] == "label_tree_refresh"
    assert refreshed.directory != initial.directory
    assert refreshed.summaries[-1]["labels"] == 2
    assert not refreshed.checkpoints  # no label matrices or model weights uploaded


def test_label_partial_resume_logs_only_new_committed_contexts(tmp_path, monkeypatch, trackers):
    cfg = label_fixture(tmp_path, monkeypatch)
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated label interruption")
        return [torch.tensor(0.25)], {}

    monkeypatch.setattr(gain_labels, "paired_gains", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        gain_labels.generate(cfg, "tree")
    assert trackers[-1].finished == ("failed", 1)
    first = trackers[-1]
    gain_labels.generate(cfg, "tree")
    resumed = trackers[-1]
    assert resumed.directory == first.directory and resumed.options["resume"]
    assert [step for step, _ in resumed.logs] == [2, 3, 3]
    assert resumed.summaries[-1]["labels"] == 3
