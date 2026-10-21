"""Evaluation telemetry must preserve saved scientific results and denominators."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stmem import evaluate, report
from stmem.config import digest, load_config
from stmem.training.checkpoints import atomic_json, file_sha256


class FakeTracker:
    instances = []

    def __init__(self, cfg, directory, **kwargs):
        self.cfg = copy.deepcopy(cfg)
        self.directory = Path(directory)
        self.options = kwargs
        self.logs = []
        self.summaries = {}
        self.error = None
        self.exited = False
        self.instances.append(self)

    def __enter__(self):
        return self

    def log(self, values, step=None):
        # Mirrors the actual tracker contract: no tensors or non-JSON payloads.
        json.dumps(values, allow_nan=False)
        if "evaluation/progress_percent" in values:
            assert len(list((self.directory / "episodes").glob("*.json"))) == step, (
                "Episode telemetry must follow durable result persistence"
            )
        self.logs.append((copy.deepcopy(values), step))

    def summary(self, values):
        json.dumps(values, allow_nan=False)
        self.summaries.update(copy.deepcopy(values))

    def __exit__(self, exc_type, exc, traceback):
        self.error = exc_type
        self.exited = True
        return False


@pytest.fixture(autouse=True)
def reset_trackers():
    FakeTracker.instances = []


def test_evaluation_scalars_keep_denominators_and_cost_semantics():
    records = [
        {
            "task": "VideoUnmask",
            "episode_idx": 0,
            "status": "success",
            "runtime_seconds": 10,
            "accelerator_memory": {"peak_allocated_bytes": 20},
            "decisions": [
                {
                    "complete_online_seconds": 1,
                    "archive_payload_bytes": 100,
                    "routing_transferred_bytes": 10,
                    "nodes_scored": 2,
                    "encoding_calls": 1,
                    "component_seconds": {"encoding": 0.1},
                },
                {
                    "complete_online_seconds": 3,
                    "archive_payload_bytes": 150,
                    "routing_transferred_bytes": 20,
                    "nodes_scored": 4,
                    "encoding_calls": 3,
                    "component_seconds": {"encoding": 0.2},
                },
            ],
        },
        {
            "task": "VideoUnmask",
            "episode_idx": 1,
            "status": "policy_error",
            "runtime_seconds": 20,
            "accelerator_memory": {"peak_allocated_bytes": 50},
            "decisions": [
                {
                    "complete_online_seconds": 5,
                    "archive_payload_bytes": 120,
                    "routing_transferred_bytes": 7,
                    "nodes_scored": 6,
                    "encoding_calls": 2,
                    "component_seconds": {"encoding": 0.5},
                }
            ],
        },
    ]
    values = evaluate.evaluation_tracking_scalars(records, [("VideoUnmask", i) for i in range(3)])
    assert values["evaluation/macro_success"] == pytest.approx(1 / 3)
    assert values["evaluation/overall/missing"] == 1
    assert values["evaluation/overall/errors"] == 1
    assert values["evaluation/decisions/complete_online_seconds/mean"] == 3
    assert values["evaluation/decisions/complete_online_seconds/p95"] == pytest.approx(4.8)
    assert values["evaluation/latency/cold_first_decision_seconds/count"] == 2
    assert values["evaluation/latency/cold_first_decision_seconds/mean"] == 3
    assert values["evaluation/latency/warm_after_first_decision_seconds/count"] == 1
    assert values["evaluation/decisions/archive_payload_bytes/max"] == 150
    assert "evaluation/decisions/archive_payload_bytes/total" not in values
    assert values["evaluation/decisions/routing_transferred_bytes/total"] == 37
    assert values["evaluation/decisions/nodes_scored/total"] == 12
    assert values["evaluation/components/encoding_seconds/total"] == pytest.approx(0.8)
    assert values["evaluation/episodes/encoding_calls_at_last_decision/total"] == 5
    assert values["evaluation/episodes/runtime_seconds/total"] == 30
    assert values["evaluation/accelerator_memory/peak_allocated_bytes/max"] == 50
    assert "evaluation/episodes/startup_seconds/mean" not in values
    assert not any("peak_reserved" in key for key in values)
    scalars = evaluate.numeric_scalars(
        {"nested": {"good": np.float64(1.25), "bad": float("nan")}, "tensor": torch.ones(())}, "x"
    )
    assert scalars == {"x/nested/good": 1.25}


@pytest.fixture
def evaluation_cli(tmp_path, monkeypatch):
    cfg = load_config("configs/debug/rollout.yaml")
    cfg["data"]["tasks"] = ["VideoUnmask"]
    cfg["evaluation"]["episode_count"] = 2
    cfg["tracking"] = {"mode": "offline", "enabled": True}
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"Mock checkpoint: no weights needed for telemetry tests")
    output = tmp_path / "evaluation"
    source = {"git_commit": "original", "source_sha256": "stable-implementation", "dirty": False}
    monkeypatch.setattr(evaluate, "load_config", lambda _: copy.deepcopy(cfg))
    monkeypatch.setattr(evaluate, "RunTracker", FakeTracker)
    monkeypatch.setattr(evaluate, "environment_manifest", lambda: {"device": "mock-cpu"})
    monkeypatch.setattr(evaluate, "source_revision", lambda: copy.deepcopy(source))
    monkeypatch.setattr(evaluate, "load_policy", lambda *_: (object(), {"config": copy.deepcopy(cfg)}))
    monkeypatch.setattr(evaluate, "OnlinePolicy", lambda *_args, **_kwargs: object())
    adapters = []

    class Adapter:
        def __init__(self, *args, **kwargs):
            self.closed = False
            adapters.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr(evaluate, "RoboMMEAdapter", Adapter)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stmem.evaluate",
            "--config",
            "ignored.yaml",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
    )
    return SimpleNamespace(cfg=cfg, checkpoint=checkpoint, output=output, source=source, adapters=adapters)


def test_evaluation_logging_and_resume_do_not_change_results(evaluation_cli, monkeypatch):
    fixture = evaluation_cli

    def run_episode(*args, **kwargs):
        if kwargs["episode_idx"] == 1:
            raise RuntimeError("Mock simulator startup failure")
        return {
            "task": kwargs["task"],
            "episode_idx": 0,
            "status": "success",
            "runtime_seconds": 12.3,
            "decisions": [{"complete_online_seconds": 0.45, "archive_payload_bytes": 1234}],
            "actions": [],
        }

    monkeypatch.setattr(evaluate, "run_episode", run_episode)
    evaluate.main()
    first = FakeTracker.instances[0]
    assert first.options["job_type"] == "evaluation"
    assert first.options["resume"] is False
    metadata = first.options["metadata"]["evaluation_manifest"]
    assert metadata["environment"] == {"device": "mock-cpu"}
    assert metadata["source"]["source_sha256"] == "stable-implementation"
    assert metadata["checkpoints"][str(fixture.checkpoint.resolve())] == file_sha256(fixture.checkpoint)
    assert first.cfg == fixture.cfg
    assert first.exited and first.error is None
    assert all(adapter.closed for adapter in fixture.adapters)
    assert first.logs == [
        ({"evaluation/success_rate_so_far_percent": None, "evaluation/progress_percent": 0.0}, 0),
        ({"evaluation/success_rate_so_far_percent": 100.0, "evaluation/progress_percent": 50.0}, 1),
        ({"evaluation/success_rate_so_far_percent": 50.0, "evaluation/progress_percent": 100.0}, 2),
    ]
    last = first.summaries["evaluation/last_episode"]
    assert last["task"] == "VideoUnmask" and last["episode_idx"] == 1
    assert last["status"] == "infrastructure_error"
    assert last["error"] == "RuntimeError: Mock simulator startup failure"
    summary = json.loads((fixture.output / "summary.json").read_text())
    assert summary["overall"]["planned"] == summary["overall"]["recorded"] == 2
    assert summary["overall"]["successes"] == summary["overall"]["errors"] == 1
    diagnostics = json.loads((fixture.output / "diagnostics.json").read_text())
    assert diagnostics["evaluation/macro_success"] == 0.5
    assert diagnostics["evaluation/decisions/complete_online_seconds/total"] == 0.45
    assert diagnostics["evaluation/episodes/runtime_seconds/total"] == 12.3
    assert not any("/decisions/" in key or "/tasks/" in key for key in first.summaries)
    snapshots = {path: path.read_bytes() for path in fixture.output.rglob("*.json")}
    fixture.cfg["tracking"]["mode"] = "disabled"
    fixture.source.update(git_commit="new-commit-same-source", dirty=True)

    def forbidden_rerun(*args, **kwargs):
        pytest.fail("Saved episodes must be reused on resume")

    monkeypatch.setattr(evaluate, "run_episode", forbidden_rerun)
    evaluate.main()
    resumed = FakeTracker.instances[1]
    assert resumed.options["resume"] is True
    assert resumed.logs == [first.logs[-1]]
    progress = resumed.summaries["evaluation/progress"]
    assert progress["new_episodes"] == 0
    assert progress["reused_episodes"] == progress["episodes_recorded"] == progress["episodes_planned"] == 2
    assert progress["errors"] == 1
    for path, contents in snapshots.items():
        assert path.read_bytes() == contents


def test_evaluation_resume_progress_includes_saved_episodes_after_a_gap(evaluation_cli, monkeypatch):
    fixture = evaluation_cli
    fixture.cfg["evaluation"]["video_cap"] = 1

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(evaluate, "run_episode", interrupt)
    with pytest.raises(KeyboardInterrupt):
        evaluate.main()
    saved_path = fixture.output / "episodes/VideoUnmask_001.json"
    atomic_json({"task": "VideoUnmask", "episode_idx": 1, "status": "timeout"}, saved_path)
    saved_bytes = saved_path.read_bytes()

    def run_episode(*args, **kwargs):
        assert kwargs["episode_idx"] == 0
        assert kwargs["video_path"] == fixture.output / "videos/VideoUnmask_000.mp4"
        # Progress already includes the later durable timeout before doing more work.
        assert FakeTracker.instances[-1].logs == [
            ({"evaluation/success_rate_so_far_percent": 0.0, "evaluation/progress_percent": 50.0}, 1)
        ]
        return {"task": kwargs["task"], "episode_idx": 0, "status": "success"}

    monkeypatch.setattr(evaluate, "run_episode", run_episode)
    evaluate.main()
    tracker = FakeTracker.instances[-1]
    assert tracker.logs[-1] == (
        {"evaluation/success_rate_so_far_percent": 50.0, "evaluation/progress_percent": 100.0},
        2,
    )
    assert tracker.summaries["evaluation/progress"]["new_episodes"] == 1
    assert tracker.summaries["evaluation/progress"]["reused_episodes"] == 1
    assert saved_path.read_bytes() == saved_bytes


def test_evaluation_dashboard_stays_two_metrics_across_all_tasks(evaluation_cli, monkeypatch):
    fixture = evaluation_cli
    fixture.cfg["data"]["tasks"] = load_config("configs/base.yaml")["data"]["tasks"]
    monkeypatch.setattr(
        evaluate,
        "run_episode",
        lambda *args, **kw: {
            "task": kw["task"],
            "episode_idx": kw["episode_idx"],
            "status": "success" if kw["episode_idx"] == 0 else "policy_error",
            "decisions": [{"component_seconds": {"encoding": 0.1}, "archive_payload_bytes": 1234}],
        },
    )
    evaluate.main()
    tracker = FakeTracker.instances[0]
    assert len(tracker.logs) == 33
    for row, step in tracker.logs:
        assert set(row) == {"evaluation/success_rate_so_far_percent", "evaluation/progress_percent"}
        assert row["evaluation/progress_percent"] == 100 * step / 32
    assert tracker.logs[-1][0]["evaluation/success_rate_so_far_percent"] == 50.0
    summary = json.loads((fixture.output / "summary.json").read_text())
    assert len(summary["tasks"]) == 16 and summary["macro_success"] == 0.5
    assert summary["overall"]["errors"] == 16


def test_empty_evaluation_shard_has_no_success_estimate(evaluation_cli, monkeypatch):
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--shard-index", "2", "--num-shards", "3"])
    evaluate.main()
    assert not evaluation_cli.adapters
    assert FakeTracker.instances[0].logs == [
        ({"evaluation/success_rate_so_far_percent": None, "evaluation/progress_percent": 100.0}, 0)
    ]
    summary = json.loads((evaluation_cli.output / "summary.json").read_text())
    assert summary["macro_success"] is None
    assert summary["overall"]["planned"] == summary["overall"]["recorded"] == 0


@pytest.mark.parametrize("failure", [ValueError, KeyboardInterrupt])
def test_evaluation_tracker_finalizes_on_load_or_episode_failure(evaluation_cli, monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure("Intentional diagnostic failure")

    monkeypatch.setattr(evaluate, "load_policy" if failure is ValueError else "run_episode", fail)
    with pytest.raises(failure):
        evaluate.main()
    tracker = FakeTracker.instances[0]
    assert tracker.exited and tracker.error is failure
    assert not (evaluation_cli.output / "complete.json").exists()
    assert all(adapter.closed for adapter in evaluation_cli.adapters)


def test_interruption_preserved_if_simulator_cleanup_also_fails(evaluation_cli, monkeypatch):
    def interrupt(adapter, *args, **kwargs):
        def close_failure():
            raise RuntimeError("Cleanup failed after interruption")

        adapter.close = close_failure
        raise KeyboardInterrupt("Original interruption")

    monkeypatch.setattr(evaluate, "run_episode", interrupt)
    with pytest.raises(KeyboardInterrupt, match="Original interruption"):
        evaluate.main()
    tracker = FakeTracker.instances[0]
    assert tracker.error is KeyboardInterrupt
    assert tracker.summaries["evaluation/interrupted_close_error"] == (
        "RuntimeError: Cleanup failed after interruption"
    )
    assert not (evaluation_cli.output / "complete.json").exists()


def test_report_cli_tracks_partial_plans_without_fabricating_missing_results(tmp_path, monkeypatch):
    directory = tmp_path / "seed0/evaluation/val/TREE/shard0"
    manifest_path = directory / "manifest.json"
    input_config = {"seed": 0, "model": {"budget": 64}, "data": {"synthetic": False}}
    input_source = {"source_sha256": "evaluation-source"}
    atomic_json(
        {
            "all_planned": [("VideoUnmask", 0), ("VideoUnmask", 1)],
            "config": input_config,
            "source": input_source,
            "checkpoints": {"compact.pt": "checkpoint-sha256"},
        },
        manifest_path,
    )
    atomic_json(
        {
            "task": "VideoUnmask",
            "episode_idx": 0,
            "status": "success",
            "model_load_seconds": 4,
            "decisions": [{"complete_online_seconds": 0.2, "archive_payload_bytes": 100}],
        },
        directory / "episodes/VideoUnmask_000.json",
    )
    monkeypatch.setattr(report, "RunTracker", FakeTracker)
    monkeypatch.setattr(report, "load_config", lambda _: {"seed": 0, "tracking": {"mode": "offline"}})
    monkeypatch.setattr(report, "source_revision", lambda: {"source_sha256": "test-source"})
    monkeypatch.setattr(report, "environment_manifest", lambda: {"device": "mock-cpu"})
    monkeypatch.setattr(sys, "argv", ["stmem.report", "--run-root", str(tmp_path)])
    report.main()
    tracker = FakeTracker.instances[0]
    assert tracker.options["job_type"] == "report"
    assert tracker.options["metadata"]["run_root"] == str(tmp_path.resolve())
    assert tracker.exited and tracker.error is None
    values = dict(key_value for row, _ in tracker.logs for key_value in row.items())
    assert values["report/seed0/TREE/overall/planned"] == 2
    assert values["report/seed0/TREE/overall/missing"] == 1
    assert values["report/seed0/TREE/macro_success"] == 0.5
    assert values["report/seed0/TREE/startup/model_load_seconds_per_shard/count"] == 1
    assert "report/seed0/CURRENT/macro_success" not in values
    assert tracker.summaries["report/input_manifest_sha256"] == {
        str(manifest_path.relative_to(tmp_path)): file_sha256(manifest_path)
    }
    assert tracker.summaries["report/input_configurations"] == {digest(input_config): input_config}
    provenance = tracker.summaries["report/input_provenance"][str(manifest_path.relative_to(tmp_path))]
    assert provenance["config_sha256"] == digest(input_config)
    assert provenance["source"] == input_source
    assert provenance["checkpoints"] == {"compact.pt": "checkpoint-sha256"}
    assert tracker.cfg["paths"]["run_root"] == str(tmp_path.resolve())
    artifact = json.loads((tmp_path / "reports/val/summary.json").read_text())
    assert artifact["seeds"]["seed0"]["methods"]["TREE"]["status"] == "partial"
    assert artifact["seeds"]["seed0"]["methods"]["CURRENT"]["macro_success"] is None


def test_report_seed_directory_uses_actual_seed_and_parent_group(tmp_path, monkeypatch):
    seed_root = tmp_path / "seed3"
    seed_root.mkdir()
    monkeypatch.setattr(report, "RunTracker", FakeTracker)
    monkeypatch.setattr(report, "load_config", lambda _: {"seed": 0, "paths": {"run_root": "runs"}})
    monkeypatch.setattr(report, "source_revision", lambda: {})
    monkeypatch.setattr(report, "environment_manifest", lambda: {})
    monkeypatch.setattr(sys, "argv", ["stmem.report", "--run-root", str(seed_root)])
    report.main()
    tracker = FakeTracker.instances[0]
    assert tracker.cfg["seed"] == 3
    assert tracker.cfg["paths"]["run_root"] == str(tmp_path.resolve())
    assert tracker.summaries["report/training_seeds"] == ["seed3"]


def test_six_smoke_readers_have_explicit_synthetic_tracking(tmp_path, monkeypatch):
    from stmem import config, tracking
    from stmem.training import data, runtime
    from stmem.evaluation import runner

    path = Path(__file__).resolve().parents[1] / "scripts/smoke.py"
    spec = importlib.util.spec_from_file_location("smoke_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = {"seed": 0, "data": {"synthetic": True}, "model": {"horizon": 8, "budget": 32}}
    episode = SimpleNamespace(conditioning=[], execution=list(range(8)), goal="synthetic diagnostic")
    monkeypatch.setattr(config, "load_config", lambda _: cfg)
    monkeypatch.setattr(config, "source_revision", lambda: {"source_sha256": "test-source"})
    monkeypatch.setattr(data, "TrainingData", lambda _: SimpleNamespace(episode=lambda _: episode))
    monkeypatch.setattr(runtime, "environment_manifest", lambda: {"device": "mock-cpu"})
    monkeypatch.setattr(runtime, "load_policy", lambda *_: (SimpleNamespace(device="cpu"), {}))
    monkeypatch.setattr(runtime, "load_router", lambda *_: (object(), {}))
    monkeypatch.setattr(tracking, "RunTracker", FakeTracker)

    class OnlinePolicy:
        def __init__(self, *args, **kwargs):
            self.observations = 0

        def reset(self, result, seed):
            assert result.current == 0 and seed == 7

        def observe(self, frame):
            self.observations += 1

        def decide(self):
            assert self.observations == 6
            assert len(FakeTracker.instances[-1].logs) == 0
            return torch.zeros(8, 8), {"decision_seconds": 0.2, "archive_payload_bytes": 1024}

    monkeypatch.setattr(runner, "OnlinePolicy", OnlinePolicy)
    for stage in ("teacher", "compact", "perceiver", "router_tree", "router_flat"):
        checkpoint = tmp_path / "seed0" / stage / "last.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(stage.encode())
    module.evaluate_readers(tmp_path, "ignored.yaml")
    assert len(FakeTracker.instances) == 6
    assert {run.options["metadata"]["method"] for run in FakeTracker.instances} == set(report.METHODS)
    for run in FakeTracker.instances:
        assert run.options["job_type"] == "evaluation_smoke"
        assert run.options["metadata"]["synthetic_debug_only"] is True
        assert run.summaries["diagnostic/finite_actions"] == 1
        assert run.summaries["diagnostic/decision_seconds"] == 0.2
        assert run.summaries["diagnostic/status"] == "passed"
        assert run.exited and run.error is None
        assert not any("success" in key for row, _ in run.logs for key in row)
    artifact = json.loads((tmp_path / "smoke_complete.json").read_text())
    assert artifact["status"] == "complete" and artifact["synthetic_debug_only"] is True
    assert set(artifact["readers"]) == set(report.METHODS)
