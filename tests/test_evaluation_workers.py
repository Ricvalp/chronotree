"""Synthetic CLI checks for worker telemetry suppression and global video caps."""

import json
import sys

import pytest

from stmem import evaluate
from stmem.evaluation.workload import video_episodes
from stmem.tracking import RunTracker
from test_tracking_evaluation import evaluation_cli as _evaluation_cli

evaluation_cli = _evaluation_cli  # Reuse the CLI fixture without importing its tests.


def test_video_cap_is_shared_across_shards_and_noise_seeds_are_stable(evaluation_cli, monkeypatch):
    fixture = evaluation_cli
    fixture.cfg["data"]["tasks"] = ["VideoUnmask", "PickXtimes"]
    fixture.cfg["evaluation"]["episode_count"] = 4
    fixture.cfg["evaluation"]["video_cap"] = 3
    base = list(sys.argv)
    calls = []

    def episode(*args, **kwargs):
        calls.append(kwargs)
        return {key: kwargs[key] for key in ("task", "episode_idx", "split")} | {
            "status": "success", "method": "FULL", "runtime_seconds": 0.01}

    monkeypatch.setattr(evaluate, "run_episode", episode)
    for index in range(4):
        output = fixture.output / f"shard{index}"
        monkeypatch.setattr(sys, "argv", [*base, "--output", str(output), "--shard-index", str(index),
                                        "--num-shards", "4"])
        evaluate.main()
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["video_episodes"] == [list(pair) for pair in video_episodes(fixture.cfg)]
    assert len(calls) == 8
    assert len({(r["task"],r["episode_idx"]) for r in calls}) == 8
    assert {(r["task"],r["episode_idx"]) for r in calls if r["video_path"]} == set(video_episodes(fixture.cfg))
    for row in calls:
        assert row["seed"] == fixture.cfg["evaluation"]["seed"] + fixture.cfg["data"]["tasks"].index(row["task"])*1000 + row["episode_idx"]


def test_worker_suppression_never_starts_wandb(evaluation_cli, monkeypatch):
    fixture = evaluation_cli
    monkeypatch.setattr(evaluate, "RunTracker", RunTracker)
    monkeypatch.setenv("WANDB_MODE", "online")
    # A disabled worker tracker must not even import the SDK.
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setattr(evaluate, "run_episode", lambda *a, **kw: {
        "task": kw["task"], "episode_idx": kw["episode_idx"], "split": "val",
        "method": "FULL", "status": "infrastructure_error", "error": "synthetic injected failure"})
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--no-tracking", "--video-cap", "0"])
    evaluate.main()
    assert not (fixture.output / "tracking.json").exists()
    summary = json.loads((fixture.output / "summary.json").read_text())
    assert summary["overall"]["errors"] == 2
    for path in (fixture.output / "episodes").glob("*.json"):
        row = json.loads(path.read_text())
        assert row["evaluation_manifest_sha256"] and row["checkpoint_sha256"] and row["noise_seed"]


def test_zero_episode_count_is_rejected_instead_of_falling_back(evaluation_cli, monkeypatch):
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--episode-count", "0"])
    with pytest.raises(ValueError, match="1–50"):
        evaluate.main()
