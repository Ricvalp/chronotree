"""Synthetic outcome fixtures; no simulator or benchmark score is produced."""

from copy import deepcopy
import json

import pytest

from stmem.evaluation.aggregation import aggregate_shards
from stmem.evaluation.workload import planned_episodes, video_episodes
from stmem.training.checkpoints import atomic_json, file_sha256


def parent_fixture(tmp_path, *, shards=4, cap=3):
    cfg = {"seed": 0, "data": {"tasks": ["PickXtimes", "VideoUnmask"]},
           "evaluation": {"episode_count": 4, "video_cap": cap, "seed": 1701},
           "tracking": {"enabled": True}}
    plan = planned_episodes(cfg)
    parent = {"config": cfg, "all_planned": plan, "num_shards": shards,
              "source": {"source_sha256": "synthetic-source"}, "checkpoints": {"/checkpoint": "weights"},
              "method": "FULL", "split": "val", "selection_family": None,
              "intervention": "none", "budget": 128, "targeting_diagnostics": False,
              "video_episodes": video_episodes(cfg), "policy_checkpoint_sha256": "weights"}
    tmp_path.mkdir(exist_ok=True)
    return parent


def write_shard(output, parent, index, *, statuses=None, done=True):
    child = deepcopy(parent)
    child["planned"] = parent["all_planned"][index::parent["num_shards"]]
    child["shard_index"] = index
    child["config"]["tracking"]["enabled"] = False
    root = output / "shards" / f"shard{index:03d}"
    root.mkdir(parents=True)
    atomic_json(child, root / "manifest.json")
    identity = file_sha256(root / "manifest.json")
    for position, (task, episode) in enumerate(child["planned"]):
        seed = parent["config"]["evaluation"]["seed"] + parent["config"]["data"]["tasks"].index(task)*1000 + episode
        status = statuses[position] if statuses else "success"
        row = {"task": task, "episode_idx": episode, "method": "FULL", "split": "val", "status": status,
               "training_seed": 0, "noise_seed": seed, "checkpoint_sha256": "weights",
               "evaluation_manifest_sha256": identity}
        atomic_json(row, root / "episodes" / f"{task}_{episode:03d}.json")
        if (task, episode) in video_episodes(parent["config"]):
            video = root / "videos" / f"{task}_{episode:03d}.mp4"
            video.parent.mkdir(exist_ok=True)
            video.write_bytes(b"synthetic link fixture, not a video")
    if done:
        atomic_json({"status": "complete", "manifest_sha256": identity,
                     "planned": len(child["planned"]), "recorded": len(child["planned"])}, root / "complete.json")
    return root


def test_missing_entire_workers_do_not_shrink_denominator(tmp_path):
    parent = parent_fixture(tmp_path)
    write_shard(tmp_path, parent, 0, statuses=["success", "infrastructure_error"])
    records, summary = aggregate_shards(tmp_path, parent)
    assert len(records) == 2
    assert summary["overall"]["planned"] == 8 and summary["overall"]["missing"] == 6
    assert summary["overall"]["errors"] == 1 and summary["overall"]["success_rate"] == 1/8
    assert summary["worker_completion"]["complete"] == 1 and summary["status"] == "partial"


def test_aggregation_covers_each_episode_once_and_limits_videos_globally(tmp_path):
    parent = parent_fixture(tmp_path)
    for index in range(4):
        write_shard(tmp_path, parent, index)
    for _ in range(2):  # repeated monitoring reuses links and completed evidence
        records, summary = aggregate_shards(tmp_path, parent)
        assert len(records) == 8 and summary["status"] == "complete"
        assert summary["worker_completion"]["complete"] == 4
        assert [(r["task"], r["episode_idx"]) for r in records] == parent["all_planned"]
    videos = list((tmp_path / "videos").glob("*.mp4"))
    assert len(videos) == 3 and all(p.is_symlink() and p.exists() for p in videos)
    assert video_episodes(parent["config"]) == [("PickXtimes",0),("VideoUnmask",0),("PickXtimes",1)]
    assert len(list(tmp_path.glob("episodes/*.json"))) == 0  # no duplicate durable episode outcomes


@pytest.mark.parametrize("field,value", [("checkpoints", {"/wrong": "wrong"}), ("method", "CURRENT"),
                                         ("num_shards", 2), ("execution_override", {"execute_actions": 8})])
def test_rejects_mixed_shard_inputs(tmp_path, field, value):
    parent = parent_fixture(tmp_path)
    root = write_shard(tmp_path, parent, 0)
    child = json.loads((root / "manifest.json").read_text())
    child[field] = value
    atomic_json(child, root / "manifest.json")
    with pytest.raises(ValueError, match="differs"):
        aggregate_shards(tmp_path, parent)


def test_rejects_copied_episode_and_false_completion(tmp_path):
    parent = parent_fixture(tmp_path)
    root = write_shard(tmp_path, parent, 0)
    path = next((root / "episodes").glob("*.json"))
    row = json.loads(path.read_text())
    row["evaluation_manifest_sha256"] = "different execution protocol"
    atomic_json(row, path)
    with pytest.raises(ValueError, match="episode manifest identity"):
        aggregate_shards(tmp_path, parent)
    path.unlink()
    with pytest.raises(ValueError, match="completion marker"):
        aggregate_shards(tmp_path, parent)


def test_rejects_duplicate_or_misassigned_episodes(tmp_path):
    parent = parent_fixture(tmp_path)
    root = write_shard(tmp_path, parent, 0)
    path = next((root / "episodes").glob("*.json"))
    (root / "episodes/copied.json").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="filename|Duplicate"):
        aggregate_shards(tmp_path, parent)
    (root / "episodes/copied.json").unlink()
    row = json.loads(path.read_text())
    row["episode_idx"] = 3
    atomic_json(row, path)
    with pytest.raises(ValueError, match="outside"):
        aggregate_shards(tmp_path, parent)


def test_router_hash_cannot_stand_in_for_policy_checkpoint(tmp_path):
    parent = parent_fixture(tmp_path)
    parent["checkpoints"]["/router"] = "router-weights"
    root = write_shard(tmp_path, parent, 0)
    path = next((root / "episodes").glob("*.json"))
    row = json.loads(path.read_text())
    row["checkpoint_sha256"] = "router-weights"
    atomic_json(row, path)
    with pytest.raises(ValueError, match="episode policy checkpoint"):
        aggregate_shards(tmp_path, parent)


def test_video_cap_and_episode_plan_reject_invalid_limits(tmp_path):
    parent = parent_fixture(tmp_path)
    cfg = parent["config"]
    for cap in [-1, True, 2.5]:
        cfg["evaluation"]["video_cap"] = cap
        with pytest.raises(ValueError, match="nonnegative integer"):
            video_episodes(cfg)
    for count in [0, 51, True]:
        with pytest.raises(ValueError, match="1–50"):
            planned_episodes(cfg, count)
    cfg["data"]["tasks"] *= 2
    with pytest.raises(ValueError, match="duplicates"):
        planned_episodes(cfg, 4)
