"""Validate and combine durable shard outcomes without narrowing denominators."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

from stmem.evaluation.metrics import episode_key, summarize_episodes
from stmem.evaluation.workload import planned_episodes, video_episodes
from stmem.training.checkpoints import atomic_json, file_sha256


@lru_cache(maxsize=256)
def _read_unchanged(path, mtime_ns, size):
    # Episode files are atomically replaced; repeated monitoring should not
    # repeatedly parse the full action/decision trace for every finished scene.
    return json.loads(Path(path).read_text())


def read_artifact(path):
    stat = path.stat()
    try:
        return _read_unchanged(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid evaluation artifact {path}: {error}") from error


def _same(actual, expected, name):
    if json.loads(json.dumps(actual)) != json.loads(json.dumps(expected)):
        raise ValueError(f"Parallel evaluation shard {name} differs from its parent manifest")


def scientific_config(cfg):
    cfg = deepcopy(cfg)
    cfg.pop("tracking", None)
    return cfg


def aggregate_shards(output, manifest):
    """Return all validated outcomes plus a summary over the full parent plan.

    An absent worker or incomplete episode remains missing. Infrastructure
    errors remain recorded failures. Duplicates or altered inputs raise, never
    selecting a convenient subset. Child completion is reported separately
    from the outcome denominator; the supervisor also checks process exits.
    """
    output = Path(output)
    cfg = manifest["config"]
    plan = planned_episodes(cfg)
    _same(manifest["all_planned"], plan, "all_planned")
    count = manifest["num_shards"]
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= len(plan):
        raise ValueError("Invalid parallel evaluation shard count")
    selected_videos = set(video_episodes(cfg))
    if "video_episodes" in manifest:
        _same(manifest["video_episodes"], video_episodes(cfg), "video_episodes")
    expected_dirs = {f"shard{index:03d}" for index in range(count)}
    actual_dirs = {p.name for p in (output / "shards").glob("*") if p.is_dir()}
    if actual_dirs - expected_dirs:
        raise ValueError(f"Unexpected evaluation shards: {sorted(actual_dirs - expected_dirs)}")
    records, seen, shard_states, video_index, episode_index = [], set(), [], [], []
    for index in range(count):
        root = output / "shards" / f"shard{index:03d}"
        path = root / "manifest.json"
        assigned = plan[index::count]
        result_paths = sorted((root / "episodes").glob("*.json"))
        if not path.exists():
            if result_paths or (root / "complete.json").exists():
                raise ValueError(f"Shard outcomes exist without a manifest: {root}")
            shard_states.append({"index": index, "status": "missing", "recorded": 0,
                                 "planned": len(assigned)})
            continue
        child = read_artifact(path)
        child_hash = file_sha256(path)
        _same(scientific_config(child["config"]), scientific_config(cfg), "config")
        _same(child["source"]["source_sha256"], manifest["source"]["source_sha256"], "source")
        for field in ("checkpoints", "method", "split", "selection_family", "intervention",
                      "budget", "targeting_diagnostics", "execution_override"):
            _same(child.get(field), manifest.get(field), field)
        _same(child["all_planned"], plan, "all_planned")
        _same(child["planned"], assigned, "planned")
        _same(child["shard_index"], index, "shard_index")
        _same(child["num_shards"], count, "num_shards")
        _same(child.get("video_episodes"), video_episodes(cfg), "video_episodes")
        if "render_device" in child:
            _same(child["render_device"], "cuda:0", "render_device")
        local_keys = set()
        for result_path in result_paths:
            row = dict(read_artifact(result_path))
            key = episode_key(row)
            if key not in assigned:
                raise ValueError(f"Episode {key} is outside shard {index}'s assigned plan")
            if key in seen:
                raise ValueError(f"Duplicate evaluation episode: {key}")
            if result_path.name != f"{key[0]}_{key[1]:03d}.json":
                raise ValueError(f"Unexpected episode result filename: {result_path}")
            _same(row.get("evaluation_manifest_sha256"), child_hash, "episode manifest identity")
            _same(row.get("method"), manifest["method"], "episode method")
            _same(row.get("split"), manifest["split"], "episode split")
            _same(row.get("training_seed"), cfg["seed"], "episode training seed")
            seed = cfg["evaluation"]["seed"] + cfg["data"]["tasks"].index(key[0]) * 1000 + key[1]
            _same(row.get("noise_seed"), seed, "episode noise seed")
            _same(row.get("checkpoint_sha256"), manifest["policy_checkpoint_sha256"],
                  "episode policy checkpoint")
            _same(row.get("execution_override"), manifest.get("execution_override"), "episode execution")
            if "status" not in row:
                raise ValueError(f"Episode lacks an outcome: {result_path}")
            seen.add(key)
            local_keys.add(key)
            row["_evaluation_shard"] = str(root)
            records.append(row)
            episode_index.append({"task": key[0], "episode_idx": key[1], "status": row["status"],
                                  "path": str(result_path.relative_to(output)), "shard": index})
            video = root / "videos" / f"{key[0]}_{key[1]:03d}.mp4"
            if video.exists():
                if key not in selected_videos:
                    raise ValueError(f"Unplanned video exceeds the global video assignment: {video}")
                destination = output / "videos" / video.name
                destination.parent.mkdir(exist_ok=True)
                relative = Path("../shards") / root.name / "videos" / video.name
                if destination.is_symlink():
                    if destination.resolve() != video.resolve():
                        raise ValueError(f"Existing video link points to another evaluation: {destination}")
                elif destination.exists():
                    raise ValueError(f"Refusing to replace existing video file: {destination}")
                else:
                    destination.symlink_to(relative)
                video_index.append({"task": key[0], "episode_idx": key[1],
                                    "path": str(destination.relative_to(output)), "shard": index})
        completed = root / "complete.json"
        status = "running"
        if completed.exists():
            marker = read_artifact(completed)
            _same(marker.get("manifest_sha256"), child_hash, "completion manifest identity")
            _same(marker.get("status"), "complete", "completion status")
            if (local_keys != set(assigned) or marker.get("planned") != len(assigned)
                    or marker.get("recorded") != len(assigned)):
                raise ValueError(f"Shard {index} completion marker does not cover all assigned episodes")
            status = "complete"
        shard_states.append({"index": index, "status": status, "recorded": len(local_keys),
                             "planned": len(assigned)})
    records.sort(key=lambda row: plan.index(episode_key(row)))
    summary = summarize_episodes(records, plan)
    summary["worker_completion"] = {
        "planned": count, "complete": sum(s["status"] == "complete" for s in shard_states),
        "shards": shard_states,
    }
    atomic_json(summary, output / "summary.json")
    atomic_json({"planned": plan, "episodes": episode_index}, output / "episode_index.json")
    atomic_json({"cap": cfg["evaluation"]["video_cap"], "planned": video_episodes(cfg),
                 "videos": video_index, "note": "Links resolve into the shard outputs; retain both directories."},
                output / "video_index.json")
    return records, summary
