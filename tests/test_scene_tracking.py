"""Exercise real offline Object3D/Table files, history, and restart semantics."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from stmem.tracking import RunTracker
from stmem.visualization.scenes import build_action_scene
from test_scenes import decision_fixture
from test_tracking import read_offline


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    for key in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_DIR"):
        monkeypatch.delenv(key, raising=False)


def config(directory):
    return {
        "seed": 0,
        "debug": True,
        "data": {"synthetic": True},
        "paths": {"run_root": str(directory)},
        "tracking": {"mode": "offline"},
    }


def test_real_offline_scenes_and_tables_survive_scalar_interleaving_and_resume(tmp_path):
    decision = decision_fixture()
    prediction = decision.target.clone()
    prediction[0, 0] = 9.0  # Show the raw-versus-commanded clipping distinction.
    decision.target_valid[0, 7] = False
    decision.target_chunk[0, 7] = 123.0  # A masked value must not appear as a target in the table.
    decision.target_valid[2:] = False
    scene = build_action_scene(decision, prediction, max_points=3, path_samples=16)
    directory = tmp_path / "run"
    for attempt, optimizer_step in enumerate((10, 8)):
        with RunTracker(config(tmp_path), directory, job_type="teacher", resume=bool(attempt)) as tracker:
            tracker.log({"train/loss": 1.0}, step=optimizer_step)
            random.seed(20)
            np.random.seed(20)
            torch.manual_seed(20)
            expected = (random.random(), np.random.random(), torch.rand(3))
            random.seed(20)
            np.random.seed(20)
            torch.manual_seed(20)
            scenes = {0: scene, 1: scene} if attempt == 0 else {0: scene}
            media_cost = tracker.log_scenes(scenes, step=optimizer_step)
            assert random.random() == expected[0]
            assert np.random.random() == expected[1]
            torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
            tracker.log({"train/loss": 0.5}, step=optimizer_step + 1)
            assert media_cost["seconds"] > 0
            assert media_cost["bytes"] > 0
            records = sorted((directory / "diagnostics").glob(f"attempt{attempt + 1}_*.json"))
            assert len(records) == 1
            diagnostic = json.loads(records[0].read_text())
            assert diagnostic["step"] == optimizer_step
            assert diagnostic["wandb_files_directory"] == str(Path(tracker.run.dir).resolve())
            metadata = diagnostic["scenes"]["diagnostics/scene_0"]
            assert metadata["predicted_actions_raw"][0][0] == 9.0
            assert metadata["predicted_actions_commanded"][0][0] < 3.0
            assert metadata["recorded_actions_raw"][0][7] == 123.0
            assert metadata["target_valid"][0][7] is False
            assert "payload" not in metadata and "points" not in metadata
            references = diagnostic["media"]
            assert set(references) == {
                f"diagnostics/{kind}_{index}" for index in scenes for kind in ("scene", "actions")
            }
            paths = {key: Path(tracker.run.dir) / value["path"] for key, value in references.items()}
            assert all(path.is_file() for path in paths.values())
            assert media_cost["bytes"] == sum(path.stat().st_size for path in paths.values())
            cloud = json.loads(paths["diagnostics/scene_0"].read_text())
            assert cloud["type"] == "lidar/beta"
            assert len(cloud["points"]) == len(scene.payload["points"])
            assert len(cloud["boxes"]) == 3
            table = json.loads(paths["diagnostics/actions_0"].read_text())
            assert table["columns"] == [
                "horizon",
                "component",
                "predicted_raw",
                "predicted_command",
                "ground_truth",
                "target_valid",
            ]
            assert len(table["data"]) == 4 * 8
            assert table["data"][0][2] == 9.0
            assert table["data"][0][3] == metadata["predicted_actions_commanded"][0][0]
            assert table["data"][7][4:] == [None, False]
            assert all(row[4:] == [None, False] for row in table["data"][16:])
    journal = [json.loads(line) for line in (directory / "tracking_history.jsonl").read_text().splitlines()]
    assert [row["history_step"] for row in journal] == list(range(6))
    assert [row["metrics"]["progress/step"] for row in journal] == [10, 10, 11, 8, 8, 9]
    for row in (journal[1], journal[4]):
        assert set(row["metrics"]) == {"diagnostics/metadata_path", "progress/step", "tracking/attempt"}
        assert Path(row["metrics"]["diagnostics/metadata_path"]).is_file()
    files = sorted(directory.glob("wandb/offline-run-*/*.wandb"))
    assert len(files) == 2
    # Old attempt metadata remains resolvable after tracking.json points at the
    # resumed SDK directory; no live run object is needed to inspect its files.
    for path in (directory / "diagnostics").glob("*.json"):
        diagnostic = json.loads(path.read_text())
        root = Path(diagnostic["wandb_files_directory"])
        assert all((root / ref["path"]).is_file() for ref in diagnostic["media"].values())
    histories, ids = [], []
    for path in files:
        for record in read_offline(path):
            if record.HasField("run"):
                ids.append(record.run.run_id)
            if record.HasField("history"):
                histories.append(
                    {
                        item.key or "/".join(item.nested_key): json.loads(item.value_json)
                        for item in record.history.item
                    }
                )
    assert len(set(ids)) == 1
    assert [row["_step"] for row in histories] == list(range(6))
    assert [row["progress/step"] for row in histories] == [10, 10, 11, 8, 8, 9]
    for row in (histories[1], histories[4]):
        assert row["diagnostics/scene_0/_type"] == "object3D-file"
        assert row["diagnostics/actions_0/_type"] == "table-file"
    assert histories[1]["diagnostics/scene_1/_type"] == "object3D-file"
    assert "diagnostics/scene_1/_type" not in histories[4]
    state = json.loads((directory / "tracking.json").read_text())
    assert state["next_history_step"] == 6 and state["status"] == "completed"


def test_scalar_api_still_rejects_media_and_scene_count_is_bounded(tmp_path):
    import wandb

    decision = decision_fixture()
    scene = build_action_scene(decision, decision.target)
    obj = wandb.Object3D(scene.payload, caption=scene.caption)
    directory = tmp_path / "run"
    with RunTracker(config(tmp_path), directory, job_type="teacher") as tracker:
        with pytest.raises(TypeError, match="CPU scalars/JSON"):
            tracker.log({"unsupported_media": obj}, step=1)
        with pytest.raises(ValueError, match="At most two"):
            tracker.log_scenes({0: scene, 1: scene, 2: scene}, step=1)
        with pytest.raises(ValueError, match="indices"):
            tracker.log_scenes({2: scene}, step=1)
        tracker.log({"train/loss": 0.5}, step=1)
    journal = [json.loads(line) for line in (directory / "tracking_history.jsonl").read_text().splitlines()]
    assert len(journal) == 1
    assert journal[0]["history_step"] == 0


def test_disabled_scene_logging_does_not_import_sdk_or_touch_scene(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def no_sdk_import(name, *args, **kwargs):
        if name == "wandb" or name.startswith("wandb."):
            pytest.fail("Disabled media logging must not import the SDK")
        return original_import(name, *args, **kwargs)

    class UntouchableScenes:
        def __bool__(self):
            pytest.fail("Disabled media logging must not inspect or render scenes")

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(builtins, "__import__", no_sdk_import)
    directory = tmp_path / "disabled"
    with RunTracker(config(tmp_path), directory, job_type="teacher") as tracker:
        assert tracker.log_scenes(UntouchableScenes(), step=10) == {"seconds": 0.0, "bytes": 0}
    assert not directory.exists()
