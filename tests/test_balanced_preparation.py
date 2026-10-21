"""Synthetic RGB-D fixtures: versioned preparation and online/offline parity."""

from copy import deepcopy
import json

import h5py
import pytest
import torch
import yaml

from stmem.config import load_config
from stmem.data import prepare
from stmem.data.center_sampling import center_sampler_settings, center_counts
from stmem.data.dataset import EpisodeDataset
from stmem.data.download import REVISION
from stmem.data.pointcloud import point_sampler_settings
from stmem.evaluation.robomme_adapter import RoboMMEAdapter, OBSERVATION_FIELDS
from test_prepare_parallel import write_raw


def test_configured_directory_and_selection_identity_reach_workers(tmp_path, monkeypatch):
    write_raw(tmp_path, "VideoUnmask", episode_ids=(0,))
    cfg = load_config("configs/teacher.yaml")
    cfg["paths"]["data_root"] = str(tmp_path)
    cfg["data"].update(tasks=["VideoUnmask"], episodes_per_task=1)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    captured = []
    monkeypatch.setattr(prepare, "run_preparation", lambda jobs, **kwargs: captured.extend(jobs))
    prepare.main(["--config", str(path)])
    job = captured.pop()
    assert job["output"] == tmp_path / "prepared_balanced_v2/VideoUnmask.h5"
    assert job["points"] == 2048 and job["center_counts"] == [32]
    assert job["point_sampling"] == point_sampler_settings(cfg)
    assert job["center_sampling"] == center_sampler_settings(cfg)
    assert not (tmp_path / "prepared").exists()
    prepare.main(["--config", str(path), "--prepared-subdir", "separate_diagnostic"])
    assert captured.pop()["output"] == tmp_path / "separate_diagnostic/VideoUnmask.h5"
    with pytest.raises(SystemExit):
        prepare.main(["--config", str(path), "--prepared-subdir", "../outside"])


def test_prepared_indices_match_online_and_reject_stale_sampling(tmp_path):
    raw = write_raw(tmp_path, "VideoUnmask", episode_ids=(0,))
    cfg = load_config("configs/debug/data.yaml")
    cfg["paths"]["data_root"] = str(tmp_path)
    cfg["model"]["points"] = 64
    output = tmp_path / cfg["data"]["prepared_subdir"] / "VideoUnmask.h5"
    # A marker represents a previous preparation; rebuilding the new version
    # must not write into that directory.
    old = tmp_path / "prepared/untouched"
    old.parent.mkdir()
    old.write_text("previous version")
    report = prepare.prepare_task(
        raw,
        output,
        "VideoUnmask",
        1,
        points=64,
        point_sampling=point_sampler_settings(cfg),
        center_sampling=center_sampler_settings(cfg),
        center_counts=center_counts(cfg),
        progress=lambda _: None,
    )
    assert old.read_text() == "previous version"
    (tmp_path / "alignment.json").write_text(
        json.dumps(
            {
                "verified": True,
                "dataset_revision": REVISION,
                "convention": "post_action_v1",
                "raw_sha256": {"VideoUnmask": report["raw_sha256"]},
            }
        )
    )
    dataset = EpisodeDataset(output.parent)
    dataset.validate_preprocessing(cfg)
    episode = dataset.load_episode(0)
    assert len(episode.conditioning) == 2
    assert not episode.conditioning[0].action_available
    assert not episode.execution[0].action_available
    offline = episode.execution[1]
    adapter = RoboMMEAdapter(
        "VideoUnmask",
        env=object(),
        points=64,
        point_sampling=point_sampler_settings(cfg),
        center_sampling=center_sampler_settings(cfg),
        center_counts=center_counts(cfg),
    )
    with h5py.File(raw, "r") as f:
        g = f["episode_0"]
        adapter.calibration = {
            f"{c}_camera_intrinsic": g["setup"][f"{c}_camera_intrinsic"][()] for c in ("front", "wrist")
        }
        obs = {field + "_list": [g["timestep_3/obs"][field][()]] for field in OBSERVATION_FIELDS}
    online = adapter.ingest(obs, {}, event_id="fixture", command=offline.previous_action)[0]
    assert torch.equal(online.points, offline.points)
    assert torch.equal(online.valid, offline.valid)
    assert torch.equal(online.proprio, offline.proprio)
    assert online.center_sampling_signature == offline.center_sampling_signature
    assert set(online.center_indices) == set(center_counts(cfg))
    for count in center_counts(cfg):
        assert torch.equal(online.center_indices[count], offline.center_indices[count])
    for section, key, value in (
        ("point_sampling", "foreground_fraction", 0.5),
        ("center_sampling", "cell_size", 0.2),
    ):
        changed = deepcopy(cfg)
        destination = (
            changed["geometry"] if section == "point_sampling" else changed["model"]["encoder_options"]
        )
        destination[section][key] = value
        with pytest.raises(ValueError, match=section):
            dataset.validate_preprocessing(changed)
    with h5py.File(output, "r+") as f:
        f["episode_0/center_indices"][str(center_counts(cfg)[0])][0, 0] = 1000
    with pytest.raises(ValueError, match="center index outside"):
        dataset.load_episode(0)


def test_legacy_manifest_cannot_satisfy_new_preprocessing(tmp_path):
    cfg = load_config("configs/teacher.yaml")
    with h5py.File(tmp_path / "VideoUnmask.h5", "w") as f:
        f.attrs["manifest"] = json.dumps({"format": "canonical_v1", "task": "VideoUnmask"})
        g = f.create_group("episode_0")
        g.create_dataset("source", data=[False])
    dataset = EpisodeDataset(tmp_path)
    with pytest.raises(ValueError, match="prepared format.*re-run prepare"):
        dataset.validate_preprocessing(cfg)
    raw = write_raw(tmp_path, "VideoUnmask", episode_ids=(0,))
    output = tmp_path / "VideoUnmask.h5"
    before = output.read_bytes()
    with pytest.raises(ValueError, match="Refusing to overwrite prepared format"):
        prepare.prepare_task(raw, output, "VideoUnmask", 1, progress=lambda _: None)
    assert output.read_bytes() == before
