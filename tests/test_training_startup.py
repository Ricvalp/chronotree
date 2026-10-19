"""Synthetic HDF5 diagnostics for action-only training initialization.

These fixtures exercise the prepared-data interface, not benchmark performance.
Full Episode decisions are the independent oracle for the optimized index.
"""

import json

import h5py
import numpy as np
import pytest
import torch

from stmem.config import load_config
from stmem.data.dataset import EpisodeDataset
from stmem.data.pointcloud import CALIBRATION_VERSION
from stmem.models.action_flow import ActionNormalizer
from stmem.training.data import TrainingData


def _fixture_config(tmp_path, *, alignment="post_action_v1", tasks=None):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["paths"]["data_root"] = str(tmp_path)
    cfg["data"].update(
        synthetic=False,
        tasks=tasks or ["VideoUnmask", "PickXtimes"],
        episodes_per_task=4,
        fraction=0.5,
        decision_stride=1,
        alignment_path=str(tmp_path / "alignment.json"),
    )
    cfg["model"].update(points=4, horizon=4)
    evidence = {
        "verified": True,
        "convention": alignment,
        "dataset_revision": "synthetic-startup-diagnostic",
        "raw_sha256": {task: task + "-diagnostic-hash" for task in cfg["data"]["tasks"]},
    }
    (tmp_path / "alignment.json").write_text(json.dumps(evidence))
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for ti, task in enumerate(cfg["data"]["tasks"]):
        manifest = {
            "format": "canonical_v1",
            "task": task,
            "dataset_revision": evidence["dataset_revision"],
            "raw_sha256": evidence["raw_sha256"][task],
            "points": cfg["model"]["points"],
            "crop": cfg["geometry"]["crop"],
            "voxel_size": cfg["geometry"]["voxel_size"],
            "max_depth_m": cfg["geometry"]["max_depth"],
            "calibration": CALIBRATION_VERSION,
            "synthetic_diagnostic_fixture": True,
        }
        with h5py.File(prepared / f"{task}.h5", "w") as f:
            f.attrs["manifest"] = json.dumps(manifest)
            # Deliberately interleaved rows verify that execution is selected by
            # source, rather than assuming that it is an array suffix.
            source = np.array([True, False, True, False, False, True, False, False, False, False])
            execution_rows = np.flatnonzero(~source)
            n = len(source)
            for ei in (10, 2, 0):
                g = f.create_group(f"episode_{ei}")
                g.attrs["goal"] = "Synthetic startup diagnostic only."
                actions = (np.arange(n * 8).reshape(n, 8) / 8 + 100 * ti + ei).astype("f4")
                actions[source] = 10000  # Conditioning commands must not enter normalization.
                masks = np.ones((n, 8), dtype=bool)
                masks[execution_rows[0]] = False  # Unobserved reset command.
                masks[execution_rows[2], :7] = False  # Gripper alone cannot start a chunk.
                masks[execution_rows[4], 7] = False  # Missing gripper does not invalidate joints.
                masks[execution_rows[3], 0] = False  # A primitive-command gap truncates chunks.
                masks[execution_rows[-1]] = False  # Repeated terminal record.
                g.create_dataset("source", data=source)
                g.create_dataset(
                    "source_index", data=np.where(source, source.cumsum(), (~source).cumsum()) - 1
                )
                g.create_dataset("raw_index", data=np.arange(n))
                g.create_dataset("actions", data=actions)
                g.create_dataset("action_valid", data=masks)
                g.create_dataset("points", data=np.zeros((n, 4, 6), dtype="f4"))
                g.create_dataset("valid", data=np.ones((n, 4), dtype=bool))
                g.create_dataset("proprio", data=np.zeros((n, 15), dtype="f4"))
    return cfg


def _dataset(cfg):
    return EpisodeDataset(
        cfg["paths"]["data_root"] + "/prepared",
        cfg["data"]["tasks"],
        cfg["model"]["horizon"],
        cfg["data"]["alignment_path"],
    )


def _full_episode_oracle(cfg):
    dataset = _dataset(cfg)
    refs, episodes, candidates = [], [], {}
    cap = int(cfg["data"]["episodes_per_task"] * cfg["data"]["fraction"])
    for task in cfg["data"]["tasks"]:
        items = []
        for dataset_index in dataset.by_task[task][:cap]:
            index = len(refs)
            refs.append(dataset_index)
            episode = dataset.load_episode(dataset_index)
            episodes.append(episode)
            for time in range(0, len(episode), cfg["data"]["decision_stride"]):
                if episode.decision(time, cfg["model"]["horizon"]).target_valid.any():
                    items.append((index, time))
        candidates[task] = items
    return refs, episodes, candidates


def _legacy_statistics(episodes):
    count = torch.zeros(8, dtype=torch.long)
    total = torch.zeros(8, dtype=torch.float64)
    squares = torch.zeros(8, dtype=torch.float64)
    for episode in episodes:
        mask = episode.action_valid.bool() & torch.isfinite(episode.actions)
        actions = episode.actions.double().masked_fill(~mask, 0)
        count += mask.sum(0)
        total += actions.sum(0)
        squares += actions.square().sum(0)
    mean = total / count.clamp_min(1)
    variance = (squares / count.clamp_min(1) - mean.square()).clamp_min(0)
    return mean.float(), variance.sqrt().clamp_min(0.01).float(), count


def _forbid_visual_loading(monkeypatch):
    def forbidden_episode(*args, **kwargs):
        pytest.fail("Startup constructed a full point-cloud episode")

    monkeypatch.setattr(EpisodeDataset, "load_episode", forbidden_episode)
    for method in ("__getitem__", "__array__", "read_direct"):
        original = getattr(h5py.Dataset, method)

        def guarded(dataset, *args, _original=original, **kwargs):
            if dataset.name.rsplit("/", 1)[-1] in {"points", "valid", "proprio"}:
                pytest.fail(f"Startup read visual payload {dataset.name}")
            return _original(dataset, *args, **kwargs)

        monkeypatch.setattr(h5py.Dataset, method, guarded)


@pytest.mark.parametrize("alignment", ["pre_action_v1", "post_action_v1"])
@pytest.mark.parametrize("horizon,stride", [(1, 1), (4, 2), (16, 4)])
def test_startup_matches_full_episode_oracle_without_visual_reads(
    tmp_path, monkeypatch, alignment, horizon, stride
):
    cfg = _fixture_config(tmp_path, alignment=alignment)
    cfg["model"]["horizon"] = horizon
    cfg["data"]["decision_stride"] = stride
    refs, episodes, candidates = _full_episode_oracle(cfg)
    expected_mean, expected_std, expected_count = _legacy_statistics(episodes)

    with monkeypatch.context() as guarded:
        _forbid_visual_loading(guarded)
        data = TrainingData(cfg)

        def forbidden_second_scan(*args, **kwargs):
            pytest.fail("Normalization reread episode actions instead of reusing startup statistics")

        guarded.setattr(EpisodeDataset, "load_episode_actions", forbidden_second_scan)
        normalizer = ActionNormalizer()
        data.fit_normalizer(normalizer)
        assert data.refs == refs
        assert list(data.candidates) == cfg["data"]["tasks"]
        assert data.candidates == candidates
        assert [data.dataset.episodes[index].key for index in data.refs] == ["episode_0", "episode_2"] * 2
        torch.testing.assert_close(normalizer.mean, expected_mean, rtol=0, atol=0)
        torch.testing.assert_close(normalizer.std, expected_std, rtol=0, atol=0)
        torch.testing.assert_close(normalizer.count, expected_count, rtol=0, atol=0)

    # Training still loads the complete observation when a decision is requested.
    index, time = data.candidates[cfg["data"]["tasks"][0]][0]
    actual = data.decision(index, time)
    expected = episodes[index].decision(time, horizon)
    torch.testing.assert_close(actual.target, expected.target, rtol=0, atol=0)
    torch.testing.assert_close(actual.target_valid, expected.target_valid)
    assert actual.current.event_id == expected.current.event_id


def test_action_only_reader_preserves_source_selection_and_alignment(tmp_path, monkeypatch):
    cfg = _fixture_config(tmp_path)
    dataset = _dataset(cfg)
    episode = dataset.load_episode(0)
    with monkeypatch.context() as guarded:
        _forbid_visual_loading(guarded)
        actions = dataset.load_episode_actions(0)
    torch.testing.assert_close(actions.actions, episode.actions, rtol=0, atol=0)
    torch.testing.assert_close(actions.action_valid, episode.action_valid)
    assert actions.execution_length == len(episode.execution) == 7
    assert actions.conditioning_length == len(episode.conditioning) == 3
    assert actions.episode_id == episode.episode_id
    assert actions.alignment == episode.alignment


def test_normalization_ignores_conditioning_masks_and_nonfinite_values(tmp_path, monkeypatch):
    cfg = _fixture_config(tmp_path, tasks=["PatternLock"])
    with h5py.File(tmp_path / "prepared/PatternLock.h5", "r+") as f:
        for group in f.values():
            source = np.asarray(group["source"])
            rows = np.flatnonzero(~source)
            masks = np.asarray(group["action_valid"])
            actions = np.asarray(group["actions"])
            masks[:, 7] = False
            actions[:, 7] = np.nan
            actions[rows[1], 0] = np.nan  # Nonfinite value with an otherwise valid mask.
            actions[rows[3], 1] = np.inf
            actions[rows[3], 0] = -np.inf  # Both invalid mask and nonfinite value.
            group["actions"][:] = actions
            group["action_valid"][:] = masks
    _, episodes, _ = _full_episode_oracle(cfg)
    expected = _legacy_statistics(episodes)
    with monkeypatch.context() as guarded:
        _forbid_visual_loading(guarded)
        data = TrainingData(cfg)
        normalizer = ActionNormalizer()
        data.fit_normalizer(normalizer)
    for value, oracle in zip((normalizer.mean, normalizer.std, normalizer.count), expected):
        torch.testing.assert_close(value, oracle, rtol=0, atol=0)
    assert torch.isfinite(normalizer.mean).all() and torch.isfinite(normalizer.std).all()
    assert normalizer.count[7] == 0 and normalizer.mean[7] == 0
    torch.testing.assert_close(normalizer.std[7], torch.tensor(0.01), rtol=0, atol=0)
    assert normalizer.count[0] < normalizer.count[2]
    assert normalizer.mean[:7].max() < 10000


@pytest.mark.parametrize("problem", ["unverified", "revision", "raw_hash", "convention"])
def test_fast_startup_still_requires_matching_verified_alignment(tmp_path, monkeypatch, problem):
    cfg = _fixture_config(tmp_path)
    path = tmp_path / "alignment.json"
    evidence = json.loads(path.read_text())
    if problem == "unverified":
        evidence["verified"] = False
    elif problem == "revision":
        evidence["dataset_revision"] = "unrelated-revision"
    elif problem == "raw_hash":
        evidence["raw_sha256"][cfg["data"]["tasks"][0]] = "unrelated-raw-file"
    else:
        evidence["convention"] = "unresolved"
    path.write_text(json.dumps(evidence))
    _forbid_visual_loading(monkeypatch)
    with pytest.raises(RuntimeError, match="alignment unresolved"):
        TrainingData(cfg)


@pytest.mark.parametrize("bound,limit", [("max_execution_frames", 6), ("max_conditioning_frames", 2)])
def test_fast_startup_rejects_episodes_exceeding_declared_bounds(tmp_path, monkeypatch, bound, limit):
    cfg = _fixture_config(tmp_path)
    cfg["model"][bound] = limit
    _forbid_visual_loading(monkeypatch)
    with pytest.raises(ValueError, match="length bound"):
        TrainingData(cfg)


@pytest.mark.parametrize(
    "name,shape,dtype",
    [
        ("points", (10, 4, 5), "f4"),
        ("valid", (10, 3), "?"),
        ("valid", (10, 4), "u1"),
        ("proprio", (10, 14), "f4"),
    ],
)
def test_fast_startup_validates_visual_metadata_without_reading_payload(
    tmp_path, monkeypatch, name, shape, dtype
):
    cfg = _fixture_config(tmp_path)
    with h5py.File(tmp_path / "prepared/VideoUnmask.h5", "r+") as f:
        group = f["episode_0"]
        del group[name]
        group.create_dataset(name, shape=shape, dtype=dtype)
    _forbid_visual_loading(monkeypatch)
    with pytest.raises(ValueError, match=name):
        TrainingData(cfg)


def test_insufficient_prepared_episodes_still_fails(tmp_path, monkeypatch):
    cfg = _fixture_config(tmp_path)
    cfg["data"]["fraction"] = 1.0  # Four required; the diagnostic fixture has three.
    _forbid_visual_loading(monkeypatch)
    with pytest.raises(ValueError, match="need 4 prepared episodes, found 3"):
        TrainingData(cfg)
