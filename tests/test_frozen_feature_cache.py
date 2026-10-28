"""Synthetic diagnostics: immutable sidecars, causal row identity and bounded I/O."""

from copy import deepcopy
import json
import os
import pickle

import h5py
import numpy as np
import pytest
import torch

from stmem.config import load_config
from stmem.data.dataset import EpisodeDataset
from stmem.data.download import REVISION
from stmem.data.feature_cache import FeatureRow, load_feature_rows
from stmem.data.prepare import prepare_task
from stmem.data.prepare_features import estimate_feature_cache, prepare_feature_task
from stmem.vision.frozen import visual_feature_signature
from stmem.training.data import TrainingData
from stmem.training.loader import PolicyBatchLoader
from test_prepare_parallel import write_raw


class PixelDescriptors:
    """Test-only RGB/pixel encoding: no claim of pretrained visual features."""

    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def sample(self, observation, camera, uv, valid):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic interrupted writer")
        features = torch.zeros(len(valid), 256, dtype=torch.float16)
        for index, name in enumerate(("front", "wrist")):
            selected = valid & (camera == index)
            pixels = uv[selected]
            rgb = observation[f"{name}_rgb"][pixels[:, 1], pixels[:, 0]]
            features[selected, :3] = torch.as_tensor(rgb.astype(np.float16))
        features[valid, 3] = camera[valid].half()
        features[valid, 4:6] = uv[valid].half()
        return features


@pytest.fixture
def cache_input(tmp_path):
    cfg = load_config("configs/debug/data.yaml")
    cfg["paths"]["data_root"] = str(tmp_path)
    cfg["model"]["points"] = 16
    cfg["model"]["visual_tokens"] = 8
    cfg["model"]["current_tokens"] = 4
    cfg["model"]["visual_features"] = {"mode": "clip_rn50_layer1"}
    cfg["data"]["tasks"] = ["VideoUnmask"]
    cfg["data"]["episodes_per_task"] = 2
    cfg["data"]["prepared_subdir"] = "prepared_balanced_v2"
    raw = write_raw(tmp_path, "VideoUnmask", episode_ids=(0, 1))
    prepared = tmp_path / "prepared_balanced_v2/VideoUnmask.h5"
    from stmem.data.pointcloud import point_sampler_settings
    from stmem.data.center_sampling import center_sampler_settings, center_counts

    prepare_task(
        raw,
        prepared,
        "VideoUnmask",
        2,
        points=16,
        point_sampling=point_sampler_settings(cfg),
        center_sampling=center_sampler_settings(cfg),
        center_counts=center_counts(cfg),
        progress=lambda _: None,
    )
    (tmp_path / "alignment.json").write_text(
        json.dumps(
            {
                "verified": True,
                "dataset_revision": REVISION,
                "convention": "post_action_v1",
            }
        )
    )
    output = tmp_path / "features_clip_rn50_layer1_v1/VideoUnmask.h5"
    return cfg, raw, prepared, output


def prepare(cache_input, **kwargs):
    cfg, raw, canonical, output = cache_input
    return prepare_feature_task(
        canonical,
        raw,
        output,
        cfg,
        progress=lambda _: None,
        extractor=kwargs.pop("extractor", PixelDescriptors()),
        **kwargs,
    )


def test_preparation_preserves_geometry_and_lazy_pixel_correspondence(cache_input, monkeypatch):
    cfg, raw, canonical, output = cache_input
    before = canonical.read_bytes()
    report = prepare(cache_input)
    assert canonical.read_bytes() == before
    assert report["complete"] and not report["diagnostic"]
    assert report["new_frames"] == 18
    assert estimate_feature_cache(canonical, cfg)["frames"] == 18
    plain = EpisodeDataset(canonical)
    dataset = EpisodeDataset(canonical, visual_features_config=cfg)
    dataset.validate_preprocessing(cfg)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Episode loading must not materialize descriptor payloads")

    monkeypatch.setattr(FeatureRow, "load", forbidden)
    episode = dataset.load_episode(0)
    original = plain.load_episode(0)
    frames = episode.conditioning + episode.execution
    for frame, baseline in zip(frames, original.conditioning + original.execution):
        assert torch.equal(frame.points, baseline.points)
        assert torch.equal(frame.valid, baseline.valid)
        assert frame.visual_feature_signature == visual_feature_signature(cfg)
        assert isinstance(frame.visual_features, FeatureRow)
    # Rows are tiny path/offset references, independent of the descriptor bytes.
    packed_refs = pickle.dumps([frame.visual_features for frame in frames])
    assert len(packed_refs) < 4096
    references = pickle.loads(packed_refs)
    selected = load_feature_rows([references[0], references[-1], None], point_count=20)
    assert selected.shape == (3, 20, 256)
    assert not selected[2].any() and not selected[:, 16:].any()
    for index, frame in enumerate((frames[0], frames[-1])):
        assert torch.equal(selected[index, :16, :3], (frame.rgb * 255).half())
        assert torch.equal(selected[index, :16, 3], frame.point_camera.half())
        assert torch.equal(selected[index, :16, 4:6], frame.point_uv.half())


def test_interrupted_cache_resumes_only_committed_rows(cache_input):
    cfg, _, canonical, output = cache_input
    with pytest.raises(RuntimeError, match="interrupted writer"):
        prepare(cache_input, extractor=PixelDescriptors(fail_at=4))
    assert not output.exists()
    partial = output.with_suffix(".partial.h5")
    with h5py.File(partial) as handle:
        assert handle["episode_0"].attrs["completed_rows"] == 3
    with pytest.raises(FileNotFoundError, match="Missing frozen descriptors"):
        EpisodeDataset(canonical, visual_features_config=cfg)
    extractor = PixelDescriptors()
    report = prepare(cache_input, extractor=extractor)
    assert extractor.calls == 15 and report["new_frames"] == 15
    assert not partial.exists()
    report = prepare(cache_input, skip_existing=True, extractor=PixelDescriptors(fail_at=1))
    assert report["skipped"]
    assert EpisodeDataset(canonical, visual_features_config=cfg).load_episode(0).conditioning


def test_cache_rejects_geometry_and_raw_identity_changes(cache_input):
    cfg, raw, canonical, _ = cache_input
    prepare(cache_input)
    wrong = deepcopy(cfg)
    wrong["geometry"]["voxel_size"] = 0.025
    with pytest.raises(ValueError, match="Prepared voxel_size"):
        estimate_feature_cache(canonical, wrong)
    with h5py.File(canonical, "r+") as handle:
        manifest = json.loads(handle.attrs["manifest"])
        manifest["point_sampling"]["foreground_fraction"] = 0.5
        handle.attrs.modify("manifest", json.dumps(manifest))
    with pytest.raises(ValueError, match="geometry/source identity"):
        EpisodeDataset(canonical, visual_features_config=cfg)
    # Raw changes are rejected before an output can be created/reused.
    with h5py.File(raw, "r+") as handle:
        handle["episode_0/timestep_0/obs/front_rgb"][0, 0, 0] = 255
    # Restore the recipe so this case reaches the raw-integrity guard.
    with h5py.File(canonical, "r+") as handle:
        manifest["point_sampling"]["foreground_fraction"] = cfg["geometry"]["point_sampling"][
            "foreground_fraction"
        ]
        handle.attrs.modify("manifest", json.dumps(manifest))
    with pytest.raises(ValueError, match="Raw RGB-D checksum"):
        prepare(cache_input, skip_existing=True)


def test_cache_rejects_changed_point_selection_and_partial_smoke(cache_input):
    cfg, _, canonical, output = cache_input
    with h5py.File(canonical, "r+") as handle:
        original = handle["episode_0/points"][0]
        changed = original.copy()
        changed[0, 0] += 0.001
        handle["episode_0/points"][0] = changed
    with pytest.raises(ValueError, match="not allowed to change geometry"):
        prepare(cache_input)
    with h5py.File(canonical, "r+") as handle:
        handle["episode_0/points"][0] = original
    output.with_suffix(".partial.h5").unlink()
    report = prepare(cache_input, max_frames=2)
    assert report["diagnostic"] and report["frames"] == 2
    with pytest.raises(ValueError, match="diagnostic feature cache"):
        EpisodeDataset(canonical, visual_features_config=cfg)


def test_missing_episode_fails_before_training_and_cache_ordering_checked(cache_input):
    cfg, _, canonical, output = cache_input
    prepare(cache_input, episodes=1)
    dataset = EpisodeDataset(canonical, visual_features_config=cfg)
    dataset.load_episode_actions(0)
    with pytest.raises(ValueError, match="lacks VideoUnmask/episode_1"):
        dataset.load_episode_actions(1)
    with h5py.File(output, "r+") as handle:
        indices = handle["episode_0/raw_index"][()]
        indices[0], indices[1] = indices[1], indices[0]
        handle["episode_0/raw_index"][:] = indices
    with pytest.raises(ValueError, match="raw_index ordering differs"):
        dataset.load_episode(0)


def test_descriptor_identity_mismatch_and_truncation_fail(cache_input):
    cfg, _, canonical, output = cache_input
    prepare(cache_input)
    dataset = EpisodeDataset(canonical, visual_features_config=cfg)
    ref = dataset.load_episode(0).conditioning[0].visual_features
    with h5py.File(output, "r+") as handle:
        manifest = json.loads(handle.attrs["manifest"])
        manifest["feature_identity"]["feature_stride"] = 8
        handle.attrs.modify("manifest", json.dumps(manifest))
    with pytest.raises(ValueError, match="backbone/preprocessing identity"):
        EpisodeDataset(canonical, visual_features_config=cfg)
    truncated = ref.__class__(
        ref.path, ref.episode, ref.row, ref.point_count, ref.signature, output.stat().st_size - 2
    )
    with pytest.raises(ValueError, match="Truncated descriptor cache"):
        truncated.load()


def test_spawn_loader_sends_lazy_descriptor_rows_and_preserves_packed_predictions(cache_input):
    """Workers prefetch geometry/reference metadata; only the consumer reads raw descriptors."""
    from stmem.models.policy import MemoryPolicy

    torch.set_num_threads(1)
    cfg, _, _, _ = cache_input
    prepare(cache_input)
    cfg["model"].update(
        width=16, heads=4, layers=1, text_layers=1, max_text_bytes=24,
        proprioception_mode="none", condition_on_previous_commands=False,
    )
    cfg["training"].update(workers=1, prefetch_factor=1, encoder_block_batch_size=2)
    data = TrainingData(cfg)
    policy = MemoryPolicy(cfg).eval()
    with PolicyBatchLoader(data, start=0, steps=2, batch=2, rank=0, world=1, microbatch=1) as loader:
        workers = list(loader._iterator._workers)
        batches = [batch for step in range(2) for batch in loader.read_update(step)]
        assert all(batch.worker_pid != os.getpid() for batch in batches)
        for batch in batches:
            packed = batch.packed
            refs = [*packed.current_visual_features,
                    *(row for block in packed.history.visual_features for row in block if row is not None)]
            assert refs and all(isinstance(ref, FeatureRow) for ref in refs)
            # Even these tiny 16-point fixtures would be 8KiB/frame as tensors;
            # transport contains path/row offsets with no persistent file handles.
            assert len(pickle.dumps(refs)) < len(refs) * 8192
            noise = torch.randn(1, cfg["model"]["horizon"], 8)
            time = torch.tensor([0.3])
            with torch.no_grad():
                received = policy(packed.metadata, noise, time, packed_batch=packed)
                expected = policy(data.sample_positions(batch.positions), noise, time)
            torch.testing.assert_close(received, expected, atol=2e-6, rtol=3e-5)
    assert all(not worker.is_alive() for worker in workers)
