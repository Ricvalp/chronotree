"""CPU correspondence/architecture invariants; fake weights are labeled synthetic."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from stmem.data.canonical import Frame
from stmem.data.pointcloud import rgbd_to_points, rgbd_to_points_with_provenance
from stmem.vision import frozen


@pytest.fixture
def rgbd():
    observation, calibration = {}, {}
    for camera, offset in (("front", 0), ("wrist", 80)):
        v, u = np.indices((8, 12))
        observation[f"{camera}_rgb"] = np.stack((u + offset, v * 20, u * v), -1).astype(np.uint8)
        depth = np.full((8, 12), 100, dtype=np.int16)
        depth[0, :2] = (-100, 0)
        observation[f"{camera}_depth"] = depth
        observation[f"{camera}_camera_extrinsic"] = np.concatenate((np.eye(3), np.zeros((3, 1))), 1)
        calibration[f"{camera}_camera_intrinsic"] = np.array([[10.0, 0, 6], [0, 10.0, 4], [0, 0, 1]])
    return observation, calibration


@pytest.mark.parametrize("sampling", ["balanced", "uniform"])
@pytest.mark.parametrize("count,voxel", [(16, 0), (32, 0.02), (256, 0)])
def test_provenance_carries_exact_selected_xyzrgb_and_observing_camera(rgbd, sampling, count, voxel):
    observation, calibration = rgbd
    kwargs = dict(count=count, crop=None, voxel_size=voxel, sampling=sampling)
    legacy, valid = rgbd_to_points(observation, calibration, **kwargs)
    points, attached_valid, camera, uv = rgbd_to_points_with_provenance(observation, calibration, **kwargs)
    assert torch.equal(points, legacy)
    assert torch.equal(valid, attached_valid)
    assert (camera[~valid] == -1).all() and (uv[~valid] == -1).all()
    for index in torch.where(valid)[0]:
        name = ("front", "wrist")[camera[index]]
        u, v = uv[index].tolist()
        expected_rgb = torch.tensor(observation[f"{name}_rgb"][v, u], dtype=torch.float32) / 255
        assert torch.equal(points[index, 3:], expected_rgb)
        k = torch.tensor(calibration[f"{name}_camera_intrinsic"], dtype=torch.float32)
        pixel = points[index, :3] @ k.T
        assert torch.allclose(pixel[:2] / pixel[2], uv[index].float(), atol=2e-6)


def test_provenance_empty_and_cropped(rgbd):
    obs, calibration = rgbd
    for name in ("front", "wrist"):
        obs[f"{name}_depth"][:] = -1
    points, valid, camera, uv = rgbd_to_points_with_provenance(obs, calibration, count=16)
    assert points.eq(0).all() and not valid.any()
    assert camera.eq(-1).all() and uv.eq(-1).all()


def test_source_pixel_sampling_uses_rf_centers_camera_and_padding():
    # Synthetic maps encode source map coordinates: channel0=x,1=y,2=camera.
    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    maps = torch.stack([torch.stack((x, y, torch.full_like(x, camera)), 0) for camera in range(2)]).float()
    uv = torch.tensor([[1, 1], [5, 9], [7, 11], [0, 255], [255, 0], [-1, -1]])
    camera = torch.tensor([0, 1, 0, 1, 0, -1])
    valid = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    result = frozen.sample_dense_features(maps, camera, uv, valid)
    expected = torch.tensor([[0, 0, 0], [1, 2, 1], [1.5, 2.5, 0], [0, 63, 1], [63, 0, 0], [0, 0, 0]])
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=0)
    with pytest.raises(ValueError, match="outside"):
        frozen.sample_dense_features(maps, camera[:1], torch.tensor([[256, 0]]), valid[:1])
    with pytest.raises(ValueError, match="unknown observing"):
        frozen.sample_dense_features(maps, torch.tensor([2]), uv[:1], valid[:1])


def test_raw_descriptor_identity_ignores_projection_and_locations():
    cfg = {
        "model": {"visual_features": {"mode": "clip_rn50_layer1"}},
        "paths": {"data_root": "/data/a", "cache_root": "/cache/a"},
    }
    spec = frozen.resolve_visual_features(cfg)
    assert spec["checkpoint"] == "/cache/a/backbones/clip_RN50.pt"
    assert spec["cache_dir"] == "/data/a/features_clip_rn50_layer1_v1"
    alternative = {
        **spec,
        "checkpoint": "/elsewhere/pretrained.pt",
        "cache_dir": "/elsewhere/cache",
        "projection_dim": 64,
    }
    assert frozen.visual_feature_signature(spec) == frozen.visual_feature_signature(alternative)
    assert frozen.resolve_visual_features({})["mode"] == "xyzrgb"
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        frozen.resolve_visual_features({"mode": "clip_rn50_layer1", "checkpoint_sha256": "wrong"})
    with pytest.raises(ValueError, match="preprocessing"):
        frozen.resolve_visual_features({"mode": "clip_rn50_layer1", "preprocessing": "resized"})


def test_pretrained_asset_missing_or_corrupt_fails_no_random_fallback(tmp_path):
    with pytest.raises(FileNotFoundError, match="Prepare them explicitly"):
        frozen.FrozenCLIPFeatures({"mode": "clip_rn50_layer1", "checkpoint": str(tmp_path / "missing.pt")})
    corrupt = tmp_path / "bad.pt"
    corrupt.write_bytes(b"invalid checkpoint")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        frozen.ensure_checkpoint(corrupt)


@pytest.fixture
def synthetic_extractor(monkeypatch):
    """Synthetic architecture-only fixture; production strictly verifies weights."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        source = frozen._CLIPLayer1()
    state = {"visual." + key: value for key, value in source.state_dict().items()}
    monkeypatch.setattr(frozen, "ensure_checkpoint", lambda path: Path(path))
    monkeypatch.setattr(
        torch.jit,
        "load",
        lambda *args, **kwargs: type("SyntheticArtifact", (), {"state_dict": lambda _: state})(),
    )
    return frozen.FrozenCLIPFeatures({"mode": "clip_rn50_layer1"})


def test_frozen_parent_train_preserves_batchnorm_and_rng(synthetic_extractor):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        encoder = synthetic_extractor
        rng_before = torch.get_rng_state().clone()
        frozen.FrozenCLIPFeatures({"mode": "clip_rn50_layer1"})
        assert torch.equal(rng_before, torch.get_rng_state())
        parent = nn.ModuleDict(
            {"encoder": encoder, "projection": nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 128))}
        )
        parent.train()
        assert not encoder.training
        assert all(not module.training for module in encoder.modules())
        assert not any(parameter.requires_grad for parameter in encoder.parameters())
        before = {name: value.clone() for name, value in encoder.named_buffers()}
        image = torch.arange(256, dtype=torch.uint8)[None, :, None, None].expand(2, -1, 256, 3).clone()
        maps = encoder.dense(image)
        assert maps.shape == (2, 256, 64, 64)
        # Frozen descriptors can directly feed trainable projection autograd.
        descriptors = maps[:, :, 10, 10]
        projected = parent["projection"](descriptors)
        projected.square().mean().backward()
        assert all(parameter.grad is None for parameter in encoder.parameters())
        assert parent["projection"][1].weight.grad.abs().sum() > 0
        assert all(torch.equal(value, before[name]) for name, value in encoder.named_buffers())
        with pytest.raises(ValueError, match="native uint8"):
            encoder.dense(torch.zeros(1, 224, 224, 3, dtype=torch.uint8))
        with pytest.raises(ValueError, match="native uint8"):
            encoder.dense(image.float())
    finally:
        torch.set_num_threads(previous_threads)


def test_frame_feature_sidecars_preserve_order_and_move():
    frame = Frame(
        torch.zeros(4, 6),
        torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        torch.zeros(15),
        torch.zeros(8),
        False,
        0,
        "execution",
        "test",
        visual_features=torch.arange(4 * 256).reshape(4, 256).half(),
        point_camera=torch.tensor([0, 1, -1, -1]),
        point_uv=torch.tensor([[0, 1], [2, 3], [-1, -1], [-1, -1]]),
        visual_feature_signature="test",
    )
    moved = frame.to("cpu")
    assert torch.equal(moved.visual_features, frame.visual_features)
    assert torch.equal(moved.point_camera, frame.point_camera)
    with pytest.raises(ValueError, match="must be supplied together"):
        replace(frame, point_uv=None)


def test_feature_precision_restores_parent_backend_flags_on_error():
    original_precision = torch.get_float32_matmul_precision()
    original_cudnn = (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        torch.set_float32_matmul_precision("high")
        with torch.backends.cudnn.flags(enabled=False, benchmark=True, deterministic=False, allow_tf32=True):
            with pytest.raises(RuntimeError, match="deliberate"):
                with frozen._feature_precision("cpu"):
                    assert torch.get_float32_matmul_precision() == "highest"
                    assert torch.backends.cudnn.enabled
                    assert not torch.backends.cudnn.benchmark
                    assert torch.backends.cudnn.deterministic
                    assert not torch.backends.cudnn.allow_tf32
                    raise RuntimeError("deliberate")
            assert torch.get_float32_matmul_precision() == "high"
            assert not torch.backends.cudnn.enabled
            assert torch.backends.cudnn.benchmark
            assert not torch.backends.cudnn.deterministic
            assert torch.backends.cudnn.allow_tf32
    finally:
        torch.set_float32_matmul_precision(original_precision)
    assert original_cudnn == (
        torch.backends.cudnn.enabled,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    )
