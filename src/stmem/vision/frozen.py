"""OpenAI CLIP RN50 stem + layer1, with explicit RGB/pixel conventions.

The early convolution architecture is adapted from OpenAI CLIP (MIT license,
Copyright 2021 OpenAI); see third_party/licenses/OpenAI-CLIP-MIT.txt. We load only these published
weights, never initialize a random stand-in when a checkpoint is unavailable.
The complete pretrained artifact is verified before torch.jit.load; no remote
code is imported and no weights are downloaded implicitly during training.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import urllib.request

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from stmem.config import digest
from stmem.data.pointcloud import CAMERA_NAMES, POINT_PROVENANCE_VERSION

CLIP_SOURCE_REVISION = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
RN50_SHA256 = "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
RN50_URL = f"https://openaipublic.azureedge.net/clip/models/{RN50_SHA256}/RN50.pt"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
FEATURE_DIM = 256
FEATURE_STRIDE = 4
FEATURE_ORIGIN = 1.0
PREPROCESSING = "clip_rgb_native256_v1"
PIXEL_SAMPLING = "rf_origin1_stride4_bilinear_border_align_corners_false_v1"
_DEFAULTS = {
    "mode": "xyzrgb",
    "backbone": "openai_clip_rn50",
    "layer": "layer1",
    "preprocessing": PREPROCESSING,
    "projection_dim": 128,
    "checkpoint": "{cache_root}/backbones/clip_RN50.pt",
    "checkpoint_sha256": RN50_SHA256,
    "cache_dir": "{data_root}/features_clip_rn50_layer1_v1",
}


def resolve_visual_features(config: Mapping | None = None) -> dict:
    """Resolve a full configuration or its model.visual_features mapping.

    Absent settings retain XYZRGB. Paths are machine-local locations, excluded
    from descriptor identity; checksum/layer/preprocessing define the content.
    """
    config = {} if config is None else config
    if not isinstance(config, Mapping):
        raise ValueError("model.visual_features must be a mapping")
    if "model" in config or "paths" in config:
        spec = config.get("model", {}).get("visual_features", {})
        paths = config.get("paths", {})
    else:
        spec, paths = config, {}
    if not isinstance(spec, Mapping):
        raise ValueError("model.visual_features must be a mapping")
    unknown = spec.keys() - _DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown frozen visual feature settings: {sorted(unknown)}")
    result = {**_DEFAULTS, **spec}
    if result["mode"] not in ("xyzrgb", "clip_rn50_layer1"):
        raise ValueError("visual_features.mode must be xyzrgb or clip_rn50_layer1")
    for key in ("backbone", "layer", "preprocessing", "checkpoint_sha256"):
        if result[key] != _DEFAULTS[key]:
            raise ValueError(f"Unsupported frozen visual feature {key}: {result[key]!r}")
    if isinstance(result["projection_dim"], bool) or not isinstance(result["projection_dim"], int):
        raise ValueError("visual_features.projection_dim must be a positive integer")
    if result["projection_dim"] < 1:
        raise ValueError("visual_features.projection_dim must be a positive integer")
    substitutions = {
        "data_root": str(paths.get("data_root", "data")),
        "cache_root": str(paths.get("cache_root", "cache")),
    }
    for key in ("checkpoint", "cache_dir"):
        value = result[key]
        if not isinstance(value, (str, os.PathLike)) or not str(value):
            raise ValueError(f"visual_features.{key} must be a nonempty path")
        try:
            result[key] = os.path.expandvars(os.path.expanduser(str(value).format(**substitutions)))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid visual_features.{key} path placeholder") from exc
    return result


def visual_feature_identity(config: Mapping | None = None) -> dict:
    spec = resolve_visual_features(config)
    if spec["mode"] == "xyzrgb":
        return {"mode": "xyzrgb"}
    return {
        "mode": spec["mode"],
        "backbone": spec["backbone"],
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "layer": spec["layer"],
        "preprocessing": spec["preprocessing"],
        "image_shape": [256, 256, 3],
        "channel_order": "RGB",
        "rgb_scale": "uint8_div255",
        "normalization_mean": list(CLIP_MEAN),
        "normalization_std": list(CLIP_STD),
        "feature_shape": [256, 64, 64],
        "feature_stride": FEATURE_STRIDE,
        "feature_origin_uv": [FEATURE_ORIGIN, FEATURE_ORIGIN],
        "pixel_sampling": PIXEL_SAMPLING,
        "point_provenance": POINT_PROVENANCE_VERSION,
        "camera_order": list(CAMERA_NAMES),
        "compute_dtype": "float32",
        "compute_precision": "fp32_no_tf32_cudnn_deterministic_v1",
        "cudnn_enabled": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_allow_tf32": False,
        "matmul_precision": "highest",
        "storage_dtype": "float16",
    }


def visual_feature_signature(config: Mapping | None = None) -> str:
    """Descriptor identity (projection weights/dimension are intentionally absent)."""
    return digest(visual_feature_identity(config))


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def ensure_checkpoint(path: str | Path, *, download: bool = False) -> Path:
    """Explicit verified download, atomic rename; never overwrite mismatched data."""
    path = Path(path)
    if path.exists():
        if not path.is_file() or file_sha256(path) != RN50_SHA256:
            raise ValueError(f"CLIP RN50 checkpoint SHA256 mismatch: {path}")
        return path
    if not download:
        raise FileNotFoundError(
            f"Missing pretrained CLIP RN50 weights: {path}. Prepare them explicitly with "
            f"python -m stmem.vision.frozen --download --output {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with urllib.request.urlopen(RN50_URL, timeout=60) as source, temporary.open("xb") as target:
            print(
                json.dumps(
                    {
                        "event": "clip_weight_download",
                        "url": RN50_URL,
                        "bytes": source.headers.get("Content-Length"),
                        "output": str(path),
                    }
                ),
                flush=True,
            )
            while block := source.read(8 * 1024 * 1024):
                target.write(block)
        if file_sha256(temporary) != RN50_SHA256:
            raise ValueError("Downloaded CLIP RN50 checkpoint failed SHA256 verification")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class _Bottleneck(nn.Module):
    """Official CLIP first-stage residual block; no stage downsampling here."""

    def __init__(self, inplanes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.Identity()
        self.conv3 = nn.Conv2d(64, 256, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = None
        if inplanes != 256:
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(1)),
                        ("0", nn.Conv2d(inplanes, 256, 1, bias=False)),
                        ("1", nn.BatchNorm2d(256)),
                    ]
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu3(x + residual)


class _CLIPLayer1(nn.Module):
    """Checkpoint-compatible early RN50; construction alone is test-only."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        self.layer1 = nn.Sequential(_Bottleneck(64), _Bottleneck(256), _Bottleneck(256))

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return self.layer1(self.avgpool(x))


def sample_dense_features(
    maps: Tensor,
    camera_indices: Tensor,
    pixel_uv: Tensor,
    valid: Tensor,
    *,
    image_shape: tuple[int, int] = (256, 256),
) -> Tensor:
    """Bilinear source-camera sampling, zero invalid points, no reprojection.

    CLIP's stride-2 padded convolution keeps the first center at input pixel 0;
    the following 2x2 average pool moves it to pixel 1 and stride 4. Thus layer1
    map cell (j,i) is centered at original (4*i+1,4*j+1). Sampling outside the
    last map center clamps to its boundary. We use no crop/resize/augmentation.
    """
    if maps.ndim != 4 or maps.shape[0] != len(CAMERA_NAMES):
        raise ValueError("Expected feature maps [2,C,Hf,Wf] in front,wrist camera order")
    count = len(camera_indices)
    if camera_indices.shape != (count,) or pixel_uv.shape != (count, 2) or valid.shape != (count,):
        raise ValueError("Expected camera[N], original pixel_uv[N,2], valid[N]")
    if camera_indices.dtype != torch.long or pixel_uv.dtype != torch.long or valid.dtype != torch.bool:
        raise ValueError("Provenance must be int64 and validity bool")
    camera_indices = camera_indices.to(maps.device)
    pixel_uv = pixel_uv.to(maps.device)
    valid = valid.to(maps.device)
    observed_camera, observed_uv = camera_indices[valid], pixel_uv[valid]
    h, w = image_shape
    if (observed_camera < 0).any() or (observed_camera >= len(CAMERA_NAMES)).any():
        raise ValueError("Valid point references an unknown observing camera")
    if (observed_uv < 0).any() or (observed_uv[:, 0] >= w).any() or (observed_uv[:, 1] >= h).any():
        raise ValueError("Valid point pixel lies outside its original RGB image")
    result = maps.new_zeros((count, maps.shape[1]), dtype=torch.float32)
    fh, fw = maps.shape[-2:]
    if (fh, fw) != (h // FEATURE_STRIDE, w // FEATURE_STRIDE):
        raise ValueError("Feature shape is incompatible with the native stride-4 RGB convention")
    for camera in range(len(CAMERA_NAMES)):
        selected = valid & (camera_indices == camera)
        uv = pixel_uv[selected].float()
        if not len(uv):
            continue
        feature_uv = (uv - FEATURE_ORIGIN) / FEATURE_STRIDE
        grid = (2 * (feature_uv + 0.5) / feature_uv.new_tensor([fw, fh]) - 1).reshape(1, 1, -1, 2)
        sampled = F.grid_sample(
            maps[camera : camera + 1].float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        result[selected] = sampled[0, :, 0].T
    return result


@contextmanager
def _feature_precision(device_type: str):
    """Isolate frozen image arithmetic from the parent policy's AMP/TF32 setup.

    The trainer reads cached descriptors; extraction runs in its own preparation
    processes or serial online adapter. Restore every process-level backend flag
    before returning to the policy. Explicit enabled=True matters: cudnn.flags()
    otherwise disables cuDNN by default and uses a matmul-precision-sensitive
    fallback convolution implementation.
    """
    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        with (
            torch.autocast(device_type=device_type, enabled=False),
            torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False),
        ):
            yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)


class FrozenCLIPFeatures(nn.Module):
    """Frozen pretrained RN50 layer1, always eval, outside the policy optimizer."""

    def __init__(self, config: Mapping, *, device: str | torch.device = "cpu"):
        super().__init__()
        self.spec = resolve_visual_features(config)
        if self.spec["mode"] != "clip_rn50_layer1":
            raise ValueError("FrozenCLIPFeatures requires visual_features.mode=clip_rn50_layer1")
        path = ensure_checkpoint(self.spec["checkpoint"])
        # Construction must not change data sampling / policy initialization RNG.
        with torch.random.fork_rng(devices=[]):
            self.backbone = _CLIPLayer1()
        artifact = torch.jit.load(str(path), map_location="cpu")
        state = artifact.state_dict()
        expected = self.backbone.state_dict()
        selected = {name: state["visual." + name] for name in expected if "visual." + name in state}
        self.backbone.load_state_dict(selected, strict=True)
        del artifact, state
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).reshape(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).reshape(1, 3, 1, 1), persistent=False)
        self.requires_grad_(False)
        self.to(device=device, dtype=torch.float32)
        self.train(False)
        self.identity = visual_feature_identity(self.spec)
        self.signature = visual_feature_signature(self.spec)

    def train(self, mode: bool = True):
        # Parent modules calling train() must never update frozen BatchNorm.
        return super().train(False)

    @torch.no_grad()
    def dense(self, rgb: Tensor | np.ndarray) -> Tensor:
        """uint8 RGB [B,256,256,3] -> FP32 raw layer1 maps [B,256,64,64]."""
        rgb = torch.as_tensor(np.asarray(rgb).copy()) if isinstance(rgb, np.ndarray) else rgb
        if rgb.ndim != 4 or rgb.shape[1:] != (256, 256, 3) or rgb.dtype != torch.uint8:
            raise ValueError("CLIP features require native uint8 RGB [B,256,256,3]; no implicit resize")
        rgb = rgb.to(device=self.mean.device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        # Match offline and online precision regardless of surrounding policy AMP.
        with _feature_precision(self.mean.device.type):
            result = self.backbone((rgb - self.mean) / self.std)
        if result.shape != (len(rgb), FEATURE_DIM, 64, 64):
            raise RuntimeError("Unexpected pretrained CLIP RN50 layer1 feature shape")
        return result

    @torch.no_grad()
    def sample(self, observation: Mapping, camera_indices: Tensor, pixel_uv: Tensor, valid: Tensor) -> Tensor:
        rgb = torch.stack(
            [torch.as_tensor(np.asarray(observation[f"{camera}_rgb"]).copy()) for camera in CAMERA_NAMES]
        )
        maps = self.dense(rgb)
        features = sample_dense_features(maps, camera_indices, pixel_uv, valid)
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite frozen CLIP point descriptors")
        return features.to(dtype=torch.float16)

    def forward(self, rgb: Tensor | np.ndarray) -> Tensor:
        return self.dense(rgb)


def main():
    parser = argparse.ArgumentParser(description="Prepare/check the pinned CLIP RN50 checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--output", help="Override the model checkpoint path")
    parser.add_argument(
        "--download", action="store_true", help="Explicitly permit the public weight download"
    )
    args = parser.parse_args()
    if args.config:
        from stmem.config import load_config

        config = load_config(args.config)
    else:
        config = {"mode": "clip_rn50_layer1"}
    spec = resolve_visual_features(config)
    path = ensure_checkpoint(args.output or spec["checkpoint"], download=args.download)
    print(
        json.dumps(
            {
                "checkpoint": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": RN50_SHA256,
                "identity": visual_feature_identity(spec),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
