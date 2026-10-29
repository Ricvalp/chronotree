"""Immutable frozen descriptors, with bounded, process-safe lazy row reads.

The cache stores pre-projection FP16 descriptors in contiguous HDF5 datasets.
Training workers pass small row references instead of placing whole episodes of
descriptors in the multiprocessing queue. Only a bounded set of the local
encoder's upcoming block batches is materialized. No HDF5 handle or mmap object
crosses a process boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch

from stmem.config import digest
from .preparation_state import IDENTITY_FIELDS


FEATURE_CACHE_VERSION = "clip_point_descriptors_v1"
FEATURE_DIM = 256


def prepared_feature_identity(manifest: dict) -> str:
    """Geometry/source recipe identity; paths and preparation walltime may move."""
    missing = IDENTITY_FIELDS - manifest.keys()
    if missing:
        raise ValueError(f"Prepared geometry manifest lacks identity fields: {sorted(missing)}")
    return digest({name: manifest[name] for name in sorted(IDENTITY_FIELDS)})


@dataclass(frozen=True)
class FeatureRow:
    """Picklable reference to one immutable, already validated descriptor row."""

    path: str
    episode: str
    row: int
    point_count: int
    signature: str
    # Absolute byte position, obtained from a contiguous HDF5 dataset. Keeping
    # this in the reference avoids reopening HDF5 for every historical frame.
    offset: int

    @property
    def shape(self):
        return (self.point_count, FEATURE_DIM)

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        return torch.float16

    @property
    def device(self):
        return torch.device("cpu")

    def load(self) -> torch.Tensor:
        return load_feature_rows([self])[0]

    def to(self, device=None, dtype=None, **kwargs) -> torch.Tensor:
        return self.load().to(device=device, dtype=dtype, **kwargs)


def load_feature_rows(
    rows: Sequence[FeatureRow | torch.Tensor | None],
    *,
    point_count: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load one bounded set of rows, zero-padding absent frames/points.

    One read-only file mapping is used per path within this call and then
    released. Returned tensors own their storage; no writable aliases to the
    cache remain. This is also the common materializer for online tensors.
    """
    with FeatureRowReader() as reader:
        result = reader.read(rows, point_count=point_count)
    return result.to(device)


class FeatureRowReader:
    """Process/thread-local immutable mappings and caller-owned CPU buffers.

    The prefetch producer keeps one reader alive for its bounded sequence of
    chunks. Only padding is zeroed: complete descriptor rows overwrite every
    byte. No mapped storage is returned, and no handle crosses a process.
    """

    def __init__(self):
        self.maps = {}

    def read(self, rows, *, point_count=None, out=None):
        if point_count is None:
            point_count = max((row.shape[0] for row in rows if row is not None), default=0)
        if point_count < 0:
            raise ValueError("point_count cannot be negative")
        shape = (len(rows), point_count, FEATURE_DIM)
        result = torch.empty(shape, dtype=torch.float16) if out is None else out
        if (
            tuple(result.shape) != shape
            or result.dtype != torch.float16
            or result.device.type != "cpu"
            or not result.is_contiguous()
        ):
            raise ValueError("Descriptor output must be a contiguous CPU FP16 buffer of the exact shape")
        array = result.numpy()
        for index, row in enumerate(rows):
            if row is None:
                result[index].zero_()
                continue
            if not isinstance(row, (FeatureRow, torch.Tensor)):
                raise TypeError("Expected a FeatureRow, descriptor Tensor, or padded None")
            if row.ndim != 2 or row.shape[1] != FEATURE_DIM or row.shape[0] > point_count:
                raise ValueError("Frozen descriptors must be [N,256] within the point padding")
            if isinstance(row, FeatureRow):
                mapping = self.maps.get(row.path)
                if mapping is None:
                    mapping = self.maps[row.path] = np.memmap(row.path, mode="r", dtype=np.uint8)
                byte_count = row.point_count * FEATURE_DIM * np.dtype("f2").itemsize
                if row.offset < 0 or row.offset + byte_count > mapping.nbytes:
                    raise ValueError(f"Truncated descriptor cache: {row.path}")
                value = np.ndarray(row.shape, dtype=np.dtype("<f2"), buffer=mapping, offset=row.offset)
                np.copyto(array[index, : row.point_count], value)
                del value
            elif isinstance(row, torch.Tensor):
                result[index, : len(row)].copy_(row.detach().to(device="cpu", dtype=torch.float16))
            if row.shape[0] < point_count:
                result[index, row.shape[0] :].zero_()
        return result

    def close(self):
        for mapping in self.maps.values():
            mapping._mmap.close()
        self.maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def feature_cache_manifest(path: Path, prepared: dict, config: dict) -> dict:
    """Validate a complete sidecar without reading its descriptor payload."""
    from stmem.vision.frozen import visual_feature_identity, visual_feature_signature

    path = Path(path)
    if path.name.endswith(".partial.h5"):
        raise ValueError("Incomplete feature sidecars cannot be consumed by a policy")
    try:
        with h5py.File(path, "r") as handle:
            manifest = json.loads(handle.attrs["manifest"])
            if manifest.get("format") != FEATURE_CACHE_VERSION:
                raise ValueError(f"Unsupported frozen feature cache format at {path}")
            if not manifest.get("complete") or manifest.get("diagnostic", False):
                raise ValueError(f"Incomplete/diagnostic feature cache cannot train a policy: {path}")
            if manifest.get("prepared_identity") != prepared_feature_identity(prepared):
                raise ValueError(f"Feature cache geometry/source identity differs from {path}")
            if manifest.get("feature_identity") != visual_feature_identity(config):
                raise ValueError(f"Feature cache backbone/preprocessing identity differs from config: {path}")
            if manifest.get("feature_signature") != visual_feature_signature(config):
                raise ValueError(f"Feature cache signature differs from config: {path}")
            layout = manifest.get("episodes", {})
            if not isinstance(layout, dict) or not layout:
                raise ValueError(f"Feature cache has no completed episodes: {path}")
            for key, rows in layout.items():
                if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
                    raise ValueError(f"Invalid feature episode length for {key}")
                group = handle[key]
                values = group["features"]
                expected_shape = (rows, int(prepared["points"]), FEATURE_DIM)
                if (
                    values.shape != expected_shape
                    or values.dtype != np.dtype("<f2")
                    or values.chunks is not None
                    or values.id.get_offset() is None
                    or int(group.attrs.get("completed_rows", -1)) != rows
                ):
                    raise ValueError(f"Invalid contiguous feature layout for {path}:{key}")
                for name, shape, dtype in (
                    ("point_camera", expected_shape[:2], "i1"),
                    ("point_uv", (*expected_shape[:2], 2), "i2"),
                    ("raw_index", (rows,), "i8"),
                    ("source_index", (rows,), "i8"),
                    ("source", (rows,), "?"),
                ):
                    if group[name].shape != shape or group[name].dtype != np.dtype(dtype):
                        raise ValueError(f"Invalid feature provenance {path}:{key}/{name}")
            return manifest
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Missing frozen descriptors {path}; run python -m stmem.data.prepare_features with this config"
        ) from error


def episode_feature_rows(path: Path, key: str, prepared_group, signature: str):
    """Return lazy feature rows and small camera/UV sidecars for one episode."""
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise ValueError(f"Feature cache lacks {key}: {path}; prepare the requested training episodes")
        group = handle[key]
        rows, points = prepared_group["points"].shape[:2]
        values = group["features"]
        if values.shape != (rows, points, FEATURE_DIM):
            raise ValueError(f"Feature rows do not match prepared episode {path}:{key}")
        for name in ("raw_index", "source_index", "source"):
            if not np.array_equal(group[name][()], prepared_group[name][()]):
                raise ValueError(f"Feature {name} ordering differs from prepared episode {path}:{key}")
        offset = values.id.get_offset()
        if offset is None or values.chunks is not None or values.dtype != np.dtype("<f2"):
            raise ValueError("Frozen descriptors must use contiguous FP16 storage")
        stride = points * FEATURE_DIM * np.dtype("f2").itemsize
        refs = [
            FeatureRow(str(path.resolve()), key, row, points, signature, offset + row * stride)
            for row in range(rows)
        ]
        # Provenance is small (~10KiB/frame). Unlike descriptor tensors, it can
        # be retained in the ordinary two-episode CPU geometry LRU.
        cameras = torch.from_numpy(np.asarray(group["point_camera"], dtype=np.int64))
        uv = torch.from_numpy(np.asarray(group["point_uv"], dtype=np.int64))
    return refs, cameras, uv
