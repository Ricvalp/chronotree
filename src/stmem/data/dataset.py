"""Per-process HDF5 reading and task-balanced episode/decision sampling."""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import random
import h5py
import numpy as np
import torch
from .canonical import (
    ACTION_DIM,
    PROPRIO_DIM,
    Frame,
    Episode,
    Decision,
    aligned_action_chunk,
    action_alignment_offset,
    valid_decision_indices,
)
from .prepare import numeric_keys, FORMAT_VERSION, PREPARED_SUBDIR


@dataclass(frozen=True)
class EpisodeRef:
    path: Path
    key: str
    task: str
    execution_length: int
    metadata: dict


@dataclass(frozen=True)
class EpisodeActions:
    """Execution supervision and source lengths; no observation payloads."""

    actions: torch.Tensor
    action_valid: torch.Tensor
    execution_length: int
    conditioning_length: int
    episode_id: str
    alignment: str


class EpisodeDataset:
    def __init__(
        self,
        root=f"data/{PREPARED_SUBDIR}",
        tasks=None,
        horizon=16,
        alignment_path=None,
        *,
        visual_features_config=None,
    ):
        self.root = Path(root)
        self.horizon = horizon
        self.observation_stride = 1
        self.action_stride = 1
        self.episodes = []
        self.by_task = {}
        data_root = self.root.parent.parent if self.root.is_file() else self.root.parent
        self.alignment_path = Path(alignment_path) if alignment_path else data_root / "alignment.json"
        self.alignment = json.loads(self.alignment_path.read_text()) if self.alignment_path.exists() else {}
        if self.root.is_file() and self.root.name.endswith(".partial.h5"):
            raise ValueError("An incomplete .partial.h5 preparation file cannot be used as a dataset")
        files = (
            sorted(path for path in self.root.glob("*.h5") if not path.name.endswith(".partial.h5"))
            if self.root.is_dir()
            else [self.root]
        )
        for path in files:
            with h5py.File(path, "r") as f:
                manifest = json.loads(f.attrs["manifest"])
                if tasks is not None and manifest["task"] not in tasks:
                    continue
                for key in numeric_keys(f, "episode_"):
                    n = int((~np.asarray(f[key]["source"])).sum())
                    ref = EpisodeRef(path, key, manifest["task"], n, manifest)
                    self.by_task.setdefault(ref.task, []).append(len(self.episodes))
                    self.episodes.append(ref)
        if not self.episodes:
            raise FileNotFoundError(f"No prepared episodes at {self.root}; run python -m stmem.data.prepare")
        self.feature_identity = None
        self.feature_paths = {}
        self.feature_manifests = {}
        self.feature_signature = None
        if visual_features_config is not None:
            self._configure_temporal_sampling(visual_features_config)
            self._configure_features(visual_features_config)

    def _configure_temporal_sampling(self, config):
        for name, section in (("observation_stride", "data"), ("action_stride", "model")):
            value = config.get(section, {}).get(name, 1)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{section}.{name} must be a positive integer")
            setattr(self, name, value)

    def _configure_features(self, config):
        from stmem.vision.frozen import (
            resolve_visual_features,
            visual_feature_identity,
            visual_feature_signature,
        )
        from .feature_cache import feature_cache_manifest
        from .stage_features import consumer_feature_cache_root

        spec = resolve_visual_features(config)
        if spec["mode"] == "xyzrgb":
            if self.feature_identity is not None:
                raise ValueError("A frozen-feature dataset cannot be reconfigured as XYZRGB")
            return
        identity = visual_feature_identity(spec)
        if self.feature_identity is not None:
            if self.feature_identity != identity:
                raise ValueError("Frozen feature identity changed after dataset construction")
            return
        for ref in self.episodes:
            if ref.task in self.feature_paths:
                continue
            path = consumer_feature_cache_root(config) / f"{ref.task}.h5"
            self.feature_manifests[ref.task] = feature_cache_manifest(path, ref.metadata, spec)
            self.feature_paths[ref.task] = path
        self.feature_identity = identity
        self.feature_signature = visual_feature_signature(spec)

    def _require_feature_episode(self, ref):
        if self.feature_identity is not None:
            layout = self.feature_manifests[ref.task]["episodes"]
            if ref.key not in layout:
                raise ValueError(
                    f"Frozen feature cache lacks {ref.task}/{ref.key}; prepare all requested training episodes"
                )

    def validate_preprocessing(self, config):
        from stmem.config import require_sampling_contract
        from .pointcloud import (
            DEFAULT_CROP,
            CALIBRATION_VERSION,
            POINT_SAMPLING_VERSION,
            point_sampler_settings,
        )
        from .center_sampling import CENTER_SAMPLING_VERSION, center_sampler_settings, center_counts

        require_sampling_contract(config)
        expected = {
            "format": FORMAT_VERSION,
            "points": config["model"]["points"],
            "crop": config.get("geometry", {}).get("crop", DEFAULT_CROP),
            "voxel_size": config.get("geometry", {}).get("voxel_size", 0.005),
            "max_depth_m": config.get("geometry", {}).get("max_depth", 4.0),
            "calibration": CALIBRATION_VERSION,
            "point_sampling_version": POINT_SAMPLING_VERSION,
            "point_sampling": point_sampler_settings(config),
            "center_sampling_version": CENTER_SAMPLING_VERSION,
            "center_sampling": center_sampler_settings(config),
            "center_counts": center_counts(config),
        }
        for ref in self.episodes:
            for key, value in expected.items():
                observed = ref.metadata.get(key, 4.0 if key == "max_depth_m" else None)
                same = (
                    np.allclose(observed, value)
                    if key in ("crop", "voxel_size", "max_depth_m") and observed is not None
                    else observed == value
                )
                if not same:
                    raise ValueError(
                        f"{ref.task}: prepared {key}={observed} differs from configuration {value}; re-run prepare with matching config"
                    )
        self._configure_temporal_sampling(config)
        self._configure_features(config)

    def __len__(self):
        return len(self.episodes)

    def _alignment_for(self, ref):
        alignment = self.alignment.get("convention", "unresolved")
        if self.alignment.get("dataset_revision") != ref.metadata["dataset_revision"]:
            alignment = "unresolved"
        if not self.alignment.get("verified", False):
            alignment = "unresolved"
        # Exact raw evidence identity prevents applying an audit to silently changed task data.
        audited_hashes = self.alignment.get("raw_sha256", {})
        if ref.task in audited_hashes and audited_hashes[ref.task] != ref.metadata["raw_sha256"]:
            alignment = "unresolved"
        return alignment

    @staticmethod
    def _validate_episode_layout(group):
        """Preserve structural checks without reading/decompressing observations."""
        source = group["source"]
        points = group["points"]
        if source.ndim != 1 or source.dtype != np.dtype(bool):
            raise ValueError(f"{group.name}: source must be bool [frames]")
        frames = source.shape[0]
        if points.ndim != 3 or points.shape[0] != frames or points.shape[-1] != 6:
            raise ValueError(f"{group.name}: points must be [frames,N,6] xyzrgb")
        shapes = {
            "valid": (frames, points.shape[1]),
            "proprio": (frames, PROPRIO_DIM),
            "actions": (frames, ACTION_DIM),
            "action_valid": (frames, ACTION_DIM),
            "source_index": (frames,),
            "raw_index": (frames,),
        }
        for name, shape in shapes.items():
            if group[name].shape != shape:
                raise ValueError(f"{group.name}: {name} must have shape {shape}")
        for name in ("valid", "action_valid"):
            if group[name].dtype != np.dtype(bool):
                raise ValueError(f"{group.name}: {name} must be bool")
        manifest = json.loads(group.file.attrs["manifest"])
        if manifest.get("format") == FORMAT_VERSION:
            counts = manifest.get("center_counts", [])
            if (
                not counts
                or "center_indices" not in group
                or set(group["center_indices"]) != {str(c) for c in counts}
            ):
                raise ValueError(f"{group.name}: missing prepared center indices; re-run prepare")
            for count in counts:
                centers = group["center_indices"][str(count)]
                if centers.shape != (frames, count) or centers.dtype != np.dtype("i8"):
                    raise ValueError(f"{group.name}: center indices must be int64 [frames,{count}]")

    def load_episode_actions(self, index):
        """Read only small source/action arrays for startup indexing/statistics."""
        ref = self.episodes[index]
        self._require_feature_episode(ref)
        alignment = self._alignment_for(ref)
        action_alignment_offset(alignment)  # Fail before exposing unverified supervision.
        with h5py.File(ref.path, "r") as f:
            group = f[ref.key]
            self._validate_episode_layout(group)
            source = np.asarray(group["source"])
            execution = ~source
            actions = torch.from_numpy(np.asarray(group["actions"])[execution])
            valid = torch.from_numpy(np.asarray(group["action_valid"])[execution])
        return EpisodeActions(actions, valid, len(actions), int(source.sum()), ref.key, alignment)

    def load_episode(self, index):
        from .center_sampling import center_sampling_signature

        ref = self.episodes[index]
        self._require_feature_episode(ref)
        alignment = self._alignment_for(ref)
        with h5py.File(ref.path, "r") as f:
            g = f[ref.key]
            self._validate_episode_layout(g)
            feature_rows, point_camera, point_uv = None, None, None
            if self.feature_identity is not None:
                from .feature_cache import episode_feature_rows

                feature_rows, point_camera, point_uv = episode_feature_rows(
                    self.feature_paths[ref.task], ref.key, g, self.feature_signature
                )
            center_signature = None
            centers = {}
            if ref.metadata.get("format") == FORMAT_VERSION:
                center_signature = center_sampling_signature(ref.metadata["center_sampling"])
                centers = {
                    int(count): torch.from_numpy(np.asarray(values))
                    for count, values in g["center_indices"].items()
                }
            source = np.asarray(g["source"])
            actions = torch.from_numpy(np.asarray(g["actions"]))
            masks = torch.from_numpy(np.asarray(g["action_valid"]))
            indices = np.asarray(g["source_index"])
            raw_indices = np.asarray(g["raw_index"])
            frames = {"conditioning": [], "execution": []}
            execution_rows = np.where(~source)[0]
            exec_position = {row: i for i, row in enumerate(execution_rows)}
            for row in range(len(source)):
                bank = "conditioning" if source[row] else "execution"
                previous = torch.zeros(8)
                available = False
                if bank == "execution" and exec_position[row] > 0 and alignment != "unresolved":
                    action_row = (
                        row if alignment == "post_action_v1" else execution_rows[exec_position[row] - 1]
                    )
                    previous = actions[action_row].clone()
                    available = bool(masks[action_row, :7].all())
                frames[bank].append(
                    Frame(
                        torch.from_numpy(np.asarray(g["points"][row])),
                        torch.from_numpy(np.asarray(g["valid"][row])),
                        torch.from_numpy(np.asarray(g["proprio"][row])),
                        previous,
                        available,
                        float(indices[row]),
                        bank,
                        f"{ref.task}/{ref.key}/{int(raw_indices[row])}",
                        int(indices[row]),
                        center_indices={count: values[row] for count, values in centers.items()},
                        center_sampling_signature=center_signature,
                        visual_features=None if feature_rows is None else feature_rows[row],
                        point_camera=None if point_camera is None else point_camera[row],
                        point_uv=None if point_uv is None else point_uv[row],
                        visual_feature_signature=self.feature_signature,
                    )
                )
            return Episode(
                frames["conditioning"],
                frames["execution"],
                actions[execution_rows],
                masks[execution_rows],
                str(g.attrs["goal"]),
                ref.task,
                ref.key,
                ref.metadata,
                alignment,
                observation_stride=self.observation_stride,
                action_stride=self.action_stride,
            )

    def get_decision(self, episode_index, decision_index, horizon=None):
        return self.load_episode(episode_index).decision(decision_index, horizon or self.horizon)

    def load_current_decisions(self, index, decision_indices, horizon=None):
        """Current-only ablation: read requested observations, never entire episodes.

        Small action/source arrays retain the exact canonical alignment and masks.
        Proprioception remains available to diagnostic plots; the current-only
        policy excludes it from every model input.
        """
        from .center_sampling import center_sampling_signature

        indices = list(decision_indices)
        if not indices:
            return []
        ref = self.episodes[index]
        self._require_feature_episode(ref)
        alignment = self._alignment_for(ref)
        action_alignment_offset(alignment)
        horizon = self.horizon if horizon is None else horizon
        with h5py.File(ref.path, "r") as handle:
            group = handle[ref.key]
            self._validate_episode_layout(group)
            execution_rows = np.flatnonzero(~np.asarray(group["source"]))
            if any(not 0 <= i < len(execution_rows) for i in indices):
                raise IndexError("invalid decision index")
            rows = sorted({int(execution_rows[i]) for i in indices})
            positions = {row: pos for pos, row in enumerate(rows)}
            actions = torch.from_numpy(np.asarray(group["actions"])[execution_rows])
            masks = torch.from_numpy(np.asarray(group["action_valid"])[execution_rows])
            source_indices = np.asarray(group["source_index"][rows])
            if any(i % self.observation_stride for i in source_indices):
                raise IndexError("decision must lie on the retained observation cadence")
            raw_indices = np.asarray(group["raw_index"][rows])
            points = torch.from_numpy(np.asarray(group["points"][rows]))
            valid = torch.from_numpy(np.asarray(group["valid"][rows]))
            proprio = torch.from_numpy(np.asarray(group["proprio"][rows]))
            centers, signature = {}, None
            if ref.metadata.get("format") == FORMAT_VERSION:
                signature = center_sampling_signature(ref.metadata["center_sampling"])
                centers = {
                    int(count): torch.from_numpy(np.asarray(values[rows]))
                    for count, values in group["center_indices"].items()
                }
            features, cameras, uv = None, None, None
            if self.feature_identity is not None:
                from .feature_cache import episode_feature_rows

                features, cameras, uv = episode_feature_rows(
                    self.feature_paths[ref.task],
                    ref.key,
                    group,
                    self.feature_signature,
                    selected_rows=rows,
                )
            frames = [
                Frame(
                    points[pos],
                    valid[pos],
                    proprio[pos],
                    torch.zeros(ACTION_DIM),
                    False,
                    float(source_indices[pos]),
                    "execution",
                    f"{ref.task}/{ref.key}/{int(raw_indices[pos])}",
                    int(source_indices[pos]),
                    center_indices={count: values[pos] for count, values in centers.items()},
                    center_sampling_signature=signature,
                    visual_features=None if features is None else features[pos],
                    point_camera=None if cameras is None else cameras[pos],
                    point_uv=None if uv is None else uv[pos],
                    visual_feature_signature=self.feature_signature,
                )
                for pos in range(len(rows))
            ]
            decisions = []
            for i in indices:
                target, target_valid = aligned_action_chunk(
                    actions, masks, i, horizon, alignment, self.action_stride
                )
                decisions.append(
                    Decision(
                        frames[positions[int(execution_rows[i])]],
                        [],
                        [],
                        target,
                        target_valid,
                        str(group.attrs["goal"]),
                        ref.task,
                        ref.key,
                        i,
                    )
                )
        return decisions

    def sample(self, rng=None):
        rng = rng or random
        task = rng.choice(sorted(self.by_task))
        idx = rng.choice(self.by_task[task])
        episode = self.load_episode(idx)
        # No supervised decisions beyond the final real expert command.
        candidates = valid_decision_indices(
            episode.action_valid,
            episode.alignment,
            horizon=self.horizon,
            stride=episode.observation_stride,
            action_stride=episode.action_stride,
        )
        if not candidates:
            raise ValueError(f"No valid action targets in {episode.episode_id}")
        return episode.decision(rng.choice(candidates), self.horizon)


# Explicitly synthetic diagnostic generator; never represents a benchmark result.
def synthetic_episode(length=12, conditioning_length=3, points=32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    actions = torch.randn(length, 8, generator=gen) * 0.1

    def make(i, source):
        p = torch.rand((points, 6), generator=gen)
        p[:, :3] = p[:, :3] * 0.3
        state = torch.zeros(15)
        state[:7] = actions[min(i, length - 1), :7]
        state[10] = 1
        return Frame(
            p,
            torch.ones(points, dtype=torch.bool),
            state,
            actions[i].clone() if source == "execution" and i > 0 else torch.zeros(8),
            source == "execution" and i > 0,
            float(i),
            source,
            f"{seed}/{source}/{i}",
        )

    return Episode(
        [make(i, "conditioning") for i in range(conditioning_length)],
        [make(i, "execution") for i in range(length)],
        actions,
        torch.ones_like(actions, dtype=torch.bool),
        "Move the object to the shown location.",
        "synthetic",
        str(seed),
        {"synthetic": True},
        "synthetic_post_action_v1",
    )
