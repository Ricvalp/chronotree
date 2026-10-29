"""Typed allowlisted observations. Privileged bookkeeping never enters a Frame."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING
import torch
from torch import Tensor

if TYPE_CHECKING:
    from stmem.data.feature_cache import FeatureRow

PROPRIO_DIM = 15  # qpos7, xyz3, quaternion wxyz4, mean finger opening1
ACTION_DIM = 8
SOURCES = ("conditioning", "execution")


def action_alignment_offset(alignment: str) -> int:
    if alignment not in {"post_action_v1", "pre_action_v1", "synthetic_post_action_v1"}:
        raise RuntimeError(
            "Action alignment unresolved. Run stmem.data.alignment with simulator replay; substantive training is blocked."
        )
    return 0 if alignment == "pre_action_v1" else 1


def valid_decision_indices(
    action_valid: Tensor, alignment: str, *, horizon: int, stride: int = 1, action_stride: int = 1
):
    """Indices with nonempty supervision, without constructing observations/chunks.

    Episode.decision masks a chunk from its first incomplete joint command on.
    Consequently, for any positive horizon a usable chunk must start with all
    seven joint dimensions available throughout its first held-action interval;
    a later command cannot rescue a gap. Strides count native recorded frames.
    The optional gripper dimension does not determine command continuity.
    """
    offset = action_alignment_offset(alignment)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (horizon, stride, action_stride)
    ):
        raise IndexError("invalid decision horizon or stride")
    usable = action_valid[offset:, :7].all(-1)
    if len(usable) < action_stride:
        return []
    if action_stride > 1:
        usable = usable.unfold(0, action_stride, 1).all(-1)
    indices = torch.arange(0, len(usable), stride, device=usable.device)
    return indices[usable[::stride]].tolist()


def retained_observations(frames, stride: int):
    """Keep phase-zero source-local observations without changing their clocks."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("observation_stride must be a positive integer")
    return [frame for frame in frames if frame.source_index % stride == 0]


@dataclass
class Frame:
    points: Tensor
    valid: Tensor
    proprio: Tensor
    previous_action: Tensor
    action_available: bool
    time: float
    source: str
    event_id: str
    source_index: int = -1
    # Geometry-only anchors into this frame's original sampled points, keyed by
    # requested quota. Invalid/padded slots are -1; never truncate another quota.
    center_indices: dict[int, Tensor] = field(default_factory=dict)
    center_sampling_signature: str | None = None
    # Optional frozen RGB descriptors BEFORE the trainable normalization/projection.
    # These share precisely the canonical sampled-point order and validity mask.
    visual_features: Tensor | FeatureRow | None = None
    point_camera: Tensor | None = None
    point_uv: Tensor | None = None
    visual_feature_signature: str | None = None

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[-1] != 6:
            raise ValueError("points must be [N,6] xyzrgb")
        if self.valid.shape != self.points.shape[:1] or self.valid.dtype != torch.bool:
            raise ValueError("valid must be bool [N] with True meaning observed")
        if self.proprio.shape != (PROPRIO_DIM,) or self.previous_action.shape != (ACTION_DIM,):
            raise ValueError("proprio/previous_action must have dimensions 15/8")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source}")
        if self.source == "conditioning" and self.action_available:
            raise ValueError("conditioning demonstrator commands are not observable")
        if not self.action_available:
            self.previous_action = torch.zeros_like(self.previous_action)
        if self.source_index < 0:
            self.source_index = int(self.time)
        if self.visual_features is not None:
            if self.visual_features.shape != (len(self.points), 256):
                raise ValueError("frozen visual_features must be [N,256] in sampled-point order")
            if not self.visual_features.dtype.is_floating_point:
                raise ValueError("frozen visual_features must be floating point")
        if (self.point_camera is None) != (self.point_uv is None):
            raise ValueError("point_camera and point_uv provenance must be supplied together")
        if self.point_camera is not None:
            if self.point_camera.shape != (len(self.points),) or self.point_camera.dtype != torch.long:
                raise ValueError("point_camera must be int64 [N]")
            if self.point_uv.shape != (len(self.points), 2) or self.point_uv.dtype != torch.long:
                raise ValueError("point_uv must be int64 [N,2]")
        for count, indices in self.center_indices.items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValueError("center_indices keys must be positive integer quotas")
            if indices.shape != (count,) or indices.dtype != torch.long:
                raise ValueError("center_indices must contain int64 [quota] tensors")
            # Full index/content checks happen once at CPU ingestion. Avoid
            # synchronizing every Frame.to(cuda) or packed-batch reconstruction.
            if indices.device.type == "cpu" and self.points.device.type == "cpu":
                occupied = indices[indices >= 0]
                if (indices < -1).any() or (occupied >= len(self.points)).any():
                    raise ValueError("center index outside sampled point cloud")
                if len(occupied.unique()) != len(occupied):
                    raise ValueError("center indices must be distinct observed points")
                if not self.valid[occupied].all() or not torch.isfinite(self.points[occupied]).all():
                    raise ValueError("center indices must reference valid finite points")

    @property
    def xyz(self):
        return self.points[:, :3]

    @property
    def rgb(self):
        return self.points[:, 3:]

    @property
    def point_valid(self):
        return self.valid

    def to(self, device):
        return replace(
            self,
            points=self.points.to(device),
            valid=self.valid.to(device),
            proprio=self.proprio.to(device),
            previous_action=self.previous_action.to(device),
            center_indices={count: indices.to(device) for count, indices in self.center_indices.items()},
            visual_features=None if self.visual_features is None else self.visual_features.to(device),
            point_camera=None if self.point_camera is None else self.point_camera.to(device),
            point_uv=None if self.point_uv is None else self.point_uv.to(device),
        )


@dataclass
class Decision:
    current: Frame
    conditioning: list[Frame]
    execution_prefix: list[Frame]
    target_chunk: Tensor
    target_valid: Tensor
    goal: str
    task: str = ""
    episode_id: str = ""
    index: int = 0

    @property
    def target(self):
        return self.target_chunk

    def to(self, device):
        return replace(
            self,
            current=self.current.to(device),
            conditioning=[f.to(device) for f in self.conditioning],
            execution_prefix=[f.to(device) for f in self.execution_prefix],
            target_chunk=self.target_chunk.to(device),
            target_valid=self.target_valid.to(device),
        )


@dataclass
class Episode:
    conditioning: list[Frame]
    execution: list[Frame]
    actions: Tensor
    action_valid: Tensor
    goal: str
    task: str = ""
    episode_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    alignment: str = "unresolved"
    # Lists/actions keep native recorded rows. Only decision construction samples
    # observations and supervised held-action targets; prepared files stay intact.
    observation_stride: int = 1
    action_stride: int = 1

    def __post_init__(self):
        for name in ("observation_stride", "action_stride"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def decision(self, index: int, horizon: int) -> Decision:
        # Unresolved samples are never made available to a substantive trainer.
        offset = action_alignment_offset(self.alignment)
        if not 0 <= index < len(self.execution) or horizon < 1:
            raise IndexError("invalid decision index or horizon")
        if self.execution[index].source_index % self.observation_stride:
            raise IndexError("decision must lie on the retained observation cadence")
        target = self.actions.new_zeros((horizon, ACTION_DIM))
        valid = torch.zeros_like(target, dtype=torch.bool)
        start = index + offset
        # Predict the first native command of each hold interval. Every native
        # command in that interval must exist: a final half interval is padding.
        available = min(horizon, max(0, (len(self.actions) - start) // self.action_stride))
        if available > 0:
            end = start + available * self.action_stride
            target[:available] = self.actions[start : end : self.action_stride]
            valid[:available] = (
                self.action_valid[start:end].reshape(available, self.action_stride, ACTION_DIM).all(1)
            )
            # A missing primitive command breaks chunk continuity; do not hop over it.
            continuity = valid[:available, :7].all(-1).to(torch.int32).cumprod(0).bool()
            valid[:available] &= continuity[:, None]
        return Decision(
            self.execution[index],
            retained_observations(self.conditioning, self.observation_stride),
            retained_observations(self.execution[:index], self.observation_stride),
            target,
            valid,
            self.goal,
            self.task,
            self.episode_id,
            index,
        )

    def __len__(self):
        return len(self.execution)
