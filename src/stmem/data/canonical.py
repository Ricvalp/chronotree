"""Typed allowlisted observations. Privileged bookkeeping never enters a Frame."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Any
import torch
from torch import Tensor

PROPRIO_DIM = 15  # qpos7, xyz3, quaternion wxyz4, mean finger opening1
ACTION_DIM = 8
SOURCES = ("conditioning", "execution")


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

    def decision(self, index: int, horizon: int) -> Decision:
        # Unresolved samples are never made available to a substantive trainer.
        if self.alignment not in {"post_action_v1", "pre_action_v1", "synthetic_post_action_v1"}:
            raise RuntimeError(
                "Action alignment unresolved. Run stmem.data.alignment with simulator replay; substantive training is blocked."
            )
        if not 0 <= index < len(self.execution) or horizon < 1:
            raise IndexError("invalid decision index or horizon")
        offset = 0 if self.alignment == "pre_action_v1" else 1
        target = self.actions.new_zeros((horizon, ACTION_DIM))
        valid = torch.zeros_like(target, dtype=torch.bool)
        start = index + offset
        available = min(horizon, len(self.actions) - start)
        if available > 0:
            target[:available] = self.actions[start : start + available]
            valid[:available] = self.action_valid[start : start + available]
            # A missing primitive command breaks chunk continuity; do not hop over it.
            continuity = valid[:available, :7].all(-1).to(torch.int32).cumprod(0).bool()
            valid[:available] &= continuity[:, None]
        return Decision(
            self.execution[index],
            self.conditioning,
            self.execution[:index],
            target,
            valid,
            self.goal,
            self.task,
            self.episode_id,
            index,
        )

    def __len__(self):
        return len(self.execution)
