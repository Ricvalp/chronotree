"""Low-dimensional observed state and available preceding commands, once per frame."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EventEncoder(nn.Module):
    def __init__(self, width: int = 256, proprio_dim: int = 15, action_dim: int = 8):
        super().__init__()
        self.proprio_dim, self.action_dim = proprio_dim, action_dim
        self.state = nn.Sequential(nn.Linear(proprio_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.action = nn.Sequential(nn.Linear(action_dim + 1, width), nn.SiLU(), nn.Linear(width, width))
        self.combine = nn.Sequential(nn.SiLU(), nn.Linear(width, width))

    def forward(self, proprio: Tensor, previous_action: Tensor, available: Tensor) -> Tensor:
        if proprio.shape[-1] != self.proprio_dim or previous_action.shape[-1] != self.action_dim:
            raise ValueError("State/action dimensions disagree with the canonical sensor contract")
        available = available.to(device=previous_action.device, dtype=torch.bool)
        # where, rather than multiplication, also removes NaNs in unavailable fields.
        action = torch.where(
            available[..., None], previous_action.float(), torch.zeros_like(previous_action).float()
        )
        if not torch.isfinite(proprio).all() or not torch.isfinite(action).all():
            raise ValueError("Available robot state/command must be finite")
        action_features = torch.cat([action, available[..., None].float()], dim=-1)
        return self.combine(self.state(proprio.float()) + self.action(action_features)).float()
