"""Low-dimensional observed state and available preceding commands, once per frame."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def command_conditioning(
    previous_action: Tensor, *, include_joint_commands: bool, include_commands: bool = True
) -> Tensor:
    """Select command inputs without modifying recorded actions or supervision.

    The joint-command ablation retains the previous gripper command. Measured
    robot state and command availability are separate fields and stay intact.
    """
    if not include_commands:
        return torch.zeros_like(previous_action)
    if include_joint_commands:
        return previous_action
    if previous_action.shape[-1] != 8:
        raise ValueError("Joint-command ablation requires seven joint targets and one gripper command")
    return torch.cat([torch.zeros_like(previous_action[..., :7]), previous_action[..., 7:]], dim=-1)


class EventEncoder(nn.Module):
    def __init__(
        self,
        width: int = 256,
        proprio_dim: int = 15,
        action_dim: int = 8,
        *,
        condition_on_previous_joint_commands: bool = True,
        condition_on_previous_commands: bool = True,
        condition_on_proprioception: bool = True,
    ):
        super().__init__()
        if not isinstance(condition_on_previous_joint_commands, bool):
            raise ValueError("model.condition_on_previous_joint_commands must be boolean")
        if not isinstance(condition_on_previous_commands, bool):
            raise ValueError("model.condition_on_previous_commands must be boolean")
        if not isinstance(condition_on_proprioception, bool):
            raise ValueError("condition_on_proprioception must be boolean")
        if not condition_on_previous_joint_commands and action_dim != 8:
            raise ValueError("Joint-command ablation requires seven joint targets and one gripper command")
        self.condition_on_previous_joint_commands = condition_on_previous_joint_commands
        self.condition_on_previous_commands = condition_on_previous_commands
        self.condition_on_proprioception = condition_on_proprioception
        self.proprio_dim, self.action_dim = proprio_dim, action_dim
        self.state = nn.Sequential(nn.Linear(proprio_dim, width), nn.SiLU(), nn.Linear(width, width))
        self.action = nn.Sequential(nn.Linear(action_dim + 1, width), nn.SiLU(), nn.Linear(width, width))
        self.combine = nn.Sequential(nn.SiLU(), nn.Linear(width, width))
        # Construct identical modules in identical order across the matched arms
        # so that suppressing a pathway does not change shared initial weights.
        self.state.requires_grad_(condition_on_proprioception)
        self.action.requires_grad_(condition_on_previous_commands)
        self.combine.requires_grad_(condition_on_proprioception or condition_on_previous_commands)

    def forward(self, proprio: Tensor, previous_action: Tensor, available: Tensor) -> Tensor:
        if proprio.shape[-1] != self.proprio_dim or previous_action.shape[-1] != self.action_dim:
            raise ValueError("State/action dimensions disagree with the canonical sensor contract")
        available = available.to(device=previous_action.device, dtype=torch.bool)
        if not self.condition_on_previous_commands:
            available = torch.zeros_like(available)
        # where, rather than multiplication, also removes NaNs in unavailable fields.
        action = torch.where(
            available[..., None], previous_action.float(), torch.zeros_like(previous_action).float()
        )
        action = command_conditioning(
            action,
            include_joint_commands=self.condition_on_previous_joint_commands,
            include_commands=self.condition_on_previous_commands,
        )
        proprio = proprio.float() if self.condition_on_proprioception else torch.zeros_like(proprio).float()
        if not torch.isfinite(proprio).all() or not torch.isfinite(action).all():
            raise ValueError("Available robot state/command must be finite")
        action_features = torch.cat([action, available[..., None].float()], dim=-1)
        # Disabled branches contribute exactly zero, including their biases.
        state_value = (
            self.state(proprio)
            if self.condition_on_proprioception
            else proprio.new_zeros(*proprio.shape[:-1], self.combine[-1].in_features)
        )
        action_value = (
            self.action(action_features)
            if self.condition_on_previous_commands
            else torch.zeros_like(state_value)
        )
        return self.combine(state_value + action_value).float()
