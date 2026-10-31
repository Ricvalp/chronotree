"""Explicit validation-only changes to the number of decoded targets executed."""

from __future__ import annotations

from copy import deepcopy

from stmem.config import validate_config
from stmem.training.runtime import validate_policy_config


def evaluation_execution_config(checkpoint_cfg, cfg, *, execute_actions=None, split="val"):
    """Preserve the learned contract while allowing one declared rollout ablation.

    The action horizon and cadence remain fixed. ``execute_actions`` counts
    predicted targets, each held for the checkpoint's native action stride.
    No checkpoint/config object supplied by the caller is mutated. The default
    path keeps the strict global policy compatibility check, including execute.
    """
    if execute_actions is None:
        validate_policy_config(checkpoint_cfg, cfg)
        return deepcopy(cfg), None
    if split != "val":
        raise ValueError("--execute-actions is a validation-only execution ablation; test remains frozen")
    horizon = checkpoint_cfg["model"]["horizon"]
    if (not isinstance(execute_actions, int) or isinstance(execute_actions, bool)
            or not 1 <= execute_actions <= horizon):
        raise ValueError(f"--execute-actions must be an integer between 1 and checkpoint horizon {horizon}")
    effective = deepcopy(cfg)
    effective["model"]["execute"] = execute_actions
    # Authorize exactly this field, locally. Architecture, horizon, sensing,
    # action/observation stride and conditioning continue to be checked.
    compatible = deepcopy(effective)
    original = checkpoint_cfg["model"]["execute"]
    compatible["model"]["execute"] = original
    validate_policy_config(checkpoint_cfg, compatible)
    validate_config(effective)
    stride = effective["model"].get("action_stride", 1)
    record = {
        "scope": "validation_execution_length_ablation",
        "units": "predicted_action_targets",
        "checkpoint_execute_actions": original,
        "execute_actions": execute_actions,
        "predicted_actions_per_chunk": horizon,
        "action_stride": stride,
        "observation_stride": effective["data"].get("observation_stride", 1),
        "checkpoint_replan_primitive_steps": original * stride,
        "replan_primitive_steps": execute_actions * stride,
        "command_interpolation": "zero_order_hold",
    }
    return effective, record
