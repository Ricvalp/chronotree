"""Fixed-current history read ablations alongside ordinary FULL rollouts.

These probes never execute their altered actions. Each uses the original
decision's current observation, instruction, clocks and exact noise tensor.
History erasure is an out-of-distribution sensitivity diagnostic; action change
alone does not establish useful semantic memory or improved task completion.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import random
import time
import traceback

import numpy as np
import torch

from stmem.evaluation.robomme_adapter import physical_action
from stmem.models.policy import Evidence
from stmem.visualization.kinematics import panda_tcp_positions


ARMS = ("FULL", "erase_conditioning", "erase_execution", "CURRENT")


@contextmanager
def _preserve_rng():
    """Diagnostics cannot consume the normal rollout's global random streams."""
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def history_read_records(records, arm):
    """Change only the records read, keeping chronological leaf order intact."""
    if arm == "FULL":
        return list(records)
    if arm == "CURRENT":
        return []
    source = {"erase_conditioning": "conditioning", "erase_execution": "execution"}.get(arm)
    if source is None:
        raise ValueError(f"Unknown probe arm: {arm}")
    return [record for record in records if record.source != source]


@torch.no_grad()
def probe_same_current(online, *, task, generator_state, executed_actions, decision_index, atol=1e-5):
    """Replay one actual decision under four reads without changing policy state.

    Call immediately after OnlinePolicy.decide(), before simulator advancement.
    ``generator_state`` is captured immediately before that actual decision.
    The return is JSON-compatible; all actions are diagnostic absolute targets.
    """
    if online.method != "FULL" or online.intervention != "none":
        raise ValueError("The reference must be ordinary FULL without interventions")
    policy = online.policy
    if policy.training:
        raise ValueError("Fixed-current probes require evaluation mode")
    started = time.perf_counter()
    original_generator = online.generator.get_state().clone()
    original_selection = online.selection_rng.getstate()
    original_clocks = online.clocks.copy()
    with _preserve_rng():
        clone = torch.Generator(device=policy.device)
        clone.set_state(generator_state)
        noise = torch.randn(
            (policy.decoder.horizon, policy.decoder.dimensions),
            device=policy.device, generator=clone,
        )
        archive = online.memory.archive
        current_records = policy.encoder.current(online.memory.current)
        evidence = Evidence(archive, current_records, online.text, online.memory.current, original_clocks.copy())
        records = list(archive.leaves)
        arms = {}
        for arm in ARMS:
            selected = history_read_records(records, arm)
            with torch.autocast(policy.device.type, dtype=online.autocast_dtype,
                                enabled=online.precision != "fp32"):
                context, bias = policy.context(evidence, selected)
                normalized = policy.decoder.integrate(
                    context, bias, steps=online.cfg["model"]["flow_steps"], noise=noise,
                )
            raw = policy.normalizer.denormalize(normalized).float().cpu().numpy()
            physical = np.stack([physical_action(action, task) for action in raw])
            xyz = panda_tcp_positions(physical[:, :7], task=task)
            arms[arm] = {
                "raw_actions": raw.tolist(), "physical_actions": physical.tolist(),
                "tcp_targets_m": xyz.tolist(), "history_leaf_count": len(selected),
                "history_source_counts": dict(Counter(record.source for record in selected)),
                "context_tokens": int(context.shape[-2]),
            }
        actual = executed_actions.detach().float().cpu().numpy()
        full = np.asarray(arms["FULL"]["raw_actions"])
        replay_error = float(np.max(np.abs(actual - full)))
        full_xyz = np.asarray(arms["FULL"]["tcp_targets_m"])
        execute = online.cfg["model"]["execute"]
        for arm, row in arms.items():
            delta = np.linalg.norm(np.asarray(row["tcp_targets_m"]) - full_xyz, axis=-1) * 1000
            row["tcp_change_mean_mm"] = float(delta.mean())
            row["tcp_change_executed_targets_mean_mm"] = float(delta[:execute].mean())
            row["tcp_change_last_target_mm"] = float(delta[-1])
            row["max_abs_joint_target_change_rad"] = float(np.max(np.abs(np.asarray(row["raw_actions"])[:, :7] - full[:, :7])))
        result = {
            "decision_index": int(decision_index),
            "execution_source_index": int(online.memory.current.source_index),
            "clocks": original_clocks, "task_goal": str(online.goal),
            "flow_steps": int(online.cfg["model"]["flow_steps"]),
            "executed_targets_per_chunk": int(execute),
            "noise_sha256": hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(),
            "full_replay_max_abs_action_error": replay_error,
            "full_replay_tolerance": atol, "full_replay_passed": replay_error <= atol,
            "controls": {"same_current_observation": True, "same_current_encoding": True,
                         "same_goal": True, "same_clocks": True, "same_flow_noise": True,
                         "history_tokens_removed_only": True,
                         "counterfactual_actions_executed": False},
            "arms": arms,
        }
    # Fail loudly if any future implementation accidentally consumes a stream
    # owned by the actual policy or edits its decision clocks.
    if not torch.equal(online.generator.get_state(), original_generator):
        raise AssertionError("History probe advanced the rollout's flow generator")
    if online.selection_rng.getstate() != original_selection or online.clocks != original_clocks:
        raise AssertionError("History probe modified the rollout's selection state or clocks")
    result["probe_seconds"] = time.perf_counter() - started
    return result


class ProbedOnlinePolicy:
    """Transparent wrapper that preserves actual rollout actions and RNG use."""

    def __init__(self, online, *, task, decision_indices=(0, 2, 5, 10, 20), atol=1e-5):
        if online.method != "FULL" or online.intervention != "none":
            raise ValueError("Wrap only the ordinary FULL baseline")
        self.online = online
        self.task = task
        self.decision_indices = frozenset(decision_indices)
        self.atol = atol
        self.probes = []

    def __getattr__(self, name):
        return getattr(self.online, name)

    @property
    def update_seconds(self):
        return self.online.update_seconds

    @update_seconds.setter
    def update_seconds(self, value):
        # run_episode updates this directly for temporally skipped frames.
        self.online.update_seconds = value

    def reset(self, reset, seed):
        self.probes = []
        return self.online.reset(reset, seed)

    def observe(self, *args, **kwargs):
        return self.online.observe(*args, **kwargs)

    def decide(self):
        index = self.online.decision_index
        state = self.online.generator.get_state().clone() if index in self.decision_indices else None
        actions, timing = self.online.decide()
        if state is not None:
            try:
                result = probe_same_current(
                    self.online, task=self.task, generator_state=state,
                    executed_actions=actions, decision_index=index, atol=self.atol,
                )
                result["status"] = "passed" if result["full_replay_passed"] else "replay_mismatch"
            except Exception as exc:
                # Probe failure must not turn an otherwise normal rollout into
                # a policy failure or prevent executing the actual decision.
                result = {"decision_index": index, "status": "error",
                          "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()}
            self.probes.append(result)
            if "probe_seconds" in result:
                timing["diagnostic_probe_seconds"] = result["probe_seconds"]
        return actions, timing
