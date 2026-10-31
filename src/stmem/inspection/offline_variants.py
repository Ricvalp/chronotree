"""Isolate previous-command channels; keep observations and FULL leaf coverage.

Artificial conditioning interventions, not executable physical counterfactuals.
Only explicit FULL record lists are consumed; original archive parents are never
used for a modified read. No token is removed or reordered.
"""

from dataclasses import replace
import numpy as np
import torch
from stmem.evaluation.robomme_adapter import JOINT_LOW, JOINT_HIGH


def edit_frame(frame, definition):
    if not frame.action_available or frame.source != "execution":
        return frame, {"changed": False, "clipped": 0}
    kind = definition["kind"]
    action = frame.previous_action.clone()
    available = True
    clipped = 0
    if kind == "identity":
        return frame, {"changed": False, "clipped": 0}
    if kind == "offset":
        joint, delta = definition["joint"], definition["delta"]
        assert 0 <= joint < 7
        requested = float(action[joint]) + delta
        actual = float(np.clip(requested, JOINT_LOW[joint], JOINT_HIGH[joint]))
        clipped = int(abs(actual - requested) > 1e-7)
        action[joint] = actual
    elif kind == "mask":
        action.zero_()
        available = False
    elif kind == "hold_qpos":
        action[:7] = frame.proprio[:7]
    elif kind == "flip_gripper":
        action[7] = -action[7]
    else:
        raise ValueError(kind)
    changed = replace(frame, previous_action=action, action_available=available)
    for field in ["points", "valid", "proprio", "center_indices"]:
        assert getattr(changed, field) is getattr(frame, field)
    for field in ["time", "source", "event_id", "source_index", "center_sampling_signature"]:
        assert getattr(changed, field) == getattr(frame, field)
    return changed, {
        "changed": available != frame.action_available or not torch.equal(action, frame.previous_action),
        "clipped": clipped,
    }


def reencode_event_records(policy, records, original_banks, changed_banks, *, force=False):
    if policy.encoder.event is None:
        assert all(record.modality != "event" for record in records)
        return list(records)
    contents = {}
    block_size = policy.encoder.block_size
    for originals, frames in zip(original_banks, changed_banks):
        assert len(originals) == len(frames)
        for start in range(0, len(frames), block_size):
            block = frames[start : start + block_size]
            old = originals[start : start + block_size]
            if not force and all(a is b for a, b in zip(block, old)):
                continue
            proprio = torch.stack([f.proprio for f in block]).to(policy.device)
            actions = torch.stack([f.previous_action for f in block]).to(policy.device)
            available = torch.tensor(
                [f.action_available and f.source == "execution" for f in block], device=policy.device
            )
            values = policy.encoder.event(proprio, actions, available)
            for f, value in zip(block, values):
                contents[f.source, f.time] = value
    result = []
    for record in records:
        assert record.is_leaf and record.count == 1
        key = record.source, record.time_anchor
        if record.modality != "event" or key not in contents:
            result.append(record)
        else:
            content = contents[key].to(record.content_sum.device)
            if force:
                assert torch.equal(content, record.content_sum), "Unchanged event reencoding differs"
            result.append(replace(record, content_sum=content, max_content=content))
    assert [r.node_id for r in result] == [r.node_id for r in records]
    assert all(new is old for new, old in zip(result, records) if old.modality == "visual")
    return result


@torch.no_grad()
def iter_variants(policy, evidence, conditioning, execution_prefix, definitions):
    records = evidence.archive.leaves
    yield "baseline", evidence, records, {"current_changed": False, "history_changed": 0, "clipped_inputs": 0}
    recomputed = reencode_event_records(
        policy, records, [conditioning, execution_prefix], [conditioning, execution_prefix], force=True
    )
    yield (
        "recomputed_event_control",
        evidence,
        recomputed,
        {
            "current_changed": False,
            "history_changed": 0,
            "clipped_inputs": 0,
            "unchanged_event_reencoding": "bit_identical",
        },
    )
    for definition in definitions:
        scope = definition["scope"]
        current = evidence.current
        info = {"current_changed": False, "history_changed": 0, "clipped_inputs": 0}
        if scope in ["current", "all"]:
            current, meta = edit_frame(current, definition)
            info["current_changed"] = bool(meta["changed"])
            info["clipped_inputs"] += meta["clipped"]
        changed_history = []
        for i, frame in enumerate(execution_prefix):
            modify = scope in ["history", "all"] or (
                scope == "recent_history" and i >= len(execution_prefix) - 8
            )
            if modify:
                changed, meta = edit_frame(frame, definition)
                info["history_changed"] += int(meta["changed"])
                info["clipped_inputs"] += meta["clipped"]
            else:
                changed = frame
            changed_history.append(changed)
        altered_records = reencode_event_records(
            policy, records, [conditioning, execution_prefix], [conditioning, changed_history]
        )
        altered = replace(evidence, current=current)
        assert altered.current_records is evidence.current_records
        assert altered.text is evidence.text and altered.clocks is evidence.clocks
        info["input_previous_action"] = current.previous_action.tolist()
        info["input_action_available"] = current.action_available
        info["available_current_command"] = evidence.current.action_available
        info["available_history_commands"] = sum(f.action_available for f in execution_prefix)
        yield definition["name"], altered, altered_records, info


def definitions(full_joint_sweep=True):
    result = []
    joints = range(7) if full_joint_sweep else [0]
    magnitudes = [0.01, 0.05] if full_joint_sweep else [0.05]
    for scope in ["current", "history", "all"]:
        for joint in joints:
            for magnitude in magnitudes:
                for sign in [-1, 1]:
                    result.append(
                        {
                            "name": f"{scope}_j{joint + 1}_{'plus' if sign > 0 else 'minus'}_{magnitude:g}rad",
                            "scope": scope,
                            "kind": "offset",
                            "joint": joint,
                            "delta": sign * magnitude,
                        }
                    )
        for kind in ["mask", "hold_qpos", "flip_gripper"]:
            result.append({"name": f"{scope}_{kind}", "scope": scope, "kind": kind})
    if full_joint_sweep:
        for joint in range(7):
            for sign in [-1, 1]:
                result.append(
                    {
                        "name": f"recent_history_j{joint + 1}_{'plus' if sign > 0 else 'minus'}_0.05rad",
                        "scope": "recent_history",
                        "kind": "offset",
                        "joint": joint,
                        "delta": 0.05 * sign,
                    }
                )
    return result
