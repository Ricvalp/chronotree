"""CPU checks protecting the recurring real-state diagnostic protocol."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stmem.data.canonical import Frame
from stmem.inspection.offline import Inspection, metrics
from stmem.inspection.offline_variants import definitions, edit_frame
from stmem.training.checkpoints import file_sha256
from stmem.visualization.kinematics import panda_tcp_positions


def test_recurring_offline_module_has_no_dependency_on_ignored_experiment_code():
    """The installed package must work without copying historical run scripts."""
    directory = Path(__file__).resolve().parents[1] / "src/stmem/inspection"
    forbidden_modules = {"run_offline", "probe_common", "ablation_invariance", "validate_offline"}
    for filename in ("offline.py", "offline_variants.py", "offline_validation.py"):
        tree = ast.parse((directory / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in forbidden_modules
            if isinstance(node, ast.Import):
                assert not any(alias.name in forbidden_modules for alias in node.names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"spec_from_file_location", "run_path"}


def frame(source="execution", available=True):
    return Frame(
        points=torch.zeros(4, 6),
        valid=torch.ones(4, dtype=torch.bool),
        proprio=torch.arange(15, dtype=torch.float32) * 0.01,
        previous_action=torch.zeros(8),
        action_available=available,
        time=64.0,
        source=source,
        event_id=f"{source}:64",
        source_index=64,
    )


@pytest.mark.parametrize("definition", definitions(full_joint_sweep=False))
def test_command_probe_preserves_observations_and_source_identity(definition):
    original = frame()
    changed, _ = edit_frame(original, definition)
    for key in ("points", "valid", "proprio", "center_indices", "visual_features"):
        assert getattr(changed, key) is getattr(original, key)
    for key in ("time", "source", "event_id", "source_index"):
        assert getattr(changed, key) == getattr(original, key)
    assert torch.equal(original.previous_action, torch.zeros(8))
    if definition["kind"] == "offset":
        assert changed.previous_action[0].item() == pytest.approx(definition["delta"])
        assert torch.equal(changed.previous_action[1:], original.previous_action[1:])


@pytest.mark.parametrize("original", [frame("conditioning", available=False), frame(available=False)])
def test_command_probe_never_invents_unavailable_demonstrator_commands(original):
    for definition in definitions(full_joint_sweep=False):
        changed, info = edit_frame(original, definition)
        assert changed is original
        assert not info["changed"]


def test_native_action_metrics_use_post_action_alignment_and_zero_order_hold():
    actions = torch.zeros(40, 8)
    actions[:, 0] = torch.arange(40) * 0.01
    actions[:, 3] = -1.6
    actions[:, 5] = 1.5
    actions[:, 7] = 1
    episode = SimpleNamespace(
        actions=actions,
        action_valid=torch.ones_like(actions, dtype=torch.bool),
        alignment="post_action_v1",
    )
    target = actions[3:35:2].clone()
    decision = SimpleNamespace(
        index=2, target_chunk=target, target_valid=torch.ones_like(target, dtype=torch.bool)
    )
    physical = target.numpy()
    xyz = panda_tcp_positions(physical[:, :7], task="VideoUnmask")
    row = metrics((physical, physical, xyz), (physical, physical, xyz), decision, episode, 2)
    native_xyz = panda_tcp_positions(actions[3:7, :7].numpy(), task="VideoUnmask")
    expected = np.linalg.norm(np.repeat(xyz, 2, axis=0)[:4] - native_xyz, axis=-1).mean() * 1000
    assert row["tcp_error_mm_first4_targets"] == 0
    assert row["native4_tcp_error_mm"] == pytest.approx(expected)
    assert expected > 0  # Matching retained targets is not matching every native expert command.
    assert row["native4_tcp_change_mm"] == 0
    np.testing.assert_array_equal(row["native_target_actions"], actions[3:35].numpy())


def test_resuming_diagnostics_requires_matching_manifest_and_checkpoint(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable synthetic fixture")
    run = Inspection(
        {
            "output": str(tmp_path),
            "labels": ["model"],
            "checkpoints": {
                "model": {
                    "path": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                    "step": 1,
                }
            },
        },
        "inspection-a",
    )
    run.validate_checkpoint("model")
    report = tmp_path / "report.json"
    report.write_text(json.dumps(run.record_identity({"status": "complete"})))
    assert run.reuse(report, True)
    with pytest.raises(FileExistsError, match="--resume"):
        run.reuse(report, False)
    report.write_text(json.dumps({"status": "complete", "inspection_manifest_sha256": "different"}))
    with pytest.raises(ValueError, match="different inspection inputs"):
        run.reuse(report, True)
    checkpoint.write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="Checkpoint changed"):
        run.validate_checkpoint("model")
