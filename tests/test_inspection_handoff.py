"""CPU-only tests of reusable handoff identity, timing and diagnostic validation."""

from copy import deepcopy
import json

import numpy as np
import pytest

from stmem.inspection import handoff
from stmem.inspection.handoff_helpers import source_for, summarize
from stmem.inspection.handoff_validation import physical, validate_trial


def test_completed_trial_reuse_requires_same_manifest_checkpoint_and_protocol(monkeypatch):
    monkeypatch.setattr(handoff, "MANIFEST_DIGEST", "manifest")
    monkeypatch.setattr(handoff, "runner_sha256", lambda: "runner")
    identity = {"sha256": "weight", "step": 50000}
    record = dict(
        task="PickXtimes",
        episode_idx=0,
        prefix_commands=16,
        expert_control=False,
        cap_total_commands=320,
        inspection_manifest_sha256="manifest",
        runner_sha256="runner",
        checkpoint_identity=identity,
        status="stopped_at_declared_diagnostic_cap",
    )
    kwargs = dict(task="PickXtimes", episode=0, prefix=16, control=False, cap=320)
    assert handoff.valid_result(record, identity, **kwargs)
    for key, value in [
        ("inspection_manifest_sha256", "another"),
        ("runner_sha256", "changed"),
        ("checkpoint_identity", {"sha256": "new"}),
        ("cap_total_commands", 160),
    ]:
        with pytest.raises(ValueError):
            handoff.valid_result({**record, key: value}, identity, **kwargs)
    assert not handoff.valid_result({**record, "status": "diagnostic_error"}, identity, **kwargs)


def _fixture(path):
    raw = np.zeros((16, 8), np.float32)
    raw[:, 7] = 1
    source = dict(
        sha256="source",
        commands=np.zeros((2, 8), np.float32),
        reset_proprio=np.zeros(18, np.float32),
        step_proprio=np.zeros((2, 18), np.float32),
    )
    diagnostic = dict(
        target_xyz=[0.0, 0.0, 0.0],
        tcp_xyz=[0.0, 0.0, 0.0],
        observed_proprio=[0.0] * 18,
        target_height_m=0.0,
        tcp_target_xy_distance_mm=0.0,
        tcp_target_distance_mm=0.0,
        target_grasped=False,
    )
    trace = []
    for step in (1, 2):
        trace.append(
            dict(
                step=step,
                phase="policy",
                command=physical(raw[0]).tolist(),
                status="success" if step == 2 else "ongoing",
                terminal=step == 2,
                observation_available=True,
                source_index=step,
                policy_observation_ingested=True,
                policy_current_source_index=step,
                policy_execution_observations=step + 1,
                same_time_expert_joint_rmse_rad=0.0,
                same_time_expert_tcp_distance_mm=0.0,
                **diagnostic,
            )
        )
    cfg = {"model": {"visual_tokens": 256, "block_size": 8}, "data": {}}
    data = dict(
        kind="real_simulator_training_scene_expert_prefix_handoff_not_benchmark_score",
        split="train",
        task="PickXtimes",
        episode_idx=0,
        variant="model",
        prefix_commands=0,
        expert_control=False,
        expert_source_sha256="source",
        status="success",
        temporal_sampling=dict(
            action_stride=1, observation_stride=1, execute=4, action_hold="zero_order_hold"
        ),
        checkpoint_identity=dict(horizon=16),
        cap_total_commands=320,
        reset_diagnostics=diagnostic,
        reset_joint_max_difference_rad=0.0,
        conditioning_frames=0,
        handoff=dict(
            step=0,
            current_source_index=0,
            execution_observations=1,
            history_leaf_counts=dict(conditioning=0, execution=0),
            diagnostics=diagnostic,
        ),
        leaves_per_history_frame=33,
        timeline=trace,
        decisions=[
            dict(step=0, chunk_raw=raw.tolist(), timing=dict(decision_index=0, archive_fine_leaves=0))
        ],
        prefix_replay_checks=[],
        video_conditioning_frames=0,
    )
    data["summary"] = summarize(data)
    path.write_text(json.dumps(data))
    return data, source, cfg


def test_saved_trace_audit_detects_wrong_commands_and_observation_leak(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    data, source, cfg = _fixture(path)
    monkeypatch.setattr("imageio_ffmpeg.count_frames_and_secs", lambda _: (3, 0.15))
    checked, counts = validate_trial(path, source, cfg)
    assert checked["status"] == "success" and counts["policy_commands"] == 2
    bad = deepcopy(data)
    bad["timeline"][0]["command"][0] = 0.2
    path.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        validate_trial(path, source, cfg)
    bad = deepcopy(data)
    bad["timeline"][0]["policy_current_source_index"] = 2
    path.write_text(json.dumps(bad))
    with pytest.raises(AssertionError):
        validate_trial(path, source, cfg)
    path.write_text(json.dumps(data))
    monkeypatch.setattr("imageio_ffmpeg.count_frames_and_secs", lambda _: (2, 0.1))
    with pytest.raises(AssertionError):
        validate_trial(path, source, cfg)


def test_grasp_flag_and_height_are_separate_diagnostic_outcomes():
    rows = []
    for step, grasp, height in [(1, False, 0.02), (2, True, 0.023), (3, False, 0.02)]:
        rows.append(
            dict(
                step=step,
                phase="policy",
                observation_available=True,
                command=[0] * 7 + [-1],
                target_grasped=grasp,
                target_height_m=height,
                tcp_target_xy_distance_mm=20.0,
            )
        )
    result = summarize(dict(task="PickXtimes", timeline=rows))
    assert result["any_target_grasp"]
    assert result["first_target_above_threshold_step"] is None
    assert result["first_close_xy_distance_mm"] == 20.0


def test_generated_expert_source_rejects_wrong_scene_and_nonfinite_commands(tmp_path):
    path = tmp_path / "prefix.json"
    record = dict(
        task="PickXtimes",
        split="train",
        episode_idx=0,
        commands=[[0.0] * 8] * 64,
        step_proprio=[[0.0] * 18] * 64,
        reset_proprio=[0.0] * 18,
    )
    path.write_text(json.dumps(record))
    result = source_for("PickXtimes", 0, {"pick_prefix": str(path)})
    assert result["commands"].shape == (64, 8)
    with pytest.raises(ValueError, match="another scene"):
        source_for("PickXtimes", 1, {"pick_prefix": str(path)})
    record["commands"][0][0] = float("nan")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Nonfinite"):
        source_for("PickXtimes", 0, {"pick_prefix": str(path)})


def test_ten_hz_trace_requires_held_commands_and_sampled_observations(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    data, source, cfg = _fixture(path)
    cfg["data"]["observation_stride"] = 2
    cfg["model"].update(action_stride=2, proprioception_mode="none", condition_on_previous_commands=False)
    data["temporal_sampling"].update(action_stride=2, observation_stride=2, execute=2)
    data["leaves_per_history_frame"] = 32
    raw = np.asarray(data["decisions"][0]["chunk_raw"], np.float32)
    raw[1, 0] = 0.3
    data["decisions"][0]["chunk_raw"] = raw.tolist()
    prototype = deepcopy(data["timeline"][0])
    data["timeline"] = []
    for step in range(1, 5):
        entry = dict(prototype)
        entry.update(
            step=step,
            source_index=step,
            command=physical(raw[(step - 1) // 2]).tolist(),
            status="success" if step == 4 else "ongoing",
            terminal=step == 4,
            policy_observation_ingested=step % 2 == 0,
            policy_current_source_index=(step // 2) * 2,
            policy_execution_observations=step // 2 + 1,
        )
        data["timeline"].append(entry)
    source["commands"] = np.zeros((4, 8), np.float32)
    source["step_proprio"] = np.zeros((4, 18), np.float32)
    data["summary"] = summarize(data)
    path.write_text(json.dumps(data))
    monkeypatch.setattr("imageio_ffmpeg.count_frames_and_secs", lambda _: (5, 0.25))
    _, counts = validate_trial(path, source, cfg)
    assert counts["policy_commands"] == 4
    data["timeline"][2]["command"] = physical(raw[0]).tolist()
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        validate_trial(path, source, cfg)
