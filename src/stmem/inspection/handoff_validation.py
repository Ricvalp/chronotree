"""Recompute handoff invariants from saved traces, independently of simulation."""

from __future__ import annotations

import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from stmem.inspection.handoff_helpers import source_for
from stmem.training.checkpoints import atomic_json, file_sha256

SCENES = (("VideoUnmask", 0), ("VideoUnmask", 1), ("PickXtimes", 0))
PREFIXES = (0, 4, 16, 32, 64)
LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], np.float32)
HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], np.float32)


def physical(action):
    action = np.asarray(action, np.float32).copy()
    assert action.shape == (8,) and np.isfinite(action).all()
    action[:7] = np.clip(action[:7], LOW, HIGH)
    action[7] = 1 if action[7] >= 0 else -1
    return action


def near(a, b, tolerance=1e-4):
    if a is None or b is None:
        assert a is b, (a, b)
    elif isinstance(a, (str, bool)):
        assert a == b, (a, b)
    else:
        assert abs(float(a) - float(b)) <= tolerance, (a, b)


def validate_trial(path, source, cfg, *, manifest_digest=None):
    """Check cadence, exact expert commands, causal counts, saved metrics and MP4."""
    path = Path(path)
    data = json.loads(path.read_text())
    assert data["kind"] == "real_simulator_training_scene_expert_prefix_handoff_not_benchmark_score"
    assert data["split"] == "train"
    if manifest_digest is not None:
        assert data["inspection_manifest_sha256"] == manifest_digest
    assert data["expert_source_sha256"] == source["sha256"]
    assert data["status"] not in ("diagnostic_error", "expert_record_exhausted_without_terminal"), (
        path,
        data.get("error"),
    )
    trace = data["timeline"]
    prefix, control = data["prefix_commands"], data["expert_control"]
    stride = data["temporal_sampling"]
    action_stride, observation_stride, execute = (
        stride["action_stride"],
        stride["observation_stride"],
        stride["execute"],
    )
    replan = action_stride * execute
    assert stride["action_hold"] == "zero_order_hold"
    if data.get("absolute_time_noise", False):
        assert execute == cfg["model"]["execute"] and replan > 0 and prefix % observation_stride == 0
    else:
        assert replan == 4
    assert action_stride == cfg["model"].get("action_stride", 1)
    assert observation_stride == cfg["data"].get("observation_stride", 1)
    assert trace and [row["step"] for row in trace] == list(range(1, len(trace) + 1))
    assert not any(row["terminal"] for row in trace[:-1])
    if data["status"] == "stopped_at_declared_diagnostic_cap":
        assert len(trace) == data["cap_total_commands"] == 320 and not trace[-1]["terminal"]
    else:
        assert trace[-1]["terminal"] and trace[-1]["status"] == data["status"]
    reset = np.asarray(data["reset_diagnostics"]["observed_proprio"], np.float32)
    reset_error = float(np.abs(reset[:7] - source["reset_proprio"][:7]).max())
    near(reset_error, data["reset_joint_max_difference_rad"])
    assert reset_error < 1e-4
    decisions = {row["step"]: row for row in data["decisions"]}
    assert len(decisions) == len(data["decisions"])
    assert list(decisions) == ([] if control else list(range(prefix, len(trace), replan)))
    if not control and decisions:
        handoff = data["handoff"]
        assert handoff["step"] == handoff["current_source_index"] == prefix
        assert handoff["execution_observations"] == prefix // observation_stride + 1
        expected_factor = (
            (
                cfg["model"]["visual_tokens"] // cfg["model"]["block_size"]
                + int(
                    cfg["model"].get("proprioception_mode", "all") == "all"
                    or cfg["model"].get("condition_on_previous_commands", True)
                )
            )
            if cfg["model"].get("history_mode", "full") != "none"
            else 0
        )
        assert data["leaves_per_history_frame"] == expected_factor
        assert handoff["history_leaf_counts"] == {
            "conditioning": expected_factor * data["conditioning_frames"],
            "execution": expected_factor * (prefix // observation_stride),
        }
        expected = source["reset_proprio"] if prefix == 0 else source["step_proprio"][prefix - 1]
        assert np.abs(np.asarray(handoff["diagnostics"]["observed_proprio"])[:7] - expected[:7]).max() < 1e-4
    expert_checks = {row["step"]: row for row in data["prefix_replay_checks"]}
    counted_expert, policy_commands, expert_commands = 0, 0, 0
    for row in trace:
        step = row["step"]
        is_expert = control or step <= prefix
        assert row["phase"] == ("expert" if is_expert else "policy")
        tcp, target = np.asarray(row["tcp_xyz"], np.float32), np.asarray(row["target_xyz"], np.float32)
        near(np.linalg.norm(tcp[:2] - target[:2]) * 1000, row["tcp_target_xy_distance_mm"])
        near(np.linalg.norm(tcp - target) * 1000, row["tcp_target_distance_mm"])
        near(target[2], row["target_height_m"])
        if row["observation_available"]:
            assert row["source_index"] == step
        else:
            assert row["terminal"]
        if not row["terminal"]:
            assert row["policy_observation_ingested"] == (step % observation_stride == 0)
            assert row["policy_current_source_index"] == (step // observation_stride) * observation_stride
            assert row["policy_execution_observations"] == step // observation_stride + 1
        if is_expert:
            assert np.array_equal(
                np.asarray(row["command"], np.float32), physical(source["commands"][step - 1])
            )
            expert_commands += 1
            if not row["terminal"]:
                error = np.abs(
                    np.asarray(row["observed_proprio"], np.float32)[:7] - source["step_proprio"][step - 1, :7]
                ).max()
                assert error < 1e-4
                near(error, expert_checks[step]["joint_max_abs_difference_rad"])
                counted_expert += 1
        else:
            decision_step = ((step - 1 - prefix) // replan) * replan + prefix
            decision = decisions[decision_step]
            offset = (step - decision_step - 1) // action_stride
            chunk = np.asarray(decision["chunk_raw"], np.float32)
            assert chunk.shape == (data["checkpoint_identity"]["horizon"], 8) and np.isfinite(chunk).all()
            assert np.array_equal(np.asarray(row["command"], np.float32), physical(chunk[offset]))
            assert decision["timing"]["decision_index"] == decision_step // replan
            if data.get("method") in ("TREE", "FLAT"):
                assert decision["timing"]["method"] == data["method"]
                assert decision["timing"]["history_tokens"] <= data["budget"]
                assert sum(decision["timing"]["represented_leaf_counts"]) == decision["timing"]["archive_fine_leaves"]
            if data.get("absolute_time_noise", False):
                from stmem.inspection.handoff import seed_for_primitive_time
                # CPU and CUDA generators have distinct state formats; saved
                # seeds are validated separately by the compact wrapper.
                assert decision.get("flow_seed") == seed_for_primitive_time(data["task"], data["episode_idx"], decision_step)
            assert decision["timing"]["archive_fine_leaves"] == data["leaves_per_history_frame"] * (
                data["conditioning_frames"] + decision_step // observation_stride
            )
            policy_commands += 1
        if step <= len(source["step_proprio"]) and not row["terminal"]:
            q, reference = np.asarray(row["observed_proprio"], np.float32), source["step_proprio"][step - 1]
            near(np.sqrt(np.mean((q[:7] - reference[:7]) ** 2)), row["same_time_expert_joint_rmse_rad"])
            near(np.linalg.norm(q[7:10] - reference[7:10]) * 1000, row["same_time_expert_tcp_distance_mm"])
    assert counted_expert == len(expert_checks)
    selected = [row for row in trace if row["phase"] == "policy" and row["observation_available"]]
    if not selected:
        selected = [row for row in trace if row["observation_available"]]
    first_close = next((row for row in selected if row["command"][7] < 0), None)
    threshold = 0.15 if data["task"] == "VideoUnmask" else 0.05
    summary = {
        "policy_commands": sum(row["phase"] == "policy" for row in trace),
        "expert_commands": sum(row["phase"] == "expert" for row in trace),
        "first_target_grasp_step": next((row["step"] for row in selected if row["target_grasped"]), None),
        "any_target_grasp": any(row["target_grasped"] for row in selected),
        "first_target_above_threshold_step": next(
            (row["step"] for row in selected if row["target_height_m"] > threshold), None
        ),
        "target_lift_threshold_m": threshold,
        "max_target_height_m": max(row["target_height_m"] for row in selected),
        "min_tcp_target_xy_distance_mm": min(row["tcp_target_xy_distance_mm"] for row in selected),
        "first_close_step": first_close["step"] if first_close else None,
        "first_close_xy_distance_mm": first_close["tcp_target_xy_distance_mm"] if first_close else None,
        "final_tcp_target_xy_distance_mm": selected[-1]["tcp_target_xy_distance_mm"],
    }
    assert set(summary) == set(data["summary"])
    for key, value in summary.items():
        near(value, data["summary"][key])
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path.parent / "rollout.mp4"))
    expected_frames = data["video_conditioning_frames"] + len(trace) + 1
    assert frames == expected_frames and abs(seconds - frames / 20) < 0.06
    return data, {
        "policy_commands": policy_commands,
        "expert_commands": expert_commands,
        "video_frames": frames,
        "video_seconds": seconds,
    }


def validate_suite(manifest, *, partial=False):
    root = Path(manifest["output"]) / "handoff"
    labels = tuple(manifest["labels"])
    sources = {scene: source_for(*scene, manifest["assets"]) for scene in SCENES}
    for (task, _episode), source in sources.items():
        asset = "prepared" if task == "VideoUnmask" else "pick_prefix"
        assert source["sha256"] == manifest["asset_hashes"][asset], asset
    paths = sorted(root.glob("*/*/*/*/result.json"))
    rows, video_rows = [], []
    cases, parity = set(), set()
    states, rng, identities = {}, {}, {}
    policy_commands = expert_commands = paired_states = paired_rng = 0
    for path in paths:
        raw = json.loads(path.read_text())
        variant, task, episode, control = (
            raw["variant"],
            raw["task"],
            raw["episode_idx"],
            raw["expert_control"],
        )
        cfg = json.loads((root / f"config_{labels[0] if control else variant}.json").read_text())
        data, counts = validate_trial(
            path,
            sources[task, episode],
            cfg,
            manifest_digest=file_sha256(Path(manifest["output"]) / "inspection_manifest.json"),
        )
        ident = data["checkpoint_identity"]
        if variant not in identities:
            assert file_sha256(ident["path"]) == ident["sha256"]
            identities[variant] = ident
        assert ident == identities[variant]
        expected_identity = manifest["checkpoints"][labels[0] if control else variant]
        assert ident["sha256"] == expected_identity["sha256"] and ident["step"] == expected_identity["step"]
        prefix = data["prefix_commands"]
        if not control:
            case = (variant, task, episode, prefix)
            assert case not in cases
            cases.add(case)
        for kind, index, diagnostic in [("reset", 0, data["reset_diagnostics"])] + (
            [("handoff", prefix, data["handoff"]["diagnostics"])] if "handoff" in data else []
        ):
            key = (task, episode, kind, index)
            measured = {
                field: np.asarray(diagnostic[field], np.float32)
                for field in ("observed_proprio", "target_xyz")
            }
            if key in states:
                for field in measured:
                    assert np.array_equal(measured[field], states[key][field]), (path, key, field)
                paired_states += 1
            else:
                states[key] = measured
        for decision in data["decisions"]:
            key = (variant, decision["step"])
            if key in rng:
                assert rng[key] == decision["rng_state_before_sha256"]
                paired_rng += 1
            else:
                rng[key] = decision["rng_state_before_sha256"]
        if "transport_parity" in data:
            result = data["transport_parity"]
            assert all(
                result[key]
                for key in ("context_bit_identical", "bias_bit_identical", "actions_bit_identical")
            )
            parity.add(variant)
        policy_commands += counts["policy_commands"]
        expert_commands += counts["expert_commands"]
        video_rows.append(
            {
                "path": str((path.parent / "rollout.mp4").relative_to(root)),
                "frames": counts["video_frames"],
                "seconds": counts["video_seconds"],
            }
        )
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "variant": variant,
                "task": task,
                "episode": episode,
                "prefix": prefix,
                "status": data["status"],
                "expert_control": control,
                "commands": len(data["timeline"]),
                "summary": data["summary"],
            }
        )
    expected = {
        (variant, task, episode, prefix)
        for variant in labels
        for task, episode in SCENES
        for prefix in PREFIXES
    }
    missing = sorted(expected - cases)
    assert cases <= expected
    controls = [row for row in rows if row["expert_control"]]
    if not partial:
        assert not missing, missing
        assert {(row["task"], row["episode"]) for row in controls} == {("VideoUnmask", 0), ("VideoUnmask", 1)}
        assert len(controls) == 2 and all(row["status"] == "success" for row in controls)
        assert parity == set(labels), parity
    report = {
        "inspection_manifest_sha256": file_sha256(Path(manifest["output"]) / "inspection_manifest.json"),
        "status": "passed_partial" if partial else "passed",
        "policy_trials": len(cases),
        "expert_controls": len(controls),
        "missing_cases": missing,
        "checked_policy_commands": policy_commands,
        "checked_expert_commands": expert_commands,
        "paired_reset_handoff_state_checks": paired_states,
        "paired_rng_state_checks": paired_rng,
        "transport_parity_variants": sorted(parity),
        "trials": rows,
        "videos": video_rows,
        "validator_sha256": file_sha256(Path(__file__)),
        "limitations": [
            "Fixed training scenes, one noise seed and correlated prefixes are not benchmark success rates.",
            "Training exposure, features, proprioception and action cadence can differ across models.",
            "Noise alignment holds across prefixes within each model; different native horizons consume different draws.",
            "This audit recomputes saved traces without a second simulation.",
        ],
    }
    atomic_json(report, root / ("validation_partial.json" if partial else "validation.json"))
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "policy_trials", "expert_controls", "checked_policy_commands")
            }
        ),
        flush=True,
    )
    return report
