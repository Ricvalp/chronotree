"""Independent CPU-only checks of saved predictions; no policy is instantiated."""

import json
import h5py
import numpy as np
from stmem.evaluation.robomme_adapter import physical_action
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.visualization.kinematics import panda_tcp_positions


def write_summary_table(summary, root):
    """Comparable native-time windows, keeping reset and later states distinct."""
    import csv

    rows = []
    for label, model in summary.items():
        groups = {(g["kind"], g["variant"], g["state_subset"]): g for g in model["groups"]}
        row = {
            "label": label,
            "step": model["step"],
            "horizon": model["horizon"],
            "action_stride": model["action_stride"],
            "native4_expert_tcp_error_mm": groups["visual", "baseline", "all_states"]["native4_tcp_error_mm"],
            "delete_current_visual_native4_change_mm": groups[
                "visual", "remove_current_visual_tokens", "after_reset"
            ]["native4_tcp_change_mm"],
            "delete_history_visual_native4_change_mm": groups[
                "visual", "remove_history_visual_tokens", "after_reset"
            ]["native4_tcp_change_mm"],
            "delete_all_history_native4_change_mm": groups["visual", "remove_all_history", "after_reset"][
                "native4_tcp_change_mm"
            ],
        }
        rows.append(row)
    atomic_json(rows, root / "summary_table.json")
    with (root / "summary_table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def validate(run):
    """Recompute FK, action alignment, holds and invariances from saved evidence."""
    ROOT = run.root
    checks = []
    DATA = run.asset("prepared")
    source_episodes = {}
    with h5py.File(DATA, "r") as handle:
        for key, group in handle.items():
            execution = ~np.asarray(group["source"])
            source_episodes[key] = (
                np.asarray(group["actions"])[execution],
                np.asarray(group["action_valid"])[execution],
            )

    def check(name, value):
        assert value, name
        checks.append(name)

    summary = {}
    row_total = 0
    expected_invariants = 0
    for label in run.labels:
        step = run.manifest["checkpoints"][label]["step"]
        report = json.loads((ROOT / label / "sensitivity.json").read_text())
        parity = json.loads((ROOT / label / "parity.json").read_text())
        h = report["horizon"]
        stride = report["action_stride"]
        sha = file_sha256(run.checkpoint(label))
        check(
            f"{label}/manifest",
            report["inspection_manifest_sha256"]
            == parity["inspection_manifest_sha256"]
            == run.manifest_sha256,
        )
        check(f"{label}/complete", report["status"] == "complete" and parity["status"] == "passed")
        check(
            f"{label}/checkpoint",
            report["checkpoint_sha256"] == parity["checkpoint_sha256"] == sha
            and report["checkpoint_step"] == parity["checkpoint_step"] == step,
        )
        check(
            f"{label}/states",
            [(s["episode"], s["index"]) for s in report["states"]]
            == [(f"episode_{ep}", i) for ep in (0, 1) for i in (0, 32, 64, 96, 128)],
        )
        check(
            f"{label}/alignment",
            report["alignment"]["verified"] and report["alignment"]["convention"] == "post_action_v1",
        )
        has_features = report["feature_signature"] is not None
        check(f"{label}/count", len(report["rows"]) == (1110 if has_features else 1070))
        check(
            f"{label}/protocol",
            report["decisions"] == [0, 32, 64, 96, 128] and report["noise_seeds"] == [1701, 1702, 1703, 1704],
        )
        check(f"{label}/data", report["prepared_sha256"] == file_sha256(DATA))
        expected_invariants += (
            (7 if not report["history_enabled"] else 3) if report["proprioception_mode"] == "none" else 0
        )
        baseline = {
            (r["kind"], r["episode"], r["index"], r["noise_seed"]): r
            for r in report["rows"]
            if r["variant"] == "baseline"
        }
        definitions = {d["name"]: d for d in report["command_definitions"]}
        for number, row in enumerate(report["rows"]):
            tag = f"{label}/row{number}"
            raw = np.array(row["raw_actions"])
            physical = np.array(row["physical_actions"])
            xyz = np.array(row["tcp_positions_m"])
            check(tag + "/shape", raw.shape == physical.shape == (h, 8) and xyz.shape == (h, 3))
            check(
                tag + "/finite",
                np.isfinite(raw).all() and np.isfinite(physical).all() and np.isfinite(xyz).all(),
            )
            correct_physical = np.stack([physical_action(a, "VideoUnmask") for a in raw])
            check(tag + "/clipping", np.array_equal(correct_physical, physical))
            check(
                tag + "/kinematics",
                np.max(np.abs(panda_tcp_positions(physical[:, :7], task="VideoUnmask") - xyz)) < 1e-10,
            )
            kind = "visual" if row["kind"] == "precision" else row["kind"]
            base = baseline[kind, row["episode"], row["index"], row["noise_seed"]]
            base_xyz = np.array(base["tcp_positions_m"])
            base_raw = np.array(base["raw_actions"])
            target = np.array(row["target_actions"])
            valid = np.array(row["target_valid"])
            source_actions, source_valid = source_episodes[row["episode"]]
            expected = np.zeros((h, 8), dtype=source_actions.dtype)
            expected_valid = np.zeros((h, 8), dtype=bool)
            continuity = True
            for position in range(h):
                start = row["index"] + 1 + position * stride
                if start + stride > len(source_actions):
                    break
                expected[position] = source_actions[start]
                interval_valid = source_valid[start : start + stride].all(0)
                continuity = continuity and interval_valid[:7].all()
                expected_valid[position] = interval_valid & continuity
            check(
                tag + "/independent_post_action_target",
                np.array_equal(target, expected) and np.array_equal(valid, expected_valid),
            )
            target_xyz = panda_tcp_positions(target[:, :7], task="VideoUnmask")
            for n in sorted({4, 16, h}):
                keep = valid[:n, :7].all(-1)
                error = np.linalg.norm(xyz[:n][keep] - target_xyz[:n][keep], axis=-1).mean() * 1000
                change = np.linalg.norm(xyz[:n] - base_xyz[:n], axis=-1).mean() * 1000
                check(tag + f"/target{n}", abs(error - row[f"tcp_error_mm_first{n}_targets"]) < 1e-8)
                check(tag + f"/targetchange{n}", abs(change - row[f"tcp_change_mm_first{n}_targets"]) < 1e-8)
            expanded = np.repeat(xyz, stride, axis=0)
            native_target = np.array(row["native_target_actions"])
            native_valid = np.array(row["native_target_valid"])
            native_start = row["index"] + 1
            check(
                tag + "/independent_native_post_action_targets",
                np.array_equal(native_target, source_actions[native_start : native_start + h * stride])
                and np.array_equal(native_valid, source_valid[native_start : native_start + h * stride]),
            )
            native_xyz = panda_tcp_positions(native_target[:, :7], task="VideoUnmask")
            for n in (4, 16):
                keep = native_valid[:n, :7].all(-1)
                error = np.linalg.norm(expanded[:n][keep] - native_xyz[:n][keep], axis=-1).mean() * 1000
                change = (
                    np.linalg.norm((expanded - np.repeat(base_xyz, stride, axis=0))[:n], axis=-1).mean()
                    * 1000
                )
                check(tag + f"/native{n}", abs(error - row[f"native{n}_tcp_error_mm"]) < 1e-8)
                check(tag + f"/nativechange{n}", abs(change - row[f"native{n}_tcp_change_mm"]) < 1e-8)
            raw_change = float(np.abs(raw.astype(np.float32) - base_raw.astype(np.float32)).max())
            check(tag + "/raw_change", raw_change == row["max_abs_raw_action_change"])
            if row["variant"] == "baseline_repeat":
                check(tag + "/repeat", np.array_equal(raw, base_raw))
            if row["kind"] == "commands":
                definition = definitions.get(row["variant"], {})
                disabled = (
                    row["index"] == 0
                    or row["variant"] == "recomputed_event_control"
                    or not report["condition_on_previous_commands"]
                )
                disabled |= not report["condition_on_previous_joint_commands"] and definition.get("kind") in (
                    "offset",
                    "hold_qpos",
                )
                if disabled:
                    check(tag + "/disabled_command", np.array_equal(raw, base_raw))
            if row["kind"] == "visual" and (
                not report["history_enabled"]
                and row["variant"]
                in [
                    "remove_history_visual_tokens",
                    "remove_all_history",
                    "erase_execution",
                    "erase_conditioning",
                    "permute_execution",
                ]
                or row["index"] == 0
                and row["variant"] in ["erase_execution", "permute_execution"]
            ):
                check(tag + "/history_control", np.array_equal(raw, base_raw))
        row_total += len(report["rows"])
        check(f"{label}/parity_states", len(parity["real_states"]) == 2)
        for row in parity["real_states"]:
            a = np.array(row["raw_actions"]["batch_fp32"])
            b = np.array(row["raw_actions"]["serial_fp32"])
            xyz_a = panda_tcp_positions(
                np.stack([physical_action(x, "VideoUnmask") for x in a])[:, :7], task="VideoUnmask"
            )
            xyz_b = panda_tcp_positions(
                np.stack([physical_action(x, "VideoUnmask") for x in b])[:, :7], task="VideoUnmask"
            )
            delta = float(np.linalg.norm(xyz_a - xyz_b, axis=-1).mean() * 1000)
            check(f"{label}/parity{row['index']}", delta < 0.01)
        gradient = parity["gradient"]
        check(f"{label}/grad_finite", all(g["finite"] for g in gradient["groups"].values()))
        check(
            f"{label}/grad_visual",
            gradient["groups"]["encoder.visual.point_mlp"]["gradient_squared_norm"] > 0,
        )
        if has_features:
            check(
                f"{label}/grad_projection",
                gradient["groups"]["encoder.visual.feature_projection"]["gradient_squared_norm"] > 0,
            )
        check(
            f"{label}/no_update",
            gradient["optimizer_updates"] == 0 and gradient["weights_bit_identical_after_backward"],
        )
        check(f"{label}/early_count", len(parity["early_prefix"]) == 12)
        for number, row in enumerate(parity["early_prefix"]):
            check(f"{label}/early{number}", row["full_chunk_tcp_difference_mm"] < 0.01)
            check(
                f"{label}/early_observations{number}",
                row["execution_observations"] == row["index"] // report["observation_stride"] + 1,
            )
            if not report["history_enabled"]:
                check(
                    f"{label}/early_no_history{number}",
                    row["history_leaves"] == row["history_frames"] == row["conditioning_frames"] == 0,
                )
        summary[label] = dict(
            step=step,
            horizon=h,
            action_stride=stride,
            rows=len(report["rows"]),
            groups=report["summary"],
            real_state_parity=parity["real_states"],
            gradient=gradient,
            max_early_prefix_tcp_difference_mm=max(
                r["full_chunk_tcp_difference_mm"] for r in parity["early_prefix"]
            ),
        )

    invariants = json.loads((ROOT / "ablation_invariance.json").read_text())
    check(
        "omitted_input_invariants/status",
        invariants["status"] == "passed"
        and len(invariants["rows"]) == expected_invariants
        and invariants["inspection_manifest_sha256"] == run.manifest_sha256,
    )
    for index, row in enumerate(invariants["rows"]):
        check(
            f"omitted_input_invariants/{index}", row["context_bit_identical"] and row["actions_bit_identical"]
        )
        check(
            f"omitted_input_invariants/{index}/source",
            row["checkpoint_sha256"] == file_sha256(run.checkpoint(row["label"])),
        )
    result = dict(
        status="passed",
        checks=len(checks),
        prediction_rows=row_total,
        real_parity_states=2 * len(run.labels),
        backward_only_probes=len(run.labels),
        early_prefix_cases=12 * len(run.labels),
        omitted_input_invariant_cases=expected_invariants,
        per_check=checks,
    )
    atomic_json(run.record_identity(result), ROOT / "validation.json")
    atomic_json(summary, ROOT / "summary.json")
    write_summary_table(summary, ROOT)
    print(json.dumps({k: v for k, v in result.items() if k != "per_check"}))
