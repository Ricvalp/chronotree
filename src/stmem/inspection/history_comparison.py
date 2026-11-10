"""Read-only CPU audit and comparison of matched teacher history diagnostics.

Only writes the requested audit files. Never modifies runs, checkpoints, policy
code, or source results. Incomplete records remain pending, not failures.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import tempfile
from urllib.parse import unquote


ERRORS = {"error", "infrastructure_error", "policy_error"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Audit:
    """Independent check accumulator; repeated comparisons cannot share state."""

    def __init__(self):
        self.checks = []

    def check(self, name, passed, **details):
        self.checks.append({"check": name, "passed": bool(passed), **details})


def load(root, audit):
    root = Path(root).resolve()
    manifest, cfg = read(root / "manifest.json"), read(root / "config.json")
    from stmem.inspection.history_use_report import _plan

    keys, _, _ = _plan(manifest)
    planned = set(keys)
    sealed_plan = (root / "identity.json").is_file()
    if sealed_plan:
        from stmem.inspection.history_resume import validate_plan

        try:
            # Use the saved request itself: validate recorded hashes/assets, not
            # equality to today's checkout. Cross-checkpoint source review is separate.
            validate_plan(root, manifest, cfg)
        except (OSError, ValueError, RuntimeError) as exc:
            audit.check(f"{root.name}:stored_identity", False, error=str(exc))
        else:
            audit.check(f"{root.name}:stored_identity", True)
    rows = {}
    for file in sorted(root.glob("results/*/episodes/*.json")):
        row = read(file)
        key = row["task"], row["episode_idx"], row["arm"]
        if key not in planned or key in rows or row["arm"] != file.parent.parent.name:
            raise ValueError(f"Unexpected, duplicate or inconsistently labeled result: {file}")
        audit.check(f"{root.name}/{file.name}:planned_unique", True)
        if sealed_plan or row.get("resume_integrity"):
            from stmem.inspection.history_resume import validate_result

            audit.check(f"{root.name}/{file.name}:sealed_plan_present", sealed_plan)
            try:
                validate_result(
                    file, dict(zip(("task", "episode_idx", "arm"), key)), manifest, cfg, video_root=root
                )
            except (OSError, ValueError, RuntimeError) as exc:
                audit.check(f"{root.name}/{file.name}:result_integrity:{key[2]}", False, error=str(exc))
            else:
                audit.check(f"{root.name}/{file.name}:result_integrity:{key[2]}", True)
        rows[key] = row
    audit.check(
        f"{root.name}:checkpoint_hash",
        sha(manifest["checkpoint"]["path"]) == manifest["checkpoint"]["sha256"],
    )
    for path, expected_hash in manifest.get("reference_results", {}).items():
        audit.check(f"{root.name}:reference_hash:{Path(path).name}", sha(path) == expected_hash)
        original = read(path)
        key = original["task"], original["episode_idx"], original["arm"]
        if original.get("resume_integrity"):
            from stmem.inspection.history_resume import validate_result

            try:
                validate_result(path, dict(zip(("task", "episode_idx", "arm"), key)), manifest, cfg)
            except (OSError, ValueError, RuntimeError) as exc:
                audit.check(f"{root.name}:reference_integrity:{Path(path).name}", False, error=str(exc))
            else:
                audit.check(f"{root.name}:reference_integrity:{Path(path).name}", True)
        copied = rows.get(key)
        audit.check(f"{root.name}:reference_present:{Path(path).name}", copied is not None)
        if copied:
            a, b = dict(original), dict(copied)
            a.pop("resume_integrity", None)
            b.pop("resume_integrity", None)
            original_video, copied_video = a.pop("video_path", None), b.pop("video_path", None)
            b.pop("reused_reference_result", None)
            audit.check(f"{root.name}:reference_identity:{Path(path).name}", a == b)
            audit.check(
                f"{root.name}:reference_video:{Path(path).name}",
                bool(original_video and copied_video) and sha(original_video) == sha(copied_video),
            )
    for path, expected_hash in manifest.get("reference_videos", {}).items():
        audit.check(f"{root.name}:reference_video_hash:{Path(path).name}", sha(path) == expected_hash)
    return {
        "root": root,
        "manifest": manifest,
        "config": cfg,
        "planned": planned,
        "rows": rows,
        "integrity": "sealed" if sealed_plan else "legacy_unsealed",
    }


def inputs(row):
    fp = row.get("task_diagnostics", {}).get("input_fingerprints") or {}
    return {
        "reset": fp.get("reset_sha256"),
        "current": (fp.get("current") or {}).get("sha256"),
        "conditioning": (fp.get("conditioning") or {}).get("sha256"),
        "goal": fp.get("goal_sha256"),
        "noise_seed": row.get("noise_seed"),
    }


def same_inputs(left, right):
    a, b = inputs(left), inputs(right)
    return {key: a[key] is not None and a[key] == b[key] for key in a}


def max_abs(a, b):
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return None
    return max((abs(x - y) for aa, bb in zip(a, b) for x, y in zip(aa, bb)), default=0.0)


def audit_run(run, audit):
    label, cfg = run["root"].name, run["config"]
    diagnostics = []
    for (task, idx, arm), row in run["rows"].items():
        case = f"{label}/{task}/{idx}/{arm}"
        audit.check(
            case + ":checkpoint", row.get("checkpoint_sha256") == run["manifest"]["checkpoint"]["sha256"]
        )
        audit.check(case + ":split", row.get("split") == "val")
        audit.check(case + ":arm_method", row.get("method") == run["manifest"]["arms"][arm]["method"])
        audit.check(
            case + ":execution_override",
            row.get("execution_override") == run["manifest"].get("execution_override"),
        )
        seed = cfg["evaluation"]["seed"] + cfg["data"]["tasks"].index(task) * 1000 + idx
        audit.check(case + ":noise_seed", row.get("noise_seed") == seed)
        audit.check(
            case + ":no_infrastructure_error", row.get("status") not in ERRORS and not row.get("error")
        )
        audit.check(case + ":official_status", row.get("status") in {"success", "fail", "timeout"})
        if row.get("status") == "timeout":
            env = row.get("environment") or {}
            audit.check(
                case + ":normal_benchmark_timeout",
                row.get("terminal_status") == "timeout"
                and row.get("primitive_commands") == env.get("requested_max_steps", -99) + 1
                and row.get("primitive_commands") == env.get("effective_max_steps", -99) - 1
                and env.get("requested_max_steps") == cfg["evaluation"]["max_steps"]
                and row.get("error") is None,
                native_commands=row.get("primitive_commands"),
                terminal_status=row.get("terminal_status"),
            )
        audit.check(
            case + ":within_wall_clock_guard",
            row.get("runtime_seconds", float("inf")) < cfg["evaluation"]["timeout_seconds"],
            runtime_seconds=row.get("runtime_seconds"),
            guard_seconds=cfg["evaluation"]["timeout_seconds"],
        )
        observer = row.get("task_diagnostics") or {}
        audit.check(case + ":observer", observer.get("status") == "complete" and not observer.get("errors"))
        audit.check(case + ":command_count", row.get("primitive_commands") == len(row.get("actions", [])))
        actions = row.get("actions", [])
        audit.check(
            case + ":finite_actions",
            all(
                len(action) == 8 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in action)
                for action in actions
            ),
        )
        audit.check(
            case + ":video",
            bool(row.get("video_path"))
            and Path(row["video_path"]).is_file()
            and Path(row["video_path"]).stat().st_size > 0,
        )
        summary = observer.get("summary") or {}
        timeline = [x for x in observer.get("timeline", []) if "tcp_xyz_m" in x]
        reset_phase = (observer.get("reset") or {}).get("current_task_index")
        phases = [x["current_task_index"] for x in timeline if "current_task_index" in x]
        phase_progress = None if reset_phase is None or not phases else max(0, max(phases) - reset_phase)
        audit.check(
            case + ":reported_phase_progress",
            summary.get("completed_subgoal_index_increase") == phase_progress,
        )
        transitions, previous = [], None
        for point in timeline:
            grasped = [a["index"] for a in point.get("candidates", []) if a.get("grasped")]
            if grasped != previous and (grasped or previous):
                transitions.append(
                    {
                        "native_command": point["command_index"],
                        "indices": grasped,
                        "phase": point.get("current_task_index"),
                    }
                )
            previous = grasped
        target_index = (observer.get("target") or {}).get("first_index")
        first_lift = next(
            (point for point in timeline if any(a.get("lifted") for a in point.get("candidates", []))), None
        )
        release = None
        if first_lift is not None:
            release = next(
                (
                    point
                    for point in timeline
                    if point["command_index"] > first_lift["command_index"]
                    and not any(a.get("grasped") for a in point.get("candidates", []))
                ),
                None,
            )
        placement = None
        controlled = [point for point in timeline if point.get("controlled_object")]
        if controlled and target_index is not None:
            last = controlled[-1]
            obj = last["controlled_object"]["xyz_m"]
            distances = {a["index"]: 1000 * math.dist(obj[:2], a["xyz_m"][:2]) for a in last["candidates"]}
            placement = {
                "native_command": last["command_index"],
                "target_index": target_index,
                "nearest_pad_index": min(distances, key=distances.get),
                "correct_pad_xy_error_mm": distances[target_index],
                "all_pad_xy_distances_mm": distances,
                "definition": "Last active sampled controlled-object XY distance to pad centers; excludes terminal auto-step, not an official success surrogate",
            }
        probes = []
        saved_probes = row.get("fixed_current_probes", [])
        if arm == "full":
            expected_probes = {
                d for d in run["manifest"]["probe_decisions"] if d < len(row.get("decisions", []))
            }
            audit.check(
                case + ":probe_coverage",
                {p.get("decision_index") for p in saved_probes} == expected_probes
                and len(saved_probes) == len(expected_probes),
                expected=sorted(expected_probes),
            )
        for p in row.get("fixed_current_probes", []):
            d = p["decision_index"]
            audit.check(
                case + f":probe{d}:passed", p.get("status") == "passed" and p.get("full_replay_passed")
            )
            audit.check(
                case + f":probe{d}:exact_full_replay", p.get("full_replay_max_abs_action_error") == 0.0
            )
            if p.get("status") != "passed":
                continue
            full = p["arms"]["FULL"]
            execute, stride = p["executed_targets_per_chunk"], cfg["model"].get("action_stride", 1)
            expected = [a for a in full["physical_actions"][:execute] for _ in range(stride)]
            actual = row["actions"][d * execute * stride : (d + 1) * execute * stride]
            err = max_abs(expected[: len(actual)], actual)
            audit.check(
                case + f":probe{d}:executed_targets_match",
                bool(actual) and err is not None and err <= 1e-7,
                max_abs_error=err,
            )
            deltas = {}
            for alternative, payload in p["arms"].items():
                xy = [1000 * math.dist(a, b) for a, b in zip(payload["tcp_targets_m"], full["tcp_targets_m"])]
                measured = statistics.mean(xy[:execute])
                audit.check(
                    case + f":probe{d}:{alternative}:tcp_recalculation",
                    abs(measured - payload["tcp_change_executed_targets_mean_mm"]) <= 1e-8,
                )
                deltas[alternative] = measured
                counts = payload["history_source_counts"]
                audit.check(
                    case + f":probe{d}:{alternative}:source_counts",
                    sum(counts.values()) == payload["history_leaf_count"],
                )
                if alternative == "erase_conditioning":
                    audit.check(case + f":probe{d}:conditioning_erased", counts.get("conditioning", 0) == 0)
                if alternative == "erase_execution":
                    audit.check(case + f":probe{d}:execution_erased", counts.get("execution", 0) == 0)
                if alternative == "CURRENT":
                    audit.check(case + f":probe{d}:current_empty", payload["history_leaf_count"] == 0)
            probes.append(
                {
                    "decision_index": d,
                    "source_index": p["execution_source_index"],
                    "noise_sha256": p["noise_sha256"],
                    "executed_tcp_changes_mm": deltas,
                    "full_replay_error": p["full_replay_max_abs_action_error"],
                }
            )
        diagnostics.append(
            {
                "task": task,
                "episode_idx": idx,
                "arm": arm,
                "status": row["status"],
                "native_commands": row.get("primitive_commands"),
                "target": observer.get("target"),
                "first_grasp_correct": summary.get("first_grasp_correct"),
                "first_lift_correct": summary.get("first_lift_correct"),
                "first_lift_command": None if first_lift is None else first_lift["command_index"],
                "first_release_after_lift": None
                if release is None
                else {
                    "native_command": release["command_index"],
                    "phase": release.get("current_task_index"),
                    "issued_gripper_command": row["actions"][release["command_index"] - 1][-1],
                },
                "phase_progress": phase_progress,
                "grasp_transitions": transitions,
                "last_active_placement": placement,
                "probe_sensitivity": probes,
                "video": row.get("video_path"),
            }
        )
    for task, idx, arm in sorted(run["planned"]):
        if arm == "full":
            continue
        full, alt = run["rows"].get((task, idx, "full")), run["rows"].get((task, idx, arm))
        if full is None or alt is None:
            continue
        audit.check(
            f"{label}/{task}/{idx}/{arm}:paired_reset",
            all(same_inputs(full, alt).values()),
            fields=same_inputs(full, alt),
        )
        if task == "SwingXtimes" and arm == "erase_conditioning":
            audit.check(
                f"{label}/SwingXtimes/{idx}:no_demo_placebo",
                full["actions"] == alt["actions"] and full["status"] == alt["status"],
            )
    return diagnostics


def counts(run):
    out = []
    groups = sorted({(task, arm) for task, idx, arm in run["planned"]})
    for task, arm in groups:
        plan = [k for k in run["planned"] if k[0] == task and k[2] == arm]
        rows = [run["rows"][k] for k in plan if k in run["rows"]]
        status = Counter(r["status"] for r in rows)
        out.append(
            {
                "task": task,
                "arm": arm,
                "planned": len(plan),
                "recorded": len(rows),
                "pending": len(plan) - len(rows),
                "success": status["success"],
                "statuses": dict(status),
                "complete_success_rate": status["success"] / len(plan) if len(rows) == len(plan) else None,
            }
        )
    return out


def compare(old, new, audit):
    audit.check("matched_planned_cases", old["planned"] == new["planned"])
    for key in ("arms", "probe_decisions", "split"):
        audit.check("matched_manifest:" + key, old["manifest"].get(key) == new["manifest"].get(key))
    for key in ("data", "geometry", "model", "training", "seed"):
        audit.check("matched_config:" + key, old["config"].get(key) == new["config"].get(key))
    for key in ("seed", "max_steps", "timeout_seconds", "episode_count"):
        audit.check(
            "matched_evaluation:" + key,
            old["config"]["evaluation"].get(key) == new["config"]["evaluation"].get(key),
        )
    audit.check(
        "matched_execution_override",
        old["manifest"]["execution_override"] == new["manifest"]["execution_override"],
    )
    comparisons = []
    for key in sorted(old["planned"] | new["planned"]):
        a, b = old["rows"].get(key), new["rows"].get(key)
        if a is None or b is None:
            comparisons.append({"task": key[0], "episode_idx": key[1], "arm": key[2], "pending": True})
            continue
        control = same_inputs(a, b)
        audit.check(
            "cross_checkpoint_reset:" + "/".join(map(str, key)), all(control.values()), fields=control
        )
        for field in (
            "action_space",
            "clock",
            "control_hz",
            "physics_hz",
            "requested_max_steps",
            "effective_max_steps",
            "observation_stride",
            "observation_hz",
            "conditioning_frames",
            "conditioning_frames_recorded",
        ):
            audit.check(
                "cross_checkpoint_environment:" + "/".join(map(str, key)) + ":" + field,
                a.get("environment", {}).get(field) == b.get("environment", {}).get(field),
            )
        ap = {
            p["decision_index"]: p for p in a.get("fixed_current_probes", []) if p.get("status") == "passed"
        }
        bp = {
            p["decision_index"]: p for p in b.get("fixed_current_probes", []) if p.get("status") == "passed"
        }
        reset_probe = None
        for d in ap.keys() & bp.keys():
            audit.check(
                "cross_checkpoint_noise:" + "/".join(map(str, key)) + f":{d}",
                ap[d]["noise_sha256"] == bp[d]["noise_sha256"],
            )
        if 0 in ap and 0 in bp:
            reset_probe = {
                arm: {
                    "old_mm": ap[0]["arms"][arm]["tcp_change_executed_targets_mean_mm"],
                    "new_mm": bp[0]["arms"][arm]["tcp_change_executed_targets_mean_mm"],
                }
                for arm in ap[0]["arms"]
            }
        comparisons.append(
            {
                "task": key[0],
                "episode_idx": key[1],
                "arm": key[2],
                "pending": False,
                "old_status": a["status"],
                "new_status": b["status"],
                "success_change": int(b["status"] == "success") - int(a["status"] == "success"),
                "old_commands": a["primitive_commands"],
                "new_commands": b["primitive_commands"],
                "input_parity": control,
                "same_state_reset_probe": reset_probe,
            }
        )
    benefits = []
    for task in sorted({k[0] for k in old["planned"]}):
        for arm in sorted({k[2] for k in old["planned"] if k[2] != "full"}):
            row = {"task": task, "control": arm}
            for label, run in (("old", old), ("new", new)):
                keys = sorted({(t, i) for t, i, a in run["planned"] if t == task and a == "full"})
                pairs = [(run["rows"].get((t, i, "full")), run["rows"].get((t, i, arm))) for t, i in keys]
                complete = all(a is not None and b is not None for a, b in pairs)
                valid = [(a, b) for a, b in pairs if a and b]
                admissible = [
                    (a, b)
                    for a, b in valid
                    if a["status"] not in ERRORS
                    and b["status"] not in ERRORS
                    and not a.get("error")
                    and not b.get("error")
                    and all(same_inputs(a, b).values())
                ]
                row[label] = {
                    "planned_pairs": len(keys),
                    "recorded_pairs": len(valid),
                    "complete": complete,
                    "valid_pairs": len(admissible),
                    "full_only_success": sum(
                        a["status"] == "success" and b["status"] != "success" for a, b in admissible
                    ),
                    "control_only_success": sum(
                        a["status"] != "success" and b["status"] == "success" for a, b in admissible
                    ),
                    "both_success": sum(a["status"] == b["status"] == "success" for a, b in admissible),
                }
                row[label]["success_benefit_percentage_points"] = (
                    100 * (row[label]["full_only_success"] - row[label]["control_only_success"]) / len(keys)
                    if complete and len(admissible) == len(keys) and keys
                    else None
                )
            benefits.append(row)
    return {"cases": comparisons, "history_benefit": benefits}


# Report rendering is defined below.


def audit_media(runs, audit):
    executable = shutil.which("ffmpeg")
    if executable is None:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    paths = {}
    for run in runs:
        for row in run["rows"].values():
            if row.get("video_path"):
                paths[Path(row["video_path"]).resolve()] = row
    results = []
    for path, row in paths.items():
        try:
            process = subprocess.run(
                [
                    executable,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostats",
                    "-progress",
                    "pipe:1",
                    "-threads",
                    "1",
                    "-i",
                    str(path),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            audit.check("video_decode:" + str(path), False, error=str(exc))
            results.append({"path": str(path), "full_decode_passed": False, "error": str(exc)})
            continue
        progress = {}
        for line in process.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                progress[key] = value
        frame_count = int(progress.get("frame", 0))
        duration = float(progress.get("out_time_us", 0)) / 1e6
        baseline = row["environment"]["conditioning_frames_recorded"] + 1 + row["primitive_commands"]
        terminal_delta = frame_count - baseline
        valid = (
            process.returncode == 0
            and not process.stderr.strip()
            and progress.get("progress") == "end"
            and frame_count > 0
        )
        audit.check("video_decode:" + str(path), valid, returncode=process.returncode, stderr=process.stderr)
        # Ordinary wrapper output contributes one current reset image and one
        # image per issued command. A missing terminal image is permitted.
        audit.check(
            "video_frame_count:" + str(path),
            terminal_delta in (-1, 0),
            decoded=frame_count,
            reset_plus_commands=baseline,
            terminal_delta=terminal_delta,
        )
        audit.check(
            "video_duration:" + str(path),
            abs(duration - frame_count / 20) <= 0.051,
            duration_seconds=duration,
            frame_count=frame_count,
        )
        results.append(
            {
                "path": str(path),
                "sha256": sha(path) if path.is_file() else None,
                "decoded_frames": frame_count,
                "duration_seconds": duration,
                "terminal_frame_delta": terminal_delta,
                "full_decode_passed": valid,
            }
        )
    return {
        "method": "Complete FFmpeg CPU decode, frame-count/20Hz-duration checks and SHA256",
        "ffmpeg": executable,
        "videos": results,
        "count": len(results),
    }


def audit_html_links(destination):
    links = re.findall(r'(?:href|src)="([^"]+)"', destination.read_text())
    missing = []
    for url in links:
        if "://" in url or url.startswith("#"):
            continue
        path = destination.parent / unquote(url.split("#", 1)[0])
        if not path.exists():
            missing.append(url)
    return {"links": len(links), "missing": missing}


def audit_recovery(run, audit):
    manifest_path = run["root"] / "recovery/manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read(manifest_path)
    status = read(run["root"] / "recovery_status.json")
    for name, digest in manifest["original_file_hashes"].items():
        audit.check("recovery:original_config_manifest:" + name, sha(run["root"] / name) == digest)
    for name, digest in manifest["initial_result_hashes"].items():
        audit.check("recovery:published_result_preserved:" + name, sha(run["root"] / name) == digest)
    original_keys = set()
    for name in manifest["initial_result_hashes"]:
        row = read(run["root"] / name)
        original_keys.add((row["task"], row["episode_idx"], row["arm"]))
    missing_keys = {(r["task"], r["episode_idx"], r["arm"]) for r in manifest["initial_missing_cases"]}
    audit.check(
        "recovery:exact_plan_partition",
        not (original_keys & missing_keys) and original_keys | missing_keys == run["planned"],
    )
    for published in status.get("published_this_session", []):
        case = published["case"]
        path = (
            run["root"]
            / "results"
            / case["arm"]
            / "episodes"
            / f"{case['task']}_{case['episode_idx']:03d}.json"
        )
        audit.check(
            "recovery:new_published_hash:" + path.name + ":" + case["arm"], sha(path) == published["sha256"]
        )
    if status.get("status") == "complete":
        audit.check("recovery:all_workers_clean", all(code == 0 for code in status["worker_exit_codes"]))
        audit.check("recovery:all_planned_records", status.get("recorded") == len(run["planned"]))
        audit.check(
            "recovery:original_hash_preservation_flag", status.get("original_result_hashes_preserved") is True
        )
    else:
        audit.check("recovery:not_failed", status.get("status") == "running")
    return {
        "reason": manifest["reason"],
        "receipt": str(manifest_path),
        "status_path": str(run["root"] / "recovery_status.json"),
        "status": status.get("status"),
        "initial_published": len(original_keys),
        "original_missing": len(missing_keys),
        "original_workers": manifest["original_workers"],
        "recovery_workers": manifest["recovery_workers"],
        "interpretation": "Session interruption is infrastructure, not a policy outcome. Only missing planned cases rerun; existing published JSON preserved byte-for-byte.",
    }


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name, delete=False
    ) as temporary:
        temporary.write(text)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _protocol(run):
    cfg, manifest = run["config"], run["manifest"]
    native = cfg["data"]["nominal_control_hz"]
    override = manifest["execution_override"]
    return {
        "predicted_targets": cfg["model"]["horizon"],
        "executed_targets": override["execute_actions"],
        "action_hz": native / cfg["model"].get("action_stride", 1),
        "observation_hz": native / cfg["data"].get("observation_stride", 1),
        "flow_steps": cfg["model"]["flow_steps"],
        "tasks": list(dict.fromkeys(task for task, _, _ in sorted(run["planned"]))),
        "planned_cases": len(run["planned"]),
        "execution_override": override,
    }


def _source_compatibility(old, new):
    a, b = old["manifest"].get("source", {}), new["manifest"].get("source", {})
    identical = bool(a.get("source_sha256")) and a.get("source_sha256") == b.get("source_sha256")
    return {
        "reference": a,
        "candidate": b,
        "identical_source_hash": identical,
        "review_required": not identical,
        "interpretation": (
            "Recorded source hashes match. Numeric/configuration checks still apply."
            if identical
            else "Recorded source hashes differ or are missing. Configuration and numeric checks do not establish "
            "implementation equivalence: review relevant encoder, policy, simulator and diagnostic changes "
            "before attributing differences solely to the checkpoint. No automatic compatibility review is claimed."
        ),
    }


def compare_history_runs(
    reference_history,
    history,
    *,
    output,
    reference_recent=None,
    recent=None,
    require_complete=False,
    validate_media=False,
):
    """Audit saved rollouts and write reports without loading models or running simulations.

    ``output`` is a directory for ``history_comparison.{json,md,html}``.
    The return includes ``complete``, ``audit_passed``, ``valid`` and ``artifacts``.
    Source differences are exposed separately as ``requires_source_review``;
    the tool does not pretend to audit unseen historical source code.
    """
    from stmem.inspection.history_comparison_report import html_report, markdown_report

    if (reference_recent is None) != (recent is None):
        raise ValueError("Supply both reference_recent and recent, or neither")
    audit = Audit()
    old, new = load(reference_history, audit), load(history, audit)
    result = {
        "scope": "Read-only audit and matched checkpoint comparison of saved validation history diagnostics",
        "old_directory": str(old["root"]),
        "new_directory": str(new["root"]),
        "old_checkpoint": old["manifest"]["checkpoint"],
        "new_checkpoint": new["manifest"]["checkpoint"],
        "protocol": _protocol(new),
        "old_counts": counts(old),
        "new_counts": counts(new),
        "old_diagnostics": audit_run(old, audit),
        "new_diagnostics": audit_run(new, audit),
        "comparison": compare(old, new, audit),
        "complete": len(old["rows"]) == len(old["planned"]) and len(new["rows"]) == len(new["planned"]),
        "source_compatibility": _source_compatibility(old, new),
        "input_integrity": {"reference": old["integrity"], "candidate": new["integrity"]},
        "recovery": {"reference": audit_recovery(old, audit), "candidate": audit_recovery(new, audit)},
    }
    result["requires_source_review"] = result["source_compatibility"]["review_required"]
    for record in result["recovery"].values():
        if record:
            result["complete"] &= record["status"] == "complete"
    recent_runs = None
    if recent is not None:
        a, b = load(reference_recent, audit), load(recent, audit)
        recent_runs = a, b
        for followup, primary in ((a, old), (b, new)):
            audit.check(
                "recent_window_arms:" + str(followup["root"]),
                set(followup["manifest"]["arms"]) == {"full", "recent_execution"},
            )
            for key in ("model", "data", "geometry", "training", "seed"):
                audit.check(
                    "recent_matches_primary_config:" + str(followup["root"]) + ":" + key,
                    followup["config"].get(key) == primary["config"].get(key),
                )
            audit.check(
                "recent_matches_primary_checkpoint:" + str(followup["root"]),
                followup["manifest"]["checkpoint"]["sha256"] == primary["manifest"]["checkpoint"]["sha256"],
            )
            audit.check(
                "recent_matches_primary_execution:" + str(followup["root"]),
                followup["manifest"]["execution_override"] == primary["manifest"]["execution_override"],
            )
            # FULL is a copied reference, not another trial or independently chosen scene.
            for key, row in followup["rows"].items():
                if key[2] == "full":
                    reference_row = primary["rows"].get(key)
                    cleaned = dict(row)
                    cleaned.pop("resume_integrity", None)
                    cleaned.pop("video_path", None)
                    cleaned.pop("reused_reference_result", None)
                    primary_cleaned = dict(reference_row or {})
                    primary_cleaned.pop("resume_integrity", None)
                    primary_cleaned.pop("video_path", None)
                    audit.check(
                        "recent_matches_primary_full:" + str(followup["root"]) + ":" + str(key),
                        cleaned == primary_cleaned,
                    )
        result["recent"] = {
            "old_directory": str(a["root"]),
            "new_directory": str(b["root"]),
            "old_counts": counts(a),
            "new_counts": counts(b),
            "old_diagnostics": audit_run(a, audit),
            "new_diagnostics": audit_run(b, audit),
            "comparison": compare(a, b, audit),
            "source_compatibility": _source_compatibility(a, b),
        }
        result["requires_source_review"] |= result["recent"]["source_compatibility"]["review_required"]
        result["complete"] &= len(a["rows"]) == len(a["planned"]) and len(b["rows"]) == len(b["planned"])
    if validate_media:
        if result["complete"]:
            result["media"] = audit_media([old, new, *(recent_runs or ())], audit)
        else:
            audit.check("complete_inputs_required_for_media_audit", False)
    result["checks"] = audit.checks
    result["failed_checks"] = [row for row in audit.checks if not row["passed"]]
    result["total_checks"] = len(audit.checks)
    result["passed_checks"] = sum(row["passed"] for row in audit.checks)
    result["audit_passed"] = not result["failed_checks"]
    result["valid"] = result["audit_passed"] and (result["complete"] or not require_complete)
    output = Path(output).resolve()
    artifacts = {
        kind: str(output / ("history_comparison." + suffix))
        for kind, suffix in [("json", "json"), ("markdown", "md"), ("html", "html")]
    }
    result["artifacts"] = artifacts
    _atomic_text(artifacts["json"], json.dumps(result, indent=2, allow_nan=False) + "\n")
    _atomic_text(artifacts["markdown"], markdown_report(result))
    _atomic_text(artifacts["html"], html_report(result, Path(artifacts["html"]), old, new, recent_runs))
    result["html_links"] = audit_html_links(Path(artifacts["html"]))
    if result["html_links"]["missing"]:
        result["valid"] = result["audit_passed"] = False
        result["failed_checks"].append(
            {"check": "html_links", "passed": False, "missing": result["html_links"]["missing"]}
        )
        result["checks"].append(result["failed_checks"][-1])
        result["total_checks"] += 1
        _atomic_text(artifacts["markdown"], markdown_report(result))
        _atomic_text(artifacts["html"], html_report(result, Path(artifacts["html"]), old, new, recent_runs))
    _atomic_text(artifacts["json"], json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-history", required=True, type=Path)
    parser.add_argument("--history", required=True, type=Path)
    parser.add_argument("--reference-recent", type=Path)
    parser.add_argument("--recent", type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="Report directory; input records are never modified"
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--validate-media", action="store_true", help="Fully decode referenced videos on CPU")
    args = parser.parse_args(argv)
    if (args.reference_recent is None) != (args.recent is None):
        parser.error("--reference-recent and --recent must be supplied together")
    result = compare_history_runs(**vars(args))
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "complete",
                    "valid",
                    "requires_source_review",
                    "passed_checks",
                    "total_checks",
                    "failed_checks",
                    "artifacts",
                )
            },
            indent=2,
        )
    )
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
