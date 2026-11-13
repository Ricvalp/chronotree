"""Explicit compact-reader jobs for the bounded local student inspection.

The coordinator owns process lifetime, locks, source checks and resumption.
These builders reuse the tested reset rendering and whole-chunk button probes;
they do not route compact students through the teacher-only inspection suite.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

from stmem.training.checkpoints import atomic_json, file_sha256

REPO = Path(__file__).resolve().parents[3]
PLACEMENTS = ("baseline", "x_plus_10cm", "y_plus_10cm", "y_minus_10cm")
NOISE_SEEDS = (2701, 2702, 2703, 2704)
BUTTON_TASKS = ("VideoRepick", "SwingXtimes", "PickXtimes")


def simulation_jobs(plan, plan_path=None, *, python=None):
    """Describe one reset job and matched TREE/FLAT rollout jobs per student.

    ``plan_path`` is accepted for coordinator interfaces, but the reused CLIs
    take explicit inputs. The parent pins this job list to its immutable plan.
    Jobs never submit to Slurm. The rollout subset is opt-in at the coordinator.
    """
    plan_path = Path(plan_path) if plan_path is not None else Path(plan["output"]) / "student_inspection.json"
    python = python or sys.executable
    protocol = plan["protocol"]
    execute = protocol.get("execute_targets", 8)
    budget = protocol.get("budget", 128)
    if execute != 8 or budget != 128:
        raise ValueError("Student simulation protocol fixes eight executed targets and B128")
    tasks = protocol.get("simulation_tasks", ["VideoRepick"])
    episodes = protocol.get("simulation_episodes", [0])
    if (not tasks or len(set(tasks)) != len(tasks) or any(task not in BUTTON_TASKS for task in tasks)):
        raise ValueError("Unsupported or repeated fixed-scene button task")
    if (not episodes or len(set(episodes)) != len(episodes)
            or any(type(episode) is not int or not 0 <= episode < 50 for episode in episodes)):
        raise ValueError("Simulation episodes must be unique official validation indices")
    output = Path(plan["output"])
    clip = str(Path(plan["assets"]["clip_checkpoint"]).resolve())
    jobs = []
    for label, entry in sorted(plan["checkpoints"].items()):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", label):
            raise ValueError("Invalid student label")
        source = str(Path(entry["path"]).resolve())
        reset = output / "reset_cube_chunks" / label
        command = [python, "-u", str(REPO / "scripts/visualize_reset_cube_chunks.py"),
                   "--checkpoint", f"{label}={source}", "--output", str(reset),
                   "--device", plan["device"], "--visual-checkpoint", clip,
                   "--method", "CURRENT", "--execute-actions", str(execute),
                   "--split", "val", "--episode", "0", "--timeout-seconds", "600",
                   "--seeds", *map(str, NOISE_SEEDS), "--placements", *PLACEMENTS]
        jobs.append({"name": f"reset_{label}", "stage": "reset", "label": label,
                     "method": "CURRENT", "command": command, "output": str(reset),
                     "expected_outputs": [str(reset / "payload.json"), str(reset / "index.html"),
                                          str(reset / f"{label}.html")],
                     "scope": "Empty-history cube-position sensitivity; no generated command is executed"})
        for task in tasks:
            for reader in ("TREE", "FLAT"):
                for episode in episodes:
                    # One episode per process makes completed outcomes reusable
                    # without repeating a published failure after interruption.
                    directory = output / "rollouts" / label / task / reader.lower() / f"episode_{episode:02d}"
                    command = [python, "-u", "-m", "stmem.inspection.button_probe",
                               "--checkpoint", source, "--output", str(directory),
                               "--task", task, "--episodes", str(episode),
                               "--method", reader, "--execute-actions", str(execute),
                               "--stable-observation-ids", "--clip-checkpoint", clip,
                               "--device", plan["device"]]
                    jobs.append({"name": f"button_{label}_{task.lower()}_{reader.lower()}_{episode:02d}",
                                 "stage": "rollouts", "label": label, "method": reader,
                                 "task": task, "episodes": [episode],
                                 "command": command, "output": str(directory),
                                 "expected_outputs": [str(directory / "results.json"), str(directory / "index.html"),
                                                      str(directory / "config.json")],
                                 "scope": "Fixed validation-scene button approach and prefix/tail diagnostic; not a benchmark score"})
    # Expert-prefix handoffs are optional, bounded training-scene diagnostics.
    # One trial per job keeps a published normal failure/cap reusable on resume.
    from stmem.inspection.student_handoff import SCENES, PREFIXES
    first_label = sorted(plan["checkpoints"])[0]
    for label in sorted(plan["checkpoints"]):
        for task, episode in SCENES:
            cases = [(reader, prefix, False) for reader in ("TREE", "FLAT") for prefix in PREFIXES]
            if label == first_label and task == "VideoUnmask":
                cases.insert(0, ("TREE", 0, True))
            for reader, prefix, control in cases:
                suffix = "expert_control" if control else f"prefix_{prefix:03d}"
                directory = output / "handoff" / label / task / reader.lower() / f"episode_{episode:02d}" / suffix
                command = [python, "-u", "-m", "stmem.inspection.student_handoff", "--plan", str(plan_path),
                           "--label", label, "--method", reader, "--task", task, "--episode", str(episode),
                           "--prefix", str(prefix), "--output", str(directory)]
                if control:
                    command.append("--expert-control")
                jobs.append({"name": f"handoff_{label}_{task.lower()}_{reader.lower()}_{episode:02d}_{suffix}",
                             "stage": "handoff", "label": label, "method": reader, "task": task,
                             "episodes": [episode], "prefix_actions": prefix, "expert_control": control,
                             "command": command, "output": str(directory),
                             "expected_outputs": [str(directory / "result.json"), str(directory / "index.html"),
                                                  str(directory / "config.json")],
                             "scope": "Training-scene expert-prefix diagnostic capped at 320 native commands; not a benchmark score"})
    return jobs


def _load(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_reset_result(plan, job):
    """Check reader/cadence/empty-history claims against the saved raw chunks."""
    entry = plan["checkpoints"][job["label"]]
    directory = Path(job["output"])
    payload = _load(directory / "payload.json")
    _require(payload.get("method") == "CURRENT", "Reset probe did not use explicit CURRENT reader")
    _require(payload.get("task") == "PickXtimes" and payload.get("split") == "val"
             and payload.get("episode_idx") == 0, "Reset scene differs from the fixed protocol")
    _require(payload.get("horizon") == 16 and payload.get("execute_steps") == 8
             and payload.get("flow_steps") == 8 and payload.get("action_stride") == 2
             and payload.get("observation_stride") == 2, "Reset action protocol differs")
    _require(payload.get("noise_seeds") == list(NOISE_SEEDS), "Reset noise seeds differ")
    _require([scene["id"] for scene in payload.get("scenes", [])] == list(PLACEMENTS),
             "Reset placement set differs")
    models = payload.get("models", [])
    _require(len(models) == 1 and models[0].get("id") == job["label"], "Reset student identity differs")
    _require(models[0].get("checkpoint_sha256") == entry["sha256"]
             and models[0].get("step") == entry["step"]
             and models[0].get("checkpoint_stage") == "compact", "Reset checkpoint identity differs")
    validation = payload.get("validation", {})
    _require(validation.get("predicted_commands_executed") == 0
             and validation.get("predicted_chunks") == len(PLACEMENTS) * len(NOISE_SEEDS)
             and validation.get("repeat_baseline_points_and_first_seed_predictions_bit_identical") is True,
             "Reset repetition/execution controls did not pass")
    baseline = payload["scenes"][0]
    rows = []
    for scene in payload["scenes"]:
        _require(scene.get("conditioning_frames") == scene.get("execution_history_frames") == 0,
                 "Reset probe unexpectedly contains history")
        _require(scene.get("goal") == baseline.get("goal")
                 and scene.get("proprio") == baseline.get("proprio"),
                 "Cube relocation changed goal or robot state")
        points_path = directory / scene["input_npz"]
        _require(points_path.resolve().is_relative_to(directory.resolve()), "Escaping reset artifact path")
        with np.load(points_path, allow_pickle=False) as arrays:
            points = arrays["points"][arrays["valid"]]
            _require(np.array_equal(points, np.asarray(scene["points"])), "Rendered point input changed")
        _require(np.isfinite(points).all(), "Nonfinite reset point input")
        for seed in NOISE_SEEDS:
            row = scene["predictions"][job["label"]][str(seed)]
            actions = np.asarray(row["raw_actions"], dtype=np.float32)
            paths = np.asarray(row["tcp_xyz"], dtype=np.float64)
            reference = np.asarray(baseline["predictions"][job["label"]][str(seed)]["tcp_xyz"])
            _require(actions.shape == (16, 8) and paths.shape == (16, 3)
                     and np.isfinite(actions).all() and np.isfinite(paths).all(), "Invalid reset actions or FK")
            timing = row["decision"]
            _require(timing.get("history_tokens") == timing.get("archive_fine_leaves") == 0,
                     "Historical tokens entered the reset grounding diagnostic")
            distance = np.linalg.norm(paths - reference, axis=-1) * 1000
            _require(np.allclose(distance, row["metrics"]["tcp_change_by_target_mm"], atol=1e-6, rtol=1e-6),
                     "Saved reset sensitivity does not match generated targets")
            rows.append({"placement": scene["id"], "noise_seed": seed,
                         "executed_prefix_tcp_change_mm": float(distance[:8].mean()),
                         "full_chunk_tcp_change_mm": float(distance.mean())})
    return {"status": "complete", "scope": job["scope"], "reader": "CURRENT",
            "historical_tokens": 0, "predicted_commands_executed": 0,
            "response_rows": rows, "html": str(directory / "index.html")}


def validate_rollout_result(plan, job):
    """Retain every planned episode; infrastructure errors are not successes."""
    entry = plan["checkpoints"][job["label"]]
    directory = Path(job["output"])
    payload = _load(directory / "results.json")
    identity = payload.get("checkpoint", {})
    _require(identity.get("sha256") == entry["sha256"] and identity.get("step") == entry["step"]
             and identity.get("stage") == "compact", "Rollout checkpoint identity differs")
    protocol = payload["protocol"]
    config = _load(directory / "config.json")
    _require(config["model"].get("flow_steps") == 8, "Rollout integration protocol differs")
    _require(protocol.get("task") == job["task"] and protocol.get("episodes") == job["episodes"]
             and protocol.get("execute_actions") == 8 and protocol.get("action_stride") == 2
             and protocol.get("horizon") == 16 and protocol.get("stable_observation_ids") is True,
             "Rollout task/cadence/identity protocol differs")
    trials = payload.get("trials", [])
    _require(sorted(trial["episode_idx"] for trial in trials) == sorted(job["episodes"]),
             "Rollout missing or duplicated planned episodes")
    statuses = Counter()
    for trial in trials:
        _require(trial.get("method") == job["method"] and trial.get("task") == job["task"]
                 and trial.get("split") == "val", "Wrong compact rollout reader or scene")
        _require(trial.get("checkpoint_sha256") == entry["sha256"], "Trial checkpoint identity differs")
        _require(trial.get("budget") == 128, "Trial history budget differs")
        _require(trial.get("execute_actions") == 8 and trial.get("action_stride") == 2,
                 "Trial action timing differs")
        statuses[trial["status"]] += 1
        _require(trial["status"] in ("success", "fail", "timeout"),
                 f"Rollout has an unresolved {trial['status']} record: {trial.get('error')}")
        video = Path(trial["video"])
        _require(video.resolve().is_relative_to(directory.resolve()) and video.is_file()
                 and video.stat().st_size > 0, "Missing or escaping rollout video")
    return {"status": "complete", "scope": job["scope"], "reader": job["method"],
            "planned": len(job["episodes"]), "recorded": len(trials), "statuses": dict(statuses),
            "success_rate": statuses["success"] / len(job["episodes"]),
            "html": str(directory / "index.html"),
            "note": "Fixed-scene diagnostic only; reported task success does not estimate benchmark success."}


def validate_simulation_job(plan, job):
    entry = plan["checkpoints"][job["label"]]
    _require(file_sha256(entry["path"]) == entry["sha256"], "Student checkpoint changed during simulation")
    for filename in job["expected_outputs"]:
        _require(Path(filename).is_file(), f"Missing simulation artifact: {filename}")
    if job["stage"] == "reset":
        result = validate_reset_result(plan, job)
    elif job["stage"] == "rollouts":
        result = validate_rollout_result(plan, job)
    elif job["stage"] == "handoff":
        from stmem.inspection.student_handoff import validate_result
        result = validate_result(plan, job)
    else:
        raise ValueError("Unknown student simulation stage")
    directory = Path(job["output"]).resolve()
    files = sorted(path for path in directory.rglob("*") if path.is_file() and not path.name.startswith("."))
    _require(all(path.resolve().is_relative_to(directory) for path in files), "Escaping simulation artifact")
    result["files"] = [str(path) for path in files]
    return result


def simulation_report_links(plan):
    """Link only reports actually produced; absent optional jobs stay absent."""
    root = Path(plan["output"]).resolve()
    result = []
    for job in simulation_jobs(plan):
        report = Path(job["output"]) / "index.html"
        if report.is_file():
            suffix = ("reset cube positions" if job["stage"] == "reset"
                      else f"{job['task']} {job['method']} episode {job['episodes'][0]}")
            if job["stage"] == "handoff":
                suffix += " expert replay control" if job["expert_control"] else f" prefix {job['prefix_actions']}"
            result.append({"label": f"{job['label']}: {suffix}", "path": str(report.resolve().relative_to(root))})
    return result


def prepare_simulation_job(plan, job):
    """Recover only unsealed interruptions, retaining every completed outcome.

    The coordinator calls this after validating its immutable plan and finding
    no completion receipt. A complete payload can have lost just its final HTML
    or receipt; regenerate presentation and reuse it. A recorded error needs
    diagnosis and is never silently rerun. Truly unfinished artifacts are moved
    intact to an attempt directory before a same-seed fresh invocation.
    """
    directory = Path(job["output"])
    root = Path(plan["output"]).resolve()
    _require(directory.resolve().is_relative_to(root), "Simulation output escapes inspection")
    if not directory.exists() or not any(directory.iterdir()):
        return {"reuse": False}
    if job["stage"] == "reset":
        payload_path = directory / "payload.json"
        if payload_path.exists():
            # A malformed unpublished reset file is an interrupted write; its
            # bytes are preserved below. A valid but incompatible file is an
            # identity error and must not be silently replaced.
            try:
                payload = _load(payload_path)
            except json.JSONDecodeError:
                payload = None
            if payload is not None:
                validate_reset_result(plan, job)
                from stmem.visualization.reset_chunks_html import write_reset_chunks_html

                if not all(Path(path).is_file() for path in job["expected_outputs"]):
                    write_reset_chunks_html(payload, directory)
                return {"reuse": True, "result": validate_simulation_job(plan, job)}
    elif job["stage"] == "rollouts":
        payload_path = directory / "results.json"
        if payload_path.exists():
            payload = _load(payload_path)  # Atomic writer: invalid JSON is corruption, not a partial record.
            trials = payload.get("trials", [])
            if not trials:
                recovered = []
                for episode in job["episodes"]:
                    path = directory / f"{job['task']}_{episode:02d}" / "result.json"
                    if path.exists():
                        recovered.append(_load(path))
                if recovered:
                    payload["trials"] = trials = recovered
                    # Per-episode outcome was published before the aggregate.
                    # Recover it rather than execute a failed episode again.
                    atomic_json(payload, payload_path)
            if trials:
                validate_rollout_result(plan, job)
                from stmem.inspection.button_report import write_report

                if not (directory / "index.html").is_file():
                    write_report(payload, directory)
                return {"reuse": True, "result": validate_simulation_job(plan, job)}
    elif job["stage"] == "handoff":
        if (directory / "result.json").exists():
            from stmem.inspection.student_handoff import validate_result, write_report
            validate_result(plan, job)
            if not (directory / "index.html").is_file():
                write_report(directory)
            return {"reuse": True, "result": validate_simulation_job(plan, job)}
    else:
        raise ValueError("Unknown student simulation stage")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = root / "interrupted_simulations" / job["name"] / f"{stamp}_{time.time_ns()}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    directory.rename(archive)
    return {"reuse": False, "preserved_interruption": str(archive)}
