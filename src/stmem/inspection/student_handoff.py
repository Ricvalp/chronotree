"""One bounded compact-student expert-prefix handoff, with immutable evidence.

Training-scene diagnostics are explicitly separate from benchmark evaluation.
An expert prefix uses real native-rate commands and observations; no duplicated
idle frames or privileged simulator labels enter policy conditioning.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from html import escape
import json
from pathlib import Path

import torch

from stmem.config import validate_config
from stmem.evaluation.runner import OnlinePolicy
from stmem.inspection import handoff
from stmem.inspection.handoff_helpers import source_for
from stmem.inspection.handoff_validation import validate_trial
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import load_policy, validate_policy_config
from stmem.vision.frozen import FrozenCLIPFeatures, resolve_visual_features

SCENES = (("VideoUnmask", 0), ("PickXtimes", 0))
PREFIXES = (0, 16, 32)
CAP = 320


class TrainingSceneCompactPolicy(OnlinePolicy):
    """Inspection-only random compact reader; production selection stays val-only.

    UNIFORM initializes the same StreamingMemory and validates the same positive
    history budget without claiming a learned router. Only the reader dispatch
    is then changed, after enforcing the fixed training-scene diagnostic scope.
    No production guard, simulator split, or checkpoint configuration is faked.
    """
    def __init__(self, policy, cfg, method, budget, *, selection_family):
        if (cfg.get("stage") != "compact" or cfg["evaluation"]["split"] != "train"
                or method not in ("TREE", "FLAT") or selection_family != method.lower()
                or budget != 128):
            raise ValueError("Explicit training-scene compact diagnostic reader required")
        super().__init__(policy, cfg, "UNIFORM", budget)
        self.method = method
        self.selection_family = selection_family


def diagnostic_config(state, assets):
    """Authorize only execute8 for this declared training-scene diagnostic."""
    if state["stage"] != "compact":
        raise ValueError("Student handoff requires a compact checkpoint")
    cfg = deepcopy(state["config"])
    if resolve_visual_features(cfg)["mode"] != "xyzrgb":
        cfg["model"]["visual_features"]["checkpoint"] = str(assets["clip_checkpoint"])
    validate_policy_config(state["config"], cfg)
    m = cfg["model"]
    if (m["horizon"] != 16 or m.get("action_stride", 1) != 2
            or cfg["data"].get("observation_stride", 1) != 2 or m["flow_steps"] != 8
            or m.get("history_mode", "full") == "none"):
        raise ValueError("Compact handoff requires H16, 10Hz, eight flow steps and history")
    original_execute = m["execute"]
    m["execute"] = 8
    cfg["stage"] = "compact"
    cfg["evaluation"]["split"] = "train"
    validate_config(cfg)
    return cfg, {"scope": "training_scene_diagnostic_not_benchmark", "checkpoint_execute": original_execute,
                 "execute_targets": 8, "action_stride": 2, "primitive_steps_per_replan": 16}


def expert_source(plan, task, episode):
    source = source_for(task, episode, plan["assets"])
    key = "prepared" if task == "VideoUnmask" else "pick_prefix"
    if source["sha256"] != plan["asset_hashes"][key]:
        raise ValueError("Expert source changed after inspection planning")
    return source


def write_report(directory):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    notes = ["Fixed training-scene expert-prefix diagnostic; not a benchmark success estimate.",
             "Prefixes count native 20Hz commands. Predicted 10Hz targets are each held for two native steps.",
             "All runs stop after at most 320 native commands; reaching that cap is not official task failure.",
             "Object/grasp diagnostics are privileged read-only labels and never policy inputs.",
             "Changes after handoff combine inherited expert state/history and subsequent policy behavior."]
    fields = {key: result.get(key) for key in ("task", "episode_idx", "variant", "checkpoint_step", "method", "budget",
                                               "prefix_commands", "expert_control", "policy_handoff_occurred",
                                               "terminal_during_expert_prefix", "status", "summary")}
    html = ("<!doctype html><html><head><meta charset='utf-8'><title>Student expert-prefix handoff</title>"
            "<style>body{font:16px system-ui;max-width:1000px;margin:30px auto;padding:0 20px}"
            "video{max-width:100%}pre{white-space:pre-wrap}</style></head><body>"
            "<h1>Compact student expert-prefix handoff</h1><ul>" +
            "".join(f"<li>{escape(note)}</li>" for note in notes) + "</ul>"
            "<video controls preload='metadata' src='rollout.mp4'></video><pre>" +
            escape(json.dumps(fields, indent=2)) + "</pre><p><a href='result.json'>Full native-command trace</a> · "
            "<a href='config.json'>Effective diagnostic configuration</a></p></body></html>")
    (directory / "index.html").write_text(html)


def validate_result(plan, job):
    directory = Path(job["output"])
    result = json.loads((directory / "result.json").read_text())
    cfg = json.loads((directory / "config.json").read_text())
    entry = plan["checkpoints"][job["label"]]
    expected = {"task": job["task"], "episode_idx": job["episodes"][0], "split": "train",
                "method": job["method"], "selection_family": job["method"].lower(), "budget": 128,
                "prefix_commands": job["prefix_actions"], "expert_control": job["expert_control"],
                "cap_total_commands": CAP, "absolute_time_noise": True, "training_scene_diagnostic": True,
                "inspection_manifest_sha256": plan["identity"]}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Handoff {key} differs from fixed protocol")
    identity = result["checkpoint_identity"]
    if (identity["sha256"] != entry["sha256"] or identity["step"] != entry["step"]
            or identity.get("stage") != "compact" or identity["horizon"] != 16):
        raise ValueError("Handoff checkpoint identity differs")
    if result["runner_sha256"] != handoff.runner_sha256():
        raise ValueError("Handoff runner source changed")
    if result["status"] not in ("success", "fail", "timeout", "stopped_at_declared_diagnostic_cap"):
        raise ValueError(f"Handoff unresolved error: {result['status']}: {result.get('error')}")
    if cfg["model"]["flow_steps"] != 8 or cfg["model"]["execute"] != 8:
        raise ValueError("Handoff decoding protocol differs")
    effective, override = diagnostic_config({"config": entry["config"], "stage": "compact"}, plan["assets"])
    # Runtime precision auto-resolution may be device dependent and is recorded.
    effective["training"]["precision"] = cfg["training"]["precision"]
    if cfg != effective or result.get("execution_override") != override:
        raise ValueError("Handoff effective checkpoint configuration changed")
    source = expert_source(plan, job["task"], job["episodes"][0])
    validated, counts = validate_trial(directory / "result.json", source, cfg, manifest_digest=plan["identity"])
    if job["expert_control"] and validated["status"] != "success":
        raise ValueError("Released-demo expert replay control did not succeed")
    for decision in validated["decisions"]:
        expected_seed = handoff.seed_for_primitive_time(job["task"], job["episodes"][0], decision["step"], "selection")
        if decision.get("selection_seed") != expected_seed:
            raise ValueError("Handoff reader-selection seed changed")
    video = Path(result["video"])
    if video.resolve() != (directory / "rollout.mp4").resolve() or not video.is_file():
        raise ValueError("Handoff video identity differs")
    return {"status": "complete", "scope": job["scope"], "reader": job["method"],
            "outcome": result["status"], "expert_control": job["expert_control"],
            "prefix_actions": job["prefix_actions"], "summary": validated["summary"], **counts,
            "html": str(directory / "index.html")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--method", choices=("TREE", "FLAT"), required=True)
    parser.add_argument("--task", choices=("VideoUnmask", "PickXtimes"), required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--prefix", type=int, choices=PREFIXES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expert-control", action="store_true")
    args = parser.parse_args(argv)
    from stmem.inspection.student_suite import verify_plan
    from stmem.inspection.student_simulation import simulation_jobs

    plan = json.loads(args.plan.read_text())
    verify_plan(plan)
    matches = [job for job in simulation_jobs(plan, args.plan) if job["stage"] == "handoff"
               and job["label"] == args.label and job["method"] == args.method and job["task"] == args.task
               and job["episodes"] == [args.episode] and job["prefix_actions"] == args.prefix
               and job["expert_control"] == args.expert_control and Path(job["output"]) == args.output]
    if len(matches) != 1:
        parser.error("Requested handoff is not a uniquely declared inspection job")
    job = matches[0]
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Handoff output already exists; coordinator must verify or preserve it before retry")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    entry = plan["checkpoints"][args.label]
    policy, state = load_policy(entry["path"], plan["device"])
    policy.eval().requires_grad_(False)
    cfg, override = diagnostic_config(state, plan["assets"])
    from stmem.doctor import choose_precision
    if cfg["training"]["precision"] == "auto":
        cfg["training"]["precision"] = choose_precision(policy.device)
    identity = {"path": entry["path"], "sha256": entry["sha256"], "step": state["step"], "stage": "compact",
                "horizon": 16, "history_mode": cfg["model"].get("history_mode", "full"),
                "action_stride": 2, "observation_stride": 2}
    atomic_json(cfg, args.output / "config.json")
    extractor = FrozenCLIPFeatures(cfg, device=policy.device) if resolve_visual_features(cfg)["mode"] != "xyzrgb" else None
    handoff.MANIFEST_DIGEST = plan["identity"]
    result = handoff.run(policy, cfg, state, identity, extractor, handoff.fast_transfer(policy),
                         expert_source(plan, args.task, args.episode),
                         task=args.task, episode=args.episode, prefix=args.prefix, variant=args.label,
                         cap=CAP, control=args.expert_control, method=args.method,
                         selection_family=args.method.lower(), budget=128, directory=args.output,
                         absolute_time_noise=True, training_scene_diagnostic=True,
                         execution_override=override, online_factory=TrainingSceneCompactPolicy)
    if file_sha256(entry["path"]) != entry["sha256"]:
        raise ValueError("Student changed during handoff")
    validate_result(plan, job)
    write_report(args.output)
    print(json.dumps({"event": "student_handoff_complete", "status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
