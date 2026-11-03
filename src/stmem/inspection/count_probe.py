"""Paired counting/history interventions on a fixed training scene, not a score.

Only expert observations, ordinary task text and the production clock enter the
policy. Privileged counts/positions label cases and measure predictions afterward.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch

from stmem.config import digest
from stmem.data.canonical import Decision, Frame
from stmem.evaluation.robomme_adapter import physical_action
from stmem.memory.frontier import validate_frontier
from stmem.models.policy import choose_records
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import load_checkpoint, load_policy, validate_policy_config
from stmem.visualization.kinematics import panda_observation_errors, panda_tcp_positions

NOISE_SEEDS = (2701, 2702, 2703, 2704)
MODES = ("natural", "fixed_current", "fixed_duration")


@dataclass
class CountCase:
    mode: str
    k: int
    current: Frame
    prefix: list[Frame]
    scene_id: str
    padding_frames: int = 0


def build_cases(source):
    """Keep causal prefixes; fix current/clock, then also match history duration.

    Natural cases have real expert current states. Fixed-current cases right-align
    real prefixes without changing their native spacing. Fixed-duration cases
    prepend repeated reset observations as an explicitly artificial idle control.
    No final-button or future expert observation can enter any case.
    """
    frames = source["execution"]
    boundaries = sorted(source["boundaries"], key=lambda b: b["completed_cycles"])
    if [b["completed_cycles"] for b in boundaries] != [1, 2, 3]:
        raise ValueError("Expected exactly three complete expert pick/place cycles")
    stride = source["checkpoint_config"]["data"].get("observation_stride", 1)
    indices = [b["frame_index"] for b in boundaries]
    if not (0 < indices[0] < indices[1] < indices[2] < len(frames)):
        raise ValueError("Cycle boundaries must increase inside the recorded trajectory")
    observed = frames[:indices[-1] + 1]
    if any(f.source != "execution" for f in observed):
        raise ValueError("Counting probe expects execution observations only")
    if [f.source_index for f in observed] != list(range(0, len(observed) * stride, stride)):
        raise ValueError("Expert retained observations do not follow the declared cadence")
    if any(f.time != f.source_index for f in observed):
        raise ValueError("Expected production source-local recorded-frame clocks")
    maximum = indices[-1]
    fixed_time = frames[maximum].time
    shared_current = replace(frames[indices[0]], time=fixed_time, source_index=int(fixed_time),
                             event_id="count-probe/shared-current")
    cases = []
    for mode in MODES:
        for k, end in zip((1, 2, 3), indices):
            if mode == "natural":
                current, prefix = frames[end], frames[:end]
                scene_id = f"natural_k{k}"
                padding = 0
            else:
                current, scene_id = shared_current, "shared_current"
                padding = maximum - end if mode == "fixed_duration" else 0
                raw = [frames[0]] * padding + frames[:end]
                offset = fixed_time - len(raw) * stride
                # Preserve real observation identity across interventions. UNIFORM
                # uses node IDs to break equal-mass ties, so renaming unchanged
                # observations would accidentally change the selected frontier.
                # Only artificial idle copies need fresh, distinct identities.
                prefix = [replace(f, time=offset + i * stride,
                                  source_index=int(offset + i * stride),
                                  event_id=f"count-probe/idle/{i}" if i < padding else f.event_id)
                          for i, f in enumerate(raw)]
            if not prefix or prefix[-1].time >= current.time:
                raise ValueError("History must strictly precede the separately encoded current")
            if len({f.event_id for f in prefix}) != len(prefix):
                raise ValueError("Repeated identities would violate exact historical coverage")
            cases.append(CountCase(mode, k, current, prefix, scene_id, padding))
    return cases


def valid_targets(k):
    """Overshooting an exact count has no valid ordinary expert continuation."""
    if k not in (1, 2, 3):
        raise ValueError("Completed count must be 1, 2 or 3")
    return range(k, 4)


def trajectory_metrics(xyz, current, button, cube, execute=8):
    """Continuous geometric proxies only: FK cannot establish button contact."""
    xyz, current, button, cube = map(lambda x: np.asarray(x, dtype=float),
                                   (xyz, current, button, cube))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not 1 <= execute <= len(xyz):
        raise ValueError("Invalid TCP trajectory/executed prefix")
    if not all(np.isfinite(v).all() for v in (xyz, current, button, cube)):
        raise ValueError("Nonfinite diagnostic geometry")
    button_distance = np.linalg.norm(xyz - button, axis=1)
    cube_distance = np.linalg.norm(xyz - cube, axis=1)
    bclose = np.linalg.norm(current - button) - button_distance[execute - 1]
    cclose = np.linalg.norm(current - cube) - cube_distance[execute - 1]
    return {
        "button_distance_end_mm": float(button_distance[-1] * 1000),
        "button_distance_min_first8_mm": float(button_distance[:execute].min() * 1000),
        "button_distance_first8_endpoint_mm": float(button_distance[execute - 1] * 1000),
        "button_xy_min_first8_mm": float(np.linalg.norm(xyz[:execute, :2] - button[:2], axis=1).min() * 1000),
        "button_closing_first8_mm": float(bclose * 1000),
        "cube_closing_first8_mm": float(cclose * 1000),
        "button_preference_first8_mm": float((bclose - cclose) * 1000),
        "tcp_displacement_first8_mm": float(np.linalg.norm(xyz[execute - 1] - current) * 1000),
        "tcp_z_change_first8_mm": float((xyz[execute - 1, 2] - current[2]) * 1000),
    }


def tensor_state_hash(state, prefixes):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        if any(name.startswith(p) for p in prefixes):
            value = value.detach().cpu().contiguous()
            result.update(name.encode())
            result.update(str((value.dtype, value.shape)).encode())
            result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def audit_checkpoint(path, expected_stage, source_config):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Incomplete checkpoint ZIP entry: {bad}")
    state = load_checkpoint(path)
    cfg = state["config"]
    if state["stage"] != expected_stage or cfg.get("debug") or cfg["data"].get("synthetic"):
        raise ValueError(f"Expected a real-data {expected_stage} checkpoint")
    validate_policy_config(source_config, cfg)
    if cfg["model"].get("proprioception_mode") != "none" or cfg["model"].get("condition_on_previous_commands", True):
        raise ValueError("This visual count probe requires the agreed no-proprio/no-command model")
    if cfg["model"]["horizon"] != 16 or cfg["model"].get("action_stride", 1) != 2:
        raise ValueError("This protocol uses 16 targets at 10 Hz, eight targets before replanning")
    for key in ("model", "training"):
        if state["provenance"][key] != digest(cfg[key]):
            raise ValueError(f"Checkpoint {key} provenance mismatch")
    for value in state["model"].values():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Checkpoint contains nonfinite parameters")
    return {"path": str(Path(path).resolve()), "sha256": file_sha256(path), "step": state["step"],
            "stage": state["stage"],
            "frozen_encoder_sha256": tensor_state_hash(state["model"], ("encoder.", "text_encoder."))}


def _position(diagnostics, name):
    for key in (name, name + "_position", name + "_xyz"):
        if key in diagnostics:
            return np.asarray(diagnostics[key], dtype=float).reshape(-1)[:3].tolist()
    raise ValueError(f"Missing diagnostic {name} position")


def summarize(predictions):
    groups = defaultdict(list)
    for row in predictions:
        groups[(row["label"], row["history_mode"], row["k"], row["x"])].append(row)
    aggregates = []
    for (label, mode, k, x), rows in groups.items():
        aggregates.append({"label": label, "history_mode": mode, "k": k, "x": x,
                           "noise_samples": len(rows), "expected_next": rows[0]["expected_next"],
                           "mean_metrics": {key: float(np.mean([r["metrics"][key] for r in rows]))
                                            for key in rows[0]["metrics"]}})
    lookup = {(r["label"], r["history_mode"], r["k"], r["x"], r["noise_seed"]): r for r in predictions}
    contrasts = []
    for row in predictions:
        label, mode, k, x, seed = (row[v] for v in ("label", "history_mode", "k", "x", "noise_seed"))
        comparisons = []
        if x == k and x < 3:
            comparisons.append(("goal_count_change_fixed_history", (label, mode, k, x+1, seed)))
        if mode != "natural" and k == x and k > 1:
            comparisons.append(("history_count_change_fixed_goal_and_current", (label, mode, k-1, x, seed)))
        for kind, other_key in comparisons:
            if other_key not in lookup:
                continue
            other = lookup[other_key]
            delta = np.asarray(row["tcp_positions_m"]) - np.asarray(other["tcp_positions_m"])
            contrasts.append({"label": label, "history_mode": mode, "kind": kind, "noise_seed": seed,
                              "press_case": {"k": k, "x": x},
                              "pick_case": {"k": other["k"], "x": other["x"]},
                              "mean_first8_tcp_change_mm": float(np.linalg.norm(delta[:8], axis=1).mean()*1000),
                              "press_minus_pick_button_preference_mm": row["metrics"]["button_preference_first8_mm"] - other["metrics"]["button_preference_first8_mm"]})
    return {"aggregates": aggregates, "paired_contrasts": contrasts,
            "interpretation": "Positive button preference means more geometric progress toward button than cube; it is not a press probability or contact test."}


@torch.no_grad()
def run_probe(source_path, teacher, student, output, device="cuda:0"):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory to preserve previous diagnostic results")
    source_path = Path(source_path).resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("task") != "PickXtimes" or source.get("split") != "train":
        raise ValueError("Only the fixed training-scene expert source is supported")
    if source.get("terminal_status") != "success" or source.get("replay_validation", {}).get("status") != "passed":
        raise ValueError("Expert source must finish successfully and pass independent replay")
    source_config = source["checkpoint_config"]
    inputs = [("teacher_full", Path(teacher), "teacher", "FULL")]
    if student:
        inputs.append(("student_uniform128", Path(student), "compact", "UNIFORM"))
    audits = {label: audit_checkpoint(path, stage, source_config) for label, path, stage, _ in inputs}
    if audits["teacher_full"]["sha256"] != source["checkpoint_sha256"]:
        raise ValueError("Teacher differs from the immutable checkpoint declared by expert generation")
    if len({entry["frozen_encoder_sha256"] for entry in audits.values()}) != 1:
        raise ValueError("Teacher/student frozen encoder weights differ; comparison would be confounded")
    cases = build_cases(source)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "running", "protocol": {"task": "PickXtimes", "scope": "fixed training-scene counterfactual diagnostic, not official evaluation",
            "execute_actions": 8, "action_stride": 2, "budget": 128, "noise_seeds": list(NOISE_SEEDS),
            "notes": ["X is the count in ordinary task language; k labels completed expert pick/place cycles.",
                      "Only k<=X is evaluated; overshooting an exact-count task has no normal expert continuation.",
                      "Natural: genuine expert current/prefix, changing only requested count within each k.",
                      "Fixed current: identical current cloud and clock, real prefixes right-aligned at unchanged frame spacing; history lengths differ.",
                      "Fixed duration: same current/clock and number of frames, padding shorter histories with repeated reset observations; this is an artificial idle control.",
                      "Fixed-current histories can have small state discontinuities; this is not an executable counterfactual rollout.",
                      "Targets are joint-angle predictions projected by FK, not executed motion or demonstrated contact.",
                      "Official planner used privileged geometry; the policy receives only ordinary allowed observations and language."]},
        "source": {"path": str(source_path), "sha256": file_sha256(source_path), "goals": source["goals"],
                   "episode_idx": source["episode_idx"], "video": str(source_path.parent / "expert.mp4"),
                   "teacher_checkpoint": str(Path(teacher).resolve())},
        "checkpoints": audits, "current_scenes": {}, "predictions": [], "case_audits": [],
        "implementation": {"probe_sha256": file_sha256(__file__),
                           "expert_generator_sha256": source["generator_sha256"],
                           "note": "Production encoder/context/decoder paths are used without modification."},
    }
    atomic_json(result, output / "results.json")
    boundary_by_count = {b["completed_cycles"]: b for b in source["boundaries"]}
    first_diagnostics = boundary_by_count[1]["diagnostics"]
    for case in cases:
        boundary = boundary_by_count[case.k]
        diag = boundary["diagnostics"] if case.mode == "natural" else first_diagnostics
        current = case.current
        fk = panda_observation_errors(current.proprio.numpy(), task="PickXtimes")
        if fk["position_error_m"] > 1e-4 or fk["rotation_error_rad"] > 1e-3:
            raise ValueError(f"Cannot plot invalid TCP kinematics: {fk}")
        result["current_scenes"][case.scene_id] = {
            "points": current.points.tolist(), "valid": current.valid.tolist(), "tcp": current.proprio[7:10].tolist(),
            "button": _position(diag, "button"), "cube": _position(diag, "cube"), "fk_check": fk,
        }
        result["case_audits"].append({"history_mode": case.mode, "k": case.k, "scene_id": case.scene_id,
            "history_frames": len(case.prefix), "padding_frames": case.padding_frames,
            "first_history_time": case.prefix[0].time, "last_history_time": case.prefix[-1].time,
            "current_time": current.time,
            "prefix_to_current_tcp_gap_mm": float(torch.linalg.vector_norm(current.proprio[7:10] - case.prefix[-1].proprio[7:10])*1000)})
    started = time.perf_counter()
    for label, path, _, method in inputs:
        policy, state = load_policy(path, device)
        policy.eval().requires_grad_(False)
        cfg = state["config"]
        precision = cfg["training"]["precision"]
        if policy.device.type == "cpu":
            precision = "fp32"
        elif precision == "auto":
            from stmem.doctor import choose_precision
            with torch.enable_grad():
                precision = choose_precision(policy.device)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        for case in cases:
            decision = Decision(case.current, [], case.prefix, torch.zeros(16, 8), torch.zeros(16, 8, dtype=torch.bool),
                                source["goals"]["3"], "PickXtimes", f"count_probe_{source['episode_idx']}", int(case.current.time))
            # Production encoders run outside decoder AMP; no expert targets enter encode/context.
            evidence = policy.encode(decision)
            selected = choose_records(evidence, method, 128)
            validate_frontier(selected.records, evidence.archive.leaves,
                              128 if method == "UNIFORM" else len(evidence.archive.leaves))
            scene = result["current_scenes"][case.scene_id]
            for x in valid_targets(case.k):
                paired_evidence = replace(evidence, text=policy.text_encoder(source["goals"][str(x)]))
                with torch.autocast(policy.device.type, dtype=dtype, enabled=precision != "fp32"):
                    context, bias = policy.context(paired_evidence, selected.records)
                    for seed in NOISE_SEEDS:
                        noise = torch.randn((16, 8), device=policy.device,
                                            generator=torch.Generator(device=policy.device).manual_seed(seed))
                        normalized = policy.decoder.integrate(context, bias, steps=cfg["model"]["flow_steps"], noise=noise.clone())
                        if seed == NOISE_SEEDS[0]:
                            repeated = policy.decoder.integrate(context, bias, steps=cfg["model"]["flow_steps"], noise=noise.clone())
                            if not torch.equal(normalized, repeated):
                                raise ValueError("Identical context/noise did not reproduce the same chunk")
                        raw = policy.normalizer.denormalize(normalized).float().cpu().numpy()
                        physical = np.stack([physical_action(a, "PickXtimes") for a in raw])
                        xyz = panda_tcp_positions(physical[:, :7], task="PickXtimes")
                        row = {"label": label, "method": method, "checkpoint_step": state["step"],
                               "history_mode": case.mode, "scene_id": case.scene_id, "k": case.k, "x": x,
                               "noise_seed": seed, "expected_next": "press_button" if x == case.k else "pick_cube",
                               "tcp_positions_m": xyz.tolist(), "physical_actions": physical.tolist(), "raw_actions": raw.tolist(),
                               "history_leaves": len(evidence.archive.leaves), "selected_history_tokens": len(selected.records),
                               "represented_leaf_count": sum(r.count for r in selected.records), "precision": precision,
                               "metrics": trajectory_metrics(xyz, scene["tcp"], scene["button"], scene["cube"])}
                        result["predictions"].append(row)
            del evidence, selected, context, bias, paired_evidence
            atomic_json(result, output / "results.json")
            print(json.dumps({"event": "count_probe_case", "label": label, "mode": case.mode, "k": case.k,
                              "predictions": len(result["predictions"]), "elapsed_seconds": time.perf_counter()-started}), flush=True)
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if file_sha256(path) != audits[label]["sha256"]:
            raise ValueError("Checkpoint changed while probing")
    if file_sha256(source_path) != result["source"]["sha256"]:
        raise ValueError("Expert source changed while probing")
    if file_sha256(__file__) != result["implementation"]["probe_sha256"]:
        raise ValueError("Probe implementation changed while running")
    for label in audits:
        unpadded = {row["noise_seed"]: row["raw_actions"] for row in result["predictions"]
                    if row["label"] == label and row["history_mode"] == "fixed_current" and row["k"] == 3}
        zero_padding = {row["noise_seed"]: row["raw_actions"] for row in result["predictions"]
                        if row["label"] == label and row["history_mode"] == "fixed_duration" and row["k"] == 3}
        if unpadded != zero_padding:
            raise ValueError("Zero-padding control changed identical context/noise predictions")
    result["zero_padding_control_bit_identical"] = True
    result.update(status="complete", elapsed_seconds=time.perf_counter()-started, summary=summarize(result["predictions"]))
    atomic_json(result, output / "results.json")
    from stmem.inspection.count_report import write_report
    report = write_report(result, output)
    print(json.dumps({"event": "complete", "report": str(report), "predictions": len(result["predictions"])}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--student", type=Path, help="Omit to run the teacher while the student is being transferred")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_probe(args.source, args.teacher, args.student, args.output, args.device)


if __name__ == "__main__":
    main()
