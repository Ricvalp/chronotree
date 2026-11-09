"""Bounded, observed-cue refinement diagnostic on a real VideoUnmask prefix.

Selection uses manually declared RGB-visible regions, never action predictions.
All matched frontiers retain every causal leaf exactly once. This is an offline
mechanism probe, not a learned reader, object segmentation, or success estimate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from stmem.config import source_revision
from stmem.data.cache import cache_identity, episode_key
from stmem.data.hierarchy_cache import validate_hierarchy_cache
from stmem.evaluation.robomme_adapter import physical_action
from stmem.inspection.cue_frontiers import prioritized_frontier, frontier_leaf_rows
from stmem.inspection.gain_repeatability import SceneAssets
from stmem.models.action_flow import masked_mse
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.label_data import LabelDataReader
from stmem.training.label_experiment import validate_models
from stmem.training.label_objectives import policy_context
from stmem.training.packed_context import columns_bank, _empty_columns
from stmem.training.runtime import load_policy
from stmem.visualization.kinematics import panda_tcp_positions


DESCRIPTIONS = {
    "uniform": "Uniform largest-region refinement; all history remains covered.",
    "cue_coarse": "Keep the entire reveal interval in its coarse summary; spend the same budget elsewhere.",
    "cue_target": "Prioritize fine anchors near the blue target during its visible reveal interval.",
    "cue_distractor": "Prioritize fine anchors near the red distractor during the same reveal interval.",
    "late_target": "Prioritize the target location AFTER occlusion; the cube is no longer visible.",
    "cue_all": "Prioritize all leaves from the reveal interval, without spatial targeting.",
    "teacher": "Full-history teacher; separate weights, reference only.",
    "current_only": "Compact policy with history removed; separate diagnostic, not a matched coverage arm.",
}


def observed_color_centers(frame):
    """Visualization-only color regions from the observed prepared cloud."""
    points = np.asarray(frame["points"])
    rgb = points[:, 3:6]
    if rgb.max() > 1.01:
        rgb = rgb / 255.0
    masks = {"blue": (rgb[:, 2] > .25) & (rgb[:, 2] > rgb[:, 0] * 1.5)
                       & (rgb[:, 2] > rgb[:, 1] * 1.5),
             "red": (rgb[:, 0] > .25) & (rgb[:, 0] > rgb[:, 1] * 1.5)
                      & (rgb[:, 0] > rgb[:, 2] * 1.5)}
    result = {}
    for color, mask in masks.items():
        # Keep the declared tabletop volume; not simulator poses or masks.
        mask &= (points[:, 2] > .005) & (points[:, 2] < .09)
        if int(mask.sum()) < 5:
            raise ValueError(f"Cannot establish {color} cue from at least five observed points")
        result[color] = {"center": np.median(points[mask, :3], axis=0).tolist(),
                         "points": int(mask.sum())}
    return result


def paired_path_metrics(a, b, normalized_a, normalized_b, valid, target_xy):
    """Paired changes across the full chunk, in metres-to-cm and normalized units."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError("Paired paths must have the same [draw,H,3] shape")
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != normalized_a.shape[1:] or normalized_a.shape != normalized_b.shape:
        raise ValueError("Action validity/shape mismatch")
    arm_valid = valid[:, :7].all(axis=-1)
    if not arm_valid.any():
        raise ValueError("No valid arm targets for path comparison")
    end = np.flatnonzero(arm_valid)[-1]
    distances = np.linalg.norm(a - b, axis=-1) * 100
    error = (normalized_a - normalized_b) ** 2
    return {
        "mean_tcp_delta_cm": distances[:, arm_valid].mean(axis=1).tolist(),
        "max_tcp_delta_cm": distances[:, arm_valid].max(axis=1).tolist(),
        "endpoint_delta_cm": distances[:, end].tolist(),
        "normalized_action_rmse": np.sqrt((error * valid).sum(axis=(1, 2)) / valid.sum()).tolist(),
        "endpoint_target_distance_a_cm": (100 * np.linalg.norm(a[:, end, :2] - target_xy, axis=-1)).tolist(),
        "endpoint_target_distance_b_cm": (100 * np.linalg.norm(b[:, end, :2] - target_xy, axis=-1)).tolist(),
    }


def load_inputs(args):
    teacher, teacher_state = load_policy(args.teacher, args.device)
    compact, compact_state = load_policy(args.compact, args.device)
    for model in (teacher, compact):
        model.freeze_encoders()
        model.eval().requires_grad_(False)
    teacher_hash, compact_hash = file_sha256(args.teacher), file_sha256(args.compact)
    if compact_state["provenance"]["dependencies"]["teacher"] != teacher_hash:
        raise ValueError("Compact checkpoint does not depend on the supplied teacher")
    cfg = deepcopy(compact_state["config"])
    # The prepared input can be a symlink; its parent/name form the public path.
    cfg["paths"]["data_root"] = str(Path(args.prepared).absolute().parent.parent)
    cfg["data"].update(prepared_subdir=Path(args.prepared).parent.name,
                       tasks=["VideoUnmask"], episodes_per_task=2, fraction=1.,
                       alignment_path=str(Path(args.alignment).resolve()))
    cfg["model"]["visual_features"]["cache_dir"] = str(Path(args.features).resolve())
    cfg["labels"] = {"contexts": 2, "budgets": args.budgets, "seed": 2026,
                     "split_fraction": .8, "candidates": 4, "exploration_fraction": .5}
    validate_models(cfg, teacher, teacher_state, compact, compact_state)
    manifest = json.loads((Path(args.encoded) / "manifest.json").read_text())
    if manifest["identity"]["teacher_sha256"] != teacher_hash:
        raise ValueError("Encoded cache has different teacher weights")
    expected = cache_identity(cfg, args.teacher, manifest["identity"]["data_identity"])
    for key in ("format", "geometry", "model_encoder", "normalization_version",
                "frozen_encoder_precision", "prefix_convention", "code"):
        if expected[key] != manifest["identity"][key]:
            raise ValueError(f"Incompatible encoded cache: {key}")
    for key in ("observation_stride", "decision_stride", "clock", "benchmark_revision", "simulator_revision"):
        if manifest["identity"]["data"].get(key) != cfg["data"].get(key):
            raise ValueError(f"Incompatible data cadence/revision: {key}")
    prepared_hash = file_sha256(args.prepared)
    if manifest.get("kind") == "bounded_real_episode_diagnostic_not_scientific_training_cache":
        receipt = json.loads((Path(args.encoded).resolve().parent / "benchmark.json").read_text())
        if prepared_hash not in receipt["provenance"]["inputs"].values():
            raise ValueError("Prepared bytes differ from the encoded cache's construction receipt")
    hierarchy = validate_hierarchy_cache(args.encoded, args.hierarchy)
    for root, record in ((args.encoded, manifest), (args.hierarchy, hierarchy)):
        for shard, entry in record["shards"].items():
            if file_sha256(Path(root) / shard) != entry["sha256"]:
                raise ValueError(f"Changed cache shard: {root}/{shard}")
    return teacher, compact, teacher_state, compact_state, cfg, {
        "teacher": {"step": teacher_state["step"], "sha256": teacher_hash, "path": str(args.teacher)},
        "compact": {"step": compact_state["step"], "sha256": compact_hash, "path": str(args.compact)},
        "prepared_sha256": prepared_hash,
        "encoded_manifest_sha256": file_sha256(Path(args.encoded) / "manifest.json"),
        "hierarchy_manifest_sha256": file_sha256(Path(args.hierarchy) / "manifest.json"),
        "alignment_sha256": file_sha256(args.alignment),
    }


def select_variants(index, nodes, roots, budget, regions, cue_end, radius):
    leaves = frontier_leaf_rows(index, roots)
    anchors = np.asarray(nodes["anchor"])[leaves]
    times = np.asarray(nodes["time_anchor"])[leaves]
    conditioning = index["source"][leaves] == "conditioning"
    early = conditioning & (times <= cue_end)
    late = conditioning & (times > cue_end)
    target = np.linalg.norm(anchors[:, :2] - np.asarray(regions["blue"]["center"][:2]), axis=-1) <= radius
    distractor = np.linalg.norm(anchors[:, :2] - np.asarray(regions["red"]["center"][:2]), axis=-1) <= radius
    target &= anchors[:, 2] < .1
    distractor &= anchors[:, 2] < .1
    preferred = {"uniform": [], "cue_target": leaves[early & target],
                 "cue_distractor": leaves[early & distractor], "late_target": leaves[late & target],
                 "cue_all": leaves[early]}
    for name, chosen in preferred.items():
        if name != "uniform" and not len(chosen):
            raise ValueError(f"No leaves satisfy the predeclared {name} region")
    # Find maximal active subtrees entirely in the reveal interval. Protect
    # exactly those, not a temporal root that also contains post-occlusion data.
    lower, upper = np.asarray(nodes["time_min"]), np.asarray(nodes["time_max"])
    protected, stack = [], list(map(int, roots))
    while stack:
        row = stack.pop()
        if index["source"][row] == "conditioning" and lower[row] >= 0 and upper[row] <= cue_end:
            protected.append(row)
        elif index["children"][row, 0] >= 0:
            stack.extend(map(int, index["children"][row]))
    rows = {name: prioritized_frontier(index, roots, budget, chosen)
            for name, chosen in preferred.items()}
    rows["cue_coarse"] = prioritized_frontier(index, roots, budget, protected_nodes=protected)
    preferred["cue_coarse"] = []
    return rows, preferred, leaves[early & target], protected


@torch.inference_mode()
def decode(policy, item, bank, seeds):
    context, bias = policy_context(policy, [item], [bank])
    normalized = []
    for seed in seeds:
        # One fixed shape per draw avoids padding/batch-size numerical effects.
        noise = torch.randn((1, *item["target"].shape), device=policy.device,
                            generator=torch.Generator(device=policy.device).manual_seed(seed))
        normalized.append(policy.decoder.integrate(context, bias, steps=8, noise=noise)[0].float().cpu())
    normalized = torch.stack(normalized)
    action = policy.normalizer.denormalize(normalized.to(policy.device)).cpu().numpy()
    physical = np.asarray([physical_action(a, "VideoUnmask") for a in action.reshape(-1, 8)]).reshape(action.shape)
    return normalized, physical, panda_tcp_positions(physical[..., :7], task="VideoUnmask")


def run(args):
    out = Path(args.output)
    if (out / "manifest.json").exists():
        raise FileExistsError("Use a new output directory to preserve measured results")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    teacher, compact, _, _, cfg, provenance = load_inputs(args)
    seeds = [910001 + 1009 * i for i in range(args.draws)]
    assets = SceneAssets(args.prepared, args.raw)
    cue_first = assets.frame("episode_0", "conditioning", 0)
    regions = observed_color_centers(cue_first)
    data = TrainingData(cfg)
    protocol = {**provenance, "scope": "One real VideoUnmask training demonstration; offline causal refinement diagnostic, not success rate",
                "noise_seeds": seeds, "decisions": args.decisions, "budgets": args.budgets,
                "cue_end_source_index": args.cue_end, "spatial_radius_m": args.radius,
                "region_max_height_m": .1,
                "observed_regions": regions, "precision": "fp32_no_tf32", "integration_steps": 8,
                "source": source_revision(), "notes": [
                    "Regions and reveal interval were fixed from observed RGB before new action comparisons.",
                    "Matched frontier arms have identical token count and exact causal leaf coverage.",
                    "Coarse means summarized, not erased. Other nodes may retain cue information.",
                    "Spatial anchors are not object masks: frozen CLIP and local neighborhood pooling mix evidence.",
                    "Teacher and current-only are separate reference diagnostics, not equal-budget reader arms.",
                    "TCP paths are forward kinematics of clipped joint targets, not executed robot trajectories.",
                    "All noise draws are retained. No outcome-based selection of a favorable seed."]}
    result = {"protocol": protocol, "scenes": []}
    atomic_json(protocol, out / "manifest.json")
    started = time.perf_counter()
    try:
        with LabelDataReader(data, cfg, args.encoded, args.hierarchy) as reader:
            group = reader.packed.hierarchy_files.get(Path(args.hierarchy) / "VideoUnmask.h5")[episode_key("VideoUnmask", "episode_0")]
            index, nodes = reader.packed._index(group), group["nodes"]
            for decision in args.decisions:
                if str(decision) not in group["decisions"] or (0, decision) not in data.candidates["VideoUnmask"]:
                    raise ValueError(f"Unavailable causal decision: {decision}")
                for budget in args.budgets:
                    item = reader.prepare({"context_id": 0, "episode_index": 0, "index": decision,
                                           "budget": budget, "seed": 2026, "time_choices": 1})
                    roots = np.asarray(group["decisions"][str(decision)]["roots"])
                    rows, preferred, target_leaves, protected = select_variants(
                        index, nodes, roots, budget, regions, args.cue_end, args.radius)
                    scene = {"id": f"decision{decision:03d}_B{budget}", "decision_index": decision,
                             "budget": budget, "goal": str(assets.prepared["episode_0"].attrs["goal"]),
                             "current": assets.frame("episode_0", "execution", decision),
                             "cue_frames": [assets.frame("episode_0", "conditioning", t) for t in (0, 16, 30, 32, 64)],
                             "expert_tcp": panda_tcp_positions(item["target"][:, :7], task="VideoUnmask").tolist(),
                             "expert_valid": item["valid"].tolist(), "variants": [], "comparisons": [],
                             "protected_cue_node_ids": [str(index["node_id"][row]) for row in protected]}
                    outputs, paths = {}, {}
                    for name in (*DESCRIPTIONS,):
                        if name == "teacher":
                            model, bank, selected = teacher, item["full"], []
                        elif name == "current_only":
                            model, bank, selected = compact, columns_bank(_empty_columns(cfg["model"]["width"]), item["clocks"], cfg["model"]["width"]), []
                        else:
                            model, selected = compact, rows[name]
                            bank = columns_bank(reader.packed._columns(nodes, selected, index=index), item["clocks"], cfg["model"]["width"])
                        normalized, actions, tcp = decode(model, item, bank, seeds)
                        outputs[name], paths[name] = normalized.numpy(), tcp
                        expert = model.normalizer.normalize(item["target"].to(model.device)).cpu().expand(len(seeds), -1, -1)
                        error = masked_mse(normalized, expert, item["valid"].expand(len(seeds), -1, -1), reduction="none")
                        geometry = []
                        for row in selected:
                            geometry.append({"id": str(index["node_id"][row]), "source": str(index["source"][row]),
                                **{key: np.asarray(nodes[key][row]).tolist() for key in
                                   ("anchor", "bounds_min", "bounds_max", "time_anchor", "time_min", "time_max", "count")}})
                        scene["variants"].append({"name": name, "description": DESCRIPTIONS[name],
                            "tokens": int(bank.content.shape[1]), "preferred_leaf_count": len(preferred.get(name, [])),
                            "exposed_preferred_leaves": len(set(map(int, selected)) & set(map(int, preferred.get(name, [])))),
                            "exposed_target_cue_leaves": (len(target_leaves) if name == "teacher" else
                                len(set(map(int, selected)) & set(map(int, target_leaves)))),
                            "selected_nodes": geometry, "actions": actions.tolist(), "tcp_paths": tcp.tolist(),
                            "metrics": {"expert_normalized_action_mse": error.tolist()},
                            "normalized_actions": normalized.tolist()})
                    for base in ("uniform", "cue_coarse"):
                        for name in DESCRIPTIONS:
                            if name != base:
                                scene["comparisons"].append({"a": base, "b": name,
                                    **paired_path_metrics(paths[base], paths[name], outputs[base], outputs[name],
                                                          item["valid"], np.asarray(regions["blue"]["center"][:2]))})
                    # Replay fixed inputs exactly once as a numerical control.
                    if not result["scenes"]:
                        selected = rows["uniform"]
                        bank = columns_bank(reader.packed._columns(nodes, selected, index=index), item["clocks"], cfg["model"]["width"])
                        repeated, _, _ = decode(compact, item, bank, seeds[:1])
                        result["checks"] = {"identical_context_max_abs_error": float(np.max(np.abs(repeated.numpy()[0] - outputs["uniform"][0])))}
                    result["scenes"].append(scene)
                    atomic_json(scene, out / "scenes" / f"{scene['id']}.json")
                    print(json.dumps({"event": "scene_complete", "scene": scene["id"], "seconds": time.perf_counter()-started}), flush=True)
    finally:
        assets.close()
    result["elapsed_seconds"] = time.perf_counter() - started
    result["summary"] = {scene["id"]: [{"a": pair["a"], "b": pair["b"],
        **{key: float(np.mean(value)) for key, value in pair.items() if key not in ("a", "b")}}
        for pair in scene["comparisons"]] for scene in result["scenes"]}
    atomic_json(result, out / "results.json")
    atomic_json({"comparisons": result["summary"], "checks": result["checks"]}, out / "summary.json")
    from stmem.inspection.cue_action_report import write_report
    print(write_report(result, out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt")
    parser.add_argument("--compact", default="runs/inspection/router_gain_audit/compact_90000.pt")
    parser.add_argument("--prepared", default="data/inspection/VideoUnmask.h5")
    parser.add_argument("--raw", default="data/raw/record_dataset_VideoUnmask.h5")
    parser.add_argument("--features", default="data/inspection/features")
    parser.add_argument("--alignment", default="data/alignment.json")
    parser.add_argument("--encoded", default="runs/hierarchy_cache_diagnostic_20261103/encoded")
    parser.add_argument("--hierarchy", default="runs/hierarchy_cache_diagnostic_20261103/hierarchy")
    parser.add_argument("--decisions", nargs="+", type=int, default=[0, 20, 64])
    parser.add_argument("--budgets", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--cue-end", type=int, default=30)
    parser.add_argument("--radius", type=float, default=.07)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.draws < 2 or args.radius <= 0 or not all(b > 0 for b in args.budgets):
        parser.error("Require at least two draws, positive radius and positive budgets")
    if args.report_only:
        from stmem.inspection.cue_action_report import write_report
        print(write_report(json.loads((Path(args.output)/"results.json").read_text()), args.output))
    else:
        run(args)


if __name__ == "__main__":
    main()
