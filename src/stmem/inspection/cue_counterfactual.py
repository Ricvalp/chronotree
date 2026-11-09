"""Swap visible blue/red cue appearance, re-encode, and compare frozen policies.

This is a bounded counterfactual on one released training episode, not a rollout
or a benchmark success rate. Geometry, current input, text and noise stay fixed.
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch

from stmem.config import source_revision
from stmem.data.dataset import EpisodeDataset
from stmem.data.pointcloud import _backproject_with_pixels
from stmem.inspection.cue_frontiers import prioritized_frontier
from stmem.inspection.cue_recolor import canonical_point_rgb, swap_red_blue
from stmem.inspection.cue_refinement import load_inputs, observed_color_centers, paired_path_metrics
from stmem.inspection.gain_repeatability import SceneAssets
from stmem.memory.flat import select_flat
from stmem.evaluation.robomme_adapter import physical_action
from stmem.training.checkpoints import atomic_json
from stmem.vision.frozen import FrozenCLIPFeatures
from stmem.visualization.kinematics import panda_tcp_positions


def spatial_pixel_mask(observation, camera, intrinsic, centers, radius):
    points, uv = _backproject_with_pixels(
        observation[f"{camera}_rgb"], observation[f"{camera}_depth"], intrinsic,
        observation[f"{camera}_camera_extrinsic"], crop=None,
    )
    keep = (points[:, 2] > .005) & (points[:, 2] < .1)
    nearby = torch.zeros(len(points), dtype=torch.bool)
    for center in centers:
        nearby |= torch.linalg.vector_norm(points[:, :2] - torch.tensor(center[:2]), dim=-1) <= radius
    uv = uv[keep & nearby]
    mask = np.zeros(observation[f"{camera}_rgb"].shape[:2], dtype=bool)
    mask[uv[:, 1], uv[:, 0]] = True
    return mask


def image_item(rgb, label):
    stream = BytesIO()
    Image.fromarray(rgb).save(stream, format="PNG")
    return {"label": label, "data_url": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()}


def prepare_frames(episode, assets, extractor, regions, args):
    originals, swapped, illustrations, audits = [], [], [], []
    maximum_feature_delta, maximum_rgb_error = 0., 0.
    for frame in episode.conditioning:
        if frame.source_index % episode.observation_stride or frame.source_index > args.cue_end:
            originals.append(frame)
            swapped.append(frame)
            continue
        raw_index = int(frame.event_id.rsplit("/", 1)[-1])
        sensors = assets.raw["episode_0"][f"timestep_{raw_index}"]["obs"]
        observation = {name: np.asarray(sensors[name]) for name in (
            "front_rgb", "wrist_rgb", "front_depth", "wrist_depth",
            "front_camera_extrinsic", "wrist_camera_extrinsic")}
        edited = dict(observation)
        counts = {}
        for camera in ("front", "wrist"):
            intrinsic = np.asarray(assets.raw["episode_0"]["setup"][f"{camera}_camera_intrinsic"])
            mask = spatial_pixel_mask(observation, camera, intrinsic,
                                      [regions[c]["center"] for c in ("blue", "red")], args.radius)
            rgb, changed = swap_red_blue(observation[f"{camera}_rgb"], mask)
            recovered, _ = swap_red_blue(rgb, mask)
            if not np.array_equal(recovered, observation[f"{camera}_rgb"]):
                raise ValueError("Color-swap round trip changed the original image")
            edited[f"{camera}_rgb"] = rgb
            counts[camera] = int(changed.sum())
        original_rgb = canonical_point_rgb(observation, frame.point_camera, frame.point_uv, frame.valid)
        changed_rgb = canonical_point_rgb(edited, frame.point_camera, frame.point_uv, frame.valid)
        error = float((original_rgb[frame.valid] - frame.rgb[frame.valid]).abs().max())
        maximum_rgb_error = max(maximum_rgb_error, error)
        if error > 1e-7:
            raise ValueError("Original raw pixels disagree with canonical point RGB")
        # Recompute both arms with the same backbone/numerics. Descriptors of
        # every point are sampled from full edited maps, including surrounding
        # points whose CLIP receptive fields overlap a changed cube pixel.
        old_features = extractor.sample(observation, frame.point_camera, frame.point_uv, frame.valid).cpu()
        new_features = extractor.sample(edited, frame.point_camera, frame.point_uv, frame.valid).cpu()
        cached = frame.visual_features.to("cpu")
        maximum_feature_delta = max(maximum_feature_delta, float((cached.float() - old_features.float()).abs().max()))
        points = frame.points.clone()
        points[frame.valid, 3:] = changed_rgb[frame.valid]
        unchanged = replace(frame, visual_features=old_features)
        changed_frame = replace(frame, points=points, visual_features=new_features)
        assert torch.equal(unchanged.xyz, changed_frame.xyz)
        assert unchanged.valid is changed_frame.valid
        assert unchanged.center_indices is changed_frame.center_indices
        originals.append(unchanged)
        swapped.append(changed_frame)
        audits.append({"source_index": frame.source_index, "raw_index": raw_index,
                       "camera_changed_pixels": counts,
                       "changed_point_rgb": int((original_rgb != changed_rgb).any(-1).sum()),
                       "changed_point_descriptors": int((old_features != new_features).any(-1).sum()),
                       "descriptor_max_abs_change": float((old_features.float() - new_features.float()).abs().max())})
        if frame.source_index in (0, 16, 30):
            for label, view, selected_frame in (("original", observation, unchanged), ("swapped", edited, changed_frame)):
                illustration = dict(assets.frame("episode_0", "conditioning", frame.source_index))
                illustration["label"] = f"{label}: reveal frame {frame.source_index}"
                illustration["points"] = selected_frame.points[selected_frame.valid].numpy().tolist()
                illustration["images"] = [image_item(view[f"{camera}_rgb"], f"{label} {camera}") for camera in ("front", "wrist")]
                illustrations.append(illustration)
    if not audits or any(not all(v > 0 for v in row["camera_changed_pixels"].values()) for row in audits):
        raise ValueError("Every retained reveal frame must contain edited pixels in both cameras")
    # Every subsequent and current observation is shared by object identity.
    return replace(episode, conditioning=originals), replace(episode, conditioning=swapped), illustrations, {
        "frames": audits, "original_rgb_max_abs_error": maximum_rgb_error,
        "cached_vs_recomputed_descriptor_max_abs_error": maximum_feature_delta,
        "color_round_trip_exact": True, "geometry_masks_centers_unchanged": True,
    }


def archive_index(archive):
    records = archive.records
    lookup = {node.node_id: i for i, node in enumerate(records)}
    children = np.asarray([[lookup[c.node_id] for c in n.children] if n.children else [-1, -1] for n in records])
    return records, {"children": children, "node_id": np.asarray([n.node_id for n in records]),
                     "count": np.asarray([n.count for n in records])}, np.asarray([lookup[n.node_id] for n in archive.roots])


def selection_ids(evidence, regions, budget, cue_end, radius):
    records, index, roots = archive_index(evidence.archive)
    protected, stack = [], list(map(int, roots))
    while stack:
        row = stack.pop()
        n = records[row]
        if n.source == "conditioning" and n.time_max <= cue_end:
            protected.append(row)
        elif n.children:
            stack.extend(index["children"][row])
    preferred = []
    for row, n in enumerate(records):
        if not n.is_leaf or n.source != "conditioning" or n.time_anchor > cue_end or n.anchor[2] >= .1:
            continue
        if any(float(torch.linalg.vector_norm(n.anchor[:2].cpu() - torch.tensor(regions[c]["center"][:2]))) <= radius
               for c in ("blue", "red")):
            preferred.append(row)
    chosen = {
        "uniform": prioritized_frontier(index, roots, budget),
        "coarse": prioritized_frontier(index, roots, budget, protected_nodes=protected),
        "both_cues": prioritized_frontier(index, roots, budget, preferred_leaves=preferred),
    }
    return {name: [records[i].node_id for i in rows] for name, rows in chosen.items()}, {
        "fine_cue_candidates": len(preferred),
        "fine_cue_node_ids": [records[row].node_id for row in preferred],
        "protected_node_ids": [records[i].node_id for i in protected],
        "exposed_fine_cue_leaves": {name: len(set(map(int, rows)) & set(preferred)) for name, rows in chosen.items()},
    }


@torch.inference_mode()
def decode(policy, evidence, records, seeds):
    context, bias = policy.context(evidence, records)
    values = []
    for seed in seeds:
        noise = torch.randn((1, policy.config["model"]["horizon"], 8), device=policy.device,
                            generator=torch.Generator(device=policy.device).manual_seed(seed))
        values.append(policy.decoder.integrate(context[None], bias[None], steps=8, noise=noise)[0].float().cpu())
    normalized = torch.stack(values)
    physical = policy.normalizer.denormalize(normalized.to(policy.device)).cpu().numpy()
    physical = np.asarray([physical_action(a, "VideoUnmask") for a in physical.reshape(-1, 8)]).reshape(physical.shape)
    return normalized.numpy(), physical, panda_tcp_positions(physical[..., :7], task="VideoUnmask")


@torch.inference_mode()
def run(args):
    out = Path(args.output)
    if (out / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    teacher, compact, _, _, cfg, provenance = load_inputs(args)
    cfg["model"]["visual_features"]["checkpoint"] = str(Path(args.clip).resolve())
    extractor = FrozenCLIPFeatures(cfg, device=args.device)
    dataset = EpisodeDataset(Path(args.prepared).parent, tasks=["VideoUnmask"],
        horizon=cfg["model"]["horizon"], alignment_path=args.alignment, visual_features_config=cfg)
    dataset.validate_preprocessing(cfg)
    episode = dataset.load_episode(next(i for i, ref in enumerate(dataset.episodes) if ref.key == "episode_0"))
    seeds = [910001 + 1009 * i for i in range(args.draws)]
    assets = SceneAssets(args.prepared, args.raw)
    started = time.perf_counter()
    try:
        regions = observed_color_centers(assets.frame("episode_0", "conditioning", 0))
        original, swapped, illustrations, checks = prepare_frames(episode, assets, extractor, regions, args)
        del extractor
        protocol = {**provenance, "scope": "One real VideoUnmask training scene with edited cue colors; offline counterfactual, not closed-loop success",
            "title": "Does changing the remembered target change the action?",
            "intervention_description": "Blue and red cube appearance is swapped in BOTH cameras throughout the retained reveal video. CLIP features and historical tokens are recomputed. Current observation, geometry, task, selected node IDs and action noise remain fixed. TCP paths are joint-target kinematics, not simulated motion.",
            "default_comparison": {"a": "teacher_original", "b": "teacher_swapped"},
            "expert_label": "Expert: original history only",
            "noise_seeds": seeds, "budget": args.budgets[0], "decision_indices": args.decisions,
            "integration_steps": 8, "precision": "fp32_no_tf32", "regions": regions,
            "source": source_revision(),
            "new_target": "original red cube region", "cue_end": args.cue_end, "radius_m": args.radius,
            "clip_sha256": "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762",
            "notes": ["The task still asks for the blue cube; after the edit blue is at the original red location.",
                "All retained reveal frames are edited, not a single frame. Later occluded frames and current input are unchanged.",
                "Saturated red/blue pixel masks are diagnostic edits only, never supplied to the models.",
                "Full-image CLIP maps are recomputed so receptive-field effects also reach surrounding sampled points.",
                "Compact selection is fixed before observing predictions and uses identical node IDs in both arms.",
                "Original expert targets are only a geometric reference; they are not valid supervision for the edited target.",
                "At reset there is no expert execution prefix. Later decisions, if requested, retain original expert approach and are less conclusive.",
                "A chunk endpoint nearer a container is only a directional proxy, not a grasp/success determination."]}
        result = {"protocol": protocol, "checks": checks, "scenes": []}
        atomic_json(protocol, out / "manifest.json")
        for decision_index in args.decisions:
            decision = original.decision(decision_index, cfg["model"]["horizon"])
            alternate = swapped.decision(decision_index, cfg["model"]["horizon"])
            assert decision.current is alternate.current and decision.goal == alternate.goal
            assert all(a is b for a, b in zip(decision.execution_prefix, alternate.execution_prefix))
            before = teacher.encode(decision)
            after = teacher.encode(alternate, current_records=before.current_records, text=before.text)
            by_id = {n.node_id: n for n in after.archive.records}
            if set(by_id) != {n.node_id for n in before.archive.records}:
                raise ValueError("Color-only edit changed hierarchy topology or node IDs")
            for n in before.archive.records:
                m = by_id[n.node_id]
                if (n.count != m.count or not torch.equal(n.anchor, m.anchor)
                        or not torch.equal(n.bounds_min, m.bounds_min) or not torch.equal(n.bounds_max, m.bounds_max)
                        or n.time_min != m.time_min or n.time_max != m.time_max):
                    raise ValueError("Color edit changed structural metadata")
            leaves = before.archive.leaves
            leaf_before = torch.stack([n.content for n in leaves]).float()
            leaf_after = torch.stack([by_id[n.node_id].content for n in leaves]).float()
            leaf_delta = leaf_after - leaf_before
            cue_mask = torch.tensor([n.source == "conditioning" and n.time_anchor <= args.cue_end
                                     for n in leaves], device=leaf_delta.device)
            changed_leaves = leaf_delta.abs().amax(-1) > 0
            if changed_leaves[~cue_mask].any():
                raise ValueError("Appearance edit changed leaves outside the complete reveal blocks")
            if not changed_leaves[cue_mask].any():
                raise ValueError("Appearance edit did not reach encoded cue content")
            leaf_check = {
                "cue_leaves": int(cue_mask.sum()), "changed_cue_leaves": int(changed_leaves[cue_mask].sum()),
                "changed_other_leaves": int(changed_leaves[~cue_mask].sum()),
                "cue_mean_content_l2_change": float(leaf_delta[cue_mask].norm(dim=-1).mean()),
                "cue_max_abs_content_change": float(leaf_delta[cue_mask].abs().max()),
                "cue_relative_content_l2_change": float(leaf_delta[cue_mask].norm() / leaf_before[cue_mask].norm().clamp_min(1e-12)),
            }
            ids, selection = selection_ids(before, regions, args.budgets[0], args.cue_end, args.radius)
            lookup = {n.node_id: n for n in before.archive.records}
            original_records = {key: [lookup[n] for n in names] for key, names in ids.items()}
            swapped_records = {key: [by_id[n] for n in names] for key, names in ids.items()}
            original_records["teacher"], swapped_records["teacher"] = before.archive.leaves, after.archive.leaves
            # A frozen random FLAT choice is another compact-reader control.
            flat_rng = random.Random(2026)
            flat_scores = {n.node_id: flat_rng.random() for n in before.archive.leaves}
            score = lambda records: [flat_scores[n.node_id] for n in records]
            original_records["flat"] = select_flat(before.archive.leaves, args.budgets[0], score).records
            swapped_records["flat"] = select_flat(after.archive.leaves, args.budgets[0], score).records
            if [n.node_id for n in original_records["flat"]] != [n.node_id for n in swapped_records["flat"]]:
                raise ValueError("Paired random flat selection changed")
            scene = {"id": f"decision{decision_index:03d}_B{args.budgets[0]}", "decision_index": decision_index,
                "goal": decision.goal, "current": assets.frame("episode_0", "execution", decision_index),
                "cue_frames": illustrations, "variants": [], "comparisons": [], "selection": selection,
                "encoded_evidence_checks": leaf_check,
                "target_regions": [{"label": "Original target (blue before)", "center": regions["blue"]["center"]},
                                   {"label": "New target (blue after)", "center": regions["red"]["center"]}],
                "expert_tcp": panda_tcp_positions(decision.target_chunk[:, :7], task="VideoUnmask").tolist(),
                "expert_valid": decision.target_valid.tolist()}
            outputs = {}
            for reader in ("teacher", "uniform", "coarse", "both_cues", "flat"):
                model = teacher if reader == "teacher" else compact
                for arm, evidence, records in (("original", before, original_records[reader]), ("swapped", after, swapped_records[reader])):
                    normalized, actions, tcp = decode(model, evidence, records, seeds)
                    name = f"{reader}_{arm}"
                    outputs[name] = normalized, tcp
                    scene["variants"].append({"name": name, "description": f"{reader} / {arm} reveal colors; fixed geometry/current/selection/noise",
                        "tokens": len(records), "selected_nodes": [], "actions": actions.tolist(), "tcp_paths": tcp.tolist(),
                        "normalized_actions": normalized.tolist(), "preferred_leaf_count": selection["fine_cue_candidates"],
                        "exposed_preferred_leaves": len({n.node_id for n in records} & set(selection["fine_cue_node_ids"])),
                        "metrics": {"old_target_endpoint_xy_cm": (100*np.linalg.norm(tcp[:, -1, :2]-np.asarray(regions["blue"]["center"][:2]), axis=-1)).tolist(),
                                    "new_target_endpoint_xy_cm": (100*np.linalg.norm(tcp[:, -1, :2]-np.asarray(regions["red"]["center"][:2]), axis=-1)).tolist()}})
                a, b = f"{reader}_original", f"{reader}_swapped"
                scene["comparisons"].append({"a": a, "b": b,
                    **paired_path_metrics(outputs[a][1], outputs[b][1], outputs[a][0], outputs[b][0],
                                          decision.target_valid, np.asarray(regions["red"]["center"][:2]))})
            if decision_index == args.decisions[0]:
                cached_decision = episode.decision(decision_index, cfg["model"]["horizon"])
                cached_evidence = teacher.encode(cached_decision, current_records=before.current_records, text=before.text)
                cached_normalized, _, cached_tcp = decode(teacher, cached_evidence, cached_evidence.archive.leaves, seeds[:1])
                checks["cached_vs_recomputed_original_teacher_max_abs_action"] = float(np.abs(cached_normalized[0]-outputs["teacher_original"][0][0]).max())
                checks["cached_vs_recomputed_original_teacher_max_tcp_mm"] = float(1000*np.linalg.norm(cached_tcp[0]-outputs["teacher_original"][1][0], axis=-1).max())
                repeat, _, _ = decode(teacher, before, original_records["teacher"], seeds[:1])
                checks["identical_input_max_abs_error"] = float(np.abs(repeat[0]-outputs["teacher_original"][0][0]).max())
            result["scenes"].append(scene)
            atomic_json(scene, out / "scenes" / f"{scene['id']}.json")
            print(json.dumps({"event": "counterfactual_complete", "decision": decision_index, "seconds": time.perf_counter()-started}), flush=True)
    finally:
        assets.close()
    result["summary"] = {s["id"]: [{"a": p["a"], "b": p["b"], **{k: float(np.mean(v)) for k,v in p.items() if k not in ("a", "b")}}
                                   for p in s["comparisons"]] for s in result["scenes"]}
    result["elapsed_seconds"] = time.perf_counter()-started
    atomic_json(result, out / "results.json")
    atomic_json({"comparisons": result["summary"], "checks": checks}, out / "summary.json")
    from stmem.inspection.cue_action_report import write_report
    print(write_report(result, out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, value in {
        "teacher": "runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt",
        "compact": "runs/inspection/router_gain_audit/compact_90000.pt",
        "prepared": "data/inspection/VideoUnmask.h5", "raw": "data/raw/record_dataset_VideoUnmask.h5",
        "features": "data/inspection/features", "alignment": "data/alignment.json",
        "encoded": "runs/hierarchy_cache_diagnostic_20261103/encoded",
        "hierarchy": "runs/hierarchy_cache_diagnostic_20261103/hierarchy",
        "clip": "cache/backbones/clip_RN50.pt", "device": "cuda:0",
    }.items():
        parser.add_argument("--"+name, default=value)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", nargs=1, type=int, default=[128])
    parser.add_argument("--decisions", nargs="+", type=int, default=[0])
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--cue-end", type=int, default=30)
    parser.add_argument("--radius", type=float, default=.07)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.draws < 2 or args.radius <= 0 or args.budgets[0] < 1:
        parser.error("Require positive budget/radius and at least two paired draws")
    if args.report_only:
        from stmem.inspection.cue_action_report import write_report
        print(write_report(json.loads((Path(args.output)/"results.json").read_text()), args.output))
    else:
        run(args)


if __name__ == "__main__":
    main()
