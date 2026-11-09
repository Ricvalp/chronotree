"""Local fixed-context gain repeatability and visual evidence diagnostic.

Uses real released demonstrations and frozen policies, never optimizes a model.
The noise varies only AFTER causal contexts and candidate moves have been fixed.
No simulator annotations enter policy inputs or candidate selection.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image
import torch

from stmem.config import source_revision
from stmem.data.cache import cache_identity, episode_key
from stmem.data.hierarchy_cache import validate_hierarchy_cache
from stmem.evaluation.robomme_adapter import physical_action
from stmem.inspection.gain_statistics import summarize_experiment, summarize_gains
from stmem.models.action_flow import masked_mse
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.label_data import LabelDataReader, LabelPlan
from stmem.training.label_experiment import validate_models
from stmem.training.label_objectives import evaluate_batch, policy_context
from stmem.training.runtime import load_policy
from stmem.visualization.kinematics import panda_tcp_positions


def fixed_noise_items(prepared, seeds, *, noop=True):
    """Shallow copies preserve every tensor and selection; only noise seed varies."""
    readers = {}
    for name, reader in prepared["readers"].items():
        readers[name] = dict(reader)
        if noop and reader["after"]:
            # An explicitly separate numerical control, not a valid refinement.
            readers[name]["after"] = [*reader["after"], reader["before"]]
            readers[name]["costs"] = torch.cat((reader["costs"], torch.ones(1)))
    return [{**prepared, "seed": int(seed), "readers": readers} for seed in seeds]


def _values(value):
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value.item() if isinstance(value, np.generic) else value


class SceneAssets:
    def __init__(self, prepared, raw):
        self.prepared = h5py.File(prepared, "r")
        self.raw = h5py.File(raw, "r")
        self.frames = {}

    def close(self):
        self.prepared.close()
        self.raw.close()

    def frame(self, episode, source, source_index):
        key = (episode, source, int(source_index))
        if key in self.frames:
            return self.frames[key]
        group = self.prepared[episode]
        rows = np.flatnonzero((np.asarray(group["source"]) == (source == "conditioning")) &
                             (np.asarray(group["source_index"]) == source_index))
        if len(rows) != 1:
            raise ValueError(f"Source frame is not unique: {key}")
        row = int(rows[0])
        points = np.asarray(group["points"][row])[np.asarray(group["valid"][row], dtype=bool)]
        raw_index = int(group["raw_index"][row])
        images = []
        for camera in ("front", "wrist"):
            rgb = np.asarray(self.raw[episode][f"timestep_{raw_index}"]["obs"][f"{camera}_rgb"])
            buf = BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
            images.append({"label": camera, "data_url": "data:image/jpeg;base64," +
                           base64.b64encode(buf.getvalue()).decode()})
        proprio = np.asarray(group["proprio"][row])
        tcp = panda_tcp_positions(proprio[:7], task="VideoUnmask")
        if not np.allclose(tcp, proprio[7:10], atol=1e-4):
            raise ValueError("Diagnostic FK disagrees with observed TCP")
        result = {"source": source, "source_index": int(source_index), "raw_index": raw_index,
                  "points": np.round(points, 5).tolist(), "images": images, "tcp": tcp.tolist()}
        self.frames[key] = result
        return result


def _record(nodes, row):
    names = ("anchor", "bounds_min", "bounds_max", "time_min", "time_max", "time_anchor",
             "count", "depth", "source", "modality", "node_id")
    result = {key: _values(nodes[key][row]) for key in names}
    return {key: (v.decode() if isinstance(v, bytes) else v) for key, v in result.items()}


def candidate_visuals(prepared, name, reader, assets):
    task, episode = prepared["episode"].split("/")
    group = reader.packed.hierarchy_files.get(reader.packed.hierarchy_root / f"{task}.h5")[episode_key(task, episode)]
    nodes = group["nodes"]
    index = reader.packed._index(group)
    lookup = {str(value): i for i, value in enumerate(index["node_id"])}
    output = []
    for node_id in prepared["readers"][name]["candidate_ids"]:
        row = lookup[node_id]
        record = _record(nodes, row)
        child_rows = [int(v) for v in index["children"][row] if v >= 0]
        pending, leaves = [row], []
        while pending:
            current = pending.pop()
            children = index["children"][current]
            if children[0] < 0:
                leaves.append(current)
            else:
                pending.extend(map(int, children))
        assert len(leaves) == int(record["count"])
        # The leaf's anchor-time identifies its centre frame. Bounds may span
        # several observed frames within the local block; they are not objects.
        times = sorted(set(float(nodes["time_anchor"][i]) for i in leaves))
        displayed_times = sorted(set(times[i] for i in (0, len(times)//2, len(times)-1)))
        frame_data = [assets.frame(episode, record["source"], int(round(t))) for t in displayed_times]
        display_rows = leaves if len(leaves) <= 1024 else [leaves[i] for i in np.linspace(0, len(leaves)-1, 1024, dtype=int)]
        output.append({"id": node_id, "operation": "split parent into children" if name == "tree" else
                       "extract leaf from residual", "record": record,
                       "children": [_record(nodes, i) for i in child_rows],
                       "leaves": [_record(nodes, i) for i in display_rows],
                       "leaf_count": len(leaves), "history_frames": frame_data})
    return output


@torch.inference_mode()
def decoded_paths(teacher, compact, prepared, seeds, *, steps=8):
    """Capture predictions using precisely the label contexts and noise protocol."""
    device = teacher.device
    noise = torch.stack([torch.randn(prepared["target"].shape, device=device,
                                   generator=torch.Generator(device=device).manual_seed(seed)) for seed in seeds])
    banks = {"teacher": (teacher, prepared["full"])}
    for name, reader in prepared["readers"].items():
        banks[f"{name}/before"] = (compact, reader["before"])
        banks.update({f"{name}/{i}": (compact, bank) for i, bank in enumerate(reader["after"])})
    outputs, paths = {}, {}
    for name, (policy, bank) in banks.items():
        context, bias = policy_context(policy, [prepared], [bank])
        predictions = []
        for start in range(0, len(seeds), 4):
            n = len(noise[start:start+4])
            predictions.append(policy.decoder.integrate(context.expand(n, -1, -1),
                bias.expand(n, *bias.shape[1:]), steps=steps, noise=noise[start:start+4]).float())
        prediction = torch.cat(predictions)
        outputs[name] = prediction.cpu()
        actions = policy.normalizer.denormalize(prediction).cpu().numpy()
        physical = np.asarray([physical_action(a, "VideoUnmask") for a in actions.reshape(-1, 8)]).reshape(actions.shape)
        paths[name] = panda_tcp_positions(physical[..., :7], task="VideoUnmask").tolist()
    gains, expert_errors = {}, {}
    valid = prepared["valid"].expand(len(seeds), -1, -1)
    expert = teacher.normalizer.normalize(prepared["target"].to(device)).cpu().expand(len(seeds), -1, -1)
    for name, output in outputs.items():
        expert_errors[name] = masked_mse(output, expert, valid, reduction="none").tolist()
    for name, reader in prepared["readers"].items():
        before = masked_mse(outputs[f"{name}/before"], outputs["teacher"], valid, reduction="none")
        gains[name] = torch.stack([(before - masked_mse(outputs[f"{name}/{i}"], outputs["teacher"], valid,
                    reduction="none")) / cost for i, cost in enumerate(reader["costs"])], dim=1).numpy()
    return paths, gains, expert_errors


def run(args):
    out = Path(args.output).resolve()
    if (out / "results.json").exists():
        raise FileExistsError(f"Completed diagnostic already exists: {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    teacher, teacher_state = load_policy(args.teacher, args.device)
    compact, compact_state = load_policy(args.compact, args.device)
    for policy in (teacher, compact):
        policy.freeze_encoders()
        policy.eval().requires_grad_(False)
    teacher_hash, compact_hash = file_sha256(args.teacher), file_sha256(args.compact)
    if compact_state["provenance"]["dependencies"]["teacher"] != teacher_hash:
        raise ValueError("Student does not depend on this teacher checkpoint")
    cfg = deepcopy(compact_state["config"])
    cfg["paths"]["data_root"] = str(Path(args.prepared).resolve().parent)
    cfg["data"].update(prepared_subdir=Path(args.prepared).name, tasks=["VideoUnmask"],
                       episodes_per_task=2, fraction=1.0, alignment_path=str(Path(args.alignment).resolve()))
    cfg["model"]["visual_features"]["cache_dir"] = str(Path(args.features).resolve())
    cfg["labels"] = {"contexts": args.contexts * 2, "budgets": [64, 128, 256], "seed": 2026,
                     "split_fraction": .8, "candidates": 4, "exploration_fraction": .5}
    validate_models(cfg, teacher, teacher_state, compact, compact_state)
    manifest = json.loads((Path(args.encoded)/"manifest.json").read_text())
    if manifest["identity"]["teacher_sha256"] != teacher_hash:
        raise ValueError("Encoded cache uses a different teacher")
    expected = cache_identity(cfg, args.teacher, manifest["identity"]["data_identity"])
    for field in ("format", "geometry", "model_encoder", "normalization_version",
                  "frozen_encoder_precision", "prefix_convention", "code"):
        if manifest["identity"][field] != expected[field]:
            raise ValueError(f"Encoded cache differs from requested policy: {field}")
    for field in ("observation_stride", "decision_stride", "clock", "benchmark_revision", "simulator_revision"):
        if manifest["identity"]["data"].get(field) != cfg["data"].get(field):
            raise ValueError(f"Encoded cache data cadence/revision differs: {field}")
    prepared_hash = file_sha256(args.prepared)
    # The existing bounded local cache has an external construction receipt.
    # Verify its original prepared bytes, even when their local path has moved.
    construction = Path(args.encoded).resolve().parent / "benchmark.json"
    if manifest.get("kind") == "bounded_real_episode_diagnostic_not_scientific_training_cache":
        evidence = json.loads(construction.read_text())
        if prepared_hash not in evidence["provenance"]["inputs"].values():
            raise ValueError("Prepared data does not match bounded cache construction evidence")
    for shard, entry in manifest["shards"].items():
        if file_sha256(Path(args.encoded)/shard) != entry["sha256"]:
            raise ValueError("Encoded cache file hash differs")
    hierarchy = validate_hierarchy_cache(args.encoded, args.hierarchy)
    for shard, entry in hierarchy["shards"].items():
        if file_sha256(Path(args.hierarchy)/shard) != entry["sha256"]:
            raise ValueError("Hierarchy hash differs")
    data = TrainingData(cfg)
    # Catalog both local episodes normally; this cache contains only episode_0.
    # Restrict diagnostic decision sampling to cached rows, never truncate a
    # selected decision's full causal history to fit the cache.
    with h5py.File(Path(args.hierarchy)/"VideoUnmask.h5", "r") as file:
        available = set(map(int, file[episode_key("VideoUnmask", "episode_0")]["decisions"]))
    data.candidates["VideoUnmask"] = [(ep, index) for ep, index in data.candidates["VideoUnmask"]
                                      if ep != 0 or index in available]
    seeds = [910001 + i * 1009 for i in range(args.draws)]
    result = {"protocol": {"kind": "real_training_scene_fixed_context_diagnostic",
        "scope": "VideoUnmask episode_0 only; not benchmark success or dataset-wide label validation",
        "noise_seeds": seeds, "objectives": ["velocity", "final_action"], "integration_steps": 8,
        "teacher": {"path": str(Path(args.teacher).resolve()), "sha256": teacher_hash, "step": teacher_state["step"]},
        "compact": {"path": str(Path(args.compact).resolve()), "sha256": compact_hash, "step": compact_state["step"]},
        "hpc_label_compact_sha256": args.hpc_compact_sha256,
        "exact_hpc_compact_file": compact_hash == args.hpc_compact_sha256,
        "precision": "fp32_no_tf32", "prepared_sha256": prepared_hash,
        "raw_visualization_sha256": file_sha256(args.raw),
        "encoded_manifest_sha256": file_sha256(Path(args.encoded)/"manifest.json"),
        "hierarchy_manifest_sha256": file_sha256(Path(args.hierarchy)/"manifest.json"),
        "alignment_sha256": file_sha256(args.alignment), "source": source_revision(),
        "selection_note": "Contexts sampled before measuring gains, time-stratified and budget-balanced; high/low chosen using first half of draws, checked on second half.",
        "notes": ["No fitting, no changed policy inputs, no future history, no test episodes.",
                  "Gain is reduction in normalized teacher-student discrepancy per added token, not task success.",
                  "Nodes are local space-time summaries, not object masks; all evidence retains exact coverage.",
                  "Action paths are forward kinematics of clipped joint targets, not simulated motion.",
                  "Velocity draws vary both Gaussian noise and flow time; final-action draws vary Gaussian initialization only.",
                  "The no-op control repeats the same context; its artificial divisor is one, not a legal refinement."]},
        "contexts": []}
    atomic_json(result["protocol"], out / "manifest.json")
    assets = SceneAssets(args.prepared, args.raw)
    started = time.perf_counter()
    max_noop = {name: 0.0 for name in ("velocity", "final_action")}
    identical_error = 0.0
    try:
        with LabelDataReader(data, cfg, args.encoded, args.hierarchy) as reader:
            for planned in LabelPlan(data, cfg).items():
                if planned["episode_index"] != 0:
                    continue
                prepared = reader.prepare(planned, router_inputs=True)
                draw_items = fixed_noise_items(prepared, seeds)
                gains = {o: {r: [] for r in prepared["readers"]} for o in ("velocity", "final_action")}
                for start in range(0, len(seeds), 4):
                    values, _ = evaluate_batch(teacher, compact, draw_items[start:start+4], integration_steps=8)
                    for objective, by_reader in values.items():
                        for name, rows in by_reader.items():
                            for value in rows:
                                max_noop[objective] = max(max_noop[objective], abs(float(value[-1])))
                                gains[objective][name].append(value[:-1].tolist())
                if planned["context_id"] == 0:
                    repeat, _ = evaluate_batch(teacher, compact, draw_items[:4], integration_steps=8)
                    for objective, by_reader in repeat.items():
                        for name, rows in by_reader.items():
                            identical_error = max(identical_error, float(np.max(np.abs(
                                np.asarray(gains[objective][name][:4])-np.asarray([v[:-1].numpy() for v in rows])))))
                task, episode = prepared["episode"].split("/")
                goal = str(assets.prepared[episode].attrs["goal"])
                paths, captured_gains, expert_errors = decoded_paths(teacher, compact, prepared, seeds)
                for name, value in prepared["readers"].items():
                    if not value["after"]:
                        continue
                    measured = np.asarray(gains["final_action"][name])
                    parity = float(np.max(np.abs(measured-captured_gains[name])))
                    # Batched padding/BLAS can cause tiny FP32 differences; expose
                    # the magnitude rather than hiding it behind a broad tolerance.
                    if not np.allclose(measured, captured_gains[name], atol=2e-5, rtol=2e-3):
                        raise ValueError(f"Action capture disagrees with canonical label gains: {parity}")
                    candidates = candidate_visuals(prepared, name, reader, assets)
                    for i, candidate in enumerate(candidates):
                        candidate.update(gains={o: np.asarray(gains[o][name])[:, i].tolist() for o in gains},
                            tcp_paths={"after": paths[f"{name}/{i}"]},
                            expert_mse=expert_errors[f"{name}/{i}"])
                    record = {"id": f"context_{planned['context_id']:03d}_{name}", "task": task,
                        "episode": episode, "decision_index": prepared["decision_index"], "reader": name,
                        "budget": prepared["budget"], "goal": goal, "noise_seeds": seeds,
                        "context_seed": prepared["seed"], "history_leaves": prepared["history_leaves"],
                        "before_tokens": value["before"].content.shape[1], "frontier_ids": value["frontier_ids"],
                        "candidate_costs": value["costs"].tolist(),
                        "current": assets.frame(episode, "execution", prepared["decision_index"]),
                        "candidates": candidates,
                        "tcp_paths": {"teacher": paths["teacher"], "before": paths[f"{name}/before"],
                            "expert": panda_tcp_positions(prepared["target"][:, :7], task=task).tolist()},
                        "expert_valid": prepared["valid"].tolist(),
                        "expert_mse": {"teacher": expert_errors["teacher"], "before": expert_errors[f"{name}/before"]},
                        "statistics": {o: summarize_gains(gains[o][name]) for o in gains},
                        "capture_gain_max_abs_error": parity,
                        "selection_note": result["protocol"]["selection_note"]}
                    result["contexts"].append(record)
                    atomic_json(record, out / "contexts" / f"{record['id']}.json")
                print(json.dumps({"context": planned["context_id"], "decision": prepared["decision_index"],
                                  "budget": prepared["budget"], "seconds": time.perf_counter()-started}), flush=True)
    finally:
        assets.close()
    result["checks"] = {"identical_seed_max_abs_error": identical_error, "no_op_max_abs_gain": max_noop,
                        "capture_gain_max_abs_error": max(r["capture_gain_max_abs_error"] for r in result["contexts"])}
    result["elapsed_seconds"] = time.perf_counter()-started
    atomic_json(result, out / "results.json")
    atomic_json(summarize_experiment(result), out / "summary.json")
    from stmem.inspection.gain_report import write_report
    print(write_report(result, out), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default="runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt")
    parser.add_argument("--compact", default="runs/inspection/compact_20261104T114857Z/checkpoint.pt")
    parser.add_argument("--prepared", default="data/inspection/VideoUnmask.h5")
    parser.add_argument("--raw", default="data/raw/record_dataset_VideoUnmask.h5")
    parser.add_argument("--features", default="data/inspection/features")
    parser.add_argument("--alignment", default="data/alignment.json")
    parser.add_argument("--encoded", default="runs/hierarchy_cache_diagnostic_20261103/encoded")
    parser.add_argument("--hierarchy", default="runs/hierarchy_cache_diagnostic_20261103/hierarchy")
    parser.add_argument("--hpc-compact-sha256", default="4fe22fd00fbc64dc740ba046d6ce1e58ade2a70580f8792e152477caad316427")
    parser.add_argument("--contexts", type=int, default=30)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-only", action="store_true", help="Regenerate HTML from existing measurements; no GPU inference")
    args = parser.parse_args()
    if args.contexts < 1 or args.draws < 4 or args.draws % 2:
        parser.error("Require positive contexts and an even number of at least four noise draws")
    if args.report_only:
        from stmem.inspection.gain_report import write_report
        result = json.loads((Path(args.output)/"results.json").read_text())
        atomic_json(summarize_experiment(result), Path(args.output)/"summary.json")
        print(write_report(result, args.output))
    else:
        run(args)


if __name__ == "__main__":
    main()
