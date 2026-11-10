"""Gain-blind context-distribution audit; never loads a policy or GPU model."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import yaml

from stmem.config import digest
from stmem.data.cache import FrozenCache, episode_key
from stmem.config import dependency
from stmem.inspection.refinement_candidates import IndexedArchive
from stmem.training.cached_loader import LightweightDecisions
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.packed_context import PackedContextReader
from stmem.training.utility_study import VARIANTS, restrict_training_pool, study_context


def context_statistics(index, roots, leaves, context):
    if context["reader"] == "CURRENT":
        return {"token_count": 0, "leaf_fraction": 0.0, "mean_depth": 0.0,
                "max_depth": 0, "source_coverage": [], "root_record_concentration": 0.0}
    archive = IndexedArchive(index, roots, leaves)
    context = archive.validate(context)
    rows = context["rows"]
    if context["reader"] == "TREE":
        depths = [int(index["depth"][row]) for row in rows]
        fine = sum(int(index["count"][row]) == 1 for row in rows)
        source_coverage = sorted({str(index["source"][row]) for row in rows})
        concentrations = []
        for root in roots:
            lo, hi = archive.intervals[int(root)]
            concentrations.append(sum(lo <= archive.intervals[row][0] < hi for row in rows))
    else:
        mask, codes, groups = archive.flat_arrays(context)
        counts = Counter(codes[~mask])
        depths = [0] * len(rows) + [1] * len(counts)
        fine = len(rows)
        source_coverage = sorted({group[0] for group in groups})
        concentrations = [sum(row in set(archive.support(root)) for row in rows) for root in roots]
    return {"token_count": context["token_count"], "leaf_fraction": fine / max(context["token_count"], 1),
            "mean_depth": float(np.mean(depths)) if depths else 0.0,
            "max_depth": max(depths, default=0), "source_coverage": source_coverage,
            "root_record_concentration": max(concentrations, default=0) / max(context["token_count"], 1)}


def audit(cfg, output, *, count=4096):
    if count < 1:
        raise ValueError("Audit count must be positive")
    output = Path(output)
    identity = {"schema_version": 1, "kind": "gain_blind_student_sampler_audit",
                "configuration": cfg, "count": count, "variants": list(VARIANTS)}
    signature = digest(identity)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Sampler audit needs a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json({**identity, "identity": signature, "status": "running"}, output / "manifest.json")
    data = TrainingData(cfg)
    pool = restrict_training_pool(data, cfg)
    cache = FrozenCache(cfg, dependency(cfg, "teacher"), data)
    light = LightweightDecisions(data)
    reader = PackedContextReader(cfg, cache.root, cfg["training"]["hierarchy_cache_root"])
    summaries = {variant: {"examples": 0, "families": Counter(), "budgets": Counter(),
                           "fallbacks": Counter(), "sources": Counter(), "template_examples": 0,
                           "within_packet": 0, "saturated": 0, "sampler_seconds": 0.0,
                           "leaf_fraction": [], "mean_depth": [], "max_depth": [],
                           "root_record_concentration": [], "generated_route_packet_lengths": Counter(),
                           "consumed_prefix_packet_lengths": Counter()}
                 for variant in VARIANTS}
    started = time.perf_counter()
    try:
        with (output / "contexts.jsonl").open("w") as handle:
            for position in range(count):
                decision = light.decision(position)
                info = decision.info
                hierarchy = reader.hierarchy_files.get(reader.hierarchy_root / f"{info.task}.h5")[
                    episode_key(info.task, info.episode_id)]
                entry = hierarchy["decisions"][str(info.index)]
                roots = reader._read(entry["roots"])
                leaves = reader._active_leaves(hierarchy, entry)
                index = reader._index(hierarchy)
                matched = {}
                for variant in VARIANTS:
                    variant_cfg = deepcopy(cfg)
                    variant_cfg["training"]["utility_study"]["variant"] = variant
                    begin = time.perf_counter()
                    context, metadata = study_context(index, roots, leaves, variant_cfg, position)
                    duration = time.perf_counter() - begin
                    statistics = context_statistics(index, roots, leaves, context)
                    row = {"position": position, "task": info.task, "episode_id": info.episode_id,
                           "index": info.index, "phase": info.phase, "variant": variant,
                           "context": context, "exposure": metadata, "statistics": statistics,
                           "sampler_seconds": duration}
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    summary = summaries[variant]
                    summary["examples"] += 1
                    summary["families"][metadata["family"]] += 1
                    summary["budgets"][statistics["token_count"]] += 1
                    summary["fallbacks"][metadata.get("fallback_reason") or "none"] += 1
                    summary["sources"].update(statistics["source_coverage"])
                    summary["template_examples"] += int(metadata["template"])
                    summary["within_packet"] += int(metadata.get("within_packet", False))
                    summary["saturated"] += int(metadata.get("saturated", False))
                    summary["sampler_seconds"] += duration
                    summary["generated_route_packet_lengths"].update(metadata.get("packet_costs", []))
                    summary["consumed_prefix_packet_lengths"].update(metadata.get("consumed_packet_costs", []))
                    for key in ("leaf_fraction", "mean_depth", "max_depth", "root_record_concentration"):
                        summary[key].append(statistics[key])
                    matched[variant] = metadata
                fields = ("family", "template", "requested_target", "initial_count", "target_count",
                          "prefix_count", "fallback_reason")
                if any(matched["S_STAGE"].get(key) != matched["S_PACKET"].get(key) for key in fields):
                    raise AssertionError("Stage/packet exposure schedules differ")
                if position % 100 == 0:
                    handle.flush()
    finally:
        light.close()
        reader.close()
    for summary in summaries.values():
        for key in ("leaf_fraction", "mean_depth", "max_depth", "root_record_concentration"):
            values = summary[key]
            summary[key] = {"mean": float(np.mean(values)), "p50": float(np.median(values)),
                            "p95": float(np.quantile(values, 0.95))}
    result = {"status": "complete", "identity": signature, "samples_per_variant": count,
              "shared_source_states": True, "policy_evaluations": 0, "pool": pool,
              "packet_length_meanings": {
                  "generated_route_packet_lengths": "All packets in the complete coarse-to-target route",
                  "consumed_prefix_packet_lengths": "Only moves reaching the sampled training prefix; final packet may be partial",
              },
              "wall_seconds": time.perf_counter() - started, "variants": summaries,
              "contexts_sha256": file_sha256(output / "contexts.jsonl")}
    atomic_json(result, output / "summary.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Resolved generated student YAML")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4096)
    args = parser.parse_args(argv)
    print(json.dumps(audit(yaml.safe_load(args.config.read_text()), args.output, count=args.count), indent=2))


if __name__ == "__main__":
    main()
