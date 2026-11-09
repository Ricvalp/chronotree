"""Exact FULL-only execution-payload permutation without rebuilding trees.

This preserves the permutation and leaf fields of
``stmem.analyze.intervene_evidence(..., 'permute_execution', seed)``. A FULL
reader consumes leaves directly, so rebuilding every parent adds no evidence
and is unnecessary for this diagnostic. This wrapper is not valid for TREE,
FLAT, UNIFORM, or PERCEIVER readers.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import random
import time

from stmem.memory.hierarchy import stable_id


def permute_execution_records(records, seed):
    """Return exactly the legacy intervention's ordered leaves, including IDs."""
    records = list(records)
    if any(not record.is_leaf or record.count != 1 for record in records):
        raise ValueError("The exact FULL permutation requires unit-mass fine leaves")
    groups = defaultdict(list)
    for leaf in records:
        if leaf.source == "execution":
            groups[leaf.modality].append(leaf)
    rng = random.Random(seed)
    mapping = {}
    for modality in sorted(groups):
        slots = sorted(groups[modality], key=lambda record: (record.time_anchor, record.node_id))
        payloads = slots.copy()
        rng.shuffle(payloads)
        for slot, payload in zip(slots, payloads):
            mapping[slot.node_id] = replace(
                slot,
                node_id=stable_id(f"permuted:{seed}", (slot.node_id, payload.node_id)),
                content_sum=payload.content_sum,
                max_content=payload.max_content,
                anchor=payload.anchor,
                bounds_min=payload.bounds_min,
                bounds_max=payload.bounds_max,
                occupancy=payload.occupancy,
            )
    return [mapping.get(record.node_id, record) for record in records]


class PermutedExecutionOnlinePolicy:
    """Same FULL permutation intervention, applied solely at the context read."""

    intervention = "permute_execution"

    def __init__(self, online, *, seed=None):
        if online.method != "FULL" or online.intervention != "none":
            raise ValueError("Wrap ordinary FULL with no preexisting intervention")
        self.online = online
        self.seed = online.cfg["evaluation"]["seed"] if seed is None else seed
        self.reads = []

    def __getattr__(self, name):
        return getattr(self.online, name)

    @property
    def update_seconds(self):
        return self.online.update_seconds

    @update_seconds.setter
    def update_seconds(self, value):
        self.online.update_seconds = value

    def reset(self, reset, seed):
        self.reads = []
        return self.online.reset(reset, seed)

    def observe(self, *args, **kwargs):
        return self.online.observe(*args, **kwargs)

    def decide(self):
        policy = self.online.policy
        original = policy.context
        had_override = "context" in policy.__dict__
        override = policy.__dict__.get("context")
        captured = []

        def permuted_context(evidence, records, **kwargs):
            started = time.perf_counter()
            if [record.node_id for record in records] != [record.node_id for record in evidence.archive.leaves]:
                raise ValueError("Expected the complete ordered FULL leaf read")
            selected = permute_execution_records(records, self.seed)
            captured.append({
                "seed": self.seed, "history_tokens": len(selected),
                "execution_tokens": sum(record.source == "execution" for record in selected),
                "conditioning_tokens": sum(record.source == "conditioning" for record in selected),
                "decision_index": int(self.online.decision_index),
                "current_source_index": int(evidence.current.source_index),
                "leaf_permutation_seconds": time.perf_counter() - started,
                "hierarchy_rebuilt": False,
                "semantics": "Exact ordered-leaf equivalent of stmem.analyze.permute_execution",
            })
            return original(evidence, selected, **kwargs)

        policy.context = permuted_context
        try:
            actions, timing = self.online.decide()
        finally:
            if had_override:
                policy.context = override
            else:
                delattr(policy, "context")
        if len(captured) != 1:
            raise RuntimeError("Expected one FULL context read per decision")
        self.reads.append(captured[0])
        timing["execution_permutation"] = captured[0]
        return actions, timing


def benchmark_cpu(*, threads=4, blocks=16, repeats=3):
    """Synthetic structural CPU microbenchmark, not rollout or model timing."""
    from types import SimpleNamespace
    import torch

    from stmem.analyze import intervene_evidence
    from stmem.memory.hierarchy import Archive
    from stmem.memory.records import leaf_record
    from stmem.models.policy import Evidence

    if min(threads, blocks, repeats) < 1:
        raise ValueError("Threads, blocks and repeats must be positive")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        archive = Archive()
        generator = torch.Generator().manual_seed(0)
        for block in range(blocks):
            archive.add_block([
                leaf_record(f"execution:{block}:{index}", torch.randn(256, generator=generator),
                            torch.rand(3, generator=generator), float(block * 16 + index // 32 * 2),
                            "execution", "visual")
                for index in range(256)
            ], f"execution:{block}")
        evidence = Evidence(archive, [], torch.zeros(256), SimpleNamespace(source_index=blocks * 16),
                            {"execution": float(blocks * 16)})
        times = {}
        for label in ("legacy_hierarchy_rebuild", "full_leaf_only"):
            samples = []
            for _ in range(repeats):
                started = time.perf_counter()
                if label == "legacy_hierarchy_rebuild":
                    result = intervene_evidence(evidence, "permute_execution", 1701).archive.leaves
                else:
                    result = permute_execution_records(archive.leaves, 1701)
                samples.append(time.perf_counter() - started)
                assert len(result) == blocks * 256
            times[label] = {"seconds": samples, "mean_seconds": sum(samples) / repeats}
        return {"scope": "Synthetic CPU structural diagnostic only; excludes model, CUDA transfer, simulator and preprocessing",
                "synthetic": True, "threads": threads, "blocks": blocks, "leaves": blocks * 256,
                "width": 256, "repeats": repeats, "torch": torch.__version__, "measurements": times}
    finally:
        torch.set_num_threads(old_threads)


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    from stmem.training.checkpoints import atomic_json

    parser = argparse.ArgumentParser(description="Synthetic CPU-only permutation microbenchmark")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark_cpu(threads=args.threads, blocks=args.blocks, repeats=args.repeats)
    if args.output is not None:
        atomic_json(result, args.output)
    print(json.dumps(result, indent=2))
