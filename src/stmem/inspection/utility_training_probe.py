"""Bounded real-cache student update timing; explicitly not a training result."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import time

import h5py
import torch
import yaml

from stmem.config import digest
from stmem.data.cache import episode_key
from stmem.models.action_flow import flow_sample, masked_mse
from stmem.models.batched_policy import records_bank
from stmem.models.policy import MemoryPolicy
from stmem.training.cached_loader import CachedArchiveReader, CachedPolicyDataset, LightweightDecisions
from stmem.training.cached_policy import CachedPolicyForward
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.runtime import load_checkpoint, validate_policy_config
from stmem.training.utility_study import TEACHER_SHA256, VARIANTS, paired_flow_randomness


def run_probe(cfg, teacher, cache_root, hierarchy_root, output, *, steps=3, microbatch=4, device="cuda"):
    if steps < 1 or steps > 10 or not 1 <= microbatch <= 16:
        raise ValueError("Diagnostic probe permits 1–10 updates and microbatch 1–16")
    teacher, output = Path(teacher), Path(output)
    if file_sha256(teacher) != TEACHER_SHA256:
        raise ValueError("Probe requires the exact accepted teacher")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Training probe requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    initial = load_checkpoint(teacher)
    validate_policy_config(initial["config"], cfg)
    cfg = deepcopy(cfg)
    cfg["stage"] = "compact"
    cfg["training"].update(context_mix={"tree": 0.45, "flat": 0.45, "uniform": 0.05, "current": 0.05},
                           intermediate_probability=0.5, hierarchy_cache_root=str(hierarchy_root),
                           utility_study={"variant": "S_STD", "manifest": "bounded_diagnostic_only",
                                          "manifest_sha256": "not_a_production_training_pool"})
    cfg["model"].update(budget=128, budgets=[64, 128, 256])
    data = TrainingData(cfg)
    available = []
    episodes = set()
    for task in data.tasks:
        with h5py.File(Path(hierarchy_root) / f"{task}.h5") as handle:
            for episode in data.by_task[task]:
                ref = data.dataset.episodes[data.refs[episode]]
                key = episode_key(task, ref.key)
                if key not in handle:
                    continue
                valid = {index for owner, index in data.candidates[task] if owner == episode}
                for index in sorted(map(int, handle[key]["decisions"]), key=int):
                    if index in valid:
                        available.append((episode, index))
                        episodes.add((task, ref.key))
    if not available:
        raise ValueError("No real decisions are covered by the diagnostic cache")
    # Same deterministic source references in every arm. This deliberate local
    # diagnostic subset never changes the production sampling/episode manifest.
    data.sample_reference = lambda position: available[(int(position) * 7) % len(available)]
    device = torch.device(device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    else:
        torch.set_num_threads(2)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    model = MemoryPolicy(cfg).to(device)
    model.load_state_dict(initial["model"])
    model.freeze_encoders()
    model.eval()
    decoder = CachedPolicyForward(model)
    light = LightweightDecisions(data)
    reader = CachedArchiveReader(cfg, cache_root)
    dataset = CachedPolicyDataset(data, cache_root)
    try:
        decision = light.decision(0)
        evidence = reader.evidence(decision)
        batch = dataset[(0,)]
        full = records_bank(model, [evidence.archive.leaves], torch.tensor([decision.clocks], device=device))
        batch = replace(batch, history=full)
        with torch.inference_mode():
            actual, bias, _ = decoder.context(batch)
            reference, reference_bias = model.context(evidence, evidence.archive.leaves)
            torch.testing.assert_close(actual[0], reference, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(bias[0], reference_bias, atol=0, rtol=0)
            noise = torch.randn(1, 16, 8, device=device, generator=torch.Generator(device=device).manual_seed(17))
            u = torch.full((1,), 0.4, device=device)
            a = model.decoder(noise, u, actual, bias)
            b = model.decoder(noise, u, reference[None], reference_bias[None])
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
            parity = {"context_max_abs": float((actual[0] - reference).abs().max()),
                      "velocity_max_abs": float((a - b).abs().max()), "history_tokens": len(evidence.archive.leaves)}
    finally:
        light.close()
        reader.close()
        dataset.close()
    rows = []
    for variant in VARIANTS:
        data.cfg["training"]["utility_study"]["variant"] = variant
        model.load_state_dict(initial["model"])
        model.train()
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1e-4)
        dataset = CachedPolicyDataset(data, cache_root)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        variant_rows = []
        try:
            for step in range(steps):
                positions = tuple(range(step * microbatch, (step + 1) * microbatch))
                started = time.perf_counter()
                loaded = dataset[positions]
                load_seconds = time.perf_counter() - started
                sync()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                action, valid = loaded.target_chunk.to(device), loaded.target_valid.to(device)
                noise, u = paired_flow_randomness(action, positions, cfg["seed"])
                x, u, target, _ = flow_sample(action, valid, model.normalizer, noise=noise, time=u)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    prediction, _ = decoder(loaded, x, u)
                    loss = masked_mse(prediction, target, valid)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                sync()
                variant_rows.append({"update": step + 1, "load_seconds": load_seconds,
                                     "update_seconds": time.perf_counter() - started,
                                     "expert_fm_loss": float(loss.detach()),
                                     "contexts": loaded.study_contexts})
            if any(value.grad is not None for module in (model.encoder, model.text_encoder)
                   for value in module.parameters()):
                raise AssertionError("Diagnostic update produced an encoder gradient")
            rows.append({"variant": variant, "updates": variant_rows,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0})
        finally:
            dataset.close()
    result = {"kind": "bounded_real_cache_student_update_diagnostic", "teacher_sha256": TEACHER_SHA256,
              "configuration_sha256": digest(cfg), "episodes": sorted(episodes),
              "available_decisions": len(available), "device": str(device),
              "precision": "bf16" if device.type == "cuda" else "fp32", "microbatch": microbatch,
              "steps": steps, "full_leaf_initialization_parity": parity, "variants": rows,
              "limitations": "One/few local episodes, serial loading, no DDP and no production study optimization. "
                             "No model weights are saved; throughput is not a full-HPC runtime estimate."}
    atomic_json(result, output / "summary.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "teacher", "cache-root", "hierarchy-root", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--microbatch", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    result = run_probe(cfg, args.teacher, args.cache_root, args.hierarchy_root, args.output,
                       steps=args.steps, microbatch=args.microbatch, device=args.device)
    print(json.dumps({"summary": str(args.output / "summary.json"), "episodes": result["episodes"],
                      "parity": result["full_leaf_initialization_parity"],
                      "variants": [{"variant": row["variant"], "peak_allocated_bytes": row["peak_allocated_bytes"],
                                    "updates": [{key: value for key, value in update.items() if key != "contexts"}
                                                for update in row["updates"]]} for row in result["variants"]]}, indent=2))


if __name__ == "__main__":
    main()
