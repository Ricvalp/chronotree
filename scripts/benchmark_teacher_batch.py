#!/usr/bin/env python3
"""Bounded synthetic teacher forward/backward comparison; no training updates.

Compare serial gradient accumulation with true batching on identical decisions,
parameters, noisy actions and flow times. Include CPU collation, transfers,
full causal history encoding, decoder and backward. Exclude dataset loading,
optimizer updates, logging and checkpoint writes. This is not a real-data
throughput measurement or a model-quality evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

import torch

from stmem.config import load_config, source_revision
from stmem.data.dataset import synthetic_episode
from stmem.models.action_flow import flow_sample, masked_mse
from stmem.models.policy import MemoryPolicy


def _lengths(value):
    try:
        lengths = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use comma-separated nonnegative integers") from error
    if not lengths or min(lengths) < 0:
        raise argparse.ArgumentTypeError("Use comma-separated nonnegative integers")
    return lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/teacher.yaml")
    parser.add_argument("--output", type=Path, default=Path("runs/batched_teacher_diagnostic/benchmark.json"))
    parser.add_argument(
        "--execution-history", type=_lengths, default=_lengths("64,96,160,256,384,512,640,768")
    )
    parser.add_argument("--conditioning-history", type=_lengths, default=_lengths("0,0,32,64,96,128,160,192"))
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if len(args.execution_history) != len(args.conditioning_history):
        parser.error("Execution and conditioning lists must describe the same number of decisions")
    if args.warmups < 0 or args.repeats < 1 or args.cpu_threads < 1:
        parser.error("Use nonnegative warmups and positive repeats/CPU threads")
    if not torch.cuda.is_available():
        parser.error("This diagnostic requires a working CUDA device")
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        parser.error("BF16 is unavailable; explicitly select --precision fp32")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    config["data"]["synthetic"] = True
    model = MemoryPolicy(config).cuda().train()
    horizon = config["model"]["horizon"]
    decisions = [
        synthetic_episode(
            execution + horizon, conditioning, config["model"]["points"], args.seed + i
        ).decision(execution, horizon)
        for i, (execution, conditioning) in enumerate(zip(args.execution_history, args.conditioning_history))
    ]
    batch_size = len(decisions)
    actions = torch.stack([decision.target_chunk for decision in decisions]).cuda()
    valid = torch.stack([decision.target_valid for decision in decisions]).cuda()
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1000)
    noisy, flow_time, target, _ = flow_sample(actions, valid, model.normalizer, generator=generator)

    def run(batched):
        model.zero_grad(set_to_none=True)
        if batched:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                prediction = model(decisions, noisy, flow_time)
                loss = masked_mse(prediction, target, valid)
            loss.backward()
            return prediction.detach().float(), loss.detach()
        predictions, losses = [], []
        for index, decision in enumerate(decisions):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                prediction = model(decision, noisy[index], flow_time[index])
                loss = masked_mse(prediction, target[index], valid[index]) / batch_size
            loss.backward()
            predictions.append(prediction.detach().float())
            losses.append(loss.detach())
        return torch.stack(predictions), torch.stack(losses).sum()

    report = {
        "diagnostic": "Synthetic forward/backward only; no optimizer updates, dataset I/O, or W&B.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_revision(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "precision": args.precision,
        "cpu_threads": args.cpu_threads,
        "seed": args.seed,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "execution_history_frames": args.execution_history,
        "conditioning_history_frames": args.conditioning_history,
        "model": config["model"],
        "encoder_block_batch_size": config["training"].get("encoder_block_batch_size", 32),
        "batch_size": batch_size,
        "timing_includes": [
            "host collation",
            "host-to-device transfer",
            "history and current encoding",
            "text encoding",
            "metadata adaptation",
            "action decoder",
            "backward",
        ],
        "timing_excludes": ["dataset loading", "optimizer updates", "logging", "checkpoint writes"],
    }
    reference = reference_loss = reference_gradients = None
    for batched, label in ((False, "serial"), (True, "batched")):
        print(json.dumps({"event": "benchmark_phase", "phase": label, "warmups": args.warmups}), flush=True)
        for _ in range(args.warmups):
            run(batched)
        torch.cuda.synchronize()
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for repeat in range(args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction, loss = run(batched)
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            durations.append(duration)
            print(
                json.dumps(
                    {"event": "benchmark_sample", "phase": label, "repeat": repeat, "seconds": duration}
                ),
                flush=True,
            )
        report[label] = {
            "seconds": durations,
            "median_seconds": statistics.median(durations),
            "examples_per_second": batch_size / statistics.median(durations),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "loss": loss.item(),
        }
        if not batched:
            reference, reference_loss = prediction.cpu(), loss.item()
            reference_gradients = {
                name: parameter.grad.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        else:
            squared_difference, squared_reference, absolute_difference = [], [], []
            if set(reference_gradients) != {
                name for name, p in model.named_parameters() if p.grad is not None
            }:
                raise AssertionError("Serial and batched paths trained different parameters")
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                difference = parameter.grad.float() - reference_gradients[name].to(parameter.device).float()
                if not torch.isfinite(difference).all():
                    raise AssertionError(f"Nonfinite gradient in {name}")
                squared_difference.append(difference.square().sum())
                squared_reference.append(reference_gradients[name].float().square().sum())
                absolute_difference.append(difference.abs().max())
            report["parity"] = {
                "prediction_max_absolute_difference": (prediction.cpu() - reference).abs().max().item(),
                "loss_absolute_difference": abs(loss.item() - reference_loss),
                "gradient_relative_l2_difference": (
                    torch.stack(squared_difference).sum().sqrt().item()
                    / torch.stack(squared_reference).sum().sqrt().clamp_min(1e-12).item()
                ),
                "gradient_max_absolute_difference": torch.stack(absolute_difference).max().item(),
                "note": "Batched GEMM/reduction order differs; FP32 is the tight numerical parity check.",
            }
    report["median_speedup"] = report["serial"]["median_seconds"] / report["batched"]["median_seconds"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
