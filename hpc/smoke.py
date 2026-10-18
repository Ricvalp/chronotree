#!/usr/bin/env python3
"""Bounded, separately invoked compute/rendering check on the actual HPC node."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import torch

from stmem.config import load_config
from stmem.doctor import diagnose, choose_precision
from stmem.data.dataset import synthetic_episode
from stmem.models.action_flow import flow_sample, masked_mse
from stmem.models.policy import MemoryPolicy
from stmem.evaluation.robomme_adapter import RoboMMEAdapter
from stmem.evaluation.timing import synchronize
from stmem.training.checkpoints import atomic_json
from stmem.tracking import RunTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--task", default="VideoUnmask")
    parser.add_argument("--config", default="configs/debug/teacher.yaml")
    parser.add_argument("--updates", type=int, choices=range(1, 9), default=3)
    parser.add_argument("--output", default="runs/hpc_smoke.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.get("debug"):
        raise ValueError("This bounded smoke requires an explicit debug profile")
    # Even when supplied a custom debug config, this probe uses synthetic policy
    # targets. Its runs must never be presented as benchmark training/evaluation.
    cfg["data"] = {**cfg.get("data", {}), "synthetic": True}
    tracking_root = Path(args.output).with_suffix("")
    report = diagnose()
    device = torch.device(report["device"])
    with RunTracker(
        cfg,
        tracking_root / "training",
        job_type="hpc-smoke-training",
        metadata={"diagnostic_only": True, "device_report": report},
    ) as training_tracker:
        model = MemoryPolicy(cfg).to(device)
        episode = synthetic_episode(8, 3, cfg["model"]["points"])
        decision = episode.decision(6, cfg["model"]["horizon"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        precision = choose_precision(device)
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        durations = []
        losses = []
        for update in range(1, args.updates + 1):
            synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=dtype, enabled=precision != "fp32"):
                x, t, target, _ = flow_sample(
                    decision.target.to(device), decision.target_valid.to(device), model.normalizer
                )
                pred = model(decision, x, t)
                loss = masked_mse(pred, target, decision.target_valid.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite smoke loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            synchronize(device)
            durations.append(time.perf_counter() - start)
            losses.append(float(loss.detach()))
            # Collection and SDK enqueueing occur after the measured update.
            training_tracker.log({"train/loss": losses[-1], "train/step_seconds": durations[-1]}, step=update)
        training_tracker.summary({"training/completed_steps": args.updates, "precision": precision})
    report["bounded_training"] = {
        "updates": args.updates,
        "seconds": durations,
        "losses": losses,
        "precision": precision,
        "synthetic_debug_only": True,
        "tracking": dict(training_tracker.identity),
        "warning": "Short debug shapes; not a core throughput or cost estimate",
    }
    if args.render:
        adapter = None
        rendering_tracker = RunTracker(
            cfg,
            tracking_root / "rendering",
            job_type="hpc-smoke-rendering",
            metadata={"diagnostic_only": True, "task": args.task, "split": "val"},
        )
        try:
            with rendering_tracker:
                try:
                    adapter = RoboMMEAdapter(args.task, episode_idx=0, split="val", points=128, max_steps=8)
                    reset = adapter.reset()
                    command = torch.cat([reset.current.proprio[:7], torch.tensor([1.0])])
                    step = adapter.step(command)
                    report["rendering"] = {
                        "status": "passed",
                        "task": args.task,
                        "conditioning_frames": len(reset.conditioning),
                        "reset_valid_points": int(reset.current.valid.sum()),
                        "step_frames": len(step.frames),
                        "terminal_status": step.status,
                        "runtime_metadata": reset.metadata,
                    }
                    rendering_tracker.log(
                        {
                            "rendering/passed": True,
                            "rendering/conditioning_frames": len(reset.conditioning),
                            "rendering/reset_valid_points": int(reset.current.valid.sum()),
                            "rendering/step_frames": len(step.frames),
                        },
                        step=1,
                    )
                    rendering_tracker.summary(report["rendering"])
                finally:
                    if adapter:
                        adapter.close()
        except Exception as exc:
            report["ok"] = False
            report["rendering"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            report.setdefault("rendering", {"status": "interrupted"})["tracking"] = dict(
                rendering_tracker.identity
            )
    atomic_json(report, args.output)
    print(f"Hardware/render smoke: {args.output}; ok={report['ok']}")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
