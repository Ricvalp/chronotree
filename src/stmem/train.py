"""Bounded staged training with deterministic sampling and coherent safe resume."""

from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from stmem.config import load_config, validate_config, run_dir, dependency, digest, source_revision
from stmem.models.policy import MemoryPolicy, choose_records
from stmem.models.action_flow import flow_sample, masked_mse
from stmem.training.data import TrainingData
from stmem.training.runtime import (
    distributed_setup,
    seed_all,
    load_checkpoint,
    validate_selection,
    validate_policy_config,
    environment_manifest,
)
from stmem.training.checkpoints import (
    atomic_save,
    atomic_json,
    capture_rng,
    restore_rng,
    mark_complete,
    file_sha256,
)
from stmem.tracking import RunTracker
from stmem.evaluation.timing import process_memory


def context_sample(policy, evidence, cfg, position):
    rng = random.Random(cfg["seed"] * 1000003 + position)
    mixture = cfg["training"]["context_mix"]
    # Exact proportions per 20 global positions, with a permuted deterministic cycle.
    families = [name.upper() for name, fraction in mixture.items() for _ in range(round(fraction * 20))]
    random.Random(cfg["seed"] + position // 20).shuffle(families)
    method = families[position % len(families)]
    budget = cfg["model"]["budgets"][(position // len(families)) % len(cfg["model"]["budgets"])]
    roots = len(evidence.archive.roots)
    if roots > budget:
        raise ValueError(f"{roots} roots exceed budget {budget}")
    if method != "CURRENT" and rng.random() < cfg["training"]["intermediate_probability"]:
        budget = rng.randint(max(roots, 1), budget)
    return choose_records(evidence, method, budget, rng=rng).records


def train_policy(cfg, stage, *, resume=None, stop_file=None):
    validate_config(cfg)
    rank = int(os.environ.get("RANK", "0"))
    directory = run_dir(cfg, stage)
    if rank == 0 and not resume and (directory / "last.pt").exists():
        raise FileExistsError(f"Existing run: {directory}; use --resume explicitly")
    with RunTracker(
        cfg,
        directory,
        job_type=stage,
        rank=rank,
        resume=bool(resume),
        metadata={"resume_checkpoint": str(resume) if resume else None},
    ) as tracker:
        result = _train_policy(cfg, stage, resume=resume, stop_file=stop_file, tracker=tracker)
        if result == 75:
            tracker.finish(status="checkpointed_partial", exit_code=75)
        return result


def _train_policy(cfg, stage, *, resume=None, stop_file=None, tracker):
    from stmem.doctor import choose_precision
    from stmem.data.cache import FrozenCache

    device, rank, world = distributed_setup()
    seed_all(cfg["seed"] + rank)
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    batch = cfg["training"]["effective_batch"]
    if cfg["training"]["microbatch"] != 1:
        raise ValueError("Variable-length reference trainer uses microbatch=1 with gradient accumulation")
    if batch % world:
        raise ValueError("Effective batch must be divisible by world size")
    data = TrainingData(cfg)
    policy = MemoryPolicy(cfg)
    dependencies = {}
    cache = None
    if stage != "teacher":
        teacher = dependency(cfg, "teacher")
        validate_selection(teacher, debug=cfg.get("debug", False) and data.synthetic)
        initial = load_checkpoint(teacher)
        validate_policy_config(initial["config"], cfg)
        policy.load_state_dict(initial["model"])
        policy.freeze_encoders()
        dependencies["teacher"] = file_sha256(teacher)
        cache = FrozenCache(cfg, teacher, data)
        dependencies["cache"] = digest(cache.manifest)
    else:
        data.fit_normalizer(policy.normalizer)
    if stage == "perceiver":
        policy.add_perceiver()
    policy.to(device).train()
    precision = cfg["training"]["precision"]
    if precision == "auto":
        precision = choose_precision(device)
    if device.type == "cpu":
        precision = "fp32"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    steps = cfg["training"]["steps"]
    warmup = cfg["training"]["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min((s + 1) / max(warmup, 1), 1.0)
        * (0.1 + 0.9 * (1 + math.cos(math.pi * min(max(s - warmup, 0) / max(steps - warmup, 1), 1))) / 2),
    )
    start = 0
    provenance = {
        "data": data.identity,
        "dependencies": dependencies,
        "model": digest(cfg["model"]),
        "seed": cfg["seed"],
        "world_size": world,
        "effective_batch": batch,
        "training": digest(cfg["training"]),
    }
    if resume:
        saved = load_checkpoint(resume)
        if saved["stage"] != stage or saved["provenance"] != provenance:
            raise ValueError("Resume stage/data/config/dependencies/world size mismatch")
        policy.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        start = saved["step"]
        restore_rng(saved["rng_by_rank"][rank])
        if start > steps:
            raise ValueError("Resume checkpoint is beyond the intended optimizer-step budget")
    directory = run_dir(cfg, stage)
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        if not resume and (directory / "last.pt").exists():
            raise FileExistsError(f"Existing run: {directory}; use --resume explicitly")
        if start < steps:
            (directory / "complete.json").unlink(missing_ok=True)
        atomic_json(cfg, directory / "config.json")
        atomic_json(
            {
                "source": source_revision(),
                "provenance": provenance,
                "precision": precision,
                "device": str(device),
                "environment": environment_manifest(),
                "synthetic": data.synthetic,
            },
            directory / "manifest.json",
        )
    # Empty historical banks/all-invalid points legitimately leave modality
    # encoders unused. Availability also changes between decisions and ranks.
    model = (
        DDP(policy, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True)
        if world > 1
        else policy
    )
    if world > 1:
        torch.distributed.barrier()
    tracker.summary(
        {
            "training/start_step": start,
            "training/intended_steps": steps,
            "training/world_size": world,
            "training/precision": precision,
            "model/parameters": sum(p.numel() for p in policy.parameters()),
            "model/trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
            "provenance": provenance,
            "environment": environment_manifest(),
        }
    )
    started = time.perf_counter()
    records_seen = 0
    allocated = int(os.environ.get("STMEM_ALLOCATED_GPUS", os.environ.get("STMEM_GPUS", world)))
    from stmem.visualization.diagnostics import PolicyScenes

    scenes = PolicyScenes(cfg, data, policy, tracker, stage=stage, cache=cache, precision=precision)

    def save(step, candidate=False):
        own_rng = capture_rng()
        states = [None] * world
        if world > 1:
            torch.distributed.all_gather_object(states, own_rng)
        else:
            states[0] = own_rng
        if rank == 0:
            payload = {
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "rng_by_rank": states,
                "step": step,
                "sampler": {"next_global_position": step * batch, "boundary": "optimizer_step"},
                "stage": stage,
                "config": cfg,
                "provenance": provenance,
                "source": source_revision(),
                "environment": environment_manifest(),
                "tracking": dict(tracker.identity),
            }
            atomic_save(payload, directory / "last.pt")
            tracker.checkpoint(directory / "last.pt", step)
            if candidate:
                path = directory / f"step_{step:08d}.pt"
                if path.exists():
                    raise FileExistsError(f"Immutable candidate already exists: {path}")
                atomic_save(payload, path, keep_previous=False)

    step = start
    consecutive_overflows = 0
    total_overflows = 0
    while step < steps:
        optimizer.zero_grad(set_to_none=True)
        local_loss = 0.0
        local_n = batch // world
        for local_index, offset in enumerate(range(rank, batch, world)):
            position = step * batch + offset
            decision = data.sample_position(position)
            action = decision.target_chunk.to(device)
            valid = decision.target_valid.to(device)
            evidence = None if cache is None else cache.evidence(decision)
            sync = model.no_sync() if world > 1 and local_index < local_n - 1 else nullcontext()
            with sync, torch.autocast(device.type, dtype=dtype, enabled=precision != "fp32"):
                if evidence is None:
                    evidence = policy.encode(decision)
                context_records = (
                    context_sample(policy, evidence, cfg, position)
                    if stage == "compact"
                    else evidence.archive.leaves
                )
                noisy, flow_time, target, _ = flow_sample(action, valid, policy.normalizer)
                if cfg["model"].get("decoder") == "regression":
                    target = policy.normalizer.normalize(action).masked_fill(~valid, 0)
                prediction = model(
                    decision,
                    noisy,
                    flow_time,
                    records=context_records,
                    evidence=evidence,
                    perceiver=stage == "perceiver",
                )
                loss = masked_mse(prediction, target, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at optimizer step {step}")
            scaler.scale(loss / local_n).backward()
            local_loss += float(loss.detach()) / local_n
            records_seen += len(evidence.archive.leaves)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), cfg["training"]["grad_clip"], error_if_nonfinite=precision != "fp16"
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if precision == "fp16" and scaler.get_scale() < old_scale:
            # GradScaler skipped the update. DDP reductions make nonfinite
            # gradients shared across ranks, so every rank retries this position.
            # Neither optimizer-step exposure nor scheduler advances on a skip.
            consecutive_overflows += 1
            total_overflows += 1
            if consecutive_overflows >= 8:
                raise FloatingPointError("Eight consecutive FP16 overflows; retry with FP32 debugging")
            continue
        consecutive_overflows = 0
        scheduler.step()
        current_step = step + 1
        flag = torch.tensor(
            [int(bool(stop_file and Path(stop_file).exists())) if rank == 0 else 0], device=device
        )
        if world > 1:
            torch.distributed.broadcast(flag, 0)
        stopped = bool(flag.item())
        checkpoint_due = (
            current_step % cfg["training"]["checkpoint_every"] == 0
            or current_step in cfg["training"]["checkpoint_steps"]
            or stopped
            or current_step == steps
        )
        if checkpoint_due:
            save(current_step, current_step in cfg["training"]["checkpoint_steps"])
        if current_step % cfg["training"]["log_every"] == 0 or stopped or current_step == steps:
            loss_tensor = torch.tensor(local_loss, device=device)
            if world > 1:
                torch.distributed.all_reduce(loss_tensor)
                loss_tensor /= world
            elapsed = time.perf_counter() - started
            record = {
                "step": current_step,
                "training_action_loss": float(loss_tensor),
                "objective": cfg["model"]["decoder"],
                "seconds": elapsed,
                "examples": (current_step - start) * batch,
                "effective_batch": batch,
                "examples_per_second": (current_step - start) * batch / elapsed,
                "allocated_gpu_hours": elapsed * allocated / 3600 if device.type == "cuda" else 0.0,
                "gpu_profile": cfg["profile"],
                "fine_leaves_processed_rank": records_seen,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm_before_clip": float(grad_norm),
                "amp_scale": scaler.get_scale(),
                "amp_overflows_this_attempt": total_overflows,
                "gpu_peak_allocated_bytes_rank": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
                "gpu_peak_reserved_bytes_rank": torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else 0,
                **process_memory(),
            }
            if rank == 0:
                with (directory / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                tracker.log({f"train/{key}": value for key, value in record.items()}, step=current_step)
        if checkpoint_due and not (stopped and current_step < steps):
            scenes.log(current_step)
        if stopped and current_step < steps:
            if rank == 0:
                atomic_json(
                    {
                        "status": "checkpointed_partial",
                        "step": current_step,
                        "checkpoint": str(directory / "last.pt"),
                    },
                    directory / "partial.json",
                )
            if world > 1:
                torch.distributed.destroy_process_group()
            return 75  # afterok dependencies must remain blocked
        step = current_step
    if rank == 0:
        mark_complete(
            directory / "complete.json",
            completed_steps=steps,
            intended_steps=steps,
            checkpoint=directory / "last.pt",
            synthetic=data.synthetic,
        )
        (directory / "partial.json").unlink(missing_ok=True)
        elapsed = time.perf_counter() - started
        tracker.summary(
            {
                "training/completed_steps": steps,
                "training/seconds_including_diagnostics": elapsed,
                "training/allocated_gpu_hours_including_diagnostics": elapsed * allocated / 3600
                if device.type == "cuda"
                else 0.0,
                "training/gpu_peak_allocated_bytes_rank": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else 0,
            }
        )
        tracker.summary(
            {"checkpoint/sha256": json.loads((directory / "complete.json").read_text())["checkpoint_sha256"]}
        )
    if world > 1:
        torch.distributed.destroy_process_group()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["teacher", "compact", "router", "perceiver"])
    parser.add_argument("--resume")
    parser.add_argument("--stop-file", default=os.environ.get("STMEM_STOP_FILE"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    stage = args.stage or cfg["stage"]
    cfg["stage"] = stage
    if stage == "router":
        from stmem.training.routers import train_router

        result = train_router(cfg, resume=args.resume, stop_file=args.stop_file)
    else:
        result = train_policy(cfg, stage, resume=args.resume, stop_file=args.stop_file)
    raise SystemExit(result)


if __name__ == "__main__":
    main()
