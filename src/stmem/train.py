"""Bounded staged training with deterministic sampling and coherent safe resume."""

from __future__ import annotations
import argparse
from contextlib import ExitStack, nullcontext
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
from stmem.training.data import TrainingData, rank_positions
from stmem.training.loader import PolicyBatchLoader
from stmem.training.forks import COMMAND_FLAG, load_fork_checkpoint, schedule_steps, validate_teacher_fork
from stmem.training.logging_options import positive_log_every, positive_progress_seconds, resolve_logging
from stmem.training.progress import TrainingProgress, print_json
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


def train_policy(
    cfg,
    stage,
    *,
    resume=None,
    fork_from=None,
    fork_sha256=None,
    stop_file=None,
    log_every=None,
    progress_every_seconds=None,
):
    validate_config(cfg)
    if resume and fork_from:
        raise ValueError("Choose either resume or fork_from, not both")
    if fork_sha256 and not fork_from:
        raise ValueError("fork_sha256 requires fork_from")
    if fork_from and stage != "teacher":
        raise ValueError("Only teacher checkpoints support matched continuations")
    logging = resolve_logging(cfg, log_every, progress_every_seconds)
    rank = int(os.environ.get("RANK", "0"))
    directory = run_dir(cfg, stage)
    if fork_from:
        if rank == 0 and directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Teacher fork needs a new empty run directory: {directory}")
        # All ranks verify the same immutable input before model construction.
        actual_hash = file_sha256(fork_from)
        if fork_sha256 is not None and fork_sha256 != actual_hash:
            raise ValueError("Teacher fork checkpoint SHA256 mismatch")
        fork_sha256 = actual_hash
    if rank == 0 and not resume and (directory / "last.pt").exists():
        raise FileExistsError(f"Existing run: {directory}; use --resume explicitly")
    with RunTracker(
        cfg,
        directory,
        job_type=stage,
        rank=rank,
        resume=bool(resume),
        metadata={
            "resume_checkpoint": str(resume) if resume else None,
            "fork_checkpoint": str(fork_from) if fork_from else None,
            "fork_sha256": fork_sha256,
            "logging": logging,
        },
    ) as tracker:
        with TrainingProgress(directory, logging["progress_every_seconds"], rank=rank) as progress:
            progress.update("device_setup", stage=stage, run_attempt=tracker.identity.get("attempt"))
            with ExitStack() as resources:
                result = _train_policy(
                    cfg,
                    stage,
                    resume=resume,
                    fork_from=fork_from,
                    fork_sha256=fork_sha256,
                    stop_file=stop_file,
                    tracker=tracker,
                    logging=logging,
                    progress=progress,
                    resources=resources,
                )
        if result == 75:
            tracker.finish(status="checkpointed_partial", exit_code=75)
        return result


def _train_policy(
    cfg,
    stage,
    *,
    resume=None,
    fork_from=None,
    fork_sha256=None,
    stop_file=None,
    tracker,
    logging,
    progress,
    resources,
):
    from stmem.doctor import choose_precision
    from stmem.data.cache import FrozenCache

    startup_started = time.perf_counter()
    device, rank, world = distributed_setup()
    seed_all(cfg["seed"] + rank)
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    batch = cfg["training"]["effective_batch"]
    if batch % world:
        raise ValueError("Effective batch must be divisible by world size")
    local_n = batch // world
    microbatch = min(cfg["training"]["microbatch"], local_n)

    def startup_progress(phase, **values):
        if rank != 0:
            return
        record = {
            "event": "training_startup",
            "stage": stage,
            "phase": phase,
            "elapsed_seconds": time.perf_counter() - startup_started,
            **values,
        }
        print_json(record)
        progress.update("startup/" + phase)
        # Resume's optimizer position is not known yet. Publish summaries, not
        # history rows with an invented optimizer step, during initialization.
        tracker.summary({f"startup/{key}": value for key, value in record.items() if key != "event"})

    def data_progress(event):
        values = dict(event)
        phase = values.pop("phase")
        values["data_elapsed_seconds"] = values.pop("elapsed_seconds")
        startup_progress(phase, **values)

    data_started = time.perf_counter()
    data = TrainingData(cfg, progress=data_progress if rank == 0 else None)
    data_seconds = time.perf_counter() - data_started
    startup_progress("model_initializing", data_seconds=data_seconds)
    policy = MemoryPolicy(cfg)
    dependencies = {}
    cache = None
    normalizer_seconds = None
    if stage != "teacher":
        startup_progress("dependencies_loading")
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
        startup_progress("normalizer_started")
        normalizer_started = time.perf_counter()
        data.fit_normalizer(policy.normalizer)
        normalizer_seconds = time.perf_counter() - normalizer_started
        startup_progress("normalizer_ready", normalizer_seconds=normalizer_seconds)
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
    decay_steps = schedule_steps(cfg["training"])
    warmup = cfg["training"]["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min((s + 1) / max(warmup, 1), 1.0)
        * (
            0.1
            + 0.9 * (1 + math.cos(math.pi * min(max(s - warmup, 0) / max(decay_steps - warmup, 1), 1))) / 2
        ),
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
        "training_engine": "padded_policy_batch_v1",
        "sampler": data.sampler_version,
        "rank_assignment": "rotating_rank_v1",
    }
    fork = None
    if resume or fork_from:
        startup_progress("fork_loading" if fork_from else "resume_loading")
        saved = load_fork_checkpoint(fork_from, fork_sha256) if fork_from else load_checkpoint(resume)
        if fork_from:
            validate_teacher_fork(saved, cfg, provenance)
            fork = {
                "source_checkpoint": str(Path(fork_from).resolve()),
                "source_sha256": fork_sha256,
                "source_step": saved["step"],
                "source_tracking_id": saved.get("tracking", {}).get("id"),
                "changes": {
                    COMMAND_FLAG: {
                        "before": saved["config"]["model"].get(COMMAND_FLAG, True),
                        "after": cfg["model"].get(COMMAND_FLAG, True),
                    }
                },
                "schedule_steps": decay_steps,
                "intended_final_step": steps,
            }
        else:
            if saved["stage"] != stage or saved["provenance"] != provenance:
                raise ValueError("Resume stage/data/config/dependencies/world size mismatch")
            fork = saved.get("fork")
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
                "logging": logging,
                "fork": fork,
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
            "training/schedule_steps": decay_steps,
            "fork": fork,
            "training/world_size": world,
            "training/precision": precision,
            "training/microbatch_size_rank": microbatch,
            "training/microbatches_per_update_rank": math.ceil(local_n / microbatch),
            "training/encoder_block_batch_size": cfg["training"]["encoder_block_batch_size"],
            "model/parameters": sum(p.numel() for p in policy.parameters()),
            "model/trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
            "provenance": provenance,
            "environment": environment_manifest(),
            "logging": logging,
        }
    )
    records_seen = torch.zeros((), device=device, dtype=torch.long)
    allocated = int(os.environ.get("STMEM_ALLOCATED_GPUS", os.environ.get("STMEM_GPUS", world)))
    from stmem.visualization.diagnostics import PolicyScenes

    scenes = PolicyScenes(cfg, data, policy, tracker, stage=stage, cache=cache, precision=precision)
    startup_progress(
        "loader_starting",
        workers_per_rank=cfg["training"].get("workers", 0),
        prefetch_factor=cfg["training"].get("prefetch_factor", 1),
    )
    loader_started = time.perf_counter()
    loader = resources.enter_context(
        PolicyBatchLoader(
            data,
            start=start,
            steps=steps,
            batch=batch,
            rank=rank,
            world=world,
            microbatch=microbatch,
            retain_for_retry=precision == "fp16",
        )
    )
    startup_metrics = {
        "data_seconds": data_seconds,
        "loader_start_seconds": time.perf_counter() - loader_started,
        "total_seconds": time.perf_counter() - startup_started,
    }
    tracker.summary(
        {
            "training/loader_workers_per_rank": loader.workers,
            "training/loader_prefetch_factor": loader.prefetch_factor,
            "training/loader_prefetch_capacity_rank": loader.prefetch_capacity,
            "training/activation_checkpointing": policy.encoder.checkpoint_blocks,
        }
    )
    if normalizer_seconds is not None:
        startup_metrics["normalizer_seconds"] = normalizer_seconds
    startup_progress("training_ready", optimizer_step=start, **startup_metrics)
    progress.update("training_ready", optimizer_steps_completed=start)
    if rank == 0:
        # The checkpoint has now been validated and the optimizer position is
        # known. This row describes startup, not a completed optimizer update.
        tracker.log({f"startup/{key}": value for key, value in startup_metrics.items()}, step=start)
    started = time.perf_counter()

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
                "sampler": {
                    "next_global_position": step * batch,
                    "boundary": "optimizer_step",
                    "version": data.sampler_version,
                    "rank_assignment": "rotating_rank_v1",
                },
                "stage": stage,
                "config": cfg,
                "provenance": provenance,
                "source": source_revision(),
                "environment": environment_manifest(),
                "tracking": dict(tracker.identity),
                "logging": logging,
                "fork": fork,
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
        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        local_loss = torch.zeros((), device=device)
        update_leaves = torch.zeros((), device=device, dtype=torch.long)
        context_tokens = torch.zeros((), device=device, dtype=torch.long)
        history_lengths = []
        data_loading_seconds = 0.0
        worker_load_seconds = 0.0
        worker_pack_seconds = 0.0
        progress.begin_update(step, local_n)
        positions = rank_positions(step, batch, rank, world)
        prepared_batches = loader.read_update(step)
        for local_index in range(0, local_n, microbatch):
            batch_positions = positions[local_index : local_index + microbatch]
            progress.update(
                "loading_batch",
                example_in_update_rank=local_index + 1,
                global_sample_position=batch_positions[0],
                microbatch_size_rank=len(batch_positions),
                task=None,
                episode_id=None,
                decision_index=None,
                conditioning_frames=None,
                execution_history_frames=None,
            )
            loading_started = time.perf_counter()
            loaded = next(prepared_batches)
            data_loading_seconds += time.perf_counter() - loading_started
            worker_load_seconds += loaded.load_seconds
            worker_pack_seconds += loaded.pack_seconds
            packed = loaded.packed
            decisions = list(packed.metadata) if cache is None else packed.decisions()
            action = packed.target_chunk.to(device)
            valid = packed.target_valid.to(device)
            lengths = [d.conditioning_frames + d.execution_history_frames for d in packed.metadata]
            history_lengths.extend(lengths)
            progress.update(
                "cache_loading" if cache is not None else "encoding_history",
                batch_tasks=[d.task for d in decisions],
                history_frames_min_rank=min(lengths),
                history_frames_max_rank=max(lengths),
                task=decisions[0].task if len(decisions) == 1 else None,
                episode_id=decisions[0].episode_id if len(decisions) == 1 else None,
                decision_index=decisions[0].index if len(decisions) == 1 else None,
                conditioning_frames=packed.metadata[0].conditioning_frames if len(decisions) == 1 else None,
                execution_history_frames=packed.metadata[0].execution_history_frames
                if len(decisions) == 1
                else None,
            )
            evidences = None if cache is None else [cache.evidence(d) for d in decisions]
            context_records = None
            if stage == "compact":
                context_records = [
                    context_sample(policy, e, cfg, position)
                    for e, position in zip(evidences, batch_positions)
                ]
            sync = model.no_sync() if world > 1 and local_index + len(decisions) < local_n else nullcontext()
            # DDP's no_sync must cover both the forward and backward. Teacher
            # encoders run inside model.forward so DDP tracks their gradients.
            with sync:
                with torch.autocast(device.type, dtype=dtype, enabled=precision != "fp32"):
                    noisy, flow_time, target, _ = flow_sample(action, valid, policy.normalizer)
                    if cfg["model"].get("decoder") == "regression":
                        target = policy.normalizer.normalize(action).masked_fill(~valid, 0)
                    prediction, stats = model(
                        decisions,
                        noisy,
                        flow_time,
                        records=context_records,
                        evidence=evidences,
                        perceiver=stage == "perceiver",
                        return_stats=True,
                        progress=progress,
                        packed_batch=packed,
                    )
                    loss = masked_mse(prediction, target, valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at optimizer step {step}")
                progress.update("backward")
                # Equal example weight, including a smaller final microbatch.
                weight = len(decisions) / local_n
                scaler.scale(loss * weight).backward()
            local_loss += loss.detach() * weight
            update_leaves += stats["history_leaves"].sum()
            context_tokens += stats["history_context_tokens"].sum()
            progress.update("microbatch_complete", examples_completed_rank=local_index + len(decisions))
        records_seen += update_leaves
        progress.update("optimizer_step")
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
        step_seconds = time.perf_counter() - update_started
        progress.finish_update(current_step)
        checkpoint_due = (
            current_step % cfg["training"]["checkpoint_every"] == 0
            or current_step in cfg["training"]["checkpoint_steps"]
            or stopped
            or current_step == steps
        )
        if checkpoint_due:
            progress.update("checkpoint")
            save(current_step, current_step in cfg["training"]["checkpoint_steps"])
        if (
            current_step == start + 1
            or current_step % logging["log_every"] == 0
            or stopped
            or current_step == steps
        ):
            progress.update("logging")
            loss_tensor = local_loss.detach().clone()
            if world > 1:
                torch.distributed.all_reduce(loss_tensor)
                loss_tensor /= world
            elapsed = time.perf_counter() - started
            record = {
                "step": current_step,
                "training_action_loss": float(loss_tensor),
                "objective": cfg["model"]["decoder"],
                "seconds": elapsed,
                "step_seconds": step_seconds,
                "data_loading_seconds_rank": data_loading_seconds,
                "data_wait_seconds_rank": data_loading_seconds,
                "data_worker_load_seconds_rank": worker_load_seconds,
                "data_worker_pack_seconds_rank": worker_pack_seconds,
                "data_prefetch_workers_rank": loader.workers,
                "data_prefetch_capacity_rank": loader.prefetch_capacity,
                "examples": (current_step - start) * batch,
                "effective_batch": batch,
                "microbatch_size_rank": microbatch,
                "microbatches_per_update_rank": math.ceil(local_n / microbatch),
                "history_frames_mean_rank": sum(history_lengths) / local_n,
                "history_frames_max_rank": max(history_lengths),
                "history_leaves_per_update_rank": int(update_leaves),
                "history_context_tokens_mean_rank": float(context_tokens) / local_n,
                "examples_per_second": (current_step - start) * batch / elapsed,
                "allocated_gpu_hours": elapsed * allocated / 3600 if device.type == "cuda" else 0.0,
                "gpu_profile": cfg["profile"],
                "fine_leaves_processed_rank": int(records_seen),
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
                print_json(record)
                tracker.log({f"train/{key}": value for key, value in record.items()}, step=current_step)
        if checkpoint_due and not (stopped and current_step < steps):
            progress.update("scene_diagnostics")
            scenes.log(current_step)
        if stopped and current_step < steps:
            progress.update("checkpointed_partial")
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
    progress.update("complete", optimizer_steps_completed=steps)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=["teacher", "compact", "router", "perceiver"])
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--resume")
    initialization.add_argument(
        "--fork-from", help="Start a separate matched teacher continuation from this complete training state"
    )
    parser.add_argument("--fork-sha256", help="Expected SHA256 of the immutable fork source checkpoint")
    parser.add_argument("--stop-file", default=os.environ.get("STMEM_STOP_FILE"))
    parser.add_argument(
        "--log-every",
        type=positive_log_every,
        help="Log scalar metrics every N optimizer updates; runtime-only override, safe to change on resume",
    )
    parser.add_argument(
        "--progress-every-seconds",
        type=positive_progress_seconds,
        help="Policy-training wall-clock heartbeat interval for stdout/progress.jsonl (disabled by default)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    stage = args.stage or cfg["stage"]
    cfg["stage"] = stage
    if (args.fork_from or args.fork_sha256) and stage != "teacher":
        parser.error("--fork-from and --fork-sha256 apply only to teacher training")
    if stage == "router":
        from stmem.training.routers import train_router

        if args.progress_every_seconds is not None:
            parser.error("--progress-every-seconds applies to teacher, compact and perceiver training")
        result = train_router(cfg, resume=args.resume, stop_file=args.stop_file, log_every=args.log_every)
    else:
        result = train_policy(
            cfg,
            stage,
            resume=args.resume,
            fork_from=args.fork_from,
            fork_sha256=args.fork_sha256,
            stop_file=args.stop_file,
            log_every=args.log_every,
            progress_every_seconds=args.progress_every_seconds,
        )
    raise SystemExit(result)


if __name__ == "__main__":
    main()
