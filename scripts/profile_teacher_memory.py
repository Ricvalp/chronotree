#!/usr/bin/env python3
"""Standalone, bounded teacher memory diagnostic; never resumes or saves training checkpoints.

Uses the production model, causal data index, packed loader, loss and DDP batch
semantics. Instrumentation is scoped to this process. Warmup, measurements and
the optional detailed trace are reported separately.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack, nullcontext
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import functools
import json
import math
import os
from pathlib import Path
import secrets
import statistics
import time
from unittest.mock import patch

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from stmem.config import load_config, source_revision, validate_config
from stmem.doctor import choose_precision, precision_test
from stmem.models import batched_policy
from stmem.models.action_flow import flow_sample, masked_mse
from stmem.models.policy import MemoryPolicy
from stmem.tracking import RunTracker
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.data import TrainingData
from stmem.training.loader import PolicyBatchLoader
from stmem.training.progress import TrainingProgress, print_json
from stmem.training.runtime import distributed_setup, environment_manifest, seed_all


@dataclass(frozen=True)
class ProfileOptions:
    output: Path | None = None
    warmup_steps: int = 10
    steps: int = 5
    trace_steps: int = 1
    scenario: str = "representative"
    cpu: bool = False

    def validate(self):
        if not 5 <= self.warmup_steps <= 10:
            raise ValueError("Use 5–10 warmup steps; the default is 10.")
        if not 1 <= self.steps <= 100:
            raise ValueError("Use 1–100 measured steps; the default is 5.")
        if self.trace_steps not in (0, 1):
            raise ValueError("trace_steps must be 0 or 1.")
        if self.scenario not in ("representative", "long-history"):
            raise ValueError("scenario must be representative or long-history.")

    @property
    def total_steps(self):
        return self.warmup_steps + self.steps + self.trace_steps

    def mode(self, step):
        if step < self.warmup_steps:
            return "warmup"
        if step < self.warmup_steps + self.steps:
            return "measure"
        return "trace"


class ProfileTrainingData(TrainingData):
    """Optional stress sampler, isolated from the production task-balanced sampler.

    Representative mode delegates exactly to TrainingData. Long-history mode
    cycles the latest valid decisions in the longest 10% of episodes per task.
    Episode lengths are used for diagnostic sampling only, never as features.
    """

    def __init__(self, cfg, scenario="representative", *, progress=None):
        super().__init__(cfg, progress=progress)
        self.scenario = scenario
        self.long_candidates = {}
        if scenario == "long-history":
            for task in self.tasks:
                latest = {}
                for episode, index in self.candidates[task]:
                    latest[episode] = max(index, latest.get(episode, -1))
                lengths = {}
                for episode in latest:
                    if self.synthetic:
                        lengths[episode] = len(self.episode(episode).conditioning)
                    else:
                        lengths[episode] = self.dataset.load_episode_actions(
                            self.refs[episode]
                        ).conditioning_length
                ordered = sorted(latest.items(), key=lambda pair: (-(lengths[pair[0]] + pair[1]), pair[0]))
                self.long_candidates[task] = ordered[: max(1, math.ceil(len(ordered) * 0.1))]

    def sample_reference(self, position):
        if self.scenario == "representative":
            return super().sample_reference(position)
        if position < 0:
            raise ValueError("Global sample positions must be nonnegative")
        task = self.tasks[position % len(self.tasks)]
        pool = self.long_candidates[task]
        return pool[(position // len(self.tasks)) % len(pool)]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class MemoryProbe:
    """Allocator counters plus CUDA events, without per-phase synchronizations.

    Phase peaks include the tensors already live on entry. The update peak is
    the maximum over ALL allocator intervals, including gaps between phases.
    CUDA events are read only after the update's final synchronization.
    """

    def __init__(self, device, progress=None):
        self.device, self.progress = device, progress
        self.cuda = device.type == "cuda"
        self.phases = []
        self.active_phase = None
        self.failed_phase = None
        self.measurement_errors = []
        self.peak_allocated = self.peak_reserved = None
        self.detailed = False
        self.enabled = True

    def current(self):
        if not self.cuda:
            return {"allocated_bytes": None, "reserved_bytes": None}
        return {
            "allocated_bytes": torch.cuda.memory_allocated(self.device),
            "reserved_bytes": torch.cuda.memory_reserved(self.device),
        }

    def _harvest(self):
        if not self.cuda:
            return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        self.peak_allocated = max(self.peak_allocated or 0, allocated)
        self.peak_reserved = max(self.peak_reserved or 0, reserved)
        return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved}

    def begin_update(self, *, enabled=True, detailed=False):
        synchronize(self.device)
        self.enabled, self.detailed = enabled, detailed
        self.phases, self.active_phase = [], None
        self.failed_phase, self.measurement_errors = None, []
        self.peak_allocated = self.peak_reserved = None
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started = time.perf_counter()

    @contextmanager
    def phase(self, name):
        if self.progress is not None:
            self.progress.update("profile/" + name)
        if not self.enabled:
            self.active_phase = name
            try:
                yield
            except BaseException:
                self.failed_phase = name
                raise
            finally:
                self.active_phase = None
            return
        if self.active_phase is not None:
            raise RuntimeError(f"Profiling phases must not overlap: {self.active_phase}, {name}")
        self._harvest()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        before = self.current()
        begin = end = None
        if self.cuda:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        started = time.perf_counter()
        self.active_phase = name
        label = torch.profiler.record_function("teacher/" + name) if self.detailed else nullcontext()
        try:
            with label:
                yield
        except BaseException:
            self.failed_phase = name
            raise
        finally:
            try:
                if end is not None:
                    end.record()
                row = {
                    "phase": name,
                    "host_seconds": time.perf_counter() - started,
                    "start_allocated_bytes": before["allocated_bytes"],
                    "end_allocated_bytes": self.current()["allocated_bytes"],
                    **self._harvest(),
                    "_events": (begin, end),
                }
                self.phases.append(row)
            except BaseException as error:
                if self.failed_phase is None:
                    raise
                self.measurement_errors.append(f"{name}: {type(error).__name__}: {error}")
            finally:
                self.active_phase = None

    def finish_update(self):
        synchronize(self.device)
        seconds = time.perf_counter() - self.started
        self._harvest()
        for row in self.phases:
            begin, end = row.pop("_events")
            row["cuda_elapsed_ms"] = begin.elapsed_time(end) if begin is not None else None
        free = total = None
        if self.cuda:
            free, total = torch.cuda.mem_get_info(self.device)
        return {
            "seconds": seconds,
            "gpu_peak_allocated_bytes": self.peak_allocated,
            "gpu_peak_reserved_bytes": self.peak_reserved,
            "gpu_free_bytes_at_step_end": free,
            "gpu_total_bytes": total,
            **{"gpu_" + key: value for key, value in self.current().items()},
        }


@contextmanager
def instrument(policy, probe):
    """Temporary function/method wrappers; no model graph or source modifications."""

    def wrap(original, name, *, major=True):
        @functools.wraps(original)
        def call(*args, **kwargs):
            context = probe.phase(name) if major else torch.profiler.record_function("teacher/" + name)
            with context:
                return original(*args, **kwargs)

        return call

    with ExitStack() as stack:
        for name, label in (
            ("encode_history_bank", "history_encoder"),
            ("encode_current_bank", "current_encoder"),
        ):
            stack.enter_context(
                patch.object(batched_policy, name, wrap(getattr(batched_policy, name), label))
            )
        stack.enter_context(
            patch.object(
                batched_policy.TokenBank,
                "adapt",
                wrap(batched_policy.TokenBank.adapt, "metadata_adapter"),
            )
        )
        for module, label in ((policy.text_encoder, "text_encoder"), (policy.decoder, "decoder")):
            stack.enter_context(patch.object(module, "forward", wrap(module.forward, label)))
        if probe.detailed:
            for module, label in (
                (policy.encoder.visual.point_mlp, "visual/point_mlp"),
                (policy.encoder.visual.edge_mlp, "visual/edge_mlp"),
                (policy.encoder.event, "events"),
            ):
                stack.enter_context(patch.object(module, "forward", wrap(module.forward, label, major=False)))
            # These functions are imported by forward_batched in their own module.
            from stmem.models import batched_spacetime

            for name in ("_canonical_cloud", "_batched_fps", "canonical_nearest"):
                stack.enter_context(
                    patch.object(
                        batched_spacetime,
                        name,
                        wrap(getattr(batched_spacetime, name), "visual/" + name, major=False),
                    )
                )
        yield


@contextmanager
def detailed_trace(directory, device):
    """Bounded allocation history plus operator trace, separate from measurements."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    cuda = device.type == "cuda"
    if cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(
        activities=activities, profile_memory=True, record_shapes=False, with_stack=False
    )
    if cuda:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=100_000, clear_history=True
        )
    original_error = None
    try:
        with profiler:
            yield
    except BaseException as error:
        original_error = error
        raise
    finally:
        # Preserve the training error even if a damaged CUDA context prevents
        # export. Try every export and always disable allocation history.
        exports = [lambda: profiler.export_chrome_trace(str(directory / "trace.json"))]
        if cuda:
            exports.extend(
                [
                    lambda: torch.cuda.memory._dump_snapshot(str(directory / "memory_snapshot.pickle")),
                    lambda: torch.cuda.memory._record_memory_history(enabled=None),
                ]
            )
        failures = []
        for export in exports:
            try:
                export()
            except Exception as error:
                failures.append(error)
        if failures:
            details = [f"{type(error).__name__}: {error}" for error in failures]
            try:
                atomic_json({"errors": details}, directory / "trace_export_errors.json")
            except OSError:
                pass
            if original_error is None:
                raise failures[0]
            original_error.add_note(f"Trace export errors: {details}")


def make_optimizer(policy, cfg):
    """Match train._train_policy; the profiling budget does not change the LR schedule."""
    t = cfg["training"]
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=t["lr"], weight_decay=t["weight_decay"]
    )
    steps, warmup = t["steps"], t["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min((s + 1) / max(warmup, 1), 1.0)
        * (0.1 + 0.9 * (1 + math.cos(math.pi * min(max(s - warmup, 0) / max(steps - warmup, 1), 1))) / 2),
    )
    return optimizer, scheduler


def run_update(policy, model, loader, optimizer, scheduler, scaler, probe, cfg, step, world, precision):
    device = probe.device
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    local_n = cfg["training"]["effective_batch"] // world
    optimizer.zero_grad(set_to_none=True)
    batches = loader.read_update(step)
    local_loss = torch.zeros((), device=device)
    history_counts, current_counts = [], []
    examples, done = [], 0
    padded_max = 0
    with instrument(policy, probe):
        while done < local_n:
            with probe.phase("data_wait"):
                loaded = next(batches)
            packed = loaded.packed
            decisions = list(packed.metadata)
            n = len(decisions)
            padded = (
                max(packed.history.blocks_per_example)
                * (policy.encoder.visual_tokens + policy.encoder.block_size)
                + policy.encoder.current_tokens
                + 3
            )
            padded_max = max(padded_max, padded)
            examples.extend(
                {
                    "task": d.task,
                    "episode_id": d.episode_id,
                    "decision_index": d.index,
                    "conditioning_frames": d.conditioning_frames,
                    "execution_history_frames": d.execution_history_frames,
                    "padded_context_slots": padded,
                }
                for d in decisions
            )
            sync = model.no_sync() if world > 1 and done + n < local_n else nullcontext()
            with sync:
                with torch.autocast(device.type, dtype=dtype, enabled=precision != "fp32"):
                    with probe.phase("action_transfer_and_flow"):
                        action = packed.target_chunk.to(device)
                        valid = packed.target_valid.to(device)
                        noisy, flow_time, target, _ = flow_sample(action, valid, policy.normalizer)
                        if cfg["model"].get("decoder") == "regression":
                            target = policy.normalizer.normalize(action).masked_fill(~valid, 0)
                    prediction, stats = model(
                        decisions, noisy, flow_time, return_stats=True, packed_batch=packed
                    )
                    loss = masked_mse(prediction, target, valid)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite profiling loss at update {step}")
                with probe.phase("backward"):
                    scaler.scale(loss * (n / local_n)).backward()
            local_loss += loss.detach() * (n / local_n)
            history_counts.append(stats["history_context_tokens"].detach())
            current_counts.append(stats["current_visual_tokens"].detach() + 3)
            done += n
        with probe.phase("optimizer"):
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), cfg["training"]["grad_clip"], error_if_nonfinite=precision != "fp16"
            )
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            updated = precision != "fp16" or scaler.get_scale() >= old_scale
            if updated:
                scheduler.step()
    memory = probe.finish_update()
    # Scalar transfers happen after the timed/synchronized update.
    history = torch.cat(history_counts)
    current = torch.cat(current_counts)
    lengths = [e["conditioning_frames"] + e["execution_history_frames"] for e in examples]
    return {
        **memory,
        "optimizer_step": step + 1,
        "optimizer_updated": updated,
        "loss": float(local_loss),
        "grad_norm_before_clip": float(grad_norm),
        "learning_rate": optimizer.param_groups[0]["lr"],
        "examples_rank": done,
        "examples": examples,
        "history_frames_mean": statistics.mean(lengths),
        "history_frames_max": max(lengths),
        "history_frames_total": sum(lengths),
        "history_tokens_mean": float(history.float().mean()),
        "history_tokens_max": int(history.max()),
        "current_aux_tokens_mean": float(current.float().mean()),
        "context_tokens_mean": float((history + current).float().mean()),
        "context_tokens_max": int((history + current).max()),
        "padded_context_slots_max": padded_max,
    }


def gather_rows(row, world):
    if world == 1:
        return [row]
    rows = [None] * world
    torch.distributed.all_gather_object(rows, row)
    return rows


def summarize(rows):
    result = {}
    for mode in ("warmup", "measure", "trace"):
        selected = [r for r in rows if r["mode"] == mode and r["optimizer_updated"]]
        if not selected:
            continue
        group = {
            "updates": len(selected),
            "seconds_mean": statistics.mean(r["seconds"] for r in selected),
            "seconds_max": max(r["seconds"] for r in selected),
            "context_tokens_mean": statistics.mean(r["context_tokens_mean"] for r in selected),
            "context_tokens_max": max(r["context_tokens_max"] for r in selected),
            "history_frames_max": max(r["history_frames_max"] for r in selected),
            "padded_context_slots_max": max(r["padded_context_slots_max"] for r in selected),
        }
        for field, reduction in (
            ("gpu_peak_allocated_bytes", max),
            ("gpu_peak_reserved_bytes", max),
            ("gpu_free_bytes_at_step_end", min),
            ("gpu_total_bytes", min),
        ):
            values = [r[field] for r in selected if r[field] is not None]
            group[field] = reduction(values) if values else None
        result[mode] = group
    return result


def write_csv(path, rows):
    if not rows:
        return
    keys = [k for k in rows[0] if k != "examples"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def publish_update(tracker, gathered, mode, step):
    metrics = {}
    for row in gathered:
        prefix = f"profile/{mode}/rank{row['rank']}/"
        for key in (
            "seconds",
            "loss",
            "gpu_peak_allocated_bytes",
            "gpu_peak_reserved_bytes",
            "gpu_free_bytes_at_step_end",
            "context_tokens_mean",
            "context_tokens_max",
            "padded_context_slots_max",
            "history_frames_max",
        ):
            if row[key] is not None:
                metrics[prefix + key] = row[key]
    for field in ("gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
        values = [r[field] for r in gathered if r[field] is not None]
        if values:
            metrics[f"profile/{mode}/{field}_max_across_ranks"] = max(values)
    metrics[f"profile/{mode}/slowest_rank_seconds"] = max(r["seconds"] for r in gathered)
    tracker.log(metrics, step=step)


def run_profile(cfg, options):
    options.validate()
    validate_config(cfg)
    if cfg.get("stage", "teacher") != "teacher":
        raise ValueError("This profiler supports the full-history teacher stage only.")
    if cfg["data"].get("split", "train") != "train":
        raise ValueError("Profiling uses training episodes only, never validation/test seeds.")
    tracking = cfg.get("tracking", {})
    if (
        not tracking.get("enabled", True)
        or (os.environ.get("WANDB_MODE") or tracking.get("mode")) == "disabled"
    ):
        raise ValueError("Profiling requires W&B logging. Use online or offline mode.")
    if options.cpu and not cfg.get("debug", False):
        raise ValueError("--cpu is for explicit debug diagnostics, not GPU memory estimates.")
    try:
        return _run_profile(cfg, options)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _run_profile(cfg, options):
    if options.cpu:
        device = torch.device("cpu")
        rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
        if world > 1:
            torch.distributed.init_process_group("gloo")
        torch.set_num_threads(2)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; --cpu with a debug config only checks profiler plumbing.")
        device, rank, world = distributed_setup()
    batch = cfg["training"]["effective_batch"]
    if batch % world:
        raise ValueError("Global effective batch must be divisible by world size.")
    local_n = batch // world
    microbatch = min(cfg["training"]["microbatch"], local_n)
    output = options.output
    if output is None:
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
        output = Path(cfg["paths"]["run_root"]) / "profiles" / ("teacher-" + name)
    names = [str(output.resolve()) if rank == 0 else None]
    if world > 1:
        torch.distributed.broadcast_object_list(names, src=0)
    output = Path(names[0])
    creation_error = None
    creation_status = [None]
    if rank == 0:
        try:
            output.mkdir(parents=True, exist_ok=False)
            atomic_json(cfg, output / "config.json")
        except Exception as error:
            creation_error = error
            creation_status[0] = f"{type(error).__name__}: {error}"
    if world > 1:
        torch.distributed.broadcast_object_list(creation_status, src=0)
    if creation_status[0] is not None:
        if creation_error is not None:
            raise creation_error
        raise RuntimeError(f"Rank 0 could not create profiling output: {creation_status[0]}")
    rank_dir = output / f"rank{rank}"
    rank_dir.mkdir(exist_ok=False)
    seed_all(cfg["seed"] + rank)
    rows, phase_rows = [], []
    probe = MemoryProbe(device)
    step, attempt, mode = 0, 0, "startup"
    try:
        with (
            RunTracker(
                cfg,
                output,
                job_type="teacher-memory-profile",
                name=output.name,
                rank=rank,
                metadata={
                    "diagnostic": True,
                    "scenario": options.scenario,
                    "synthetic": cfg["data"].get("synthetic", False),
                },
            ) as tracker,
            TrainingProgress(rank_dir, 30, rank=rank) as progress,
        ):
            probe.progress = progress

            def startup(event):
                progress.update("profile/startup/" + event["phase"])
                if rank == 0:
                    print_json({"event": "profile_startup", **event})

            data = ProfileTrainingData(cfg, options.scenario, progress=startup)
            policy = MemoryPolicy(cfg)
            data.fit_normalizer(policy.normalizer)
            policy.to(device).train()
            precision = cfg["training"]["precision"]
            if device.type == "cpu":
                precision = "fp32"
            elif precision == "auto":
                precision = choose_precision(device)
            else:
                tested = precision_test(
                    device,
                    getattr(torch, {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}[precision]),
                )
                if not tested["ok"]:
                    raise RuntimeError(f"Requested precision test failed: {tested}")
            scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
            optimizer, scheduler = make_optimizer(policy, cfg)
            model = (
                DDP(
                    policy,
                    device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=True,
                )
                if world > 1
                else policy
            )
            manifest = {
                "kind": "standalone_teacher_memory_diagnostic",
                "synthetic": data.synthetic,
                "scenario": options.scenario,
                "options": {**asdict(options), "output": str(output)},
                "source": source_revision(),
                "profiler_script_sha256": file_sha256(Path(__file__)),
                "environment": environment_manifest(),
                "data_identity": data.identity,
                "precision": precision,
                "world_size": world,
                "effective_batch": batch,
                "microbatch_rank": microbatch,
                "parameters": sum(p.numel() for p in policy.parameters()),
                "initialization": "fresh weights; no training checkpoint read or written",
                "sampler": data.sampler_version
                if options.scenario == "representative"
                else "latest_valid_decision_in_longest_10_percent_of_episodes_per_task",
                "memory_history_max_entries": 100_000 if options.trace_steps and device.type == "cuda" else 0,
                "notes": [
                    "Warmup and detailed trace are excluded from measured-step summaries.",
                    "Per-phase peaks include memory already live on entry; do not add phase peaks.",
                    "Phase GPU durations use stream events; overlapping stream durations are not additive.",
                    "CUDA allocator statistics exclude allocations made outside PyTorch (e.g. NCCL).",
                    "Free memory is sampled at step end, not at the exact peak.",
                    "Detailed tracing adds overhead; its time/memory is reported separately.",
                    "CPU debug memory fields are null and cannot estimate GPU capacity.",
                    "No checkpoint writes or point-cloud scene logging are included in update timing.",
                ],
            }
            if rank == 0:
                atomic_json(manifest, output / "manifest.json")
                tracker.summary({"profile/manifest": manifest, "profile/output": str(output)})
            if world > 1:
                torch.distributed.barrier()
            with (
                PolicyBatchLoader(
                    data,
                    start=0,
                    steps=options.total_steps,
                    batch=batch,
                    rank=rank,
                    world=world,
                    microbatch=microbatch,
                    retain_for_retry=precision == "fp16",
                ) as loader,
                (rank_dir / "steps.jsonl").open("w") as step_stream,
                (rank_dir / "phases.jsonl").open("w") as phase_stream,
            ):
                step, attempt, overflows = 0, 0, 0
                while step < options.total_steps:
                    mode = options.mode(step)
                    # Keep completed baseline measurements usable even if the
                    # subsequent, higher-overhead trace fails.
                    if mode == "trace":
                        atomic_json(summarize(rows), rank_dir / "measurements.json")
                    progress.begin_update(step, local_n)
                    attempt += 1
                    trace = detailed_trace(rank_dir, device) if mode == "trace" else nullcontext()
                    with trace:
                        probe.begin_update(enabled=mode != "warmup", detailed=mode == "trace")
                        row = run_update(
                            policy,
                            model,
                            loader,
                            optimizer,
                            scheduler,
                            scaler,
                            probe,
                            cfg,
                            step,
                            world,
                            precision,
                        )
                    row.update(rank=rank, mode=mode, attempt=attempt)
                    rows.append(row)
                    step_stream.write(json.dumps(row) + "\n")
                    step_stream.flush()
                    for phase in probe.phases:
                        record = {
                            "rank": rank,
                            "mode": mode,
                            "optimizer_step": step + 1,
                            "attempt": attempt,
                            **phase,
                        }
                        phase_rows.append(record)
                        phase_stream.write(json.dumps(record) + "\n")
                    phase_stream.flush()
                    gathered = gather_rows(row, world)
                    if rank == 0:
                        publish_update(tracker, gathered, mode, step + 1)
                        print_json(
                            {
                                "event": "profile_update",
                                "mode": mode,
                                "optimizer_step": step + 1,
                                "optimizer_updated": row["optimizer_updated"],
                                "slowest_rank_seconds": max(r["seconds"] for r in gathered),
                                "peak_allocated_bytes_by_rank": [
                                    r["gpu_peak_allocated_bytes"] for r in gathered
                                ],
                            }
                        )
                    if row["optimizer_updated"]:
                        step += 1
                        overflows = 0
                        progress.finish_update(step)
                    else:
                        overflows += 1
                        if overflows >= 8:
                            raise FloatingPointError("Eight consecutive FP16 overflows in profiling.")
            local_summary = {
                "rank": rank,
                "device": str(device),
                "phases": sorted({p["phase"] for p in phase_rows}),
                **summarize(rows),
            }
            atomic_json(local_summary, rank_dir / "summary.json")
            write_csv(rank_dir / "steps.csv", rows)
            write_csv(rank_dir / "phases.csv", phase_rows)
            all_summaries = gather_rows(local_summary, world)
            summary = {
                "status": "completed",
                "kind": manifest["kind"],
                "synthetic": data.synthetic,
                "scenario": options.scenario,
                "output": str(output),
                "warmup_steps": options.warmup_steps,
                "measured_steps": options.steps,
                "trace_steps": options.trace_steps,
                "world_size": world,
                "ranks": all_summaries,
                "max_measured_peak_allocated_bytes_across_ranks": max(
                    (r["measure"]["gpu_peak_allocated_bytes"] for r in all_summaries),
                    default=None,
                )
                if device.type == "cuda"
                else None,
            }
            if rank == 0:
                atomic_json(summary, output / "summary.json")
                tracker.summary({"profile/result": summary})
                print_json({"event": "profile_complete", "output": str(output)})
            return summary
    except BaseException as error:
        failure = {
            "status": "failed",
            "rank": rank,
            "mode": mode,
            "optimizer_step": step + 1,
            "attempt": attempt,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_phase": probe.failed_phase,
            "completed_attempts": len(rows),
            "measurement_errors": probe.measurement_errors,
        }
        try:
            probe._harvest()
            failure.update(
                **probe.current(),
                peak_allocated_bytes=probe.peak_allocated,
                peak_reserved_bytes=probe.peak_reserved,
            )
            if device.type == "cuda" and isinstance(error, torch.cuda.OutOfMemoryError):
                # No allocation history is enabled for baseline measurements;
                # this snapshot still shows the allocator state at failure.
                torch.cuda.memory._dump_snapshot(str(rank_dir / "oom_snapshot.pickle"))
        except Exception as snapshot_error:
            failure["snapshot_error"] = f"{type(snapshot_error).__name__}: {snapshot_error}"
        try:
            atomic_json(failure, rank_dir / "failure.json")
            write_csv(rank_dir / "steps.csv", rows)
            write_csv(rank_dir / "phases.csv", phase_rows)
        except Exception as export_error:
            error.add_note(f"Could not save failure diagnostics: {export_error}")
        # Never publish a successful summary or write to a teacher stage directory.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/teacher.yaml")
    parser.add_argument(
        "--output", type=Path, help="Fresh output directory; default RUN_ROOT/profiles/unique-name."
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=10, help="Warmup optimizer updates (5–10; default 10)."
    )
    parser.add_argument(
        "--steps", type=int, default=5, help="Measured optimizer updates after warmup (default 5)."
    )
    parser.add_argument("--trace-steps", type=int, choices=(0, 1), default=1)
    parser.add_argument("--scenario", choices=("representative", "long-history"), default="representative")
    parser.add_argument(
        "--cpu", action="store_true", help="Explicit debug plumbing check; no GPU memory results."
    )
    args = parser.parse_args()
    options = ProfileOptions(**{key: value for key, value in vars(args).items() if key != "config"})
    options.validate()
    run_profile(load_config(args.config), options)


if __name__ == "__main__":
    main()
