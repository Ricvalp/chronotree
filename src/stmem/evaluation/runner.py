"""Streaming, action-chunk closed-loop execution with explicit denominators."""

from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
from stmem.memory.streaming import StreamingMemory, CurrentObservationMemory
from stmem.models.policy import Evidence, choose_records
from stmem.models.router import CachedScorer
from stmem.evaluation.timing import DecisionTimer, latency_summary, accelerator_memory, process_memory


@contextmanager
def episode_timeout(seconds):
    if not hasattr(signal, "SIGALRM") or seconds <= 0:
        yield
        return

    def timeout(*_):
        raise TimeoutError("Episode exceeded declared wall-clock timeout")

    old = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class OnlinePolicy:
    def __init__(
        self,
        policy,
        cfg,
        method,
        budget,
        *,
        router=None,
        router_hash="",
        selection_family=None,
        intervention="none",
    ):
        self.policy = policy.eval().requires_grad_(False)
        self.cfg = cfg
        self.method = method
        self.budget = budget
        self.observation_stride = cfg.get("data", {}).get("observation_stride", 1)
        from stmem.doctor import choose_precision

        self.precision = cfg["training"]["precision"]
        if self.precision == "auto":
            self.precision = choose_precision(policy.device)
        if policy.device.type == "cpu":
            self.precision = "fp32"
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
            self.precision
        ]
        if method == "PERCEIVER" and (policy.resampler is None or len(policy.resampler.latents) != budget):
            raise ValueError(
                "PERCEIVER requires its adapted checkpoint and exactly its trained latent budget"
            )
        m = cfg["model"]
        self.history_enabled = m.get("history_mode", "full") != "none"
        if not self.history_enabled and method not in ("FULL", "CURRENT"):
            raise ValueError("A current-only checkpoint supports FULL/CURRENT evaluation only")
        self.memory = (
            StreamingMemory(
                policy.encoder,
                m["block_size"],
                archive_device=m["archive_device"],
                max_execution_frames=m["max_execution_frames"],
                max_conditioning_frames=m["max_conditioning_frames"],
            )
            if self.history_enabled
            else CurrentObservationMemory()
        )
        if method not in ("FULL", "CURRENT", "PERCEIVER"):
            self.memory.validate_budget(budget)
        self.scorer = CachedScorer(router, router_hash) if router is not None else None
        if method in ("TREE", "FLAT") and self.scorer is None and selection_family is None:
            raise ValueError("Learned TREE/FLAT evaluation requires its trained router")
        self.selection_family = selection_family
        self.intervention = intervention
        self.goal = ""
        self.text = None
        self.clocks = {}
        self.update_seconds = 0.0
        self.decision_index = 0

    @torch.no_grad()
    def reset(self, reset, seed):
        with DecisionTimer(self.policy.device) as timer:
            conditioning = (
                [frame for frame in reset.conditioning if frame.source_index % self.observation_stride == 0]
                if self.history_enabled
                else []
            )
            if reset.current.source_index % self.observation_stride:
                raise ValueError("Reset current observation must lie on the declared sampling phase")
            self.memory.reset(conditioning, reset.current)
            self.goal = reset.goal
            self.text = self.policy.text_encoder(reset.goal)
            self.clocks = {
                "conditioning": conditioning[-1].time if conditioning else 0.0,
                "execution": reset.current.time if self.history_enabled else 0.0,
            }
            self.generator = torch.Generator(device=self.policy.device).manual_seed(seed)
            self.selection_rng = random.Random(seed)
            self.decision_index = 0
            self.update_seconds = 0.0
            if self.scorer:
                self.scorer.keys.clear()
                self.scorer.feature_transferred_bytes = self.scorer.key_transferred_bytes = 0
                self.scorer.query_transferred_bytes = self.scorer.keys_encoded = 0
        self.reset_encoding_seconds = timer.seconds

    @torch.no_grad()
    def observe(self, frame, preprocessing_seconds=0.0):
        if frame.source_index % self.observation_stride:
            # Full-rate injected adapters are supported without inserting their
            # dropped observations into the memory. Count their actual cost.
            self.update_seconds += preprocessing_seconds
            return
        with DecisionTimer(self.policy.device) as timer:
            self.memory.observe(frame)
        self.update_seconds += timer.seconds + preprocessing_seconds

    @torch.no_grad()
    def decide(self):
        policy = self.policy
        memory = self.memory
        previous_transfer = self.scorer.transferred_bytes if self.scorer else 0
        previous_keys = self.scorer.keys_encoded if self.scorer else 0
        with DecisionTimer(policy.device) as timer:
            with timer.component("prefix_and_current_encoding"):
                archive = memory.archive
                current = policy.encoder.current(memory.current)
                self.clocks["execution"] = memory.current.time if self.history_enabled else 0.0
                evidence = Evidence(archive, current, self.text, memory.current, self.clocks.copy())
                if self.intervention != "none":
                    from stmem.analyze import intervene_evidence

                    evidence = intervene_evidence(evidence, self.intervention, self.cfg["evaluation"]["seed"])
                    archive = evidence.archive
            with timer.component("routing_keys_and_query"):
                if self.scorer:
                    candidates = archive.records if self.method == "TREE" else archive.leaves
                    self.scorer.retain(candidates)
                    self.scorer.prepare(candidates)
                    self.scorer.set_query(policy.routing_query(evidence, self.budget))
            with timer.component("selection"):
                selected = choose_records(
                    evidence, self.method, self.budget, scorer=self.scorer, rng=self.selection_rng
                )
            with timer.component("selected_transfer_and_adapter"):
                with torch.autocast(
                    policy.device.type, dtype=self.autocast_dtype, enabled=self.precision != "fp32"
                ):
                    context, bias = policy.context(
                        evidence, selected.records, perceiver=self.method == "PERCEIVER"
                    )
            with timer.component("flow_decode"):
                with torch.autocast(
                    policy.device.type, dtype=self.autocast_dtype, enabled=self.precision != "fp32"
                ):
                    normalized = policy.decoder.integrate(
                        context, bias, steps=self.cfg["model"]["flow_steps"], generator=self.generator
                    )
                actions = policy.normalizer.denormalize(normalized)
            with timer.component("storage_accounting"):
                storage = memory.storage
                storage.update(process_memory())
        result = timer.as_dict()
        result.update(selected.stats)
        result.update(storage)
        result.update(
            {
                "history_tokens": len(selected.records)
                if self.method != "PERCEIVER"
                else min(self.budget, len(archive.leaves))
                if not archive.leaves
                else self.budget,
                "represented_leaf_counts": [r.count for r in selected.records],
                "resolution_depths": [r.depth for r in selected.records],
                "routing_key_bytes": self.scorer.storage_bytes if self.scorer else 0,
                "routing_transferred_bytes_cumulative": self.scorer.transferred_bytes if self.scorer else 0,
                "routing_transferred_bytes": self.scorer.transferred_bytes - previous_transfer
                if self.scorer
                else 0,
                "new_keys_encoded": self.scorer.keys_encoded - previous_keys if self.scorer else 0,
                "decoder_precision": self.precision,
                "frozen_encoder_precision": "fp32",
                "selected_transfer_payload_bytes": sum(
                    (r.content_sum.numel() + 23) * 4
                    for r in selected.records
                    if r.content_sum.device != policy.device
                ),
                "archive_fine_leaves": len(archive.leaves),
                "history_mode": "full" if self.history_enabled else "none",
                "observation_update_seconds": self.update_seconds,
                "complete_online_seconds": timer.seconds + self.update_seconds,
                "decision_index": self.decision_index,
            }
        )
        self.update_seconds = 0.0
        self.decision_index += 1
        return actions, result


def run_episode(
    adapter,
    online,
    *,
    task,
    episode_idx,
    split,
    seed,
    timeout_seconds=1800,
    video_path=None,
    checkpoint_hash="",
    router_hash="",
):
    """Invalid outputs and simulator errors remain failed planned episodes."""
    result = {
        "task": task,
        "episode_idx": episode_idx,
        "split": split,
        "status": "error",
        "method": online.method,
        "actions": [],
        "decisions": [],
        "error": None,
        "checkpoint_sha256": checkpoint_hash,
        "router_sha256": router_hash,
        "noise_seed": seed,
        "training_seed": online.cfg["seed"],
        "budget": online.budget,
    }
    result["intervention"] = online.intervention
    started = time.perf_counter()
    if torch.device(online.policy.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(online.policy.device)
    video = None
    try:
        with episode_timeout(timeout_seconds):
            if video_path:
                import imageio.v2 as imageio

                Path(video_path).parent.mkdir(parents=True, exist_ok=True)
                video = imageio.get_writer(video_path, fps=20)
            reset = adapter.reset()
            online.reset(reset, seed)
            result["environment"] = reset.metadata
            action_stride = online.cfg["model"].get("action_stride", 1)
            observation_stride = online.cfg.get("data", {}).get("observation_stride", 1)
            native_hz = reset.metadata.get("control_hz")
            result["temporal_sampling"] = {
                "native_control_hz": native_hz,
                "action_stride": action_stride,
                "observation_stride": observation_stride,
                "action_hz": None if native_hz is None else native_hz / action_stride,
                "observation_hz": None if native_hz is None else native_hz / observation_stride,
                "predicted_actions_per_chunk": online.cfg["model"]["horizon"],
                "replan_primitive_steps": online.cfg["model"]["execute"] * action_stride,
                "command_interpolation": "zero_order_hold",
                "clock": "native_source_local_recorded_frame",
            }
            nominal_hz = online.cfg.get("data", {}).get("nominal_control_hz")
            if (
                (action_stride > 1 or observation_stride > 1)
                and native_hz is not None
                and nominal_hz is not None
            ):
                if not np.isclose(native_hz, nominal_hz):
                    raise ValueError("Runtime control rate differs from the sampled training-data contract")
            result["startup_seconds"] = time.perf_counter() - started
            result["reset_encoding_seconds"] = online.reset_encoding_seconds
            if video:
                for rgb in adapter.last_rgb:
                    video.append_data(rgb)
            steps = 0
            effective = int(reset.metadata.get("effective_max_steps", online.cfg["evaluation"]["max_steps"]))
            # Wrapper owns its exact limit; a defensive bound must not extend it.
            while steps < effective:
                chunk, timing = online.decide()
                if chunk.shape != (online.cfg["model"]["horizon"], 8) or not torch.isfinite(chunk).all():
                    raise ValueError("Policy emitted nonfinite or incorrectly shaped action chunk")
                result["decisions"].append(timing)
                # The simulator keeps its native cadence. Coarse absolute
                # joint/gripper targets are held for action_stride native steps.
                commands = chunk[: online.cfg["model"]["execute"]].repeat_interleave(action_stride, dim=0)
                for action in commands:
                    step = adapter.step(action)
                    result["actions"].append(
                        np.asarray(step.command if step.command is not None else action.cpu()).tolist()
                    )
                    steps += 1
                    # Termination is checked BEFORE accessing potentially empty observations.
                    if step.done:
                        for frame in step.frames:
                            online.observe(
                                frame, getattr(adapter, "last_preprocess_seconds", 0.0) / len(step.frames)
                            )
                        if video:
                            for rgb in adapter.last_rgb:
                                video.append_data(rgb)
                        result["status"] = (
                            "success"
                            if step.status == "success"
                            else "error"
                            if step.status == "error"
                            else "timeout"
                            if step.truncated or step.status == "timeout"
                            else "fail"
                        )
                        result["terminal_status"] = step.status
                        result["error"] = step.error
                        return result
                    if not step.frames and not getattr(step, "observation_skipped", False):
                        raise RuntimeError("Ongoing simulator step returned no allowed observation")
                    if not step.frames:
                        online.update_seconds = getattr(online, "update_seconds", 0.0) + getattr(
                            adapter, "last_preprocess_seconds", 0.0
                        )
                    for frame in step.frames:
                        online.observe(
                            frame, getattr(adapter, "last_preprocess_seconds", 0.0) / len(step.frames)
                        )
                    if video:
                        for rgb in adapter.last_rgb:
                            video.append_data(rgb)
                    if steps >= effective:
                        result["status"] = "timeout"
                        return result
            result["status"] = "timeout"
    except TimeoutError as exc:
        result["status"] = "timeout"
        result["error"] = str(exc)
    except Exception as exc:
        result["status"] = "policy_error" if isinstance(exc, (ValueError, FloatingPointError)) else "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        if video:
            video.close()
        result["runtime_seconds"] = time.perf_counter() - started
        result["primitive_commands"] = len(result["actions"])
        result["latency"] = latency_summary([r["complete_online_seconds"] for r in result["decisions"]])
        result["accelerator_memory"] = accelerator_memory(online.policy.device)
    return result
