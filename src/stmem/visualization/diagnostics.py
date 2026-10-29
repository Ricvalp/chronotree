"""Two fixed training examples, evaluated only at checkpoint boundaries."""

from __future__ import annotations

from dataclasses import replace
import time
import warnings

import torch

from stmem.models.policy import choose_records
from stmem.tracking import _preserve_rng
from stmem.visualization.scenes import build_action_scene


def fixed_policy_examples(data, count=2):
    """Load fixed probes without changing the training sampler or episode LRU."""
    by_episode = {}
    for task in data.tasks:
        for episode, index in data.candidates[task]:
            by_episode.setdefault(episode, []).append(index)
    episodes = list(by_episode)
    chosen = episodes[:1] if count == 1 else episodes[:: max(1, len(episodes) - 1)]
    history_enabled = getattr(data, "history_enabled", True)
    loaded_episode = None
    loaded_index = None

    def example(episode, index):
        nonlocal loaded_episode, loaded_index
        if not history_enabled and not data.synthetic:
            return data.dataset.load_current_decisions(data.refs[episode], [index], data.horizon)[0]
        if loaded_index == episode:
            loaded = loaded_episode
        elif data.synthetic:
            from stmem.data.dataset import synthetic_episode

            task, number, seed = data.refs[episode]
            options = data.cfg["data"]
            loaded = synthetic_episode(
                options.get("synthetic_execution_frames", 12),
                options.get("synthetic_conditioning_frames", 5),
                data.cfg["model"]["points"],
                seed,
            )
            loaded.task, loaded.episode_id = task, f"episode_{number}"
            loaded.observation_stride = getattr(data, "observation_stride", 1)
            loaded.action_stride = getattr(data, "action_stride", 1)
        else:
            loaded = data.dataset.load_episode(data.refs[episode])
        loaded_episode, loaded_index = loaded, episode
        decision = loaded.decision(index, data.horizon)
        return decision if history_enabled else replace(decision, conditioning=[], execution_prefix=[])

    examples = []
    for episode in chosen:
        candidates = by_episode[episode]
        # Fixed early decision, independent of model results. Full-history models
        # retain every available frame; current-only models read just this row.
        index = next((i for i in candidates if i >= 2 * data.horizon), candidates[0])
        examples.append(example(episode, index))
    if len(examples) < count and len(episodes) == 1:
        alternatives = [i for i in by_episode[episodes[0]] if i != examples[0].index]
        if alternatives:
            examples.append(example(episodes[0], alternatives[0]))
    return examples


class PolicyScenes:
    def __init__(self, cfg, data, policy, tracker, *, stage, cache=None, precision="fp32"):
        options = cfg.get("tracking", {}).get("scenes", {})
        self.enabled = tracker.enabled and options.get("enabled", True)
        self.cfg, self.data, self.policy, self.tracker = cfg, data, policy, tracker
        self.stage, self.cache, self.precision = stage, cache, precision
        self.count, self.max_points = options.get("count", 2), options.get("max_points", 1024)
        self.decisions = None
        self.failed = set()
        self.seconds = 0.0
        self.events = 0
        if self.enabled and data.synthetic:
            self.enabled = False
            tracker.summary(
                {"diagnostics/scenes_skipped": "Synthetic proprioception has no calibrated Panda TCP"}
            )

    def _examples(self):
        return fixed_policy_examples(self.data, self.count)

    def log(self, step):
        if not self.enabled:
            return
        started = time.perf_counter()
        modes = [(module, module.training) for module in self.policy.modules()]
        scenes, skipped = {}, {}
        method = {"teacher": "FULL", "compact": "UNIFORM", "perceiver": "PERCEIVER"}[self.stage]
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.precision]
        with _preserve_rng():
            try:
                self.policy.eval()
                with (
                    torch.inference_mode(),
                    torch.autocast(self.policy.device.type, dtype=dtype, enabled=self.precision != "fp32"),
                ):
                    if self.decisions is None:
                        self.decisions = self._examples()
                    for index, decision in enumerate(self.decisions):
                        if index in self.failed:
                            continue
                        # Targets are used only by the visualization, never inference.
                        inputs = replace(
                            decision,
                            target_chunk=torch.zeros_like(decision.target_chunk),
                            target_valid=torch.zeros_like(decision.target_valid),
                        )
                        # Online/cached evidence is FP32; autocast only the reader
                        # and action decoder, as in the deployed policy path.
                        with torch.autocast(self.policy.device.type, enabled=False):
                            evidence = (
                                self.cache.evidence(inputs) if self.cache else self.policy.encode(inputs)
                            )
                        records = choose_records(evidence, method, self.cfg["model"]["budget"]).records
                        noise = torch.randn(
                            (self.cfg["model"]["horizon"], 8),
                            generator=torch.Generator().manual_seed(self.cfg["seed"] * 1009 + index),
                        ).to(self.policy.device)
                        prediction = (
                            self.policy.predict(
                                evidence, records, noise=noise, perceiver=self.stage == "perceiver"
                            )
                            .float()
                            .cpu()
                        )
                        try:
                            scene = build_action_scene(decision, prediction, max_points=self.max_points)
                        except ValueError as exc:
                            # Bad calibration/malformed targets must be visible; don't
                            # emit a misleading plot or retry every checkpoint forever.
                            if torch.isfinite(prediction).all():
                                self.failed.add(index)
                            skipped[str(index)] = str(exc)
                            warnings.warn(f"Point-cloud diagnostic {index} skipped: {exc}", stacklevel=2)
                            continue
                        scene.metadata.update(
                            context_method=method,
                            context_budget=self.cfg["model"]["budget"],
                            context_records=len(records),
                            fixed_noise_seed=self.cfg["seed"] * 1009 + index,
                            diagnostic="training_examples_only",
                            optimizer_step=int(step),
                        )
                        scenes[index] = scene
                media = self.tracker.log_scenes(scenes, step=step)
            finally:
                # Preserve even deliberately mixed submodule train/eval modes.
                for module, training in modes:
                    module.training = training
        elapsed = time.perf_counter() - started
        self.seconds += elapsed
        self.events += 1
        self.tracker.log(
            {
                "diagnostics/scenes_seconds": elapsed,
                "diagnostics/scenes_media_seconds": media["seconds"],
                "diagnostics/scenes_bytes": media["bytes"],
                "diagnostics/scenes_logged": len(scenes),
                "diagnostics/scenes_skipped": skipped,
            },
            step=step,
        )
        self.tracker.summary(
            {
                "diagnostics/scenes_seconds_total": self.seconds,
                "diagnostics/scenes_events": self.events,
                "diagnostics/scenes_failed": sorted(self.failed),
            }
        )
