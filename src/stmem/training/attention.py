"""Fixed, target-free inference probes; never alter the training forward pass."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from stmem.models.attention_diagnostics import CrossAttentionDiagnostics, GROUP_NAMES
from stmem.models.policy import choose_records
from stmem.tracking import _preserve_rng


def context_groups(evidence, records, device, *, current_state_enabled=True):
    """Exactly the token order produced by MemoryPolicy.context, without padding."""
    ids = {name: index for index, name in enumerate(GROUP_NAMES)}
    groups = [ids["current_visual"]] * len(evidence.current_records)
    if current_state_enabled:
        groups.append(ids["current_state"])
    groups.extend(ids[name] for name in ("goal", "clock"))
    groups.extend(
        ids["history_visual" if record.modality == "visual" else "history_event"] for record in records
    )
    return torch.tensor(groups, dtype=torch.long, device=device)


def mean_metrics(rows):
    """Average only defined conditional entropies; mass includes empty groups."""
    result = {}
    for key in sorted({key for row in rows for key in row}):
        values = [row[key] for row in rows if row.get(key) is not None]
        if values:
            result[key] = sum(values) / len(values)
    return result


class PolicyAttention:
    def __init__(self, cfg, policy, tracker, scenes, *, stage, every, cache=None, precision="fp32"):
        self.cfg, self.policy, self.tracker, self.scenes = cfg, policy, tracker, scenes
        self.stage, self.every, self.cache, self.precision = stage, every, cache, precision
        self.enabled = tracker.enabled and every > 0 and stage in ("teacher", "compact")
        self.seconds, self.events = 0.0, 0
        if tracker.enabled:
            tracker.summary(
                {
                    "attention/protocol": {
                        "version": "fixed_inference_cross_attention_v1",
                        "enabled": self.enabled,
                        "every_optimizer_steps": every,
                        "examples": scenes.count,
                        "selection": "same_fixed_training_examples_as_pointcloud_diagnostics",
                        "synthetic_diagnostic": scenes.data.synthetic,
                        "flow_evaluation": "last_euler_step",
                        "averaging": "per_head_query_then_layers_then_scenes",
                        "encoder_precision": "fp32",
                        "decoder_precision": precision,
                        "mass_baseline": "uniform_over_valid_context_tokens",
                        "entropy": "conditional_within_group_divided_by_log_token_count",
                        "empty_group_entropy": "undefined_omitted",
                        "singleton_entropy": 0,
                        "perceiver_note": "disabled: resampled tokens mix visual and event modalities",
                    }
                }
            )

    def log(self, step):
        if not self.enabled:
            return
        started = time.perf_counter()
        policy = self.policy
        modes = [(module, module.training) for module in policy.modules()]
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.precision]
        method = "FULL" if self.stage == "teacher" else "UNIFORM"
        calls = 1 if self.cfg["model"].get("decoder") == "regression" else self.cfg["model"]["flow_steps"]
        examples = []
        with _preserve_rng():
            try:
                policy.eval()
                with torch.inference_mode():
                    if self.scenes.decisions is None:
                        self.scenes.decisions = self.scenes._examples()
                    for index, decision in enumerate(self.scenes.decisions):
                        inputs = replace(
                            decision,
                            target_chunk=torch.zeros_like(decision.target_chunk),
                            target_valid=torch.zeros_like(decision.target_valid),
                        )
                        # Recompute learned content at current weights. Only raw
                        # fixed inputs persist between probes, never stale tokens.
                        with torch.autocast(policy.device.type, enabled=False):
                            evidence = self.cache.evidence(inputs) if self.cache else policy.encode(inputs)
                        records = choose_records(evidence, method, self.cfg["model"]["budget"]).records
                        groups = context_groups(
                            evidence,
                            records,
                            policy.device,
                            current_state_enabled=policy.current_state_enabled,
                        )
                        noise_seed = self.cfg["seed"] * 1009 + index
                        noise = torch.randn(
                            (self.cfg["model"]["horizon"], 8),
                            generator=torch.Generator().manual_seed(noise_seed),
                        ).to(policy.device)
                        with (
                            torch.autocast(policy.device.type, dtype=dtype, enabled=self.precision != "fp32"),
                            CrossAttentionDiagnostics(
                                policy.decoder, groups, capture_on_call=calls
                            ) as capture,
                        ):
                            prediction = policy.predict(evidence, records, noise=noise)
                        if not torch.isfinite(prediction).all():
                            raise FloatingPointError("Nonfinite fixed-scene attention probe prediction")
                        examples.append(
                            {
                                "scene": index,
                                "task": decision.task,
                                "episode_id": decision.episode_id,
                                "decision_index": decision.index,
                                "conditioning_frames": len(decision.conditioning),
                                "execution_history_frames": len(decision.execution_prefix),
                                "fixed_noise_seed": noise_seed,
                                "context_method": method,
                                "context_tokens": len(groups),
                                "metrics": capture.metrics(),
                                "layers": capture.layer_metrics(),
                            }
                        )
            finally:
                for module, training in modes:
                    module.training = training
        # Metric conversion completes GPU work before this timing is recorded.
        elapsed = time.perf_counter() - started
        self.seconds += elapsed
        self.events += 1
        metrics = mean_metrics([example["metrics"] for example in examples])
        row = {
            "step": int(step),
            "attempt": self.tracker.identity.get("attempt"),
            "seconds": elapsed,
            "flow_time": None if self.cfg["model"].get("decoder") == "regression" else (calls - 1) / calls,
            "examples": examples,
            "metrics": metrics,
        }
        directory = Path(self.tracker.directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "attention.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.tracker.log(
            {
                **{f"attention/{key.replace('/', '_')}": value for key, value in metrics.items()},
                "attention/seconds": elapsed,
                "attention/scenes": len(examples),
            },
            step=step,
        )
        self.tracker.summary({"attention/seconds_total": self.seconds, "attention/events": self.events})
