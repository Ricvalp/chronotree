"""Shared encoder/action-policy interface for all six readers."""

from __future__ import annotations
from dataclasses import dataclass
import math
import random
import torch
from torch import nn
from stmem.models.action_flow import ActionFlow, ActionNormalizer, RegressionHead
from stmem.models.event_encoder import command_conditioning
from stmem.models.metadata_adapter import MetadataAdapter
from stmem.models.text_encoder import TextEncoder
from stmem.models.perceiver import Perceiver
from stmem.models.router import query_features
from stmem.vision.frozen import resolve_visual_features


@dataclass
class Evidence:
    archive: object
    current_records: list
    text: torch.Tensor
    current: object
    clocks: dict


class MemoryPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        from stmem.models.spacetime_encoder import BlockEncoder

        m = config["model"]
        self.config = config
        self.condition_on_previous_joint_commands = m.get("condition_on_previous_joint_commands", True)
        self.condition_on_previous_commands = m.get("condition_on_previous_commands", True)
        self.proprioception_mode = m.get("proprioception_mode", "all")
        self.history_mode = m.get("history_mode", "full")
        if self.history_mode not in ("full", "none"):
            raise ValueError("model.history_mode must be full or none")
        self.history_enabled = self.history_mode == "full"
        self.clock_enabled = self.history_enabled
        if self.proprioception_mode not in ("all", "current_only", "none"):
            raise ValueError("model.proprioception_mode must be all, current_only, or none")
        if not isinstance(self.condition_on_previous_commands, bool):
            raise ValueError("model.condition_on_previous_commands must be boolean")
        if not self.history_enabled and (
            self.proprioception_mode != "none" or self.condition_on_previous_commands
        ):
            raise ValueError(
                "Current-only policies require proprioception_mode=none and "
                "condition_on_previous_commands=false"
            )
        self.current_state_enabled = self.proprioception_mode != "none" or self.condition_on_previous_commands
        self.encoder = BlockEncoder(
            width=m["width"],
            visual_tokens=m["visual_tokens"],
            block_size=m["block_size"],
            current_tokens=m["current_tokens"],
            variant=m.get("encoder", "joint"),
            checkpoint_blocks=m.get("checkpoint_blocks", False),
            condition_on_previous_joint_commands=self.condition_on_previous_joint_commands,
            condition_on_previous_commands=self.condition_on_previous_commands,
            proprioception_mode=self.proprioception_mode,
            history_enabled=self.history_enabled,
            visual_features=resolve_visual_features(config),
            **m.get("encoder_options", {}),
        )
        self.text_encoder = TextEncoder(
            m["width"], m["heads"], m.get("text_layers", 2), m.get("max_text_bytes", 256)
        )
        self.adapter = MetadataAdapter(m["width"])
        self.proprio = (
            nn.Sequential(nn.Linear(24, m["width"]), nn.SiLU(), nn.Linear(m["width"], m["width"]))
            if self.history_enabled
            else None
        )
        if self.proprio is not None:
            self.proprio.requires_grad_(self.current_state_enabled)
        self.clock = nn.Linear(2, m["width"]) if self.clock_enabled else None
        decoder = RegressionHead if m.get("decoder") == "regression" else ActionFlow
        self.decoder = decoder(m["width"], m["heads"], m["layers"], m["horizon"])
        self.normalizer = ActionNormalizer()
        self.resampler = None
        self.encoders_frozen = False

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze_encoders(self):
        self.encoders_frozen = True
        for module in (self.encoder, self.text_encoder):
            module.eval()
            module.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.encoders_frozen:
            self.encoder.eval()
            self.text_encoder.eval()
        return self

    def add_perceiver(self):
        m = self.config["model"]
        self.resampler = Perceiver(m["width"], m["heads"], m["budget"]).to(self.device)

    def encode(self, decision, archive=None, current_records=None, text=None):
        from stmem.memory.streaming import build_archive

        if not self.history_enabled:
            from stmem.memory.hierarchy import Archive

            archive = Archive()
        elif archive is None:
            archive = build_archive(
                decision.conditioning,
                decision.execution_prefix,
                self.encoder,
                self.config["model"]["block_size"],
            )
        if current_records is None:
            current_records = self.encoder.current(decision.current)
        if text is None:
            text = self.text_encoder(decision.goal)
        clocks = {
            "conditioning": decision.conditioning[-1].time
            if self.history_enabled and decision.conditioning
            else 0.0,
            "execution": decision.current.time if self.history_enabled else 0.0,
        }
        return Evidence(archive, current_records, text, decision.current, clocks)

    def routing_query(self, evidence, budget):
        if evidence.current_records:
            mean = torch.stack([r.content.to(self.device) for r in evidence.current_records]).mean(0)
        else:
            mean = torch.zeros(self.config["model"]["width"], device=self.device)
        return query_features(
            mean,
            evidence.text.to(self.device),
            evidence.current,
            evidence.clocks,
            budget,
            condition_on_previous_joint_commands=self.condition_on_previous_joint_commands,
            condition_on_previous_commands=self.condition_on_previous_commands,
            proprioception_mode=self.proprioception_mode,
        )

    def encode_current_state(self, state):
        """Shared current-state input contract for serial and packed policies."""
        if (
            not self.condition_on_previous_joint_commands
            or not self.condition_on_previous_commands
            or self.proprioception_mode == "none"
        ):
            state = torch.cat(
                [
                    state[..., :15]
                    if self.proprioception_mode != "none"
                    else torch.zeros_like(state[..., :15]),
                    command_conditioning(
                        state[..., 15:23],
                        include_joint_commands=self.condition_on_previous_joint_commands,
                        include_commands=self.condition_on_previous_commands,
                    ),
                    state[..., 23:]
                    if self.condition_on_previous_commands
                    else torch.zeros_like(state[..., 23:]),
                ],
                dim=-1,
            )
        return self.proprio(state)

    def context(self, evidence, records, *, perceiver=False):
        current = self.adapter(
            evidence.current_records, evidence.clocks, self.device, timeless=not self.history_enabled
        )
        if not self.history_enabled:
            if perceiver:
                raise ValueError("Current-only policies have no history to resample")
            current = torch.cat([current, evidence.text.to(self.device)[None]], 0)
            return current, torch.zeros(len(current), device=self.device)
        f = evidence.current
        state = torch.cat(
            [
                f.proprio.to(self.device),
                f.previous_action.to(self.device),
                torch.tensor([float(f.action_available)], device=self.device),
            ]
        ).float()
        clock = torch.tensor(
            [evidence.clocks["conditioning"] / 100.0, evidence.clocks["execution"] / 100.0],
            device=self.device,
        )
        pieces = [current]
        if self.current_state_enabled:
            pieces.append(self.encode_current_state(state)[None])
        pieces.extend([evidence.text.to(self.device)[None], self.clock(clock)[None]])
        current = torch.cat(pieces, 0)
        history = self.adapter(records, evidence.clocks, self.device)
        if self.config["model"].get("mass_bias", True):
            bias = torch.tensor([math.log(r.count) for r in records], device=self.device)
        else:
            bias = torch.zeros(len(records), device=self.device)
        if perceiver:
            if self.resampler is None:
                raise RuntimeError("PERCEIVER requires its separately adapted checkpoint")
            history = self.resampler(history, bias)
            bias = torch.zeros(len(history), device=self.device)
        return torch.cat([current, history]), torch.cat([torch.zeros(len(current), device=self.device), bias])

    def forward(
        self,
        decision,
        noisy_action,
        time,
        *,
        records=None,
        evidence=None,
        perceiver=False,
        return_stats=False,
        progress=None,
        packed_batch=None,
    ):
        if isinstance(decision, (list, tuple)):
            from stmem.models.batched_policy import batch_context

            if not decision:
                raise ValueError("A policy batch must contain at least one decision")
            if records is not None and evidence is None:
                raise ValueError("Batched selected records require the matching evidence banks")
            context, bias, stats = batch_context(
                self,
                decision,
                evidences=evidence,
                record_lists=records,
                perceiver=perceiver,
                progress=progress,
                packed_batch=packed_batch,
            )
            prediction = self.decoder(noisy_action, time, context, bias)
            return (prediction, stats) if return_stats else prediction
        if packed_batch is not None:
            raise ValueError("Packed policy inputs require a list of decisions")
        evidence = evidence if evidence is not None else self.encode(decision)
        records = evidence.archive.leaves if records is None else records
        context, bias = self.context(evidence, records, perceiver=perceiver)
        return self.decoder(noisy_action, time, context, bias)

    @torch.no_grad()
    def predict(self, evidence, records, *, noise=None, generator=None, perceiver=False):
        context, bias = self.context(evidence, records, perceiver=perceiver)
        action = self.decoder.integrate(
            context, bias, steps=self.config["model"]["flow_steps"], noise=noise, generator=generator
        )
        return self.normalizer.denormalize(action)


def choose_records(evidence, method, budget, *, scorer=None, rng=None):
    from stmem.memory.frontier import select_tree, select_uniform
    from stmem.memory.flat import select_flat
    from stmem.memory.frontier import Selection

    method = method.upper()
    if method in ("FULL", "PERCEIVER"):
        records = evidence.archive.leaves
        return Selection(
            records,
            {
                "method": method,
                "records_fetched": len(records),
                "nodes_scored": 0,
                "leaves_scored": 0,
                "active_tokens": len(records),
            },
        )
    if method == "CURRENT":
        return Selection([], {"method": method})
    if method == "UNIFORM":
        return select_uniform(evidence.archive.roots, budget)
    if scorer is None:
        rng = rng or random.Random(0)
        # A fixed random score per encountered record; never redraw prior tree scores.
        scores = {}

        def scorer(records):
            return [scores.setdefault(r.node_id, rng.random()) for r in records]

    if method == "TREE":
        return select_tree(evidence.archive.roots, budget, scorer)
    if method == "FLAT":
        return select_flat(evidence.archive.leaves, budget, scorer)
    raise ValueError(f"Unknown reader {method}")
