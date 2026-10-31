"""Bounded real-training-state inspection of immutable checkpoints.

No optimizer update, simulator execution, test seed or benchmark score. All
source histories obey each checkpoint's own observation/action cadence.
"""

from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from stmem.analyze import intervene_evidence
from stmem.data.canonical import action_alignment_offset
from stmem.data.dataset import EpisodeDataset
from stmem.evaluation.robomme_adapter import ResetResult, physical_action
from stmem.evaluation.runner import OnlinePolicy
from stmem.models.action_flow import flow_sample, masked_mse
from stmem.models.batched_policy import batch_context, pack_policy_batch
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import load_policy
from stmem.visualization.kinematics import panda_tcp_positions
from stmem.inspection.offline_variants import definitions, iter_variants

DECISIONS = [0, 32, 64, 96, 128]
SEEDS = [1701, 1702, 1703, 1704]


@dataclass
class Inspection:
    manifest: dict
    manifest_sha256: str

    @property
    def root(self):
        return Path(self.manifest["output"]) / "offline"

    @property
    def labels(self):
        return self.manifest["labels"]

    @property
    def device(self):
        return self.manifest.get("device", "cuda:0")

    def asset(self, name):
        return Path(self.manifest["assets"][name])

    def checkpoint(self, label):
        return Path(self.manifest["checkpoints"][label]["path"])

    def record_identity(self, report):
        report["inspection_manifest_sha256"] = self.manifest_sha256
        return report

    def validate_checkpoint(self, label):
        expected = self.manifest["checkpoints"][label]
        if file_sha256(self.checkpoint(label)) != expected["sha256"]:
            raise ValueError(f"Checkpoint changed: {label}")

    def reuse(self, path, resume):
        if not path.exists():
            return False
        value = json.loads(path.read_text())
        if value.get("inspection_manifest_sha256") != self.manifest_sha256:
            raise ValueError(f"Output belongs to different inspection inputs: {path}")
        if not resume:
            raise FileExistsError(f"Output exists; use --resume to reuse completed checks: {path}")
        return value.get("status") in ("complete", "passed")


def data_for(cfg, run):
    cfg = deepcopy(cfg)
    if "visual_features" in cfg["model"]:
        cfg["model"]["visual_features"]["cache_dir"] = str(run.asset("features"))
    dataset = EpisodeDataset(
        run.asset("prepared"),
        tasks=["VideoUnmask"],
        horizon=cfg["model"]["horizon"],
        alignment_path=run.asset("alignment"),
        visual_features_config=cfg,
    )
    dataset.validate_preprocessing(cfg)
    episode_indices = {ref.key: i for i, ref in enumerate(dataset.episodes) if ref.task == "VideoUnmask"}
    episodes = [dataset.load_episode(episode_indices[f"episode_{i}"]) for i in (0, 1)]
    assert [episode.episode_id for episode in episodes] == ["episode_0", "episode_1"]
    return cfg, dataset, episodes


def canonical_decision(episode, index, cfg):
    decision = episode.decision(index, cfg["model"]["horizon"])
    if cfg["model"].get("history_mode", "full") == "none":
        decision = replace(decision, conditioning=[], execution_prefix=[])
    return decision


def decode(policy, context, bias, noise, amp=False):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        normalized = policy.decoder.integrate(
            context, bias, steps=policy.config["model"]["flow_steps"], noise=noise
        )
    raw = policy.normalizer.denormalize(normalized).float().cpu().numpy()
    physical = np.stack([physical_action(a, "VideoUnmask") for a in raw.reshape(-1, 8)])
    return raw, physical, panda_tcp_positions(physical[:, :7], task="VideoUnmask")


def metrics(prediction, baseline, decision, episode, stride):
    raw, physical, xyz = prediction
    braw, bphysical, bxyz = baseline
    target = decision.target_chunk.numpy()
    valid = decision.target_valid.numpy()
    target_xyz = panda_tcp_positions(target[:, :7], task="VideoUnmask")
    result = {"max_abs_raw_action_change": float(np.abs(raw - braw).max())}
    for n in sorted({4, 16, len(raw)}):
        keep = valid[:n, :7].all(-1)
        result[f"tcp_error_mm_first{n}_targets"] = float(
            np.linalg.norm(xyz[:n][keep] - target_xyz[:n][keep], axis=-1).mean() * 1000
        )
        result[f"tcp_change_mm_first{n}_targets"] = float(
            np.linalg.norm(xyz[:n] - bxyz[:n], axis=-1).mean() * 1000
        )
        result[f"joint_error_rmse_first{n}_targets"] = float(
            np.sqrt(np.square(physical[:n, :7][keep] - target[:n, :7][keep]).mean())
        )
    expanded = np.repeat(physical, stride, axis=0)
    expanded_xyz = np.repeat(xyz, stride, axis=0)
    base_xyz = np.repeat(bxyz, stride, axis=0)
    offset = action_alignment_offset(episode.alignment)
    start = decision.index + offset
    native_target = episode.actions[start : start + len(expanded)].numpy()
    native_valid = episode.action_valid[start : start + len(expanded)].numpy()
    native_xyz = panda_tcp_positions(native_target[:, :7], task="VideoUnmask")
    for n in (4, 16):
        keep = native_valid[:n, :7].all(-1)
        result[f"native{n}_tcp_error_mm"] = float(
            np.linalg.norm(expanded_xyz[:n][keep] - native_xyz[:n][keep], axis=-1).mean() * 1000
        )
        result[f"native{n}_tcp_change_mm"] = float(
            np.linalg.norm(expanded_xyz[:n] - base_xyz[:n], axis=-1).mean() * 1000
        )
        result[f"native{n}_joint_error_rmse"] = float(
            np.sqrt(np.square(expanded[:n, :7][keep] - native_target[:n, :7][keep]).mean())
        )
        gv = native_valid[:n, 7]
        result[f"native{n}_gripper_sign_errors"] = int(
            np.sum((expanded[:n, 7][gv] > 0) != (native_target[:n, 7][gv] > 0))
        )
    return {
        **result,
        "raw_actions": raw.tolist(),
        "physical_actions": physical.tolist(),
        "tcp_positions_m": xyz.tolist(),
        "target_actions": target.tolist(),
        "target_valid": valid.tolist(),
        "native_target_actions": native_target.tolist(),
        "native_target_valid": native_valid.tolist(),
    }


def visual_variants(policy, evidence):
    records = evidence.archive.leaves
    no_current = replace(evidence, current_records=[])
    no_visual_history = [r for r in records if r.modality != "visual"]
    result = {
        "baseline": (evidence, records),
        "remove_current_visual_tokens": (no_current, records),
        "remove_history_visual_tokens": (evidence, no_visual_history),
        "remove_all_visual_tokens": (no_current, no_visual_history),
        "remove_all_history": (evidence, []),
    }
    for mode in ["erase_execution", "erase_conditioning", "permute_execution"]:
        altered = intervene_evidence(evidence, mode, seed=9127)
        assert altered.current is evidence.current
        result[mode] = altered, altered.archive.leaves
    points = evidence.current.points.clone()
    points[:, 3:] = 0.5
    gray = replace(evidence.current, points=points)
    result["neutralize_current_RGB_channels_only"] = (
        replace(evidence, current=gray, current_records=policy.encoder.current(gray)),
        records,
    )
    if evidence.current.visual_features is not None:
        zero = replace(
            evidence.current,
            visual_features=torch.zeros(evidence.current.visual_features.shape, dtype=torch.float16),
        )
        result["zero_current_frozen_descriptors"] = (
            replace(evidence, current=zero, current_records=policy.encoder.current(zero)),
            records,
        )
    return result


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["kind"], row["variant"]].append(row)
    summary = []
    for (kind, variant), group in sorted(grouped.items()):
        for split, selected in [
            ("all_states", group),
            ("reset", [r for r in group if r["index"] == 0]),
            ("after_reset", [r for r in group if r["index"] > 0]),
        ]:
            numeric = {
                key: float(np.mean([r[key] for r in selected]))
                for key, value in group[0].items()
                if isinstance(value, (int, float))
                and key.startswith(("tcp_", "native", "max_abs_", "joint_"))
            }
            summary.append(
                dict(kind=kind, variant=variant, state_subset=split, samples=len(selected), **numeric)
            )
    return summary


@torch.inference_mode()
def sensitivity(run, label):
    started = time.perf_counter()
    policy, state = load_policy(run.checkpoint(label), run.device)
    cfg, dataset, episodes = data_for(state["config"], run)
    stride = cfg["model"].get("action_stride", 1)
    command_defs = definitions(full_joint_sweep=False)
    report = dict(
        kind="real_training_state_diagnostic_not_heldout_accuracy",
        label=label,
        status="running",
        checkpoint_sha256=file_sha256(run.checkpoint(label)),
        checkpoint_step=state["step"],
        horizon=cfg["model"]["horizon"],
        action_stride=stride,
        observation_stride=cfg["data"].get("observation_stride", 1),
        history_enabled=policy.history_enabled,
        proprioception_mode=policy.proprioception_mode,
        condition_on_previous_commands=policy.condition_on_previous_commands,
        condition_on_previous_joint_commands=policy.condition_on_previous_joint_commands,
        decisions=DECISIONS,
        noise_seeds=SEEDS,
        command_definitions=command_defs,
        prepared_sha256=file_sha256(run.asset("prepared")),
        alignment=dataset.alignment,
        feature_cache=str(run.asset("features")) if dataset.feature_identity else None,
        feature_signature=dataset.feature_signature,
        states=[],
        rows=[],
        notes=[
            "Two released VideoUnmask training episodes; not held-out action accuracy or benchmark success.",
            "Native windows expand 10Hz targets by zero-order hold at20Hz, matching the actual executor.",
            "First4 target metrics span0.2s at20Hz and0.4s at10Hz; native4 is0.2s in every model.",
            "Action/history perturbations are artificial and can be out of distribution.",
            "Neutral RGB changes only canonical RGB channels, retaining frozen descriptors; zero descriptors is a separate intervention.",
            "Empty history interventions in current-only are invariance controls, not evidence of learned history use.",
        ],
    )
    destination = run.root / label
    destination.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        for index in DECISIONS:
            decision = canonical_decision(episode, index, cfg)
            evidence = policy.encode(decision)
            variants = visual_variants(policy, evidence)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                visual_contexts = {
                    name: policy.context(ev, records) for name, (ev, records) in variants.items()
                }
            command_contexts = {}
            command_controls = {}
            for name, ev, records, controls in iter_variants(
                policy, evidence, decision.conditioning, decision.execution_prefix, command_defs
            ):
                command_contexts[name] = policy.context(ev, records)
                command_controls[name] = controls
            report["states"].append(
                dict(
                    episode=episode.episode_id,
                    index=index,
                    conditioning_frames=len(decision.conditioning),
                    execution_history_frames=len(decision.execution_prefix),
                    history_leaves=len(evidence.archive.leaves),
                    current_visual_tokens=len(evidence.current_records),
                    history_visual_tokens=sum(r.modality == "visual" for r in evidence.archive.leaves),
                    command_controls=command_controls,
                )
            )
            for seed in SEEDS:
                noise = torch.randn(
                    (cfg["model"]["horizon"], 8),
                    device=policy.device,
                    generator=torch.Generator(device=policy.device).manual_seed(seed),
                )
                for kind, contexts, amp in [
                    ("visual", visual_contexts, True),
                    ("commands", command_contexts, False),
                ]:
                    baseline = decode(policy, *contexts["baseline"], noise, amp)
                    for name, context in contexts.items():
                        pred = baseline if name == "baseline" else decode(policy, *context, noise, amp)
                        row = dict(
                            episode=episode.episode_id,
                            index=index,
                            noise_seed=seed,
                            kind=kind,
                            variant=name,
                            **metrics(pred, baseline, decision, episode, stride),
                        )
                        if kind == "commands":
                            definition = next((d for d in command_defs if d["name"] == name), {})
                            unchanged = (
                                name == "recomputed_event_control"
                                or index == 0
                                or not policy.condition_on_previous_commands
                            )
                            unchanged |= not policy.condition_on_previous_joint_commands and definition.get(
                                "kind"
                            ) in ("offset", "hold_qpos")
                            if unchanged:
                                assert row["max_abs_raw_action_change"] == 0, (
                                    label,
                                    name,
                                    index,
                                    row["max_abs_raw_action_change"],
                                )
                        if kind == "visual" and (
                            not policy.history_enabled
                            and name
                            in [
                                "remove_history_visual_tokens",
                                "remove_all_history",
                                "erase_execution",
                                "erase_conditioning",
                                "permute_execution",
                            ]
                            or index == 0
                            and name in ["erase_execution", "permute_execution"]
                        ):
                            assert row["max_abs_raw_action_change"] == 0, (label, name, index)
                        report["rows"].append(row)
                    if seed == SEEDS[0]:
                        repeat = decode(policy, *contexts["baseline"], noise, amp)
                        assert np.array_equal(repeat[0], baseline[0])
                        report["rows"].append(
                            dict(
                                episode=episode.episode_id,
                                index=index,
                                noise_seed=seed,
                                kind=kind,
                                variant="baseline_repeat",
                                **metrics(repeat, baseline, decision, episode, stride),
                            )
                        )
                if seed == SEEDS[0]:
                    fp32_baseline = decode(policy, *command_contexts["baseline"], noise, False)
                    bf16_baseline = decode(policy, *visual_contexts["baseline"], noise, True)
                    report["rows"].append(
                        dict(
                            episode=episode.episode_id,
                            index=index,
                            noise_seed=seed,
                            kind="precision",
                            variant="fp32_vs_production_bf16",
                            **metrics(fp32_baseline, bf16_baseline, decision, episode, stride),
                        )
                    )
            report["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(run.record_identity(report), destination / "sensitivity_partial.json")
            print(
                json.dumps(
                    dict(
                        event="state_complete",
                        label=label,
                        episode=episode.episode_id,
                        index=index,
                        rows=len(report["rows"]),
                        elapsed=report["elapsed_seconds"],
                    )
                ),
                flush=True,
            )
            del evidence, variants, visual_contexts, command_contexts
    report["summary"] = summarize(report["rows"])
    report["status"] = "complete"
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(run.record_identity(report), destination / "sensitivity.json")
    atomic_json(report["summary"], destination / "summary.json")


def action_difference(actual, reference, stride):
    a = np.stack([physical_action(x, "VideoUnmask") for x in actual])
    b = np.stack([physical_action(x, "VideoUnmask") for x in reference])
    xyz = panda_tcp_positions(a[:, :7], task="VideoUnmask") - panda_tcp_positions(
        b[:, :7], task="VideoUnmask"
    )
    return dict(
        max_abs_raw_difference=float(np.abs(actual - reference).max()),
        first4_targets_tcp_difference_mm=float(np.linalg.norm(xyz[:4], axis=-1).mean() * 1000),
        native4_tcp_difference_mm=float(
            np.linalg.norm(np.repeat(xyz, stride, axis=0)[:4], axis=-1).mean() * 1000
        ),
        full_chunk_tcp_difference_mm=float(np.linalg.norm(xyz, axis=-1).mean() * 1000),
    )


def parity(run, label):
    started = time.perf_counter()
    policy, state = load_policy(run.checkpoint(label), run.device)
    cfg, dataset, episodes = data_for(state["config"], run)
    stride = cfg["model"].get("action_stride", 1)
    destination = run.root / label
    destination.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="running",
        checkpoint_sha256=file_sha256(run.checkpoint(label)),
        label=label,
        checkpoint_step=state["step"],
        action_stride=stride,
        observation_stride=cfg["data"].get("observation_stride", 1),
        real_states=[],
        early_prefix=[],
        gradient={},
    )
    for index in [0, 64]:
        decision = canonical_decision(episodes[0], index, cfg)
        packed = pack_policy_batch([decision], cfg["model"]["block_size"], model_config=cfg["model"])
        noise = torch.randn(
            (cfg["model"]["horizon"], 8),
            device=policy.device,
            generator=torch.Generator(device=policy.device).manual_seed(1701),
        )
        outputs, contexts = {}, {}
        with torch.no_grad():
            for mode in [
                "serial_fp32",
                "batch_fp32",
                "serial_all_bf16",
                "batch_all_bf16",
                "online_production",
            ]:
                batched = mode.startswith("batch")
                amp = "bf16" in mode
                policy.train(batched)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    if batched:
                        context, bias, _ = batch_context(policy, [decision], packed_batch=packed)
                    else:
                        evidence = policy.encode(decision)
                        with torch.autocast(
                            "cuda", dtype=torch.bfloat16, enabled=amp or mode == "online_production"
                        ):
                            context, bias = policy.context(evidence, evidence.archive.leaves)
                outputs[mode] = decode(
                    policy,
                    context,
                    bias,
                    noise[None] if batched else noise,
                    amp or mode == "online_production",
                )[0].reshape(-1, 8)
                c, b = (context[0], bias[0]) if batched else (context, bias)
                contexts[mode] = c[torch.isfinite(b)].float().cpu().numpy()
            pairs = [
                ("batch_fp32", "serial_fp32"),
                ("batch_all_bf16", "serial_all_bf16"),
                ("online_production", "batch_all_bf16"),
                ("online_production", "serial_fp32"),
            ]
            row = dict(
                index=index,
                context_tokens=len(contexts["serial_fp32"]),
                comparisons={},
                raw_actions={k: v.tolist() for k, v in outputs.items()},
            )
            for actual, reference in pairs:
                difference = action_difference(outputs[actual], outputs[reference], stride)
                difference["context_max_abs_difference"] = float(
                    np.abs(contexts[actual] - contexts[reference]).max()
                )
                row["comparisons"][actual + "_vs_" + reference] = difference
            assert row["comparisons"]["batch_fp32_vs_serial_fp32"]["full_chunk_tcp_difference_mm"] < 0.01
            report["real_states"].append(row)
        if index == 64:
            # Keep original frozen/inactive parameter flags. Backprop only; never construct an optimizer.
            before = {name: p.detach().cpu().clone() for name, p in policy.named_parameters()}
            policy.train()
            policy.zero_grad(set_to_none=True)
            target = packed.target_chunk.to(policy.device)
            valid = packed.target_valid.to(policy.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                noisy, flow_time, velocity, _ = flow_sample(
                    target,
                    valid,
                    policy.normalizer,
                    noise=noise[None],
                    time=torch.tensor(0.5, device=policy.device),
                )
                prediction = policy(list(packed.metadata), noisy, flow_time, packed_batch=packed)
                loss = masked_mse(prediction, velocity, valid)
            loss.backward()
            groups = {}
            for name, parameter in policy.named_parameters():
                key = (
                    ".".join(name.split(".")[:3])
                    if name.startswith("encoder.visual")
                    else ".".join(name.split(".")[:2])
                )
                group = groups.setdefault(
                    key,
                    dict(parameters=0, requires_grad_parameters=0, gradient_squared_norm=0.0, finite=True),
                )
                group["parameters"] += parameter.numel()
                group["requires_grad_parameters"] += parameter.numel() if parameter.requires_grad else 0
                if parameter.grad is not None:
                    gradient = parameter.grad.float()
                    group["gradient_squared_norm"] += float(gradient.square().sum())
                    group["finite"] &= bool(torch.isfinite(gradient).all())
                assert torch.equal(parameter.detach().cpu(), before[name]), name
            assert all(g["finite"] for g in groups.values())
            assert groups["encoder.visual.point_mlp"]["gradient_squared_norm"] > 0
            report["gradient"] = dict(
                index=index,
                loss=float(loss.detach()),
                flow_time=0.5,
                groups=groups,
                weights_bit_identical_after_backward=True,
                optimizer_updates=0,
            )
            policy.zero_grad(set_to_none=True)
            del loss, prediction, before
        atomic_json(run.record_identity(report), destination / "parity_partial.json")
    # Real early prefixes at identical raw observations, not independently rerendered simulator states.
    fp32_cfg = deepcopy(cfg)
    fp32_cfg["training"]["precision"] = "fp32"
    for control in ["genuine", "no_conditioning_structural_control"]:
        for episode in episodes if control == "genuine" else episodes[:1]:
            conditioning = episode.conditioning if control == "genuine" else []
            online = OnlinePolicy(policy, fp32_cfg, "FULL", cfg["model"]["budget"])
            seed = 4711
            policy.eval()
            online.reset(ResetResult(conditioning, episode.execution[0], episode.goal, {}), seed)
            generator = torch.Generator(device=policy.device).manual_seed(seed)
            decisions, noises, references, counts = [], [], [], []
            with torch.no_grad():
                for index in range(13):
                    if index:
                        online.observe(episode.execution[index])
                    if index not in [0, 4, 8, 12]:
                        continue
                    decision = canonical_decision(episode, index, cfg)
                    if control != "genuine":
                        decision = replace(decision, conditioning=[])
                    decisions.append(decision)
                    noises.append(
                        torch.randn((cfg["model"]["horizon"], 8), device=policy.device, generator=generator)
                    )
                    actions, timing = online.decide()
                    references.append(actions.cpu().numpy())
                    counts.append(
                        dict(
                            history_leaves=timing["archive_fine_leaves"],
                            execution_observations=online.memory.execution_observations,
                            current_time=online.memory.current.time,
                        )
                    )
                    assert (
                        online.memory.execution_observations
                        == index // cfg["data"].get("observation_stride", 1) + 1
                    )
                    assert online.memory.current.event_id == decision.current.event_id
                packed = pack_policy_batch(decisions, cfg["model"]["block_size"], model_config=cfg["model"])
                policy.train()
                context, bias, stats = batch_context(policy, list(packed.metadata), packed_batch=packed)
                actual = (
                    policy.normalizer.denormalize(
                        policy.decoder.integrate(
                            context, bias, steps=cfg["model"]["flow_steps"], noise=torch.stack(noises)
                        )
                    )
                    .cpu()
                    .numpy()
                )
                for position, index in enumerate([0, 4, 8, 12]):
                    difference = action_difference(actual[position], references[position], stride)
                    assert difference["full_chunk_tcp_difference_mm"] < 0.01, (label, index, difference)
                    assert int(stats["history_leaves"][position]) == counts[position]["history_leaves"]
                    report["early_prefix"].append(
                        dict(
                            index=index,
                            episode=episode.episode_id,
                            control=control,
                            conditioning_frames=len(decisions[position].conditioning),
                            history_frames=len(decisions[position].execution_prefix),
                            **counts[position],
                            **difference,
                            packed_raw_actions=actual[position].tolist(),
                            streaming_raw_actions=references[position].tolist(),
                        )
                    )
            atomic_json(run.record_identity(report), destination / "parity_partial.json")
    report["status"] = "passed"
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(run.record_identity(report), destination / "parity.json")
    print(
        json.dumps(dict(event="parity_complete", label=label, seconds=report["elapsed_seconds"])), flush=True
    )


def altered_state(frame):
    changed = replace(
        frame, proprio=torch.roll(frame.proprio, 1) + torch.arange(15, dtype=frame.proprio.dtype) * 0.01
    )
    assert changed.points is frame.points and changed.visual_features is frame.visual_features
    assert not torch.equal(changed.proprio, frame.proprio)
    return changed


@torch.inference_mode()
def invariants(run):
    started = time.perf_counter()
    report = dict(
        scope="Artificial omitted-input invariance at real VideoUnmask train0 decision64; no commands executed.",
        noise_seed=1701,
        precision="fp32",
        rows=[],
    )
    for label in run.labels:
        sensitivity_report = json.loads((run.root / label / "sensitivity.json").read_text())
        if sensitivity_report["proprioception_mode"] != "none":
            continue
        policy, state = load_policy(run.checkpoint(label), run.device)
        cfg, dataset, episodes = data_for(state["config"], run)
        episode = episodes[0]
        decision = canonical_decision(episode, 64, cfg)
        original = policy.encode(decision)
        context, bias = policy.context(original, original.archive.leaves)
        noise = torch.randn(
            (cfg["model"]["horizon"], 8),
            device=policy.device,
            generator=torch.Generator(device=policy.device).manual_seed(1701),
        )
        reference = decode(policy, context, bias, noise, False)[0]
        variants = {
            "current_proprioception": replace(decision, current=altered_state(decision.current)),
            "history_proprioception": replace(
                decision,
                conditioning=[altered_state(f) for f in decision.conditioning],
                execution_prefix=[altered_state(f) for f in decision.execution_prefix],
            ),
            "current_and_history_proprioception": replace(
                decision,
                current=altered_state(decision.current),
                conditioning=[altered_state(f) for f in decision.conditioning],
                execution_prefix=[altered_state(f) for f in decision.execution_prefix],
            ),
        }
        if not policy.history_enabled:
            full = episode.decision(64, cfg["model"]["horizon"])
            variants["inject_retained_history"] = full
            variants["inject_history_with_altered_proprioception"] = replace(
                full,
                conditioning=[altered_state(f) for f in full.conditioning],
                execution_prefix=[altered_state(f) for f in full.execution_prefix],
            )
            variants["current_timestamp_and_source_index"] = replace(
                decision,
                current=replace(
                    decision.current,
                    time=decision.current.time + 1000.0,
                    source_index=decision.current.source_index + 1000,
                ),
            )
        for name, changed in variants.items():
            ev = policy.encode(changed)
            c, b = policy.context(ev, ev.archive.leaves)
            actions = decode(policy, c, b, noise, False)[0]
            assert torch.equal(c, context) and torch.equal(b, bias), (label, name, "context differs")
            assert np.array_equal(actions, reference), (label, name, "actions differ")
            report["rows"].append(
                dict(
                    label=label,
                    checkpoint_step=state["step"],
                    checkpoint_sha256=file_sha256(run.checkpoint(label)),
                    intervention=name,
                    context_bit_identical=True,
                    actions_bit_identical=True,
                    input_conditioning_frames=len(changed.conditioning),
                    input_execution_history_frames=len(changed.execution_prefix),
                    encoded_history_leaves=len(ev.archive.leaves),
                    applicable_nonempty_history=bool(changed.conditioning or changed.execution_prefix),
                )
            )
        if not policy.history_enabled:
            altered = replace(original, clocks={"conditioning": 9831.0, "execution": 4512.0})
            c, b = policy.context(altered, altered.archive.leaves)
            actions = decode(policy, c, b, noise, False)[0]
            assert torch.equal(c, context) and torch.equal(b, bias)
            assert np.array_equal(actions, reference)
            report["rows"].append(
                dict(
                    label=label,
                    intervention="context_clocks",
                    context_bit_identical=True,
                    actions_bit_identical=True,
                    checkpoint_step=state["step"],
                    checkpoint_sha256=file_sha256(run.checkpoint(label)),
                )
            )
    report["status"] = "passed"
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(run.record_identity(report), run.root / "ablation_invariance.json")
    print(json.dumps(dict(status="passed", cases=len(report["rows"]), seconds=report["elapsed_seconds"])))


def protocol(run):
    return run.record_identity(
        {
            "scope": "Bounded real-training-state diagnostics; no optimizer updates, simulation, benchmark score or test seeds.",
            "task": "VideoUnmask",
            "training_episodes": [0, 1],
            "native_decision_indices": DECISIONS,
            "paired_flow_noise_seeds": SEEDS,
            "visual_probe_precision": "FP32 encoder/text; BF16 metadata context and flow decoder, matching production online policy on CUDA.",
            "previous_command_probe_precision": "FP32 throughout, matching the preceding command sensitivity audit.",
            "native_action_metrics": "Targets use zero-order hold for model.action_stride native20Hz commands; native4 spans0.2s and native16 spans0.8s for all models.",
            "history_contract": "Retain phase-zero source-local observation cadence; early-prefix parity ingests every original observation through production OnlinePolicy.",
            "intervention_contract": "Artificial encoded-token deletion/payload permutation and previous-command perturbations are not physically rendered counterfactuals. Conditioning commands remain unavailable.",
            "parity": "Real episode0 states0/64: serial/packed FP32, all-BF16 and production. Early-prefix episodes0/1 at0/4/8/12 plus no-conditioning structural control on episode0.",
            "gradient": "One BF16 backward-only probe at episode0 decision64; no optimizer and all weights checked bit-identical afterward.",
            "prepared_data": str(run.asset("prepared")),
            "prepared_sha256": file_sha256(run.asset("prepared")),
            "alignment": str(run.asset("alignment")),
            "alignment_sha256": file_sha256(run.asset("alignment")),
            "frozen_cache": str(run.asset("features")),
            "limitations": [
                "Training-state fit is not held-out accuracy; four noise draws are correlated repeats.",
                "Different architectures and training exposure are not an isolated controlled ablation.",
                "Deletion sensitivity does not establish semantic usefulness or closed-loop corrective ability.",
            ],
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label")
    parser.add_argument(
        "--mode", choices=["all", "sensitivity", "parity", "invariants", "validate"], default="all"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = Inspection(json.loads(args.manifest.read_text()), file_sha256(args.manifest))
    if args.label is not None and args.label not in run.labels:
        parser.error("--label is not present in the inspection manifest")
    if args.label is not None and args.mode in ("invariants", "validate"):
        parser.error("Aggregate invariants/validation require all manifest labels")
    if args.mode != "validate" and (not run.device.startswith("cuda") or not torch.cuda.is_available()):
        parser.error("This matched BF16 inspection protocol requires a working CUDA device")
    labels = [args.label] if args.label else run.labels
    torch.set_num_threads(2)
    run.root.mkdir(parents=True, exist_ok=True)
    for label in labels:
        run.validate_checkpoint(label)
    atomic_json(protocol(run), run.root / "protocol.json")
    for label in labels:
        for mode, function in (("sensitivity", sensitivity), ("parity", parity)):
            if args.mode not in ("all", mode):
                continue
            if not run.reuse(run.root / label / f"{mode}.json", args.resume):
                function(run, label)
            torch.cuda.empty_cache()
    if args.mode in ("all", "invariants") and args.label is None:
        if not run.reuse(run.root / "ablation_invariance.json", args.resume):
            invariants(run)
    if args.mode in ("all", "validate") and args.label is None:
        from stmem.inspection.offline_validation import validate

        validate(run)
    for label in labels:
        run.validate_checkpoint(label)


if __name__ == "__main__":
    main()
