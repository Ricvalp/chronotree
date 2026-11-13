"""Exploratory local utility measurements for transferred compact students.

This path deliberately does not impersonate the sealed 256-state HPC study.
It encodes a bounded set of local released demonstrations with the students'
shared frozen encoders and reuses the production causal hierarchy, exact FLAT
residuals and gain-blind refinement constructors. Expert FM is the primary
target. Teacher agreement is absent unless an exact matching teacher is supplied.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import re
import time

import numpy as np
import torch

from stmem.config import digest
from stmem.data.dataset import EpisodeDataset
from stmem.inspection.refinement_candidates import IndexedArchive
from stmem.inspection.refinement_measurements import (
    _context_batches, _losses, _noise, _precision, _validate_draws,
    action_mask_summary, state_fingerprint,
)
from stmem.inspection.refinement_statistics import tie_set
from stmem.models.action_flow import flow_sample
from stmem.models.batched_policy import TokenBank
from stmem.training.label_objectives import policy_context
from stmem.training.packed_context import (
    SCALAR_FIELDS, STRING_FIELDS, TENSOR_FIELDS, columns_bank, flat_residual_columns,
)

SCHEMA = "local_compact_utility_v1"


def open_local_dataset(config, assets):
    """Redirect only data locations, retaining checkpoint preprocessing/cadence."""
    cfg = deepcopy(config)
    if "visual_features" in cfg["model"]:
        cfg["model"]["visual_features"]["cache_dir"] = str(assets["features"])
    dataset = EpisodeDataset(assets["prepared"], tasks=["VideoUnmask"],
                             horizon=cfg["model"]["horizon"],
                             alignment_path=assets["alignment"], visual_features_config=cfg)
    dataset.validate_preprocessing(cfg)
    return dataset


def select_local_states(dataset, config, *, maximum=8):
    """Gain-blind source-index quantiles, complete targets, fixed local episodes.

    These are exploratory diagnostic training scenes. Do not describe this
    selection as the held-out primary utility panel or as unseen by the teacher.
    No loss, target position or semantic annotation enters the selection rule.
    """
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 32:
        raise ValueError("Local state cap must be between 1 and 32")
    selected_episodes = [i for i, ref in enumerate(dataset.episodes)
                         if ref.task == "VideoUnmask" and ref.key in {"episode_0", "episode_1"}]
    selected_episodes.sort(key=lambda i: dataset.episodes[i].key)
    if len(selected_episodes) != 2:
        raise ValueError("Local utility diagnostic requires VideoUnmask episodes 0 and 1")
    candidates = []
    per_episode = math.ceil(maximum / len(selected_episodes))
    for episode_index in selected_episodes:
        episode = dataset.load_episode(episode_index)
        eligible = []
        for index, frame in enumerate(episode.execution):
            if frame.source_index % episode.observation_stride:
                continue
            decision = episode.decision(index, config["model"]["horizon"])
            # VideoUnmask has all eight structural action coordinates available.
            if bool(decision.target_valid.all()):
                eligible.append(index)
        if not eligible:
            raise ValueError(f"No complete-horizon decisions in {episode.episode_id}")
        offsets = np.linspace(0, len(eligible) - 1, min(per_episode, len(eligible)), dtype=int)
        rows = []
        for offset in offsets:
            index = eligible[int(offset)]
            decision = episode.decision(index, config["model"]["horizon"])
            phase = ("early" if offset < len(eligible) / 3 else
                     "middle" if offset < 2 * len(eligible) / 3 else "late")
            state = {"task": episode.task, "episode_id": episode.episode_id,
                     "episode_index": episode_index, "decision_index": index,
                     "source_index": decision.current.source_index, "phase": phase,
                     "conditioning_frames": len(decision.conditioning),
                     "execution_history_frames": len(decision.execution_prefix),
                     "partition": "exploratory_local_training_scenes",
                     "selection": "complete_horizon_source_order_quantiles_before_losses"}
            state["state_id"] = digest([SCHEMA, episode.task, episode.episode_id, index])[:24]
            rows.append({"state": state, "decision": decision})
        candidates.append(rows)
    # Round-robin keeps both episodes represented when the cap is odd.
    return [row for offset in range(per_episode) for rows in candidates
            for row in rows[offset:offset + 1]][:maximum]


def _record_columns(records, width):
    columns = {}
    for field in TENSOR_FIELDS:
        dimension = width if field in {"content_sum", "max_content"} else 3
        columns[field] = (torch.stack([getattr(record, field) for record in records]).detach().cpu().float()
                          if records else torch.empty(0, dimension))
    for field in SCALAR_FIELDS:
        columns[field] = np.asarray([getattr(record, field) for record in records])
    for field in STRING_FIELDS:
        columns[field] = np.asarray([getattr(record, field) for record in records], dtype=object)
    return columns


def _take(columns, rows):
    return {name: value[rows] for name, value in columns.items()}


def _record_visuals(columns):
    keys = ("node_id", "source", "modality", "count", "anchor", "bounds_min", "bounds_max",
            "time_min", "time_max", "time_anchor", "depth", "disconnected")
    result = []
    for row in range(len(columns["count"])):
        item = {}
        for key in keys:
            value = columns[key][row]
            item[key] = value.tolist() if isinstance(value, (np.ndarray, torch.Tensor)) else (
                value.item() if isinstance(value, np.generic) else value)
        result.append(item)
    return result


@torch.inference_mode()
def prepare_local_state(policy, decision, *, seed, budgets=(128,), packet_cost=8,
                        max_candidates=4, audit_candidates=4):
    """Encode one causal decision once; reuse the same banks across students.

    The caller must verify shared frozen encoder/normalizer identities before
    reusing this payload with another checkpoint. No decoder output enters the
    candidate construction. CPU banks can be stored as trusted local artifacts.
    """
    _validate_frozen(policy)
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
    started = time.perf_counter()
    width = policy.config["model"]["width"]
    with _precision(policy.device, "fp32"):
        evidence = policy.encode(decision)
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
    encoded = time.perf_counter()
    roots = evidence.archive.roots
    # Use the exact archive leaf order for FP32 residual reduction parity.
    leaves = evidence.archive.leaves
    records, seen, pending = [], set(), list(reversed(roots))
    while pending:
        record = pending.pop()
        if record.node_id in seen:
            continue
        seen.add(record.node_id)
        records.append(record)
        pending.extend(reversed(record.children))
    lookup = {record.node_id: row for row, record in enumerate(records)}
    columns = _record_columns(records, width)
    index = {"children": np.asarray([[lookup[child.node_id] for child in record.children]
                                      if record.children else [-1, -1] for record in records],
                                     dtype=np.int64).reshape(-1, 2),
             **{key: columns[key].numpy() if torch.is_tensor(columns[key]) else columns[key]
                for key in ("node_id", "source", "modality", "count", "anchor", "time_anchor")}}
    leaf_rows = np.asarray([lookup[leaf.node_id] for leaf in leaves], dtype=np.int64)
    archive = IndexedArchive(index, [lookup[root.node_id] for root in roots], leaf_rows)
    if any(record.time_max >= decision.current.time for record in leaves if record.source == "execution"):
        raise ValueError("Current or future execution evidence leaked into local history")
    clocks = [evidence.clocks["conditioning"], evidence.clocks["execution"]]
    leaf_columns = _take(columns, leaf_rows)
    current_columns = _record_columns(evidence.current_records, width)
    item = {"task": decision.task, "episode": f"{decision.task}/{decision.episode_id}",
            "decision_index": decision.index, "target": decision.target_chunk.detach().cpu().float(),
            "valid": decision.target_valid.detach().cpu().bool(),
            "current_state": torch.cat([decision.current.proprio.detach().cpu(),
                                         decision.current.previous_action.detach().cpu(),
                                         torch.tensor([float(decision.current.action_available)])]).float(),
            "clocks": clocks, "history_leaves": len(leaves), "text": evidence.text.detach().cpu().float(),
            "current": columns_bank(current_columns, clocks, width),
            "full": columns_bank(leaf_columns, clocks, width)}
    if not action_mask_summary(item)["full_horizon"]:
        raise ValueError("Local primary utility states require complete target horizons")
    panel = archive.diagnostic_panel(seed, budgets=budgets, packet_size=packet_cost,
                                     max_candidates=max_candidates, audit_candidates=audit_candidates)
    panel["capability"] = {}
    for reader in ("TREE", "FLAT"):
        for budget, mode in ((128, "ordinary"), (128, "packet"), (256, "ordinary")):
            name = f"{reader.lower()}_{mode}{budget}"
            try:
                if mode == "ordinary":
                    context = archive.ordinary(reader, budget, seed + budget)
                else:
                    route = archive.route(reader, budget, seed + budget, mode="packet")
                    context = route["contexts"][min(budget, len(leaves))]
                panel["contexts"][context["context_id"]] = archive.describe(context)
                panel["capability"][name] = context["context_id"]
            except ValueError as error:
                panel["capability"][name] = {"omitted": str(error)}
    full = archive.context("TREE", leaf_rows)
    panel["contexts"][full["context_id"]] = archive.describe(full)
    panel["capability"]["all_leaves"] = full["context_id"]
    contexts, visuals = {}, {}
    for key, spec in panel["contexts"].items():
        archive.validate(spec)
        selected = (_take(columns, spec["rows"]) if spec["reader"] == "TREE" else
                    flat_residual_columns(leaf_columns, *archive.flat_arrays(spec)))
        contexts[key] = columns_bank(selected, clocks, width)
        if contexts[key].content.shape[1] != spec["token_count"]:
            raise ValueError("Hydrated compact token count differs from the fixed context")
        visuals[key] = _record_visuals(selected)
    # Erasures deliberately violate complete coverage and change token count.
    # Keep them outside the allocation candidates and never call them gains.
    panel["interventions"] = {}
    baseline = panel["capability"].get("tree_ordinary128")
    if isinstance(baseline, str):
        baseline_records, baseline_bank = visuals[baseline], contexts[baseline]
        for name, remove in (("current_only", {"conditioning", "execution"}),
                             ("erase_conditioning", {"conditioning"}),
                             ("erase_execution", {"execution"})):
            keep = torch.tensor([record["source"] not in remove for record in baseline_records], dtype=torch.bool)
            key = f"intervention/{name}"
            contexts[key] = TokenBank(*(getattr(baseline_bank, field)[:, keep] for field in
                                        ("content", "metadata", "valid", "log_count")))
            visuals[key] = [record for record, retained in zip(baseline_records, keep.tolist()) if retained]
            panel["interventions"][name] = {"context_id": key, "baseline": baseline,
                                              "tokens": int(keep.sum()), "exact_coverage": False,
                                              "note": "Input-erasure sensitivity only; no allocation gain or router target"}
    scene = decision.current.points[decision.current.valid].detach().cpu()
    if len(scene) > 2048:
        scene = scene[torch.linspace(0, len(scene) - 1, 2048).long()]
    return {"schema": SCHEMA, "item": item, "contexts": contexts, "panel": panel,
            "visual": {"scene_points": scene, "current_anchors": current_columns["anchor"],
                       "context_records": visuals},
            "state_fingerprint": state_fingerprint(item), "encoding_precision": "fp32",
            "timings": {"encode_and_hierarchy_seconds": encoded - started,
                        "context_assembly_seconds": time.perf_counter() - encoded,
                        "prepare_wall_seconds": time.perf_counter() - started},
            "scope": "Exploratory local training-scene diagnostic; not the sealed primary study"}


def context_visual_records(prepared, key):
    return prepared["visual"]["context_records"][key]


def _validate_frozen(policy):
    if not policy.encoders_frozen or any(module.training for module in policy.modules()) or any(
            parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError("Utility measurements require frozen encoders, eval mode and gradients disabled")


@torch.inference_mode()
def evaluate_student_state(policy, prepared, draws, *, checkpoint_hash, teacher=None,
                           teacher_hash=None, prefix_targets=8, batch_size=16,
                           precision="fp32", numerical_checks=True, maximum_padded_tokens=32768):
    """Raw expert-FM utility with paired draws; teacher agreement is optional.

    Expert targets never construct contexts. The unchanged-context and
    serial/packed controls estimate the numerical floor separately from noise.
    """
    _validate_frozen(policy)
    _validate_draws(draws)
    if not re.fullmatch(r"[a-f0-9]{64}", checkpoint_hash):
        raise ValueError("An exact student checkpoint SHA256 is required")
    item, contexts = prepared["item"], prepared["contexts"]
    horizon = item["target"].shape[0]
    if not 1 <= prefix_targets <= horizon or batch_size < 1 or not contexts:
        raise ValueError("Invalid prefix, batch size or empty context panel")
    mask_summary = action_mask_summary(item)
    if not mask_summary["full_horizon"]:
        raise ValueError("Local primary utility states must have complete target horizons")
    if teacher is not None:
        _validate_frozen(teacher)
        if teacher.device != policy.device or not re.fullmatch(r"[a-f0-9]{64}", str(teacher_hash)):
            raise ValueError("Teacher requires the same device and an exact checkpoint hash")
        for name in ("mean", "std", "count"):
            if not torch.equal(getattr(teacher.normalizer, name), getattr(policy.normalizer, name)):
                raise ValueError("Teacher/student normalization differs")
    elif teacher_hash is not None:
        raise ValueError("A teacher hash cannot substitute for an available matching teacher")
    device = policy.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    noise, u = _noise(item, draws, device)
    flow_times = u.cpu().tolist()
    valid = item["valid"].to(device)[None].expand(len(draws), -1, -1)
    target = item["target"].to(device)[None].expand(len(draws), -1, -1)
    x, _, velocity, _ = flow_sample(target, valid, policy.normalizer, noise=noise, time=u)
    teacher_prediction = None
    teacher_expert = None
    with _precision(device, precision):
        if teacher is not None:
            teacher_context, teacher_bias = policy_context(teacher, [item], [item["full"]])
            teacher_prediction = torch.cat([teacher.decoder(x[d:d+1], u[d:d+1], teacher_context,
                                                            teacher_bias).float() for d in range(len(draws))])
            if not torch.isfinite(teacher_prediction).all():
                raise FloatingPointError("Nonfinite teacher velocity")
            teacher_expert = torch.stack(_losses(teacher_prediction, velocity, valid, prefix_targets), dim=1).cpu().tolist()
            del teacher_context, teacher_bias
        rows, predictions = [], {}
        check_key = min(contexts)
        measured_contexts = dict(contexts)
        duplicate = "__duplicate_noop__"
        if numerical_checks:
            measured_contexts[duplicate] = contexts[check_key]
        for keys in _context_batches(measured_contexts, batch_size, maximum_padded_tokens):
            context, bias = policy_context(policy, [item] * len(keys), [measured_contexts[key] for key in keys])
            for index, draw in enumerate(draws):
                prediction = policy.decoder(x[index:index+1].expand(len(keys), -1, -1),
                                             u[index:index+1].expand(len(keys)), context, bias).float()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite student velocity")
                losses = _losses(prediction, velocity[index:index+1].expand_as(prediction),
                                  valid[index:index+1].expand_as(prediction), prefix_targets)
                values = torch.stack(losses, dim=1).cpu().tolist()
                agreement = (torch.stack(_losses(prediction, teacher_prediction[index:index+1].expand_as(prediction),
                                                  valid[index:index+1].expand_as(prediction), prefix_targets),
                                         dim=1).cpu().tolist() if teacher_prediction is not None else None)
                for position, key in enumerate(keys):
                    if numerical_checks and key in (check_key, duplicate) and index < 2:
                        predictions[key, index] = prediction[position].detach().cpu()
                    if key == duplicate:
                        continue
                    rows.append({"context_id": key, "student_hash": checkpoint_hash, **draw,
                                 "flow_time": flow_times[index],
                                 "expert_full": values[position][0], "expert_prefix": values[position][1],
                                 "teacher_full": agreement[position][0] if agreement else None,
                                 "teacher_prefix": agreement[position][1] if agreement else None,
                                 "teacher_expert_full": teacher_expert[index][0] if teacher_expert else None,
                                 "teacher_expert_prefix": teacher_expert[index][1] if teacher_expert else None})
            del context, bias
        numerical = {"status": "not_requested"}
        if numerical_checks:
            context, bias = policy_context(policy, [item], [contexts[check_key]])
            differences, noop, loss_differences, noop_losses = [], [], [], []
            row_lookup = {(row["context_id"], row["bank"], row["draw_index"]): row for row in rows}
            for index, draw in enumerate(draws[:2]):
                serial = policy.decoder(x[index:index+1], u[index:index+1], context, bias).float()
                differences.append(float((serial[0].cpu() - predictions[check_key, index]).abs().max()))
                noop.append(float((predictions[check_key, index] - predictions[duplicate, index]).abs().max()))
                loss = _losses(serial, velocity[index:index+1], valid[index:index+1], prefix_targets)[0][0]
                previous = row_lookup[check_key, draw["bank"], draw["draw_index"]]["expert_full"]
                loss_differences.append(abs(float(loss.cpu()) - previous))
                repeated_loss = _losses(predictions[duplicate, index][None].to(device),
                                        velocity[index:index+1], valid[index:index+1], prefix_targets)[0][0]
                noop_losses.append(abs(float(repeated_loss.cpu()) - previous))
            numerical = {"status": "measured", "noop_prediction_max_abs": max(noop),
                         "noop_loss_max_abs": max(noop_losses),
                         "serial_packed_prediction_max_abs": max(differences),
                         "serial_packed_loss_max_abs": max(loss_differences),
                         "same_budget_delta_tokens": 0,
                         "note": "Numerical floor, not noise confidence or practical effect size"}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"schema": SCHEMA, "rows": sorted(rows, key=lambda r: (r["context_id"], r["bank"], r["draw_index"])),
            "numerical": numerical, "mask": mask_summary, "precision": precision,
            "teacher_hash": teacher_hash, "teacher_agreement_available": teacher is not None,
            "state_fingerprint": prepared.get("state_fingerprint", state_fingerprint(item)),
            "timings": {"wall_seconds": time.perf_counter() - started,
                        "compact_model_examples": len(measured_contexts) * len(draws) +
                        (min(2, len(draws)) if numerical_checks else 0),
                        "teacher_model_examples": len(draws) if teacher is not None else 0,
                        "contexts_adapted": len(measured_contexts) + int(numerical_checks),
                        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}}


def summarize_utility(panel, rows, *, epsilon=1e-6):
    """Discovery selects; independent confirmation measures gains and regret.

    Preserve signs and all candidate/control denominators. No teacher-less row
    is relabeled as teacher agreement. Tied discovery winners receive averaged
    confirmation credit, rather than arbitrary confident ranking labels.
    """
    grouped = defaultdict(list)
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Utility tie tolerance must be finite and nonnegative")
    for row in rows:
        grouped[row["context_id"], row["bank"]].append(row)
    means = {key: float(np.mean([row["expert_full"] for row in values])) for key, values in grouped.items()}
    teacher_available = bool(rows) and all(row.get("teacher_full") is not None for row in rows)
    teacher_means = ({key: float(np.mean([row["teacher_full"] for row in values]))
                      for key, values in grouped.items()} if teacher_available else {})
    summary, gain_rows = [], []
    for family in panel["families"]:
        cost = family["final_budget"] - family["background"]
        base = family["base"]
        candidates = family["candidates"]
        required = [base, *family["controls"].values(),
                    *(candidate["prefixes"][str(step)] for candidate in candidates for step in (1, cost))]
        if any((key, bank) not in means for key in required for bank in ("discovery", "confirmation")):
            raise ValueError("A utility family is missing fixed context/draw measurements")
        if not candidates:
            continue
        first = {c["candidate_id"]: means[c["prefixes"]["1"], "discovery"] for c in candidates}
        completed = {c["candidate_id"]: means[c["prefixes"][str(cost)], "discovery"] for c in candidates}
        confirmation = {c["candidate_id"]: means[c["prefixes"][str(cost)], "confirmation"] for c in candidates}
        choices = {"immediate": tie_set(first, epsilon=epsilon), "packet": tie_set(completed, epsilon=epsilon),
                   "low": tie_set({key: -value for key, value in completed.items()}, epsilon=epsilon)}
        if teacher_available:
            choices["teacher"] = tie_set({c["candidate_id"]: teacher_means[c["prefixes"][str(cost)], "discovery"]
                                          for c in candidates}, epsilon=epsilon)
        selected = {name: float(np.mean([confirmation[key] for key in keys])) for name, keys in choices.items()}
        controls = {name: means[key, "confirmation"] for name, key in family["controls"].items()}
        random_loss = float(np.mean(list(confirmation.values())))
        confirmation_best = tie_set(confirmation, epsilon=epsilon)
        summary.append({"reader": family["reader"], "family": family["panel"],
                        "background": family["background"], "budget": family["final_budget"],
                        "candidate_count": len(candidates), "selection": choices,
                        "selection_objective": "discovery_expert_full", "tie_policy": "uniform_average",
                        "base_loss": means[base, "confirmation"], "packet_loss": selected["packet"],
                        "immediate_loss": selected["immediate"], "low_selected_loss": selected["low"],
                        "teacher_selected_loss": selected.get("teacher"),
                        "expert_teacher_selection_disagreement": bool(not set(choices["packet"]) & set(choices["teacher"]))
                        if teacher_available and len(choices["packet"]) == len(choices["teacher"]) == 1 else None,
                        "random_loss": random_loss, "controls": controls,
                        "packet_vs_random": random_loss - selected["packet"],
                        "packet_vs_immediate": selected["immediate"] - selected["packet"],
                        "packet_vs_low": selected["low"] - selected["packet"],
                        "packet_vs_dispersed": controls["dispersed"] - selected["packet"] if "dispersed" in controls else None,
                        "winner_reversal": bool(not set(choices["packet"]) & set(confirmation_best))
                        if len(choices["packet"]) == len(confirmation_best) == 1 else None,
                        "ranking_eligible": len(candidates) >= 2, "delta_tokens": 0,
                        "diagnostic_selection_not_deployed_router": True})
        for candidate in candidates:
            for kind, increment in (("first", 1), ("complete", cost)):
                context = candidate["prefixes"][str(increment)]
                gains = {bank: means[base, bank] - means[context, bank] for bank in ("discovery", "confirmation")}
                lookup = {row["draw_index"]: row["expert_full"] for row in grouped[context, "confirmation"]}
                paired = [row["expert_full"] - lookup[row["draw_index"]] for row in grouped[base, "confirmation"]]
                gain_rows.append({"reader": family["reader"], "family": family["panel"],
                                  "background": family["background"], "candidate_id": candidate["candidate_id"],
                                  "kind": kind, "delta_tokens": increment,
                                  "discovery_gain": gains["discovery"], "confirmation_gain": gains["confirmation"],
                                  "teacher_discovery_gain": teacher_means[base, "discovery"] - teacher_means[context, "discovery"]
                                  if teacher_available else None,
                                  "teacher_confirmation_gain": teacher_means[base, "confirmation"] - teacher_means[context, "confirmation"]
                                  if teacher_available else None,
                                  "gain_per_token": gains["confirmation"] / increment,
                                  "confirmation_positive_draw_fraction": float(np.mean(np.asarray(paired) > epsilon)),
                                  "confirmation_noise_sd": float(np.std(paired, ddof=1)) if len(paired) > 1 else None})
    capability = {name: ({"omitted": key["omitted"]} if isinstance(key, dict) else
                         {"expert_full": means[key, "confirmation"], "context_id": key,
                          "tokens": panel["contexts"][key]["token_count"]})
                  for name, key in panel.get("capability", {}).items()}
    interventions = {name: {**spec, "expert_full": means[spec["context_id"], "confirmation"],
                            "baseline_expert_full": means[spec["baseline"], "confirmation"],
                            "loss_change": means[spec["context_id"], "confirmation"] -
                            means[spec["baseline"], "confirmation"]}
                     for name, spec in panel.get("interventions", {}).items()}
    teacher_reference = {"available": teacher_available}
    if teacher_available:
        # Each context repeats the same reference; count each draw only once.
        reference = {row["draw_index"]: row["teacher_expert_full"] for row in rows if row["bank"] == "confirmation"}
        teacher_reference["expert_full"] = float(np.mean(list(reference.values())))
    return {"selection": summary, "gains": gain_rows, "capability": capability,
            "interventions": interventions, "teacher_reference": teacher_reference,
            "eligibility": panel["eligibility"],
            "teacher_agreement_available": teacher_available,
            "scope": "Exploratory local fixed-state diagnostics; no benchmark success or router fit"}
