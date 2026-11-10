"""Frozen, paired expert-FM and teacher-agreement measurements.

This module deliberately does not select candidates or interpret gains.  The
caller supplies a sealed gain-blind context manifest, and receives raw losses.
Teacher references depend on the state/draw/teacher/precision, never on the
student, so the same references can be reused across the matched student matrix.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import re
import time

import numpy as np
import torch

from stmem.evaluation.robomme_adapter import physical_action
from stmem.models.action_flow import flow_sample, masked_mse
from stmem.training.label_objectives import policy_context
from stmem.visualization.kinematics import panda_tcp_positions, panda_variant


SCHEMA = "refinement_measurements_v1"
FLOW_CONVENTION = "flow_sample_masked_invalid_inputs_v1"
GENERATION_CONVENTION = "unmasked_gaussian_explicit_euler_v1"
RNG_CONVENTION = "sha256_state_bank_draw_cpu_torch_generator_v1"
PRECISIONS = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def draw_seed(experiment_seed, task, episode, decision_index, bank, draw_index):
    """No reader, student, candidate, budget, batch or worker enters this seed."""
    if not isinstance(bank, str) or not bank:
        raise ValueError("A nonempty independent draw-bank name is required")
    if int(draw_index) < 0 or int(decision_index) < 0:
        raise ValueError("Draw and source-local decision indices must be nonnegative")
    value = [RNG_CONVENTION, int(experiment_seed), str(task), str(episode),
             int(decision_index), bank, int(draw_index)]
    return int(_json_hash(value)[:16], 16) & ((1 << 63) - 1)


def draw_plan(experiment_seed, task, episode, decision_index, *, discovery=4,
              confirmation=4, extra_banks=None):
    """Predeclare independent banks; extensions require a new explicit name."""
    banks = {"discovery": discovery, "confirmation": confirmation}
    for bank, count in (extra_banks or {}).items():
        if bank in banks:
            raise ValueError("An extension must be a new independent draw bank")
        banks[bank] = count
    rows = []
    for bank, count in banks.items():
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 64:
            raise ValueError("Each draw bank must contain 1 to 64 predeclared draws")
        for index in range(count):
            rows.append({"bank": bank, "draw_index": index,
                         "seed": draw_seed(experiment_seed, task, episode,
                                           decision_index, bank, index)})
    return rows


def _validate_draws(draws):
    if not draws:
        raise ValueError("At least one predeclared draw is required")
    labels, seeds = set(), set()
    for row in draws:
        key = str(row["bank"]), int(row["draw_index"])
        seed = int(row["seed"])
        if not key[0] or key[1] < 0 or not 0 <= seed < (1 << 63):
            raise ValueError("Invalid draw identity")
        if key in labels or seed in seeds:
            raise ValueError("Draw banks must have unique identities and independent seeds")
        labels.add(key)
        seeds.add(seed)


def _tensor_hash_update(hasher, name, tensor):
    value = torch.as_tensor(tensor).detach().cpu().contiguous()
    hasher.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
    # Byte viewing also supports bfloat16, which NumPy cannot represent directly.
    hasher.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def state_fingerprint(item):
    """Bind reference predictions to actual evidence, target and causal clocks."""
    hasher = hashlib.sha256()
    for name in ("current", "full"):
        for field in ("content", "metadata", "valid", "log_count"):
            _tensor_hash_update(hasher, name + "/" + field, getattr(item[name], field))
    for name in ("text", "current_state", "clocks", "target", "valid"):
        _tensor_hash_update(hasher, name, item[name])
    hasher.update(str(int(item["history_leaves"])).encode())
    return hasher.hexdigest()


def action_mask_summary(item):
    """Separate terminal/unavailable targets; do not change the FM convention."""
    valid = torch.as_tensor(item["valid"]).bool().cpu()
    if valid.ndim != 2 or valid.shape != item["target"].shape:
        raise ValueError("A state needs matching unbatched [H,D] target and mask")
    task = str(item.get("task", str(item.get("episode", "")).split("/")[0]))
    structural = torch.ones(valid.shape[-1], dtype=torch.bool)
    if task in ("PatternLock", "RouteStick"):
        if valid.shape[-1] != 8:
            raise ValueError("Stick-task action convention requires eight canonical dimensions")
        structural[-1] = False
        if valid[:, -1].any():
            raise ValueError("The stick-task gripper must remain structurally unavailable")
    if "structural_valid" in item:
        declared = torch.as_tensor(item["structural_valid"]).bool().cpu()
        if declared.shape != structural.shape or not torch.equal(declared, structural):
            raise ValueError("Declared structural action mask differs from the task convention")
    terminal = (~valid) & structural[None]
    return {"structural_valid": structural.tolist(),
            "structurally_unavailable_coordinates": int((~structural).sum() * len(valid)),
            "terminal_or_unavailable_coordinates": int(terminal.sum()),
            "valid_coordinates": int(valid.sum()),
            "full_horizon": bool(not terminal.any()),
            "flow_input_convention": FLOW_CONVENTION}


@contextmanager
def _precision(device, precision):
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported explicit precision: {precision}")
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device.type, dtype=PRECISIONS[precision], enabled=precision != "fp32"):
            yield
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


@contextmanager
def _timed(timings, key, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - started


def _validate_models(teacher, student, teacher_hash, student_hash):
    for name, model, identity in (("teacher", teacher, teacher_hash), ("student", student, student_hash)):
        if not re.fullmatch(r"[0-9a-f]{64}", str(identity)):
            raise ValueError(f"{name} requires an exact checkpoint SHA256")
        if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Utility measurement requires every model in eval mode with gradients disabled")
        if not model.encoders_frozen:
            raise ValueError("Utility measurements require frozen encoders")
    if teacher.device != student.device:
        raise ValueError("Teacher and student must share the measurement device")
    for field in ("mean", "std", "count"):
        if not torch.equal(getattr(teacher.normalizer, field), getattr(student.normalizer, field)):
            raise ValueError("Teacher/student normalization differs")


def _noise(item, draws, device):
    # CPU generation makes draw bytes invariant to worker GPU assignment.
    noises, times = [], []
    for row in draws:
        generator = torch.Generator(device="cpu").manual_seed(int(row["seed"]))
        noises.append(torch.randn(item["target"].shape, generator=generator, dtype=torch.float32))
        times.append(torch.rand((), generator=generator, dtype=torch.float32))
    return torch.stack(noises).to(device), torch.stack(times).to(device)


def _losses(prediction, target, valid, prefix_targets):
    prefix = valid.clone()
    prefix[:, prefix_targets:] = False
    return (masked_mse(prediction, target, valid, reduction="none"),
            masked_mse(prediction, target, prefix, reduction="none"))


def _context_batches(contexts, maximum_batch, maximum_padded_tokens):
    if maximum_batch < 1 or maximum_padded_tokens < 1:
        raise ValueError("Inference batch/token limits must be positive")
    batch, length = [], 0
    for key in sorted(contexts, key=lambda k: (contexts[k].content.shape[1], k)):
        size = int(contexts[key].content.shape[1])
        if batch and (len(batch) >= maximum_batch or
                      (len(batch) + 1) * max(length, size) > maximum_padded_tokens):
            yield batch
            batch, length = [], 0
        batch.append(key)
        length = max(length, size)
    if batch:
        yield batch


def reference_payload_fingerprint(reference):
    hasher = hashlib.sha256()
    for name in ("predictions", "times", "expert_full", "expert_prefix"):
        _tensor_hash_update(hasher, name, reference[name])
    return hasher.hexdigest()


@torch.inference_mode()
def evaluate_state(teacher, student, item, contexts, draws, *, teacher_hash, student_hash,
                   precision="fp32", compact_batch_size=32, prefix_targets=10,
                   maximum_padded_tokens=32768, reference=None, reference_identity=None,
                   retain_predictions=False, reference_only=False):
    """Measure one frozen causal state and return raw per-context/draw rows.

    ``contexts`` maps stable manifest IDs to single-example TokenBanks. Adapted
    contexts are kept only for one bounded batch, reused for every paired draw,
    then released. A long full-leaf context is run alone, never truncated.

    The returned ``teacher_reference`` consists of small CPU tensors and a
    checksum. It may be serialized by the experiment runner and reused with a
    different student. ``reference_identity`` supplies cache/source identity
    from the outer manifest; any change invalidates reference reuse.
    """
    started = time.perf_counter()
    _validate_models(teacher, student, teacher_hash, student_hash)
    _validate_draws(draws)
    device = teacher.device
    mask_summary = action_mask_summary(item)
    horizon = item["target"].shape[0]
    if not 1 <= prefix_targets <= horizon or (not contexts and not reference_only):
        raise ValueError("Need contexts and an executed prefix within the predicted horizon")
    if precision not in PRECISIONS:
        raise ValueError("Invalid precision")
    if not mask_summary["valid_coordinates"]:
        raise ValueError("Cannot measure a state with no valid expert action coordinates")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings = {}
    with _timed(timings, "identity_seconds", device):
        identity = {
            "schema": SCHEMA, "teacher_hash": teacher_hash, "state": state_fingerprint(item),
            "draws": draws, "precision": precision, "tf32": False,
            "prefix_targets": prefix_targets, "flow_convention": FLOW_CONVENTION,
            "rng_convention": RNG_CONVENTION, "dependencies": reference_identity,
        }
        identity_hash = _json_hash(identity)
    noise, u = _noise(item, draws, device)
    valid = item["valid"].to(device).bool()[None].expand(len(draws), -1, -1)
    actions = item["target"].to(device)[None].expand(len(draws), -1, -1)
    x, _, target, _ = flow_sample(actions, valid, teacher.normalizer, noise=noise, time=u)
    if reference is None:
        with _precision(device, precision):
            with _timed(timings, "teacher_context_seconds", device):
                full_context, full_bias = policy_context(teacher, [item], [item["full"]])
            predictions = []
            with _timed(timings, "teacher_forward_seconds", device):
                # One full-history example at a time bounds attention memory.
                for draw in range(len(draws)):
                    predictions.append(teacher.decoder(x[draw:draw+1], u[draw:draw+1],
                                                       full_context, full_bias).float())
            teacher_prediction = torch.cat(predictions)
        teacher_full, teacher_prefix = _losses(teacher_prediction, target, valid, prefix_targets)
        reference = {"identity": identity, "identity_hash": identity_hash,
                     "predictions": teacher_prediction.cpu(), "times": u.cpu(),
                     "expert_full": teacher_full.cpu(), "expert_prefix": teacher_prefix.cpu()}
        reference["payload_hash"] = reference_payload_fingerprint(reference)
        del full_context, full_bias, predictions
    else:
        if reference.get("identity_hash") != identity_hash or reference.get("identity") != identity:
            raise ValueError("Teacher reference identity changed; do not reuse stale predictions")
        if reference.get("payload_hash") != reference_payload_fingerprint(reference):
            raise ValueError("Teacher reference payload checksum mismatch")
        if not torch.equal(reference["times"], u.cpu()) or reference["predictions"].shape != target.shape:
            raise ValueError("Teacher reference draw or shape mismatch")
        teacher_prediction = reference["predictions"].to(device)
        teacher_full, teacher_prefix = _losses(teacher_prediction, target, valid, prefix_targets)
        if not torch.equal(teacher_full.cpu(), reference["expert_full"]) or not torch.equal(
                teacher_prefix.cpu(), reference["expert_prefix"]):
            raise ValueError("Teacher reference loss mismatch")
        timings["teacher_context_seconds"] = timings["teacher_forward_seconds"] = 0.0
    if not torch.isfinite(teacher_prediction).all():
        raise FloatingPointError("Nonfinite teacher velocity prediction")
    if reference_only:
        timings.update(wall_seconds=time.perf_counter() - started,
                       compact_model_examples=0,
                       teacher_model_examples=len(draws) if timings["teacher_forward_seconds"] else 0,
                       contexts_adapted=0,
                       gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                       gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0)
        return {"rows": [], "teacher_reference": reference, "timings": timings,
                "mask": mask_summary, "precision": precision, "predictions": {}}
    flow_times = u.cpu().tolist()
    teacher_full_values = teacher_full.cpu().tolist()
    teacher_prefix_values = teacher_prefix.cpu().tolist()
    rows, retained = [], {}
    with _precision(device, precision):
        for keys in _context_batches(contexts, compact_batch_size, maximum_padded_tokens):
            with _timed(timings, "compact_context_seconds", device):
                context, bias = policy_context(student, [item] * len(keys), [contexts[key] for key in keys])
            for draw_index, draw in enumerate(draws):
                with _timed(timings, "compact_forward_seconds", device):
                    prediction = student.decoder(x[draw_index:draw_index+1].expand(len(keys), -1, -1),
                                                 u[draw_index:draw_index+1].expand(len(keys)),
                                                 context, bias).float()
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Nonfinite compact velocity prediction")
                expert_full, expert_prefix = _losses(prediction, target[draw_index:draw_index+1].expand_as(prediction),
                                                     valid[draw_index:draw_index+1].expand_as(prediction), prefix_targets)
                agreement_full, agreement_prefix = _losses(
                    prediction, teacher_prediction[draw_index:draw_index+1].expand_as(prediction),
                    valid[draw_index:draw_index+1].expand_as(prediction), prefix_targets)
                # A single small batched transfer, rather than four CUDA scalar
                # synchronizations for every candidate. Loss reduction is FP32.
                values = torch.stack((expert_full, expert_prefix, agreement_full, agreement_prefix), dim=1).cpu().tolist()
                saved_predictions = prediction.cpu() if retain_predictions else None
                for position, key in enumerate(keys):
                    rows.append({"student_hash": student_hash, "context_id": key, **draw,
                                 "flow_time": flow_times[draw_index],
                                 "expert_full": values[position][0],
                                 "expert_prefix": values[position][1],
                                 "teacher_full": values[position][2],
                                 "teacher_prefix": values[position][3],
                                 "teacher_expert_full": teacher_full_values[draw_index],
                                 "teacher_expert_prefix": teacher_prefix_values[draw_index]})
                    if retain_predictions:
                        retained[key, draw["bank"], draw["draw_index"]] = saved_predictions[position]
            del context, bias
    rows.sort(key=lambda row: (row["context_id"], row["bank"], row["draw_index"]))
    timings.update(wall_seconds=time.perf_counter() - started,
                   compact_model_examples=len(contexts) * len(draws),
                   teacher_model_examples=len(draws) if timings["teacher_forward_seconds"] else 0,
                   contexts_adapted=len(contexts),
                   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                   gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0)
    return {"rows": rows, "teacher_reference": reference, "timings": timings,
            "mask": mask_summary, "precision": precision, "predictions": retained}


def _maximum_prediction_difference(left, right):
    if left.keys() != right.keys():
        raise ValueError("Numerical comparison requires identical contexts and draws")
    return max(float((left[key] - right[key]).abs().max()) for key in left)


@torch.inference_mode()
def initial_full_leaf_parity(teacher, initialized_student, item, draws, *, teacher_hash,
                            student_hash, precision="fp32", prefix_targets=10,
                            atol=2e-6, reference_identity=None):
    """Check cached teacher-to-Stage-P initialization, before any student update.

    This is deliberately a numerical check rather than a guarantee that a
    trained compact decoder will retain its teacher's full-leaf behavior.
    """
    result = evaluate_state(teacher, initialized_student, item, {"full": item["full"]}, draws,
                            teacher_hash=teacher_hash, student_hash=student_hash,
                            precision=precision, prefix_targets=prefix_targets,
                            reference_identity=reference_identity, retain_predictions=True)
    differences = []
    for index, draw in enumerate(draws):
        prediction = result["predictions"]["full", draw["bank"], draw["draw_index"]]
        expected = result["teacher_reference"]["predictions"][index]
        differences.append(float((prediction - expected).abs().max()))
    maximum = max(differences)
    return {"passed": maximum <= atol, "max_abs_prediction_difference": maximum,
            "atol": float(atol), "precision": precision, "draws": len(draws),
            "history_leaves": int(item["history_leaves"]),
            "maximum_teacher_mse": max(row["teacher_full"] for row in result["rows"]),
            "note": "Initialization parity only; no inference about a trained student's full-history capability."}


@torch.inference_mode()
def numerical_controls(teacher, student, item, contexts, draws, *, teacher_hash, student_hash,
                       precision="fp32", prefix_targets=10, maximum_contexts=3,
                       reference_identity=None):
    """Duplicate/no-op, serial/packed/reversed order and higher-precision probes.

    No-op controls have no move cost. Returned loss differences estimate a
    numerical floor, not uncertainty or a practical effect threshold.
    """
    if maximum_contexts < 1:
        raise ValueError("Numerical controls need at least one context")
    keys = sorted(contexts)[:maximum_contexts]
    if not keys:
        raise ValueError("Numerical controls need at least one context")
    banks = {key: contexts[key] for key in keys}
    duplicate = "__duplicate_noop__"
    if duplicate in banks:
        raise ValueError("Reserved numerical-control context ID")
    banks[duplicate] = banks[keys[0]]
    common = dict(teacher_hash=teacher_hash, student_hash=student_hash, precision=precision,
                  prefix_targets=prefix_targets, retain_predictions=True,
                  reference_identity=reference_identity)
    packed = evaluate_state(teacher, student, item, banks, draws[:2], compact_batch_size=len(banks), **common)
    serial = evaluate_state(teacher, student, item, dict(reversed(list(banks.items()))), draws[:2],
                            compact_batch_size=1, reference=packed["teacher_reference"], **common)
    noop_prediction = max(float((packed["predictions"][keys[0], d["bank"], d["draw_index"]] -
                                 packed["predictions"][duplicate, d["bank"], d["draw_index"]]).abs().max())
                          for d in draws[:2])
    lookup = {(r["context_id"], r["bank"], r["draw_index"]): r for r in packed["rows"]}
    noop_loss = max(abs(lookup[keys[0], d["bank"], d["draw_index"]]["expert_full"] -
                        lookup[duplicate, d["bank"], d["draw_index"]]["expert_full"]) for d in draws[:2])
    serial_lookup = {(r["context_id"], r["bank"], r["draw_index"]): r for r in serial["rows"]}
    loss_floor = max(abs(r["expert_full"] - serial_lookup[k]["expert_full"]) for k, r in lookup.items())
    result = {"noop_prediction_max_abs": noop_prediction, "noop_expert_loss_abs": noop_loss,
              "serial_packed_prediction_max_abs": _maximum_prediction_difference(
                  packed["predictions"], serial["predictions"]),
              "serial_packed_expert_loss_max_abs": loss_floor,
              "raw_gain_numerical_floor": max(noop_loss, loss_floor),
              "contexts": len(banks), "draws": len(draws[:2]),
              "precision": precision, "same_budget_delta_tokens": 0,
              "note": "Numerical differences, not confidence intervals or practical effect thresholds."}
    if precision != "fp32":
        high = evaluate_state(teacher, student, item, banks, draws[:2], compact_batch_size=1,
                              **{**common, "precision": "fp32"})
        result["higher_precision_prediction_max_abs"] = _maximum_prediction_difference(
            serial["predictions"], high["predictions"])
        high_lookup = {(r["context_id"], r["bank"], r["draw_index"]): r for r in high["rows"]}
        result["higher_precision_expert_loss_max_abs"] = max(
            abs(r["expert_full"] - high_lookup[k]["expert_full"]) for k, r in serial_lookup.items())
        result["extra_model_examples"] = high["timings"]["compact_model_examples"] + high["timings"]["teacher_model_examples"]
    else:
        result["higher_precision_note"] = "Already FP32 with TF32 disabled; no reduced-precision comparison requested."
        result["extra_model_examples"] = 0
    result["model_examples"] = sum(r["timings"]["compact_model_examples"] +
                                   r["timings"]["teacher_model_examples"] for r in (packed, serial)) + result["extra_model_examples"]
    return result


@torch.inference_mode()
def generated_state(teacher, student, item, contexts, seeds, *, teacher_hash, student_hash,
                    task=None, precision="fp32", integration_steps=8, prefix_targets=10,
                    compact_batch_size=16, maximum_padded_tokens=32768):
    """Pure-noise chunk diagnostics with one fixed context throughout Euler.

    Physical commands use the actual task's clipping/gripper convention. FK is
    a predicted target path, never a simulated trajectory or success result.
    """
    _validate_models(teacher, student, teacher_hash, student_hash)
    task = task or item.get("task") or str(item.get("episode", "")).split("/")[0]
    robot = panda_variant(task)  # Reject unknown tasks rather than assuming a gripper.
    if integration_steps != 8:
        raise ValueError("This diagnostic's frozen generation protocol requires eight Euler steps")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Fresh independent generation seeds are required")
    if not 1 <= prefix_targets <= item["target"].shape[0]:
        raise ValueError("Invalid generated-action prefix")
    action_mask_summary({**item, "task": task})
    device = teacher.device
    noise = torch.stack([torch.randn(item["target"].shape,
                                   generator=torch.Generator().manual_seed(int(seed))) for seed in seeds]).to(device)
    valid = item["valid"].to(device).bool()[None].expand(len(seeds), -1, -1)
    expert = teacher.normalizer.normalize(item["target"].to(device))[None].expand_as(noise)
    output, timings = {}, {}

    def collect(model, key, prediction):
        full_loss, prefix_loss = _losses(prediction, expert, valid, prefix_targets)
        actions = model.normalizer.denormalize(prediction).cpu().numpy()
        physical = np.asarray([physical_action(row, task) for row in actions.reshape(-1, 8)]).reshape(actions.shape)
        tail_mask = valid.clone()
        tail_mask[:, :prefix_targets] = False
        tail_loss = masked_mse(prediction, expert, tail_mask, reduction="none")
        output[key] = {"normalized_actions": prediction.cpu().numpy(),
                       "physical_actions": physical,
                       "tcp_paths": panda_tcp_positions(physical[..., :7], task=task),
                       "expert_full": full_loss.cpu().tolist(), "expert_prefix": prefix_loss.cpu().tolist(),
                       "expert_tail": tail_loss.cpu().tolist(),
                       "gripper_commands": None if robot == "panda_stick" else physical[..., 7].tolist(),
                       "joint_clipping_count": int(np.count_nonzero(actions[..., :7] != physical[..., :7]))}

    with _precision(device, precision):
        with _timed(timings, "teacher_context_seconds", device):
            context, bias = policy_context(teacher, [item], [item["full"]])
        predictions = []
        with _timed(timings, "teacher_integration_seconds", device):
            for draw in range(len(seeds)):
                predictions.append(teacher.decoder.integrate(context, bias, steps=8,
                                                              noise=noise[draw:draw+1]).float())
        collect(teacher, "teacher_full", torch.cat(predictions))
        del context, bias, predictions
        if "teacher_full" in contexts:
            raise ValueError("teacher_full is reserved for the full-history reference")
        for keys in _context_batches(contexts, compact_batch_size, maximum_padded_tokens):
            with _timed(timings, "compact_context_seconds", device):
                context, bias = policy_context(student, [item] * len(keys), [contexts[key] for key in keys])
            predictions = []
            with _timed(timings, "compact_integration_seconds", device):
                for draw in range(len(seeds)):
                    predictions.append(student.decoder.integrate(
                        context, bias, steps=8, noise=noise[draw:draw+1].expand(len(keys), -1, -1)).float())
            stacked = torch.stack(predictions)
            for position, key in enumerate(keys):
                collect(student, key, stacked[:, position])
            del context, bias, predictions, stacked
    teacher_actions = torch.as_tensor(output["teacher_full"]["normalized_actions"])
    for value in output.values():
        full, prefix = _losses(torch.as_tensor(value["normalized_actions"]), teacher_actions,
                              valid.cpu(), prefix_targets)
        value["teacher_full"] = full.tolist()
        value["teacher_prefix"] = prefix.tolist()
    comparisons = {}
    for left, right in (("base", "immediate_completed"), ("base", "packet_completed"),
                        ("immediate_completed", "packet_completed"), ("packet_completed", "dispersed")):
        if left not in output or right not in output:
            continue
        a, b = output[left], output[right]
        full, prefix = _losses(torch.as_tensor(a["normalized_actions"]), torch.as_tensor(b["normalized_actions"]),
                              valid.cpu(), prefix_targets)
        distance = np.linalg.norm(a["tcp_paths"] - b["tcp_paths"], axis=-1) * 1000
        comparisons[f"{left}_vs_{right}"] = {
            "normalized_action_mse_full": full.tolist(), "normalized_action_mse_prefix": prefix.tolist(),
            "tcp_target_distance_mm_by_target": distance.tolist(),
            "tcp_target_distance_mm_prefix_mean": distance[:, :prefix_targets].mean(1).tolist(),
            "tcp_target_distance_mm_tail_mean": (distance[:, prefix_targets:].mean(1).tolist()
                                                  if prefix_targets < distance.shape[1] else None)}
    timings["model_examples"] = (1 + len(contexts)) * len(seeds) * integration_steps
    return {"contexts": output, "timings": timings, "seeds": list(map(int, seeds)),
            "task": task, "robot": robot, "precision": precision, "student_hash": student_hash,
            "teacher_hash": teacher_hash, "integration_steps": integration_steps,
            "generation_convention": GENERATION_CONVENTION, "rng_convention": RNG_CONVENTION,
            "prefix_targets": prefix_targets, "comparisons": comparisons,
            "note": "Predicted joint-target FK paths; no dynamics, contact or task-success inference."}
