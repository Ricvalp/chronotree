"""Synthetic CPU checks of the utility study's actual scientific comparisons."""

# ruff: noqa: F811
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from test_cached_context_loader import fixture  # noqa: F401
from test_packed_context import indexed as indexed
from test_label_data import _configure
from test_label_objectives import policies
from stmem.inspection import refinement_measurements as measurements
from stmem.inspection.refinement_generated import load_frozen_policy
from stmem.models.action_flow import ActionNormalizer, flow_sample, masked_mse
from stmem.models.batched_policy import TokenBank
from stmem.training.label_data import LabelDataReader, LabelPlan
from stmem.training.label_objectives import policy_context


TEACHER = "a" * 64
STUDENT = "b" * 64


def payload(indexed):
    cfg, data, teacher, cache, root = indexed
    _configure(cfg)
    teacher, student = policies(teacher)
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        item = reader.prepare(LabelPlan(data, cfg).items()[2])
    contexts = {"tree_base": item["readers"]["tree"]["before"],
                "flat_base": item["readers"]["flat"]["before"], "full": item["full"]}
    for family in ("tree", "flat"):
        contexts.update({f"{family}_{i}": bank for i, bank in enumerate(item["readers"][family]["after"])})
    draws = measurements.draw_plan(7, "TaskA", "episode_0", 2, discovery=2, confirmation=2)
    return teacher, student, item, contexts, draws


def evaluate(teacher, student, item, contexts, draws, **kwargs):
    return measurements.evaluate_state(teacher, student, item, contexts, draws,
                                       teacher_hash=TEACHER, student_hash=STUDENT, prefix_targets=1, **kwargs)


def row_map(result):
    return {(r["context_id"], r["bank"], r["draw_index"]): r for r in result["rows"]}


def test_actual_flow_targets_and_independent_noise_banks(indexed):
    teacher, student, item, contexts, draws = payload(indexed)
    result = evaluate(teacher, student, item, contexts, draws)
    assert len(set(d["seed"] for d in draws)) == 4
    assert measurements.draw_plan(7, "TaskA", "episode_0", 2, discovery=2, confirmation=2) == draws
    assert measurements.draw_plan(7, "TaskA", "episode_0", 3, discovery=2, confirmation=2) != draws
    rows = row_map(result)
    for draw in draws:
        generator = torch.Generator().manual_seed(draw["seed"])
        noise = torch.randn(item["target"].shape, generator=generator)
        time = torch.rand((), generator=generator)
        x, u, velocity_target, _ = flow_sample(item["target"], item["valid"], teacher.normalizer,
                                               noise=noise, time=time)
        for key, bank in contexts.items():
            with torch.inference_mode():
                context, bias = policy_context(student, [item], [bank])
                prediction = student.decoder(x[None], u[None], context, bias)[0]
            wanted = masked_mse(prediction, velocity_target, item["valid"])
            got = rows[key, draw["bank"], draw["draw_index"]]
            assert got["flow_time"] == float(time)
            assert got["expert_full"] == pytest.approx(float(wanted), abs=2e-6, rel=3e-5)
    assert result["timings"]["contexts_adapted"] == len(contexts)
    assert result["timings"]["teacher_model_examples"] == len(draws)


def test_serial_batch_order_noop_and_reusable_teacher_reference(indexed):
    teacher, student, item, contexts, draws = payload(indexed)
    contexts["noop"] = contexts["tree_base"]
    batched = evaluate(teacher, student, item, contexts, draws, retain_predictions=True)
    serial = evaluate(teacher, student, item, dict(reversed(list(contexts.items()))), draws,
                      compact_batch_size=1, retain_predictions=True,
                      reference=batched["teacher_reference"])
    assert serial["timings"]["teacher_model_examples"] == 0
    for key, a in row_map(batched).items():
        b = row_map(serial)[key]
        for metric in ("expert_full", "expert_prefix", "teacher_full", "teacher_prefix"):
            assert a[metric] == pytest.approx(b[metric], abs=2e-6, rel=3e-5)
    for draw in draws:
        torch.testing.assert_close(batched["predictions"]["noop", draw["bank"], draw["draw_index"]],
                                   batched["predictions"]["tree_base", draw["bank"], draw["draw_index"]],
                                   atol=2e-6, rtol=3e-5)
    # New student may reuse the teacher but produces new student-specific rows.
    other = deepcopy(student)
    other.decoder.out[-1].bias.add_(0.2)
    changed = measurements.evaluate_state(teacher, other, item, contexts, draws,
                                          teacher_hash=TEACHER, student_hash="c" * 64,
                                          prefix_targets=1, reference=batched["teacher_reference"])
    assert changed["teacher_reference"]["identity_hash"] == batched["teacher_reference"]["identity_hash"]
    assert changed["rows"][0]["student_hash"] != batched["rows"][0]["student_hash"]
    assert changed["rows"][0]["expert_full"] != batched["rows"][0]["expert_full"]


def test_reference_rejects_changed_evidence_precision_draws_and_corruption(indexed):
    teacher, student, item, contexts, draws = payload(indexed)
    result = evaluate(teacher, student, item, contexts, draws, reference_identity={"cache": "original"})
    reference = result["teacher_reference"]
    altered = deepcopy(item)
    altered["target"][0, 0] += 1
    for options in ({"item": altered}, {"reference_identity": {"cache": "changed"}},
                    {"draws": draws[:-1]}, {"precision": "bf16"}):
        settings = dict(item=item, draws=draws, reference_identity={"cache": "original"}, **{})
        settings.update(options)
        with pytest.raises(ValueError, match="identity"):
            evaluate(teacher, student, settings.pop("item"), contexts, settings.pop("draws"),
                     reference=reference, **settings)
    corrupt = deepcopy(reference)
    corrupt["predictions"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="checksum"):
        evaluate(teacher, student, item, contexts, draws, reference=corrupt,
                 reference_identity={"cache": "original"})


def test_reference_only_computation_does_not_run_student_context_or_decoder(indexed, monkeypatch):
    teacher, student, item, contexts, draws = payload(indexed)
    def reject(*_args, **_kwargs):
        raise AssertionError("Student decoder should not run while building shared teacher references")
    monkeypatch.setattr(student.decoder, "forward", reject)
    result = evaluate(teacher, student, item, {}, draws, reference_only=True)
    assert result["rows"] == []
    assert result["timings"]["teacher_model_examples"] == len(draws)
    assert result["timings"]["compact_model_examples"] == 0
    assert result["timings"]["contexts_adapted"] == 0


def test_frozen_and_initial_full_leaf_contract(indexed):
    teacher, student, item, contexts, draws = payload(indexed)
    initialized = deepcopy(teacher)
    parity = measurements.initial_full_leaf_parity(teacher, initialized, item, draws,
                                                    teacher_hash=TEACHER, student_hash=STUDENT, prefix_targets=1)
    assert parity["passed"] and parity["maximum_teacher_mse"] < 1e-12
    student.decoder.train()
    with pytest.raises(ValueError, match="eval mode"):
        evaluate(teacher, student, item, contexts, draws)
    student.eval()
    student.decoder.out[-1].bias.requires_grad_(True)
    with pytest.raises(ValueError, match="gradients disabled"):
        evaluate(teacher, student, item, contexts, draws)


def test_generated_loader_freezes_teacher_checkpoint_before_backend(indexed, tmp_path):
    cfg, _, _, _, _ = indexed
    teacher, student, item, contexts, draws = payload(indexed)
    paths = [tmp_path / "teacher.pt", tmp_path / "compact.pt"]
    for path, model, stage in zip(paths, (teacher, student), ("teacher", "compact")):
        torch.save({"stage": stage, "config": cfg, "model": model.state_dict()}, path)
    # Exercise the actual checkpoint/model loader. Teacher checkpoints are
    # trainable by default, unlike compact checkpoints, so eval() alone fails.
    loaded = [load_frozen_policy(path, "cpu")[0] for path in paths]
    for model in loaded:
        assert model.encoders_frozen
        assert not any(module.training for module in model.modules())
        assert not any(parameter.requires_grad for parameter in model.parameters())
    result = measurements.generated_state(loaded[0], loaded[1], item, {"base": contexts["tree_base"]},
                                          [draws[0]["seed"]], teacher_hash=TEACHER,
                                          student_hash=STUDENT, task="PickXtimes", prefix_targets=1)
    assert np.isfinite(result["contexts"]["base"]["normalized_actions"]).all()


def test_numerical_controls_measure_noop_without_fake_move_cost(indexed):
    teacher, student, item, contexts, draws = payload(indexed)
    result = measurements.numerical_controls(teacher, student, item, contexts, draws,
                                             teacher_hash=TEACHER, student_hash=STUDENT, prefix_targets=1)
    assert result["same_budget_delta_tokens"] == 0
    assert result["noop_prediction_max_abs"] <= 2e-6
    assert result["raw_gain_numerical_floor"] <= 2e-6
    assert result["serial_packed_prediction_max_abs"] <= 2e-6


def test_expert_and_teacher_objectives_can_reproducibly_prefer_opposite_contexts(monkeypatch):
    class Decoder(nn.Module):
        def forward(self, x, u, context, bias):
            return context[:, :1, :1].expand_as(x)

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.normalizer = ActionNormalizer()
            self.decoder = Decoder()
            self.encoders_frozen = True
            self.device = torch.device("cpu")
            self.eval().requires_grad_(False)

    def bank(value):
        return TokenBank(torch.full((1, 1, 1), value), torch.zeros(1, 1, 23),
                         torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1))

    monkeypatch.setattr(measurements, "policy_context", lambda policy, items, histories:
                        (torch.cat([b.content for b in histories]), None))
    draws = [{"bank": "discovery", "draw_index": 0, "seed": 55}]
    noise = torch.randn(2, 8, generator=torch.Generator().manual_seed(55))
    item = {"target": noise, "valid": torch.ones(2, 8, dtype=torch.bool),
            "current": bank(0.0), "full": bank(3.0), "text": torch.zeros(1),
            "current_state": torch.zeros(24), "clocks": (0.0, 1.0), "history_leaves": 1}
    result = evaluate(Policy(), Policy(), item, {"before": bank(0.0), "after": bank(2.0)}, draws)
    rows = {row["context_id"]: row for row in result["rows"]}
    assert rows["before"]["expert_full"] - rows["after"]["expert_full"] == pytest.approx(-4.0)
    assert rows["before"]["teacher_full"] - rows["after"]["teacher_full"] == pytest.approx(8.0)


@pytest.mark.parametrize("task,gripper", [("VideoUnmask", True), ("RouteStick", False)])
def test_generated_chunks_use_task_specific_physical_actions_and_fk(indexed, task, gripper):
    teacher, student, item, contexts, draws = payload(indexed)
    item = deepcopy(item)
    item["task"] = task
    if not gripper:
        item["valid"][:, 7] = False
    result = measurements.generated_state(teacher, student, item, {"base": contexts["tree_base"]}, [77, 89],
                                          teacher_hash=TEACHER, student_hash=STUDENT,
                                          prefix_targets=1)
    assert result["integration_steps"] == 8
    assert result["contexts"]["base"]["tcp_paths"].shape == (2, 2, 3)
    assert np.isfinite(result["contexts"]["base"]["tcp_paths"]).all()
    commands = result["contexts"]["base"]["physical_actions"][..., 7]
    assert set(np.unique(commands)) <= ({-1.0, 1.0} if gripper else {0.0})
    assert (result["contexts"]["base"]["gripper_commands"] is not None) == gripper
    # Ordinary inference starts from unmasked pure noise and is independent of
    # expert targets/masks except for the final diagnostic losses.
    changed = deepcopy(item)
    changed["target"] += 100
    again = measurements.generated_state(teacher, student, changed, {"base": contexts["tree_base"]}, [77, 89],
                                         teacher_hash=TEACHER, student_hash=STUDENT, prefix_targets=1)
    np.testing.assert_array_equal(result["contexts"]["base"]["normalized_actions"],
                                  again["contexts"]["base"]["normalized_actions"])


def test_structural_gripper_absence_is_not_terminal_padding():
    item = {"task": "RouteStick", "target": torch.zeros(16, 8), "valid": torch.ones(16, 8, dtype=torch.bool)}
    item["valid"][:, 7] = False
    summary = measurements.action_mask_summary(item)
    assert summary["full_horizon"]
    assert summary["structurally_unavailable_coordinates"] == 16
    assert summary["terminal_or_unavailable_coordinates"] == 0
    item["valid"][-1, :7] = False
    summary = measurements.action_mask_summary(item)
    assert not summary["full_horizon"]
    assert summary["terminal_or_unavailable_coordinates"] == 7
