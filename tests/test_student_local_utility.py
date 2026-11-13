"""Synthetic CPU invariants for local student label-utility diagnostics."""
# ruff: noqa: F811
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_cached_context_loader import fixture  # noqa: F401
from stmem.inspection import student_utility as utility
from stmem.inspection.refinement_measurements import draw_plan
from stmem.memory.flat import FlatState
from stmem.models.action_flow import ActionNormalizer, flow_sample, masked_mse
from stmem.models.batched_policy import TokenBank, records_bank


def test_prepared_contexts_preserve_actual_causal_records_and_flat_residuals(fixture):
    cfg, data, policy, _ = fixture
    policy.freeze_encoders()
    decision = data.episode(0).decision(6, cfg["model"]["horizon"])
    assert len(decision.execution_prefix) % cfg["model"]["block_size"] == 1
    payload = utility.prepare_local_state(policy, decision, seed=71, budgets=(8,), packet_cost=2)
    assert payload["contexts"]["intervention/current_only"].content.shape[1] == 0
    for name, removed in (("erase_conditioning", "conditioning"), ("erase_execution", "execution")):
        spec = payload["panel"]["interventions"][name]
        assert not spec["exact_coverage"]
        assert spec["context_id"] not in payload["panel"]["contexts"]
        assert all(record["source"] != removed for record in payload["visual"]["context_records"][spec["context_id"]])
    with torch.inference_mode():
        evidence = policy.encode(decision)
    record_map = {r.node_id: r for r in evidence.archive.records}
    for name, spec in payload["panel"]["contexts"].items():
        if spec["reader"] == "TREE":
            records = [record_map[node] for node in spec["node_ids"]]
        else:
            records = FlatState(evidence.archive.leaves, set(spec["node_ids"])).records
        with torch.inference_mode():
            expected = records_bank(policy, [records], torch.tensor([payload["item"]["clocks"]]))
        actual = payload["contexts"][name]
        assert actual.content.shape[1] == spec["token_count"]
        for field in ("content", "metadata", "valid", "log_count"):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), atol=2e-6, rtol=2e-6)
        assert sorted(leaf for record in records for leaf in record.leaf_ids) == sorted(record_map[node].node_id
            for node in record_map if record_map[node].is_leaf)
    altered = replace(decision, target_chunk=decision.target_chunk + 50)
    repeat = utility.prepare_local_state(policy, altered, seed=71, budgets=(8,), packet_cost=2)
    assert repeat["panel"] == payload["panel"]
    for key in payload["contexts"]:
        torch.testing.assert_close(repeat["contexts"][key].content, payload["contexts"][key].content, rtol=0, atol=0)


def test_gain_blind_state_selection_respects_cadence_and_excludes_terminal_padding(fixture):
    cfg, data, _, _ = fixture
    episodes = [replace(data.episode(i), task="VideoUnmask", episode_id=f"episode_{i}") for i in (0, 1)]
    dataset = SimpleNamespace(episodes=[SimpleNamespace(task="VideoUnmask", key=f"episode_{i}") for i in (0, 1)],
                              load_episode=lambda index: episodes[index])
    rows = utility.select_local_states(dataset, cfg, maximum=7)
    assert len(rows) == 7
    assert {r["state"]["episode_id"] for r in rows} == {"episode_0", "episode_1"}
    assert all(r["decision"].target_valid.all() for r in rows)
    assert all(r["decision"].current.source_index % 2 == 0 for r in rows)
    changed = deepcopy(episodes)
    for episode in changed:
        episode.actions += 1000
    dataset.load_episode = lambda index: changed[index]
    assert [r["state"] for r in rows] == [r["state"] for r in utility.select_local_states(dataset, cfg, maximum=7)]


class ToyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalizer = ActionNormalizer()
        self.decoder = ToyDecoder()
        self.encoders_frozen = True
        self.device = torch.device("cpu")
        self.eval().requires_grad_(False)


class ToyDecoder(nn.Module):
    def forward(self, x, u, context, bias):
        return context[:, :1, :1].expand_as(x) + x * 0.25 + u[:, None, None]


def bank(value):
    return TokenBank(torch.full((1, 1, 1), float(value)), torch.zeros(1, 1, 23),
                     torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1))


def toy_payload():
    return {"item": {"task": "VideoUnmask", "target": torch.arange(16).reshape(2, 8).float() / 10,
                     "valid": torch.ones(2, 8, dtype=torch.bool), "current": bank(0), "full": bank(1),
                     "text": torch.zeros(1), "current_state": torch.zeros(24), "clocks": [0., 1.],
                     "history_leaves": 1},
            "contexts": {"before": bank(0), "after": bank(1)}}


def test_expert_losses_are_real_targets_with_paired_noise_and_no_fabricated_teacher(monkeypatch):
    monkeypatch.setattr(utility, "policy_context", lambda policy, items, histories:
                        (torch.cat([b.content for b in histories]), None))
    policy, prepared = ToyPolicy(), toy_payload()
    draws = draw_plan(73, "VideoUnmask", "episode_0", 16, discovery=2, confirmation=2)
    result = utility.evaluate_student_state(policy, prepared, draws, checkpoint_hash="b" * 64, prefix_targets=1)
    assert not result["teacher_agreement_available"]
    assert all(row["teacher_full"] is None for row in result["rows"])
    assert result["timings"]["teacher_model_examples"] == 0
    assert result["numerical"]["noop_prediction_max_abs"] == 0
    assert result["numerical"]["noop_loss_max_abs"] == 0
    for row in result["rows"]:
        generator = torch.Generator().manual_seed(row["seed"])
        noise, u = torch.randn(2, 8, generator=generator), torch.rand((), generator=generator)
        item = prepared["item"]
        x, u, target, _ = flow_sample(item["target"], item["valid"], policy.normalizer, noise=noise, time=u)
        prediction = policy.decoder(x[None], u[None], prepared["contexts"][row["context_id"]].content, None)[0]
        assert row["expert_full"] == pytest.approx(float(masked_mse(prediction, target, item["valid"])), abs=1e-6)
    reverse = deepcopy(prepared)
    reverse["contexts"] = dict(reversed(list(reverse["contexts"].items())))
    other = utility.evaluate_student_state(policy, reverse, draws, checkpoint_hash="b" * 64,
                                           prefix_targets=1, batch_size=1)
    assert other["rows"] == result["rows"]


def test_frozen_teacher_identity_and_normalizer_are_required(monkeypatch):
    policy, prepared = ToyPolicy(), toy_payload()
    draws = draw_plan(2, "VideoUnmask", "episode_0", 0, discovery=1, confirmation=1)
    with pytest.raises(ValueError, match="cannot substitute"):
        utility.evaluate_student_state(policy, prepared, draws, checkpoint_hash="b" * 64,
                                       teacher_hash="a" * 64, prefix_targets=1)
    teacher = ToyPolicy()
    teacher.normalizer.mean += 1
    with pytest.raises(ValueError, match="normalization"):
        utility.evaluate_student_state(policy, prepared, draws, checkpoint_hash="b" * 64,
                                       teacher=teacher, teacher_hash="a" * 64, prefix_targets=1)
    policy.train()
    with pytest.raises(ValueError, match="eval mode"):
        utility.evaluate_student_state(policy, prepared, draws, checkpoint_hash="b" * 64, prefix_targets=1)


def test_optional_teacher_agreement_can_disagree_with_actual_expert_utility(monkeypatch):
    monkeypatch.setattr(utility, "policy_context", lambda policy, items, histories:
                        (torch.cat([b.content for b in histories]), None))
    monkeypatch.setattr(ToyDecoder, "forward", lambda self, x, u, context, bias:
                        context[:, :1, :1].expand_as(x))
    prepared = toy_payload()
    prepared["item"]["full"] = bank(3.)
    prepared["contexts"]["after"] = bank(2.)
    prepared["item"]["target"] = torch.randn(2, 8, generator=torch.Generator().manual_seed(55))
    result = utility.evaluate_student_state(ToyPolicy(), prepared,
        [{"bank": "discovery", "draw_index": 0, "seed": 55}], checkpoint_hash="b" * 64,
        teacher=ToyPolicy(), teacher_hash="a" * 64, prefix_targets=1)
    rows = {row["context_id"]: row for row in result["rows"]}
    assert result["teacher_agreement_available"]
    assert rows["before"]["teacher_expert_full"] == pytest.approx(9.)
    assert rows["before"]["expert_full"] - rows["after"]["expert_full"] == pytest.approx(-4.)
    assert rows["before"]["teacher_full"] - rows["after"]["teacher_full"] == pytest.approx(8.)


def test_discovery_selection_cannot_follow_confirmation_winner_and_ties_average():
    panel = {"families": [{"reader": "TREE", "panel": "packet", "background": 120,
                           "final_budget": 128, "base": "base", "controls": {"dispersed": "control"},
                           "candidates": [{"candidate_id": name, "prefixes": {"1": name, "8": name}}
                                          for name in ("a", "b")]}], "eligibility": []}
    rows = [{"context_id": key, "bank": bank_name, "draw_index": draw, "expert_full": value,
             "teacher_full": None}
            for bank_name, values in (("discovery", {"base": 4., "a": 0., "b": 2., "control": 3.}),
                                      ("confirmation", {"base": 4., "a": 2., "b": 0., "control": 3.}))
            for key, value in values.items() for draw in range(2)]
    summary = utility.summarize_utility(panel, rows)
    selected = summary["selection"][0]
    assert selected["selection"]["packet"] == ["a"]
    assert selected["packet_loss"] == 2
    assert selected["packet_vs_random"] == -1
    assert selected["winner_reversal"] is True
    for row in rows:
        if row["bank"] == "discovery" and row["context_id"] == "b":
            row["expert_full"] = 0.
    tie = utility.summarize_utility(panel, rows)["selection"][0]
    assert tie["selection"]["packet"] == ["a", "b"]
    assert tie["packet_loss"] == 1
    assert tie["winner_reversal"] is None
