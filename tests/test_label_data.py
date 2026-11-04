"""Synthetic diagnostics: indexed label preparation preserves scientific inputs."""

# Pytest injects the imported shared fixtures by parameter name.
# ruff: noqa: F811

from copy import deepcopy
from dataclasses import fields
import random

import h5py
import pytest
import torch

from test_cached_context_loader import fixture  # noqa: F401
from test_packed_context import indexed as indexed  # Explicit pytest fixture re-export.
from stmem.data.cache import episode_key, write_episode_cache
from stmem.data.hierarchy_cache import write_episode_hierarchy
from stmem.memory.flat import FlatState
from stmem.memory.frontier import split_frontier, validate_frontier
from stmem.models.batched_policy import records_bank
from stmem.models.policy import MemoryPolicy, choose_records
from stmem.models.router import CachedScorer, LocalRouter
from stmem.training.label_data import LabelDataReader, LabelPlan


def _configure(cfg):
    cfg["labels"].update(contexts=24, refresh_contexts=12, budgets=[8, 12, 16], candidates=4)
    return cfg


def _bank_equal(actual, expected):
    for field in fields(actual):
        torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name), rtol=0, atol=0)


def _reference(cfg, item, evidence, reader, scorer=None):
    rng = random.Random(item["seed"])
    rng.randrange(item["time_choices"])
    learned = scorer is not None and rng.random() >= cfg["labels"]["exploration_fraction"]
    if learned:
        records = evidence.archive.records if reader == "tree" else evidence.archive.leaves
        scorer.retain(records)
        scorer.prepare(records)
    cap = rng.randint(max(len(evidence.archive.roots), 1),
                      max(len(evidence.archive.roots), item["budget"] - 1))
    before = choose_records(evidence, reader, cap, scorer=scorer if learned else None, rng=rng).records
    if len(before) >= item["budget"]:
        return before, [], []
    if reader == "tree":
        candidates = [record for record in before if record.children]
        rng.shuffle(candidates)
        candidates = candidates[:cfg["labels"]["candidates"]]
        after = [split_frontier(before, record) for record in candidates]
    else:
        state = FlatState(tuple(evidence.archive.leaves), {record.node_id for record in before if record.is_leaf})
        state = state.normalize_singletons()
        before = state.records
        candidates = [record for record in state.unselected if state.move_cost(record) > 0]
        rng.shuffle(candidates)
        candidates = candidates[:cfg["labels"]["candidates"]]
        after = [state.move(record).normalize_singletons().records for record in candidates]
    return before, after, candidates


def test_plan_preserves_reference_samples_without_constructing_decisions(indexed, monkeypatch):
    cfg, data, _, _, _ = indexed
    _configure(cfg)
    expected = []
    for refresh in (False, True):
        contexts = cfg["labels"]["refresh_contexts" if refresh else "contexts"]
        rows = []
        for context_id in range(contexts):
            task = data.tasks[context_id % len(data.tasks)]
            episode_index = data.by_task[task][(context_id // len(data.tasks)) % len(data.by_task[task])]
            episode = data.episode(episode_index)
            times = [index for index in range(0, len(episode), data.stride)
                     if episode.decision(index, data.horizon).target_valid.any()]
            stratum = (context_id // len(cfg["labels"]["budgets"])) % 3
            lo, hi = len(times) * stratum // 3, len(times) * (stratum + 1) // 3
            seed = cfg["labels"]["seed"] + context_id + (1000000 if refresh else 0)
            rows.append((episode_index, random.Random(seed).choice(times[lo:hi] or times), seed))
        expected.append(rows)
    monkeypatch.setattr(data, "episode", lambda *_: pytest.fail("Plan must use the action-only index"))
    for refresh, reference in zip((False, True), expected):
        plan = LabelPlan(data, cfg, refresh)
        actual = plan.items()
        assert [(row["episode_index"], row["index"], row["seed"]) for row in actual] == reference
        shards = [plan.items(shard_index=rank, shard_count=4) for rank in range(4)]
        assert sorted([row for shard in shards for row in shard], key=lambda row: row["context_id"]) == actual
        assert plan.items([5, 0]) == [actual[5], actual[0]]


@pytest.mark.parametrize("no_state", [False, True])
def test_indexed_labels_match_reference_full_before_after_and_candidate_inputs(indexed, no_state):
    cfg, data, policy, cache, root = indexed
    _configure(cfg)
    if no_state:
        cfg["model"].update(proprioception_mode="none", condition_on_previous_commands=False)
        policy = MemoryPolicy(cfg).eval().requires_grad_(False)
        with torch.no_grad():
            for task in data.tasks:
                with h5py.File(cache.root / f"{task}.h5", "w") as base:
                    for episode_index in data.by_task[task]:
                        episode = data.episode(episode_index)
                        key = episode_key(task, episode.episode_id)
                        write_episode_cache(base.create_group(key), episode, policy, data.stride, data.horizon)
                with h5py.File(cache.root / f"{task}.h5", "r") as base, h5py.File(root / f"{task}.h5", "w") as out:
                    for key in base:
                        write_episode_hierarchy(base[key], out.create_group(key), {"synthetic_diagnostic": True},
                                                {"identity": {"model_encoder": cfg["model"], "data": cfg["data"]}})
    prototype = LocalRouter(cfg["model"]["width"], cfg["model"]["router_width"])
    counts = {"tree": 0, "flat": 0}
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        for item in LabelPlan(data, cfg).items():
            actual = reader.prepare(item)
            decision = data.episode(item["episode_index"]).decision(item["index"], data.horizon)
            evidence = cache.evidence(decision)
            clocks = torch.tensor([[evidence.clocks["conditioning"], evidence.clocks["execution"]]])
            torch.testing.assert_close(actual["target"], decision.target_chunk, rtol=0, atol=0)
            torch.testing.assert_close(actual["valid"], decision.target_valid, rtol=0, atol=0)
            torch.testing.assert_close(actual["query"], policy.routing_query(evidence, item["budget"]), rtol=0, atol=0)
            _bank_equal(actual["current"], records_bank(policy, [evidence.current_records], clocks))
            _bank_equal(actual["full"], records_bank(policy, [evidence.archive.leaves], clocks))
            assert actual["history_leaves"] == len(evidence.archive.leaves)
            assert actual["history_leaves"] <= item["history_leaf_bound"]
            def tensor_bytes(value):
                if isinstance(value, torch.Tensor):
                    return value.numel() * value.element_size()
                if isinstance(value, dict):
                    return sum(tensor_bytes(child) for child in value.values())
                if isinstance(value, list):
                    return sum(tensor_bytes(child) for child in value)
                if hasattr(value, "__dataclass_fields__"):
                    return sum(tensor_bytes(getattr(value, field.name)) for field in fields(value))
                return 0
            assert tensor_bytes(actual) <= item["estimated_bytes"]
            for family in ("tree", "flat"):
                before, after, candidates = _reference(cfg, item, evidence, family)
                prepared = actual["readers"][family]
                _bank_equal(prepared["before"], records_bank(policy, [before], clocks))
                assert len(prepared["after"]) == len(after)
                for got, expected in zip(prepared["after"], after):
                    validate_frontier(expected, evidence.archive.leaves, item["budget"])
                    _bank_equal(got, records_bank(policy, [expected], clocks))
                torch.testing.assert_close(prepared["candidate"], prototype.candidate_input(candidates), rtol=0, atol=0)
                assert prepared["candidate"].shape[0] == len(after)
                assert (prepared["costs"] > 0).all()
                counts[family] += len(after)
    assert all(count > 0 for count in counts.values())
    assert not reader.packed.base_files.values and not reader.packed.hierarchy_files.values


def test_indexed_flat_does_not_call_quadratic_reference_move_cost(indexed, monkeypatch):
    cfg, data, _, cache, root = indexed
    _configure(cfg)
    monkeypatch.setattr(FlatState, "move_cost", lambda *_: pytest.fail("Quadratic reference path"))
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        actual = reader.prepare(LabelPlan(data, cfg).items()[0])
    assert "flat" in actual["readers"]


def test_refresh_objective_scoped_scorers_match_frozen_reference(indexed):
    cfg, data, policy, cache, root = indexed
    _configure(cfg)
    cfg["labels"]["exploration_fraction"] = 0.0
    router = LocalRouter(cfg["model"]["width"], cfg["model"]["router_width"]).eval().requires_grad_(False)
    scorers = {name: CachedScorer(deepcopy(router), name) for name in ("velocity/tree", "action/flat")}
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        for item in LabelPlan(data, cfg, refresh=True).items():
            actual = reader.prepare(item, scorers)
            decision = data.episode(item["episode_index"]).decision(item["index"], data.horizon)
            evidence = cache.evidence(decision)
            clocks = torch.tensor([[evidence.clocks["conditioning"], evidence.clocks["execution"]]])
            assert set(actual["readers"]) == set(scorers)
            for name in scorers:
                scorer = CachedScorer(deepcopy(router), name)
                scorer.set_query(policy.routing_query(evidence, item["budget"]))
                before, after, candidates = _reference(cfg, item, evidence, name.rsplit("/", 1)[-1], scorer)
                _bank_equal(actual["readers"][name]["before"], records_bank(policy, [before], clocks))
                for got, expected in zip(actual["readers"][name]["after"], after):
                    _bank_equal(got, records_bank(policy, [expected], clocks))
                torch.testing.assert_close(actual["readers"][name]["candidate"], router.candidate_input(candidates),
                                           rtol=0, atol=0)


def test_requested_prefix_mismatch_is_rejected(indexed):
    cfg, data, _, cache, root = indexed
    _configure(cfg)
    item = LabelPlan(data, cfg).items()[0]
    episode = data.episode(item["episode_index"])
    key = episode_key(episode.task, episode.episode_id)
    with h5py.File(root / f"{episode.task}.h5", "a") as file:
        file[key]["decisions"][str(item["index"])].attrs["prefix_key"] = "future_block"
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        with pytest.raises(ValueError, match="cadence"):
            reader.prepare(item)


def test_history_inputs_do_not_depend_on_future_supervision(indexed):
    cfg, data, _, cache, root = indexed
    _configure(cfg)
    item = LabelPlan(data, cfg).items()[0]
    with LabelDataReader(data, cfg, cache.root, root) as reader:
        before = reader.prepare(item)
        # The current aligned prior command stays unchanged. Change only later
        # expert targets in the small action arrays, never observed features.
        episode = reader.decisions._episode(item["episode_index"])
        episode.actions = episode.actions.clone()
        episode.actions[item["index"] + 1:] += 99
        after = reader.prepare(item)
    assert not torch.equal(before["target"], after["target"])
    for name in ("query", "current_state", "text", "clocks"):
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    for name in ("current", "full"):
        _bank_equal(before[name], after[name])
    for family in ("tree", "flat"):
        _bank_equal(before["readers"][family]["before"], after["readers"][family]["before"])
        torch.testing.assert_close(before["readers"][family]["candidate"],
                                   after["readers"][family]["candidate"], rtol=0, atol=0)
