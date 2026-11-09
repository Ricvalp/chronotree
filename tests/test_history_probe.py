from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from stmem.inspection.history_probe import ARMS, ProbedOnlinePolicy, probe_same_current
from stmem.models.policy import Evidence


class Decoder:
    horizon = 16
    dimensions = 8

    def __init__(self):
        self.steps = []

    def integrate(self, context, bias, *, steps=8, noise=None, generator=None):
        self.steps.append(steps)
        if noise is None:
            noise = torch.randn((self.horizon, self.dimensions), generator=generator)
        return noise * .001 + context.sum() * .001


class FakePolicy:
    device = torch.device("cpu")
    training = False

    def __init__(self):
        self.decoder = Decoder()
        self.context_calls = []
        self.normalizer = SimpleNamespace(denormalize=lambda value: value)
        self.encoder = SimpleNamespace(current=self.encode_current)

    def encode_current(self, current):
        # Deliberately exercise global RNG restoration, while keeping deterministic
        # values. A real frozen encoder should not need randomness at all.
        torch.rand(())
        random.random()
        np.random.random()
        return current.tokens

    def context(self, evidence, records):
        self.context_calls.append((evidence, list(records)))
        values = [1., 2., evidence.clocks["conditioning"], evidence.clocks["execution"]]
        values.extend(record.value for record in records)
        context = torch.tensor(values)[:, None]
        return context, torch.zeros(len(context))


class FakeOnline:
    method = "FULL"
    intervention = "none"
    precision = "fp32"
    autocast_dtype = torch.float32
    goal = "pick the blue cube"
    cfg = {"model": {"flow_steps": 8, "execute": 10}}

    def __init__(self, empty=False):
        self.policy = FakePolicy()
        self.text = torch.tensor([3.])
        self.clocks = {"conditioning": 20., "execution": 40.}
        self.generator = torch.Generator().manual_seed(123)
        self.selection_rng = random.Random(123)
        self.decision_index = 0
        leaves = [] if empty else [
            SimpleNamespace(node_id="conditioning:0", source="conditioning", value=6.),
            SimpleNamespace(node_id="execution:0", source="execution", value=4.),
        ]
        self.memory = SimpleNamespace(archive=SimpleNamespace(leaves=leaves),
                                      current=SimpleNamespace(source_index=40, tokens=["current"]))

    def reset(self, reset, seed):
        self.generator.manual_seed(seed)
        self.decision_index = 0
        return reset

    def observe(self, *args, **kwargs):
        return args, kwargs

    def decide(self):
        evidence = Evidence(self.memory.archive, self.policy.encoder.current(self.memory.current),
                            self.text, self.memory.current, self.clocks)
        context, bias = self.policy.context(evidence, self.memory.archive.leaves)
        actions = self.policy.normalizer.denormalize(self.policy.decoder.integrate(
            context, bias, steps=8, generator=self.generator,
        ))
        self.decision_index += 1
        self.last_return = actions, {"decision_index": self.decision_index - 1}
        return self.last_return


def test_shared_noise_replay_and_unchanged_inputs_rng():
    online = FakeOnline()
    before = online.generator.get_state().clone()
    actual, _ = online.decide()
    after = online.generator.get_state().clone()
    py_rng, np_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    original_clocks = online.clocks.copy()
    archive = online.memory.archive
    originals = archive.leaves.copy()
    row = probe_same_current(online, task="VideoUnmask", generator_state=before,
                             executed_actions=actual, decision_index=0)
    assert row["full_replay_passed"] and row["full_replay_max_abs_action_error"] == 0
    assert torch.equal(online.generator.get_state(), after)
    assert random.getstate() == py_rng and torch.equal(torch.get_rng_state(), torch_rng)
    now_np = np.random.get_state()
    assert now_np[0] == np_rng[0] and np.array_equal(now_np[1], np_rng[1]) and now_np[2:] == np_rng[2:]
    assert online.memory.archive is archive and archive.leaves == originals
    assert online.clocks == original_clocks
    probe_calls = online.policy.context_calls[-4:]
    evidence = probe_calls[0][0]
    assert all(call[0] is evidence for call in probe_calls)
    assert evidence.current is online.memory.current
    assert evidence.current_records is online.memory.current.tokens and evidence.text is online.text
    assert [r.source for r in probe_calls[1][1]] == ["execution"]
    assert [r.source for r in probe_calls[2][1]] == ["conditioning"]
    assert probe_calls[3][1] == []
    assert online.policy.decoder.steps == [8] * 5
    assert row["arms"]["FULL"]["tcp_change_mean_mm"] == 0
    assert row["arms"]["CURRENT"]["tcp_change_mean_mm"] > 0


def test_empty_history_has_identical_predictions_in_every_arm():
    online = FakeOnline(empty=True)
    before = online.generator.get_state().clone()
    actual, _ = online.decide()
    row = probe_same_current(online, task="VideoRepick", generator_state=before,
                             executed_actions=actual, decision_index=0)
    assert all(row["arms"][arm]["tcp_change_mean_mm"] == 0 for arm in ARMS)


def test_wrapper_returns_original_actions_and_preserves_future_stream():
    reference, observed = FakeOnline(), FakeOnline()
    wrapped = ProbedOnlinePolicy(observed, task="VideoUnmask", decision_indices=(0, 2))
    for _ in range(4):
        expected, _ = reference.decide()
        actual, timing = wrapped.decide()
        assert torch.equal(actual, expected)
        assert actual is observed.last_return[0] and timing is observed.last_return[1]
        assert torch.equal(reference.generator.get_state(), observed.generator.get_state())
        assert reference.decision_index == observed.decision_index
    assert len(wrapped.probes) == 2
    assert all(r["status"] == "passed" for r in wrapped.probes)
    marker = object()
    assert wrapped.reset(marker, 123) is marker and wrapped.probes == []


def test_probe_failure_remains_separate_from_normal_rollout(monkeypatch):
    import stmem.inspection.history_probe as module

    online = FakeOnline()
    wrapped = ProbedOnlinePolicy(online, task="VideoUnmask", decision_indices=(0,))
    def broken(*args, **kwargs):
        raise RuntimeError("Diagnostic deliberately unavailable")
    monkeypatch.setattr(module, "probe_same_current", broken)
    actual, _ = wrapped.decide()
    assert actual is online.last_return[0]
    assert wrapped.probes[0]["status"] == "error"
    assert "Diagnostic deliberately unavailable" in wrapped.probes[0]["error"]


def test_nonfull_reference_rejected():
    online = FakeOnline()
    online.intervention = "erase_conditioning"
    with pytest.raises(ValueError, match="ordinary FULL"):
        ProbedOnlinePolicy(online, task="VideoUnmask")
