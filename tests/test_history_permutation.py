from dataclasses import fields
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stmem.analyze import intervene_evidence
from stmem.inspection.history_permutation import PermutedExecutionOnlinePolicy, permute_execution_records
from stmem.memory.hierarchy import Archive
from stmem.memory.records import MemoryRecord, aggregate, leaf_record
from stmem.models.action_flow import ActionFlow
from stmem.models.policy import Evidence


def make_evidence():
    archive = Archive()
    index = 0
    for source, block, completed in (("conditioning", 0, True), ("execution", 0, True),
                                     ("execution", 1, True), ("execution", 2, False)):
        leaves = []
        for modality in ("visual", "event"):
            for within in range(3):
                index += 1
                # Repeated times test the required node-ID tie-breaking.
                time = float(block * 10 + within // 2)
                anchor = torch.tensor([index * .01, index * .02, index * .03])
                leaf = leaf_record(f"{source}:{modality}:{index}", torch.arange(8).float() + index,
                                   anchor, time, source, modality, bounds_min=anchor - .1,
                                   bounds_max=anchor + .2, time_min=time - .25, time_max=time + .5,
                                   completed=completed, occupancy=index / 100)
                leaves.append(leaf)
        if completed:
            archive.add_block(leaves, f"{source}:{block}")
        else:
            archive.set_pending(source, leaves)
    return Evidence(archive, [object()], torch.tensor([1.]),
                    SimpleNamespace(source_index=44), {"conditioning": 1., "execution": 44.})


def assert_records_equal(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b):
        for field in fields(MemoryRecord):
            lv, rv = getattr(left, field.name), getattr(right, field.name)
            if isinstance(lv, torch.Tensor):
                assert torch.equal(lv, rv), field.name
            else:
                assert lv == rv, field.name


@pytest.mark.parametrize("seed", [0, 1701, 9187])
def test_exact_parity_all_leaf_fields_completed_and_pending(seed):
    ev = make_evidence()
    original = ev.archive.leaves
    before = {record.node_id: record.content_sum.clone() for record in original}
    expected = intervene_evidence(ev, "permute_execution", seed)
    actual = permute_execution_records(original, seed)
    assert_records_equal(actual, expected.archive.leaves)
    assert all(new is old for new, old in zip(actual, original) if old.source == "conditioning")
    assert all(torch.equal(record.content_sum, before[record.node_id]) for record in original)
    assert ev.current is expected.current and ev.clocks is expected.clocks


def test_global_random_streams_unchanged_and_no_parent_aggregation(monkeypatch):
    ev = make_evidence()
    python, numpy, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    def forbidden(*args, **kwargs):
        raise AssertionError("No hierarchy may be rebuilt")
    monkeypatch.setattr(Archive, "add_block", forbidden)
    monkeypatch.setattr(Archive, "set_pending", forbidden)
    actual = permute_execution_records(ev.archive.leaves, 1701)
    assert len(actual) == len(ev.archive.leaves)
    assert random.getstate() == python and np.array_equal(np.random.get_state()[1], numpy[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_identical_attention_and_flow_output_with_same_noise():
    ev = make_evidence()
    expected = intervene_evidence(ev, "permute_execution", 1701).archive.leaves
    actual = permute_execution_records(ev.archive.leaves, 1701)
    # Real action flow sees the exact same ordered latent context and bias.
    decoder = ActionFlow(width=8, heads=2, layers=1, horizon=4).eval()
    noise = torch.randn(4, 8, generator=torch.Generator().manual_seed(77))
    context = torch.stack([r.content_sum for r in actual])
    legacy = torch.stack([r.content_sum for r in expected])
    bias = torch.zeros(len(actual))
    assert torch.equal(decoder.integrate(context, bias, steps=8, noise=noise),
                       decoder.integrate(legacy, bias, steps=8, noise=noise))


def test_empty_execution_is_identity_and_parents_are_rejected():
    ev = make_evidence()
    records = [r for r in ev.archive.leaves if r.source == "conditioning"]
    assert all(a is b for a, b in zip(permute_execution_records(records, 5), records))
    parent = aggregate(records[:2], "parent", children=True)
    with pytest.raises(ValueError, match="unit-mass"):
        permute_execution_records([parent], 5)


class Policy:
    def __init__(self):
        self.fail = False
        self.received = None

    def context(self, evidence, records):
        self.received = (evidence, records)
        if self.fail:
            raise RuntimeError("context failed")
        return torch.stack([record.content_sum for record in records]).sum()


class Online:
    method, intervention, decision_index, update_seconds = "FULL", "none", 0, 0.
    cfg = {"evaluation": {"seed": 1701}}

    def __init__(self):
        self.policy, self.evidence = Policy(), make_evidence()
        self.generator = torch.Generator().manual_seed(4)

    def decide(self):
        value = self.policy.context(self.evidence, self.evidence.archive.leaves)
        result = torch.randn((4, 8), generator=self.generator) + value
        self.decision_index += 1
        return result, {}

    def reset(self, reset, seed):
        self.generator.manual_seed(seed)
        return reset


def test_wrapper_restoration_inputs_and_generator_advancement():
    online = Online()
    clone = torch.Generator().manual_seed(4)
    torch.randn((4, 8), generator=clone)
    wrapped = PermutedExecutionOnlinePolicy(online)
    _, timing = wrapped.decide()
    assert "context" not in online.policy.__dict__
    assert online.policy.received[0] is online.evidence
    expected = intervene_evidence(online.evidence, "permute_execution", 1701)
    assert_records_equal(online.policy.received[1], expected.archive.leaves)
    assert torch.equal(online.generator.get_state(), clone.get_state())
    assert timing["execution_permutation"]["hierarchy_rebuilt"] is False
    assert wrapped.intervention == "permute_execution" and online.intervention == "none"
    wrapped.update_seconds = 4.
    assert online.update_seconds == 4.
    online.policy.fail = True
    with pytest.raises(RuntimeError, match="context failed"):
        wrapped.decide()
    assert "context" not in online.policy.__dict__


def test_wrong_reader_or_prior_intervention_rejected():
    online = Online()
    online.method = "TREE"
    with pytest.raises(ValueError, match="ordinary FULL"):
        PermutedExecutionOnlinePolicy(online)
