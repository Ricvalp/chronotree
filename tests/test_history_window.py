from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from stmem.inspection.history_window import RecentExecutionOnlinePolicy, select_recent_execution_records


def record(index, source="execution", pending=False):
    return SimpleNamespace(node_id=f"{source}:{index}", source=source, count=1,
                           depth=0, content_sum=torch.tensor([float(index)]),
                           time_min=float(index), time_max=float(index), time_anchor=float(index),
                           completed=not pending)


def evidence(block_count=3, pending=True):
    conditioning = [record(50, "conditioning"), record(51, "conditioning")]
    blocks = {"conditioning": conditioning}
    for block in range(block_count):
        blocks[f"execution{block}"] = [record(2 * block), record(2 * block + 1)]
    pend = [record(2 * block_count, pending=True)] if pending else []
    archive = SimpleNamespace(completed_blocks=blocks, pending={"execution": pend})
    leaves = [leaf for values in blocks.values() for leaf in values] + pend
    archive.leaves = leaves
    return SimpleNamespace(archive=archive, current=SimpleNamespace(source_index=100),
                           current_records=object(), text=object(), clocks={"execution": 100.}), leaves


def test_latest_complete_blocks_pending_and_conditioning_preserve_original_leaves():
    ev, leaves = evidence()
    selected, info = select_recent_execution_records(ev, leaves, 2)
    assert [r.node_id for r in selected] == [r.node_id for r in leaves[:2] + leaves[4:]]
    assert all(any(r is original for original in leaves) for r in selected)
    assert info["retained_completed_execution_block_ids"] == ["execution1", "execution2"]
    assert info["execution_recorded_time_bounds"] == [2., 6.]
    assert info["pending_execution_leaves"] == 1 and info["conditioning_tokens"] == 2
    assert ev.archive.leaves is leaves and len(leaves) == 9


@pytest.mark.parametrize("blocks,pending", [(0, False), (0, True), (1, True), (2, False)])
def test_no_change_before_window_is_exceeded(blocks, pending):
    ev, leaves = evidence(blocks, pending)
    selected, _ = select_recent_execution_records(ev, leaves, 2)
    assert len(selected) == len(leaves) and all(a is b for a, b in zip(selected, leaves))


class FakePolicy:
    device = torch.device("cpu")

    def __init__(self):
        self.called = None
        self.raise_error = False

    def context(self, ev, records):
        self.called = (ev, records)
        if self.raise_error:
            raise RuntimeError("context failure")
        return sum(float(r.content_sum[0]) for r in records)


class FakeOnline:
    method = "FULL"
    intervention = "none"
    decision_index = 0
    update_seconds = 0.

    def __init__(self, ev, leaves):
        self.policy = FakePolicy()
        self.ev, self.leaves = ev, leaves
        self.generator = torch.Generator().manual_seed(7)

    def decide(self):
        value = self.policy.context(self.ev, self.leaves)
        actions = torch.randn((2, 8), generator=self.generator) + value
        self.decision_index += 1
        return actions, {"history_tokens": len(self.leaves)}

    def reset(self, value, seed):
        self.generator.manual_seed(seed)
        self.decision_index = 0
        return value


def test_wrapper_preserves_context_arguments_restores_method_and_does_not_consume_noise():
    ev, leaves = evidence()
    online = FakeOnline(ev, leaves)
    expected_generator = torch.Generator().manual_seed(7)
    expected_noise = torch.randn((2, 8), generator=expected_generator)
    wrapped = RecentExecutionOnlinePolicy(online, completed_blocks=2)
    py, np_state, state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    actual, timing = wrapped.decide()
    selected, _ = select_recent_execution_records(ev, leaves, 2)
    assert torch.equal(actual, expected_noise + sum(float(r.content_sum[0]) for r in selected))
    assert torch.equal(online.generator.get_state(), expected_generator.get_state())
    assert "context" not in online.policy.__dict__
    assert online.policy.called[0] is ev
    assert timing["history_tokens"] == 7 and timing["original_archived_history_tokens"] == 9
    assert torch.equal(torch.get_rng_state(), state) and random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    wrapped.update_seconds = 4.
    assert online.update_seconds == 4.
    marker = object()
    assert wrapped.reset(marker, 7) is marker and wrapped.reads == []


def test_context_is_restored_when_underlying_decision_fails():
    ev, leaves = evidence()
    online = FakeOnline(ev, leaves)
    online.policy.raise_error = True
    wrapped = RecentExecutionOnlinePolicy(online)
    with pytest.raises(RuntimeError, match="context failure"):
        wrapped.decide()
    assert "context" not in online.policy.__dict__ and wrapped.reads == []


def test_existing_instance_context_override_is_restored():
    ev, leaves = evidence()
    online = FakeOnline(ev, leaves)
    def original(ev, records):
        return 1.
    online.policy.context = original
    RecentExecutionOnlinePolicy(online).decide()
    assert online.policy.context is original


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_invalid_window_rejected(value):
    ev, leaves = evidence()
    with pytest.raises(ValueError):
        RecentExecutionOnlinePolicy(FakeOnline(ev, leaves), completed_blocks=value)
