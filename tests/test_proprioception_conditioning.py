"""Synthetic diagnostics for complete state/command removal, not performance evidence."""

from copy import deepcopy

import pytest
import torch

from stmem.config import load_config, validate_config
from stmem.data.cache import cache_identity
from stmem.data.dataset import synthetic_episode
from stmem.memory.streaming import StreamingMemory
from stmem.models.batched_policy import batch_context, pack_policy_batch
from stmem.models.event_encoder import EventEncoder
from stmem.models.policy import MemoryPolicy, choose_records
from stmem.models.router import LocalRouter
from stmem.training.attention import context_groups
from stmem.training.runtime import load_policy, validate_policy_config


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(2830)


def config(mode="none", commands=False):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"].update(
        width=16,
        heads=4,
        layers=1,
        text_layers=1,
        max_text_bytes=24,
        block_size=3,
        current_tokens=5,
        visual_tokens=7,
        horizon=4,
        encoder_options={"neighbor_cap": 4, "center_chunk": 3},
        proprioception_mode=mode,
        condition_on_previous_commands=commands,
    )
    cfg["training"]["encoder_block_batch_size"] = 2
    return cfg


def decisions():
    return [
        synthetic_episode(9, 4, 7, seed=2831).decision(6, 4),
        synthetic_episode(7, 1, 4, seed=2832).decision(2, 4),
        synthetic_episode(5, 0, 0, seed=2833).decision(0, 4),
    ]


def perturb(values, *, state=True, commands=True, scope="both"):
    """Mutate raw state before encoding, including the end-effector anchor fields."""
    changed = deepcopy(values)
    for decision in changed:
        frames = []
        if scope in ("both", "current"):
            frames.append(decision.current)
        if scope in ("both", "history"):
            frames.extend([*decision.conditioning, *decision.execution_prefix])
        for frame in frames:
            if state:
                frame.proprio.add_(torch.linspace(-4, 4, 15))
            if commands:
                # Includes all seven joints, gripper and the availability flag.
                frame.previous_action.add_(torch.linspace(10, 20, 8))
                frame.action_available = not frame.action_available
    return changed


def predict(policy, values, noisy, time, path):
    if path == "serial":
        return torch.stack([policy(d, noisy[i], time[i]) for i, d in enumerate(values)])
    if path == "packed":
        packed = pack_policy_batch(values, policy.encoder.block_size)
        return policy(list(packed.metadata), noisy, time, packed_batch=packed)
    if path == "cached":
        evidence = [policy.encode(d) for d in values]
        return policy(values, noisy, time, evidence=evidence)
    return policy(values, noisy, time)


@pytest.mark.parametrize("path", ["serial", "batched", "packed", "cached"])
def test_no_proprio_and_no_commands_are_exactly_invariant_in_all_paths(path):
    policy = MemoryPolicy(config()).eval()
    original = decisions()
    changed = perturb(original)
    noisy, time = torch.randn(3, 4, 8), torch.tensor([0.2, 0.4, 0.7])
    with torch.no_grad():
        before = predict(policy, original, noisy, time, path)
        after = predict(policy, changed, noisy, time, path)
    torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["all", "current_only", "none"])
def test_complete_commands_removed_in_all_proprioception_modes(mode):
    policy = MemoryPolicy(config(mode)).eval()
    original = decisions()
    changed = perturb(original, state=False)
    noisy, time = torch.randn(3, 4, 8), torch.tensor([0.2, 0.4, 0.7])
    with torch.no_grad():
        before = predict(policy, original, noisy, time, "packed")
        after = predict(policy, changed, noisy, time, "packed")
    torch.testing.assert_close(before, after, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["all", "current_only", "none"])
def test_proprioception_mode_respects_current_and_history_boundaries(mode):
    policy = MemoryPolicy(config(mode)).eval()
    original = decisions()
    noisy, time = torch.randn(3, 4, 8), torch.tensor([0.2, 0.4, 0.7])
    with torch.no_grad():
        before = predict(policy, original, noisy, time, "serial")
        current = predict(policy, perturb(original, commands=False, scope="current"), noisy, time, "serial")
        history = predict(policy, perturb(original, commands=False, scope="history"), noisy, time, "serial")
    if mode == "none":
        torch.testing.assert_close(before, current, atol=0, rtol=0)
    else:
        assert (before - current).abs().max() > 1e-6
    if mode != "all":
        torch.testing.assert_close(before, history, atol=0, rtol=0)
    else:
        assert (before - history).abs().max() > 1e-6


@pytest.mark.parametrize("mode", ["all", "current_only", "none"])
@pytest.mark.parametrize("commands", [False, True])
def test_serial_packed_streaming_parity_and_token_counts(mode, commands):
    policy = MemoryPolicy(config(mode, commands)).eval()
    values = decisions()
    noisy, time = torch.randn(3, 4, 8), torch.tensor([0.2, 0.4, 0.7])
    with torch.no_grad():
        serial = predict(policy, values, noisy, time, "serial")
        packed = pack_policy_batch(values, policy.encoder.block_size)
        actual, stats = policy(list(packed.metadata), noisy, time, packed_batch=packed, return_stats=True)
        torch.testing.assert_close(serial, actual, atol=2e-6, rtol=3e-5)
        for index, decision in enumerate(values):
            offline = policy.encode(decision)
            memory = StreamingMemory(policy.encoder)
            if decision.execution_prefix:
                memory.reset(decision.conditioning, decision.execution_prefix[0])
                for frame in [*decision.execution_prefix[1:], decision.current]:
                    memory.observe(frame)
            else:
                memory.reset(decision.conditioning, decision.current)
            streaming = policy.encode(decision, archive=memory.archive)
            assert [r.node_id for r in offline.archive.leaves] == [
                r.node_id for r in streaming.archive.leaves
            ]
            assert len(offline.archive.leaves) == stats["history_leaves"][index]
            prediction = policy(decision, noisy[index], time[index], evidence=streaming)
            torch.testing.assert_close(prediction, serial[index], atol=0, rtol=0)
            events = [r for r in offline.archive.leaves if r.modality == "event"]
            expected = len(decision.conditioning) + len(decision.execution_prefix)
            assert len(events) == (expected if mode == "all" or commands else 0)
            if mode != "all":
                assert all(torch.count_nonzero(r.anchor) == 0 for r in events)
            context, bias = policy.context(offline, offline.archive.leaves)
            groups = context_groups(
                offline,
                offline.archive.leaves,
                policy.device,
                current_state_enabled=policy.current_state_enabled,
            )
            assert len(context) == len(bias) == len(groups)
            assert len(context) == len(offline.current_records) + len(offline.archive.leaves) + 2 + int(
                policy.current_state_enabled
            )


@pytest.mark.parametrize("mode", ["none", "current_only"])
def test_historical_state_cannot_leak_through_command_only_event_anchors(mode):
    policy = MemoryPolicy(config(mode, commands=True)).eval()
    original = decisions()
    changed = perturb(original, commands=False, scope="history")
    noisy, time = torch.randn(3, 4, 8), torch.tensor([0.2, 0.4, 0.7])
    with torch.no_grad():
        before = predict(policy, original, noisy, time, "serial")
        after = predict(policy, changed, noisy, time, "serial")
        batched = predict(policy, changed, noisy, time, "packed")
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    torch.testing.assert_close(before, batched, atol=2e-6, rtol=3e-5)


@pytest.mark.parametrize("method", ["TREE", "FLAT"])
def test_no_state_rebuilt_archive_router_and_predictions_are_invariant(method):
    policy = MemoryPolicy(config()).eval()
    router = LocalRouter(width=16).eval()
    original = decisions()[0]
    changed = perturb([original])[0]
    noise = torch.randn(4, 8)
    with torch.no_grad():
        predictions, archives, queries, selections = [], [], [], []
        for value in (original, changed):
            evidence = policy.encode(value)
            query = policy.routing_query(evidence, 12)
            def scorer(records):
                return router(query, router.candidate_input(records)).tolist()

            selection = choose_records(evidence, method, 12, scorer=scorer)
            predictions.append(policy.predict(evidence, selection.records, noise=noise))
            archives.append(router.candidate_input(evidence.archive.records))
            queries.append(query)
            selections.append([r.node_id for r in selection.records])
    torch.testing.assert_close(queries[0], queries[1], atol=0, rtol=0)
    torch.testing.assert_close(archives[0], archives[1], atol=0, rtol=0)
    torch.testing.assert_close(predictions[0], predictions[1], atol=0, rtol=0)
    assert selections[0] == selections[1]


def test_shared_modules_initialize_identically_and_unused_modules_are_frozen():
    torch.manual_seed(102)
    control = MemoryPolicy(config("all"))
    torch.manual_seed(102)
    ablated = MemoryPolicy(config("none"))
    assert control.state_dict().keys() == ablated.state_dict().keys()
    for key in control.state_dict():
        torch.testing.assert_close(control.state_dict()[key], ablated.state_dict()[key], atol=0, rtol=0)
    assert not any(p.requires_grad for p in ablated.proprio.parameters())
    assert not any(p.requires_grad for p in ablated.encoder.event.parameters())
    assert all(p.requires_grad for p in control.proprio.parameters())
    assert all(p.requires_grad for p in control.encoder.event.state.parameters())
    assert not any(p.requires_grad for p in control.encoder.event.action.parameters())


def test_no_proprio_backward_and_zero_history_reset_are_valid():
    policy = MemoryPolicy(config())
    values = decisions()
    noisy, time = torch.randn(3, 4, 8), torch.tensor([0.2, 0.4, 0.7])
    output = policy(values, noisy, time)
    output.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.encoder.visual.parameters())
    assert all(p.grad is None for p in policy.proprio.parameters())
    assert all(p.grad is None for p in policy.encoder.event.parameters())
    with torch.no_grad():
        context, bias, stats = batch_context(policy, [values[-1]])
    assert int(torch.isfinite(bias).sum()) == 2  # Instruction and clock; visual slots are masked padding.
    assert torch.count_nonzero(context[~torch.isfinite(bias)]) == 0
    assert stats["history_leaves"].tolist() == [0]


def test_disabled_event_channels_ignore_nonfinite_raw_values():
    encoder = EventEncoder(width=16, condition_on_proprioception=False)
    state, action = torch.randn(2, 15), torch.randn(2, 8)
    available = torch.tensor([True, False])
    before = encoder(state, action, available)
    state.fill_(float("nan"))
    torch.testing.assert_close(before, encoder(state, action, available), atol=0, rtol=0)
    encoder = EventEncoder(width=16, condition_on_previous_commands=False)
    state = torch.randn(2, 15)
    before = encoder(state, action, available)
    action.fill_(float("nan"))
    torch.testing.assert_close(before, encoder(state, action, ~available), atol=0, rtol=0)


def test_checkpoint_restores_no_proprio_and_command_removal(tmp_path):
    policy = MemoryPolicy(config()).eval()
    path = tmp_path / "synthetic.pt"
    torch.save({"stage": "teacher", "config": policy.config, "model": policy.state_dict()}, path)
    restored, _ = load_policy(path, device="cpu")
    restored.eval()
    original = decisions()[0]
    changed = perturb([original])[0]
    noisy = torch.randn(4, 8)
    with torch.no_grad():
        before = policy(original, noisy, torch.tensor(0.3))
        after = restored(changed, noisy, torch.tensor(0.3))
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_omitted_legacy_options_preserve_weights_and_predictions():
    legacy = config("all", True)
    legacy["model"].pop("proprioception_mode")
    legacy["model"].pop("condition_on_previous_commands")
    torch.manual_seed(301)
    implicit = MemoryPolicy(legacy).eval()
    torch.manual_seed(301)
    explicit = MemoryPolicy(config("all", True)).eval()
    explicit.load_state_dict(implicit.state_dict(), strict=True)
    values = decisions()
    noisy, time = torch.randn(3, 4, 8), torch.tensor([0.2, 0.4, 0.7])
    with torch.no_grad():
        before = implicit(values, noisy, time)
        after = explicit(values, noisy, time)
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_no_proprio_cannot_read_future_frames_or_target_commands():
    policy = MemoryPolicy(config()).eval()
    episode = synthetic_episode(9, 4, 7, seed=2834)
    before = episode.decision(3, 4)
    changed = deepcopy(episode)
    for frame in changed.execution[4:]:
        frame.proprio.add_(123)
        frame.previous_action.add_(111)
        frame.points.add_(100)
    changed.actions[4:] = -10
    after = changed.decision(3, 4)
    noisy, time = torch.randn(1, 4, 8), torch.tensor([0.3])
    with torch.no_grad():
        original = policy([before], noisy, time)
        perturbed = policy([after], noisy, time)
    torch.testing.assert_close(original, perturbed, atol=0, rtol=0)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_complete_command_flag_requires_boolean(value):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"]["condition_on_previous_commands"] = value
    with pytest.raises(ValueError, match="condition_on_previous_commands"):
        validate_config(cfg)


@pytest.mark.parametrize("value", ["current", "ALL", False, None, 0])
def test_proprioception_mode_validation(value):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"]["proprioception_mode"] = value
    with pytest.raises(ValueError, match="proprioception_mode"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "key,default,changed",
    [
        ("condition_on_previous_commands", True, False),
        ("proprioception_mode", "all", "none"),
        ("proprioception_mode", "all", "current_only"),
    ],
)
def test_checkpoint_contract_normalizes_legacy_and_rejects_changed_state_contract(key, default, changed):
    baseline = load_config("configs/debug/real_teacher.yaml")
    baseline["model"][key] = default
    legacy = deepcopy(baseline)
    legacy["model"].pop(key)
    validate_policy_config(legacy, baseline)
    validate_policy_config(baseline, legacy)
    altered = deepcopy(baseline)
    altered["model"][key] = changed
    for recorded, requested in ((baseline, altered), (altered, legacy)):
        with pytest.raises(ValueError, match=key):
            validate_policy_config(recorded, requested)


@pytest.mark.parametrize("key,bad", [
    ("condition_on_previous_commands", 1), ("condition_on_previous_commands", "true"),
    ("proprioception_mode", "current"), ("proprioception_mode", None),
])
def test_checkpoint_validator_rejects_invalid_new_input_flags_even_when_both_equal(key, bad):
    cfg = load_config("configs/debug/real_teacher.yaml")
    cfg["model"][key] = bad
    with pytest.raises(ValueError, match=key):
        validate_policy_config(cfg, deepcopy(cfg))


@pytest.mark.parametrize(
    "key,default,changed",
    [
        ("condition_on_previous_commands", True, False),
        ("proprioception_mode", "all", "none"),
    ],
)
def test_frozen_leaf_cache_identity_includes_state_and_command_contract(
    tmp_path, monkeypatch, key, default, changed
):
    monkeypatch.setattr("stmem.data.cache.source_revision", lambda: {"source_sha256": "diagnostic"})
    teacher = tmp_path / "checkpoint.pt"
    teacher.write_bytes(b"synthetic cache identity test; not a checkpoint")
    baseline = load_config("configs/debug/real_teacher.yaml")
    baseline["model"][key] = default
    ablated = deepcopy(baseline)
    ablated["model"][key] = changed
    before = cache_identity(baseline, teacher, "synthetic")
    after = cache_identity(ablated, teacher, "synthetic")
    assert before["model_encoder"][key] == default
    assert after["model_encoder"][key] == changed
    assert before != after
