"""Synthetic diagnostics for the selective previous-joint-command ablation."""

from copy import deepcopy

import pytest
import torch

from stmem.config import load_config, validate_config
from stmem.data.cache import cache_identity
from stmem.data.dataset import synthetic_episode
from stmem.models.batched_policy import pack_policy_batch
from stmem.models.event_encoder import EventEncoder
from stmem.models.policy import MemoryPolicy
from stmem.training.runtime import load_policy, validate_policy_config


FLAG = "condition_on_previous_joint_commands"


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(1827)


def _config(enabled=False):
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
    )
    if enabled is None:
        cfg["model"].pop(FLAG, None)
    else:
        cfg["model"][FLAG] = enabled
    cfg["training"]["encoder_block_batch_size"] = 2
    return cfg


def _decisions():
    return [
        synthetic_episode(9, 4, 7, seed=1828).decision(6, 4),
        synthetic_episode(7, 1, 4, seed=1829).decision(2, 4),
    ]


def _change(decisions, channel, scope):
    changed = deepcopy(decisions)
    for decision in changed:
        frames = [decision.current] if scope == "current" else decision.execution_prefix
        for frame in frames:
            if channel == "state":
                # Measured joints must remain available even when commanded joints are removed.
                frame.proprio[:7].add_(0.8)
            elif frame.action_available:
                if channel == "joints":
                    frame.previous_action[:7].add_(torch.linspace(0.5, 1.1, 7))
                else:
                    frame.previous_action[7].add_(1.0)
    return changed


def _predict(model, decisions, noisy, times, path):
    if path == "serial":
        return torch.stack([model(d, noisy[i], times[i]) for i, d in enumerate(decisions)])
    if path == "packed":
        packed = pack_policy_batch(decisions, model.config["model"]["block_size"])
        return model(list(packed.metadata), noisy, times, packed_batch=packed)
    if path == "cached":
        evidences = [model.encode(d) for d in decisions]
        return model(
            decisions,
            noisy,
            times,
            evidence=evidences,
            records=[e.archive.leaves for e in evidences],
        )
    return model(decisions, noisy, times)


@pytest.mark.parametrize("path", ["serial", "batched", "packed", "cached"])
@pytest.mark.parametrize("scope", ["current", "history"])
def test_joint_ablation_is_exact_in_all_policy_paths(path, scope):
    decisions = _decisions()
    changed = _change(decisions, "joints", scope)
    noisy, times = torch.randn(2, 4, 8), torch.tensor([0.3, 0.7])
    model = MemoryPolicy(_config()).eval()
    with torch.no_grad():
        before = _predict(model, decisions, noisy, times, path)
        after = _predict(model, changed, noisy, times, path)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    for original, modified in zip(decisions, changed):
        torch.testing.assert_close(original.target_chunk, modified.target_chunk, atol=0, rtol=0)
        torch.testing.assert_close(original.target_valid, modified.target_valid, atol=0, rtol=0)


@pytest.mark.parametrize("enabled", [True, None])
@pytest.mark.parametrize("scope", ["current", "history"])
def test_default_and_explicit_baseline_still_use_joint_commands(enabled, scope):
    decisions = _decisions()
    changed = _change(decisions, "joints", scope)
    noisy, times = torch.randn(2, 4, 8), torch.tensor([0.3, 0.7])
    model = MemoryPolicy(_config(enabled)).eval()
    with torch.no_grad():
        before = model(decisions, noisy, times)
        after = model(changed, noisy, times)
    assert (before - after).abs().max() > 1e-6


@pytest.mark.parametrize("channel", ["gripper", "state"])
@pytest.mark.parametrize("scope", ["current", "history"])
def test_ablation_preserves_gripper_commands_and_measured_state(channel, scope):
    decisions = _decisions()
    changed = _change(decisions, channel, scope)
    noisy, times = torch.randn(2, 4, 8), torch.tensor([0.3, 0.7])
    model = MemoryPolicy(_config()).eval()
    with torch.no_grad():
        before = _predict(model, decisions, noisy, times, "packed")
        after = _predict(model, changed, noisy, times, "packed")
    assert (before - after).abs().max() > 1e-6


def test_event_encoder_masks_unavailable_fields_and_only_disabled_joint_gradients():
    encoder = EventEncoder(width=16, condition_on_previous_joint_commands=False)
    state = torch.randn(2, 15, requires_grad=True)
    action = torch.randn(2, 8, requires_grad=True)
    available = torch.tensor([True, False])
    before = encoder(state, action, available)
    changed = action.detach().clone()
    changed[1].fill_(float("nan"))
    torch.testing.assert_close(before, encoder(state, changed, available), atol=0, rtol=0)
    before.square().sum().backward()
    assert torch.count_nonzero(action.grad[:, :7]) == 0
    assert action.grad[0, 7].abs() > 0
    assert torch.count_nonzero(action.grad[1]) == 0
    assert torch.count_nonzero(state.grad[:, :7]) > 0


def test_current_state_encoder_keeps_availability_and_measured_joint_gradients():
    model = MemoryPolicy(_config())
    state = torch.randn(2, 24)
    state[:, 23] = 1
    state.requires_grad_()
    model.encode_current_state(state).square().sum().backward()
    assert torch.count_nonzero(state.grad[:, 15:22]) == 0
    assert torch.count_nonzero(state.grad[:, 22]) > 0
    assert torch.count_nonzero(state.grad[:, :7]) > 0
    before = model.encode_current_state(state.detach())
    unavailable = state.detach().clone()
    unavailable[:, 23] = 0
    assert not torch.equal(before, model.encode_current_state(unavailable))


def test_routing_query_cannot_reintroduce_disabled_commands():
    model = MemoryPolicy(_config()).eval()
    decision = _decisions()[0]
    with torch.no_grad():
        evidence = model.encode(decision)
        before = model.routing_query(evidence, 12)
        evidence.current = _change([decision], "joints", "current")[0].current
        after = model.routing_query(evidence, 12)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        evidence.current = _change([decision], "gripper", "current")[0].current
        assert not torch.equal(before, model.routing_query(evidence, 12))
        evidence.current = _change([decision], "state", "current")[0].current
        assert not torch.equal(before, model.routing_query(evidence, 12))


def test_checkpoint_loading_preserves_ablation_for_inference(tmp_path):
    model = MemoryPolicy(_config()).eval()
    checkpoint = tmp_path / "synthetic_teacher.pt"
    torch.save({"stage": "teacher", "config": model.config, "model": model.state_dict()}, checkpoint)
    restored, state = load_policy(checkpoint, device="cpu")
    restored.eval()
    assert state["config"]["model"][FLAG] is False
    decision = _decisions()[0]
    changed = _change([decision], "joints", "history")[0]
    changed = _change([changed], "joints", "current")[0]
    noise = torch.randn(4, 8)
    with torch.no_grad():
        evidence = model.encode(decision)
        expected = model.predict(evidence, evidence.archive.leaves, noise=noise)
        evidence = restored.encode(changed)
        actual = restored.predict(evidence, evidence.archive.leaves, noise=noise)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_inspector_encoder_honors_the_same_command_ablation():
    from stmem.visualization.inspect_encoder import _encoder

    cfg = _config()
    policy = MemoryPolicy(cfg).eval()
    inspector = _encoder(cfg)
    inspector.load_state_dict(policy.encoder.state_dict())
    original = _decisions()[0]
    changed = _change([original], "joints", "history")[0]
    with torch.no_grad():
        expected = policy.encoder.encode_frames(original.execution_prefix[:3])
        actual = inspector.encode_frames(changed.execution_prefix[:3])
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual):
        assert left.node_id == right.node_id
        torch.testing.assert_close(left.content, right.content, atol=0, rtol=0)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_command_conditioning_flag_requires_boolean(value):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"][FLAG] = value
    with pytest.raises(ValueError, match=FLAG):
        validate_config(cfg)


def test_checkpoint_contract_normalizes_omitted_default_and_rejects_changed_conditioning():
    baseline = load_config("configs/debug/real_teacher.yaml")
    baseline["model"][FLAG] = True
    old = deepcopy(baseline)
    old["model"].pop(FLAG)
    validate_policy_config(old, baseline)
    validate_policy_config(baseline, old)
    ablated = deepcopy(baseline)
    ablated["model"][FLAG] = False
    for recorded, requested in [(baseline, ablated), (ablated, baseline), (old, ablated)]:
        with pytest.raises(ValueError, match=FLAG):
            validate_policy_config(recorded, requested)


def test_frozen_cache_identity_records_command_conditioning(tmp_path, monkeypatch):
    monkeypatch.setattr("stmem.data.cache.source_revision", lambda: {"source_sha256": "diagnostic"})
    teacher = tmp_path / "checkpoint.pt"
    teacher.write_bytes(b"synthetic cache identity test; not a checkpoint")
    baseline = load_config("configs/debug/real_teacher.yaml")
    baseline["model"][FLAG] = True
    ablated = deepcopy(baseline)
    ablated["model"][FLAG] = False
    before = cache_identity(baseline, teacher, "synthetic")
    after = cache_identity(ablated, teacher, "synthetic")
    assert before["model_encoder"][FLAG] is True
    assert after["model_encoder"][FLAG] is False
    assert before != after


def test_from_scratch_recipe_matches_low_lr_baseline_except_command_conditioning():
    baseline = load_config("configs/teacher_b200_b128_m8_blocks32.yaml")
    ablated = load_config("configs/teacher_b200_b128_m8_blocks32_no_previous_joint_commands.yaml")
    validate_config(ablated)
    assert ablated["model"][FLAG] is False
    assert ablated["training"]["steps"] == 100000
    assert ablated["training"]["lr"] == 0.0001
    assert ablated["training"]["effective_batch"] == 128
    assert ablated["training"]["microbatch"] == 8
    assert ablated["training"]["encoder_block_batch_size"] == 32
    ablated["model"][FLAG] = True
    assert ablated == baseline
