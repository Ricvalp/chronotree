"""Synthetic input-contract tests; random descriptors are not pretrained evidence."""

from copy import deepcopy

import pytest
import torch

from stmem.config import load_config, validate_config
from stmem.data.dataset import synthetic_episode
from stmem.memory.streaming import StreamingMemory
from stmem.models.batched_policy import pack_policy_batch
from stmem.models.policy import MemoryPolicy
from stmem.training.runtime import validate_policy_config
from stmem.vision.frozen import visual_feature_signature


def configuration(mode="none"):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["model"].update(
        width=16,
        heads=4,
        layers=1,
        text_layers=1,
        max_text_bytes=24,
        block_size=3,
        visual_tokens=6,
        current_tokens=3,
        horizon=4,
        proprioception_mode=mode,
        condition_on_previous_commands=False,
        visual_features={"mode": "clip_rn50_layer1", "projection_dim": 128},
        encoder_options={"neighbor_cap": 4, "center_chunk": 3},
    )
    cfg["training"]["encoder_block_batch_size"] = 2
    return cfg


def episode(cfg, seed=3):
    ep = synthetic_episode(length=7, conditioning_length=4, points=9, seed=seed)
    gen = torch.Generator().manual_seed(seed + 100)
    for frame in ep.conditioning + ep.execution:
        frame.visual_features = torch.randn(9, 256, generator=gen).half()
        frame.visual_feature_signature = visual_feature_signature(cfg)
    return ep


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(1)
    torch.manual_seed(900)


@pytest.mark.parametrize("mode", ["all", "current_only", "none"])
def test_serial_packed_streaming_and_descriptor_projection_gradients(mode):
    cfg = configuration(mode)
    policy = MemoryPolicy(cfg)
    ep = episode(cfg)
    d = ep.decision(4, 4)
    noisy, time = torch.randn(4, 8), torch.tensor(0.4)
    serial = policy(d, noisy, time)
    packed = pack_policy_batch([d], policy.encoder.block_size)
    rebuilt = packed.decisions()[0]
    torch.testing.assert_close(rebuilt.current.visual_features, d.current.visual_features)
    batch = policy(packed.metadata, noisy[None], time[None], packed_batch=packed)[0]
    torch.testing.assert_close(batch, serial, atol=2e-6, rtol=2e-5)
    batch.square().mean().backward()
    for param in policy.encoder.visual.feature_projection.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()
        assert param.grad.abs().sum() > 0
    stream = StreamingMemory(policy.encoder)
    stream.reset(ep.conditioning, ep.execution[0])
    for frame in ep.execution[1:5]:
        stream.observe(frame)
    evidence = policy.encode(d, archive=stream.archive)
    with torch.no_grad():
        online = policy(d, noisy, time, evidence=evidence)
    torch.testing.assert_close(online, serial, atol=2e-6, rtol=2e-5)


def test_frozen_features_cannot_change_sampling_and_no_prop_is_invariant():
    cfg = configuration()
    policy = MemoryPolicy(cfg).eval()
    ep = episode(cfg)
    d = ep.decision(4, 4)
    changed = deepcopy(d)
    for frame in [changed.current, *changed.conditioning, *changed.execution_prefix]:
        frame.proprio.fill_(float("nan"))
        frame.previous_action.fill_(float("nan"))
        frame.action_available = not frame.action_available
    noisy, time = torch.randn(4, 8), torch.tensor(0.4)
    with torch.no_grad():
        before, after = policy(d, noisy, time), policy(changed, noisy, time)
    torch.testing.assert_close(after, before, atol=0, rtol=0)
    visually_changed = deepcopy(d)
    visually_changed.current.visual_features = visually_changed.current.visual_features.flip(0)
    first = policy.encoder.current(d.current)
    second = policy.encoder.current(visually_changed.current)
    assert any(not torch.equal(a.content, b.content) for a, b in zip(first, second))
    for a, b in zip(first, second):
        for field in ("anchor", "bounds_min", "bounds_max"):
            torch.testing.assert_close(getattr(a, field), getattr(b, field), atol=0, rtol=0)


def test_missing_and_incompatible_features_fail_explicitly():
    cfg = configuration()
    policy = MemoryPolicy(cfg)
    frame = episode(cfg).execution[0]
    frame.visual_feature_signature = "wrong"
    with pytest.raises(ValueError, match="identity"):
        policy.encoder.current(frame)
    legacy = deepcopy(cfg)
    cfg["model"]["encoder_options"]["center_sampling"] = load_config("configs/debug/teacher.yaml")["model"][
        "encoder_options"
    ]["center_sampling"]
    legacy = deepcopy(cfg)
    del legacy["model"]["visual_features"]
    with pytest.raises(ValueError, match="visual_features"):
        validate_policy_config(legacy, cfg)
    different = deepcopy(cfg)
    different["model"]["visual_features"]["projection_dim"] = 64
    with pytest.raises(ValueError, match="projection_dim"):
        validate_policy_config(cfg, different)


def test_required_arms_share_initial_parameters_and_training_recipe():
    prop, noprop = [load_config(f"configs/teacher_fv_{name}.yaml") for name in ("prop", "no_prop")]
    for cfg in (prop, noprop):
        validate_config(cfg)
        assert (cfg["model"]["horizon"], cfg["model"]["execute"]) == (32, 4)
        assert cfg["training"]["checkpoint_steps"] == [25000, 50000, 100000]
        assert cfg["training"]["effective_batch"] == 128
        assert cfg["training"]["steps"] == 100000
    assert prop["training"] == noprop["training"]
    torch.manual_seed(0)
    first = MemoryPolicy(prop)
    torch.manual_seed(0)
    second = MemoryPolicy(noprop)
    assert first.state_dict().keys() == second.state_dict().keys()
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], atol=0, rtol=0)
