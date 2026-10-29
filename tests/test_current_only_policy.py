"""Synthetic invariants for the explicit current-scene/task-only diagnostic."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from stmem.config import load_config
from stmem.data.dataset import synthetic_episode
from stmem.models.batched_policy import batch_context, pack_policy_batch
from stmem.models.policy import MemoryPolicy


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(2930)


def config():
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
        history_mode="none",
        proprioception_mode="none",
        condition_on_previous_commands=False,
    )
    return cfg


def decisions():
    return [
        synthetic_episode(9, 4, 7, seed=2931).decision(6, 4),
        synthetic_episode(7, 1, 4, seed=2932).decision(2, 4),
        synthetic_episode(5, 0, 0, seed=2933).decision(0, 4),
    ]


def predict(policy, values, noisy, time, path):
    if path == "serial":
        return torch.stack([policy(d, noisy[i], time[i]) for i, d in enumerate(values)])
    if path == "packed":
        packed = pack_policy_batch(values, policy.encoder.block_size, model_config=policy.config["model"])
        return policy(list(packed.metadata), noisy, time, packed_batch=packed)
    if path == "cached":
        return policy(values, noisy, time, evidence=[policy.encode(d) for d in values])
    return policy(values, noisy, time)


@pytest.mark.parametrize("path", ["serial", "batched", "packed", "cached"])
def test_current_only_cannot_read_history_state_commands_or_progress(path):
    policy = MemoryPolicy(config()).eval()
    values = decisions()
    changed = deepcopy(values)
    for value in changed:
        # History is irrelevant even if it cannot be encoded or validated.
        value.conditioning = [object()]
        value.execution_prefix = [object(), object()]
        value.current.proprio.fill_(float("nan"))
        value.current.previous_action.fill_(float("nan"))
        value.current.action_available = not value.current.action_available
        value.current.time += 829
        value.current.source_index += 829
        value.current.event_id = "different bookkeeping"
        value.target_chunk.fill_(float("nan"))
        value.target_valid.fill_(False)
    noisy, time = torch.randn(3, 4, 8), torch.tensor([0.2, 0.4, 0.7])
    with (
        torch.no_grad(),
        patch(
            "stmem.models.batched_policy.encode_history_bank", side_effect=AssertionError("history encoded")
        ),
        patch.object(policy.encoder, "encode_frames", side_effect=AssertionError("history encoded")),
    ):
        before = predict(policy, values, noisy, time, path)
        after = predict(policy, changed, noisy, time, path)
    torch.testing.assert_close(before, after, atol=0, rtol=0)


def test_current_only_has_no_history_modules_or_raw_history_payload():
    policy = MemoryPolicy(config())
    assert not policy.history_enabled and not policy.clock_enabled and not policy.current_state_enabled
    assert policy.encoder.event is None and policy.proprio is None and policy.clock is None
    assert not policy.encoder.visual.raw_tau_t.requires_grad
    packed = pack_policy_batch(decisions(), policy.encoder.block_size, model_config=policy.config["model"])
    assert packed.history.points.numel() == 0
    assert packed.history.visual_features == []
    assert packed.history_lengths == [0, 0, 0]
    assert packed.history.blocks_per_example == [0, 0, 0]
    assert torch.count_nonzero(packed.clocks) == 0
    assert all(not value.conditioning and not value.execution_prefix for value in packed.decisions())
    for decision in decisions():
        evidence = policy.encode(decision)
        assert not evidence.archive.leaves
        assert evidence.clocks == {"conditioning": 0.0, "execution": 0.0}
        context, bias = policy.context(evidence, [])
        assert len(context) == len(evidence.current_records) + 1
        assert len(context) == len(bias)
    with pytest.raises(RuntimeError, match="History encoding is disabled"):
        policy.encoder.encode_frames(decisions()[0].conditioning)


def test_current_only_serial_batched_packed_parity_and_gradients():
    policy = MemoryPolicy(config()).eval()
    values = decisions()
    noisy, time = torch.randn(3, 4, 8), torch.tensor([0.2, 0.4, 0.7])
    with torch.no_grad():
        serial = predict(policy, values, noisy, time, "serial")
        for path in ("batched", "packed", "cached"):
            torch.testing.assert_close(
                predict(policy, values, noisy, time, path), serial, atol=2e-6, rtol=3e-5
            )
    policy.train()
    packed = pack_policy_batch(values, policy.encoder.block_size, model_config=policy.config["model"])
    output, stats = policy(list(packed.metadata), noisy, time, packed_batch=packed, return_stats=True)
    assert stats["history_leaves"].tolist() == [0, 0, 0]
    assert stats["history_context_tokens"].tolist() == [0, 0, 0]
    output.square().mean().backward()
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_current_only_is_sensitive_to_current_scene_and_goal():
    policy = MemoryPolicy(config()).eval()
    values = decisions()[:1]
    noisy, time = torch.randn(1, 4, 8), torch.tensor([0.4])
    scene = deepcopy(values)
    scene[0].current.points[:, :3] += torch.tensor([0.2, -0.1, 0.1])
    goal = deepcopy(values)
    goal[0].goal = "a different visible task"
    with torch.no_grad():
        original = predict(policy, values, noisy, time, "serial")
        assert (predict(policy, scene, noisy, time, "serial") - original).abs().max() > 1e-6
        assert (predict(policy, goal, noisy, time, "serial") - original).abs().max() > 1e-6


def test_current_only_empty_cloud_uses_finite_goal_only_context():
    policy = MemoryPolicy(config()).eval()
    value = decisions()[-1]
    context, bias, stats = batch_context(policy, [value])
    assert torch.isfinite(context).all()
    assert int(torch.isfinite(bias).sum()) == 1
    assert stats["current_visual_tokens"].item() == 0
    assert stats["history_context_tokens"].item() == 0


def test_current_only_frozen_descriptors_match_dense_and_never_read_history(tmp_path):
    from stmem.data.feature_cache import FeatureRow
    from stmem.vision.frozen import visual_feature_signature

    cfg = config()
    cfg["model"]["visual_features"] = {"mode": "clip_rn50_layer1", "projection_dim": 128}
    signature = visual_feature_signature(cfg)
    values = decisions()[:2]
    for value in values:
        for frame in [*value.conditioning, *value.execution_prefix]:
            frame.visual_feature_signature = signature
            frame.visual_features = FeatureRow(
                str(tmp_path / "history-must-not-be-read.bin"),
                "synthetic",
                0,
                len(frame.points),
                signature,
                0,
            )
        value.current.visual_feature_signature = signature
        value.current.visual_features = torch.randn(len(value.current.points), 256).half()
    cached = deepcopy(values)
    path = tmp_path / "current-descriptors.bin"
    offset = 0
    with path.open("wb") as stream:
        for index, value in enumerate(cached):
            features = value.current.visual_features
            stream.write(features.numpy().tobytes())
            value.current.visual_features = FeatureRow(
                str(path),
                "synthetic",
                index,
                len(value.current.points),
                signature,
                offset,
            )
            offset += features.numel() * features.element_size()
    policy = MemoryPolicy(cfg).eval()
    noisy, time = torch.randn(2, 4, 8), torch.tensor([0.3, 0.7])
    with torch.no_grad():
        dense = predict(policy, values, noisy, time, "serial")
        for path in ("serial", "batched", "packed", "cached"):
            torch.testing.assert_close(
                predict(policy, cached, noisy, time, path), dense, atol=2e-6, rtol=3e-5
            )
    predict(policy, cached, noisy, time, "packed").square().mean().backward()
    for parameter in policy.encoder.visual.feature_projection.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_current_only_structure_inspection_has_no_history_or_event_modules():
    from stmem.visualization.encoder_structure import build_structure

    policy = MemoryPolicy(config()).eval()
    episode = synthetic_episode(9, 4, 7, seed=2931)
    with patch.object(policy.encoder, "encode_frames", side_effect=AssertionError("history encoded")):
        structure = build_structure(episode, policy.encoder, 6)
    assert len(structure["blocks"]) == 1
    assert structure["archive"] == {"roots": [], "leaf_ids": [], "nodes": []}
    assert not structure["config"]["history_enabled"]
    assert not structure["config"]["condition_on_previous_commands"]
    assert structure["episode"]["conditioning_frames"] == 0
    assert structure["episode"]["execution_prefix_frames"] == 0
    assert structure["current"]["raw_state_diagnostic_only"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("history_mode", "window"),
        ("proprioception_mode", "current_only"),
        ("condition_on_previous_commands", True),
    ],
)
def test_current_only_contract_rejects_conflicting_inputs(field, value):
    cfg = config()
    cfg["model"][field] = value
    with pytest.raises(ValueError):
        MemoryPolicy(cfg)
