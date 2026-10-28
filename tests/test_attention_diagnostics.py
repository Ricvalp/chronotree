"""Synthetic diagnostics for bounded attention summaries, not policy quality."""

from copy import deepcopy
import math

import pytest
import torch

from stmem.models.action_flow import ActionFlow
from stmem.models.attention_diagnostics import (
    GROUP_NAMES,
    CrossAttentionDiagnostics,
    cross_attention_statistics,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(107)


def _dense(q, k, bias, groups, names):
    allowed = torch.isfinite(bias)
    safe_k = k.masked_fill(~allowed[:, None, :, None], 0)
    logits = q @ safe_k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    logits = (logits + bias[:, None, None]).masked_fill(~allowed[:, None, None], -torch.inf)
    weights = torch.softmax(logits, -1)
    result = {}
    log_weights = torch.where(weights > 0, weights.log(), 0)
    entropy = -(weights * log_weights).sum(-1) / allowed.sum(-1).log()[:, None, None]
    result["entropy/all"] = entropy.mean()
    selection = {name: [i] for i, name in enumerate(names)}
    selection.update(current=[0, 1], history=[4, 5])
    for name, ids in selection.items():
        keep = allowed & torch.stack([groups == index for index in ids]).any(0)
        selected = weights.masked_fill(~keep[:, None, None], 0)
        mass = selected.sum(-1)
        count = keep.sum(-1)
        baseline = count / allowed.sum(-1)
        relative = selected / mass[..., None]
        log_relative = torch.where(relative > 0, relative.log(), 0)
        entropy = -(relative * log_relative).sum(-1)
        normalized = entropy / count.clamp_min(2).log()[:, None, None]
        normalized = normalized.masked_fill((count <= 1)[:, None, None], 0)
        result[f"mass/{name}"] = mass.mean()
        result[f"uniform_mass/{name}"] = baseline.mean()
        valid = count > 0
        if valid.any():
            result[f"relative_mass/{name}"] = (mass[valid] / baseline[valid, None, None]).mean()
            result[f"entropy/{name}"] = normalized[valid].mean()
    return result


@pytest.mark.parametrize("chunk_size", [1, 3, 2048])
def test_streaming_groups_entropy_and_mass_match_dense_attention(chunk_size):
    q, k = torch.randn(2, 3, 4, 5), torch.randn(2, 3, 19, 5)
    groups = torch.tensor([[0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 0, -1, -1]] * 2)
    bias = torch.randn(2, 19) * 0.2
    bias[:, -2:] = -torch.inf
    k[:, :, -2:] = float("nan")
    actual = cross_attention_statistics(q, k, bias, groups, key_chunk_size=chunk_size)
    expected = _dense(q, k, bias, groups, GROUP_NAMES)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=3e-5, atol=3e-6, msg=key)
    assert sum(actual[f"mass/{name}"] for name in GROUP_NAMES) == pytest.approx(1)


def test_entropy_is_computed_per_head_before_averaging():
    # Two sharply focused heads attend to opposite keys. Entropy of their
    # average distribution would be one, but each head's entropy is near zero.
    q = torch.ones(1, 2, 1, 1)
    k = torch.tensor([[[[30.0], [-30.0]], [[-30.0], [30.0]]]])
    stats = cross_attention_statistics(q, k, None, torch.tensor([0, 0]), ("visual",), key_chunk_size=1)
    assert stats["mass/visual"] == pytest.approx(1)
    assert stats["entropy/visual"] < 1e-20
    assert stats["entropy/all"] < 1e-20


def test_uniform_attention_and_singleton_entropy_have_expected_baselines():
    q, k = torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 6, 4)
    groups = torch.tensor([0, 0, 0, 1, 2, 2])
    stats = cross_attention_statistics(q, k, None, groups, ("three", "one", "two"), key_chunk_size=2)
    for name, count in [("three", 3), ("one", 1), ("two", 2)]:
        assert stats[f"mass/{name}"] == pytest.approx(count / 6)
        assert stats[f"uniform_mass/{name}"] == pytest.approx(count / 6)
        assert stats[f"relative_mass/{name}"] == pytest.approx(1)
        assert stats[f"entropy/{name}"] == pytest.approx(0 if count == 1 else 1)


def test_log_mass_bias_is_included_and_empty_groups_are_explicit():
    q, k = torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 3, 2)
    bias = torch.tensor([[0.0, math.log(3), -torch.inf]])
    stats = cross_attention_statistics(q, k, bias, torch.tensor([0, 1, -1]), ("a", "b", "empty"))
    assert stats["mass/a"] == pytest.approx(0.25)
    assert stats["mass/b"] == pytest.approx(0.75)
    assert stats["uniform_mass/b"] == pytest.approx(0.5)
    assert stats["relative_mass/b"] == pytest.approx(1.5)
    assert stats["mass/empty"] == stats["uniform_mass/empty"] == 0
    assert "entropy/empty" not in stats and "relative_mass/empty" not in stats


@pytest.mark.parametrize("key_count", [0, 5])
def test_all_padding_is_finite_and_not_fabricated_attention(key_count):
    q = torch.randn(2, 2, 3, 4)
    k = torch.full((2, 2, key_count, 4), float("nan"))
    bias = torch.full((2, key_count), -torch.inf)
    groups = torch.full((2, key_count), -1, dtype=torch.long)
    stats = cross_attention_statistics(q, k, bias, groups, ("visual", "state"), key_chunk_size=2)
    assert all(torch.isfinite(value) and value == 0 for value in stats.values())
    assert "entropy/visual" not in stats and "entropy/state" not in stats


def test_group_entropy_remains_stable_when_its_global_mass_underflows():
    q = torch.ones(1, 1, 1, 1)
    k = torch.tensor([[[[1000.0], [1000.0], [-1000.0], [-1000.0], [-1000.0]]]])
    stats = cross_attention_statistics(
        q, k, None, torch.tensor([0, 0, 1, 1, 1]), ("a", "b"), key_chunk_size=1
    )
    assert stats["mass/a"] == pytest.approx(1)
    assert stats["mass/b"] == 0
    assert stats["entropy/a"] == pytest.approx(1)
    assert stats["entropy/b"] == pytest.approx(1)


def test_mixed_empty_groups_use_per_example_counts_and_defined_entropy_only():
    q, k = torch.zeros(2, 2, 3, 4), torch.zeros(2, 2, 6, 4)
    groups = torch.tensor([[0, 0, 1, 2, 3, -1], [0, 0, 1, 2, 3, 4]])
    bias = torch.zeros(2, 6)
    bias[0, -1] = -torch.inf
    stats = cross_attention_statistics(q, k, bias, groups)
    assert stats["mass/history"] == pytest.approx(1 / 12)
    assert stats["uniform_mass/history"] == pytest.approx(1 / 12)
    assert stats["relative_mass/history"] == pytest.approx(1)
    assert stats["entropy/history"] == 0  # One token in the only present group.
    assert stats["mass/current"] == pytest.approx((3 / 5 + 3 / 6) / 2)
    assert stats["entropy/current"] == pytest.approx(1)
    assert "entropy/history_event" not in stats


def test_hook_preserves_forward_gradients_rng_and_state_dict():
    decoder = ActionFlow(width=16, heads=4, layers=2, horizon=5)
    reference = deepcopy(decoder)
    noise = torch.randn(2, 5, 8)
    context = torch.randn(2, 11, 16)
    bias = torch.zeros(2, 11)
    bias[:, -1] = -torch.inf
    groups = torch.tensor([0, 0, 1, 2, 3, 4, 4, 4, 5, 5, -1])
    rng = torch.get_rng_state().clone()
    saved_keys = set(decoder.state_dict())
    with CrossAttentionDiagnostics(decoder, groups, key_chunk_size=3) as capture:
        prediction = decoder(noise, torch.tensor([0.2, 0.7]), context, bias)
        prediction.square().sum().backward()
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    expected = reference(noise, torch.tensor([0.2, 0.7]), context, bias)
    expected.square().sum().backward()
    torch.testing.assert_close(prediction, expected, atol=0, rtol=0)
    assert set(decoder.state_dict()) == saved_keys
    for (name, actual), (_, target) in zip(decoder.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(actual.grad, target.grad, atol=0, rtol=0, msg=name)
    assert len(capture.layer_metrics()) == 2
    stats = capture.metrics()
    assert sum(stats[f"mass/{name}"] for name in GROUP_NAMES) == pytest.approx(1)
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in decoder.modules())
    restored = deepcopy(reference)
    restored.load_state_dict(decoder.state_dict(), strict=True)


def test_capture_targets_final_euler_evaluation_only_and_preserves_sampling():
    decoder = ActionFlow(width=16, heads=4, layers=2, horizon=5)
    context = torch.randn(1, 8, 16)
    bias = torch.zeros(1, 8)
    groups = torch.tensor([0, 0, 1, 2, 3, 4, 4, 5])
    noise = torch.randn(1, 5, 8)
    expected = decoder.integrate(context, bias, steps=8, noise=noise)
    with CrossAttentionDiagnostics(decoder, groups, capture_on_call=8, key_chunk_size=2) as capture:
        actual = decoder.integrate(context, bias, steps=8, noise=noise)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    stats = capture.metrics()
    x = noise.clone()
    with torch.no_grad():
        for index in range(7):
            x = x + decoder(x, torch.tensor(index / 8), context, bias) / 8
        with CrossAttentionDiagnostics(decoder, groups, key_chunk_size=2) as final:
            decoder(x, torch.tensor(7 / 8), context, bias)
    assert stats == final.metrics()
    with CrossAttentionDiagnostics(decoder, groups, capture_on_call=9) as incomplete:
        decoder.integrate(context, bias, steps=8, noise=noise)
    with pytest.raises(RuntimeError, match="not captured"):
        incomplete.metrics()


def test_hooks_are_removed_when_observed_operation_raises():
    decoder = ActionFlow(width=16, heads=4, layers=1, horizon=5)
    with pytest.raises(RuntimeError, match="deliberate"):
        with CrossAttentionDiagnostics(decoder, torch.tensor([0])):
            raise RuntimeError("deliberate")
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in decoder.modules())


def test_empty_context_returns_no_fabricated_mass_when_projection_hooks_do_not_run():
    decoder = ActionFlow(width=16, heads=4, layers=1, horizon=5)
    with CrossAttentionDiagnostics(decoder, torch.empty(0, dtype=torch.long)) as capture:
        decoder(torch.randn(1, 5, 8), torch.tensor(0.5), torch.zeros(1, 0, 16), None)
    assert all(value == 0 for value in capture.metrics().values())


@pytest.mark.parametrize("labels", [torch.tensor([0, -1]), torch.tensor([0, 6]), torch.tensor([0.0, 1.0])])
def test_malformed_group_labels_fail_instead_of_dropping_attention(labels):
    with pytest.raises(ValueError):
        cross_attention_statistics(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 2, 2), None, labels)
