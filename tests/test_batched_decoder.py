"""Batching preserves per-decision predictions, masks and gradient weights."""

from copy import deepcopy

import pytest
import torch

from stmem.models.action_flow import (
    ActionFlow,
    ActionNormalizer,
    Attention,
    RegressionHead,
    flow_sample,
    masked_mse,
)
from stmem.models.perceiver import Perceiver
from stmem.models.text_encoder import TextEncoder


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(194)


def contexts(lengths, width=16):
    content = torch.randn(len(lengths), max(lengths), width)
    bias = torch.zeros(len(lengths), max(lengths))
    for index, length in enumerate(lengths):
        bias[index, length:] = -torch.inf
        # Invalid values must not contaminate a valid example's prediction.
        content[index, length:] = float("nan")
    return content, bias


@pytest.mark.parametrize("decoder_type", [ActionFlow, RegressionHead])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_decoder_batch_matches_separate_examples_and_gradients(decoder_type, batch_size):
    model = decoder_type(width=16, heads=4, layers=2, horizon=3)
    separate = deepcopy(model)
    lengths = [2, 5, 3][:batch_size]
    context, bias = contexts(lengths)
    noise = torch.randn(batch_size, 3, 8)
    times = torch.linspace(0.2, 0.8, batch_size)
    target = torch.randn_like(noise)
    valid = torch.ones_like(noise, dtype=torch.bool)
    valid[0, -1] = False
    result = model(noise, times, context, bias)
    assert result.shape == noise.shape
    expected = torch.stack(
        [
            separate(noise[i], times[i], context[i, :length], bias[i, :length])
            for i, length in enumerate(lengths)
        ]
    )
    torch.testing.assert_close(result, expected, atol=5e-7, rtol=2e-5)
    masked_mse(result, target, valid).backward()
    sum(masked_mse(expected[i], target[i], valid[i]) / batch_size for i in range(batch_size)).backward()
    for actual, reference in zip(model.parameters(), separate.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, reference.grad, atol=7e-7, rtol=4e-5)


def test_action_loss_weights_examples_equally_with_terminal_padding():
    prediction = torch.zeros(3, 4, 8, requires_grad=True)
    target = torch.stack([torch.full((4, 8), 2.0), torch.ones(4, 8), torch.zeros(4, 8)])
    valid = torch.zeros_like(prediction, dtype=torch.bool)
    valid[0, 0, 0] = True
    valid[1] = True
    target[~valid] = float("nan")
    losses = masked_mse(prediction, target, valid, reduction="none")
    torch.testing.assert_close(losses, torch.tensor([4.0, 1.0, 0.0]))
    loss = masked_mse(prediction, target, valid)
    torch.testing.assert_close(loss, torch.tensor(5 / 3))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[~valid]) == 0


@pytest.mark.parametrize("batch_size", [1, 4])
def test_flow_samples_independent_times_and_correct_interpolation(batch_size):
    actions = torch.randn(batch_size, 3, 8)
    valid = torch.ones_like(actions, dtype=torch.bool)
    valid[0, -1] = False
    normalizer = ActionNormalizer()
    x, time, target, noise = flow_sample(actions, valid, normalizer)
    assert time.shape == (batch_size,)
    assert len(torch.unique(time)) == batch_size
    clean = normalizer.normalize(actions)
    torch.testing.assert_close(
        x, ((1 - time[:, None, None]) * noise + time[:, None, None] * clean).masked_fill(~valid, 0)
    )
    torch.testing.assert_close(target, (clean - noise).masked_fill(~valid, 0))
    for i in range(batch_size):
        single = flow_sample(actions[i], valid[i], normalizer, noise=noise[i], time=time[i])
        torch.testing.assert_close(x[i], single[0])
        torch.testing.assert_close(target[i], single[2])
    shared, times, _, _ = flow_sample(actions, valid, normalizer, noise=noise, time=torch.tensor(0.5))
    assert times.shape == (batch_size,)
    torch.testing.assert_close(shared, (0.5 * (noise + clean)).masked_fill(~valid, 0))


@pytest.mark.parametrize("decoder_type", [ActionFlow, RegressionHead])
def test_batched_integration_retains_batch_dimension(decoder_type):
    model = decoder_type(width=16, heads=4, layers=1, horizon=3)
    context, bias = contexts([2])
    noise = torch.randn(1, 3, 8)
    batched = model.integrate(context, bias, noise=noise, steps=2)
    single = model.integrate(context[0], bias[0], noise=noise[0], steps=2)
    assert batched.shape == (1, 3, 8)
    assert single.shape == (3, 8)
    torch.testing.assert_close(batched[0], single)
    assert model.integrate(context, bias, steps=2).shape == (1, 3, 8)


def test_text_padding_matches_unbatched_outputs_and_gradients():
    model = TextEncoder(16, heads=4, layers=2, max_bytes=12)
    separate = deepcopy(model)
    text = ["", "a", "grab the blue object", "é robot"]
    result = model(text)
    expected = torch.stack([separate(item) for item in text])
    torch.testing.assert_close(result, expected, atol=5e-7, rtol=1e-5)
    weights = torch.randn_like(result)
    (result * weights).sum().backward()
    (expected * weights).sum().backward()
    for actual, reference in zip(model.parameters(), separate.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, reference.grad, atol=3e-6, rtol=6e-5)
    assert model(["a"]).shape == (1, 16)
    assert model("a").shape == (16,)


def test_perceiver_masked_batch_matches_single_histories_and_gradients():
    model = Perceiver(width=16, heads=4, budget=3)
    separate = deepcopy(model)
    history, bias = contexts([2, 0, 5])
    history.requires_grad_()
    result, valid = model(history, bias, return_mask=True)
    expected = torch.stack(
        [separate(history[0, :2], bias[0, :2]), torch.zeros(3, 16), separate(history[2], bias[2])]
    )
    assert result.shape == (3, 3, 16)
    assert valid.tolist() == [[True] * 3, [False] * 3, [True] * 3]
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected)
    result.square().sum().backward(retain_graph=True)
    expected.square().sum().backward()
    assert torch.isfinite(history.grad).all()
    assert torch.count_nonzero(history.grad[1]) == 0
    for actual, reference in zip(model.parameters(), separate.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, reference.grad, atol=2e-6, rtol=3e-5)
    empty = model(history[:1, :0], bias[:1, :0], return_mask=True)
    assert empty[0].shape == (1, 3, 16)
    assert not empty[1].any()
    assert model(history[0, :0], bias[0, :0]).shape == (0, 16)


def test_attention_all_empty_rows_return_zero_and_finite_gradients():
    attention = Attention(16, 4)
    query = torch.randn(2, 3, 16, requires_grad=True)
    context = torch.full((2, 5, 16), float("nan"), requires_grad=True)
    bias = torch.full((2, 5), -torch.inf)
    result = attention(query, context, bias)
    assert torch.count_nonzero(result) == 0
    result.sum().backward()
    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(context.grad).all()
    for parameter in attention.parameters():
        assert torch.isfinite(parameter.grad).all()
