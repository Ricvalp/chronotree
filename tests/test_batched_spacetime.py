"""Synthetic diagnostics for batched local geometry, masks and gradients."""

from copy import deepcopy
from dataclasses import fields

import pytest
import torch

from stmem.models.spacetime_encoder import EncodedLeaves, SpacetimeEncoder


def _inputs():
    generator = torch.Generator().manual_seed(413)
    points = torch.rand(3, 3, 11, 6, generator=generator)
    points[..., :3] *= 0.2
    valid = torch.rand(3, 3, 11, generator=generator) > 0.25
    valid[1, 2] = False  # A padded final frame.
    valid[2, 1, 2:] = False  # Fewer points than center candidates.
    times = torch.tensor([[0.0, 1.0, 2.0], [0.0, 3.0, 3.0], [5.0, 7.0, 11.0]])
    return points, valid, times


def _encoder(**kwargs):
    torch.manual_seed(29)
    return SpacetimeEncoder(width=12, neighbor_cap=5, center_chunk=3, **kwargs)


def _assert_fields_close(actual, expected):
    for field in fields(EncodedLeaves):
        torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name), msg=field.name)


@pytest.mark.parametrize("framewise", [False, True])
@pytest.mark.parametrize("quotas", [2, [3, 1, 4]])
def test_vectorized_blocks_match_individual_reference_forward_and_gradients(framewise, quotas):
    points, valid, times = _inputs()
    batched = _encoder(framewise=framewise)
    reference = deepcopy(batched)
    input_batch = points.clone().requires_grad_()
    input_reference = points.clone().requires_grad_()
    actual = batched.forward_batched(input_batch, valid, times, quotas)
    expected = [reference(p, v, t, quotas) for p, v, t in zip(input_reference, valid, times)]
    _assert_fields_close(
        actual,
        EncodedLeaves(*(torch.stack([getattr(x, f.name) for x in expected]) for f in fields(EncodedLeaves))),
    )
    actual.content.square().sum().backward()
    sum(output.content.square().sum() for output in expected).backward()
    torch.testing.assert_close(input_batch.grad, input_reference.grad, atol=1e-6, rtol=3e-5)
    for (name, left), (_, right) in zip(batched.named_parameters(), reference.named_parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=3e-6, rtol=3e-5, msg=name)


def test_padding_nonfinite_values_and_other_batch_members_cannot_change_valid_outputs():
    points, valid, times = _inputs()
    encoder = _encoder()
    baseline = encoder.forward_batched(points, valid, times, [3, 2, 4])
    padded = torch.full((3, 3, 19, 6), float("nan"))
    padded[:, :, :11] = points
    padded_valid = torch.zeros((3, 3, 19), dtype=torch.bool)
    padded_valid[:, :, :11] = valid
    padded[:, :, :11][~valid] = float("inf")
    _assert_fields_close(encoder.forward_batched(padded, padded_valid, times, [3, 2, 4]), baseline)
    points[1:] *= 999
    altered = encoder.forward_batched(points, valid, times, [3, 2, 4])
    for f in fields(EncodedLeaves):
        torch.testing.assert_close(getattr(altered, f.name)[0], getattr(baseline, f.name)[0])


@pytest.mark.parametrize("framewise", [False, True])
def test_point_permutations_and_equal_distance_neighbor_cap_are_padding_invariant(framewise):
    # Distinct colors at identical coordinates force ties right across the cap.
    points = torch.zeros(2, 2, 9, 6)
    points[..., 3] = torch.arange(9) / 9
    points[..., 4] = torch.tensor([0.2, 0.9])[:, None, None]
    valid = torch.ones(2, 2, 9, dtype=torch.bool)
    times = torch.zeros(2, 2)
    encoder = _encoder(framewise=framewise)
    baseline = encoder.forward_batched(points, valid, times, 3)
    permutation = torch.tensor([4, 7, 2, 8, 0, 1, 5, 3, 6])
    _assert_fields_close(encoder.forward_batched(points[:, :, permutation], valid, times, 3), baseline)
    padded = torch.cat([points[:, :, permutation], torch.zeros(2, 2, 12, 6)], dim=2)
    padded_valid = torch.cat([valid, torch.zeros(2, 2, 12, dtype=torch.bool)], dim=2)
    _assert_fields_close(encoder.forward_batched(padded, padded_valid, times, 3), baseline)
    expected = [encoder(p, v, t, 3) for p, v, t in zip(points, valid, times)]
    _assert_fields_close(
        baseline,
        EncodedLeaves(*(torch.stack([getattr(x, f.name) for x in expected]) for f in fields(EncodedLeaves))),
    )


def test_masked_future_frames_match_observed_prefix_and_do_not_leak_features():
    points, valid, times = _inputs()
    encoder = _encoder()
    prefix = encoder.forward_batched(points[:, :2], valid[:, :2], times[:, :2], [3, 2])
    valid[:, 2] = False
    points[:, 2] = 1e9
    padded = encoder.forward_batched(points, valid, times, [3, 2, 4])
    for f in fields(EncodedLeaves):
        torch.testing.assert_close(getattr(padded, f.name)[:, :5], getattr(prefix, f.name))
    assert not padded.valid[:, 5:].any()
    assert torch.count_nonzero(padded.content[:, 5:]) == 0


def test_all_invalid_rows_return_finite_zero_content_and_finite_zero_gradients():
    encoder = _encoder()
    points = torch.full((2, 3, 5, 6), float("nan"), requires_grad=True)
    # Nonfinite points are excluded even if the external mask says True.
    result = encoder.forward_batched(points, torch.ones(2, 3, 5, dtype=torch.bool), torch.zeros(2, 3), 4)
    assert not result.valid.any()
    for f in fields(EncodedLeaves):
        assert torch.isfinite(getattr(result, f.name)).all()
    assert torch.count_nonzero(result.content) == 0
    assert torch.count_nonzero(result.occupancy) == 0
    result.content.sum().backward()
    assert torch.equal(points.grad, torch.zeros_like(points.grad))
    for parameter in encoder.parameters():
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))


@pytest.mark.parametrize("shape,quota", [((2, 3, 0, 6), 4), ((2, 3, 5, 6), 0), ((0, 3, 5, 6), 4)])
def test_empty_shapes_retain_declared_leaf_slots(shape, quota):
    result = _encoder().forward_batched(
        torch.zeros(shape), torch.zeros(shape[:-1], dtype=torch.bool), torch.zeros(shape[:2]), quota
    )
    assert result.content.shape == (shape[0], shape[1] * quota, 12)
    assert not result.valid.any()


def test_batched_forward_never_reads_tensor_values_on_cpu(monkeypatch):
    points, valid, times = _inputs()
    encoder = _encoder()

    def forbidden(*args, **kwargs):
        raise AssertionError("Batched pooling may not synchronize tensor values with the CPU")

    for method in ("cpu", "item", "numpy", "tolist", "__bool__"):
        monkeypatch.setattr(torch.Tensor, method, forbidden)
    encoder.forward_batched(points, valid, times, [3, 2, 4])
