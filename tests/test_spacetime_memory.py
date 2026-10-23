"""Synthetic CPU diagnostics for sparse Gaussian gradients and saved state."""

from copy import deepcopy
from dataclasses import fields

import pytest
import torch

from stmem.models.spacetime_encoder import EncodedLeaves, SpacetimeEncoder


@pytest.mark.parametrize("framewise", [False, True])
def test_selected_neighbor_gradients_match_independent_dense_scalar_reference(framewise):
    """The scalar encoder is the unchanged dense-distance mathematical reference.

    Include occupancy in the objective: its Gaussian gradient bypasses the
    content softmax and would expose an accidentally detached selected distance.
    Coordinates and clocks also require gradients here, although training data
    does not, so selecting neighbors cannot discard those derivative paths.
    """
    generator = torch.Generator().manual_seed(211)
    points = torch.rand(3, 3, 13, 6, generator=generator)
    points[..., :3] *= 0.2
    # Distinct colors at identical locations force ties across the neighbor cap.
    points[0, :, :, :3] = 0.05
    valid = torch.ones(3, 3, 13, dtype=torch.bool)
    valid[1, 1, 7:] = False
    valid[2] = False  # Entirely empty block must produce finite masked gradients.
    points[~valid] = float("nan")
    times = torch.tensor([[0.0, 1.0, 3.0], [2.0, 3.0, 6.0], [0.0, 1.0, 2.0]])
    quotas = [3, 1, 4]
    centers = torch.full((3, 3, max(quotas)), -1, dtype=torch.long)
    for frame, quota in enumerate(quotas):
        centers[:, frame, :quota] = torch.arange(quota)
    centers[1, 2, -1] = -1  # Empty center slot in an otherwise occupied frame.

    torch.manual_seed(42)
    batched = SpacetimeEncoder(
        width=12,
        neighbor_cap=5,
        center_chunk=3,
        temporal_radius=1.5,
        framewise=framewise,
    )
    scalar = deepcopy(batched)
    left_points, right_points = (points.clone().requires_grad_() for _ in range(2))
    left_times, right_times = (times.clone().requires_grad_() for _ in range(2))
    actual = batched.forward_batched(left_points, valid, left_times, quotas, center_indices=centers)
    reference_rows = [
        scalar(
            cloud,
            mask,
            clock,
            quotas,
            center_indices=[indices[frame, :quota] for frame, quota in enumerate(quotas)],
        )
        for cloud, mask, clock, indices in zip(right_points, valid, right_times, centers)
    ]
    expected = EncodedLeaves(
        *(
            torch.stack([getattr(row, field.name) for row in reference_rows])
            for field in fields(EncodedLeaves)
        )
    )
    for field in fields(EncodedLeaves):
        torch.testing.assert_close(getattr(actual, field.name), getattr(expected, field.name), msg=field.name)

    weights = torch.linspace(0.2, 1.7, actual.content.numel()).reshape_as(actual.content)

    def objective(result):
        return (result.content * weights).square().sum() + 0.17 * result.occupancy.square().sum()

    objective(actual).backward()
    objective(expected).backward()
    for left, right in ((left_points, right_points), (left_times, right_times)):
        assert torch.isfinite(left.grad).all()
        torch.testing.assert_close(left.grad, right.grad, atol=4e-6, rtol=5e-5)
    for (name, left), (_, right) in zip(batched.named_parameters(), scalar.named_parameters()):
        assert left.grad is not None and right.grad is not None, name
        assert torch.isfinite(left.grad).all(), name
        torch.testing.assert_close(left.grad, right.grad, atol=4e-6, rtol=5e-5, msg=name)
    assert batched.raw_tau_x.grad.abs() > 0
    if not framewise:
        assert batched.raw_tau_t.grad.abs() > 0
        assert left_times.grad.abs().sum() > 0
    assert torch.count_nonzero(left_points.grad[~valid]) == 0


def test_training_pooling_does_not_save_dense_center_point_geometry():
    """Backward needs selected geometry, not every center-to-cloud distance.

    Observations are constants as in training. Point-cloud feature MLP inputs
    are legitimately retained; identify only center-by-all-points geometry,
    whose all-points dimension is distinct from width and the neighbor cap.
    """
    generator = torch.Generator().manual_seed(212)
    blocks, frames, points_per_frame, quota = 2, 3, 17, 4
    points = torch.rand(blocks, frames, points_per_frame, 6, generator=generator)
    points[..., :3] *= 0.1
    valid = torch.ones(points.shape[:-1], dtype=torch.bool)
    times = torch.arange(frames, dtype=torch.float32).expand(blocks, -1)
    centers = torch.arange(quota).expand(blocks, frames, -1)
    encoder = SpacetimeEncoder(width=12, neighbor_cap=5, center_chunk=3)
    saved_shapes = []

    def save(tensor):
        if tensor.is_floating_point():
            saved_shapes.append(tuple(tensor.shape))
        # Detach only the hook payload, not the computation: avoids reference
        # cycles between an output's grad_fn and its saved tensor wrapper.
        return tensor.detach()

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        result = encoder.forward_batched(points, valid, times, quota, center_indices=centers)
        (result.content.square().sum() + result.occupancy.square().sum()).backward()

    all_points = frames * points_per_frame
    centers_per_block = frames * quota
    dense_geometry = [
        shape
        for shape in saved_shapes
        if len(shape) in (3, 4)
        and shape[0] == blocks
        and 0 < shape[1] <= centers_per_block
        and shape[2] == all_points
    ]
    assert not dense_geometry, f"Autograd retained dense center-to-cloud geometry: {dense_geometry}"
    assert encoder.raw_tau_x.grad is not None and torch.isfinite(encoder.raw_tau_x.grad)
    assert encoder.raw_tau_t.grad is not None and torch.isfinite(encoder.raw_tau_t.grad)
