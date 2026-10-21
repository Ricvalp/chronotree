"""Geometry invariants; synthetic cases are diagnostics, not task-performance evidence."""

from pathlib import Path

import numpy as np
import pytest
import torch

from stmem.data.center_sampling import (
    center_counts,
    center_sampler_settings,
    center_sampling_signature,
    prepare_center_indices,
    resolve_center_sampling,
    select_center_indices,
    select_cluster_centers_with_diagnostics,
    select_stratified_fps_centers,
)
from stmem.data.pointcloud import point_sampler_settings, rgbd_to_points, sample_points


@pytest.fixture(autouse=True, scope="module")
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def plane_points(side=31):
    axis = torch.linspace(-0.5, 0.5, side)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    p = torch.zeros(side * side, 6)
    p[:, 0], p[:, 1] = x.flatten(), y.flatten()
    return p


def test_sampling_reserves_elevated_geometry_without_removing_flat_surfaces():
    table = plane_points(101)
    generator = torch.Generator().manual_seed(17)
    elevated = torch.rand(600, 6, generator=generator)
    elevated[:, :2] = elevated[:, :2] * 0.08 - 0.04
    elevated[:, 2] = elevated[:, 2] * 0.15 + 0.01
    points = torch.cat([table, elevated])
    sampled, valid = sample_points(points, 128, voxel_size=0)
    assert valid.all()
    assert int((sampled[:, 2] > 0.005).sum()) >= 96
    assert (sampled[:, 2] == 0).any()
    assert len(torch.unique(sampled, dim=0)) == 128

    # Sparse elevated geometry is retained, and unused quota goes to the table.
    sampled, valid = sample_points(torch.cat([table, elevated[:7]]), 128, voxel_size=0)
    assert valid.all() and int((sampled[:, 2] > 0.005).sum()) == 7
    for point in elevated[:7]:
        assert (sampled == point).all(-1).any()


def test_sampling_balances_spatial_cells_instead_of_density():
    generator = torch.Generator().manual_seed(31)
    dense = torch.rand(1000, 6, generator=generator)
    sparse = torch.rand(30, 6, generator=generator)
    dense[:, :2] = 0.01 + dense[:, :2] * 0.07
    sparse[:, :2] = 0.21 + sparse[:, :2] * 0.07
    dense[:, 2] = sparse[:, 2] = 0.02
    sampled, valid = sample_points(torch.cat([dense, sparse]), 40, voxel_size=0, foreground_fraction=1)
    assert valid.all()
    assert int((sampled[:, 0] > 0.2).sum()) == 20
    assert int((sampled[:, 0] < 0.1).sum()) == 20


def test_sampling_is_order_invariant_rng_neutral_and_padding_is_unobserved():
    points = torch.rand(400, 6, generator=torch.Generator().manual_seed(91))
    before = torch.random.get_rng_state().clone()
    a, ma = sample_points(points, 64)
    b, mb = sample_points(points.flip(0), 64)
    assert torch.equal(a, b) and torch.equal(ma, mb)
    assert torch.equal(before, torch.random.get_rng_state())
    invalid = torch.full((3, 6), float("nan"))
    result, valid = sample_points(torch.cat([points[:2], invalid]), 8, voxel_size=0)
    assert int(valid.sum()) == 2 and not result[~valid].any()
    empty, mask = sample_points(torch.empty(0, 6), 8)
    assert not mask.any() and not empty.any()


def test_clustering_moves_table_corner_seeds_to_representative_interiors():
    points = plane_points()
    initial = select_stratified_fps_centers(points, 32)
    selected, report = select_cluster_centers_with_diagnostics(points, 32)
    xyz = points[:, :3].numpy().astype(np.float64)
    _, inverse, counts = np.unique(np.floor(xyz / 0.1), axis=0, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]

    def distortion(indices):
        distance = ((xyz[:, None] - xyz[indices.numpy()][None]) ** 2).sum(-1).min(-1)
        return float(np.dot(weights, distance))

    assert distortion(selected) < distortion(initial)
    assert int((points[initial, :2].abs() > 0.49).all(-1).sum()) == 4
    assert not (points[selected, :2].abs() > 0.49).all(-1).any()
    objective = report["global_stage"]["weighted_squared_distortion"]
    assert np.isclose(objective[0], distortion(initial))
    assert np.isclose(objective[-1], distortion(selected))
    assert np.all(np.diff(objective) <= 1e-10)


def test_centers_are_unique_observed_permutation_invariant_and_stratified():
    generator = torch.Generator().manual_seed(103)
    points = torch.rand(400, 6, generator=generator)
    points[:200, 2] = 0
    before = torch.random.get_rng_state().clone()
    chosen = select_center_indices(points, 32)
    permutation = torch.randperm(len(points), generator=generator)
    shuffled = select_center_indices(points[permutation], 32)
    assert len(chosen) == len(chosen.unique()) == 32
    assert (points[chosen[:24], 2] > 0.005).all()
    assert torch.equal(points[chosen], points[permutation][shuffled])
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("count", [0, 1, 7, 40])
def test_coincident_and_sparse_clouds_do_not_fabricate_centers(count):
    points = torch.zeros(count, 6)
    points[:, 2] = 0.1
    points[:, 3] = torch.arange(count) / max(count, 1)
    chosen = select_center_indices(points, 32)
    assert len(chosen) == min(count, 32)
    assert len(chosen.unique()) == len(chosen)
    assert chosen.dtype == torch.long


def test_prepared_centers_exclude_padding_and_keep_original_indices():
    points = torch.rand(12, 6, generator=torch.Generator().manual_seed(51))
    valid = torch.zeros(12, dtype=torch.bool)
    valid[[2, 5, 8, 10]] = True
    points[5, 0] = float("nan")
    result = prepare_center_indices(points, valid, [2, 8])
    assert result[2].shape == (2,) and result[8].shape == (8,)
    assert set(result[8][result[8] >= 0].tolist()) == {2, 8, 10}
    assert int((result[8] == -1).sum()) == 5
    assert set(result[2].tolist()) <= {2, 8, 10}
    assert not (result[8] < -1).any()


def test_settings_and_signatures_resolve_the_effective_geometry():
    assert center_counts({}) == [32]
    assert center_counts({"model": {"block_size": 4, "visual_tokens": 10, "current_tokens": 12}}) == [
        2,
        3,
        12,
    ]
    assert point_sampler_settings({})["sampling"] == "balanced"
    assert center_sampler_settings({})["method"] == "balanced_cluster"
    assert center_sampling_signature(None) == center_sampling_signature({"table_z": 0})
    assert center_sampling_signature(None) != center_sampling_signature({"cell_size": 0.2})
    assert center_sampling_signature(None) != center_sampling_signature({"method": "fps"})
    assert resolve_center_sampling({"cell_size": 0.1})["max_iterations"] == 20
    with pytest.raises(ValueError, match="at least block_size"):
        center_counts({"model": {"block_size": 8, "visual_tokens": 4}})
    with pytest.raises(ValueError, match="Unknown"):
        center_sampler_settings({"model": {"encoder_options": {"center_sampling": {"object_mask": True}}}})
    with pytest.raises(ValueError, match="Unknown"):
        point_sampler_settings({"geometry": {"point_sampling": {"target_id": 20}}})


def test_optional_real_reviewed_preview_parity():
    """Cross-check the independently saved real preview when local artifacts exist."""
    root = Path(__file__).resolve().parents[1]
    preview = root / "runs/center_comparison32"
    captures = root / "runs/sampling_preview/capture"
    if not (preview / "traces/pick80_current_cluster.npz").exists():
        pytest.skip("Optional bounded real preview artifacts are not present")
    scenes = [
        ("pick80", "PickXtimes_val0", "execution", 80, list(range(72, 80))),
        ("pick40", "PickXtimes_val0", "execution", 40, list(range(32, 40))),
        ("video24", "VideoUnmask_ep0", "execution", 24, list(range(16, 24))),
        ("video64", "VideoUnmask_ep0", "execution", 64, list(range(56, 64))),
        ("video_conditioning", "VideoUnmask_ep0", "conditioning", 24, list(range(56, 64))),
    ]
    for name, directory, historical_source, decision, history in scenes:
        for context, times in [("current", [decision]), ("history", history)]:
            with np.load(preview / f"traces/{name}_{context}_cluster.npz") as trace:
                for frame_id, time in enumerate(times):
                    source = "execution" if context == "current" else historical_source
                    stem = f"frame_{time:03d}" if directory.startswith("Pick") else f"{source}_{time:03d}"
                    with np.load(captures / directory / f"{stem}.npz") as raw:
                        sampled, valid = rgbd_to_points(dict(raw), dict(raw))
                    expected = trace["points"][trace["point_frame_indices"] == frame_id]
                    assert np.array_equal(sampled[valid].numpy(), expected)
                    selected = select_center_indices(sampled[valid], 32)
                    offset = int(np.flatnonzero(trace["point_frame_indices"] == frame_id)[0])
                    expected_indices = trace["center_point_indices"][
                        trace["center_frame_indices"] == frame_id
                    ]
                    assert np.array_equal(selected.numpy() + offset, expected_indices)
