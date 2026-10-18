from dataclasses import replace
import random
from types import SimpleNamespace

import pytest
import torch

from stmem.data.canonical import Frame, Episode
from stmem.memory.records import aggregate, leaf_record
from stmem.memory.hierarchy import Archive, local_tree, maximum_root_count
from stmem.memory.frontier import select_tree, select_uniform, select_random, validate_frontier
from stmem.memory.flat import FlatState, select_flat
from stmem.memory.streaming import StreamingMemory, build_archive
from stmem.models.spacetime_encoder import BlockEncoder


def make_leaf(i, source="execution", modality="visual", completed=True):
    content = torch.tensor([i, 2 * i + 1, -i], dtype=torch.float32)
    anchor = torch.tensor([float(i), 0.0, 0.0])
    return leaf_record(
        f"{source}:{modality}:{i}", content, anchor, float(i), source, modality, completed=completed
    )


def make_frame(i, source="execution", points=12):
    rng = torch.Generator().manual_seed(100 + i)
    xyzrgb = torch.rand(points, 6, generator=rng)
    xyzrgb[:, :3] *= 0.15
    return Frame(
        xyzrgb,
        torch.ones(points, dtype=torch.bool),
        torch.rand(15, generator=rng),
        torch.rand(8, generator=rng),
        source == "execution" and i > 0,
        float(i),
        source,
        f"{source}:{i}",
    )


def small_encoder(**kwargs):
    return BlockEncoder(
        width=12, visual_tokens=4, current_tokens=5, block_size=2, neighbor_cap=8, center_chunk=3, **kwargs
    )


def test_binary_counter_coverage_sums_bounds_and_budget():
    archive = Archive()
    all_leaves = []
    for source in ("conditioning", "execution"):
        for block in range(7):
            leaves = [
                make_leaf(block * 3 + i, source, modality)
                for modality in ("visual", "event")
                for i in range(3)
            ]
            archive.add_block(leaves, f"{source}:{block}")
            all_leaves.extend(leaves)
    assert len(archive.roots) == 12  # popcount(7)=3, two sources and modalities
    by_id = {r.node_id: r for r in all_leaves}
    for record in archive.records:
        members = [by_id[i] for i in record.leaf_ids]
        assert record.count == len(members)
        torch.testing.assert_close(record.content_sum, torch.stack([r.content for r in members]).sum(0))
        torch.testing.assert_close(record.anchor, torch.stack([r.anchor for r in members]).mean(0))
        torch.testing.assert_close(record.bounds_min, torch.stack([r.bounds_min for r in members]).amin(0))
        torch.testing.assert_close(record.bounds_max, torch.stack([r.bounds_max for r in members]).amax(0))
        assert record.time_min == min(r.time_min for r in members)
        assert record.time_max == max(r.time_max for r in members)
    for budget in (12, 13, 21, 84, 100):
        seen = []

        def score(records):
            seen.extend(r.node_id for r in records)
            return [-1e6 for r in records]

        result = select_tree(archive.roots, budget, score)
        assert len(result.records) == min(budget, len(all_leaves))
        assert len(set(seen)) == len(seen)
        assert result.stats["nodes_scored"] <= len(archive.roots) + 2 * max(0, budget - len(archive.roots))
        validate_frontier(result.records, all_leaves, budget)
        validate_frontier(select_uniform(archive.roots, budget), all_leaves, budget)
        validate_frontier(select_random(archive.roots, budget, random.Random(2)), all_leaves, budget)
    with pytest.raises(ValueError, match="all 12"):
        select_tree(archive.roots, 11, lambda records: [0] * len(records))
    assert select_uniform([], 0).records == []


def test_aggregation_is_differentiable_and_before_any_adapter():
    content = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)
    leaves = [
        replace(make_leaf(i), content_sum=content * (i + 1), max_content=content * (i + 1)) for i in range(4)
    ]
    root = local_tree(leaves)
    torch.testing.assert_close(root.content, content * 2.5)
    root.content.sum().backward()
    torch.testing.assert_close(content.grad, torch.full_like(content, 2.5))
    with pytest.raises(ValueError, match="sources or modalities"):
        aggregate([make_leaf(0), make_leaf(1, modality="event")], "bad")


@pytest.mark.parametrize("sizes", [(1,), (2,), (1, 6), (2, 2), (3, 1, 5, 2), (8, 7, 6, 5)])
def test_flat_exact_residuals_fill_capacity_and_zero_cost(sizes):
    groups = [(s, m) for s in ("conditioning", "execution") for m in ("visual", "event")]
    leaves = [make_leaf(j, *groups[i]) for i, n in enumerate(sizes) for j in range(n)]
    by_id = {r.node_id: r for r in leaves}
    for budget in range(len(sizes), len(leaves) + 2):
        calls = []

        def score(records):
            calls.append(len(records))
            return [float(i % 3) for i in range(len(records))]

        result = select_flat(leaves, budget, score)
        assert len(result) == min(budget, len(leaves))
        assert calls == ([] if len(leaves) <= budget else [len(leaves)])
        validate_frontier(result, leaves, budget)
        for record in result:
            members = [by_id[i] for i in record.leaf_ids]
            torch.testing.assert_close(record.content, torch.stack([r.content for r in members]).mean(0))
            assert record.count == len(members)
            if record.residual_leaf_ids:
                assert record.disconnected
    state = FlatState(tuple(leaves)).normalize_singletons()
    for leaf in state.unselected:
        assert state.move_cost(leaf) > 0
        moved = state.move(leaf)
        assert moved.token_count - state.token_count == state.move_cost(leaf)
    assert select_flat([], 0, lambda _: []).records == []


def test_encoder_point_permutation_masking_unit_mass_and_gradient():
    torch.manual_seed(9)
    encoder = small_encoder(checkpoint_blocks=True).train()
    frames = [make_frame(0), make_frame(1)]
    original = encoder.encode_frames(frames)
    permuted = []
    for frame in frames:
        order = torch.randperm(len(frame.points))
        permuted.append(replace(frame, points=frame.points[order], valid=frame.valid[order]))
    shuffled = encoder.encode_frames(permuted)
    assert [r.node_id for r in shuffled] == [r.node_id for r in original]
    for left, right in zip(original, shuffled):
        torch.testing.assert_close(left.content, right.content)
        torch.testing.assert_close(left.anchor, right.anchor)
        assert left.count == 1
    loss = sum(r.content.square().mean() for r in original)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.visual.parameters())
    assert all(torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None)
    invalid = [
        replace(f, points=torch.full_like(f.points, torch.nan), valid=torch.zeros_like(f.valid))
        for f in frames
    ]
    leaves = encoder.encode_frames(invalid)
    assert len(leaves) == 2 and all(r.modality == "event" for r in leaves)
    pooled = encoder.visual([f.points for f in invalid], [f.valid for f in invalid], [0.0, 1.0], [2, 2])
    assert not pooled.valid.any() and torch.equal(pooled.content, torch.zeros_like(pooled.content))


def test_conditioning_commands_unavailable_and_no_cacheable_absolute_age():
    torch.manual_seed(1)
    encoder = small_encoder().eval()
    conditioning = [make_frame(i, "conditioning") for i in range(2)]
    first = encoder.encode_frames(conditioning)
    changed = [replace(f, previous_action=torch.full((8,), 1000.0)) for f in conditioning]
    second = encoder.encode_frames(changed)
    for a, b in zip(first, second):
        torch.testing.assert_close(a.content, b.content)
    shifted = [replace(f, time=f.time + 100.0) for f in conditioning]
    shifted_records = encoder.encode_frames(shifted)
    for a, b in zip(first, shifted_records):
        torch.testing.assert_close(a.content, b.content)
        assert b.time_min - a.time_min == 100


def test_streaming_parity_causality_prefix_replacement_and_source_reset():
    torch.manual_seed(5)
    encoder = small_encoder().eval()
    conditioning = [make_frame(i, "conditioning") for i in range(3)]
    execution = [make_frame(i) for i in range(7)]
    online = StreamingMemory(encoder)
    online.reset(conditioning, execution[0])
    prior_prefix_ids = set()
    for index, frame in enumerate(execution):
        if index:
            assert online.observe(frame)
        assert not online.observe(frame)
        actual = online.archive
        expected = build_archive(conditioning, execution[:index], encoder)
        assert [r.node_id for r in actual.leaves] == [r.node_id for r in expected.leaves]
        assert [r.node_id for r in actual.roots] == [r.node_id for r in expected.roots]
        for a, b in zip(actual.records, expected.records):
            torch.testing.assert_close(a.content, b.content)
            assert a.content.device.type == "cpu" and not a.content.requires_grad
            if a.source == "execution":
                assert a.time_max < frame.time
        prefix_ids = {r.node_id for r in actual.leaves if not r.completed}
        if index > 0 and index % 2 == 0:
            assert not prefix_ids and not prior_prefix_ids.intersection(r.node_id for r in actual.leaves)
        prior_prefix_ids = prefix_ids
        validate_frontier(select_uniform(actual.roots, 64), actual.leaves, 64)
        # Future perturbations cannot affect a previously constructed prefix.
        altered = execution[:index] + [
            replace(f, points=f.points * 500, previous_action=f.previous_action * -20)
            for f in execution[index:]
        ]
        causal = build_archive(conditioning, altered[:index], encoder)
        for a, b in zip(causal.leaves, expected.leaves):
            torch.testing.assert_close(a.content, b.content)
    assert online.storage["pending_raw_bytes"] == 0
    online.reset([], make_frame(0))
    assert online.leaves == [] and online.roots == [] and online.execution_observations == 1


def test_framewise_matched_leaf_count_and_temporal_support():
    torch.manual_seed(10)
    joint, framewise = small_encoder(), small_encoder(variant="framewise")
    framewise.load_state_dict(joint.state_dict())
    frames = [make_frame(i) for i in range(2)]
    left, right = joint.encode_frames(frames), framewise.encode_frames(frames)
    assert len(left) == len(right)
    assert all(r.time_min == r.time_max for r in right)
    assert any(r.time_min < r.time_max for r in left if r.modality == "visual")


def test_supported_budget_bound_accounts_for_intermediate_prefixes():
    # At final length16 there is one root/group; worst intermediate15 has four.
    assert maximum_root_count(16, 0, 2) == 8
    assert maximum_root_count(16, 16, 2) == 14
    with pytest.raises(ValueError, match="worst supported root bound"):
        StreamingMemory(small_encoder(), max_execution_frames=16, max_conditioning_frames=16).validate_budget(
            8
        )


def test_hdf5_frozen_cache_matches_differentiable_and_streaming_prefixes(tmp_path):
    import h5py
    from stmem.data.cache import FrozenCache, episode_key, write_episode_cache

    torch.manual_seed(19)
    encoder = small_encoder().eval()
    episode = Episode(
        [make_frame(i, "conditioning") for i in range(3)],
        [make_frame(i) for i in range(8)],
        torch.zeros(8, 8),
        torch.ones(8, 8, dtype=torch.bool),
        "pick the observed object",
        "synthetic",
        "0",
        alignment="synthetic_post_action_v1",
    )
    policy = SimpleNamespace(encoder=encoder, text_encoder=lambda goal: torch.arange(12).float())
    with torch.no_grad(), h5py.File(tmp_path / "synthetic.h5", "w") as f:
        write_episode_cache(
            f.create_group(episode_key(episode.task, episode.episode_id)), episode, policy, 1, 2
        )
    # Test real frozen-cache read/reconstruction without requiring a training run.
    cache = FrozenCache.__new__(FrozenCache)
    cache.cfg = {"model": {"block_size": 2}}
    cache.root = tmp_path
    online = StreamingMemory(encoder)
    online.reset(episode.conditioning, episode.execution[0])
    for index in range(7):
        if index:
            online.observe(episode.execution[index])
        decision = episode.decision(index, 2)
        cached = cache.evidence(decision)
        direct = build_archive(decision.conditioning, decision.execution_prefix, encoder)
        for archive in (direct, online.archive):
            assert [r.node_id for r in cached.archive.leaves] == [r.node_id for r in archive.leaves]
            assert [r.node_id for r in cached.archive.roots] == [r.node_id for r in archive.roots]
            for a, b in zip(cached.archive.records, archive.records):
                torch.testing.assert_close(a.content, b.content)
                assert a.completed == b.completed
        for a, b in zip(cached.current_records, encoder.current(decision.current)):
            torch.testing.assert_close(a.content, b.content)
        cached_read = select_uniform(cached.archive.roots, 12)
        online_read = select_uniform(online.roots, 12)
        assert [r.node_id for r in cached_read] == [r.node_id for r in online_read]
        for a, b in zip(cached_read, online_read):
            torch.testing.assert_close(a.content, b.content)
