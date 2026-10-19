"""Synthetic invariants for exact encoder inspection; no success evaluation."""

from copy import deepcopy
from dataclasses import replace
import json
import random

import numpy as np
import pytest
import torch

from stmem.data.canonical import Episode, Frame
from stmem.memory.hierarchy import walk_records
from stmem.memory.streaming import build_archive
from stmem.models.spacetime_encoder import BlockEncoder, SpacetimeEncoder
from stmem.visualization.encoder_structure import build_structure


def _frame(index, source="execution", count=7):
    generator = torch.Generator().manual_seed(200 + index)
    points = torch.rand(count, 6, generator=generator)
    points[:, :3] *= 0.15
    valid = torch.ones(count, dtype=torch.bool)
    valid[-1] = False
    return Frame(
        points,
        valid,
        torch.rand(15, generator=generator),
        torch.rand(8, generator=generator),
        source == "execution" and index > 0,
        float(index),
        source,
        f"{source}:{index}",
    )


def _episode():
    return Episode(
        [_frame(index, "conditioning") for index in range(5)],
        [_frame(index) for index in range(8)],
        torch.zeros(8, 8),
        torch.ones(8, 8, dtype=torch.bool),
        "synthetic diagnostic",
        "synthetic",
        "0",
        alignment="synthetic_post_action_v1",
    )


def _encoder(**kwargs):
    return BlockEncoder(
        width=12,
        block_size=2,
        visual_tokens=5,
        current_tokens=4,
        neighbor_cap=4,
        center_chunk=2,
        **kwargs,
    )


def _capture(storage):
    def collect(trace):
        storage.append(
            {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in trace.items()
            }
        )

    return collect


@pytest.mark.parametrize("checkpoint_blocks", [False, True])
def test_trace_preserves_exact_outputs_and_parameter_gradients(checkpoint_blocks):
    torch.manual_seed(45)
    plain = _encoder(checkpoint_blocks=checkpoint_blocks).train()
    traced = deepcopy(plain)
    frames = [_frame(0), _frame(1)]
    baseline = plain.encode_frames(frames)
    snapshots = []
    observed = traced.encode_frames(frames, trace=_capture(snapshots))
    assert len(snapshots) == 1
    for left, right in zip(baseline, observed):
        assert left.node_id == right.node_id
        assert torch.equal(left.content, right.content)
    sum(record.content.square().sum() for record in baseline).backward()
    sum(record.content.square().sum() for record in observed).backward()
    for (name, left), (_, right) in zip(plain.named_parameters(), traced.named_parameters()):
        assert torch.equal(left.grad, right.grad), name
    assert all(not value.requires_grad for value in snapshots[0].values() if isinstance(value, torch.Tensor))


@pytest.mark.parametrize("framewise", [False, True])
def test_trace_is_exact_graph_used_for_pooling_with_normalized_gaussian_weights(framewise):
    encoder = SpacetimeEncoder(
        width=8,
        neighbor_cap=5,
        center_chunk=2,
        framewise=framewise,
        spatial_radius=0.2,
        temporal_radius=2.0,
    )
    frames = [_frame(0), _frame(1)]
    # A nonfinite point is excluded even if the external validity mask says True.
    frames[0].points[0, 0] = float("nan")
    snapshots = []
    result = encoder(
        [frame.points for frame in frames],
        [frame.valid for frame in frames],
        [0.0, 1.0],
        [3, 2],
        trace=_capture(snapshots),
    )
    graph = snapshots[0]
    assert set(graph).isdisjoint({"content", "features", "edge", "hidden_state"})
    cloud, indices, mask = graph["points"], graph["neighbor_indices"], graph["neighbor_mask"]
    assert len(cloud) == 11
    assert torch.isfinite(cloud).all()
    weights = graph["weights"]
    assert torch.equal(weights[~mask], torch.zeros_like(weights[~mask]))
    torch.testing.assert_close(weights.sum(-1), result.valid.float())
    dx = cloud[indices, :3] - graph["anchor"][:, None, :]
    dt = graph["point_times"][indices] - graph["center_times"][:, None]
    logits = -(dx.square().sum(-1) / encoder.tau_x + dt.square() / encoder.tau_t)
    expected = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
    torch.testing.assert_close(weights, expected)
    assert (dx.square().sum(-1)[mask] <= encoder.spatial_radius**2).all()
    assert (dt.abs()[mask] <= encoder.temporal_radius).all()
    if framewise:
        assert torch.equal(
            graph["point_frame_indices"][indices][mask],
            graph["center_frame_indices"][:, None].expand_as(indices)[mask],
        )
    features = encoder.point_mlp(cloud)
    edge = encoder.edge_mlp(torch.cat([features[indices], dx, dt[..., None]], dim=-1))
    torch.testing.assert_close(result.content, (weights[..., None] * edge).sum(1))
    torch.testing.assert_close(graph["anchor"], cloud[graph["center_point_indices"], :3])


@pytest.mark.parametrize("quota", [0, 3])
def test_empty_or_all_invalid_points_emit_complete_empty_trace(quota):
    encoder = SpacetimeEncoder(width=8)
    snapshots = []
    result = encoder(
        [torch.full((4, 6), float("nan"))],
        [torch.zeros(4, dtype=torch.bool)],
        [0.0],
        [quota],
        trace=_capture(snapshots),
    )
    graph = snapshots[0]
    assert graph["points"].shape == (0, 6)
    assert graph["neighbor_indices"].shape == (quota, 0)
    assert graph["neighbor_mask"].shape == (quota, 0)
    assert graph["weights"].shape == (quota, 0)
    assert not result.valid.any()
    assert torch.equal(result.content, torch.zeros_like(result.content))


def test_payload_uses_exact_shared_archive_with_completed_conditioning_and_pending_execution():
    episode, encoder = _episode(), _encoder()
    payload = build_structure(episode, encoder, 5, detail_blocks=1)
    with torch.no_grad():
        archive = build_archive(episode.conditioning, episode.execution[:5], encoder)
    roots = archive.roots
    records = walk_records(roots)
    assert payload["archive"]["roots"] == [record.node_id for record in roots]
    assert payload["archive"]["leaf_ids"] == [record.node_id for record in archive.leaves]
    nodes = {node["id"]: node for node in payload["archive"]["nodes"]}
    for record in records:
        node = nodes[record.node_id]
        assert node["leaf_ids"] == list(record.leaf_ids)
        assert node["children"] == [child.node_id for child in record.children]
        assert node["count"] == record.count == len(node["leaf_ids"])
        assert node["anchor"] == record.anchor.tolist()
        assert (node["source"], node["modality"]) == record.group
        assert node["completed"] == record.completed
        for child_id in node["children"]:
            child = nodes[child_id]
            assert (child["source"], child["modality"]) == (node["source"], node["modality"])
    history = [block for block in payload["blocks"] if block["kind"] == "history"]
    assert len(history) == 6
    assert [block["detailed"] for block in history] == [False, False, True, False, False, True]
    assert history[2]["completed"]  # Short final conditioning block is final, not pending.
    assert not history[-1]["completed"]
    assert [[frame["quota"] for frame in block["frames"]] for block in history] == [
        [3, 2],
        [3, 2],
        [3],
        [3, 2],
        [3, 2],
        [3],
    ]
    for block in history:
        for modality in ("visual", "event"):
            local_root = nodes[block["local_roots"][modality]]
            assert set(local_root["leaf_ids"]) == set(block[f"{modality}_leaf_ids"])
            assert local_root["block_id"] == block["id"]
    assert any(node["kind"] == "temporal" for node in nodes.values())
    assert any(node["kind"] == "local" for node in nodes.values())
    assert set(payload["current"]["leaf_ids"]).isdisjoint(payload["archive"]["leaf_ids"])
    assert payload["blocks"][-1]["frames"][0]["event_id"] == "execution:5"
    assert payload["blocks"][-1]["event_leaf_ids"] == []
    json.dumps(payload, allow_nan=False)


def test_payload_omits_targets_future_and_features_and_preserves_rng_modes_and_gradients():
    episode, encoder = _episode(), _encoder()
    encoder.train()
    encoder.visual.edge_mlp.eval()  # Preserve heterogeneous mode flags exactly.
    modes = {name: module.training for name, module in encoder.named_modules()}
    for parameter in encoder.parameters():
        parameter.grad = torch.full_like(parameter, 0.25)
    grads = [parameter.grad.clone() for parameter in encoder.parameters()]
    torch_state, python_state, numpy_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
    payload = build_structure(episode, encoder, 3)
    altered = replace(
        episode,
        actions=object(),
        action_valid=object(),
        execution=episode.execution[:4]
        + [
            replace(frame, points=frame.points * 1000, proprio=frame.proprio * -999)
            for frame in episode.execution[4:]
        ],
    )
    assert build_structure(altered, encoder, 3) == payload
    assert {name: module.training for name, module in encoder.named_modules()} == modes
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    for parameter, grad in zip(encoder.parameters(), grads):
        assert torch.equal(parameter.grad, grad)
    serialized = json.dumps(payload)
    for forbidden in ("target_chunk", "target_valid", "content_sum", "max_content", "previous_action"):
        assert forbidden not in serialized
    for block in payload["blocks"]:
        if not block["detailed"]:
            assert block["details"] is None
            continue
        detail = block["details"]
        assert len(detail["points"]) == sum(frame["valid_points"] for frame in block["frames"])
        for event in detail["events"]:
            assert event["anchor"] == event["proprio"][7:10]
            if block["source"] == "conditioning":
                assert not event["action_available"]


def test_no_history_and_zero_detail_blocks_still_show_current_and_all_invalid_events():
    episode, encoder = _episode(), _encoder()
    episode.conditioning = [
        replace(
            frame,
            points=torch.full_like(frame.points, float("nan")),
            valid=torch.zeros_like(frame.valid),
        )
        for frame in episode.conditioning
    ]
    payload = build_structure(episode, encoder, 0, detail_blocks=0)
    assert all(node["modality"] == "event" for node in payload["archive"]["nodes"])
    assert all(not block["detailed"] for block in payload["blocks"][:-1])
    assert payload["blocks"][-1]["detailed"]
    assert all(block["source"] == "conditioning" for block in payload["blocks"][:-1])
    for block in payload["blocks"][:-1]:
        assert block["local_roots"]["visual"] is None
    empty = build_structure(replace(episode, conditioning=[]), encoder, 0)
    assert empty["archive"] == {"roots": [], "leaf_ids": [], "nodes": []}
    assert len(empty["blocks"]) == 1


@pytest.mark.parametrize("index,detail_blocks", [(-1, 2), (8, 2), (True, 2), (0, -1), (0, 1.5), (0, True)])
def test_invalid_inspection_requests_are_rejected(index, detail_blocks):
    with pytest.raises(ValueError):
        build_structure(_episode(), _encoder(), index, detail_blocks=detail_blocks)


def test_current_cannot_duplicate_history_or_go_backwards_in_source_time():
    episode = _episode()
    episode.execution[2] = replace(episode.execution[2], event_id=episode.execution[1].event_id)
    with pytest.raises(ValueError, match="also occurs"):
        build_structure(episode, _encoder(), 2)
    episode.execution[2] = replace(episode.execution[2], event_id="unique", time=0.0)
    with pytest.raises(ValueError, match="clock"):
        build_structure(episode, _encoder(), 2)


def test_pending_block_replacement_matches_shared_archive_at_completion_boundary():
    episode, encoder = _episode(), _encoder()
    prefix_ids = set()
    for index in (3, 4, 5, 6):
        payload = build_structure(episode, encoder, index, detail_blocks=1)
        with torch.no_grad():
            archive = build_archive(episode.conditioning, episode.execution[:index], encoder)
        assert payload["archive"]["leaf_ids"] == [record.node_id for record in archive.leaves]
        nodes = payload["archive"]["nodes"]
        pending = {node["id"] for node in nodes if not node["completed"] and node["kind"] == "leaf"}
        assert bool(pending) == bool(index % encoder.block_size)
        if not index % encoder.block_size:
            assert prefix_ids.isdisjoint(payload["archive"]["leaf_ids"])
        prefix_ids = pending


def test_all_invalid_detailed_centers_have_no_invented_points_or_neighbors():
    episode, encoder = _episode(), _encoder()
    frame = episode.execution[0]
    episode.execution[0] = replace(frame, valid=torch.zeros_like(frame.valid))
    payload = build_structure(episode, encoder, 0)
    details = payload["blocks"][-1]["details"]
    assert details["points"] == []
    assert len(details["centers"]) == encoder.current_tokens
    for center in details["centers"]:
        assert not center["valid"] and not center["candidate_valid"]
        assert center["point_index"] is None and center["leaf_id"] is None
        assert center["neighbor_indices"] == center["neighbor_mask"] == center["weights"] == []
    json.dumps(payload, allow_nan=False)


def test_nonfinite_serialized_observation_is_explicitly_rejected():
    episode = _episode()
    episode.execution[0].proprio[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        build_structure(episode, _encoder(), 0)
