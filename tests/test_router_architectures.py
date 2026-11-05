"""Architecture ablations retain causal inputs and exact cached scoring."""

import pytest
import torch

from stmem.memory.frontier import select_tree, validate_frontier
from stmem.memory.hierarchy import local_tree
from stmem.memory.records import aggregate, leaf_record
from stmem.models.router import CachedScorer, LocalRouter, build_router, current_token_features
from stmem.training.runtime import load_router


VARIANTS = ("dot", "mlp", "children", "current_attention", "children_current_attention")


def config(variant):
    return {"reader": "tree", "model": {
        "width": 16, "router_width": 8, "router_architecture": variant,
        "router_attention_heads": 2, "router_query_tokens": 3, "router_attention_layers": 2,
    }}


def records():
    generator = torch.Generator().manual_seed(71)
    leaves = [leaf_record(str(i), torch.randn(16, generator=generator),
                          torch.tensor([i / 10, 0.0, 0.0]), float(i), "execution", "visual")
              for i in range(8)]
    root = local_tree(leaves)

    def walk(record):
        return [record] + [node for child in record.children for node in walk(child)]

    return leaves, root, walk(root)


def test_default_factory_preserves_original_checkpoint_and_score():
    torch.manual_seed(9)
    legacy = LocalRouter(16, 8)
    cfg = config("dot")
    del cfg["model"]["router_architecture"]
    factory = build_router(cfg)
    factory.load_state_dict(legacy.state_dict(), strict=True)
    query, candidate = torch.randn(5, 59), torch.randn(5, 55)
    expected = ((legacy.query(query) * legacy.key(candidate)).sum(-1) / 8 ** .5
                + legacy.bias(candidate[:, -23:]).squeeze(-1))
    torch.testing.assert_close(factory(query, candidate), expected, rtol=0, atol=0)
    assert set(factory.state_dict()) == set(legacy.state_dict())


@pytest.mark.parametrize("variant", VARIANTS)
def test_cached_and_batched_scores_match_and_traversal_uses_no_towers(variant):
    torch.manual_seed(5)
    router = build_router(config(variant)).eval().requires_grad_(False)
    leaves, root, archive = records()
    parents = [record for record in archive if record.children]
    current = current_token_features(leaves[:3], {"execution": 8}, width=16, device="cpu")
    query = torch.randn(59)
    kwargs = {"current": current} if router.requires_current else {}
    children = router.child_input(parents) if router.requires_children else None
    expected = router(query, router.candidate_input(parents), children=children, **kwargs)
    paired_kwargs = {"current": current[None].expand(len(parents), -1, -1)} if router.requires_current else {}
    paired = router(query[None].expand(len(parents), -1), router.candidate_input(parents),
                    children=children, **paired_kwargs)
    torch.testing.assert_close(paired, expected, atol=3e-7, rtol=3e-6)
    cached = CachedScorer(router, "synthetic")
    cached.prepare(archive, chunk_size=3)
    cached.set_query(query, **kwargs)
    torch.testing.assert_close(torch.tensor(cached(parents)), expected, atol=3e-7, rtol=3e-6)
    encoded = cached.keys_encoded
    cached.prepare(archive)
    assert cached.keys_encoded == encoded
    assert encoded == (len(parents) if router.requires_children else len(archive))

    def forbidden(*args, **kwargs):
        raise AssertionError("Traversal must only score cached projections")

    router.encode_query = router.encode_candidates = forbidden
    selected = select_tree([root], 5, cached)
    validate_frontier(selected.records, leaves, 5)
    assert len(selected.records) == 5
    assert cached.storage_bytes > 0


@pytest.mark.parametrize("variant", ["current_attention", "children_current_attention"])
def test_current_attention_ignores_padding_and_mean_but_uses_geometry_and_task(variant):
    torch.manual_seed(27)
    router = build_router(config(variant)).eval()
    query = torch.randn(2, 59)
    current = torch.randn(2, 4, 39)
    valid = torch.tensor([[True, True, False, False], [False, False, False, False]])
    actual = router.encode_query(query, current=current, current_valid=valid)
    perm = torch.tensor([2, 0, 3, 1])
    permuted = router.encode_query(query, current=current[:, perm], current_valid=valid[:, perm])
    torch.testing.assert_close(actual, permuted, atol=2e-6, rtol=2e-6)
    altered = current.clone()
    altered[~valid] = float("nan")
    torch.testing.assert_close(actual, router.encode_query(query, current=altered, current_valid=valid))
    query_without_mean = query.clone()
    query_without_mean[:, :16] += 10000
    torch.testing.assert_close(actual, router.encode_query(query_without_mean, current=current, current_valid=valid))
    moved = current.clone()
    moved[0, 0, 16:19] += 5  # explicit current anchor
    assert not torch.allclose(actual[0], router.encode_query(query, current=moved, current_valid=valid)[0])
    task = query.clone()
    task[:, 16:32] *= -1
    assert not torch.allclose(actual, router.encode_query(task, current=current, current_valid=valid))
    empty = router.encode_query(query, current=torch.empty(2, 0, 39), current_valid=torch.empty(2, 0, dtype=torch.bool))
    assert torch.isfinite(empty).all()


def test_children_preserve_information_lost_by_parent_and_ignore_child_order():
    torch.manual_seed(82)
    router = build_router(config("children"))

    def parent(vectors, name):
        leaves = [leaf_record(f"{name}_{i}", torch.tensor(v + [0.] * 14),
                              torch.tensor([float(i), 0., 0.]), float(i), "execution", "visual")
                  for i, v in enumerate(vectors)]
        return aggregate(leaves, node_id=name, children=tuple(leaves))

    left = parent([[1., 1.], [-1., -1.]], "left")
    right = parent([[1., -1.], [-1., 1.]], "right")
    candidates = router.candidate_input([left, right])
    torch.testing.assert_close(candidates[0], candidates[1], rtol=0, atol=0)
    children = router.child_input([left, right])
    keys, _ = router.encode_candidates(candidates, children=children)
    swapped, _ = router.encode_candidates(candidates, children=children.flip(1))
    torch.testing.assert_close(keys, swapped, rtol=0, atol=0)
    assert not torch.allclose(keys[0], keys[1], atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_variants_receive_gradients_and_reload(tmp_path, variant):
    torch.manual_seed(18)
    cfg = config(variant)
    router = build_router(cfg)
    inputs = {"query_input": torch.randn(4, 59), "candidate_input": torch.randn(4, 55)}
    if router.requires_children:
        inputs["children"] = torch.randn(4, 2, 55)
    if router.requires_current:
        inputs["current"] = torch.randn(4, 3, 39)
    prediction = router(**inputs)
    prediction.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in router.parameters())
    assert router.score_head[0].weight.grad.abs().sum() > 0
    if router.requires_children:
        assert router.child_encoder[0].weight.grad.abs().sum() > 0
    if router.requires_current:
        assert router.current_projection[0].weight.grad.abs().sum() > 0
    path = tmp_path / "router.pt"
    torch.save({"config": cfg, "model": router.state_dict()}, path)
    loaded, _ = load_router(path)
    # MHA may choose different eval/training kernels; checkpoint tensors remain
    # exact, while floating-point forward parity allows their rounding error.
    for name, value in router.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(prediction, loaded(**inputs), rtol=2e-6, atol=2e-7)


def test_child_architecture_cannot_be_mislabeled_as_flat():
    cfg = config("children")
    cfg["reader"] = "flat"
    with pytest.raises(ValueError, match="TREE"):
        build_router(cfg)
