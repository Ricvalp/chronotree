"""Synthetic architecture fits exercise optimization, resume, and row grouping.

Rich feature construction/provenance is tested separately in test_router_inputs;
these fixtures isolate the trainer with explicit deterministic synthetic inputs.
"""

from copy import deepcopy

import numpy as np
import pytest
import torch

from stmem.config import run_dir
from stmem.models.router import build_router
from stmem.training import router_inputs, routers
from test_router_diagnostics import _assert_rng_equal
from test_router_objectives import _config, _write_shards
from test_tracking_routers import RecordingTracker, _assert_identical_optimization


VARIANTS = ("dot", "mlp", "children", "current_attention", "children_current_attention")


def synthetic_inputs(cfg, paths, merged):
    generator = np.random.default_rng(150)
    contexts, inverse = np.unique(merged["context_id"], return_inverse=True)
    width = cfg["model"]["width"]
    current = generator.normal(size=(len(contexts), 3, width + 23)).astype("f4")
    valid = np.ones(current.shape[:2], dtype=bool)
    valid[::2, -1] = False
    valid[-1] = False
    return {
        "current": current, "current_valid": valid, "row_to_context": inverse,
        "children": generator.normal(size=(len(merged["gain"]), 2, 2 * width + 23)).astype("f4"),
    }, {"schema_version": "synthetic-test", "identity": "fixed-causal-input-fixture"}


@pytest.fixture
def fixture_tracker(monkeypatch):
    RecordingTracker.instances = []
    monkeypatch.setattr(routers, "RunTracker", RecordingTracker)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(routers, "environment_manifest", lambda: {"device": "cpu-synthetic-test"})
    monkeypatch.setattr(router_inputs, "load_router_inputs", synthetic_inputs)
    return RecordingTracker.instances


def experiment(path, variant):
    cfg = _config(path, "final_action")
    cfg["model"].update(router_architecture=variant, router_attention_heads=2,
                        router_query_tokens=3, router_attention_layers=1)
    cfg["training"].update(router_diagnostics=True, steps=4, checkpoint_steps=[4],
                           router_fit_contexts=2, log_every=2)
    return cfg


def checkpoint(cfg):
    return torch.load(run_dir(cfg, "router") / "last.pt", map_location="cpu", weights_only=False)


@pytest.mark.parametrize("variant", VARIANTS)
def test_architecture_training_and_resume_are_exact(tmp_path, fixture_tracker, variant):
    cfg = experiment(tmp_path / "resumed", variant)
    _write_shards(cfg)
    stop = tmp_path / "stop"
    stop.touch()
    assert routers.train_router(cfg, stop_file=stop) == 75
    stop.unlink()
    source = run_dir(cfg, "router") / "last.pt"
    assert checkpoint(cfg)["step"] == 1
    assert routers.train_router(cfg, resume=source) == 0
    actual = checkpoint(cfg)
    assert actual["step"] == 4
    assert actual["fit_ids_sha256"]
    row = fixture_tracker[-1].logs[-1][1]
    assert row["train_labels"] == 4
    assert row["validation_labels"] == 8
    assert np.isfinite(row["training_full_gain_loss"])
    assert np.isfinite(row["conditional_label_validation_loss"])
    for prefix in ("training", "validation"):
        assert row[prefix + "_top_choice_label_regret"] >= -1e-7
        assert row[prefix + "_selected_label_gain"] <= row[prefix + "_best_candidate_label_gain"] + 1e-7
    metadata = fixture_tracker[-1].options["metadata"]
    assert metadata["router_architecture"] == variant
    assert metadata["router_parameters"] == sum(p.numel() for p in build_router(cfg).parameters())
    assert metadata["available_train_labels"] == 8
    if variant in ("dot", "mlp"):
        assert actual["router_input_identity"] is None
    else:
        assert actual["router_input_identity"]["identity"] == "fixed-causal-input-fixture"

    control = deepcopy(cfg)
    control["paths"]["run_root"] = str(tmp_path / "uninterrupted")
    assert routers.train_router(control) == 0
    expected = checkpoint(control)
    _assert_identical_optimization(actual, expected)
    _assert_rng_equal(actual, expected)


@pytest.mark.parametrize("variant", ["current_attention", "children_current_attention"])
def test_deduplicated_queries_match_rowwise_forward_and_gradients(tmp_path, variant):
    cfg = experiment(tmp_path, variant)
    _write_shards(cfg)
    merged, paths, *_ = routers._load_labels(cfg)
    arrays, _ = synthetic_inputs(cfg, paths, merged)
    query = torch.from_numpy(merged["query"])
    candidate = torch.from_numpy(merged["candidate"])
    mapping = torch.as_tensor(arrays["row_to_context"], dtype=torch.long)
    _, first = np.unique(arrays["row_to_context"], return_index=True)
    inputs = {key: torch.from_numpy(value) for key, value in arrays.items()}
    inputs["context_query"] = query[first]
    torch.manual_seed(39)
    deduplicated = build_router(cfg)
    rowwise = deepcopy(deduplicated)
    ids = torch.tensor([3, 2, 3, 0, 1, 7, 6])
    prediction = routers._router_prediction(deduplicated, query, candidate, ids, inputs)
    direct = rowwise(query[ids], candidate[ids],
                     children=inputs["children"][ids] if rowwise.requires_children else None,
                     current=inputs["current"][mapping[ids]],
                     current_valid=inputs["current_valid"][mapping[ids]])
    torch.testing.assert_close(prediction, direct, rtol=3e-6, atol=3e-7)
    weights = torch.tensor([1., -.2, .5, .8, -.3, 1., .2])
    (prediction * weights).sum().backward()
    (direct * weights).sum().backward()
    for (name, left), (right_name, right) in zip(deduplicated.named_parameters(), rowwise.named_parameters()):
        assert name == right_name
        assert left.grad is not None and right.grad is not None, name
        torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=3e-6, msg=name)


def test_training_subset_is_fixed_train_only_and_preserves_validation(tmp_path, fixture_tracker):
    cfg = experiment(tmp_path, "mlp")
    _write_shards(cfg)
    merged, _, _, contexts, _ = routers._load_labels(cfg)
    train = np.flatnonzero(merged["partition"] == 0)
    validation = np.flatnonzero(merged["partition"] == 1)
    rng = np.random.get_state()
    first = routers._fit_training_ids(cfg, contexts, train)
    second = routers._fit_training_ids(cfg, contexts, train[::-1])
    np.testing.assert_array_equal(first, second)
    after = np.random.get_state()
    np.testing.assert_array_equal(rng[1], after[1])
    assert rng[2:] == after[2:]
    assert len(first) == 4 and len(set(merged["context_id"][first])) == 2
    assert not set(first).intersection(validation)
    for context in np.unique(merged["context_id"][first]):
        assert set(np.flatnonzero(merged["context_id"] == context)).issubset(first)
    assert routers.train_router(cfg) == 0
    source = run_dir(cfg, "router") / "last.pt"
    assert fixture_tracker[-1].logs[-1][1]["validation_labels"] == len(validation)
    changed = deepcopy(cfg)
    changed["training"].update(router_fit_contexts=3, steps=5)
    with pytest.raises(ValueError, match="subset changed"):
        routers.train_router(changed, resume=source)

    changed = deepcopy(cfg)
    changed["training"]["steps"] = 5
    for seed in range(100):
        changed["seed"] = seed
        if not np.array_equal(first, routers._fit_training_ids(changed, contexts, train)):
            break
    with pytest.raises(ValueError, match="IDs changed"):
        routers.train_router(changed, resume=source)


def test_enriched_input_identity_cannot_change_on_resume(tmp_path, fixture_tracker, monkeypatch):
    cfg = experiment(tmp_path, "children_current_attention")
    _write_shards(cfg)
    assert routers.train_router(cfg) == 0
    source = run_dir(cfg, "router") / "last.pt"

    def changed(*args):
        arrays, ident = synthetic_inputs(*args)
        return arrays, {**ident, "identity": "different-inputs"}

    monkeypatch.setattr(router_inputs, "load_router_inputs", changed)
    cfg["training"]["steps"] = 5
    with pytest.raises(ValueError, match="richer input identity changed"):
        routers.train_router(cfg, resume=source)


def test_top_choice_regret_uses_tie_expectation_and_valid_candidates():
    contexts = [np.array([0, 0, 0, 1, 1])]
    groups = routers._context_groups(contexts, np.arange(5))
    scores = torch.tensor([2., 2., 0., 4., 0.])
    gain = torch.tensor([1., 3., 5., 2., 10.])
    metrics = routers._choice_metrics(scores, gain, groups, 2., "validation")
    assert metrics["validation_selected_label_gain"] == 2
    assert metrics["validation_best_candidate_label_gain"] == 7.5
    assert metrics["validation_random_candidate_label_gain"] == 4.5
    assert metrics["validation_top_choice_label_regret"] == 5.5
    assert metrics["validation_top_choice_label_regret_scaled"] == 2.75
    tied = routers._choice_metrics(torch.zeros(5), gain, groups, 2., "validation")
    assert tied["validation_selected_label_gain"] == tied["validation_random_candidate_label_gain"]
    oracle = routers._choice_metrics(gain, gain, groups, 2., "validation")
    assert oracle["validation_top_choice_label_regret"] == 0


def test_choice_groups_do_not_merge_refresh_contexts_or_singletons():
    groups, valid = routers._context_groups([np.array([10, 10, 11]), np.array([10, 10])], np.arange(5))
    torch.testing.assert_close(groups, torch.tensor([[0, 1], [3, 4]]))
    assert valid.all()
