"""Synthetic CPU diagnostics for signed gains, tail selection and provenance."""
from copy import deepcopy
import csv
import gzip
import json

import numpy as np
import pytest

from stmem.config import digest
from stmem.inspection.refinement_measurements import draw_plan
from stmem.inspection.utility_gain_distribution import (
    _loss_arrays, collect, concentration, distribution, plots, repeatability, run, summarize,
)
from stmem.memory.utility_routing import feature_spec
from stmem.training.checkpoints import atomic_json
from stmem.training.utility_router_data import (
    SCHEMA, label_identity, labels_identity, measured_arrays, publish_identity, write_state,
)


def synthetic_dataset(root, *, leak=False):
    states = [dict(state_id=f"s{i}", task="SyntheticTask", episode_id=f"episode_{0 if leak else i}",
                   episode_index=i, decision_index=8, partition=part)
              for i, part in enumerate(("fit", "development"))]
    protocol = {"packet_max_cost": 8, "candidate_limit": 4}
    study = {"experiment_id": "synthetic-diagnostic", "source": {"source_sha256": "synthetic"},
             "states": states, "student": {"sha256": "a"*64, "step": 95000},
             "configuration": {"model": {"width": 4}}, "protocol": protocol}
    identity = label_identity(study, 2)
    publish_identity(root, identity)
    for worker, state in enumerate(states):
        draws = draw_plan(19, state["task"], state["episode_id"], 8, discovery=2, confirmation=2)
        contexts, arrays = [], {}
        for reader in ("TREE", "FLAT"):
            panel = {"reader": reader, "context_id": reader+state["state_id"], "base": "base", "packet_cost": 4,
                "background_count": 8, "prefix_stratum": 0, "controls": {"uniform": "uniform"},
                "candidates": [{"candidate_id": f"c{k}", "cost": 4, "first": f"f{k}", "packet": f"p{k}"}
                               for k in range(4)]}
            gains = [0., .2, -.4, .001]
            rows = []
            for j, draw in enumerate(draws):
                scale = [1., 1.1, .7, .9][j]
                for name, gain in [("base", 0.), ("uniform", .02),
                        *[(f"f{k}", g) for k, g in enumerate(gains)],
                        *[(f"p{k}", g*2) for k, g in enumerate(gains)]]:
                    rows.append({"context_id": name, **draw, "expert_full": 2.-scale*gain,
                                 "expert_prefix": 3.-scale*gain*.5, "teacher_full": 1.})
            values = measured_arrays(rows, panel, draws)
            spec = feature_spec(4)
            values["query"] = np.zeros(spec["query_dim"], np.float32)
            values["candidates"] = np.zeros((4, spec["candidate_dim"]), np.float32)
            prefix = reader + "_"
            arrays.update({prefix + k: v for k, v in values.items()})
            contexts.append({**panel, "array_prefix": prefix})
        record = write_state(root, state, identity, contexts, arrays, exclusions=[], timings={}, draws=draws)
        atomic_json({"schema": SCHEMA, "identity_sha256": digest(identity), "worker_index": worker,
            "planned_state_ids": [state["state_id"]], "completed_state_ids": [state["state_id"]],
            "states": {state["state_id"]: record}, "status": "complete"}, root / f"worker_{worker}.json")
    return identity


def test_raw_sign_units_partitions_and_all_candidate_rows(tmp_path):
    root = tmp_path / "labels"
    synthetic_dataset(root)
    bound, groups, excluded = collect(root, tmp_path / "candidates.csv.gz")
    assert excluded == {} and len(groups) == 4
    views, payloads = summarize(groups, 1e-4)
    assert len(views) == len(payloads) == 32
    total = views["fit/TREE/full/packet/total"]
    unit = views["fit/TREE/full/packet/per_token"]
    assert total["counts"] == {"candidates": 4, "contexts": 1, "states": 1, "episodes": 1, "tasks": 1}
    assert total["confirmation"]["mean"] == pytest.approx(unit["confirmation"]["mean"] * 4)
    assert total["confirmation"]["positive_fraction"] == .5
    assert total["confirmation"]["negative_fraction"] == .25
    assert total["confirmation"]["exact_zero_fraction"] == .25
    assert total["actual_router_target"] and not unit["actual_router_target"]
    # Prefix uses the stored prefix mask/reduction, not a slice of full loss.
    first = views["fit/TREE/full/first/total"]["confirmation"]["mean"]
    prefix = views["fit/TREE/prefix/first/total"]["confirmation"]["mean"]
    assert prefix == pytest.approx(first*.5)
    assert views["fit/TREE/full/first/per_token"]["confirmation"] == views["fit/TREE/full/first/total"]["confirmation"]
    with gzip.open(tmp_path / "candidates.csv.gz", "rt") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16  # One wide row per candidate, not eight rows per draw.
    assert {row["partition"] for row in rows} == {"fit", "development"}
    assert bound["dataset_sha256"]


def test_discovery_tail_is_not_selected_using_confirmation_and_handles_ties():
    discovery = np.array([10., 1., 1., -9., 0.])
    confirmation = np.array([-2., 100., 100., 2., 0.])
    result = repeatability(discovery, confirmation, 1e-4)
    top = result["tails"]["positive"]["0.1"]
    assert top["selected_count"] == 1 and top["discovery_mean"] == 10.
    assert top["confirmation_mean"] == -2. and top["confirmation_positive_fraction"] == 0
    assert result["tails"]["negative"]["0.1"]["confirmation_mean"] == 2.
    # Every candidate has equal weight despite repeated values.
    assert distribution(np.array([0., 1., 1.]), 0)["positive_fraction"] == pytest.approx(2/3)
    assert concentration(np.array([0., 1., 1.]))["absolute"]["0.1"]["share"] == .5


@pytest.mark.parametrize("values", [np.array([]), np.zeros(5), -np.ones(5), np.ones(5)])
def test_empty_zero_constant_and_negative_only_are_finite_and_honest(values):
    summary = distribution(values, 0)
    tails = repeatability(values, values, 0)
    result = {"summary": summary, "tails": tails, "concentration": concentration(values), "plots": plots(values, values, 0)}
    json.dumps(result, allow_nan=False)
    for panel in result["plots"]:
        assert panel["layout"]["title"]["text"]
        assert panel["layout"]["xaxis"]["title"]["text"]
        assert panel["layout"]["yaxis"]["title"]["text"]
    assert tails["pearson"] is None
    if not np.maximum(values, 0).sum():
        assert result["concentration"]["positive"]["0.01"]["share"] is None
    if len(values) and np.all(values < 0):
        assert summary["negative_fraction"] == 1
        assert tails["tails"]["positive"]["0.1"]["selected_count"] == 0


def test_end_to_end_cpu_report_renders_without_network_and_preserves_labels(tmp_path):
    root, output = tmp_path / "labels", tmp_path / "report"
    synthetic_dataset(root)
    before = labels_identity(root)["dataset_sha256"]
    result = run(root, output)
    assert result["status"] == "complete"
    assert labels_identity(root)["dataset_sha256"] == before
    summary = json.loads((output / "summary.json").read_text())
    assert summary["study_id"] == "synthetic-diagnostic" and summary["dataset_sha256"] == before
    assert not summary["independent_candidate_trials"]
    rendered = (output / "index.html").read_text()
    assert "Plotly.react" in rendered and '<script src="' not in rendered
    assert "does not establish a power law" in rendered
    assert "confirmation" in rendered and "Per-token" in rendered
    assert "synthetic-diagnostic" in (output / "summary.json").read_text()
    receipt = json.loads((output / "complete.json").read_text())
    assert set(receipt["artifacts"]) == {"index.html", "report.md", "summary.json", "candidates.csv.gz"}
    with pytest.raises(FileExistsError):
        run(root, output)


def test_corruption_missing_receipts_and_episode_leakage_cannot_publish(tmp_path):
    root = tmp_path / "labels"
    synthetic_dataset(root)
    with (root / "states/s0.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        run(root, tmp_path / "invalid")
    assert not (tmp_path / "invalid/complete.json").exists()
    other = tmp_path / "missing"
    synthetic_dataset(other)
    (other / "worker_1.json").unlink()
    with pytest.raises(FileNotFoundError):
        run(other, tmp_path / "missing_report")
    leak = tmp_path / "leak"
    synthetic_dataset(leak, leak=True)
    with pytest.raises(ValueError, match="Episode leakage"):
        run(leak, tmp_path / "leak_report")
    assert not (tmp_path / "leak_report/complete.json").exists()


def test_noise_banks_must_match_raw_draws(tmp_path):
    from stmem.training.utility_router_data import read_state
    root = tmp_path / "labels"
    identity = synthetic_dataset(root)
    metadata, arrays = read_state(root, identity["states"][0], identity)
    context = metadata["contexts"][0]
    changed = deepcopy(metadata)
    changed["draws"][-1]["seed"] = changed["draws"][0]["seed"]
    with pytest.raises(ValueError, match="draw seeds"):
        _loss_arrays(context, arrays, changed)
    changed_arrays = deepcopy(arrays)
    changed_arrays[context["array_prefix"]+"draw_seeds"][-1] = changed["draws"][0]["seed"]
    with pytest.raises(ValueError, match="overlap"):
        _loss_arrays(context, changed_arrays, changed)
    changed = deepcopy(arrays)
    changed[context["array_prefix"]+"packet_prefix_loss"][0, 0] = np.nan
    with pytest.raises(ValueError, match="candidate losses"):
        _loss_arrays(context, changed, metadata)


def test_equal_state_and_task_weighting_do_not_duplicate_candidate_weights(tmp_path):
    root = tmp_path / "labels"
    synthetic_dataset(root)
    _, groups, _ = collect(root, tmp_path / "rows.csv.gz")
    group = groups["fit", "TREE"]
    original_mean = group["arrays"]["full_packet_confirmation"].mean()
    group["arrays"] = {key: np.concatenate((values, values[:1])) for key, values in group["arrays"].items()}
    group["state_rows"] = np.append(group["state_rows"], "another_state")
    group["task_rows"] = np.append(group["task_rows"], "another_task")
    group["tasks"].add("another_task")
    views, _ = summarize(groups, 1e-4)
    row = views["fit/TREE/full/packet/total"]
    assert row["confirmation"]["near_zero_fraction"] == .4
    for weighting in ("equal_task", "equal_state"):
        values = row["weighting_sensitivity"][weighting]["confirmation"]
        assert values["near_zero_fraction"] == .625
        assert values["mean"] == pytest.approx(original_mean/2)
    assert row["baseline_raw_expert_loss"]["confirmation"]["mean"] == 2.
