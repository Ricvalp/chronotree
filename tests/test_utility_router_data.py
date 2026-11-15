"""Synthetic CPU checks for fresh expert utility labels and exact shard receipts."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from test_cached_context_loader import fixture  # noqa: F401 -- synthetic shared fixture
from test_packed_context import indexed  # noqa: F401
from test_label_objectives import policies

from test_refinement_candidates import forest
from stmem.config import digest
from stmem.inspection.refinement_candidates import IndexedArchive
from stmem.inspection.refinement_measurements import draw_plan
from stmem.memory.utility_routing import feature_spec
from stmem.training.checkpoints import atomic_json
from stmem.training.utility_router_data import (
    SCHEMA, label_identity, labels_identity, load_label_rows, measured_arrays,
    publish_identity, reachable_panels, write_state,
)


def protocol():
    return {"seed": 17, "budget": 32, "packet_max_cost": 8, "candidate_limit": 4,
            "route_prefixes_per_state": 2, "discovery_draws": 2, "confirmation_draws": 2}


def state(name="a", partition="fit"):
    return {"state_id": name, "task": "TaskA", "episode_id": "episode_0",
            "episode_index": 0, "decision_index": 4, "partition": partition}


def archive():
    index, roots, leaves = forest((64, 48), geometry=True)
    return IndexedArchive(index, roots, leaves)


def test_reachable_panels_are_early_and_late_with_exact_matched_contexts():
    history = archive()
    result = reachable_panels(history, state(), protocol())
    assert result["exclusions"] == []
    assert len(result["panels"]) == 4
    for reader in ("TREE", "FLAT"):
        rows = [row for row in result["panels"] if row["reader"] == reader]
        assert rows[0]["background_count"] < 17 <= rows[1]["background_count"]
        for row in rows:
            assert len(row["route_prefix"]) == row["background_count"] - row["initial_count"]
            counts = {result["contexts"][c["packet"]]["token_count"] for c in row["candidates"]}
            counts |= {result["contexts"][key]["token_count"] for key in row["controls"].values()}
            assert counts == {row["final_count"]}
    for context in result["contexts"].values():
        assert history.validate(context) == context
    # Future expert actions/noise do not participate in construction.
    altered = dict(state(), target=[[999.]], noise=[[42.]], eventual_duration=100000)
    again = reachable_panels(history, altered, protocol())
    for a, b in zip(result["panels"], again["panels"]):
        assert a["base"] == b["base"] and a["candidates"] == b["candidates"]


def raw_rows(panel, draws):
    contexts = {panel["base"], *panel["controls"].values()}
    contexts.update(candidate[key] for candidate in panel["candidates"] for key in ("first", "packet"))
    values = {key: 1.0 + i / 100 for i, key in enumerate(sorted(contexts))}
    return [{"context_id": key, **draw, "expert_full": values[key],
             "expert_prefix": values[key] * .7, "teacher_full": 3. - values[key]}
            for key in contexts for draw in draws]


def test_targets_use_expert_loss_and_confirmation_cannot_change_training_targets():
    panel = reachable_panels(archive(), state(), protocol())["panels"][0]
    draws = draw_plan(17, "TaskA", "episode_0", 4, discovery=2, confirmation=2)
    rows = raw_rows(panel, draws)
    arrays = measured_arrays(rows, panel, draws)
    expected = (arrays["background_loss"][None] - arrays["packet_loss"]).mean(1)
    np.testing.assert_array_equal(arrays["discovery_packet_gain"], expected)
    teacher = (arrays["background_teacher_loss"][None] - arrays["packet_teacher_loss"]).mean(1)
    np.testing.assert_allclose(expected, -teacher)
    changed = deepcopy(rows)
    for row in changed:
        if row["bank"] == "confirmation":
            row["expert_full"] = 1000 - row["expert_full"]
    changed_arrays = measured_arrays(changed, panel, draws)
    np.testing.assert_array_equal(changed_arrays["discovery_packet_gain"], arrays["discovery_packet_gain"])
    assert not np.array_equal(changed_arrays["confirmation_packet_gain"], arrays["confirmation_packet_gain"])


def make_dataset(tmp_path):
    states = [state("first"), state("second", "development")]
    study = {"experiment_id": "study", "source": {"source_sha256": "s"}, "states": states,
             "student": {"sha256": "x" * 64, "step": 95000},
             "configuration": {"model": {"width": 4}}, "protocol": protocol()}
    identity = label_identity(study, 2)
    publish_identity(tmp_path, identity)
    for worker, entry in enumerate(states):
        panel = reachable_panels(archive(), entry, protocol())["panels"][0]
        draws = draw_plan(17, entry["task"], entry["episode_id"], 4, discovery=2, confirmation=2)
        values = measured_arrays(raw_rows(panel, draws), panel, draws)
        spec = feature_spec(4)
        values["query"] = np.zeros(spec["query_dim"], np.float32)
        values["candidates"] = np.zeros((len(panel["candidates"]), spec["candidate_dim"]), np.float32)
        arrays = {"c0_" + key: value for key, value in values.items()}
        record = write_state(tmp_path, entry, identity, [dict(panel, array_prefix="c0_")], arrays,
                             exclusions=[], timings={}, draws=draws)
        receipt = {"schema": SCHEMA, "identity_sha256": digest(identity), "worker_index": worker,
                   "planned_state_ids": [entry["state_id"]], "completed_state_ids": [entry["state_id"]],
                   "states": {entry["state_id"]: record}, "status": "complete"}
        atomic_json(receipt, tmp_path / f"worker_{worker}.json")
    return identity


def test_complete_receipts_separate_partitions_and_preserve_candidate_arrays(tmp_path):
    identity = make_dataset(tmp_path)
    bound = labels_identity(tmp_path)
    assert bound["identity"] == identity
    fitting = load_label_rows(tmp_path, "TREE", "fit")
    development = load_label_rows(tmp_path, "TREE", "development")
    assert len(fitting) == len(development) == 1
    assert fitting[0]["candidates"].ndim == 2
    assert fitting[0]["candidate_records"][0]["cost"] == fitting[0]["packet_cost"]
    assert fitting[0]["partition"] == "fit" and development[0]["partition"] == "development"
    (tmp_path / "worker_1.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_label_rows(tmp_path, "TREE", "fit")


def test_changed_payload_and_reassigned_shards_cannot_be_reused(tmp_path):
    identity = make_dataset(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        publish_identity(tmp_path, {**identity, "workers": 3})
    with (tmp_path / "states/first.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        labels_identity(tmp_path)


def test_complete_cpu_label_generation_reuses_actual_cache_and_flow_targets(indexed, tmp_path, monkeypatch):  # noqa: F811
    """Exercise model inference, current/history hydration, both readers and resume."""
    from stmem.training import utility_router_data as module
    from stmem.training import utility_router_study as preparation
    from stmem.training.utility_routers import build_table, evaluate, train

    cfg, data, teacher, cache, hierarchy = indexed
    teacher, student = policies(teacher)
    cfg = deepcopy(cfg)
    cfg["tracking"] = {"enabled": False, "mode": "disabled"}
    spec = {**protocol(), "budget": 16, "packet_max_cost": 4,
            "workers": 1, "execute_targets": 1, "precision": "fp32", "key_width": 8,
            "fit": {"steps": 1, "batch": 4, "checkpoint_every": 1, "log_every": 1, "eval_every": 1}}
    states = [dict(state("first"), episode_index=data.by_task["TaskA"][0], decision_index=2),
              dict(state("second", "development"), task="TaskB",
                   episode_index=data.by_task["TaskB"][0], decision_index=2)]
    study = {"experiment_id": "synthetic-diagnostic", "source": {"source_sha256": "synthetic"},
             "states": states, "configuration": cfg, "protocol": spec, "data_identity": data.identity,
             "student": {"name": "S_STD", "checkpoint": "student", "sha256": "b" * 64, "step": 95000},
             "dependencies": {"teacher": "teacher", "teacher_sha256": "a" * 64,
                              "encoded": str(cache.root), "hierarchy": str(hierarchy)}}
    monkeypatch.setattr(preparation, "verify_study", lambda _path: study)
    monkeypatch.setattr(module, "TrainingData", lambda _cfg: data)
    monkeypatch.setattr(module, "load_policy", lambda path, _device:
                        (teacher if path == "teacher" else student,
                         {"stage": "teacher" if path == "teacher" else "compact", "config": cfg}))
    args = SimpleNamespace(study=tmp_path / "study.json", output=tmp_path / "labels",
                           workers=1, worker_index=0, device="cpu", batch_size=3, max_hours=1)
    atomic_json(study, args.study)
    receipt = module.generate(args)
    assert receipt["status"] == "complete" and len(receipt["completed_state_ids"]) == 2
    before = labels_identity(args.output)
    for reader in ("TREE", "FLAT"):
        rows = load_label_rows(args.output, reader)
        assert len(rows) == 4
        for row in rows:
            assert row["query"].shape == (feature_spec(cfg["model"]["width"])["query_dim"],)
            assert np.isfinite(row["packet_loss"]).all()
            assert row["packet_loss"].shape[1] == 4
        table = build_table(rows, study, reader)
        assert len(table["fit_ids"]) and len(table["development_ids"])
        assert table["label_inventory"][0] == before["dataset_sha256"]
        for objective in ("immediate", "packet"):
            run = tmp_path / "fits" / reader / objective
            assert train(args.study, run, reader=reader, target=objective, device="cpu") == 0
            summary = evaluate(args.study, run, reader=reader, target=objective, device="cpu")
            assert summary["status"] == "complete" and summary["not_closed_loop_success"]
    assert module.generate(args)["status"] == "complete"
    assert labels_identity(args.output)["dataset_sha256"] == before["dataset_sha256"]
    # A duplicate allocation must not overwrite the active owner's receipt.
    previous = (args.output / "worker_0.json").read_bytes()
    with module._lock(args.output / "worker_0.lock"):
        with pytest.raises(BlockingIOError):
            module.generate(args)
    assert (args.output / "worker_0.json").read_bytes() == previous
