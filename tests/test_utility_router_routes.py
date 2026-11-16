"""CPU synthetic invariants for the full-route development diagnostic."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_cached_context_loader import fixture  # noqa: F401 -- synthetic shared fixtures
from test_packed_context import indexed  # noqa: F401
from test_label_objectives import policies
from test_utility_router_online import evidence

from stmem.evaluation.utility_router import LiveIndexedArchive, UtilityOnlineRouter
from stmem.inspection import utility_router_routes as module
from stmem.inspection.refinement_data import UtilityDataReader, stable_seed
from stmem.inspection.refinement_measurements import draw_plan
from stmem.memory.utility_routing import build_scorer
from stmem.models.policy import choose_records
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.packed_context import _take, columns_bank


def routers(width=8):
    torch.manual_seed(412)
    return {name: build_scorer(width, key_width=8).eval().requires_grad_(False) for name in module.ROUTERS}


def routing_protocol():
    return {"budget": 16, "candidate_limit": 4, "packet_max_cost": 4,
            "discovery_draws": 2, "confirmation_draws": 2, "execute_targets": 1,
            "seed": 56, "precision": "fp32", "workers": 1, "batch_size": 3,
            "scope": "synthetic_full_route_diagnostic"}


def test_all_seven_routes_match_actual_online_selection_and_global_uniform():
    ev = evidence()
    clocks = torch.tensor([2., 10.])
    live = LiveIndexedArchive(ev.archive, 8, clocks)
    live.take = lambda rows: _take(live.columns, np.asarray(rows, dtype=np.int64))
    # The current observation uses the same immutable record representation as
    # the online route, without constructing or exposing any expert target.
    from stmem.evaluation.utility_router import record_columns
    current = columns_bank(record_columns(ev.current_records, 8), clocks, 8)
    models, protocol, seed = routers(), routing_protocol(), 42
    contexts, methods = module.construct_routes(live, current, ev.text, clocks, models, protocol, seed)
    assert set(methods) == set(module.METHODS)
    for method, details in methods.items():
        context = contexts[details["context_id"]]
        assert context["token_count"] == 16
        expected_records = live.selected_records(context)
        if method == "uniform":
            expected = choose_records(ev, "UNIFORM", 16)
        else:
            reader, selection = method.split("_")
            online = UtilityOnlineRouter(models.get(method, models[reader + "_packet"]),
                                         reader=reader.upper(), budget=16, protocol=protocol,
                                         selection="random" if selection == "random" else "learned")
            expected = online.select(ev, seed=seed)
        assert {r.node_id for r in expected.records} == {r.node_id for r in expected_records}
        covered = [leaf for record in expected_records for leaf in record.leaf_ids]
        assert len(covered) == len(set(covered)) == len(ev.archive.leaves)
    assert len({context["coverage_hash"] for context in contexts.values()}) == 1
    assert methods["uniform"]["route"]["controller"] == "global_uniform_heap"


@pytest.mark.parametrize("empty", [False, True])
def test_routes_saturate_short_history_and_support_empty_causal_history(empty):
    ev = evidence()
    if empty:
        ev.archive.clear()
    clocks = torch.tensor([2., 10.])
    live = LiveIndexedArchive(ev.archive, 8, clocks)
    live.take = lambda rows: _take(live.columns, np.asarray(rows, dtype=np.int64))
    from stmem.evaluation.utility_router import record_columns
    current = columns_bank(record_columns(ev.current_records, 8), clocks, 8)
    contexts, methods = module.construct_routes(live, current, ev.text, clocks, routers(),
                                               {**routing_protocol(), "budget": 128}, 73)
    assert len(methods) == 7
    assert {context["token_count"] for context in contexts.values()} == {len(ev.archive.leaves)}


def prepared_manifest(tmp_path):
    """Structurally production-size manifest; no trained data/GPU use."""
    source = {"source_sha256": "a" * 64, "git": "1" * 40}
    states = [{"state_id": str(i), "partition": "development", "task": "task", "episode_id": str(i),
               "episode_index": i, "decision_index": 4} for i in range(128)]
    study = {"experiment_id": "child-pilot", "source": source, "configuration": {}, "data_identity": {},
             "student": {"name": "S_STD", "step": 95000, "sha256": "s", "checkpoint": "student"},
             "protocol": {"seed": 10, "budget": 128, "execute_targets": 8, "flow_steps": 8,
                          "candidate_limit": 8, "packet_max_cost": 8, "controller": "packet"},
             "states": states + [{"state_id": "fit", "partition": "fit"}], "dependencies": {}}
    study_path, source_receipt, validation_path = (tmp_path / name for name in ("study.json", "source.json", "verified.json"))
    atomic_json(study, study_path)
    atomic_json(source, source_receipt)
    validation = {"status": "complete", "study_path": str(study_path), "study_sha256": file_sha256(study_path),
                  "experiment_id": study["experiment_id"], "source_sha256": source["source_sha256"],
                  "source_receipt": str(source_receipt), "source_receipt_sha256": file_sha256(source_receipt)}
    atomic_json(validation, validation_path)
    runtime = {"path": str(validation_path), "sha256": file_sha256(validation_path),
               "source_sha256": source["source_sha256"]}
    files = {}
    for name in module.ROUTERS:
        path = tmp_path / (name + ".pt")
        path.write_bytes(b"synthetic checkpoint identity only")
        reader, target = name.split("_")
        files[name] = {"checkpoint": str(path), "sha256": file_sha256(path), "reader": reader,
                       "target": target, "step": 5000}
    manifest = module.build_manifest(study_path, study, runtime, files,
                                     {"source_sha256": "b" * 64, "git": "2" * 40})
    path = tmp_path / "manifest.json"
    atomic_json(manifest, path)
    return path, manifest, study, validation_path


def test_new_stage_binds_all_states_routers_original_runtime_and_fresh_noise(tmp_path, monkeypatch):
    from stmem.training import utility_router_study
    path, manifest, study, _ = prepared_manifest(tmp_path)
    calls = []
    def verify(_path, **kwargs):
        calls.append(kwargs)
        return study
    monkeypatch.setattr(utility_router_study, "verify_study", verify)
    monkeypatch.setattr(module, "source_revision", lambda: manifest["source"])
    assert module.verify_manifest(path) == manifest
    assert calls == [{"verify_source": False, "verify_dependencies": True}]
    assert len(manifest["states"]) == 128
    assert manifest["protocol"]["seed"] == stable_seed(study["protocol"]["seed"], "complete_routes_v1")
    original = draw_plan(study["protocol"]["seed"], "task", "episode", 4)
    fresh = draw_plan(manifest["protocol"]["seed"], "task", "episode", 4)
    assert not {r["seed"] for r in original} & {r["seed"] for r in fresh}
    changed = deepcopy(manifest)
    changed["states"].pop()
    atomic_json(module.seal_manifest(changed), path)
    with pytest.raises(ValueError, match="protocol differ"):
        module.verify_manifest(path)


def test_invalid_original_verification_receipt_fails_before_source_bypass(tmp_path, monkeypatch):
    from stmem.training import utility_router_study
    path, manifest, _, validation_path = prepared_manifest(tmp_path)
    monkeypatch.setattr(module, "source_revision", lambda: manifest["source"])
    monkeypatch.setattr(utility_router_study, "verify_study", lambda *a, **k: pytest.fail("old source bypass reached"))
    validation_path.write_text("{}")
    with pytest.raises(ValueError, match="receipt changed"):
        module.verify_manifest(path)


def test_complete_cpu_route_measurement_resume_report_and_corruption(indexed, tmp_path, monkeypatch):  # noqa: F811
    cfg, data, original, cache, hierarchy = indexed
    teacher, student = policies(original)
    cfg = deepcopy(cfg)
    cfg["tracking"] = {"enabled": False, "mode": "disabled"}
    protocol = routing_protocol()
    states = [{"state_id": task, "task": task, "episode_id": "episode_0", "episode_index": data.by_task[task][0],
               "decision_index": 2, "partition": "development"} for task in ("TaskA", "TaskB")]
    manifest = {"schema": module.SCHEMA, "experiment_id": "explicit-synthetic-diagnostic", "states": states,
                "configuration": cfg, "protocol": protocol, "data_identity": data.identity,
                "student": {"name": "S_STD", "checkpoint": "student", "sha256": "b" * 64, "step": 95000},
                "dependencies": {"teacher": "teacher", "teacher_sha256": "a" * 64,
                                 "encoded": str(cache.root), "hierarchy": str(hierarchy)}}
    models = routers(cfg["model"]["width"])
    monkeypatch.setattr(module, "verify_manifest", lambda _path: manifest)
    monkeypatch.setattr(module, "TrainingData", lambda _cfg: data)
    monkeypatch.setattr(module, "_load_routers", lambda _manifest, _device: models)
    monkeypatch.setattr(module, "load_policy", lambda path, _device:
                        (teacher if path == "teacher" else student,
                         {"stage": "teacher" if path == "teacher" else "compact", "config": cfg}))
    args = SimpleNamespace(manifest=tmp_path / "manifest.json", output=tmp_path / "routes", workers=1,
                           worker_index=0, device="cpu", batch_size=3, max_hours=1)
    atomic_json(manifest, args.manifest)
    receipt = module.measure(args)
    assert receipt["status"] == "complete" and receipt["completed"] == 2
    before = (args.output / "states/TaskA.json").read_bytes()
    assert module.verify_results(manifest, args.output)["status"] == "complete"
    result = module.report(args.manifest, args.output)
    assert result["status"] == "complete" and result["not_closed_loop_success"]
    assert len(result["methods"]) == 14 and len(result["paired_comparisons"]) == 24
    for record in result["state_metrics"]:
        assert all(np.isfinite(record[key]) for key in module.METRICS)
    assert "complete budget-filling routes" in (args.output / "report.md").read_text()
    # Actual cached query/context construction must also agree with the frozen
    # context reader used by existing policy measurements.
    reader = UtilityDataReader(data, cfg, cache.root, hierarchy)
    try:
        saved = module.read_state(args.output, manifest, states[0])
        item, expected = reader.prepare(states[0], saved["contexts"])
        _, group, archive = reader.archive(states[0])
        live = module.CachedRouteArchive(reader, group, archive, cfg["model"]["width"], item["clocks"])
        for context_id, context in saved["contexts"].items():
            actual = live.bank(context)
            for field in ("content", "metadata", "log_count", "valid"):
                torch.testing.assert_close(getattr(actual, field), getattr(expected[context_id], field))
    finally:
        reader.close()
    assert module.measure(args)["status"] == "complete"
    assert (args.output / "states/TaskA.json").read_bytes() == before
    previous = (args.output / "worker_0.json").read_bytes()
    with module.lock(args.output / "worker_0.lock"):
        with pytest.raises(BlockingIOError):
            module.measure(args)
    assert (args.output / "worker_0.json").read_bytes() == previous
    payload = module.read_json(args.output / "states/TaskA.json")
    payload["rows"][0]["expert_full"] += 1
    payload["state_sha256"] = module.digest({k: v for k, v in payload.items() if k != "state_sha256"})
    atomic_json(payload, args.output / "states/TaskA.json")
    with pytest.raises(ValueError, match="checksum"):
        module.measure(args)
    assert (args.output / "worker_0.json").read_bytes() == previous
    (args.output / "states/TaskA.json").write_bytes(before)
    payload = module.read_json(args.output / "states/TaskA.json")
    payload["rows"][0]["expert_full"] += 1
    atomic_json(payload, args.output / "states/TaskA.json")
    with pytest.raises(ValueError, match="sealed content"):
        module.verify_results(manifest, args.output)


def test_partial_report_is_read_only(tmp_path, monkeypatch):
    manifest = {"schema": module.SCHEMA, "experiment_id": "synthetic", "protocol": {"workers": 1},
                "states": [{"state_id": "a"}]}
    monkeypatch.setattr(module, "verify_manifest", lambda _path: manifest)
    path = tmp_path / "manifest.json"
    atomic_json(manifest, path)
    before = path.read_bytes()
    result = module.report(path, tmp_path)
    assert result["status"] == "partial" and result["completed"] == 0
    assert list(tmp_path.iterdir()) == [path] and path.read_bytes() == before
