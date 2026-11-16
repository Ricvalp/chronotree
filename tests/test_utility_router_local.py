"""CPU-only portable execution checks; all policy/evidence fixtures are synthetic."""
# ruff: noqa: F811
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_cached_context_loader import fixture  # noqa: F401
from test_packed_context import indexed as indexed
from test_label_objectives import policies
from test_utility_router_routes import routers, routing_protocol
from stmem.inspection import utility_router_local as local
from stmem.inspection import utility_router_bundle as transfer
from stmem.inspection.refinement_data import UtilityDataReader
from stmem.training.checkpoints import atomic_json, atomic_save, file_sha256


@pytest.fixture
def portable(indexed, tmp_path, monkeypatch):
    cfg, data, original, cache, hierarchy = indexed
    cfg = deepcopy(cfg)
    cfg["tracking"] = {"enabled": False, "mode": "disabled"}
    teacher, student = policies(original)
    root, output = tmp_path / "bundle", tmp_path / "analysis"
    root.mkdir()
    models = {}
    for name, policy, stage in (("teacher", teacher, "teacher"), ("S_STD", student, "compact"),
                                ("S_STAGE", student, "compact"), ("S_PACKET", student, "compact")):
        path = root / (name + ".pt")
        torch.save({"stage": stage, "step": 95000, "config": cfg, "model": policy.state_dict()}, path)
        models[name] = {"path": path.name, "sha256": file_sha256(path), "step": 95000}
    raw = {"state_id": "a", "task": "TaskA", "episode_id": "episode_0", "episode_index": data.by_task["TaskA"][0],
           "decision_index": 2, "partition": "development"}
    state = {**raw, "task": "PickXtimes"}
    reader = UtilityDataReader(data, cfg, cache.root, hierarchy)
    item, _ = reader.prepare(raw, {})
    item.update(task="PickXtimes", episode="PickXtimes/episode_0")
    _, group, archive = reader.archive(raw)
    live = local.routes.CachedRouteArchive(reader, group, archive, cfg["model"]["width"], item["clocks"])
    base = archive.coarse("TREE")
    bank = live.bank(base)
    aliases = {name: base["context_id"] for name in ("base", "immediate_completed", "packet_completed", "dispersed")}
    banks = {name: {alias: bank for alias in aliases} for name in ("S_STD", "S_STAGE", "S_PACKET")}
    case = {"state_id": "a", "state": state, "reader": "TREE", "background": base["token_count"],
            "final_budget": base["token_count"], "generation_seeds": [112, 113],
            "students": {name: {"contexts": aliases} for name in banks}, "selection_reasons": ["synthetic fixture"]}
    plan = {"plan_id": "original-synthetic-plan", "students": {name: models[name] for name in banks},
            "cases": [case], "precision": "fp32", "integration_steps": 8, "execute_targets": 1,
            "native_commands_per_target": 2, "scope": "synthetic_diagnostic", "selection_audit": {}}
    bundle = {"bundle_id": "synthetic-evidence-bundle", "configuration": cfg, "models": models,
              "protocol": routing_protocol(), "states": [state],
              "student": {"name": "S_STD", "step": 95000, "sha256": models["S_STD"]["sha256"]},
              "generated_cases": [case], "generated_plan": plan}
    # Export and reload the actual cached state using weights-only portable
    # serialization. Synthetic TaskA uses the PickXtimes Panda convention for
    # generated-action FK; no real-data outcome is implied by this fixture.
    payload = transfer.export_state(reader, raw, cfg,
        {name: {alias: base for alias in aliases} for name in banks})
    payload["state"] = state
    payload["item"].update(task="PickXtimes", episode="PickXtimes/episode_0")
    state_path = root / "state.pt"
    atomic_save(transfer._encode(payload), state_path)
    bundle["state_files"] = {"a": {"path": state_path.name, "sha256": file_sha256(state_path),
        "bytes": state_path.stat().st_size, "state": state}}
    monkeypatch.setattr(transfer, "verify_bundle", lambda _root: bundle)
    monkeypatch.setattr(local, "_routers", lambda *_: routers(cfg["model"]["width"]))
    manifest = local.adopt(bundle, output, 3)
    yield SimpleNamespace(root=root, output=output, manifest=manifest, bundle=bundle, live=live)
    reader.close()


def test_actual_frozen_model_routes_generated_actions_and_reports(portable):
    p = portable
    result = local.run_routes(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=False)
    assert result["status"] == "complete" and result["completed"] == 1
    result = local.run_generated(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=False)
    assert result["status"] == "complete" and result["completed"] == 3
    summary = local.report(p.bundle, p.manifest, p.output)
    assert summary["status"] == "complete"
    assert (p.output / "routes/report.md").exists()
    assert (p.output / "generated/index.html").exists()
    assert "Predicted" in (p.output / "generated/S_STD/a.json").read_text()
    with np.load(p.output / "generated/S_STD/a.npz", allow_pickle=False) as arrays:
        assert np.isfinite(arrays["packet_completed/tcp_paths"]).all()
    report = local.read_json(p.output / "routes/summary.json")
    assert report["not_closed_loop_success"] and len(report["methods"]) == 14


def test_complete_resume_skips_inference_and_binds_committed_bytes(portable, monkeypatch):
    p = portable
    local.run_routes(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=False)
    local.run_generated(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=False)
    old = (p.output / "routes/states/a.json").read_bytes()
    monkeypatch.setattr(local, "_policies", lambda *_: pytest.fail("Completed routes reran models"))
    monkeypatch.setattr(local, "load_frozen_policy", lambda *_: pytest.fail("Completed generation reran models"))
    assert local.run_routes(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=True)["status"] == "complete"
    assert local.run_generated(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=True)["status"] == "complete"
    assert (p.output / "routes/states/a.json").read_bytes() == old
    with (p.output / "generated/S_STD/a.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        local.generated_status(p.bundle, p.manifest, p.output)


def test_partial_routes_require_resume_and_preserve_identity(portable, monkeypatch):
    p = portable
    result = local.run_routes(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1e-15, resume=False)
    assert result["status"] == "partial" and result["completed"] == 0
    with pytest.raises(ValueError, match="use --resume"):
        local.run_routes(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=False)
    assert local.run_routes(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=True)["status"] == "complete"
    with pytest.raises(ValueError, match="identity changed"):
        local.adopt(p.bundle, p.output, 4)
    record = local.read_json(p.output / "routes/states/a.json")
    record["rows"][0]["expert_full"] += 1
    record["state_sha256"] = local.digest({k: v for k, v in record.items() if k != "state_sha256"})
    atomic_json(record, p.output / "routes/states/a.json")
    receipt = (p.output / "routes/worker_0.json").read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        local.run_routes(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=True)
    assert (p.output / "routes/worker_0.json").read_bytes() == receipt


def test_status_without_results_is_read_only(portable, capsys):
    p = portable
    output = p.output.parent / "not_started"
    assert local.main(["--bundle", str(p.root), "--output", str(output), "--mode", "status"]) == 0
    assert not output.exists()
    assert "not_started" in capsys.readouterr().out


def test_orphan_generation_cannot_be_reused_for_another_student(portable):
    p = portable
    local.run_generated(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=False)
    path = p.output / "generated/S_STD/a.npz"
    saved = local.read_json(path.with_suffix(".json"))
    saved["student"] = "S_PACKET"
    atomic_json(saved, path.with_suffix(".json"))
    (p.output / "generated/receipt.json").unlink()
    with pytest.raises(ValueError, match="another case, model or protocol"):
        local.run_generated(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=True)


def test_failed_resume_keeps_existing_committed_generation_cells(portable, monkeypatch):
    p = portable
    local.run_generated(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=False)
    path = p.output / "generated/receipt.json"
    receipt = local.read_json(path)
    receipt.update(status="partial", completed=1, completed_cells=receipt["planned_cells"][:1])
    name, state_id = receipt["completed_cells"][0]
    key = f"{name}/{state_id}"
    guard = receipt["files"][key]
    receipt["files"] = {key: guard}
    atomic_json(receipt, path)
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("synthetic device unavailable")
    monkeypatch.setattr(local, "load_frozen_policy", unavailable)
    with pytest.raises(RuntimeError, match="device unavailable"):
        local.run_generated(p.root, p.bundle, p.manifest, p.output, device="cpu", max_hours=1, resume=True)
    after = local.read_json(path)
    assert after["status"] == "failed" and after["completed"] == 1 and after["files"] == {key: guard}
