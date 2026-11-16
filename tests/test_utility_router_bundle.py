"""Synthetic CPU-only portable evidence, relocation and integrity checks."""
import shutil

import pytest
import torch

from test_cached_context_loader import fixture  # noqa: F401 -- synthetic fixtures
from test_packed_context import indexed  # noqa: F401
from stmem.inspection import utility_router_bundle as module
from stmem.inspection.refinement_data import UtilityDataReader
from stmem.inspection.utility_router_routes import CachedRouteArchive, ROUTERS, construct_routes
from stmem.memory.utility_routing import build_scorer
from stmem.training.checkpoints import atomic_json, atomic_save, file_sha256


def state(data):
    index, t = data.by_task[data.tasks[0]][0], 2
    return {"state_id": "synthetic", "task": data.tasks[0], "episode_id": "episode_0",
            "episode_index": index, "decision_index": t, "partition": "development"}


def test_roundtrip_preserves_cached_banks_selected_contexts_and_routes(indexed, tmp_path):  # noqa: F811
    cfg, data, _policy, encoded, hierarchy = indexed
    record = state(data)
    reader = UtilityDataReader(data, cfg, encoded.root, hierarchy)
    try:
        _, group, archive = reader.archive(record)
        selected = {"base": archive.coarse("TREE"), "flat": archive.coarse("FLAT")}
        payload = module.export_state(reader, record, cfg, {"S_STD": selected})
        path = tmp_path / "states/synthetic.pt"
        atomic_save(module._encode(payload), path)
        manifest = {"state_files": {"synthetic": {"path": "states/synthetic.pt", "sha256": file_sha256(path),
                                                   "bytes": path.stat().st_size, "state": record}}}
        item, portable, banks = module.load_state(tmp_path, manifest, "synthetic")
        original_item, original_banks = reader.prepare(record, selected)
        for name in ("current", "full"):
            for field in module.BANK_FIELDS:
                assert torch.equal(getattr(item[name], field), getattr(original_item[name], field))
        for alias in selected:
            for field in module.BANK_FIELDS:
                assert torch.equal(getattr(banks["S_STD"][alias], field), getattr(original_banks[alias], field))
        for name in ("target", "valid", "text", "clocks", "current_state"):
            assert torch.equal(torch.as_tensor(item[name]), torch.as_tensor(original_item[name]))
        live = CachedRouteArchive(reader, group, archive, cfg["model"]["width"], item["clocks"])
        routers = {name: build_scorer(live.width, 8).eval().requires_grad_(False) for name in ROUTERS}
        protocol = {"budget": 8, "candidate_limit": 4, "packet_max_cost": 3}
        before, methods = construct_routes(live, item["current"], item["text"], item["clocks"], routers, protocol, 19)
        after, new_methods = construct_routes(portable, item["current"], item["text"], item["clocks"], routers, protocol, 19)
        assert set(before) == set(after)
        assert {k: v["context_id"] for k, v in methods.items()} == {k: v["context_id"] for k, v in new_methods.items()}
        for key in before:
            for field in module.BANK_FIELDS:
                assert torch.equal(getattr(live.bank(before[key]), field), getattr(portable.bank(after[key]), field))
        shutil.move(str(tmp_path / "states"), str(tmp_path / "relocated"))
        manifest["state_files"]["synthetic"]["path"] = "relocated/synthetic.pt"
        module.load_state(tmp_path, manifest, "synthetic")
        with (tmp_path / "relocated/synthetic.pt").open("ab") as stream:
            stream.write(b"corrupt")
        with pytest.raises(ValueError, match="checksum"):
            module.load_state(tmp_path, manifest, "synthetic")
    finally:
        reader.close()


def test_safe_relative_paths_reject_escapes_and_symlinks(tmp_path):
    with pytest.raises(ValueError, match="relative"):
        module.bundle_path(tmp_path, "../elsewhere")
    with pytest.raises(ValueError, match="relative"):
        module.bundle_path(tmp_path, "/absolute")
    (tmp_path / "link").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="symlink"):
        module.bundle_path(tmp_path, "link/file")


def test_complete_export_is_relocatable_resumable_and_keeps_original_partitions(indexed, tmp_path, monkeypatch):  # noqa: F811
    """Exercise real cache reads/export; only inaccessible HPC preflight is mocked."""
    from hpc import refinement_generated
    cfg, data, _policy, encoded, hierarchy = indexed
    repo = tmp_path / "repo"
    repo.mkdir()
    code_root = repo / "old_source"
    code_root.mkdir()
    atomic_json({"source_sha256": "old"}, code_root / ".utility-source.json")
    source = {"source_sha256": "synthetic-current", "git": "synthetic", "dirty": False}
    monkeypatch.setattr(module, "source_revision", lambda: source)
    paths = {}
    for name in ("teacher", "S_STD", "S_STAGE", "S_PACKET", *ROUTERS):
        folder = repo / name
        folder.mkdir()
        path = folder / "checkpoint.pt"
        path.write_bytes(("synthetic checkpoint " + name).encode())
        paths[name] = path
        atomic_json({"status": "complete"}, folder / "complete.json")
    states = [{**state(data), "state_id": f"state_{i}"} for i in range(128)]
    deps = {"teacher": str(paths["teacher"]), "teacher_sha256": file_sha256(paths["teacher"]), "teacher_step": 140000,
            "encoded": str(encoded.root), "hierarchy": str(hierarchy)}
    parent = {"experiment_id": "parent", "configuration": cfg, "dependencies": deps}
    protocol = {"seed": 34, "budget": 128, "execute_targets": 8, "flow_steps": 8,
                "controller": "packet", "fit": {"steps": 5000}}
    study = {"experiment_id": "child", "parent_experiment_id": "parent", "configuration": cfg,
             "dependencies": deps, "data_identity": data.identity, "states": states, "protocol": protocol,
             "student": {"name": "S_STD", "step": 95000, "checkpoint": str(paths["S_STD"]),
                         "sha256": file_sha256(paths["S_STD"])}}
    reader = UtilityDataReader(data, cfg, encoded.root, hierarchy)
    try:
        _, _, archive = reader.archive(states[0])
        context = archive.coarse("TREE")
    finally:
        reader.close()
    parent_path, study_path, case_path = (repo / filename for filename in ("parent.json", "study.json", "generated_plan.json"))
    atomic_json(parent, parent_path)
    atomic_json(study, study_path)
    case = {"state_id": states[0]["state_id"], "state": {**states[0], "partition": "train"},
            "generation_seeds": [1, 2], "students": {name: {"contexts": {"base": context["context_id"]}}
            for name in ("S_STD", "S_STAGE", "S_PACKET")}}
    generated = {"plan_id": "generated", "experiment_id": "parent", "cases": [case], "measurement_inputs": [],
                 "students": {name: {"checkpoint": str(paths[name]), "sha256": file_sha256(paths[name]), "step": 95000}
                 for name in ("S_STD", "S_STAGE", "S_PACKET")}}
    atomic_json(generated, case_path)
    panel = {"experiment_id": "parent", "contexts": {context["context_id"]: context}}
    panel["panel_sha256"] = module.digest(panel)
    atomic_json(panel, repo / "contexts/state_0.json")
    pilot_path, generated_launch = repo / "pilot_plan.json", repo / "generated_launch.json"
    pilot = {"study": str(study_path), "code_root": str(code_root)}
    atomic_json(pilot, pilot_path)
    atomic_json({"code_root": str(code_root)}, generated_launch)
    inputs = {"pilot": pilot, "study": study, "parent": parent, "generated": generated,
              "routers": {(name.split("_")[0].upper(), name.split("_")[1]): paths[name] for name in ROUTERS},
              "pilot_path": pilot_path, "parent_path": parent_path, "case_path": case_path,
              "generated_launch_path": generated_launch, "preflight": {"status": "complete", "synthetic": True}}
    monkeypatch.setattr(module, "validate_inputs", lambda *args: inputs)
    monkeypatch.setattr(module, "TrainingData", lambda config: data)
    monkeypatch.setattr(refinement_generated, "complete_student", lambda *_: False)
    root = tmp_path / "original"
    manifest = module.export_bundle(repo, root)
    assert len(manifest["state_files"]) == 128
    assert manifest["states"][0]["partition"] == "development"
    assert manifest["generated_cases"][0]["state"]["partition"] == "train"
    assert module.verify_bundle(root) == manifest
    item, _archive, contexts = module.load_state(root, manifest, "state_0")
    assert set(contexts) == {"S_STD", "S_STAGE", "S_PACKET"}
    before = (root / "states/state_0.pt").read_bytes()
    (root / "SHA256SUMS").unlink()
    monkeypatch.setattr(module, "validate_inputs", lambda *_: pytest.fail("Completed resume reread HPC"))
    assert module.export_bundle(repo, root) == manifest
    assert (root / "SHA256SUMS").is_file()
    assert (root / "states/state_0.pt").read_bytes() == before
    destination = tmp_path / "moved"
    shutil.move(str(root), destination)
    assert module.verify_bundle(destination) == manifest
    receipt = module.read_json(destination / "complete.json")
    receipt["status"] = "running"
    atomic_json(receipt, destination / "complete.json")
    with pytest.raises(ValueError, match="not complete"):
        module.verify_bundle(destination)


def test_immutable_copy_rejects_changed_source_or_destination(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"frozen")
    root = tmp_path / "bundle"
    root.mkdir()
    first = module._copy(source, root, "model.pt")
    assert module._copy(source, root, "model.pt") == first
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        module._copy(source, root, "model.pt", expected=first["sha256"])
    with pytest.raises(ValueError, match="differs"):
        module._copy(source, root, "model.pt")
