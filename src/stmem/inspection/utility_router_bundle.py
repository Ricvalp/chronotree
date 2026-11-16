"""Portable, checksummed evidence for the fixed development router diagnostics.

CPU export adopts unchanged, verified HPC studies. Only already encoded causal
states are copied; no raw data, feature recomputation, training or GPU inference
is performed. Relative bundle paths permit relocation without rewriting studies.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from stmem.config import digest, source_revision
from stmem.inspection.refinement_candidates import IndexedArchive
from stmem.inspection.refinement_data import UtilityDataReader, read_json, stable_seed
from stmem.inspection.refinement_measurements import action_mask_summary
from stmem.inspection.utility_router_routes import CachedRouteArchive, METHODS, ROUTERS
from stmem.models.batched_policy import TokenBank
from stmem.training.checkpoints import atomic_json, atomic_save, file_sha256
from stmem.training.data import TrainingData
from stmem.training.packed_context import _take, columns_bank, flat_residual_columns

SCHEMA = "utility_router_bundle_v1"
STATE_SCHEMA = "utility_router_bundle_state_v1"
BANK_FIELDS = ("content", "metadata", "valid", "log_count")
REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def seal(value):
    body = {k: v for k, v in value.items() if k != "bundle_id"}
    return {**body, "bundle_id": digest(body)}


def route_protocol(study):
    protocol = deepcopy(study["protocol"])
    protocol.update(seed=stable_seed(protocol["seed"], "complete_routes_v1"), discovery_draws=4, confirmation_draws=4,
                    workers=1, batch_size=32, methods=list(METHODS), scope="fixed_development_full_route_diagnostic",
                    fresh_draw_namespace="complete_routes_v1")
    return protocol


def bundle_path(root, relative):
    """Only relative regular files inside this bundle may be dereferenced."""
    relative = PurePosixPath(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Bundle file path must remain relative and contained")
    root = Path(root).resolve()
    target = root.joinpath(*relative.parts)
    if any(path.is_symlink() for path in (target, *target.parents) if path != root.parent and path.is_relative_to(root)):
        raise ValueError("Bundle files cannot be symlinks")
    if not target.resolve().is_relative_to(root):
        raise ValueError("Bundle file escapes its root")
    return target


def _encode(value):
    if isinstance(value, TokenBank):
        return {"__bank__": {key: _encode(getattr(value, key)) for key in BANK_FIELDS}}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, np.ndarray):
        payload = value.tolist() if value.dtype.kind in "OUS" else torch.from_numpy(value.copy())
        return {"__array__": payload, "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported portable evidence value: {type(value).__name__}")


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"__bank__"}:
            return TokenBank(**{key: _decode(item) for key, item in value["__bank__"].items()})
        if set(value) == {"__array__", "dtype", "shape"}:
            data = value["__array__"]
            data = data.numpy() if isinstance(data, torch.Tensor) else data
            return np.asarray(data, dtype=np.dtype(value["dtype"])).reshape(value["shape"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


class BundleArchive:
    """Dense row view preserving names, leaf order and exact cached reductions."""
    def __init__(self, record):
        self.archive = IndexedArchive(record["index"], record["roots"], record["leaves"])
        self.columns = record["columns"]
        self.width, self.clocks = int(record["width"]), record["clocks"]
        self.leaf_columns = self.take(self.archive.leaves)
        self.payload_bytes = sum(x.numel() * x.element_size() for x in self.columns.values()
                                 if isinstance(x, torch.Tensor))
        self.input_seconds = 0.0
        if self.archive.coverage != record["coverage_hash"]:
            raise ValueError("Portable causal coverage differs from exported cache")
        execution = self.leaf_columns["source"] == "execution"
        if execution.any() and np.max(self.leaf_columns["time_max"][execution]) >= float(self.clocks[1]):
            raise ValueError("Current/future execution evidence in portable archive")

    def take(self, rows):
        return _take(self.columns, np.asarray(rows, dtype=np.int64))

    def bank(self, context):
        context = self.archive.validate(context)
        columns = (self.take(context["rows"]) if context["reader"] == "TREE" else
                   flat_residual_columns(self.leaf_columns, *self.archive.flat_arrays(context)))
        return columns_bank(columns, self.clocks, self.width)


def export_state(reader, state, configuration, generated_specs=None):
    """Materialize exactly one fixed causal state; preserve original row IDs."""
    _, group, archive = reader.archive(state)
    item, _ = reader.prepare(state, {})
    if not action_mask_summary(item)["full_horizon"]:
        raise ValueError("Fixed diagnostic state must have a full expert action horizon")
    live = CachedRouteArchive(reader, group, archive, configuration["model"]["width"], item["clocks"])
    rows = np.asarray(sorted(archive.active), dtype=np.int64)
    mapping = {int(row): i for i, row in enumerate(rows)}
    index = {key: np.asarray(value)[rows] for key, value in archive.index.items()}
    index["children"] = np.asarray([[mapping[int(child)] if child >= 0 else -1 for child in pair]
                                   for pair in index["children"]], dtype=np.int64).reshape(-1, 2)
    body = {"index": index, "columns": live.columns, "width": live.width, "clocks": item["clocks"],
            "roots": [mapping[int(row)] for row in archive.roots],
            "leaves": [mapping[int(row)] for row in archive.leaves],
            "coverage_hash": archive.coverage, "original_rows": rows}
    portable = BundleArchive(body)
    generated_contexts, generated_identities = {}, {}
    for student, contexts in (generated_specs or {}).items():
        generated_contexts[student], generated_identities[student] = {}, {}
        for alias, old_spec in contexts.items():
            old_spec = archive.validate(old_spec)
            new = portable.archive.context(old_spec["reader"], [mapping[int(row)] for row in old_spec["rows"]])
            if any(new[key] != old_spec[key] for key in ("context_id", "coverage_hash", "token_count")):
                raise ValueError("Dense export changed a selected generated-action context")
            bank, original = portable.bank(new), live.bank(old_spec)
            for field in BANK_FIELDS:
                if not torch.equal(getattr(bank, field), getattr(original, field)):
                    raise ValueError("Portable generated context differs from cached evidence")
            generated_contexts[student][alias] = bank
            generated_identities[student][alias] = {"original": old_spec, "portable": new}
    full = columns_bank(portable.leaf_columns, item["clocks"], live.width)
    for field in BANK_FIELDS:
        if not torch.equal(getattr(full, field), getattr(item["full"], field)):
            raise ValueError("Portable full-history bank differs from cached reference")
    return {"schema": STATE_SCHEMA, "state": deepcopy(state), "item": item, "archive": body,
            "generated_contexts": generated_contexts, "generated_identities": generated_identities}


def load_state(root, manifest, state_id):
    entry = manifest["state_files"][state_id]
    path = bundle_path(root, entry["path"])
    if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
        raise ValueError("Portable state payload checksum changed")
    payload = _decode(torch.load(path, map_location="cpu", weights_only=True))
    if payload.get("schema") != STATE_SCHEMA or payload["state"] != entry["state"]:
        raise ValueError("Portable state identity changed")
    live = BundleArchive(payload["archive"])
    item = payload["item"]
    if item["history_leaves"] != len(live.archive.leaves) or not action_mask_summary(item)["full_horizon"]:
        raise ValueError("Portable state action mask or history count changed")
    for student, aliases in payload["generated_contexts"].items():
        for alias, bank in aliases.items():
            spec = payload["generated_identities"][student][alias]
            if any(spec["original"][key] != spec["portable"][key]
                   for key in ("context_id", "coverage_hash", "token_count")):
                raise ValueError("Portable generated context identity changed")
            expected = live.bank(spec["portable"])
            if any(not torch.equal(getattr(bank, field), getattr(expected, field)) for field in BANK_FIELDS):
                raise ValueError("Portable selected context differs from its causal allocation")
    return item, live, payload["generated_contexts"]


def verify_bundle(root, *, verify_source=True):
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != SCHEMA or seal(manifest) != manifest:
        raise ValueError("Portable bundle manifest changed")
    if verify_source and manifest["source"]["source_sha256"] != source_revision()["source_sha256"]:
        raise ValueError("Diagnostic source differs from exported bundle; use the same committed code")
    complete = read_json(root / "complete.json")
    if (complete.get("status") != "complete" or complete.get("bundle_id") != manifest["bundle_id"]
            or complete.get("manifest_sha256") != file_sha256(root / "manifest.json")):
        raise ValueError("Portable bundle is not complete or its receipt changed")
    for relative, entry in manifest["files"].items():
        path = bundle_path(root, relative)
        if not path.is_file() or path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Portable bundle checksum mismatch: {relative}")
    study, parent, generated = manifest["study"], manifest["parent"], manifest["generated_plan"]
    if (manifest["states"] != [row for row in study["states"] if row["partition"] == "development"]
            or len(manifest["states"]) != 128 or len({s["state_id"] for s in manifest["states"]}) != 128):
        raise ValueError("Portable bundle must retain all 128 fixed development states")
    if (study["parent_experiment_id"] != parent["experiment_id"]
            or generated["experiment_id"] != parent["experiment_id"]
            or manifest["generated_cases"] != generated["cases"]):
        raise ValueError("Portable parent, pilot or generated case plan differs")
    if (manifest["models"]["S_STD"]["sha256"] != study["student"]["sha256"]
            or manifest["models"]["teacher"]["sha256"] != parent["dependencies"]["teacher_sha256"]):
        raise ValueError("Portable model identities differ from frozen study")
    expected_models = {"teacher", "S_STD", "S_STAGE", "S_PACKET", *ROUTERS}
    if set(manifest["models"]) != expected_models:
        raise ValueError("Portable bundle must contain the teacher, three students and four routers")
    for entry in (*manifest["models"].values(), *manifest.get("assets", {}).values(), *manifest["state_files"].values()):
        if manifest["files"].get(entry["path"]) != {key: entry[key] for key in ("sha256", "bytes")}:
            raise ValueError("Portable input entry differs from sealed file inventory")
    for name in ("S_STD", "S_STAGE", "S_PACKET"):
        if any(manifest["models"][name][key] != generated["students"][name][key] for key in ("sha256", "step")):
            raise ValueError("Portable generated-action student differs from original case plan")
    for name, expected in (("pilot_manifest", study), ("parent_manifest", parent), ("generated_plan", generated)):
        if read_json(bundle_path(root, f"originals/{name}.json")) != expected:
            raise ValueError("Portable manifest differs from its unchanged original")
    if (manifest["protocol"] != route_protocol(study) or manifest["configuration"] != study["configuration"]
            or manifest["data_identity"] != study["data_identity"]):
        raise ValueError("Portable runtime protocol/configuration differs from the fixed pilot")
    expected_states = {state["state_id"]: state for state in manifest["states"]}
    for case in generated["cases"]:
        original = expected_states.setdefault(case["state_id"], case["state"])
        if any(original[key] != case["state"][key] for key in ("task", "episode_id", "episode_index", "decision_index")):
            raise ValueError("Generated and development state identities differ")
    if (set(expected_states) != set(manifest["state_files"])
            or any(entry["state"] != expected_states[key] for key, entry in manifest["state_files"].items())):
        raise ValueError("Portable state inventory does not cover the fixed diagnostic union")
    return manifest


def validate_inputs(repo, pilot_name, generated_name):
    from hpc.utility_jobs import runtime_environment
    from hpc.utility_routers import evaluation_inputs, evaluation_study
    from hpc.utility_source import verify_study_source

    repo = Path(repo).resolve()
    pilot_path = repo / "hpc/generated" / pilot_name / "plan.json"
    pilot = read_json(pilot_path)
    study = evaluation_study(pilot)
    _, _, routers = evaluation_inputs(pilot, study)
    generated_launch_path = repo / "hpc/generated" / generated_name / "plan.json"
    generated_launch = read_json(generated_launch_path)
    code_root = Path(generated_launch["code_root"])
    parent_path, case_path = Path(generated_launch["manifest"]), Path(generated_launch["case_plan"])
    receipt = verify_study_source(code_root, parent_path)
    command = ("import json,sys; from stmem.inspection.refinement_generated import verify_plan; "
               "p,m=verify_plan(sys.argv[1]); print(json.dumps({'plan':p,'manifest':m}))")
    result = subprocess.run([pilot["python"], "-c", command, str(case_path)], cwd=code_root,
                            env=runtime_environment(code_root, cpu=True), check=True, text=True, capture_output=True)
    verified = json.loads(result.stdout)
    parent, generated = verified["manifest"], verified["plan"]
    if (study["parent_manifest_sha256"] != file_sha256(parent_path)
            or study["parent_experiment_id"] != parent["experiment_id"]
            or generated["students"]["S_STD"]["sha256"] != study["student"]["sha256"]):
        raise ValueError("Generated-action cases and router pilot use different frozen studies/students")
    states = [row for row in study["states"] if row["partition"] == "development"]
    if (len(states) != 128 or len({row["state_id"] for row in states}) != 128
            or set(generated["students"]) != {"S_STD", "S_STAGE", "S_PACKET"}
            or not 1 <= len(generated["cases"]) <= 16):
        raise ValueError("Export needs all128 development states and the three-student generated plan")
    if (study["student"]["step"] != 95000 or study["student"]["name"] != "S_STD"
            or study["protocol"]["budget"] != 128 or study["protocol"]["execute_targets"] != 8
            or study["protocol"]["controller"] != "packet"):
        raise ValueError("Export requires the fixed S_STD95k B128 execute8 packet pilot")
    if (any(student["step"] != 95000 for student in generated["students"].values())
            or generated["integration_steps"] != 8 or generated["generated_draws"] != 2
            or generated["execute_targets"] != 8):
        raise ValueError("Generated plan must preserve all three95k students, two draws, eight integration/executed targets")
    preflight = {"status": "complete", "pilot_plan": str(pilot_path), "pilot_plan_sha256": file_sha256(pilot_path),
                 "pilot_manifest_sha256": file_sha256(pilot["study"]), "pilot_source": study["source"],
                 "pilot_runtime_receipt_sha256": file_sha256(Path(pilot["code_root"]) / ".utility-source.json"),
                 "parent_manifest_sha256": file_sha256(parent_path), "parent_source": parent["source"],
                 "generated_plan_sha256": file_sha256(case_path), "generated_plan_id": generated["plan_id"],
                 "generated_runtime_receipt": receipt,
                 "scope": "Original study and generated plan validated under their recorded unchanged runtimes"}
    return {"pilot": pilot, "study": study, "parent": parent, "generated": generated, "routers": routers,
            "pilot_path": pilot_path, "parent_path": parent_path, "case_path": case_path,
            "generated_launch_path": generated_launch_path, "preflight": preflight}


def _copy(source, root, relative, *, expected=None):
    source, target = Path(source), bundle_path(root, relative)
    before = file_sha256(source)
    if expected is not None and before != expected:
        raise ValueError(f"Frozen source checksum changed: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if file_sha256(target) != before:
            raise ValueError(f"Existing exported file differs: {relative}; choose another bundle")
    else:
        temporary = target.with_name(target.name + ".partial")
        shutil.copyfile(source, temporary)
        if file_sha256(temporary) != before or file_sha256(source) != before:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Source changed while exporting: {source}")
        os.replace(temporary, target)
    return {"path": relative, "sha256": before, "bytes": target.stat().st_size,
            "source_path": str(source.resolve())}


def write_checksums(root, manifest):
    """Finish/recover checksum publication after the complete receipt is durable."""
    root = Path(root)
    paths = [*sorted(manifest["files"]), "manifest.json", "complete.json", "export_request.json", "export_progress.json"]
    temporary = root / "SHA256SUMS.partial"
    temporary.write_text("".join(f"{file_sha256(root / path)}  {path}\n" for path in paths))
    os.replace(temporary, root / "SHA256SUMS")


def export_bundle(repo, output, *, pilot_name="utility_router_std95k", generated_name="utility140k_generated95k"):
    from hpc.refinement_generated import complete_student
    from stmem.inspection.refinement_utility import read_panel
    from stmem.vision.frozen import resolve_visual_features

    repo, output = Path(repo).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    def progress(phase, **values):
        print(json.dumps({"event": "utility_bundle_export", "phase": phase,
                          "elapsed_seconds": time.monotonic() - started, **values}), flush=True)

    with (output / ".export.lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "complete.json").exists():
            existing = verify_bundle(output)
            if existing["request"] != {"repo": str(repo), "pilot_name": pilot_name, "generated_name": generated_name}:
                raise ValueError("Completed bundle belongs to a different export request")
            write_checksums(output, existing)
            return existing
        progress("verifying_original_studies")
        inputs = validate_inputs(repo, pilot_name, generated_name)
        study, parent, generated = (inputs[key] for key in ("study", "parent", "generated"))
        cfg = study["configuration"]
        source = source_revision()
        request = {"repo": str(repo), "pilot_name": pilot_name, "generated_name": generated_name}
        identity = {"request": request, "source": source, "preflight": inputs["preflight"]}
        request_path = output / "export_request.json"
        if request_path.exists() and read_json(request_path) != identity:
            raise ValueError("Export inputs/source changed; choose a fresh bundle directory")
        atomic_json(identity, request_path)
        progress_path = output / "export_progress.json"
        recorded = read_json(progress_path) if progress_path.exists() else {}
        models, files, assets = {}, {}, {}
        def copy(source_path, relative, expected=None):
            entry = _copy(source_path, output, relative, expected=expected)
            files[relative] = {key: entry[key] for key in ("sha256", "bytes")}
            return entry
        progress("copying_frozen_models")
        models["teacher"] = {**copy(parent["dependencies"]["teacher"], "models/teacher.pt",
                                     parent["dependencies"]["teacher_sha256"]),
                             "step": parent["dependencies"]["teacher_step"], "stage": "teacher"}
        for name, value in generated["students"].items():
            path = study["student"]["checkpoint"] if name == "S_STD" else value["checkpoint"]
            models[name] = {**copy(path, f"models/{name}.pt", value["sha256"]),
                            "step": value["step"], "stage": "compact"}
        for (reader, target), path in inputs["routers"].items():
            name = reader.lower() + "_" + target
            models[name] = {**copy(path, f"models/{name}.pt"), "reader": reader, "target": target,
                            "step": study["protocol"]["fit"]["steps"], "stage": "router"}
            for relative in ("complete.json", "metrics.jsonl", "tracking.json", "tracking_history.jsonl", "evaluation/summary.json"):
                artifact = Path(path).parent / relative
                if artifact.is_file():
                    copy(artifact, f"fits/{name}/{relative}")
        visual = resolve_visual_features(cfg)
        if visual["mode"] != "xyzrgb":
            assets["clip"] = copy(visual["checkpoint"], "assets/clip_RN50.pt", visual["checkpoint_sha256"])
        for key, path in (("pilot_plan", inputs["pilot_path"]), ("pilot_manifest", inputs["pilot"]["study"]),
                          ("parent_manifest", inputs["parent_path"]), ("generated_plan", inputs["case_path"]),
                          ("generated_launch", inputs["generated_launch_path"])):
            copy(path, f"originals/{key}.json")
        copy(Path(inputs["pilot"]["code_root"]) / ".utility-source.json", "originals/pilot_source_receipt.json")
        copy(Path(read_json(inputs["generated_launch_path"])["code_root"]) / ".utility-source.json",
             "originals/generated_source_receipt.json")
        atomic_json(inputs["preflight"], output / "originals/preflight.json")
        files["originals/preflight.json"] = {"sha256": file_sha256(output / "originals/preflight.json"),
                                           "bytes": (output / "originals/preflight.json").stat().st_size}
        cases = {case["state_id"]: case for case in generated["cases"]}
        panels = {}
        for state_id, case in cases.items():
            panels[state_id] = read_panel(inputs["parent_path"].parent, state_id, parent["experiment_id"])
            copy(inputs["parent_path"].parent / "contexts" / f"{state_id}.json", f"originals/contexts/{state_id}.json")
        for offset, entry in enumerate(generated["measurement_inputs"]):
            if Path(entry["path"]).stem in cases:
                copy(entry["path"], f"originals/measurements/{offset:04d}.npz", entry["sha256"])
                copy(Path(entry["path"]).with_suffix(".json"), f"originals/measurements/{offset:04d}.json", entry["metadata_sha256"])
        progress("opening_existing_encoded_cache")
        data = TrainingData(cfg)
        if data.identity != study["data_identity"]:
            raise ValueError("Export action-index identity differs from pilot")
        reader = UtilityDataReader(data, cfg, study["dependencies"]["encoded"], study["dependencies"]["hierarchy"])
        states = [row for row in study["states"] if row["partition"] == "development"]
        union = {state["state_id"]: state for state in states}
        for state_id, case in cases.items():
            if state_id not in union:
                union[state_id] = case["state"]
            elif any(union[state_id][key] != case["state"][key]
                     for key in ("task", "episode_id", "episode_index", "decision_index")):
                raise ValueError("Generated case state differs from development state")
        state_files = {}
        try:
            for offset, (state_id, state) in enumerate(union.items()):
                relative = f"states/{state_id}.pt"
                path = bundle_path(output, relative)
                if state_id in recorded:
                    entry = recorded[state_id]
                    if (entry["state"] != state or entry["path"] != relative
                            or file_sha256(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]):
                        raise ValueError("Previously exported state changed")
                else:
                    specs = {}
                    if state_id in cases:
                        for student, details in cases[state_id]["students"].items():
                            specs[student] = {alias: panels[state_id]["contexts"][context_id]
                                              for alias, context_id in details["contexts"].items()}
                    payload = export_state(reader, state, cfg, specs)
                    atomic_save(_encode(payload), path, keep_previous=False)
                    entry = {"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size, "state": state}
                    recorded[state_id] = entry
                    atomic_json(recorded, progress_path)
                state_files[state_id] = entry
                files[relative] = {key: entry[key] for key in ("sha256", "bytes")}
                progress("state_exported", completed=offset + 1, planned=len(union), state_id=state_id,
                         exported_state_bytes=sum(entry["bytes"] for entry in state_files.values()))
        finally:
            reader.close()
        results = inputs["case_path"].parent / "generated_plan_results"
        copied_results = all(complete_student(inputs["case_path"], student) for student in generated["students"])
        if copied_results:
            for path in sorted(results.rglob("*")):
                if path.is_file() and not path.is_symlink() and not path.name.endswith(".lock"):
                    copy(path, "existing_generated_results/" + path.relative_to(results).as_posix())
        protocol = route_protocol(study)
        if source_revision()["source_sha256"] != source["source_sha256"]:
            raise ValueError("Source changed during export; use a fresh bundle with fixed code")
        manifest = seal({"schema": SCHEMA, "request": request, "source": source, "study": study, "parent": parent,
                         "generated_plan": generated, "generated_cases": generated["cases"],
                         "configuration": cfg, "data_identity": study["data_identity"], "dependencies": study["dependencies"],
                         "student": study["student"], "protocol": protocol, "states": states, "state_files": state_files,
                         "models": models, "assets": assets, "files": files, "preflight": inputs["preflight"],
                         "existing_generated_results": copied_results,
                         "scope": "Exact fixed development evidence; original manifests retained unchanged; no final-partition claim",
                         "total_payload_bytes": sum(entry["bytes"] for entry in files.values())})
        atomic_json(manifest, output / "manifest.json")
        atomic_json({"status": "complete", "bundle_id": manifest["bundle_id"],
                     "manifest_sha256": file_sha256(output / "manifest.json")}, output / "complete.json")
        write_checksums(output, manifest)
        verify_bundle(output)
        progress("complete", output=str(output), states=len(union), bytes=manifest["total_payload_bytes"],
                 gib=manifest["total_payload_bytes"] / 1024**3)
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--pilot-name", default="utility_router_std95k")
    parser.add_argument("--generated-name", default="utility140k_generated95k")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    result = verify_bundle(args.output) if args.verify else export_bundle(args.repo, args.output,
                                      pilot_name=args.pilot_name, generated_name=args.generated_name)
    print(json.dumps({"status": "complete", "bundle_id": result["bundle_id"],
                      "output": str(args.output.resolve()), "bytes": result["total_payload_bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
