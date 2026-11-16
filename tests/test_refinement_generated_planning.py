"""Synthetic CPU checks: NPZ read caching cannot change the scientific plan."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hpc import refinement_generated_planning as adapter
from stmem.config import digest
from stmem.inspection import refinement_generated as generated
from stmem.inspection import refinement_utility as utility
from stmem.inspection.refinement_data import TEACHER_REFERENCE
from stmem.training.checkpoints import atomic_json, file_sha256


def quiet(*args, **kwargs):
    pass


def test_cached_rows_preserve_values_types_order_and_decode_each_column_once(tmp_path, monkeypatch):
    path = tmp_path / "rows.npz"
    rows = [{"text": f"ctx_{i}", "integer": i, "float": i / 3,
             "boolean": i % 2 == 0, "bytes": f"raw_{i}".encode()} for i in range(35)]
    utility.save_rows(path, rows, {"identity": "synthetic"})
    calls = []
    original = np.lib.npyio.NpzFile.__getitem__

    def getitem(self, key):
        calls.append(key)
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", getitem)
    baseline = utility.load_rows(path)
    assert len(calls) == len(rows) * len(rows[0])
    calls.clear()
    original_numpy, original_loader, original_np_load = utility.np, utility.load_rows, np.load
    with adapter.cached_row_loading(utility, emit=quiet):
        actual = utility.load_rows(path)
        assert utility.np is not np and np.load is original_np_load
    assert utility.np is original_numpy and utility.load_rows is original_loader
    assert calls == list(rows[0])
    assert actual == baseline
    assert [list(row) for row in actual[0]] == [list(row) for row in baseline[0]]
    assert [[type(value) for value in row.values()] for row in actual[0]] == [
        [type(value) for value in row.values()] for row in baseline[0]]


def test_empty_arrays_and_checksum_failure_keep_original_behavior(tmp_path):
    path = tmp_path / "empty.npz"
    utility.save_rows(path, [], {"identity": "synthetic"})
    before = utility.load_rows(path)
    original_numpy, original_loader = utility.np, utility.load_rows
    with adapter.cached_row_loading(utility, emit=quiet):
        assert utility.load_rows(path) == before
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        utility.load_rows(path)
    events = []
    with pytest.raises(ValueError, match="checksum mismatch"):
        with adapter.cached_row_loading(utility, emit=lambda phase, **fields: events.append((phase, fields))):
            utility.load_rows(path)
    assert utility.np is original_numpy and utility.load_rows is original_loader
    assert events[-1][1]["status"] == "failed"


def test_cache_released_and_npy_forwarded_without_changes(tmp_path):
    path = tmp_path / "one.npz"
    np.savez_compressed(path, values=np.arange(3))
    proxy = adapter.ColumnCachingNumpy(np)
    assert proxy.asarray is np.asarray
    cached = proxy.load(path, allow_pickle=False)
    with cached as arrays:
        assert arrays.files == ["values"]
        assert arrays["values"] is arrays["values"]
        assert cached._columns
    assert not cached._columns and cached._archive.zip is None
    cached = proxy.load(path, allow_pickle=False)
    cached["values"]
    cached.close()
    assert not cached._columns and cached._archive.zip is None
    npy = tmp_path / "one.npy"
    np.save(npy, np.arange(3))
    assert np.array_equal(proxy.load(npy, allow_pickle=False), np.load(npy, allow_pickle=False))


def test_progress_first_every_eight_files_and_cli_forwarding(tmp_path, monkeypatch, capsys):
    path = tmp_path / "empty.npz"
    utility.save_rows(path, [], {"identity": "synthetic"})
    forwarded = []

    def scientific_main(argv):
        forwarded.extend(argv)
        for _ in range(17):
            utility.load_rows(path)
        return 0

    monkeypatch.setattr(generated, "main", scientific_main)
    progress_path = tmp_path / "progress.jsonl"
    args = ["--mode", "plan", "--plan", "same.json", "--max-cases", "16"]
    assert adapter.main([*args, "--progress-file", str(progress_path)]) == 0
    assert forwarded == args
    events = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert events == [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    started = [event["file_number"] for event in events if event["phase"] == "rows_started"]
    completed = [event["file_number"] for event in events if event["phase"] == "rows_completed"]
    assert started == completed == [1, 8, 16]
    assert events[0]["phase"] == "planner_import_started"
    assert events[-1]["phase"] == "planner_finished"
    assert events[-2]["files"] == 17


def synthetic_study(tmp_path):
    students = ["S_STD", "S_STAGE", "S_PACKET"]
    states = [{"state_id": f"state_{i}", "task": "VideoUnmask", "episode_id": f"episode_{i}",
               "decision_index": 40, "partition": "train", "cache_eligible": True} for i in range(2)]
    manifest = {"experiment_id": "synthetic-only", "reference": dict(TEACHER_REFERENCE),
                "primary_endpoints": students, "states": states,
                "protocol": {"seed": 17, "precision": "fp32", "gain_epsilon": 1e-6,
                             "execute_targets": 8}}
    manifest_path = tmp_path / "manifest.json"
    atomic_json(manifest, manifest_path)
    candidates = [{"candidate_id": name, "support_hash": name,
                   "prefixes": {"1": f"{name}/first", "8": f"{name}/packet"}} for name in ("a", "b")]
    families = [{"reader": "TREE", "panel": "packet", "background": 120, "final_budget": 128,
                 "base": "base", "controls": {"dispersed": "dispersed"}, "candidates": candidates}]
    for state in states:
        panel = {"experiment_id": manifest["experiment_id"], "families": families}
        panel["panel_sha256"] = digest(panel)
        atomic_json(panel, tmp_path / "contexts" / f"{state['state_id']}.json")
    checkpoints = {}
    for student in students:
        checkpoint = tmp_path / f"{student}.pt"
        checkpoint.write_bytes(f"synthetic checkpoint {student}".encode())
        checkpoints[student] = checkpoint
        identity = {"mode": "main", "experiment_id": manifest["experiment_id"], "partition": "train",
                    "checkpoint_sha256": file_sha256(checkpoint), "step": 95000, "endpoint": 100000,
                    "initialization": "accepted_teacher_weights"}
        folder = tmp_path / "measurements" / student / "step_00095000/main/train"
        atomic_json(identity, folder / "identity.json")
        values = {"base": 10.0, "dispersed": 9.0, "a/first": 5.0, "a/packet": 8.0,
                  "b/first": 9.0, "b/packet": 2.0}
        for state in states:
            rows = [{"context_id": context, "bank": bank, "draw_index": draw,
                     "expert_full": value, "expert_prefix": value / 2,
                     "teacher_full": 10 - value, "teacher_prefix": 5 - value / 2}
                    for context, value in values.items()
                    for bank in ("discovery", "confirmation") for draw in range(2)]
            utility.save_rows(folder / f"{state['state_id']}.npz", rows, {"identity": identity})
    return manifest, manifest_path, checkpoints


def test_whole_generated_plan_is_byte_identical_to_original_planner(tmp_path, monkeypatch):
    manifest, manifest_path, checkpoints = synthetic_study(tmp_path)
    # Only environment provenance/dependency validation is synthetic. The full
    # actual planner still hashes, loads and selects all common measured cases.
    monkeypatch.setattr(generated, "verify_manifest", lambda *_: manifest)
    monkeypatch.setattr(generated, "source_revision", lambda: {"source_sha256": "synthetic"})
    output = tmp_path / "plan.json"
    options = {"checkpoints": checkpoints, "max_cases": 2, "limited_check": True}
    original = generated.create_plan(manifest_path, output, **options)
    original_bytes = output.read_bytes()
    assert len(original["cases"]) == 2
    with adapter.cached_row_loading(utility, emit=quiet):
        optimized = generated.create_plan(manifest_path, output, **options)
    assert optimized == original
    assert optimized["plan_id"] == original["plan_id"]
    assert output.read_bytes() == original_bytes
