"""CPU-only report checks using explicitly synthetic artifact fixtures."""

import hashlib
import json

import pytest

from stmem.inspection.report import build_report, offline_rows, validate_links
from stmem.training.checkpoints import atomic_json, file_sha256


def report_fixture(tmp_path):
    bundle = tmp_path / "bundle"
    model = bundle / "model_a"
    model.mkdir(parents=True)
    checkpoint = model / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic report fixture, not model weights")
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (bundle / "SHA256SUMS").write_text(f"{sha}  model_a/checkpoint.pt\n")
    output = tmp_path / "output"
    output.mkdir()
    manifest = {
        "output": str(output), "bundle": str(bundle), "labels": ["model_a"],
        "checkpoints": {"model_a": {"sha256": sha}},
        "protocol": {"scope": "Synthetic report fixture", "handoff_scenes": [["VideoUnmask", 0]],
                     "handoff_prefixes": [0]},
    }
    atomic_json(manifest, output / "inspection_manifest.json")
    identity = file_sha256(output / "inspection_manifest.json")
    atomic_json({"inspection_manifest_sha256": identity, "checkpoints": {
        "model_a": {"step": 17, "config": {"model": {"horizon": 16}}, "provenance": {}},
    }}, output / "checkpoint_audit.json")
    return manifest, output, identity


def test_partial_report_labels_missing_stages_and_has_no_dead_links(tmp_path):
    manifest, output, _ = report_fixture(tmp_path)
    page = build_report(manifest, allow_partial=True)
    assert "PARTIAL" in page.read_text() and "pending" in page.read_text()
    assert validate_links(output)["status"] == "passed"
    with pytest.raises(ValueError, match="incomplete"):
        build_report(manifest)


def test_report_rejects_results_from_another_checkpoint_bundle(tmp_path):
    manifest, output, _ = report_fixture(tmp_path)
    (output / "offline").mkdir()
    atomic_json({"status": "passed", "inspection_manifest_sha256": "another inspection"},
                output / "offline/validation.json")
    with pytest.raises(ValueError, match="offline validation belongs"):
        build_report(manifest, allow_partial=True)
    assert not (output / "index.html").exists()


def test_report_rejects_failed_validation_even_in_partial_mode(tmp_path):
    manifest, output, identity = report_fixture(tmp_path)
    (output / "offline").mkdir()
    atomic_json({"status": "failed", "inspection_manifest_sha256": identity},
                output / "offline/validation.json")
    with pytest.raises(ValueError, match="did not pass"):
        build_report(manifest, allow_partial=True)


def test_offline_report_preserves_native_time_window_and_missing_descriptors(tmp_path):
    root = tmp_path / "offline"
    root.mkdir()
    atomic_json([
        {"label": "frozen", "native4_expert_tcp_error_mm": 7.5,
         "delete_current_visual_native4_change_mm": 11.4,
         "delete_history_visual_native4_change_mm": 74.3},
        {"label": "xyzrgb", "native4_expert_tcp_error_mm": 5.9,
         "delete_current_visual_native4_change_mm": 1.1,
         "delete_history_visual_native4_change_mm": 18.7},
    ], root / "summary_table.json")
    groups = [
        {"kind": "visual", "variant": "zero_current_frozen_descriptors", "state_subset": subset,
         "native4_tcp_change_mm": value} for subset, value in [("reset", 107.7), ("after_reset", 39.8)]
    ]
    atomic_json({"frozen": {"groups": groups}, "xyzrgb": {"groups": []}}, root / "summary.json")
    rows = offline_rows(tmp_path)
    assert rows["frozen"]["all_states_native4_expert_tcp_error_mm"] == 7.5
    assert rows["frozen"]["after_reset_native4_zero_descriptors_change_mm"] == 39.8
    assert rows["xyzrgb"]["after_reset_native4_zero_descriptors_change_mm"] is None


def test_report_link_validation_detects_missing_video(tmp_path):
    (tmp_path / "index.html").write_text('<video src="missing.mp4"></video>')
    with pytest.raises(ValueError, match="missing local links"):
        validate_links(tmp_path)
    assert json.loads((tmp_path / "report_validation.json").read_text())["missing"] == [
        {"page": "index.html", "target": "missing.mp4"},
    ]
