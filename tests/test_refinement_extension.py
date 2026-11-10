"""Synthetic bounded confirmation-extension controls; no inference or Slurm."""
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from stmem.config import digest
from stmem.inspection.refinement_utility import (
    confirmation_extension, measurement_directory, measurement_draws, save_rows,
)
from stmem.training.checkpoints import file_sha256


def fixture(tmp_path):
    state = {"state_id": "state", "task": "VideoUnmask", "episode_id": "episode_0",
             "decision_index": 100, "partition": "train", "cache_eligible": True}
    manifest = {"experiment_id": "fixed", "states": [state],
                "protocol": {"precision": "fp32", "seed": 1, "additional_draw_cap": 16,
                             "discovery_draws": 4, "confirmation_draws": 4}}
    base = measurement_directory(tmp_path, "S0", 90000, "main", "train")
    base.mkdir(parents=True)
    identity = {"experiment_id": "fixed", "student": "S0", "checkpoint_sha256": "frozen",
                "step": 90000, "partition": "train", "precision": "fp32", "mode": "main"}
    (base / "identity.json").write_text(json.dumps(identity))
    panel = {"experiment_id": "fixed", "state_id": "state", "contexts": {}}
    panel["panel_sha256"] = digest(panel)
    (tmp_path / "contexts").mkdir()
    (tmp_path / "contexts/state.json").write_text(json.dumps(panel))
    rows = [{"context_id": "candidate", "bank": bank, "draw_index": 0, "expert_full": .1}
            for bank in ("discovery", "confirmation")]
    save_rows(base / "state.npz", rows, {"identity": identity, "state": state,
                                        "panel_sha256": panel["panel_sha256"]})
    kwargs = {"student": "S0", "step": 90000, "checkpoint_hash": "frozen", "partition": "train",
              "base_directory": base, "additional_draws": 8}
    return manifest, state, base, kwargs


def test_extension_binds_primary_without_modification_and_uses_fresh_bank(tmp_path):
    manifest, state, base, kwargs = fixture(tmp_path)
    hashes = {path.name: file_sha256(path) for path in base.iterdir()}
    extension = confirmation_extension(manifest, tmp_path, **kwargs)
    assert extension["bank"] == "confirmation_extension" and extension["draws"] == 8
    assert extension["base_state_artifacts"]["state"]["rows_sha256"] == hashes["state.npz"]
    assert hashes == {path.name: file_sha256(path) for path in base.iterdir()}
    primary_draws = measurement_draws(manifest["protocol"], state)
    extra_draws = measurement_draws(manifest["protocol"], state, extension)
    assert len(extra_draws) == 8
    assert {draw["bank"] for draw in extra_draws} == {"confirmation_extension"}
    assert not {draw["seed"] for draw in primary_draws} & {draw["seed"] for draw in extra_draws}
    assert extra_draws == measurement_draws(manifest["protocol"], state, extension)
    assert base != measurement_directory(tmp_path, "S0", 90000, "extension", "train")
    assert "raw_only" in extension["analysis_policy"]


def test_extension_caps_and_binds_exact_student_partition_precision(tmp_path):
    manifest, _, base, kwargs = fixture(tmp_path)
    for count in (0, 17):
        with pytest.raises(ValueError, match="1..16"):
            confirmation_extension(manifest, tmp_path, **{**kwargs, "additional_draws": count})
    manifest["protocol"]["additional_draw_cap"] = 4
    with pytest.raises(ValueError, match="1..4"):
        confirmation_extension(manifest, tmp_path, **kwargs)
    manifest["protocol"]["additional_draw_cap"] = 16
    for key, value in (("student", "S_PACKET"), ("checkpoint_hash", "other"),
                       ("partition", "validation"), ("step", 30000)):
        with pytest.raises(ValueError, match="differs"):
            confirmation_extension(manifest, tmp_path, **{**kwargs, key: value})
    identity = json.loads((base / "identity.json").read_text())
    identity["mode"] = "extension"
    (base / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="original"):
        confirmation_extension(manifest, tmp_path, **kwargs)


def test_extension_detects_primary_artifact_changes_and_missing_confirmation(tmp_path):
    manifest, _, base, kwargs = fixture(tmp_path)
    before = confirmation_extension(manifest, tmp_path, **kwargs)
    metadata = json.loads((base / "state.json").read_text())
    # A changed primary artifact creates a different extension identity on resume.
    rows = [{"context_id": "candidate", "bank": bank, "draw_index": 0, "expert_full": .2}
            for bank in ("discovery", "confirmation")]
    save_rows(base / "state.npz", rows, metadata)
    after = confirmation_extension(manifest, tmp_path, **kwargs)
    assert digest(before) != digest(after)
    save_rows(base / "state.npz", rows[:1], metadata)
    with pytest.raises(ValueError, match="both original banks"):
        confirmation_extension(manifest, tmp_path, **kwargs)


def test_hpc_extension_dry_run_preserves_untyped_gpu_and_pins_original_rows(tmp_path, monkeypatch):
    module_path = Path(__file__).parents[1] / "hpc/refinement_utility.py"
    spec = importlib.util.spec_from_file_location("utility_extension_hpc", module_path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    manifest, _, base, _ = fixture(tmp_path)
    (tmp_path / "src").mkdir()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint for dry-run only")
    manifest["dependencies"] = {"compact": str(checkpoint)}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(launcher, "read_site", lambda _: {"STMEM_REPO_ROOT": str(tmp_path), "STMEM_PYTHON": "python"})
    monkeypatch.setattr(launcher, "verify_manifest", lambda _: manifest)
    calls = []
    def run(command, **_kwargs):
        calls.append(command)
        assert command[:2] == ["bash", "-n"]
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(launcher.subprocess, "run", run)
    launcher.main(["--stage", "extension", "--manifest", str(manifest_path), "--name", "fixture",
                   "--extension-base", str(base), "--additional-draws", "8", "--shards", "2", "--dry-run"])
    plans = list((tmp_path / "hpc/generated/fixture").glob("*.json"))
    assert len(plans) == 1
    plan = json.loads(plans[0].read_text())
    assert plan["commands"][0][-4:] == ["--extension-base", str(base), "--additional-draws", "8"]
    assert plan["inputs"][str(base / "state.json")] == file_sha256(base / "state.json")
    script = Path(plan["script"]).read_text()
    assert "#SBATCH --gres=gpu:1" in script and "#SBATCH --array=0-1%2" in script
    assert "--exclusive" not in script and '--worker-index "${SLURM_ARRAY_TASK_ID:-0}"' in script
    assert len(calls) == 1
