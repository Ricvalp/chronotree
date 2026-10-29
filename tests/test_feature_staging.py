"""Synthetic sidecars: staging is bounded, immutable and identity preserving."""

from copy import deepcopy
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import pytest
import torch

from stmem.data import stage_features
from stmem.data.dataset import EpisodeDataset
from stmem.training.data import TrainingData
from test_frozen_feature_cache import cache_input as _cache_input, prepare
from test_training_logging_hpc import planning as _planning


cache_input = _cache_input
planning = _planning


@pytest.fixture
def staging(cache_input, tmp_path, monkeypatch):
    monkeypatch.delenv("STMEM_FEATURE_CACHE_ROOT", raising=False)
    prepare(cache_input)
    scratch = tmp_path / "local scratch"
    scratch.mkdir()
    return cache_input, scratch


def test_stage_exact_bytes_and_runtime_paths_preserve_training_identity(staging, monkeypatch):
    (cfg, _, canonical, source), scratch = staging
    original_config = deepcopy(cfg)
    original = TrainingData(cfg)
    result = stage_features.stage_feature_cache(cfg, scratch)
    destination = Path(result["path"]) / source.name
    assert result["copied_bytes"] == source.stat().st_size and result["reused_bytes"] == 0
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o222 == 0
    record = json.loads((destination.parent / "staging.json").read_text())
    assert record["files"]["VideoUnmask"]["sha256"] == sha256(source.read_bytes()).hexdigest()
    monkeypatch.setenv("STMEM_FEATURE_CACHE_ROOT", str(destination.parent))
    relocated = TrainingData(cfg)
    assert relocated.identity == original.identity and cfg == original_config
    left = original.dataset.load_episode(0).conditioning[0].visual_features
    right = relocated.dataset.load_episode(0).conditioning[0].visual_features
    assert Path(right.path) == destination and Path(left.path) == source
    torch.testing.assert_close(left.load(), right.load(), rtol=0, atol=0)


def test_reuse_performs_no_copy_or_full_file_rehash(staging, monkeypatch):
    (cfg, _, _, source), scratch = staging
    first = stage_features.stage_feature_cache(cfg, scratch)

    def forbidden(*args, **kwargs):
        raise AssertionError("Verified unchanged sidecars must reuse metadata without rehashing")

    monkeypatch.setattr(stage_features, "_copy_verified", forbidden)
    monkeypatch.setattr(stage_features, "_sha256", forbidden)
    again = stage_features.stage_feature_cache(cfg, scratch)
    assert again["path"] == first["path"] and again["copied_bytes"] == 0
    assert again["reused_bytes"] == source.stat().st_size


def test_source_change_uses_new_namespace_and_changed_destination_is_rejected(staging):
    (cfg, _, _, source), scratch = staging
    first = stage_features.stage_feature_cache(cfg, scratch)
    old_path = Path(first["path"]) / source.name
    original_bytes = old_path.read_bytes()
    with h5py.File(source, "r+") as handle:
        handle["episode_0/features"][0, 0, 0] += 1
    second = stage_features.stage_feature_cache(cfg, scratch)
    assert second["path"] != first["path"] and old_path.read_bytes() == original_bytes
    destination = Path(second["path"]) / source.name
    destination.chmod(0o644)
    with destination.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        stream.write(b"x")
    altered = destination.read_bytes()
    with pytest.raises(ValueError, match="changed or is untracked"):
        stage_features.stage_feature_cache(cfg, scratch)
    assert destination.read_bytes() == altered


def test_capacity_failure_occurs_before_copy(staging, monkeypatch):
    (cfg, _, _, source), scratch = staging
    original = source.read_bytes()
    monkeypatch.setattr(stage_features.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(OSError, match="Insufficient scratch space"):
        stage_features.stage_feature_cache(cfg, scratch)
    assert not list(scratch.rglob("*.h5")) and source.read_bytes() == original


def test_checksum_failure_does_not_publish_partial_file(staging, monkeypatch):
    (cfg, _, _, source), scratch = staging
    original = source.read_bytes()
    monkeypatch.setattr(stage_features, "_sha256", lambda _: "0" * 64)
    with pytest.raises(OSError, match="checksum mismatch"):
        stage_features.stage_feature_cache(cfg, scratch)
    assert not list(scratch.rglob("*.h5"))
    assert not list(scratch.rglob("*.partial.*"))
    assert source.read_bytes() == original


def test_interruption_after_atomic_copy_recovers_only_verified_contents(staging, monkeypatch):
    (cfg, _, _, source), scratch = staging

    def interrupted_record(*args):
        raise RuntimeError("Synthetic interrupted staging record")

    with monkeypatch.context() as scoped:
        scoped.setattr(stage_features, "_atomic_json", interrupted_record)
        with pytest.raises(RuntimeError, match="interrupted staging record"):
            stage_features.stage_feature_cache(cfg, scratch)
    paths = list(scratch.rglob("*.h5"))
    assert len(paths) == 1 and paths[0].read_bytes() == source.read_bytes()
    result = stage_features.stage_feature_cache(cfg, scratch)
    assert result["copied_bytes"] == 0 and result["reused_bytes"] == source.stat().st_size
    record = paths[0].parent / "staging.json"
    assert json.loads(record.read_text())["files"]["VideoUnmask"]["sha256"]
    record.unlink()
    paths[0].chmod(0o644)
    with paths[0].open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        stream.write(b"x")
    with pytest.raises(ValueError, match="failed checksum verification"):
        stage_features.stage_feature_cache(cfg, scratch)
    assert not record.exists()


@pytest.mark.parametrize("failure", ["diagnostic", "wrong_geometry", "symlink", "inside_source"])
def test_invalid_source_or_destination_rejected_before_copy(staging, failure):
    (cfg, _, canonical, source), scratch = staging
    if failure == "diagnostic":
        with h5py.File(source, "r+") as handle:
            manifest = json.loads(handle.attrs["manifest"])
            manifest["diagnostic"] = True
            handle.attrs.modify("manifest", json.dumps(manifest))
        expected = "diagnostic feature cache"
    elif failure == "wrong_geometry":
        with h5py.File(canonical, "r+") as handle:
            manifest = json.loads(handle.attrs["manifest"])
            manifest["points"] += 1
            handle.attrs.modify("manifest", json.dumps(manifest))
        expected = "geometry/source identity"
    elif failure == "symlink":
        (scratch / "chronotree_features").symlink_to(source.parent, target_is_directory=True)
        expected = "Unsafe feature staging directory"
    else:
        scratch = source.parent
        expected = "outside the source"
    original = source.read_bytes()
    with pytest.raises(ValueError, match=expected):
        stage_features.stage_feature_cache(cfg, scratch)
    assert source.read_bytes() == original


def _concurrent_stage(config, scratch, gate, queue):
    try:
        gate.wait(20)
        queue.put(stage_features.stage_feature_cache(config, scratch))
    except Exception as exc:
        queue.put({"error": repr(exc)})


def test_concurrent_jobs_copy_once_with_shared_lock(staging):
    (cfg, _, _, source), scratch = staging
    context = multiprocessing.get_context("spawn")
    gate, queue = context.Event(), context.Queue()
    workers = [context.Process(target=_concurrent_stage, args=(cfg, scratch, gate, queue)) for _ in range(2)]
    try:
        for worker in workers:
            worker.start()
        gate.set()
        results = [queue.get(timeout=45) for _ in workers]
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        assert all("error" not in result for result in results), results
        assert len({result["path"] for result in results}) == 1
        assert sum(result["copied_bytes"] for result in results) == source.stat().st_size
        assert sum(result["reused_bytes"] for result in results) == source.stat().st_size
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=10)
        queue.close()


def test_runtime_cli_prints_only_path_and_override_still_validates_identity(staging, monkeypatch, capsys):
    (cfg, _, canonical, source), scratch = staging
    config_path = scratch / "config.json"
    config_path.write_text(json.dumps(cfg))
    for key in ("STMEM_DATA_ROOT", "STMEM_CACHE_ROOT", "STMEM_RUN_ROOT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(sys, "argv", ["stage_features", "--config", str(config_path),
                                      "--scratch-root", str(scratch), "--print-path"])
    stage_features.main()
    captured = capsys.readouterr()
    destination = Path(captured.out.strip())
    assert (destination / source.name).is_file()
    assert json.loads(captured.err.splitlines()[-1])["status"] == "ready"
    monkeypatch.setenv("STMEM_FEATURE_CACHE_ROOT", "relative/path")
    with pytest.raises(ValueError, match="absolute directory"):
        EpisodeDataset(canonical, visual_features_config=cfg)
    monkeypatch.setenv("STMEM_FEATURE_CACHE_ROOT", str(destination))
    altered = deepcopy(cfg)
    altered["model"]["visual_features"]["feature_stride"] = 8
    with pytest.raises(ValueError):
        EpisodeDataset(canonical, visual_features_config=altered)


def test_planner_stages_only_at_runtime_before_torchrun(planning, tmp_path):
    _, invoke, _, _, _ = planning
    site = tmp_path / "site.env"
    scratch = tmp_path / "must not be created on login"
    site.write_text(site.read_text() + f"STMEM_SCRATCH_ROOT='{scratch}'\n")
    plan = invoke("--only", "seed0_teacher", "--profile", "b200_4gpu",
                  "--teacher-config", "configs/teacher_fv_no_prop_10hz.yaml")
    script_path = Path(plan["jobs"][0]["script"])
    script = script_path.read_text()
    assert 'if [[ -n "${STMEM_SCRATCH_ROOT:-}" ]]; then' in script
    assert ' --scratch-root "$STMEM_SCRATCH_ROOT" --print-path)' in script
    assert "export STMEM_FEATURE_CACHE_ROOT" in script
    assert script.index("-m stmem.data.stage_features") < script.index("-m torch.distributed.run")
    assert script.count("-m stmem.data.stage_features") == 1
    assert not scratch.exists()
    subprocess.run(["bash", "-n", script_path], check=True)


def test_one_stage_template_has_valid_conditional_staging():
    path = Path(__file__).resolve().parents[1] / "hpc/jobs/common.sh"
    script = path.read_text()
    assert 'if [[ -n "${STMEM_SCRATCH_ROOT:-}" ]]; then' in script
    assert script.index("-m stmem.data.stage_features") < script.index("-m torch.distributed.run")
    subprocess.run(["bash", "-n", path], check=True)
