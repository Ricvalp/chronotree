"""Checkpoint compatibility and alignment evidence must follow explicit paths."""

from copy import deepcopy
import json
import h5py
import pytest
import torch
import yaml
from stmem.config import load_config, validate_config
from stmem.data.paths import alignment_path
from stmem.data.prepare import prepare_task
from stmem.training.runtime import validate_policy_config


def test_policy_loader_defaults_and_debug_override():
    for path in ("configs/teacher.yaml", "configs/compact.yaml", "configs/perceiver.yaml"):
        cfg = load_config(path)
        validate_config(cfg)
        assert cfg["training"]["workers"] == 2
        assert cfg["training"]["prefetch_factor"] == 1
        assert cfg["training"]["effective_batch"] == 64
        assert cfg["training"]["microbatch"] == 16
        assert cfg["model"]["checkpoint_blocks"] is False
    debug = load_config("configs/debug/teacher.yaml")
    validate_config(debug)
    assert debug["training"]["workers"] == 0
    assert debug["model"]["checkpoint_blocks"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("workers", -1),
        ("workers", 1.5),
        ("workers", True),
        ("workers", "2"),
        ("workers", None),
        ("prefetch_factor", 0),
        ("prefetch_factor", -1),
        ("prefetch_factor", 1.5),
        ("prefetch_factor", True),
        ("prefetch_factor", "1"),
        ("prefetch_factor", None),
    ],
)
def test_invalid_loader_settings_fail_before_worker_creation(key, value):
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["training"][key] = value
    with pytest.raises(ValueError, match=f"training.{key}"):
        validate_config(cfg)


def test_loader_settings_can_be_omitted_for_explicit_configs():
    cfg = load_config("configs/debug/teacher.yaml")
    del cfg["training"]["workers"]
    del cfg["training"]["prefetch_factor"]
    validate_config(cfg)


def test_real_debug_rollout_preserves_checkpoint_contract():
    teacher = load_config("configs/debug/real_teacher.yaml")
    rollout = load_config("configs/debug/rollout.yaml")
    validate_policy_config(teacher, rollout)
    changed = deepcopy(rollout)
    changed["model"].update(
        budget=128,
        budgets=[64, 128, 256],
        max_execution_frames=3000,
        archive_device="cuda",
        checkpoint_blocks=False,
    )
    validate_policy_config(teacher, changed)


@pytest.mark.parametrize(
    "key",
    [
        "points",
        "block_size",
        "visual_tokens",
        "current_tokens",
        "width",
        "layers",
        "horizon",
        "execute",
        "flow_steps",
    ],
)
def test_policy_rejects_shape_compatible_protocol_change(key):
    cfg = load_config("configs/debug/real_teacher.yaml")
    changed = deepcopy(cfg)
    changed["model"][key] += 1
    with pytest.raises(ValueError, match=f"model.{key}"):
        validate_policy_config(cfg, changed)


def test_policy_rejects_geometry_and_encoder_options():
    cfg = load_config("configs/debug/real_teacher.yaml")
    changed = deepcopy(cfg)
    changed["geometry"]["voxel_size"] *= 2
    with pytest.raises(ValueError, match="geometry"):
        validate_policy_config(cfg, changed)
    changed = deepcopy(cfg)
    changed["model"]["encoder_options"]["temporal_denominator"] = 2
    with pytest.raises(ValueError, match="encoder_options"):
        validate_policy_config(cfg, changed)


def test_alignment_default_relocates_and_explicit_override_survives(tmp_path):
    cfg = {"paths": {"data_root": str(tmp_path)}, "data": {"alignment_path": "data/alignment.json"}}
    assert alignment_path(cfg) == tmp_path / "alignment.json"
    cfg["data"]["alignment_path"] = str(tmp_path / "audit/verified.json")
    assert alignment_path(cfg) == tmp_path / "audit/verified.json"
    cfg["data"]["alignment_path"] = "{data_root}/audit/verified.json"
    assert alignment_path(cfg) == tmp_path / "audit/verified.json"


def test_alignment_cli_writes_configured_registry_without_simulator(tmp_path, monkeypatch):
    from stmem.data import alignment

    root = tmp_path / "data"
    (root / "raw").mkdir(parents=True)
    (root / "manifests").mkdir()
    with h5py.File(root / "raw/record_dataset_VideoUnmask.h5", "w") as f:
        f.create_group("episode_0")
        f.create_group("episode_1")
    (root / "manifests/VideoUnmask.json").write_text(json.dumps({"revision": alignment.REVISION}))
    output = tmp_path / "evidence/registry.json"
    config = tmp_path / "audit.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "paths": {"data_root": str(root)},
                "data": {"tasks": ["VideoUnmask"], "alignment_path": str(output)},
            }
        )
    )
    monkeypatch.setattr(
        alignment, "replay_episode", lambda *a, **k: {"post_action_supported": True, "comparisons": []}
    )
    alignment.main(["--config", str(config)])
    assert json.loads(output.read_text())["verified"]
    assert not (root / "alignment.json").exists()


def test_mismatched_cache_config_fails_before_creating_cache(tmp_path, monkeypatch):
    from stmem.data.cache import build_cache
    import stmem.training.runtime as runtime

    teacher = load_config("configs/debug/real_teacher.yaml")
    request = deepcopy(teacher)
    request["model"]["block_size"] *= 2
    request["paths"]["cache_root"] = str(tmp_path / "cache")
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"config": teacher}, checkpoint)
    monkeypatch.setattr(runtime, "validate_selection", lambda *a, **k: None)
    with pytest.raises(ValueError, match="block_size"):
        build_cache(request, checkpoint)
    assert not (tmp_path / "cache").exists()


def test_prepare_rejects_corrupted_raw_member_before_output(tmp_path):
    raw = tmp_path / "corrupt.h5"
    raw.write_bytes(b"corrupted")
    output = tmp_path / "prepared/task.h5"
    with pytest.raises(ValueError, match="download-manifest SHA256"):
        prepare_task(raw, output, "VideoUnmask", 2, expected_raw_sha256="0" * 64)
    assert not output.parent.exists()
