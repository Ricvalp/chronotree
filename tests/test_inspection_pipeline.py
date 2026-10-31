"""CPU checks for immutable transfer validation and repeatable inspection setup."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

from stmem.config import digest
from stmem.inspection import common
from stmem.inspection.pipeline import inspection_lock
from stmem.models.policy import MemoryPolicy
from stmem.training.checkpoints import capture_rng
from test_batched_training import _config


def bundle(tmp_path, contents=b"checkpoint diagnostic fixture"):
    root = tmp_path / "bundle"
    (root / "model_a").mkdir(parents=True)
    path = root / "model_a/checkpoint.pt"
    path.write_bytes(contents)
    sha = hashlib.sha256(contents).hexdigest()
    (root / "SHA256SUMS").write_text(f"{sha}  model_a/checkpoint.pt\n")
    return root


def test_transfer_checksum_detects_partial_and_changed_files(tmp_path):
    root = bundle(tmp_path)
    result = common.discover_checkpoints(root)
    assert list(result) == ["model_a"]
    assert Path(result["model_a"]["path"]).is_file()
    (root / "model_a/checkpoint.pt").write_bytes(b"interrupted rsync")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        common.discover_checkpoints(root)


@pytest.mark.parametrize("name", ["../checkpoint.pt", "model_a/../checkpoint.pt", "/tmp/checkpoint.pt"])
def test_checksum_manifest_cannot_escape_bundle(tmp_path, name):
    root = bundle(tmp_path)
    (root / "SHA256SUMS").write_text(f"{'0' * 64}  {name}\n")
    with pytest.raises(ValueError, match="Unsupported"):
        common.discover_checkpoints(root)


def test_checkpoint_symlink_cannot_escape_bundle(tmp_path):
    root = bundle(tmp_path)
    inside = root / "model_a/checkpoint.pt"
    outside = tmp_path / "outside.pt"
    inside.replace(outside)
    inside.symlink_to(outside)
    with pytest.raises(ValueError, match="inside the bundle"):
        common.discover_checkpoints(root)


def test_second_launcher_cannot_write_the_same_output(tmp_path):
    with inspection_lock(tmp_path / "output"):
        with pytest.raises(RuntimeError, match="Another inspection"):
            with inspection_lock(tmp_path / "output"):
                pytest.fail("second lock acquired")
    with inspection_lock(tmp_path / "output"):
        pass


def test_resume_keeps_manifest_immutable_and_rejects_changed_assets(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(common, "audit_checkpoint", lambda entry: {**entry, "step": 17})
    asset = tmp_path / "data.h5"
    asset.write_bytes(b"fixed real-data fixture identity")
    values = {name: str(asset) for name in (
        "prepared", "alignment", "features", "clip_checkpoint", "pick_prefix", "replay_commands",
    )}
    config = tmp_path / "assets.yaml"
    config.write_text(yaml.safe_dump(values))
    manifest, path = common.prepare_manifest(root, output, config)
    original = path.read_bytes()
    same, same_path = common.prepare_manifest(root, output, config, resume=True)
    assert same == manifest and same_path == path and path.read_bytes() == original
    with pytest.raises(FileExistsError, match="use --resume"):
        common.prepare_manifest(root, output, config)
    asset.write_bytes(b"different observations")
    with pytest.raises(ValueError, match="changed"):
        common.prepare_manifest(root, output, config, resume=True)
    assert path.read_bytes() == original


def test_checkpoint_audit_validates_actual_model_optimizer_and_provenance(tmp_path, monkeypatch):
    """Synthetic serialized state, explicitly not a successful real-data run."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = _config(tmp_path / "fake_run", batch=2, steps=3)
    # Audit checks real-run declarations; data is never read in this serialization fixture.
    cfg["debug"] = False
    cfg["data"]["synthetic"] = False
    cfg["training"].update(warmup_steps=1, lr=0.0001)
    with torch.random.fork_rng(devices=[]):
        policy = MemoryPolicy(cfg)
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.0001)
        for parameter in policy.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        state = {
            "config": cfg, "stage": "teacher", "model": policy.state_dict(),
            "optimizer": optimizer.state_dict(), "step": 1, "scheduler": {"last_epoch": 1},
            "rng_by_rank": [capture_rng()],
            "provenance": {"model": digest(cfg["model"]), "training": digest(cfg["training"]),
                           "effective_batch": 2, "world_size": 1, "sampler": "fixture",
                           "rank_assignment": "rotating_rank_v1"},
            "sampler": {"next_global_position": 2, "boundary": "optimizer_step", "version": "fixture",
                        "rank_assignment": "rotating_rank_v1"},
        }
    root = bundle(tmp_path)
    path = root / "model_a/checkpoint.pt"

    def save(payload):
        torch.save(payload, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / "SHA256SUMS").write_text(f"{sha}  model_a/checkpoint.pt\n")
        return common.discover_checkpoints(root)["model_a"]

    result = common.audit_checkpoint(save(state))
    assert result["status"] == "passed" and result["step"] == 1
    assert all(result["checks"].values())
    corrupt = deepcopy(state)
    corrupt["scheduler"]["last_epoch"] = 2
    with pytest.raises(ValueError, match="Scheduler step"):
        common.audit_checkpoint(save(corrupt))
    corrupt = deepcopy(state)
    next(iter(corrupt["model"].values())).fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        common.audit_checkpoint(save(corrupt))
    (path.parent / "source.json").write_text(json.dumps({"step": 0, "sha256": "wrong"}))
    with pytest.raises(ValueError, match="source.json"):
        common.audit_checkpoint(save(state))
