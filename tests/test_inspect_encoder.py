"""Inspector selection, alignment, checkpoint provenance, and CPU CLI boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import random

import h5py
import numpy as np
import pytest
import torch
import yaml

from stmem.config import load_config
from stmem.data.pointcloud import CALIBRATION_VERSION
from stmem.training.checkpoints import file_sha256
from stmem.visualization import inspect_encoder as inspector


@pytest.fixture
def inspect_data(tmp_path, monkeypatch):
    for name in ("STMEM_DATA_ROOT", "STMEM_RUN_ROOT", "STMEM_CACHE_ROOT"):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "data"
    prepared = root / "prepared/VideoUnmask.h5"
    prepared.parent.mkdir(parents=True)
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["paths"]["data_root"] = str(root)
    cfg["data"].update(synthetic=False, tasks=["VideoUnmask"], episodes_per_task=2)
    cfg["model"].update(points=4, width=16, current_tokens=4)
    manifest = {
        "task": "VideoUnmask",
        "dataset_revision": "fixture-revision",
        "raw_sha256": "a" * 64,
        "points": 4,
        "crop": cfg["geometry"]["crop"],
        "voxel_size": cfg["geometry"]["voxel_size"],
        "max_depth_m": cfg["geometry"]["max_depth"],
        "calibration": CALIBRATION_VERSION,
        "diagnostic_fixture": True,
    }
    with h5py.File(prepared, "w") as file:
        file.attrs["manifest"] = json.dumps(manifest)
        for key in ("episode_0", "episode_3"):
            group = file.create_group(key)
            group.attrs.update(task="VideoUnmask", episode_id=key, goal="Synthetic fixture goal")
            cloud = np.zeros((8, 4, 6), dtype=np.float32)
            cloud[..., :3] = [[0, 0, 0.1], [0.02, 0, 0.1], [0, 0.02, 0.1], [0.02, 0.02, 0.1]]
            cloud[..., 3:] = [0.1, 0.4, 0.6]
            cloud[:, :, 0] += np.arange(8, dtype=np.float32)[:, None] / 100
            group["points"] = cloud
            group["valid"] = np.ones((8, 4), dtype=bool)
            proprio = np.zeros((8, 15), dtype=np.float32)
            proprio[:, 10] = 1
            group["proprio"] = proprio
            group["actions"] = np.zeros((8, 8), dtype=np.float32)
            mask = np.ones((8, 8), dtype=bool)
            mask[:3] = False
            group["action_valid"] = mask
            group["source"] = np.asarray([True, True, False, False, False, False, False, False])
            group["source_index"] = np.asarray([0, 1, 0, 1, 2, 3, 4, 5], dtype=np.int64)
            group["raw_index"] = np.arange(8, dtype=np.int64)
            group["completed"] = np.zeros(8, dtype=bool)
    # A selected-task inspection must never open another task's in-progress/corrupt file.
    (prepared.parent / "PatternLock.h5").write_bytes(b"Not a readable HDF5 file")
    alignment = root / "alignment.json"
    alignment.write_text(
        json.dumps(
            {
                "verified": True,
                "convention": "post_action_v1",
                "dataset_revision": "fixture-revision",
                "raw_sha256": {"VideoUnmask": "a" * 64},
            }
        )
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return cfg, config_path, prepared, alignment


def test_selected_raw_episode_cpu_geometry_and_rng_provenance(inspect_data, monkeypatch):
    cfg, _, prepared, alignment = inspect_data
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    threads = torch.get_num_threads()
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda: pytest.fail("Inspector must not initialize CUDA"))
    payload = inspector.inspect_encoder(
        cfg, task="VideoUnmask", episode_key="3", decision_index=2, detail_blocks=1
    )
    assert payload["episode"]["episode_id"] == "episode_3"
    assert payload["episode"]["execution_prefix_frames"] == 2
    assert payload["episode"]["conditioning_frames"] == 2
    assert len([block for block in payload["blocks"] if block["kind"] == "current"]) == 1
    provenance = payload["provenance"]
    assert provenance["encoder_parameters"] == "configuration_defaults"
    assert "not a trained model" in provenance["encoder_label"]
    assert provenance["checkpoint_sha256"] is None
    assert provenance["device"] == "cpu"
    assert provenance["prepared_sha256"] == file_sha256(prepared)
    assert provenance["alignment_sha256"] == file_sha256(alignment)
    assert provenance["alignment_convention"] == "post_action_v1"
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    assert torch.get_num_threads() == threads
    assert json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("style", ["policy", "encoder"])
def test_checkpoint_encoder_weights_and_learned_bandwidth_are_used(inspect_data, tmp_path, style):
    cfg, _, _, _ = inspect_data
    with inspector._diagnostic_runtime(0, 1):
        encoder = inspector._encoder(cfg)
        encoder.visual.raw_tau_x.fill_(-2.0)
        encoder.visual.raw_tau_t.fill_(2.0)
    expected_x, expected_t = float(encoder.visual.tau_x), float(encoder.visual.tau_t)
    checkpoint = tmp_path / f"{style}.pt"
    weights = encoder.state_dict()
    state = {"config": cfg, "stage": "teacher", "step": 2}
    if style == "policy":
        state["model"] = {f"encoder.{key}": value for key, value in weights.items()}
        state["model"]["decoder.unused_for_inspection"] = torch.zeros(1)
    else:
        state["encoder"] = weights
    torch.save(state, checkpoint)
    payload = inspector.inspect_encoder(
        cfg,
        task="VideoUnmask",
        episode_key="episode_0",
        decision_index=2,
        checkpoint=checkpoint,
        detail_blocks=0,
    )
    assert payload["config"]["tau_x"] == pytest.approx(expected_x)
    assert payload["config"]["tau_t"] == pytest.approx(expected_t)
    assert payload["provenance"]["encoder_parameters"] == "checkpoint"
    assert payload["provenance"]["checkpoint_sha256"] == file_sha256(checkpoint)
    assert payload["provenance"]["checkpoint_step"] == 2
    assert payload["provenance"]["checkpoint_debug"] is True


@pytest.mark.parametrize(
    "failure", ["partial", "shape_config", "alignment", "raw_identity", "episode", "decision"]
)
def test_invalid_data_or_selection_is_rejected(inspect_data, tmp_path, failure):
    cfg, _, prepared, alignment = inspect_data
    kwargs = {"task": "VideoUnmask", "episode_key": "0", "decision_index": 2}
    message = ""
    if failure == "partial":
        partial = prepared.with_suffix(".partial.h5")
        partial.symlink_to(prepared)
        kwargs["prepared"] = partial
        message = "in-progress"
    elif failure == "shape_config":
        cfg["model"]["points"] = 8
        message = "differs from configuration"
    elif failure in ("alignment", "raw_identity"):
        evidence = json.loads(alignment.read_text())
        if failure == "alignment":
            evidence["verified"] = False
        else:
            evidence["raw_sha256"]["VideoUnmask"] = "b" * 64
        alignment.write_text(json.dumps(evidence))
        message = "alignment unresolved"
    elif failure == "episode":
        kwargs["episode_key"] = "1"
        message = "episode_1 is absent"
    elif failure == "decision":
        kwargs["decision_index"] = 6
        message = "outside execution frames"
    with pytest.raises((ValueError, RuntimeError), match=message):
        inspector.inspect_encoder(cfg, **kwargs)


@pytest.mark.parametrize("failure", ["config", "weights", "missing_config"])
def test_checkpoint_incompatibility_is_rejected(inspect_data, tmp_path, failure):
    cfg, _, _, _ = inspect_data
    checkpoint_cfg = copy.deepcopy(cfg)
    with inspector._diagnostic_runtime(0, 1):
        weights = inspector._encoder(cfg).state_dict()
    state = {"config": checkpoint_cfg, "encoder": weights}
    if failure == "config":
        checkpoint_cfg["model"]["block_size"] *= 2
        message = "contract mismatch"
    elif failure == "weights":
        state["encoder"] = {}
        message = "Missing key"
    else:
        del state["config"]
        message = "must include its config"
    checkpoint = tmp_path / "invalid.pt"
    torch.save(state, checkpoint)
    with pytest.raises((ValueError, RuntimeError), match=message):
        inspector.inspect_encoder(
            cfg, task="VideoUnmask", episode_key="0", decision_index=2, checkpoint=checkpoint
        )


def test_cli_writes_html_and_complete_adjacent_json_with_explicit_relocated_inputs(
    inspect_data, tmp_path, monkeypatch
):
    from stmem.visualization import structure_html

    cfg, config_path, prepared, alignment = inspect_data
    output = tmp_path / "nested/inspection.html"
    captured = {}

    def write_html(payload, path):
        captured.update(payload)
        Path(path).write_text("<!doctype html><title>Inspector fixture</title>")

    monkeypatch.setattr(structure_html, "write_structure_html", write_html)
    cfg["paths"]["data_root"] = str(tmp_path / "absent-default-root")
    config_path.write_text(yaml.safe_dump(cfg))
    inspector.main(
        [
            "--config",
            str(config_path),
            "--task",
            "VideoUnmask",
            "--episode",
            "3",
            "--decision",
            "2",
            "--detail-blocks",
            "1",
            "--prepared",
            str(prepared),
            "--alignment",
            str(alignment),
            "--output",
            str(output),
            "--threads",
            "1",
        ]
    )
    assert output.exists()
    saved = json.loads(output.with_suffix(".json").read_text())
    assert saved == captured
    assert saved["provenance"]["episode_id"] == "episode_3"
    assert saved["provenance"]["prepared_path"] == str(prepared.resolve())


@pytest.mark.parametrize("extra", [["--threads", "5"], ["--episode", "-1"], ["--decision", "-1"]])
def test_cli_argument_errors_are_clear(inspect_data, tmp_path, extra):
    _, config_path, _, _ = inspect_data
    with pytest.raises(SystemExit) as failure:
        inspector.main(
            [
                "--config",
                str(config_path),
                "--task",
                "VideoUnmask",
                "--output",
                str(tmp_path / "x.html"),
                *extra,
            ]
        )
    assert failure.value.code == 2


@pytest.mark.parametrize(
    "collision",
    [
        "alignment_direct",
        "alignment_symlink",
        "alignment_hardlink",
        "config_symlink",
        "prepared_symlink",
        "prepared_hardlink",
        "checkpoint_symlink",
    ],
)
def test_cli_cannot_overwrite_inputs_through_either_output(inspect_data, tmp_path, monkeypatch, collision):
    import os
    from stmem.visualization import structure_html

    cfg, config_path, prepared, alignment = inspect_data
    output = tmp_path / "inspection.html"
    checkpoint = None
    if collision == "alignment_direct":
        output = alignment.with_suffix(".html")
        source = alignment
    elif collision.startswith("alignment_"):
        source = alignment
        if collision.endswith("symlink"):
            output.with_suffix(".json").symlink_to(source)
        else:
            os.link(source, output.with_suffix(".json"))
    elif collision == "config_symlink":
        source = config_path
        output.with_suffix(".json").symlink_to(source)
    elif collision.startswith("prepared_"):
        source = prepared
        if collision.endswith("symlink"):
            output.symlink_to(source)
        else:
            os.link(source, output)
    else:
        source = checkpoint = tmp_path / "encoder.pt"
        with inspector._diagnostic_runtime(0, 1):
            weights = inspector._encoder(cfg).state_dict()
        torch.save({"config": cfg, "encoder": weights}, checkpoint)
        output.symlink_to(source)
    before = source.read_bytes()
    monkeypatch.setattr(
        structure_html,
        "write_structure_html",
        lambda *args: pytest.fail("Refuse collision before any output write"),
    )
    args = ["--config", str(config_path), "--task", "VideoUnmask", "--decision", "2", "--output", str(output)]
    if checkpoint is not None:
        args.extend(["--checkpoint", str(checkpoint)])
    with pytest.raises(ValueError, match="would overwrite an input"):
        inspector.main(args)
    assert source.read_bytes() == before
