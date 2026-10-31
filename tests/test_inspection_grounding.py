"""CPU safeguards for recurring physical-cube and native-cadence diagnostics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stmem.inspection import grounding
from stmem.inspection.grounding_helpers import cube_mask, verified_existing
from stmem.training.checkpoints import file_sha256


def test_cube_labels_respect_orientation_and_do_not_mutate_points():
    # A non-cubic box makes a ninety-degree rotation observable.
    cube = dict(p=[1, 2, 3], q=[2**-0.5, 0, 0, 2**-0.5], half_size=[0.2, 0.05, 0.05])
    points = np.array([[1, 2.15, 3, 0, 0, 1], [1.15, 2, 3, 0, 0, 1], [1, 2, 3.1, 0, 0, 1]])
    original = points.copy()
    np.testing.assert_array_equal(cube_mask(points, cube, tolerance=0), [True, False, False])
    np.testing.assert_array_equal(points, original)


def test_10hz_command_expansion_holds_joint_targets_instead_of_interpolating():
    raw = np.zeros((16, 8), dtype=np.float32)
    raw[:, 0] = np.linspace(-0.1, 0.1, 16)
    raw[:, 3], raw[:, 5], raw[:, 7] = -1.6, 1.5, 1
    result = grounding.physical_prediction(raw, {"model": {"action_stride": 2}})
    assert result["native_commands"].shape == (32, 8)
    np.testing.assert_array_equal(result["native_commands"][::2], result["physical_actions"])
    np.testing.assert_array_equal(result["native_commands"][1::2], result["physical_actions"])
    np.testing.assert_array_equal(result["native_tcp"][::2], result["tcp_targets"])
    np.testing.assert_array_equal(result["first4_native_tcp"], result["native_tcp"][:4])
    assert not np.array_equal(result["native_tcp"][1], result["native_tcp"][2])


def test_noise_matches_absolute_native_replanning_time():
    policy = SimpleNamespace(device=torch.device("cpu"), config={"model": {"horizon": 16}})
    generator = torch.Generator().manual_seed(2701)
    for step in range(0, 84, 4):
        expected = torch.randn(16, 8, generator=generator)
        assert torch.equal(grounding.fixed_noise(policy, step, 2701), expected)
    assert not torch.equal(grounding.fixed_noise(policy, 0, 2701), grounding.fixed_noise(policy, 0, 2702))


def test_resume_requires_completed_matching_identity(tmp_path):
    path = tmp_path / "result.json"
    identity = {"inspection_manifest_sha256": "manifest-a", "checkpoint_sha256": "checkpoint-a"}
    assert verified_existing(path, identity, resume=True) is None
    path.write_text(json.dumps(dict(identity, diagnostic_complete=False)))
    assert verified_existing(path, identity, resume=True) is None
    with pytest.raises(FileExistsError):
        verified_existing(path, identity, resume=False)
    path.write_text(json.dumps(dict(identity, diagnostic_complete=True)))
    assert verified_existing(path, identity, resume=True)["diagnostic_complete"]
    with pytest.raises(ValueError, match="identity mismatch"):
        verified_existing(path, dict(identity, checkpoint_sha256="checkpoint-b"), resume=True)


def test_reset_resume_recovers_interruption_after_generator_completion(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"diagnostic fixture")
    sha = file_sha256(checkpoint)
    manifest = dict(
        output=str(tmp_path),
        device="cpu",
        assets={},
        checkpoints={
            "model": dict(path=str(checkpoint), sha256=sha, step=1, config={"model": {}, "data": {}})
        },
    )
    directory = tmp_path / "reset_cube_chunks/model"
    directory.mkdir(parents=True)
    identity = dict(inspection_manifest_sha256="manifest", checkpoint_sha256=sha)
    (directory / "job.json").write_text(json.dumps(identity))
    (directory / "payload.json").write_text(json.dumps({"models": [{"checkpoint_sha256": sha}]}))
    calls = []

    def generate(args):
        calls.append(args)
        return {"models": [{"checkpoint_sha256": sha}]}

    monkeypatch.setattr(grounding, "reset_generator", lambda: SimpleNamespace(generate=generate))
    result = grounding.run_reset(manifest, "model", "manifest", resume=True)
    assert result["diagnostic_complete"] and len(calls) == 1
    assert result["observation_stride"] == result["action_stride"] == 1
    grounding.run_reset(manifest, "model", "manifest", resume=True)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="identity mismatch"):
        grounding.run_reset(manifest, "model", "other-manifest", resume=True)
