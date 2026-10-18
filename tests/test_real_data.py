"""Optional deterministic regression on the explicitly downloaded small subset."""

from pathlib import Path
import h5py
import pytest
import torch
from stmem.config import load_config
from stmem.data.dataset import EpisodeDataset
from stmem.evaluation.robomme_adapter import RoboMMEAdapter

ROOT = Path("data")
pytestmark = pytest.mark.skipif(
    not (ROOT / "alignment.json").exists() or not (ROOT / "prepared/VideoUnmask.h5").exists(),
    reason="real small subset/alignment audit has not been prepared",
)


def test_resolved_real_alignment_and_offline_online_sensor_parity():
    dataset = EpisodeDataset(ROOT / "prepared", tasks=["VideoUnmask"])
    dataset.validate_preprocessing(load_config("configs/debug/data.yaml"))
    episode = dataset.load_episode(0)
    assert episode.alignment == "post_action_v1" and len(episode.conditioning) == 66
    assert not episode.execution[0].action_available and not episode.action_valid[0].any()
    with h5py.File(ROOT / "raw/record_dataset_VideoUnmask.h5", "r") as f:
        g = f["episode_0"]
        raw = g["timestep_67"]
        assert torch.equal(
            episode.decision(0, 3).target_chunk[0], torch.as_tensor(raw["action/joint_action"][()]).flatten()
        )
        assert torch.equal(episode.execution[1].previous_action, episode.actions[1])
        # Policy receives exactly the same allowlisted sensors via online transport.
        adapter = RoboMMEAdapter("VideoUnmask", env=object(), points=256)
        adapter.calibration = {
            f"{c}_camera_intrinsic": g["setup"][f"{c}_camera_intrinsic"][()] for c in ("front", "wrist")
        }
        online = {
            k + "_list": [raw["obs"][k][()]]
            for k in (
                "front_rgb",
                "wrist_rgb",
                "front_depth",
                "wrist_depth",
                "front_camera_extrinsic",
                "wrist_camera_extrinsic",
                "joint_state",
                "eef_state",
                "gripper_state",
            )
        }
        frame = adapter.ingest(online, {}, event_id="regression", command=episode.actions[1])[0]
        assert torch.equal(frame.points, episode.execution[1].points)
        assert torch.equal(frame.valid, episode.execution[1].valid)
        assert torch.equal(frame.proprio, episode.execution[1].proprio)
        assert torch.equal(frame.previous_action, episode.execution[1].previous_action)
    # Terminal suffixes are unavailable targets, while the first completing command remains valid.
    assert episode.action_valid[161, :7].all()
    assert not episode.action_valid[162:].any()
    assert not episode.decision(161, 8).target_valid.any()
