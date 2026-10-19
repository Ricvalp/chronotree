"""Prepared-file reuse requires exact identity, complete layout, and one writer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from stmem.data.preparation_state import preparation_lock, verify_prepared


@pytest.fixture
def prepared(tmp_path):
    output = tmp_path / "VideoUnmask.h5"
    expected = {
        "format": "canonical_v1",
        "task": "VideoUnmask",
        "dataset_revision": "revision-a",
        "raw_sha256": "a" * 64,
        "raw_path": "/old/location/raw.h5",
        "points": 3,
        "crop": ((-1, -1, 0), (1, 1, 1)),
        "voxel_size": 0.005,
        "max_depth_m": 4.0,
        "calibration": "cv_world_to_camera_mm_v1",
        "clock": "source_local_recorded_frame",
        "physical_continuity_verified": False,
        "nominal_control_hz": 20,
        "alignment": "unresolved",
        "episodes": [],
    }
    embedded = copy.deepcopy(expected)
    embedded["episodes"] = [
        {
            "episode_id": "episode_0",
            "conditioning_frames": 2,
            "execution_frames": 2,
            "first_completed_raw_index": 4,
            "terminal_suffix_records": 0,
            "seed": 0,
            "difficulty": "easy",
            "raw_indices_contiguous": False,
        }
    ]
    with h5py.File(output, "w") as file:
        file.attrs["manifest"] = json.dumps(embedded)
        group = file.create_group("episode_0")
        group.attrs.update(task="VideoUnmask", episode_id="episode_0", goal="Remember the object")
        for name, shape, dtype in (
            ("points", (4, 3, 6), "f4"),
            ("valid", (4, 3), "?"),
            ("proprio", (4, 15), "f4"),
            ("actions", (4, 8), "f4"),
            ("action_valid", (4, 8), "?"),
        ):
            group.create_dataset(name, shape=shape, dtype=dtype)
        group.create_dataset("source", data=np.asarray([True, True, False, False]))
        group.create_dataset("completed", data=np.asarray([False, False, False, True]))
        group.create_dataset("raw_index", data=np.asarray([0, 1, 2, 4], dtype=np.int64))
        group.create_dataset("source_index", data=np.asarray([0, 1, 0, 1], dtype=np.int64))
    return output, expected, {"episode_0": 4}


def test_matching_file_reusable_after_relocation_without_sidecar_or_payload_reads(prepared, monkeypatch):
    output, expected, layout = prepared
    expected["raw_path"] = "/new/location/raw.h5"
    snapshot = output.read_bytes()
    original = h5py.Dataset.__getitem__
    payloads = {"points", "valid", "proprio", "actions", "action_valid"}

    def forbid_payload(self, *args, **kwargs):
        assert Path(self.name).name not in payloads, "Reuse checking must not read point-cloud/robot payloads"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbid_payload)
    report = verify_prepared(output, expected, layout)
    assert report["raw_path"] == "/old/location/raw.h5"
    assert report["episodes"][0]["execution_frames"] == 2
    assert report["alignment"] == "unresolved"
    assert not output.with_suffix(".manifest.json").exists()
    assert output.read_bytes() == snapshot


@pytest.mark.parametrize(
    "key,value",
    [
        ("format", "canonical_v2"),
        ("task", "VideoRepick"),
        ("dataset_revision", "revision-b"),
        ("raw_sha256", "b" * 64),
        ("points", 1024),
        ("crop", [[-2, -1, 0], [1, 1, 1]]),
        ("voxel_size", 0.0050000001),
        ("max_depth_m", 3.0),
        ("calibration", "changed"),
        ("clock", "primitive_commands"),
        ("physical_continuity_verified", True),
        ("nominal_control_hz", 30),
        ("alignment", "post_action_v1"),
    ],
)
def test_changed_identity_cannot_reuse_diagnostic_file(prepared, key, value):
    output, expected, layout = prepared
    original = output.read_bytes()
    expected[key] = value
    assert verify_prepared(output, expected, layout) is None
    assert output.read_bytes() == original


@pytest.mark.parametrize("layout", [{"episode_0": 5}, {"episode_0": 4, "episode_1": 4}, {"episode_1": 4}])
def test_exact_episode_subset_and_frame_counts_are_required(prepared, layout):
    output, expected, _ = prepared
    assert verify_prepared(output, expected, layout) is None


@pytest.mark.parametrize(
    "missing",
    [
        "points",
        "valid",
        "proprio",
        "actions",
        "action_valid",
        "source",
        "completed",
        "raw_index",
        "source_index",
    ],
)
def test_missing_required_dataset_rebuilds(prepared, missing):
    output, expected, layout = prepared
    with h5py.File(output, "r+") as file:
        del file["episode_0"][missing]
    assert verify_prepared(output, expected, layout) is None


@pytest.mark.parametrize(
    "name,shape,dtype",
    [
        ("points", (4, 3, 6), "f8"),
        ("valid", (4, 3), "u1"),
        ("proprio", (4, 14), "f4"),
        ("actions", (4, 7), "f4"),
        ("action_valid", (4, 8), "i1"),
        ("source_index", (4,), "i4"),
    ],
)
def test_wrong_shape_or_dtype_rebuilds(prepared, name, shape, dtype):
    output, expected, layout = prepared
    with h5py.File(output, "r+") as file:
        group = file["episode_0"]
        del group[name]
        group.create_dataset(name, shape=shape, dtype=dtype)
    assert verify_prepared(output, expected, layout) is None


@pytest.mark.parametrize(
    "change",
    [
        "missing_manifest",
        "invalid_json",
        "duplicate_episode_report",
        "wrong_report_count",
        "wrong_completion",
        "missing_goal",
        "wrong_task",
        "conditioning_resumes",
        "source_clock",
        "raw_clock",
        "extra_episode",
        "soft_link",
    ],
)
def test_malformed_or_incomplete_state_rebuilds(prepared, change):
    output, expected, layout = prepared
    with h5py.File(output, "r+") as file:
        group = file["episode_0"]
        manifest = json.loads(file.attrs["manifest"])
        if change == "missing_manifest":
            del file.attrs["manifest"]
        elif change == "invalid_json":
            file.attrs["manifest"] = "{unfinished"
        elif change == "duplicate_episode_report":
            manifest["episodes"].append(manifest["episodes"][0])
            file.attrs["manifest"] = json.dumps(manifest)
        elif change == "wrong_report_count":
            manifest["episodes"][0]["execution_frames"] = 3
            file.attrs["manifest"] = json.dumps(manifest)
        elif change == "wrong_completion":
            manifest["episodes"][0]["first_completed_raw_index"] = 2
            file.attrs["manifest"] = json.dumps(manifest)
        elif change == "missing_goal":
            del group.attrs["goal"]
        elif change == "wrong_task":
            group.attrs["task"] = "VideoRepick"
        elif change == "conditioning_resumes":
            group["source"][:] = [True, False, True, False]
        elif change == "source_clock":
            group["source_index"][:] = [0, 1, 2, 3]
        elif change == "raw_clock":
            group["raw_index"][:] = [0, 1, 1, 4]
        elif change == "extra_episode":
            file.create_group("episode_1")
        elif change == "soft_link":
            del group["action_valid"]
            group["action_valid"] = h5py.SoftLink("/episode_0/valid")
    assert verify_prepared(output, expected, layout) is None


def test_missing_unreadable_or_truncated_output_is_not_modified(prepared, tmp_path):
    _, expected, layout = prepared
    output = tmp_path / "missing.h5"
    assert verify_prepared(output, expected, layout) is None
    output.write_bytes(b"not a complete HDF5 file")
    assert verify_prepared(output, expected, layout) is None
    assert output.read_bytes() == b"not a complete HDF5 file"
    assert verify_prepared(tmp_path, expected, layout) is None


def test_incomplete_expected_identity_is_a_caller_error(prepared):
    output, expected, layout = prepared
    del expected["raw_sha256"]
    with pytest.raises(ValueError, match="identity"):
        verify_prepared(output, expected, layout)


@pytest.mark.parametrize("stale_field", ["output", "prepared_bytes"])
def test_skip_repairs_relocated_sidecar_then_preserves_matching_files(prepared, tmp_path, stale_field):
    from stmem.data.download import sha256
    from stmem.data.prepare import prepare_task

    output, expected, _ = prepared
    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as file:
        episode = file.create_group("episode_0")
        for index in (0, 1, 2, 4):
            episode.create_group(f"timestep_{index}")
    raw_hash = sha256(raw)
    with h5py.File(output, "r+") as file:
        embedded = json.loads(file.attrs["manifest"])
        embedded["raw_sha256"] = raw_hash
        file.attrs["manifest"] = json.dumps(embedded)
    sidecar = output.with_suffix(".manifest.json")
    stale = {**embedded, "output": str(output.resolve()), "prepared_bytes": output.stat().st_size}
    stale[stale_field] = "/previous/location/VideoUnmask.h5" if stale_field == "output" else 1
    sidecar.write_text(json.dumps(stale))
    original_hdf5 = (output.read_bytes(), output.stat().st_mtime_ns)

    def reuse():
        return prepare_task(
            raw,
            output,
            "VideoUnmask",
            1,
            points=expected["points"],
            crop=expected["crop"],
            voxel_size=expected["voxel_size"],
            max_depth_m=expected["max_depth_m"],
            dataset_revision=expected["dataset_revision"],
            expected_raw_sha256=raw_hash,
            skip_existing=True,
            progress=lambda event: None,
        )

    result = reuse()
    repaired = json.loads(sidecar.read_text())
    assert result["skipped"] is True
    assert repaired["output"] == result["output"] == str(output.resolve())
    assert repaired["prepared_bytes"] == result["prepared_bytes"] == output.stat().st_size
    assert all(repaired[key] == value for key, value in embedded.items())
    assert (output.read_bytes(), output.stat().st_mtime_ns) == original_hdf5
    # Extra valid report fields are retained when every required value matches.
    repaired["elapsed_seconds"] = 12.5
    sidecar.write_text(json.dumps(repaired))
    matching_sidecar = (sidecar.read_bytes(), sidecar.stat().st_mtime_ns)
    assert reuse()["skipped"] is True
    assert (sidecar.read_bytes(), sidecar.stat().st_mtime_ns) == matching_sidecar
    assert (output.read_bytes(), output.stat().st_mtime_ns) == original_hdf5


def test_lock_prevents_second_writer_and_releases_after_exception(tmp_path):
    output = tmp_path / "prepared/task.h5"
    child = (
        "from pathlib import Path\n"
        "import sys\n"
        "from stmem.data.preparation_state import preparation_lock\n"
        "try:\n"
        "    with preparation_lock(Path(sys.argv[1])): pass\n"
        "except RuntimeError as error:\n"
        "    print(error)\n"
        "    raise SystemExit(23)\n"
    )
    with pytest.raises(ValueError, match="writer failed"):
        with preparation_lock(output) as lock_path:
            inode = lock_path.stat().st_ino
            result = subprocess.run(
                [sys.executable, "-c", child, str(output)], capture_output=True, text=True, timeout=10
            )
            assert result.returncode == 23
            assert "Another preparation writer" in result.stdout
            assert output.name + ".lock" in result.stdout
            with pytest.raises(RuntimeError, match="output lock"):
                with preparation_lock(output):
                    pass
            raise ValueError("writer failed")
    assert lock_path.is_file() and lock_path.stat().st_ino == inode
    with preparation_lock(output) as reacquired:
        assert reacquired.stat().st_ino == inode
    assert not output.exists()


def test_crashed_process_releases_os_lock_without_removing_lockfile(tmp_path):
    output = tmp_path / "task.h5"
    child = (
        "from pathlib import Path\n"
        "import os, sys\n"
        "from stmem.data.preparation_state import preparation_lock\n"
        "with preparation_lock(Path(sys.argv[1])):\n"
        "    os._exit(0)\n"
    )
    subprocess.run([sys.executable, "-c", child, str(output)], check=True, timeout=10)
    with preparation_lock(output) as lock_path:
        assert lock_path.is_file()
