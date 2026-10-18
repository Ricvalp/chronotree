"""Prepare explicitly limited episodes from whole task HDF5 archives."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import time
import h5py
import numpy as np
from .download import TASKS, REVISION, sha256
from .pointcloud import DEFAULT_CROP, CALIBRATION_VERSION, rgbd_to_points, proprioception

FORMAT_VERSION = "canonical_v1"


def numeric_keys(group, prefix):
    keys = [k for k in group if re.fullmatch(re.escape(prefix) + r"\d+", k)]
    return sorted(keys, key=lambda k: int(k[len(prefix) :]))


def text_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_goal(setup):
    raw = np.asarray(setup["task_goal"][()]).reshape(-1)
    return text_value(raw[0]) if len(raw) else ""


def raw_episode_metadata(group):
    steps = numeric_keys(group, "timestep_")
    demos = [bool(np.asarray(group[k]["info"]["is_video_demo"][()]).reshape(-1)[0]) for k in steps]
    if any(demos[i] and not demos[i - 1] for i in range(1, len(demos))):
        raise ValueError("conditioning resumes after execution; unsupported source boundary")
    completed = [bool(np.asarray(group[k]["info"].get("is_completed", False)).reshape(-1)[0]) for k in steps]
    return steps, demos, completed


def prepare_task(
    raw_path: Path,
    output: Path,
    task: str,
    episodes: int | None,
    *,
    points=1024,
    crop=DEFAULT_CROP,
    voxel_size=0.005,
    max_depth_m=4.0,
    dataset_revision=REVISION,
    expected_raw_sha256=None,
):
    started = time.monotonic()
    raw_hash = sha256(raw_path)
    if expected_raw_sha256 is not None and raw_hash != expected_raw_sha256:
        raise ValueError(f"Extracted HDF5 failed its download-manifest SHA256: {raw_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.h5")
    report = {
        "format": FORMAT_VERSION,
        "task": task,
        "dataset_revision": dataset_revision,
        "raw_sha256": raw_hash,
        "raw_path": str(raw_path.resolve()),
        "points": points,
        "crop": crop,
        "voxel_size": voxel_size,
        "max_depth_m": max_depth_m,
        "calibration": CALIBRATION_VERSION,
        "clock": "source_local_recorded_frame",
        "physical_continuity_verified": False,
        "nominal_control_hz": 20,
        "alignment": "unresolved",
        "episodes": [],
    }
    with h5py.File(raw_path, "r") as src, h5py.File(temporary, "w") as dst:
        keys = numeric_keys(src, "episode_")
        if episodes is not None:
            keys = keys[:episodes]
        for key in keys:
            raw = src[key]
            setup = raw["setup"]
            steps, demos, completed = raw_episode_metadata(raw)
            calib = {f"{c}_camera_intrinsic": setup[f"{c}_camera_intrinsic"][()] for c in ("front", "wrist")}
            group = dst.create_group(key)
            group.attrs.update(goal=read_goal(setup), task=task, episode_id=key)
            n = len(steps)
            for name, shape, dtype in [
                ("points", (n, points, 6), "f4"),
                ("valid", (n, points), "?"),
                ("proprio", (n, 15), "f4"),
                ("actions", (n, 8), "f4"),
                ("action_valid", (n, 8), "?"),
            ]:
                group.create_dataset(
                    name,
                    shape=shape,
                    dtype=dtype,
                    compression="gzip",
                    compression_opts=1,
                    chunks=(1, *shape[1:]),
                )
            group.create_dataset("source", data=np.asarray(demos, dtype=bool))
            group.create_dataset("completed", data=np.asarray(completed, dtype=bool))
            group.create_dataset("raw_index", data=[int(k[9:]) for k in steps])
            seen_complete = False
            source_index = {"conditioning": 0, "execution": 0}
            source_indices = []
            for i, key_step in enumerate(steps):
                step = raw[key_step]
                obs = {
                    k: step["obs"][k][()]
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
                cloud, valid = rgbd_to_points(
                    obs, calib, count=points, crop=crop, voxel_size=voxel_size, max_depth_m=max_depth_m
                )
                group["points"][i] = cloud.numpy()
                group["valid"][i] = valid.numpy()
                group["proprio"][i] = proprioception(obs).numpy()
                action = np.zeros(8, dtype=np.float32)
                mask = np.zeros(8, dtype=bool)
                try:
                    a = np.asarray(step["action"]["joint_action"][()], dtype=np.float32).reshape(-1)
                    if len(a) not in (7, 8):
                        raise ValueError("joint action requires 7 or 8 dimensions")
                    action[: len(a)] = a
                    mask[: len(a)] = np.isfinite(a)
                except (ValueError, KeyError, TypeError):
                    pass
                if demos[i] or seen_complete:
                    mask[:] = False
                if not demos[i] and (i == 0 or demos[i - 1]):
                    mask[:] = False  # reset's automatic initial command
                if (
                    i > 0
                    and not demos[i]
                    and not demos[i - 1]
                    and int(key_step[9:]) - int(steps[i - 1][9:]) != 1
                ):
                    mask[:] = False
                if task in ("PatternLock", "RouteStick"):
                    mask[7] = False
                    action[7] = 0
                # The first completed observation follows its final valid command.
                if not demos[i] and completed[i]:
                    seen_complete = True
                action[~np.isfinite(action)] = 0
                group["actions"][i] = action
                group["action_valid"][i] = mask
                source = "conditioning" if demos[i] else "execution"
                source_indices.append(source_index[source])
                source_index[source] += 1
            group.create_dataset("source_index", data=source_indices)
            episode_report = {
                "episode_id": key,
                "conditioning_frames": sum(demos),
                "execution_frames": len(demos) - sum(demos),
                "first_completed_raw_index": next(
                    (int(steps[i][9:]) for i, c in enumerate(completed) if c), None
                ),
                "terminal_suffix_records": sum(1 for i, c in enumerate(completed) if c) - 1
                if any(completed)
                else 0,
                "seed": int(setup["seed"][()]),
                "difficulty": text_value(setup["difficulty"][()]),
                "raw_indices_contiguous": all(int(b[9:]) - int(a[9:]) == 1 for a, b in zip(steps, steps[1:])),
            }
            report["episodes"].append(episode_report)
            print(json.dumps(episode_report), flush=True)
        dst.attrs["manifest"] = json.dumps(report)
    temporary.replace(output)
    report.update(
        elapsed_seconds=time.monotonic() - started,
        prepared_bytes=output.stat().st_size,
        output=str(output.resolve()),
    )
    output.with_suffix(".manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", choices=TASKS)
    p.add_argument("--config")
    p.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Read only this many episodes per task (archive download is still whole-task)",
    )
    p.add_argument("--data-root", default=os.getenv("STMEM_DATA_ROOT", "data"))
    p.add_argument("--points", type=int)
    p.add_argument("--voxel-size", type=float)
    args = p.parse_args(argv)
    if args.episodes is not None and args.episodes < 1:
        p.error("--episodes must be positive")
    config = {}
    if args.config:
        from stmem.config import load_config

        config = load_config(args.config)
        args.data_root = config["paths"]["data_root"]
        args.tasks = args.tasks or config["data"]["tasks"]
        if args.episodes is None:
            args.episodes = config["data"]["episodes_per_task"]
    if not args.tasks:
        p.error("--tasks or --config is required")
    geometry = config.get("geometry", {})
    points = args.points or config.get("model", {}).get("points", 1024)
    voxel_size = args.voxel_size if args.voxel_size is not None else geometry.get("voxel_size", 0.005)
    root = Path(args.data_root)
    for task in args.tasks:
        candidates = list((root / "raw").rglob(f"record_dataset_{task}.h5"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one raw file for {task}, found {candidates}")
        manifest_path = root / "manifests" / f"{task}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing download integrity manifest {manifest_path}; use download --archive PATH to verify an existing archive"
            )
        download_manifest = json.loads(manifest_path.read_text())
        revision = download_manifest["revision"]
        expected_raw_sha256 = next(
            (
                member["sha256"]
                for member in download_manifest.get("members", [])
                if Path(member["path"]).name == candidates[0].name and member.get("sha256")
            ),
            None,
        )
        print(
            json.dumps(
                prepare_task(
                    candidates[0],
                    root / "prepared" / f"{task}.h5",
                    task,
                    args.episodes,
                    points=points,
                    voxel_size=voxel_size,
                    crop=geometry.get("crop", DEFAULT_CROP),
                    max_depth_m=geometry.get("max_depth", 4.0),
                    dataset_revision=revision,
                    expected_raw_sha256=expected_raw_sha256,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
