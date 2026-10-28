"""Attach frozen CLIP descriptors to existing prepared XYZRGB points.

This command never changes canonical geometry or its alignment evidence. It
recovers validated raw camera/pixel correspondences for the existing prepared
points and writes separate, resumable descriptor/provenance sidecars.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from stmem.config import load_config, require_sampling_contract
from stmem.training.checkpoints import atomic_json
from stmem.vision.frozen import (
    FrozenCLIPFeatures,
    resolve_visual_features,
    visual_feature_identity,
    visual_feature_signature,
)
from .download import sha256
from .feature_cache import (
    FEATURE_CACHE_VERSION,
    FEATURE_DIM,
    feature_cache_manifest,
    prepared_feature_identity,
)
from .pointcloud import canonical_point_provenance, point_sampler_settings
from .prepare import FORMAT_VERSION, numeric_keys
from .preparation_state import preparation_lock


def _emit(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def _layout(prepared, episodes, max_frames):
    keys = numeric_keys(prepared, "episode_")
    if episodes is not None:
        if episodes < 1 or len(keys) < episodes:
            raise ValueError(f"Requested {episodes} episodes but prepared data contains {len(keys)}")
        keys = keys[:episodes]
    result = {}
    remaining = max_frames
    for key in keys:
        rows = len(prepared[key]["source"])
        if remaining is not None:
            rows = min(rows, remaining)
            remaining -= rows
        if rows:
            result[key] = rows
        if remaining == 0:
            break
    if not result:
        raise ValueError("No prepared frames were selected")
    return result


def _check_config_geometry(manifest, config):
    geometry = config.get("geometry", {})
    for name, expected in (
        ("points", config["model"]["points"]),
        ("crop", geometry.get("crop", [[-0.9, -0.75, -0.05], [0.75, 0.75, 1.25]])),
        ("voxel_size", geometry.get("voxel_size", 0.005)),
        ("max_depth_m", geometry.get("max_depth", 4.0)),
        ("point_sampling", point_sampler_settings(config)),
    ):
        actual = manifest.get(name)
        # JSON normalizes tuples and lists without approximate geometry checks.
        if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
            raise ValueError(f"Prepared {name} differs from feature config; use matching canonical geometry")
    if manifest.get("format") != FORMAT_VERSION:
        raise ValueError("Frozen features require the explicit balanced-v2 canonical geometry format")


def estimate_feature_cache(prepared_path, config, *, episodes=None, max_frames=None):
    """Read only manifests/layouts; no RGB-D decoding, weights or GPU required."""
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    with h5py.File(prepared_path, "r") as prepared:
        manifest = json.loads(prepared.attrs["manifest"])
        _check_config_geometry(manifest, config)
        layout = _layout(prepared, episodes, max_frames)
    frames, points = sum(layout.values()), int(manifest["points"])
    # Descriptor FP16 + source camera int8 + original UV int16, plus three small
    # frame metadata arrays. HDF5 headers add a small, measured overhead.
    bytes_per_frame = points * (FEATURE_DIM * 2 + 1 + 2 * 2) + 8 + 8 + 1
    return {
        "task": manifest["task"],
        "episodes": layout,
        "frames": frames,
        "descriptor_bytes_per_frame": points * FEATURE_DIM * 2,
        "estimated_bytes": frames * bytes_per_frame,
        "estimated_gib": frames * bytes_per_frame / 2**30,
        "diagnostic": max_frames is not None,
    }


def _frame_provenance(canonical, raw_episode, row, calibration, manifest):
    """Attach raw pixels to immutable canonical rows, never a newly sampled cloud."""
    raw_index = int(canonical["raw_index"][row])
    sensors = raw_episode[f"timestep_{raw_index}"]["obs"]
    observation = {
        name: sensors[name][()]
        for name in (
            "front_rgb",
            "wrist_rgb",
            "front_depth",
            "wrist_depth",
            "front_camera_extrinsic",
            "wrist_camera_extrinsic",
        )
    }
    points = torch.from_numpy(canonical["points"][row])
    valid = torch.from_numpy(canonical["valid"][row])
    try:
        camera, uv = canonical_point_provenance(
            observation,
            calibration,
            points,
            valid,
            max_depth_m=manifest["max_depth_m"],
            voxel_size=manifest["voxel_size"],
        )
    except ValueError as error:
        raise ValueError(
            f"Cannot recover raw pixels for canonical {manifest['task']}/{canonical.name.rsplit('/', 1)[-1]}"
            f"/{raw_index}: {error}; feature preparation is not allowed to change geometry"
        ) from error
    return observation, valid, camera, uv


def check_feature_geometry(prepared_path, raw_path, config, *, episodes=None, max_frames):
    """Bounded read-only pixel check; no CLIP, descriptor writes or whole-file hashing."""
    if max_frames is None or max_frames < 1:
        raise ValueError("Read-only geometry checks require a positive max_frames bound")
    estimate = estimate_feature_cache(prepared_path, config, episodes=episodes, max_frames=max_frames)
    started = time.monotonic()
    checked = 0
    with h5py.File(prepared_path, "r") as prepared, h5py.File(raw_path, "r") as raw:
        manifest = json.loads(prepared.attrs["manifest"])
        for key, count in estimate["episodes"].items():
            calibration = {
                f"{camera}_camera_intrinsic": raw[key]["setup"][f"{camera}_camera_intrinsic"][()]
                for camera in ("front", "wrist")
            }
            for row in range(count):
                _frame_provenance(prepared[key], raw[key], row, calibration, manifest)
                checked += 1
    report = {
        "event": "feature_geometry_checked",
        "task": manifest["task"],
        "frames_checked": checked,
        "elapsed_seconds": time.monotonic() - started,
        "raw_checksum_verified": False,
        "descriptors_written": False,
    }
    _emit(report)
    return report


def prepare_feature_task(
    prepared_path,
    raw_path,
    output,
    config,
    *,
    episodes=None,
    max_frames=None,
    device="cuda",
    skip_existing=False,
    extractor=None,
    progress=None,
):
    """One task, one writer, atomic completion; matching partial files resume.

    ``max_frames`` creates a clearly diagnostic sidecar which the training
    dataset refuses. A complete small-episode preparation is reusable only for
    those episodes; missing requested training episodes fail during indexing.
    ``extractor`` is an injection point for bounded synthetic invariant tests.
    """
    prepared_path, raw_path, output = map(Path, (prepared_path, raw_path, output))
    if output.resolve() in (prepared_path.resolve(), raw_path.resolve()):
        raise ValueError("Frozen descriptors must use a separate output path")
    if output.name.endswith(".partial.h5"):
        raise ValueError("Choose a final .h5 output; partial filenames are managed by preparation")
    spec = resolve_visual_features(config)
    if spec["mode"] == "xyzrgb":
        raise ValueError("Feature preparation requires model.visual_features.mode: clip_rn50_layer1")
    progress = progress or _emit
    started = time.monotonic()
    estimate = estimate_feature_cache(prepared_path, config, episodes=episodes, max_frames=max_frames)
    progress({"event": "feature_cache_estimate", **estimate, "output": str(output)})
    with h5py.File(prepared_path, "r") as prepared:
        prepared_manifest = json.loads(prepared.attrs["manifest"])
    manifest = {
        "format": FEATURE_CACHE_VERSION,
        "task": prepared_manifest["task"],
        "prepared_identity": prepared_feature_identity(prepared_manifest),
        "raw_sha256": prepared_manifest["raw_sha256"],
        "feature_identity": visual_feature_identity(spec),
        "feature_signature": visual_feature_signature(spec),
        "episodes": estimate["episodes"],
        "points": prepared_manifest["points"],
        "diagnostic": max_frames is not None,
        "complete": False,
    }
    # Integrity is checked even when reusing a completed cache. This may read a
    # large raw file but never loads it into RAM or rebuilds training geometry.
    last_hash = started

    def hash_progress(done):
        nonlocal last_hash
        now = time.monotonic()
        if now - last_hash >= 10:
            progress(
                {
                    "event": "feature_raw_hash",
                    "task": manifest["task"],
                    "bytes_done": done,
                    "bytes_total": raw_path.stat().st_size,
                }
            )
            last_hash = now

    if sha256(raw_path, progress=hash_progress) != manifest["raw_sha256"]:
        raise ValueError("Raw RGB-D checksum differs from the prepared geometry's source")
    temporary = output.with_suffix(".partial.h5")
    with preparation_lock(output):
        if skip_existing and output.exists() and not manifest["diagnostic"]:
            cached = feature_cache_manifest(output, prepared_manifest, spec)
            if cached["episodes"] != manifest["episodes"]:
                raise ValueError(
                    "Existing feature cache episode selection differs; choose another output directory"
                )
            report = {
                **cached,
                "skipped": True,
                "output": str(output.resolve()),
                "bytes": output.stat().st_size,
                "elapsed_seconds": time.monotonic() - started,
            }
            progress({"event": "feature_task_skipped", **report})
            return report
        if output.exists():
            with h5py.File(output, "r") as existing:
                old = json.loads(existing.attrs["manifest"])
            if any(old.get(key) != value for key, value in manifest.items() if key != "complete"):
                raise ValueError("Refusing to overwrite an incompatible feature cache; use a new cache_dir")
            raise FileExistsError(
                f"Completed feature caches are immutable: {output}; use --skip-existing to verify/reuse "
                "or choose a new output directory"
            )
        mode = "r+" if temporary.exists() else "w"
        with h5py.File(temporary, mode) as dst:
            if mode == "r+":
                previous = json.loads(dst.attrs["manifest"])
                if any(previous.get(key) != value for key, value in manifest.items() if key != "complete"):
                    raise ValueError(f"Partial feature cache has a different identity/selection: {temporary}")
            else:
                dst.attrs["manifest"] = json.dumps(manifest, sort_keys=True)
            extractor = extractor if extractor is not None else FrozenCLIPFeatures(spec, device=device)
            processed = sum(
                int(dst[key].attrs.get("completed_rows", 0)) for key in manifest["episodes"] if key in dst
            )
            resumed = processed
            extraction_started = last_progress = time.monotonic()
            with h5py.File(prepared_path, "r") as prepared, h5py.File(raw_path, "r") as raw:
                for key, count in manifest["episodes"].items():
                    canonical, raw_episode = prepared[key], raw[key]
                    calibration = {
                        f"{camera}_camera_intrinsic": raw_episode["setup"][f"{camera}_camera_intrinsic"][()]
                        for camera in ("front", "wrist")
                    }
                    group = dst.require_group(key)
                    required = {"features", "point_camera", "point_uv", "raw_index", "source_index", "source"}
                    if set(group) != required or "completed_rows" not in group.attrs:
                        if int(group.attrs.get("completed_rows", 0)):
                            raise ValueError(
                                f"Committed partial episode has missing datasets: {temporary}:{key}"
                            )
                        # A writer interrupted during dataset creation has no
                        # committed rows; reconstruct that empty episode safely.
                        del dst[key]
                        group = dst.create_group(key)
                    if "features" not in group:
                        group.create_dataset(
                            "features", shape=(count, manifest["points"], FEATURE_DIM), dtype="<f2"
                        )
                        group.create_dataset("point_camera", shape=(count, manifest["points"]), dtype="i1")
                        group.create_dataset("point_uv", shape=(count, manifest["points"], 2), dtype="i2")
                        for name in ("raw_index", "source_index", "source"):
                            group.create_dataset(name, data=canonical[name][:count])
                        group.attrs["completed_rows"] = 0
                        dst.flush()
                    start = int(group.attrs["completed_rows"])
                    if not 0 <= start <= count:
                        raise ValueError(f"Invalid partial completion counter in {temporary}:{key}")
                    for row in range(start, count):
                        observation, valid, camera, uv = _frame_provenance(
                            canonical, raw_episode, row, calibration, prepared_manifest
                        )
                        features = (
                            extractor.sample(observation, camera, uv, valid)
                            .detach()
                            .to(device="cpu", dtype=torch.float16)
                        )
                        if features.shape != (manifest["points"], FEATURE_DIM):
                            raise ValueError("Frozen extractor returned an incompatible descriptor shape")
                        if not torch.isfinite(features).all() or torch.count_nonzero(features[~valid]):
                            raise ValueError(
                                "Descriptors must be finite and invalid-point padding must be zero"
                            )
                        group["features"][row] = features.numpy()
                        group["point_camera"][row] = camera.numpy().astype(np.int8)
                        group["point_uv"][row] = uv.numpy().astype(np.int16)
                        # Flush payload before advancing the durable row counter;
                        # a killed writer may repeat a row, never skip one.
                        dst.flush()
                        group.attrs.modify("completed_rows", row + 1)
                        dst.flush()
                        processed += 1
                        now = time.monotonic()
                        if now - last_progress >= 10 or processed == estimate["frames"]:
                            elapsed = now - extraction_started
                            rate = (processed - resumed) / max(elapsed, 1e-9)
                            progress(
                                {
                                    "event": "feature_frame_progress",
                                    "task": manifest["task"],
                                    "episode": key,
                                    "row": row,
                                    "frames_done": processed,
                                    "frames_total": estimate["frames"],
                                    "frames_per_second": rate,
                                    "eta_seconds": (estimate["frames"] - processed) / max(rate, 1e-9),
                                    "elapsed_seconds": now - started,
                                }
                            )
                            last_progress = now
            manifest["complete"] = True
            dst.attrs.modify("manifest", json.dumps(manifest, sort_keys=True))
            dst.flush()
        temporary.replace(output)
    elapsed = time.monotonic() - started
    report = {
        **manifest,
        **estimate,
        "complete": True,
        "output": str(output.resolve()),
        "bytes": output.stat().st_size,
        "elapsed_seconds": elapsed,
        "extraction_seconds": time.monotonic() - extraction_started,
        "new_frames": processed - resumed,
        "frames_per_second": (processed - resumed) / max(time.monotonic() - extraction_started, 1e-9),
    }
    atomic_json(report, output.with_suffix(".manifest.json"))
    progress({"event": "feature_task_complete", **report})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Diagnostic only: cap total frames per task; cannot train from this output",
    )
    parser.add_argument(
        "--output-dir", help="Override sidecar destination (record it as visual_features.cache_dir to train)"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report selected frame counts and uncompressed cache bytes only",
    )
    parser.add_argument(
        "--check-geometry",
        action="store_true",
        help="Read-only CPU check of raw pixels for canonical points; requires --max-frames",
    )
    args = parser.parse_args(argv)
    if (
        args.threads < 1
        or (args.episodes is not None and args.episodes < 1)
        or (args.max_frames is not None and args.max_frames < 1)
    ):
        parser.error("--threads, --episodes and --max-frames must be positive")
    if args.check_geometry and (args.max_frames is None or args.dry_run):
        parser.error("--check-geometry requires --max-frames and cannot be combined with --dry-run")
    config = load_config(args.config)
    require_sampling_contract(config)
    spec = resolve_visual_features(config)
    if spec["mode"] == "xyzrgb":
        parser.error("Config must select frozen CLIP visual features")
    root = Path(config["paths"]["data_root"])
    prepared_root = root / config["data"].get("prepared_subdir", "prepared_balanced_v2")
    output_root = Path(args.output_dir or spec["cache_dir"])
    if output_root.resolve() == prepared_root.resolve():
        parser.error("Feature sidecars must use a directory separate from canonical geometry")
    tasks = args.tasks or config["data"]["tasks"]
    if len(tasks) != len(set(tasks)):
        parser.error("Duplicate task names are not supported")
    episodes = args.episodes or config["data"]["episodes_per_task"]
    jobs, estimates = [], []
    for task in tasks:
        if task not in config["data"]["tasks"]:
            parser.error(f"Task {task!r} is not in this configuration")
        prepared = prepared_root / f"{task}.h5"
        estimate = estimate_feature_cache(prepared, config, episodes=episodes, max_frames=args.max_frames)
        estimates.append(estimate)
        if not args.dry_run:
            raw = list((root / "raw").rglob(f"record_dataset_{task}.h5"))
            if len(raw) != 1:
                raise FileNotFoundError(f"Expected one raw RGB-D HDF5 for {task}, found {raw}")
            jobs.append((prepared, raw[0], output_root / f"{task}.h5"))
    _emit(
        {
            "event": "feature_preparation_plan",
            "tasks": estimates,
            "total_frames": sum(item["frames"] for item in estimates),
            "estimated_bytes": sum(item["estimated_bytes"] for item in estimates),
            "output_dir": str(output_root),
            "dry_run": args.dry_run,
            "check_geometry": args.check_geometry,
            "note": (
                "Read-only bounded pixel correspondence check; no CLIP, descriptor writes or full-file hashing"
                if args.check_geometry
                else "FP16 sampled descriptors before trainable projection; one sequential writer on the selected device"
            ),
        }
    )
    if args.dry_run:
        return estimates
    torch.set_num_threads(args.threads)
    if args.check_geometry:
        return [
            check_feature_geometry(prepared, raw, config, episodes=episodes, max_frames=args.max_frames)
            for prepared, raw, _ in jobs
        ]
    extractor = FrozenCLIPFeatures(spec, device=args.device)
    return [
        prepare_feature_task(
            prepared,
            raw,
            output,
            config,
            episodes=episodes,
            max_frames=args.max_frames,
            device=args.device,
            skip_existing=args.skip_existing,
            extractor=extractor,
        )
        for prepared, raw, output in jobs
    ]


if __name__ == "__main__":
    main()
