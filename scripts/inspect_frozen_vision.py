"""Bounded source-pixel/feature inspection; no policy training or benchmark score.

Shows native RGB, selected observed pixels and a three-component PCA rendering
of the frozen layer1 maps. PCA colors are an explanatory projection, not object
classes or evidence that the eventual trained policy will use the descriptors.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from matplotlib.patches import Rectangle

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from stmem.config import load_config, source_revision
from stmem.data.pointcloud import (
    CAMERA_NAMES,
    point_sampler_settings,
    rgbd_to_points,
    rgbd_to_points_with_provenance,
)
from stmem.vision.frozen import FrozenCLIPFeatures, sample_dense_features


def frame_inputs(args, config):
    if args.raw_h5:
        with h5py.File(args.raw_h5, "r") as source:
            group = source[f"episode_{args.episode}"]
            calibration = {
                f"{c}_camera_intrinsic": group["setup"][f"{c}_camera_intrinsic"][()] for c in CAMERA_NAMES
            }
            for frame in args.frames:
                record = group[f"timestep_{frame}"]
                observation = {
                    f"{camera}_{field}": record["obs"][f"{camera}_{field}"][()]
                    for camera in CAMERA_NAMES
                    for field in ("rgb", "depth", "camera_extrinsic")
                }
                info = {
                    "kind": "released_training_frame",
                    "task": args.task,
                    "episode": args.episode,
                    "raw_frame": frame,
                    "source": "conditioning"
                    if bool(np.asarray(record["info"]["is_video_demo"][()]).reshape(-1)[0])
                    else "execution",
                    "raw_h5": str(args.raw_h5.resolve()),
                }
                yield (
                    f"{args.task}_train{args.episode}_frame{frame:04d}",
                    observation,
                    calibration,
                    info,
                    None,
                )
        return
    from stmem.evaluation.robomme_adapter import OBSERVATION_FIELDS, RoboMMEAdapter, numpy_value
    from stmem.data.center_sampling import center_counts, center_sampler_settings

    adapter = RoboMMEAdapter(
        args.sim_task,
        args.episode,
        args.split,
        points=config["model"]["points"],
        max_steps=8,
        crop=config["geometry"]["crop"],
        voxel_size=config["geometry"]["voxel_size"],
        max_depth_m=config["geometry"]["max_depth"],
        point_sampling=point_sampler_settings(config),
        center_sampling=center_sampler_settings(config),
        center_counts=center_counts(config),
    )
    try:
        reset = adapter.reset()
        observation = {
            key: numpy_value(adapter.last_raw_observation[key + "_list"][-1]) for key in OBSERVATION_FIELDS
        }
        cube = None
        if args.sim_task == "PickXtimes":
            base = adapter.env.unwrapped
            cube = {
                "position": numpy_value(base.target_cube.pose.p).reshape(-1, 3)[0],
                "quaternion": numpy_value(base.target_cube.pose.q).reshape(-1, 4)[0],
                "half_size": np.broadcast_to(numpy_value(base.cube_half_size), (3,)),
                "color_name": str(base.target_color_name),
            }
        info = {
            "kind": "simulator_reset_diagnostic_no_policy_commands",
            "task": args.sim_task,
            "episode": args.episode,
            "split": args.split,
            "source": "execution",
            "conditioning_frames": len(reset.conditioning),
            "runtime": reset.metadata,
        }
        yield (
            f"{args.sim_task}_{args.split}{args.episode}_reset",
            observation,
            adapter.calibration,
            info,
            cube,
        )
    finally:
        adapter.close()


def cube_membership(points, cube):
    if cube is None:
        return np.zeros(len(points), dtype=bool)
    w, x, y, z = cube["quaternion"] / np.linalg.norm(cube["quaternion"])
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    in_box = np.all(np.abs((points[:, :3] - cube["position"]) @ rotation) <= cube["half_size"] + 0.003, -1)
    channel = {"red": 0, "green": 1, "blue": 2}[cube["color_name"]]
    color = np.all(
        points[:, channel + 3, None] > points[:, [3 + c for c in range(3) if c != channel]] + 0.15, -1
    )
    return in_box & color


def pca_projection(maps, sampled):
    values = maps.permute(0, 2, 3, 1).reshape(-1, maps.shape[1]).float().cpu()
    mean = values.mean(0)
    centered = values - mean
    _, vectors = torch.linalg.eigh(centered.T @ centered / len(centered))
    basis = vectors[:, -3:].flip(1)
    for index in range(3):
        if basis[basis[:, index].abs().argmax(), index] < 0:
            basis[:, index] *= -1
    colors = centered @ basis
    lo = torch.quantile(colors, 0.02, dim=0)
    hi = torch.quantile(colors, 0.98, dim=0)
    scale = (hi - lo).clamp_min(1e-8)
    dense_colors = ((colors - lo) / scale).clamp(0, 1).reshape(2, 64, 64, 3).numpy()
    point_colors = (((sampled.float().cpu() - mean) @ basis - lo) / scale).clamp(0, 1).numpy()
    return dense_colors, point_colors


def pointcloud_html(path, points, point_colors, cube_mask, name):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("Inspection HTML requires uv sync --extra visualization") from exc
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Canonical XYZRGB (unchanged)", "Same points, frozen descriptor PCA colors"),
    )
    for column, colors in enumerate((points[:, 3:], point_colors), 1):
        color = [f"rgb({r},{g},{b})" for r, g, b in (np.clip(colors, 0, 1) * 255).astype(np.uint8)]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"size": 2, "color": color},
                name="Observed points",
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        if cube_mask.any():
            xyz = points[cube_mask, :3]
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="markers",
                    marker={"size": 5, "color": "magenta", "symbol": "circle-open"},
                    name="Cube (diagnostic label only)",
                    showlegend=column == 1,
                    legendgroup="cube",
                    visible="legendonly",
                ),
                row=1,
                col=column,
            )
    fig.update_scenes(
        aspectmode="data", xaxis_title="world X (m)", yaxis_title="world Y (m)", zaxis_title="world Z (m)"
    )
    fig.update_layout(
        title=f"{name}: per-camera frozen feature correspondence",
        height=650,
        margin={"t": 100},
        annotations=list(fig.layout.annotations)
        + [
            dict(
                text="PCA colors are a visualization, not semantic labels. Magenta labels use diagnostic-only cube geometry.",
                x=0.5,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
        ],
    )
    fig.write_html(path, include_plotlyjs=True)


def inspect_frame(name, observation, calibration, info, cube, config, extractor, output, repeats):
    kwargs = dict(
        count=config["model"]["points"],
        crop=config["geometry"]["crop"],
        voxel_size=config["geometry"]["voxel_size"],
        max_depth_m=config["geometry"]["max_depth"],
        **point_sampler_settings(config),
    )
    started = time.perf_counter()
    points, valid, camera, uv = rgbd_to_points_with_provenance(observation, calibration, **kwargs)
    geometry_seconds = time.perf_counter() - started
    canonical, canonical_valid = rgbd_to_points(observation, calibration, **kwargs)
    if not (torch.equal(points, canonical) and torch.equal(valid, canonical_valid)):
        raise AssertionError("Provenance changed the canonical point selection")
    rgb = np.stack([observation[c + "_rgb"] for c in CAMERA_NAMES])
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        features = extractor.sample(observation, camera, uv, valid)
        if features.is_cuda:
            torch.cuda.synchronize(features.device)
        times.append(time.perf_counter() - started)
    maps = extractor.dense(rgb)
    expected = sample_dense_features(maps, camera, uv, valid).half()
    if not torch.equal(features, expected):
        raise AssertionError("Dense map and production sampled-feature path disagree")
    dense_colors, colors = pca_projection(maps, features)
    mask = valid.numpy()
    cube_mask = cube_membership(points.numpy(), cube) & mask
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), constrained_layout=True)
    reprojection_errors, color_errors = [], []
    for camera_id, name_camera in enumerate(CAMERA_NAMES):
        selected = mask & (camera.numpy() == camera_id)
        pix = uv.numpy()[selected]
        xyz = points.numpy()[selected, :3]
        calibration_k = np.asarray(calibration[f"{name_camera}_camera_intrinsic"]).reshape(3, 3)
        extrinsic = np.asarray(observation[f"{name_camera}_camera_extrinsic"]).reshape(3, 4)
        projected = (xyz @ extrinsic[:, :3].T + extrinsic[:, 3]) @ calibration_k.T
        projected = projected[:, :2] / projected[:, 2:]
        reprojection_errors.extend(np.linalg.norm(projected - pix, axis=-1).tolist())
        lookup = observation[name_camera + "_rgb"][pix[:, 1], pix[:, 0]].astype(np.float32) / 255
        color_errors.extend(np.abs(lookup - points.numpy()[selected, 3:]).flatten().tolist())
        axes[camera_id, 0].imshow(observation[name_camera + "_rgb"])
        axes[camera_id, 0].set_title(f"{name_camera}: original RGB")
        axes[camera_id, 1].imshow(observation[name_camera + "_rgb"])
        axes[camera_id, 1].scatter(pix[:, 0], pix[:, 1], s=6, c=colors[selected], linewidths=0)
        axes[camera_id, 1].set_title(f"{selected.sum()} selected source pixels; descriptor PCA colors")
        axes[camera_id, 2].imshow(
            dense_colors[camera_id], origin="upper", extent=(-1, 255, 255, -1), interpolation="nearest"
        )
        axes[camera_id, 2].set_title("64×64 layer1 feature map (PCA)")
        cube_uv = uv.numpy()[cube_mask & (camera.numpy() == camera_id)]
        if len(cube_uv):
            for col in (0, 1, 2):
                low, high = cube_uv.min(0) - 3, cube_uv.max(0) + 3
                axes[camera_id, col].add_patch(
                    Rectangle(low, *(high - low), linewidth=1.2, edgecolor="magenta", facecolor="none")
                )

        for axis in axes[camera_id]:
            axis.set(
                xlim=(-0.5, 255.5),
                ylim=(255.5, -0.5),
                xlabel="original u (pixels)",
                ylabel="original v (pixels)",
            )
    fig.suptitle(
        f"{name}\nFrozen CLIP RN50 layer1; per-scene PCA is a visualization, not segmentation", fontsize=13
    )
    png = output / f"{name}.png"
    fig.savefig(png, dpi=145)
    plt.close(fig)
    pointcloud_html(output / f"{name}.html", points.numpy()[mask], colors[mask], cube_mask[mask], name)
    camera_counts = {
        name_camera: int((valid & (camera == camera_id)).sum())
        for camera_id, name_camera in enumerate(CAMERA_NAMES)
    }
    result = {
        **info,
        "name": name,
        "points": len(points),
        "valid_points": int(valid.sum()),
        "camera_counts": camera_counts,
        "canonical_xyzrgb_bitwise_equal": True,
        "dense_sample_path_bitwise_equal": True,
        "feature_shape": list(maps.shape),
        "feature_dtype": str(features.dtype),
        "feature_finite": bool(torch.isfinite(features).all()),
        "point_rgb_lookup_max_abs_error": max(color_errors, default=0),
        "reprojection_max_error_pixels": max(reprojection_errors, default=0),
        "geometry_seconds": geometry_seconds,
        "feature_seconds_repeated": times,
        "feature_warm_median_seconds": float(np.median(times[1:] if repeats > 1 else times)),
        "descriptor_bytes_per_frame": features.numel() * features.element_size(),
        "dense_map_fp16_bytes_per_frame": maps.numel() * 2,
        "png": png.name,
        "html": f"{name}.html",
    }
    if cube is not None:
        result["cube_diagnostic"] = {
            "selected_points": int(cube_mask.sum()),
            "label_definition": "observed points inside oriented simulator cube box +3mm and matching RGB; metrics only, never encoder input",
            "camera_point_counts": {
                c: int((cube_mask & (camera.numpy() == i)).sum()) for i, c in enumerate(CAMERA_NAMES)
            },
        }
        if cube_mask.any():
            cube_features = features.cpu().float()[torch.from_numpy(cube_mask)]
            table_mask = valid & (points[:, 2].abs() < 0.003)
            if table_mask.any():
                table_features = features.cpu().float()[table_mask]
                result["cube_diagnostic"]["mean_cube_vs_table_cosine_similarity"] = float(
                    torch.nn.functional.cosine_similarity(
                        cube_features.mean(0)[None], table_features.mean(0)[None]
                    )
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--raw-h5", type=Path)
    source.add_argument("--sim-task", choices=("PickXtimes", "VideoUnmask"))
    parser.add_argument("--task", default="VideoUnmask", help="Bookkeeping name for raw data")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--frames", type=int, nargs="+", default=[0, 66, 100], help="Original raw timestep numbers"
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--checkpoint", help="Override pretrained CLIP weight path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    torch.set_num_threads(2)
    config = deepcopy(load_config(args.config))
    spec = config.setdefault("model", {}).setdefault("visual_features", {})
    spec["mode"] = "clip_rn50_layer1"
    if args.checkpoint:
        spec["checkpoint"] = args.checkpoint
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extractor = FrozenCLIPFeatures(config, device=args.device)
    results = [
        inspect_frame(name, obs, calib, info, cube, config, extractor, args.output_dir, args.repeats)
        for name, obs, calib, info, cube in frame_inputs(args, config)
    ]
    report = {
        "kind": "frozen_feature_correspondence_diagnostic_not_policy_evaluation",
        "identity": extractor.identity,
        "signature": extractor.signature,
        "frozen_backbone_parameters": sum(p.numel() for p in extractor.parameters()),
        "frozen_backbone_gradients_enabled": any(p.requires_grad for p in extractor.parameters()),
        "device": str(next(extractor.parameters()).device),
        "device_name": torch.cuda.get_device_name(args.device)
        if str(args.device).startswith("cuda")
        else "cpu",
        "source_revision": source_revision(),
        "results": results,
        "limitations": [
            "PCA colors do not measure semantic object separability or policy use.",
            "Tiny frame sample, not dataset throughput; shared HPC IO and parallel extraction are unmeasured.",
            "No trained frozen-vision checkpoint or task-success claim.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
