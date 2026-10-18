"""Inspect raw schema/source boundaries and save two-camera geometry diagnostics."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import h5py
import numpy as np
from .pointcloud import backproject, DEFAULT_CROP
from .prepare import numeric_keys, raw_episode_metadata


def calibration_diagnostic(path, task, output, episode=0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "task": task,
        "coordinate_convention": "CV world-to-camera extrinsic inverted; signed millimetres -> metres",
        "crop": DEFAULT_CROP,
        "frames": [],
    }
    with h5py.File(path, "r") as f:
        key = numeric_keys(f, "episode_")[episode]
        g = f[key]
        keys, demos, completed = raw_episode_metadata(g)
        ex = [i for i, d in enumerate(demos) if not d]
        chosen = [0, ex[0], ex[min(len(ex) - 1, 32)]]
        fig = plt.figure(figsize=(15, 9))
        for row, idx in enumerate(chosen):
            raw = g[keys[idx]]
            obs = raw["obs"]
            clouds = {}
            item = {
                "raw_index": int(keys[idx][9:]),
                "source": "conditioning" if demos[idx] else "execution",
                "cameras": {},
            }
            for c in ("front", "wrist"):
                k = g["setup"][f"{c}_camera_intrinsic"][()]
                e = obs[f"{c}_camera_extrinsic"][()]
                cloud = backproject(
                    obs[f"{c}_rgb"][()], obs[f"{c}_depth"][()], k, e, crop=DEFAULT_CROP
                ).numpy()
                clouds[c] = cloud
                xyz = cloud[:, :3]
                camera = xyz @ e[:, :3].T + e[:, 3]
                reprojection = camera @ k.T
                pixels = reprojection[:, :2] / reprojection[:, 2:]
                pixel_residual = np.abs(pixels - np.round(pixels)).max() if len(pixels) else None
                item["cameras"][c] = {
                    "valid_cropped_points": len(cloud),
                    "xyz_min": xyz.min(0).tolist() if len(xyz) else None,
                    "xyz_max": xyz.max(0).tolist() if len(xyz) else None,
                    "table_plane_fraction_abs_z_below_5mm": float(np.mean(np.abs(xyz[:, 2]) < 0.005))
                    if len(xyz)
                    else None,
                    "max_reprojection_pixel_rounding_error": float(pixel_residual)
                    if pixel_residual is not None
                    else None,
                }
            ax = fig.add_subplot(3, 3, row * 3 + 1)
            ax.imshow(np.concatenate([obs["front_rgb"][()], obs["wrist_rgb"][()]], axis=1))
            ax.set_title(f"{key} {keys[idx]} ({item['source']})")
            ax.axis("off")
            ax = fig.add_subplot(3, 3, row * 3 + 2, projection="3d")
            for c, color in [("front", "tab:cyan"), ("wrist", "tab:orange")]:
                cloud = clouds[c][:: max(1, len(clouds[c]) // 6000)]
                ax.scatter(*cloud[:, :3].T, s=0.2, c=color, label=c, alpha=0.5)
            ax.scatter([-0.615], [0], [0], c="black", s=30, marker="x", label="Panda base")
            ax.set(
                xlim=(-0.9, 0.75),
                ylim=(-0.75, 0.75),
                zlim=(-0.05, 1.25),
                xlabel="world x (m)",
                ylabel="world y (m)",
                zlabel="world z (m)",
            )
            ax.view_init(25, -60)
            ax.legend(markerscale=5, fontsize=7)
            ax = fig.add_subplot(3, 3, row * 3 + 3)
            for c, color in [("front", "tab:cyan"), ("wrist", "tab:orange")]:
                xyz = clouds[c][:: max(1, len(clouds[c]) // 6000), :3]
                ax.scatter(xyz[:, 0], xyz[:, 2], s=0.2, color=color, label=c, alpha=0.5)
            ax.axhline(0, c="black", lw=0.7)
            ax.set(xlim=(-0.9, 0.75), ylim=(-0.05, 1.25), xlabel="world x (m)", ylabel="world z (m)")
            ax.set_aspect("equal")
            report["frames"].append(item)
        fig.tight_layout()
        fig.savefig(output / f"{task}_calibration.png", dpi=160)
        plt.close(fig)
        report["episode_id"] = key
        report["conditioning_frames"] = sum(demos)
        report["execution_frames"] = len(demos) - sum(demos)
        report["terminal_completed_records"] = sum(completed)
    (output / f"{task}_calibration.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--task", default="VideoUnmask")
    p.add_argument("--data-root", default=os.getenv("STMEM_DATA_ROOT", "data"))
    p.add_argument("--output")
    args = p.parse_args(argv)
    if args.config:
        from stmem.config import load_config

        config = load_config(args.config)
        args.data_root = config.get("paths", {}).get("data_root", args.data_root)
        args.task = config.get("data", {}).get("tasks", [args.task])[0]
    root = Path(args.data_root)
    files = list((root / "raw").rglob(f"record_dataset_{args.task}.h5"))
    if len(files) != 1:
        raise FileNotFoundError("Download/extract a task archive first")
    print(
        json.dumps(calibration_diagnostic(files[0], args.task, args.output or root / "diagnostics"), indent=2)
    )


if __name__ == "__main__":
    main()
