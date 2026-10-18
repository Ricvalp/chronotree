"""Resolve one revision-wide action convention using real reset/step replay.

The tested convention is explicit, never selected independently per sample.
Verified evidence is required by EpisodeDataset before action decisions are exposed.
"""

from __future__ import annotations
import argparse
import json
import os
import time
import traceback
from pathlib import Path
import h5py
import numpy as np
from .download import REVISION, sha256
from .prepare import numeric_keys, raw_episode_metadata
from .paths import alignment_path

BENCHMARK_REVISION = "6503b26e00ff338611fceb039417fbc64826ccc8"
SIMULATOR_REVISION = "07be6fbc66350ddca200abfb0a11b692f078f7fd"


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2)))


def replay_episode(path, task, episode_key, *, steps=32, points=256, video_path=None):
    from stmem.evaluation.robomme_adapter import RoboMMEAdapter

    adapter = RoboMMEAdapter(task, int(episode_key[8:]), "train", points=points, max_steps=1300)
    video = None
    started = time.monotonic()
    try:
        with h5py.File(path, "r") as f:
            group = f[episode_key]
            keys, demos, completed = raw_episode_metadata(group)
            execution = [k for k, d in zip(keys, demos) if not d]
            result = adapter.reset()
            raw_reset = adapter.last_raw_observation
            row0 = group[execution[0]]
            # reset has already executed its automatic initial command. Never replay it.
            reset_error = rms(result.current.proprio[:7], row0["obs/joint_state"][()])
            cond_expected = sum(demos)
            conditioning_errors = [
                rms(frame.proprio[:7], group[k]["obs/joint_state"][()])
                for frame, k in zip(result.conditioning, [k for k, d in zip(keys, demos) if d])
            ]
            camera_error = {
                c: rms(raw_reset[f"{c}_camera_extrinsic_list"][-1], row0[f"obs/{c}_camera_extrinsic"][()])
                for c in ("front", "wrist")
            }
            if video_path is not None:
                import imageio.v2 as imageio

                Path(video_path).parent.mkdir(parents=True, exist_ok=True)
                video = imageio.get_writer(video_path, fps=20)
                for frame in adapter.last_rgb:
                    video.append_data(frame)
            previous = np.asarray(raw_reset["joint_state_list"][-1])
            comparisons = []
            status = "ongoing"
            max_commands = len(execution) - 1 if steps <= 0 else min(steps, len(execution) - 1)
            for j in range(1, max_commands + 1):
                stored = group[execution[j]]
                action = np.asarray(stored["action/joint_action"][()], dtype=np.float32).reshape(-1)
                if action.shape == (7,):
                    action = np.r_[action, 0].astype(np.float32)
                # Chunk primitive execution remains exactly one env.step per command.
                step = adapter.step(action)
                status = step.status
                if step.frames:
                    observed = np.asarray(adapter.last_raw_observation["joint_state_list"][-1])
                    target = stored["obs/joint_state"][()]
                    comparisons.append(
                        {
                            "raw_index": int(execution[j][9:]),
                            "stored_vs_pre_rmse": rms(target, previous),
                            "stored_vs_post_rmse": rms(target, observed),
                            "terminal": step.done,
                        }
                    )
                    previous = observed
                    if video:
                        for image in adapter.last_rgb:
                            video.append_data(image)
                if step.done:
                    break
            pre = np.asarray([x["stored_vs_pre_rmse"] for x in comparisons])
            post = np.asarray([x["stored_vs_post_rmse"] for x in comparisons])
            # Fixed global convention supported only if post-step residual is low and
            # distinguishable from pre-step at moving controls; static controls carry no timing evidence.
            nonterminal = np.asarray([not x["terminal"] for x in comparisons], dtype=bool)
            moving = (pre > 1e-4) & nonterminal
            verified = bool(
                reset_error < 1e-4
                and len(result.conditioning) == cond_expected
                and len(comparisons) >= 3
                and moving.sum() >= 3
                and post[nonterminal].max() < 1e-4
                and np.median(pre[moving]) > 10 * max(float(np.median(post[moving])), 1e-7)
            )
            return {
                "episode_id": episode_key,
                "conditioning_expected": cond_expected,
                "conditioning_returned": len(result.conditioning),
                "conditioning_joint_max_rmse": max(conditioning_errors, default=0.0),
                "reset_initial_execution_joint_rmse": reset_error,
                "reset_extrinsic_rmse": camera_error,
                "compared_commands": len(comparisons),
                "median_pre_rmse": float(np.median(pre)) if len(pre) else None,
                "median_post_rmse": float(np.median(post)) if len(post) else None,
                "max_post_rmse": float(post.max()) if len(post) else None,
                "nonterminal_max_post_rmse": float(post[nonterminal].max()) if nonterminal.any() else None,
                "terminal_protocol_note": "Continuous wrapper executes an automatic extra terminal step; terminal observation state is excluded from active-decision timing evidence.",
                "post_action_supported": verified,
                "status": status,
                "effective_metadata": adapter.metadata,
                "elapsed_seconds": time.monotonic() - started,
                "comparisons": comparisons,
                "video_path": str(video_path) if video_path else None,
            }
    finally:
        if video:
            video.close()
        adapter.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config")
    p.add_argument("--task", default="VideoUnmask")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--steps", type=int, default=32, help="0 runs to terminal or end of expert recording")
    p.add_argument("--data-root", default=os.getenv("STMEM_DATA_ROOT", "data"))
    p.add_argument("--output")
    p.add_argument("--video", action="store_true")
    args = p.parse_args(argv)
    config = {}
    if args.config:
        from stmem.config import load_config

        config = load_config(args.config)
        args.data_root = config.get("paths", {}).get("data_root", args.data_root)
        args.task = config.get("data", {}).get("tasks", [args.task])[0]
    root = Path(args.data_root)
    config.setdefault("paths", {})["data_root"] = str(root)
    output = Path(args.output) if args.output else alignment_path(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "verified": False,
        "convention": "unresolved",
        "dataset_revision": REVISION,
        "benchmark_revision": BENCHMARK_REVISION,
        "simulator_revision": SIMULATOR_REVISION,
        "candidate_convention": "post_action_v1",
        "reset_initial_action": "not_replayed; not observable policy input",
        "raw_sha256": {},
        "episodes": [],
    }
    try:
        files = list((root / "raw").rglob(f"record_dataset_{args.task}.h5"))
        if len(files) != 1:
            raise FileNotFoundError(f"Expected raw {args.task} HDF5 under {root}/raw")
        path = files[0]
        source_manifest = json.loads((root / "manifests" / f"{args.task}.json").read_text())
        if source_manifest["revision"] != REVISION:
            raise ValueError(
                "This replay convention audit only registers the explicitly pinned dataset revision"
            )
        report["raw_sha256"][args.task] = sha256(path)
        with h5py.File(path, "r") as f:
            keys = numeric_keys(f, "episode_")[: args.episodes]
        for key in keys:
            result = replay_episode(
                path,
                args.task,
                key,
                steps=args.steps,
                video_path=root / "diagnostics" / f"{args.task}_{key}_expert.mp4" if args.video else None,
            )
            report["episodes"].append(result)
            print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2), flush=True)
        report["verified"] = bool(
            len(keys) >= 2 and all(e["post_action_supported"] for e in report["episodes"])
        )
        if report["verified"]:
            report["convention"] = "post_action_v1"
        else:
            report["blocker"] = (
                "Replay evidence does not yet distinguish the global post-action convention within required tolerance."
            )
    except Exception as exc:
        report["blocker"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"])
    temp = output.with_suffix(".partial.json")
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(output)
    print(
        json.dumps(
            {
                "verified": report["verified"],
                "convention": report["convention"],
                "output": str(output),
                "blocker": report.get("blocker"),
            },
            indent=2,
        )
    )
    if not report["verified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
