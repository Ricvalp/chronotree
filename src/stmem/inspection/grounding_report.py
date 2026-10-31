"""Independent saved-array/media validation and dynamic grounding reports."""

from __future__ import annotations
import html
import json
from pathlib import Path

import imageio_ffmpeg
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stmem.training.checkpoints import file_sha256
from stmem.visualization.kinematics import panda_tcp_positions

ROOT = None
MODELS = {}
HISTORY_ENABLED = {}
LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], np.float32)
HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], np.float32)
checks = 0


def check(value, message):
    global checks
    assert value, message
    checks += 1


def read(path):
    return json.loads(path.read_text())


def physical(raw):
    raw = np.asarray(raw, dtype=np.float32).copy()
    raw[:, :7] = np.clip(raw[:, :7], LOW, HIGH)
    raw[:, 7] = np.where(raw[:, 7] >= 0, 1, -1)
    return raw


def validate_prediction(row, horizon, stride):
    raw = np.asarray(row["raw_actions"], dtype=np.float32)
    check(raw.shape == (horizon, 8) and np.isfinite(raw).all(), "raw shape and finite")
    command = physical(raw)
    check(
        np.array_equal(command, np.asarray(row["physical_actions"], np.float32)),
        "joint clipping and gripper binarization",
    )
    native = np.repeat(command, stride, axis=0)
    check(np.array_equal(native, np.asarray(row["native_commands"], np.float32)), "zero order hold expansion")
    tcp = panda_tcp_positions(native[:, :7], task="PickXtimes")
    check(np.allclose(tcp, row["native_tcp"], atol=1e-7, rtol=1e-6), "native command FK")
    check(np.allclose(tcp[:4], row["first4_native_tcp"], atol=1e-7, rtol=1e-6), "physical0.2s FK")
    if "attention" in row:
        attention = row["attention"]
        check(attention["hook_control"] == "bit_identical", "attention observational only")
        metrics = attention["metrics"]
        groups = ("current_visual", "current_state", "goal", "clock", "history_visual", "history_event")
        check(
            np.isclose(sum(metrics["mass/" + group] for group in groups), 1.0, atol=1e-5),
            "attention mass normalization",
        )
        check(
            all(-1e-6 <= value <= 1.000001 for key, value in metrics.items() if key.startswith("entropy/")),
            "entropy bounded",
        )


def validate_coverage(value, path, variant, observation_stride):
    for bank, stats in value["banks"].items():
        indices = stats["source_indices"]
        check(all(i % observation_stride == 0 for i in indices), "retained cadence phase")
        check(indices == sorted(set(indices)), "unique ordered source indices")
        if bank == "latest_history_block":
            check(HISTORY_ENABLED[variant], "no fabricated history branch")
            check(max(indices) < value["step"], "current is excluded from history")
        else:
            check(indices == [value["step"]], "current observation has exact current native timestamp")
        trace = np.load(path / f"step_{value['step']:03d}_{bank}_trace.npz")
        points, pf = trace["points"], trace["point_frame_indices"]
        membership = np.zeros(len(points), bool)
        for f, (p, q, half) in enumerate(
            zip(trace["cube_positions"], trace["cube_quaternions"], trace["cube_half_sizes"])
        ):
            w, x, y, z = q / np.linalg.norm(q)
            rotation = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            mask = pf == f
            xyzrgb = points[mask]
            inside = (np.abs((xyzrgb[:, :3] - p) @ rotation) <= half + 0.003).all(-1)
            blue = (xyzrgb[:, 5] > xyzrgb[:, 3] + 0.15) & (xyzrgb[:, 5] > xyzrgb[:, 4] + 0.15)
            membership[mask] = inside & blue
        mass = (trace["weights"] * (membership[trace["neighbor_indices"]] & trace["neighbor_mask"])).sum(-1)
        check(int(membership.sum()) == stats["cube_points"], "independent point membership")
        check(np.allclose(mass, stats["cube_mass_per_token"]), "independent Gaussian pooling mass")
        check(
            int(((mass > 0.5) & trace["valid"]).sum()) == stats["cube_majority_tokens"],
            "independent cube majority token count",
        )


def build_reports(manifest, manifest_sha256):
    global ROOT, MODELS, HISTORY_ENABLED, checks
    from stmem.inspection.reset_report import build_reset_report

    ROOT = Path(manifest["output"])
    audit = read(ROOT / "checkpoint_audit.json")
    MODELS = {
        label: (
            manifest["checkpoints"][label]["step"],
            audit["checkpoints"][label]["config"]["model"]["horizon"],
            audit["checkpoints"][label]["config"]["model"].get("action_stride", 1),
            audit["checkpoints"][label]["config"]["data"].get("observation_stride", 1),
        )
        for label in manifest["labels"]
    }
    HISTORY_ENABLED = {
        label: audit["checkpoints"][label]["config"]["model"].get("history_mode", "full") != "none"
        for label in MODELS
    }
    checks = 0
    reset_validation = build_reset_report(manifest, manifest_sha256)
    summary = {"kind": "bounded_scene_diagnostics_not_benchmark_score", "models": {}}
    for variant, (step, horizon, stride, observation_stride) in MODELS.items():
        checkpoint = Path(manifest["checkpoints"][variant]["path"])
        identity = file_sha256(checkpoint)
        rollout_dir = ROOT / "rollouts" / variant / "rollout"
        rollout = read(rollout_dir / "result.json")
        rendering = read(ROOT / "grounding" / variant / "rendered_cube/result.json")
        for result in (rollout, rendering):
            check(
                result["inspection_manifest_sha256"] == manifest_sha256 and result["diagnostic_complete"],
                "manifest identity and completion",
            )
            check(
                result["checkpoint_sha256"] == identity and result["checkpoint_step"] == step,
                "checkpoint identity",
            )
            check(result["temporal_sampling"]["action_stride"] == stride, "native action stride")
            check(
                result["temporal_sampling"]["observation_stride"] == observation_stride,
                "native observation stride",
            )
            check(result["temporal_sampling"]["horizon"] == horizon, "native horizon")
        check(rendering["status"] == "complete", "all shifted scenes completed")
        baseline = rendering["scenes"]["baseline"]
        for name, scene in rendering["scenes"].items():
            check(scene["max_replay_proprio_error"] < 1e-5, "fixed robot replay")
            for t, state in scene["states"].items():
                check(
                    state["input_signature"] == baseline["states"][t]["input_signature"],
                    "robot action time text equality",
                )
                actual_shift = np.asarray(state["cube"]["p"]) - baseline["states"][t]["cube"]["p"]
                check(
                    np.allclose(actual_shift, scene["shift_xyz_m"], atol=1e-4, rtol=0),
                    "physical cube translation",
                )
                check(state["coverage"]["banks"]["current"]["cube_points"] > 0, "visible target cube")
                validate_coverage(
                    state["coverage"],
                    ROOT / "grounding" / variant / "rendered_cube" / name,
                    variant,
                    observation_stride,
                )
                if not HISTORY_ENABLED[variant]:
                    check(state["history_tokens"] == 0, "current-only no history tokens")
                for precision, rows in state["predictions"].items():
                    ref_rows = baseline["states"][t]["predictions"][precision]
                    changes = []
                    for row, reference in zip(rows, ref_rows):
                        check(row["seed"] == reference["seed"], "paired intervention noise")
                        validate_prediction(row, horizon, stride)
                        delta = np.asarray(row["native_tcp"]) - reference["native_tcp"]
                        changes.append(np.linalg.norm(delta[:4], axis=-1).mean() * 1000)
                    check(
                        np.isclose(
                            np.mean(changes),
                            rendering["summary"][name][t][precision]["first4_native_0p2s"][
                                "mean_tcp_change_mm"
                            ],
                        ),
                        "independent response metric",
                    )
        timeline = rollout["timeline"]
        check(0 < len(timeline) <= rollout["cap_native_commands"], "bounded actual commands")
        check(
            [r["step"] for r in timeline] == list(range(1, len(timeline) + 1)), "consecutive native commands"
        )
        decision_map = {d["step"]: d for d in rollout["decisions"]}
        interval = rollout["temporal_sampling"]["execute_targets"] * stride
        check(interval == 4, "equal physical replanning interval")
        for row in timeline:
            index = row["step"] - 1
            raw = decision_map[index // interval * interval]["raw_actions"]
            native = np.repeat(physical(raw), stride, axis=0)
            check(
                np.array_equal(native[index % interval], np.asarray(row["command"], np.float32)),
                "executed production chunk with ZOH",
            )
            check(
                row["memory_current_source_index"] == row["step"] // observation_stride * observation_stride,
                "memory current sample timing",
            )
            if not row["done"]:
                check(
                    row["observed_source_indices"]
                    == ([row["step"]] if row["step"] % observation_stride == 0 else []),
                    "cadence retained observations",
                )
            observed = row["observed_proprio"]
            if observed is not None:
                check(
                    np.isclose(
                        np.linalg.norm(np.asarray(observed)[7:9] - row["cube"]["p"][:2]),
                        row["actual_tcp_cube_xy_distance_m"],
                    ),
                    "observed cube XY error",
                )
                check(
                    np.isclose(
                        np.linalg.norm(np.asarray(row["command_tcp"]) - observed[7:10]),
                        row["command_tracking_error_m"],
                    ),
                    "actuator tracking error",
                )
            check(
                np.allclose(
                    panda_tcp_positions(np.asarray(row["command"])[:7], task="PickXtimes"),
                    row["command_tcp"],
                    atol=1e-7,
                ),
                "command FK",
            )
        for cov in rollout["coverage"]:
            validate_coverage(cov, rollout_dir, variant, observation_stride)
        media = Path(rollout["video"])
        frame_count, seconds = imageio_ffmpeg.count_frames_and_secs(str(media))
        check(frame_count == len(timeline) + 1, "reset plus every native frame")
        check(abs(seconds - frame_count / 20) < 0.06, "20Hz video")
        first = next((r for r in timeline if r["command"][7] < 0), None)
        info = dict(
            step=step,
            horizon=horizon,
            action_stride=stride,
            observation_stride=observation_stride,
            precision=rollout["precision"],
            status=rollout["status"],
            native_commands=len(timeline),
            video_frames=frame_count,
            grasped=any(r["grasped"] for r in timeline),
            grasp_flag_commands=[r["step"] for r in timeline if r["grasped"]],
            max_cube_lift_mm=max(r["cube_lift_from_reset_m"] for r in timeline) * 1000,
            first_close_command=None if first is None else first["step"],
            first_close_observed_xy_mm=None
            if first is None
            else first.get("actual_tcp_cube_xy_distance_m", float("nan")) * 1000,
            first_close_commanded_xy_mm=None
            if first is None
            else first["command_tcp_cube_xy_distance_m"] * 1000,
            first_close_tracking_mm=None
            if first is None
            else first.get("command_tracking_error_m", float("nan")) * 1000,
            state80_response_mm={
                name: {
                    precision: v["80"][precision]["first4_native_0p2s"]["mean_tcp_change_mm"]
                    for precision in v["80"]
                }
                for name, v in rendering["summary"].items()
                if name != "baseline"
            },
            state80_first0p8s_response_mm={
                name: {
                    precision: v["80"][precision]["first16_native_0p8s"]["mean_tcp_change_mm"]
                    for precision in v["80"]
                }
                for name, v in rendering["summary"].items()
                if name != "baseline"
            },
            state80_full_horizon_response_mm={
                name: {precision: v["80"][precision]["full"]["mean_tcp_change_mm"] for precision in v["80"]}
                for name, v in rendering["summary"].items()
                if name != "baseline"
            },
            state80_coverage=baseline["states"]["80"]["coverage"],
            state80_attention={
                precision: rows[0]["attention"]["metrics"]
                for precision, rows in baseline["states"]["80"]["predictions"].items()
            },
        )
        info["state80_response_by_native_command_mm"] = {}
        info["state80_endpoint_response_mm"] = {}
        for scene in ("x_plus_10cm", "y_plus_10cm"):
            info["state80_response_by_native_command_mm"][scene] = {}
            info["state80_endpoint_response_mm"][scene] = {}
            for precision, rows in rendering["scenes"][scene]["states"]["80"]["predictions"].items():
                references = baseline["states"]["80"]["predictions"][precision]
                delta = np.stack(
                    [np.asarray(row["native_tcp"]) - ref["native_tcp"] for row, ref in zip(rows, references)]
                )
                response = np.linalg.norm(delta, axis=-1).mean(0) * 1000
                info["state80_response_by_native_command_mm"][scene][precision] = response.tolist()
                info["state80_endpoint_response_mm"][scene][precision] = float(response[-1])
        reset_payload = ROOT / "reset_cube_chunks" / variant / "payload.json"
        if reset_payload.exists():
            payload = read(reset_payload)
            scene = next(s for s in payload["scenes"] if s["id"] == "baseline")
            pred = next(iter(scene["predictions"].values()))["2701"]["raw_actions"]
            error = float(np.max(np.abs(np.asarray(rollout["decisions"][0]["raw_actions"]) - pred)))
            info["reset_visualization_raw_action_max_difference"] = error
            check(error < 1e-6, "production initial chunk matches independent reset visualization")
        summary["models"][variant] = info
    summary.update(
        status="passed",
        checks=checks,
        inspection_manifest_sha256=manifest_sha256,
        reset_validation_checks=reset_validation["checks"],
        notes=[
            "No model or simulator execution in this validator. All calculations use saved arrays.",
            "Checkpoints differ in training exposure, visual features, history/proprio inputs, and cadence.",
            "Cube translations are artificial matched-state diagnostics. Single capped rollouts are not benchmark success rates.",
            "TenHz action targets are held for two20Hz commands, not interpolated; first4-native response spans0.2seconds.",
        ],
    )
    render_report(summary)
    summary["checks"] = checks
    (ROOT / "grounding/validation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(event="grounding_validation", status="passed", checks=checks, models=list(summary["models"]))
        ),
        flush=True,
    )
    return summary


def render_report(summary):
    def number(value):
        return "—" if value is None else f"{value:.2f}"

    rows, cards = [], []
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, scene, title in zip(axes, ("x_plus_10cm", "y_plus_10cm"), ("Cube X +100 mm", "Cube Y +100 mm")):
        for name, model in summary["models"].items():
            values = model["state80_response_by_native_command_mm"][scene]
            curve = values[model["precision"]]
            axis.plot(np.arange(1, len(curve) + 1) / 20, curve, label=name)
        axis.axvspan(
            0,
            0.2,
            color="gray",
            alpha=0.12,
            label="Executed before replan" if scene == "x_plus_10cm" else None,
        )
        axis.set(title=title, xlabel="Time into held action chunk (seconds)", xlim=(0, 1.6))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean commanded TCP displacement (mm)")
    axes[1].legend(fontsize=8)
    figure.suptitle("Matched replay state 80: response to physically moving the cube")
    figure.tight_layout()
    figure.savefig(ROOT / "rollouts/state80_response_by_time.png", dpi=160)
    plt.close(figure)
    for name, model in summary["models"].items():
        x = model["state80_response_mm"]["x_plus_10cm"][model["precision"]]
        y = model["state80_response_mm"]["y_plus_10cm"][model["precision"]]
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{model['step']:,}</td><td>{number(x)}</td><td>{number(y)}</td><td>{model['first_close_command']}</td><td>{number(model['first_close_observed_xy_mm'])}</td><td>{'yes' if model['grasped'] else 'no'}</td></tr>"
        )
        cards.append(f'''<article><h2>{html.escape(name)} · {model["step"]:,} updates</h2>
<video controls preload="metadata" poster="{name}/rollout/poster.png" src="{name}/rollout/PickXtimes_val0_first160.mp4"></video>
<p>{model["horizon"]} targets, {20 / model["action_stride"]:g} Hz targets; 20 Hz simulation. Replans every 0.2 s.</p>
<p>First close: command {model["first_close_command"]}; observed XY miss {number(model["first_close_observed_xy_mm"])} mm; actuator tracking {number(model["first_close_tracking_mm"])} mm. Any simulator grasp flag: {"yes" if model["grasped"] else "no"}; maximum cube rise {number(model["max_cube_lift_mm"])} mm. A brief grasp flag does not establish a successful lift.</p>
<a href="{name}/rollout/result.json">Rollout trace</a> · <a href="../grounding/{name}/rendered_cube/result.json">Relocation, attention and coverage</a></article>''')
    comparison = ROOT / "rollouts/model_comparison.mp4"
    readers = [
        imageio.get_reader(ROOT / "rollouts" / name / "rollout/PickXtimes_val0_first160.mp4")
        for name in MODELS
    ]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    last = [None] * len(readers)
    frames = max(model["video_frames"] for model in summary["models"].values())
    writer = imageio.get_writer(comparison, fps=20, macro_block_size=2)
    try:
        for step in range(frames):
            canvas = Image.new("RGB", (1024, 288 * ((len(MODELS) + 1) // 2)), "#172231")
            for index, (name, reader) in enumerate(zip(MODELS, readers)):
                if step < summary["models"][name]["video_frames"]:
                    last[index] = reader.get_data(step)
                if step == (
                    summary["models"][name]["first_close_command"]
                    or min(80, summary["models"][name]["video_frames"] - 1)
                ):
                    Image.fromarray(last[index]).save(ROOT / "rollouts" / name / "rollout/poster.png")
                tile = Image.fromarray(last[index]).resize((512, 256))
                x, y = (index % 2) * 512, (index // 2) * 288
                canvas.paste(tile, (x, y + 32))
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (x + 9, y + 7),
                    f"{name} | {summary['models'][name]['step']:,} updates | command {step}",
                    font=font,
                    fill="white",
                )
            if step == min(100, frames - 1):
                canvas.save(ROOT / "rollouts/comparison_poster.png")
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()
        for reader in readers:
            reader.close()
    comparison_frames, comparison_seconds = imageio_ffmpeg.count_frames_and_secs(str(comparison))
    check(comparison_frames == frames and abs(comparison_seconds - frames / 20) < 0.06, "comparison media")
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PickXtimes grounding comparison</title>
<style>body{font:16px system-ui;max-width:1280px;margin:32px auto;padding:0 20px;background:#f8fafc;color:#182333}h1{font-size:30px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}article{background:white;border:1px solid #dbe2ec;padding:20px;border-radius:12px}video{width:100%;background:#111}table{border-collapse:collapse;width:100%;margin:24px 0}th,td{text-align:left;border-bottom:1px solid #dbe2ec;padding:10px}a{color:#0758b0}.scroll{overflow:auto}@media(max-width:800px){.cards{grid-template-columns:1fr}body{margin:16px auto}}</style>
<h1>PickXtimes: trained policy comparison</h1><p>Validation scene 0. Each fresh rollout is capped at 160 native 20 Hz commands (8 seconds). These are bounded diagnostics, not benchmark success rates.</p>
<p>The separate relocation probe moves the real cube by 100 mm along X/Y, rerenders both cameras, then replays the same robot commands to state 80. Predictions are not executed. Response measures the mean change in the first 0.2 seconds of commanded TCP targets, averaged over four paired noise draws. The table uses each model's production decoding precision; FP32 is retained in JSON.</p>
<p>Training exposure and architecture/input choices differ across checkpoints. Changes cannot be attributed to any single factor.</p>
<h2>Synchronized comparison</h2><video controls preload="metadata" poster="comparison_poster.png" src="model_comparison.mp4"></video>
<h2>Response across the predicted horizon</h2><p>The shaded first 0.2 seconds is actually executed before replanning. Full horizon durations differ across configurations. Later targets can respond more strongly than immediate targets, so the full response is retained here.</p><img style="width:100%" src="state80_response_by_time.png" alt="Cube relocation response across time into the action chunk">
<div class="scroll"><table><thead><tr><th>Model</th><th>Updates</th><th>X+100mm response(mm)</th><th>Y+100mm response(mm)</th><th>First close</th><th>XY miss(mm)</th><th>Any grasp flag</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table></div><div class="cards">"""
        + "".join(cards)
        + """</div><p><a href="../grounding/validation.json">Independent array/media validation and detailed summary</a></p></html>"""
    )
    (ROOT / "rollouts/index.html").write_text(page)
    (ROOT / "grounding/index.html").write_text(
        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../rollouts/index.html"><a href="../rollouts/index.html">Open grounding and rollout comparison</a>'
    )
