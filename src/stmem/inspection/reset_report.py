"""Independent saved-array checks and offline HTML for reset cube probes."""

from copy import deepcopy
import base64
import html
import json
from pathlib import Path
import re

import numpy as np

from stmem.training.checkpoints import file_sha256
from stmem.visualization.kinematics import panda_observation_errors, panda_tcp_pose
from stmem.visualization.reset_chunks_html import write_reset_chunks_html

LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], np.float32)
HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], np.float32)
PLACEMENTS = ["baseline", "x_plus_10cm", "y_plus_10cm", "y_minus_10cm"]
STYLE = """body{font:16px system-ui,sans-serif;background:#f3f6fa;color:#203047;margin:0}
main{max-width:1240px;margin:auto;padding:30px}p{line-height:1.6}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.card,section{background:white;border:1px solid #dce4ee;border-radius:12px;padding:22px;margin:14px 0}.card{text-decoration:none;color:#2055a2}.card h2{font-size:20px}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #e3e9ee}.scroll{overflow-x:auto}.muted{color:#526578}
@media(max-width:700px){main{padding:15px}.cards{grid-template-columns:1fr}}"""


def validate_reset_payload(payload, directory, info, manifest_sha256, reference=None):
    """Recompute camera/robot matching, physical commands, FK and response."""
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def close(a, b, tol=1e-9):
        return np.allclose(a, b, atol=tol, rtol=0)

    model = payload["models"][0]
    name = model["id"]
    horizon, execute = payload["horizon"], payload["execute_steps"]
    stride = payload["action_stride"]
    check("checkpoint", file_sha256(info["path"]) == model["checkpoint_sha256"] == info["sha256"])
    check("manifest", payload["inspection_manifest_sha256"] == manifest_sha256)
    check("completed", payload["diagnostic_complete"])
    check("step", model["step"] == info["step"])
    check("paired-noise", payload["noise_seeds"] == [2701, 2702, 2703, 2704])
    check("placements", [s["id"] for s in payload["scenes"]] == PLACEMENTS)
    check("native-replanning", execute * stride == 4)
    baseline = payload["scenes"][0]
    rows = []
    for si, scene in enumerate(payload["scenes"]):
        prefix = scene["id"]
        with np.load(directory / scene["input_npz"], allow_pickle=False) as arrays:
            check(
                prefix + "/observed-input", np.array_equal(arrays["points"][arrays["valid"]], scene["points"])
            )
        check(prefix + "/finite-points", np.isfinite(scene["points"]).all())
        check(
            prefix + "/reset-history",
            scene["conditioning_frames"]
            == scene["execution_history_frames"]
            == scene["time"]
            == scene["source_index"]
            == 0,
        )
        check(
            prefix + "/commands-unavailable",
            not scene["action_available"] and not np.any(scene["previous_action"]),
        )
        fixed = [
            "proprio",
            "previous_action",
            "action_available",
            "time",
            "goal",
            "calibration",
            "robot_base_pose",
            "other_objects_diagnostic_only",
        ]
        check(prefix + "/fixed-inputs", all(scene[k] == baseline[k] for k in fixed))
        shift = np.asarray(scene["cube"]["p"]) - np.asarray(baseline["cube"]["p"])
        check(prefix + "/physical-relocation", close(shift, scene["shift_xyz_m"], 1e-4))
        check(prefix + "/cube-visible", scene["cube_point_count"] > 0)
        check(
            prefix + "/camera-embedding",
            base64.b64decode(scene["rgb_data_uri"].split(",", 1)[1])
            == (directory / (scene["id"] + "_cameras.png")).read_bytes(),
        )
        fk_error = panda_observation_errors(
            np.asarray(scene["proprio"]), task="PickXtimes", base_pose=np.asarray(scene["robot_base_pose"])
        )
        check(
            prefix + "/observed-FK",
            fk_error["position_error_m"] < 1e-5 and fk_error["rotation_error_rad"] < 1e-3,
        )
        if reference is not None:
            other = reference["scenes"][si]
            check(
                prefix + "/same-scene-across-models",
                all(scene[k] == other[k] for k in fixed + ["points", "cube", "rgb_data_uri"]),
            )
        by_noise = []
        for seed in payload["noise_seeds"]:
            prediction = scene["predictions"][name][str(seed)]
            ref = baseline["predictions"][name][str(seed)]
            raw = np.asarray(prediction["raw_actions"], np.float32)
            physical = np.asarray(prediction["physical_actions"], np.float32)
            check(
                prefix + f"/{seed}/shape",
                raw.shape == physical.shape == (horizon, 8) and np.isfinite(raw).all(),
            )
            expected = raw.copy()
            expected[:, :7] = np.clip(expected[:, :7], LOW, HIGH)
            expected[:, 7] = np.where(expected[:, 7] >= 0, 1, -1)
            check(prefix + f"/{seed}/physical-command", np.array_equal(expected, physical))
            pose = panda_tcp_pose(
                physical[:, :7], task="PickXtimes", base_pose=np.asarray(scene["robot_base_pose"])
            )
            check(
                prefix + f"/{seed}/FK",
                close(pose, prediction["tcp_pose"]) and close(pose[:, :3, 3], prediction["tcp_xyz"]),
            )
            delta = (pose[:, :3, 3] - np.asarray(ref["tcp_xyz"])) * 1000
            distances = np.linalg.norm(delta, axis=-1)
            previous = prediction["metrics"]
            check(
                prefix + f"/{seed}/response",
                close(distances, previous["tcp_change_by_target_mm"])
                and close(distances[:execute].mean(), previous["first4_tcp_change_mean_mm"])
                and close(distances.mean(), previous["full_chunk_tcp_change_mean_mm"]),
            )
            check(
                prefix + f"/{seed}/empty-context",
                prediction["decision"]["history_tokens"]
                == prediction["decision"]["archive_fine_leaves"]
                == prediction["decision"]["decision_index"]
                == 0,
            )
            native_delta = np.repeat(delta, stride, axis=0)
            native_distances = np.linalg.norm(native_delta, axis=-1)
            unit = shift / np.linalg.norm(shift) if np.linalg.norm(shift) else np.zeros(3)
            by_noise.append(
                {
                    "seed": seed,
                    "executed_0p2s_tcp_change_mean_mm": float(native_distances[:4].mean()),
                    "first_0p8s_tcp_change_mean_mm": float(native_distances[:16].mean()),
                    "full_chunk_tcp_change_mean_mm": float(native_distances.mean()),
                    "endpoint_tcp_change_mm": float(native_distances[-1]),
                    "executed_0p2s_signed_response_along_shift_mm": float((native_delta[:4] @ unit).mean()),
                    "endpoint_signed_response_along_shift_mm": float(native_delta[-1] @ unit),
                    "native_tcp_change_by_command_mm": native_distances.tolist(),
                    "gripper_flips_executed_window": int(
                        np.sum(physical[:execute, 7] != np.asarray(ref["physical_actions"])[:execute, 7])
                    ),
                }
            )
        means = {
            key: float(np.mean([r[key] for r in by_noise]))
            for key, value in by_noise[0].items()
            if key != "seed" and isinstance(value, (int, float))
        }
        rows.append(
            {
                "scene": scene["id"],
                "label": scene["label"],
                "cube_points": scene["cube_point_count"],
                "mean_over_noise": means,
                "by_noise": by_noise,
            }
        )
    check(
        "complete-chunks",
        payload["validation"]["predicted_chunks"] == 16
        and payload["validation"]["predicted_commands_executed"] == 0,
    )
    check(
        "repeat-determinism",
        payload["validation"]["repeat_baseline_points_and_first_seed_predictions_bit_identical"],
    )
    return dict(
        label=f"{name.replace('_', ' ')} · {info['step']:,} updates",
        step=info["step"],
        sha256=info["sha256"],
        horizon=horizon,
        execute_targets=execute,
        action_stride=stride,
        observation_stride=payload["observation_stride"],
        chunk_seconds=horizon * stride / 20,
        history_mode=model["history_mode"],
        scenes=rows,
    ), checks


def build_reset_report(manifest, manifest_sha256):
    root = Path(manifest["output"]) / "reset_cube_chunks"
    models, payloads, check_names = {}, {}, []
    reference = None
    for name in manifest["labels"]:
        directory = root / name
        payload = json.loads((directory / "payload.json").read_text())
        model, checks = validate_reset_payload(
            payload, directory, manifest["checkpoints"][name], manifest_sha256, reference
        )
        models[name], payloads[name] = model, payload
        check_names.extend(name + "/" + check for check in checks)
        reference = reference or payload
    for name, payload in payloads.items():
        enriched = deepcopy(payload)
        model = models[name]
        enriched["models"][0]["label"] = model["label"]
        enriched["notes"].extend(
            [
                f"Action targets are spaced at {20 / model['action_stride']:g} Hz; {model['execute_targets']} targets cover four native commands (0.2 seconds) before replanning.",
                "The 10 Hz executor holds each predicted joint target for two native commands; there is no interpolation.",
                "Raw first4_tcp_change_mean_mm means the executed target prefix; the comparison summary uses equal physical windows.",
            ]
        )
        directory = root / name
        (directory / "render_payload.json").write_text(json.dumps(enriched, indent=2) + "\n")
        write_reset_chunks_html(enriched, directory)
        path = directory / f"{name}.html"
        viewer = path.read_text().replace(
            '<a href="index.html" class="brand">', '<a href="../index.html" class="brand">'
        )
        nav = "".join(
            f'<a href="../{ident}/{ident}.html">{html.escape(other["label"])}</a>'
            for ident, other in models.items()
        )
        viewer = re.sub(
            r'(<nav class="model-nav" aria-label="Checkpoint">).*?(</nav>)',
            lambda m: m.group(1) + nav + m.group(2),
            viewer,
            flags=re.S,
        )
        if enriched["models"][0]["visual_features"]["mode"] != "xyzrgb":
            viewer = viewer.replace(
                "The robot is visible where a camera sees it.",
                "The robot is visible where a camera sees it. This model also receives frozen CLIP descriptors at these sampled points; the displayed colors show RGB, not those descriptors.",
            )
        viewer = viewer.replace(
            "The same noise is reused across all cube positions and checkpoints.",
            "Noise is paired across cube positions within this checkpoint. Matching seed numbers do not equalize different horizon shapes across checkpoints.",
        ).replace(
            "differences between checkpoints reflect their learned weights.",
            "checkpoints may differ in visual inputs, horizon, cadence, and trained weights.",
        )
        path.write_text(viewer)
    limitations = [
        "One PickXtimes validation reset plus artificial cube translations, not a benchmark success rate.",
        "No predicted commands are executed in the reset probe; FK targets are not observed robot motion.",
        "Training exposure, visual features, proprioception, history and action cadence differ; this is not an isolated architecture ablation.",
        "Noise is paired across cube positions within each model. Matching seed numbers does not equalize different horizons or action cadences.",
    ]
    result = dict(
        status="passed",
        inspection_manifest_sha256=manifest_sha256,
        checks=len(check_names),
        check_names=check_names,
        prediction_chunks=len(models) * 16,
        discrete_targets=sum(m["horizon"] * 16 for m in models.values()),
        models=models,
        limitations=limitations,
    )
    (root / "independent_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "summary.json").write_text(
        json.dumps(dict(models=models, limitations=limitations), indent=2) + "\n"
    )
    render_index(root, models, limitations)
    return result


def render_index(root, models, limitations):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, scene_id in zip(axes, PLACEMENTS[1:]):
        for model in models.values():
            scene = next(s for s in model["scenes"] if s["scene"] == scene_id)
            response = np.mean([r["native_tcp_change_by_command_mm"] for r in scene["by_noise"]], axis=0)
            axis.plot(np.arange(1, len(response) + 1) / 20, response, label=model["label"])
        axis.axvspan(0, 0.2, color="gray", alpha=0.12)
        axis.set(title=scene_id.replace("_", " "), xlabel="Time into predicted chunk (s)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean commanded TCP displacement (mm)")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Reset: response to physically moving the cube by 100 mm")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(root / f"response_by_time.{suffix}", dpi=160)
    plt.close(fig)
    cards = "".join(
        f'<a class="card" href="{name}/{name}.html"><h2>{html.escape(model["label"])}</h2><p>{model["horizon"]} targets at {20 / model["action_stride"]:g} Hz · {model["chunk_seconds"]:g} s predicted</p><strong>Open four-position point cloud →</strong></a>'
        for name, model in models.items()
    )
    rows = []
    for model in models.values():
        for scene in model["scenes"][1:]:
            m = scene["mean_over_noise"]
            rows.append(
                f"<tr><td>{html.escape(model['label'])}</td><td>{html.escape(scene['label'])}</td><td>{m['executed_0p2s_tcp_change_mean_mm']:.3f}</td><td>{m['first_0p8s_tcp_change_mean_mm']:.3f}</td><td>{m['full_chunk_tcp_change_mean_mm']:.3f}</td><td>{m['endpoint_signed_response_along_shift_mm']:+.3f}</td></tr>"
            )
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>First action chunks · cube-position comparison</title><style>{STYLE}</style><main><p class="muted">CHRONOTREE · PICKXTIMES VALIDATION SCENE 0</p><h1>Does the first action chunk follow the cube?</h1><p>{len(models)} checkpoints, four physically rendered cube positions, and four paired noise seeds per model.</p><div class="cards">{cards}</div><section><h2>How to read the viewers</h2><p>Colored boxes show cube positions; matching paths show forward kinematics of predicted joint targets. Large circles mark the targets executed before replanning: four at 20 Hz, two at 10 Hz. Both cover 0.2 seconds. The 10 Hz executor holds each target twice. These lines join predicted targets and do not show executed robot motion.</p><p>Select <b>Action targets</b> to inspect small differences. Later targets can respond more strongly than the initial near-stationary prefix. Every point-cloud viewer works offline.</p></section><section><h2>Response across the predicted horizon</h2><img style="width:100%" src="response_by_time.png" alt="TCP response by time into the predicted chunk"><div class="scroll"><table><thead><tr><th>Model</th><th>Cube shift</th><th>First 0.2 s, mm</th><th>First 0.8 s, mm</th><th>Full chunk, mm</th><th>Endpoint along shift, mm</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section><section><h2>Scope</h2><p>{" ".join(html.escape(x) for x in limitations)}</p><p><a href="summary.json">Numerical summary</a> · <a href="independent_validation.json">Independent validation</a> · <a href="../index.html">Overall report</a></p></section></main></html>"""
    (root / "index.html").write_text(document)
