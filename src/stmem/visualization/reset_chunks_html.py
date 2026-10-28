"""Offline point-cloud views of matched first-chunk cube-position probes.

The input contains measured simulator observations and already generated action
chunks. This renderer changes neither observations nor predictions. FK target
positions are displayed at their actual scale; they are not executed trajectories.
"""

from __future__ import annotations

import html
from importlib.resources import files
import json
from pathlib import Path
import re


def _encoded(value: dict) -> str:
    result = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                               ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        result = result.replace(character, escaped)
    return result


def _validate(payload: dict) -> None:
    if payload.get("schema_version") != 1 or payload.get("decision_step") != 0:
        raise ValueError("Reset-chunk HTML requires schema_version=1 and decision_step=0")
    for field in ("models", "scenes", "noise_seeds", "horizon", "execute_steps", "precision"):
        if not payload.get(field):
            raise ValueError(f"Reset-chunk payload is missing {field}")
    ids = [model["id"] for model in payload["models"]]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", ident) for ident in ids):
        raise ValueError("Model IDs must be unique safe filename components")
    scene_ids = [scene["id"] for scene in payload["scenes"]]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Scene IDs must be unique")
    if any(payload["scenes"][0]["shift_xyz_m"]):
        raise ValueError("The first scene must be the unshifted reference")
    if not 1 <= payload["execute_steps"] <= payload["horizon"]:
        raise ValueError("Invalid executed horizon")
    for scene in payload["scenes"]:
        if not scene["points"] or any(len(point) != 6 for point in scene["points"]):
            raise ValueError("Scene points must be nonempty XYZRGB records")
        if len(scene["current_tcp_pose"]) != 4:
            raise ValueError("Each scene needs the observed world TCP transform")
        for ident in ids:
            for seed in payload["noise_seeds"]:
                pred = scene["predictions"][ident][str(seed)]
                if len(pred["tcp_xyz"]) != payload["horizon"]:
                    raise ValueError("Every prediction must retain the entire action horizon")
                for key in ("raw_actions", "physical_actions"):
                    if len(pred[key]) != payload["horizon"] or any(len(row) != 8 for row in pred[key]):
                        raise ValueError(f"Invalid {key} shape")
    _encoded(payload)  # Reject NaN/Inf instead of silently producing broken Javascript.


def write_reset_chunks_html(payload: dict, output_dir: str | Path) -> dict[str, Path]:
    """Write an index and one self-contained interactive HTML per checkpoint.

    All libraries, point clouds and predictions are embedded. The pages work
    directly from disk, with no HTTP server or network access. ``scenes[0]`` is
    the unshifted reference used by the displacement chart.
    """
    _validate(payload)
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise RuntimeError("Install the visualization extra to render offline point clouds") from exc
    assets = files("stmem.visualization")
    style = assets.joinpath("reset_chunks_viewer.css").read_text(encoding="utf-8")
    viewer = assets.joinpath("reset_chunks_viewer.js").read_text(encoding="utf-8")
    plotly = get_plotlyjs().replace("</script", "<\\/script")
    encoded = _encoded(payload)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    nav = "".join(f'<a href="{m["id"]}.html">{html.escape(m["label"])}</a>' for m in payload["models"])
    episode_label = f'{html.escape(str(payload["task"]).upper())} · {html.escape(str(payload["split"]).upper())} SCENE {int(payload["episode_idx"])}'
    outside = [html.escape(s["label"]) for s in payload["scenes"] if s.get("inside_original_cube_center_sampling_xy_bounds") is False]
    outside_note = (" Outside the original cube-center spawn range: " + ", ".join(outside) + ".") if outside else ""
    cards = []
    for model in payload["models"]:
        ident, label = model["id"], html.escape(model["label"])
        condition = "Joint-command conditioning enabled" if model["condition_on_previous_joint_commands"] else "Joint-command conditioning disabled"
        cards.append(f'<a class="model-card" href="{ident}.html"><span class="eyebrow">{model["step"]:,} TRAINING STEPS</span><h2>{label}</h2><p>{condition}</p><strong>Open interactive point cloud →</strong></a>')
        document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{label} · First action chunk · PickXtimes</title><style>{style}</style></head>
<body><div class="shell">
<header class="topbar"><a href="index.html" class="brand">CHRONOTREE <span>/ First action chunk</span></a><span class="offline">Self-contained · offline</span></header>
<nav class="model-nav" aria-label="Checkpoint">{nav}</nav>
<section class="intro"><div class="eyebrow">{episode_label} · FIRST DECISION AFTER RESET</div><h1>{label}</h1>
<p>Move the cube, keep the robot and noise fixed. Compare the first <b>{payload["horizon"]} predicted joint-target poses</b> for each cube position.</p>
<div class="badges"><span>{model["step"]:,} training steps</span><span>{condition}</span><span>No execution history or available previous commands</span><span id="precision-badge"></span></div></section>
<div id="runtime-error" role="alert" hidden></div><noscript><p>JavaScript is required for the interactive point cloud.</p></noscript>
<section class="workspace"><aside class="controls">
<label for="scene">Visible point-cloud scene</label><select id="scene"></select><p class="help">Only this scene’s observed RGB points are displayed. Colored boxes identify the alternative cube positions.</p>
<label for="seed">Matched flow-noise seed</label><select id="seed"></select><p class="help">The same noise is reused across all cube positions and checkpoints.</p>
<div class="divider"></div><label class="check"><input id="all-paths" type="checkbox" checked>Compare all cube positions</label>
<label class="check"><input id="show-cloud" type="checkbox" checked>Show observed points</label>
<label class="check"><input id="show-cubes" type="checkbox" checked>Show cube boxes</label>
<label class="check"><input id="show-axes" type="checkbox">Show target orientations</label>
<label class="check"><input id="show-labels" type="checkbox">Number all target points</label>
<label for="point-size" class="range-label">Point size <output id="point-size-value">3</output></label><input id="point-size" type="range" min="1" max="7" step="0.5" value="3">
<div class="divider"></div><div id="scene-info" class="scene-info"></div><div id="scene-legend" class="scene-legend"></div>
<details class="camera-details"><summary>Front + wrist camera images</summary><img id="camera-image" alt="Rendered front and wrist views of the selected cube position"></details>
</aside><div class="stage"><div class="plot-toolbar"><span>World coordinates · metres · equal axis scale</span><div class="view-buttons"><button data-view="world">Whole scene</button><button data-view="actions">Action targets</button><button data-view="cube">Cube area</button><button data-view="top">Top view</button></div></div>
<div id="cloud" aria-label="Interactive observed RGB point cloud and predicted action target positions"></div>
<div class="plot-caption"><span><b>Large circles:</b> first {payload["execute_steps"]} targets · <b>small diamonds:</b> remaining targets · <b>black diamond:</b> observed TCP at reset.</span><span>Drag to orbit · scroll to zoom · hover for target details</span></div>
<div class="important">Lines connect the {payload["horizon"]} discrete targets in command order. They are <b>not an executed robot trajectory or interpolated actions</b>. All paths use their true world scale. Cube boxes are diagnostic annotations, never policy inputs.</div>
</div></section>
<section class="response-section"><div class="section-heading"><div><h2>How much does the generated chunk change?</h2><p>Per-command TCP displacement from the unshifted-cube prediction, using the same noise. This separate chart uses millimetres; the 3D view above is never magnified.</p></div><span class="reference-tag">Reference: original cube position</span></div>
<div id="response" aria-label="TCP displacement from baseline in millimetres by command"></div>
<div class="table-scroll"><table><thead><tr><th>Cube position</th><th>Cube displacement</th><th>Mean TCP change<br>first {payload["execute_steps"]}</th><th>Mean TCP change<br>all {payload["horizon"]}</th><th>Largest TCP change</th><th>Gripper changes<br>all {payload["horizon"]}</th></tr></thead><tbody id="metrics"></tbody></table></div></section>
<section class="explain"><article><h2>What am I looking at?</h2><p>The RGB cloud is the sampled input reconstructed from both simulator cameras at the first policy decision. The robot is visible where a camera sees it. The black diamond marks its current tool center point (TCP).</p><p>Each colored path converts one predicted chunk of seven absolute joint targets into world TCP positions using the Panda’s forward kinematics. Colors identify cube positions, not time. Hover a point to inspect its command number, gripper command and joint targets.</p></article>
<article><h2>What stays fixed?</h2><p>The reset configuration, instruction, robot state, previous-command availability and flow-noise sample are held fixed. Only the target cube is relocated before the simulator renders the reset observation. Each changed scene is rendered and sampled anew.</p><p>All models receive no previous commands at reset; differences between checkpoints reflect their learned weights. No predicted actions are executed. Artificial cube relocations are a sensitivity diagnostic, not official validation episodes or task-success scores.{outside_note}</p></article>
<article><h2>How to compare</h2><p>Start with all paths visible, then use <b>Action targets</b> to inspect overlap. Change the visible scene to see where the cube moved. The response chart reveals changes too small to distinguish at the full scene scale.</p><p>Orientation axes, when enabled, show local X/red, Y/green and Z/blue at commands 1, {payload["execute_steps"]}, 8, 12 and {payload["horizon"]}. They do not depict gripper opening. The fresh 16k checkpoint has less training than the matched 64,978-step continuations.</p></article></section>
<details class="provenance"><summary>Protocol, checkpoint identity and export</summary><p id="notes"></p><pre id="provenance"></pre><button id="download">Download complete JSON data</button></details>
<footer>Chronotree · first-decision cube-position sensitivity · generated from measured simulator observations</footer></div>
<script id="reset-chunks-data" type="application/json">{encoded}</script><script>window.RESET_CHUNKS_MODEL={json.dumps(ident)};</script><script>{plotly}</script><script>{viewer}</script></body></html>'''
        path = output_dir / f"{ident}.html"
        path.write_text(document, encoding="utf-8")
        paths[ident] = path
    index = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PickXtimes · first action chunk · cube-position comparison</title><style>{style}</style></head><body><div class="shell index-shell"><header class="topbar"><span class="brand">CHRONOTREE <span>/ First action chunk</span></span><span class="offline">Self-contained · offline</span></header><section class="intro"><div class="eyebrow">{episode_label} · FIRST DECISION AFTER RESET</div><h1>Does the first action chunk follow the cube?</h1><p>Inspect the same reset scene with {len(payload["scenes"])} cube positions. Each interactive point cloud overlays all {payload["horizon"]} predicted TCP targets, with matched noise and an unchanged robot state.</p></section><div class="model-grid">{''.join(cards)}</div><section class="index-guide"><h2>Open a checkpoint above</h2><p>Choose a visible cube-position scene and keep <b>Compare all cube positions</b> enabled. Colored boxes locate the cubes; matching paths show their generated chunks. Large circles emphasize the first {payload["execute_steps"]} targets that normal rollout would execute before replanning.</p><p>The paths are forward-kinematic positions of discrete joint targets, not executed motion. The separate millimetre chart reveals small changes without enlarging the 3D paths. Every HTML file embeds its point clouds, predictions and plotting library; it opens directly from disk without internet.</p><p><b>Comparison scope:</b> the two 64,978-step continuations are matched. The fresh ablation is at 16,000 steps. No policy actions are executed, and these artificial cube relocations are diagnostics, not benchmark episodes.</p></section><footer>All files contain measured first-decision predictions · no interpolation of actions · no future observations</footer></div></body></html>'''
    paths["index"] = output_dir / "index.html"
    paths["index"].write_text(index, encoding="utf-8")
    return paths
