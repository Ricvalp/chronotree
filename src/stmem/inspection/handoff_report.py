"""Build plots, a video gallery and matched-model videos from saved traces only."""

from __future__ import annotations

from collections import defaultdict
import csv
import math
import html
import json
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from stmem.training.checkpoints import file_sha256

ROOT = None
LABELS = {}
COLORS = {0: "#334155", 4: "#3b82f6", 16: "#f59e0b", 32: "#0d9488", 64: "#8b5cf6"}


def outcome(run):
    return (
        f"Diagnostic cap ({run['cap_total_commands']})"
        if run["status"] == "stopped_at_declared_diagnostic_cap"
        else run["status"]
    )


def collect():
    runs = []
    for path in sorted(ROOT.glob("*/*/*/*/result.json")):
        r = json.loads(path.read_text())
        if r.get("kind") != "real_simulator_training_scene_expert_prefix_handoff_not_benchmark_score":
            continue
        r["_path"] = str(path.relative_to(ROOT))
        r["_video"] = str((path.parent / "rollout.mp4").relative_to(ROOT))
        runs.append(r)
    return runs


def row(run):
    timeline = run["timeline"]
    first_grasp = next((r for r in timeline if r["target_grasped"]), None)
    grasp_at_handoff = run.get("handoff", {}).get("diagnostics", {}).get("target_grasped")
    return {
        "task": run["task"],
        "episode": run["episode_idx"],
        "variant": run["variant"],
        "checkpoint_step": run["checkpoint_step"],
        "prefix": run["prefix_commands"],
        "expert_control": run["expert_control"],
        "status": run["status"],
        "total_commands": len(timeline),
        **run["summary"],
        "highest_active_subtask_index": max((r.get("current_task_index", 0) for r in timeline), default=0),
        "grasp_at_handoff": grasp_at_handoff,
        "first_grasp_in_full_trace_step": None if first_grasp is None else first_grasp["step"],
        "first_grasp_in_full_trace_phase": None if first_grasp is None else first_grasp["phase"],
        "first_grasp_only_after_handoff": bool(first_grasp is not None and first_grasp["phase"] == "policy"),
        "replay_max_joint_error_rad": max(
            (r["joint_max_abs_difference_rad"] for r in run["prefix_replay_checks"]), default=0
        ),
        "video": run["_video"],
        "result": run["_path"],
    }


def plot_group(key, runs):
    task, episode, variant = key
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    for r in sorted(runs, key=lambda r: r["prefix_commands"]):
        rows = [x for x in r["timeline"] if x["observation_available"]]
        x = np.array([z["step"] for z in rows])
        prefix = r["prefix_commands"]
        c = COLORS[prefix]
        for ax, field in zip(axes, ("tcp_target_xy_distance_mm", "target_height_m"), strict=True):
            y = np.array([z[field] for z in rows])
            ax.plot(x, y, color=c, label=f"Prefix {prefix}" + (" (secondary)" if prefix == 64 else ""))
            pre = x <= prefix
            ax.plot(x[pre], y[pre], color=c, lw=5, alpha=0.25)
            if len(x):
                value = r["handoff"]["diagnostics"][field]
                ax.scatter([prefix], [value], color=c, s=25)
    axes[0].set_ylabel("TCP–target XY distance (mm)")
    axes[1].set_ylabel("Target height (m)")
    axes[1].axhline(0.15 if task == "VideoUnmask" else 0.05, color="#64748b", ls=":")
    axes[1].set_xlabel("Primitive execution step (conditioning excluded)")
    axes[0].legend(ncols=3, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{task} training scene {episode} · {LABELS[variant]}")
    fig.text(
        0.5,
        0.015,
        "Dots mark policy takeover; thick segments are expert prefixes. Fixed noise seed 2701.\n"
        "Small training-scene diagnostic; a 320-step cap is not an official failure.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    target = ROOT / "figures" / f"{task}_ep{episode}_{variant}"
    target.parent.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(target.with_suffix("." + ext), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(target.with_suffix(".png").relative_to(ROOT))


def comparisons(runs):
    lookup = {
        (r["task"], r["episode_idx"], r["variant"], r["prefix_commands"]): r
        for r in runs
        if not r["expert_control"]
    }
    output = ROOT / "comparisons"
    output.mkdir(exist_ok=True)
    records = []
    for task, ep in (("VideoUnmask", 0), ("VideoUnmask", 1), ("PickXtimes", 0)):
        for prefix in (0, 32, 64):
            keys = [(task, ep, variant, prefix) for variant in LABELS]
            if not all(key in lookup for key in keys):
                continue
            selected = [lookup[key] for key in keys]
            lengths = [imageio_ffmpeg.count_frames_and_secs(str(ROOT / r["_video"]))[0] for r in selected]
            movie = output / f"{task}_ep{ep}_prefix{prefix:03d}_models.mp4"
            identity = {
                "task": task,
                "episode": ep,
                "prefix": prefix,
                "variants_row_major": list(LABELS),
                "source_videos": [r["_video"] for r in selected],
                "source_video_sha256": [file_sha256(ROOT / r["_video"]) for r in selected],
                "checkpoint_sha256": [r["checkpoint_identity"]["sha256"] for r in selected],
                "source_frames": lengths,
                "alignment": "absolute execution time",
                "fps": 20,
                "ended_runs": "hold final frame with explicit outcome",
                "exposure_caveat": "Architectures, action horizons/cadence and training exposure differ; not a matched ablation.",
            }
            meta = movie.with_suffix(".json")
            if movie.exists() and meta.exists() and json.loads(meta.read_text()) == identity:
                records.append({**identity, "video": str(movie.relative_to(ROOT))})
                continue
            readers = [
                imageio.get_reader(ROOT / r["_video"], input_params=["-threads", "1"]) for r in selected
            ]
            last = [None] * len(selected)
            writer = imageio.get_writer(
                movie, fps=20, quality=8, macro_block_size=16, ffmpeg_params=["-threads", "2"]
            )
            try:
                for i in range(max(lengths)):
                    canvas = Image.new("RGB", (1024, 64 + math.ceil(len(selected) / 2) * 304), "#111827")
                    draw = ImageDraw.Draw(canvas)
                    phase = (
                        f"conditioning frame {i + 1}"
                        if i < selected[0]["video_conditioning_frames"]
                        else f"execution step {i - selected[0]['video_conditioning_frames']}"
                    )
                    draw.text(
                        (10, 8), f"{task} training scene {ep} · prefix {prefix} · {phase}", fill="white"
                    )
                    draw.text(
                        (10, 26),
                        "Panels follow the listed checkpoint order. Exposure and architecture differ.",
                        fill="#cbd5e1",
                    )
                    for j, (reader, count, r) in enumerate(zip(readers, lengths, selected, strict=True)):
                        if i < count:
                            last[j] = reader.get_next_data()
                        panel = Image.fromarray(last[j]).convert("RGB")
                        label = ImageDraw.Draw(panel)
                        label.rectangle((0, 0, panel.width, 16), fill="#111827")
                        label.text((7, 2), LABELS[r["variant"]], fill="white")
                        if i >= count:
                            label.rectangle((0, panel.height - 22, panel.width, panel.height), fill="#243247")
                            label.text(
                                (7, panel.height - 18),
                                f"FINAL FRAME: {outcome(r)} at {len(r['timeline'])}",
                                fill="white",
                            )
                        canvas.paste(panel, ((j % 2) * 512, 64 + (j // 2) * 304))
                    writer.append_data(np.asarray(canvas))
            finally:
                writer.close()
                for reader in readers:
                    reader.close()
            meta.write_text(json.dumps(identity, indent=2) + "\n")
            records.append({**identity, "video": str(movie.relative_to(ROOT))})
    return records


def write_gallery(runs, figures, comparison):
    groups = defaultdict(list)
    experts = []
    for r in runs:
        if r["expert_control"]:
            experts.append(r)
        else:
            groups[r["task"], r["episode_idx"], r["variant"]].append(r)

    def card(r):
        summary = r["summary"]
        p = r["prefix_commands"]
        seek = (r["video_conditioning_frames"] + p) / 20
        grasp = summary["first_target_grasp_step"]
        miss = summary["first_close_xy_distance_mm"]
        handoff_grasp = r.get("handoff", {}).get("diagnostics", {}).get("target_grasped", False)
        grasp_note = " (already grasped at takeover)" if handoff_grasp else ""
        return f'''<article><h3>{"Expert-only control" if r["expert_control"] else f"{p} expert commands → policy"}</h3>
<p class="outcome">{html.escape(outcome(r))}</p><video controls preload="metadata" poster="{html.escape(str(Path(r["_video"]).with_name("poster.jpg")))}" data-seek="{seek}"><source src="{html.escape(r["_video"])}" type="video/mp4"></video>
<p><button onclick="jump(this)">Jump to takeover</button> <a href="{html.escape(r["_video"])}">MP4</a> · <a href="{html.escape(r["_path"])}">Measurements</a></p>
<p>First simulator grasp flag during {"expert replay" if r["expert_control"] else "policy control"}: <b>{"none observed" if grasp is None else "step " + str(grasp)}</b>{grasp_note}<br>First-close XY distance: <b>{"no close command" if miss is None else f"{miss:.1f} mm"}</b></p></article>'''

    sections = []
    for key, entries in sorted(groups.items()):
        task, episode, variant = key
        figure = figures[key]
        sections.append(
            f'''<section data-task="{task}" data-model="{variant}"><h2>{task} · training scene {episode} · {LABELS[variant]}</h2><div class="cards">'''
            + "".join(card(r) for r in sorted(entries, key=lambda r: r["prefix_commands"]))
            + f'''</div><details><summary>Distance and target-height traces</summary><a href="{figure}"><img src="{figure}" alt="Measured trajectory traces"></a> <a href="{figure[:-3]}pdf">PDF</a></details></section>'''
        )
    links = "".join(
        f'<li><a href="{r["video"]}">{r["task"]} scene {r["episode"]}, prefix {r["prefix"]}: {len(LABELS)} models, matched physical time</a></li>'
        for r in comparison
    )
    options = "".join(f'<option value="{v}">{label}</option>' for v, label in LABELS.items())
    doc = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Checkpoint comparison: closed-loop handoff</title>
<style>body{font:16px/1.5 system-ui;color:#182236;background:#f5f7fb;margin:0}main{max-width:1400px;margin:auto;padding:28px}h1{line-height:1.2}section{margin:42px 0}h2{font-size:23px}.notice{background:#e3edf7;border-left:4px solid #3a78ab;padding:14px 18px}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}article{background:white;border:1px solid #dce3ed;border-radius:9px;padding:15px;overflow:hidden}article h3{margin:0}video{width:100%;background:#111827}a{color:#145ea8}button,select{font:inherit;padding:6px 10px;border:1px solid #bdc9d8;border-radius:5px;background:white}details img{max-width:100%}.outcome{font-weight:600}summary,button{cursor:pointer}[hidden]{display:none!important}@media(max-width:750px){.cards{grid-template-columns:1fr}main{padding:16px}}</style><main><h1>Checkpoint comparison: expert-prefix handoff</h1>
<p>Actual closed-loop simulation on the same three training scenes and five expert-prefix lengths as the earlier inspection. Each video retains the live conditioning sequence and every observation.</p>
<p class="notice"><b>Checkpoint steps and model labels are recorded in each section.</b> Architecture, training exposure, action horizons and cadence differ. All replan every four native commands (0.2 seconds); 10 Hz targets are held twice at the native 20 Hz control interface. Noise is paired across prefix arms within each model; H32 native-horizon draws differ after reset. These fixed scenes and one noise seed are not a validation success-rate estimate. Reaching the 320-command diagnostic cap is not an official failure.</p>
<p>Each model keeps its saved visual features, history, proprioception and action cadence. Exact input settings are in its saved configuration and per-trial measurements. Videos display the supplied conditioning for comparison even when a current-only model does not receive it.</p>
<p>VideoUnmask targets the container hiding the cube; PickXtimes targets the cube. Yellow labels mark expert control and teal labels mark policy control. Prefix 64 is the previously declared later-handoff secondary arm.</p>
<p><a href="summary.csv">Summary CSV</a> · <a href="summary.json">Summary JSON</a> · <a href="validation.json">Consistency checks</a></p>"""
    if links:
        doc += "<h2>Matched-time comparison videos</h2><ul>" + links + "</ul>"

    doc += f"""<p><label>Task <select id="task" onchange="filter()"><option value="all">All tasks</option><option>VideoUnmask</option><option>PickXtimes</option></select></label> <label>Model <select id="model" onchange="filter()"><option value="all">All checkpoints</option>{options}</select></label></p>"""
    doc += "".join(sections)
    doc += (
        '<details><summary>Successful full expert replay controls (VideoUnmask scenes 0 and 1)</summary><div class="cards">'
        + "".join(card(r) for r in experts)
        + "</div></details>"
    )
    doc += """</main><script>function jump(b){const v=b.closest('article').querySelector('video');const seek=()=>{v.currentTime=Number(v.dataset.seek)};if(v.readyState>=1)seek();else{v.addEventListener('loadedmetadata',seek,{once:true});v.load()}}function filter(){const t=document.getElementById('task').value,m=document.getElementById('model').value;for(const s of document.querySelectorAll('section')){s.hidden=(t!=='all'&&s.dataset.task!==t)||(m!=='all'&&s.dataset.model!==m);if(s.hidden)for(const v of s.querySelectorAll('video'))v.pause()}}</script></html>"""
    (ROOT / "index.html").write_text(doc)


def make_posters(runs):
    for run in runs:
        source = ROOT / run["_video"]
        reader = imageio.get_reader(source, input_params=["-threads", "1"])
        try:
            index = run["video_conditioning_frames"] + min(run["prefix_commands"], len(run["timeline"]))
            frame = reader.get_data(index)
        finally:
            reader.close()
        imageio.imwrite(source.with_name("poster.jpg"), frame)


def build_report(manifest):
    global ROOT, LABELS
    ROOT = Path(manifest["output"]) / "handoff"
    LABELS = {}
    for variant in manifest["labels"]:
        cfg = json.loads((ROOT / f"config_{variant}.json").read_text())
        step = manifest["checkpoints"][variant]["step"]
        hz = 20 / cfg["model"].get("action_stride", 1)
        LABELS[variant] = f"{variant.replace('_', ' ')} · {step:,} steps · {hz:g} Hz"
    runs = collect()
    rows = [row(run) for run in runs]
    planned = len(LABELS) * 3 * 5
    summary = {
        "kind": "fixed_training_scene_handoff_not_benchmark_score",
        "policy_trials": sum(not run["expert_control"] for run in runs),
        "expert_controls": sum(run["expert_control"] for run in runs),
        "planned_policy_trials": planned,
        "models": LABELS,
        "runs": rows,
    }
    overview = []
    for variant in LABELS:
        for task, episode in (("VideoUnmask", 0), ("VideoUnmask", 1), ("PickXtimes", 0)):
            group = [
                run
                for run in rows
                if not run["expert_control"]
                and run["variant"] == variant
                and run["task"] == task
                and run["episode"] == episode
            ]
            if not group:
                continue
            reset = next((run for run in group if run["prefix"] == 0), None)
            overview.append(
                {
                    "variant": variant,
                    "task": task,
                    "episode": episode,
                    "completed_prefixes": sorted(run["prefix"] for run in group),
                    "successful_prefixes": sorted(
                        run["prefix"] for run in group if run["status"] == "success"
                    ),
                    "grasp_prefixes": sorted(run["prefix"] for run in group if run["any_target_grasp"]),
                    "lift_prefixes": sorted(
                        run["prefix"] for run in group if run["first_target_above_threshold_step"] is not None
                    ),
                    "first_pick_place_cycle_prefixes": sorted(
                        run["prefix"]
                        for run in group
                        if task == "PickXtimes" and run["highest_active_subtask_index"] >= 2
                    ),
                    "button_stage_prefixes": sorted(
                        run["prefix"]
                        for run in group
                        if task == "PickXtimes" and run["highest_active_subtask_index"] >= 4
                    ),
                    "from_reset": reset,
                }
            )
    summary["by_model_scene"] = overview
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (ROOT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for run in runs:
        if not run["expert_control"]:
            groups[run["task"], run["episode_idx"], run["variant"]].append(run)
    figures = {key: plot_group(key, group) for key, group in groups.items()}
    comparison = comparisons(runs)
    make_posters(runs)
    write_gallery(runs, figures, comparison)
    validate_media(manifest, comparison)
    print(
        json.dumps(
            {
                "policy_trials": summary["policy_trials"],
                "expert_controls": summary["expert_controls"],
                "comparison_videos": len(comparison),
                "gallery": str(ROOT / "index.html"),
            }
        ),
        flush=True,
    )
    return summary


def validate_media(manifest, comparison):
    from html.parser import HTMLParser
    from urllib.parse import unquote, urlsplit
    from stmem.training.checkpoints import atomic_json, file_sha256

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            self.links.extend(value for name, value in attrs if name in ("href", "src", "poster") and value)

    validation = json.loads((ROOT / "validation.json").read_text())
    assert validation["status"] == "passed"
    assert len(validation["videos"]) == len(manifest["labels"]) * 15 + 2
    movies = []
    for entry in validation["videos"]:
        path = ROOT / entry["path"]
        movies.append({**entry, "sha256": file_sha256(path), "kind": "actual_rollout"})
    assert len(comparison) == 9
    for entry in comparison:
        path = ROOT / entry["video"]
        count, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
        assert count == max(entry["source_frames"])
        assert abs(seconds - count / 20) < 0.06
        assert entry["variants_row_major"] == list(manifest["labels"])
        reader = imageio.get_reader(path, input_params=["-threads", "1"])
        try:
            preview = reader.get_data(min(150, count - 1))
        finally:
            reader.close()
        imageio.imwrite(path.with_suffix(".png"), preview)
        movies.append(
            {
                "path": entry["video"],
                "frames": count,
                "seconds": seconds,
                "sha256": file_sha256(path),
                "kind": "aligned_comparison",
            }
        )
    links = Links()
    links.feed((ROOT / "index.html").read_text())
    checked = []
    for link in sorted(set(links.links)):
        url = urlsplit(link)
        if url.scheme or not url.path:
            continue
        assert (ROOT / unquote(url.path)).is_file(), link
        checked.append(link)
    report = {
        "inspection_manifest_sha256": file_sha256(Path(manifest["output"]) / "inspection_manifest.json"),
        "status": "passed",
        "rollout_videos": len(validation["videos"]),
        "comparison_videos": len(comparison),
        "checked_links": checked,
        "videos": movies,
        "notes": [
            "All rollout frames counted and timing verified independently.",
            "Comparison videos align native time, holding labelled final frames after termination.",
            "Representative comparison frames and all posters decoded from actual saved videos.",
        ],
    }
    atomic_json(report, ROOT / "media_validation.json")
    return report
