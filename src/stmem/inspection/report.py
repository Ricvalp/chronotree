"""Measured, label-independent HTML report for recurring checkpoint inspections."""

from __future__ import annotations

import csv
from html import escape
from html.parser import HTMLParser
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

from stmem.inspection.common import discover_checkpoints, read_json
from stmem.training.checkpoints import atomic_json, file_sha256


def number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def offline_rows(output):
    """Read the reusable table, or the original inspection's equivalent fields."""
    rows = read_json(Path(output) / "offline/summary_table.json")
    aliases = {
        "native4_expert_tcp_error_mm": "all_states_native4_expert_tcp_error_mm",
        "delete_current_visual_native4_change_mm": "after_reset_native4_current_deletion_change_mm",
        "delete_history_visual_native4_change_mm": "after_reset_native4_history_vision_deletion_change_mm",
    }
    models = read_json(Path(output) / "offline/summary.json")
    result = {}
    for row in rows:
        label = row.get("label", row.get("model"))
        if not label or label in result:
            raise ValueError("Offline summary has missing or duplicate model labels")
        row = dict(row)
        for new, old in aliases.items():
            if new in row:
                row[old] = row[new]
        descriptor = next((g["native4_tcp_change_mm"] for g in models.get(label, {}).get("groups", [])
                           if (g["kind"], g["variant"], g["state_subset"]) ==
                           ("visual", "zero_current_frozen_descriptors", "after_reset")), None)
        row.setdefault("after_reset_native4_zero_descriptors_change_mm", descriptor)
        result[label] = row
    return result


def matched_comparisons(audit, ground, offline, previous):
    """Only compare an earlier checkpoint with the same scientific provenance."""
    if previous is None:
        return []
    previous = Path(previous).resolve()
    old_audit = read_json(previous / "checkpoint_audit.json")["checkpoints"]
    old_ground = read_json(previous / "grounding/validation.json")["models"]
    old_offline = offline_rows(previous)
    rows = []
    for label, current in audit.items():
        old = old_audit.get(label)
        if old is None or any(current["provenance"].get(k) != old["provenance"].get(k)
                              for k in ("model", "training", "data", "effective_batch", "seed")):
            continue
        if label not in ground or label not in old_ground or label not in offline or label not in old_offline:
            continue
        rows.append({
            "model": label, "previous_step": old["step"], "step": current["step"],
            "previous_first_close_xy_mm": old_ground[label]["first_close_observed_xy_mm"],
            "first_close_xy_mm": ground[label]["first_close_observed_xy_mm"],
            "previous_expert_tcp_error_mm": old_offline[label]["all_states_native4_expert_tcp_error_mm"],
            "expert_tcp_error_mm": offline[label]["all_states_native4_expert_tcp_error_mm"],
            "scope": "Same scientific configuration and fixed diagnostic protocol; unequal training exposure",
        })
    return rows


def validate_links(output):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.targets = []

        def handle_starttag(self, tag, attrs):
            self.targets.extend(v for k, v in attrs if k in {"href", "src", "poster"} and v)

    checked, missing = [], []
    for page in sorted(output.rglob("*.html")):
        parser = Links()
        parser.feed(page.read_text())
        for target in sorted(set(parser.targets)):
            url = urlsplit(target)
            if url.scheme or not url.path or url.path.startswith("javascript:"):
                continue
            path = page.parent / unquote(url.path)
            item = {"page": str(page.relative_to(output)), "target": target}
            (checked if path.exists() else missing).append(item)
    result = {"status": "passed" if not missing else "failed", "checked_links": len(checked), "missing": missing}
    atomic_json(result, output / "report_validation.json")
    if missing:
        raise ValueError(f"Report has missing local links: {missing[:5]}")
    return result


def build_report(manifest, *, compare_to=None, allow_partial=False):
    output = Path(manifest["output"])
    identity = file_sha256(output / "inspection_manifest.json")
    audit_report = read_json(output / "checkpoint_audit.json")
    if audit_report.get("inspection_manifest_sha256") != identity:
        raise ValueError("Checkpoint audit belongs to a different inspection manifest")
    audit = audit_report["checkpoints"]
    if set(audit) != set(manifest["labels"]):
        raise ValueError("Checkpoint audit labels differ from the inspection manifest")
    status_path = output / "pipeline_status.json"
    if status_path.exists():
        from stmem.inspection.pipeline import stage_code_hash
        for stage, record in read_json(status_path).get("stages", {}).items():
            if record.get("code_sha256") != stage_code_hash(stage):
                raise ValueError(f"{stage} diagnostic code changed; choose a new output directory")
    paths = {
        "offline": "offline/validation.json", "grounding": "grounding/validation.json",
        "reset": "reset_cube_chunks/independent_validation.json", "handoff": "handoff/validation.json",
        "media": "handoff/media_validation.json",
    }
    validations = {name: read_json(output / path) for name, path in paths.items() if (output / path).exists()}
    for name, validation in validations.items():
        if validation.get("inspection_manifest_sha256") != identity:
            raise ValueError(f"{name} validation belongs to a different inspection manifest")
        if validation.get("status") != "passed":
            raise ValueError(f"{name} validation did not pass; finish the stage with --resume")
    complete = set(validations) == set(paths) and all(r.get("status") == "passed" for r in validations.values())
    if not complete and not allow_partial:
        raise ValueError("Inspection validation is incomplete; finish the missing stages with --resume")
    current = discover_checkpoints(manifest["bundle"])
    if set(current) != set(manifest["labels"]):
        raise ValueError("Checkpoint bundle labels changed during inspection")
    for label, info in current.items():
        if info["sha256"] != manifest["checkpoints"][label]["sha256"]:
            raise ValueError("Checkpoint changed during inspection")
    offline = offline_rows(output) if "offline" in validations else {}
    ground = validations.get("grounding", {}).get("models", {})
    handoff = read_json(output / "handoff/summary.json") if "handoff" in validations else {"runs": []}
    expected = len(manifest["labels"]) * len(manifest["protocol"]["handoff_scenes"]) * len(manifest["protocol"]["handoff_prefixes"])
    if complete and (handoff.get("policy_trials") != expected or handoff.get("expert_controls") != 2):
        raise ValueError("Missing planned expert-prefix trials or controls")
    comparisons = matched_comparisons(audit, ground, offline, compare_to)
    summary = {
        "status": "complete" if complete else "partial", "scope": manifest["protocol"]["scope"],
        "inspection_manifest_sha256": file_sha256(output / "inspection_manifest.json"),
        "checkpoints": {label: {"step": audit[label]["step"], "sha256": current[label]["sha256"],
                                "config": audit[label]["config"], "grounding": ground.get(label),
                                "offline": offline.get(label)} for label in manifest["labels"]},
        "handoff": handoff, "comparisons": comparisons,
        "validation": {name: {k: v for k, v in report.items() if k not in {"per_check", "models", "videos", "checked_links"}}
                       for name, report in validations.items()},
    }
    atomic_json(summary, output / "summary.json")
    if comparisons:
        with (output / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
    def link(path, title):
        if (output / path).exists():
            return f'<a href="{escape(path)}">{escape(title)}</a>'
        return f'<span class="muted">{escape(title)} (pending)</span>'

    cards, rollout_rows, sensitivity_rows, coverage_rows = [], [], [], []
    for label in manifest["labels"]:
        cfg = audit[label]["config"]
        model, step = cfg["model"], audit[label]["step"]
        stride = model.get("action_stride", 1)
        cards.append(f'<div class="card"><strong>{escape(label)}</strong>'
                     f'<p>{step:,} updates · {model["horizon"]} targets at {20/stride:g} Hz</p>'
                     + link(f"reset_cube_chunks/{label}/{label}.html", "Open four-position point cloud →")
                     + '</div>')
        if label in ground:
            g = ground[label]
            rollout_rows.append([label, f"{step:,}", g["first_close_command"], number(g["first_close_observed_xy_mm"]),
                                 number(g["first_close_commanded_xy_mm"]), number(g["first_close_tracking_mm"]),
                                 number(g["max_cube_lift_mm"]), "Yes" if g["grasped"] else "No"])
            coverage = g["state80_coverage"]["banks"]["current"]
            response = g["state80_response_mm"]
            coverage_rows.append([label, coverage["cube_points"], coverage["cube_majority_tokens"],
                                  number(response["x_plus_10cm"]["bf16"], 3), number(response["y_plus_10cm"]["bf16"], 3)])
        if label in offline:
            o = offline[label]
            sensitivity_rows.append([label, number(o["all_states_native4_expert_tcp_error_mm"]),
                                 number(o["after_reset_native4_current_deletion_change_mm"]),
                                 number(o["after_reset_native4_history_vision_deletion_change_mm"]),
                                 number(o["after_reset_native4_zero_descriptors_change_mm"])])
    handoff_rows = []
    for task, episode in manifest["protocol"]["handoff_scenes"]:
        cells = [f"{task} train {episode}"]
        for label in manifest["labels"]:
            trials = [r for r in handoff["runs"] if r["variant"] == label and r["task"] == task
                      and r["episode"] == episode and not r["expert_control"]]
            successful = sorted(r["prefix"] for r in trials if r["status"] == "success")
            cells.append(", ".join(map(str, successful)) if successful else "None" if trials else "Pending")
        handoff_rows.append(cells)
    compare_html = ""
    if comparisons:
        rows = [[r["model"], f'{r["previous_step"]:,} → {r["step"]:,}',
                 f'{number(r["previous_first_close_xy_mm"])} → {number(r["first_close_xy_mm"])}',
                 f'{number(r["previous_expert_tcp_error_mm"])} → {number(r["expert_tcp_error_mm"])}'] for r in comparisons]
        compare_html = '<section><h2>Compared with the previous inspection</h2>' + table(
            ["Model", "Training updates", "First-close XY miss (mm)", "Expert-state TCP error (mm)"], rows
        ) + '<p>Only matching model, data and training fingerprints are included. These are fixed-scene diagnostics, not evidence of generalization across the benchmark.</p><a href="comparison.csv">Download comparison CSV</a></section>'
    plot_html = ""
    if ground and offline:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [name for name in manifest["labels"] if name in ground and name in offline]
        figure, axes = plt.subplots(1, 3, figsize=(13, 4.3))
        panels = [
            ("First-close TCP–cube XY miss", "mm (lower is closer)", [ground[n]["first_close_observed_xy_mm"] for n in labels]),
            ("Expert-state action fit, next 0.2 s", "TCP error (mm)", [offline[n]["all_states_native4_expert_tcp_error_mm"] for n in labels]),
            ("Remove current visual tokens", "Change in next 0.2 s (mm)", [offline[n]["after_reset_native4_current_deletion_change_mm"] for n in labels]),
        ]
        for axis, (title, ylabel, values) in zip(axes, panels):
            values = [np.nan if v is None else v for v in values]
            bars = axis.bar(range(len(labels)), values, color=["#4265a5", "#8b5ba5", "#c8842c", "#16877f"][:len(labels)])
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
            axis.set(title=title, ylabel=ylabel, xticks=range(len(labels)),
                     xticklabels=[n.replace("_", "\n") + f'\n{audit[n]["step"]//1000}k' for n in labels])
            axis.spines[["top", "right"]].set_visible(False)
            axis.margins(y=0.18)
        figure.tight_layout()
        for suffix in ("png", "svg"):
            figure.savefig(output / f"comparison_metrics.{suffix}", dpi=160)
        plt.close(figure)
        plot_html = '<section><h2>Different measurements answer different questions</h2><img src="comparison_metrics.png" alt="Cube targeting, expert action fit and current-vision dependence"><p>A larger response to token deletion establishes dependence on visual input; it does not establish correct object localization.</p><a href="comparison_metrics.svg">Download vector plot</a></section>'
    style = '''*{box-sizing:border-box}body{margin:0;font:16px system-ui;background:#f3f6fb;color:#203247}main{max-width:1280px;margin:auto;padding:32px}h1{font-size:clamp(28px,4vw,42px)}p{line-height:1.65}a{color:#185fa9}section,.card{background:white;border:1px solid #dce4ef;border-radius:12px;padding:24px;margin:20px 0}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.card{margin:0;text-decoration:none;color:inherit}.card strong,.card span{color:#185fa9}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;vertical-align:top;padding:11px;border-bottom:1px solid #e4eaf1}th{background:#f6f8fc}img,video{width:100%;height:auto;border-radius:7px}.links{display:flex;gap:12px;flex-wrap:wrap}.links a{background:#e4edf9;padding:12px;border-radius:7px;text-decoration:none}.muted{color:#536a82}code{overflow-wrap:anywhere;font-size:12px}@media(max-width:700px){main{padding:16px}section,.card{padding:16px}}'''
    movie = '<video controls preload="metadata" poster="rollouts/comparison_poster.png" src="rollouts/model_comparison.mp4"></video>' if (output / 'rollouts/model_comparison.mp4').exists() else '<p>Rollout videos pending.</p>'
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Chronotree checkpoint inspection</title><style>{style}</style><main>
<header><p class="muted">CHRONOTREE · {escape(Path(manifest['bundle']).name)} · {summary['status'].upper()}</p><h1>Checkpoint inspection</h1><p>{len(manifest['labels'])} immutable checkpoints. Input integrity checks passed. This report separates action fit, visual dependence, cube targeting and complete task success.</p><p class="muted">Fixed validation/training scenes and unequal training exposure. These are diagnostics, not benchmark success rates or an isolated architecture comparison.</p></header>
<nav class="links">{link('reset_cube_chunks/index.html', 'Interactive point clouds')}{link('rollouts/index.html', 'Fresh rollout videos')}{link('handoff/index.html', 'Expert-prefix videos')}<a href="summary.json">Numerical results</a></nav>
{plot_html}{compare_html}
<section><h2>First action chunk at four cube positions</h2><div class="cards">{''.join(cards)}</div><p>PickXtimes validation scene 0, original cube position and X+10 cm / Y+10 cm / Y−10 cm. Both cameras are physically rerendered. Robot state, other objects and paired noise are fixed. The paths are forward kinematics of generated joint targets; counterfactual predictions are not executed.</p><p>Large target markers cover the next 0.2 seconds before replanning. The 10 Hz policy holds each target for two native commands. Later chunk targets are displayed separately so a small first-target response is not mistaken for complete visual blindness.</p></section>
<section><h2>Fresh PickXtimes validation rollout</h2><p>Each policy executes its own commands for up to 160 native steps (8 seconds). Observed XY miss is measured after the first close command; commanded XY miss uses the FK target and cube position before that command. Tracking error is 3D. These are TCP-center distances, not jaw clearance or complete grasp-pose errors.</p>{table(['Model','Updates','First close','Observed XY miss (mm)','Commanded XY miss (mm)','Tracking (mm)','Max cube rise (mm)','Any grasp flag'], rollout_rows)}<p>A brief grasp flag or a small cube displacement does not establish a sustained lift. Reaching the diagnostic cap is not an official task failure.</p>{movie}</section>
<section><h2>Cube coverage and correction from a later state</h2><p>The same saved off-target robot trajectory is replayed to native command 80, with the cube physically shifted by 100 mm. Responses below cover the imminent four native commands and average four paired noise draws. Object membership is used only to measure coverage, never as model input.</p>{table(['Model','Sampled cube points','Cube-majority current tokens','X-shift response (mm)','Y-shift response (mm)'], coverage_rows)}{link('grounding/validation.json', 'Grounding measurements and validation')}</section>
<section><h2>Expert-state fit and input sensitivity</h2><p>Released VideoUnmask training episodes 0/1, decisions 0/32/64/96/128, four paired noises. Fit averages all ten states; perturbation responses average the eight states after reset. Values below cover the same native 0.2-second window for all target cadences.</p>{table(['Model','Expert TCP error (mm)','Delete current vision: change (mm)','Delete history vision: change (mm)','Zero current descriptors: change (mm)'], sensitivity_rows)}<p>Artificial deletions may be out of distribution. The saved checks also cover previous commands, measured state, omitted inputs, packed/serial inference, early histories, precision changes and gradients without optimizer updates.</p>{link('offline/summary_table.csv', 'CSV')} · {link('offline/validation.json', 'Independent validation')}</section>
<section><h2>Expert-prefix handoffs</h2><p>{handoff.get('policy_trials',0)} of {expected} policy trials and {handoff.get('expert_controls',0)} of two expert replay controls. Cells list expert-prefix lengths yielding official task success before the 320-native-command cap.</p>{table(['Training scene',*manifest['labels']],handoff_rows)}<p>Prefixes 0/4/16/32/64 are correlated conditions on the same scenes. Partial grasp/lift/subtask progress is shown separately in the video gallery. Pick prefixes come from the verified official planner; VideoUnmask prefixes come from released demonstrations.</p>{link('handoff/index.html', 'All outcomes and videos')}</section>
<section><h2>Integrity and reproducibility</h2><p>SHA256, ZIP CRC, finite tensors, strict model/optimizer loads, sampler/scheduler boundaries and per-rank RNG records were checked. Model weights were not modified. No training, Slurm submission or test-seed tuning was performed by this inspection.</p><a href="checkpoint_audit.json">Checkpoint audit</a> · <a href="inspection_manifest.json">Immutable protocol and data identities</a> · <a href="report_validation.json">Report link validation</a><p>Reproduce with <code>uv run python scripts/inspect_checkpoints.py --bundle {escape(os.path.relpath(manifest['bundle']))} --resume</code>.</p></section></main></html>'''
    (output / "index.html").write_text(document)
    # The validation artifact links to itself; create it before checking links.
    atomic_json({"status": "checking"}, output / "report_validation.json")
    validation = validate_links(output)
    validation.update(inspection_status=summary["status"], missing_validations=sorted(set(paths)-set(validations)))
    atomic_json(validation, output / "report_validation.json")
    return output / "index.html"
