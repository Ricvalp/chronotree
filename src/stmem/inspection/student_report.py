"""Human-readable summaries of the exploratory local student utility screen.

Plots preserve signed effects and show repeated-noise variability. Aggregated
means first average candidates within a state, then states within episodes and
episodes within tasks. No population confidence interval is claimed from the
two local episodes, and loss-selected packets are not called learned routers.
"""
from __future__ import annotations

from collections import defaultdict
from html import escape
import json
import math
from pathlib import Path
import textwrap

import numpy as np

from stmem.training.checkpoints import atomic_json

SCHEMA = "local_student_diagnostic_report_v1"


def _number(value):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Diagnostic summaries require finite measurements")
    return value


def _episode_mean(rows, field):
    states = defaultdict(list)
    for row in rows:
        value = _number(row.get(field))
        if value is not None:
            states[row["task"], row["episode_id"], row["state_id"]].append(value)
    episodes = defaultdict(list)
    for (task, episode, _), values in states.items():
        episodes[task, episode].append(float(np.mean(values)))
    tasks = defaultdict(list)
    for (task, _), values in episodes.items():
        tasks[task].append(float(np.mean(values)))
    return {"mean": float(np.mean([np.mean(values) for values in tasks.values()])) if tasks else None,
            "states": len(states), "episodes": len(episodes), "tasks": len(tasks)}


def _group_summaries(rows, keys, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [{**dict(zip(keys, key)), "rows": len(values),
             "metrics": {field: _episode_mean(values, field) for field in metrics}}
            for key, values in sorted(groups.items())]


def summarize_local_measurements(records):
    """Summarize stored per-state results without rescoring or selecting cases."""
    gains, selections, capabilities, interventions, numerical = [], [], [], [], []
    omissions, states, seen = [], [], set()
    for record in records:
        state = record["state"]
        shared = {"student": record["label"], "task": state["task"], "episode_id": state["episode_id"],
                  "state_id": state["state_id"], "decision_index": state.get("decision_index"),
                  "phase": state.get("phase", "unspecified")}
        identity = tuple(shared[key] for key in ("student", "task", "episode_id", "state_id"))
        if identity in seen:
            raise ValueError("Duplicate student/state measurement would double-count evidence")
        seen.add(identity)
        states.append(shared)
        epsilon = _number(record.get("effective_tie_epsilon", 1e-6))
        if epsilon is None or epsilon < 0:
            raise ValueError("Diagnostic tie deadband must be nonnegative")
        summary = record["summary"]
        for row in summary.get("gains", []):
            # In the single-move audit, 'first' and 'complete' describe the
            # same operation. Count and plot that observation just once.
            if row["kind"] == "complete" and row["delta_tokens"] == 1:
                continue
            discovery, confirmation = _number(row["discovery_gain"]), _number(row["confirmation_gain"])
            if discovery is None or confirmation is None:
                raise ValueError("A measured gain needs discovery and confirmation values")
            nontrivial = abs(discovery) > epsilon and abs(confirmation) > epsilon
            gains.append({**shared, **row, "epsilon": epsilon,
                          "discovery_near_zero": abs(discovery) <= epsilon,
                          "confirmation_near_zero": abs(confirmation) <= epsilon,
                          "sign_agreement_non_ties": ((discovery > 0) == (confirmation > 0))
                          if nontrivial else None,
                          "confirmation_gain_above_deadband": confirmation > epsilon})
        for row in summary.get("selection", []):
            choices = row["selection"]
            selections.append({**shared, **row,
                               "discovery_packet_tie": len(choices["packet"]) > 1,
                               "discovery_immediate_tie": len(choices["immediate"]) > 1})
        for name, row in summary.get("capability", {}).items():
            if "omitted" in row:
                omissions.append({**shared, "comparison": name, "reason": row["omitted"]})
            else:
                capabilities.append({**shared, "context": name, **row})
        teacher = summary.get("teacher_reference", {})
        if teacher.get("available") and teacher.get("expert_full") is not None:
            capabilities.append({**shared, "context": "teacher_full", "expert_full": teacher["expert_full"],
                                 "tokens": None})
        for name, row in summary.get("interventions", {}).items():
            interventions.append({**shared, "intervention": name, **row})
        numeric = record.get("numerical", {})
        numerical.append({**shared, **numeric, "effective_tie_epsilon": epsilon})
    gain_summaries = _group_summaries(
        gains, ("student", "reader", "family", "background", "kind", "delta_tokens"),
        ("discovery_gain", "confirmation_gain", "confirmation_noise_sd", "confirmation_positive_draw_fraction",
         "discovery_near_zero", "confirmation_near_zero", "sign_agreement_non_ties",
         "confirmation_gain_above_deadband"))
    eligible = [row for row in selections if row.get("ranking_eligible")]
    selection_summaries = _group_summaries(
        eligible, ("student", "reader", "family", "budget"),
        ("packet_loss", "random_loss", "packet_vs_random", "packet_vs_immediate", "packet_vs_dispersed",
         "packet_vs_low", "discovery_packet_tie", "discovery_immediate_tie", "winner_reversal",
         "expert_teacher_selection_disagreement"))
    numerical_summaries = []
    for student in sorted({row["student"] for row in numerical}):
        rows = [row for row in numerical if row["student"] == student]
        values = {}
        for field in ("noop_prediction_max_abs", "noop_loss_max_abs", "serial_packed_prediction_max_abs",
                      "serial_packed_loss_max_abs", "effective_tie_epsilon"):
            measured = [_number(row.get(field)) for row in rows if row.get(field) is not None]
            values[field] = max(measured) if measured else None
        numerical_summaries.append({"student": student, "states": len(rows), "maximum": values})
    return {"schema": SCHEMA, "students": sorted({row["student"] for row in states}),
            "measured_student_state_cells": len(states), "gain_summaries": gain_summaries,
            "selection_summaries": selection_summaries,
            "capability": _group_summaries(capabilities, ("student", "context"), ("expert_full", "tokens")),
            "interventions": _group_summaries(interventions, ("student", "intervention"),
                                               ("expert_full", "baseline_expert_full", "loss_change", "tokens")),
            "numerical": numerical_summaries, "omissions": omissions,
            "plot_rows": {"gains": gains, "selections": selections},
            "aggregation": "Candidates within states, states within episodes, episodes within tasks; task means averaged equally",
            "uncertainty": "No population confidence intervals from two local episodes; candidates/noise draws are correlated",
            "notes": [
                "Signed gain is expert FM loss before minus after; negative means refinement worsened the surrogate.",
                "Discovery chooses a packet; independent confirmation measures it. This is not a learned-router result.",
                "Noise SD describes repeated paired flow-noise draws within a fixed state, not episode uncertainty.",
                "Winner reversal is defined only when discovery and confirmation each have one untied winner.",
                "Numerical near-ties are not confident positive or negative routing labels.",
                "Absolute capability loss and generated actions must support any claim that allocation is useful.",
                "Input deletion changes the observation distribution and coverage; it is a sensitivity probe, not a router label.",
                "Generated actions and two training scenes do not estimate benchmark success or unseen-episode generalization.",
            ]}


def _formatted(value):
    return "—" if value is None else f"{value:.5g}"


def write_diagnostic_figures(root, summary):
    """Generate static plots and an evidence page entirely from stored results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary.get("schema") != SCHEMA:
        raise ValueError("Unsupported local student report schema")
    root = Path(root)
    folder = root / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    students = summary["students"]
    plots = []
    plt.rcParams.update({"font.size": 10})

    def axes_grid(title):
        count = max(1, len(students))
        figure, axes = plt.subplots(1, count, figsize=(5 * count, 4.7), squeeze=False)
        figure.suptitle(textwrap.fill(title, width=54 * count), fontsize=12)
        if not students:
            axes[0, 0].text(.5, .5, "No completed measurements", ha="center", transform=axes[0, 0].transAxes)
        return figure, axes[0]

    def save(figure, filename, label, caption):
        figure.tight_layout(rect=(0, 0, 1, .88))
        figure.savefig(folder / filename, dpi=150, bbox_inches="tight")
        plt.close(figure)
        plots.append({"label": label, "path": f"diagnostics/{filename}", "caption": caption})

    figure, axes = axes_grid("Signed refinement gains: discovery versus independent confirmation")
    for axis, student in zip(axes, students):
        points = [row for row in summary["plot_rows"]["gains"] if row["student"] == student]
        for reader, color in (("TREE", "#315e9b"), ("FLAT", "#c66731")):
            for kind, marker in (("first", "o"), ("complete", "^")):
                rows = [row for row in points if row["reader"] == reader and row["kind"] == kind]
                if rows:
                    axis.scatter([r["discovery_gain"] for r in rows], [r["confirmation_gain"] for r in rows],
                                 color=color, marker=marker, s=20, alpha=.6, label=f"{reader} {kind}")
        values = [row[field] for row in points for field in ("discovery_gain", "confirmation_gain")]
        limit = max([abs(value) for value in values] + [1e-8]) * 1.08
        axis.plot([-limit, limit], [-limit, limit], "--", color="#969da5", linewidth=1)
        axis.axhline(0, color="#9ba5ae", linewidth=.7)
        axis.axvline(0, color="#9ba5ae", linewidth=.7)
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), title=student,
                 xlabel="Discovery mean signed gain", ylabel="Confirmation mean signed gain")
        if points:
            axis.legend(fontsize=8)
    save(figure, "gain_repeatability.png", "Gain repeatability",
         "Each point is one candidate/state comparison; these correlated points are not independent episodes. "
         "The diagonal represents identical discovery and confirmation effects. Both negative and positive gains remain visible.")

    figure, axes = axes_grid("Within-state noise variability versus the confirmed refinement gain")
    for axis, student in zip(axes, students):
        points = [row for row in summary["plot_rows"]["gains"] if row["student"] == student]
        for kind, marker, color in (("first", "o", "#315e9b"), ("complete", "^", "#c66731")):
            rows = [row for row in points if row["kind"] == kind and row.get("confirmation_noise_sd") is not None]
            if rows:
                axis.scatter([r["confirmation_noise_sd"] for r in rows],
                             [r["confirmation_gain"] for r in rows], s=20, alpha=.6,
                             marker=marker, color=color, label=kind)
        axis.axhline(0, color="#9ba5ae", linewidth=.7)
        axis.set(title=student, xlabel="SD across paired confirmation draws",
                 ylabel="Mean signed confirmation gain")
        if points:
            axis.legend(fontsize=8)
    save(figure, "gain_noise.png", "Gain versus flow-noise variation",
         "A gain small relative to repeated-noise variation may be difficult to rank reliably from one draw. "
         "These SDs are descriptive; a small number of draws does not establish precise per-candidate significance.")

    figure, axes = axes_grid("Absolute expert FM loss under capability controls (lower is better)")
    for axis, student in zip(axes, students):
        rows = [row for row in summary["capability"] if row["student"] == student]
        rows = [row for row in rows if row["metrics"]["expert_full"]["mean"] is not None]
        axis.barh([row["context"].replace("_", " ") for row in rows],
                  [row["metrics"]["expert_full"]["mean"] for row in rows], color="#507ba4")
        axis.set(title=student, xlabel="Episode-averaged confirmation expert FM loss")
        axis.invert_yaxis()
    save(figure, "capability.png", "Absolute capability losses",
         "Controls share state/noise. Full-leaf student input can be outside compact training support and is not a guaranteed upper bound. "
         "Missing controls are listed explicitly. The teacher appears only when its exact checkpoint was supplied.")

    figure, axes = axes_grid("History erasure controls: change in expert FM loss")
    for axis, student in zip(axes, students):
        rows = [row for row in summary["interventions"] if row["student"] == student]
        rows = [row for row in rows if row["metrics"]["loss_change"]["mean"] is not None]
        axis.barh([row["intervention"].replace("_", " ") for row in rows],
                  [row["metrics"]["loss_change"]["mean"] for row in rows], color="#628479")
        axis.axvline(0, color="#52606d", linewidth=.8)
        axis.set(title=student, xlabel="Loss after deletion − ordinary TREE/B128 loss")
        axis.invert_yaxis()
    save(figure, "history_sensitivity.png", "History-bank sensitivity",
         "Positive values mean deletion worsened expert FM fit. These interventions change coverage and may be out of distribution; "
         "they are not exact-budget refinement labels and do not by themselves prove semantic memory use.")

    atomic_json(summary, folder / "summary.json")
    tables = []
    headers = ("Student", "Reader", "Family", "Budget", "Packet tie fraction", "Untied reversal fraction", "Reversal states")
    rows = []
    for row in summary["selection_summaries"]:
        metrics = row["metrics"]
        rows.append([row["student"], row["reader"], row["family"], str(row["budget"]),
                     _formatted(metrics["discovery_packet_tie"]["mean"]),
                     _formatted(metrics["winner_reversal"]["mean"]), str(metrics["winner_reversal"]["states"])])
    tables.append(("Discovery ties and winner reversals", headers, rows))
    headers = ("Student", "Context", "Expert loss", "States", "Episodes", "Mean history records")
    rows = [[row["student"], row["context"], _formatted(row["metrics"]["expert_full"]["mean"]),
             str(row["metrics"]["expert_full"]["states"]), str(row["metrics"]["expert_full"]["episodes"]),
             _formatted(row["metrics"]["tokens"]["mean"])] for row in summary["capability"]]
    tables.append(("Absolute capability measurements", headers, rows))
    headers = ("Student", "States", "No-op prediction max", "Serial/packed loss max", "Max tie deadband")
    rows = [[row["student"], str(row["states"]), _formatted(row["maximum"]["noop_prediction_max_abs"]),
             _formatted(row["maximum"]["serial_packed_loss_max_abs"]),
             _formatted(row["maximum"]["effective_tie_epsilon"])] for row in summary["numerical"]]
    tables.append(("Numerical floor", headers, rows))
    paragraphs = ["These results test whether candidate allocations contain reproducible expert-action utility. "
                  "Inspect confirmation gains, absolute prediction quality and generated chunks together; a large gain alone can reflect a poor base context.",
                  summary["aggregation"], summary["uncertainty"], *summary["notes"]]
    page = ["<!doctype html><html><head><meta charset='utf-8'><title>Student routing-label evidence</title>",
            "<style>body{font:16px/1.5 system-ui;max-width:1450px;margin:28px auto;padding:0 20px;color:#263545}"
            "img{width:100%;height:auto;border:1px solid #d9e0e6}table{border-collapse:collapse}"
            "th,td{padding:7px 12px;border-bottom:1px solid #d9e0e6;text-align:left}.note{background:#eff4f8;padding:14px}</style></head><body>",
            "<h1>Evidence for routing-label usefulness</h1><p><a href='../index.html'>Inspection overview</a> · "
            "<a href='summary.json'>Complete numerical summary and plotted points</a></p>",
            "<div class='note'>" + "".join(f"<p>{escape(text)}</p>" for text in paragraphs[:3]) + "</div>"]
    markdown = ["# Evidence for routing-label usefulness", "", *paragraphs[:3], ""]
    for plot in plots:
        image = Path(plot["path"]).name
        page.extend([f"<h2>{escape(plot['label'])}</h2><img src='{escape(image, quote=True)}' alt='{escape(plot['label'], quote=True)}'>",
                     f"<p>{escape(plot['caption'])}</p>"])
        markdown.extend([f"## {plot['label']}", "", f"![{plot['label']}]({image})", "", plot["caption"], ""])
    for title, headers, rows in tables:
        page.extend([f"<h2>{escape(title)}</h2><table><tr>" + "".join(f"<th>{escape(c)}</th>" for c in headers) + "</tr>",
                     *["<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in rows], "</table>"])
        markdown.extend([f"## {title}", "", "| " + " | ".join(headers) + " |",
                         "| " + " | ".join("---" for _ in headers) + " |",
                         *["| " + " | ".join(row) + " |" for row in rows], ""])
    page.append("<h2>Interpretation limits and omissions</h2><ul>" + "".join(f"<li>{escape(text)}</li>" for text in paragraphs[3:]) + "</ul>")
    page.append("<pre>" + escape(json.dumps(summary["omissions"], indent=2)) + "</pre></body></html>")
    (folder / "index.html").write_text("\n".join(page))
    (folder / "report.md").write_text("\n".join([*markdown, *[f"- {text}" for text in paragraphs[3:]], ""]))
    return [{"label": "Routing-label evidence: gains, repeatability, capability and sensitivity", "path": "diagnostics/index.html"},
            *[{"label": plot["label"], "path": plot["path"]} for plot in plots]]
