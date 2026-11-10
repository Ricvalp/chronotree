"""Rendering for saved teacher history comparisons; never runs a policy."""

from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote


LIMITATIONS = [
    "History erasure/windowing can be out of distribution. Action sensitivity alone is not useful semantic memory.",
    "Across checkpoints only reset probes share the same state. Later FULL trajectories can diverge; do not compare pooled later sensitivities as a controlled training effect.",
    "Missing or erroneous pairs do not supply a measured history benefit. Success counts retain the planned denominator.",
    "FULL shadow predictions are checked against issued held commands; TCP changes are independently recomputed from saved target coordinates.",
    "Normal timeouts are checked against the recorded simulator limits. Wall-clock cutoffs and infrastructure errors invalidate the paired audit.",
    "Permutation comparisons are not included: reset UUIDs can change same-time leaf ordering, so equal shuffle seeds alone do not ensure matched interventions.",
    "A fixed small validation diagnostic is neither an all-task benchmark nor an inference-speed benchmark. Timings include rendering, observers and probes.",
]


def protocol_text(protocol):
    return (
        f"{protocol['predicted_targets']} predicted targets at {protocol['action_hz']:g} Hz; "
        f"{protocol['observation_hz']:g} Hz observations; execute {protocol['executed_targets']}; "
        f"{protocol['flow_steps']} flow integration steps."
    )


def _amount(value):
    return "unavailable" if value is None else f"{value:+.1f}"


def markdown_report(result):
    old_step, new_step = result["old_checkpoint"]["step"], result["new_checkpoint"]["step"]
    lines = [
        "# Matched teacher history comparison",
        "",
        f"Reference: **{old_step:,} updates**; candidate: **{new_step:,} updates**.",
        "",
        f"Complete: **{result['complete']}**. Audit valid: **{result['audit_passed']}**.",
        "",
        protocol_text(result["protocol"]),
        "",
        "| Task | Arm | Reference successes / planned | Candidate successes / planned | Recorded reference / candidate |",
        "|---|---|---:|---:|---:|",
    ]
    old = {(r["task"], r["arm"]): r for r in result["old_counts"]}
    new = {(r["task"], r["arm"]): r for r in result["new_counts"]}
    for key in sorted(old.keys() | new.keys()):
        a, b = old.get(key), new.get(key)

        def show(row):
            return "not planned" if row is None else f"{row['success']}/{row['planned']}"

        recorded = f"{a['recorded'] if a else 0} / {b['recorded'] if b else 0}"
        lines.append(f"| {key[0]} | {key[1]} | {show(a)} | {show(b)} | {recorded} |")
    lines += [
        "",
        "## History benefit within each checkpoint",
        "",
        "FULL success minus control success, in percentage points; only complete, valid matched pairs contribute.",
        "",
        "| Task | Control | Reference benefit | Candidate benefit |",
        "|---|---|---:|---:|",
    ]
    for row in result["comparison"]["history_benefit"]:
        lines.append(
            f"| {row['task']} | {row['control']} | "
            f"{_amount(row['old']['success_benefit_percentage_points'])} | "
            f"{_amount(row['new']['success_benefit_percentage_points'])} |"
        )
    if result.get("recent"):
        lines += [
            "",
            "## Recent-history follow-up",
            "",
            "FULL reference rows are reused from the primary study, not new trials.",
            "",
            "| Task | Arm | Reference successes / planned | Candidate successes / planned |",
            "|---|---|---:|---:|",
        ]
        prior = {(r["task"], r["arm"]): r for r in result["recent"]["old_counts"]}
        for row in result["recent"]["new_counts"]:
            a = prior.get((row["task"], row["arm"]))
            previous = "not planned" if a is None else f"{a['success']}/{a['planned']}"
            lines.append(f"| {row['task']} | {row['arm']} | {previous} | {row['success']}/{row['planned']} |")
    lines += [
        "",
        "## Audit and interpretation",
        "",
        f"{result['passed_checks']}/{result['total_checks']} checks passed.",
        "",
        result["source_compatibility"]["interpretation"],
        "",
    ]
    if result.get("recent"):
        lines += ["Recent-window sources: " + result["recent"]["source_compatibility"]["interpretation"], ""]
    lines += [
        "Input seals: " + json.dumps(result["input_integrity"], sort_keys=True) + ". "
        "Legacy records are audited without claiming retroactive integrity seals.",
        "",
    ]
    lines.extend("- " + text for text in LIMITATIONS)
    for label, recovery in result["recovery"].items():
        if recovery:
            lines += [
                "",
                f"{label.title()} recovery: {recovery['initial_published']} original published cases "
                f"preserved; {recovery['original_missing']} originally missing cases recovered. "
                "Recovery receipts and preserved hashes are checked. This is infrastructure history, not a task outcome.",
            ]
    if result.get("media"):
        lines += [
            "",
            f"Media: {result['media']['count']} referenced videos decoded. Copied FULL references "
            "are included in this media count; it is not a count of independent trials.",
        ]
    if result["failed_checks"]:
        lines += ["", "### Failed checks", ""] + ["- " + r["check"] for r in result["failed_checks"]]
    return "\n".join(lines) + "\n"


def html_report(result, destination, old, new, recent=None):
    def href(path):
        return quote(Path(os.path.relpath(Path(path).resolve(), destination.parent.resolve())).as_posix())

    def link(path, label):
        return f'<a href="{href(path)}">{escape(str(label))}</a>'

    def table(rows, headers):
        return (
            '<div class="tablewrap"><table><thead><tr>'
            + "".join(f"<th>{escape(h)}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
            + "</tbody></table></div>"
        )

    old_step, new_step = result["old_checkpoint"]["step"], result["new_checkpoint"]["step"]
    labels = [f"Reference {old_step:,}", f"Candidate {new_step:,}"]

    def outcome_counts(before, after):
        a = {(r["task"], r["arm"]): r for r in before}
        b = {(r["task"], r["arm"]): r for r in after}
        rows = []
        for key in sorted(a.keys() | b.keys()):
            values = []
            for records in (a, b):
                r = records.get(key)
                values.append(
                    "not planned"
                    if r is None
                    else f"{r['success']}/{r['planned']} ({r['recorded']} recorded)"
                )
            rows.append([escape(key[0]), escape(key[1]), *values])
        return table(rows, ["Task", "History read", *labels])

    def video_link(row, label):
        path = row.get("video_path")
        return link(path, label) if path and Path(path).is_file() else escape(label) + " (video missing)"

    def case_table(comparison, a, b):
        rows = []
        for case in comparison["cases"]:
            key = case["task"], case["episode_idx"], case["arm"]
            if case["pending"]:
                rows.append(
                    [escape(key[0]), str(key[1]), escape(key[2]), "pending pair", "", "", "unavailable"]
                )
                continue
            ar, br = a["rows"][key], b["rows"][key]
            rows.append(
                [
                    escape(key[0]),
                    str(key[1]),
                    escape(key[2]),
                    video_link(ar, case["old_status"]),
                    video_link(br, case["new_status"]),
                    f"{case['old_commands']} → {case['new_commands']}",
                    "Matched" if all(case["input_parity"].values()) else "<strong>Mismatch</strong>",
                ]
            )
        return table(rows, ["Task", "Scene", "Arm", *labels, "Native commands", "Reset/noise"])

    def video(row, label):
        path = row.get("video_path")
        if not path or not Path(path).is_file():
            return "<p>Video missing.</p>"
        start = (row.get("environment") or {}).get("conditioning_frames_recorded", 0) / 20
        return (
            f"<div><h4>{escape(label)} · {escape(row['status'])}</h4>"
            f'<video controls preload="none" src="{href(path)}"></video>'
            f'<button data-start="{start}" onclick="const v=this.previousElementSibling;'
            f'v.currentTime=Number(this.dataset.start);v.play()">Play policy execution ({start:.2f}s)</button></div>'
        )

    videos = []
    for case in result["comparison"]["cases"]:
        if case["pending"] or (case["arm"] != "full" and case["success_change"] == 0):
            continue
        key = case["task"], case["episode_idx"], case["arm"]
        a, b = old["rows"][key], new["rows"][key]
        videos.append(
            f"<details><summary>{escape(key[0])} · scene {key[1]} · {escape(key[2])}: "
            f"{escape(a['status'])} → {escape(b['status'])}</summary>"
            f"<p>{escape((b.get('task_diagnostics') or {}).get('goal', ''))}</p>"
            f'<div class="videos">{video(a, labels[0])}{video(b, labels[1])}</div></details>'
        )
    benefit_rows = []
    for row in result["comparison"]["history_benefit"]:
        benefit_rows.append(
            [
                escape(row["task"]),
                escape(row["control"]),
                *[_amount(row[k]["success_benefit_percentage_points"]) for k in ("old", "new")],
            ]
        )
    benefits = table(benefit_rows, ["Task", "Control", *(x + " benefit (pp)" for x in labels)])
    recent_section = ""
    if recent:
        recent_section = (
            "<section><h2>Recent-history follow-up</h2>"
            "<p>FULL reference rows are reused, not additional trials. The manifest records the window.</p>"
            + outcome_counts(result["recent"]["old_counts"], result["recent"]["new_counts"])
            + case_table(result["recent"]["comparison"], *recent)
            + "</section>"
        )
    reset_rows = []
    for case in result["comparison"]["cases"]:
        for arm, row in (case.get("same_state_reset_probe") or {}).items():
            if arm != "FULL":
                reset_rows.append(
                    [
                        escape(case["task"]),
                        str(case["episode_idx"]),
                        escape(arm),
                        f"{row['old_mm']:.3f}",
                        f"{row['new_mm']:.3f}",
                    ]
                )
    reset = table(reset_rows, ["Task", "Scene", "Removed history", *(x + " change (mm)" for x in labels)])
    file_links = []
    for label, run in zip(labels, (old, new)):
        gallery = run["root"] / "index.html"
        if gallery.is_file():
            file_links.append(link(gallery, label + " gallery"))
    file_links += [
        link(destination.with_suffix(".json"), "Audit JSON"),
        link(destination.with_suffix(".md"), "Text summary"),
    ]
    limitations = "".join("<li>" + escape(x) + "</li>" for x in LIMITATIONS)
    failures = "".join("<li>" + escape(x["check"]) + "</li>" for x in result["failed_checks"])
    provenance = escape(
        json.dumps(
            {
                "reference": result["old_checkpoint"],
                "candidate": result["new_checkpoint"],
                "protocol": result["protocol"],
                "sources": result["source_compatibility"],
                "recent_sources": (result.get("recent") or {}).get("source_compatibility"),
                "input_integrity": result["input_integrity"],
                "recovery": result["recovery"],
            },
            indent=2,
        )
    )
    status = "complete" if result["complete"] else "incomplete — pending cases are not failures"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Teacher history: {old_step:,} → {new_step:,}</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#172a3a;background:#f4f7fa;margin:0}}main{{max-width:1180px;margin:auto;padding:28px}}h1{{font-size:2rem;line-height:1.2}}section,details,.intro{{background:white;border:1px solid #d6e0e8;border-radius:12px;padding:20px;margin:18px 0}}summary{{cursor:pointer;font-weight:650}}a{{color:#175d9d}}.tablewrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.92rem}}th,td{{padding:9px 12px;border-bottom:1px solid #e0e7ed;text-align:left;white-space:nowrap}}th{{background:#edf3f8}}.videos{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}video{{width:100%;background:#101820;min-height:180px}}button{{border:1px solid #aabecb;background:#edf4fb;border-radius:5px;padding:8px;cursor:pointer}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem}}@media(max-width:720px){{main{{padding:12px}}.videos{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>Teacher history: {old_step:,} → {new_step:,} updates</h1>
<div class="intro"><p>{escape(protocol_text(result["protocol"]))}</p><p>Status: <strong>{status}</strong>.
<strong>{result["passed_checks"]}/{result["total_checks"]} checks passed.</strong></p><p>{" · ".join(file_links)}</p>
<p>{escape(result["source_compatibility"]["interpretation"])}</p></div>
<section><h2>Success by task and history read</h2>{outcome_counts(result["old_counts"], result["new_counts"])}</section>
<section><h2>History benefit</h2><p>FULL success minus control success, in percentage points. Only complete valid pairs provide a benefit estimate.</p>{benefits}</section>
{recent_section}
<section><h2>Matched scene outcomes</h2><p>Statuses link to complete videos. Later trajectories may diverge.</p>{case_table(result["comparison"], old, new)}</section>
<section><h2>FULL and changed-success cases</h2><p>Use “Play policy execution” to skip the expert demonstration at the beginning.</p>{"".join(videos)}</section>
<section><h2>Same-state sensitivity at reset</h2><p>Mean Euclidean TCP-target displacement over the executed prefix when history is removed. Targets are predicted FK positions, not realized robot movement. Reset scenes and flow noise are matched across checkpoints; larger displacement is not evidence of correct cue use.</p>{reset}</section>
<section><h2>Limits and controls</h2><ul>{limitations}</ul>{"<h3>Failed checks</h3><ul>" + failures + "</ul>" if failures else ""}</section>
<details><summary>Checkpoint, execution and recovery provenance</summary><pre>{provenance}</pre></details>
</main></body></html>"""
