"""Rebuild utility tables/plots from immutable rows, without model inference."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from stmem.inspection.refinement_statistics import context_means, episode_bootstrap
from stmem.training.checkpoints import atomic_json, file_sha256


def _format(value):
    if value is None:
        return "not measured"
    if isinstance(value, dict):
        if value.get("mean") is None:
            return "not measured"
        interval = value.get("interval")
        return f"{value['mean']:.6g}" + (f" [{interval[0]:.6g}, {interval[1]:.6g}]" if interval else " (interval unavailable)")
    return str(value)


def assess_hypotheses(manifest, results, output):
    """Narrow offline findings; never infer behavior from a velocity-loss proxy."""
    expected = {row["state_id"] for row in manifest["states"] if row.get("cache_eligible", True)}
    exclusions = [row for row in manifest["states"] if not row.get("cache_eligible", True)]
    accounted_exclusions = all(row.get("cache_ineligible_reason") for row in exclusions)
    training_ids = {(row.get("initialization"), row.get("declared_endpoint"), row.get("step"))
                    for row in results["inventory"] if row["student"] != "S0" and row.get("primary_endpoint")}
    matched_training = len(training_ids) == 1 and all(value is not None for value in next(iter(training_ids), ()))
    complete = all(any(row["student"] == student and row.get("primary_endpoint") and row["mode"] == "main"
                       for row in results["inventory"]) and expected <= {
                           row["state"]["state_id"] for row in results["analyses"] if row["state"]["student"] == student}
                   for student in ("S0", "S_STD", "S_STAGE", "S_PACKET")) and bool(expected) and accounted_exclusions and matched_training
    hypotheses = {f"H{i}": {"status": "inconclusive", "reason":
                   "Completed frozen panel; this hypothesis needs additional qualified evidence" if complete else
                   "Primary matched endpoint panel incomplete; current results are diagnostic or partial"}
                  for i in range(1, 9)}
    hypotheses["H4"]["reason"] = "Optional equal-budget interaction panel not activated"
    hypotheses["H5"]["reason"] = "Offline resolution losses cannot establish correct semantic choices; behavioral evidence required"
    hypotheses["H8"]["reason"] = "Conditional learned-router/crossed adaptation stage not activated"
    population = results["population"]
    if complete:
        hypotheses["H1"]["reason"] = "A2 candidate/decision prevalence and threshold sensitivity reported; no universal practical-utility threshold is assumed"
        hypotheses["H6"]["reason"] = "Objective conflicts reported with episode clustering; reproducibility of semantic conflict requires separate inspection"
        for name, candidates in (("H2", {k: v for k, v in population.get("backgrounds", {}).items()
                                          if "/validation/" in k}),
                                 ("H3", {k: v["packet_vs_immediate"] for k, v in population["selection"].items()
                                          if "/validation/" in k and "/packet/" in k})):
            positive = [key for key, value in candidates.items() if value.get("interval") and value["interval"][0] > 0]
            negative = [key for key, value in candidates.items() if value.get("interval") and value["interval"][1] <= 0]
            if positive:
                hypotheses[name] = {"status": "supported", "scope": positive,
                    "reason": "Positive episode-clustered confirmation interval in these predeclared groups; supervised surrogate evidence only, unadjusted for multiple comparisons"}
            elif candidates and len(negative) == len(candidates):
                hypotheses[name] = {"status": "not supported", "scope": negative,
                    "reason": "All available qualified confirmation intervals exclude a positive effect in the tested groups"}
        positive = [key for key, values in population.get("student_pairs", {}).items() if "/validation/" in key
                    and all(values[field].get("interval") and values[field]["interval"][0] > 0 for field in
                            ("absolute_packet_improvement", "allocation_benefit_improvement"))]
        if positive:
            hypotheses["H7"] = {"status": "supported", "scope": positive,
                "reason": "Matched states improve absolute selected loss and within-student allocation benefit; one paired training seed and an offline surrogate only, unadjusted for multiple comparisons"}
    for value in hypotheses.values():
        value["eligible_states"] = len(expected)
        value["predeclared_excluded_states"] = len(exclusions)
    assessment_path = Path(output) / "hypothesis_assessments.json"
    if assessment_path.exists():
        assessments = json.loads(assessment_path.read_text())
        if assessments.get("experiment_id") != manifest["experiment_id"]:
            raise ValueError("Manual hypothesis assessment belongs to another experiment")
        for key, value in assessments.get("hypotheses", {}).items():
            if key not in hypotheses or value.get("status") not in ("supported", "not supported", "inconclusive") \
                    or not value.get("reason") or not value.get("scope") or not value.get("evidence"):
                raise ValueError("Manual assessment requires valid hypothesis/status, reason, scope and evidence")
            hypotheses[key] = {**value, "assessment_source": "explicit evidence-linked manual assessment"}
    return hypotheses


def student_training_artifacts(analyses):
    """Read only hash-identified local runs, never infer their training curves."""
    records = {}
    for analysis in analyses:
        student = analysis["state"]["student"]
        if student == "S0" or student in records:
            continue
        identity = analysis.get("identity", {})
        path = Path(identity.get("checkpoint_path", "__not_available__"))
        if not path.is_file():
            records[student] = {"status": "training_artifacts_unavailable"}
            continue
        if file_sha256(path) != identity.get("checkpoint_sha256"):
            raise ValueError("Training-artifact checkpoint changed since measurement")
        endpoint = int(identity["step"])
        curves = {}
        metrics_path = path.parent / "metrics.jsonl"
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines():
                row = json.loads(line)
                if 0 < row.get("step", 0) <= endpoint:
                    curves[row["step"]] = {key: row[key] for key in
                        ("step", "training_action_loss", "step_seconds", "allocated_gpu_hours", "seconds") if key in row}
        latest = {}
        for strata_path in sorted(path.parent.glob("study_strata_rank*.jsonl")):
            with strata_path.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    if 0 < row.get("step", 0) <= endpoint:
                        key = (strata_path.name, row["step"], row["dimension"], row["value"])
                        latest[key] = (row["count"], row["mean_expert_fm_loss"])
        strata = defaultdict(lambda: [0, 0.0])
        for (_, _, dimension, value), (count, loss) in latest.items():
            strata[(dimension, value)][0] += count
            strata[(dimension, value)][1] += count * loss
        records[student] = {"status": "available" if curves else "metrics_unavailable",
            "checkpoint": str(path), "checkpoint_sha256": identity["checkpoint_sha256"], "endpoint": endpoint,
            "learning_curve": [curves[key] for key in sorted(curves)],
            "realized_strata": [{"dimension": dim, "value": value, "examples": count,
                "mean_training_expert_loss": loss / count} for (dim, value), (count, loss) in sorted(strata.items()) if count],
            "note": "Repeated logged steps use their latest record; losses describe training examples, not held-out quality"}
    return records


def runtime_projection(manifest, analyses, planned):
    """A declared smoke's observed cost scenarios, never a training forecast.

    Components scale by their actual work unit. The range captures observed
    smoke heterogeneity, not a confidence interval or an HPC speed guarantee.
    """
    result = {"status": "not_estimated", "planned": planned,
              "reason": "Requires the declared 8–16-state multi-task real-data smoke",
              "scope": "four frozen students; shared teacher references; excludes optimization"}
    smoke = {row["state"]["state_id"]: row for row in analyses
             if row.get("identity", {}).get("mode") == "smoke"
             and not row["identity"].get("limited_check", False)}
    rows = list(smoke.values())
    if (manifest.get("kind") != "matched_student_refinement_utility" or not planned
            or not 8 <= len(rows) <= 16 or len({r["state"]["task"] for r in rows}) < 3
            or len({(r["state"]["task"], r["state"]["episode_id"]) for r in rows}) < 3):
        return result
    samples = {"cache_assembly_seconds_per_state": [], "compact_seconds_per_example": [],
               "teacher_seconds_per_example": []}
    for row in rows:
        timing = row["timings"]
        samples["cache_assembly_seconds_per_state"].append(timing["cache_context_assembly_seconds"])
        compact = timing.get("compact_model_examples", 0)
        teacher = timing.get("teacher_model_examples", 0)
        if compact:
            samples["compact_seconds_per_example"].append(
                (timing.get("compact_context_seconds", 0) + timing.get("compact_forward_seconds", 0)) / compact)
        if teacher:
            samples["teacher_seconds_per_example"].append(
                (timing.get("teacher_context_seconds", 0) + timing.get("teacher_forward_seconds", 0)) / teacher)
    if any(not value for value in samples.values()):
        result["reason"] = "Smoke lacks fresh teacher/compact timings; reused references cannot predict their original cost"
        return result
    state_count = len(planned.get("context_counts", {}))
    work = {"cache_assembly_seconds_per_state": state_count * 4,
            "compact_seconds_per_example": planned["compact_model_examples_four_students"],
            "teacher_seconds_per_example": planned["teacher_model_examples_shared"]}
    components = {key: (np.quantile(value, [0, .5, 1]) * work[key]).tolist() for key, value in samples.items()}
    result.update(status="smoke_cost_scenarios", reason=None, states=len(rows),
                  component_seconds_min_median_max=components,
                  total_serial_seconds_min_median_max=np.sum(list(components.values()), axis=0).tolist(),
                  note="Observed min/median/max unit costs, not uncertainty intervals. Histories and storage may differ; "
                       "parallel allocations may contend. Excludes startup, candidate construction, numerical controls, "
                       "extra confirmation, generated actions, simulation and reports. No training duration is predicted.")
    return result


def write_report(manifest, results, output):
    started = time.perf_counter()
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from stmem.inspection.refinement_data import read_json

    output = Path(output)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    analyses = results["analyses"]
    gains = [r for a in analyses for r in a["gains"]]
    selections = [r for a in analyses for r in a["selection"]]
    pairs = [r for a in analyses for r in a["backgrounds"]]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for family, kind, label in (("immediate_audit", "first", "A2 unrestricted"),
                                ("packet", "first", "A1 packet first move"), ("packet", "complete", "packet")):
        values = [r["confirmation_gain"] for r in gains if r["family"] == family and r["kind"] == kind]
        if values:
            axes[0, 0].hist(values, bins=30, alpha=.45, label=label)
    axes[0, 0].set(title="Signed confirmation gain (descriptive)", xlabel="Raw expert FM loss reduction")
    if gains:
        axes[0, 0].legend(fontsize=8)
        axes[0, 1].scatter([r["discovery_gain"] for r in gains], [r["confirmation_gain"] for r in gains], s=7, alpha=.4)
        axes[1, 2].scatter([r["teacher_confirmation_gain"] for r in gains],
                           [r["confirmation_gain"] for r in gains], s=7, alpha=.4)
    axes[0, 1].set(title="Discovery / confirmation", xlabel="Discovery gain", ylabel="Confirmation gain")
    grouped = defaultdict(list)
    for r in selections:
        if r["family"] == "packet":
            grouped[(r["student"], r["reader"], r["partition"], str(r["budget"]))].append(r)
    for i, (key, rows) in enumerate(sorted(grouped.items())):
        estimate = episode_bootstrap(rows, "packet_vs_immediate")
        mean, interval = estimate["mean"], estimate["interval"]
        axes[0, 2].errorbar(i, mean, yerr=None if interval is None else
                          np.array([[max(0, mean - interval[0])], [max(0, interval[1] - mean)]]), fmt="o")
    axes[0, 2].set_xticks(range(len(grouped)), ["/".join(k) for k in sorted(grouped)], rotation=60, ha="right", fontsize=7)
    axes[0, 2].set(title="Equal-budget packet vs immediate", ylabel="Confirmation loss improvement")
    if pairs:
        axes[1, 0].hist([r["coarse_minus_rich"] for r in pairs], bins=25)
    axes[1, 0].set(title="Same operation, coarse minus rich gain", xlabel="Raw expert FM loss difference")
    for a in analyses[:8]:
        panel = read_json(output / "contexts" / f"{a['state']['state_id']}.json")
        means = context_means(a["raw_rows"])
        for family in panel["families"]:
            if family["panel"] != "packet" or not family["candidates"]:
                continue
            # Gain-blind first candidate for the figure; never a confirmation winner.
            c = family["candidates"][0]
            xs, ys = [0], [means[(family["base"], "confirmation")]["expert_full"]]
            for increment, key in sorted(c["prefixes"].items(), key=lambda p: int(p[0])):
                xs.append(int(increment))
                ys.append(means[(key, "confirmation")]["expert_full"])
            axes[1, 1].plot(xs, ys, alpha=.5)
            break
    axes[1, 1].set(title="Fixed candidate prefix curves (first 8 states)", xlabel="Added tokens", ylabel="Expert FM loss")
    axes[1, 2].set(title="Expert / teacher disagreement", xlabel="Teacher agreement gain", ylabel="Expert FM gain")
    for ax in axes.flat:
        ax.axhline(0, color="black", linewidth=.5)
    fig.savefig(plots / "utility.png", dpi=160)
    fig.savefig(plots / "utility.pdf")
    plt.close(fig)
    hypotheses = assess_hypotheses(manifest, results, output)
    atomic_json(hypotheses, output / "hypotheses.json")
    training = student_training_artifacts(analyses)
    atomic_json(training, output / "student_training.json")
    lines = ["# Student training × refinement utility pilot", "", f"Experiment: `{manifest['experiment_id']}`.", "",
             f"Scope: **{manifest['kind']}**. Source: `{manifest['source']['git']}` / `{manifest['source']['source_sha256']}`.", "",
             "These are frozen expert-FM utility measurements, not benchmark success. Diagnostic loss-selected packets are not deployed routers.", "",
             "## Required student matrix", "", "| Student | Measurement stage | Measured states | Status |", "|---|---|---:|---|"]
    lines.extend(f"| {r['student']} | {r['mode']} | {r['measured_states']} | {r['status']} |" for r in results["inventory"])
    if results.get("confirmation_extensions"):
        lines += ["", "Independent confirmation extensions are separate raw evidence. They do not enter primary plots, selections or intervals, and original discovery choices remain fixed.", "",
                  "| Student | Step | Partition | Draws | States | Status | Directory |", "|---|---:|---|---:|---:|---|---|"]
        for row in results["confirmation_extensions"]:
            lines.append(f"| {row['student']} | {row['step']} | {row['partition']} | {row['draws']} | "
                         f"{row['measured_states']} | {row['status']} | `{row['directory']}` |")
    training_steps = manifest.get("protocol", {}).get("student_training", {}).get("steps", 30000)
    lines += ["", f"This study declares {training_steps:,} updates for each teacher-initialized student. Earlier numbered snapshots are development-only on the fitting partition; final confirmation uses each checkpoint's recorded primary endpoint. Existing studies without the extended schedule retain their original 30k endpoint. An explicitly configured S0 continuation uses its separately declared common endpoint. Initializations and training durations are never pooled in matched contrasts. S0 has a different prior training history.", "",
              "## Absolute loss and selection improvement", "",
              "Intervals resample episodes within tasks after averaging repeated draws/states. They do not measure training-seed robustness.", "",
              "| Student / partition / reader / panel / budget | Background | Immediate-selected | Packet-selected | Teacher-selected | Random | UNIFORM | Dispersed | Discovery-low |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in sorted(results["population"]["selection"].items()):
        lines.append(f"| {name} | " + " | ".join(_format(row.get(field)) for field in
                     ("base_loss", "immediate_loss", "packet_loss", "teacher_selected_loss", "random_loss",
                      "uniform_loss", "dispersed_loss", "low_selected_loss")) + " |")
    lines += ["", "| Same grouping | Packet vs immediate | Packet vs UNIFORM | Packet vs dispersed |", "|---|---:|---:|---:|"]
    for name, row in sorted(results["population"]["selection"].items()):
        lines.append(f"| {name} | {_format(row['packet_vs_immediate'])} | {_format(row['packet_vs_uniform'])} | {_format(row['packet_vs_dispersed'])} |")
    lines += ["", "Positive improvements mean lower confirmation loss at exactly the same final token count. Ties receive uniform average credit; all-negative pools remain included.", "",
              "## First-ten-target secondary control", "",
              "These losses use the same H16-discovery-selected tie sets. Prefix loss never reselects candidates.", "",
              "| Group | Background | Immediate-selected | Packet-selected | Teacher-selected | Random | UNIFORM | Dispersed | Packet vs immediate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in sorted(results["population"]["selection"].items()):
        fields = ("base_prefix_loss", "immediate_prefix_loss", "packet_prefix_loss", "teacher_selected_prefix_loss",
                  "random_prefix_loss", "uniform_prefix_loss", "dispersed_prefix_loss", "prefix_packet_vs_immediate")
        lines.append(f"| {name} | " + " | ".join(_format(row.get(field)) for field in fields) + " |")
    thresholds = results["population"].get("relative_thresholds", manifest.get("protocol", {}).get(
        "relative_thresholds", [.005, .01, .05]))
    threshold_labels = " | ".join(f"≤{100 * value:g}%" for value in thresholds)
    lines += ["", "## Prevalence, null effects and all-negative pools", "",
              "Candidate frequencies are averaged within states/episodes; decision-context frequencies are separate. Numerical sign repeatability is descriptive and is not statistical significance. Relative thresholds below are sensitivity choices, not established success thresholds.", "",
              f"Numerical gain deadband: `{results['population'].get('gain_epsilon', manifest.get('protocol', {}).get('gain_epsilon', 1e-6)):g}`.", "",
              f"| Student / partition / reader / family / kind | Operations | Stable positive | Stable harmful | Direction reversal | {threshold_labels} | Teacher/expert conflict |",
              "|---|" + "---:|" * (5 + len(thresholds))]
    for name, row in sorted(results["population"].get("prevalence", {}).items()):
        fields = ("stable_positive", "stable_harmful", "direction_reversal",
                  *(f"small_relative_{float(value)}" for value in thresholds), "teacher_expert_conflict")
        lines.append(f"| {name} | {row['candidate_measurements']} | " +
                     " | ".join(_format(row[field]) for field in fields) + " |")
    lines += ["", "| Decision grouping / background | Contexts | Any stable positive | All stable harmful | No confirmation-positive |",
              "|---|---:|---:|---:|---:|"]
    for name, row in sorted(results["population"].get("decision_prevalence", {}).items()):
        lines.append(f"| {name} | {row['decision_contexts']} | " + " | ".join(_format(row[field]) for field in
                     ("any_stable_positive", "all_stable_harmful", "no_confirmed_positive")) + " |")
    lines += ["", "| Selection grouping | All-negative discovery pool | All-negative confirmation pool | Packet tie fraction |",
              "|---|---:|---:|---:|"]
    for name, row in sorted(results["population"]["selection"].items()):
        lines.append(f"| {name} | " + " | ".join(_format(row.get(field)) for field in
                     ("all_negative_discovery", "all_negative_confirmation", "packet_tie_fraction")) + " |")
    lines += ["", "## Matched student comparisons", "",
              "Same states, reader, final budget, initialization, optimizer step and declared endpoint only. Positive values favor the second arm. S0 is excluded from this matched-training contrast.", "",
              "| Contrast / partition / reader / panel / budget / training identity | Absolute packet-loss improvement | Absolute random-allocation improvement | Allocation-benefit improvement |",
              "|---|---:|---:|---:|"]
    for name, row in sorted(results["population"].get("student_pairs", {}).items()):
        lines.append(f"| {name} | " + " | ".join(_format(row[field]) for field in
                     ("absolute_packet_improvement", "absolute_random_improvement", "allocation_benefit_improvement")) + " |")
    omissions = results["population"].get("student_pair_omissions", [])
    if omissions:
        lines += ["", f"Suppressed {len(omissions)} unmatched or unidentified decision-context comparisons. Reasons and both training identities are retained in `analysis.json` under `population.student_pair_omissions`."]
    lines += ["", "## Same-operation background effect", "", "| Student / partition / reader / kind | Coarse minus rich gain |", "|---|---:|"]
    for name, row in sorted(results["population"].get("backgrounds", {}).items()):
        lines.append(f"| {name} | {_format(row)} |")
    lines += ["", "## Gain strata", "",
              "One-way strata avoid an unsupported full cross-product. Every row retains its number of eligible operations, states and episode-clustered interval. A1 packets and A2 unrestricted moves remain separate.", "",
              "| Student / partition / reader / family / kind / stratum | Operations | States | Expert gain | First-ten gain | Relative gain | Paired draw SD |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in sorted(results["population"].get("gain_strata", {}).items()):
        lines.append(f"| {name} | {row['candidate_measurements']} | {row['states']} | " +
                     " | ".join(_format(row[field]) for field in
                                ("confirmation_gain", "prefix_confirmation_gain", "relative_gain", "confirmation_noise_sd")) + " |")
    lines += ["",
              "## Capability references", "", "| Student | State | Context | Confirmation expert loss |", "|---|---|---|---:|"]
    for a in analyses:
        means = context_means(a["raw_rows"])
        for name, key in a["capability"].items():
            value = means.get((key, "confirmation"), {}).get("expert_full") if isinstance(key, str) else None
            lines.append(f"| {a['state']['student']} | {a['state']['state_id']} | {name} | {_format(value)} |")
        teacher = {row["draw_index"]: row["teacher_expert_full"] for row in a["raw_rows"]
                   if row["bank"] == "confirmation" and "teacher_expert_full" in row}
        if teacher:
            lines.append(f"| {a['state']['student']} | {a['state']['state_id']} | Teacher FULL reference | "
                         f"{np.mean(list(teacher.values())):.6g} |")
    lines += ["", "All-leaf student inputs may be outside compact-training support and are not an upper bound. Teacher references are stored once per state/draw.", "",
              "## Hypotheses", "", "| Hypothesis | Assessment | Reason |", "|---|---|---|"]
    lines.extend(f"| {key} | {value['status']} | {value['reason']}"
                 + (f" Scope: {', '.join(value['scope'])}" if value.get("scope") else "") + " |"
                 for key, value in hypotheses.items())
    lines += ["", "## Student training artifacts", "",
              "Curves use only updates through the measured endpoint and are imported only from a hash-verified checkpoint directory. Realized strata include all available rank files and deduplicate repeated steps. Detailed curves and exposure counts are in `student_training.json`.", "",
              "| Student | Artifact status | Endpoint | Logged updates | Strata |", "|---|---|---:|---:|---:|"]
    for student in ("S_STD", "S_STAGE", "S_PACKET"):
        value = training.get(student, {})
        lines.append(f"| {student} | {value.get('status', 'not_measured')} | {value.get('endpoint', '—')} | "
                     f"{len(value.get('learning_curve', []))} | {len(value.get('realized_strata', []))} |")
    curves = [(name, value["learning_curve"]) for name, value in training.items() if value.get("learning_curve")]
    if curves:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for name, curve in curves:
            measured = [row for row in curve if "training_action_loss" in row]
            ax.plot([row["step"] for row in measured], [row["training_action_loss"] for row in measured],
                    label=name, linewidth=.7, alpha=.7)
        ax.set(xlabel="Optimizer update", ylabel="Training expert FM loss", title="Fixed-endpoint learning curves")
        ax.legend()
        fig.savefig(plots / "student_learning.png", dpi=160)
        plt.close(fig)
        lines += ["", "![Fixed-endpoint student learning curves](plots/student_learning.png)"]
    lines += ["", "## Eligibility and boundaries", "",
              f"Preselected states: {len(manifest['states'])}; task counts and exclusions are retained in `manifest.json`.", "",
              "Every selected diagnostic episode and the complete router-validation partition are excluded from new student/router fitting. Development and final partitions are reported separately. Existing teacher/S0 exposure is not claimed unseen.", "",
              "The terminal/unavailable-input convention is unchanged. Full-H16 expert loss selects candidates; first-ten loss is secondary. Structural gripper masks are recorded independently of terminal padding.", ""]
    eligibility_path = output / "contexts_complete.json"
    if eligibility_path.exists():
        context_plan = read_json(eligibility_path)
        if context_plan.get("experiment_id") != manifest["experiment_id"]:
            raise ValueError("Eligibility denominators belong to another experiment")
        eligibility = defaultdict(lambda: {"total": 0, "eligible": 0, "ranking": 0, "reasons": defaultdict(int)})
        for row in context_plan.get("eligibility", []):
            group = eligibility[(row["reader"], row["panel"], row.get("final_budget", "audit128"))]
            group["total"] += 1
            group["eligible"] += int(row.get("eligible", False))
            group["ranking"] += int(row.get("ranking_eligible", False))
            if row.get("reason"):
                group["reasons"][row["reason"]] += 1
        lines += ["| Reader / family / budget | Preselected states | Eligible | Ranking-eligible | Exclusion reasons |",
                  "|---|---:|---:|---:|---|"]
        for key, row in sorted(eligibility.items(), key=lambda value: str(value[0])):
            lines.append(f"| {'/'.join(map(str, key))} | {row['total']} | {row['eligible']} | {row['ranking']} | "
                         + "; ".join(f"{reason}: {count}" for reason, count in sorted(row["reasons"].items())) + " |")
    exclusions = defaultdict(int)
    for row in manifest.get("sampling_exclusions", []):
        exclusions[row.get("reason", "unspecified")] += 1
    if exclusions:
        lines += ["", "State-sampling exclusions: " + "; ".join(f"{k}: {v}" for k, v in sorted(exclusions.items())) + "."]
    lines += ["", "## Compute", "",
              "| Student | States | Compact examples | Measurement wall seconds | Cache/assembly seconds | Compact adaptation seconds | Compact forward seconds | Teacher forward seconds | Peak GPU GiB |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for student in ("S0", "S_STD", "S_STAGE", "S_PACKET"):
        chosen = [a for a in analyses if a["state"]["student"] == student]
        timings = [a["timings"] for a in chosen]
        seconds = sum(t["wall_seconds"] for t in timings)
        examples = sum(t["compact_model_examples"] for t in timings)
        lines.append(f"| {student} | {len(chosen)} | {examples} | {seconds:.3f} | "
                     f"{sum(t['cache_context_assembly_seconds'] for t in timings):.3f} | "
                     f"{sum(t.get('compact_context_seconds', 0) for t in timings):.3f} | "
                     f"{sum(t.get('compact_forward_seconds', 0) for t in timings):.3f} | "
                     f"{sum(t.get('teacher_forward_seconds', 0) for t in timings):.3f} | "
                     f"{max([t['gpu_peak_allocated_bytes'] for t in timings] or [0])/2**30:.3f} |")
    planned_path = output / "contexts_complete.json"
    planning = read_json(planned_path) if planned_path.exists() else {}
    # Main measurements supersede smoke rows in scientific tables, but the
    # original smoke timing remains useful for checking the compute forecast.
    smoke_metadata = []
    for rows_path in sorted((output / "measurements").glob("*/step_*/smoke/*/*.npz")):
        metadata_path = rows_path.with_suffix(".json")
        if not metadata_path.exists():
            continue
        metadata = read_json(metadata_path)
        if metadata["identity"]["experiment_id"] != manifest["experiment_id"] or metadata["rows_sha256"] != file_sha256(rows_path):
            raise ValueError("Smoke timing artifact belongs to a changed measurement")
        smoke_metadata.append(metadata)
    projection = runtime_projection(manifest, smoke_metadata or analyses, planning)
    atomic_json(projection, output / "runtime_projection.json")
    if projection["status"] == "smoke_cost_scenarios":
        hours = [value / 3600 for value in projection["total_serial_seconds_min_median_max"]]
        lines += ["", f"Frozen four-student panel cost scenarios from the declared smoke: "
                  f"**{hours[0]:.2f} / {hours[1]:.2f} / {hours[2]:.2f} serial hours** "
                  "using observed minimum/median/maximum unit costs.", "", projection["note"]]
    else:
        lines += ["", "Runtime projection: **not estimated**. " + projection["reason"] + "."]
    mask_groups = defaultdict(lambda: [0, 0, 0, 0])
    for analysis in analyses:
        mask = analysis.get("mask")
        if mask is None:
            continue
        value = mask_groups[(analysis["state"]["student"], analysis["state"]["task"])]
        for i, count in enumerate((1, mask["valid_coordinates"], mask["structurally_unavailable_coordinates"],
                                   mask["terminal_or_unavailable_coordinates"])):
            value[i] += count
    if mask_groups:
        lines += ["", "## Action-mask accounting", "",
                  "Counts are target coordinates, counted once per state, not once per candidate/draw. "
                  "Structural gripper absence is separate from terminal or missing targets on otherwise available joints.", "",
                  "| Student / task | States | Valid coordinates | Structurally absent | Terminal/missing available coordinates |",
                  "|---|---:|---:|---:|---:|"]
        lines.extend(f"| {'/'.join(key)} | " + " | ".join(map(str, value)) + " |"
                     for key, value in sorted(mask_groups.items()))
    lines += ["", "A one-task cache check cannot predict HPC all-task throughput: history lengths, storage and GPU differ. Training compute is unmeasured until the jobs run.", "",
              "## Artifacts", "", "![Reproducible utility plots](plots/utility.png)", "",
              "`analysis.json` preserves raw per-draw losses, selections, eligibility groupings and episode-aware statistics. `contexts/` stores fixed supports, moves, costs and exact-coverage identities. `measurements/` contains compressed draw arrays and checksums.", "",
              "Regenerate without inference: `python -m stmem.inspection.refinement_utility --mode report --manifest PATH/manifest.json`.", ""]
    (output / "report.md").write_text("\n".join(lines))
    atomic_json({"experiment_id": manifest["experiment_id"], "report_seconds": time.perf_counter() - started,
                 "measured_states": len(analyses), "model_inference": False}, output / "report_receipt.json")
