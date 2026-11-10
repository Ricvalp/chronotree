"""Discovery-only choices and episode-clustered utility summaries.

No maximum over confirmation results is reported as achieved performance.
Candidates and repeated noise draws are not independent episodes.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from stmem.config import digest


def finite_mean(values):
    values = list(values)
    return float(np.mean(values)) if values else None


def tie_set(values, *, epsilon=1e-6, minimize=True):
    if not values:
        return []
    if epsilon < 0 or not all(math.isfinite(float(v)) for v in values.values()):
        raise ValueError("Selections require finite losses and nonnegative numerical deadband")
    best = (min if minimize else max)(values.values())
    return sorted(k for k, v in values.items() if abs(v - best) <= epsilon)


def select_and_confirm(discovery, confirmation, *, epsilon=1e-6):
    """Equal probability over the discovery tie set; confirmation never selects."""
    chosen = tie_set(discovery, epsilon=epsilon)
    if set(discovery) != set(confirmation):
        raise ValueError("Confirmation must cover exactly the fixed candidate pool")
    return {"selected_ids": chosen, "tie_count": len(chosen),
            "discovery_loss": finite_mean(discovery[i] for i in chosen),
            "confirmation_loss": finite_mean(confirmation[i] for i in chosen),
            "random_confirmation_loss": finite_mean(confirmation.values()),
            "tie_policy": "uniform_average_over_discovery_near_ties"}


def episode_bootstrap(rows, field, *, draws=1000, seed=2701):
    """Average states within episodes, episodes within tasks; resample episodes."""
    states = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None and math.isfinite(float(value)):
            states[(row["task"], row["episode_id"], row.get("state_id", "unspecified"))].append(float(value))
    episodes = defaultdict(list)
    for (task, episode, _), values in states.items():
        episodes[(task, episode)].append(float(np.mean(values)))
    tasks = defaultdict(list)
    for (task, _), values in sorted(episodes.items()):
        tasks[task].append(float(np.mean(values)))
    if not tasks:
        return {"mean": None, "interval": None, "episodes": 0, "tasks": 0}
    mean = float(np.mean([np.mean(values) for values in tasks.values()]))
    # A singleton episode cannot supply population uncertainty for that task.
    if min(map(len, tasks.values())) < 2:
        interval = None
    else:
        rng = np.random.default_rng(seed)
        samples = np.stack([rng.choice(values, (draws, len(values)), replace=True).mean(1)
                            for values in tasks.values()]).mean(0)
        interval = np.quantile(samples, [.025, .975]).tolist()
    return {"mean": mean, "interval": interval, "episodes": len(episodes), "tasks": len(tasks),
            "unit": "episode clusters stratified by task; one paired training seed"}


def context_means(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["context_id"], row["bank"])].append(row)
    return {key: {field: finite_mean(row[field] for row in values) for field in
                  ("expert_full", "expert_prefix", "teacher_full", "teacher_prefix")}
            for key, values in groups.items()}


def analyze_panel(state, panel, rows, *, epsilon=1e-6):
    """Panel schema is normalized by the runner; all IDs refer to fixed contexts."""
    means = context_means(rows)
    groups, gains, backgrounds = [], [], []
    for group in panel["groups"]:
        base = group["base"]
        if (base, "confirmation") not in means or (base, "discovery") not in means:
            continue
        candidates = [c for c in group["candidates"] if all((c[k], bank) in means
                      for k in ("first", "complete") for bank in ("discovery", "confirmation"))]
        for candidate in candidates:
            for kind, cost in (("first", 1), ("complete", group["cost"])):
                key = candidate[kind]
                values = {bank: means[(base, bank)]["expert_full"] - means[(key, bank)]["expert_full"]
                          for bank in ("discovery", "confirmation")}
                baseline = means[(base, "confirmation")]["expert_full"]
                paired_draws = []
                for base_row in rows:
                    if base_row["context_id"] == base and base_row["bank"] == "confirmation":
                        other = next((r for r in rows if r["context_id"] == key and r["bank"] == "confirmation"
                                      and r["draw_index"] == base_row["draw_index"]), None)
                        if other:
                            paired_draws.append(base_row["expert_full"] - other["expert_full"])
                gains.append({**state, "reader": group["reader"], "family": group["family"],
                              "background": group["background"], "candidate_id": candidate["id"],
                              "paired_background": group.get("paired", False),
                              **{k: candidate.get(k) for k in ("source", "depth", "support_count")},
                              "kind": kind, "delta_tokens": cost,
                              "discovery_gain": values["discovery"], "confirmation_gain": values["confirmation"],
                              "prefix_discovery_gain": means[(base, "discovery")]["expert_prefix"]
                              - means[(key, "discovery")]["expert_prefix"],
                              "prefix_confirmation_gain": means[(base, "confirmation")]["expert_prefix"]
                              - means[(key, "confirmation")]["expert_prefix"],
                              "baseline": baseline,
                              "relative_gain": values["confirmation"] / baseline if baseline > epsilon else None,
                              "relative_ill_conditioned": baseline <= epsilon,
                              "confirmation_noise_sd": float(np.std(paired_draws, ddof=1)) if len(paired_draws) > 1 else None,
                              "teacher_confirmation_gain": means[(base, "confirmation")]["teacher_full"]
                              - means[(key, "confirmation")]["teacher_full"],
                              "gain_per_token": values["confirmation"] / cost})
        if not candidates:
            continue
        confirmation = {c["id"]: means[(c["complete"], "confirmation")]["expert_full"] for c in candidates}
        selected = {}
        for name, context_key, metric in (("immediate", "first", "expert_full"),
                                          ("packet", "complete", "expert_full"),
                                          ("teacher", "complete", "teacher_full")):
            discovery = {c["id"]: means[(c[context_key], "discovery")][metric] for c in candidates}
            selected[name] = select_and_confirm(discovery, confirmation, epsilon=epsilon)
        low_discovery = {c["id"]: -means[(c["complete"], "discovery")]["expert_full"] for c in candidates}
        selected["low"] = select_and_confirm(low_discovery, confirmation, epsilon=epsilon)
        selected["low"]["selection_rule"] = "largest_discovery_expert_full_loss"
        selected["low"]["discovery_loss"] *= -1
        control = {name: means[(key, "confirmation")]["expert_full"] for name, key in group["controls"].items()
                   if (key, "confirmation") in means}
        prefix_losses = {c["id"]: means[(c["complete"], "confirmation")]["expert_prefix"] for c in candidates}
        prefix_selected = {name: finite_mean(prefix_losses[key] for key in choice["selected_ids"])
                           for name, choice in selected.items()}
        prefix_control = {name: means[(key, "confirmation")]["expert_prefix"] for name, key in group["controls"].items()
                          if (key, "confirmation") in means}
        row = {**state, "reader": group["reader"], "family": group["family"],
               "background": group["background"], "budget": group["background"] + group["cost"],
               "candidate_count": len(candidates), "selection": selected, "controls": control,
               "base_loss": means[(base, "confirmation")]["expert_full"],
               "packet_loss": selected["packet"]["confirmation_loss"],
               "immediate_loss": selected["immediate"]["confirmation_loss"],
               "teacher_selected_loss": selected["teacher"]["confirmation_loss"],
               "random_loss": selected["packet"]["random_confirmation_loss"],
               "low_selected_loss": selected["low"]["confirmation_loss"],
               "uniform_loss": control.get("uniform"), "dispersed_loss": control.get("dispersed"),
               "packet_vs_immediate": selected["immediate"]["confirmation_loss"]
               - selected["packet"]["confirmation_loss"],
               "packet_vs_random": selected["packet"]["random_confirmation_loss"]
               - selected["packet"]["confirmation_loss"],
               "packet_vs_uniform": control.get("uniform", float("nan"))
               - selected["packet"]["confirmation_loss"],
               "packet_vs_dispersed": control.get("dispersed", float("nan"))
               - selected["packet"]["confirmation_loss"],
               "selection_objective": "discovery_expert_full_H16",
               "base_prefix_loss": means[(base, "confirmation")]["expert_prefix"],
               "packet_prefix_loss": prefix_selected["packet"],
               "immediate_prefix_loss": prefix_selected["immediate"],
               "teacher_selected_prefix_loss": prefix_selected["teacher"],
               "random_prefix_loss": finite_mean(prefix_losses.values()),
               "uniform_prefix_loss": prefix_control.get("uniform"),
               "dispersed_prefix_loss": prefix_control.get("dispersed"),
               "low_selected_prefix_loss": prefix_selected["low"],
               "prefix_controls": prefix_control,
               "prefix_packet_vs_immediate": prefix_selected["immediate"] - prefix_selected["packet"],
               "all_negative_discovery": float(all(means[(base, "discovery")]["expert_full"]
                   - means[(c["complete"], "discovery")]["expert_full"] < -epsilon for c in candidates)),
               "all_negative_confirmation": float(all(means[(base, "confirmation")]["expert_full"]
                   - means[(c["complete"], "confirmation")]["expert_full"] < -epsilon for c in candidates)),
               "no_positive_confirmation": float(all(means[(base, "confirmation")]["expert_full"]
                   - means[(c["complete"], "confirmation")]["expert_full"] <= epsilon for c in candidates)),
               "packet_tie_fraction": float(selected["packet"]["tie_count"] > 1),
               "delta_tokens": 0, "diagnostic_selection_not_deployed_router": True}
        confirmation_best = tie_set(confirmation, epsilon=epsilon)
        row["winner_reversal"] = (float(not set(selected["packet"]["selected_ids"]) & set(confirmation_best))
                                  if len(confirmation_best) == len(selected["packet"]["selected_ids"]) == 1 else None)
        for field in ("packet_vs_uniform", "packet_vs_dispersed"):
            if not math.isfinite(row[field]):
                row[field] = None
        groups.append(row)
    paired = defaultdict(dict)
    for row in gains:
        if row["family"] == "packet" and row["paired_background"]:
            paired[(row["reader"], row["candidate_id"], row["kind"])][row["background"]] = row
    for (reader, candidate, kind), values in paired.items():
        if 56 in values and 120 in values:
            backgrounds.append({**state, "reader": reader, "candidate_id": candidate, "kind": kind,
                                "coarse_minus_rich": values[56]["confirmation_gain"] - values[120]["confirmation_gain"]})
    return {"selection": groups, "gains": gains, "backgrounds": backgrounds}


def population_summary(analyses, *, epsilon=1e-6, relative_thresholds=(.005, .01, .05)):
    if not math.isfinite(float(epsilon)) or epsilon < 0:
        raise ValueError("Population gain deadband must be finite and nonnegative")
    relative_thresholds = tuple(float(value) for value in relative_thresholds)
    if not relative_thresholds or len(set(relative_thresholds)) != len(relative_thresholds) or any(
            not math.isfinite(value) or value < 0 for value in relative_thresholds):
        raise ValueError("Relative thresholds must be distinct, finite and nonnegative")
    threshold_fields = tuple(f"small_relative_{value}" for value in relative_thresholds)
    results = {}
    grouped = defaultdict(list)
    for analysis in analyses:
        for row in analysis["selection"]:
            grouped[(row.get("student", "S0"), row["partition"], row["reader"], row["family"], row["budget"])].append(row)
    for key, rows in grouped.items():
        results["/".join(map(str, key))] = {field: episode_bootstrap(rows, field) for field in (
            "base_loss", "packet_loss", "immediate_loss", "teacher_selected_loss", "random_loss",
            "low_selected_loss", "uniform_loss", "dispersed_loss",
            "packet_vs_immediate", "packet_vs_random", "packet_vs_uniform", "packet_vs_dispersed", "winner_reversal",
            "base_prefix_loss", "packet_prefix_loss", "immediate_prefix_loss", "teacher_selected_prefix_loss",
            "random_prefix_loss", "uniform_prefix_loss", "dispersed_prefix_loss", "low_selected_prefix_loss",
            "prefix_packet_vs_immediate", "all_negative_discovery",
            "all_negative_confirmation", "no_positive_confirmation", "packet_tie_fraction")}
    prevalence = {}
    grouped_gains = defaultdict(list)
    for analysis in analyses:
        for row in analysis["gains"]:
            grouped_gains[(row.get("student", "S0"), row["partition"], row["reader"], row["family"], row["kind"])].append(row)
    for key, rows in grouped_gains.items():
        annotated = []
        for row in rows:
            values = {**row,
                      "stable_positive": float(row["discovery_gain"] > epsilon and row["confirmation_gain"] > epsilon),
                      "stable_harmful": float(row["discovery_gain"] < -epsilon and row["confirmation_gain"] < -epsilon),
                      "direction_reversal": float(row["discovery_gain"] * row["confirmation_gain"] < 0)
                      if min(abs(row["discovery_gain"]), abs(row["confirmation_gain"])) > epsilon else None,
                      "teacher_expert_conflict": float(row["teacher_confirmation_gain"] * row["confirmation_gain"] < 0)
                      if min(abs(row["teacher_confirmation_gain"]), abs(row["confirmation_gain"])) > epsilon else None}
            for threshold in relative_thresholds:
                values[f"small_relative_{threshold}"] = (float(abs(row["relative_gain"]) <= threshold)
                                                         if row["relative_gain"] is not None else None)
            annotated.append(values)
        prevalence["/".join(map(str, key))] = {"candidate_measurements": len(rows), **{
            field: episode_bootstrap(annotated, field) for field in ("stable_positive", "stable_harmful",
                "direction_reversal", "teacher_expert_conflict", *threshold_fields)}}
    pair_omissions = []
    student_pairs = paired_student_summary(analyses, omissions=pair_omissions)
    return {"selection": results, "prevalence": prevalence,
            "gain_strata": gain_strata(analyses), "backgrounds": paired_background_summary(analyses),
            "student_pairs": student_pairs, "student_pair_omissions": pair_omissions,
            "decision_prevalence": decision_prevalence(analyses, epsilon=epsilon),
            "gain_epsilon": epsilon, "relative_thresholds": list(relative_thresholds),
            "note": "Sign repeatability is descriptive, not significance. Deadband is numerical, not practical utility."}


def gain_strata(analyses):
    """One-way strata retain denominators without a mostly empty Cartesian grid."""
    groups = defaultdict(list)
    for analysis in analyses:
        for row in analysis["gains"]:
            depth, support = row.get("depth"), row.get("support_count")
            strata = {"task": row["task"], "phase": row.get("phase", "unknown"),
                      "background": row["background"], "source": row.get("source", "unknown"),
                      "depth": "unknown" if depth is None else ("0-3" if depth < 4 else "4-7" if depth < 8 else "8+"),
                      "descendants": "unknown" if support is None else
                      ("1-32" if support <= 32 else "33-256" if support <= 256 else "257+")}
            base = (row.get("student", "S0"), row["partition"], row["reader"], row["family"], row["kind"])
            for dimension, value in strata.items():
                groups[(*base, dimension, str(value))].append(row)
    return {"/".join(map(str, key)): {"candidate_measurements": len(rows),
            "states": len({r["state_id"] for r in rows}),
            **{field: episode_bootstrap(rows, field) for field in
               ("confirmation_gain", "prefix_confirmation_gain", "relative_gain", "confirmation_noise_sd")}}
            for key, rows in sorted(groups.items())}


def paired_background_summary(analyses):
    groups = defaultdict(list)
    for analysis in analyses:
        for row in analysis["backgrounds"]:
            groups[(row.get("student", "S0"), row["partition"], row["reader"], row["kind"])].append(row)
    return {"/".join(key): episode_bootstrap(rows, "coarse_minus_rich") for key, rows in sorted(groups.items())}


def decision_prevalence(analyses, *, epsilon=1e-6):
    """Decision frequency is separate from candidate-operation frequency."""
    contexts = defaultdict(list)
    for analysis in analyses:
        for row in analysis["gains"]:
            key = (row.get("student", "S0"), row["partition"], row["reader"], row["family"],
                   row["kind"], row["background"], row["state_id"])
            contexts[key].append(row)
    populations = defaultdict(list)
    for key, rows in contexts.items():
        values = {**rows[0], "any_stable_positive": float(any(r["discovery_gain"] > epsilon and
            r["confirmation_gain"] > epsilon for r in rows)), "all_stable_harmful": float(all(
            r["discovery_gain"] < -epsilon and r["confirmation_gain"] < -epsilon for r in rows)),
            "no_confirmed_positive": float(all(r["confirmation_gain"] <= epsilon for r in rows))}
        populations[key[:-1]].append(values)
    return {"/".join(map(str, key)): {"decision_contexts": len(rows),
            **{field: episode_bootstrap(rows, field) for field in
               ("any_stable_positive", "all_stable_harmful", "no_confirmed_positive")}}
            for key, rows in sorted(populations.items())}


def paired_student_summary(analyses, *, omissions=None):
    """Match state and training identity; suppress missing or mismatched identities.

    A 15k development model cannot be a matched comparator for a 30k endpoint,
    nor can teacher-initialized and S0-initialized arms be compared as one study.
    Valid continuation endpoints need not equal the primary schedule's 30k.
    """
    identity_fields = ("experiment_id", "initialization", "initialization_sha256",
                       "training_manifest_sha256", "step", "endpoint", "precision", "mode")
    contexts = defaultdict(lambda: defaultdict(list))
    omissions = [] if omissions is None else omissions
    for analysis in analyses:
        identity = analysis.get("identity", {})
        for row in analysis["selection"]:
            student = row.get("student", "S0")
            if student not in ("S_STD", "S_STAGE", "S_PACKET"):
                continue
            missing = [field for field in identity_fields if identity.get(field) is None]
            invalid_steps = any(not isinstance(identity.get(field), int)
                                or isinstance(identity.get(field), bool) or identity[field] < 1
                                for field in ("step", "endpoint"))
            if missing or invalid_steps or identity.get("student") != student:
                omissions.append({"reason": "missing_or_invalid_training_identity", "student": student,
                                  "state_id": row["state_id"], "missing_fields": missing})
                continue
            key = (row["partition"], row["reader"], row["family"], row["budget"], row["state_id"])
            contexts[key][student].append((row, {field: identity[field] for field in identity_fields}))
    groups = defaultdict(list)
    group_identities = {}
    for key, rows in contexts.items():
        for baseline, candidate in (("S_STD", "S_STAGE"), ("S_STAGE", "S_PACKET")):
            for a, ai in rows.get(baseline, []):
                for b, bi in rows.get(candidate, []):
                    differing = [field for field in identity_fields if ai[field] != bi[field]]
                    if differing:
                        omissions.append({"reason": "unmatched_training_identity",
                                          "contrast": baseline + "→" + candidate, "state_id": key[-1],
                                          "partition": key[0], "reader": key[1], "family": key[2],
                                          "budget": key[3], "differing_fields": differing,
                                          "baseline_identity": ai, "candidate_identity": bi})
                        continue
                    paired = {**b, "absolute_packet_improvement": a["packet_loss"] - b["packet_loss"],
                              "absolute_random_improvement": a["random_loss"] - b["random_loss"],
                              "allocation_benefit_improvement": b["packet_vs_random"] - a["packet_vs_random"]}
                    group = (baseline + "→" + candidate, *key[:-1], ai["initialization"],
                             f"step_{ai['step']}", f"endpoint_{ai['endpoint']}", digest(ai)[:12])
                    groups[group].append(paired)
                    group_identities[group] = ai
    return {"/".join(map(str, key)): {"matched_identity": group_identities[key], **{
            field: episode_bootstrap(rows, field) for field in
            ("absolute_packet_improvement", "absolute_random_improvement", "allocation_benefit_improvement")}}
            for key, rows in sorted(groups.items())}
