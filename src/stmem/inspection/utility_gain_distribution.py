"""Describe every recorded expert-FM refinement, without new policy inference.

A candidate is one observed (decision, partial history, proposed refinement)
comparison, averaged within each independent noise bank. It is not an episode,
noise draw, independent trial, generated action, or closed-loop success.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import html
import json
import math
from pathlib import Path
import time

import numpy as np

from stmem.config import digest
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.utility_router_data import labels_identity, read_state

SCHEMA = "utility_gain_distribution_v1"
BANKS = ("discovery", "confirmation")
OPERATIONS = ("first", "packet")
SCOPES = ("full", "prefix")
FRACTIONS = (.01, .05, .10)
QUANTILES = (0., .01, .05, .25, .5, .75, .95, .99, 1.)


def _progress(phase, **values):
    print(json.dumps({"event": "utility_gain_distribution", "phase": phase, **values}), flush=True)


def _correlation(a, b):
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def distribution(values, epsilon):
    """All candidate rows carry equal weight, including zero and negative gains."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected a finite vector of gains")
    if not len(values):
        return {"count": 0, "mean": None, "quantiles": {}, "near_zero_fraction": None,
                "positive_fraction": None, "negative_fraction": None, "exact_zero_fraction": None,
                "positive_magnitude_sum": 0., "negative_magnitude_sum": 0., "absolute_magnitude_sum": 0.}
    return {"count": len(values), "mean": float(values.mean()), "std": float(values.std()),
            "quantiles": {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))},
            "near_zero_fraction": float(np.mean(np.abs(values) <= epsilon)),
            "positive_fraction": float(np.mean(values > epsilon)),
            "negative_fraction": float(np.mean(values < -epsilon)),
            "exact_zero_fraction": float(np.mean(values == 0)),
            "positive_magnitude_sum": float(np.maximum(values, 0).sum()),
            "negative_magnitude_sum": float(np.maximum(-values, 0).sum()),
            "absolute_magnitude_sum": float(np.abs(values).sum())}


def concentration(values):
    values = np.asarray(values, dtype=np.float64)
    result = {}
    for name, magnitude in (("absolute", np.abs(values)), ("positive", np.maximum(values, 0)),
                            ("negative", np.maximum(-values, 0))):
        ordered = np.sort(magnitude)[::-1]
        total = float(ordered.sum())
        result[name] = {str(fraction): {"selected_count": min(len(values), math.ceil(fraction * len(values))),
            "share": float(ordered[:math.ceil(fraction * len(values))].sum() / total) if total else None}
            for fraction in FRACTIONS}
    return result


def repeatability(discovery, confirmation, epsilon):
    """Choose the tail using discovery only; then inspect its fresh confirmation."""
    discovery = np.asarray(discovery, dtype=np.float64)
    confirmation = np.asarray(confirmation, dtype=np.float64)
    if discovery.shape != confirmation.shape or discovery.ndim != 1:
        raise ValueError("Paired noise banks must contain identical candidate rows")
    result = {"pearson": _correlation(discovery, confirmation),
              "tie_rule": "stable recorded candidate order; boundary ties are reported", "tails": {}}
    for name, score in (("absolute", np.abs(discovery)), ("positive", discovery), ("negative", -discovery)):
        eligible = np.arange(len(discovery)) if name == "absolute" else np.flatnonzero(score > epsilon)
        order = eligible[np.argsort(-score[eligible], kind="stable")]
        result["tails"][name] = {}
        for fraction in FRACTIONS:
            # Fraction of ALL candidate rows; do not silently change to a
            # fraction of the positive subpopulation.
            chosen = order[:math.ceil(fraction * len(discovery))]
            d, c = discovery[chosen], confirmation[chosen]
            result["tails"][name][str(fraction)] = {
                "selected_count": len(chosen), "eligible_count": len(eligible),
                "boundary_score": float(score[chosen[-1]]) if len(chosen) else None,
                "boundary_tie_count": int(np.sum(score[eligible] == score[chosen[-1]])) if len(chosen) else 0,
                "discovery_mean": float(d.mean()) if len(d) else None,
                "confirmation_mean": float(c.mean()) if len(c) else None,
                "confirmation_positive_fraction": float(np.mean(c > epsilon)) if len(c) else None,
                "confirmation_negative_fraction": float(np.mean(c < -epsilon)) if len(c) else None,
                "same_nonzero_sign_fraction": float(np.mean(
                    ((d > epsilon) & (c > epsilon)) | ((d < -epsilon) & (c < -epsilon)))) if len(c) else None,
            }
    return result


def _loss_arrays(context, arrays, metadata):
    prefix = context["array_prefix"]
    values = {key[len(prefix):]: value for key, value in arrays.items() if key.startswith(prefix)}
    banks = np.asarray(values["draw_bank"])
    expected = np.asarray([draw["bank"] for draw in metadata["draws"]])
    if banks.ndim != 1 or not np.array_equal(banks, expected) or set(banks) != set(BANKS):
        raise ValueError("Noise bank identities do not match committed draws")
    if not np.array_equal(values["draw_seeds"], [int(draw["seed"]) for draw in metadata["draws"]]):
        raise ValueError("Raw draw seeds disagree with committed metadata")
    pairs = [(draw["bank"], int(draw["draw_index"])) for draw in metadata["draws"]]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Repeated noise draw identity")
    seeds = {bank: {int(draw["seed"]) for draw in metadata["draws"] if draw["bank"] == bank} for bank in BANKS}
    if seeds["discovery"] & seeds["confirmation"]:
        raise ValueError("Discovery and confirmation noise banks overlap")
    n = len(context["candidates"])
    result = {}
    for scope in SCOPES:
        suffix = "loss" if scope == "full" else "prefix_loss"
        background = np.asarray(values["background_" + suffix], dtype=np.float64)
        if background.shape != (len(banks),) or not np.isfinite(background).all() or np.any(background < 0):
            raise ValueError("Invalid raw expert baseline losses")
        for bank in BANKS:
            mask = banks == bank
            result[f"{scope}_{bank}_background"] = np.full(n, background[mask].mean())
        for operation in OPERATIONS:
            after = np.asarray(values[f"{operation}_{suffix}"], dtype=np.float64)
            if after.shape != (n, len(banks)) or not np.isfinite(after).all() or np.any(after < 0):
                raise ValueError("Invalid raw expert candidate losses")
            paired = background[None, :] - after
            for bank in BANKS:
                delta = paired[:, banks == bank]
                result[f"{scope}_{operation}_{bank}"] = delta.mean(axis=1)
                result[f"{scope}_{operation}_{bank}_draw_std"] = delta.std(axis=1, ddof=1) if delta.shape[1] > 1 else np.full(n, np.nan)
                if scope == "full" and not np.array_equal(delta.mean(1), values[f"{bank}_{operation}_gain"]):
                    raise ValueError("Raw paired gains disagree with stored router targets")
    return result


def _columns():
    fields = ["partition", "reader", "task", "episode_id", "state_id", "decision_index", "context_id",
              "candidate_id", "packet_cost", "background_tokens", "prefix_stratum"]
    for scope in SCOPES:
        fields += [f"{scope}_{bank}_background" for bank in BANKS]
        for operation in OPERATIONS:
            for bank in BANKS:
                fields += [f"{scope}_{operation}_{bank}", f"{scope}_{operation}_{bank}_draw_std"]
    return fields


def collect(labels, csv_path, *, bound=None):
    """Validate the complete dataset once; retain scalar gains, never features."""
    labels = Path(labels)
    bound = labels_identity(labels) if bound is None else bound
    identity = bound["identity"]
    groups, episodes, contexts_seen = {}, {}, set()
    excluded = defaultdict(int)
    with gzip.open(csv_path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_columns())
        writer.writeheader()
        for number, state in enumerate(identity["states"]):
            if state["partition"] not in ("fit", "development"):
                raise ValueError("Unexpected label partition")
            episode_key = (state["task"], state["episode_id"])
            if episode_key in episodes and episodes[episode_key] != state["partition"]:
                raise ValueError("Episode leakage between fitting and development labels")
            episodes[episode_key] = state["partition"]
            metadata, arrays = read_state(labels, state, identity)
            original = bound["states"][state["state_id"]]
            if (original["npz_sha256"] != metadata["npz_sha256"] or original["json_sha256"] !=
                    file_sha256(labels / "states" / (state["state_id"] + ".json"))):
                raise ValueError("Label state changed after inventory validation")
            for exclusion in metadata["exclusions"]:
                excluded[f"{state['partition']}/{exclusion['reader']}/{exclusion['reason']}"] += 1
            for context in metadata["contexts"]:
                key = (state["partition"], context["reader"])
                if context["reader"] not in ("TREE", "FLAT"):
                    raise ValueError("Unknown reader")
                context_key = (state["state_id"], context["reader"], context["context_id"])
                if context_key in contexts_seen:
                    raise ValueError("Repeated refinement context")
                contexts_seen.add(context_key)
                cost = context["packet_cost"]
                if isinstance(cost, bool) or not isinstance(cost, int) or not 1 <= cost <= identity["protocol"]["packet_max_cost"]:
                    raise ValueError("Packet token cost is outside the declared protocol")
                ids = [str(row["candidate_id"]) for row in context["candidates"]]
                if len(set(ids)) != len(ids) or not 1 <= len(ids) <= identity["protocol"]["candidate_limit"]:
                    raise ValueError("Invalid candidate identity/count")
                if any(row["cost"] != cost for row in context["candidates"]):
                    raise ValueError("Candidate token costs disagree with their matched panel")
                values = _loss_arrays(context, arrays, metadata)
                group = groups.setdefault(key, {"arrays": defaultdict(list), "contexts": set(),
                    "states": set(), "episodes": set(), "tasks": set(), "task_chunks": [], "state_chunks": []})
                group["contexts"].add(context_key)
                group["states"].add(state["state_id"])
                group["episodes"].add(episode_key)
                group["tasks"].add(state["task"])
                group["task_chunks"].append(np.full(len(ids), state["task"], dtype=object))
                group["state_chunks"].append(np.full(len(ids), state["state_id"], dtype=object))
                group["arrays"]["packet_cost"].append(np.full(len(ids), cost, dtype=np.float64))
                for name, value in values.items():
                    group["arrays"][name].append(value)
                common = {field: state[field] for field in ("partition", "task", "episode_id", "state_id", "decision_index")}
                common.update(reader=context["reader"], context_id=context["context_id"], packet_cost=cost,
                              background_tokens=context["background_count"], prefix_stratum=context["prefix_stratum"])
                for candidate, name in enumerate(ids):
                    row = {field: float(value[candidate]) if np.isfinite(value[candidate]) else None for field, value in values.items()}
                    writer.writerow({**common, "candidate_id": name, **row})
            if number % 128 == 0 or number + 1 == len(identity["states"]):
                _progress("reading_labels", states=number + 1, total=len(identity["states"]))
    for group in groups.values():
        group["arrays"] = {key: np.concatenate(chunks) for key, chunks in group["arrays"].items()}
        group["task_rows"] = np.concatenate(group.pop("task_chunks"))
        group["state_rows"] = np.concatenate(group.pop("state_chunks"))
        group["counts"] = {key: len(group[key]) for key in ("contexts", "states", "episodes", "tasks")}
        group["counts"]["candidates"] = len(group["arrays"]["packet_cost"])
    return bound, groups, dict(excluded)


def _thin(x, y, maximum=2048):
    if len(x) <= maximum:
        return np.asarray(x).tolist(), np.asarray(y).tolist()
    # Include tail ranks logarithmically as well as the bulk. Descriptive
    # statistics/histograms remain exact; this only reduces HTML line vertices.
    ids = np.unique(np.concatenate((np.linspace(0, len(x)-1, maximum//2, dtype=int),
        np.geomspace(1, len(x), maximum//4).astype(int)-1,
        len(x)-np.geomspace(1, len(x), maximum//4).astype(int))))
    return np.asarray(x)[ids].tolist(), np.asarray(y)[ids].tolist()


def _trace(x, y, name, **kwargs):
    return {"x": np.asarray(x).tolist(), "y": np.asarray(y).tolist(), "name": name, **kwargs}


def plots(discovery, confirmation, epsilon):
    """Small plotting payload; every histogram bin and statistic uses all rows."""
    d, c = np.asarray(discovery), np.asarray(confirmation)
    n = len(c)
    both = np.concatenate((d, c))
    extent = float(np.max(np.abs(both))) if n else 0.
    limit = max(extent, epsilon, 1e-12)
    near_limit = max(float(np.quantile(np.abs(both), .5)), epsilon, limit * 1e-6) if n else limit
    panels = []
    for label, span in (("Signed gain distribution (all candidates)", limit),
                        ("Near-zero detail (same total-candidate denominator)", near_limit)):
        bins = np.linspace(-span, span, 81)
        traces = []
        for name, values in zip(BANKS, (d, c)):
            counts, _ = np.histogram(values, bins=bins)
            traces.append(_trace((bins[:-1]+bins[1:])/2, counts / max(1, n), name,
                                 type="bar", opacity=.65, width=float(bins[1]-bins[0])))
        panels.append({"data": traces, "layout": {"title": label, "barmode": "overlay",
            "xaxis": {"title": "Signed gain (positive = lower expert FM loss)"},
            "yaxis": {"title": "Fraction of all candidate refinements"}}})
    traces = []
    for name, values in zip(BANKS, (d, c)):
        values = np.sort(values)
        x, y = _thin(values, np.arange(1, n+1)/max(n, 1))
        traces.append(_trace(x, y, name, type="scatter", mode="lines", line={"shape": "hv"}))
    panels.append({"data": traces, "layout": {"title": "Empirical cumulative distribution",
        "xaxis": {"title": "Signed gain"}, "yaxis": {"title": "P(gain ≤ x)", "range": [0, 1]}}})
    traces = []
    for name, magnitude in (("absolute", np.abs(c)), ("positive", c[c > 0]), ("negative magnitude", -c[c < 0])):
        magnitude = magnitude[magnitude > 0]
        values, counts = np.unique(magnitude, return_counts=True)
        survival = np.cumsum(counts[::-1])[::-1] / max(1, n)
        x, y = _thin(values, survival)
        traces.append(_trace(x, y, name, type="scatter", mode="lines", line={"shape": "hv"}))
    panels.append({"data": traces, "layout": {"title": "Confirmation tails — log/log CCDF (zeros excluded)",
        "xaxis": {"title": "Positive gain / negative magnitude / |gain|", "type": "log"},
        "yaxis": {"title": "Fraction of ALL candidates ≥ magnitude", "type": "log"}}})
    traces = [_trace([0, 1], [0, 1], "equal magnitude reference", type="scatter", mode="lines", line={"dash": "dash"})]
    for name, magnitude in (("absolute", np.abs(c)), ("positive", np.maximum(c, 0)), ("negative", np.maximum(-c, 0))):
        values = np.sort(magnitude)[::-1]
        if values.sum():
            x, y = _thin(np.arange(1, n+1)/n, np.cumsum(values)/values.sum())
            traces.append(_trace([0, *x], [0, *y], name, type="scatter", mode="lines"))
    panels.append({"data": traces, "layout": {"title": "Concentration of confirmation effect magnitude",
        "xaxis": {"title": "Fraction of candidates, largest magnitude first", "range": [0, 1]},
        "yaxis": {"title": "Fraction of total magnitude", "range": [0, 1]}}})
    ids = np.argsort(d, kind="stable")
    if len(ids) > 5000:
        ids = ids[np.unique(np.linspace(0, len(ids)-1, 5000, dtype=int))]
    panels.append({"data": [_trace(d[ids], c[ids], "paired candidate", type="scatter", mode="markers",
        marker={"size": 4, "opacity": .35}), _trace([-limit, limit], [-limit, limit], "identity",
        type="scatter", mode="lines", line={"dash": "dash"})], "layout": {
        "title": f"Repeatability: {len(ids):,}/{n:,} displayed; statistics use ALL",
        "xaxis": {"title": "Discovery gain"}, "yaxis": {"title": "Independent confirmation gain"}}})
    # Raw JavaScript Plotly 3+ requires title objects; Python's legacy-string
    # coercion does not run when these payloads go directly to Plotly.react.
    for panel in panels:
        layout = panel["layout"]
        layout["title"] = {"text": layout["title"]}
        for axis in ("xaxis", "yaxis"):
            layout[axis]["title"] = {"text": layout[axis]["title"]}
    return panels


def summarize(groups, epsilon):
    views, payloads = {}, {}
    for (partition, reader), group in sorted(groups.items(), key=lambda item: (item[0][0] != "fit", item[0][1])):
        arrays = group["arrays"]
        weights = {}
        for name, rows in (("equal_state", group["state_rows"]), ("equal_task", group["task_rows"])):
            unique, inverse, counts = np.unique(rows, return_inverse=True, return_counts=True)
            weights[name] = 1. / (counts[inverse] * len(unique))
        for scope in SCOPES:
            for operation in OPERATIONS:
                for unit in ("total", "per_token"):
                    cost = arrays["packet_cost"] if unit == "per_token" and operation == "packet" else 1.
                    d = arrays[f"{scope}_{operation}_discovery"] / cost
                    c = arrays[f"{scope}_{operation}_confirmation"] / cost
                    key = "/".join((partition, reader, scope, operation, unit))
                    summary = {"counts": group["counts"], "epsilon": epsilon,
                        "gain_definition": "paired before-minus-after expert FM loss" + (" / added token count" if unit == "per_token" else ""),
                        "actual_router_target": scope == "full" and unit == "total",
                        "discovery": distribution(d, epsilon), "confirmation": distribution(c, epsilon),
                        "concentration_confirmation": concentration(c),
                        "repeatability": repeatability(d, c, epsilon),
                        "weighting_sensitivity": {name: {bank: {
                            "mean": float(np.sum(values*weight)),
                            "near_zero_fraction": float(np.sum((np.abs(values) <= epsilon)*weight)),
                            "positive_fraction": float(np.sum((values > epsilon)*weight)),
                            "negative_fraction": float(np.sum((values < -epsilon)*weight))}
                            for bank, values in zip(BANKS, (d, c))} for name, weight in weights.items()},
                        "baseline_raw_expert_loss": {bank: distribution(arrays[f"{scope}_{bank}_background"], 0.) for bank in BANKS},
                        "per_task": {task: {"discovery": distribution(d[group["task_rows"] == task], epsilon),
                                            "confirmation": distribution(c[group["task_rows"] == task], epsilon),
                                            "baseline_raw_expert_loss": {bank: distribution(
                                                arrays[f"{scope}_{bank}_background"][group["task_rows"] == task], 0.) for bank in BANKS}}
                                     for task in sorted(group["tasks"])}}
                    views[key] = summary
                    payloads[key] = {"summary": {k: v for k, v in summary.items() if k != "per_task"},
                                     "plots": plots(d, c, epsilon)}
    return views, payloads


def _plotly_js():
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as error:
        raise RuntimeError("Plotly is required: uv pip install --python .venv/bin/python 'plotly==6.3.1'") from error
    return get_plotlyjs()


def render_html(payloads, summary, path):
    data = json.dumps(payloads, allow_nan=False).replace("</", "<\\/")
    controls = "".join(f'<label>{html.escape(name)} <select id="{name}"></select></label>'
                       for name in ("partition", "reader", "horizon", "refinement", "units"))
    chart_nodes = "".join(f'<div id="plot{i}" class="plot"></div>' for i in range(6))
    source = html.escape(summary["dataset_sha256"])
    study = html.escape(summary["study_id"])
    document = """<!doctype html><html><head><meta charset="utf-8"><title>Refinement gain distributions</title>
<style>body{font:15px system-ui,sans-serif;margin:24px;color:#17202a;max-width:1700px}h1{font-size:26px}label{display:inline-block;margin:8px 18px 8px 0}select{padding:7px}.grid{display:grid;grid-template-columns:repeat(2,minmax(350px,1fr))}.plot{height:440px}pre{white-space:pre-wrap;background:#f2f5f8;padding:16px}small{overflow-wrap:anywhere}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<script>""" + _plotly_js() + """</script></head><body>
<h1>Individual refinement effects</h1>
<p>Positive gain means reduced expert flow-matching loss; negative gain means worse prediction. These are paired loss differences, not generated-action changes or success rates. Both noise banks average the same candidate across their own draws.</p>
<p>All candidate rows are included in distributions and statistics. Related candidates, contexts and frames are correlated; no independent-sample confidence interval is claimed. Fit/development and TREE/FLAT remain separate. The log/log tail is descriptive: a straight-looking segment does not establish a power law.</p>
<p>The current router fits <b>total, full-chunk discovery gain</b> (with a training-fitted scalar normalization). Per-token and executed-prefix views are additional diagnostics. Immediate = first +1-token move; packet = completed recorded-cost refinement.</p>
""" + "<p>Study: <code>" + study + "</code>; student step " + str(summary["student_step"]) + ".</p>" + controls + """<p id="counts"></p><div class="grid">""" + chart_nodes + """</div>
<h2>Exact descriptive statistics and discovery-selected tail checks</h2>
<p>Near-zero means |gain| ≤ the explicitly configured epsilon. Positive/negative fractions use ±epsilon. Tail concentration is the share of summed magnitudes, not additive causal benefit: overlapping refinements cannot be added together. Tail repeatability selects using discovery only, then inspects fresh confirmation.</p>
<pre id="stats"></pre><small>Dataset SHA256: """ + source + """. Full candidate rows: candidates.csv.gz; all task breakdowns and provenance: summary.json.</small>
<script>const views=""" + data + """;
const controls=['partition','reader','horizon','refinement','units'];
const keys=Object.keys(views);const options=controls.map((_,i)=>[...new Set(keys.map(k=>k.split('/')[i]))]);
controls.forEach((name,i)=>{const e=document.getElementById(name);options[i].forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=v==='first'?'immediate':v;e.appendChild(o)});e.onchange=render;});
function render(){let key=controls.map(k=>document.getElementById(k).value).join('/');let v=views[key];if(!v){document.getElementById('counts').textContent='No feasible candidate contexts for this selection.';for(let i=0;i<6;i++)Plotly.purge('plot'+i);document.getElementById('stats').textContent='';return;}
let c=v.summary.counts;document.getElementById('counts').textContent=`${c.candidates.toLocaleString()} candidates; ${c.contexts} contexts; ${c.states} decisions; ${c.episodes} task/episode pairs; ${c.tasks} tasks. Epsilon=${v.summary.epsilon}.`;
v.plots.forEach((p,i)=>Plotly.react('plot'+i,p.data,{...p.layout,margin:{l:75,r:20,t:100,b:70},legend:{orientation:'h',x:0,y:1.02,xanchor:'left',yanchor:'bottom'},template:'plotly_white'},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg',filename:key.replaceAll('/','_')+'_'+i}}));document.getElementById('stats').textContent=JSON.stringify(v.summary,null,2);}
render();</script></body></html>"""
    path.write_text(document)


def report_text(summary):
    lines = ["# Individual refinement effect distributions", "",
        f"Study: `{summary['study_id']}`. Dataset: `{summary['dataset_sha256']}`. Student step: {summary['student_step']}.", "",
        "Descriptive candidate-weighted expert FM gains. Positive means improved prediction. No policy inference or retraining was performed.", "",
        "Each row averages paired before/after losses within one noise bank; candidates are not independent episodes. Confirmation and fitting partitions are not interchangeable: confirmation is fresh noise, development is the episode split.", "",
        f"Near-zero threshold: `|gain| <= {summary['epsilon']}` in the selected gain units. No fitted power-law claim.", "",
        "The current router regression target is total full-chunk discovery gain with a training-only scale. Per-token views are additional diagnostics.", "",
        "| Partition / reader / horizon / credit | Candidates | Near-zero confirmation | Positive confirmation | Top 1% absolute share | Top-positive 1% confirmed positive |",
        "|---|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "undefined" if value is None else f"{100*value:.2f}%"
    for key, view in summary["views"].items():
        if not key.endswith("/total"):
            continue
        tails = view["repeatability"]["tails"]["positive"]["0.01"]
        lines.append(f"| {key.removesuffix('/total')} | {view['counts']['candidates']} | "
            f"{percent(view['confirmation']['near_zero_fraction'])} | {percent(view['confirmation']['positive_fraction'])} | "
            f"{percent(view['concentration_confirmation']['absolute']['0.01']['share'])} | {percent(tails['confirmation_positive_fraction'])} |")
    lines += ["", "Open `index.html` for signed histograms, near-zero detail, ECDF, positive/negative/absolute tail CCDFs, concentration curves and paired-noise scatter plots.",
        "", "`candidates.csv.gz` contains all per-candidate total gains, paired-draw standard deviations, baseline losses and identities. `summary.json` includes per-task distributions and discovery-selected top 1%, 5%, 10% repeatability.",
        "", "Concentration measures sums of magnitudes across recorded comparisons, not additive intervention benefits. Candidate distributions retain the label generator's sampling weights and are not a census of every possible archive refinement.",
        "", "Plot vertices/scatter may be thinned for display; histogram counts, quantiles, concentration and repeatability statistics use every recorded candidate. No success, generalization, power-law or independent-candidate significance claim follows from these plots."]
    return "\n".join(lines) + "\n"


def run(labels, output, *, epsilon=1e-4):
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("Near-zero epsilon must be finite and nonnegative")
    labels, output = Path(labels).resolve(), Path(output).resolve()
    if output == labels or labels in output.parents:
        raise ValueError("Write analysis outside the immutable label dataset")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Analysis output must be new or empty; use a fresh output directory")
    _plotly_js()  # Fail before hashing/reading the dataset if optional plotting dependency is absent.
    started = time.perf_counter()
    _progress("validating_complete_label_inventory", labels=str(labels))
    def validation_progress(done, total):
        if done % 128 == 0 or done == total:
            _progress("validating_labels", states=done, total=total)
    bound = labels_identity(labels, progress=validation_progress)
    output.mkdir(parents=True, exist_ok=True)
    _, groups, exclusions = collect(labels, output / "candidates.csv.gz", bound=bound)
    _progress("summarizing", groups=len(groups))
    views, payloads = summarize(groups, epsilon)
    identity = bound["identity"]
    summary = {"schema": SCHEMA, "status": "complete", "scope": "descriptive expert-FM candidate utility; no closed-loop claim",
        "labels": str(labels), "dataset_sha256": bound["dataset_sha256"], "label_identity_sha256": bound["identity_sha256"],
        "study_id": identity["study_id"], "student_sha256": identity["student_sha256"],
        "student_step": identity["student_step"], "source_of_labels": identity["source"],
        "protocol": identity["protocol"], "epsilon": epsilon, "planned_states": len(identity["states"]),
        "exclusions": exclusions, "views": views, "independent_candidate_trials": False,
        "candidate_csv_sha256": file_sha256(output / "candidates.csv.gz"),
        "analysis_source_sha256": file_sha256(Path(__file__)), "elapsed_seconds": time.perf_counter()-started}
    render_html(payloads, summary, output / "index.html")
    (output / "report.md").write_text(report_text(summary))
    atomic_json(summary, output / "summary.json")
    atomic_json({"schema": SCHEMA, "status": "complete", "dataset_sha256": bound["dataset_sha256"],
        "artifacts": {name: file_sha256(output / name) for name in ("summary.json", "index.html", "report.md", "candidates.csv.gz")},
        "summary_digest": digest(summary)}, output / "complete.json")
    return {"status": "complete", "output": str(output), "dataset_sha256": bound["dataset_sha256"],
            "groups": len(groups), "planned_states": len(identity["states"]), "elapsed_seconds": time.perf_counter()-started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="Existing complete expert-FM pilot labels directory")
    parser.add_argument("--output", type=Path, required=True, help="Fresh analysis directory outside labels")
    parser.add_argument("--epsilon", type=float, default=1e-4,
                        help="Explicit descriptive near-zero threshold in displayed gain units (default 0.0001)")
    args = parser.parse_args(argv)
    result = run(args.labels, args.output, epsilon=args.epsilon)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
