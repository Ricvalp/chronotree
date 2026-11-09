"""Offline views of measured memory-refinement gains and their repeatability.

This renderer never calls a policy, changes a frontier, or invents semantic
object labels. Geometry and target paths must come from the diagnostic runner.
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re

import numpy as np

from stmem.inspection.gain_statistics import summarize_gains


def _encoded(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, replacement in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                                   ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(character, replacement)
    return text


def _array(value, trailing_shape, name):
    values = np.asarray(value, dtype=float)
    if values.size and (values.ndim < len(trailing_shape)
                        or values.shape[-len(trailing_shape):] != trailing_shape
                        or not np.isfinite(values).all()):
        raise ValueError(f"{name} must be finite with trailing shape {trailing_shape}")
    return values


def _validate_frames(frames):
    for frame in frames:
        if frame.get("points") is not None:
            points = _array(frame["points"], (6,), "point cloud")
            if points.size and points.ndim != 2:
                raise ValueError("Point clouds must be [N,6] XYZRGB")
        if frame.get("tcp") is not None:
            values = _array(frame["tcp"], (3,), "observed TCP")
            if values.shape != (3,):
                raise ValueError("Observed TCP must be [3]")
        for image in frame.get("images", []):
            if not re.fullmatch(r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=\s]+",
                                image["data_url"]):
                raise ValueError("Camera images must be embedded PNG/JPEG/WebP data URLs")


def validate_result(result):
    """Require matched draws, finite measurements, and explicit diagnostic scope."""
    protocol = result.get("protocol", {})
    seeds = protocol.get("noise_seeds", [])
    if not protocol.get("scope") or len(seeds) < 4 or len(set(seeds)) != len(seeds):
        raise ValueError("A diagnostic scope and at least four unique paired noise seeds are required")
    contexts = result.get("contexts", [])
    if not contexts or len({str(c["id"]) for c in contexts}) != len(contexts):
        raise ValueError("Nonempty contexts with unique IDs are required")
    for context in contexts:
        candidates = context.get("candidates", [])
        if len(candidates) < 2 or len({str(c["id"]) for c in candidates}) != len(candidates):
            raise ValueError("Each context needs at least two distinct candidate moves")
        objectives = set(candidates[0]["gains"])
        if not objectives:
            raise ValueError("Measured candidate gains are required")
        _validate_frames([context.get("current", {})])
        _validate_frames([{"images": context.get("reference_images", [])}])
        for candidate in candidates:
            if set(candidate["gains"]) != objectives:
                raise ValueError("Every candidate must have the same measured objectives")
            for gains in candidate["gains"].values():
                values = np.asarray(gains, dtype=float)
                if values.shape != (len(seeds),) or not np.isfinite(values).all():
                    raise ValueError("Every candidate gain must retain all paired finite draws")
            _validate_frames(candidate.get("history_frames", []))
            if candidate.get("expert_mse") is not None:
                values = np.asarray(candidate["expert_mse"], dtype=float)
                if values.shape != (len(seeds),) or not np.isfinite(values).all():
                    raise ValueError("Expert MSE must retain all paired finite draws")
            for path in candidate.get("tcp_paths", {}).values():
                values = _array(path, (3,), "candidate target path")
                if values.size and (values.ndim != 3 or values.shape[0] != len(seeds)):
                    raise ValueError("Candidate target paths must be [draws,H,3]")
        for name, path in context.get("tcp_paths", {}).items():
            values = _array(path, (3,), "target path")
            if values.size and (values.ndim != (2 if name == "expert" else 3)
                                or (name != "expert" and values.shape[0] != len(seeds))):
                raise ValueError("Target paths must match the declared paired draws")
        if context.get("expert_valid") is not None:
            valid = np.asarray(context["expert_valid"])
            horizon = len(context.get("tcp_paths", {}).get("expert", []))
            if valid.dtype != np.bool_ or valid.shape not in ((horizon,), (horizon, 8)):
                raise ValueError("Expert validity must be boolean [H] or [H,8]")
        for values in context.get("expert_mse", {}).values():
            values = np.asarray(values, dtype=float)
            if values.shape != (len(seeds),) or not np.isfinite(values).all():
                raise ValueError("Reference expert MSE must retain all paired finite draws")
    _encoded(result)


def summarize_context(context, objective="final_action"):
    """Select extremes using the first half only; assess them on the second half.

    The held-out draws share the scene/frontier and are not held-out episodes.
    We deliberately retain negative-gain candidates and exact ties.
    """
    candidates = context["candidates"]
    gains = np.asarray([candidate["gains"][objective] for candidate in candidates], dtype=float)
    epsilon = context.get("statistics", {}).get(objective, {}).get("epsilon", 1e-6)
    measured = summarize_gains(gains.T, epsilon=epsilon)
    split = gains.shape[1] // 2
    discovery, confirmation = gains[:, :split], gains[:, split:]
    means = discovery.mean(axis=1)
    high, low = int(means.argmax()), int(means.argmin())
    if high == low:
        low = (high + 1) % len(candidates)
    is_best = gains >= gains.max(axis=0, keepdims=True) - epsilon
    win_share = (is_best / is_best.sum(axis=0, keepdims=True)).mean(axis=1)
    rows = []
    for index, candidate in enumerate(candidates):
        rows.append({
            "id": candidate["id"], "discovery_mean": float(means[index]),
            "confirmation_mean": float(confirmation[index].mean()),
            "all_mean": float(gains[index].mean()),
            "standard_deviation": float(gains[index].std(ddof=1)),
            "positive_fraction": float((gains[index] > epsilon).mean()),
            "best_fraction": float(win_share[index]),
        })
    return {"objective": objective, "discovery_draws": split,
            "confirmation_draws": gains.shape[1] - split,
            "high_id": candidates[high]["id"], "low_id": candidates[low]["id"],
            "confirmation_difference_mean": measured["validation"]["high_minus_low_mean_gain"],
            "confirmation_high_beats_low_fraction": measured["validation"]["high_beats_low_probability"],
            "candidates": rows}


def _filename(index, context):
    task = re.sub(r"[^A-Za-z0-9_-]", "_", str(context.get("task", "context")))
    reader = re.sub(r"[^A-Za-z0-9_-]", "_", str(context.get("reader", "reader")))
    return f"{index:03d}_{task}_{reader}.html"


def _checkpoint_note(protocol):
    if protocol.get("exact_hpc_compact_file") is not False:
        return ""
    step = protocol.get("compact", {}).get("step", "unknown")
    return (f"Checkpoint scope: the local student is at step {step:,}. " if isinstance(step, int)
            else f"Checkpoint scope: the local student step is {step}. ") + (
                "Its weights differ from the student used to generate the HPC router labels. "
                "This is a mechanism diagnostic with the available checkpoint, not an exact "
                "reproduction of those label-training results.")


def write_report(result: dict, output_dir: str | Path) -> Path:
    """Write a self-contained index and one interactive page per decision/reader."""
    validate_result(result)
    from plotly.offline import get_plotlyjs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plotly = get_plotlyjs().replace("</script", "<\\/script")
    cards, table_rows, destinations = [], [], {}
    for index, context in enumerate(result["contexts"]):
        objectives = list(context["candidates"][0]["gains"])
        objective = "final_action" if "final_action" in objectives else objectives[0]
        summary = summarize_context(context, objective)
        title = (f"{context.get('task', 'Task')} · {context.get('episode', 'episode')} · "
                 f"decision {context.get('decision_index', '?')} · {context.get('reader', 'reader')}")
        payload = {"protocol": result["protocol"], "context": context}
        page = _PAGE.replace("__STYLE__", _STYLE).replace("__TITLE__", escape(title))
        checkpoint_note = _checkpoint_note(result["protocol"])
        page = page.replace("__CHECKPOINT_NOTE__", (
            f'<p class="warning">{escape(checkpoint_note)}</p>' if checkpoint_note else ""))
        page = page.replace("__PLOTLY__", plotly).replace("__DATA__", _encoded(payload))
        page = page.replace("__SCRIPT__", _SCRIPT)
        filename = _filename(index, context)
        destinations[str(context["id"])] = filename
        (output_dir / filename).write_text(page, encoding="utf-8")
        delta = summary["confirmation_difference_mean"]
        fraction = summary["confirmation_high_beats_low_fraction"]
        note = context.get("selection_note", "Included in the declared diagnostic sample.")
        cards.append(
            f'<a class="context-card" href="{filename}"><span class="eyebrow">'
            f'{escape(str(context.get("reader", "")))} · B={escape(str(context.get("budget", "?")))}'
            f'</span><h2>{escape(title)}</h2><p>{escape(str(context.get("goal", "")))}</p>'
            f'<p>Discovery-selected high minus low, on held-out draws: <b>{delta:+.3g}</b>.'
            f' High wins on <b>{fraction:.0%}</b> of those draws.</p>'
            f'<p class="muted">{escape(str(note))}</p><strong>Inspect scene, evidence and actions →</strong></a>'
        )
        table_rows.append(
            f'<tr><td><a href="{filename}">{escape(str(context["id"]))}</a></td>'
            f'<td>{escape(str(context.get("reader", "?")))}</td>'
            f'<td>{escape(str(context.get("decision_index", "?")))}</td>'
            f'<td>{escape(str(context.get("budget", "?")))}</td>'
            f'<td>{delta:+.3g}</td><td>{fraction:.0%}</td></tr>'
        )
    notes = "".join(f"<li>{escape(str(note))}</li>" for note in result["protocol"].get("notes", []))
    scope = escape(str(result["protocol"]["scope"]))
    checkpoint_note = _checkpoint_note(result["protocol"])
    checkpoint_html = f'<p class="warning">{escape(checkpoint_note)}</p>' if checkpoint_note else ""
    featured = []
    for example in result.get("featured", []):
        ident = str(example["id"])
        if ident not in destinations:
            raise ValueError(f"Featured context does not exist: {ident}")
        featured.append(
            f'<a class="context-card" href="{destinations[ident]}"><span class="eyebrow">'
            f'{escape(ident)}</span><h2>{escape(str(example["title"]))}</h2>'
            f'<p>{escape(str(example["note"]))}</p><strong>Open this example →</strong></a>'
        )
    featured_html = (f'<section class="card"><h2>Suggested examples to inspect</h2>'
                     f'<p>Examples selected for interpretation after the diagnostic. They are illustrative, '
                     f'not a representative subset or an independent test set; every measured context '
                     f'remains available below.</p><div class="cards">{"".join(featured)}</div></section>'
                     if featured else "")
    examples_html = (f'<section class="card"><h2>All {len(table_rows)} measured contexts</h2>'
                     '<p>Final-action high/low choices use discovery draws only. The table reports their '
                     'comparison on held-out noise draws in the same scene. Discovery candidates tied '
                     'within the declared numerical tolerance share selection credit; a tied held-out '
                     'contrast receives half win credit. Individual pages show specific A/B representatives.</p><div class="table-scroll">'
                     '<table><thead><tr><th>Context</th><th>Reader</th><th>Decision</th><th>Budget</th>'
                     '<th>Held-out high − low</th><th>High wins (+½ ties)</th></tr></thead><tbody>'
                     + "".join(table_rows) + '</tbody></table></div></section>'
                     if len(cards) > 8 or featured else '<section class="cards">' + "".join(cards) + '</section>')
    index_page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Memory gain repeatability · Chronotree</title>
<style>{_STYLE}</style></head><body><main><header><div class="eyebrow">CHRONOTREE · LOCAL REAL-DATA DIAGNOSTIC</div>
<h1>Which historical refinements help—and does the answer repeat?</h1><p>{scope}</p>{checkpoint_html}
<p>Each card fixes the current scene, teacher, student, selected history and candidate moves.
Only the shared action-noise draw changes. High/low candidates are selected using the first half of the draws;
the card reports their comparison on the remaining draws. This split tests noise repeatability in the same scene,
not generalization to new episodes.</p></header>{featured_html}{examples_html}
<section class="card"><h2>What a gain means</h2><p>Positive gain means that this refinement reduces the measured
teacher–student discrepancy per additional history token. Negative gain means it increases that discrepancy.
These values are not task success rates, object-importance probabilities, or causal evidence that a visible object
explains the improvement. All available leaves remain represented before and after the move.</p>
<p>Historical clouds illustrate representative observed frames. Anchors and bounds describe archived tokens;
their boxes do not identify object membership or the exact points used by the encoder. Target paths show
forward kinematics of joint targets, not executed simulation.</p><ul>{notes}</ul>
<details><summary>Protocol and checkpoint identities</summary><pre>{escape(json.dumps(result['protocol'], indent=2))}</pre></details>
</section></main></body></html>'''
    path = output_dir / "index.html"
    path.write_text(index_page, encoding="utf-8")
    return path


_STYLE = r'''
:root{--ink:#193343;--muted:#5a7180;--line:#d9e3e8;--blue:#176e95;--green:#17805a;--orange:#c56d21}
*{box-sizing:border-box}body{margin:0;background:#f2f6f8;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1600px;margin:auto;padding:28px 24px 64px}h1{font-size:clamp(26px,3.6vw,40px);line-height:1.15;margin:10px 0 16px}
h2{font-size:20px;margin:0 0 10px}h3{font-size:16px;margin:0 0 8px}p{margin:10px 0}a{color:var(--blue)}
.eyebrow{font-size:12px;letter-spacing:.11em;font-weight:750;color:var(--blue)}.muted{color:var(--muted)}
.card,.context-card{background:white;border:1px solid var(--line);border-radius:13px;padding:21px;margin:18px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(350px,1fr));gap:18px}.context-card{display:block;text-decoration:none;color:var(--ink);margin:0}
.context-card:hover{border-color:var(--blue);box-shadow:0 3px 14px #1b526b12}.context-card strong{color:var(--blue)}
.controls{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}label{font-size:12px;font-weight:650;color:var(--muted)}
select{display:block;width:100%;padding:9px;margin-top:5px;border:1px solid #afc2cd;border-radius:6px;background:white;color:var(--ink)}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:18px}.plot3d{height:480px}.gain-plot{height:350px}#current-plot{height:580px}
.fact-row{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.fact{background:#edf4f7;padding:8px 11px;border-radius:7px;font-size:13px}
.camera-row{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}.camera-row figure{margin:0;flex:1;min-width:150px;max-width:310px}
.camera-row img{width:100%;border-radius:7px;border:1px solid var(--line)}figcaption{font-size:12px;color:var(--muted)}
.historical{border:1px solid var(--line);padding:16px;border-radius:9px;min-width:0}.selected-a{border-top:4px solid var(--green)}.selected-b{border-top:4px solid var(--orange)}
.note{background:#f0f6f8;padding:12px;border-radius:7px;font-size:13px}.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}
.warning{background:#fff6df;border-left:4px solid #bd8020;padding:12px 15px;border-radius:5px}
td,th{padding:9px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#f2f6f8}tbody tr.high{background:#eef9f2}tbody tr.low{background:#fff6ec}
summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:450px;overflow:auto}
#error{background:#fff0ed;color:#922f25;padding:12px}button{border:1px solid #b5c6ce;border-radius:5px;background:#f3f8fa;padding:7px 12px;cursor:pointer;font:inherit}
.topbar{display:flex;justify-content:space-between;gap:16px;align-items:center}.small{font-size:12px}.green{color:var(--green)}.orange{color:var(--orange)}
@media(max-width:900px){main{padding:18px 12px}.plots{grid-template-columns:1fr}.controls{grid-template-columns:1fr 1fr}.card{padding:15px}.plot3d{height:420px}#current-plot{height:500px}}
'''


_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__ · Memory gains</title>
<style>__STYLE__</style><script>__PLOTLY__</script></head><body><main>
<div class="topbar"><a href="index.html">← All diagnostic examples</a><span class="muted small">Self-contained · works offline</span></div>
<header><div class="eyebrow">CHRONOTREE · FIXED SCENE / FIXED HISTORY / PAIRED NOISE</div><h1>__TITLE__</h1>
<p id="goal"></p><p class="muted" id="scope"></p>__CHECKPOINT_NOTE__</header>
<div id="error" role="alert" hidden></div>
<section class="card" id="interpretive-section" hidden><h2>What to inspect in this example</h2><p id="interpretive-note"></p></section>
<section class="card" id="reference-section" hidden><h2>Initial conditioning reference</h2><p>These recorded camera views help identify the scene before objects were hidden. They are an interpretive reference, not the current observation or an object mask.</p><div class="camera-row" id="reference-images"></div></section>
<section class="card"><div class="controls"><label>Gain objective<select id="objective"></select></label>
<label>Paired noise draw<select id="draw"></select></label>
<label class="green">Candidate A<select id="candidate-a"></select></label>
<label class="orange">Candidate B<select id="candidate-b"></select></label></div>
<div class="fact-row" id="facts"></div><p class="note" id="selection-note"></p>
<p class="muted">Defaults choose the highest and lowest mean gain on discovery draws only. Change the dropdowns to inspect every sampled candidate, including negative or unstable moves. Candidate A is green; B is orange.</p>
</section>
<section class="card"><h2>Current observation and predicted action targets</h2><p>All curves use the selected paired noise draw. The current scene and history selection are identical; each candidate changes one historical refinement.</p>
<div class="controls"><label>View<select id="view"><option value="full">Full scene</option><option value="actions">Action area</option><option value="top">Top view</option></select></label></div>
<div id="current-plot"></div><div class="camera-row" id="current-images"></div>
<p class="note">Lines connect discrete TCP positions computed from clipped joint targets. They are not executed trajectories, collision tests, or evidence of task success. Gains use normalized eight-component actions before controller clipping; these paths show only TCP XYZ from the seven arm joints, so orientation and gripper differences are not visible. Expert targets are a separate reference; unavailable or padded expert targets are hidden. The paths always correspond to final-action inference, even when the gain chart is set to the velocity objective.</p>
</section>
<section class="card"><h2>Which historical evidence is being refined?</h2>
<p>Each panel shows a representative historical frame alongside the candidate's token anchors and spatial bounds. These are token summaries, not object masks. The temporal span may cover more frames than the one displayed.</p>
<div class="controls"><label>Historical view<select id="history-view"><option value="candidate">Candidate area</option><option value="full">Whole historical frame</option></select></label></div>
<div class="plots"><article class="historical selected-a"><h3 id="history-a-title"></h3><label>Representative observed frame<select id="frame-a"></select></label><div class="plot3d" id="history-a"></div><div id="meta-a" class="small"></div><div class="camera-row" id="images-a"></div></article>
<article class="historical selected-b"><h3 id="history-b-title"></h3><label>Representative observed frame<select id="frame-b"></select></label><div class="plot3d" id="history-b"></div><div id="meta-b" class="small"></div><div class="camera-row" id="images-b"></div></article></div>
<p class="muted">For TREE, one parent is replaced by its two children; a line joins each child anchor to the parent anchor. For FLAT, one leaf is exposed individually and its source/modality residual is updated. Neither operation removes the remaining history. Descendant anchor dots are shown for orientation and are not all separate active context tokens. Anchors and boxes do not depict exact encoder pooling support or object membership. Historical times are source-local frame indices.</p></section>
<section class="card"><h2>Does candidate utility repeat across noise draws?</h2><p id="objective-description"></p><div class="gain-plot" id="gain-plot"></div>
<p class="muted">The dotted divider separates discovery and held-out noise draws. The action input is regenerated for every draw, but each comparison shares that draw across teacher, baseline student and all candidate students. Positive gain reduces teacher–student error per extra token; negative gain increases it.</p>
<div class="table-scroll"><table><thead><tr><th>Candidate</th><th>Discovery mean</th><th>Held-out mean</th><th>All-draw SD</th><th>Positive draws</th><th>Best-candidate share</th></tr></thead><tbody id="gain-table"></tbody></table></div>
<p class="muted" id="tie-note"></p></section>
<section class="card" id="expert-section" hidden><h2>Does teacher agreement also improve agreement with the expert?</h2><p>This separate diagnostic uses masked normalized eight-component action MSE against the recorded expert chunk, averaged over held-out noise draws. It is not the gain-label objective and is not task success. Lower MSE is better; a positive change versus the unrefined student means the move makes expert agreement worse.</p><div class="table-scroll"><table><thead><tr><th>Prediction</th><th>Held-out expert MSE</th><th>Change versus student before</th><th>Draws with higher expert error</th></tr></thead><tbody id="expert-table"></tbody></table></div></section>
<section class="card"><h2>Protocol and limits</h2><p id="interpretation"></p><ul id="notes"></ul>
<details><summary>Context measurements and checkpoint identities</summary><pre id="provenance"></pre></details>
<button id="download">Download this example's JSON</button></section>
</main><script type="application/json" id="payload">__DATA__</script><script>__SCRIPT__</script></body></html>'''


_SCRIPT = r'''
'use strict';
const DATA=JSON.parse(document.getElementById('payload').textContent),C=DATA.context,P=DATA.protocol,el=id=>document.getElementById(id);
const palette=['#17805a','#c56d21','#7954ac','#2284ad','#b74f81','#7c792b','#4d7475','#b45e53'];
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length,sd=a=>Math.sqrt(a.reduce((s,x)=>s+(x-mean(a))**2,0)/(a.length-1));
const fmt=x=>Number.isFinite(x)?x.toExponential(3):'unavailable',source=s=>s===1?'conditioning':s===0?'execution':String(s??'source unavailable');
const drawCount=P.noise_seeds.length,split=Math.floor(drawCount/2),choices=C.candidates;
const epsilon=()=>C.statistics?.[el('objective').value]?.epsilon??1e-6;
const opts={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
function options(id,values,format=x=>x){const node=el(id),prior=node.value;node.replaceChildren();for(const value of values){const o=document.createElement('option');o.value=String(value);o.textContent=format(value);node.appendChild(o);}if(values.map(String).includes(prior))node.value=prior;}
function index(id){return Number(el(id).value);}
function gain(i){return choices[i].gains[el('objective').value];}
function sceneLayout(id,focus=null,top=false){
 const axes=['X','Y','Z'].map(a=>({title:a+' (m)',gridcolor:'#dce5ea',backgroundcolor:'#f8fafb'}));
 if(focus?.length){for(let k=0;k<3;k++){const v=focus.map(p=>p[k]).filter(Number.isFinite),lo=Math.min(...v),hi=Math.max(...v),pad=Math.max(.035,(hi-lo)*.16);axes[k].range=[lo-pad,hi+pad];}}
 return {margin:{l:0,r:0,t:5,b:0},scene:{aspectmode:'data',xaxis:axes[0],yaxis:axes[1],zaxis:axes[2],camera:{eye:top?{x:0,y:0,z:2.5}:{x:1.4,y:-1.7,z:1.2},up:{x:0,y:top?1:0,z:top?0:1}}},uirevision:id,legend:{orientation:'h',y:-.02,font:{size:11}}};
}
function cloud(points,name='Observed RGB-D cloud'){
 return {type:'scatter3d',mode:'markers',name,x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),marker:{size:2,opacity:.6,color:points.map(p=>'rgb('+p.slice(3,6).map(v=>Math.round(255*Math.max(0,Math.min(1,v)))).join(',')+')')},hoverinfo:'skip'};
}
function dots(points,name,color,size=6,symbol='circle'){
 return {type:'scatter3d',mode:'markers',name,x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),marker:{size,color,symbol},hovertemplate:name+'<br>%{x:.3f}, %{y:.3f}, %{z:.3f} m<extra></extra>'};
}
function line(points,name,color,dash='solid',width=5){
 return {type:'scatter3d',mode:'lines+markers',name,x:points.map(p=>p?.[0]??null),y:points.map(p=>p?.[1]??null),z:points.map(p=>p?.[2]??null),customdata:points.map((_,i)=>i+1),line:{color,width,dash},marker:{size:3,color},hovertemplate:name+'<br>target %{customdata}<br>%{x:.3f}, %{y:.3f}, %{z:.3f} m<extra></extra>'};
}
function imageRow(id,images=[]){const node=el(id);node.replaceChildren();for(const item of images){const f=document.createElement('figure'),im=document.createElement('img'),cap=document.createElement('figcaption');im.src=item.data_url;im.alt=item.label??'Recorded camera image';cap.textContent=item.label??'Recorded camera image';f.append(im,cap);node.appendChild(f);}}
function currentPlot(){
 const d=index('draw'),a=index('candidate-a'),b=index('candidate-b'),traces=[],focus=[],current=C.current??{};
 if(current.points?.length)traces.push(cloud(current.points));
 if(current.tcp){traces.push(dots([current.tcp],'Observed current TCP','#172e3e',8,'diamond'));focus.push(current.tcp);}
 const expert=C.tcp_paths?.expert?.map((p,i)=>{const valid=C.expert_valid?.[i];return valid===undefined||(Array.isArray(valid)?valid.every(Boolean):valid)?p:null;});
 for(const [path,name,color,dash] of [[C.tcp_paths?.teacher?.[d],'Teacher · full history','#283b47','dash'],[C.tcp_paths?.before?.[d],'Student · before move','#2377ae','dot'],[choices[a].tcp_paths?.after?.[d],'Student · candidate A','#17805a','solid'],[choices[b].tcp_paths?.after?.[d],'Student · candidate B','#c56d21','solid'],[expert,'Expert target chunk','#995aab','dash']]){if(path?.some(Boolean)){traces.push(line(path,name,color,dash));focus.push(...path.filter(Boolean));}}
 const view=el('view').value;Plotly.react('current-plot',traces,sceneLayout('current-'+view,view==='actions'?focus:null,view==='top'),opts);
 imageRow('current-images',current.images);
}
function recordAnchor(r){return r.anchor??r.mean_anchor??null;}
function bounds(r){return [r.bounds_min??r.lower??r.bbox_min,r.bounds_max??r.upper??r.bbox_max];}
function box(r,name,color){const [lo,hi]=bounds(r);if(!lo||!hi)return null;const corners=Array.from({length:8},(_,i)=>[i&1?hi[0]:lo[0],i&2?hi[1]:lo[1],i&4?hi[2]:lo[2]]),points=[];for(let i=0;i<8;i++)for(const mask of [1,2,4])if(!(i&mask))points.push(corners[i],corners[i|mask],null);const trace=line(points,name,color,'dot',2);trace.mode='lines';trace.hoverinfo='skip';return trace;}
function metadata(r){const tmin=r.time_min??r.t_min,tmax=r.time_max??r.t_max;return `${source(r.source)} · ${r.count??'?'} descendant leaves · source time ${tmin??'?'}–${tmax??'?'} · modality ${r.modality??'visual'}`;}
function historyPlot(which,reset=false){
 const ci=index('candidate-'+which),candidate=choices[ci],frames=candidate.history_frames??[],field='frame-'+which,color=which==='a'?'#17805a':'#c56d21';
 const old=el(field).value;options(field,frames.map((_,i)=>i),i=>`${source(frames[i].source)} · frame ${frames[i].source_index??'?'}${frames[i].label?' · '+frames[i].label:''}`);if(reset)el(field).value='0';else if(old)el(field).value=old;
 const frame=frames[index(field)]??{},record=candidate.record??{},traces=[],anchor=recordAnchor(record),leaves=candidate.leaves??[];
 if(frame.points?.length)traces.push(cloud(frame.points,'Representative historical frame'));
 const leafPoints=leaves.map(l=>Array.isArray(l)?l:recordAnchor(l)).filter(Boolean);if(leafPoints.length)traces.push(dots(leafPoints,'Descendant anchors across candidate interval','#8b9da7',3));
 const bbox=box(record,'Candidate bounds',color);if(bbox)traces.push(bbox);if(anchor)traces.push(dots([anchor],'Candidate mean anchor',color,8,'diamond'));
 for(const [i,child] of (candidate.children??[]).entries()){const p=recordAnchor(child),cc=i===0?'#a149a3':'#3264b5';if(p){traces.push(dots([p],`Child ${i+1} anchor`,cc,7));if(anchor){const edge=line([anchor,p],`Parent → child ${i+1}`,cc,'solid',3);edge.mode='lines';traces.push(edge);}}const cb=box(child,`Child ${i+1} bounds`,cc);if(cb)traces.push(cb);}
 el('history-'+which+'-title').textContent=`Candidate ${which.toUpperCase()}: ${candidate.id} · ${candidate.operation??C.reader}`;
 const displayed=leafPoints.length,total=candidate.total_descendant_leaves??candidate.leaf_count??record.count??displayed;
 el('meta-'+which).textContent=metadata(record)+`. Showing ${displayed} of ${total} descendant anchors. `+(frames.length?`${frames.length} representative frame(s), not the complete time span.`:'No representative raw frame was available.');
 const [lo,hi]=bounds(record),focus=el('history-view').value==='candidate'?[anchor,lo,hi,...(candidate.children??[]).map(recordAnchor)].filter(Boolean):null;
 imageRow('images-'+which,frame.images);Plotly.react('history-'+which,traces,sceneLayout('history-'+which+'-'+el('history-view').value,focus),opts);
}
function summaries(){return choices.map((candidate,i)=>{const g=gain(i),disc=g.slice(0,split),held=g.slice(split),best=g.map((v,j)=>{const column=choices.map((_,k)=>gain(k)[j]),max=Math.max(...column);return v>=max-epsilon()?1/column.filter(x=>x>=max-epsilon()).length:0;});return {id:candidate.id,disc:mean(disc),held:mean(held),sd:sd(g),positive:g.filter(x=>x>epsilon()).length/drawCount,best:mean(best)};});}
function expertTable(){
 const before=C.expert_mse?.before,teacher=C.expert_mse?.teacher;if(!before?.length)return;el('expert-section').hidden=false;const a=index('candidate-a'),b=index('candidate-b'),tbody=el('expert-table');tbody.replaceChildren();
 const rows=[['Student · before move',before,-1],['Teacher · full history',teacher,-2],...choices.map((c,i)=>['Student · '+c.id,c.expert_mse,i])];
 for(const [name,values,i] of rows){if(!values?.length)continue;const held=values.slice(split),delta=held.map((v,j)=>v-before[split+j]),tr=document.createElement('tr');if(i===a)tr.className='high';else if(i===b)tr.className='low';for(const text of [name,fmt(mean(held)),i===-1?'reference':fmt(mean(delta)),i===-1?'—':`${delta.filter(x=>x>0).length}/${delta.length}`]){const td=document.createElement('td');td.textContent=text;tr.appendChild(td);}tbody.appendChild(tr);}
}
function gainPlot(){
 const a=index('candidate-a'),b=index('candidate-b'),d=index('draw'),rows=summaries(),traces=choices.map((candidate,i)=>({type:'scatter',mode:'lines+markers',name:String(candidate.id),x:gain(i).map((_,j)=>j+1),y:gain(i),line:{color:i===a?'#17805a':i===b?'#c56d21':palette[(i+2)%palette.length],width:i===a||i===b?3:1,dash:i===a||i===b?'solid':'dot'},marker:{size:i===a||i===b?6:4},customdata:P.noise_seeds,hovertemplate:'Draw %{x} · noise seed %{customdata}<br>gain %{y:.5g}<extra>%{fullData.name}</extra>'}));
 Plotly.react('gain-plot',traces,{margin:{l:80,r:20,t:20,b:48},xaxis:{title:'Paired noise draw',dtick:1},yaxis:{title:'Gain / added history token',zeroline:true,zerolinecolor:'#586c78'},legend:{orientation:'h',y:1.12},shapes:[{type:'line',x0:split+.5,x1:split+.5,y0:0,y1:1,yref:'paper',line:{color:'#788c98',dash:'dot'}},{type:'line',x0:d+1,x1:d+1,y0:0,y1:1,yref:'paper',line:{color:'#d5b23a',width:1}}],uirevision:'gain-'+el('objective').value},opts);
 const tbody=el('gain-table');tbody.replaceChildren();for(let i=0;i<rows.length;i++){const r=rows[i],tr=document.createElement('tr');if(i===a)tr.className='high';else if(i===b)tr.className='low';for(const value of [String(r.id),fmt(r.disc),fmt(r.held),fmt(r.sd),(100*r.positive).toFixed(0)+'%',(100*r.best).toFixed(1)+'%']){const td=document.createElement('td');td.textContent=value;tr.appendChild(td);}tbody.appendChild(tr);}
 const delta=gain(a).slice(split).map((v,i)=>v-gain(b)[split+i]),facts=[`A discovery mean: ${fmt(rows[a].disc)}`,`B discovery mean: ${fmt(rows[b].disc)}`,`Held-out A − B: ${fmt(mean(delta))}`,`A beats B: ${delta.filter(v=>v>epsilon()).length}/${delta.length} · near ties: ${delta.filter(v=>Math.abs(v)<=epsilon()).length}/${delta.length}`];el('facts').replaceChildren();for(const text of facts){const node=document.createElement('span');node.className='fact';node.textContent=text;el('facts').appendChild(node);}
 el('tie-note').textContent=`Standard deviation describes variability across draws, not uncertainty across episodes. Positive gains exceed ${epsilon()}; candidates within ${epsilon()} of the best gain share win credit. This is a numerical deadband, not a statistical significance threshold. Individual A/B comparisons use the displayed candidates; the index averages discovery-tied candidate sets. Candidates are not semantic object-importance scores.`;
 el('objective-description').textContent=el('objective').value==='velocity'?'Velocity labels compare teacher and student flow velocities at the same expert-conditioned noisy action and sampled flow time. Each draw changes both noise and flow time.':'Final-action labels compare teacher and student normalized final chunks after independent integration from the same initial noise. They use '+(P.integration_steps??8)+' integration steps; each draw is a separate initial-noise sample.';
}
function defaults(){const rows=summaries(),m=rows.map(r=>r.disc);let a=m.indexOf(Math.max(...m)),b=m.indexOf(Math.min(...m));if(a===b)b=(a+1)%m.length;el('candidate-a').value=String(a);el('candidate-b').value=String(b);}
function refresh(reset=false){try{el('error').hidden=true;currentPlot();historyPlot('a',reset);historyPlot('b',reset);gainPlot();expertTable();}catch(e){el('error').hidden=false;el('error').textContent=e.stack??String(e);}}
function init(){
 const objectives=Object.keys(choices[0].gains);options('objective',objectives,x=>x==='final_action'?'Final denoised actions':x==='velocity'?'Flow velocities':x);if(objectives.includes('final_action'))el('objective').value='final_action';
 options('draw',P.noise_seeds.map((_,i)=>i),i=>`${i+1}: seed ${P.noise_seeds[i]} · ${i<split?'discovery':'held-out'}`);
 for(const which of ['a','b'])options('candidate-'+which,choices.map((_,i)=>i),i=>String(choices[i].id));defaults();
 el('goal').textContent='Task instruction: '+(C.goal??'unavailable');el('scope').textContent=P.scope;
 if(C.interpretation){el('interpretive-section').hidden=false;el('interpretive-note').textContent=C.interpretation;}
 if(C.reference_images?.length){el('reference-section').hidden=false;imageRow('reference-images',C.reference_images);}
 el('selection-note').textContent=`Discovery draws 1–${split}; held-out draws ${split+1}–${drawCount}. `+(C.selection_note??P.selection_note??'All sampled candidates are retained.');
 el('interpretation').textContent='This is a local real-training-scene diagnostic. The held-out split is over noise draws in the same fixed scene, not over episodes. Gains measure agreement with the frozen full-history teacher per added history token; they do not measure task success. A historical image can suggest a semantic explanation, but it cannot establish which object or event caused a gain. No oracle labels enter the policy.';
 for(const note of P.notes??[]){const li=document.createElement('li');li.textContent=note;el('notes').appendChild(li);}
 const {current:ignored,candidates:removed,tcp_paths:paths,...contextMetadata}=C;el('provenance').textContent=JSON.stringify({protocol:P,context:contextMetadata,candidates:choices.map(c=>({id:c.id,operation:c.operation,record:c.record,children:c.children,statistics:c.statistics}))},null,2);
 el('objective').addEventListener('change',()=>{defaults();refresh(true);});for(const id of ['draw','view','history-view'])el(id).addEventListener('change',()=>refresh());for(const which of ['a','b']){el('candidate-'+which).addEventListener('change',()=>refresh(true));el('frame-'+which).addEventListener('change',()=>refresh());}
 el('download').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(DATA,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='gain_repeatability_'+String(C.id).replace(/[^A-Za-z0-9_-]/g,'_')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});refresh(true);
}
init();
'''
