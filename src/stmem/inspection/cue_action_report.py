"""Self-contained views of controlled history refinement and action chunks.

All geometry and predictions are supplied by the measured diagnostic runner.
The viewer neither changes evidence nor infers object labels from token anchors.
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re

import numpy as np


def _finite(value, shape, label):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return array


def _frame(frame):
    points = np.asarray(frame.get("points", []), dtype=float)
    if points.size and (points.ndim != 2 or points.shape[1] != 6 or not np.isfinite(points).all()):
        raise ValueError("Point clouds must be finite [N,6] XYZRGB")
    if "tcp" in frame:
        _finite(frame["tcp"], (3,), "Observed TCP")
    for image in frame.get("images", []):
        if not re.fullmatch(r"data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=\s]+", image["data_url"]):
            raise ValueError("Source images must be embedded image data URLs")


def validate_result(result):
    """Reject mismatched draws or horizons rather than visually pairing them."""
    protocol = result.get("protocol", {})
    seeds = protocol.get("noise_seeds", [])
    if not protocol.get("scope") or len(seeds) < 1 or len(set(seeds)) != len(seeds):
        raise ValueError("Require diagnostic scope and unique paired noise seeds")
    scenes = result.get("scenes", [])
    if not scenes or len({scene["id"] for scene in scenes}) != len(scenes):
        raise ValueError("Require nonempty scenes with unique IDs")
    for scene in scenes:
        _frame(scene["current"])
        for frame in scene.get("cue_frames", []):
            _frame(frame)
        for region in scene.get("target_regions", []):
            _finite(region["center"], (3,), "Annotated region center")
            if not isinstance(region.get("label"), str) or not region["label"]:
                raise ValueError("Annotated regions require descriptive labels")
        variants = scene.get("variants", [])
        names = {variant["name"] for variant in variants}
        if len(variants) < 2 or len(names) != len(variants):
            raise ValueError("Require at least two uniquely named variants")
        horizon = None
        for variant in variants:
            actions = np.asarray(variant["actions"], dtype=float)
            if actions.ndim != 3 or actions.shape[0] != len(seeds) or actions.shape[-1] != 8:
                raise ValueError("Actions must have shape [paired draws,H,8]")
            if actions.shape[1] < 1 or not np.isfinite(actions).all():
                raise ValueError("Actions must be nonempty and finite")
            if horizon is None:
                horizon = actions.shape[1]
            if actions.shape[1] != horizon:
                raise ValueError("Variants must share their action horizon")
            _finite(variant["tcp_paths"], (len(seeds), horizon, 3), "TCP paths")
            for node in variant.get("selected_nodes", []):
                _finite(node["anchor"], (3,), "Selected node anchor")
                for field in ("bounds_min", "bounds_max"):
                    if field in node:
                        _finite(node[field], (3,), field)
        if "expert_tcp" in scene:
            _finite(scene["expert_tcp"], (horizon, 3), "Expert TCP path")
        if "expert_valid" in scene:
            valid = np.asarray(scene["expert_valid"])
            if valid.dtype != np.bool_ or valid.shape not in ((horizon,), (horizon, 8)):
                raise ValueError("Expert validity must be boolean [H] or [H,8]")
        for comparison in scene.get("comparisons", []):
            if comparison["a"] not in names or comparison["b"] not in names:
                raise ValueError("Comparison refers to an absent variant")
    # This also rejects nonfinite values in supplementary metrics/provenance.
    json.dumps(result, allow_nan=False)


def _encoded(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for source, destination in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                                ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(source, destination)
    return text


def write_report(result, output_dir):
    """Render measured paths and evidence in a single offline HTML artifact."""
    validate_result(result)
    from plotly.offline import get_plotlyjs

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = result["protocol"]
    page = _PAGE.replace("__SCOPE__", escape(protocol["scope"]))
    page = page.replace("__TITLE__", escape(protocol.get("title", "Does exposing the cue change the action?")))
    page = page.replace("__INTERVENTION__", escape(protocol.get("intervention_description", (
        "The current observation, task and sampled action noise stay fixed across variants. "
        "Refinement changes how finely history is represented: coarse summaries still contain their descendants. "
        "These curves are joint-target forward kinematics, not executed motion or success measurements."
    ))))
    page = page.replace("__STYLE__", _STYLE).replace("__SCRIPT__", _SCRIPT)
    page = page.replace("__PLOTLY__", get_plotlyjs().replace("</script", "<\\/script"))
    page = page.replace("__PAYLOAD__", _encoded(result))
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>__STYLE__</style></head><body>
<main><header><p class="eyebrow">FROZEN-POLICY DIAGNOSTIC</p><h1>__TITLE__</h1>
<p>__SCOPE__</p><p class="caution">__INTERVENTION__</p></header>
<section class="controls"><label>Decision / experiment<select id="sceneSelect"></select></label>
<label>Paired noise draw<select id="draw"></select></label>
<label>Action-path comparison<select id="comparison"></select></label></section>
<section class="card"><h2 id="sceneTitle"></h2><p id="goal"></p><p id="sceneNote" class="muted"></p>
<div id="paths" class="plot"></div><p id="pairNote" class="muted"></p>
<div id="delta" class="deltaPlot"></div><p class="muted">All shown variants use the same Gaussian draw.
Click a legend entry to hide it; double-click to isolate it. Path markers are predicted target positions in order.
The dotted black path is the valid expert target chunk, when provided. The dashed variant is the baseline;
the solid variant is the comparison. The separation plot makes small differences visible without shifting the paths.
TCP curves are joint-target forward kinematics, not physically executed motion or task success.</p>
<p id="regionNote" class="muted"></p><div id="currentImages" class="images"></div></section>
<section class="card"><h2>Historical evidence and selected centers</h2>
<div class="controls"><label>Historical frame<select id="frame"></select></label>
<label>History selection to inspect<select id="variant"></select></label></div>
<div id="history" class="plot"></div><p id="historyNote" class="muted"></p><div id="historyImages" class="images"></div>
<p id="pairedImageNote" class="muted"></p><div id="pairedHistoryImages" class="images"></div>
<p class="muted">Centers and bounds describe encoded regions, not object masks or exact attention weights.
Node time intervals can overlap several observations. Only provided historical frames are shown here.</p></section>
<section class="card"><h2>What each variant changes</h2><div class="scroll"><table id="variantTable"></table></div></section>
<section class="card"><h2>Measured changes in the complete chunk</h2><p class="muted">Numbers are supplied by the diagnostic runner.
For per-draw arrays, the table reports the mean and range across the declared paired draws. A small distance change is not a behavioral switch.</p>
<div class="scroll"><table id="comparisonTable"></table></div></section>
<details class="card"><summary>Protocol, checks and interpretation</summary><ul id="notes"></ul><pre id="protocol"></pre><pre id="summary"></pre></details>
</main><script>__PLOTLY__</script><script type="application/json" id="payload">__PAYLOAD__</script><script>__SCRIPT__</script></body></html>"""

_STYLE = """*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#172333;font:16px/1.55 system-ui,sans-serif}main{max-width:1360px;margin:auto;padding:32px 24px 60px}h1{font-size:clamp(28px,4vw,44px);line-height:1.12;margin:8px 0 16px}h2{font-size:23px;margin:0 0 10px}.eyebrow{font-size:12px;letter-spacing:.14em;font-weight:750;color:#3155aa}.muted{color:#566376;font-size:14px}.caution{padding:16px 20px;background:#fff5db;border-left:4px solid #c68e20;border-radius:7px}.card{margin:22px 0;padding:24px;background:white;border:1px solid #dce2eb;border-radius:14px;box-shadow:0 5px 22px #1d324508}.controls{display:flex;gap:20px;flex-wrap:wrap;margin:16px 0}label{font-size:13px;font-weight:650;display:flex;flex-direction:column;gap:6px;flex:1;min-width:180px}select{font:inherit;font-weight:450;padding:9px;border:1px solid #bec9d9;background:#fff;border-radius:6px;max-width:100%}.plot{height:600px}.deltaPlot{height:240px}.images{display:flex;gap:14px;flex-wrap:wrap}.images figure{margin:0}.images img{width:256px;max-width:100%;border-radius:8px;border:1px solid #dce2eb}.images figcaption{font-size:13px;color:#566376}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #e1e6ed;vertical-align:top}th{background:#f4f6fa;white-space:nowrap}td:first-child{font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}summary{cursor:pointer;font-weight:700}@media(max-width:700px){main{padding:18px 12px}.card{padding:16px}.plot{height:440px}}"""

_SCRIPT = r"""const data=JSON.parse(document.getElementById('payload').textContent);
const el=id=>document.getElementById(id);
const colors=['#2563eb','#e4572e','#069c72','#9c43c8','#d79913','#3c758f','#ce4180','#576673'];
const pretty=name=>name.replaceAll('_',' ');
const opt=(select,value,label)=>{const e=document.createElement('option');e.value=value;e.textContent=label;select.append(e)};
data.scenes.forEach((s,i)=>opt(el('sceneSelect'),i,`${s.id} · decision ${s.decision_index}`));
const primaryScene=data.scenes.findIndex(s=>s.id==='decision000_B128');if(primaryScene>=0)el('sceneSelect').value=primaryScene;
data.protocol.noise_seeds.forEach((seed,i)=>opt(el('draw'),i,`${i+1} / ${data.protocol.noise_seeds.length} · seed ${seed}`));
const scene=()=>data.scenes[Number(el('sceneSelect').value)];
function images(id,frame){const box=el(id);box.replaceChildren();for(const image of frame.images||[]){const fig=document.createElement('figure'),img=document.createElement('img'),caption=document.createElement('figcaption');img.src=image.data_url;img.alt=image.label||'Observed RGB frame';caption.textContent=(frame.label?frame.label+' · ':'')+(image.label||'Observed RGB frame');fig.append(img,caption);box.append(fig)}}
function cloud(frame){const p=frame.points||[];return{type:'scatter3d',mode:'markers',x:p.map(q=>q[0]),y:p.map(q=>q[1]),z:p.map(q=>q[2]),name:'Observed cloud',showlegend:false,marker:{size:2,opacity:.48,color:p.map(q=>`rgb(${q.slice(3,6).map(v=>Math.round(v<=1?v*255:v)).join(',')})`)},hoverinfo:'skip'}}
function pathTrace(path,label,color,dash){return{type:'scatter3d',mode:'lines+markers',x:path.map(p=>p[0]),y:path.map(p=>p[1]),z:path.map(p=>p[2]),name:label,line:{color,width:6,dash:dash||'solid'},marker:{color,size:3},text:path.map((_,i)=>`Target ${i+1}`),hovertemplate:'%{text}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f}) m<extra>%{fullData.name}</extra>'}}
const layout=id=>({margin:{l:0,r:0,t:10,b:0},paper_bgcolor:'white',uirevision:id,legend:{orientation:'h',x:0,y:1.02},scene:{aspectmode:'data',xaxis:{title:'X (m)'},yaxis:{title:'Y (m)'},zaxis:{title:'Z (m)'},camera:{eye:{x:1.45,y:-1.7,z:1.3}}}});
function renderPaths(){const s=scene(),draw=Number(el('draw').value),traces=[cloud(s.current)],choice=el('comparison').value,pair=choice==='all'?null:(s.comparisons||[])[Number(choice)];s.variants.forEach((v,i)=>{if(!pair||[pair.a,pair.b].includes(v.name))traces.push(pathTrace(v.tcp_paths[draw],pretty(v.name),colors[i%colors.length],pair&&v.name===pair.a?'dash':'solid'))});if(s.expert_tcp){let p=s.expert_tcp;if(s.expert_valid)p=p.filter((_,i)=>{const valid=s.expert_valid[i];return Array.isArray(valid)?valid.slice(0,7).every(Boolean):valid});traces.push(pathTrace(p,data.protocol.expert_label||'Expert targets','#111827','dot'))}if(s.current.tcp){const p=s.current.tcp;traces.push({type:'scatter3d',mode:'markers',x:[p[0]],y:[p[1]],z:[p[2]],name:'Observed TCP',marker:{size:7,color:'#111827',symbol:'diamond'}})}for(const [i,region] of (s.target_regions||[]).entries()){const p=region.center;traces.push({type:'scatter3d',mode:'markers',x:[p[0]],y:[p[1]],z:[p[2]],name:region.label,marker:{size:8,symbol:'cross',color:i===0?'#2248c7':'#b91c1c'},hovertemplate:'%{fullData.name}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f}) m<extra></extra>'})}el('regionNote').textContent=(s.target_regions||[]).length?'Crosses mark diagnostic locations supplied from the observed cue; they are annotations, not extra policy inputs.':'';Plotly.react('paths',traces,layout(s.id),{responsive:true,displaylogo:false});images('currentImages',s.current);el('delta').style.display=pair?'block':'none';if(pair){const a=s.variants.find(v=>v.name===pair.a),b=s.variants.find(v=>v.name===pair.b),distance=a.tcp_paths[draw].map((p,i)=>1000*Math.hypot(...p.map((v,j)=>v-b.tcp_paths[draw][i][j])));el('pairNote').textContent=`This draw: mean TCP-target separation ${(distance.reduce((a,b)=>a+b,0)/distance.length).toFixed(2)} mm; final-target separation ${distance.at(-1).toFixed(2)} mm. No trajectory offset or visual exaggeration is applied.`;Plotly.react('delta',[{x:distance.map((_,i)=>i+1),y:distance,type:'scatter',mode:'lines+markers',line:{color:'#3155aa'},hovertemplate:'Target %{x}: %{y:.3f} mm<extra></extra>'}],{margin:{l:65,r:20,t:10,b:48},xaxis:{title:'Predicted target index',dtick:2},yaxis:{title:'TCP separation (mm)',rangemode:'tozero'},paper_bgcolor:'white'},{responsive:true,displaylogo:false})}else el('pairNote').textContent='All variants are shown. Select a paired comparison to inspect small differences without overlapping all paths.'}
function renderHistory(){const s=scene(),frame=(s.cue_frames||[])[Number(el('frame').value)],variant=s.variants[Number(el('variant').value)];if(!frame){Plotly.react('history',[],layout(s.id+'history'),{responsive:true,displaylogo:false});el('historyNote').textContent='No representative historical frames supplied.';el('historyImages').replaceChildren();el('pairedHistoryImages').replaceChildren();el('pairedImageNote').textContent='';return}const nodes=(variant.selected_nodes||[]).filter(n=>{if(n.source!=null&&frame.source!=null&&n.source!==frame.source)return false;const lo=n.time_min??n.time_anchor,hi=n.time_max??n.time_anchor;return frame.source_index==null||lo==null||(lo<=frame.source_index&&hi>=frame.source_index)});const traces=[cloud(frame)];if(nodes.length)traces.push({type:'scatter3d',mode:'markers',x:nodes.map(n=>n.anchor[0]),y:nodes.map(n=>n.anchor[1]),z:nodes.map(n=>n.anchor[2]),name:'Selected centers spanning this frame',marker:{size:6,color:colors[Number(el('variant').value)%colors.length],symbol:'diamond'},text:nodes.map(n=>`Leaves: ${n.count??'?'}<br>Time: ${n.time_min??n.time_anchor??'?'}–${n.time_max??n.time_anchor??'?'}`),hovertemplate:'%{text}<extra></extra>'});Plotly.react('history',traces,layout(s.id+'history'),{responsive:true,displaylogo:false});el('historyNote').textContent=((variant.selected_nodes||[]).length?`${pretty(variant.name)} · ${nodes.length} selected nodes have observed-time support spanning this frame.`:(variant.tokens>0?`${pretty(variant.name)} uses ${variant.tokens} historical tokens; their node geometry is not included in this view.`:`${pretty(variant.name)} has no historical tokens.`))+` ${frame.label?frame.label+' · ':''}${frame.description||''}`;images('historyImages',frame);const paired=(s.cue_frames||[]).find(f=>f!==frame&&f.label&&frame.label&&f.label!==frame.label&&f.source===frame.source&&f.source_index===frame.source_index);el('pairedImageNote').textContent=paired?'Same historical frame under the paired input: '+paired.label:'';images('pairedHistoryImages',paired||{})}
function table(id,headers,rows){const t=el(id);t.replaceChildren();const head=document.createElement('tr');for(const name of headers){const th=document.createElement('th');th.textContent=name;head.append(th)}t.append(head);for(const values of rows){const row=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=String(value??'—');row.append(td)}t.append(row)}}
function valueText(value){if(Array.isArray(value)){const a=value.flat(Infinity).filter(x=>typeof x==='number');if(a.length)return `${(a.reduce((a,b)=>a+b,0)/a.length).toPrecision(4)} [${Math.min(...a).toPrecision(3)}, ${Math.max(...a).toPrecision(3)}]`;return JSON.stringify(value)}if(typeof value==='number')return value.toPrecision(4);if(value&&typeof value==='object')return JSON.stringify(value);return value??'—'}
function renderScene(){const s=scene();el('sceneTitle').textContent=`${s.id} · decision ${s.decision_index}`;el('goal').textContent=s.goal||'';el('sceneNote').textContent=s.description||s.note||'';el('frame').replaceChildren();(s.cue_frames||[]).forEach((f,i)=>opt(el('frame'),i,`${f.label?f.label+' · ':''}${f.source||'history'} frame ${f.source_index??i}${f.description?' · '+f.description:''}`));el('variant').replaceChildren();s.variants.forEach((v,i)=>opt(el('variant'),i,pretty(v.name)));const target=s.variants.findIndex(v=>v.name==='cue_target');if(target>=0)el('variant').value=target;const variantFields=[['tokens','History tokens'],['preferred_leaf_count','Preferred leaves'],['exposed_preferred_leaves','Preferred exposed individually'],['exposed_target_cue_leaves','Target-cue leaves exposed individually']].filter(([key])=>s.variants.some(v=>v[key]!=null));table('variantTable',['Variant',...variantFields.map(x=>x[1]),'What changes'],s.variants.map(v=>[pretty(v.name),...variantFields.map(([key])=>v[key]),v.description]));const comparisons=s.comparisons||[],fields=[...new Set(comparisons.flatMap(c=>Object.keys(c).filter(k=>!['a','b'].includes(k))))];table('comparisonTable',['Pair',...fields.map(pretty)],comparisons.map(c=>[`${pretty(c.a)} → ${pretty(c.b)}`,...fields.map(k=>valueText(c[k]))]));el('comparison').replaceChildren();comparisons.forEach((c,i)=>opt(el('comparison'),i,`${pretty(c.a)} → ${pretty(c.b)}`));opt(el('comparison'),'all','All variants');const preferred=data.protocol.default_comparison||{a:'cue_coarse',b:'cue_target'};const primary=comparisons.findIndex(c=>c.a===preferred.a&&c.b===preferred.b);if(primary>=0)el('comparison').value=primary;else if(!comparisons.length)el('comparison').value='all';renderPaths();renderHistory()}
el('sceneSelect').onchange=renderScene;el('draw').onchange=renderPaths;el('comparison').onchange=renderPaths;el('frame').onchange=renderHistory;el('variant').onchange=renderHistory;
for(const note of data.protocol.notes||[]){const li=document.createElement('li');li.textContent=note;el('notes').append(li)}
el('protocol').textContent=JSON.stringify(data.protocol,null,2);el('summary').textContent=JSON.stringify(data.summary||{},null,2);renderScene();
"""
