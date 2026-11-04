"""Standalone viewer for recorded button-hover diagnostics, never task scoring.

Only saved predictions and observations are rendered. Predicted joint-target FK
is distinguished from actual ensuing motion and from physical button contact.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _encoded(value):
    result = json.dumps(value, default=_plain, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for char, replacement in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                              ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        result = result.replace(char, replacement)
    return result


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")


def validate_result(result):
    """Check action/trajectory consistency without inventing a press criterion."""
    if not isinstance(result.get("protocol"), dict) or not result.get("trials"):
        raise ValueError("Button report requires a protocol and recorded trials")
    for trial in result["trials"]:
        decisions = trial.get("decisions", [])
        if not decisions:
            # Infrastructure errors must remain visible rather than disappearing.
            continue
        steps = [row["native_step"] for row in decisions]
        if any(b <= a for a, b in zip(steps, steps[1:])):
            raise ValueError("Decision native steps must strictly increase")
        execute = trial.get("execute_actions", result["protocol"].get("execute_actions", 8))
        for row in decisions:
            targets = row["target_tcp_xyz"]
            horizon = len(targets)
            if not 1 <= execute <= horizon:
                raise ValueError("Executed prefix exceeds predicted horizon")
            _array(targets, (horizon, 3), "target_tcp_xyz")
            _array(row["physical_actions"], (horizon, 8), "physical_actions")
            for field in ("tcp_xyz", "button_xyz"):
                _array(row["diagnostics"][field], (3,), f"diagnostics.{field}")
            points = np.asarray(row.get("points", []), dtype=float)
            if points.size and (points.ndim != 2 or points.shape[1] != 6 or not np.isfinite(points).all()):
                raise ValueError("Current scene points must be finite [N,6] XYZRGB")
        timeline = trial.get("timeline", [])
        if any(b["native_step"] <= a["native_step"] for a, b in zip(timeline, timeline[1:])):
            raise ValueError("Timeline native steps must strictly increase")
        for row in timeline:
            for field in ("tcp_xyz", "button_xyz"):
                if row.get(field) is not None:
                    _array(row[field], (3,), f"timeline.{field}")


def write_report(result, output: Path) -> Path:
    """Write an offline HTML with the exact recorded measurements embedded."""
    from plotly.offline import get_plotlyjs

    validate_result(result)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(result)
    for trial in payload["trials"]:
        # The canonical result JSON retains all raw records. The interactive
        # display needs neither per-leaf timing lists nor duplicate raw actions.
        for row in trial.get("decisions", []):
            row.pop("timing", None)
            row.pop("raw_actions", None)
            if len(row.get("points", [])):
                row["points"] = np.asarray(row["points"], dtype=float).round(5).tolist()
        for record in [trial, *trial.get("branches", [])]:
            for key in ("video", "result_path"):
                value = record.get(key)
                if value and Path(value).is_absolute():
                    record[key] = os.path.relpath(value, output)
    page = _PAGE.replace("__PLOTLY__", get_plotlyjs().replace("</script", "<\\/script"))
    page = page.replace("__DATA__", _encoded(payload))
    path = output / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Button approach and discarded action targets · Chronotree</title>
<style>
:root{--ink:#1b3042;--muted:#61717e;--line:#d8e1e7;--blue:#136d9b}*{box-sizing:border-box}
body{margin:0;background:#f2f5f7;color:var(--ink);font:15px/1.5 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:28px 22px 60px}
h1{font-size:clamp(26px,4vw,40px);line-height:1.15;margin:8px 0 12px}h2{font-size:20px;margin:0 0 10px}p{margin:9px 0}.eyebrow{font-size:12px;letter-spacing:.12em;color:var(--blue);font-weight:700}
.card{background:white;border:1px solid var(--line);border-radius:12px;padding:20px;margin:18px 0}.muted{color:var(--muted)}
.controls{display:grid;grid-template-columns:2fr 2fr 1fr;gap:14px}label{font-size:12px;font-weight:650;color:var(--muted)}select{display:block;width:100%;padding:8px;margin-top:5px;border:1px solid #b3c4ce;border-radius:6px;background:white;color:var(--ink)}
.slider{display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center;margin:18px 0}input[type=range]{width:100%}button{font:inherit;background:#f5f9fb;border:1px solid #b3c4ce;border-radius:5px;padding:6px 10px;cursor:pointer}
#status{font-weight:650;min-height:25px}#plot{height:650px}.plots{display:grid;grid-template-columns:1fr 1fr;gap:12px}.traceplot{height:320px}.facts{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}.fact{background:#f1f6f9;border-radius:7px;padding:8px 12px;font-size:13px}
video{width:100%;max-width:1000px;background:#101a22}a{color:var(--blue)}summary{cursor:pointer;font-weight:650}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:450px;overflow:auto}#error{background:#fff0ed;color:#922f25;padding:12px}.branch-videos{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px;margin-top:18px}.branch-videos article{padding:12px;border:1px solid var(--line);border-radius:8px}.branch-videos h3{font-size:15px;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{padding:9px;text-align:left;border-bottom:1px solid var(--line)}.table-scroll{overflow:auto}
@media(max-width:780px){main{padding:18px 10px}.card{padding:13px}.controls,.plots{grid-template-columns:1fr}#plot{height:500px}}
</style><script>__PLOTLY__</script></head><body><main>
<header><div class="eyebrow">CHRONOTREE · REAL SIMULATOR DIAGNOSTIC</div>
<h1>What does the policy predict above the button?</h1>
<p>Compare the whole generated chunk with the prefix sent to the controller and the motion that actually followed.</p>
<p class="muted">This is a targeted diagnostic, not a benchmark success rate. Colored target paths are forward kinematics of clipped absolute joint targets, not executed robot motion. A target near the button does not establish contact or a successful press.</p></header>
<section class="card"><div class="controls"><label>Model and reader<select id="model"></select></label><label>Scene and rollout<select id="trial"></select></label><label>View<select id="view"><option value="action">Button and action area</option><option value="full">Full current scene</option></select></label><label>Prediction at this observed state<select id="prediction"></select></label></div>
<div class="slider"><button id="previous" aria-label="Previous decision">←</button><input type="range" id="decision" min="0" max="0" value="0" step="1" aria-label="Decision"><button id="next" aria-label="Next decision">→</button></div>
<div id="status" aria-live="polite"></div><div class="facts" id="facts"></div><div id="error" hidden role="alert"></div><div id="plot"></div>
<p class="muted" id="legend-note">Blue: targets in the executed prefix. Orange dashed: later targets discarded at the next replan. Green: actual subsequent TCP motion. Black: current TCP. Red: button cap, used only for measurement. Execution stops immediately on termination.</p>
<p class="muted">Solid target segments connect consecutive target poses for readability; the simulator may follow a different path. Additional noise/model probes were never executed and share the baseline rollout's observed history. Display cloud coordinates are rounded to 0.01 mm; the saved result JSON retains original precision. Target trajectories and measurement values remain unrounded.</p>
<div class="plots"><div class="traceplot" id="distance"></div><div class="traceplot" id="height"></div></div>
<details><summary>Exact decision measurements</summary><pre id="measurements"></pre></details></section>
<section class="card"><h2>Recorded rollout</h2><p id="video-caption" class="muted"></p><video id="video" controls preload="metadata"></video><p><button id="seek">Jump to selected decision</button> <a id="video-link">Open MP4</a> <a id="result-link" hidden>Recorded JSON</a></p><p class="muted">The video shows actual simulation, including any initial conditioning frames. Diagnostic labels and button positions are never added to policy inputs.</p></section>
<section class="card" id="branch-section" hidden><h2>Execute the saved tail from the same simulator state</h2><p>These short branches replay the recorded commands to the selected state, verify simulator-state agreement, then execute saved predictions. The replan-8 and execute-16 branches share the initial eight targets. The teacher branch uses a separately generated teacher chunk from the same observed history.</p><p class="muted">A successful branch is evidence about this particular state, not a success-rate estimate. No button controller or hidden task label generates its commands. Geometric endpoint measurements may include the wrapper's automatic terminal step; success comes from the simulator's official status.</p><div class="table-scroll"><table><thead><tr><th>Native step</th><th>Branch</th><th>Outcome</th><th>Actual descent (mm)</th><th>Button progress (mm)</th><th>Cap depth (mm)</th><th>Replay state error</th></tr></thead><tbody id="branch-summary"></tbody></table></div><div class="branch-videos" id="branch-videos"></div></section>
<section class="card"><h2>Recorded trials</h2><div class="table-scroll"><table><thead><tr><th>Model</th><th>Task / episode</th><th>Checkpoint</th><th>Reader</th><th>Execute</th><th>Status</th><th>Decisions</th></tr></thead><tbody id="summary"></tbody></table></div>
<details><summary>Protocol and limitations</summary><pre id="protocol"></pre></details><details id="extras" hidden><summary>Additional probes or branch results</summary><pre id="extras-data"></pre></details></section>
</main><script type="application/json" id="payload">__DATA__</script><script>
'use strict';
const DATA=JSON.parse(document.getElementById('payload').textContent),el=id=>document.getElementById(id);
const modelKey=t=>JSON.stringify([t.label,t.method,t.checkpoint_step]);
const execute=t=>t.execute_actions??DATA.protocol.execute_actions??8;
const stride=t=>t.action_stride??DATA.protocol.action_stride??2;
const hz=()=>DATA.protocol.native_control_hz??DATA.protocol.native_hz??20;
const defaultDecision=t=>Math.max(0,t.decisions.findIndex(d=>d.probe_reason==='hover'));
const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
function choices(id,values,format){const select=el(id),prior=select.value;select.replaceChildren();for(const v of values){const o=document.createElement('option');o.value=String(v);o.textContent=format(v);select.appendChild(o);}if(values.map(String).includes(prior))select.value=prior;}
function line(points,name,color,dash='solid',indices=null){return {type:'scatter3d',mode:'lines+markers',name,x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),customdata:indices??points.map((_,i)=>i),line:{color,width:dash==='solid'?6:4,dash},marker:{color,size:4},hovertemplate:name+'<br>index %{customdata}<br>%{x:.3f}, %{y:.3f}, %{z:.3f} m<extra></extra>'};}
function marker(xyz,name,color,symbol){return {type:'scatter3d',mode:'markers',name,x:[xyz[0]],y:[xyz[1]],z:[xyz[2]],marker:{color,size:8,symbol},hovertemplate:name+'<br>%{x:.3f}, %{y:.3f}, %{z:.3f} m<extra></extra>'};}
function showFacts(values){el('facts').replaceChildren();for(const value of values){const span=document.createElement('span');span.className='fact';span.textContent=value;el('facts').appendChild(span);}}
function showBranches(t){
 const rows=t.branches??[];el('branch-section').hidden=!rows.length;el('branch-summary').replaceChildren();el('branch-videos').replaceChildren();
 const fixed=x=>typeof x==='number'?x.toFixed(2):'—';
 for(const b of rows){const tr=document.createElement('tr');for(const value of [b.native_step,b.mode,b.status,fixed(b.actual_descent_mm),fixed(b.actual_button_progress_mm),fixed(b.maximum_button_depth_mm),b.replay_simulator_max_error??'unverified']){const td=document.createElement('td');td.textContent=String(value);tr.appendChild(td);}el('branch-summary').appendChild(tr);
 if(b.video){const article=document.createElement('article'),title=document.createElement('h3'),video=document.createElement('video'),link=document.createElement('a'),seek=document.createElement('button');title.textContent=`Step ${b.native_step} · ${b.mode} · ${b.status}`;video.controls=true;video.preload='metadata';video.src=b.video;link.href=b.video;link.textContent='Open MP4';seek.textContent='Jump to branch';seek.addEventListener('click',()=>{video.currentTime=(b.prefix_video_frames??0)/hz();});article.append(title,video,seek,document.createTextNode(' '),link);el('branch-videos').appendChild(article);}}
}
function plotTimeline(t,d){
 const rows=(t.timeline??[]).filter(r=>r.tcp_xyz&&r.button_xyz&&!r.terminal),x=rows.map(r=>r.native_step/hz());
 const selected=d.native_step/hz(),shape={type:'line',x0:selected,x1:selected,y0:0,y1:1,yref:'paper',line:{color:'#8354b5',width:2,dash:'dot'}};
 const layout=title=>({title:{text:title,font:{size:15}},margin:{l:58,r:15,t:42,b:45},xaxis:{title:'Execution time (s)'},yaxis:{title:'mm'},shapes:[shape],legend:{orientation:'h',y:-.25,font:{size:11}},uirevision:el('trial').value,hovermode:'x unified'});
 const dist={x,y:rows.map(r=>1000*distance(r.tcp_xyz,r.button_xyz)),mode:'lines',name:'Actual TCP → button',line:{color:'#298666'}};
 const xy={x,y:rows.map(r=>1000*Math.hypot(r.tcp_xyz[0]-r.button_xyz[0],r.tcp_xyz[1]-r.button_xyz[1])),mode:'lines',name:'Horizontal error',line:{color:'#2877a5'}};
 Plotly.react('distance',[dist,xy],layout('Actual button distance'),{responsive:true,displaylogo:false});
 const z={x,y:rows.map(r=>1000*r.tcp_xyz[2]),mode:'lines',name:'Actual TCP height',line:{color:'#298666'}};
 const button={x,y:rows.map(r=>1000*r.button_xyz[2]),mode:'lines',name:'Button-cap height',line:{color:'#b84438'}};
 const targets={x:d.target_tcp_xyz.map((_,i)=>(d.native_step+(i+1)*stride(t))/hz()),y:d.target_tcp_xyz.map(p=>1000*p[2]),mode:'lines+markers',name:'Selected chunk target heights',line:{color:'#e4922b',dash:'dot'}};
 Plotly.react('height',[z,button,targets],layout('Actual and predicted world height'),{responsive:true,displaylogo:false});
}
function refreshTrial(){
 const indices=DATA.trials.map((t,i)=>[t,i]).filter(([t])=>modelKey(t)===el('model').value).map(([,i])=>i);
 choices('trial',indices,i=>{const t=DATA.trials[i];return `${t.task} / ${t.episode_idx} · execute ${execute(t)} · ${t.variant??t.status}`;});
 const t=DATA.trials[Number(el('trial').value)];el('decision').max=Math.max(0,t.decisions.length-1);el('decision').value=defaultDecision(t);refresh();
}
function refresh(){try{
 el('error').hidden=true;
 const t=DATA.trials[Number(el('trial').value)],index=Number(el('decision').value),d=t.decisions[index];
 if(el('branch-section').dataset.trial!==el('trial').value){showBranches(t);el('branch-section').dataset.trial=el('trial').value;}
 const video=el('video');if(t.video){if(video.getAttribute('src')!==t.video)video.src=t.video;video.hidden=false;el('video-link').href=t.video;el('video-link').hidden=false;el('seek').disabled=!d;}
 else{video.removeAttribute('src');video.hidden=true;el('video-link').hidden=true;el('seek').disabled=true;}
 el('video-caption').textContent=`${t.label} · ${t.task} episode ${t.episode_idx} · ${t.status}`;
 el('result-link').hidden=!t.result_path;if(t.result_path)el('result-link').href=t.result_path;
 const extras={};for(const key of ['probes','branches','summary','error'])if(t[key]!==undefined)extras[key]=t[key];
 el('extras').hidden=Object.keys(extras).length===0;el('extras-data').textContent=JSON.stringify(extras,null,2);
 if(!d){el('status').textContent='No policy decisions recorded for this trial.';for(const id of ['plot','distance','height'])Plotly.purge(id);showFacts([]);el('measurements').textContent=JSON.stringify(t,null,2);return;}
 choices('prediction',[-1,...(d.probes??[]).map((_,i)=>i)],i=>i<0?'Production draw · actually used':`${d.probes[i].label} / ${d.probes[i].method??''} · noise ${d.probes[i].noise_seed} · never executed`);
 const probeIndex=Number(el('prediction').value),probe=probeIndex<0?null:d.probes[probeIndex];
 const n=execute(t),diag=d.diagnostics,targets=(probe??d).target_tcp_xyz,next=t.decisions[index+1];
 const end=Math.min(next?.native_step??Infinity,d.native_step+n*stride(t));
 const actual=(t.timeline??[]).filter(r=>r.native_step>d.native_step&&r.native_step<=end&&r.tcp_xyz&&!r.terminal);
 const points=d.points??[],traces=[];
 if(points.length)traces.push({type:'scatter3d',mode:'markers',name:'Current RGB-D cloud',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),marker:{size:2,opacity:.55,color:points.map(p=>'rgb('+p.slice(3).map(c=>Math.round(Math.max(0,Math.min(1,c))*255)).join(',')+')')},hoverinfo:'skip'});
 traces.push(marker(diag.tcp_xyz,'Current TCP','#1b3042','diamond'),marker(diag.button_xyz,'Button cap (label)','#b84438','cross'));
 traces.push(line([diag.tcp_xyz,...targets.slice(0,n)],`Targets 1–${n}: ${probe?'counterfactual prefix':'executed prefix'}`,'#147ca8','solid'));
 if(n<targets.length)traces.push(line(targets.slice(n-1),`Targets ${n+1}–${targets.length}: ${probe?'counterfactual tail':'discarded tail'}`,'#e4932c','dash',targets.slice(n-1).map((_,i)=>n+i)));
 if(actual.length)traces.push(line([diag.tcp_xyz,...actual.map(r=>r.tcp_xyz)],'Actual ensuing TCP motion','#279769','solid',[d.native_step,...actual.map(r=>r.native_step)]));
 const currentDistance=distance(diag.tcp_xyz,diag.button_xyz)*1000,prefixDistance=distance(targets[n-1],diag.button_xyz)*1000,lastDistance=distance(targets.at(-1),diag.button_xyz)*1000;
 showFacts([`Current distance: ${currentDistance.toFixed(1)} mm`,`Target ${n}: ${prefixDistance.toFixed(1)} mm`,`Target ${targets.length}: ${lastDistance.toFixed(1)} mm`,`Cap depression: ${(1000*(diag.button_depth_m??0)).toFixed(2)} mm`]);
 el('status').textContent=`Decision ${index+1}/${t.decisions.length} · native step ${d.native_step} (${(d.native_step/hz()).toFixed(2)} s) · ${diag.task_name??'phase unavailable'}`;
 el('legend-note').textContent=probe?'Counterfactual prediction: blue and orange targets were never executed. Green is the original policy’s actual motion from this same observed state. Black is the current TCP; red is the measured button cap.':'Production prediction: blue is the prefix issued before the next replan; orange dashed targets are discarded. Green is actual ensuing motion. Execution stops immediately on termination.';
 const axes=['X','Y','Z'].map(a=>({title:a+' (m)',gridcolor:'#dce4e9',backgroundcolor:'#f8fafc'}));
 if(el('view').value==='action'){const focus=[diag.tcp_xyz,diag.button_xyz,...targets,...actual.map(r=>r.tcp_xyz)];for(let j=0;j<3;j++){const v=focus.map(p=>p[j]),lo=Math.min(...v),hi=Math.max(...v),pad=Math.max(.025,.15*(hi-lo));axes[j].range=[lo-pad,hi+pad];}}
 const spans=axes.map(a=>a.range?a.range[1]-a.range[0]:1),scale=Math.cbrt(spans.reduce((a,b)=>a*b,1));
 const aspect=el('view').value==='action'?{aspectmode:'manual',aspectratio:Object.fromEntries(['x','y','z'].map((a,i)=>[a,spans[i]/scale]))}:{aspectmode:'data'};
 Plotly.react('plot',traces,{margin:{l:0,r:0,t:10,b:0},scene:{...aspect,xaxis:axes[0],yaxis:axes[1],zaxis:axes[2],camera:{eye:{x:1.4,y:-1.8,z:1.1}}},uirevision:'button-'+el('view').value,legend:{orientation:'h',y:-.03,font:{size:11}}},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:'button_chunk',scale:2}});
 plotTimeline(t,{...d,target_tcp_xyz:targets});const {points:ignored,...measurements}=d;el('measurements').textContent=JSON.stringify(probe?{selected_probe:probe,baseline_decision:measurements}:measurements,null,2);
 }catch(error){el('error').hidden=false;el('error').textContent=String(error);}}
choices('model',[...new Set(DATA.trials.map(modelKey))],key=>{const [label,method,step]=JSON.parse(key);return `${label} / ${method} · ${Number(step).toLocaleString()} steps`;});
el('model').addEventListener('change',refreshTrial);el('trial').addEventListener('change',()=>{const t=DATA.trials[Number(el('trial').value)];el('decision').max=Math.max(0,t.decisions.length-1);el('decision').value=defaultDecision(t);refresh();});
el('view').addEventListener('change',refresh);el('decision').addEventListener('input',refresh);el('prediction').addEventListener('change',refresh);
for(const [id,offset] of [['previous',-1],['next',1]])el(id).addEventListener('click',()=>{el('decision').value=Math.min(Number(el('decision').max),Math.max(0,Number(el('decision').value)+offset));refresh();});
el('seek').addEventListener('click',()=>{const t=DATA.trials[Number(el('trial').value)],d=t.decisions[Number(el('decision').value)];if(d)el('video').currentTime=(d.native_step+(t.video_start_frames??t.conditioning_video_frames??0))/hz();});
el('protocol').textContent=JSON.stringify(DATA.protocol,null,2);
for(const t of DATA.trials){const tr=document.createElement('tr');for(const value of [t.label,`${t.task} / ${t.episode_idx}`,t.checkpoint_step,t.method,execute(t),t.status,t.decisions?.length??0]){const td=document.createElement('td');td.textContent=String(value);tr.appendChild(td);}el('summary').appendChild(tr);}
refreshTrial();
</script></body></html>'''
