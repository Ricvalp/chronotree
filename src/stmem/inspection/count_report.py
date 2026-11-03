"""Offline HTML for the expert-count intervention diagnostic, never task scoring.

The viewer consumes already generated predictions. It does not modify inputs,
classify button presses from FK, or execute any predicted action.
"""

from __future__ import annotations

from collections import defaultdict
from html import escape
import json
import math
import os
from pathlib import Path

import numpy as np


MODES = {"natural", "fixed_current", "fixed_duration"}
METRICS = (
    ("button_preference_first8_mm", "Button vs cube progress, mm"),
    ("button_distance_end_mm", "Final target → button, mm"),
    ("button_distance_min_first8_mm", "Closest first 8 targets → button, mm"),
    ("tcp_displacement_first8_mm", "TCP displacement at target 8, mm"),
)


def _encoded(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, replacement in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                                   ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(character, replacement)
    return text


def _array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def validate_result(result):
    """Reject broken comparisons rather than drawing plausible but wrong paths."""
    protocol = result["protocol"]
    if protocol["task"] != "PickXtimes":
        raise ValueError("This count diagnostic is defined for PickXtimes")
    if protocol["execute_actions"] != 8 or protocol["action_stride"] != 2:
        raise ValueError("This report expects eight executed targets at action stride two")
    if not protocol.get("scope") or not protocol.get("noise_seeds"):
        raise ValueError("A diagnostic scope and declared noise seeds are required")
    scenes = result["current_scenes"]
    if not scenes or not result["predictions"]:
        raise ValueError("Count report requires current scenes and predictions")
    for name, scene in scenes.items():
        points = np.asarray(scene["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 6 or not len(points):
            raise ValueError(f"{name}: points must be nonempty [N,6] XYZRGB")
        if not np.isfinite(points).all():
            raise ValueError(f"{name}: points must be finite")
        valid = scene.get("valid", [True] * len(points))
        if len(valid) != len(points) or any(not isinstance(v, bool) for v in valid) or not any(valid):
            raise ValueError(f"{name}: valid must be a nonempty boolean mask matching points")
        for field in ("tcp", "button", "cube"):
            _array(scene[field], (3,), f"{name}.{field}")
    identities = set()
    horizons = set()
    for row in result["predictions"]:
        if row["scene_id"] not in scenes or row["history_mode"] not in MODES:
            raise ValueError("Unknown prediction scene or history mode")
        if row["noise_seed"] not in protocol["noise_seeds"]:
            raise ValueError("Prediction uses an undeclared noise seed")
        if any(isinstance(row[k], bool) or not isinstance(row[k], int) for k in ("k", "x")):
            raise ValueError("History and requested counts must be integers")
        if not 0 <= row["k"] <= row["x"]:
            raise ValueError("Count comparisons require 0 <= completed k <= requested X")
        expected = "press_button" if row["k"] == row["x"] else "pick_cube"
        if row["expected_next"] != expected:
            raise ValueError("Expected next phase disagrees with k and X")
        identity = tuple(row[k] for k in (
            "label", "method", "checkpoint_step", "history_mode", "scene_id", "k", "x", "noise_seed",
        ))
        if identity in identities:
            raise ValueError("Duplicate model/context/count/noise prediction")
        identities.add(identity)
        horizon = len(row["physical_actions"])
        if horizon < protocol["execute_actions"]:
            raise ValueError("Prediction horizon is shorter than the executed prefix")
        _array(row["physical_actions"], (horizon, 8), "physical_actions")
        _array(row["tcp_positions_m"], (horizon, 3), "tcp_positions_m")
        horizons.add(horizon)
        for key, value in row.get("metrics", {}).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"Nonfinite metric: {key}")
    if len(horizons) != 1:
        raise ValueError("All compared action chunks must have the same horizon")
    _encoded(result)
    return horizons.pop()


def aggregate_rows(result):
    """Report means over available scenes/noise seeds, with denominators visible."""
    groups = defaultdict(list)
    for row in result["predictions"]:
        key = tuple(row[k] for k in ("label", "method", "checkpoint_step", "history_mode", "k", "x"))
        groups[key].append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        record = dict(zip(("label", "method", "checkpoint_step", "history_mode", "k", "x"), key))
        record.update(
            samples=len(rows), scenes=len({r["scene_id"] for r in rows}),
            noise_seeds=len({r["noise_seed"] for r in rows}), expected_next=rows[0]["expected_next"],
        )
        for metric, _ in METRICS:
            values = [r.get("metrics", {}).get(metric) for r in rows]
            # Never present a partial subset as the mean of the complete group.
            record[metric] = None if any(v is None for v in values) else float(np.mean(values))
        summaries.append(record)
    return summaries


def _table(result):
    headers = ["Model / reader", "Step", "History", "k done", "X requested", "Expert next phase",
               "Predictions / scenes / seeds", *(label for _, label in METRICS)]
    rows = []
    for row in aggregate_rows(result):
        values = [f"{row['label']} / {row['method']}", f"{row['checkpoint_step']:,}", row["history_mode"],
                  row["k"], row["x"], row["expected_next"].replace("_", " "),
                  f"{row['samples']} / {row['scenes']} / {row['noise_seeds']}"]
        values += ["—" if row[key] is None else f"{row[key]:.2f}" for key, _ in METRICS]
        rows.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>")
    return ('<div class="table-scroll"><table><thead><tr>'
            + "".join(f"<th>{escape(h)}</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")


def _video(source, output):
    path = source.get("video")
    if not path:
        return ""
    path = Path(path)
    if path.is_absolute():
        path = Path(os.path.relpath(path, output))
    href = escape(path.as_posix(), quote=True)
    return (f'<details><summary>Expert source video</summary><video controls preload="metadata" '
            f'src="{href}"></video><p><a href="{href}">Open video</a></p></details>')


def write_report(result: dict, output: Path) -> Path:
    """Write a standalone interactive point-cloud viewer and numerical table."""
    horizon = validate_result(result)
    from plotly.offline import get_plotlyjs

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = result["protocol"]
    notes = [
        "This is a controlled expert-history diagnostic, not a benchmark success rate.",
        "Curves show forward kinematics of absolute joint targets, not executed robot motion. "
        "Distances to the button do not establish contact or a successful press.",
        "The first eight targets would execute before replanning: 0.8 seconds at 10 Hz. "
        "Later targets are predictions only; each target is held for two native 20 Hz commands.",
        "Changing the goal with history fixed tests the requested-count condition. Comparing histories "
        "at fixed current observation is an artificial intervention; history content and timing controls "
        "must be interpreted using the recorded protocol.",
        "Button/cube markers are diagnostic labels only and are not supplied to the policy.",
        *protocol.get("notes", []),
    ]
    descriptions = "".join(f"<li>{escape(str(note))}</li>" for note in notes)
    plotly = get_plotlyjs().replace("</script", "<\\/script")
    page = _PAGE.replace("__PLOTLY__", plotly).replace("__DATA__", _encoded(result))
    page = page.replace("__TASK__", escape(protocol["task"]))
    page = page.replace("__SCOPE__", escape(protocol["scope"]))
    page = page.replace("__HORIZON__", str(horizon)).replace("__NOTES__", descriptions)
    page = page.replace("__TABLE__", _table(result)).replace("__VIDEO__", _video(result.get("source", {}), output))
    destination = output / "index.html"
    destination.write_text(page, encoding="utf-8")
    return destination


_PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Expert count and next action · __TASK__</title>
<style>
:root{color-scheme:light;--ink:#183142;--muted:#5e7080;--line:#d8e1e7;--blue:#185f8f}
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:1440px;margin:auto;padding:30px 24px 64px}h1{font-size:clamp(26px,4vw,40px);line-height:1.15;margin:8px 0 16px}
h2{font-size:21px;margin:0 0 12px}.eyebrow{font-size:12px;letter-spacing:.13em;color:var(--blue);font-weight:700}
p{margin:10px 0}.muted{color:var(--muted)}.card{background:white;border:1px solid var(--line);border-radius:14px;padding:22px;margin:20px 0}
.controls{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.controls label{font-size:12px;font-weight:650;color:var(--muted)}
select{display:block;width:100%;padding:9px;margin-top:5px;background:white;border:1px solid #b5c4cf;border-radius:7px;color:var(--ink)}
#plot{height:650px;margin-top:14px}#status{padding:10px 0;font-weight:600}#goal-list{padding-left:20px}
.key{display:flex;gap:20px;flex-wrap:wrap;font-size:13px}.solid,.dashed{display:inline-block;width:36px;vertical-align:middle;border-top:3px solid var(--blue);margin-right:6px}.dashed{border-top-style:dashed}
.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{text-align:left;padding:10px;border-bottom:1px solid var(--line);white-space:nowrap}th{background:#f0f5f8}tbody tr:hover{background:#f8fbfd}
video{max-width:900px;width:100%;margin-top:16px}summary{cursor:pointer;font-weight:650}a{color:var(--blue)}
#error{background:#fff0ef;color:#851f1a;padding:14px;white-space:pre-wrap}li{margin:8px 0}
@media(max-width:760px){main{padding:20px 12px}.card{padding:14px}.controls{grid-template-columns:repeat(2,minmax(0,1fr))}#plot{height:520px}}
</style><script>__PLOTLY__</script></head><body><main>
<header><div class="eyebrow">CHRONOTREE · __TASK__ · TEACHER / STUDENT DIAGNOSTIC</div>
<h1>Does the completed count change the next action?</h1><p>Hold the current scene fixed and inspect how the predicted chunk changes with the requested count <b>X</b> and the <b>k</b> completed pick/place cycles in history.</p>
<p class="muted">__SCOPE__</p></header>
<section class="card"><div class="controls">
<label>Model and reader<select id="model"></select></label><label>History construction<select id="mode"></select></label>
<label>Current scene<select id="scene"></select></label><label>Paired noise seed<select id="seed"></select></label>
<label>Compare<select id="compare"><option value="x">Requested X · hold k fixed</option><option value="k">History k · hold X fixed</option></select></label>
<label><span id="fixed-label">Completed cycles k</span><select id="fixed"></select></label>
<label>View<select id="view"><option value="action">Action area</option><option value="full">Full scene</option></select></label></div>
<div id="status" aria-live="polite"></div><div class="key"><span><i class="solid"></i>First 8 targets: would execute</span><span><i class="dashed"></i>Targets 9–__HORIZON__: prediction only</span></div>
<div id="error" role="alert" hidden></div><div id="plot"></div><ul id="goal-list"></ul>
<p class="muted">Rotate by dragging; zoom with the wheel. Double-click a legend item to isolate a path. “Action area” frames the robot TCP, button, cube and compared target paths; “Full scene” shows the entire input cloud. This changes only the view, never the policy input or predictions. Axes use world meters and equal spatial scale. “Expert next phase” is the task rule, not a classification of the predicted motion.</p></section>
<section class="card"><h2>Mean geometric measurements</h2><p>Means across the available paired noise seeds and scenes within each model / history / k / X group. The denominator is shown; different rows can have different scene coverage. “Button vs cube progress” is the reduction in distance to the button minus the reduction in distance to the cube, measured at target 8. Positive values favor geometric progress toward the button; this is not a probability. A smaller button distance alone does not establish a press.</p>__TABLE__</section>
<section class="card"><h2>How to interpret this test</h2><ul>__NOTES__</ul>__VIDEO__</section>
</main><script type="application/json" id="payload">__DATA__</script><script>
'use strict';
const DATA = JSON.parse(document.getElementById('payload').textContent);
const el = id => document.getElementById(id);
const controls = ['model','mode','scene','seed','compare','fixed','view'];
const colors = ['#007a9d','#d05a28','#7c4bb3','#2f8759','#c23c73','#9a791a'];
const modelKey = r => JSON.stringify([r.label,r.method,r.checkpoint_step]);
function options(id, values, format=x=>x){
  const select=el(id), previous=select.value;
  select.replaceChildren();
  for(const value of values){const option=document.createElement('option');option.value=String(value);option.textContent=format(value);select.appendChild(option);}
  if(values.map(String).includes(previous))select.value=previous;
}
function unique(rows,key){return [...new Set(rows.map(r=>r[key]))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b)));}
function marker(name,xyz,color,symbol='circle',size=7,text=name){return {type:'scatter3d',mode:'markers+text',name,text:[text],textposition:'top center',textfont:{color,size:12},x:[xyz[0]],y:[xyz[1]],z:[xyz[2]],marker:{color,size,symbol},hovertemplate:name+'<br>%{x:.3f}, %{y:.3f}, %{z:.3f} m<extra></extra>'};}
function pathTrace(points,name,color,dash,size,group,firstIndex){
  return {type:'scatter3d',mode:'lines+markers',name,legendgroup:group,showlegend:dash==='solid',
    x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),
    customdata:points.map((_,i)=>firstIndex+i),line:{color,width:dash==='solid'?6:3,dash},
    marker:{color,size},hovertemplate:name+'<br>target %{customdata}<br>%{x:.3f}, %{y:.3f}, %{z:.3f} m<extra></extra>'};
}
function refresh(){
 try{
  el('error').hidden=true;
  let rows=DATA.predictions.filter(r=>modelKey(r)===el('model').value);
  options('mode',['natural','fixed_current','fixed_duration'].filter(mode=>rows.some(r=>r.history_mode===mode)),x=>({natural:'Natural expert prefix',fixed_current:'Fixed current observation',fixed_duration:'Fixed current and duration'})[x]);
  rows=rows.filter(r=>r.history_mode===el('mode').value);
  options('scene',unique(rows,'scene_id'));
  rows=rows.filter(r=>r.scene_id===el('scene').value);
  options('seed',unique(rows,'noise_seed'));
  rows=rows.filter(r=>r.noise_seed===Number(el('seed').value));
  const compare=el('compare').value, fixed=compare==='x'?'k':'x';
  el('fixed-label').textContent=compare==='x'?'Completed cycles k':'Requested count X';
  options('fixed',unique(rows,fixed));
  rows=rows.filter(r=>r[fixed]===Number(el('fixed').value)).sort((a,b)=>a[compare]-b[compare]);
  if(!rows.length)throw new Error('No predictions for the selected comparison.');
  const scene=DATA.current_scenes[el('scene').value];
  const points=scene.points.filter((_,i)=>!scene.valid||scene.valid[i]);
  const traces=[{type:'scatter3d',mode:'markers',name:'Current RGB-D point cloud',
    x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),
    marker:{size:2,opacity:.55,color:points.map(p=>'rgb('+p.slice(3).map(c=>Math.round(Math.max(0,Math.min(1,c))*255)).join(',')+')')},hoverinfo:'skip'},
    marker('Current robot TCP',scene.tcp,'#152b3b','diamond',8,'TCP'),
    marker('Button label',scene.button,'#cf353b','cross',8,'Button'),marker('Cube label',scene.cube,'#9a7413','square',7,'Cube')];
  const goals=el('goal-list');goals.replaceChildren();
  for(const [index,row] of rows.entries()){
    const color=colors[index%colors.length],phase=row.expected_next==='press_button'?'press button':'pick cube';
    const name=`k=${row.k}, X=${row.x} · expert next: ${phase}`;
    const first=[scene.tcp,...row.tcp_positions_m.slice(0,DATA.protocol.execute_actions)];
    traces.push(pathTrace(first,name,color,'solid',5,name,0));
    const rest=row.tcp_positions_m.slice(DATA.protocol.execute_actions-1);
    if(rest.length>1)traces.push(pathTrace(rest,name+' · later targets',color,'dash',2.5,name,DATA.protocol.execute_actions));
    const li=document.createElement('li');li.style.color=color;
    li.textContent=`k=${row.k}, X=${row.x}: ${DATA.source?.goals?.[String(row.x)]||'(goal text unavailable)'}`;
    goals.appendChild(li);
  }
  const model=JSON.parse(el('model').value);
  el('status').textContent=`${model[0]} / ${model[1]} · ${Number(model[2]).toLocaleString()} steps · ${rows.length} paired paths · noise ${el('seed').value}`;
  const axis={title:'X (m)',backgroundcolor:'#f8fafc',gridcolor:'#dce4e9',zerolinecolor:'#c4cfd7'};
  const axes=[{...axis},{...axis,title:'Y (m)'},{...axis,title:'Z (m)'}];
  if(el('view').value==='action'){
    const focus=[scene.tcp,scene.button,scene.cube,...rows.flatMap(r=>r.tcp_positions_m)];
    for(let dimension=0;dimension<3;dimension++){
      const values=focus.map(p=>p[dimension]),lo=Math.min(...values),hi=Math.max(...values);
      const padding=Math.max(.025,(hi-lo)*.15);
      axes[dimension].range=[lo-padding,hi+padding];axes[dimension].autorange=false;
    }
  }else{for(const value of axes)value.autorange=true;}
  Plotly.react('plot',traces,{margin:{l:0,r:0,t:15,b:0},paper_bgcolor:'white',font:{family:'system-ui',color:'#183142'},
    uirevision:'count-probe-'+el('view').value,scene:{aspectmode:'data',xaxis:axes[0],yaxis:axes[1],zaxis:axes[2],
    camera:{eye:{x:1.4,y:-1.8,z:1.2}}},legend:{orientation:'h',y:-.03,font:{size:11}}},
    {responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',filename:'count_probe',scale:2}});
 }catch(error){el('error').hidden=false;el('error').textContent=String(error);}
}
options('model',[...new Set(DATA.predictions.map(modelKey))],key=>{const [label,method,step]=JSON.parse(key);return `${label} / ${method} · ${Number(step).toLocaleString()} steps`;});
for(const id of controls)el(id).addEventListener('change',refresh);
refresh();
</script></body></html>'''
