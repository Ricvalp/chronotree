"""Offline report for ordinary VideoUnmaskSwap rollouts and observed choices.

Simulator object identities are visualization-only diagnostics. Official task
outcomes retain the entire planned denominator, including errors and timeouts.
"""

from __future__ import annotations

from collections import Counter
from html import escape
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from stmem.evaluation.metrics import summarize_episodes


def _plan(manifest):
    entries = manifest.get("planned_episodes", manifest.get("all_planned", manifest.get("planned")))
    if not isinstance(entries, list) or not entries:
        raise ValueError("Rollout report requires an explicit nonempty planned episode list")
    plan = []
    for entry in entries:
        if isinstance(entry, dict):
            task, index = entry.get("task", "VideoUnmaskSwap"), entry["episode_idx"]
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            task, index = entry
        else:
            task, index = manifest.get("task", "VideoUnmaskSwap"), entry
        if task != "VideoUnmaskSwap" or not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError("Expected nonnegative VideoUnmaskSwap episode indices")
        plan.append((task, index))
    if len(plan) != len(set(plan)):
        raise ValueError("Planned episode list contains duplicates")
    return plan


def _xyz(value, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite XYZ vector")
    return array


def _layout_ranks(diagnostic):
    """World-coordinate ranks, deliberately not image-left/right claims."""
    bins = (diagnostic.get("reset") or {}).get("bins", [])
    if not bins:
        return {}
    identifiers = [item["index"] for item in bins]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Reset layout contains duplicate bin identities")
    positions = np.stack([_xyz(item["xyz_m"], "Reset bin center") for item in bins])
    # Equal coordinates share a rank; no screen-coordinate interpretation.
    ranks = {}
    for row, identifier in enumerate(identifiers):
        ranks[str(identifier)] = {
            "x_rank": 1 + int((positions[:, 0] < positions[row, 0]).sum()),
            "y_rank": 1 + int((positions[:, 1] < positions[row, 1]).sum()),
            "count": len(bins), "xyz_m": positions[row].tolist(),
        }
    for row in diagnostic.get("timeline", []):
        if "tcp_xyz_m" in row:
            _xyz(row["tcp_xyz_m"], "Observed TCP")
    return ranks


def _video(output, record):
    candidate = record.get("video_path")
    if candidate:
        path = Path(candidate)
        if not path.is_absolute():
            # A relative path can be output-relative or repository-relative.
            resolved = path.resolve()
            path = resolved if resolved.is_relative_to(output) else output / path
    else:
        path = output / "videos" / f"VideoUnmaskSwap_{record['episode_idx']:03d}.mp4"
    path = path.resolve()
    if not path.is_relative_to(output):
        raise ValueError("Video must be inside the report directory so relative links remain portable")
    return quote(path.relative_to(output).as_posix()) if path.is_file() else None


def load_report(output_dir):
    """Read durable results; never silently drop missing or failed episodes."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    plan = _plan(manifest)
    by_key = {}
    for path in sorted((output / "episodes").glob("*.json")):
        record = json.loads(path.read_text())
        key = (record.get("task"), record.get("episode_idx"))
        if key not in plan or key in by_key:
            raise ValueError("Unexpected or duplicate episode result")
        if not isinstance(record.get("status"), str):
            raise ValueError("Episode result lacks an official outcome")
        by_key[key] = record
    summary = summarize_episodes(by_key.values(), plan)
    rows = []
    for task, index in plan:
        record = by_key.get((task, index))
        if record is None:
            rows.append({"task": task, "episode_idx": index, "status": "pending", "pending": True})
            continue
        diagnostic = record.get("choice_diagnostics") or {}
        active_samples = sum("tcp_xyz_m" in sample for sample in diagnostic.get("timeline", []))
        # Preserve all supplied annotations, but avoid duplicating large native
        # commands and full decision-timing records in a visualization payload.
        rows.append({
            "task": task, "episode_idx": index, "status": record["status"], "pending": False,
            "error": record.get("error"), "runtime_seconds": record.get("runtime_seconds"),
            "primitive_commands": record.get("primitive_commands", len(record.get("actions", []))),
            "noise_seed": record.get("noise_seed"), "choice_diagnostics": diagnostic,
            "observer_available": active_samples > 0,
            "layout_ranks": _layout_ranks(diagnostic), "video": _video(output, record),
        })
    choices = [row["choice_diagnostics"].get("summary", {}) for row in rows
               if not row["pending"] and row["observer_available"]]
    choice_summary = {
        "episodes_with_observer_summary": sum(bool(value) for value in choices),
        "first_grasp_observed": sum(bool(value.get("first_grasp_bin_indices")) for value in choices),
        "correct_first_grasp": sum(value.get("first_grasp_correct") is True for value in choices),
        "first_lift_observed": sum(bool(value.get("first_lift_bin_indices")) for value in choices),
        "correct_first_lift": sum(value.get("first_lift_correct") is True for value in choices),
        "no_approach_observed": sum(bool(value) and value.get("first_approach_bin_index") is None for value in choices),
        "first_approach_world_x_ranks": dict(Counter(
            str(row["layout_ranks"][str(index)]["x_rank"])
            for row in rows if not row["pending"] and row["observer_available"]
            for index in [row["choice_diagnostics"].get("summary", {}).get("first_approach_bin_index")]
            if index is not None and str(index) in row["layout_ranks"]
        )),
    }
    result = {"manifest": manifest, "summary": summary, "choice_summary": choice_summary, "episodes": rows}
    json.dumps(result, allow_nan=False)
    return result


def _encoded(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for source, destination in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                                ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(source, destination)
    return text


def write_report(output_dir):
    """Write index.html; video URLs remain relative to this artifact directory."""
    result = load_report(output_dir)
    from plotly.offline import get_plotlyjs

    scope = result["manifest"].get("scope", "Ordinary VideoUnmaskSwap simulation episodes; no input interventions")
    page = _PAGE.replace("__SCOPE__", escape(str(scope))).replace("__STYLE__", _STYLE)
    page = page.replace("__PLOTLY__", get_plotlyjs().replace("</script", "<\\/script"))
    page = page.replace("__DATA__", _encoded(result)).replace("__SCRIPT__", _SCRIPT)
    destination = Path(output_dir) / "index.html"
    destination.write_text(page, encoding="utf-8")
    return destination


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VideoUnmaskSwap: which container does the policy choose?</title><style>__STYLE__</style></head><body><main>
<header><p class="eyebrow">ORDINARY CLOSED-LOOP ROLLOUTS</p><h1>Which container does the policy choose?</h1><p>__SCOPE__</p>
<p class="caution">Official success and geometric choice diagnostics are different measurements. Simulator identities annotate this report only; they are not policy inputs.
World X/Y ranks describe the layout at execution start after the conditioning video. They do not mean screen-left or screen-right.</p></header>
<section id="stats" class="stats"></section><p id="progress" class="muted"></p>
<section class="card"><h2>All planned episodes</h2><div class="scroll"><table id="episodes"></table></div>
<p class="muted">“No approach” means no event met the declared proximity threshold, not a missing result. Approach is a proxy; grasp/lift evidence is shown separately.
The first grasp is the simulator's first grasp/contact flag; it can be momentary and does not establish a successful pickup.
Actor indices identify containers within one episode and are not fixed physical locations across episodes.</p></section>
<section class="card"><h2>Target and first grasp/lift locations across configurations</h2><div id="locations" class="plot"></div>
<p class="muted">Dots mark intended target centers at execution start; triangles mark the first grasped container and crosses the first lifted container at their starting locations.
Grasp and lift are separate events. A wrong pickup can terminate the episode before an active observation records its lift;
“none observed” does not establish that no lift occurred. Pending episodes contribute no choice marker.</p></section>
<section class="card"><div class="controls"><h2 id="episodeTitle"></h2><label>Inspect episode<select id="episodeSelect"></select></label></div>
<p id="episodeNote"></p><p id="eventNote" class="muted"></p><div class="panels"><div><video id="video" controls preload="metadata"></video><p id="videoNote" class="muted"></p><a id="videoLink">Open or download MP4</a></div><div id="trajectory" class="plot"></div></div>
<p class="muted">The overhead trace is the actual observed TCP trajectory, not predicted target kinematics. Container markers use execution-start positions;
containers may subsequently move. Hover markers to identify each container and its world-coordinate rank.</p>
<details><summary>Observed events and diagnostic definitions</summary><pre id="diagnostics"></pre></details></section>
<details class="card"><summary>Frozen protocol and complete planned-denominator summary</summary><pre id="protocol"></pre><pre id="summary"></pre></details>
</main><script>__PLOTLY__</script><script type="application/json" id="payload">__DATA__</script><script>__SCRIPT__</script></body></html>"""

_STYLE = """*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#172638;font:15px/1.55 system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:32px 24px 64px}h1{font-size:clamp(28px,4vw,44px);line-height:1.13;margin:8px 0 18px}h2{font-size:22px;margin:0 0 14px}.eyebrow{font-size:12px;font-weight:750;letter-spacing:.14em;color:#3566a4}.caution{background:#fff4d8;border-left:4px solid #c38a20;padding:14px 18px;border-radius:6px}.card{background:#fff;border:1px solid #dce3ec;border-radius:14px;margin:24px 0;padding:24px;box-shadow:0 4px 18px #20384a07}.muted{font-size:13px;color:#59677a}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:24px}.stat{background:#fff;border:1px solid #dce3ec;border-radius:10px;padding:16px}.stat strong{display:block;font-size:28px;color:#214d80}.stat span{font-size:13px;color:#59677a}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 9px;border-bottom:1px solid #e2e7ef;text-align:left;vertical-align:top}th{background:#f4f7fa;white-space:nowrap}td:first-child{font-weight:700}.success{color:#168653}.error,.policy_error,.infrastructure_error{color:#b42e32}.timeout,.fail{color:#9b6712}.pending{color:#6e7f91}.controls{display:flex;align-items:center;justify-content:space-between;gap:20px;flex-wrap:wrap}label{font-size:13px;font-weight:600;display:flex;flex-direction:column;gap:5px}select{font:inherit;padding:8px 12px;border:1px solid #bcc8d9;border-radius:6px;background:white}.plot{height:440px}.panels{display:grid;grid-template-columns:1fr 1fr;gap:20px}video{width:100%;max-height:440px;border-radius:8px;background:#101820}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:650}a{color:#245ba4}@media(max-width:850px){main{padding:18px 12px}.card{padding:16px}.stats{grid-template-columns:repeat(2,1fr)}.panels{grid-template-columns:1fr}}"""

_SCRIPT = r"""const data=JSON.parse(document.getElementById('payload').textContent),el=id=>document.getElementById(id);
const diagnostic=r=>r.choice_diagnostics||{},summary=r=>diagnostic(r).summary||{},target=r=>diagnostic(r).target||{},bins=r=>diagnostic(r).reset?.bins||[];
const yesno=v=>v===true?'Yes':v===false?'No':'—';
function rank(r,index){const v=(r.layout_ranks||{})[String(index)];return v?`X ${v.x_rank}/${v.count}, Y ${v.y_rank}/${v.count}`:'—'}
function choice(r,index,empty='None observed'){return index==null?empty:`bin ${index} (${rank(r,index)})`}
function listChoice(r,value){return value?.length?value.map(i=>choice(r,i)).join('; '):'None observed'}
const total=data.summary.overall,c=data.choice_summary;
for(const [value,label] of [[`${total.successes}/${total.planned}`,'Official successes / planned'],[`${total.recorded}/${total.planned}`,'Recorded episodes / planned'],[`${c.correct_first_grasp}/${c.first_grasp_observed}`,'Correct first grasp / observed first grasp'],[`${c.no_approach_observed}`,'Recorded episodes with no approach event']]){const box=document.createElement('div');box.className='stat';const strong=document.createElement('strong'),span=document.createElement('span');strong.textContent=value;span.textContent=label;box.append(strong,span);el('stats').append(box)}
el('progress').textContent=total.complete?`All ${total.planned} planned episodes are recorded. Errors (${total.errors}) and timeouts (${total.timeouts}) remain in the success denominator.`:`Progress report: ${total.missing} planned episodes remain pending. Success/planned is a lower bound until every episode finishes. Errors and timeouts remain included.`;
const headers=['Episode','Official status','Intended first color','Target spatial rank','First approach proxy','First grasp','Correct first grasp','First lift','Correct first lift'];
const head=document.createElement('tr');for(const name of headers){const th=document.createElement('th');th.textContent=name;head.append(th)}el('episodes').append(head);
data.episodes.forEach((r,i)=>{const s=summary(r),t=target(r),row=document.createElement('tr'),values=[r.episode_idx,r.status,t.color||'—',rank(r,t.first_bin_index),r.pending?'Pending':r.observer_available?choice(r,s.first_approach_bin_index,'No approach'):'Diagnostics unavailable',r.pending?'Pending':r.observer_available?listChoice(r,s.first_grasp_bin_indices):'Diagnostics unavailable',yesno(s.first_grasp_correct),r.pending?'Pending':r.observer_available?listChoice(r,s.first_lift_bin_indices):'Diagnostics unavailable',yesno(s.first_lift_correct)];values.forEach((value,j)=>{const td=document.createElement('td');if(j===0){const a=document.createElement('a');a.href='#episodeTitle';a.textContent=value;a.onclick=()=>{el('episodeSelect').value=i;renderEpisode()};td.append(a)}else td.textContent=value;if(j===1)td.className=r.status;row.append(td)});el('episodes').append(row);const option=document.createElement('option');option.value=i;option.textContent=`Episode ${r.episode_idx} · ${r.status}`;el('episodeSelect').append(option)});
const layout=title=>({title:{text:title,font:{size:15}},margin:{l:55,r:15,t:40,b:50},paper_bgcolor:'white',xaxis:{title:'World X (m)'},yaxis:{title:'World Y (m)',scaleanchor:'x',scaleratio:1},legend:{orientation:'h',y:-.23}});
const scatter=(points,name,color,symbol='circle',texts=[])=>({type:'scatter',mode:'markers',x:points.map(p=>p[0]),y:points.map(p=>p[1]),name,marker:{size:11,color,symbol},text:texts,hovertemplate:'%{text}<br>X %{x:.3f}, Y %{y:.3f} m<extra>%{fullData.name}</extra>'});
function renderLocations(){const intended=[],grasped=[],lifted=[],itext=[],gtext=[],ltext=[];for(const r of data.episodes){const t=target(r),s=summary(r),lookup=new Map(bins(r).map(b=>[b.index,b]));if(lookup.has(t.first_bin_index)){intended.push(lookup.get(t.first_bin_index).xyz_m);itext.push(`Episode ${r.episode_idx}: intended ${t.color||'target'}`)}for(const i of s.first_grasp_bin_indices||[]){if(lookup.has(i)){grasped.push(lookup.get(i).xyz_m);gtext.push(`Episode ${r.episode_idx}: first grasp bin ${i}; correct ${yesno(s.first_grasp_correct)}`)}}for(const i of s.first_lift_bin_indices||[]){if(lookup.has(i)){lifted.push(lookup.get(i).xyz_m);ltext.push(`Episode ${r.episode_idx}: first lift bin ${i}; correct ${yesno(s.first_lift_correct)}`)}}}Plotly.react('locations',[scatter(intended,'Intended first target','#178477','circle',itext),scatter(grasped,'First grasped container','#c26332','triangle-up',gtext),scatter(lifted,'First lifted container','#8d3c92','x',ltext)],layout('Execution-start positions in common world coordinates'),{responsive:true,displaylogo:false})}
function renderEpisode(){const r=data.episodes[Number(el('episodeSelect').value)],d=diagnostic(r),s=summary(r),t=target(r);el('episodeTitle').textContent=`Episode ${r.episode_idx} · ${r.status}`;el('episodeNote').textContent=r.pending?'This planned episode has not produced a durable result.':`Requested color order: ${(t.ordered_colors||[t.color||'unavailable']).join(' → ')}; first target ${choice(r,t.first_bin_index,'unknown')}. ${r.primitive_commands??'?'} native commands. ${r.error?'Recorded error: '+r.error:''}`;el('eventNote').textContent=r.pending?'':!r.observer_available?'No measured execution diagnostics are available for this episode.':`First approach: ${choice(r,s.first_approach_bin_index,'none observed')}; first grasp: ${listChoice(r,s.first_grasp_bin_indices)}; first lift: ${listChoice(r,s.first_lift_bin_indices)}. Correct first lift: ${yesno(s.first_lift_correct)}. Any correct lift observed: ${yesno(s.any_correct_lift)}. Completed subgoal index increase: ${s.completed_subgoal_index_increase??'unavailable'}.`;const video=el('video');video.pause();if(r.video){video.src=r.video;video.style.display='block';el('videoLink').href=r.video;el('videoLink').style.display='inline';el('videoNote').textContent='Ordinary rollout video. Keep the videos/ directory with this HTML file.'}else{video.removeAttribute('src');video.load();video.style.display='none';el('videoLink').removeAttribute('href');el('videoLink').style.display='none';el('videoNote').textContent=r.pending?'Video pending.':'No saved video found for this episode.'}const traces=[],start=d.reset?.tcp_xyz_m,timeline=(d.timeline||[]).filter(v=>Array.isArray(v.tcp_xyz_m)),path=[...(start?[start]:[]),...timeline.map(v=>v.tcp_xyz_m)];if(path.length)traces.push({type:'scatter',mode:'lines',x:path.map(v=>v[0]),y:path.map(v=>v[1]),name:'Observed TCP',line:{color:'#315aa6',width:2}});if(start)traces.push(scatter([start],'Initial TCP','#315aa6','triangle-up',['Before first policy command']));for(const bin of bins(r)){const intended=bin.index===t.first_bin_index;traces.push(scatter([bin.xyz_m],`${intended?'Target · ':''}bin ${bin.index}`,intended?'#168653':'#7b8797',intended?'star':'square',[`bin ${bin.index}; ${rank(r,bin.index)}${intended?'; requested '+t.color:''}`]))}for(const [field,label,color] of [['first_approach_command','First approach proxy','#d18c22'],['first_grasp_command','First grasp','#8347b5'],['first_lift_command','First lift','#bf3d50']]){const when=s[field],event=timeline.find(v=>v.command_index===when);if(event)traces.push(scatter([event.tcp_xyz_m],label,color,'diamond',[`Command ${when}`]))}Plotly.react('trajectory',traces,layout('Actual TCP path and starting container locations'),{responsive:true,displaylogo:false});el('diagnostics').textContent=JSON.stringify({definitions:d.definitions,target:t,summary:s,reset:d.reset,errors:d.errors},null,2)}
el('video').onerror=()=>{if(el('video').getAttribute('src'))el('videoNote').textContent='This browser could not decode the video. The original MP4 is available using the link below.'};el('episodeSelect').onchange=renderEpisode;el('protocol').textContent=JSON.stringify(data.manifest,null,2);el('summary').textContent=JSON.stringify({official:data.summary,choices:data.choice_summary},null,2);renderLocations();renderEpisode();
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    print(write_report(parser.parse_args().output_dir))
