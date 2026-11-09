"""Paired closed-loop history interventions, with explicit planned denominators.

This report separates official outcomes from fixed-current action probes.
History deletion is an out-of-distribution diagnostic, not semantic proof.
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from urllib.parse import quote

from stmem.evaluation.metrics import summarize_episodes


_ERRORS = {"error", "infrastructure_error", "policy_error"}
# run_episode writes imageio.get_writer(video_path, fps=20), independently of
# retained-observation frequency. A contract test checks the actual writer.
_RECORDED_VIDEO_FPS = 20.0
_LABELS = {
    "full": "Full history",
    "erase_conditioning": "Without conditioning history",
    "erase_execution": "Without execution history",
    "current_only": "Current only",
    "recent_execution": "Recent execution blocks + conditioning",
    "permute_execution": "Execution content shuffled across time slots",
}


def _plan(manifest):
    entries = manifest.get("planned_cases")
    if not isinstance(entries, list) or not entries:
        raise ValueError("An explicit nonempty planned_cases list is required")
    keys = []
    for entry in entries:
        task, index, arm = entry.get("task"), entry.get("episode_idx"), entry.get("arm")
        if (not isinstance(task, str) or not task or not isinstance(arm, str) or not arm
                or not isinstance(index, int) or isinstance(index, bool) or index < 0
                or "/" in arm or "\\" in arm or arm in {".", ".."}):
            raise ValueError("Invalid planned task, episode index, or arm")
        keys.append((task, index, arm))
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate planned case")
    arms = list(dict.fromkeys(key[2] for key in keys))
    if "full" not in arms:
        raise ValueError("Matched interventions require the full reference arm")
    scenes = [(task, index) for task, index, arm in keys if arm == "full"]
    if any({(task, index) for task, index, a in keys if a == arm} != set(scenes) for arm in arms):
        raise ValueError("Every arm must plan exactly the same task/episode scenes")
    return keys, ["full", *[arm for arm in arms if arm != "full"]], scenes


def _video(output, record, arm):
    candidate = record.get("video_path")
    if candidate:
        path = Path(candidate)
        if not path.is_absolute():
            resolved = path.resolve()
            path = resolved if resolved.is_relative_to(output) else output / path
    else:
        path = output / "videos" / arm / f"{record['task']}_{record['episode_idx']:03d}.mp4"
    path = path.resolve()
    if not path.is_relative_to(output):
        raise ValueError("Video must remain inside the report directory")
    return quote(path.relative_to(output).as_posix()) if path.is_file() else None


def _fingerprints(row):
    return (row.get("task_diagnostics") or {}).get("input_fingerprints") or {}


def _video_execution_start(record):
    # Use native recorded frames, not the subsampled model-conditioning count.
    frames = (record.get("environment") or {}).get("conditioning_frames_recorded")
    if frames is None:
        return None
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
        raise ValueError("Recorded conditioning video length must be nonnegative frames")
    return frames / _RECORDED_VIDEO_FPS


def _parity(left, right):
    a, b = _fingerprints(left), _fingerprints(right)

    def compare(x, y):
        return "unknown" if x is None or y is None else "matched" if x == y else "mismatch"

    result = {
        "reset": compare(a.get("reset_sha256"), b.get("reset_sha256")),
        "current": compare((a.get("current") or {}).get("sha256"), (b.get("current") or {}).get("sha256")),
        "conditioning": compare((a.get("conditioning") or {}).get("sha256"), (b.get("conditioning") or {}).get("sha256")),
        "goal": compare(a.get("goal_sha256"), b.get("goal_sha256")),
        "noise_seed": compare(left.get("noise_seed"), right.get("noise_seed")),
    }
    result["status"] = "mismatch" if "mismatch" in result.values() else "matched" if all(v == "matched" for v in result.values()) else "unknown"
    return result


def _paired(rows, scenes, arm):
    counts = dict.fromkeys([
        "planned", "pending_pairs", "error_pairs", "input_mismatch_pairs", "completed_pairs",
        "verified_input_pairs", "unknown_input_pairs", "both_success", "full_only_success",
        "control_only_success", "both_unsuccessful",
    ], 0)
    cases = []
    for task, index in scenes:
        full, control = rows[(task, index, "full")], rows[(task, index, arm)]
        parity = _parity(full, control)
        counts["planned"] += 1
        if full["pending"] or control["pending"]:
            outcome = "pending_pairs"
        elif full["status"] in _ERRORS or control["status"] in _ERRORS:
            outcome = "error_pairs"
        elif parity["status"] == "mismatch":
            outcome = "input_mismatch_pairs"
        else:
            counts["completed_pairs"] += 1
            counts["verified_input_pairs" if parity["status"] == "matched" else "unknown_input_pairs"] += 1
            success_a, success_b = full["status"] == "success", control["status"] == "success"
            outcome = ("both_success" if success_a and success_b else "full_only_success" if success_a
                       else "control_only_success" if success_b else "both_unsuccessful")
        counts[outcome] += 1
        cases.append({"task": task, "episode_idx": index, "full_status": full["status"],
                      "control_status": control["status"], "outcome": outcome, "input_parity": parity})
    return {**counts, "cases": cases}


def load_report(output_dir):
    """Read durable records, including absent cases as explicitly pending."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    keys, arms, scenes = _plan(manifest)
    records, record_paths = {}, {}
    for path in sorted((output / "results").glob("*/episodes/*.json")):
        arm = path.parent.parent.name
        record = json.loads(path.read_text())
        key = (record.get("task"), record.get("episode_idx"), arm)
        if key not in keys or key in records or record.get("arm", arm) != arm:
            raise ValueError("Unexpected, duplicated, or inconsistently labeled result")
        if not isinstance(record.get("status"), str):
            raise ValueError("Result lacks official status")
        records[key] = record
        record_paths[key] = quote(path.relative_to(output).as_posix())
    rows = {}
    for task, index, arm in keys:
        record = records.get((task, index, arm))
        row = {"task": task, "episode_idx": index, "arm": arm, "pending": record is None,
               "status": "pending" if record is None else record["status"]}
        if record is not None:
            diagnostic = record.get("task_diagnostics") or {}
            row.update({
                "goal": record.get("goal_text", record.get("goal", diagnostic.get("goal"))),
                "noise_seed": record.get("noise_seed"), "error": record.get("error"),
                "runtime_seconds": record.get("runtime_seconds"),
                "primitive_commands": record.get("primitive_commands", len(record.get("actions", []))),
                "task_diagnostics": diagnostic,
                "fixed_current_probes": record.get("fixed_current_probes", []),
                "video": _video(output, record, arm),
                "policy_video_start_seconds": _video_execution_start(record),
                "raw_result": record_paths[(task, index, arm)],
            })
        rows[(task, index, arm)] = row
    by_arm = {
        arm: summarize_episodes([record for (task, index, a), record in records.items() if a == arm], scenes)
        for arm in arms
    }
    tasks = list(dict.fromkeys(task for task, _ in scenes))
    paired = {arm: _paired(rows, scenes, arm) for arm in arms if arm != "full"}
    paired_by_task = {
        task: {arm: _paired(rows, [key for key in scenes if key[0] == task], arm)
               for arm in arms if arm != "full"}
        for task in tasks
    }
    summary = {"planned_cases": len(keys), "recorded_cases": len(records),
               "complete": len(records) == len(keys), "per_arm": by_arm,
               "paired": paired, "paired_by_task": paired_by_task,
               "interpretation": "History modification can be an OOD intervention. Paired outcome changes show dependence under the declared intervention, not semantic understanding. Errors and missing pairs are not counted as a lost success."}
    result = {"manifest": manifest, "arms": arms, "tasks": tasks,
              "scenes": [{"task": task, "episode_idx": index,
                          "arms": {arm: rows[(task, index, arm)] for arm in arms}} for task, index in scenes],
              "summary": summary}
    json.dumps(result, allow_nan=False)
    return result


def summarize(output_dir):
    """Return official per-arm and matched-case outcomes without writing files."""
    return load_report(output_dir)["summary"]


def _encoded(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for source, destination in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                                ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(source, destination)
    return text


def _html_payload(result):
    """Keep displayed evidence; retain detailed traces only in linked raw JSON.

    A fresh shallow structure avoids mutating load_report's full diagnostics.
    Fingerprint equality and success counts have already been computed using
    complete records before this presentation-only compaction.
    """
    scenes = []
    for scene in result["scenes"]:
        arms = {}
        for arm, row in scene["arms"].items():
            diagnostic = row.get("task_diagnostics") or {}
            compact = {key: diagnostic[key] for key in (
                "version", "status", "goal", "target", "summary", "errors", "definitions", "observer_seconds",
            ) if key in diagnostic}
            fingerprints = diagnostic.get("input_fingerprints") or {}
            if fingerprints:
                compact["input_fingerprints"] = {
                    key: fingerprints[key] for key in ("reset_sha256", "goal_sha256") if key in fingerprints
                }
                for key, fields in (("current", ("sha256",)), ("conditioning", ("sha256", "count"))):
                    if key in fingerprints:
                        values = fingerprints[key] or {}
                        compact["input_fingerprints"][key] = {field: values[field] for field in fields if field in values}
            arms[arm] = {**row, "task_diagnostics": compact}
        scenes.append({**scene, "arms": arms})
    return {**result, "scenes": scenes}


def write_report(output_dir):
    result = _html_payload(load_report(output_dir))
    scope = result["manifest"].get("scope", "Paired closed-loop history diagnostics; fixed validation episodes, no seed search")
    page = _PAGE.replace("__SCOPE__", escape(str(scope))).replace("__STYLE__", _STYLE)
    page = page.replace("__DATA__", _encoded(result)).replace("__LABELS__", _encoded(_LABELS)).replace("__SCRIPT__", _SCRIPT)
    destination = Path(output_dir) / "index.html"
    destination.write_text(page, encoding="utf-8")
    return destination


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Does the teacher need history to complete the task?</title><style>__STYLE__</style></head><body><main>
<header><p class="eyebrow">PAIRED CLOSED-LOOP DIAGNOSTIC</p><h1>Does the teacher need history to complete the task?</h1><p>__SCOPE__</p>
<p class="caution">A change in predicted actions is not a change in task success. These are separate measurements below.
Changing or deleting history can be an out-of-distribution intervention: even a loss of success does not by itself prove which historical fact the policy used.
Simulator object identities and privileged task-state annotations are diagnostic only, never policy inputs.</p></header>
<section id="stats" class="stats"></section><p id="progress" class="muted"></p>
<section class="card"><h2>Official success on all planned scenes</h2><div class="scroll"><table id="success"></table></div>
<p class="muted">Every cell uses the complete planned denominator. Pending cases make current rates lower bounds. Errors remain unsuccessful in these rates.</p></section>
<section class="card"><h2>What changes under the history intervention?</h2><div class="scroll"><table id="paired"></table></div>
<p class="muted">Lost = full succeeds and control fails or times out; gained = the reverse. Missing, errored, or mismatched-input pairs are listed separately and are not evidence of lost competence.
“Verified input pairs” require matching reset/current/conditioning/goal fingerprints and action-noise seed. These checks concern the original inputs before the declared intervention.</p><div id="examples"></div></section>
<section class="card"><h2>Every planned scene</h2><div class="scroll"><table id="scenes"></table></div></section>
<section class="card"><div class="controls"><h2 id="sceneTitle">Inspect a scene</h2><label>Task<select id="task"></select></label><label>Episode<select id="episode"></select></label>
<button id="playPolicy">Play policy execution</button><button id="playAll">Play including demonstration</button><button id="pauseAll">Pause all</button></div><p id="sceneGoal"></p>
<div id="videos" class="videos"></div><p class="muted">Videos share the same initial scene. Trajectories diverge after different commands; matching playback times does not imply matching physical states.
Saved videos retain the simulator's conditioning demonstration even in controls that withhold those frames from the policy. The policy-start note distinguishes demonstration from policy execution.
Keep videos/ with this HTML file. Direct MP4 links are available if your browser cannot decode them.</p>
<details><summary>Reset parity and observed task events</summary><pre id="events"></pre></details></section>
<section class="card"><h2>Fixed-current action probes — a separate measurement</h2>
<p class="muted">These probes hold the current observation and action noise fixed at states along the full-history trajectory. They measure changes to predicted chunks, not alternate rollout success.
Missing probes mean unmeasured, not zero sensitivity.</p><div id="probes"></div></section>
<details class="card"><summary>Frozen protocol and complete measured summary</summary><pre id="manifest"></pre><pre id="summary"></pre></details>
</main><script type="application/json" id="payload">__DATA__</script><script type="application/json" id="labels">__LABELS__</script><script>__SCRIPT__</script></body></html>"""

_STYLE = """*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#172638;font:15px/1.55 system-ui,sans-serif}main{max-width:1500px;margin:auto;padding:32px 24px 64px}h1{font-size:clamp(28px,4vw,44px);line-height:1.15;margin:8px 0 18px}h2{font-size:22px;margin:0 0 14px}h3{font-size:17px;margin:0 0 6px}.eyebrow{font-size:12px;font-weight:750;letter-spacing:.14em;color:#3566a4}.caution{background:#fff4d8;border-left:4px solid #c38a20;padding:14px 18px;border-radius:6px}.card{background:#fff;border:1px solid #dce3ec;border-radius:14px;margin:24px 0;padding:24px}.muted{font-size:13px;color:#59677a}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:24px}.stat{background:#fff;border:1px solid #dce3ec;border-radius:10px;padding:16px}.stat strong{display:block;font-size:26px;color:#214d80}.stat span{font-size:13px;color:#59677a}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:10px 9px;border-bottom:1px solid #e2e7ef;text-align:left;vertical-align:top}th{background:#f4f7fa}td:first-child{font-weight:650}.success{color:#168653}.error,.policy_error,.infrastructure_error,.mismatch{color:#b42e32}.timeout,.fail{color:#9b6712}.pending,.unknown{color:#6e7f91}.controls{display:flex;align-items:center;gap:16px;flex-wrap:wrap}.controls h2{flex:1}label{font-size:13px;font-weight:600;display:flex;flex-direction:column;gap:5px}select,button{font:inherit;padding:8px 12px;border:1px solid #bcc8d9;border-radius:6px;background:white}button{cursor:pointer;color:#245ba4}.videos{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}.videoCard{border:1px solid #dce3ec;border-radius:10px;padding:14px}video{width:100%;max-height:400px;border-radius:6px;background:#101820}pre{font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:650}a{color:#245ba4}.example{background:#edf4fb;padding:10px 14px;border-radius:6px;margin:8px 0}@media(max-width:800px){main{padding:18px 12px}.card{padding:16px}.stats{grid-template-columns:repeat(2,1fr)}.videos{grid-template-columns:1fr}}"""

_SCRIPT = r"""const data=JSON.parse(document.getElementById('payload').textContent),labels=JSON.parse(document.getElementById('labels').textContent),el=id=>document.getElementById(id),label=a=>labels[a]||a;
function node(tag,text,cls){const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n}
function table(target,headers,rows){const t=typeof target==='string'?el(target):target;t.replaceChildren();const h=node('tr');headers.forEach(v=>h.append(node('th',v)));t.append(h);rows.forEach(values=>{const r=node('tr');values.forEach(v=>{const cell=node('td');cell.append(v instanceof Node?v:document.createTextNode(String(v??'—')));r.append(cell)});t.append(r)})}
const s=data.summary;for(const a of data.arms){const c=s.per_arm[a].overall,box=node('div',null,'stat');box.append(node('strong',`${c.recorded?c.successes:'—'}/${c.planned}`),node('span',label(a)+' official successes'));el('stats').append(box)}
el('progress').textContent=`${s.recorded_cases}/${s.planned_cases} planned rollouts recorded. ${s.complete?'All results are present.':'Progress report; pending cases remain in every denominator.'}`;
const rate=c=>c.recorded?`${c.successes}/${c.planned} (${(100*c.successes/c.planned).toFixed(1)}%) · ${c.missing} pending, ${c.errors} errors`:`Not measured · 0/${c.planned} recorded`;
table('success',['Task',...data.arms.map(label)],[...data.tasks.map(t=>[t,...data.arms.map(a=>rate(s.per_arm[a].tasks[t]))]),['All planned scenes',...data.arms.map(a=>rate(s.per_arm[a].overall))]]);
const pairedRows=[];for(const a of data.arms.filter(a=>a!=='full')){for(const task of [...data.tasks,null]){const p=task?s.paired_by_task[task][a]:s.paired[a];pairedRows.push([label(a),task||'All scenes',`${p.completed_pairs}/${p.planned}`,p.full_only_success,p.control_only_success,p.both_success,p.both_unsuccessful,`${p.verified_input_pairs}/${p.completed_pairs}`,p.pending_pairs,p.error_pairs,p.input_mismatch_pairs])}}table('paired',['Control','Task','Completed pairs / planned','Lost successes','Gained successes','Both succeed','Both unsuccessful','Verified input pairs','Pending pairs','Error pairs','Input mismatch'],pairedRows);
function choose(task,index){el('task').value=task;setEpisodes(index);el('sceneTitle').scrollIntoView({behavior:'smooth',block:'start'})}
let examples=0;for(const [a,p] of Object.entries(s.paired)){for(const c of p.cases){if(c.outcome==='full_only_success'){const box=node('div',null,'example'),link=node('a',`${c.task} episode ${c.episode_idx}`);link.href='#sceneTitle';link.onclick=()=>choose(c.task,c.episode_idx);box.append(link,document.createTextNode(`: full succeeded; ${label(a)} ${c.control_status}. Input parity: ${c.input_parity.status}. This is an intervention result, not automatic semantic proof.`));el('examples').append(box);examples++}}}if(!examples)el('examples').append(node('p','No recorded full-success → control-failure pair yet.','muted'));
table('scenes',['Task / episode',...data.arms.map(label)],data.scenes.map(scene=>{const link=node('a',`${scene.task} / ${scene.episode_idx}`);link.href='#sceneTitle';link.onclick=()=>choose(scene.task,scene.episode_idx);return[link,...data.arms.map(a=>node('span',scene.arms[a].status,scene.arms[a].status))]}));
for(const task of data.tasks){const option=node('option',task);option.value=task;el('task').append(option)}
function scene(){return data.scenes.find(v=>v.task===el('task').value&&v.episode_idx===Number(el('episode').value))}
function setEpisodes(selected){el('episode').replaceChildren();for(const v of data.scenes.filter(v=>v.task===el('task').value)){const option=node('option',`Episode ${v.episode_idx}`);option.value=v.episode_idx;el('episode').append(option)}if(selected!=null)el('episode').value=selected;renderScene()}
function renderProbes(full){const container=el('probes');container.replaceChildren();const probes=full.fixed_current_probes||[];if(!probes.length){container.append(node('p','No fixed-current action probe is recorded for this scene.','muted'));return}const measured=probes.filter(p=>p.status==='passed').length;container.append(node('p',`${measured}/${probes.length} probes passed the full-action replay check. Only passed probes should be interpreted. The reference and controls share the current observation, task, clocks and initial noise.`));const probeArms=[...new Set(probes.flatMap(p=>Object.keys(p.arms||{})).map(a=>a==='FULL'?'full':a==='CURRENT'?'current_only':a))];if(probeArms.some(a=>!data.arms.includes(a)))container.append(node('p','These saved full-run probes also include controls outside this report’s rollout comparison. Each probe arm is labeled explicitly; no fixed-current result for an unlisted arm is implied.','muted'));const wrapper=node('div',null,'scroll'),tab=node('table'),rows=[],number=v=>Number.isFinite(v)?v.toFixed(3):'—';for(const p of probes){for(const [a,v] of Object.entries(p.arms||{})){rows.push([p.decision_index??'—',p.status,label(a==='FULL'?'full':a==='CURRENT'?'current_only':a),v.history_leaf_count,v.context_tokens,number(v.tcp_change_mean_mm),number(v.tcp_change_executed_targets_mean_mm),number(v.tcp_change_last_target_mm),number(v.max_abs_joint_target_change_rad)])}}table(tab,['Decision','Probe status','Arm','History leaves','Context tokens','Mean TCP change, all targets (mm)','Mean TCP change, executed prefix (mm)','Last-target TCP change (mm)','Max joint-target change (rad)'],rows);wrapper.append(tab);container.append(wrapper);for(const [i,probe] of probes.entries()){const details=node('details'),title=node('summary',`Probe ${i+1} · decision ${probe.decision_index??'unreported'} · ${probe.status??'unknown'} · raw values and replay audit`);details.append(title,node('pre',JSON.stringify(probe,null,2)));container.append(details)}}
function renderScene(){const selected=scene();if(!selected)return;document.querySelectorAll('video').forEach(v=>v.pause());el('videos').replaceChildren();el('sceneTitle').textContent=`${selected.task} · episode ${selected.episode_idx}`;const goals=[...new Set(Object.values(selected.arms).map(v=>v.goal).filter(Boolean))];el('sceneGoal').textContent=goals.length?`Goal: ${goals.join(' | ')}`:'Goal has not been recorded yet.';const details={};for(const a of data.arms){const r=selected.arms[a],card=node('div',null,'videoCard'),heading=node('h3',label(a)),status=node('p',`${r.status}${r.primitive_commands!=null?' · '+r.primitive_commands+' native commands':''}`,r.status);card.append(heading,status);if(r.video){const video=node('video');video.controls=true;video.preload='metadata';video.src=r.video;const start=r.policy_video_start_seconds;if(Number.isFinite(start)){video.dataset.policyStart=start;card.append(node('p',start>0?`Conditioning demonstration: first ${start.toFixed(2)}s. Policy execution starts at ${start.toFixed(2)}s.`:'No conditioning demonstration. Policy execution starts at 0s.','muted'))}else card.append(node('p','Policy-start time was not recorded; this video may include a conditioning demonstration.','muted'));const note=node('p','', 'muted');video.onerror=()=>note.textContent='Browser decoding unavailable; use the original MP4 link below.';const link=node('a','Open / download MP4');link.href=r.video;card.append(video,note,link)}else card.append(node('p',r.pending?'Result and video pending.':'No saved video available.','muted'));if(r.raw_result){const line=node('p'),link=node('a','Complete episode JSON (including frame fingerprints and trace)');link.href=r.raw_result;line.append(link);card.append(line)}if(r.error)card.append(node('p','Recorded error: '+r.error,'error'));el('videos').append(card);const diag=r.task_diagnostics||{};const match=a==='full'?null:s.paired[a].cases.find(c=>c.task===selected.task&&c.episode_idx===selected.episode_idx);details[a]={status:r.status,goal:r.goal,input_parity:match?.input_parity,input_fingerprints:diag.input_fingerprints,target:diag.target,summary:diag.summary,errors:diag.errors}}el('events').textContent=JSON.stringify(details,null,2);renderProbes(selected.arms.full)}
el('task').onchange=()=>setEpisodes();el('episode').onchange=renderScene;el('playPolicy').onclick=()=>document.querySelectorAll('video[data-policy-start]').forEach(v=>{v.currentTime=Number(v.dataset.policyStart);v.play().catch(()=>{})});el('playAll').onclick=()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})});el('pauseAll').onclick=()=>document.querySelectorAll('video').forEach(v=>v.pause());el('manifest').textContent=JSON.stringify(data.manifest,null,2);el('summary').textContent=JSON.stringify(s,null,2);setEpisodes();
"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    print(write_report(parser.parse_args().output_dir))
