"""Generated-action witnesses for local compact-reader utility diagnostics.

Selection uses expert FM discovery losses only. Generation starts from fresh,
unmasked Gaussian noise, uses the actual compact context throughout Euler, and
does not require a teacher checkpoint. FK paths are joint-target diagnostics,
not executed trajectories or estimates of benchmark success.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import torch

from stmem.evaluation.robomme_adapter import physical_action
from stmem.inspection.refinement_measurements import (
    GENERATION_CONVENTION, RNG_CONVENTION, _context_batches, _losses, _precision,
    _tensor_hash_update, _timed, action_mask_summary,
)
from stmem.inspection.refinement_statistics import tie_set
from stmem.models.action_flow import masked_mse
from stmem.training.checkpoints import file_sha256
from stmem.training.label_objectives import policy_context
from stmem.visualization.kinematics import panda_tcp_positions, panda_variant

SCHEMA = "local_student_behavior_v1"


def select_behavior_contexts(panel, rows, *, reader, final_budget=128, epsilon=1e-6,
                             random_seed=0):
    """Pick witnesses within one predeclared state/family, never across states.

    The caller fixes the states/readers before measuring any losses. All
    completed packet alternatives have the same final token budget. A stable
    representative is used for visualization of a discovery near-tie; the
    complete tie sets are retained and must not be treated as clear rankings.
    """
    families = [family for family in panel["families"]
                if family["reader"] == reader and family["panel"] == "packet"
                and int(family["final_budget"]) == final_budget]
    if len(families) != 1:
        raise ValueError("Need exactly one eligible predeclared packet family")
    family = families[0]
    cost = int(final_budget - family["background"])
    candidates = family["candidates"]
    if cost < 1 or len(candidates) < 2:
        raise ValueError("Generated ranking witnesses require two distinct positive-cost candidates")
    values = defaultdict(dict)
    for row in rows:
        if row["bank"] != "discovery":
            continue
        identity = int(row["draw_index"])
        if identity in values[row["context_id"]]:
            raise ValueError("Duplicate discovery context/draw")
        loss = float(row["expert_full"])
        if not np.isfinite(loss):
            raise ValueError("Behavior selections need finite expert discovery losses")
        values[row["context_id"]][identity] = loss
    ids = [family["base"], *family["controls"].values()]
    ids += [candidate["prefixes"][str(prefix)] for candidate in candidates for prefix in (1, cost)]
    if any(not values[key] for key in ids):
        raise ValueError("Missing discovery rows for the fixed packet family")
    if len({tuple(sorted(values[key])) for key in ids}) != 1:
        raise ValueError("Every behavior alternative must use the same discovery draws")
    means = {key: float(np.mean(list(values[key].values()))) for key in ids}
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Candidate IDs must be distinct")
    first = {key: means[candidate["prefixes"]["1"]] for key, candidate in by_id.items()}
    complete = {key: means[candidate["prefixes"][str(cost)]] for key, candidate in by_id.items()}
    ties = {"immediate_completed": tie_set(first, epsilon=epsilon),
            "packet_best": tie_set(complete, epsilon=epsilon),
            "packet_worst": tie_set(complete, epsilon=epsilon, minimize=False)}
    representatives = {name: candidates[0] for name, candidates in ties.items()}
    representatives["random_packet"] = random.Random(random_seed).choice(sorted(by_id))
    aliases = {"base": family["base"], **{
        name: by_id[key]["prefixes"][str(cost)] for name, key in representatives.items()},
        **{name: key for name, key in family["controls"].items() if name in ("dispersed", "uniform")}}
    return {"reader": reader, "background": int(family["background"]), "final_budget": final_budget,
            "packet_cost": cost, "aliases": aliases, "tie_sets": ties,
            "representatives": representatives, "candidate_ids": sorted(by_id),
            "discovery_completed_losses": complete, "discovery_first_losses": first,
            "discovery_alias_losses": {name: means[key] for name, key in aliases.items()},
            "random_seed": int(random_seed), "selection_bank": "discovery",
            "objective": "expert_full", "tie_epsilon": epsilon,
            "tie_policy": "first stable representative; retain complete discovery near-tie sets",
            "scope": "Within-state loss-selected witnesses, not deployed learned routing or held-out task success"}


def _check_model(policy, checkpoint_hash):
    if not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint_hash)):
        raise ValueError("Generated actions require an exact checkpoint SHA256")
    if (not policy.encoders_frozen or any(module.training for module in policy.modules())
            or any(parameter.requires_grad for parameter in policy.parameters())):
        raise ValueError("Generated-action diagnostics require a frozen eval-mode policy")


def history_geometry(bank):
    """Display actual selected metadata, including time/source/count and bounds.

    These are record anchors/supports, not object membership or attention maps.
    Index positions follow the shared 23-dimensional metadata schema.
    """
    metadata = bank.metadata[0, bank.valid[0]].detach().float().cpu().numpy()
    if metadata.ndim != 2 or metadata.shape[1] != 23 or not np.isfinite(metadata).all():
        raise ValueError("Expected finite selected-record metadata [N,23]")
    return {"anchors": metadata[:, :3].tolist(), "bounds_min": metadata[:, 3:6].tolist(),
            "bounds_max": metadata[:, 6:9].tolist(),
            "time_min": (metadata[:, 9] * 100).tolist(), "time_max": (metadata[:, 10] * 100).tolist(),
            "counts": np.rint(np.exp(metadata[:, 13].astype(np.float64))).astype(np.int64).tolist(),
            "sources": ["conditioning" if row[18] > .5 else "execution" for row in metadata],
            "modalities": ["visual" if row[20] > .5 else "event" for row in metadata],
            "disconnected": (metadata[:, 17] > .5).tolist()}


@torch.inference_mode()
def generate_compact_actions(policy, item, contexts, seeds, *, checkpoint_hash, task=None,
                             precision="fp32", prefix_targets=8, batch_size=8,
                             maximum_padded_tokens=32768, integration_steps=8,
                             teacher=None, teacher_hash=None):
    """Integrate real compact contexts from paired pure noise and measure paths.

    Expert actions/masks enter metrics only; neither determines initial noise,
    context selection, nor generated commands. The optional teacher must be the
    separately verified dependency; the caller validates its encoder identity.
    """
    _check_model(policy, checkpoint_hash)
    task = task or item.get("task") or str(item.get("episode", "")).split("/")[0]
    robot = panda_variant(task)
    mask_summary = action_mask_summary({**item, "task": task})
    if integration_steps != 8:
        raise ValueError("The frozen behavior protocol requires eight Euler integration steps")
    if not contexts or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Need compact contexts and distinct generation seeds")
    if any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
           or not 0 <= int(seed) < (1 << 63) for seed in seeds):
        raise ValueError("Generation seeds must be nonnegative 63-bit integers")
    horizon = int(item["target"].shape[0])
    if not 1 <= prefix_targets <= horizon or not mask_summary["valid_coordinates"]:
        raise ValueError("Need valid expert coordinates and a prefix within the horizon")
    if "teacher_full" in contexts:
        raise ValueError("teacher_full is reserved for an actual optional teacher reference")
    if teacher is not None:
        _check_model(teacher, teacher_hash)
        if teacher.device != policy.device:
            raise ValueError("Teacher and student must share the generation device")
        for field in ("mean", "std", "count"):
            if not torch.equal(getattr(policy.normalizer, field), getattr(teacher.normalizer, field)):
                raise ValueError("Teacher/student action normalization differs")
    elif teacher_hash is not None:
        raise ValueError("A teacher hash without a teacher does not define a reference")
    device = policy.device
    noise = torch.stack([torch.randn(item["target"].shape, dtype=torch.float32,
                                   generator=torch.Generator(device="cpu").manual_seed(int(seed)))
                         for seed in seeds]).to(device)
    noise_hash = hashlib.sha256()
    _tensor_hash_update(noise_hash, "initial_unmasked_noise", noise)
    valid = item["valid"].to(device).bool()[None].expand(len(seeds), -1, -1)
    expert = policy.normalizer.normalize(item["target"].to(device))[None].expand_as(noise)
    output, timings = {}, {}

    def collect(model, key, prediction):
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite generated action chunk")
        full, prefix = _losses(prediction, expert, valid, prefix_targets)
        tail_mask = valid.clone()
        tail_mask[:, :prefix_targets] = False
        tail = masked_mse(prediction, expert, tail_mask, reduction="none")
        actions = model.normalizer.denormalize(prediction).cpu().numpy()
        physical = np.asarray([physical_action(row, task) for row in actions.reshape(-1, 8)]).reshape(actions.shape)
        output[key] = {"normalized_actions": prediction.cpu().numpy(), "physical_actions": physical,
                       "tcp_paths": panda_tcp_positions(physical[..., :7], task=task),
                       "expert_full": full.cpu().tolist(), "expert_prefix": prefix.cpu().tolist(),
                       "expert_tail": tail.cpu().tolist() if prefix_targets < horizon else None,
                       "gripper_commands": None if robot == "panda_stick" else physical[..., 7].tolist(),
                       "joint_clipping_count": int(np.count_nonzero(actions[..., :7] != physical[..., :7]))}

    with _precision(device, precision):
        for keys in _context_batches(contexts, batch_size, maximum_padded_tokens):
            with _timed(timings, "compact_context_seconds", device):
                context, bias = policy_context(policy, [item] * len(keys), [contexts[key] for key in keys])
            with _timed(timings, "compact_integration_seconds", device):
                predictions = [policy.decoder.integrate(
                    context, bias, steps=integration_steps,
                    noise=noise[draw:draw+1].expand(len(keys), -1, -1).clone()).float()
                    for draw in range(len(seeds))]
                stacked = torch.stack(predictions)
            for position, key in enumerate(keys):
                collect(policy, key, stacked[:, position])
            del context, bias, predictions, stacked
        if teacher is not None:
            with _timed(timings, "teacher_context_seconds", device):
                context, bias = policy_context(teacher, [item], [item["full"]])
            with _timed(timings, "teacher_integration_seconds", device):
                predictions = [teacher.decoder.integrate(context, bias, steps=integration_steps,
                                                         noise=noise[draw:draw+1].clone()).float()
                               for draw in range(len(seeds))]
            collect(teacher, "teacher_full", torch.cat(predictions))
    counts = {key: int(bank.valid.sum()) for key, bank in contexts.items()}
    if teacher is not None:
        counts["teacher_full"] = int(item["full"].valid.sum())
        for value in output.values():
            full, prefix = _losses(torch.as_tensor(value["normalized_actions"]),
                                  torch.as_tensor(output["teacher_full"]["normalized_actions"]),
                                  valid.cpu(), prefix_targets)
            value.update(teacher_full=full.tolist(), teacher_prefix=prefix.tolist())
    comparisons = {}
    pairs = [("base", "packet_best"), ("packet_best", "packet_worst"),
             ("immediate_completed", "packet_best"), ("packet_best", "dispersed"),
             ("packet_best", "random_packet")]
    for left, right in pairs:
        if left not in output or right not in output:
            continue
        a, b = output[left], output[right]
        full, prefix = _losses(torch.as_tensor(a["normalized_actions"]),
                              torch.as_tensor(b["normalized_actions"]), valid.cpu(), prefix_targets)
        distance = np.linalg.norm(a["tcp_paths"] - b["tcp_paths"], axis=-1) * 1000
        comparisons[f"{left}_vs_{right}"] = {
            "same_history_token_count": counts[left] == counts[right],
            "normalized_action_mse_full": full.tolist(), "normalized_action_mse_prefix": prefix.tolist(),
            "tcp_target_distance_mm_by_target": distance.tolist(),
            "tcp_target_distance_mm_prefix_mean": distance[:, :prefix_targets].mean(1).tolist(),
            "tcp_target_distance_mm_tail_mean": (distance[:, prefix_targets:].mean(1).tolist()
                                                 if prefix_targets < horizon else None)}
    expert_actions = item["target"].detach().cpu().numpy()
    physical_expert = np.asarray([physical_action(row, task) for row in expert_actions])
    timings["model_examples"] = (len(contexts) + int(teacher is not None)) * len(seeds) * integration_steps
    return {"schema": SCHEMA, "contexts": output, "comparisons": comparisons, "timings": timings,
            "seeds": list(map(int, seeds)), "initial_noise_sha256": noise_hash.hexdigest(),
            "task": task, "robot": robot, "precision": precision, "checkpoint_hash": checkpoint_hash,
            "teacher_hash": teacher_hash, "teacher_status": "measured" if teacher is not None else "not_supplied",
            "prefix_targets": prefix_targets, "integration_steps": integration_steps,
            "generation_convention": GENERATION_CONVENTION, "rng_convention": RNG_CONVENTION,
            "history_token_counts": counts, "history_geometry": {key: history_geometry(bank)
                                                                    for key, bank in contexts.items()},
            "current_visual_anchors": item["current"].metadata[0, item["current"].valid[0], :3].cpu().tolist(),
            "current_tcp": item["current_state"][7:10].cpu().tolist(),
            "expert_tcp_path": panda_tcp_positions(physical_expert[:, :7], task=task).tolist(),
            "expert_tcp_valid": item["valid"][:, :7].all(-1).cpu().tolist(),
            "expert_physical_actions": physical_expert.tolist(), "mask": mask_summary,
            "note": "Predicted joint-target FK paths, not realized motion/contact/success. A different valid action may have higher expert MSE."}


def write_behavior_case(path, result, *, state, student, selection, scene_points=None,
                        identity=None):
    """Write raw arrays, checksummed JSON and one self-contained interactive HTML.

    ``path`` is an NPZ path (other suffixes are replaced). The enclosing runner
    owns the state/checkpoint lock and resume identity; this function never
    selects another case or drops a completed comparison on numerical failure.
    """
    from stmem.inspection.refinement_generated import _write_result

    path = Path(path).with_suffix(".npz")
    if result.get("schema") != SCHEMA:
        raise ValueError("Unsupported student generated-action result schema")
    if set(result["contexts"]) - {"teacher_full"} != set(selection["aliases"]):
        raise ValueError("Generated contexts do not match the frozen discovery selection")
    for alias in selection["aliases"]:
        expected_count = selection["background"] if alias == "base" else selection["final_budget"]
        if result["history_token_counts"][alias] != expected_count:
            raise ValueError("Generated historical token count differs from the declared allocation")
    metadata = {"schema": SCHEMA, "state": state, "student": student, "selection": selection,
                "identity": identity, "scope": "exploratory_local_compact_student_diagnostic",
                "simulation_status": "not_run"}
    if scene_points is not None:
        points = np.asarray(scene_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] not in (3, 6) or not np.isfinite(points).all():
            raise ValueError("Display scene points must be finite [N,3] or XYZRGB [N,6]")
        # Deterministic display-only stride; this does not alter encoded inputs.
        points = points[::max(1, int(np.ceil(len(points) / 8192)))]
        metadata["scene_points"] = points.tolist()
    _write_result(path, result, metadata)
    report_behavior_case(path)
    return {"arrays": str(path), "json": str(path.with_suffix(".json")),
            "html": str(path.with_suffix(".html"))}


def report_behavior_case(path):
    """Rebuild the viewer from verified artifacts without inference."""
    from plotly.offline import get_plotlyjs

    path = Path(path).with_suffix(".npz")
    payload = json.loads(path.with_suffix(".json").read_text())
    if payload.get("schema") != SCHEMA or payload["arrays_sha256"] != file_sha256(path):
        raise ValueError("Generated behavior schema/checksum mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        payload["paths"] = {key.split("/")[0]: arrays[key].tolist()
                            for key in arrays.files if key.endswith("/tcp_paths")}
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    page = _HTML.replace("__PLOTLY__", get_plotlyjs().replace("</script", "<\\/script")).replace("__DATA__", encoded)
    path.with_suffix(".html").write_text(page)
    return path.with_suffix(".html")


_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Student history-allocation actions</title>
<style>body{font:16px/1.5 system-ui;color:#243047;margin:24px;max-width:1500px}select{font:inherit;padding:6px;margin:4px}#plot{height:700px}.note{background:#edf3fa;padding:14px;border-left:4px solid #3572ab}pre{white-space:pre-wrap;max-height:450px;overflow:auto}table{border-collapse:collapse}th,td{padding:7px 14px;border-bottom:1px solid #ccd5df;text-align:right}th:first-child,td:first-child{text-align:left}</style></head><body>
<h1>Generated action chunks under different history allocations</h1>
<p class="note">Every alternative uses the same current observation, task and fresh Gaussian noise. Best/worst packets were chosen using discovery expert FM losses, not these generated paths or confirmation losses. Lines show TCP positions of predicted joint targets; they are not simulated trajectories or evidence of task success. The solid segment is the executed prefix, the dashed segment the discarded tail. Expert actions are a recorded reference, not a unique correct solution.</p>
<h2 id="title"></h2><label>Generation draw <select id="draw"></select></label>
<label>Show historical records for <select id="context"></select></label>
<label><input id="history" type="checkbox" checked> Historical anchors</label>
<label><input id="bounds" type="checkbox"> Selected-record bounding boxes</label>
<label>Source <select id="source"><option value="all">Both banks</option><option value="conditioning">Conditioning</option><option value="execution">Execution</option></select></label>
<p id="status"></p><div id="plot"></div><div id="table"></div>
<p>Historical markers are aggregate-record anchors, not object masks. Their source-local time intervals and leaf counts appear on hover. A bounding box may cover disconnected evidence; it does not indicate an object or occupied volume. Current point colors are display RGB when supplied; otherwise gray dots are current visual-token anchors. Observed TCP/recorded expert values are visualization labels, not extra policy inputs.</p>
<details><summary>Selection, repeatability and action-difference metadata</summary><pre id="metrics"></pre></details>
<script>__PLOTLY__</script><script>
const data=__DATA__,byId=id=>document.getElementById(id),colors={base:'#88929e',immediate_completed:'#4569b0',packet_best:'#ce512e',packet_worst:'#8a57b6',random_packet:'#9c8a43',dispersed:'#337f85',uniform:'#557a39',teacher_full:'#142e49'};
data.seeds.forEach((seed,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${i+1} (seed ${seed})`;byId('draw').appendChild(o)});
Object.keys(data.history_geometry).forEach(name=>{const o=document.createElement('option');o.value=name;o.textContent=`${name} (${data.history_token_counts[name]} records)`;byId('context').appendChild(o)});if(data.history_geometry.packet_best)byId('context').value='packet_best';
byId('title').textContent=`${data.student} · ${data.state.task||data.task} ${data.state.episode_id||''} · decision ${data.state.decision_index??''} · ${data.selection.reader} B${data.selection.final_budget}`;
function trace(points,name,color,mode='lines+markers',dash='solid'){return {type:'scatter3d',mode,x:points.map(p=>p?p[0]:null),y:points.map(p=>p?p[1]:null),z:points.map(p=>p?p[2]:null),name,line:{color,width:5,dash},marker:{color,size:3},connectgaps:false};}
function boxPoints(lo,hi){const p=[];for(let axis=0;axis<3;axis++)for(let a=0;a<2;a++)for(let b=0;b<2;b++){const q=[...lo],r=[...lo],other=[0,1,2].filter(x=>x!==axis);q[other[0]]=r[other[0]]=a?hi[other[0]]:lo[other[0]];q[other[1]]=r[other[1]]=b?hi[other[1]]:lo[other[1]];r[axis]=hi[axis];p.push(q,r,null)}return p;}
function render(){const draw=Number(byId('draw').value),prefix=data.prefix_targets,traces=[],name=byId('context').value,geo=data.history_geometry[name],source=byId('source').value;
const cloud=data.scene_points||data.current_visual_anchors;const scene=trace(cloud,data.scene_points?'Current sampled point cloud':'Current visual-token anchors','#aeb8c1','markers');scene.marker.size=data.scene_points?1.4:3;scene.opacity=.45;if(data.scene_points&&cloud.length&&cloud[0].length===6){const high=Math.max(...cloud.slice(0,1000).flatMap(p=>p.slice(3))),scale=high<=1.001?255:1;scene.marker.color=cloud.map(p=>`rgb(${p.slice(3).map(v=>Math.round(Math.max(0,Math.min(255,v*scale)))).join(',')})`)}traces.push(scene);
traces.push(trace([data.current_tcp],'Observed TCP','#111111','markers'));
const expert=data.expert_tcp_path.map((p,i)=>data.expert_tcp_valid[i]?p:null);traces.push(trace(expert.slice(0,prefix),'Recorded expert · prefix','#169b48'));if(expert.length>prefix)traces.push(trace(expert.slice(prefix-1),'Recorded expert · tail','#169b48','lines+markers','dash'));
for(const [key,paths] of Object.entries(data.paths)){const p=paths[draw];traces.push(trace(p.slice(0,prefix),key+` · first ${prefix}`,colors[key]||'#333'));if(p.length>prefix)traces.push(trace(p.slice(prefix-1),key+' · tail',colors[key]||'#333','lines+markers','dash'));}
if(byId('history').checked){const ids=geo.anchors.map((p,i)=>i).filter(i=>source==='all'||geo.sources[i]===source);const h=trace(ids.map(i=>geo.anchors[i]),'Selected history · '+name,colors[name]||'#333','markers');h.marker.size=ids.map(i=>Math.min(12,4+Math.log2(geo.counts[i])));h.marker.symbol='diamond';h.customdata=ids.map(i=>[geo.sources[i],geo.modalities[i],geo.time_min[i],geo.time_max[i],geo.counts[i],geo.disconnected[i]]);h.hovertemplate='%{customdata[0]} / %{customdata[1]}<br>Source time %{customdata[2]:.2f}–%{customdata[3]:.2f} s<br>%{customdata[4]} leaves; disconnected=%{customdata[5]}<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>';traces.push(h);if(byId('bounds').checked){const boxes=ids.flatMap(i=>boxPoints(geo.bounds_min[i],geo.bounds_max[i]));const b=trace(boxes,'Historical support bounds',colors[name]||'#333','lines');b.line.width=1;b.opacity=.2;traces.push(b)}}
Plotly.react('plot',traces,{margin:{l:0,r:0,t:10,b:0},scene:{aspectmode:'data',xaxis:{title:'World x (m)'},yaxis:{title:'World y (m)'},zaxis:{title:'World z (m)'}},legend:{orientation:'h'},uirevision:data.state.state_id||'same-state'},{responsive:true,displaylogo:false});
byId('status').textContent=`${data.selection.background}-record base; completed alternatives ${data.selection.final_budget} records. ${prefix} executed targets; ${data.integration_steps} integration steps. Fresh generation is independent of the discovery and confirmation FM draws. Teacher reference: ${data.teacher_status}.`;
const table=document.createElement('table'),head=document.createElement('tr');['Context','Records','Generated expert MSE','Prefix MSE','Tail MSE'].forEach(x=>{const th=document.createElement('th');th.textContent=x;head.appendChild(th)});table.appendChild(head);Object.entries(data.metrics).forEach(([key,v])=>{const tr=document.createElement('tr');[key,data.history_token_counts[key],v.expert_full[draw],v.expert_prefix[draw],v.expert_tail===null?'—':v.expert_tail[draw]].forEach((value,i)=>{const td=document.createElement('td');td.textContent=i>1&&typeof value==='number'?value.toPrecision(5):value;tr.appendChild(td)});table.appendChild(tr)});byId('table').replaceChildren(table);byId('metrics').textContent=JSON.stringify({selection:data.selection,comparisons:data.comparisons,note:data.note,identity:data.identity},null,2);}
['draw','context','source','history','bounds'].forEach(id=>byId(id).onchange=render);render();</script></body></html>"""
