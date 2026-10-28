(() => {
  "use strict";
  const P = JSON.parse(document.getElementById("reset-chunks-data").textContent);
  const M = P.models.find(m => m.id === window.RESET_CHUNKS_MODEL);
  const $ = id => document.getElementById(id);
  const colors = ["#2563eb", "#d97706", "#9333ea", "#0d9488", "#e11d48", "#0891b2", "#9a6700", "#52525b", "#be185d"];
  const cloud = $("cloud"), response = $("response");
  let activeView = "world", camera = null, ready = false, rendering = Promise.resolve();
  const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const xyz = rows => ({x: rows.map(p=>p[0]), y: rows.map(p=>p[1]), z: rows.map(p=>p[2])});
  const norm = a => Math.hypot(...a), minus = (a,b) => a.map((v,i)=>v-b[i]);
  const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
  const pred = (s,seed) => s.predictions[M.id][String(seed)];
  const color = i => colors[i % colors.length];
  const currentScene = () => P.scenes[Number($("scene").value)];
  const currentSeed = () => Number($("seed").value);
  const fmt = (v, n=3) => Number(v).toFixed(n);
  const error = e => {$("runtime-error").hidden=false; $("runtime-error").textContent=`Unable to render this artifact: ${e.message || e}`; window.__resetChunksError=String(e);};

  function cubeCorners(cube) {
    const [w,x,y,z] = cube.q, n = Math.hypot(w,x,y,z), [a,b,c,d] = [w,x,y,z].map(v=>v/n);
    const R = [[1-2*(c*c+d*d),2*(b*c-a*d),2*(b*d+a*c)],[2*(b*c+a*d),1-2*(b*b+d*d),2*(c*d-a*b)],[2*(b*d-a*c),2*(c*d+a*b),1-2*(b*b+c*c)]];
    const half = typeof cube.half_size === "number" ? [cube.half_size,cube.half_size,cube.half_size] : cube.half_size;
    return Array.from({length:8},(_,i)=>{
      const q=half.map((v,j)=>(i&(1<<j))?v:-v);
      return cube.p.map((v,j)=>v+R[j].reduce((s,r,k)=>s+r*q[k],0));
    });
  }
  function cubeTrace(scene, i, selected) {
    const corners=cubeCorners(scene.cube), lines=[];
    for(let k=0;k<8;k++) for(let bit=1;bit<=4;bit*=2) if(!(k&bit)) lines.push(corners[k],corners[k|bit],[null,null,null]);
    return {type:"scatter3d",mode:"lines",...xyz(lines),line:{color:color(i),width:selected?5:3,dash:selected?"solid":"dash"},opacity:selected?1:.55,name:`Cube: ${scene.label}`,showlegend:false,hoverinfo:"skip"};
  }
  function targetTraces(scene, i, seed, selected) {
    const pr=pred(scene,seed), c=color(i), d=pr.tcp_xyz;
    const hover=d.map((p,k)=>`${esc(scene.label)}<br><b>Target ${k+1} / ${P.horizon}</b>${k<P.execute_steps?" · before next replan":""}<br>TCP: ${p.map(v=>fmt(v,4)).join(", ")} m<br>Gripper: ${pr.physical_actions[k][7]>0?"open (+1)":"close (−1)"}<br>Joint targets (rad):<br>${pr.physical_actions[k].slice(0,7).map(v=>fmt(v,4)).join(", ")}`);
    const shared={type:"scatter3d",name:scene.label,legendgroup:scene.id,line:{color:c},marker:{color:c},hovertemplate:"%{hovertext}<extra></extra>"};
    const traces=[{...shared,...xyz(d),mode:$("show-labels").checked?"lines+markers+text":"lines+markers",text:$("show-labels").checked?d.map((_,k)=>String(k+1)):undefined,textfont:{size:10,color:c},textposition:"top center",line:{color:c,width:selected?5:3},marker:{color:c,size:3,symbol:"diamond",opacity:selected?1:.75},hovertext:hover,showlegend:true},
      {...shared,...xyz(d.slice(0,P.execute_steps)),mode:"lines+markers",line:{color:c,width:selected?8:6},marker:{color:c,size:selected?6:5,symbol:"circle",line:{color:"white",width:.7}},hovertext:hover.slice(0,P.execute_steps),showlegend:false}];
    if($("show-axes").checked && selected && pr.tcp_pose) {
      const axes=["#dc2626","#16a34a","#2563eb"], indices=[...new Set([0,P.execute_steps-1,7,11,P.horizon-1])].filter(k=>k<P.horizon);
      for(let axis=0;axis<3;axis++){
        const segments=[];
        for(const k of indices){const T=pr.tcp_pose[k], p=d[k], end=p.map((v,j)=>v+.025*T[j][axis]);segments.push(p,end,[null,null,null]);}
        traces.push({type:"scatter3d",mode:"lines",...xyz(segments),line:{color:axes[axis],width:3},name:`Local ${"XYZ"[axis]}`,showlegend:false,hoverinfo:"skip"});
      }
    }
    return traces;
  }
  function bounds(points, minExtent=.08, padding=.06) {
    const lows=[0,1,2].map(i=>Math.min(...points.map(p=>p[i]))), highs=[0,1,2].map(i=>Math.max(...points.map(p=>p[i])));
    return lows.map((lo,i)=>{const center=(lo+highs[i])/2, half=Math.max(minExtent,highs[i]-lo)/2+padding;return [center-half,center+half];});
  }
  function ranges(seed) {
    if(activeView==="cube") return currentScene().cube.p.map(v=>[v-.18,v+.18]);
    const targets=P.scenes.flatMap(s=>pred(s,seed).tcp_xyz), tcp=currentScene().current_tcp_pose.slice(0,3).map(row=>row[3]);
    if(activeView==="actions") {
      const extra=[];
      if($("show-axes").checked) {
        const pr=pred(currentScene(),seed);
        if(pr.tcp_pose) for(const k of [...new Set([0,P.execute_steps-1,7,11,P.horizon-1])].filter(k=>k<P.horizon))
          for(let axis=0;axis<3;axis++) extra.push(pr.tcp_xyz[k].map((v,j)=>v+.025*pr.tcp_pose[k][j][axis]));
      }
      return bounds([...targets,tcp,...extra],.025,.008);
    }
    // Fixed ranges across all checkpoints and cube/noise selections for honest visual comparison.
    const everyTarget=P.scenes.flatMap(s=>P.models.flatMap(m=>P.noise_seeds.flatMap(n=>s.predictions[m.id][String(n)].tcp_xyz)));
    return bounds([...P.scenes.flatMap(s=>s.points.map(p=>p.slice(0,3))),...everyTarget,tcp],.1,.025);
  }
  function sceneMetadata(scene,index) {
    const displacement=norm(scene.shift_xyz_m)*1000;
    $("scene-info").innerHTML=`<strong>${esc(scene.label)}</strong><br>Cube shift: ${scene.shift_xyz_m.map(v=>fmt(v*1000,0)).join(", ")} mm<br>Cube center: ${scene.cube.p.map(v=>fmt(v,3)).join(", ")} m<br>${scene.points.length.toLocaleString()} observed points${scene.cube_point_count!==undefined?` · ${scene.cube_point_count} cube points`:""}${scene.inside_original_cube_center_sampling_xy_bounds===false?'<p style="color:#9a530b"><b>Outside the original cube-center spawn range.</b></p>':""}`;
    $("scene-legend").innerHTML=P.scenes.map((s,i)=>`<div class="legend-row ${i===index?"selected":""}"><span class="swatch" style="background:${color(i)}"></span>${esc(s.label)}${i===index?" · shown cloud":""}</div>`).join("");
    $("camera-image").src=scene.rgb_data_uri || "";
    $("camera-image").parentElement.hidden=!scene.rgb_data_uri;
  }
  function chartAndTable(seed,selected) {
    const base=pred(P.scenes[0],seed), lines=[];let max=0;
    $("metrics").innerHTML=P.scenes.map((s,i)=>{
      const pr=pred(s,seed), changes=pr.tcp_xyz.map((p,k)=>1000*norm(minus(p,base.tcp_xyz[k]))), flips=pr.physical_actions.filter((a,k)=>a[7]!==base.physical_actions[k][7]).length;
      max=Math.max(max,...changes);
      lines.push({type:"scatter",mode:"lines+markers",name:s.label,x:changes.map((_,k)=>k+1),y:changes,line:{color:color(i),width:i===selected?3:1.7},marker:{size:i===selected?6:4},hovertemplate:`${esc(s.label)}<br>Target %{x}: %{y:.4f} mm<extra></extra>`});
      return `<tr class="${i===selected?"selected":""}"><td><span class="swatch" style="background:${color(i)}"></span>${esc(s.label)}</td><td>${fmt(norm(s.shift_xyz_m)*1000,0)} mm</td><td>${fmt(mean(changes.slice(0,P.execute_steps)))} mm</td><td>${fmt(mean(changes))} mm</td><td>${fmt(Math.max(...changes))} mm</td><td>${flips} / ${P.horizon}</td></tr>`;
    }).join("");
    const layout={margin:{l:72,r:30,t:18,b:50},paper_bgcolor:"white",plot_bgcolor:"white",font:{family:"system-ui, sans-serif",size:11,color:"#53667a"},xaxis:{title:{text:"Predicted command within first chunk"},tickmode:"linear",dtick:1,range:[.75,P.horizon+.25],gridcolor:"#e8edf4",zeroline:false},yaxis:{title:{text:"TCP change from baseline (mm)"},range:[0,Math.max(.01,max*1.15)],gridcolor:"#e8edf4",zeroline:false},showlegend:false,hovermode:"closest",shapes:[{type:"rect",xref:"x",yref:"paper",x0:.75,x1:P.execute_steps+.5,y0:0,y1:1,fillcolor:"#edf3fb",line:{width:0},layer:"below"}],uirevision:"response"};
    return Plotly.react(response,lines,layout,{responsive:true,displaylogo:false,modeBarButtonsToRemove:["select2d","lasso2d"]});
  }
  async function draw() {
    const index=Number($("scene").value), scene=P.scenes[index], seed=currentSeed(), traces=[];
    sceneMetadata(scene,index);
    if($("show-cloud").checked) traces.push({type:"scatter3d",mode:"markers",...xyz(scene.points),marker:{size:Number($("point-size").value),color:scene.points.map(p=>`rgb(${p.slice(3,6).map(v=>Math.round(Math.max(0,Math.min(1,v))*255)).join(",")})`),opacity:.9},name:"Observed RGB-D points",hovertemplate:"Observed point<br>x %{x:.4f} m<br>y %{y:.4f} m<br>z %{z:.4f} m<extra></extra>",showlegend:false});
    const indices=$("all-paths").checked?P.scenes.map((_,i)=>i):[index];
    // Draw selected predictions last so they remain distinguishable at overlap.
    const ordered=indices.filter(i=>i!==index).concat(index);
    for(const i of ordered){const s=P.scenes[i];if($("show-cubes").checked)traces.push(cubeTrace(s,i,i===index));traces.push(...targetTraces(s,i,seed,i===index));}
    const tcp=scene.current_tcp_pose.slice(0,3).map(row=>row[3]);
    traces.push({type:"scatter3d",mode:"markers+text",...xyz([tcp]),marker:{color:"#172638",size:7,symbol:"diamond",line:{color:"white",width:2}},text:["Observed TCP"],textfont:{color:"#172638",size:11},textposition:"bottom right",name:"Observed TCP at reset",hovertemplate:"Observed TCP at reset<br>x %{x:.4f} m<br>y %{y:.4f} m<br>z %{z:.4f} m<extra></extra>",showlegend:false});
    const r=ranges(seed), axis=(i)=>({title:{text:`${"XYZ"[i]} (m)`,font:{size:11}},range:r[i],backgroundcolor:"#fbfcfe",gridcolor:"#dce5ee",zerolinecolor:"#c5d3e3",showbackground:true,tickfont:{size:10}});
    const defaultCamera={eye:{x:1.25,y:-1.65,z:1.05},up:{x:0,y:0,z:1},center:{x:0,y:0,z:0}};
    const layout={margin:{l:0,r:0,t:0,b:0},paper_bgcolor:"white",font:{family:"system-ui,sans-serif",color:"#53667a",size:11},scene:{xaxis:axis(0),yaxis:axis(1),zaxis:axis(2),aspectmode:"data",camera:camera||defaultCamera,dragmode:"orbit"},legend:{orientation:"h",x:.02,y:.02,bgcolor:"rgba(255,255,255,.86)",font:{size:10},itemclick:false,itemdoubleclick:false},uirevision:activeView,showlegend:true};
    await Promise.all([Plotly.react(cloud,traces,layout,{responsive:true,displaylogo:false,scrollZoom:true,toImageButtonOptions:{format:"png",filename:`${M.id}_${scene.id}_seed${seed}`},modeBarButtonsToRemove:["resetCameraLastSave3d"]}),chartAndTable(seed,index)]);
    if(!ready){
      cloud.on("plotly_relayout", event=>{if(event["scene.camera"])camera=event["scene.camera"];});
      cloud.on("plotly_webglcontextlost",()=>error("The browser lost its WebGL context. Reload or enable hardware acceleration."));
      ready=true;
    }
    window.__resetChunksReady=true;
    window.__resetChunksState={model:M.id,scene:scene.id,seed,traceCount:traces.length,pointCount:scene.points.length};
  }
  function update(){window.__resetChunksReady=false;rendering=rendering.then(draw).catch(error);}
  function preset(view){
    activeView=view;
    camera=view==="top"?{eye:{x:0,y:0,z:2.4},up:{x:0,y:1,z:0},center:{x:0,y:0,z:0}}:null;
    document.querySelectorAll("[data-view]").forEach(b=>b.classList.toggle("active",b.dataset.view===view));
    // Clear Plotly's preserved camera only for an explicit preset request.
    if(ready)Plotly.relayout(cloud,{"scene.camera":camera||{eye:{x:1.25,y:-1.65,z:1.05},up:{x:0,y:0,z:1},center:{x:0,y:0,z:0}}});
    update();
  }
  try {
    if(!M)throw Error("Checkpoint identity is absent from the payload");
    P.scenes.forEach((s,i)=>$("scene").add(new Option(s.label,i)));
    P.noise_seeds.forEach(seed=>$("seed").add(new Option(String(seed),seed)));
    $("precision-badge").textContent=`Encoder ${P.precision.encoder} · context/decoder ${P.precision.context_decoder}`;
    document.querySelectorAll(".model-nav a").forEach(a=>a.classList.toggle("active",a.getAttribute("href")===`${M.id}.html`));
    ["scene","seed","all-paths","show-cloud","show-cubes","show-axes","show-labels"].forEach(id=>$(id).addEventListener("change",update));
    $("point-size").addEventListener("input",()=>{$("point-size-value").value=$("point-size").value;update();});
    document.querySelectorAll("[data-view]").forEach(b=>b.addEventListener("click",()=>preset(b.dataset.view)));
    document.querySelector('[data-view="world"]').classList.add("active");
    $("notes").textContent=(P.notes||[]).join(" ");
    $("provenance").textContent=JSON.stringify({task:P.task,split:P.split,episode_idx:P.episode_idx,decision_step:P.decision_step,checkpoint:M,precision:P.precision,noise_seeds:P.noise_seeds,horizon:P.horizon,execute_steps:P.execute_steps,metadata:P.metadata||P.provenance},null,2);
    $("download").addEventListener("click",()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(P,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="reset_cube_chunks.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
    update();
  } catch(e){error(e);}
})();
