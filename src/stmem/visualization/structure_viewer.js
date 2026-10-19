/* Offline inspector. Never recompute model neighborhoods or archive topology. */
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const error = reason => {
    const banner = $("runtime-error");
    banner.hidden = false;
    banner.textContent = "The structure inspector could not finish rendering. " +
      "Try a browser with WebGL enabled. The embedded JSON remains available through Export JSON.\n" +
      String(reason && reason.message ? reason.message : reason);
    document.documentElement.dataset.inspectorError = "true";
  };
  window.addEventListener("error", event => error(event.error || event.message));
  window.addEventListener("unhandledrejection", event => error(event.reason));
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  };
  const number = (value, places = 3) => Number.isFinite(value)
    ? Number(value).toLocaleString("en", {maximumFractionDigits: places}) : "—";
  const precise = value => Number.isFinite(value) ? Number(value).toPrecision(7) : "—";
  const titleCase = value => String(value).replace(/(^|[_ /])([a-z])/g, (_, a, b) => a + b.toUpperCase());
  const short = value => String(value).length > 52 ? String(value).slice(0, 23) + "…" + String(value).slice(-24) : String(value);
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const option = (select, value, label) => {
    const node = element("option", "", label);
    node.value = value;
    select.append(node);
    return node;
  };
  const kv = (container, name, value) => {
    const item = element("div");
    item.append(element("dt", "", name), element("dd", "", value));
    container.append(item);
  };
  const displaySource = block => block.kind === "current" ? "current" : block.source;
  let payload;
  try { payload = JSON.parse($("structure-data").textContent); } catch (reason) { error(reason); return; }
  $("download").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: "application/json"}));
    const anchor = element("a");
    anchor.href = url;
    anchor.download = "chronotree-structure.json";
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  if (!window.Plotly) { error("The bundled Plotly runtime did not initialize."); return; }
  const detailed = payload.blocks.filter(block => block.detailed && block.details);
  const nodes = new Map(payload.archive.nodes.map(node => [node.id, node]));
  const blocks = new Map(payload.blocks.map(block => [block.id, block]));
  const leafOwners = new Map();
  payload.blocks.forEach(block => block.leaf_ids.forEach(id => leafOwners.set(id, block.id)));
  const sources = ["conditioning", "execution", "current"].filter(source => detailed.some(block => displaySource(block) === source));
  const firstSource = sources.includes("execution") ? "execution" : sources[0];
  const state = {
    view: "neighborhood", source: firstSource, block: null, center: 0, frame: "all", separation: 0,
    pointSize: 2, within: true, cross: true, events: true, centers: true,
    hierarchyGroup: "all", hierarchyLabels: false, node: null, event: null,
  };
  const plotConfig = {
    responsive: true, displaylogo: false, scrollZoom: true,
    modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d", "select2d", "autoScale2d"],
    toImageButtonOptions: {format: "png", filename: "chronotree-structure", scale: 2},
  };
  let playing = null, rendering = false, requested = false;
  function stopAnimation() {
    if (playing) clearInterval(playing);
    playing = null;
    $("play").textContent = "▶";
    $("play").setAttribute("aria-label", "Animate observed frames");
  }
  function selectBlock(id) {
    state.block = id;
    state.frame = "all";
    state.event = null;
    const block = blocks.get(id);
    const centers = block.details.centers;
    state.center = Math.max(0, centers.findIndex(center => center.valid));
    $("frame").replaceChildren();
    option($("frame"), "all", `All ${block.frames.length} frames`);
    block.frames.forEach((frame, index) => option($("frame"), index,
      `Frame ${frame.source_index} · t=${number(frame.time)}`));
    $("frame").value = "all";
    $("play").disabled = block.frames.length < 2;
    $("center").replaceChildren();
    centers.forEach((center, index) => option($("center"), index,
      `${String(center.index).padStart(2, "0")} · frame ${block.frames[center.frame_index]?.source_index ?? center.frame_index}` +
      (center.valid ? ` · ${center.neighbor_mask.filter(Boolean).length} neighbors` : " · empty / no leaf")));
    $("center").value = state.center;
    $("center").disabled = !centers.length;
    $("block-description").textContent = block.kind === "current"
      ? `Current observation · ${payload.config.current_tokens} center slots · outside the archive.`
      : `${block.completed ? "Completed block" : "Pending execution prefix"} · ${block.frames.length}/${payload.config.block_size} frames · ${block.visual_leaf_ids.length} visual + ${block.event_leaf_ids.length} event leaves.`;
  }
  function selectSource(source) {
    stopAnimation();
    state.source = source;
    $("source").value = source;
    $("block").replaceChildren();
    const available = detailed.filter(block => displaySource(block) === source);
    available.forEach((block, index) => option($("block"), block.id,
      source === "current" ? `Current · t=${number(block.frames[0].time)}` :
        `${block.completed ? "Block" : "Pending"} · t=${number(block.frames[0].time)}–${number(block.frames.at(-1).time)}`));
    const chosen = available.at(-1);
    if (chosen) { $("block").value = chosen.id; selectBlock(chosen.id); }
  }
  function setFrame(value) {
    state.frame = value;
    $("frame").value = value;
    state.event = null;
    if (value !== "all") {
      const centers = blocks.get(state.block).details.centers;
      const candidate = centers.findIndex(center => center.frame_index === Number(value) && center.valid);
      if (candidate >= 0) { state.center = candidate; $("center").value = candidate; }
    }
    requestRender();
  }
  function switchView(view) {
    stopAnimation();
    state.view = view;
    for (const name of ["neighborhood", "hierarchy"]) {
      const active = view === name;
      $("tab-" + name).classList.toggle("active", active);
      $("tab-" + name).setAttribute("aria-selected", active);
      $(name + "-view").hidden = !active;
      $(name + "-controls").hidden = !active;
    }
    requestRender();
  }
  function selectedPoint(point, time, firstTime) {
    return [point[0] + (time - firstTime) * state.separation, point[1], point[2]];
  }
  function trace3d(coords, extra) {
    return {type: "scatter3d", x: coords.map(p => p[0]), y: coords.map(p => p[1]),
      z: coords.map(p => p[2]), showlegend: false, ...extra};
  }
  function colors(points) {
    return points.map(point => `rgb(${point.slice(3, 6).map(c => Math.round(Math.max(0, Math.min(1, c)) * 255)).join(",")})`);
  }
  function panelHeader(title, subtitle, badge, pending = false) {
    const heading = element("div", "selection-heading");
    const description = element("div");
    description.append(element("h2", "", title), element("div", "selection-subtitle", subtitle));
    heading.append(description, element("span", "pill" + (pending ? " pending" : ""), badge));
    $("selection").replaceChildren(heading);
  }
  function eventPanel(event, block) {
    panelHeader("Robot-event leaf", event.leaf_id, "State + available command");
    const stats = element("dl", "detail-stats");
    kv(stats, "Source frame", block.frames[event.frame_index].source_index);
    kv(stats, "Recorded time", number(event.time));
    kv(stats, "Command available", event.action_available ? "Yes" : "No");
    kv(stats, "Leaf mass", "1");
    $("selection").append(stats, element("p", "help", "This event encodes observed robot state and a separately masked preceding command. It is not repeated on every point and has no point-pooling edges."));
    if (event.proprio) $("selection").append(element("p", "inline-code", "Observed proprioception: [" + event.proprio.map(v => number(v, 5)).join(", ") + "]"));
  }
  function centerPanel(center, block, visibleEdges) {
    if (!center) {
      panelHeader("No visual centers", "This block has no exported center slots.", "Empty evidence");
      return;
    }
    const details = block.details;
    const allowed = center.neighbor_indices.map((point, index) => ({point, index,
      allowed: Boolean(center.neighbor_mask[index]), weight: center.weights[index]}));
    const actual = allowed.filter(row => row.allowed);
    const cross = actual.filter(row => details.point_frame_indices[row.point] !== center.frame_index);
    panelHeader(`Visual center ${center.index}`, center.leaf_id || "No archived leaf: empty or invalid neighborhood",
      center.valid ? (block.kind === "current" ? "Current scene" : block.completed ? "Completed evidence" : "Pending prefix") : "Invalid / zero content",
      block.kind !== "current" && !block.completed);
    const stats = element("dl", "detail-stats");
    kv(stats, "Source frame", block.frames[center.frame_index]?.source_index ?? center.frame_index);
    kv(stats, "Allowed neighbors", `${actual.length} / ${allowed.length} slots`);
    kv(stats, "Cross-frame", cross.length);
    kv(stats, "Weight sum", number(actual.reduce((sum, row) => sum + row.weight, 0), 7));
    kv(stats, "Edges shown", visibleEdges);
    $("selection").append(stats);
    const note = element("p", "help", state.frame === "all"
      ? "All valid finite block points are shown. Edge filters affect visibility only; the table retains every exported capped-neighbor slot."
      : "The frame filter limits the RGB context cloud and center display. Selected incoming neighbors remain visible, including cross-frame support.");
    $("selection").append(note);
    const more = element("details");
    more.append(element("summary", "details-toggle", `Inspect all ${allowed.length} neighbor slots · indices, masks and exact weights`));
    const wrap = element("div", "edge-table-wrap");
    const table = element("table");
    const head = element("thead"), row = element("tr");
    ["Slot", "Point index", "Source frame", "Δt", "Allowed", "Weight α"].forEach(label => row.append(element("th", "", label)));
    head.append(row); table.append(head);
    const body = element("tbody");
    for (const item of allowed) {
      const row = element("tr", item.allowed ? "" : "masked-row");
      const frame = details.point_frame_indices[item.point];
      const values = [item.index, item.point, block.frames[frame]?.source_index ?? "—",
        Number.isFinite(details.point_times[item.point]) ? number(details.point_times[item.point] - center.time) : "—",
        item.allowed ? "yes" : "masked", precise(item.weight)];
      values.forEach(value => row.append(element("td", "", value)));
      body.append(row);
    }
    table.append(body); wrap.append(table); more.append(wrap); $("selection").append(more);
  }
  function bindCloudClicks() {
    const plot = $("cloud");
    if (plot._chronotreeBound) return;
    plot._chronotreeBound = true;
    plot.on("plotly_click", event => {
      const data = event.points?.[0]?.customdata;
      if (!data) return;
      if (data.type === "center") {
        state.center = data.index; state.event = null; $("center").value = data.index;
        requestRender();
      } else if (data.type === "event") {
        state.event = data.index; requestRender();
      }
    });
    plot.on("plotly_webglcontextlost", () => error("WebGL context lost; reload or try a browser with hardware acceleration."));
  }
  async function drawCloud() {
    const block = blocks.get(state.block);
    if (!block) throw new Error("No detailed point-cloud block was exported.");
    const details = block.details;
    const firstTime = block.frames[0].time;
    const visibleFrame = frame => state.frame === "all" || frame === Number(state.frame);
    const visiblePoints = details.points.map((point, index) => index).filter(index => visibleFrame(details.point_frame_indices[index]));
    const traces = [trace3d(visiblePoints.map(index => selectedPoint(details.points[index], details.point_times[index], firstTime)), {
      name: "Observed RGB", mode: "markers", meta: {kind: "observed_points", block_id: block.id},
      marker: {size: state.pointSize, color: colors(visiblePoints.map(index => details.points[index])), opacity: .66},
      customdata: visiblePoints.map(index => ({type: "point", index, frame: details.point_frame_indices[index]})),
      text: visiblePoints.map(index => `Observed point ${index}<br>source frame ${block.frames[details.point_frame_indices[index]].source_index}<br>t=${number(details.point_times[index])} recorded frames` +
        `<br>World xyz=[${details.points[index].slice(0, 3).map(value => number(value, 5)).join(", ")}] m`),
      hovertemplate: "%{text}" + (state.separation ? "<br>Display x=%{x:.3f} m (time offset applied)" : "") + "<extra></extra>",
    })];
    const center = details.centers[state.center];
    let shownEdges = 0;
    if (center?.candidate_valid) {
      const target = selectedPoint(center.anchor, center.time, firstTime);
      const edges = center.neighbor_indices.map((point, index) => ({point, index, weight: center.weights[index]}))
        .filter(row => center.neighbor_mask[row.index] && (details.point_frame_indices[row.point] === center.frame_index ? state.within : state.cross));
      shownEdges = edges.length;
      const maxWeight = Math.max(...edges.map(edge => edge.weight), 1e-12);
      // Bucketing line color is display only. Every edge retains its actual weight.
      const buckets = Array.from({length: 6}, () => ({x: [], y: [], z: [], text: []}));
      edges.forEach(edge => {
        const source = selectedPoint(details.points[edge.point], details.point_times[edge.point], firstTime);
        const bucket = buckets[Math.min(5, Math.floor(edge.weight / maxWeight * 5))];
        for (const [name, dimension] of [["x", 0], ["y", 1], ["z", 2]]) bucket[name].push(source[dimension], target[dimension], null);
        const label = `Point ${edge.point} → center ${center.index}<br>weight α=${precise(edge.weight)}<br>` +
          (details.point_frame_indices[edge.point] === center.frame_index ? "same frame" : "cross frame");
        bucket.text.push(label, label, "");
      });
      const palette = ["#96b5d2", "#6da5ca", "#3c91bd", "#167da7", "#116b8f", "#0a5374"];
      buckets.forEach((bucket, index) => {
        if (bucket.x.length) traces.push({type: "scatter3d", ...bucket, mode: "lines", showlegend: false,
          meta: {kind: "incoming_edges", edge_count: bucket.x.length / 3},
          line: {color: palette[index], width: 2 + index * .35}, opacity: .5 + index * .07,
          hovertemplate: "%{text}<extra></extra>"});
      });
      if (edges.length) traces.push(trace3d(edges.map(edge => selectedPoint(details.points[edge.point], details.point_times[edge.point], firstTime)), {
        mode: "markers", name: "Allowed neighbors", meta: {kind: "selected_neighbors"},
        marker: {size: state.pointSize + 1.6, color: edges.map(edge => edge.weight), colorscale: "Viridis", cmin: 0, cmax: maxWeight,
          colorbar: {title: {text: "Weight α", font: {size: 10}}, thickness: 9, len: .38, x: .98, y: .25, tickfont: {size: 9}}, showscale: true, opacity: .96},
        text: edges.map(edge => `Point ${edge.point}<br>weight α=${precise(edge.weight)}<br>source frame ${block.frames[details.point_frame_indices[edge.point]].source_index}` +
          `<br>World xyz=[${details.points[edge.point].slice(0, 3).map(value => number(value, 5)).join(", ")}] m`),
        hovertemplate: "%{text}<extra></extra>",
      }));
      if (state.centers) traces.push(trace3d([target], {
        mode: "markers", name: "Selected center", meta: {kind: "selected_center"},
        marker: {size: 10, color: "#f59d50", symbol: "diamond", line: {color: "#fff", width: 1}},
        customdata: [{type: "center", index: state.center}], hovertemplate: `Selected center ${center.index}<extra></extra>`,
      }));
    }
    if (state.centers) {
      const centers = details.centers.map((item, index) => ({...item, uiIndex: index})).filter(item => item.candidate_valid && visibleFrame(item.frame_index));
      traces.push(trace3d(centers.map(item => selectedPoint(item.anchor, item.time, firstTime)), {
        mode: "markers", name: "Visual centers", meta: {kind: "centers"},
        marker: {size: 5.5, color: centers.map(item => item.uiIndex === state.center ? "#f59d50" : css("--accent")), symbol: "diamond", opacity: 1, line: {color: css("--panel"), width: 1}},
        customdata: centers.map(item => ({type: "center", index: item.uiIndex})),
        text: centers.map(item => `Visual center ${item.index}<br>frame ${block.frames[item.frame_index].source_index}<br>${item.neighbor_mask.filter(Boolean).length} allowed neighbors`),
        hovertemplate: "%{text}<extra></extra>",
      }));
    }
    if (state.events) {
      const events = details.events.map((item, index) => ({...item, uiIndex: index})).filter(item => visibleFrame(item.frame_index));
      if (events.length) traces.push(trace3d(events.map(item => selectedPoint(item.anchor, item.time, firstTime)), {
        mode: "markers", name: "Robot events", meta: {kind: "robot_events"},
        marker: {size: 7, color: css("--purple"), symbol: "square", line: {color: css("--panel"), width: 1}},
        customdata: events.map(item => ({type: "event", index: item.uiIndex})),
        text: events.map(item => `Robot-event leaf<br>frame ${block.frames[item.frame_index].source_index}<br>preceding command ${item.action_available ? "available" : "unavailable"}`),
        hovertemplate: "%{text}<extra></extra>",
      }));
      if (block.kind === "current" && payload.current.proprio) {
        const anchor = payload.current.proprio.slice(7, 10);
        traces.push(trace3d([selectedPoint(anchor, firstTime, firstTime)], {
          mode: "markers", meta: {kind: "current_proprio"},
          marker: {size: 7, color: css("--purple"), symbol: "square"},
          hovertemplate: "Current observed end-effector position<br>Direct proprioception input; not an archive event leaf<extra></extra>",
        }));
      }
    }
    const axis = name => ({title: {text: name, font: {size: 11}}, gridcolor: css("--line"), zerolinecolor: css("--line"),
      tickfont: {size: 9, color: css("--muted")}, showbackground: false});
    const layout = {
      paper_bgcolor: css("--plot"), plot_bgcolor: css("--plot"), font: {family: "system-ui, sans-serif", color: css("--ink")},
      margin: {l: 0, r: 0, t: 5, b: 0}, showlegend: false,
      scene: {xaxis: axis(state.separation ? "x + display time offset (m)" : "x (m)"), yaxis: axis("y (m)"), zaxis: axis("z (m)"),
        aspectmode: "data", bgcolor: css("--plot"), dragmode: "orbit", camera: {eye: {x: 1.5, y: -1.8, z: 1.25}}},
      uirevision: "cloud-" + block.id, hoverlabel: {font: {family: "system-ui, sans-serif", size: 11}},
    };
    await Plotly.react($("cloud"), traces, layout, plotConfig);
    bindCloudClicks();
    $("cloud-caption").textContent = `${number(visiblePoints.length)} / ${number(details.points.length)} context points · ${shownEdges} incoming edges shown` +
      (state.separation ? ` · display offset ${state.separation.toFixed(2)} m/frame` : " · world coordinates");
    $("view-status").textContent = `${titleCase(displaySource(block))} · ${block.completed ? "complete" : "pending prefix"}`;
    if (state.event !== null && details.events[state.event]) eventPanel(details.events[state.event], block);
    else centerPanel(center, block, shownEdges);
  }
  const groupKey = node => `${node.source}/${node.modality}`;
  const groupNames = {"conditioning/visual": "Conditioning · visual", "conditioning/event": "Conditioning · robot events",
    "execution/visual": "Execution · visual", "execution/event": "Execution · robot events"};
  function forestLayout(group) {
    const eligibleRoots = payload.archive.roots.map(id => nodes.get(id)).filter(Boolean)
      .filter(node => group === "all" || groupKey(node) === group);
    const orderedGroups = ["conditioning/visual", "conditioning/event", "execution/visual", "execution/event"]
      .filter(key => eligibleRoots.some(root => groupKey(root) === key));
    const positions = new Map(), spans = [], included = [], edgePairs = [];
    let cursor = 0;
    function visit(id, depth, ancestors) {
      if (ancestors.has(id)) throw new Error("Archive export contains a cyclic child link: " + id);
      if (positions.has(id)) throw new Error("Archive export assigns a node to multiple roots: " + id);
      const node = nodes.get(id);
      if (!node) throw new Error("Archive child is missing from the exported nodes: " + id);
      const next = new Set(ancestors); next.add(id);
      let x;
      if (node.children.length) {
        const children = node.children.map(child => visit(child, depth + 1, next));
        x = children.reduce((sum, point) => sum + point.x, 0) / children.length;
        node.children.forEach(child => edgePairs.push([id, child]));
      } else { x = cursor; cursor += 1; }
      const position = {x, y: -depth};
      positions.set(id, position); included.push(node);
      return position;
    }
    orderedGroups.forEach(key => {
      const start = cursor;
      eligibleRoots.filter(root => groupKey(root) === key).forEach(root => { visit(root.id, 0, new Set()); cursor += 1.2; });
      spans.push({key, min: start - .8, max: cursor - .4});
      cursor += Math.max(3, (cursor - start) * .06);
    });
    return {positions, spans, included, edgePairs};
  }
  function nodePanel(node) {
    if (!node) {
      panelHeader("Select an archive node", "Choose a parent or fine leaf to inspect the actual stored structure.", "Exact binary edges");
      $("selection").append(element("p", "help", "The full causal archive is exported. Current-scene tokens are intentionally outside this hierarchy; inspect them in the separate Current scene point-cloud view."));
      return;
    }
    panelHeader(`${titleCase(node.kind)} · ${titleCase(node.source)} / ${node.modality === "event" ? "robot event" : "visual"}`,
      node.id, node.completed ? "Completed" : "Pending prefix", !node.completed);
    const stats = element("dl", "detail-stats");
    kv(stats, "Descendant leaves", number(node.count));
    kv(stats, "Observed interval", `${number(node.time_min)}–${number(node.time_max)}`);
    kv(stats, "Tree depth", node.depth);
    kv(stats, "Temporal scale", node.scale);
    kv(stats, "Children", node.children.length);
    $("selection").append(stats);
    const owners = [...new Set(node.leaf_ids.map(id => leafOwners.get(id)).filter(Boolean))];
    $("selection").append(element("p", "help", `${owners.length} block${owners.length === 1 ? "" : "s"} · anchor [${node.anchor.map(v => number(v, 4)).join(", ")}] m · Gaussian occupancy ${number(node.occupancy, 5)} (not leaf mass).`));
    $("selection").append(element("p", "inline-code", `World bounds (metres): min [${node.bounds_min.map(value => number(value, 5)).join(", ")}] · max [${node.bounds_max.map(value => number(value, 5)).join(", ")}].`));
    if (node.children.length) {
      const links = element("div", "node-links");
      node.children.forEach((id, index) => {
        const button = element("button", "button node-link", `Child ${index + 1}: ${short(id)}`);
        button.title = id;
        button.addEventListener("click", () => { state.node = id; requestRender(); });
        links.append(button);
      });
      $("selection").append(links);
    }
    if (owners.length === 1 && blocks.get(owners[0])?.details) {
      const button = element("button", "button", "Open this block’s point cloud");
      button.style.marginTop = "12px";
      button.addEventListener("click", () => {
        const block = blocks.get(owners[0]); selectSource(displaySource(block));
        $("block").value = block.id; selectBlock(block.id);
        const found = block.details.centers.findIndex(center => center.leaf_id === node.id);
        if (found >= 0) { state.center = found; $("center").value = found; }
        switchView("neighborhood");
      });
      $("selection").append(button);
    } else if (owners.length === 1) $("selection").append(element("p", "help", "This earlier block is included in the full hierarchy; point-neighborhood details were not requested for it."));
  }
  async function drawHierarchy() {
    const graph = forestLayout(state.hierarchyGroup);
    const edgeTrace = {type: "scattergl", mode: "lines", x: [], y: [], hoverinfo: "skip", showlegend: false,
      line: {color: css("--line"), width: 1.2}, meta: {kind: "actual_child_edges", edge_count: graph.edgePairs.length}};
    graph.edgePairs.forEach(([parent, child]) => {
      edgeTrace.x.push(graph.positions.get(parent).x, graph.positions.get(child).x, null);
      edgeTrace.y.push(graph.positions.get(parent).y, graph.positions.get(child).y, null);
    });
    const roots = new Set(payload.archive.roots);
    const nodeTrace = {type: "scattergl", mode: state.hierarchyLabels ? "markers+text" : "markers", showlegend: false,
      x: graph.included.map(node => graph.positions.get(node.id).x), y: graph.included.map(node => graph.positions.get(node.id).y),
      customdata: graph.included.map(node => node.id), meta: {kind: "actual_archive_nodes", node_count: graph.included.length},
      text: graph.included.map(node => node.kind === "leaf" ? String(node.id).split(":").slice(-2).join(":") : `${node.count}`),
      textposition: "bottom center", textfont: {size: 8, color: css("--muted")},
      hovertext: graph.included.map(node => `${titleCase(node.kind)} · ${groupNames[groupKey(node)] || groupKey(node)}<br>${node.count} descendant leaves · t=${number(node.time_min)}–${number(node.time_max)}<br>${node.completed ? "completed" : "pending prefix"}<br>${short(node.id)}`),
      hovertemplate: "%{hovertext}<extra></extra>",
      marker: {size: graph.included.map(node => roots.has(node.id) ? 12 : node.kind === "leaf" ? 5 : 8),
        symbol: graph.included.map(node => node.kind === "temporal" ? "square" : node.kind === "local" ? "diamond" : "circle"),
        color: graph.included.map(node => !node.completed ? css("--amber") : node.kind === "temporal" ? css("--teal") : node.kind === "local" ? css("--purple") : css("--accent")),
        line: {width: .5, color: css("--panel")}, opacity: .94},
    };
    const traces = [edgeTrace, nodeTrace];
    if (state.node && graph.positions.has(state.node)) {
      const position = graph.positions.get(state.node);
      traces.push({type: "scattergl", mode: "markers", x: [position.x], y: [position.y], showlegend: false,
        marker: {size: 20, color: "rgba(0,0,0,0)", line: {width: 2, color: css("--amber")}},
        hoverinfo: "skip", meta: {kind: "selected_archive_node"}});
    }
    const annotations = graph.spans.map(span => ({x: (span.min + span.max) / 2, y: 1.4, xref: "x", yref: "y",
      text: groupNames[span.key] || span.key, showarrow: false, font: {size: 11, color: css("--muted")}}));
    if (!graph.included.length) annotations.push({text: "No historical leaves at this causal decision", x: .5, y: .5,
      xref: "paper", yref: "paper", showarrow: false, font: {size: 14, color: css("--muted")}});
    await Plotly.react($("hierarchy"), traces, {
      paper_bgcolor: css("--plot"), plot_bgcolor: css("--plot"), font: {family: "system-ui, sans-serif", color: css("--ink")},
      margin: {l: 45, r: 20, t: 20, b: 45}, showlegend: false, annotations,
      xaxis: {showgrid: false, zeroline: false, showticklabels: false, title: {text: "Leaf order within actual binary trees", font: {size: 10, color: css("--muted")}}},
      yaxis: {showgrid: false, zeroline: false, dtick: 1, tickformat: "~s", tickfont: {size: 9, color: css("--muted")}, title: {text: "Root → descendants", font: {size: 10}}},
      dragmode: "pan", uirevision: "hierarchy-" + state.hierarchyGroup,
      shapes: graph.spans.map((span, index) => ({type: "rect", xref: "x", yref: "paper", x0: span.min, x1: span.max,
        y0: 0, y1: 1, line: {width: 0}, fillcolor: index % 2 ? css("--subtle") : css("--plot"), layer: "below"})),
    }, plotConfig);
    const plot = $("hierarchy");
    if (!plot._chronotreeBound) {
      plot._chronotreeBound = true;
      plot.on("plotly_click", event => {
        const id = event.points?.[0]?.customdata;
        if (nodes.has(id)) { state.node = id; requestRender(); }
      });
      plot.on("plotly_webglcontextlost", () => error("The hierarchy WebGL context was lost; reload the artifact."));
    }
    $("hierarchy-caption").textContent = `${number(graph.included.length)} exact nodes · ${number(graph.edgePairs.length)} child edges`;
    $("view-status").textContent = `${payload.archive.roots.length} historical roots · ${payload.current.leaf_ids.length} separate current leaves`;
    nodePanel(nodes.get(state.node));
  }
  async function requestRender() {
    requested = true;
    if (rendering) return;
    rendering = true;
    document.documentElement.dataset.inspectorRendering = "true";
    try {
      while (requested) {
        requested = false;
        if (state.view === "neighborhood") await drawCloud(); else await drawHierarchy();
      }
      document.documentElement.dataset.inspectorReady = "true";
    } catch (reason) { error(reason); }
    finally { rendering = false; document.documentElement.dataset.inspectorRendering = "false"; }
  }
  function initialize() {
    const provenance = payload.provenance || {};
    const checkpoint = provenance.checkpoint_sha256 || provenance.checkpoint?.sha256 || provenance.encoder_parameters === "checkpoint";
    $("parameter-badge").textContent = checkpoint ? "Bandwidths from checkpoint" : "Configuration bandwidths · no trained checkpoint";
    $("parameter-badge").classList.add(checkpoint ? "checkpoint" : "default");
    const summaries = [
      ["Task", payload.episode.task, true], ["Episode / decision", `${payload.episode.episode_id} / ${payload.episode.decision_index}`, true],
      ["Observed history", `${payload.episode.conditioning_frames + payload.episode.execution_prefix_frames} frames`],
      ["History leaves", number(payload.archive.leaf_ids.length)], ["Detailed blocks", number(detailed.length)],
      ["Current center slots", number(payload.config.current_tokens)],
    ];
    summaries.forEach(([label, value, small]) => {
      const item = element("div", "summary-item");
      item.append(element("span", "summary-label", label), element("span", "summary-value" + (small ? " small" : ""), value));
      $("summary").append(item);
    });
    const parameters = [
      ["Encoder", payload.config.variant], ["Block length", `${payload.config.block_size} frames`],
      ["History slots / block", payload.config.visual_tokens], ["τₓ denominator", `${number(payload.config.tau_x, 6)} m²`],
      ["τₜ denominator", `${number(payload.config.tau_t, 5)} frame²`], ["Spatial radius", `${number(payload.config.spatial_radius)} m`],
      ["Temporal radius", `${number(payload.config.temporal_radius)} frames`], ["Neighbor cap", payload.config.neighbor_cap],
    ];
    parameters.forEach(([name, value]) => $("parameters").append(element("dt", "", name), element("dd", "", value)));
    sources.forEach(source => option($("source"), source, source === "current" ? "Current scene · separate path" : titleCase(source)));
    option($("hierarchy-group"), "all", "All source / modality groups");
    [...new Set(payload.archive.nodes.map(groupKey))].sort().forEach(key => option($("hierarchy-group"), key, groupNames[key] || key));
    $("scope-description").textContent = provenance.scope || "Full causal binary hierarchy; detailed neighborhoods only for requested blocks and the separate current scene.";
    $("provenance").textContent = JSON.stringify({episode: payload.episode, config: payload.config, provenance,
      export_counts: {archive_nodes: payload.archive.nodes.length, archive_roots: payload.archive.roots.length,
        history_leaves: payload.archive.leaf_ids.length, detailed_blocks: detailed.length,
        detailed_points: detailed.reduce((sum, block) => sum + block.details.points.length, 0),
        allowed_neighborhood_edges: detailed.reduce((sum, block) => sum + block.details.centers.reduce((n, center) => n + center.neighbor_mask.filter(Boolean).length, 0), 0)}}, null, 2);
    $("footer-counts").textContent = `Schema ${payload.schema_version} · no external assets or network required`;
    $("source").addEventListener("change", event => { selectSource(event.target.value); requestRender(); });
    $("block").addEventListener("change", event => { stopAnimation(); selectBlock(event.target.value); requestRender(); });
    $("frame").addEventListener("change", event => { stopAnimation(); setFrame(event.target.value); });
    $("center").addEventListener("change", event => { state.center = Number(event.target.value); state.event = null; requestRender(); });
    $("separation").addEventListener("input", event => {
      state.separation = Number(event.target.value); $("separation-value").textContent = `${state.separation.toFixed(2)} m/frame`; requestRender();
    });
    $("point-size").addEventListener("input", event => {
      state.pointSize = Number(event.target.value); $("point-size-value").textContent = state.pointSize.toFixed(1); requestRender();
    });
    for (const name of ["within", "cross", "events", "centers"]) $(name).addEventListener("change", event => { state[name] = event.target.checked; requestRender(); });
    $("play").addEventListener("click", () => {
      if (playing) { stopAnimation(); return; }
      $("play").textContent = "Ⅱ"; $("play").setAttribute("aria-label", "Pause frame animation");
      const advance = () => { if (!rendering) setFrame(String(state.frame === "all" ? 0 : (Number(state.frame) + 1) % blocks.get(state.block).frames.length)); };
      advance(); playing = setInterval(advance, 800);
    });
    $("tab-neighborhood").addEventListener("click", () => switchView("neighborhood"));
    $("tab-hierarchy").addEventListener("click", () => switchView("hierarchy"));
    $("hierarchy-group").addEventListener("change", event => { state.hierarchyGroup = event.target.value; state.node = null; requestRender(); });
    $("hierarchy-labels").addEventListener("change", event => { state.hierarchyLabels = event.target.checked; requestRender(); });
    $("hierarchy-reset").addEventListener("click", () => Plotly.relayout($("hierarchy"), {"xaxis.autorange": true, "yaxis.autorange": true}));
    $("camera-reset").addEventListener("click", () => Plotly.relayout($("cloud"), {"scene.camera": {eye: {x: 1.5, y: -1.8, z: 1.25}}}));
    $("theme-toggle").addEventListener("click", () => {
      document.documentElement.dataset.theme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
      requestRender();
    });
    if (window.ResizeObserver) new ResizeObserver(() => {
      const plot = $(state.view === "neighborhood" ? "cloud" : "hierarchy");
      if (plot.data) Plotly.Plots.resize(plot).catch(error);
    }).observe(document.querySelector(".stage"));
    selectSource(firstSource);
    // Read-only inspection/debug surface also supports reproducible browser checks.
    window.chronotreeInspector = {payload, state, requestRender,
      get rendering() { return rendering; }, get playing() { return Boolean(playing); }};
    requestRender();
  }
  try { initialize(); } catch (reason) { error(reason); }
})();
