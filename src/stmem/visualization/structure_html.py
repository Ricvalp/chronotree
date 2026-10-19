"""Standalone, offline rendering of exported encoder and archive structure.

The renderer consumes exact backend neighborhoods and tree edges. Display controls
never select model inputs, recompute neighbors, or change Gaussian weights.
"""

from __future__ import annotations

import html
from importlib.resources import files
import json
from pathlib import Path


def _script_json(payload: dict) -> str:
    """Keep arbitrary diagnostic text inside a non-executable JSON script tag."""
    value = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for character, escape in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        value = value.replace(character, escape)
    return value


def write_structure_html(payload: dict, path: str | Path) -> Path:
    """Write an offline ``file://`` inspector with embedded Plotly and payload.

    ``payload`` follows ``build_structure`` schema version 1 and may include an
    additional JSON-serializable ``provenance`` object supplied by the CLI.
    No network resources, model features, or target actions are added here.
    """
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Structure HTML requires a schema_version=1 payload")
    for field in ("config", "episode", "blocks", "archive", "current"):
        if field not in payload:
            raise ValueError(f"Structure payload is missing {field}")
    encoded = _script_json(payload)
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise RuntimeError(
            "Offline structure visualization requires the optional visualization extra: "
            "uv sync --locked --extra visualization (also retain --extra simulator if needed)"
        ) from exc
    assets = files("stmem.visualization")
    style = assets.joinpath("structure_viewer.css").read_text(encoding="utf-8")
    viewer = assets.joinpath("structure_viewer.js").read_text(encoding="utf-8")
    plotly = get_plotlyjs().replace("</script", "<\\/script")
    task = html.escape(str(payload.get("episode", {}).get("task", "Episode")))
    document = f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>{task} · Chronotree structure inspector</title>
  <style>{style}</style>
</head>
<body>
  <div id="runtime-error" class="runtime-error" role="alert" hidden></div>
  <noscript><div class="runtime-error">JavaScript is required to inspect this offline artifact. The complete data is embedded in this file.</div></noscript>
  <div class="shell">
    <header class="topbar">
      <div class="brand"><span class="brand-mark" aria-hidden="true">◈</span> CHRONOTREE <span class="brand-divider"></span><span class="brand-subtitle">Structure inspector</span></div>
      <div class="top-actions"><span class="offline-badge"><span></span> Offline artifact</span><button id="download" class="button subtle">Export JSON</button><button id="theme-toggle" class="button icon-button" aria-label="Switch color theme">◐</button></div>
    </header>
    <section class="intro">
      <div><div class="eyebrow">OBSERVATIONS → LOCAL EVIDENCE → CAUSAL MEMORY</div><h1>See what the encoder connects.</h1><p class="subtitle">Inspect observed point clouds, exact Gaussian neighborhoods, and the archive’s actual binary structure.</p></div>
      <div id="parameter-badge" class="parameter-badge"></div>
    </section>
    <div id="summary" class="summary-strip" aria-label="Artifact summary"></div>
    <section class="workspace">
      <div class="workspace-heading"><div class="tabs" role="tablist" aria-label="Structure views"><button id="tab-neighborhood" role="tab" aria-selected="true" aria-controls="neighborhood-view" class="tab active">01 <span>Local neighborhoods</span></button><button id="tab-hierarchy" role="tab" aria-selected="false" aria-controls="hierarchy-view" class="tab">02 <span>Memory hierarchy</span></button></div><span id="view-status" class="view-status"></span></div>
      <div class="workspace-body">
        <aside class="controls" aria-label="Inspection controls">
          <div id="neighborhood-controls">
            <section class="control-section"><h2>Evidence window</h2><label for="source">Source</label><select id="source"></select><label for="block">Block</label><select id="block"></select><p id="block-description" class="help"></p></section>
            <section class="control-section"><h2>Observed frames</h2><div class="frame-row"><select id="frame" aria-label="Visible frame"></select><button id="play" class="button play-button" aria-label="Animate observed frames">▶</button></div><label for="separation" class="range-label">Time separation <output id="separation-value">0.00 m/frame</output></label><input id="separation" type="range" min="0" max="0.5" step="0.01" value="0"><p class="help emphasized">Display only. Offsets x by recorded-frame time; model coordinates and weights remain unchanged.</p><label for="point-size" class="range-label">Point size <output id="point-size-value">2.0</output></label><input id="point-size" type="range" min="1" max="6" step="0.5" value="2"></section>
            <section class="control-section"><h2>Incoming neighborhood</h2><label for="center">Visual center</label><select id="center"></select><p class="help">Click a center in the cloud or choose it here.</p><label class="check"><input id="within" type="checkbox" checked><span>Same-frame edges</span></label><label class="check"><input id="cross" type="checkbox" checked><span>Cross-frame edges</span></label><label class="check"><input id="events" type="checkbox" checked><span>Robot-event leaves</span></label><label class="check"><input id="centers" type="checkbox" checked><span>Visual centers</span></label></section>
          </div>
          <div id="hierarchy-controls" hidden>
            <section class="control-section"><h2>Archive groups</h2><label for="hierarchy-group">Source / modality</label><select id="hierarchy-group"></select><p class="help">Each source and modality forms its own forest. Edges are recorded child links, not spatial tracking.</p><label class="check"><input id="hierarchy-labels" type="checkbox"><span>Show leaf labels</span></label><button id="hierarchy-reset" class="button wide">Fit complete forest</button></section>
            <section class="control-section"><h2>Node key</h2><div class="legend-item"><span class="legend-shape temporal"></span>Temporal merge</div><div class="legend-item"><span class="legend-shape local"></span>Local binary parent</div><div class="legend-item"><span class="legend-shape leaf"></span>Fine evidence leaf</div><div class="legend-item"><span class="legend-shape pending"></span>Pending prefix</div><p class="help">Select any node to inspect its exact children, leaf count, observed-time bounds, and block membership.</p></section>
          </div>
          <section class="control-section last"><h2>Encoding parameters</h2><div id="parameters" class="parameter-list"></div><p class="help">Spatial coordinates: metres. Time: source-local recorded frames, not inferred physical seconds.</p></section>
        </aside>
        <div class="stage">
          <div id="neighborhood-view" role="tabpanel" aria-labelledby="tab-neighborhood"><div class="stage-toolbar"><div><span class="legend-dot rgb"></span>Observed RGB <span class="legend-dot center"></span>Visual centers <span class="legend-dot event"></span>Robot events</div><button id="camera-reset" class="button subtle">Reset camera</button></div><div id="cloud" class="plot cloud-plot" aria-label="Interactive 3D observed point cloud"></div><div class="stage-caption"><span id="cloud-caption"></span><span>Drag to orbit · scroll to zoom</span></div></div>
          <div id="hierarchy-view" role="tabpanel" aria-labelledby="tab-hierarchy" hidden><div class="stage-toolbar"><div id="hierarchy-caption"></div><span class="help">Current scene stays separate</span></div><div id="hierarchy" class="plot hierarchy-plot" aria-label="Interactive memory hierarchy"></div><div class="stage-caption"><span>Layout shows tree order, not physical distance or object identity.</span><span>Drag to pan · scroll to zoom</span></div></div>
          <div id="selection" class="selection-panel" aria-live="polite"></div>
        </div>
      </div>
    </section>
    <section class="explanation-grid"><article><span class="section-number">01</span><h2>Local evidence</h2><p>Each visual token pools a bounded set of observed points. Shown edges and normalized weights come from the encoder export; controls only change what you see.</p><div class="formula">α ∝ exp(−‖x − c‖² / τ<sub>x</sub> − (t − τ)² / τ<sub>t</sub>)</div></article><article><span class="section-number">02</span><h2>Recoverable hierarchy</h2><p>Completed blocks form local binary trees, then temporal forests. Parents retain their children. Pending execution evidence uses only its observed prefix; the current scene is supplied separately.</p></article><article><span class="section-number">03</span><h2>Scientific scope</h2><p>No object tracks or inferred correspondences are drawn. Gaussian edges express pooling neighborhoods. Temporal separation is a display transform, and archive count is descendant-leaf mass, not Gaussian occupancy.</p></article></section>
    <details class="provenance"><summary>Provenance, scope, and complete export</summary><p id="scope-description" class="help"></p><pre id="provenance"></pre></details>
    <footer><span>Chronotree · exact structure, inspectable evidence</span><span id="footer-counts"></span></footer>
  </div>
  <script id="structure-data" type="application/json">{encoded}</script>
  <script>{plotly}</script>
  <script>{viewer}</script>
</body>
</html>
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path
