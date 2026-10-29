"""Build the editable draw.io teacher diagram and its interactive HTML export.

No dataset, CUDA allocation, training, or remote service is needed. Layer layout
is curated here; dimensions, counts, settings and source fingerprints come from
the repository. Review the layout when the implementation changes, then regenerate.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from stmem.config import load_config
from stmem.data.center_sampling import CENTER_SAMPLING_VERSION, center_sampler_settings
from stmem.data.pointcloud import POINT_SAMPLING_VERSION, point_sampler_settings
from stmem.models.policy import MemoryPolicy
from stmem.training.logging_options import resolve_logging
from stmem.vision.frozen import _CLIPLayer1, visual_feature_identity


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "diagrams"
STEM = "teacher_architecture"
H32_CONFIG = "configs/teacher_b200_b128_m8_blocks32_no_previous_joint_commands_h32.yaml"
FV_CONFIGS = {
    "fv_prop": "configs/teacher_fv_prop.yaml",
    "fv_no_prop": "configs/teacher_fv_no_prop.yaml",
    "fv_prop_10hz": "configs/teacher_fv_prop_10hz.yaml",
    "fv_no_prop_10hz": "configs/teacher_fv_no_prop_10hz.yaml",
}
PAGES = (
    ("overview", "01 · Overview"),
    ("encoders", "02 · Encoders"),
    ("decoder", "03 · Action decoder"),
    ("context", "04 · Context & size"),
    ("frozen", "05 · Frozen vision"),
)
COLORS = {
    "data": ("#F1F5F9", "#64748B"),
    "visual": ("#EFF6FF", "#2563EB"),
    "event": ("#F0FDFA", "#0F766E"),
    "text": ("#F5F3FF", "#7C3AED"),
    "decoder": ("#FFFBEB", "#B45309"),
    "note": ("#FFFFFF", "#CBD5E1"),
}
SOURCES = [
    "configs/base.yaml",
    "configs/teacher.yaml",
    "configs/teacher_b200.yaml",
    "configs/teacher_b200_b128_m8_blocks32.yaml",
    "configs/teacher_b200_b128_m8_blocks32_no_previous_joint_commands.yaml",
    H32_CONFIG,
    *FV_CONFIGS.values(),
    "src/stmem/config.py",
    "src/stmem/data/pointcloud.py",
    "src/stmem/data/center_sampling.py",
    "src/stmem/data/canonical.py",
    "src/stmem/data/prepare.py",
    "src/stmem/data/prepare_features.py",
    "src/stmem/data/dataset.py",
    "src/stmem/data/feature_cache.py",
    "src/stmem/vision/frozen.py",
    "src/stmem/evaluation/robomme_adapter.py",
    "src/stmem/evaluation/runner.py",
    "src/stmem/models/policy.py",
    "src/stmem/models/spacetime_encoder.py",
    "src/stmem/models/batched_spacetime.py",
    "src/stmem/models/batched_policy.py",
    "src/stmem/models/event_encoder.py",
    "src/stmem/models/text_encoder.py",
    "src/stmem/models/metadata_adapter.py",
    "src/stmem/models/action_flow.py",
    "src/stmem/models/attention_diagnostics.py",
    "src/stmem/training/data.py",
    "src/stmem/training/loader.py",
    "src/stmem/training/attention.py",
    "src/stmem/training/logging_options.py",
    "src/stmem/visualization/diagnostics.py",
    "src/stmem/tracking.py",
    "src/stmem/train.py",
]


def count(module):
    return sum(p.numel() for p in module.parameters())


def snapshot():
    cfg = load_config(ROOT / "configs" / "teacher.yaml")
    policy = MemoryPolicy(cfg)
    h32_cfg = load_config(ROOT / H32_CONFIG)
    h32_policy = MemoryPolicy(h32_cfg)
    frozen_variants = {}
    for name, path in FV_CONFIGS.items():
        variant_cfg = load_config(ROOT / path)
        variant_policy = MemoryPolicy(variant_cfg)
        frozen_variants[name] = {
            "configuration": path,
            "data": variant_cfg["data"],
            "model": variant_cfg["model"],
            "training": variant_cfg["training"],
            "total_policy_parameters": count(variant_policy),
            "trainable_policy_parameters": sum(
                p.numel() for p in variant_policy.parameters() if p.requires_grad
            ),
            "inactive_policy_parameters": sum(
                p.numel() for p in variant_policy.parameters() if not p.requires_grad
            ),
            "feature_projection_parameters": count(variant_policy.encoder.visual.feature_projection),
            "external_frozen_encoder_parameters": count(_CLIPLayer1()),
            "visual_feature_identity": visual_feature_identity(variant_cfg),
            "current_state_token": variant_policy.current_state_enabled,
            "historical_event_tokens": variant_policy.encoder.history_events_enabled,
        }
    if cfg["model"]["encoder"] != "joint" or cfg["model"]["decoder"] != "flow":
        raise ValueError("Update the curated diagram for this encoder/decoder variant.")
    if cfg["model"]["visual_tokens"] % cfg["model"]["block_size"]:
        raise ValueError("Update the token-count formula for uneven per-frame visual quotas.")
    if center_sampler_settings(cfg)["method"] != "balanced_cluster":
        raise ValueError("Update the curated center-selection diagram for this sampling method.")
    if point_sampler_settings(cfg)["sampling"] != "balanced":
        raise ValueError("Update the curated observed-point sampling diagram for this method.")
    if cfg["model"]["current_tokens"] != cfg["model"]["visual_tokens"] // cfg["model"]["block_size"]:
        raise ValueError("Update the shared per-frame center-cache diagram for unequal quotas.")
    if not all(p.requires_grad for p in policy.parameters()):
        raise ValueError("Update the trainable/frozen labels for the current teacher.")
    parts = {
        "Shared visual encoder": count(policy.encoder.visual),
        "Historical event encoder": count(policy.encoder.event),
        "Goal text encoder": count(policy.text_encoder),
        "Shared metadata adapter": count(policy.adapter),
        "Current state MLP": count(policy.proprio),
        "Source clock projection": count(policy.clock),
        "Action flow decoder": count(policy.decoder),
    }
    assert sum(parts.values()) == count(policy)
    return (
        cfg,
        policy,
        {
            "configuration": "configs/teacher.yaml",
            "model": cfg["model"],
            "geometry": cfg["geometry"],
            "prepared_subdir": cfg["data"]["prepared_subdir"],
            "center_sampling": center_sampler_settings(cfg),
            "center_sampling_version": CENTER_SAMPLING_VERSION,
            "point_sampling": point_sampler_settings(cfg),
            "point_sampling_version": POINT_SAMPLING_VERSION,
            "training": cfg["training"],
            "parameters": parts,
            "total_parameters": count(policy),
            "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
            "fp32_weight_bytes": 4 * count(policy),
            "attention_diagnostics": {
                "default_interval_optimizer_steps": resolve_logging(cfg)["attention_every"],
                "capture_flow_time": (cfg["model"]["flow_steps"] - 1) / cfg["model"]["flow_steps"],
                "changes_model_parameters": False,
            },
            "variants": {
                **frozen_variants,
                "no_previous_joint_commands_h32": {
                    "configuration": H32_CONFIG,
                    "model": h32_cfg["model"],
                    "training": h32_cfg["training"],
                    "total_parameters": count(h32_policy),
                    "extra_parameters_vs_core": count(h32_policy) - count(policy),
                },
            },
            "source_sha256": {
                path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES
            },
        },
    )


class Page:
    def __init__(self, file, key, title, subtitle, height=1240):
        self.key, self.height, self.serial = key, height, 0
        diagram = ET.SubElement(file, "diagram", id=key, name=dict(PAGES)[key])
        model = ET.SubElement(
            diagram,
            "mxGraphModel",
            dx="1800",
            dy=str(height),
            grid="0",
            gridSize="10",
            guides="1",
            tooltips="1",
            connect="1",
            arrows="1",
            fold="1",
            page="1",
            pageScale="1",
            pageWidth="1800",
            pageHeight=str(height),
            math="0",
            shadow="0",
            background="#FFFFFF",
            adaptiveColors="none",
        )
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", value="Architecture", parent="0")
        ET.SubElement(self.root, "mxCell", id="notes", value="Reading notes", parent="0")
        self.text(48, 30, 1690, 55, title, size=36, bold=True)
        self.text(48, 92, 1690, 44, subtitle, size=19, color="#475569")
        for i, (target, label) in enumerate(PAGES):
            selected = target == key
            stride = 1728 / len(PAGES)
            self.card(
                48 + stride * i,
                153,
                stride - 24,
                46,
                label,
                tone="visual" if selected else "note",
                link=target,
                size=17,
                tooltip="Click to open this diagram page. Hover over blocks for implementation details.",
                padding=11,
            )
        self.text(
            48,
            height - 48,
            1690,
            30,
            (
                "CHRONOTREE  /  Matched FV-Prop / FV-NoProp experiment     •     "
                if key == "frozen"
                else "CHRONOTREE  /  Teacher · configs/teacher.yaml     •     "
            )
            + "Click page tabs or linked blocks · hover for details · all shapes are editable",
            size=14,
            color="#64748B",
        )

    def ident(self):
        self.serial += 1
        return f"{self.key}-{self.serial}"

    def text(self, x, y, w, h, content, size=18, color="#0F172A", bold=False, parent="1"):
        cell = ET.SubElement(
            self.root,
            "mxCell",
            id=self.ident(),
            value=html.escape(content).replace("\n", "<br>"),
            style=f"text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;"
            f"overflow=hidden;fontFamily=Arial;fontSize={size};fontColor={color};"
            f"fontStyle={1 if bold else 0};spacing=0;",
            vertex="1",
            parent=parent,
        )
        ET.SubElement(
            cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), attrib={"as": "geometry"}
        )
        return cell.attrib["id"]

    def card(
        self,
        x,
        y,
        w,
        h,
        title,
        lines=(),
        *,
        tone="data",
        link=None,
        tooltip="",
        size=19,
        padding=16,
        parent="1",
    ):
        fill, stroke = COLORS[tone]
        content = f"<b>{html.escape(title)}</b>"
        if lines:
            content += "<br><br>" + "<br>".join(html.escape(str(line)) for line in lines)
        ident = self.ident()
        attrs = {"id": ident, "label": content, "tooltip": tooltip or "\n".join([title, *lines])}
        if link:
            attrs["link"] = f"data:page/id,{link}"
        obj = ET.SubElement(self.root, "object", attrs)
        cell = ET.SubElement(
            obj,
            "mxCell",
            style=f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
            f"align=left;verticalAlign=top;spacing={padding};fontFamily=Arial;"
            f"fontSize={size};fontColor=#0F172A;shadow=0;",
            vertex="1",
            parent=parent,
        )
        ET.SubElement(
            cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), attrib={"as": "geometry"}
        )
        return ident

    def edge(self, source, target, label="", tone="data", entry=""):
        cell = ET.SubElement(
            self.root,
            "mxCell",
            id=self.ident(),
            value=html.escape(label),
            style=f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;"
            f"jettySize=auto;html=1;endArrow=block;endFill=1;"
            f"strokeColor={COLORS[tone][1]};strokeWidth=2;"
            "fontFamily=Arial;fontSize=15;fontColor=#475569;labelBackgroundColor=#FFFFFF;" + entry,
            edge="1",
            parent="1",
            source=source,
            target=target,
        )
        ET.SubElement(cell, "mxGeometry", relative="1", attrib={"as": "geometry"})

    def section(self, x, y, w, label):
        self.text(x, y, w, 35, label, size=19, bold=True, color="#475569")


def overview(file, cfg, facts):
    m = cfg["model"]
    w, f, v, c, h = (m[k] for k in ("width", "block_size", "visual_tokens", "current_tokens", "horizon"))
    p = Page(
        file,
        "overview",
        "The full-history teacher",
        "One causal decision → evidence tokens → action-query flow decoder → a chunk of robot commands",
    )
    for x, width, title in (
        (48, 300, "01  AVAILABLE OBSERVATIONS"),
        (440, 340, "02  SHARED ENCODERS"),
        (875, 390, "03  CONTEXT TOKENS"),
        (1380, 372, "04  ACTION PREDICTION"),
    ):
        p.section(x, 225, width, title)
    history = p.card(
        48,
        285,
        300,
        180,
        "History available at decision t",
        [
            "Conditioning observations",
            "Execution frames strictly before t",
            "XYZRGB + state + available",
            "preceding commands",
        ],
        tooltip="Source: training/data.py and batched_policy.py. Conditioning and execution "
        "are separate source-local histories. Future observations never enter the context. "
        "model.condition_on_previous_joint_commands=false zeros the seven commanded joints "
        "in both current and historical inputs; measured state and the previous gripper command remain.",
    )
    enc = p.card(
        440,
        285,
        340,
        180,
        "Historical visual + event",
        [
            f"Separate {f}-frame blocks per source",
            f"Up to {v} visual + {f} event leaves",
            f"Each token has {w} features",
            "Shared metadata adapter",
        ],
        tone="visual",
        link="encoders",
    )
    bank = p.card(
        875,
        285,
        390,
        180,
        "Full fine-leaf history",
        [
            "All causal blocks and partial prefixes",
            "No teacher history token budget",
            "No learned router / parent selection",
            "Padding is masked",
        ],
        tone="visual",
        link="context",
        tooltip="Teacher training consumes all fine leaves. The compact TREE/FLAT readers and "
        "their token budgets are later stages, not components of this teacher forward pass.",
    )
    current = p.card(
        48,
        565,
        300,
        165,
        "Current observation at t",
        [
            f"Up to {m['points']:,} XYZRGB points",
            "Robot state + preceding command",
            "Two source-local clocks",
        ],
    )
    cur_enc = p.card(
        440,
        565,
        340,
        165,
        "Current observation encoders",
        [
            f"Same visual encoder → ≤{c} tokens",
            "Same metadata adapter on visual",
            "State MLP + clock projection",
        ],
        tone="event",
        link="context",
    )
    cur_bank = p.card(
        875,
        565,
        390,
        165,
        f"Current + auxiliary · ≤{c + 3} tokens",
        [f"≤{c} current visual tokens", "1 state + 1 goal + 1 clock", f"Width {w} throughout"],
        tone="event",
        link="context",
    )
    goal = p.card(48, 825, 300, 115, "Task instruction", ["UTF-8 goal text from the episode"], tone="text")
    text = p.card(
        440,
        825,
        340,
        115,
        "Byte-level text Transformer",
        [f"{m['text_layers']} layers → 1 pooled goal token"],
        tone="text",
        link="encoders",
    )
    noise = p.card(
        1380,
        285,
        372,
        125,
        "Noisy action chunk + flow time",
        [f"[B, {h}, 8] and u ∈ [0, 1]", "Projected to action queries"],
        tone="decoder",
        link="decoder",
    )
    dec = p.card(
        1380,
        515,
        372,
        225,
        f"{m['layers']} action-query decoder blocks",
        [
            f"{h} queries · width {w} · {m['heads']} heads",
            "Query self-attention",
            "Cross-attention to all context",
            f"Feed-forward {w} → {4 * w} → {w}",
        ],
        tone="decoder",
        link="decoder",
    )
    output = p.card(
        1380,
        835,
        372,
        125,
        f"Velocity · [B, {h}, 8]",
        [f"Inference: {m['flow_steps']} Euler evaluations", f"Denormalize; execute {m['execute']} commands"],
        tone="decoder",
        link="decoder",
    )
    for a, b, tone in (
        (history, enc, "visual"),
        (enc, bank, "visual"),
        (current, cur_enc, "event"),
        (cur_enc, cur_bank, "event"),
        (goal, text, "text"),
        (text, cur_bank, "text"),
        (cur_bank, dec, "event"),
        (noise, dec, "decoder"),
        (dec, output, "decoder"),
    ):
        p.edge(a, b, tone=tone)
    # The automatic top entry would cross the noisy-action input card.
    p.edge(bank, dec, tone="visual", entry="entryX=0;entryY=0.2;entryDx=0;entryDy=0;")
    p.card(
        48,
        1030,
        525,
        112,
        f"{facts['total_parameters']:,} trainable parameters",
        ["All encoders and the decoder train from scratch.", "Hover over blocks for details."],
        tone="note",
        size=18,
        link="context",
        parent="notes",
    )
    p.card(
        614,
        1030,
        525,
        112,
        "Causality is defined at the decision",
        ["A past block may pool jointly across its frames.", "Current visual evidence stays separate."],
        tone="note",
        size=18,
        parent="notes",
    )
    p.card(
        1180,
        1030,
        572,
        112,
        "History is read by action queries",
        [
            "No dense self-attention over the full history.",
            f"Decoder self-attention uses only its {h} queries.",
        ],
        tone="note",
        size=18,
        parent="notes",
    )


def encoders(file, cfg, policy):
    m, visual = cfg["model"], policy.encoder.visual
    w, f, v, c = (m[k] for k in ("width", "block_size", "visual_tokens", "current_tokens"))
    center_settings = center_sampler_settings(cfg)
    point_settings = point_sampler_settings(cfg)
    foreground_centers = int(c * center_settings["foreground_fraction"])
    p = Page(
        file,
        "encoders",
        "How observations become evidence tokens",
        "Blue: shared visual encoder   ·   Teal: historical events   ·   Violet: task instruction",
        height=1770,
    )
    p.section(48, 230, 350, "A0  OBSERVED GEOMETRY")
    p.section(440, 230, 1312, "A1  SHARED VISUAL ENCODER  /  History and current frame")
    raw = p.card(
        48,
        300,
        300,
        220,
        "Calibrated point cloud",
        [
            f"N ≤ {m['points']:,} · XYZRGB + mask",
            f"{point_settings['foreground_fraction']:.0%} elevated; rest broad-scene",
            f"XY cells {point_settings['cell_size']:g} m; voxels {cfg['geometry']['voxel_size'] * 1000:g} mm",
            f"Elevated: world z > {point_settings['table_z'] + point_settings['height_margin']:g} m",
            "Front + wrist; fixed crop",
        ],
        tone="data",
        size=18,
        tooltip="pointcloud.py: observed RGB-D is backprojected into calibrated world coordinates. "
        "Point sampling reserves an elevated-geometry quota and retains broad-scene coverage. "
        "No simulator object IDs, masks, or hidden poses are inputs.",
    )
    centers = p.card(
        48,
        555,
        300,
        260,
        "Frame-local center indices",
        [
            f"≤{c} anchors; target {foreground_centers} elevated + {c - foreground_centers} broad",
            "Cell-balanced centroid refinement",
            "Snap to observed point positions",
            f"XYZ cells: {center_settings['cell_size']:g} m",
            f"≤{center_settings['max_iterations']} iterations per stage",
            "Prepared once; shared online rule",
        ],
        tone="data",
        size=18,
        tooltip="center_sampling.py: deterministic stratified FPS initializes two refinement "
        "stages. Each occupied world XYZ cell has total weight one. Elevated anchors are refined "
        "first, then held fixed while broad-scene anchors refine residual coverage. Each centroid "
        "is snapped to an observed member point. The cached indices depend only on this frame. "
        "Preparation stores center_indices/<quota>; CPU loader workers reuse them. Online RGB-D "
        "observations use the same selector once per acquired frame.",
    )
    point = p.card(
        440,
        300,
        300,
        145,
        "Point MLP",
        [f"Linear 6 → {w}", f"SiLU → Linear {w} → {w}"],
        tone="visual",
        tooltip="spacetime_encoder.py: SpacetimeEncoder.point_mlp. "
        f"{count(visual.point_mlp):,} trainable parameters; applied independently to each point.",
    )
    neighbors = p.card(
        440,
        555,
        300,
        230,
        "Gather bounded neighborhoods",
        [
            "Reuse prepared center indices",
            f"{v // f}/historical or {c}/current frame",
            f"≤{visual.neighbor_cap} neighbors per center",
            f"Radius {visual.spatial_radius:g} m; Δt ≤ {visual.temporal_radius:g} frames",
            "Same source, example and block",
        ],
        tone="visual",
        size=18,
        tooltip="spacetime_encoder.py + batched_spacetime.py. The shared per-frame observed-point "
        "indices choose anchors; the encoder constructs exact bounded Gaussian neighborhoods "
        "from the currently available block or prefix. Pooling neighbors are not clustering "
        "assignments. Temporal distance uses source-local recorded frames, not seconds. "
        "Centers are geometric samples, not semantic object or robot keypoints. "
        "Batched neighbor ranking runs without autograd; selected-neighbor distances "
        "are then evaluated with gradients for the learned bandwidths and observed inputs.",
    )
    gather = p.card(
        820,
        415,
        270,
        170,
        "Gather local edge features",
        [f"Point feature {w}", "+ relative XYZ (3)", f"+ relative time (1) = {w + 4}"],
        tone="visual",
    )
    edge = p.card(
        1150,
        415,
        250,
        170,
        "Edge MLP",
        [f"Linear {w + 4} → {w}", f"SiLU → Linear {w} → {w}"],
        tone="visual",
        tooltip="spacetime_encoder.py: SpacetimeEncoder.edge_mlp. "
        f"{count(visual.edge_mlp):,} trainable parameters.",
    )
    pool = p.card(
        1465,
        415,
        287,
        215,
        "Gaussian weighted pool",
        [
            "Softmax over valid neighbors",
            f"1 vector of width {w}/center",
            "Learned positive τx and τt",
            f"Initial τx={float(visual.tau_x.detach()):.3g} m²; "
            f"τt={float(visual.tau_t.detach()):.3g} frames²",
        ],
        tone="visual",
        size=18,
        tooltip="Gaussian score = −||ΔXYZ||²/τx − Δt²/τt; softmax-weighted sum of edge features. "
        "τx and τt use softplus plus a small positive floor. This is one local pooling stage.",
    )
    for a, b in (
        (raw, point),
        (raw, centers),
        (centers, neighbors),
        (point, gather),
        (neighbors, gather),
        (gather, edge),
        (edge, pool),
    ):
        p.edge(a, b, tone="visual")
    p.card(
        820,
        680,
        932,
        125,
        f"Historical block: up to {v} visual leaves  •  Current: up to {c}",
        [
            f"Separate {f}-frame blocks per source; prefixes contain only observed frames.",
            "Only points and center indices are prepared; learned content is recomputed.",
        ],
        tone="note",
        size=18,
        parent="notes",
    )
    p.section(48, 930, 1600, "B  EVENT ENCODER  /  One token per historical frame")
    state = p.card(
        48,
        990,
        360,
        120,
        "Observed state · 15 values",
        ["7 joints + end-effector XYZ (3)", "+ quaternion (4) + gripper (1)"],
        tone="event",
        size=18,
    )
    command = p.card(
        48,
        1160,
        360,
        140,
        "Previous command",
        [
            "7 joints + gripper + 1 availability",
            "Joint fields zeroed when disabled",
            "Conditioning command unavailable",
        ],
        tone="event",
        size=18,
        tooltip="model.condition_on_previous_joint_commands defaults to true. False removes "
        "only the seven commanded joint targets; the gripper command and availability flag remain. "
        "Unavailable commands, including conditioning/reset commands, are always zeroed.",
    )
    state_mlp = p.card(
        490, 990, 320, 120, "State branch", [f"Linear 15 → {w} → SiLU", f"Linear {w} → {w}"], tone="event"
    )
    action_mlp = p.card(
        490, 1160, 320, 120, "Command branch", [f"Linear 9 → {w} → SiLU", f"Linear {w} → {w}"], tone="event"
    )
    combine = p.card(
        910,
        1075,
        345,
        135,
        "Add branches",
        [f"SiLU → Linear {w} → {w}", "One event token per frame"],
        tone="event",
        tooltip=f"event_encoder.py: EventEncoder. {count(policy.encoder.event):,} parameters. "
        "Unavailable preceding commands are zeroed before their branch projection. "
        "The optional joint-command ablation applies before this projection in serial and batched paths.",
    )
    event_out = p.card(
        1370,
        1075,
        382,
        135,
        "Attach metadata",
        ["Same adapter as the visual tokens", "Adds location / time / source fields"],
        tone="event",
        link="context",
    )
    for a, b in (
        (state, state_mlp),
        (command, action_mlp),
        (state_mlp, combine),
        (action_mlp, combine),
        (combine, event_out),
    ):
        p.edge(a, b, tone="event")
    p.section(48, 1350, 1600, "C  GOAL TEXT ENCODER  /  Trained from scratch")
    byte = p.card(
        48,
        1420,
        300,
        170,
        "UTF-8 bytes + start token",
        [f"At most {m['max_text_bytes']} bytes", "Vocabulary: 257 IDs", "Padding is masked"],
        tone="text",
    )
    emb = p.card(
        425,
        1420,
        340,
        170,
        "Token + position embeddings",
        [f"Token table: 257 × {w}", f"Position table: {m['max_text_bytes'] + 1} × {w}"],
        tone="text",
    )
    tx = p.card(
        845,
        1420,
        470,
        210,
        f"{m['text_layers']} Transformer encoder layers",
        [
            f"{m['heads']}-head self-attention · width {w}",
            f"FFN: {w} → {2 * w} → {w} with ReLU",
            "Residual connections + post LayerNorm",
            "Dropout = 0; no causal text mask",
        ],
        tone="text",
        tooltip="text_encoder.py: TextEncoder. Standard PyTorch TransformerEncoderLayer "
        "(norm_first=False). Instruction text is available as a whole at each decision.",
    )
    goal = p.card(
        1395,
        1420,
        357,
        170,
        "Masked mean → LayerNorm",
        [f"One goal vector · [B, 1, {w}]", "Goes directly to context", "No metadata adapter"],
        tone="text",
    )
    for a, b in ((byte, emb), (emb, tx), (tx, goal)):
        p.edge(a, b, tone="text")


def decoder(file, cfg, policy, facts):
    m = cfg["model"]
    w, h, n, heads = (m[k] for k in ("width", "horizon", "layers", "heads"))
    p = Page(
        file,
        "decoder",
        "Action-query conditional flow matching",
        "Training predicts a velocity field once; inference integrates that field to produce robot commands.",
        height=2110,
    )
    noisy = p.card(
        48,
        270,
        340,
        150,
        f"Noisy action · [B, {h}, 8]",
        [f"Linear 8 → {w}", f"+ learned {h} × {w} positions"],
        tone="decoder",
    )
    time = p.card(
        48,
        495,
        340,
        175,
        "Flow interpolation time u",
        ["16 sine + 16 cosine features", f"Linear 32 → {w} → SiLU", f"Linear {w} → {w}"],
        tone="decoder",
        tooltip="action_flow.py: ActionFlow.time_in. u is the flow interpolation time; "
        "it is independent of the source-local episode clocks in the observation context.",
    )
    query = p.card(
        480,
        350,
        330,
        150,
        "Add query embeddings",
        [f"{h} action-query tokens", f"Shape [B, {h}, {w}]"],
        tone="decoder",
    )
    blocks = p.card(
        910,
        270,
        390,
        270,
        f"{n} decoder blocks in sequence",
        [
            f"Block 1 → … → Block {n}",
            f"Each preserves [B, {h}, {w}]",
            f"{count(policy.decoder.layers[0]):,} parameters/block",
            "Independent weights per block",
            "Every block reads the same context",
        ],
        tone="decoder",
    )
    context = p.card(
        910,
        620,
        390,
        135,
        "Full observation context",
        [f"[B, L, {w}] + validity mask", "All fine leaves + current/auxiliary"],
        tone="visual",
        link="context",
    )
    out = p.card(
        1410,
        305,
        342,
        180,
        "Output projection",
        [f"LayerNorm → Linear {w} → 8", f"Velocity vθ · [B, {h}, 8]", "7 arm joints + gripper"],
        tone="decoder",
        tooltip="Output is velocity in normalized action space, not a Cartesian end-effector "
        "position or a raw joint command. The flow solver and action normalizer convert it.",
    )
    for a, b, tone in (
        (noisy, query, "decoder"),
        (time, query, "decoder"),
        (query, blocks, "decoder"),
        (context, blocks, "visual"),
        (blocks, out, "decoder"),
    ):
        p.edge(a, b, tone=tone)
    p.card(
        1410,
        565,
        342,
        190,
        "Attention dimensions",
        [
            f"{heads} heads × {w // heads} features/head",
            f"Self: {h} queries ↔ {h} queries",
            f"Cross: {h} queries → L context",
            "No history-to-history attention",
        ],
        tone="note",
        size=18,
        parent="notes",
    )

    attention = facts["attention_diagnostics"]
    p.card(
        48,
        1800,
        1704,
        225,
        "ATTENTION DIAGNOSTICS  /  Fixed training-scene inference probes",
        [
            f"Every {attention['default_interval_optimizer_steps']} optimizer updates by default; "
            f"fixed noise, final Euler evaluation at u = {attention['capture_flow_time']:g}.",
            "Cross-attention mass: current visual/state, goal/clock, historical visual/robot-event tokens.",
            "Conditional entropy is normalized within each group before averaging heads, queries and layers.",
            "Uniform token-count baselines account for longer histories. Attention weight alone is not causal importance.",
            "Detached Q/K statistics leave SDPA and learned parameters unchanged; runtime cadence is resume-safe.",
        ],
        tone="note",
        size=19,
        tooltip="training/attention.py and models/attention_diagnostics.py. Fixed probes do not expose "
        "expert targets to inference. Diagnostic work is separate from the optimization update and timed explicitly.",
        parent="notes",
    )
    p.section(48, 820, 1650, "INSIDE EACH DECODER BLOCK  /  Pre-normalized residual sublayers")
    a = p.card(
        48,
        890,
        490,
        195,
        "1  Query self-attention",
        [
            "y = LayerNorm(x)",
            f"Q, K, V from the {h} query tokens",
            f"{heads} heads → output projection",
            "x ← x + SelfAttention(y, y)",
        ],
        tone="decoder",
    )
    b = p.card(
        655,
        890,
        490,
        195,
        "2  Context cross-attention",
        [
            "Q = projected LayerNorm(x)",
            "K, V = projected observation context",
            "Invalid context slots have −∞ bias",
            "x ← x + CrossAttention(Q, context)",
        ],
        tone="decoder",
    )
    c = p.card(
        1262,
        890,
        490,
        195,
        "3  Feed-forward network",
        ["y = LayerNorm(x)", f"Linear {w} → {4 * w} → GELU", f"Linear {4 * w} → {w}", "x ← x + FFN(y)"],
        tone="decoder",
    )
    p.edge(a, b, tone="decoder")
    p.edge(b, c, tone="decoder")
    p.card(
        48,
        1160,
        810,
        325,
        "TRAINING  /  One sampled flow time per example",
        [
            "y = normalized ground-truth action chunk",
            "ε ∼ Normal(0, I);   u ∼ Uniform(0, 1)",
            "xᵤ = (1 − u) ε + u y;   target velocity = y − ε",
            "One decoder evaluation predicts vθ(xᵤ, u | context).",
            "MSE over valid action dimensions within each example;",
            "then average examples equally. Padding has zero loss.",
            "Future actions are supervision; future observations are excluded.",
        ],
        tone="note",
        size=19,
        parent="notes",
    )
    p.card(
        920,
        1160,
        832,
        325,
        "INFERENCE  /  Receding-horizon control",
        [
            "Start with a Gaussian action chunk at flow time u = 0.",
            f"Apply {m['flow_steps']} Euler steps with the same decoder weights.",
            f"Δu = 1/{m['flow_steps']};   x ← x + Δu · vθ(x, u | context)",
            f"Denormalize the resulting {h} × 8 action chunk.",
            f"Execute the first {m['execute']} commands, then observe and replan.",
            "The observation context is shared across the flow evaluations.",
            "Training does not run this whole Euler loop for each example.",
        ],
        tone="note",
        size=19,
        parent="notes",
    )
    variant = facts["variants"]["no_previous_joint_commands_h32"]
    vm, vt = variant["model"], variant["training"]
    p.card(
        48,
        1540,
        1704,
        210,
        f"OPTIONAL H{vm['horizon']} TEACHER  /  Previous commanded joints disabled",
        [
            f"Chunk [B, {vm['horizon']}, 8]; learned positions [{vm['horizon']}, {vm['width']}]. "
            f"Execute {vm['execute']} commands before replanning; control cadence is unchanged.",
            f"{variant['total_parameters']:,} parameters "
            f"(+{variant['extra_parameters_vs_core']:,} versus the core teacher). "
            "All visual/history encoders and causal boundaries are preserved.",
            f"4 B200s: effective batch {vt['effective_batch']}, microbatch {vt['microbatch']}/GPU, "
            f"encoder block batch {vt['encoder_block_batch_size']}; "
            f"LR {vt['lr']:g}, {vt['steps']:,} updates.",
            f"This is a separate fresh-training experiment. The main diagram remains the H{h} core configuration.",
        ],
        tone="note",
        size=19,
        tooltip=variant["configuration"] + ". Prepared frame geometry is reused. "
        "Future actions remain supervision only; target padding stays masked. "
        "Previous gripper command and measured state remain available.",
        parent="notes",
    )


def context(file, cfg, policy, facts):
    m, t = cfg["model"], cfg["training"]
    w, f, v, c = (m[k] for k in ("width", "block_size", "visual_tokens", "current_tokens"))
    per_frame = v // f + 1
    local_four = t["effective_batch"] // 4
    micro_four = min(t["microbatch"], local_four)
    micro_one = min(t["microbatch"], t["effective_batch"])
    passes_four = (local_four + micro_four - 1) // micro_four
    passes_one = (t["effective_batch"] + micro_one - 1) // micro_one
    p = Page(
        file,
        "context",
        "Context assembly, tensor shapes & model size",
        "B = examples in this device pass   ·   Hc = conditioning frames   ·   He = strictly past execution frames",
        height=1860,
    )
    p.section(
        48,
        230,
        1650,
        "SHARED METADATA ADAPTER  /  Applied to historical visual/event leaves and current visual tokens",
    )
    content = p.card(48, 295, 330, 115, "Token content", [f"Linear {w} → {w}"], tone="visual")
    meta = p.card(
        48,
        460,
        330,
        145,
        "23 metadata values",
        [f"Linear 23 → {w} → SiLU", f"Linear {w} → {w}"],
        tone="visual",
        tooltip="metadata_adapter.py: geometry anchor/bounds (9), source-local times (2), "
        "ages (2), log mass/depth/scale (3), completed/disconnected flags (2), "
        "source one-hot (2), modality one-hot (2), singleton flag (1). Total 23.",
    )
    adapt = p.card(
        475,
        365,
        330,
        145,
        "Add → LayerNorm",
        [f"Adapted token of width {w}", f"{count(policy.adapter):,} shared parameters"],
        tone="visual",
    )
    p.card(
        930,
        295,
        822,
        310,
        "Current auxiliary tokens · three separate branches",
        [
            "State: 15 observed + 8 preceding action + 1 availability = 24",
            "Previous-joint input: "
            + ("enabled" if m.get("condition_on_previous_joint_commands", True) else "disabled")
            + " in the core configuration",
            "When disabled: zero 7 joints; retain gripper + availability.",
            f"Linear 24 → {w} → SiLU → Linear {w} → {w}  ⇒  1 token",
            "Goal: pooled text encoder output  ⇒  1 token",
            f"Clock: [conditioning, execution] / 100 → Linear 2 → {w}  ⇒  1 token",
            "These three tokens bypass the metadata adapter.",
            f"Concatenate with ≤{c} adapted current visual tokens.",
        ],
        tone="event",
        size=19,
        tooltip="policy.py and batched_policy.py share the current-state projection. "
        "model.condition_on_previous_joint_commands controls current and historical command inputs "
        "together. Observed joint angles, TCP pose, measured gripper state and all prediction targets remain.",
    )
    p.edge(content, adapt, tone="visual")
    p.edge(meta, adapt, tone="visual")
    p.card(
        48,
        670,
        804,
        250,
        "Valid context length · before padding",
        [
            f"Lvalid ≤ {per_frame}(Hc + He) + {c + 3}",
            f"Each historical frame: ≤{v // f} visual + 1 event token",
            f"Each full {f}-frame block: ≤{v} visual + {f} event = {v + f}",
            f"Current/auxiliary: ≤{c} visual + 3",
            f"Teacher retains all causal history; budget={m['budget']} is for later readers.",
        ],
        tone="visual",
        size=19,
    )
    p.card(
        930,
        670,
        822,
        250,
        "Allocated context length · batched tensor",
        [
            f"Lpad = {v + f} · maxᵢ(⌈Hcᵢ/{f}⌉ + ⌈Heᵢ/{f}⌉) + {c + 3}",
            f"Context tensor: [B, Lpad, {w}]; mask: [B, Lpad]",
            "Partial source blocks and short examples occupy padded slots.",
            "Attention masks every invalid slot; no history truncation.",
            "Teacher leaf mass = 1, so valid log-mass attention bias = 0.",
        ],
        tone="visual",
        size=19,
        tooltip="batched_policy.py: full_history_context / packed context. Conditioning and execution "
        "are blocked separately per example. Only the fine-leaf bank is assembled; hierarchy parents "
        "are not built in the teacher tensor path.",
    )
    p.section(48, 985, 820, "TRAINABLE PARAMETERS  /  Counted from a CPU model instance")
    p.section(970, 985, 780, "CURRENT CORE TRAINING CONFIGURATION")
    table_y = 1040
    for i, (name, value) in enumerate(facts["parameters"].items()):
        fill = "data" if i % 2 == 0 else "note"
        p.card(48, table_y + i * 48, 565, 48, name, tone=fill, size=17, padding=12)
        p.card(613, table_y + i * 48, 260, 48, f"{value:,}", tone=fill, size=17, padding=12)
    total_y = table_y + len(facts["parameters"]) * 48
    p.card(
        48,
        total_y + 8,
        565,
        58,
        "TOTAL · all trainable, initialized from scratch",
        tone="visual",
        size=17,
        padding=15,
    )
    p.card(613, total_y + 8, 260, 58, f"{facts['total_parameters']:,}", tone="visual", size=20, padding=15)
    p.card(
        970,
        1040,
        782,
        410,
        "Decision batching and compute",
        [
            f"Global effective batch: {t['effective_batch']} decisions/update",
            f"Microbatch cap: {t['microbatch']} decisions/device/pass",
            f"4 GPUs: {micro_four} decisions/device/pass × {passes_four} pass/update",
            f"1 GPU: {micro_one} decisions/pass × {passes_one} passes/update",
            f"Historical encoder block batch: {t['encoder_block_batch_size']}",
            f"CPU loader workers: {t['workers']}/rank; prefetch factor: {t['prefetch_factor']}",
            f"Activation checkpointing: {str(m['checkpoint_blocks']).lower()}",
            f"{t['steps']:,} optimizer updates · AdamW · learning rate {t['lr']:g}",
            "Auto precision prefers BF16; geometry/reductions remain FP32.",
            "CPU workers pack prepared points/center indices, not learned features.",
        ],
        tone="note",
        size=19,
        parent="notes",
    )
    mb = facts["fp32_weight_bytes"] / 1e6
    p.card(
        48,
        1510,
        825,
        225,
        "Weights are small; history activations can be large",
        [
            f"FP32 parameter storage: {mb:.2f} MB",
            f"Weights + gradients + two FP32 Adam moments: ~{4 * mb:.2f} MB",
            "These figures exclude activations, attention workspace and CUDA.",
            "Autocast does not change the stored FP32 parameter count.",
            "Each DDP rank holds a replica; GPU memory is not pooled.",
        ],
        tone="note",
        size=18,
        parent="notes",
    )
    p.card(
        970,
        1510,
        782,
        225,
        "Scope and source of truth",
        [
            "This diagram describes the joint-encoder, flow-decoder teacher.",
            "Prepared data: calibrated XYZRGB, frame-local centers, causal state.",
            "Compact TREE/FLAT/router/Perceiver are subsequent stages.",
            "Action normalizer: no trainable weights; fitted statistics are buffers.",
            "Code + config are authoritative. Regenerate after architecture edits.",
        ],
        tone="note",
        size=18,
        parent="notes",
    )


def frozen_vision(file, facts):
    """Required matched experiment, separate from the core XYZRGB defaults."""
    prop, no_prop = (facts["variants"][key] for key in ("fv_prop", "fv_no_prop"))
    p = Page(
        file,
        "frozen",
        "Frozen RGB features · matched state-input experiment",
        "FV-Prop and FV-NoProp share visual evidence, initial shared weights, clocks, action cadence and training budget",
        height=2160,
    )
    p.section(48, 225, 1704, "01  SAME FRAME-LOCAL VISUAL PIPELINE IN BOTH ARMS")
    source = p.card(
        48,
        285,
        310,
        250,
        "Observed RGB-D only",
        [
            "Front + wrist RGB, each 256 × 256",
            "Backproject → crop → voxel/sample",
            "Canonical XYZRGB stays [2048, 6]",
            "Save source camera + original (u,v)",
            "Padding validity follows each point",
        ],
        size=18,
        tooltip="Online pointcloud.py carries source camera/pixel indices through geometry-only selection. Offline feature preparation recovers pixels for unchanged canonical points, verifying exact RGB and recorded-depth backprojection within 1 micrometre per coordinate; missing or ambiguous matches fail. No object labels or unseen pixels are used.",
    )
    clip = p.card(
        430,
        285,
        365,
        250,
        "Frozen CLIP RN50 · layer1",
        [
            "RGB normalization; no crop/resize",
            "3-convolution stem + average pool",
            "First 3 residual bottlenecks",
            "Per-camera [256, 64, 64] feature map",
            "Stride 4; origin at input pixel (1,1)",
            "Eval mode, including BatchNorm",
        ],
        tone="visual",
        size=18,
        tooltip="vision/frozen.py loads the SHA256-pinned OpenAI RN50 artifact, retaining stem and layer1 only. Parameters have no gradients; training uses cached descriptors, online rollout executes the frozen image encoder.",
    )
    descriptors = p.card(
        875,
        285,
        365,
        250,
        "Sample source-camera features",
        [
            "Bilinear sample at saved pixel (u,v)",
            "Raw descriptors [2048, 256]",
            "FP16 sidecar; invalid rows zero",
            "Identity includes weights + mapping",
            "Read only requested episode/frame",
            "No trainable projection in cache",
        ],
        tone="visual",
        size=18,
        tooltip="feature_cache.py aligns each descriptor with the exact prepared point index. Offline and online paths share normalization, sampling and FP16 descriptor rounding. No future frame enters an individual image encoding.",
    )
    projection = p.card(
        1320,
        285,
        432,
        250,
        "Trainable point representation",
        [
            "LayerNorm 256 → Linear 256 → 128",
            "Concatenate XYZ + RGB + descriptor",
            "Point MLP: 134 → 256 → 256",
            "Same edge MLP + Gaussian pooling",
            "Distances/anchors use only XYZ",
            "Geometry and reductions remain FP32",
        ],
        tone="visual",
        size=18,
        tooltip="spacetime_encoder.py and batched_spacetime.py. The new descriptor projection and point encoder are optimized; the frozen RGB backbone is excluded from the optimizer. Shared center indices, quotas, neighborhoods and temporal groups do not depend on descriptors.",
    )
    p.edge(source, clip, "RGB", tone="visual")
    p.edge(clip, descriptors, "dense map", tone="visual")
    p.edge(descriptors, projection, "raw features", tone="visual")
    p.card(
        48,
        580,
        1704,
        130,
        "Geometry, visibility and causality stay matched",
        [
            "Both arms: ≤2048 observed points/frame; nominal 75% above-table quota; ≤32 balanced centers/frame; 256 visual leaves/full 8-frame block.",
            "Conditioning and execution remain separate; history strictly precedes current t. Current appears once. No removed state slot becomes a visual token.",
            "Robot pixels, depth and calibration remain. Visual XYZ/time/source metadata and clock tokens remain; their ablation is a separate follow-up.",
        ],
        tone="note",
        size=18,
    )
    p.section(
        48,
        760,
        1704,
        "02  COMPLETE COMMAND REMOVAL IN BOTH ARMS  /  MEASURED-STATE INPUT IS THE CONTROLLED DIFFERENCE",
    )
    p.card(
        48,
        820,
        804,
        350,
        "FV-Prop · required matched control",
        [
            "Measured joints, TCP pose and gripper state retained.",
            "Current state MLP gets [15 measured values, 9 fixed zeros].",
            "Historical event content uses measured-state branch only.",
            "Historical event anchor/bounds retain measured TCP XYZ.",
            "Previous joints, gripper command and availability all disabled.",
            "≤33(Hc + He) + 35 valid context tokens; masked batch padding.",
            f"{prop['trainable_policy_parameters']:,} active trainable policy parameters",
            f"{prop['inactive_policy_parameters']:,} inactive stored parameters; frozen RGB encoder separate",
        ],
        tone="event",
        size=19,
        tooltip="configs/teacher_fv_prop.yaml. model.proprioception_mode=all, condition_on_previous_commands=false. Inactive command-branch weights are initialized for matching but frozen and excluded from optimization.",
    )
    p.card(
        930,
        820,
        822,
        350,
        "FV-NoProp · primary candidate",
        [
            "No explicit numerical robot state reaches the learned policy.",
            "Current state token removed; goal and clock tokens retained.",
            "Historical event tokens removed, including state-derived XYZ.",
            "Router numerical state/command inputs are fixed zeros.",
            "Previous joints, gripper command and availability all disabled.",
            "≤32(Hc + He) + 34 valid context tokens; masked batch padding.",
            f"{no_prop['trainable_policy_parameters']:,} active trainable policy parameters",
            f"{no_prop['inactive_policy_parameters']:,} inactive stored parameters; frozen RGB encoder separate",
        ],
        tone="visual",
        size=19,
        tooltip="configs/teacher_fv_no_prop.yaml. model.proprioception_mode=none. Raw-state mutation checks rebuild archives and metadata. Zeroing only the state MLP would leave historical TCP anchors as a leak; these tokens are removed entirely.",
    )
    t = prop["training"]
    p.card(
        48,
        1225,
        1704,
        190,
        "03  BASE 20 Hz RECIPE · SAME DECODER, FRESH TRAINING AND DEVELOPMENT PROTOCOL",
        [
            "FULL teacher reads every causal visual leaf. Action flow predicts H=32 native 20 Hz joint commands; execute K=4, then replan.",
            f"4 B200s · effective batch {t['effective_batch']} · microbatch {t['microbatch']}/GPU · encoder block batch {t['encoder_block_batch_size']} · activation checkpointing off",
            f"AdamW, LR {t['lr']:g} · {t['steps']:,} updates · compare 25k / 50k / 100k · same seed, sample order and fixed validation scenes",
            "Autonomous first grasp/lift and target error are diagnostics; full task success remains distinct. No new masks, expert recovery or target-pose inputs.",
        ],
        tone="decoder",
        size=19,
        link="decoder",
    )
    p.card(
        48,
        1465,
        804,
        230,
        "Resource and interpretation boundaries",
        [
            f"External frozen image encoder: {prop['external_frozen_encoder_parameters']:,} parameters",
            "FP16 descriptors: 1 MiB/frame before HDF5 compression.",
            "Two full FP32 feature maps: 8 MiB transient/frame.",
            "Cache I/O and online CLIP latency must be measured.",
            "FV-NoProp vs FV-Prop isolates explicit state removal",
            "under frozen vision, not the benefit of visual pretraining.",
        ],
        tone="note",
        size=18,
        parent="notes",
    )
    p.section(48, 1745, 1704, "04  OPTIONAL 10 Hz OBSERVATION + ACTION EXPERIMENT")
    temporal = facts["variants"]["fv_prop_10hz"]
    tm, td = temporal["model"], temporal["data"]
    p.card(
        48,
        1805,
        1704,
        230,
        "teacher_fv_prop_10hz.yaml / teacher_fv_no_prop_10hz.yaml · fresh, separate runs",
        [
            f"Retain every {td['observation_stride']}nd source frame; original clocks remain 0, 2, 4, … in native 20 Hz units. Conditioning/execution phases stay separate.",
            f"History blocks retain {tm['block_size']} sampled frames and ≤{tm['visual_tokens']} visual leaves. The full retained history remains available; no eight-block truncation.",
            f"Predict H={tm['horizon']} absolute targets at 10 Hz; hold each for {tm['action_stride']} native simulator steps. Execute {tm['execute']} targets = 4 native steps, then replan.",
            "Post-action target phase: a[t+1], a[t+3], …; mask all intervening invalid commands and incomplete hold intervals. Check termination every native step.",
            "Reuse full-rate XYZRGB and CLIP caches. Same per-frame centers, optimizer and batch settings. Fewer history frames, not fewer centers per retained frame.",
            "This changes the temporal experiment: nominal chunk coverage stays ~1.6 s; eight-frame blocks cover ~0.8 s. Preserve original time metadata and causal boundaries.",
        ],
        tone="decoder",
        size=18,
        tooltip="configs/teacher_fv_prop_10hz.yaml and teacher_fv_no_prop_10hz.yaml. observation_stride=2, action_stride=2, horizon=16, execute=2, block_size=8. The simulator still steps at20Hz; every native command/terminal status is counted. Sparse action supervision never jumps over invalid primitive commands. Existing20Hz experiments remain separate.",
    )
    p.card(
        930,
        1465,
        822,
        230,
        "Legacy defaults and later readers",
        [
            "Existing XYZRGB checkpoint loading and token counts remain.",
            "Old seven-joint ablation still retains command gripper/mask.",
            "These new arms use the stronger complete-command switch.",
            "Optional current_only mode retains current measured state",
            "and removes historical state; it is not a required third run.",
            "TREE/FLAT must later reuse the same selected recipe/leaves.",
        ],
        tone="note",
        size=18,
        parent="notes",
    )


def validate_xml(xml):
    """Check references and page bounds without requiring the draw.io renderer."""
    file = ET.fromstring(xml)
    targets = {d.attrib["id"] for d in file.findall("diagram")}
    for diagram in file.findall("diagram"):
        model = diagram.find("mxGraphModel")
        root = model.find("root")
        ids = [e.attrib["id"] for e in root.iter() if "id" in e.attrib]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {diagram.attrib['name']}")
        for obj in root.findall("object"):
            if link := obj.attrib.get("link"):
                if link.removeprefix("data:page/id,") not in targets:
                    raise ValueError(f"Broken page link: {link}")
        for cell in root.iter("mxCell"):
            for key in ("parent", "source", "target"):
                if key in cell.attrib and cell.attrib[key] not in ids:
                    raise ValueError(f"Broken reference: {cell.attrib}")
            if cell.attrib.get("vertex") == "1":
                geo = cell.find("mxGeometry")
                x, y, w, h = (float(geo.attrib[k]) for k in ("x", "y", "width", "height"))
                if x < 0 or y < 0 or x + w > 1800 or y + h > int(model.attrib["pageHeight"]):
                    raise ValueError(f"Vertex outside page: {ET.tostring(cell)}")


def viewer(xml, total):
    options = {
        "highlight": "#2563eb",
        "nav": True,
        "resize": False,
        "toolbar": "pages zoom layers lightbox",
        "edit": "_blank",
        "toolbar-position": "top",
        "center": True,
        "show-tooltip-icons": True,
        "show-link-icons": True,
        "xml": xml,
    }
    data = html.escape(json.dumps(options, ensure_ascii=False), quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chronotree · Interactive teacher architecture</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f1f5f9;color:#0f172a;
font:16px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{background:white;border-bottom:1px solid #cbd5e1;padding:22px 32px;
display:flex;align-items:center;justify-content:space-between;gap:24px;flex-wrap:wrap}}
.eyebrow{{font-size:12px;letter-spacing:.15em;color:#475569;font-weight:700}}
h1{{font-size:26px;line-height:1.2;margin:4px 0 8px}}p{{margin:0;color:#475569}}
nav{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
a{{color:#1d4ed8;text-decoration:none}}a:hover{{text-decoration:underline}}
.button{{border:1px solid #cbd5e1;border-radius:8px;padding:9px 14px;background:white}}
main{{padding:18px 24px}}.hint{{margin:0 0 14px;font-size:14px}}
.mxgraph{{width:100%;height:calc(100vh - 235px);min-height:480px;border:1px solid #cbd5e1;
border-radius:8px;background:white;overflow:auto}}
footer{{padding:0 24px 18px;font-size:13px;color:#64748b}}
#load-status{{padding:20px}}@media(max-width:700px){{header{{padding:16px}}main{{padding:12px}}
.mxgraph{{height:70vh}}h1{{font-size:23px}}}}
</style>
</head>
<body>
<header><div><div class="eyebrow">CHRONOTREE / MODEL REFERENCE</div>
<h1>The full-history teacher</h1>
<p>{total:,} core trainable parameters · Five linked views · Frozen-vision experiment included</p></div>
<nav aria-label="Diagram files">
<a class="button" href="teacher_architecture.drawio" download>Download editable .drawio</a>
<a class="button" href="../architecture_diagram.md">Reading guide</a>
</nav></header>
<main>
<p class="hint">Use the diagram’s page tabs to explore. Hover to reveal the toolbar; zoom and drag to inspect
details. Hover over blocks for notes. The lightbox toolbar also opens the draw.io editor.</p>
<div id="diagram" class="mxgraph" style="width:100%;height:calc(100vh - 235px);overflow:auto"
data-mxgraph="{data}">
<p id="load-status">Loading the draw.io viewer… Internet access is needed for the viewer.
You can also open the downloaded .drawio file in draw.io Desktop.</p></div>
</main>
<footer>Generated from configs/teacher.yaml and the implementation. No dataset or run is loaded.
The diagram is embedded in this file; viewer code loads from diagrams.net.</footer>
<script>
window.onDrawioViewerLoad = function() {{
  const container = document.getElementById('diagram');
  container.textContent = '';
  GraphViewer.createViewerForElement(container, function(viewer) {{
    window.chronotreeViewer = viewer;
    const fit = function() {{
      window.requestAnimationFrame(function() {{
        viewer.graph.fit(24);
        viewer.graph.center(true, true);
        viewer.graph.initialViewState = {{
          scale: viewer.graph.view.scale,
          translate: viewer.graph.view.translate.clone()
        }};
      }});
    }};
    viewer.addListener('graphChanged', fit);
    window.addEventListener('resize', fit);
    fit();
  }});
}};
</script>
<script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
</body>
</html>
"""


def build():
    cfg, policy, facts = snapshot()
    file = ET.Element(
        "mxfile",
        host="app.diagrams.net",
        agent="Chronotree architecture generator",
        pages=str(len(PAGES)),
    )
    overview(file, cfg, facts)
    encoders(file, cfg, policy)
    decoder(file, cfg, policy, facts)
    context(file, cfg, policy, facts)
    frozen_vision(file, facts)
    ET.indent(file, space="  ")
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(file, encoding="unicode") + "\n"
    validate_xml(xml)
    return {
        f"{STEM}.drawio": xml,
        f"{STEM}.html": viewer(xml, facts["total_parameters"]),
        f"{STEM}.json": json.dumps(facts, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the committed diagrams are stale.")
    args = parser.parse_args()
    artifacts = build()
    if args.check:
        stale = [
            name
            for name, text in artifacts.items()
            if not (OUT / name).exists() or (OUT / name).read_text() != text
        ]
        if stale:
            raise SystemExit(
                "Stale architecture artifacts: "
                + ", ".join(stale)
                + "\nReview layers, then run: uv run python scripts/generate_teacher_diagram.py"
            )
        print("Architecture artifacts match the code/config snapshot; XML references and bounds are valid.")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, text in artifacts.items():
            (OUT / name).write_text(text)
            print((OUT / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
