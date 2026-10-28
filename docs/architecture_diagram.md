# Interactive teacher architecture

Open [teacher_architecture.html](diagrams/teacher_architecture.html) in a browser,
or open [teacher_architecture.drawio](diagrams/teacher_architecture.drawio) in
draw.io Desktop or [diagrams.net](https://app.diagrams.net/). Download the files
first if your repository browser displays their source. The HTML embeds the
diagram and loads the official viewer from diagrams.net, so its viewer needs
internet access; the native `.drawio` file can be opened in draw.io Desktop offline.

The diagram was opened through the official
[draw.io MCP](https://github.com/jgraph/drawio-mcp) `open_drawio_xml` tool. No MCP
installation is needed to view or edit the saved diagram.

## Reading the diagram

Use the four navigation tabs at the top of a page, or the viewer's page selector.
Hover over the diagram to reveal the toolbar. Zoom, drag to pan, and hover over
blocks for source references and implementation notes. Linked blocks open a more
detailed page. The toolbar also offers layers and a larger lightbox; the lightbox
offers editing in draw.io. The optional **Reading notes** layer
contains explanatory cards; hiding it leaves the main architecture visible.

| Page | What it shows |
|---|---|
| Overview | How one causal decision becomes historical/current evidence and then a predicted action chunk. |
| Encoders | Prepared point sampling and center indices, point and edge MLPs, Gaussian pooling, historical event branches, and the byte-level text Transformer. |
| Action decoder | Query projections, each residual attention/FFN sublayer, and the distinction between training flow matching and inference Euler integration. |
| Context & size | The shared metadata adapter, current auxiliary branches, valid versus allocated context lengths, parameter counts, and core batching settings. |

Blue blocks represent visual/history processing, teal represents state and
current evidence, violet represents goal text, and amber represents action
decoding. Boxes are computational stages, and arrows carry tensors or features.
This is an architecture diagram, not a sampled point cloud, a particular task
episode, a measured GPU trace, or a visualization of trained attention weights.

The scope is the **full-history teacher** in `configs/teacher.yaml`. Its encoders
and action decoder are trained from scratch; exact parameter counts appear on
**Context & size**. All causal fine leaves are retained; the compact-reader budget
does not cap teacher history. Stacking decoder blocks does not imply stacking
graph-convolution layers: the visual encoder uses one local
edge-MLP and Gaussian pooling stage, while the decoder applies attention between
action queries and the resulting context. Spatial centers are samples of observed
points, not semantic robot keypoints. Their deterministic selection is separate
from learned visual features and Gaussian pooling.

`B` denotes examples in a device's current pass; `Hc` and `He` count available
conditioning and strictly past execution frames. Padding is masked throughout.
The parameter table counts the model itself; the memory estimate explicitly
excludes potentially large history activations, temporary workspaces and CUDA.
Training settings describe core repository defaults, not a live HPC run or an
arbitrarily overridden generated job configuration.

The core teacher runs for 100,000 optimizer updates with effective batch 256.
Its microbatch setting of 64 is a per-device maximum: four GPUs use 64 decisions
each in one pass per update, while one GPU accumulates four passes of 64.
Historical encoder blocks are processed in batches of up to 128.

The core configuration enables `model.condition_on_previous_joint_commands`.
Setting it to `false` zeros the seven previous commanded joint targets in both
the current state MLP and every historical event token. The previous gripper
command and its availability flag remain, as do measured joint positions, TCP
pose and measured gripper state. Unavailable commands at reset or in the
conditioning video are still masked. Serial, packed training and live rollout
paths use the same setting. The ablation preserves the 24-value current input,
the nine-value historical command input, token counts and action targets; it
does not require preparing the geometry again. This is an optional experiment,
not a change to the core defaults shown by the diagram.

The **Action decoder** page also labels the optional
[`teacher_b200_b128_m8_blocks32_no_previous_joint_commands_h32.yaml`](../configs/teacher_b200_b128_m8_blocks32_no_previous_joint_commands_h32.yaml)
experiment. It predicts 32 consecutive commands while still executing four
before replanning at the same native control cadence. Its action tensor is
`[B, 32, 8]` and its learned position table is `[32, 256]`, giving 6,102,794
parameters (4,096 more than the core teacher). All observation encoders and
causal history boundaries remain the same. This fresh run retains the ablated
teacher's effective batch 128, microbatch eight per GPU, encoder block batch 32,
learning rate 0.0001 and 100,000 updates on four B200s. Prepared frame geometry is
reused; targets and terminal masks are assembled for the configured horizon.
The main diagram and its core parameter table remain the 16-action default;
the JSON snapshot records the longer-horizon variant separately.

The decoder page also describes the optional attention diagnostics. Every ten
optimizer updates by default, fixed training-scene probes generate action
chunks from fixed noise and measure cross-attention at the final Euler
evaluation (`u=7/8` for eight integration steps). The telemetry distinguishes
current visual/state, goal/clock and historical visual/event tokens. It reports
group mass, token-count baselines and normalized conditional entropy, computed
within each head before averaging. This instrumentation does not add model
parameters or change the decoder's SDPA computation. Its interval is a runtime
logging control, independent of checkpoint architecture/training hashes.

## Prepared geometry and token counts

The revised preparation writes `prepared_balanced_v2/`, preserving the previous
`prepared/` files. It backprojects the two RGB-D views, applies the calibrated
workspace crop and 5 mm voxel sampling, and retains up to **2,048 points per
frame**. A nominal 75% quota prioritizes points above world `z = 0.005 m`, balanced
over 10 cm XY cells; the remaining quota covers the broader scene. This is a
geometric height distinction, not an object or table segmentation mask.

Each frame receives **up to 32 observed-point centers**: a nominal 24 elevated plus
eight broad-scene anchors. Deterministic stratified FPS initializes spatially
balanced centroid refinement. Occupied 10 cm XYZ cells receive equal total
weight; updated centers snap to observed member points. The elevated stage runs
first, then its centers remain fixed during the broad-scene stage. Each stage
has a maximum of 20 iterations. This geometry uses no learned weights or
simulator object labels.

Preparation stores the frame-local indices in `center_indices/32`. Both current
and historical paths reuse those indices; CPU loader workers pack them with the
points. Live observations use the same shared selector once per acquired frame.
The teacher still computes learned point/edge features and Gaussian pooling on
every training pass. Clustering assignments do not become pooling edges: those
edges follow the encoder's fixed radius, temporal bound, and nearest-neighbor cap.

| Quantity | Core shape or limit |
|---|---:|
| Point input per frame | `[2048, 6]` XYZRGB plus validity mask |
| Current visual tokens | 32 |
| Historical visual tokens per eight-frame block | 256 |
| Historical robot-event tokens per full block | 8 |
| Total fine leaves per full block | 264 |
| Valid teacher context | At most `33 × (Hc + He) + 35` tokens |

The current observation stays outside history. Conditioning and execution use
separate blocks, and an incomplete execution block contains only its observed
prefix. TREE and FLAT subsequently consume the exact same frozen fine leaves;
the revised geometry does not relax causal alignment or leaf coverage.

## Keeping it current

Layer labels and layout are maintained in
[`scripts/generate_teacher_diagram.py`](../scripts/generate_teacher_diagram.py).
The generator reads the core configuration, instantiates the model on CPU to
count its parameters, and fingerprints the implementation files. It does not
read the dataset, allocate a GPU, run training, or call an external service.

After reviewing the generator against an architecture change, run from the
repository root:

```bash
uv run python scripts/generate_teacher_diagram.py
uv run python scripts/generate_teacher_diagram.py --check
```

Commit the generator and regenerated `.drawio`, `.html`, and `.json` together.
The JSON is an auditable configuration/count/source-hash snapshot. `--check`
detects stale artifacts and validates XML IDs, references, page links and bounds;
it cannot prove the prose matches a newly changed algorithm. Inspect all four
pages after editing their layout. Native draw.io shapes remain editable, but
manual diagram edits must be reflected in the generator before regeneration.

## Verification

On 2026-10-20, the official draw.io MCP server v1.10.0 enumerated all four pages
and opened the native XML. Chromium rendered the four pages without browser
errors or text overflowing its shapes. The HTML checks exercised page selection,
internal page links, zoom, fit, dragging, source tooltips, the Reading notes layer,
the larger lightbox, and opening all four pages in the draw.io editor. CPU
generation, the artifact freshness/structure check,
and Ruff checks passed. This validates the diagram and viewer, not training
throughput or learned model quality.

An earlier update set the core teacher to 100k steps, effective batch 128, microbatch
cap 64 and encoder block batch 128, all three artifacts were regenerated.
The revised **Context & size** page was opened in Chromium and its batching
labels inspected; the text fits its shapes and the browser reported no errors.

On 2026-10-21, the geometry update regenerated all three artifacts with 2,048
points, 32 prepared centers per frame, 256 historical visual tokens per full
block, and the current effective batch of 256. Chromium rendered all four pages;
the updated encoder and context layouts, page navigation, internal links, zoom,
fit, dragging, source tooltips, and layer dialog were checked. The browser
reported no errors, and text bounds fit their shapes on every page. The artifact
freshness/structure check and Ruff check passed. The model still has 6,098,698
trainable parameters: the added geometry is deterministic and adds tokens, not
learned parameters. These checks do not measure training memory or throughput.

On 2026-10-26, the optional previous-joint-command ablation was added to the
historical event and current auxiliary descriptions. All three artifacts were
regenerated; the core default remains enabled and the parameter count remains
6,098,698. Chromium rendered all four pages without browser errors or overflowing
text bounds. The changed encoder and context layouts were visually inspected.
The artifact freshness/structure check and Ruff check passed. This verifies the
documentation layout, not the effect of the ablation on policy quality.

On 2026-10-28, the optional 32-action teacher received a separately labeled note
on the **Action decoder** page and a configuration/count record in the JSON
snapshot. All three artifacts were regenerated while retaining the core
16-action diagram. The snapshot/XML check and Ruff passed. Chromium rendered
the changed decoder page; its text bounds fit every shape and the full page was
visually inspected. This verifies the diagram, not B200 memory or throughput.

Later on 2026-10-28, a separate attention-telemetry note was added below the H32
experiment. All three artifacts were regenerated, with no architecture or
parameter-count change. The snapshot/XML and Ruff checks passed. Chromium
opened the actual HTML and rendered the decoder page; the added note and all
other text fit their shapes, and the full page was visually inspected. Local
layout evidence is retained in `runs/architecture_diagram/attention/`.
