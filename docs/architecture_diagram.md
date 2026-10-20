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
| Encoders | Point and edge MLPs, FPS and Gaussian pooling, historical event branches, and the byte-level text Transformer. |
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
points, not semantic robot keypoints.

`B` denotes examples in a device's current pass; `Hc` and `He` count available
conditioning and strictly past execution frames. Padding is masked throughout.
The parameter table counts the model itself; the memory estimate explicitly
excludes potentially large history activations, temporary workspaces and CUDA.
Training settings describe core repository defaults, not a live HPC run or an
arbitrarily overridden generated job configuration.

The core teacher runs for 100,000 optimizer updates with effective batch 128.
Its microbatch setting of 64 is a per-device maximum: four GPUs use 32 decisions
each in one pass per update, while one GPU accumulates two passes of 64.
Historical encoder blocks are processed in batches of up to 128.

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

After updating the core teacher to 100k steps, effective batch 128, microbatch
cap 64 and encoder block batch 128, all three artifacts were regenerated.
The revised **Context & size** page was opened in Chromium and its batching
labels inspected; the text fits its shapes and the browser reported no errors.
