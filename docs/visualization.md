# Inspect the encoder's point clouds and space-time structure

The offline inspector exports an interactive HTML file and an adjacent JSON
record for one decision in a prepared episode. It runs the real `BlockEncoder`
and the shared streaming archive builder on CPU. It does not train, evaluate a
policy, invoke the simulator, modify prepared data, or start a W&B run.

## Install and export

Install the optional plotting dependency in the existing environment, retaining
the simulator packages used by alignment and evaluation:

```bash
uv sync --locked --extra simulator --extra visualization --group dev
```

On the cluster, start from the repository and load your existing site settings:

```bash
source hpc/jobs/common.sh

"$STMEM_PYTHON" -m stmem.visualization.inspect_encoder \
  --config configs/teacher.yaml \
  --task VideoUnmask --episode 0 --decision 16 \
  --detail-blocks 2 \
  --output "$STMEM_RUN_ROOT/visualizations/VideoUnmask_ep0_t16.html"
```

This reads `VideoUnmask.h5` under `$STMEM_DATA_ROOT/prepared`, using the verified
`alignment.json` in the data root. Wait until this particular task has its final
`.h5` file; the other tasks may still be preparing. An in-progress `.partial.h5`
file is rejected. The preprocessing settings must match the config, just as in
training. The command uses one CPU thread by default and requires no GPU.

Copy the HTML file to your computer and open it in a browser with WebGL enabled.
It contains the plotting library and data, so it needs no server or network
connection. The JSON is an audit record and is not needed to open the HTML.

To inspect an incomplete execution block, export decision 20:

```bash
"$STMEM_PYTHON" -m stmem.visualization.inspect_encoder \
  --config configs/teacher.yaml \
  --task VideoUnmask --episode 0 --decision 20 \
  --detail-blocks 2 \
  --output "$STMEM_RUN_ROOT/visualizations/VideoUnmask_ep0_t20.html"
```

With the core block size of eight, decision 16 has two completed execution
blocks; decision 20 also has a pending block containing observations 16–19.
The observation at the selected decision is always encoded separately as the
current frame. These are execution-frame indices, not optimizer steps or raw
HDF5 frame indices. `--episode 0` selects the raw episode ID `episode_0`.

For a second scene, change `--episode 0` to `--episode 1` and use a distinct
output filename. Use `--detail-blocks 4` to expose point neighborhoods for more
historical blocks. Every export retains the full available causal hierarchy;
this flag only limits which blocks include their detailed point arrays and
pooling edges. Larger decisions and more detailed blocks take more CPU time and
produce larger files. The loader indexes metadata in the selected task file,
loads the selected episode, and hashes that prepared file for provenance.
Use `--detail-blocks 0` to keep only the current cloud alongside the full history
hierarchy.

## What the views represent

The inspector connects the following stages:

1. **Prepared observations.** The valid finite XYZRGB points that actually enter
   the encoder, after its canonical point ordering. Invalid/padded points are
   excluded. The exported detailed blocks have no extra display downsampling.
2. **Space-time pooling.** Farthest-point-sampled centers and the exact neighbor
   indices, masks and normalized Gaussian weights from an encoder forward pass.
   Selecting a center exposes its incoming point edges, including neighbors in
   different frames of the same observed block. The core configuration uses four
   centers per historical frame (32 per full eight-frame block) and 64 centers
   for the current observation. Sparse inputs may yield fewer valid centers.
3. **Local trees and temporal forest.** The actual binary archive nodes, child
   edges, bounding boxes, timestamps and leaf counts. Visual local trees split
   centers spatially; event local trees follow time. Completed blocks merge
   through the temporal binary-counter forest. A pending execution prefix has
   its own temporary local trees and has not entered the completed forest.

Conditioning and execution remain separate, as do visual and robot-event
records. Event anchors are observed robot TCP positions. Their inputs are
observed robot state and available preceding commands; conditioning commands
are unavailable. Current-frame visual tokens stay outside the historical tree.
The inspector does not invent object tracks or a separate nearest-neighbor
graph between frames, and it does not show learned retrieval decisions.

Use the scene controls to rotate/zoom, select a block and pooling center, filter
frames, and compare within-frame and cross-frame edges. Separating frames along
the display's time axis changes only the layout. Actual coordinates are world
metres, and timestamps are **source-local recorded-frame units**, not seconds.
The hierarchy view is useful for checking leaf coverage and which completed or
pending blocks a parent summarizes.

The Gaussian kernel is
`exp(-squared_xyz_distance / tau_x - squared_time_distance / tau_t)`, normalized
over the allowed capped neighborhood. The core defaults are a 0.3 m spatial
radius, eight recorded frames of temporal radius and a cap of 128 neighbors per
center. The file reports the actual resolved parameters. The center can pool
from either side of its timestamp within an already observed block: causality
is enforced by the available history prefix, not an additional per-center
past-only mask. Future execution observations never enter that history.

## Inspect a trained checkpoint

Without `--checkpoint`, the file is explicitly labeled **configuration-default
geometry**, with initial bandwidths and a randomly initialized encoder. This is
sufficient to inspect input points, FPS centers and the configured graph/tree
construction. It is not a trained representation or policy result.

After teacher training saves a checkpoint, append the following to the export
command:

```bash
  --checkpoint "$STMEM_RUN_ROOT/seed0/teacher/step_00025000.pt"
```

Use a config compatible with that checkpoint. The inspector strictly loads the
encoder state and reports checkpoint identity, stage, step and learned
bandwidths. A generated teacher config can be passed through `--config` as well.
Load only trusted project checkpoints. Learned feature vectors and expert
targets are not part of the point-cloud display. The geometry is computed on
CPU; exact distance ties need not choose bit-identical neighbors on other
devices.

## Explicit local paths

For an existing prepared subset outside the configured data root:

```bash
uv run python -m stmem.visualization.inspect_encoder \
  --config configs/teacher.yaml \
  --prepared runs/preparation_real/prepared/VideoUnmask.h5 \
  --alignment data/alignment.json \
  --task VideoUnmask --episode 0 --decision 20 \
  --detail-blocks 2 \
  --output runs/visualizations/VideoUnmask_ep0_t20.html
```

This example uses the already prepared local 1,024-point subset. For the local
256-point debug subset, instead use `--config configs/debug/data.yaml` and
`--prepared data/prepared/VideoUnmask.h5`. Do not substitute debug preprocessing
for a core training configuration.

## Reading the supplied VideoUnmask example

Episode 0 asks the robot to watch the conditioning video and then lift the
container hiding the blue cube. The front camera looks down onto three white
containers on a wood table; the wrist view contains parts of the fingers. The
HTML contains sampled visible surfaces, not a rendered full robot. Purple
markers locate the observed tool center independently of those surface points.

The default execution view selects visual center 0 and draws only that center's
incoming neighborhood. Deterministic FPS begins at the lexicographically first
point, which is a table corner here. The pending block at decision 20 contains
four overlapping frames (16–19): its selected center has 83 valid edges, 20
within the center's frame and 63 across frames. With no display time separation,
many repeated table points and edges overlap. Change the center selector or
increase time separation to inspect these connections; they are pooling edges,
not robot links or action trajectories.

This example also exposes a real representation limitation. Most prepared
points describe the tabletop, and all historical visual neighborhoods miss the
elevated object points at the default four-centers-per-frame quota and 0.3 m
radius. The separate current-frame path includes them. This is a measured input
coverage problem, not a claim that the task has been learned or that object
evidence is absent from the original cameras. See the
[coverage report](evidence/encoder_coverage_inspection.json) and
[implementation status](status.md). Initial MLP weights do not cause the hard
geometric exclusion; a checkpoint alone cannot change the fixed radius or FPS
quota.
