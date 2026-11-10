# Repeating the local checkpoint inspection

For teacher continuations that also need the matched history comparisons, use
the [complete teacher workflow](teacher_inspection_workflow.md). It wraps the
usual checks below, adds history controls, and supports automatic resume and
comparison against the fixed 104k reference.

[`scripts/inspect_checkpoints.py`](../scripts/inspect_checkpoints.py) runs the
usual checkpoint checks, input-sensitivity probes, cube-position visualizations,
and bounded simulator rollouts. It produces a local HTML report and videos. It
does not train, submit Slurm jobs, or replace the official evaluation suite.

For a transferred bundle on this workstation, run:

```bash
cd /home/rvalperga/chronotree
uv run python scripts/inspect_checkpoints.py \
  --bundle runs/inspection/eval_20261031T093812Z
```

For the next transfer, replace the bundle path. The default output is the bundle
path with `_analysis` appended, so this example writes
`runs/inspection/eval_20261031T093812Z_analysis/`. Open its `index.html` when the
command finishes. The point-cloud viewers work offline; keep the output directory
together so the report can find its videos, images, and numerical results.

If that output already exists, use `--resume`:

```bash
uv run python scripts/inspect_checkpoints.py \
  --bundle runs/inspection/eval_20261031T093812Z \
  --resume
```

The launcher checks checkpoint hashes, asset identities, protocol, device, and
policy source against the saved manifest. Recorded stage code identities and installed runtime versions must
also match when resuming. `runtime_identity.json` records the observed Python,
PyTorch and simulator versions separately from the immutable protocol. Completed results are reused and validated; incomplete
work is retried. A changed checkpoint, asset, or scientific implementation needs
a new output directory. Do not replace checkpoint files inside an existing
bundle. A lock prevents two launchers from writing the same output concurrently.

## Checkpoint bundle and local assets

The expected transfer layout is:

```text
runs/inspection/eval_<timestamp>/
  SHA256SUMS
  history_h16/
    checkpoint.pt
    source.json          # optional origin/step/hash metadata
  history_h32/
    checkpoint.pt
  frozen_10hz/
    checkpoint.pt
```

Labels are not fixed: use lowercase names starting with a letter and containing
letters, digits, or underscores. `SHA256SUMS` must contain one entry per checkpoint
in the form `<sha256>  <label>/checkpoint.pt`. Transfer the checksums produced for
the HPC snapshots along with the snapshots; recomputing them only after transfer
would not verify the transfer against the original files. Checkpoint paths must
stay inside the bundle. The script rejects missing files, checksum failures,
unsupported bundles, and synthetic/debug checkpoints.

The simulator environment and dependencies must already be installed. The full
workflow requires a working CUDA device and RoboMME assets for the fixed scenes.
`uv run python -m stmem.doctor` checks the actual device. There is no automatic
full-dataset download or preparation in this workflow.

[`configs/inspection.yaml`](../configs/inspection.yaml) supplies six local assets:

| Key | Default path | Purpose |
| --- | --- | --- |
| `prepared` | `data/inspection/VideoUnmask.h5` | Prepared real VideoUnmask episodes 0 and 1, including the observations and recorded expert commands used by the offline and handoff checks. |
| `alignment` | `data/alignment.json` | Verified revision-level action/observation alignment record for the local data. |
| `features` | `data/inspection/features` | Frozen visual descriptors matching the prepared data and checkpoint preprocessing/feature identity. |
| `clip_checkpoint` | `cache/backbones/clip_RN50.pt` | Frozen CLIP RN50 weights for newly rendered simulator observations. |
| `pick_prefix` | `data/inspection/pick_prefix.json` | Previously generated and verified official-planner prefix for PickXtimes training scene 0. |
| `replay_commands` | `data/inspection/replay_commands.json` | Saved off-target robot commands and observed states for the matched later-state cube-relocation probe. This is deliberately not an expert prefix. |

Relative paths resolve from the repository root, not the YAML file's directory.
The local `data/inspection/` entries are links to existing diagnostic **data**;
the data, features, weights, and generated outputs remain ignored by Git. The
reusable implementation imports tracked code, not Python helpers from older
ignored inspection directories. Keep the link targets available, or provide a
YAML file with the same six keys through `--assets-config`. Asset files, and files
inside asset directories, are hashed into the manifest. All six assets are
required at preflight even when selecting a subset of stages.

The checkpoint audit checks SHA256 and ZIP CRC, finite model/optimizer tensors,
strict model and optimizer loading, optimizer tensor shapes and update counts,
sampler boundaries, scheduler state and saved learning rate, and per-rank RNG
records. Any optional `source.json` must agree with the checkpoint step and hash.
Checkpoint configurations are preserved; only local data/backbone locations are
redirected to the supplied assets.

## Workflow options

| Option | Behavior |
| --- | --- |
| `--bundle PATH` | Required immutable checkpoint bundle. |
| `--output PATH` | Explicit output directory; otherwise `<bundle>_analysis`. |
| `--assets-config PATH` | Override `configs/inspection.yaml`. |
| `--device cuda:0` | Device for inference and simulation; default `cuda:0`. |
| `--resume` | Continue an existing matching inspection and reuse validated completed results. |
| `--audit-only` | Check checkpoints/assets and write the manifest/audit without GPU inference. |
| `--stages offline grounding handoff` | Select stages, in the supplied order. These three are the default. A subset produces a report explicitly marked partial. |
| `--report-only` | Rebuild the overall report from completed, validated results without rerunning simulator/inference stages. |
| `--compare-to PATH` | Add comparisons with an earlier analysis directory where model labels and scientific provenance match. Different configurations are not treated as matched continuations. |

A CPU-safe transfer audit is:

```bash
uv run python scripts/inspect_checkpoints.py \
  --bundle runs/inspection/eval_20261031T093812Z \
  --audit-only
```

This creates the output manifest, so the subsequent full run needs `--resume`.
Leave the default `--device cuda:0` for this audit if the full run will use that
device; the audit itself does not execute CUDA inference. If an inspection
already exists, add `--resume` to the audit as well.

To run the offline checks and cube diagnostics first:

```bash
uv run python scripts/inspect_checkpoints.py \
  --bundle runs/inspection/eval_20261031T093812Z \
  --stages offline grounding --resume
```

To finish the full suite, run the normal command with `--resume`. To rebuild only
the report after all stages have finished:

```bash
uv run python scripts/inspect_checkpoints.py \
  --bundle runs/inspection/eval_20261031T093812Z \
  --report-only --resume
```

The launcher prints a progress notice every 30 seconds. Full stage output is in
`logs/offline.log`, `logs/grounding.log`, and `logs/handoff.log`; `pipeline_status.json`
records stage progress separately from the immutable `inspection_manifest.json`.
For example:

```bash
tail -n 60 runs/inspection/eval_20261031T093812Z_analysis/logs/handoff.log
```

These local diagnostic subprocesses disable W&B and clear inherited HPC
`STMEM_*` and distributed-rank settings. They use the current checkout and the
manifest's explicit paths. The results stay in the local report directory.

## Fixed diagnostic protocol

**Offline checks** use released VideoUnmask training episodes 0 and 1 at native
decision indices 0, 32, 64, 96, and 128, with paired noise seeds 1701–1704. They
measure expert action/TCP fit and sensitivity to current/history visual evidence,
previous commanded actions, and measured state. For frozen-feature models,
neutral RGB and zeroed frozen descriptors are separate interventions. Additional
checks cover omitted-input invariants, serial/packed inference, precision,
early-history alignment, and one backward-only gradient probe. No optimizer
update occurs, and weights are checked for equality afterward.

**Reset cube-position checks** use PickXtimes validation scene 0. The real cube is
placed at baseline, X +10 cm, Y +10 cm, and Y −10 cm; both cameras are rerendered.
The robot, task, timing, and other objects are held fixed. Four paired noise draws
2701–2704 produce the first action chunks. None of these predictions is executed.
The standalone HTML files show sampled scene points, the cube, and forward
kinematics of the predicted physical joint targets. Lines connect target
positions; they are not simulated motion. Large markers indicate targets that
would execute before the next replan. Displayed colors are RGB, not CLIP
feature channels.

**Later-state grounding checks** move the cube to baseline/X +10 cm/Y +10 cm and
replay the same saved off-target commands. States 0, 40, and 80 are inspected with
four paired draws in FP32 and production precision. Coverage includes cube points,
Gaussian pooling mass, and cube-majority tokens in the current frame and latest
history block; attention is also recorded. Object poses/membership are diagnostic
labels only, never model inputs or a sampling guide. The fixed replay separates
scene-position effects from changes in the robot trajectory.

**Fresh closed-loop checks** run each model on PickXtimes validation scene 0 with
seed 2701, capped at 160 native commands. They save videos, first-close targeting
and tracking errors, grasp flags, cube lift, and input-coverage traces. Each
checkpoint uses its own production observation/action cadence. At a 20 Hz native
simulator rate, a 10 Hz target is held for two commands; there is no interpolation.
The compared configurations replan every four native commands (0.2 seconds).
Responses over the first 0.2 seconds and 0.8 seconds are reported separately from
full-chunk responses because predicted horizon durations can differ.

**Expert-prefix handoffs** use VideoUnmask training scenes 0 and 1 and PickXtimes
training scene 0. Prefix lengths are 0, 4, 16, 32, and 64 native commands, followed
by the learned policy, with a total cap of 320 native commands. VideoUnmask
prefixes come from the released demonstrations; the PickXtimes prefix comes from
the saved verified official-planner record. Two VideoUnmask full-expert replay
controls are included. There are 15 policy trials per checkpoint and two shared
expert controls. Videos and partial grasp/lift/subtask progress are retained even
when official task success is not reached within the cap.

## Outputs and interpretation

| Output | Contents |
| --- | --- |
| `index.html`, `summary.json` | Overall report and numerical summary. |
| `checkpoint_audit.json`, `inspection_manifest.json` | Checkpoint configurations, integrity checks, immutable input/protocol identities. |
| `offline/summary_table.csv`, `offline/summary_table.json` | Expert-state fit and sensitivity tables. |
| `offline/validation.json` | Independent offline diagnostic validation. |
| `reset_cube_chunks/index.html` | Reset comparison and links to one interactive four-position point-cloud viewer per checkpoint. |
| `reset_cube_chunks/<label>/<label>.html` | Standalone point-cloud viewer with paired-noise selection. |
| `reset_cube_chunks/independent_validation.json` | Recomputed physical actions, FK, scene matching, and reset-response checks. |
| `rollouts/index.html`, `rollouts/model_comparison.mp4` | Fresh-rollout gallery and synchronized comparison video. |
| `grounding/validation.json` | Recomputed grounding/cadence/coverage/media checks and detailed rollout measurements. |
| `handoff/index.html`, `handoff/summary.json` | Prefix-handoff outcomes and videos. |
| `handoff/validation.json`, `handoff/media_validation.json` | Handoff replay/action/timing checks and video validation. |
| `report_validation.json` | Final report link/asset validation. |

This suite answers whether conditioning changes predictions, whether a cube
survives visual encoding, whether production timing matches the intended
protocol, and whether the policy can grasp/lift on these fixed scenes. It does
not estimate benchmark success rate or generalization. Offline fit uses training
states. Prefix conditions and paired noise draws are correlated, not independent
episodes. Artificial input deletions may themselves be out of distribution.

A grasp flag alone does not establish a successful lift, and a lift does not
establish full task success. Prefix handoff success within the declared cap is
reported separately from partial progress. A small first-chunk displacement at
reset can coexist with stronger later targets; compare equal physical-time
windows before drawing conclusions from different horizon lengths. Checkpoints
with different training exposure, features, proprioception, history, or cadence
are not a controlled single-factor ablation. Test seeds are not used or tuned by
this inspection.
