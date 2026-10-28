# Frozen vision with and without proprioception

This implements [the experiment brief](../chronotree_frozen_vision_no_proprio_experiment.md).
The comparison asks whether removing numerical robot state improves targeting
when both policies receive the same frozen pretrained visual descriptors. It
does not isolate visual pretraining's benefit; that needs an additional matched
XYZRGB arm. Full feature preparation and HPC training remain experiment execution
steps, not completed implementation checks.

## Configuration and architecture

| Setting | FV-Prop | FV-NoProp |
| --- | --- | --- |
| Configuration | `configs/teacher_fv_prop.yaml` | `configs/teacher_fv_no_prop.yaml` |
| Frozen visual backbone | OpenAI CLIP RN50 through `layer1` | Same |
| Current/historical measured state | Retained | Removed |
| Previous commanded joints, gripper, availability | All removed | All removed |
| Prediction/execution horizon | 32 / 4 native 20 Hz commands | Same |
| Updates / peak learning rate | 100,000 / 0.0001 | Same |
| Effective batch / microbatch per GPU | 128 / 8 | Same |
| Encoder block batch | 32 | Same |
| GPUs | Four B200s, `aiq`, untyped `--gres=gpu:4` | Same |
| Comparison checkpoints | 25k, 50k, 100k | Same |

FV-NoProp inherits FV-Prop. Make common capacity or schedule changes in
`teacher_fv_prop.yaml`. Both train freshly with matching seeds, data order and
initial shared-module weights. Four GPUs with microbatch eight use four
accumulation passes per optimizer update. Activation checkpointing remains off.
Legacy XYZRGB configurations and checkpoint state dictionaries retain their
previous behavior.

| Parameter count | FV-Prop | FV-NoProp |
| --- | ---: | ---: |
| Total policy parameters | 6,168,970 | 6,168,970 |
| Trainable policy parameters | 6,100,618 | 5,892,746 |
| Disabled policy parameters (`requires_grad=False`) | 68,352 | 276,224 |
| Separate frozen CLIP early-stage parameters | 244,576 | 244,576 |
| Total policy + early visual backbone | 6,413,546 | 6,413,546 |

Unused state/command modules are retained but frozen to preserve matching shared
initialization; their outputs do not reach the model. Frozen visual parameter
counts cover the executed stem/layer1 only, not CLIP's discarded later stages.

Each native 256×256 RGB image independently produces a `[256,64,64]` layer1 map.
Original observing-camera and integer-pixel identities follow the selected
XYZRGB points through crop, voxelization and sampling. Bilinear lookup uses
feature coordinate `((u-1)/4,(v-1)/4)`, border padding and `align_corners=False`,
matching the early-convolution receptive fields. CLIP RGB normalization, camera
order, coordinate mapping and extraction precision are versioned. There is no
image crop, random geometric augmentation or future-frame encoder.

The backbone remains frozen and in eval mode. Its FP32 extraction contract uses
cuDNN enabled, deterministic algorithms, benchmarking disabled, TF32 disabled
and highest float32 matmul precision, restoring surrounding global settings.
Online inference extracts each arriving frame; training reads raw frozen FP16
descriptors. Trainable LayerNorm and a 256→128 projection precede the point MLP,
which also receives original XYZRGB. Projection outputs are never cached.
Geometric selection, 32 historical centers/frame, current visual quotas,
neighborhoods and temporal groups remain matched. Features never influence point
selection or sorting.

FV-NoProp removes the current state token and historical robot-event tokens,
including state-derived end-effector anchors. Their slots are not reassigned to
visual tokens. Router inputs follow the same switches. FV-Prop retains measured
state but excludes every previous-command field, including availability. The
optional `model.proprioception_mode: current_only` path supports a later
experiment. Visible robot geometry, calibration, clocks and visual-token metadata
remain available in both arms. A metadata ablation is a separate future study.

## Data and storage

```text
/hpc/home/phi/rvalperga/data/robomme-data/
  raw/record_dataset_<task>.h5            original RGB-D demonstrations
  prepared_balanced_v2/<task>.h5         existing XYZRGB + centers
  alignment.json                        existing verified alignment evidence
  features_clip_rn50_layer1_v1/<task>.h5  new descriptors + pixel provenance
```

Only the last directory is new. Do not regenerate balanced-v2 geometry or repeat
the alignment audit. Feature preparation repeats deterministic point selection
and requires exact equality with existing XYZRGB/validity arrays before attaching
features. It verifies the raw SHA256. Cache identity covers source/geometry
recipe, CLIP checksum, layer, preprocessing, shape, pixel correspondence and
precision contract. Mismatched caches fail explicitly.

Raw FP16 descriptors cost **1 MiB/frame** at 2,048 points; provenance adds about
10 KiB/frame. Budget approximately **98.6 GiB per 100,000 frames**, plus small
HDF5 headers. Estimate the actual dataset below before allocating storage.
Descriptors are contiguous and uncompressed for bounded memory-mapped reads.
Workers enqueue lightweight row references; only the next encoder block batch
is materialized. Whole descriptor episodes are not retained in the prefetch
queue.

Writers use `<task>.partial.h5` and completed-row counters. A matching partial
file resumes. Completed files are immutable: `--skip-existing` verifies/reuses
them. Different recipes require a new `visual_features.cache_dir`. `--max-frames`
creates a diagnostic cache which training rejects; use a separate output folder
for those probes.

## HPC workflow

### 1. Load the site and download the public weights once

After pushing/pulling the implementation:

```bash
cd /hpc/home/phi/rvalperga/chronotree
set -a
source hpc/site_b200_4gpu.env
set +a
export STMEM_SITE_FILE="$STMEM_REPO_ROOT/hpc/site_b200_4gpu.env"

"$STMEM_PYTHON" -m stmem.vision.frozen --download \
  --output "$STMEM_CACHE_ROOT/backbones/clip_RN50.pt"
```

The official CLIP download is checksum-verified. Compute jobs reuse the shared
file without Internet access. Existing W&B credentials are reused. Both arms
have distinct run directories and tracking identities in the same project;
there is no need for another site file.

### 2. Estimate storage and inspect features

This reads manifests/layouts only and needs no GPU:

```bash
"$STMEM_PYTHON" -m stmem.data.prepare_features \
  --config configs/teacher_fv_prop.yaml --episodes 100 --dry-run
```

This bounded inspection uses one allocated GPU and writes HTML/PNG overlays,
without altering data:

```bash
srun --partition=aiq --gres=gpu:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G --time=00:10:00 \
  "$STMEM_PYTHON" scripts/inspect_frozen_vision.py \
    --config configs/teacher_fv_prop.yaml \
    --sim-task PickXtimes --split val --episode 0 \
    --output-dir runs/inspection/fv_pick_reset_features
```

The report checks pixels, RGB lookup and reprojection and visualizes dense-map
variation. PCA colors are not object masks or proof of learned cube sensitivity.

### 3. Prepare the shared sidecars

After checking storage capacity:

```bash
mkdir -p hpc/logs
export STMEM_CONFIG="$STMEM_REPO_ROOT/configs/teacher_fv_prop.yaml"
sbatch --array=0-15%4 --export=ALL hpc/jobs/prepare_features.sbatch
```

One array element prepares one task, in configuration order; up to four run
simultaneously. Each requests one untyped GPU in `aiq`, four CPUs, 16 GB RAM and
24 hours. This parallelizes across tasks with independent writers/backbones.
Logs are `hpc/logs/stmem_features_<array-job>_<task-index>.out` and `.err`.
Progress includes throughput, ETA and bytes. Rerunning resumes matching partial
files and verifies/skips completed tasks. Wait for all 16 final `.h5` files.

### 4. Measure longest-history capacity

Microbatch eight is the retained recipe, not a proven full-history B200 memory
limit. This bounded profiling job uses long released training histories:

```bash
"$STMEM_PYTHON" hpc/profile_memory.py \
  --site hpc/site_b200_4gpu.env --profile b200_4gpu \
  --config configs/teacher_fv_prop.yaml --name fv_prop_long_history \
  --scenario long-history --warmup-steps 1 --steps 1 --trace-steps 0 \
  --walltime 00:30:00 --submit
```

FV-Prop has more state tokens and is the natural first capacity check. Inspect
memory and timings before long training. If needed, reduce the common microbatch
or encoder block batch in `teacher_fv_prop.yaml`, preserving effective batch 128,
and repeat with a new profiler name. Both arms inherit the same change. Encoder
block batch controls how many independent local blocks run together; it does
not increase history length or token counts. It bounds a pooling invocation and
descriptor transfer, but **does not bound all activations retained for backward**
across that microbatch's history. The full-history capacity check still matters.

### 5. Render and submit both fresh teachers

Render first:

```bash
for arm in prop no_prop; do
  "$STMEM_PYTHON" hpc/plan.py \
    --site hpc/site_b200_4gpu.env --profile b200_4gpu \
    --teacher-config "configs/teacher_fv_${arm}.yaml" \
    --run-root "runs/teacher_fv_${arm}" \
    --output-dir "hpc/generated/teacher_fv_${arm}" \
    --plan core_seed0 --only seed0_teacher \
    --log-every 1 --attention-every 10 --progress-every-seconds 30 --dry-run
done
```

Inspect each generated `plan.json`, `configs/seed0_teacher.yaml` and
`jobs/seed0_teacher.sbatch`: four untyped GPUs, `aiq`, 64 CPUs, 128 GB RAM and
48 hours. Generated snapshots are separate from source configurations. Submit:

```bash
for arm in prop no_prop; do
  "$STMEM_PYTHON" hpc/plan.py \
    --site hpc/site_b200_4gpu.env --profile b200_4gpu \
    --teacher-config "configs/teacher_fv_${arm}.yaml" \
    --run-root "runs/teacher_fv_${arm}" \
    --output-dir "hpc/generated/teacher_fv_${arm}" \
    --plan core_seed0 --only seed0_teacher \
    --log-every 1 --attention-every 10 --progress-every-seconds 30 --submit
done
```

These independent jobs may run simultaneously when resources permit. Outputs:

```text
runs/teacher_fv_prop/seed0/teacher/
runs/teacher_fv_no_prop/seed0/teacher/
```

Planned snapshots are `step_00025000.pt`, `step_00050000.pt` and
`step_00100000.pt`; `last.pt` tracks resumable progress. After a safe stop, resume
the same arm with the same planner command plus `--resume --submit`, preserving
its directories and scientific configuration. Neither arm starts from an old
XYZRGB or previous-joint-only checkpoint.

Use **`--only seed0_teacher`**. The generic core DAG has a different two-checkpoint
selection protocol and downstream cache/router stages. It does not implement
this three-checkpoint FV comparison.

### 6. Evaluate both snapshots on one H200 each, saving ten videos

This submits both 25k checkpoints on the same fixed validation episodes 0 and 1
for all 16 tasks. Repeat with `STMEM_EVAL_STEP=50000` and `100000` when available.
Each run copies an immutable snapshot into a new evaluation directory:

```bash
cd /hpc/home/phi/rvalperga/chronotree
set -a
source hpc/site_b200_4gpu.env
set +a
export STMEM_SITE_FILE=/dev/null
export STMEM_METHOD=FULL STMEM_SPLIT=val
export STMEM_EVAL_STEP=25000
printf -v STMEM_STEP_FILE 'step_%08d.pt' "$STMEM_EVAL_STEP"

for arm in prop no_prop; do
  export STMEM_RUN_ROOT="$STMEM_REPO_ROOT/runs/teacher_fv_${arm}"
  export STMEM_EVAL_OUTPUT="$STMEM_RUN_ROOT/seed0/evaluation/FULL_val_step${STMEM_EVAL_STEP}_videos10"
  mkdir -p "$STMEM_RUN_ROOT/seed0/evaluation"
  mkdir "$STMEM_EVAL_OUTPUT" || break
  cp "$STMEM_RUN_ROOT/seed0/teacher/$STMEM_STEP_FILE" \
     "$STMEM_EVAL_OUTPUT/checkpoint.pt" || break
  export STMEM_CHECKPOINT="$STMEM_EVAL_OUTPUT/checkpoint.pt"
  export STMEM_CONFIG="$STMEM_EVAL_OUTPUT/config.json"

  "$STMEM_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from stmem.training.runtime import load_checkpoint

state = load_checkpoint(os.environ['STMEM_CHECKPOINT'])
assert state['step'] == int(os.environ['STMEM_EVAL_STEP'])
cfg = state['config']
cfg['paths'].update(
    data_root=os.environ['STMEM_DATA_ROOT'],
    cache_root=os.environ['STMEM_CACHE_ROOT'],
    run_root=os.environ['STMEM_RUN_ROOT'],
)
cfg['model']['visual_features']['checkpoint'] = str(
    Path(os.environ['STMEM_CACHE_ROOT']) / 'backbones/clip_RN50.pt'
)
cfg['evaluation'].update(episode_count=2, video_cap=10)
Path(os.environ['STMEM_CONFIG']).write_text(json.dumps(cfg, indent=2) + '\n')
print('Evaluation checkpoint:', os.environ['STMEM_CHECKPOINT'])
print('Evaluation configuration:', os.environ['STMEM_CONFIG'])
PY
  test -s "$STMEM_CONFIG" || break
  sbatch --partition=gpuq --gres=gpu:1 \
    --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=32G --time=06:00:00 \
    --output="$STMEM_EVAL_OUTPUT/slurm-%j.out" \
    --error="$STMEM_EVAL_OUTPUT/slurm-%j.err" --export=ALL \
    hpc/jobs/evaluate.sbatch \
      --output "$STMEM_EVAL_OUTPUT" --episode-count 2 --targeting-diagnostics
done
```

The explicit scheduler options override the generic evaluation template. The
B200 site supplies paths/credentials; the job requests `gpuq` with **untyped
`--gres=gpu:1`**. `STMEM_SITE_FILE=/dev/null` prevents `common.sh` from resetting
the exported run/configuration paths. It does not change the checkout or create
a worktree. Reload the B200 site as in step 1 before another preparation array.

Videos are in `<evaluation-output>/videos/`. The cap is ten episodes per submitted
shard, chosen in task/episode order—not ten per task. `episodes/` holds complete
records, `summary.json` official success, `diagnostics.json` timing/resources,
and `targeting.json` a compact target-inspection report. Evaluation remains
grouped separately from training in the same W&B project.

Choose among 25k/50k/100k by task-macro official success on this fixed validation
subset, breaking ties toward the earlier checkpoint. Apply the same rule to
both arms and record selected checkpoint hashes. Official test remains reserved
for the frozen protocol and requires the repository's test manifest.

For a faster development-only check, use a separate output directory and set
`cfg['data']['tasks'] = ['PickXtimes', 'VideoUnmask']` in the snapshot-config
script. Keep these tasks/seeds identical between arms. Four development episodes
are not the all-task selection score.

## Targeting diagnostics and sensitivity HTML

`--targeting-diagnostics` reads object poses and `agent.is_grasping` strictly as
evaluation labels. It returns the exact original reset/step objects, does not
call task-evaluation functions, and adds nothing to `Frame` or policy context.
Unsupported tasks are `not_applicable`; diagnostic failures remain separate
from official outcomes. The existing W&B dashboard keeps its two progress and
success-rate charts; details stay in artifacts.

PickXtimes tracks `target_cube`. VideoUnmask tracks **`bin_0`**, the first requested
container, rather than its hidden cube. This initial target stays fixed if a
later subtask concerns another container. Reports distinguish official task
success, contact grasp, new grasp after reset, height crossing and grasp-plus-lift.
The pinned height thresholds are cube origin >5 cm and container origin >15 cm.
VideoUnmask's official pickup uses height alone, so strict contact-plus-lift is
an additional diagnostic. Distances are TCP-to-actor-origin XY/3D distances,
not surface/contact gaps. Tracking error compares FK of the clipped joint target
with observed post-command TCP and includes controller lag/dynamics. Per-step
trajectories and read-only subgoal-index progress are retained.

Terminal wrapper states are excluded because of its automatic extra step. A
terminal-only grasp can therefore be unobserved even when official success is
true. Observer runtime and diagnostic errors are explicit.

On a GPU allocation, compare matched trained snapshots at physically displaced
reset scenes, with four common noise seeds:

```bash
"$STMEM_PYTHON" scripts/visualize_reset_cube_chunks.py \
  --checkpoint "fv_prop=$STMEM_REPO_ROOT/runs/teacher_fv_prop/seed0/teacher/step_00050000.pt" \
  --checkpoint "fv_no_prop=$STMEM_REPO_ROOT/runs/teacher_fv_no_prop/seed0/teacher/step_00050000.pt" \
  --visual-checkpoint "$STMEM_CACHE_ROOT/backbones/clip_RN50.pt" \
  --split val --episode 0 --seeds 2701 2702 2703 2704 \
  --output runs/inspection/fv_reset_cube_val0_step50000
```

Repeat on fixed episode 1 with a new output folder. This generates H32 chunks
without executing them and without stale execution history. Inspect both the
first four and later targets: initial holding can make the first four nearly
static. FK overlays are commanded poses, not guaranteed feasible trajectories.
Closed-loop timelines cover approach/close/lift phases. Later current-image-only
relocations must be labeled separately because unchanged history would contain
conflicting object positions.

## Validation and remaining uncertainty

Focused CPU checks cover unchanged geometry selection; frozen parameters/BN;
projection gradients; complete command removal; raw-state mutations that rebuild
derived metadata; serial/packed/streaming behavior; causality; lazy cache
resumption/rejection; legacy compatibility; and unchanged evaluation telemetry.
Use `uv run pytest` and `uv run python -m stmem.doctor` on the actual device.
Synthetic checks remain labeled diagnostics.

The full CPU regression suite passes **724 tests**, with one skipped check whose
legacy default prepared-data path is absent. The actual released subset was
tested separately at its explicit inspection path. The device doctor passes
FP32, BF16 and FP16. Both teacher planner dry-runs resolve to distinct roots and
the intended four-GPU `aiq` resources; no HPC jobs were submitted.

| Final bounded check | Result |
| --- | --- |
| CPU regression | 724 passed, 1 skipped; real subset checked separately |
| Actual-device doctor | FP32, BF16 and FP16 passed |
| Real feature cache | 464 frames; 491,310,800 bytes |
| Local preparation | 30.25 s total; 26.64 s extraction; 17.42 frames/s |
| Real optimizer smoke | One update/arm, 190 history frames; about 5.52 GB peak |
| Production simulator smoke | Four episodes, 36 issued commands, 12 H32 decisions with K4; four videos |
| Conditioning/reset | All 66 VideoUnmask conditioning frames retained; PickXtimes reset has empty history |
| Final cached/online features | Bitwise equal for inspected real frame, also under outer BF16/high matmul |
| Legacy checkpoint loading | Actual H16 16k checkpoint strict-loads with 6,098,698 policy parameters |

The legacy load preserves XYZRGB, full measured proprioception and its existing
previous-joint-command ablation; it does not silently apply complete command
removal or frozen vision to an old checkpoint.

The final strict-FP32 sidecar for two released VideoUnmask episodes contains
464 frames and **491,310,800 bytes**. On the RTX 5000 Ada, the bounded preparation
took 30.25 seconds including integrity work, or 26.64 seconds / **17.42 frames/s**
for extraction. This is a local measurement with warm I/O, not an HPC estimate.
For an inspected conditioning frame, cached and online raw descriptors match
bitwise, also under outer BF16 autocast and high matmul precision. Exact geometry,
camera and pixel identities match. Earlier implementation caches that allowed
ambient TF32 are superseded and rejected by the final identity contract.

Each arm completed one real FP32 optimizer update at microbatch/effective batch
one with **190 history frames**. The updates took 0.80–0.84 seconds excluding
loader wait (about 0.93–0.96 seconds with surrounding measured work), with about
5.52 GB peak allocated GPU memory. Historical token counts were 6,080 for
FV-NoProp versus 6,270 for FV-Prop: the difference is exactly 190 event tokens,
with unchanged visual coverage.

Both one-update checkpoints also completed production evaluation on fixed
PickXtimes and VideoUnmask validation scene 0 with an intentionally tiny requested
eight-step limit. All four runs produced finite H32 chunks, used K4 replanning,
wrote videos and targeting reports, and had no feature/policy/simulator errors.
They timed out as expected after nine external commands under the wrapper's
effective limit, with the terminal state excluded from targeting aggregates.
VideoUnmask retained all 66 conditioning frames. These are execution checks,
not comparative performance results.

Final generated evidence is under `runs/inspection/fv_implementation/`: the
`data/features_strictfp32/` sidecar, `train_{prop,no_prop}_strictfp32/`,
`rollout_{prop,no_prop}_strictfp32/`, and
`rollout_smoke_strictfp32_verification.json`. Feature overlays are under
`runs/inspection/frozen_vision_20261028/`. These large/local artifacts remain
ignored by git.

Small local updates and deliberately short rollouts establish execution paths,
not full-history four-B200 capacity or targeting improvement. A frozen backbone,
removal of state or lower imitation loss does not guarantee improved autonomous
grasping. Compare targeting, grasp/lift, commanded versus observed trajectories
and full success together. Delay costly router-label generation and other
memory-method training until the teacher recipe is useful.
