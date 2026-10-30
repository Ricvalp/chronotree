# Four-checkpoint inspection — 30 October 2026

The transferred checkpoints are complete and load correctly. Frozen vision and
the current-only model show stronger visual dependence and better approaches in
the inspected PickXtimes validation scene, but neither achieves a sustained lift
there within the 160-command window. Current-only succeeds from reset on both
inspected VideoUnmask training scenes. Frozen vision with history also picks up
the cube from reset on the inspected PickXtimes training scene, without completing
the entire task inside the longer diagnostic cap.

These are bounded diagnostics on fixed scenes, not official benchmark success
rates. The models differ in training exposure, visual features, proprioception,
history, action cadence and sometimes horizon; their comparison does not isolate
one architectural cause.

## Checkpoints and artifacts

Inputs are immutable snapshots in `runs/inspection/eval_20261030/`. Results and
reproduction scripts are in `runs/inspection/eval_20261030_analysis/`.

| Checkpoint | Updates | Visual input | History / proprioception | Horizon / target cadence |
|---|---:|---|---|---|
| `history_h16` | 60,000 | XYZRGB | Full / measured state | 16 / 20 Hz |
| `history_h32` | 39,000 | XYZRGB | Full / measured state | 32 / 20 Hz |
| `frozen_10hz` | 27,000 | Frozen CLIP | Full / none | 16 / 10 Hz |
| `current_only` | 100,000 | Frozen CLIP | None / none | 16 / 10 Hz |

All four replan after four native 20 Hz commands. The 10 Hz models predict two
targets for that interval and hold each for two native commands; they do not
interpolate. Their observations are also sampled at 10 Hz. The two XYZRGB models
remove previous joint commands but retain preceding gripper/availability inputs;
the frozen-vision models remove all previous command channels.

All four SHA256SUMS match. ZIP CRCs, finite model/optimizer tensors, strict model
and optimizer loads, scheduler/sampler steps and per-rank RNG records pass.
Checkpoint identities and complete saved configurations are retained in
`checkpoint_audit.json`. The local RTX 5000 Ada compute doctor passed FP32, BF16
and FP16. Simulation rendering was exercised separately by the real rollouts.

Open `index.html` for the overall report, `reset_cube_chunks/index.html` for
interactive point clouds, `rollouts/index.html` for fresh validation-scene videos,
and `handoff/index.html` for expert-prefix outcomes and videos.

## Cube position and fresh closed-loop approaches

The reset visualization physically renders the original PickXtimes validation
scene 0 and X+10 cm, Y+10 cm and Y−10 cm cube placements. Both cameras, production
sampling and frozen features where applicable are recomputed. Robot state, goal,
other objects, clocks and paired noise remain fixed. No privileged cube labels
enter the policy, and none of these counterfactual predictions is executed.

There are 64 generated chunks and 1,280 targets, with four noise seeds per
checkpoint/placement. The cube contributes 147–179 visible sampled points.
The first 0.2 seconds of targets change only about 0.5–1.5 mm for a 100 mm cube
translation. Later reset targets respond more strongly: current-only's complete
1.6-second chunk changes by 18–24 mm on average, with endpoint changes around
55–65 mm. Thus the reset-only first-target probe understates its later predicted
response. FK targets and lines in these viewers are not executed trajectories.

The later-state probe replays an earlier policy's identical off-target command
prefix under each cube relocation. At native command 80, the immediate response
remains small even for the frozen-vision models:

| Model | X+100 mm: first 0.2 s response | Y+100 mm: first 0.2 s response |
|---|---:|---:|
| History H16 | 0.488 mm | 0.433 mm |
| History H32 | 0.703 mm | 0.518 mm |
| Frozen history | 2.234 mm | 1.212 mm |
| Current-only | 1.722 mm | 1.608 mm |

These are production BF16 means over four paired noise draws. Full-horizon mean
responses at that state are approximately 0.5–2.5 mm; the stronger late-target
response at reset does not persist there. All models see 123 cube points and two
cube-majority current tokens. The latest retained historical block has 15 such
tokens for H16/H32 and 11 for frozen history; current-only correctly has none.

Separate fresh PickXtimes validation-scene rollouts execute each policy's own
commands for up to 160 native commands:

| Model | First close command | Observed TCP–cube XY miss | Max cube rise | Grasp flag |
|---|---:|---:|---:|---|
| History H16 | 92 | 86.74 mm | ~0 mm | No |
| History H32 | 98 | 139.86 mm | ~0 mm | No |
| Frozen history | 109 | 23.60 mm | 2.19 mm | Briefly at 109–112 |
| Current-only | 117 | 37.30 mm | 1.25 mm | No |

All reach the declared diagnostic cap; none demonstrates a sustained lift.
The first-close observed miss uses post-command TCP and cube positions. The
commanded miss uses the FK target and pre-command cube position. These are
TCP-center distances, not jaw-clearance or full grasp-pose errors. Commanded
misses are also substantial, so actuator tracking alone does not explain them.
Every initial raw chunk exactly matches the corresponding reset viewer's first
noise seed. Independent validation covers 6,568 checks, including cadence,
causality, FK, pooling, paired scene identity and media integrity.

## Offline sensitivity, omitted inputs and numerical checks

The unchanged real-data protocol uses released VideoUnmask training episodes
0/1, native decision indices 0/32/64/96/128 and four paired flow-noise seeds.
Native-command metrics account for the 10 Hz executor's zero-order hold.

| Model | Expert TCP error, first 0.2 s | Delete current vision: change | Delete history vision: change |
|---|---:|---:|---:|
| History H16 | 6.70 mm | 0.97 mm | 22.46 mm |
| History H32 | 9.16 mm | 1.07 mm | 32.27 mm |
| Frozen history | 11.54 mm | 15.69 mm | 76.07 mm |
| Current-only | 9.32 mm | 210.21 mm | 0 mm |

Action fit averages all ten states; deletion responses use the eight states after
reset. These are training-state fit and artificial perturbations, not held-out
accuracy or proof of correct cube localization. Current-only's history deletion
is an architectural invariance control. Zeroing its frozen descriptors changes
the same imminent targets by about 55 mm after reset.

Previous-joint perturbations are invariant for all models. Both no-proprioception
models also ignore all command/availability changes and current/historical
measured proprioception changes. Current-only ignores injected nonempty history,
altered frame timestamps/source indices and changed context clocks.

Packed/serial FP32 full-chunk mean TCP differences are at most 0.0000514 mm;
early-prefix cases are below 0.000295 mm. Production versus all-BF16 native
first-four mean differences span 0.386–1.556 mm across eight fixed-state cases.
Visual MLP and frozen-feature projection gradients are finite and nonzero.
No optimizer update was performed. These fixed-input checks do not establish
that numerical differences cannot accumulate over a closed-loop trajectory.

Independent validation passes 65,983 checks over 4,360 saved predictions, eight
real-state parity cases, four backward-only probes, 48 early-prefix cases and ten
explicit omitted-input invariance cases. Data/cache identities and the exact
protocol remain in `offline/`.

## Expert-prefix handoffs

The suite uses the same VideoUnmask training scenes 0/1 and PickXtimes training
scene 0, prefixes of 0/4/16/32/64 genuine expert commands, and a total cap of 320
native commands. Pick uses the existing verified official-planner prefix; it is
not represented as a released demonstration. Both full recorded VideoUnmask
expert controls succeed. Noise is aligned by absolute decision time within each
model; H16/H32 sequential streams differ after reset because their native noise
tensor sizes differ.

| Scene / official task success | History H16 | History H32 | Frozen history | Current-only |
|---|---|---|---|---|
| VideoUnmask train 0 | Prefixes 32, 64 | Prefixes 32, 64 | All five | All five |
| VideoUnmask train 1 | Prefixes 32, 64 | Prefixes 16, 32, 64 | Prefixes 4, 16, 32, 64 | All five |
| PickXtimes train 0 | None within cap | None within cap | None within cap | None within cap |

Current-only's two VideoUnmask successes from reset are promising seen-scene
behavior, not evidence that history is unnecessary or that it generalizes to
unseen hidden-object configurations. Frozen history grasps/lifts the Pick cube
from reset and reaches its first placement stage. H16 reaches the button stage
after prefixes 16/32/64, but does not achieve official success within the cap.
These partial accomplishments are reported separately from task success.

The current-only Pick trial from reset has a brief contact, not a successful
pickup. First closure occurs at command 87, at cube height but 19.26 mm off-center.
Grasp flags appear at commands 88–92; it reopens at 93 after 0.30 seconds without
a sustained lift. Its minimum XY distance of 1.02 mm occurred while the TCP was
about 294 mm above the cube. Its maximum cube rise occurred before closure with
the gripper open. The contact sheet and CPU-derived trace review are in
`handoff/current_only_pick_review/`; 1,725 independent checks validate these
measurements and the extracted video frames. These observations do not identify
a unique underlying cause.

All 60 policy trials and two expert controls complete their declared protocols.
The final consistency audit checks 12,037 policy commands and 2,700 paired RNG
states. Videos, gallery, per-trial JSON, media checks and exact prefix outcomes
are retained in `handoff/`. Correlated prefixes on three fixed training scenes
must not be reported as a benchmark success rate.

## Reproducibility and limitations

The input checkpoint hashes are rechecked after analysis. Diagnostic adaptations,
original helper references, configs and independently recomputed measurements are
retained under the ignored analysis directory. No training, Slurm submission,
test-seed tuning, production architecture change or checkpoint mutation occurred.
Some diagnostic processes overlapped on the workstation after memory headroom
was checked; their elapsed times are not performance comparisons.

Interactive controls and desktop/mobile layouts were inspected with local
Chromium. The bundled browser lacks H264 playback; video integrity is established
separately by frame counting and decoding. Official all-task validation remains
the separate HPC evaluation campaign using the same immutable snapshots.
