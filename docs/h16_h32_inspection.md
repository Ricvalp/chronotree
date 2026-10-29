# H16/H32 checkpoint inspection — 29 October 2026

Both transferred checkpoints are complete and load correctly. H16's offline
action fit has improved since the earlier 16k inspection, but the inspected
PickXtimes policies still respond weakly to cube relocation and miss the cube
in their own closed-loop approaches. These bounded diagnostics do not establish
that the architecture cannot learn visual grounding.

## Checkpoints and integrity

The transferred bundle is
`runs/inspection/h16_h32_20261029T124616Z/`. Analysis snapshots and results are in
`runs/inspection/h16_h32_analysis_20261029/`.

| Checkpoint | Updates | Predicted actions | Executed before replanning | SHA-256 |
|---|---:|---:|---:|---|
| H16 | 41,000 | 16 | 4 | `7341cec3985807358662f384df66a9773a9744f7b5e15e9e0343c6b7c50d8a5b` |
| H32 | 20,000 | 32 | 4 | `24e98e053f392f808aecbdb84f4a738e42c64c36b78817650796ad8d97760fc9` |

Both use XYZ+RGB, 20 Hz actions/observations, full causal history, measured robot
state and the preceding gripper command. Previous seven-joint commands are
disabled. These are not the newer frozen-CLIP/no-proprioception experiments.
They have different training exposure, so their comparison does not isolate
the effect of action horizon. Normalizers are bit-identical; the only model
configuration difference between these two snapshots is the horizon.

The supplied SHA256SUMS match. ZIP CRC checks, finite model/optimizer tensors,
strict model/optimizer loading, scheduler/sampler step consistency and per-rank
RNG records pass. Source files were unchanged while checked. The complete audit
is `checkpoint_audit.json`; predictions use pinned inspection copies.

## First chunks with four cube positions

`reset_cube_chunks/index.html` links separate interactive H16 and H32 views.
Each includes the original cube position and X+10 cm, Y+10 cm, Y−10 cm variants
of PickXtimes validation scene 0. Both camera observations are rerendered after
moving the actual simulator cube. All four cube centers are within the nominal
spawn XY bounds, but the paired relocations remain artificial diagnostics,
not additional official episode seeds.

Robot state, other objects, task text, calibration and source clocks are held
fixed. The same four flow-noise seeds are used, and the first 16 noise vectors
also match across H16/H32. Baseline repetition is bit-identical. All four scenes
contain visible cube points (168, 167, 179 and 147 respectively).

| Response to a 100 mm cube displacement | H16 41k | H32 20k |
|---|---:|---:|
| Mean displacement of the first four TCP targets | 0.85–1.00 mm | 0.67–1.14 mm |
| Mean displacement of the common first 16 targets | 0.80–1.09 mm | 0.89–1.82 mm |

Ranges span the three translated scenes; each measurement averages four paired
noise draws. The response component along the cube shift is tiny and has mixed
signs. Some later H32 targets respond more strongly (full-chunk mean change
5.57 mm for X+10 cm), but these targets lie beyond the four commands executed
before replanning. The initial chunk makes little motion even in the baseline,
so this reset-only probe cannot determine an eventual grasp location.

The HTML displays FK of the clipped joint commands, including target
orientations. Lines connect targets in command order; they are not executed
trajectories or extra interpolated actions. The first four targets are
highlighted. Cube boxes are diagnostic annotations and never enter the policy.
No predictions are executed in this visualization probe. Validation passes 239
independent checks over 32 chunks/768 targets, plus browser/WebGL/control checks.

## Later-state grounding and closed-loop approaches

At matched replay state 80, physically moving the cube by 100 mm while retaining
the same robot trajectory gives the following FP32 first-four-target changes:

| Cube relocation | H16 41k | H32 20k |
|---|---:|---:|
| X +100 mm | 0.368 mm | 0.611 mm |
| Y +100 mm | 0.100 mm | 0.125 mm |

Thus weak position sensitivity is also present later in the approach, not just
at reset. These are paired inference probes; their altered predictions are not
executed. The independent fresh rollouts execute each policy's own chunks with
the same scene and original production seed protocol:

| PickXtimes validation scene 0, cap 160 commands | H16 41k | H32 20k |
|---|---:|---:|
| First close command | 85 | 85 |
| Observed TCP–cube XY distance at first close | 129.29 mm | 321.59 mm |
| Commanded TCP–cube XY distance at first close | 129.96 mm | 317.92 mm |
| Observed–commanded TCP distance at first close | 3.29 mm | 4.39 mm |
| Cube lifted | No | No |

Both reach the diagnostic cap, which is distinct from an official task outcome.
The large targeting errors are already in the predicted joint targets and are
much larger than tracking error. H16 improves over the earlier 16k model's
464 mm first-close miss in this scene, but still misses by about 13 cm.
`rollouts/index.html` provides both videos. Grounding/rollout checks passed
3,281 independent assertions, including replay identity, FK, native horizons,
executed commands and media integrity.

The cube survives tokenization: both models have two current cube-majority
tokens and 15 in the latest history block at state 80. Adapted cube/table
features remain distinguishable. Current visual attention mass is 3.60%/1.92%
for H16/H32, but mass alone is not causal importance; the physical relocation
probe is the stronger evidence for weak cube-position response.

## Offline action, history, command and numerical diagnostics

The preserved protocol uses real VideoUnmask training episodes 0/1, decisions
0/32/64/96/128 and four paired noise draws. This is training-state fit, not
held-out action accuracy. First-four expert TCP error is 7.95 mm for H16 and
12.71 mm for H32. H16's earlier 16k error was 11.63 mm.

After execution starts, deleting current visual tokens changes the first four
targets by only 1.04/1.16 mm, whereas deleting historical vision changes them by
24.03/40.19 mm. Reset and later states are reported separately: VideoUnmask
already supplies a conditioning video at reset, even with empty execution
history. Deletion/permutation are artificial interventions, not proof that a
particular history representation is semantically useful.

Perturbing removed joint-command channels gives bit-identical predictions in
both checkpoints. Retained gripper/availability inputs still affect outputs.
FP32 packed/serial parity passes (maximum inspected first-four TCP discrepancy
about 0.000039 mm), as do early-prefix checks. Production versus all-BF16
same-input differences are 0.49–1.22 mm. Visual gradients are finite/nonzero;
backward probes perform no optimizer updates. Independent validation passes
9,305 checks over 2,120 prediction rows, four real-state parity cases, two
backward-only probes and 24 early-prefix cases.

Detailed summaries and exact protocol adaptations are saved in `offline/`.
No long training, HPC submission, test-seed tuning or checkpoint selection is
performed by this inspection.

## Expert-prefix handoffs

The preserved protocol runs VideoUnmask training scenes 0/1 and PickXtimes
training scene 0 with prefixes of 0/4/16/32/64 expert commands, followed by each
policy's own closed-loop actions. Each trial has a 320-total-command cap. Flow
noise is paired by absolute decision time. The two full VideoUnmask expert
replay controls succeed. Pick uses the previously generated and verified
official-planner prefix, not a claimed released demonstration identity.

| Scene/outcome within the diagnostic cap | H16 41k | H32 20k |
|---|---|---|
| VideoUnmask scene 0, task success | Prefixes 32 and 64 | Prefix 64 |
| VideoUnmask scene 1, task success | Prefixes 32 and 64 | Prefix 64 |
| PickXtimes scene 0, grasp and lift | Prefixes 32 and 64 | Prefix 64 |
| PickXtimes scene 0, full task success | None | None |
| Any of these three scenes, task success from reset | None | None |

The Pick cube is not already grasped at handoff. H16 completes the first
pick/place cycle after the two successful-grasp prefixes and reaches the
second picking subtask. H32 reaches the first placement subtask after prefix
64. These are partial task accomplishments, not full success.

H16's shorter successful VideoUnmask handoffs improve over its earlier 16k
checkpoint, which succeeded only at the tested prefix 64. This still does not
resolve the failures from reset. Conditions on the same three scenes are
correlated and are not an official validation success-rate estimate. All 30
policy trials and both expert controls completed their declared protocols;
the videos and detailed comparisons are in `handoff/index.html`.
Independent validation checks 7,583 executed policy commands against their
generated chunks, 44 identical paired reset/handoff states and six exact
transport-parity cases. All 32 rollout videos and nine comparison videos pass
media validation; all 88 gallery links resolve. The overall report passes
31-link and desktop/mobile browser checks. Old/new reset comparison inputs are
bit-identical across points, masks, proprioception, preceding actions and centers.
The bundled Chromium passed gallery layout/filter checks but lacks H264 support,
so video seeking was not browser-tested there; FFmpeg independently decoded and
verified every saved video.

## Open the results

- `runs/inspection/h16_h32_analysis_20261029/index.html`: overall report.
- `reset_cube_chunks/index.html`: four-position first-chunk point clouds.
- `rollouts/index.html`: fresh 160-command PickXtimes videos.
- `handoff/index.html`: expert-prefix videos and per-condition outcomes.

Only the visualization generator changed in tracked production tooling: the
optional `--placements` list selects four scenes without changing the original
five-scene default, and first-16 metrics now remain first-16 for longer chunks.
Existing kinematics/structure HTML tests pass (8 tests), along with Ruff and
whitespace checks. The analysis wrappers, original-script hashes and literal
path/horizon adaptations are retained with the ignored experiment artifacts.
