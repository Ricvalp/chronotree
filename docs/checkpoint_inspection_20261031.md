# Three-checkpoint inspection — 31 October 2026

All three checkpoints in `runs/inspection/eval_20261031T093812Z/` passed transfer
and state-integrity checks. The complete local report is
[`index.html`](../runs/inspection/eval_20261031T093812Z_analysis/index.html).
The recurring workflow is documented in
[`checkpoint_inspection.md`](checkpoint_inspection.md).

These are fixed-scene diagnostics, with unequal training exposure and several
configuration differences. They are not official RoboMME benchmark success
rates or an isolated test of the frozen backbone. No training or Slurm job was
started and no checkpoint was modified.

## Checkpoints and protocol

| Label | Updates | Visual input | Predicted targets |
| --- | ---: | --- | --- |
| `history_h16` | 82,000 | XYZRGB, measured proprioception, no previous joint commands | 16 at 20 Hz |
| `history_h32` | 60,000 | XYZRGB, measured proprioception, no previous joint commands | 32 at 20 Hz |
| `frozen_10hz` | 60,000 | Frozen CLIP, no proprioception or previous commands | 16 at 10 Hz |

All three use full history and replan after four native 20 Hz commands. The
10 Hz model holds each target for two native commands, without interpolation.
The XYZRGB models still receive the permitted gripper/availability information;
their disabled previous-joint-command perturbations are exactly invariant.

The retained protocol matches the preceding inspection: VideoUnmask expert
states from training episodes 0/1, four paired flow-noise draws per state,
current/history/command/proprioception interventions, packing and precision
parity, early-prefix checks, and backward-only gradient checks. PickXtimes
validation scene 0 supplies four physically rerendered cube placements at
reset, later fixed-trajectory probes, and an eight-second fresh rollout.
Expert-prefix handoffs use three training scenes, five prefixes per scene,
and a 320-native-command cap. Two full expert replays serve as controls.

## Fresh PickXtimes validation rollout

| Model | First close (native command) | Observed TCP–cube XY miss | Maximum cube rise | Grasp detected |
| --- | ---: | ---: | ---: | --- |
| H16 | 90 | 298.14 mm | No lift | No |
| H32 | 112 | 110.20 mm | No lift | No |
| Frozen 10 Hz | 103 | 17.31 mm | 133.93 mm | Yes, commands 103–160 |

The frozen model now picks up and holds the cube in this fixed validation
scene. The saved frame at command 160 visibly confirms the lifted cube. Its
first-close commanded XY miss is 22.44 mm and joint-target tracking error at
that step is 2.41 mm. These TCP-center measurements are not jaw-clearance or
complete grasp-pose errors. The trial stops after eight seconds; a successful
pickup is not complete PickXtimes success.

Compared with the prior matched checkpoints, the first-close observed XY miss
changed from 86.74 to 298.14 mm for H16, 139.86 to 110.20 mm for H32, and 23.60
to 17.31 mm for frozen vision. The older frozen checkpoint only bumped the cube
about 2.19 mm in this validation trial. Training longer therefore did not improve
closed-loop targeting uniformly across models.

## Expert-state fit and visual dependence

All distances below refer to the same first 0.2 seconds of native commands.
Expert error averages ten expert states; intervention effects average the eight
states after reset and four paired noise draws.

| Model | Previous → current expert TCP error | Delete current visual tokens: prediction change | Delete history visual tokens: prediction change |
| --- | ---: | ---: | ---: |
| H16 | 6.70 → 5.97 mm | 1.18 mm | 18.76 mm |
| H32 | 9.16 → 8.24 mm | 1.35 mm | 33.66 mm |
| Frozen 10 Hz | 11.54 → 7.57 mm | 11.47 mm | 74.30 mm |

All three fit these expert states better. Frozen vision remains much more
sensitive to current visual content; zeroing its current frozen descriptors
changes the imminent prediction by 39.82 mm. Deletion is an artificial,
potentially out-of-distribution intervention: response magnitude does not by
itself establish useful object localization.

The real cube-relocation probes remain necessary. At the later off-target
replay state 80, moving the cube by 100 mm changes the first 0.2 seconds by less
than 1 mm for every model. The cube is represented in the sampled points and
current visual tokens; this is not explained simply by having no cube samples.
At reset, some models respond substantially later in the predicted chunk while
the imminent execution window changes much less. The four-position viewers
show both windows explicitly.

FP32 serial/packed inference differs by at most 0.0011 mm in the measured full
TCP paths. Early-history parity is also below 0.002 mm. The tracked replacement
for the previous ad hoc inspection reproduced the old H32 decision-64,
noise-1701 raw action chunk bit-for-bit. No optimizer update occurs during the
gradient probes.

## Expert-prefix handoffs

The cells list prefixes yielding official task success within the diagnostic
cap. Prefix lengths are native 20 Hz commands, not model target indices.

| Training scene | H16 | H32 | Frozen 10 Hz |
| --- | --- | --- | --- |
| VideoUnmask 0 | 32, 64 | 64 | 0, 4, 16, 32, 64 |
| VideoUnmask 1 | 32, 64 | 4, 16, 32, 64 | 0, 4, 16, 32, 64 |
| PickXtimes 0 | None | None | None |

Both full VideoUnmask expert controls succeed. Frozen vision now succeeds from
reset on both VideoUnmask scenes; the previous checkpoint failed scene 1 from
reset. The XYZRGB models still miss from reset on all three training scenes.

On PickXtimes training scene 0, frozen vision does **not** grasp from reset at
60k, although its 27k checkpoint lifted the cube on that same probe. Prefixes
4/16/32/64 now allow it to grasp, lift and complete the first pick/place cycle;
prefix 64 reaches the button stage. H16 lifts after prefixes 32/64 and H32 after
prefix 64. None finishes the whole task within the cap. These results temper the
successful frozen-model pickup on the separate validation scene: reliability
remains dependent on the scene and preceding trajectory.

## Reproducibility and validation

The new tracked entry point is `scripts/inspect_checkpoints.py`. It discovers
bundle labels from `SHA256SUMS`, validates actual checkpoint state, hashes all
local diagnostic assets, and records an immutable manifest. Stage logs and
progress live separately. Resume reuses matching completed stages and reruns
their independent audits; changed inputs, scientific source, recorded stage
code or runtime identity require a fresh output. Interrupt handling terminates
child process groups before releasing the output lock.

Measured checks for this bundle:

- 3,250 prediction rows and 52,729 offline checks, including 36 early-prefix
  cases, six real packing/precision states, three gradient probes, and three
  omitted-input invariants.
- 383 reset checks and 4,978 grounding/media checks.
- 45 policy handoffs and two expert controls; 9,349 policy commands, 1,357 expert
  commands, 74 paired-state checks and 2,104 paired-RNG checks audited.
- 47 handoff videos, nine handoff comparison videos, three fresh rollout
  videos, and one fresh rollout comparison video, with frame/timing checks.
- 55 CPU tests pass. Ruff and the architecture diagram snapshot check pass;
  production model architecture and diagram contents were unchanged.
- Desktop/mobile browser checks pass for the overall report, four-position
  viewers and galleries. Cached Chromium lacks H264 playback, so media decoding
  and frame counts are verified with FFmpeg separately from browser layout.

The actual RTX 5000 Ada device passed the device doctor. The runtime sidecar
records the environment observed when the new launcher was exercised; it does
not claim retrospective version proof for results generated earlier in the
same session. Data, weights, rendered artifacts and logs remain Git-ignored.
