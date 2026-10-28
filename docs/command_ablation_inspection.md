# Previous-joint-command ablation inspection — 28 October 2026

The matched 15k continuation experiment verifies that the joint-command input
ablation works, but does not fix the weak response to cube position in the
inspected PickXtimes scene. Removing those inputs improves several expert-prefix
handoffs relative to the normally continued model. It does not establish reliable
control from reset or a general improvement over the original parent.

## Checkpoints and comparison

| Model | Optimizer step | Previous joint targets | Status |
|---|---:|---|---|
| Original lower-LR parent | 49,978 | Available | Prior diagnostic artifacts reused |
| Control continuation | 64,978 | Available | Full matched +15k experiment |
| Ablated continuation | 64,978 | Removed | Full matched +15k experiment |
| Fresh ablation | 16,000 | Removed | Earlier training; not exposure-matched |

The two continuations share the exact parent SHA256
`0217288395612d1973e868297349995d09aaab19a6caf5473702830a8528878b`,
data, normalization, sample order contract, peak LR 0.0001, 100k LR horizon and
15k-update continuation budget. Their scientific configurations differ only in
`model.condition_on_previous_joint_commands`. This removes the seven prior joint
targets from current and historical inputs. Measured robot state, prior gripper
commands, availability flags, text and visual history remain available.

All three new files pass ZIP CRC, finite model/optimizer tensors, strict model and
optimizer loading, scheduler/sampler/RNG consistency and lineage checks. Each has
6,098,698 parameters. Normalization tensors are bit-identical across the new
checkpoints and the parent. Immutable inspection copies and exact identities are
recorded in `runs/inspection/command_ablation_20261028/checkpoint_audit.json`.

## Main measurements

| Measurement | Control 64,978 | Ablated 64,978 | Fresh ablation 16,000 |
|---|---:|---:|---:|
| First-four expert-command TCP error, VideoUnmask training states | 5.543 mm | 6.743 mm | 11.629 mm |
| First-four TCP response to physically moving the cube +100 mm X, Pick state 80 | 0.148 mm | 0.083 mm | 0.171 mm |
| Same response to +100 mm Y | 0.053 mm | 0.088 mm | 0.140 mm |
| First close command, fresh Pick validation rollout | 112 | 98 | 76 |
| Observed TCP–cube XY distance at first close | 83.085 mm | 119.360 mm | 464.144 mm |
| Commanded TCP–cube XY distance at first close | 79.065 mm | 118.343 mm | 461.800 mm |
| Observed–commanded TCP distance at first close | 4.670 mm | 1.336 mm | 5.953 mm |

Offline errors use the same two real VideoUnmask training episodes, decision
indices 0/32/64/96/128 and four paired flow-noise draws. They are not held-out
accuracy. The original parent's first-four error was 5.620 mm. The cube-relocation
probe uses PickXtimes validation scene 0, rerenders both cameras, and replays the
same recorded robot commands. Robot states and command histories match across
cube locations; altered predictions are not executed. Measurements average the
first four predicted TCP positions over four paired draws in FP32.

The separate 160-command Pick rollouts execute each model's own predictions with
the same reset and noise seed. All three reach the declared diagnostic cap
without lifting the cube. Their large misses are already present in the commanded
targets, rather than being mainly actuator tracking error. A 160-command cap is
not an official task-success evaluation.

Cube input coverage remains intact: at the matched state 80, 123 current cube
points are visible, with two current tokens and 15 tokens in the latest historical
block receiving more than half their pooling mass from diagnostic cube points.
Cube and table features remain distinguishable. Attention moves toward the current
robot-state/gripper token in the ablated continuation, but attention weights alone
do not establish causal reliance or identify a complete failure mechanism.

## Command, history and numerical checks

Rerunning the prior offline protocol produced 3,180 prediction rows. The full seven-joint
PickXtimes command sweep adds 1,376 rows per model. Changing only past joint targets
produces bit-identical full action chunks in both ablated checkpoints, including
current, historical, combined and recent-history scopes. Retained gripper and
availability interventions still change predictions. Thus the intended channels
are actually removed, rather than merely labeled as removed in a configuration.

Removing historical vision changes predictions and increases expert-action error;
the policies are not wholly insensitive to vision. Current-vision deletion has
much larger effects at reset than after execution history exists, so those states
are reported separately. Removing or permuting robot/visual histories remains an
artificial input intervention, not a demonstration that those histories are
semantically useful in closed loop.

Six real-state packed/serial comparisons, three backward-only probes and 36
early-prefix cases pass. The largest FP32 first-four TCP parity differences are
about 0.000060 mm for the real-state probe and 0.000066 mm for early prefixes.
Visual MLP and bandwidth gradients are finite and nonzero. No optimizer update
is performed. Production online versus all-BF16 training comparisons differ by
0.368–1.374 mm in the inspected real-state cases. These same-input, single-decision
measurements do not point to batching, duplicated reset history or numerical
precision as the explanation for the tens-to-hundreds-of-millimetres targeting
errors. They do not rule out numerical differences accumulating in closed loop.

## Expert-prefix handoffs

The protocol uses VideoUnmask training episodes 0/1 and PickXtimes training
episode 0, prefixes 0/4/16/32/64, seed 2701 aligned by absolute decision time,
FULL history and the same 320-total-command cap. VideoUnmask targets the container
hiding a cube. Pick uses the previously generated and verified official-planner
prefix, not a claimed released demonstration. Every post-handoff policy action is
executed; every returned observation enters causal memory. No object is grasped
at handoff in these conditions.

For the matched continuations:

- Control succeeds only on VideoUnmask episode 0 with a 64-command prefix.
- The ablated continuation succeeds with prefixes 32 and 64 in both VideoUnmask
  episodes. These four successes recover the parent's previously measured
  outcomes; they do not surpass them.
- Neither continuation succeeds from reset on any of the three scenes.
- On PickXtimes, the ablated continuation completes the first pick/place cycle
  after prefixes 16/32 and reaches the second placement stage after prefix 64.
  Control fails without grasping after prefixes 16/32; after prefix 64 it grasps
  and reaches only the first placement stage before official failure. Neither
  completes the full PickXtimes task within this protocol.

Official failure, success and reaching the diagnostic cap are separate outcomes.
The five prefixes on one scene are correlated conditions, not independent
benchmark episodes. The two full recorded VideoUnmask expert controls succeed
after 161 and 152 commands. Parent comparisons reuse preserved artifacts whose
checkpoint hash, runner, precision, noise, budget and handoff states were checked
against the new protocol.

The fresh 16k model succeeds after a 64-command prefix in both VideoUnmask
episodes, but not after the shorter prefixes. It grasps the Pick cube after
prefixes 16 and 64 without completing the full task. None of the three new
checkpoints succeeds from reset on the three inspected scenes within the
320-command window. All 45 planned policy trials and both recorded-expert
controls completed their declared protocols; the galleries retain their videos.

## Artifacts and validation

### Initial action chunks for relocated cubes

`reset_cube_chunks/index.html` provides interactive point-cloud views of the first
policy decision after reset for all three checkpoints. Each model's standalone
HTML embeds the exact 2,048-point current observation, front/wrist camera preview,
and all 16 predicted TCP target poses. Five scenes use the original cube position
and translations of ±10 cm along world X/Y. Each position is physically rendered
from both cameras; only one selected scene's observed cloud is displayed at a
time. The cube wireframes are diagnostic annotations, not policy inputs.

Predictions use the production `OnlinePolicy.reset()` / `decide()` path: FP32
encoders and BF16 context/decoder, eight flow steps, with four selectable paired
noise seeds (2701–2704). Previous commands are unavailable and execution history
is empty in all models at this first decision. Checkpoint differences here reflect
learned weights, rather than removal of an available command at inference time.
The displayed target positions and orientations come from FK of the actual
clipped joint targets; connecting lines show target order, not simulated motion
or additional interpolated commands. The first four commands are emphasized
because deployment would replan after those. No predicted command is executed.

The first-four TCP response to a 100 mm cube translation averages 0.43–1.21 mm
across the four paired noise draws, depending on checkpoint and direction. The
first chunk begins near the initial raised robot pose; this reset-only result
does not determine where a complete subsequent rollout will grasp. These
production-precision measurements differ from the preceding FP32/state-80 probe.
The X−10 cm placement lies outside the official sampler's cube-center X interval;
all relocated scenes are explicitly labeled artificial diagnostic configurations.

Robot state, goal, clocks, camera calibration and other object poses match across
placements. The settled cube translations match the requested offsets within
0.1 mm; FK agrees with the observed TCP within 0.01 mm. A repeated baseline gives
bit-identical sampled points and first-seed action chunks. Payloads, full inputs,
camera images and checkpoint identities accompany the HTML in `reset_cube_chunks/`.
Independent CPU validation passes 397 checks over 60 chunks / 960 targets; every
baseline seed-2701 chunk matches the preceding production rollout bit-for-bit.
Five existing kinematics tests and the device/architecture checks also pass.

Reproduce into a new ignored output directory:

```bash
uv run --extra simulator --extra visualization python scripts/visualize_reset_cube_chunks.py \
  --checkpoint control=runs/inspection/command_ablation_20261028/checkpoints/control.pt \
  --checkpoint continuation_no_commands=runs/inspection/command_ablation_20261028/checkpoints/continuation_no_commands.pt \
  --checkpoint fresh_no_commands=runs/inspection/command_ablation_20261028/checkpoints/fresh_no_commands.pt \
  --output runs/inspection/reset_cube_chunks_repeat
```

Rebuild the HTML from saved data without simulation or inference:

```bash
uv run --extra visualization python scripts/visualize_reset_cube_chunks.py \
  --output runs/inspection/command_ablation_20261028/reset_cube_chunks --render-only
```

### Complete inspection artifacts

All generated artifacts are under the ignored directory
`runs/inspection/command_ablation_20261028/`:

- `report/index.html`: comparison plots, quantitative tables and coverage.
- `report/summary.json` and `report/validation.json`: independent CPU metric
  recomputation, checkpoint/protocol identities and completion accounting.
- `comparisons/index.html`: synchronized three-model Pick video, first-close
  frames and trajectory plots.
- `handoff/index.html`: 47 annotated rollout videos, nine synchronized three-model
  comparisons and per-trial results.
- `offline/`: real-data prediction, command, history, precision and gradient checks.
- `visual/`, `grounding/`, `command_pick/`: matched scene probes and rollouts.
- `doctor.json`: actual RTX 5000 Ada FP32/BF16/FP16 compute checks, all passing.
- `invariants.log`: 91 CPU tests passing; the architecture snapshot check passes.

The independent result audit passes all 82 checks with no pending cases or errors.
Handoff validation covers 11,475 issued policy commands, exact expert replay,
causal history counts, matched reset/handoff states and video frame counts. All
100 report links and 124 handoff-gallery links resolve.

Original diagnostic scripts, prior results, source data and checkpoint files remain
unchanged. The wrappers redirect checkpoint/output paths and model labels while
retaining the previous mathematical probes. An initial command-probe provenance
path error occurred before any simulated reset or prediction; its logs are kept,
the helper-file path was corrected, and the same three conditions rerun. All 19
visual modes subsequently completed successfully. No training, Slurm submission,
test-seed tuning or production policy/configuration change was performed.

## Executing all sixteen actions — bounded follow-up

The `execute16_comparison/` follow-up executes all 16 targets before replanning,
compared with the current four-command executor, for each of the three frozen
checkpoints. Six runs use the same PickXtimes validation scene 0 and a cap of
320 primitive commands (16 seconds of active simulator control). Command frequency
remains 20 Hz and physics remains 100 Hz. Prediction horizon remains 16; this
changes the feedback/replanning interval from 0.2 to 0.8 seconds, without changing
the time spacing of the predicted commands.

| Model | First-close XY miss, execute 4 | First-close XY miss, execute 16 | Grasp or lift within cap |
|---|---:|---:|---|
| Control continuation, 64,978 | 83.1 mm | 91.0 mm | Neither arm |
| No-command continuation, 64,978 | 119.4 mm | 87.5 mm | Neither arm |
| Fresh no-command model, 16,000 | 464.1 mm | 174.9 mm | Neither arm |

All six reach the diagnostic cap; none reports official success or failure before
that cap. The altered trajectories and smaller misses for two checkpoints do not
establish improved task success. Results concern one fixed scene and noise stream,
not an independent six-episode benchmark sample.

The noise draws are matched by absolute decision time: at primitive time `t`, both
arms use draw `t // 4 + 1` from seed 2701. Execute-16 skips unused draws rather than
confounding different execution horizons with different noise at shared decision
times. Initial full chunks match the preceding reset probe, paired first-four
commands/observations are identical, and each execute-four run reproduces its
preceding 160-command rollout bit-for-bit. Every actual observation is ingested
once; the current frame remains outside historical leaves. A diagnostic-only
batched CPU transfer was rechecked against the production adapter at nine real
history states with bit-identical contexts, biases and actions. No model weights,
production defaults or training settings changed.

The gallery at `execute16_comparison/index.html` includes six original videos and
three synchronized execute-four/execute-sixteen comparisons. Independent CPU
validation passes 11,924 checks across all 1,920 commands, including FK, causal
history counts, paired noise, earlier-rollout reproduction and video frame counts.
All nine videos contain the reset frame plus 320 post-command frames at 20 FPS.

### Why the first chunk is small

CPU analysis of the available expert trajectories is in `action_cadence_analysis/`.
The PickXtimes example is the previously generated official-planner training-scene
prefix, not a released PickXtimes demonstration: no released Pick HDF5 is available
locally. VideoUnmask examples are the released training episodes 0 and 1.

| Expert source | Initial unchanged arm commands | First-16 target TCP displacement | First-32 displacement | Next-16 displacement after decision 16 |
|---|---:|---:|---:|---:|
| Generated PickXtimes train 0 prefix | 7 | 29.0 mm | 243.3 mm | 222.9 mm |
| Released VideoUnmask train 0 | 12 | 9.7 mm | 222.2 mm | 217.6 mm |
| Released VideoUnmask train 1 | 12 | 7.4 mm | 202.2 mm | 198.7 mm |

These are FK target displacements, not accumulated path lengths or observed robot
displacements. The generated Pick expert's observed displacement after sixteen
commands is about 20.4 mm. The execute-sixteen control policy moves about 18.9 mm
after sixteen and 187 mm after thirty-two commands. The scenes differ, so this is
not an expert-action accuracy comparison. It shows that initial holds and ramp-up
are an important explanation for the quiet reset chunk. General recorded-frame
timestamps must still not be treated as wall time across omitted `NO RECORD`
intervals; the measured live simulation retains the 20 Hz command cadence.

### Possible training experiments — not implemented or run

Increasing the prediction horizon to 32 consecutive targets at 20 Hz would cover
1.6 seconds while preserving the existing action protocol. Retaining execute-four
would preserve a 0.2-second feedback interval. This is the cleaner first test of
whether a longer supervised future helps; it does not assume the visual-targeting
failure is caused by temporal horizon.

Lower-frequency target training is also compatible with absolute joint angles.
For example, sixteen targets at 10 Hz can supervise the original commanded joints
at `t+2, t+4, ..., t+32`. Deployment then needs an explicit matching hold or joint
interpolation scheme at the existing 20 Hz interface. These schemes have different
dynamics and need a recorded-expert replay check, including discrete gripper
transitions. Actual intermediate observations and issued commands must still enter
the causal memory. Sending the subsampled targets at the current one-target-per-
primitive-step pace would instead accelerate the intended trajectory. The current
controller does not automatically interpolate this new coarse trajectory.

`data.decision_stride` only selects which training decision starts to use; changing
it does not change spacing inside the target chunk. A temporal-target experiment
requires explicit changes to target construction and execution semantics.

The distinction between prediction and execution horizons follows the standard
[receding-horizon action-chunk formulation](https://arxiv.org/html/2303.04137v5).
The position-target interpretation is documented by
[ManiSkill's joint-position controller](https://maniskill.readthedocs.io/en/latest/user_guide/concepts/controllers.html)
and verified against the locally pinned controller implementation.
