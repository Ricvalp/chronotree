# Training and rollout distribution audit — 26 October 2026

The episode-start representation matches between training and closed-loop
rollout. There is no fabricated repeated initial state or fake `t-1` observation.
The main relevant distribution change is expert-generated command/state history
during training versus policy-generated history during rollout. The measured
dependence on historical commanded actions makes error amplification a plausible
explanation for meaningful motion with weak object targeting; it is not yet a
demonstrated sole cause.

This is a bounded diagnostic of the two near-50k teachers (49,978 updates at peak
LR 0.0001 and 49,896 at peak LR 0.001). No training, new closed-loop experiment,
Slurm submission, or production-code/configuration change was made. Scripts and
large measurements are under the ignored directory
`runs/inspection/reset_distribution_audit/`.

## What history contains

Let `o_t` denote the execution observation after primitive command `t`. Reset
already executes one benchmark-internal initial command; its value is not exposed
to the policy. Both paths mark that initial preceding command unavailable, using
zero values and a false availability flag.

| Decision | Current input | Execution-history observations | First target command |
|---:|---|---|---:|
| 0 | `o_0` | Empty | 1 |
| 4 | `o_4` | `o_0, o_1, o_2, o_3` | 5 |
| 8 | `o_8` | `o_0` through `o_7` | 9 |

Thus `t` is supplied separately once and `t-1` is included among the genuinely
observed historical frames when it exists. There is no fixed two-frame stack
that needs filling at reset. Every primitive observation is retained, even
though planning occurs every four commands. Pending blocks contain only their
observed prefix; no future/full-block feature is substituted.

The entire supplied conditioning video, if present, is additionally available
from the first decision in a separate source bank. Its demonstrator commands
remain unavailable during both training and rollout. Both inspected VideoUnmask
episodes contain 66 conditioning frames. Without a supplied conditioning video,
the first decision has no historical observations at all.

The governing paths are `Episode.decision` in `src/stmem/data/canonical.py`,
`EpisodeDataset.load_episode`, `RoboMMEAdapter.reset/ingest`, and
`StreamingMemory.reset/observe`. Repeated event delivery is ignored; equal images
from distinct physical steps remain distinct observations. Batch padding uses
zero-filled values plus invalid frame/point/attention masks. Padded timestamp
storage repeats the final timestamp, but this does not create a valid frame.

## Measured evidence

- **Raw inputs:** transported all 66 conditioning frames and execution frames
  0–12 from each of two real VideoUnmask training episodes through the production
  adapter and compared them with v2 preparation: **158 frames**, bit-identical
  point clouds, valid masks, centers, measured states, physically processed prior
  commands, availability flags, source tags and clocks. Goal text also matches.
  Twenty-four repeated step-event deliveries were correctly ignored. This
  isolates recorded-sensor transport; it is not an independent simulator render.
- **Trained-model early prefixes:** packed training assembly versus actual
  `OnlinePolicy` streaming assembly, at indices 0/4/8/12, both checkpoints and
  both real episodes, with matched noise and FP32 on both paths. All 16 genuine
  cases passed. Maximum mean TCP difference was **0.000063 mm** over the first
  four commands and **0.000126 mm** over all sixteen. Another eight artificial
  no-conditioning cases passed (conditioning removed identically on both paths,
  solely to test empty-history structure). These are not benchmark results.
- **Sampler:** an action/source-only scan of all **100 VideoUnmask training
  episodes** found 3,576 valid stride-four decisions, including all 100 resets
  (**2.796%**) and all 400 first-four decisions (**11.186%**). Reset is not
  excluded. The actual task-balanced sampler visits shuffled decision candidates,
  not uniformly sampled episodes. Other tasks' fractions were not measured.
- **Prior startup accuracy:** on the two inspected expert trajectories, four
  paired noise draws gave mean first-four reset TCP errors of **2.73 mm** for
  LR 0.0001 and **11.01 mm** for LR 0.001. Corresponding non-reset errors were
  6.11 and 7.60 mm. This small training-state diagnostic does not establish a
  general startup weakness or validate unseen reset scenes.
- **Checks:** 17 reset/causal tests passed (one optional debug-data test skipped
  because that preparation is absent); 59 batching/decoder/geometry tests passed;
  34 sampler/startup tests passed. Suites have some overlapping coverage.
  The actual-device doctor passed FP32/BF16/FP16 on the local RTX 5000 Ada.
  Early-prefix output metrics and frame/leaf counts were independently recomputed
  from saved action arrays in `validation.json`.

Prior actual simulator alignment evidence also matches initial qpos, conditioning
qpos and reset camera extrinsics on these episodes; see [data_contract.md](data_contract.md).
That simulator check was not rerun here. Neither bounded local evidence nor source
inspection certifies all HPC files or all validation scenes.

## Other shifts and discrepancies

**Expert versus generated trajectories is the strongest current hypothesis.**
Training commands lead toward successful grasps. An incorrect generated approach
is instead fed back during rollout alongside its resulting state. A history in
which the robot performs a lifting motion but the cube remains on the table can
be absent from successful demonstrations. Accurate prediction on expert histories
does not establish recovery from this situation. Measured robot-state history
also contains motion cues, so removing commands cannot eliminate every shortcut.
The absolute execution clock could additionally correlate with expert progress;
whether the network uses it to advance through unsuccessful motions is untested.

**Cadence and ordinary preprocessing match.** Both configurations sample decision
indices at stride four, predict sixteen commands, and execute four. Both cameras,
calibration, point/center sampling, action normalization, text encoding and
current/history separation use shared or checked paths. There is no BatchNorm or
nonzero model dropout mismatch. The 313 valid expert commands in the two local
VideoUnmask episodes already satisfy rollout clipping/gripper conversion exactly.
The controller still operates open-loop for four commands before replanning;
that can amplify prediction error without constituting an indexing bug.

**Numerical precision differs.** Training uses BF16 autocast through the teacher,
whereas online visual/event/text encoders run FP32 and adapters/decoder use BF16.
The earlier real-checkpoint probe measured about 0.40–1.76 mm first-four TCP
differences, far smaller than the inspected approximately 105 mm grasp miss.
This does not prove rounding never matters near a contact threshold. The new
FP32 parity test deliberately separates structural parity from this difference.

**There is a separate action-decoder input mismatch.** `flow_sample` zeros noisy
action inputs wherever target validity is false, whereas `ActionFlow.integrate`
generates and updates every action slot. The decoder mixes action dimensions and
attends across horizon positions, so those slots can influence supervised joints.

- PatternLock/RouteStick have an unused gripper target throughout training, but
  inference generates that latent channel before discarding its final value.
- Near-terminal partial chunks have zeroed unavailable tails during training,
  but generated tails during inference. Seven of 79 locally prepared decisions
  have partial horizons.

The mismatch and computational coupling were verified with an explicitly
synthetic CPU probe; its performance impact on trained policies was not measured.
It does not explain an early PickXtimes approach where every action dimension and
the full target horizon are valid. Public task-level unused dimensions can be
handled consistently at inference; future terminal validity must never be passed
as an oracle mask. No decoder correction is implemented by this audit.

**Inference is not the flow-training objective.** Eight Euler integration steps
and freshly sampled noise at replanning introduce solver approximation and
sampling variability. Lower flow loss need not imply proportionate improvement
in sampled chunks or closed-loop success. Solver-step and execution-horizon
experiments remain separate, unrun controls.

## Recommended experiment

The first learning experiment should isolate commanded-action conditioning rather
than add repeated reset frames or bundle a sampling change into it.

1. Branch two fresh runs from the **same lower-LR near-50k checkpoint**. Match
   initial weights, optimizer/scheduler treatment, data draws, random seeds,
   effective batch, precision and fixed update budget (for example 5k updates
   as an initial screen). One branch continues normally.
2. In the treatment, make previous commanded-action inputs unavailable in
   **both current input and every historical event token, during both training
   and evaluation**. Use the zero-plus-unavailable convention. Keep measured
   joint positions, TCP pose, finger opening, all visual history, conditioning,
   clocks, token counts, leaf coverage, targets/masks and normalization intact.
   This masks all eight command channels; it is a declared input ablation, not
   deletion of robot-event observations. Record the setting in checkpoint identity.
3. Keep startup sampling, architecture, LR and execution horizon unchanged.
   Compare on the same predeclared official validation scenes and noise seeds,
   including PickXtimes. Measure actual grasp/lift and task success, first-close
   miss distance, and whether a rerendered cube relocation produces a correction
   in the appropriate direction. Simulator object geometry is diagnostic-only.
   Flow loss and sensitivity alone do not determine the winner.

The continued control is necessary: improvement over the untouched checkpoint
could otherwise be ordinary additional training. A negative short adaptation
does not rule out a benefit from learning without this shortcut from scratch.
All-command masking also removes gripper targets while retaining observed finger
opening; a later per-channel ablation could separate that effect if necessary.

A cheaper complementary diagnostic is expert-prefix handoff: on the same
**training scenes with recorded experts**, compare policy takeover after zero,
four, sixteen and thirty-two real expert steps. Feed every real observation and
issued command through the normal adapter. Improvement after handoff would show
benefit from a successful prefix, but would not uniquely identify a reset defect:
progress, states and information all change. Track whether targeting error grows
again after takeover. Released validation expert actions are not assumed to
exist; this is not an official validation success measurement. Local data support
VideoUnmask; PickXtimes expert diagnostics would use the HPC training data.

If this specifically implicates early decisions, test startup oversampling as a
separate factor. Current evidence supports neither inventing duplicate history
nor masking commands only at evaluation as a repair.

## Reproducible local artifacts

Under `runs/inspection/reset_distribution_audit/`:

- `raw_reset_input_parity.py` / `.json`: actual recorded sensor/input comparisons.
- `early_prefix_parity.py` / `.json`: trained-model packed/streaming checks.
- `count_training_starts.py`, `sampling_full_videounmask_counts.json`: whole-task counts.
- `action_distribution_audit.py`, `action_distribution.json`: expert command handling.
- `invalid_action_slot_probe.py` / `.json`: explicitly synthetic decoder-input probe.
- `reset_history_review.md`, `input_shift_review.md`, `sampling_experiment_review.md`:
  detailed source references, checks and limitations.
- `doctor.json`, `validation.json`: device and result-consistency evidence.

Run the diagnostic scripts with `uv run python PATH`; the early-prefix script
requires the existing local near-50k checkpoint snapshots and a working GPU.
Generated data and weights remain ignored. No proposed learning or handoff
experiment has been launched.

Subsequent authorized work ran the simulation handoff experiment; see
[expert_prefix_handoff.md](expert_prefix_handoff.md). It also established that the
two inspected successful VideoUnmask demonstrations contain failed grasp attempts
and retries. Absence of an on-policy data-aggregation pipeline must not be
interpreted as absence of every recovery example in the released demonstrations.
