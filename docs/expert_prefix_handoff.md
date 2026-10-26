# Expert-prefix handoff in simulation — 26 October 2026

Giving the policy a real expert approach prefix substantially improves these
closed-loop training-scene diagnostics. Both near-50k checkpoints succeed on
both inspected VideoUnmask scenes after 32 expert commands, while all zero-,
four- and sixteen-command arms fail to grasp the target within the declared
320-command window. PickXtimes also benefits, but neither checkpoint completes
the full repeated pick/place/button task within that window.

These are **three fixed training configurations, two checkpoints, and one paired
action-noise seed**, not an official validation/test score or independent samples
of 30 environments. All reported policy actions after handoff were actually
executed in the simulator, and all resulting observations entered causal memory.
There were no optimizer updates or changes to production policy code/configs.

## What was run

- VideoUnmask train episodes 0 and 1: exact official seeds 6000/6100, easy. Replay
  the released recorded expert commands from the locally prepared demonstrations.
  Both supply 66 real conditioning frames. The target is a **container hiding a
  cube**, not the cube itself.
- PickXtimes train episode 0: exact official seed 1000, easy. Generate a prefix
  using the pinned benchmark's `solve_pickup` and Panda planner, then replay it
  in a fresh identical simulator before handoff. This is a regenerated expert
  trajectory, **not** a claim of identity with the released HDF5 commands. No raw
  PickXtimes download was needed, no seeds were retried, and no failure-recovery
  behavior was injected into generation.
- Checkpoints: 49,978 updates at peak LR 0.0001 and 49,896 at peak LR 0.001,
  immutable copies under `runs/inspection/analysis_v2_near50k/`. SHA256 values:
  `0217288395612d1973e868297349995d09aaab19a6caf5473702830a8528878b`
  and `0402d45f0b9def057bc3c1c1385d20d8c27550dd61619a5e7446073d9e871828`.
- Primary prefixes: **0, 4, 16, 32 primitive commands**. Added a separately
  declared **64-command secondary condition** after inspecting expert stages:
  0/4 are stationary, 16 is early motion, 32 is approach, and 64 is closer to the
  object but still before expert grasp. No object was grasped at any handoff.
- 24 primary plus six secondary policy rollouts, and two full recorded-expert
  controls. Each policy trial stops on an official terminal result or a common
  **320 total-command diagnostic cap**, including the prefix. A reached cap is
  not relabeled as official task failure or official simulator timeout.
- Production inference convention: FP32 encoders, BF16 adapter/decoder,
  16-command prediction, eight Euler integration steps, four commands executed
  before replanning. FULL memory retains all conditioning/execution evidence.

## Results

| Scene/model | Prefix 0 | Prefix 4 | Prefix 16 | Prefix 32 | Prefix 64, secondary |
|---|---|---|---|---|---|
| VideoUnmask, both scenes / LR 0.0001 | No target grasp | No target grasp | No target grasp | Both succeed | Both succeed |
| VideoUnmask, both scenes / LR 0.001 | No target grasp | No target grasp | No target grasp | Both succeed | Both succeed |
| PickXtimes / LR 0.0001 | No cube grasp | No cube grasp | First pick/place completed; second pickup not completed | Same | Same |
| PickXtimes / LR 0.001 | No cube grasp | No cube grasp | No cube grasp | Cube picked/lifted; placement not completed | Same |

Every non-success cell describes the state at the fixed diagnostic cap. None is
a claim that continued execution could never succeed.

For VideoUnmask, the 32-command handoffs reach official success at total commands
99–105, i.e. **67–73 autonomous policy commands after handoff**. The expert has not
already picked up the container: at handoff 32 the TCP is still 112 mm / 79 mm
away horizontally from the respective target centers, and about 356 / 362 mm
above the world origin. The 64-command handoffs succeed after another 29–33
policy commands. The pure expert controls succeed after 161 / 152 commands.

For the lower-LR PickXtimes model, the 16/32/64 arms establish contact grasp and
lift the cube to approximately 145–147 mm world height, then pass the official
first placement stage. At command 320 the active stage remains the second
pickup. The higher-LR model's 32/64 arms lift the cube to approximately 155–157
mm, but remain at the first placement stage. The target cube starts at about
20 mm height. No PickXtimes trial completes the required full sequence.

The first-close horizontal distance is a secondary descriptive measurement,
not a grasp-success metric. In VideoUnmask it is distance to the container actor
center, not a verified optimal grasp point. Contact and official task outcome
provide the stronger evidence.

## Interpretation and an important correction

This shows that the trained policy can successfully finish an approach and grasp
after an appropriate expert prefix. A generic inability to execute grasping
motions is therefore insufficient to explain the observed failures. It also
shows that simply appending four genuine stationary observations/commands does
not solve the problem.

The result is consistent with dependence on a useful motion trajectory: an
expert's recent commands and measured states establish the approach, and the
policy can continue it. It does **not** isolate the commanded-action channel,
because handoff changes measured robot state, time, visual observations and
history together. It does not establish a reset-padding bug or prove that the
policy responds appropriately to object relocation. The prior isolated command
and cube-position sensitivity tests remain necessary evidence for that question.

There is a correction to any blanket description that these data lack recovery
behavior: **both inspected recorded VideoUnmask demonstrations themselves contain
an unsuccessful first grasp and a later successful retry**. Episode 0 first
closes at command 73 without grasping, opens again, and establishes contact grasp
at 142; episode 1 first closes at 68 and establishes grasp at 132 after retrying.
Successful demonstrations can contain intermediate failures. They still do not
provide corrective coverage of arbitrary states visited by our policy.

The next controlled learning test remains normal continuation versus removing
commanded-action inputs consistently from both current and historical event
inputs during training and evaluation, with measured state/visual history intact.
This handoff result strengthens the motivation to test trajectory dependence;
it does not make that treatment a guaranteed remedy.

## Protocol and checks

Reset's automatic initial command is excluded from the replay. Prefix length k
means recorded/regenerated commands 1 through k have been issued, each through
the normal adapter. At handoff, current is observation k, execution history is
0 through k−1, and the complete conditioning bank is retained. No stored point
cloud is substituted into a simulator rollout. Expert joint states are checked
against every replayed nonterminal prefix step with a fixed 1e−4 rad maximum
absolute tolerance; resets are checked too.

Noise seed 2701 is paired by **absolute decision index**. The generator advances
k/4 draws during an expert prefix, without computing or executing policy actions,
so every arm uses the same noise at the same absolute replan time. One noise seed
is a deliberate bounded screen, not robustness to stochastic sampling.

Generated Pick prefixes were independently replayed before use. Reset plus all
32/64 steps match generation exactly in measured proprioception, target cube
position/quaternion and both camera extrinsics. The first 32 commands/states of
the 64-step generation match the original shorter prefix bit-for-bit.

Object poses, grasp predicates and existing task-progress fields are used only
for metrics and captions; they never enter the policy. Metric collection does
not call state-changing task evaluation functions. Empty terminal observations
are flagged and excluded from position aggregates.

The diagnostic initially encountered expensive individual CPU-to-GPU transfers
for archived records. A script-local equivalent transfer batches the same FP32
content/metadata tensors before the unchanged adapter operations. Six real
checkpoint/prefix comparisons produce bit-identical contexts, attention biases
and predicted chunks. An additional actual-rollout comparison confirms all 196
common commands/states and 49 predicted chunks are bit-identical. No production
optimization was applied. Concurrent unrelated GPU work and this diagnostic
transport change mean timings are not a performance benchmark.

The interrupted serial-transfer attempt and initial diagnostic precision-probe
setup error are retained separately under `attempts/` / diagnostic logs, excluded
from scientific trial counts. All final primary and secondary trials completed
their declared termination/cap protocol. No failed scene was replaced with another
seed. The initial precision setup was corrected by resolving automatic precision
outside the no-gradient rollout, so the device doctor's backward test could run.

## Videos, data and reproduction

Open `runs/inspection/expert_prefix_handoff/index.html` for the local gallery.
It groups every recorded rollout by task, scene, checkpoint and prefix length,
with jump-to-handoff controls and corresponding trajectory plots.

Four-way comparison videos under `comparisons/` show the lower-LR checkpoint
with prefixes 0/16/32/64 at matching absolute execution times. Shorter successful
videos freeze on their final success frame, explicitly labelled. Each clip shows
front and wrist cameras; yellow indicates conditioning/expert control, teal the
policy. There are 32 individual rollout/control videos plus generated-prefix
previews and the comparison videos.

Detailed JSONs live at `TASK/episode_N/VARIANT/prefix_K/result.json`, with separate
`expert/expert_control/` folders. `summary.json` / `.csv`, plots, method/source
reviews and validation reports are beside the gallery. Large generated artifacts
remain ignored by git.

From the repository root, with existing local data, weights and simulator assets:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
uv run python runs/inspection/expert_prefix_handoff/generated_pick/generate_prefix.py
uv run python runs/inspection/expert_prefix_handoff/generated_pick/generate_prefix.py \
  --steps 64 --output runs/inspection/expert_prefix_handoff/generated_pick64

for variant in lr_0001 lr_001; do
  uv run python runs/inspection/expert_prefix_handoff/run_handoff.py \
    --task VideoUnmask --variant "$variant" --expert-controls
  uv run python runs/inspection/expert_prefix_handoff/run_handoff.py \
    --task PickXtimes --episodes 0 --variant "$variant"
  uv run python runs/inspection/expert_prefix_handoff/run_later_handoff.py --variant "$variant"
done

uv run python runs/inspection/expert_prefix_handoff/validate_results.py \
  --expected-policy-trials 30 --expected-expert-controls 2
uv run python runs/inspection/expert_prefix_handoff/summarize_handoff.py --comparison-videos
```

The runner reuses completed results in the same output directories; use a fresh
study directory for an independent repetition. This study did not run new
training or submit HPC jobs.
