# Button-hover diagnostic — 4 November 2026

The 83k compact student can predict a descent toward the button in targets 9–16
while keeping targets 1–8 near its current elevated pose. Replanning after eight
targets then discards that descent. This was observed in genuine closed-loop
VideoRepick rollouts, and one saved student chunk completed the task when its
otherwise discarded tail was executed from the exact same simulator state.

This finding explains a specific observed failure mode. It does not establish
that every timeout has this cause or that executing 16 targets globally improves
the benchmark. All scenes below are official **validation** episodes used for
diagnosis, not training or test-seed tuning. No weights were updated.

Open the [focused interactive report](../runs/inspection/button_83k_20261104/focus/index.html)
for the two stalls, teacher predictions and replay videos. It starts at the
first student hover; its decision slider contains selected windows around the
stalls. The [full report](../runs/inspection/button_83k_20261104/index.html)
retains all 20 recorded trajectories, including the explicitly exploratory
UUID-default execution-16 pilot. The
[machine-readable summary](../runs/inspection/button_83k_20261104/summary.json)
records input hashes, outcomes, branch results and matched reset parity.

## Immutable inputs and scope

| Policy | Step | Reader | Checkpoint SHA-256 |
|---|---:|---|---|
| Shared compact student | 83,000 | UNIFORM, historical budget 128 | `b787dfab69e430c1b85ff9dc4bbd9ae1817dc6646ef5a13473ddc2e7e17c5335` |
| Accepted teacher | 104,110 | FULL | `eb57ccd4c131888a5f5d6225a3a0cbe5cd14b3afc5c6ff0a1a17ee319cd70918` |

Local checkpoint paths:

```text
runs/inspection/compact_20261104T114857Z/checkpoint.pt
runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt
```

Both have identical frozen learned visual/text encoder weights, checked by hash
`d0693bdd283d45f8a46b57339ec7e911d6469d7d48f8b9e5a66ac562fca50773`.
They use frozen CLIP RN50 layer-1 features, no proprioception or previous-command
conditioning, 16 predicted absolute joint/gripper targets at 10 Hz, and observed
history at 10 Hz. The initial rollouts execute eight targets per prediction,
each held for two native 20 Hz commands: 0.8 seconds before replanning. The whole
prediction spans 1.6 seconds.

The diagnostic uses production `OnlinePolicy.decide()` and `run_episode()`.
Instrumentation records all predicted targets before the executor truncates
the chunk. It preserves the normal task text, causal observations, precision,
flow integration, and episode noise seed:
`evaluation.seed + original_task_index * 1000 + episode_idx`.
The task list is not shortened before calculating that seed.

Button position, cap depth, simulator task bookkeeping and robot pose are
**annotations only**. They never enter the policy or generate commands. The
observer reads `env.timestep` and the corresponding task name without invoking
task predicates or `evaluate()`, which could advance simulator bookkeeping.

## Original closed-loop observations

Source artifacts:

```text
runs/inspection/button_83k_20261104/student/results.json
runs/inspection/button_83k_20261104/teacher/results.json
```

| VideoRepick validation episode | Student, execute 8 | Native commands | Teacher, execute 8 | Native commands |
|---:|---|---:|---|---:|
| 0 | Timeout | 1,301 | Success | 247 |
| 1 | Timeout | 1,301 | Success | 469 |
| 2 | Timeout | 1,301 | Fail | 96 |
| 3 | Fail | 424 | Timeout | 1,301 |

There were no recorded infrastructure/policy exceptions in these eight
rollouts. These are individual diagnostic outcomes, not a replacement for the
all-task benchmark score. The student reached the final button phase in
episodes 0 and 1. No final-button probe was triggered in episodes 2 or 3; those
failures must not be attributed to the same mechanism from this evidence.

At the student's first hover in episode 0, native command 256, the actual TCP
was 121.5 mm from the button cap. Its production chunk predicted only 3.6 mm
of descent by target 8, but 112.3 mm by target 16. Target 8 remained 118.1 mm
from the cap, while target 16 was 11.1 mm away. The next replan discarded the
later targets. The actual rollout remained elevated and eventually timed out.

Episode 1 showed the same pattern at command 560: the production chunk predicted
0.1 mm of descent by target 8 and 113.5 mm by target 16. The observations support
**postponed descent during replanning**, rather than a policy that never predicts
movement toward the button. Forward kinematics alone does not prove contact.

## Four noise draws on the same observed histories

At entry to the button phase, the first detected hover, and a later hover,
additional predictions use independent noise seeds 2701–2704. These do not
advance the production flow or selection RNGs. Student and teacher receive
the same encoded observations, current observation, goal and clocks. The
teacher reads FULL history and the student reads UNIFORM-128; this comparison
therefore does not isolate the action head from the reader.

The table gives mean target descent across those four independent draws, in mm;
positive means lower TCP target height than the current observed TCP. These are
counterfactual predicted targets, not executed motion.

| Episode / native command | Student target 8 | Student target 16 | Teacher target 8 | Teacher target 16 |
|---|---:|---:|---:|---:|
| 0 / 256, first hover | 4.35 | 110.80 | 122.86 | 107.38 |
| 0 / 416, later hover | −6.54 | 93.62 | 79.53 | 118.30 |
| 1 / 560, first hover | 13.25 | 119.73 | 117.45 | 94.41 |
| 1 / 720, later hover | 2.17 | 105.18 | 112.62 | 111.08 |

The postponement persists across these noise samples. The accepted teacher
generally places more of the descent inside the first eight targets on the
student's own observed history. This argues against an inability of the visual
inputs to support a button-directed prediction in these states. It does not
establish which training or compression change caused the student behavior.

## Verified simulator branches

For each branch, the simulator is reset to the same official scene and all saved
prefix commands are replayed. Before any alternative commands are issued, the
full simulator state, observed robot state and active task index are checked
against the saved snapshot. Every branch below had **zero maximum simulator
state difference and zero proprioception difference** at handoff.

The three interventions are:

- `replan8`: replay the next two recorded eight-target prefixes from the baseline.
- `execute16`: execute all 16 targets from the first saved student prediction.
- `teacher16`: execute the saved teacher prediction for noise seed 2701 from
  this same observed history.

The first two branches have exactly the same initial 16 native commands. They
differ only when the baseline would replace targets 9–16 with a new prediction.
The teacher branch is a separate comparison and does not share that prefix.
Each branch executes at most 32 native commands (1.6 seconds), stopping
immediately on a terminal outcome.

| Episode / command | Branch | Official outcome within saved chunk | Actual descent (mm) | Button-distance reduction (mm) | Maximum recorded cap depth (mm) |
|---|---|---|---:|---:|---:|
| 0 / 256 | replan8 | Ongoing | −9.18 | −9.15 | 0.49 |
| 0 / 256 | execute16 | **Success** | 105.71 | 93.57 | 12.04 |
| 0 / 256 | teacher16 | **Success** | 103.46 | 93.02 | 9.82 |
| 0 / 416 | replan8 | Ongoing | −3.35 | −3.38 | 0.49 |
| 0 / 416 | execute16 | Ongoing | 73.18 | 73.21 | 0.49 |
| 0 / 416 | teacher16 | **Success** | 103.40 | 91.78 | 11.93 |
| 1 / 560 | replan8 | Ongoing | −1.44 | −1.98 | 0.49 |
| 1 / 560 | execute16 | Ongoing | 104.18 | 99.64 | 2.86 |
| 1 / 560 | teacher16 | **Success** | 111.80 | 106.77 | 3.89 |
| 1 / 720 | replan8 | Ongoing | −0.09 | −0.22 | 0.49 |
| 1 / 720 | execute16 | Ongoing | 88.88 | 86.90 | 0.49 |
| 1 / 720 | teacher16 | **Success** | 112.62 | 101.40 | 10.05 |

“Ongoing” means the bounded saved chunk ended without a terminal result; it is
not a full-rollout failure. These deliberately selected states are not an
independent episode sample, so the branch counts are **not success rates**.

The native continuous wrapper can execute an automatic extra terminal step.
Branch endpoint TCP/depth measurements may therefore include that extra step,
and cap depths are sampled observations rather than continuous contact traces.
They must not be used to infer an exact press threshold or controller tracking
error. Success is taken directly from the official simulator status. The HTML
actual-motion timelines exclude terminal observations for this reason.

The exact-state student rescue at episode 0 / command 256 is the strongest
causal evidence: an already predicted, otherwise discarded descent completed
the task without an oracle controller or newly generated action.

## UUID nuisance and the paired execution-horizon control

Production reset currently creates a random transport UUID. Frame event IDs
include it; block and hierarchy node IDs hash those event IDs. UNIFORM refines
greatest leaf mass first, breaking equal-mass ties by node ID. Consequently,
identical observations and flow noise can still produce different UNIFORM
contexts across fresh resets. A preliminary global execute-16 rollout cannot
be treated as a clean paired comparison with the original execute-8 run.
This nuisance does not affect the saved-command branches above.

`--stable-observation-ids` provides a **diagnostic-only** control. It remaps
incoming reset/step event identities to `button-diagnostic/reset` and
`button-diagnostic/step/N`; production ingestion still adds the observation
index and performs deduplication. The identity sequence resets with each
episode, contains no task/episode label, and changes no sensor value, timing,
observation cadence or model parameter. Production evaluation is unchanged.

Paired execute-8 and execute-16 runs on the same four fixed validation scenes
were recorded under:

```text
runs/inspection/button_83k_20261104/matched_execute8/
runs/inspection/button_83k_20261104/matched_execute16/
```

The first point cloud and raw first-chunk predictions were verified bit-identical
between the two execution settings in **all four scenes**, as were their first
16 executed native commands. Later noise draws use the production sequential
generator: after the replanning schedules diverge, they match by decision index,
not by absolute native time. These completed
matched results are:

| VideoRepick validation episode | Execute 8 | Native commands | Execute 16 | Native commands |
|---:|---|---:|---|---:|
| 0 | Timeout | 1,301 | Timeout | 1,301 |
| 1 | Fail | 338 | **Success** | 479 |
| 2 | Fail | 99 | **Success** | 746 |
| 3 | Fail | 119 | Timeout | 1,301 |

Thus two of these four fixed diagnostic scenes succeeded with execute-16,
versus none with execute-8. This small, targeted comparison is not evidence of
an all-task improvement. Executing 16 targets from reset changes earlier
manipulation as well as the button phase. The execute-16 timeouts in episodes
0 and 3 were still in the first-pick phase, not final-button hesitation. In the
original execute-8 episode 2 timeout, the simulator was still waiting for the
third correct-cube pickup. Visually hovering near a button does not by itself
establish that the preceding task has been completed.

An earlier execute-16 pilot under `student_execute16/` also produced two
successes, but used fresh UUIDs. It is retained as exploratory evidence and is
not the matched comparison above. The exact-state saved-chunk rescue and the
global execution-horizon comparison answer different questions; one does not
guarantee the other will succeed.

## Reproduce the diagnostics

Use fresh output directories: the scripts refuse to overwrite existing results.
These are bounded local simulation commands, not training or Slurm submission.

```bash
cd /home/rvalperga/chronotree

uv run python -m stmem.inspection.button_probe \
  --checkpoint runs/inspection/compact_20261104T114857Z/checkpoint.pt \
  --teacher runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt \
  --task VideoRepick --episodes 0 1 2 3 --method UNIFORM \
  --execute-actions 8 \
  --clip-checkpoint cache/backbones/clip_RN50.pt \
  --output runs/inspection/button_hover_recheck/student

uv run python -m stmem.inspection.button_probe \
  --checkpoint runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt \
  --task VideoRepick --episodes 0 1 2 3 --method FULL \
  --execute-actions 8 \
  --clip-checkpoint cache/backbones/clip_RN50.pt \
  --output runs/inspection/button_hover_recheck/teacher
```

Replay the **existing recorded** student snapshots, whose probe indices are
known. A newly generated rollout may have probes at different native steps.

```bash
uv run python -m stmem.inspection.button_branch \
  --trial runs/inspection/button_83k_20261104/student/VideoRepick_00/result.json \
  --config runs/inspection/button_83k_20261104/student/config.json \
  --steps 256 416 --modes replan8 execute16 teacher16 \
  --output runs/inspection/button_hover_recheck/branches_ep0

uv run python -m stmem.inspection.button_branch \
  --trial runs/inspection/button_83k_20261104/student/VideoRepick_01/result.json \
  --config runs/inspection/button_83k_20261104/student/config.json \
  --steps 560 720 --modes replan8 execute16 teacher16 \
  --output runs/inspection/button_hover_recheck/branches_ep1
```

For a new matched global execution-horizon control, run both variants with
stable observation identities:

```bash
for execute in 8 16; do
  uv run python -m stmem.inspection.button_probe \
    --checkpoint runs/inspection/compact_20261104T114857Z/checkpoint.pt \
    --task VideoRepick --episodes 0 1 2 3 --method UNIFORM \
    --execute-actions "$execute" --stable-observation-ids \
    --clip-checkpoint cache/backbones/clip_RN50.pt \
    --output "runs/inspection/button_hover_recheck/matched_execute${execute}"
done
```

Each probe directory contains resolved `config.json`, `results.json`, per-scene
`result.json`, full `decisions.jsonl`, recorded MP4, saved probe snapshots and
`index.html`. The HTML distinguishes target FK from actual motion and makes
additional teacher/noise predictions explicitly counterfactual. Branch videos
show a short identical replay prefix followed by the alternative saved commands.
Display point-cloud coordinates are rounded to 0.01 mm; canonical JSON keeps
the original values. Model targets and reported metrics are unrounded.

## Verification completed

```bash
uv run pytest -q tests/test_button_probe.py tests/test_button_report.py
```

**18 CPU tests passed.** The checks cover withheld-tail geometry, actual
production prefix execution and zero-order holds, immediate terminal stopping,
read-only phase annotations, independent probe RNGs, identical branch prefixes,
simulator-state comparison, stable-ID sensor invariance/deduplication, and
report payload preservation. Ruff also passed for the report and tests.

The actual student report was opened in cached Chromium through Playwright:
point-cloud rendering, default hover selection, teacher/noise selection and
branch-video elements loaded without JavaScript errors. Visual layout was
inspected, including equal spatial scale in the zoomed action view. These
interface checks do not replace the real simulator measurements above.

The expanded verification also included `tests/test_temporal_rollout.py` and
`tests/test_compact_continuation.py`: **36 tests passed**, including the deployed
encoder-cache identity regression. The actual RTX 5000 Ada device doctor passed
FP32, BF16 and FP16. `scripts/generate_teacher_diagram.py --check` passed; model
architecture and production defaults were not changed by these diagnostics.
