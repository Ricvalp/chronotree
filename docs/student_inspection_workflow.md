# Local compact-student inspection and routing-label diagnostics

[`scripts/inspect_students.py`](../scripts/inspect_students.py) inspects the
transferred S_STD, S_STAGE and S_PACKET checkpoints with their compact-reader
contexts. It measures whether particular history refinements give repeatable
expert-action improvements, then checks whether those allocations change
ordinary generated action chunks. The existing teacher-only
`inspect_checkpoints.py` and `inspect_teacher.py` are not used for these students.

This is an **exploratory local diagnostic**. It does not train a student/router,
submit Slurm jobs, download data, or replace the sealed 256-state HPC utility
study and its declared 100k endpoints. Latest student snapshots may have
different update counts; the report records this and does not present the
comparison as a matched training-recipe result.

## Run the transferred bundle

The laptop transfer creates this layout on the workstation:

```text
runs/inspection/eval_students_TIMESTAMP/
  SHA256SUMS
  s_std/checkpoint.pt
  s_stage/checkpoint.pt
  s_packet/checkpoint.pt
```

Run on the workstation, substituting the actual bundle name:

```bash
cd /home/rvalperga/chronotree

uv run python scripts/inspect_students.py \
  --bundle runs/inspection/eval_students_TIMESTAMP \
  --detach
```

The default runs **utility measurements and reset cube-position viewers**.
`--detach` prints the coordinator PID and launcher log. Its default four-hour
bound applies to child-stage execution; checkpoint and asset verification
precede that execution bound. The script writes completed state results before
continuing and can resume after a time limit or interruption.

For all four implemented stages, including the button-approach simulations
and expert-prefix handoffs,
choose the complete stage list at the first invocation:

```bash
uv run python scripts/inspect_students.py \
  --bundle runs/inspection/eval_students_TIMESTAMP \
  --stages utility reset rollouts handoff \
  --detach
```

The optional `rollouts` stage uses one fixed VideoRepick validation scene per
student and per reader: six trials for three students, with TREE and FLAT,
B128, eight executed targets and eight flow integration steps. It saves videos
and full predicted chunks around the button approach. This is a diagnostic
subset, not a new all-task success estimate. Infrastructure failures remain
errors and do not become policy failures or successful completion receipts.

The optional `handoff` stage uses two fixed **training** scenes, VideoUnmask episode 0
and PickXtimes episode 0. For each student it evaluates TREE and FLAT after 0, 16 or 32
native expert commands: 12 trials per student. A shared full expert replay on
VideoUnmask episode 0 checks the released demonstration against the simulator, giving
37 trials for all three students. PickXtimes has a short official-planner prefix
rather than a complete released expert recording, so it has no full replay
control. Each trial records its expert source and exact replay checks.

Handoffs retain real observations from the expert prefix; they do not synthesize
idle frames. They use B128, eight executed targets, two native steps per target,
eight flow integration steps, and a cap of 320 native commands in total. Reaching
that diagnostic cap is recorded separately from simulator failure. Flow-noise
and random-reader seeds depend on the task, episode and absolute primitive time,
so models/readers share draws at the same time. Each trial saves a video and
native-command/chunk traces. The controller never receives the privileged
object or grasp labels used in the report. These probes test whether taking
over from an expert approach helps; they do not estimate benchmark success.

The 37 handoffs are deliberately opt-in and may exceed the default four-hour
coordinator bound. Completed trials remain sealed; resume the same command to
continue the remaining trials. A recorded infrastructure error is surfaced for
diagnosis, never silently rerun or counted as a policy failure.

To inspect the plan first, use the same options with `--dry-run` instead of
`--detach`. This validates checkpoints and assets and prints the jobs without
model inference or simulation. `--audit-only` writes the verified inspection
plan without running the stages. A subsequent execution of an audited plan
requires `--resume`.

## Resume, progress and reports

Results default to the bundle path with `_student_inspection` appended:

```bash
cat runs/inspection/eval_students_TIMESTAMP_student_inspection/status.json
tail -n 40 runs/inspection/eval_students_TIMESTAMP_student_inspection/logs/measure_s_std.log
```

Repeat the original command with `--resume` to continue. Preserve all scientific
options, including the original stage list:

```bash
uv run python scripts/inspect_students.py \
  --bundle runs/inspection/eval_students_TIMESTAMP \
  --stages utility reset rollouts handoff \
  --resume --detach
```

Resume verifies checkpoint, source, asset and protocol identities and reuses
only sealed results. Do not replace weights inside an existing bundle. A new
checkpoint, stage selection, budget, draw count, numerical protocol or source
revision requires a new output directory. The coordinator owns a lock and
cleans up child process groups on interruption.

To rebuild the landing report from stored results without running inference or
simulation, repeat the same options with `--resume --report-only`, omitting
`--detach`. This also regenerates the static utility-diagnostic figures from
the stored measurements. Individual generated-action HTML files can be rebuilt from their
NPZ/JSON artifacts through
`stmem.inspection.student_behavior.report_behavior_case(path)`; their array
checksums are verified first.

## What is measured

The default local state panel contains up to eight complete-H16 decisions from
the existing **VideoUnmask episodes 0 and 1**, sampled across source-time
quantiles before reading any policy losses. Both episodes remain represented.
Their membership in the students' optimization pools is not established, so
the report does not call them held out. Geometry, alignment, frozen CLIP
descriptors and cadence must agree with the checkpoint configuration.

All students must have identical frozen encoder and action-normalizer tensors
and the same accepted-teacher dependency. Their shared causal observations are
encoded once locally. The workflow does not load the old 104k teacher's learned
cache or require transferring the full HPC caches. Future observations and
future complete-block encodings never enter the available history.

For every state, both TREE and FLAT use a fixed gain-blind candidate manifest:

- Default final history budget B128; current observation/task/clock tokens are
  outside this budget.
- Up to four coherent eight-move packets, their immediate first moves, and
  equal-budget controls, using the existing exact-coverage constructors.
- An unrestricted single-move audit and capability contexts, including ordinary
  and detailed B128, B256 and all leaves where feasible. Ineligible comparisons
  have explicit reasons rather than silently replaced scenes or dropped roots.
- Eight discovery and eight independent confirmation noise/time draws, paired
  across every candidate, reader and student.
- Primary expert FM loss from the real masked velocity target; signed gains,
  near-ties, winner reversals and per-draw variation remain available.
- Unchanged-context and serial/packed numerical checks distinguish numerical
  differences from sampled-noise variation.
- Source-bank deletion controls measure sensitivity while clearly retaining
  their status as potentially out-of-distribution interventions.

Discovery selects the best first move, best completed packet and low-gain
packet. Independent confirmation measures those choices. The landing report's
positive `packet_vs_random`, `packet_vs_immediate` or `packet_vs_dispersed`
values mean **lower confirmation expert FM loss** for the discovery-selected
packet. These are offline diagnostic selections, not the performance of a
trained router. Random-choice comparison averages the fixed candidate pool;
its samples and repeated draws are not independent episodes.

## Generated actions and the HTML viewers

Two evenly spaced members of the predeclared state list are the common
generated-action cases for all students. With the default eight-state panel,
these include an early and a late decision from different episodes. They are
selected before any gains are measured; only the
within-state alternative choices use discovery loss. Each reader/budget case
starts from two fresh shared Gaussian seeds and runs eight Euler integration
steps with its chosen compact context fixed throughout generation.

Viewers compare the base context, completed immediate-selected packet,
discovery-best and discovery-worst packets, a fixed random packet and available
equal-budget controls. They show:

- The actual current sampled point cloud and observed TCP for orientation.
- Recorded expert and generated joint-target TCP paths, with the first eight
  targets solid and the remaining targets dashed.
- Selected historical record anchors and optional support bounds, with
  source-bank, source-local time interval and represented leaf count on hover.
- Full-chunk, prefix and tail generated-action errors and paired action/path
  differences.

Historical markers are not object masks or attention weights. Paths are
forward kinematics of clipped absolute joint targets, using the task's audited
robot variant. They do not simulate contact or demonstrate task completion.
A different valid action can have a higher expert-action error. Current RGB is
for display; it does not replace the model's frozen visual descriptors.

The `reset` stage separately renders the PickXtimes cube at baseline and three
offset positions with four paired noise seeds. It explicitly verifies empty
history and uses CURRENT for this reset-only equivalence; subsequent student
rollouts use their actual TREE/FLAT readers. No generated reset chunk is
executed. These self-contained point-cloud viewers answer whether the first
chunk responds to cube position.

Utility losses and offline generated-action witnesses use FP32 to measure small
differences consistently. Simulator probes retain the checkpoint's resolved
evaluation precision and record it. An offline loss preference is therefore
evidence to examine, not a promised closed-loop improvement.

## Optional teacher reference and limits

Expert utility does not require a teacher checkpoint. To also measure teacher
agreement, add `--teacher /absolute/path/to/exact_teacher140k.pt` at the first
invocation. Its SHA256 must match the dependency recorded by every student,
and its encoder/normalizer tensors must match. An older teacher is rejected.
Without this file, reports explicitly mark teacher agreement as unmeasured.

All local assets are configured by `configs/inspection.yaml`; the six existing
asset entries remain required at preflight. Simulator stages require working
RoboMME/SAPIEN assets and CUDA. HTML generation uses the optional `visualization`
dependency (Plotly), embedded offline in each viewer. No network fetch is
required to view those pages.

Two local episodes and latest checkpoints cannot establish broad label
quality, student-recipe superiority or benchmark generalization. The useful
screen is whether a discovery preference survives independent noise and
changes a generated action in a plausible direction, with a visible numerical
margin. Positive local findings motivate the full matched study; negative
findings remain local and do not prove that all retrieval is useless.

## Output map

| Path inside `<bundle>_student_inspection/` | Contents |
|---|---|
| `index.html`, `summary.md`, `summary.json` | Cross-student utility results, viewer links and limitations |
| `diagnostics/index.html`, `diagnostics/summary.json` | Repeatability, gain/noise plots, capability and erasure tables, ties and numerical floors |
| `diagnostics/*.png`, `diagnostics/report.md` | Reproducible static figures and a shareable detailed report |
| `student_inspection.json`, `states.json` | Immutable checkpoints/configuration/protocol and gain-blind states |
| `prepared/`, `contexts/` | Shared local causal evidence and candidate manifests |
| `measurements/<student>/<state>.json` | Raw paired losses, numerical controls, gains and capability results |
| `generated/<student>/<state>_<reader>_b128.html` | Interactive history-allocation/action witness |
| `generated/<student>/*.npz`, `*.json` | Generated arrays, selection metadata and checksums |
| `reset_cube_chunks/<student>/` | Four-position reset viewers and numerical results |
| `rollouts/<student>/VideoRepick/<reader>/` | Optional button-approach videos, chunks and outcomes |
| `handoff/<student>/<task>/<reader>/` | Optional expert-prefix videos, replay checks and native-command traces |
| `logs/`, `completed/`, `status.json` | Execution logs, verified completion receipts and progress |

Generated data, checkpoints and reports stay ignored. The diagnostic changes
neither the policy architecture nor its action/context semantics, so no
architecture-diagram regeneration is needed.

## Implementation verification

The final focused CPU suite passed **154 tests**, covering compact-reader
protocols, unchanged historical memory/current observation/clocks, paired noise,
exact candidate coverage, independent confirmation, episode-aware reporting,
complete result identities, held-command timing, transport parity, and resume
behavior. Ruff and architecture snapshot/XML checks passed. The actual RTX
5000 Ada device passed the FP32/BF16/FP16 doctor checks.

An end-to-end one-state utility smoke with the old 90k compact checkpoint passed
with its exact old teacher reference and without a teacher. Both produced TREE
and FLAT action viewers; resuming reused the saved tensors, measurements and
action arrays without rerunning inference. A separate causal-prefix check used
33 conditioning frames and 49 execution-history frames, including one frame in
the pending execution block. Serial/batched loss differences were at most
1.01e-7 in these checks; no-op prediction differences were zero. A real
PickXtimes reset also produced the compact CURRENT viewer. These are bounded
implementation checks, not evidence about the new three students or label
quality across tasks.

A bounded GPU/simulator smoke used the existing **old 90k compact checkpoint**,
PickXtimes training episode 0, TREE/B128, H16, eight executed targets, and eight
flow integration steps. It decoded one chunk, executed exactly one native
command, and saved a video and trace. This verifies the execution path; it is
neither a result for the newly transferred students nor a success-rate test.
The normal handoff protocol remains capped at 320 commands, and was not run as
part of that one-command implementation smoke.
