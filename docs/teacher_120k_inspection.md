# Teacher continuation inspection — 10 November 2026

The transferred frozen-vision/no-proprioception teacher is at **120,000 updates**,
15,890 beyond the previously inspected 104,110-step checkpoint. Its SHA256 is
`f5cd82b7b35f807af31a1078a9634b941667e936ad68e0050c722cd2a8171ecf`.
The transfer checksum, ZIP CRC, finite tensors, strict model/optimizer loading,
optimizer/scheduler/sampler steps, per-rank RNG records and constant `1e-5` LR
checks pass. Saved scientific configurations and training provenance match.

This is a bounded diagnostic comparison, not a new all-task success rate.
Official test seeds are untouched. No training or production policy changes are
part of this inspection.

## Completed usual inspection

The existing reusable launcher completed its offline, grounding and handoff
stages. It retains the established four-native-command replanning interval:
two predicted 10 Hz targets, each held for two native 20 Hz simulator commands.
This differs from the execute-ten history study described below.

| Measurement | 104,110 updates | 120,000 updates |
|---|---:|---:|
| Expert-state TCP error over first 0.2 s | 5.404 mm | 5.607 mm |
| Expert-state TCP error over first 0.8 s | 7.645 mm | 7.364 mm |
| Same-state 0.8 s error with conditioning history erased | 20.871 mm | 20.293 mm |
| Same-state 0.8 s error with execution history erased | 61.473 mm | 62.858 mm |
| Fresh PickXtimes first-close XY miss | 4.278 mm | 11.499 mm |
| Fresh PickXtimes maximum cube rise | 134.985 mm | 133.518 mm |
| VideoUnmask expert-prefix handoff successes | 10/10 | 9/10 |
| PickXtimes expert-prefix trials with grasp/lift | 5/5 | 5/5 |

Expert-state entries use the same ten VideoUnmask training states and four
paired noise seeds per state, with FP32 visual/text encoding and BF16
context/decoder computation. They are training-state prediction diagnostics;
the four noise draws are not independent task trials. Native 16 commands span
0.8 seconds, not the entire 16-target, 1.6-second predicted chunk.

The fresh PickXtimes scene still has a sustained grasp/lift, confirmed in actual
video frames, but first closure is less accurately centered. Moving the cube
100 mm at the fixed later state still changes imminent predictions by only
about 1 mm; later chunk targets can respond more strongly. Forward-kinematics
paths in the HTML viewers are predictions, not executed trajectories.

The new VideoUnmask failure is training episode 1 after four expert commands.
It lifts a wrong container: the target remains at height 0.05331 m while another
container reaches 0.16599 m. The old checkpoint succeeds in that same prefix
case. This is a paired selection regression, not an infrastructure failure.
All five PickXtimes handoffs still stop at the 320-command diagnostic cap
without full task success, despite meaningful grasp/lift/subtask progress.

The usual findings are mixed and do not establish broad improvement. Erasing
history still markedly worsens expert-action fit, much as it did at 104k.

## History comparison protocol

The paired study uses validation episodes 0–3 on SwingXtimes, VideoRepick and
VideoPlaceOrder, with FULL, erased conditioning, erased execution and CURRENT
reads. The same teacher architecture, observations, goal, clocks, eight Euler
steps and noise-seed protocol are retained. Each chunk predicts 16 targets at
10 Hz and executes ten before replanning. A separate Swing follow-up retains
two completed eight-frame execution blocks plus the pending prefix.

The main study contains 48 planned rollouts, plus four fresh recent-window
rollouts. Four FULL reference rows in the follow-up are copied after identity
checks and are not additional trials. An early two-worker launch was deliberately
interrupted to increase parallelism; its four completed rows are excluded from
the final comparison and match their four-worker counterparts bit-for-bit in
actions and reset fingerprints.

Closed-loop history interventions can change later robot states. Fixed-current
shadow probes separately preserve observation and noise within each checkpoint.
Across checkpoints, only reset shadow probes share the same physical state;
later pooled sensitivity magnitudes must not be presented as a common-state
effect of training. The usual offline probes above supply common expert states.

The previous permutation follow-up is not repeated as an exactly matched
comparison: reset UUIDs affect same-time node ordering, so equal permutation
seeds would not guarantee the same intervention. No production tie-breaking
convention was changed for this evaluation.

## What the completed history comparisons show

All 48 primary and four fresh recent-window trials are complete, with no policy
errors or wall-clock cutoffs. Every timeout reaches the normal 1,301-native-command
wrapper limit. The primary success results are:

| Task | FULL at 104,110 | FULL at 120,000 | Without initial video at 104,110 | Without initial video at 120,000 |
|---|---:|---:|---:|---:|
| SwingXtimes | 4/4 | 4/4 | 4/4 | 4/4 |
| VideoRepick | 1/4 | 1/4 | 0/4 | 0/4 |
| VideoPlaceOrder | 0/4 | 0/4 | 3/4 | 3/4 |

SwingXtimes has no initial conditioning video. Its corresponding erasure arm
is a placebo and matches FULL action-for-action. Removing execution history
or all history produces 0/4 on each of the three tasks at both checkpoints.
These are interventions on the same teacher; CURRENT is not a separately
trained policy. Current visual observations, instruction and clock inputs
remain available, while the selected historical records are removed.

**SwingXtimes:** the new teacher still needs history to finish these scenes,
but shows no additional success benefit over the old teacher. Retaining only
two completed eight-frame blocks plus the pending prefix also gives 0/4 at
both checkpoints. The recent-window and FULL runs match for the first 60
native commands, before the first history removal. At 120k, the shortened
history leads to early grasp/lift or release failures; no trial completes a
full right-left swing pair. One trial releases the cube onto the first right
waypoint rather than maintaining a grasp through the pair. This
is evidence for useful dependence on older observations, not a demonstration
of learned counting independently of movement continuity.

**VideoRepick:** the successful scene changes from episode 1 to episode 2;
there is one gained and one lost success. The new teacher picks the correct
cube and completes all three pickup/drop cycles in episode 2. Removing the
initial video in that same scene still permits picking the correct cube but
produces zero completed cycles and a premature move toward the finish button.
That supports a role for the demonstration in sequencing, without establishing
that it caused correct cube identification. Conversely, episode 1 now completes
only one of three cycles before attempting to finish; the old teacher completed
all three. Episode 3 completes only two cycles with both checkpoints; the new
teacher approaches and descends onto the button area before a terminal failure,
whereas the old teacher times out. The exact simulator failure flag was not
recorded, so this is a behavioral interpretation supported by frames and
subgoal progress, not a logged failure-reason diagnosis.

**VideoPlaceOrder:** FULL fails all four scenes, and deleting the initial
video succeeds in three, at both checkpoints. The new teacher correctly
grasps and lifts the cube but approaches a wrong destination in every scene.
For episodes 0–2, the wrong destination is unchanged from 104k. In episode 3,
the old cube finishes between the intended pad and a neighboring pad, outside
the success tolerance; the new cube goes decisively to a different wrong pad.
This remains evidence against reliable use of the initial ordering cue.
With execution history or all history removed, none of the four new
VideoPlaceOrder scenes reaches a grasp/lift of the requested cube. Those
failures therefore cannot be interpreted solely as mistakes in recalling
the destination order.

| VideoPlaceOrder episode | Intended pad | New approached pad | Final distance to approached / intended pad |
|---|---:|---:|---:|
| 0 | 0 | 1 | 13.4 / 267.7 mm |
| 1 | 3 | 0 | 14.2 / 260.4 mm |
| 2 | 0 | 1 | 12.2 / 191.0 mm |
| 3 | 1 | 0 | 11.3 / 338.1 mm |

Distances use saved actual simulated cube positions, not predicted
forward-kinematics targets. Privileged target and subgoal labels are read-only
diagnostic observations; they were never supplied to either policy.

At identical reset states, removing the initial video changes the predicted
first-ten-target TCP path by a mean 12.375 → 10.327 mm in VideoRepick and
26.945 → 23.721 mm in VideoPlaceOrder (104k → 120k). Each entry averages
Euclidean target displacement over ten targets and then over four matched
scenes, with one shared initial flow-noise tensor per scene. These are predicted
FK positions, not realized robot displacement or a repeated-noise estimate.
Swing reset sensitivity is correctly zero because no historical observations
are available yet. Greater or smaller prediction sensitivity alone does not
measure whether the policy uses the correct historical cue.

These fixed four-scene results do not establish that continued training cannot
help. They show no improvement in the measured history-related success benefit
after these additional 15,890 updates. History erasure/windowing is itself an
out-of-distribution intervention, and useful dependence on history is weaker
evidence than reliable semantic recall. There is no basis here to replace the
immutable 104k teacher or regenerate its caches; broader matched validation
would be needed to justify that decision.

## Artifacts and verification

- [Matched 104k/120k comparison and paired videos](../runs/inspection/eval_teacher_20261110T152129Z_analysis_planning/history_comparison.html)
- [New primary history-intervention gallery](../runs/inspection/history_use_teacher120k_20261110_4workers/index.html)
- [Usual inspection report and videos](../runs/inspection/eval_teacher_20261110T152129Z_analysis/index.html)
- [Four-position cube point clouds](../runs/inspection/eval_teacher_20261110T152129Z_analysis/reset_cube_chunks/index.html)
- [Usual comparison data and compatibility audit](../runs/inspection/eval_teacher_20261110T152129Z_analysis_planning/usual_comparison.json)
- [Recent-history gallery](../runs/inspection/history_recent_swing_teacher120k_20261110/index.html)
- [History audit and machine-readable comparisons](../runs/inspection/eval_teacher_20261110T152129Z_analysis_planning/history_comparison.json)
- [Per-scene behavior and inspected frame evidence](../runs/inspection/eval_teacher_20261110T152129Z_analysis_planning/history_behavior_comparison.json)

The primary four-worker process session received an external SIGTERM after
32 results had been published. All 32 results and the original manifest/config
were preserved byte-for-byte. The 16 missing cases were recovered with the
unchanged worker implementation, original seeds/settings and two parallel
workers, with separate recovery receipts. All recovered workers exited zero.
The interruption is infrastructure history, not a failed task trial. The
earlier four duplicated trials from the deliberate resource restart are not
included in any denominator.

The recovered primary results are recorded in W&B run `3efbb871a2432c20`; the
recent-history follow-up is `0669a6483a398f4b`, both in `ricvalp/chronotree-routers`.

The usual offline validation passed 17,423 checks; grounding passed 1,660.
Handoff validation covers 15 policy trials and two expert controls, 2,332 policy
commands, 661 expert commands, 14 paired-state and 506 paired-RNG checks.
All 102 usual-report links resolve. Both checkpoint hashes and all six local
asset groups were independently rechecked. The actual RTX 5000 Ada passed the
compute doctor in FP32, BF16 and FP16; rendering was exercised by the rollouts.

The final paired-history audit passed **4,363/4,363 checks**. It validated
matched reset fingerprints, configurations, seeds, actually issued held actions,
same-current replay, independently recomputed TCP sensitivities, intervention
source counts, recovery preservation and normal terminal accounting. All 112
referenced old/new videos fully decoded (133,237 frames), with expected frame
counts/durations; the references include eight copied FULL videos, so this is
not 112 independent trials. All 141 comparison-viewer links resolve. The behavior
review covers 32 paired VideoRepick/VideoPlaceOrder cases and the Swing follow-up,
with selected actual frames inspected to support the interpretations above.

Overall source hashes differ from the older inspection, but the teacher forward
and usual diagnostic implementations were reviewed as unchanged. Protocol,
asset hashes and recorded runtime identities match. Added router-only paths
and checkpoint/resume infrastructure do not alter these teacher computations.

Reproduce the usual inspection into its existing matching output with:

```bash
uv run python scripts/inspect_checkpoints.py \
  --bundle runs/inspection/eval_teacher_20261110T152129Z \
  --compare-to runs/inspection/eval_20261102T120621Z_analysis \
  --resume
```

Reproduce the saved-result comparison and its full media audit with:

```bash
python3 runs/inspection/eval_teacher_20261110T152129Z_analysis_planning/compare_history.py \
  --new-recent runs/inspection/history_recent_swing_teacher120k_20261110 \
  --require-complete --validate-media
```

The run-specific comparison helpers, JSON records and media are diagnostic
artifacts under ignored `runs/inspection/`; the scientific findings are retained
in this note. No production implementation or architecture changed.
