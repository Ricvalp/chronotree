# Expert-count history diagnostic

This bounded experiment tests the hypothesis that button hovering reflects an
unreliable transition from repeating an action to finishing the task. It changes
the required count in normal task language and the completed repetitions in
expert visual history. It does not train, select test seeds, or report benchmark
success. No production architecture or diagram settings change.

## Task and input contract

PickXtimes is used instead of BinFill. BinFill requests per-color cube counts in
a bin, leaving a different visible scene as cubes disappear. In SwingXtimes,
one repetition is a complete right/left cycle and putting down the cube precedes
the button. PickXtimes returns the same cube to the same target after every
pick/place cycle, making its cycle boundary a cleaner comparison.

All three tasks state their target counts in task language; they do not use a
reset conditioning video for the count. The probe uses ordinary official
PickXtimes wording for X=1,2,3. After k completed cycles, k=X means the next expert
phase is button pressing, while k<X means another pickup. Cells k>X are excluded:
they have already violated the exact-count task and have no ordinary correct
expert continuation.

The fixed scene is training episode 1, seed 1101, easy, naturally X=3. It was
chosen from metadata as the first easy three-cycle scene, without evaluating
policy outcomes. The official planner generates three complete cycles and a
successful final button control. Privileged object/phase information is used
only by that expert and diagnostic annotations. The model receives canonical
observations, its usual metadata, and normal task text. Future reference frames,
expert commands, phase labels and counts are excluded from policy inputs.

## Comparisons

The protocol has three histories for each valid (k,X) pair, with four paired
noise seeds (2701–2704): 72 predicted chunks per model.

1. **Natural:** genuine expert current observation and causal prefix at each
   cycle boundary. Changing X at fixed k changes only the language input.
2. **Fixed current:** use exactly the same current observation after cycle 1
   for all k. Right-align each genuine expert prefix to the same current clock
   without changing its frame spacing. History lengths still differ. This is a
   counterfactual input, with a small possible physical discontinuity.
3. **Fixed duration:** also equalize history length and timestamps by prepending
   repeated reset observations to shorter prefixes. These observations model an
   artificial idle interval, not a newly simulated trajectory. Changes under
   this control can reflect timing, repetition, content/mass or unusual inputs;
   they do not by themselves prove a counting shortcut. These are literal
   repeated initial frames with new timestamps, not masked padding or simulator
   waiting: 136 added frames (13.6 seconds) for k=1, 68 (6.8 seconds) for k=2,
   and none for k=3.

The teacher uses FULL leaves. The student uses UNIFORM with a 128-token budget,
matching its interim evaluation. Their learned frozen encoder and text-encoder
weights must match exactly. Each selected history is checked for exact leaf
coverage. Encoding runs in FP32 and adaptation/decoding uses the checkpoint's
production precision. Chunks have 16 targets at 10 Hz; the first eight targets
are highlighted as the executed prefix from the accepted HPC evaluation.

HTML curves show forward kinematics of target joint angles, not executed
trajectories. Button-directed progress is a geometric quantity, not a press
probability. A complete press cannot be inferred from these curves. Each case
also verifies exact repeatability with identical noise and context.

## Commands

The local accepted teacher is already present. To regenerate the expert source
in a **new** output directory:

```bash
uv run python -m stmem.inspection.count_expert \
  --checkpoint runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt \
  --output runs/inspection/count_probe_expert_new \
  --clip-checkpoint cache/backbones/clip_RN50.pt
```

The verified source from this implementation is already saved at
`runs/inspection/count_probe_expert_train1_v2/source.pt`; no regeneration is
necessary. To compare teacher and the transferred 15k student:

```bash
uv run python -m stmem.inspection.count_probe \
  --source runs/inspection/count_probe_expert_train1_v2/source.pt \
  --teacher runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt \
  --student runs/inspection/count_probe/student_15k.pt \
   --output runs/inspection/count_probe_teacher_student_new
```

Omit `--student` for a teacher-only run. The output must be new or empty; existing
results are never overwritten. Outputs are `results.json` and a standalone
interactive `index.html`. The linked expert video remains in the source directory.
The checkpoint audit verifies ZIP integrity, finite tensors, configuration hashes,
and strict model loading; the source teacher hash is pinned too.

Transfer on the **laptop**, using peano and workstation SSH aliases:

```bash
mkdir -p "$HOME/Downloads/chronotree_count_probe"
rsync -avP \
  peano:/hpc/home/phi/rvalperga/chronotree/runs/memory_fv_seed0_compact_hierarchy/seed0/compact/step_00015000.pt \
  "$HOME/Downloads/chronotree_count_probe/student_15k.pt"
ssh workstation 'mkdir -p /home/rvalperga/chronotree/runs/inspection/count_probe'
rsync -avP "$HOME/Downloads/chronotree_count_probe/student_15k.pt" \
  workstation:/home/rvalperga/chronotree/runs/inspection/count_probe/student_15k.pt
```

## Measured checks and preliminary teacher result

On the local RTX 5000 Ada, the device doctor passed FP32/BF16/FP16 checks. The
expert executed 538 native commands and produced 270 retained observations.
Cycle boundaries were native steps 208,344,480; one physical hold after cycle 1
aligned the boundary to the 10 Hz observation cadence. The expert button control
succeeded. Independent replay matched all 539 reset/command observations exactly
in robot state, cube pose, target/button positions, button depth and camera
extrinsics. No seed retry was used. Boundary TCP poses differ by at most 0.624 mm
and cube positions by at most 0.861 mm, before the exact-current intervention.

Source SHA256:
`819a19125bc732da38fb7b325c464c9b6fc99dfc6603f04931290c1b1220db7f`.

The 104,110-step accepted teacher produced all 72 chunks. Natural-history mean
progress toward the button after the first eight targets was:

| Completed k | Requested X | Expected next phase | Button distance reduction |
| --- | --- | --- | ---: |
| 1 | 1 | Button | +60.3 mm |
| 1 | 2 | Pickup | −5.2 mm |
| 1 | 3 | Pickup | −5.5 mm |
| 2 | 2 | Button | +48.6 mm |
| 2 | 3 | Pickup | −3.4 mm |
| 3 | 3 | Button | +49.1 mm |

Thus predictions respond appropriately to the requested count on these natural
expert states. This does not establish successful closed-loop pressing. With
exact current/clock fixed, changing k=2 to k=3 at X=3 changes the first-eight TCP
targets by a mean 28.4 mm over paired noise. Padding all histories to the same
duration weakens the distinction: k=1,X=3 moves 52.2 mm closer to the button,
versus −5.5 mm with its natural history. This is evidence of sensitivity to the
intervention, not a proof of duration-only reasoning; repeated idle frames are
an artificial distribution shift.

Teacher artifacts:
[interactive report](../runs/inspection/count_probe_teacher/index.html),
[full results](../runs/inspection/count_probe_teacher/results.json),
[expert video](../runs/inspection/count_probe_expert_train1_v2/expert.mp4).

CPU checks cover causality, current/clock invariance, equal-duration padding,
overshoot exclusion, matching noise comparisons, geometric metrics, encoder
fingerprints and report validation. `tests/test_count_probe.py` and
`tests/test_count_report.py`: 15 tests passed. Ruff and the architecture snapshot
check passed. Generated data, weights and HTML remain ignored.

## Transferred 15k student comparison

The student arrived at `runs/inspection/count_probe/student_15k.pt`. Its ZIP,
finite tensors, model/training provenance and strict model load passed. It is a
15,000-step compact checkpoint with SHA256
`22e77c01281123398613b95f791e4a10faf68188c1d43a831a0bb261ad2e511b`.
The frozen visual/text encoder state hash matches the accepted teacher exactly.

The completed comparison contains 144 predictions, 72 per model. Teacher FULL
uses 3,328–7,680 historical leaves; student UNIFORM uses exactly 128 summaries
covering the same available leaves. Natural expert-prefix results are:

| Completed k | Requested X | Teacher button progress | Student button progress |
| --- | --- | ---: | ---: |
| 1 | 1 | +60.3 mm | +62.5 mm |
| 1 | 2 | −5.2 mm | −2.3 mm |
| 1 | 3 | −5.5 mm | −3.4 mm |
| 2 | 2 | +48.6 mm | +62.2 mm |
| 2 | 3 | −3.4 mm | +0.5 mm |
| 3 | 3 | +49.1 mm | +66.8 mm |

Progress is reduction in TCP–button distance at target eight, averaged over
four paired noise seeds. It measures a predicted target, not actual contact.
The student retains the expected phase distinction on these natural histories;
this does not establish abstract visual counting because count, time and history
length remain correlated.

With the current observation fixed and histories retimed to the three-cycle
clock, k=2,X=3 should still lead to another pickup. The student instead predicts
58.9 mm of progress toward the button, versus 0.36 mm for the teacher. This
intervention changes timing as well as the tiny natural differences in current
state, so it cannot attribute the problem specifically to visual compression.
With repeated-reset padding the student approaches the button in all six cells
(61.9–70.1 mm progress). Padding also changes visual content mass and sometimes
encoder block alignment; it is an artificial stress test, not a pure clock test.
The observed closed-loop button-hover failure remains untested directly.

The first student comparison exposed a diagnostic identity nuisance: case-specific
renaming of unchanged expert frames changed deterministic UNIFORM tie breaks.
The corrected probe preserves real observation IDs across interventions and
assigns new IDs only to inserted idle copies. Production selection is unchanged.
All 144 predictions were rerun in a fresh output directory. The zero-padding
k=3 control now reproduces identical raw actions bit for bit across the two
fixed-current modes, for both models and all four noise seeds. Natural teacher
and student predictions are unchanged by the correction. The 15 CPU tests pass,
including identity preservation; no optimizer update was performed.

Use the corrected artifacts:
[teacher/student interactive report](../runs/inspection/count_probe_teacher_student_15k_v2/index.html),
[full paired results](../runs/inspection/count_probe_teacher_student_15k_v2/results.json).
