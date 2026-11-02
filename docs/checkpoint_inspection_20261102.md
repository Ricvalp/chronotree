# Three-checkpoint inspection — 2 November 2026

The existing reusable inspection script completed on all three transferred
checkpoints in `runs/inspection/eval_20261102T120621Z/`. Transfer hashes, ZIP CRC,
finite tensors, strict state loading, optimizer/scheduler/sampler and RNG checks
passed. The complete report is
[`index.html`](../runs/inspection/eval_20261102T120621Z_analysis/index.html).

This is the same fixed-scene diagnostic protocol as the
[31 October inspection](checkpoint_inspection_20261031.md), not an official
benchmark success-rate evaluation. All models replan after four native 20 Hz
commands (0.2 seconds); the 10 Hz model executes two targets, each held twice.
This differs from the recent HPC evaluations requesting eight executed targets.
No new training, Slurm submission, or test-seed tuning was performed.

```bash
uv run python scripts/inspect_checkpoints.py \
  --bundle runs/inspection/eval_20261102T120621Z \
  --compare-to runs/inspection/eval_20261031T093812Z_analysis
```

Add `--resume` to reuse and validate these completed outputs. Workflow options
and the exact protocol are documented in [checkpoint_inspection.md](checkpoint_inspection.md).

## Changes since the previous checkpoints

Expert-state error is the mean TCP error over the imminent 0.2 seconds on ten
VideoUnmask training states with paired noise. Closing error is the observed
TCP–cube XY distance after the first close in one fresh PickXtimes validation
rollout, capped at 160 native commands. These are different measurements.

| Model | Previous → current updates | Expert-state TCP error | First-close XY error | Current validation pickup |
| --- | ---: | ---: | ---: | --- |
| Frozen vision, 16 targets at 10 Hz | 60,000 → 104,110 | 7.57 → 5.40 mm | 17.31 → 4.28 mm | Grasp and 135.0 mm cube rise |
| XYZRGB H16, 20 Hz | 82,000 → 114,093 | 5.97 → 6.91 mm | 298.14 → 299.36 mm | No grasp or lift |
| XYZRGB H32, 20 Hz | 60,000 → 91,859 | 8.24 → 6.48 mm | 110.20 → 8.02 mm | Grasp and 125.5 mm cube rise |

Frozen vision retains its validation pickup with closer targeting. H32 now
grasps and lifts in this scene, where it previously missed. Saved command-160
images visibly confirm both lifts. Neither short pickup establishes completion
of the multi-pick task. H16 remains unsuccessful in this validation probe.

Model/data/training/effective-batch/seed provenance and continuation records match
each model's previous checkpoint. Protocol, asset hashes and substantive runtime
identity match too. Source revisions differ (`f389dab` versus `6d6e51b`); reviewed
changes concern parallel evaluation and an optional renderer-binding path unused
by this inspection, rather than models or preprocessing. Training exposure and
several architectural/input settings still differ between models.

## Expert-prefix handoffs

The cells list native-command prefix lengths that led to full task success
within the 320-command diagnostic cap. Prefix zero starts directly from reset.

| Training scene | Frozen 10 Hz | H16 | H32 |
| --- | --- | --- | --- |
| VideoUnmask 0 | 0, 4, 16, 32, 64 | 64 | 64 |
| VideoUnmask 1 | 0, 4, 16, 32, 64 | 32, 64 | 4, 32, 64 |
| PickXtimes 0 | None | None | None |

Both full VideoUnmask expert controls succeed. Frozen vision retains all ten
VideoUnmask handoff successes. H16 loses the scene-0 prefix-32 success; H32 loses
the scene-1 prefix-16 success. Improvement is therefore not uniform.

On PickXtimes training scene 0, frozen vision and H32 now grasp and lift at all
five prefixes, including reset, and reach the button stage. Neither completes
the task within the cap. Both previously failed to grasp from reset on this
training scene. H16 grasps/lifts only after prefixes 32/64 and reaches the first
completed pick/place cycle, without reaching the button stage. Partial progress
is distinct from full task success; capped trials are not full benchmark episodes.

## Visual dependence and cube relocation

Removing current visual tokens changes imminent predictions by 11.20 mm for
frozen vision, 1.50 mm for H16 and 1.43 mm for H32. Removing history visual tokens
changes them by 65.08, 16.54 and 34.33 mm respectively. These artificial deletion
tests measure dependence, not successful localization.

In the four rerendered reset positions, translating the cube by 100 mm changes
the first 0.2 seconds by approximately 0.9–1.5 mm. Frozen/H32 full-chunk endpoints
change by roughly 42–52 mm, similar to the previous inspection. Those later
targets are counterfactual predictions and are not executed in these viewers.
H16's endpoint changes only 2–4 mm, but its predicted horizon is half as long.
At replay state 80, imminent responses remain small (roughly 1 mm in BF16), so
improved pickup in the unshifted scenes does not establish robust correction
after an off-target approach.

## Artifacts and validation

- [Four-position point clouds](../runs/inspection/eval_20261102T120621Z_analysis/reset_cube_chunks/index.html)
- [Fresh rollout videos](../runs/inspection/eval_20261102T120621Z_analysis/rollouts/index.html)
- [Expert-prefix video gallery](../runs/inspection/eval_20261102T120621Z_analysis/handoff/index.html)
- [Machine-readable results](../runs/inspection/eval_20261102T120621Z_analysis/summary.json)

Measured validation: 3,250 offline predictions and 52,729 checks, including 36
early-prefix cases, six parity states, three backward-only probes and three
omitted-input invariants; 383 reset checks; 4,978 grounding/media checks; 45 policy
handoffs and two expert controls, auditing 9,745 policy commands, 1,357 expert
commands, 74 paired-state checks and 2,200 paired-RNG checks. All passed.

The output contains 60 MP4 files: 47 handoff videos, nine handoff comparisons,
three fresh rollouts and one fresh-rollout comparison. Media validation passed,
and all 228 report links resolved. The actual RTX 5000 Ada passed the device
doctor. Offline, grounding and handoff stages took approximately 144, 134 and
1,380 seconds respectively. Data, weights, logs and generated media remain
Git-ignored; no implementation changes were needed.
