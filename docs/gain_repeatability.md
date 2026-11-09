# Local gain repeatability and visual evidence

The user requested noise-repeatability and interpretable high/low-gain examples,
without the small memorization experiment. This diagnostic performs inference
only. It does not modify policies, production labels, the sampler, or training.

## Measured run: 9 November 2026

Open `runs/inspection/gain_repeatability_20261109/index.html`. Its featured links
lead to interactive point clouds, source RGB frames, candidate anchors/bounds,
TREE parent/child edges, teacher/student/expert TCP target paths, and every
noise-draw gain. All 60 reader-context pages remain accessible, including weak,
negative and unstable cases. HTML and Plotly are self-contained and work offline.

Inputs and scope:

- Exact selected full-history teacher: step 104110, SHA256
  `eb57ccd4c131888a5f5d6225a3a0cbe5cd14b3afc5c6ff0a1a17ee319cd70918`.
- Available local compact: step 83000, SHA256
  `b787dfab69e430c1b85ff9dc4bbd9ae1817dc6646ef5a13473ddc2e7e17c5335`.
  **This differs from the 90000-step compact used for the HPC router labels**, SHA
  `4fe22fd00fbc64dc740ba046d6ce1e58ade2a70580f8792e152477caad316427`.
- One real released training demonstration: VideoUnmask episode_0. Thirty
  decision/context pairs cover 21 distinct decision indices, three budgets
  (64/128/256), and both TREE and FLAT. These are not 30 independent scenes.
- Four random candidate moves per reader-context, sixteen paired Gaussian draws,
  both flow-velocity and eight-step final-action objectives. The first eight
  draws select high/low candidates; the last eight assess those choices.
- No trained router is needed: these are **measured gains, not router scores**.
  No memorization training, simulator rollout, test seed, or cluster submission.

Each context is constructed once with the canonical `LabelDataReader`; only the
prepared item's noise seed changes. Before/after refinements and the teacher use
the same draw. Velocity draws also change flow time; final-action integration
starts from pure Gaussian noise. Targets only define velocity/noise mixtures,
validity masks, and separate expert-error diagnostics. They never enter routing
features or memory selection.

| Reader/objective | Mean cross-draw pair-order agreement | Discovery-high beats discovery-low on held-out mean |
|---|---:|---:|
| TREE velocity | 70.9% | 23/30 |
| TREE final actions | 74.2% | 21/30 |
| FLAT velocity | 69.4% | 27/30 |
| FLAT final actions | 67.2% | 20/30 |

These are descriptive statistics conditional on one episode and the sampled
contexts, not confidence intervals over scenes or estimates of robot success.
The measurements show repeatable signal and substantial noise, not pure noise.
They do not establish that noise is the cause of the HPC router's validation
failure, particularly with the compact-checkpoint difference.

## Useful examples

- `036_VideoUnmask_tree.html`: at decision 20, B=64, a conditioning-frame-64
  refinement around the container hiding the blue target has higher gain than a
  distractor-container refinement in all eight fresh draws. Held-out mean gains
  are approximately +1.79e-4 and -6.35e-5. The target cube is **already covered**
  in the candidate frame; the initial video establishes its location. This is
  plausible target-region utility, not proof of an object-specific feature.
- `021_VideoUnmask_flat.html`: a target-container leaf beats a table leaf in all
  eight fresh draws, but both mean gains are negative. Higher gain can mean a less
  harmful refinement; it does not necessarily mean an improvement over the
  existing residual summaries.
- `032_VideoUnmask_tree.html`: the discovery-selected high-gain candidate loses
  to the low-gain candidate in seven of eight fresh draws. The contexts and
  candidate moves are unchanged.
- `057_VideoUnmask_flat.html`: a table-region leaf wins all sixteen draws against
  the other sampled candidates, but makes expert MSE slightly worse on seven of
  eight held-out draws. The target-container candidate slightly improves expert
  MSE while scoring poorly by teacher agreement. Effects are small. This is an
  example of objective disagreement, not evidence that table detail is generally
  more important to task success.

The featured examples were chosen after inspecting results for explanation. The
full sample and selection protocol are retained, so these examples are not
presented as an unbiased distribution of outcomes. RGB-based spatial descriptions
are visualization-only annotations; no simulator object pose or semantic mask
was supplied to the policies.

Point clouds show actual representative observations. Token bounds/anchors and
descendant centres are not exact pooling masks or object segmentation. TCP curves
show forward kinematics of clipped joint targets, not physically executed
trajectories. Labels compare normalized eight-dimensional actions before
clipping, including orientation-related joints and gripper commands that a TCP
XYZ curve alone cannot show.

## Re-run locally

The defaults select the above local checkpoints and existing diagnostic cache:

```bash
uv run python -m stmem.inspection.gain_repeatability \
  --contexts 30 --draws 16 \
  --output "runs/inspection/gain_repeatability_$(date -u +%Y%m%dT%H%M%SZ)"
```

Use a new directory for new measurements. Regenerate HTML without inference:

```bash
uv run python -m stmem.inspection.gain_repeatability \
  --output runs/inspection/gain_repeatability_20261109 --report-only
```

The code checks teacher/student frozen encoders and normalization, teacher/cache
identity, geometry/tokenization/cadence, content hashes, the original prepared
data receipt, verified alignment, and exact causal prefix counts. Raw RGB is
used only for visualization. An identical-input replay and an identical-context
no-op control both produced zero gain difference. Separately decoding action
paths in different batches differed from canonical gain evaluation by at most
2.38e-5; reported statistics always use the canonical outputs. The larger batch
parity difference occurs in a large-gain, high-variance example. Tiny differences
should not receive semantic interpretation.

Actual-device `stmem.doctor` passed FP32/BF16/FP16 on the RTX 5000 Ada. The measured
diagnostic uses FP32 with TF32 disabled. Forty-five focused CPU tests passed;
the HTML was also exercised in Chromium. No model/context
architecture changes were made, and the architecture diagram check passed.

## Obtain the exact HPC compact checkpoint

Run on the laptop, retaining the existing local 83k snapshot:

```bash
mkdir -p ~/Desktop/chronotree_local/router_gain_audit
rsync -avP \
  peano:/hpc/home/phi/rvalperga/chronotree/hpc/generated/memory_fv_seed0_routers/frozen/compact.pt \
  ~/Desktop/chronotree_local/router_gain_audit/compact_90000.pt

ssh workstation 'mkdir -p /home/rvalperga/chronotree/runs/inspection/router_gain_audit'
rsync -avP ~/Desktop/chronotree_local/router_gain_audit/compact_90000.pt \
  workstation:/home/rvalperga/chronotree/runs/inspection/router_gain_audit/
```

Then the same local command can use
`--compact runs/inspection/router_gain_audit/compact_90000.pt` and a new output
directory. This matches policy weights; a production-label replay would still
need the production sampling plan/contexts. The local diagnostic deliberately
uses its bounded time-stratified VideoUnmask sample.

## Follow-up: expose the actual visible-cube cue

The original context036 example did not isolate revelation of target identity.
The blue cube is visible in source frames 0–31, then covered from 32 onward. At
10 Hz, the 16 retained reveal frames 0, 2, …, 30 contribute 512 leaves. In that example,
all conditioning frames 0–62 were represented by one 1,024-leaf temporal summary;
the highlighted split concerned only two leaves at frame 64, after occlusion.
The earlier cue remained available in summarized form. Its target-refinement
changed predicted TCP targets only 0.354 mm on average, with no target switch.

The transferred 90k compact now matches production labels exactly (SHA above).
The new `stmem.inspection.cue_refinement` diagnostic uses that model and the same
teacher/cache. It fixes current observation, instruction, eight integration
steps and eight paired noise draws, and evaluates decisions 0/20/64 at budgets
64/128/256. All six equal-budget frontiers retain exact causal coverage:

- Uniform largest-region refinement.
- Keep all reveal frames in a protected 512-leaf summary, refine other regions.
- Expose 32 fine anchors near the visible blue cube across the reveal interval.
- Expose 32 fine anchors near the red distractor over the same interval.
- Refine the target location only after occlusion.
- Distribute detail over the whole reveal interval.

Selection uses regions derived from observed prepared RGB, a 70 mm XY radius and
anchor height below 100 mm. It uses no prediction, teacher gain, hidden object pose
or semantic mask. These are diagnostic annotations, not a learned selector.
Frozen CLIP receptive fields and local pooling mean anchors are not isolated
object features. Teacher full-history and compact current-only references are
shown separately; they are not equal-budget coverage arms.

At the primary reset decision and 128-token budget, exposing the actual target
cue changes TCP targets by **1.506 mm on average**, and the final target by
**2.857 mm** relative to the protected-coarse baseline. Across all nine cases,
that refinement does not change which observed cube center is nearest the chunk
endpoint. This is only a directional proxy, not a grasp or task-success result.
Different noise draws produce much larger changes: for reset/B128, mean pairwise
full-path displacement under the coarse baseline is 37.10 mm. This descriptive
pairwise measure does not treat the correlated pairs as independent samples.

Larger changes are possible from other reallocations: the covered-target arm
shifts one reset/B128 draw's endpoint by 145 mm and makes its nearest region switch
from blue to red. Thus large sensitivity alone is not evidence of useful memory.
All arms and draws are retained in the viewer; no successful-looking draw was
selected to represent the outcome.

Open `runs/inspection/cue_refinement_20261109/index.html` for current clouds,
paired action paths, original reveal images and selected historical anchors.
The default is reset/B128. Curves are clipped joint-target forward kinematics,
not executed trajectories. The coarse summary still contains cue evidence, so
these results do not distinguish sufficient coarse representation from weak use
of target identity. A stronger future test would consistently swap blue/red
appearance throughout the reveal video, re-encode RGB/CLIP features, and keep
current observation and action noise fixed. That content intervention has not
been run here.

Rerun with a fresh output directory:

```bash
uv run python -m stmem.inspection.cue_refinement \
  --decisions 0 20 64 --budgets 64 128 256 --draws 8 \
  --output "runs/inspection/cue_refinement_$(date -u +%Y%m%dT%H%M%SZ)"
```

The identical-input replay produced zero numerical difference. 44 focused CPU
tests passed; actual-device doctor passed FP32/BF16/FP16. No training, simulator
rollout, test seed, benchmark tuning or cluster submission occurred.

## Follow-up: swap the remembered target identity

The user authorized the stronger appearance counterfactual. The verified output
is `runs/inspection/cue_counterfactual_verified_20261109/index.html`.
This uses the same exact 90k student and 104110-step teacher, one VideoUnmask
training episode at reset, and 16 fixed paired Gaussian draws. No simulation
was executed and no weights were updated.

In both cameras, saturated blue/red pixels are exchanged inside observed RGB-D
regions around the two cubes during all 16 retained reveal frames (0,2,…,30).
The spatial mask uses observed backprojection, a 70 mm XY radius and height
5–100 mm; it is an offline edit, never a policy feature. The edit preserves
lighting through an R/B channel permutation and is exactly reversible. Green
and background pixels remain unchanged. The goal still asks for the blue cube,
so the intended target moves to the original red location, 237.6 mm away.

Recompute full CLIP maps and all sampled-point descriptors in edited frames,
not just the changed pixels' descriptors. Update canonical point RGB at its
original camera/pixel correspondences, preserving point geometry, validity,
center indices, time, proprioception and actions. Re-encode complete history
blocks and all parent summaries. Reuse the exact same current records and task
embedding. Original and altered inputs use identical hierarchy node IDs and
matched selections; the student is tested with uniform TREE, protected coarse
reveal history, fine detail around both cubes, and fixed random FLAT at B128.
The teacher uses every leaf. Original expert paths are visual references only;
they are not counterfactual supervision.

| Model/read | Mean TCP path change | Mean final-target change |
|---|---:|---:|
| Full-history teacher | 0.524 mm | 1.459 mm |
| Student uniform | 0.049 mm | 0.070 mm |
| Student coarse reveal | 0.017 mm | 0.030 mm |
| Student fine detail around both cues | 0.762 mm | 1.758 mm |
| Student random FLAT | 0.100 mm | 0.192 mm |

No student changes its nearest-of-two endpoint region in any of the 16 draws.
The teacher changes one assignment toward the old region, not the new target.
These are directional proxies: a chunk may end high above or between objects.
Individual differences can exceed the means (maximum pointwise teacher change
15.73 mm), so do not describe every draw as submillimetre. The correct conclusion
is no intended target redirection in the first chunk of this scene; later phases
and other scenes have not been tested with this intervention.

Input controls are exact: original camera pixels reproduce canonical RGB;
recomputed unchanged CLIP descriptors and teacher predictions match cached-input
results; identical-input prediction replay is unchanged. Eighty of the 512 cue
leaves change, with 14.9% relative aggregate content L2 change; no other leaf
changes. Both cameras' edited images were visually inspected in the HTML.
This supports weak use of target identity in the first chunk, even by the
teacher, rather than merely too little fine detail in the student. It does not
establish that the teacher ignores memory throughout an entire rollout.

Rerun with a new output directory:

```bash
uv run python -m stmem.inspection.cue_counterfactual \
  --draws 16 --budgets 128 --decisions 0 \
  --output "runs/inspection/cue_counterfactual_$(date -u +%Y%m%dT%H%M%SZ)"
```

35 focused CPU tests passed, covering reversible appearance editing, original
camera/UV ownership, observed-depth spatial restrictions, exact-coverage
frontiers, metric units and HTML input validation. Browser checks confirmed
paired images, comparison controls and unchanged current imagery. No model
architecture or scientific training defaults changed.

## Limits of the counterfactual: other cues and scene memorization

A follow-up source/data audit does not establish that the cue video is the only
possible predictor of target location. The pinned VideoUnmask generator places
containers sequentially and always asks for `bin_0` first; the public goal names
its subsequently shuffled cube color. Later container placement uses rejection
sampling around earlier containers. Geometry/material are shared, but there is
a possible layout/generation-order correlation. The model does not receive bin
IDs, simulator poses, episode seeds, subgoals or completion flags. This actual
checkpoint disables proprioception and previous-command conditioning, and reset
uses a fixed robot pose/action before the first policy decision.

Observed RGB-D measurements across all 100 released training scenes are saved
in `runs/inspection/cue_leak_audit_20261109/dataset_layout.json`. The first target
color is blue/green/red in 21/33/46 scenes; target centers have 99 distinct XY
positions at 1 mm rounding. Among the three colored locations, target X rank
is lowest/middle/highest in 34/30/36 scenes, and Y rank in 30/38/32. Thus a fixed
absolute location or always-left/right rule is not the dataset rule. Target is
closest to scene origin in 47/100 scenes, a descriptive layout correlation, not
proof the policy exploits it. These ranks exclude additional empty containers
in harder scenes.

The pixel audit found no remaining blue-dominant pixels around the original
target after editing reveal frames 0/16/30 in either camera, even when including
dim pixels. Covered frames 32/64 and current raw frame 66 also contain none in
that region. This rules out an obvious remaining blue patch in those inspected
frames, not all possible visual correlations.

The counterfactual is a training-scene diagnostic. A mapping memorized from the
full layout to the original target, a geometric shortcut, or a shared initial
movement can all explain its weak first-chunk response. Do not conclude from it
that the policy cannot infer the original target elsewhere or always chooses
the same container across scenes. A stronger follow-up needs held-out layouts,
fixed geometry with independently varied cue-defined targets, and observations
of later target-directed motion; no such new rollout was run in this audit.

## Ordinary VideoUnmaskSwap validation configurations — 2026-11-09

Following the single-scene counterfactual, ran normal closed-loop rollouts on
official validation configurations 0–9, without any history, appearance, goal,
current-observation or simulator-state interventions. This check uses the
selected full-history teacher at step 104110, SHA256
`eb57ccd4c131888a5f5d6225a3a0cbe5cd14b3afc5c6ff0a1a17ee319cd70918`,
with 16 predicted targets, 10 executed targets per replan, action and observation
stride 2, and eight FM integration steps. Noise seeds retain the normal full-task
list indexing. Two workers shared the local RTX 5000 Ada; all ten completed, with
no errors or timeouts. This is not a router or compact-reader result.

| Configuration | First requested container | First contact grasp | Official outcome |
|---|---|---|---|
| 0 | 2 (green) | 1 | fail |
| 1 | 1 (red) | 2 | fail |
| 2 | 1 (green) | 0 | fail |
| 3 | 2 (red) | 1 | fail |
| 4 | 1 (blue) | 0 | fail |
| 5 | 0 (green) | 1 | fail |
| 6 | 2 (green) | 2, briefly; later 3 | fail |
| 7 | 0 (blue) | 1 | fail |
| 8 | 1 (red), then 0 (green) | 1, then 0 | success |
| 9 | 1 (green) | 0 | fail |

Container indices are zero-based simulator creation identities within each
scene, not left/right labels or policy inputs. Choice observations are read-only
and excluded from the policy. First-grasp coordinates span X=-0.121…+0.171m and
Y=-0.177…+0.126m and their within-scene spatial ranks vary. Thus the policy does
not always pick one fixed identity, one fixed spatial rank, or one world
location in this sample. However, 1/10 official success and eight initially
wrong grasps provide no evidence of reliable target tracking. Configuration 6's
correct initial contact lasts one retained observation at command 122, never
lifts the requested container, and later transitions to a distractor. Only
configuration 8 completes the requested pickup sequence.

All failures stop before completing the first requested pickup. Do not
interpret the absence of a recorded wrong-container lift above 0.15 m as no
choice: official failure can occur at the threshold-crossing terminal step,
whose wrapper-generated observation is deliberately excluded. Earlier grasp
contacts and the terminal outcome are available. Successful choice in one scene
cannot establish that history caused the choice; conversely these ten scenes
cannot prove complete history blindness or memorization. Pre-swap poses were
not recorded numerically, so an original-location shortcut remains an untested
hypothesis, with reveal-and-swap videos available for qualitative inspection.

Interactive report with overhead paths and all ten videos:
`runs/inspection/video_unmask_swap_normal10_20261109/index.html`.
The same videos/results are logged at
https://wandb.ai/ricvalp/chronotree-routers/runs/c16dbb018a75f627 .

Reproduce into a fresh directory (actual simulator, no data download):

```bash
uv run python -m stmem.inspection.swap_rollouts \
  --checkpoint runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt \
  --output runs/inspection/video_unmask_swap_normal10_repeat \
  --episodes 10 --workers 2 --execute-actions 10
```

13 observer/report CPU tests pass, including unchanged adapter return objects,
report-only hidden state, terminal exclusion, planned denominators and video
paths. Architecture diagram snapshot check passes unchanged. No training,
Slurm submission, or test-seed tuning was performed.

## Paired history-use rollouts on other tasks — 2026-11-09

The normal VideoUnmaskSwap results did not establish useful memory, so the next
diagnostic separates execution-history dependence from use of the initial
demonstration. Use the same full-history teacher at step 104110 and the same
checkpoint hash as above. These are teacher results, not compact or router
results. Frozen visual features and no explicit proprioception or previous
commands are preserved. Historical inputs here are encoded visual observations.

The primary comparison fixes official validation indices 0–3 on SwingXtimes,
VideoRepick and VideoPlaceOrder before seeing the outcomes. All four arms use
the same model, natural simulator reset, goal, clock and flow-noise seed:

- FULL: ordinary full-history action-head read.
- Erase conditioning: remove the initial demonstration's historical records.
- Erase execution: remove prior rollout observations' historical records.
- CURRENT: the same teacher with an empty historical read. This is not the
  separately trained current-only model.

The simulator always plays the original demonstration; no scene, target, RGB,
geometry or expert sequence is swapped. Later closed-loop states can diverge
because actions differ. A separate shadow probe on FULL decisions 0, 2, 5, 10
and 20 holds current observation, goal, clocks and exact FM noise fixed while
reading each alternative history. It executes no counterfactual action and
preserves all policy and global RNG state. Exact FULL replay and reset-input
fingerprints serve as controls. Simulator task states and actor poses are saved
only by a read-only observer and never supplied to the policy.

Execution is 16 predicted targets, 10 executed targets, action and observation
stride 2 on the native 20 Hz clock, and eight flow-integration steps. All arms
retain the canonical full-task-list noise-seed indexing and the same native
episode limit. These small selected validation subsets are diagnostic results,
not an estimate of all-task benchmark performance.

Two follow-ups reuse the four original FULL Swing results after checking their
hashes and model/input/execution settings, adding four fresh rollouts each:

- Recent execution: keep the last two complete eight-observation execution
  blocks plus the pending block, and all conditioning. At this replanning
  cadence the window stabilizes at 16, 18, 20 or 22 retained observations,
  roughly two seconds. Encoded contents are not recomputed.
- Permuted execution: retain every historical payload and geometric descriptor,
  token count, modality/source count and original time slot; change the
  association between execution payloads and time slots. This is the existing
  encoded-evidence permutation, applied after local spatiotemporal encoding.
  It does not remove temporal information already embedded inside each latent.

Commands use fresh output directories and log to `chronotree-routers`:

```bash
uv run python -m stmem.inspection.history_use \
  --checkpoint runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt \
  --output runs/inspection/history_use_repeat \
  --episodes 4 --workers 2 --execute-actions 10

uv run python -m stmem.inspection.history_use \
  --checkpoint runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt \
  --output runs/inspection/history_recent_swing_repeat \
  --episodes 4 --workers 2 --execute-actions 10 \
  --recent-swing --reference-output runs/inspection/history_use_repeat

uv run python -m stmem.inspection.history_use \
  --checkpoint runs/inspection/eval_20261102T120621Z/frozen_10hz/checkpoint.pt \
  --output runs/inspection/history_permuted_swing_repeat \
  --episodes 4 --workers 2 --execute-actions 10 \
  --permuted-swing --reference-output runs/inspection/history_use_repeat
```

Reports retain every result, raw action/observer/probe JSON and video:
`runs/inspection/history_use_20261109/index.html`,
`runs/inspection/history_recent_swing_20261109/index.html`, and
`runs/inspection/history_permuted_swing_20261109/index.html`.

### Completed measurements

All 56 fresh simulations completed: 48 primary cases and four per follow-up.
The eight additional FULL rows in the follow-up reports are reused references,
not additional simulations. There are no infrastructure or observer errors.
All 12 primary scenes have identical reset-input fingerprints across their four
arms. All 52 fixed-current probes pass with zero FULL replay action error.
Every timeout reaches the normal 1,301-command task limit; none is a diagnostic
wall-clock timeout.

| Task (validation 0–3) | FULL | Erase conditioning | Erase execution | CURRENT |
|---|---:|---:|---:|---:|
| SwingXtimes | 4/4 | 4/4 (no conditioning exists) | 0/4 | 0/4 |
| VideoRepick | 1/4 | 0/4 | 0/4 | 0/4 |
| VideoPlaceOrder | 0/4 | 3/4 | 0/4 | 0/4 |

For the same four Swing scenes, recent execution succeeds 0/4 and permuted
execution succeeds 0/4, versus matching FULL 4/4. The main W&B run is
https://wandb.ai/ricvalp/chronotree-routers/runs/64e4a3a59e27defe ; recent-window
and permutation runs are `c96dac2cafc8168b` and `3b53e510e23e6a82` in the same
project. HTML reports and every video were browser-checked after completion.
The raw aggregate, per-task behavior reviews and final audit remain alongside
the reports. These results establish useful history dependence under the
specified interventions, with stronger evidence for movement continuity than
reliable semantic retrieval of a demonstrated identity or destination.

### Interpretable paired cases

SwingXtimes has no initial conditioning video. FULL and erase-conditioning are
therefore an exact no-op pair: all four complete executed action sequences and
outcomes match, and all four succeed. Erase-execution and CURRENT both fail to
grasp the cube and reach the normal 1,301-command task limit in every scene.
Keeping recent history allows all four scenes to grasp and lift the intended
cube, but all four then open the gripper before completing the first swing.
The first 60 native commands match FULL exactly; divergence begins at command
61, when the recent-window intervention first discards an older block. Open
commands occur at 162, 142, 142 and 126, respectively. The recent-window outcomes
are three task timeouts and one failure. This supports useful dependence on
older visual history for movement continuity, not a claim about remembering
the number of completed swings: these failures occur before one complete swing.

Preserving every token while permuting execution payloads across time slots
also gives 0/4 Swing successes, all normal 1,301-command timeouts with no error.
Scenes 0 and 2 never grasp; scene 1 briefly contacts the correct cube but never
lifts it. Scene 3 eventually lifts the correct cube at command 242, enters the
first swing phase at 244, then loses the grasp at 250 and stalls. No scene
completes a swing. In all four, the first 20 native commands match FULL exactly,
and divergence begins at command 21, the first replan with historical input.
Historical token counts match FULL at every shared decision. This removes the
token-count explanation for these performance differences, but scrambling
content/time association remains an out-of-distribution intervention.

VideoRepick validation 1 provides a successful-sequence example. FULL grasps and
releases the correct cube three times (native command intervals 78–132,
204–256 and 326–378), then successfully presses the button at command 471.
Erasing the initial demonstration still selects the same correct cube, but
keeps its first grasp from command 76 to the last active observation at 152,
carries the cube toward the button and fails at 153. This is a sequencing
failure rather than wrong-object selection. In hard validation scene 3, FULL
correctly picks cube 14 of 15 twice before timing out during the third cycle;
the erased-demonstration control never lifts a cube and briefly contacts the
wrong cube 13. This latter example is a useful identity-selection hint, but
neither arm completes that scene. Actor indices are report-only, not model inputs.

VideoPlaceOrder validation 1 is a counterexample to uniformly beneficial
conditioning. FULL places the cube at the wrong pad 0; the erased-demonstration
control places it at requested pad 3 and succeeds. Last active XY distance from
the requested pad is 25.0 cm for FULL versus 3.2 cm for the erased arm. Both
versions choose the same wrong pad in scene 0. The erased-demonstration successes
in scenes 1–3 all choose a positive-X/negative-Y table region; scene 0's desired
pad lies elsewhere. This is compatible with a location preference, not proof
that the video-free policy somehow recovered the demonstrated ordinal order.
Visual review also does not support a consistent last-demonstrated-pad rule:
scene 1 shows pads 3, 2, 0, 1 and FULL selects the third (0), whereas scene 2
shows 1, 0 and FULL selects the first (1), although the second (0) is requested.
These order reconstructions are visual annotations of the retained videos.
FULL's fourth PlaceOrder scene is a near-correct placement rather than a wrong
pad: the cube is released about 6.03 cm from the requested pad center, outside
the task's 5 cm success threshold, and remains near that position until timeout.
Thus FULL has three wrong-pad failures and one near-correct-pad timeout here.

Videos include the original conditioning before execution. The viewer labels
this explicitly and can start each video at the policy's first command:
VideoRepick scene 1 has 13.25 seconds of conditioning, VideoPlaceOrder scene 1
has 54.35 seconds. Do not attribute these expert demonstration movements to the
policy. Decoded paired contact sheets and observer reviews are retained in the
main output directory.

Implementation validation: 52 CPU tests pass across observers, fixed-current
probes, recent windows, permutation parity, HTML reports and existing routing
invariants. The architecture snapshot check passes unchanged and Ruff is clean.
Future permutation runs use an exact ordered-leaf implementation, avoiding
unneeded parent reconstruction; all leaf fields/IDs and shared-noise flow
outputs match the previous intervention. This campaign used the previous
intervention throughout. A synthetic 4,096-leaf CPU check measured 0.1405 versus
0.0131 seconds for this structural operation only. Full-history per-record
metadata preparation/transfers dominate the long failed rollouts, so this is
not an end-to-end speedup claim. No production architecture or training changes.
