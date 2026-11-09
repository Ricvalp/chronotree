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
