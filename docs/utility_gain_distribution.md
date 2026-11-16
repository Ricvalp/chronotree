# Distribution of refinement effects

This analysis reads the completed `expert_fm_packet_labels_v1` dataset used by
the S_STD95k router pilot. It uses every recorded candidate, not only winning
candidates or examples selected for visual inspection. It requires no GPU,
teacher/student inference, raw demonstrations or encoded cache. The original
labels, training splits and checkpoints remain unchanged.

## Run on the HPC

Pull the implementation before submitting the new closed-loop evaluations, then
run the CPU report independently of those jobs:

```bash
cd /hpc/home/phi/rvalperga/chronotree

CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python scripts/plot_utility_gains.py \
  --labels runs/utility_router_std95k/labels \
  --output runs/utility_router_std95k/gain_distribution \
  --epsilon 0.0001
```

The HTML renderer uses the existing optional visualization dependency. If it is
missing from the HPC environment, install the repository's pinned version:

```bash
uv pip install --python .venv/bin/python "plotly==6.3.1"
```

The script validates complete worker receipts and state payload checksums before
publishing the report. Progress messages distinguish validation from analysis.
The resulting `index.html` is a standalone interactive file; copy it to a laptop
and open it in a browser. `summary.json` and `report.md` contain numerical
summaries, and the compressed candidate table retains individual effects for
further analysis. Generated outputs stay under ignored `runs/`.

`--epsilon` declares the descriptive near-zero band in displayed loss units;
0.0001 is the default, not a calibrated task-relevance threshold. Distribution
curves and quantiles retain all raw values. Use a fresh output directory when
changing this analysis setting; the script refuses to overwrite an existing
nonempty report directory.

## What an effect means

For a fixed decision, history context, candidate and shared flow-noise/time draw:

```text
gain = expert FM loss before refinement - expert FM loss after refinement
```

Positive gains improve prediction; negative gains worsen it. The main plots use
full-chunk expert FM loss, and executed-prefix plots restrict that loss to the
first eight action targets. These are velocity-prediction losses under noisy
expert actions, not changes in simulator success or final generated-action MSE.

An **immediate** label measures the first single-token refinement. A **packet**
label measures the completed allocation of up to eight additional tokens. The
pilot trains on raw total gains with a training-fitted scale. Any per-token view
is an additional descriptive normalization, not a different training target.
Dividing packet gain by its cost does not recover the unmeasured intermediate
single-move effects inside that packet.

The report separates TREE/FLAT, fitting/development states and immediate/packet
effects. It retains independent discovery and confirmation banks, each averaging
four paired noise draws in this pilot. Descriptive candidate-weighted summaries
correspond to the training row sampling; task/state-balanced summaries help check
whether unequal candidate pools dominate the picture.

## Reading the plots

- **Signed histogram and empirical cumulative distribution:** reveal the mass
  near zero and both helpful and harmful refinements. A declared numerical
  near-zero cutoff is not a task-relevance or success threshold.
- **Tail probabilities:** show the fraction exceeding each effect magnitude,
  with positive, negative and absolute tails on logarithmic axes.
- **Concentration curves and top fractions:** show how much positive, negative
  or absolute effect magnitude comes from the largest 1%, 5% and 10% of
  candidates. A large absolute concentration can include harmful refinements.
- **Independent repeatability:** rank or select using discovery gains, then
  inspect the same candidates' confirmation gains. Large discovery-only effects
  may reflect noise; confirmation must not be used to choose that subset.

A long tail or a straight-looking log-log curve alone does not establish a power
law. Task mixtures, baseline difficulty, context size and measurement noise can
also produce a broad distribution. This report describes the gain-blind sampled
candidate distribution, not every possible refinement or an unbiased census of
all demonstration frames. Candidates and noise draws share states and episodes;
they are not independent experimental repetitions.

## Validation and data availability

The current pilot's label files are on the HPC and are absent from the workstation.
Older local S0 measurements cannot substitute for this dataset. Implementation
checks use explicitly synthetic CPU fixtures; no empirical claim about sparsity,
tail shape or a power law is made before the real report is generated.

Implementation verification: ten CPU tests cover raw signed gains, packet-cost
normalization, independent-bank selection, constant/zero/negative-only cases,
all-candidate inclusion, corrupt/incomplete inputs, episode split leakage and
standalone rendering. Five existing label-data regression tests and 32 launcher/
router-evaluation tests passed. Ruff and the unchanged teacher-diagram snapshot
check passed. The synthetic HTML was inspected in headless Chromium with GPU
rendering disabled; plots use SVG, with legible titles/axes and no WebGL dependency.
No production label analysis, GPU inference or Slurm submission ran locally.
