# Matched random TREE/FLAT compact checkpoint selection

This is the Stage P checkpoint comparison prescribed by
`PROMPT_CHRONOTREE.md` §4.1.1 and §9.3. The two candidates are the existing
15,000- and 30,000-update checkpoints of **one shared compact policy**.
No weights or routers are trained by these evaluation jobs.

For each candidate, run two closed-loop evaluations:

| Context family | Historical context supplied to the action head |
| --- | --- |
| Random TREE | A valid mixed-resolution frontier chosen using seeded random refinement scores. Parents summarize entire subtrees; chosen records cover every available leaf exactly once. |
| Random FLAT | Randomly selected fine leaves plus one residual summary for each nonempty source/modality group of unselected leaves. Selected leaves and residuals cover every available leaf exactly once. |

These are **random context families**, not the final learned TREE and FLAT
readers. They test whether the shared decoder and adapter work with both kinds
of compressed context used in training. UNIFORM, used in the recent interim
evaluations, is a third, deterministic coverage frontier; its success alone is
not the planned selection criterion.

Fix the history budget at **128 tokens**, including FLAT residuals and TREE
summaries. Current visual, goal and clock tokens are additional and unchanged.
Short histories or indivisible tree refinements may use fewer tokens. Both
families preserve causal evidence and exact leaf coverage; they choose different
resolutions of that evidence, rather than discarding unselected history.

All four jobs use the first **four official validation episodes per task**,
64 episodes across 16 tasks, the same episode/noise seed setting from the
checkpoint, the same model/sensing configuration, and **eight executed targets
per chunk**. For these checkpoints each 10 Hz target is held for two native
20 Hz commands, so replanning occurs after 0.8 seconds unless the episode ends.
The predicted horizon remains 16. Selection randomness resets from the episode
seed independently of flow sampling. Different decisions can lead to different
later trajectories: matched means the same initial scenes and evaluation
protocol, not identical subsequent observations or random TREE/FLAT token sets.

## Submit the four evaluations

Push the implementation and pull it on the HPC first. No new site file or
dataset/cache preparation is needed. These commands explicitly select `gpuq`
with **untyped `gpu:4`**, not `gpu:h200:4`, and do not use `--exclusive`.

The subshell checks that both immutable candidate files exist before submitting
anything. If 30k has not been saved yet, wait for it; do not substitute a changing
`last.pt` for this predeclared comparison.

```bash
cd /hpc/home/phi/rvalperga/chronotree
git pull

(
  set -e
  compact_dir=runs/memory_fv_seed0_compact_hierarchy/seed0/compact
  for step in 15000 30000; do
    printf -v checkpoint '%s/step_%08d.pt' "$compact_dir" "$step"
    test -f "$checkpoint" || { echo "Checkpoint not ready: $checkpoint"; exit 1; }
  done

  for step in 15000 30000; do
    printf -v checkpoint '%s/step_%08d.pt' "$compact_dir" "$step"
    for family in tree flat; do
      .venv/bin/python hpc/evaluate.py \
        --site hpc/site_b200_4gpu.env \
        --checkpoint "$checkpoint" \
        --name "compact_${step}_${family}_b128_exec8" \
        --method "${family^^}" --selection-family "$family" --budget 128 \
        --partition gpuq --gpu-type "" \
        --gpus 4 --workers-per-gpu 4 --cpus 64 --mem 128G \
        --walltime 06:00:00 \
        --episode-count 4 --video-cap 10 --execute-actions 8 \
        --submit
    done
  done
)
```

This submits **four independent jobs**, each requesting four GPUs and running
16 simulator workers. Each evaluates 64 episodes and saves up to 10 videos:
256 rollouts and up to 40 videos in total. Slurm may run them concurrently or
queue them; the site file's planner concurrency limit does not coordinate these
four standalone submissions. Use `--dry-run` instead of `--submit` to render
without submitting, then repeat with the same names and options to submit.

Each plan snapshots its named checkpoint. Output directories are:

```text
runs/evaluation/compact_15000_tree_b128_exec8/
runs/evaluation/compact_15000_flat_b128_exec8/
runs/evaluation/compact_30000_tree_b128_exec8/
runs/evaluation/compact_30000_flat_b128_exec8/
```

Each contains `summary.json`, `manifest.json`, `complete.json` once finished,
and `videos/` links into its `shards/` directories. Slurm logs are under
`hpc/generated/NAME/logs/`; per-worker logs are under the output's `workers/`.
W&B names include the method, `random-selection`, and checkpoint step. These
evaluation jobs do not alter the training checkpoints. Resume partial jobs
with the same command/name/options plus `--resume`, after the prior allocation
has stopped; completed jobs should not be resubmitted.

## Select the shared checkpoint after all four jobs finish

For each candidate the score is:

```text
0.5 × TREE macro_success + 0.5 × FLAT macro_success
```

Higher is better; ties choose the earlier predeclared candidate. The same
selected weights are then used for both learned routers. A better random TREE
score by itself does not establish that a learned TREE router will outperform
FLAT; that comparison comes after router supervision and fitting.

Run this lightweight selection command on the HPC login node:

```bash
.venv/bin/python -m stmem.training.select_checkpoint \
  --candidates \
    runs/memory_fv_seed0_compact_hierarchy/seed0/compact/step_00015000.pt \
    runs/memory_fv_seed0_compact_hierarchy/seed0/compact/step_00030000.pt \
  --evaluations \
    runs/evaluation/compact_15000_tree_b128_exec8 \
    runs/evaluation/compact_15000_flat_b128_exec8 \
    runs/evaluation/compact_30000_tree_b128_exec8 \
    runs/evaluation/compact_30000_flat_b128_exec8 \
  --output runs/memory_fv_seed0_compact_hierarchy/seed0/compact/selected.pt
```

The resolver verifies checkpoint hashes, complete planned episode coverage,
matching family/method labels, absence of learned routers, and identical
budgets, seeds, execution settings and source code. It creates `selected.pt`
and `selected.selection.json` with scores and evidence paths. It refuses to
replace an existing selection with different weights. These 64-scene scores
remain a small validation comparison, not a final test-set result.

## Implementation checks and limits

Parallel and serial evaluators now reject ambiguous random-selection requests;
the HPC launcher forwards the family and budget and binds them to the named
plan's identity. Parallel manifests and worker manifests retain the same family.
CPU fixtures exercise both families through supervision, aggregation, immutable
resume, and balanced checkpoint resolution, including rejection of mismatched
budgets, seeds, execution settings, incomplete workers and wrong checkpoint
identities. No encoder, hierarchy, context selection algorithm or training
mixture changed. The diagram's source snapshot is regenerated for evaluator
fingerprint changes; its architecture and layout remain unchanged.

Validation commands:

```bash
uv run pytest tests/test_compact_selection.py tests/test_parallel_evaluation.py \
  tests/test_parallel_evaluation_hpc.py tests/test_evaluation_aggregation.py \
  tests/test_evaluation_execution.py tests/test_cache_runtime_compatibility.py
uv run python scripts/generate_teacher_diagram.py --check
uv run python -m stmem.doctor
```

Measured locally: **121 tests passed**; the architecture check passed, and the
RTX 5000 Ada doctor passed FP32, BF16 and FP16 compute checks. The local 15k
checkpoint was also read to confirm compact stage, budget 128, horizon 16,
stored execute 2 and action stride 2; the submitted override is therefore needed.
No Slurm jobs or full rollouts were run as part of this implementation. Actual
four-GPU runtime and TREE/FLAT success rates must be measured on the HPC.
