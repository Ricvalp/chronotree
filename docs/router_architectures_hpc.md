# Fixed-label router architecture experiments

These are fresh architecture fits on the existing initial label dataset. The
teacher, compact policy, encoders, hierarchy, target gains and episode split
stay fixed. The default objective is final-action discrepancy with the original
eight-step solver. No raw preparation, teacher inference, label refresh or
full-history attention is added.

The default suite contains eight training runs:

| Architecture | TREE | FLAT | Change |
|---|---|---|---|
| `dot` | Yes | Yes | Fresh original two-tower baseline |
| `mlp` | Yes | Yes | Nonlinear query/candidate interaction, same inputs |
| `children` | Yes | No | MLP scorer also receives both children individually |
| `current_attention` | Yes | Yes | MLP scorer uses task-conditioned current-token attention |
| `children_current_attention` | Yes | No | Both information changes |

Compare the child and current-attention variants against `mlp` to isolate added
information; compare `mlp` against `dot` to isolate the scoring change. Children
are a TREE-specific diagnostic, not a matched TREE-versus-FLAT performance
claim. All fits use the original batch size, learning rate and weight decay,
and 100,000 fresh optimizer updates by default. They do not resume the old
5,000-step routers, whose provenance is checked and snapshotted only to verify
the fixed label dataset. No measured improvement is claimed before these jobs
run.

## Submit the suite

After the updated code has been pushed and pulled onto the HPC:

```bash
cd /hpc/home/phi/rvalperga/chronotree

.venv/bin/python hpc/router_architectures.py \
  --site hpc/site_b200_4gpu.env \
  --experiment memory_fv_seed0_routers \
  --name memory_fv_seed0_router_arch_final8 \
  --objective final_action \
  --steps 100000 \
  --wandb-project chronotree-routers \
  --dry-run
```

This validates the original experiment's immutable initial configurations,
5k checkpoints and labels, then writes the plan, new configurations and batch
scripts. It does not contact Slurm. Submit with the same settings:

```bash
.venv/bin/python hpc/router_architectures.py \
  --site hpc/site_b200_4gpu.env \
  --experiment memory_fv_seed0_routers \
  --name memory_fv_seed0_router_arch_final8 \
  --objective final_action \
  --steps 100000 \
  --wandb-project chronotree-routers \
  --submit
```

The site file supplies repository/environment/account locations. The launcher
explicitly requests **`gpuq`, `--gres=gpu:1`, one node, one task**, and no
`--exclusive`. It does not request a GPU model, despite the site's B200 filename.
It overrides any previous `WANDB_PROJECT` with **`chronotree-routers`**. No shell
`source` is needed when invoking the repository's Python directly as above.

The submitted work consists of:

- One four-element input-reconstruction array. Each allocation requests one
  untyped GPU, eight CPUs, 32 GB RAM and two hours. Reconstruction is CPU work;
  the GPU reservation allows it to use the available GPU partition. The four
  indices are independent and can run simultaneously. Each reconstructs one
  TREE and one FLAT sidecar using already encoded observations and hierarchy.
- Eight independent training jobs, each requesting one untyped GPU, four CPUs,
  16 GB RAM and two hours. `dot` and `mlp` need no sidecars and may start
  immediately. The other four jobs wait for successful input reconstruction.

Thus there are twelve allocations in total and at most eight simultaneously
eligible allocations in this dependency graph. Resource/time requests are
provisional; reconstruction and attention-variant timings have not been
measured on the full HPC dataset. They are not predicted GPU-hour costs.

The reconstruction does **not** recompute gain targets. It replays the original
decision/refinement sampling plan, verifies the old query/candidate rows and
episode partitions, and stores the missing current-token and child inputs
separately. The original label files are read-only. Completed input shards are
validated and reused on restart.

Source/configuration hashes are checked at allocation startup. Keep the
checkout fixed while these jobs are queued or running; an isolated checkout is
appropriate if other code work must continue in parallel.

## Locations and progress

```text
hpc/generated/memory_fv_seed0_router_arch_final8/
  plan.json
  configs/
  jobs/
  logs/                         # .out and .err, including array indices
  submitted.json                # job IDs and dependency commands

runs/memory_fv_seed0_router_arch_final8/
  inputs/{tree,flat}/            # shared immutable reconstructed inputs
  <architecture>/<reader>/seed0/router_<reader>/
    metrics.jsonl
    config.json
    last.pt
    step_*.pt
    tracking.json
```

Read-only progress:

```bash
.venv/bin/python hpc/router_architectures.py \
  --site hpc/site_b200_4gpu.env \
  --name memory_fv_seed0_router_arch_final8 \
  --status

squeue -u "$USER"
```

`--status` reads recorded filesystem state; it neither submits jobs nor queries
Slurm. To continue interrupted allocations, use the original submission
command plus `--resume`. Completed fits are skipped, active job IDs are not
duplicated, and each interrupted fit resumes its own `last.pt` and W&B run.
An ambiguous submission receipt is an error requiring reconciliation rather
than a blind duplicate submission.
Dependent fits use Slurm's `--kill-on-invalid-dep=yes`, so a failed input array
cancels its waiting fits instead of leaving them pending indefinitely. Resume
after that cancellation has propagated. This behavior is defined by the
[Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html).

## Optional small memorization suite

This is a separate diagnostic using 256 whole training contexts; validation
episodes remain held out. It uses a separate name/output directory and should
not be interpreted as end-to-end policy performance.

```bash
.venv/bin/python hpc/router_architectures.py \
  --site hpc/site_b200_4gpu.env \
  --experiment memory_fv_seed0_routers \
  --name memory_fv_seed0_router_arch_tiny256 \
  --objective final_action \
  --fit-contexts 256 --steps 10000 \
  --wandb-project chronotree-routers \
  --submit
```

For only the cheapest fits, add `--variants dot mlp`; no reconstruction jobs
are then needed. `--readers tree` limits the suite to TREE. These choices,
training duration, resource requests and source fingerprint are immutable for
a named experiment. Use a new `--name` when changing them.

Compare full training and conditional label-validation loss/ranking, constant
baselines and the label-based top-choice metrics. The ordinary minibatch loss
is not the full fitting-set loss. Held-out label performance is conditional on
the frozen policies/encoders, which may already have seen those episodes;
it is not a benchmark success rate. Closed-loop and routing-latency evaluation
remain necessary before claiming better control or efficiency.

## Matched closed-loop check

The completed final-action architecture screen's highest FLAT-dot ranking
accuracy was 52.33% at 2,000 updates. That checkpoint was not retained: this
trainer does not save a validation-best checkpoint automatically. Its retained
50,000-update snapshot reached 51.98%, the best ranking accuracy among the
retained suite snapshots. These are label-ranking scores, not task success.
Evidence: W&B run `d422829bd22e63d1`, retrieved 2026-11-09.

Evaluate that router against random FLAT selection using the exact frozen
90k compact policy used for its labels. Both retain the same residual summaries,
128-token budget, four validation episodes per task, ten executed targets per
chunk, and checkpoint-defined action solver and seeds. Action-noise and random
selection RNGs are separate. The trajectories can diverge after selection affects
the action, as intended. This comparison measures the effect of learned ranking;
it does not remove history or replace the compact policy with the teacher.

After updating the HPC checkout, run in Bash:

```bash
cd /hpc/home/phi/rvalperga/chronotree

EVAL_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
ROUTER_EVAL_ARGS=(
  --site hpc/site_b200_4gpu.env
  --checkpoint hpc/generated/memory_fv_seed0_routers/frozen/compact.pt
  --method FLAT --budget 128 --execute-actions 10
  --partition gpuq --gpu-type ""
  --gpus 4 --workers-per-gpu 4 --cpus 64 --mem 128G
  --walltime 06:00:00 --episode-count 4 --video-cap 10
  --wandb-project chronotree-routers
)

.venv/bin/python hpc/evaluate.py "${ROUTER_EVAL_ARGS[@]}" \
  --name "router_dot50k_execute10_${EVAL_STAMP}" \
  --router-checkpoint runs/memory_fv_seed0_router_arch_final8/dot/flat/seed0/router_flat/step_00050000.pt \
  --submit

.venv/bin/python hpc/evaluate.py "${ROUTER_EVAL_ARGS[@]}" \
  --name "random_flat90k_execute10_${EVAL_STAMP}" \
  --selection-family flat \
  --submit
```

The jobs are independent. Each requests four untyped GPUs on gpuq, 64 CPUs,
128 GB RAM, and runs up to 16 simulations concurrently. Neither requests
`--exclusive`. Each evaluates 64 episodes and saves at most ten videos.
The explicit W&B project overrides the site's `WANDB_PROJECT`.

For each name, results are in `runs/evaluation/<name>/summary.json` and
`runs/evaluation/<name>/videos/`; Slurm logs are in
`hpc/generated/<name>/logs/`. Video links refer to files in the corresponding
`shards/`, which must be kept. W&B records separate evaluation runs.

The launcher snapshots the policy and router under
`runs/evaluation_inputs/<name>/`, checks reader family and exact compact SHA256,
and records both hashes in the plan. It never follows a moving `last.pt` on
resume. Reusing a name requires the original source paths and identical
experimental settings; use the original arguments plus `--resume` for an
interrupted evaluation. Replacing `--submit` with `--dry-run` only snapshots
and renders. Keep source code unchanged while evaluations are queued/running.
