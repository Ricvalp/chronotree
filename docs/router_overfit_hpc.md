# Longer fitting on fixed router labels

This is a user-requested fitting diagnostic, separate from the matched
5k/refresh/10k router experiment. It tests whether more optimizer updates can
fit the existing initial labels. It does not generate additional labels, train
the teacher/compact policy, or add validation episodes to the fitting set.

The default is TREE/velocity, the run inspected on 4 November 2026. Its initial
5,000 updates took 16.4 seconds on an H200; validation loss stayed at about
10.37 and final within-context ordering agreement was 50.15%. The three other
initial routers had final ordering agreements between 50.33% and 51.88%.
These are observations from the user's W&B runs, not evidence that longer
training will help. Downloaded W&B inspection records are in the ignored
`runs/inspection/router_loss_10f7ad712dcf45d0/` inspection directory.

## Submit on the HPC

Finish the original router dependency chain before pulling source changes into
its checkout: queued stages verify the source fingerprint. The original
pipeline's read-only status command is:

```bash
cd /hpc/home/phi/rvalperga/chronotree
.venv/bin/python hpc/train_routers.py \
  --site hpc/site_b200_4gpu.env --name memory_fv_seed0_routers --status
```

Once the updated code is on the HPC:

```bash
.venv/bin/python hpc/overfit_router.py \
  --site hpc/site_b200_4gpu.env \
  --experiment memory_fv_seed0_routers \
  --reader tree --objective velocity \
  --steps 100000 \
  --wandb-project chronotree-routers \
  --submit
```

Omit `--submit` to snapshot inputs and render without contacting Slurm.
The default request is **one untyped GPU on gpuq, four CPUs, 16 GB RAM, one
hour**, with no `--exclusive`. No additional GPU speeds up this small fit by
default. The step count is a total: the diagnostic continues the immutable
`step_00005000.pt` to 100,000, adding 95,000 updates. Batch size, learning rate,
weight decay, architecture, episode partition, gain scaling, optimizer state,
and sampling RNG are preserved. The router uses constant AdamW learning rate;
inherited policy scheduler fields do not create a new schedule.

The diagnostic uses the separate W&B project **`chronotree-routers`**, even if
the site file specifies `WANDB_PROJECT=chronotree`. `--wandb-project` explicitly
controls that destination. It has its own W&B identity and default name
`seed0-router-overfit-tree-velocity`. Outputs are under:

```text
runs/memory_fv_seed0_routers_overfit_tree_velocity_100k/seed0/router_tree/
hpc/generated/memory_fv_seed0_routers_overfit_tree_velocity_100k/
```

The generated directory contains configuration, plan, frozen source checkpoint,
Slurm script and logs. Original checkpoints and label HDF5 files are read-only
inputs. The source checkpoint and label hashes are checked before fitting.
Resume an interrupted diagnostic with the same command plus `--resume` after
its previous allocation has ended.

## Interpret the curves

The unmodified source checkpoint is evaluated before further fitting. Every
1,000 updates, the diagnostic evaluates both complete label partitions:

- `training_full_gain_loss`: loss over all fitting rows, not one minibatch.
- `training_within_context_rank_agreement`: candidate ordering on fitting rows.
- `conditional_label_validation_loss`: loss over the held-out episode partition.
- `validation_within_context_rank_agreement`: candidate ordering on that split.
- `baseline/{training,validation}_zero_gain_loss`: predict zero for every row.
- `baseline/{training,validation}_median_gain_loss`: predict the signed median
  fitting-set gain for every row. This median is fitted on training rows only;
  it is a simple reference, not claimed to be the optimal Huber-loss constant.

The ordinary `training_gain_loss` remains the sampled minibatch loss. Losses
retain the original training-only target scale. Ranking ignores near-tied true
gains and never compares candidates across decision contexts. Diagnostic passes
are eval-only, do not consume sampling RNG, and do not alter optimizer updates.

If full-training loss falls and training ranking improves while validation
does not, the model is memorizing rather than generalizing. If neither improves,
longer fitting alone has not solved the problem; inspect label consistency,
feature scale, capacity, and small-subset fit next. A single longer TREE fit is
not a matched reader comparison or a new benchmark result. Any downstream
TREE/FLAT claim needs equal training budgets and a fixed evaluation protocol.

## Verification

CPU tests cover fixed-label provenance, isolated continuation, source guards,
Slurm rendering, project override, baseline separation, exact optimizer/RNG
parity with diagnostics enabled, and interruption recovery. A bounded synthetic
CUDA check on the RTX 5000 Ada continued a two-step router to six steps and
matched an uninterrupted six-step fit exactly (weights and optimizer state).
It peaked at 17.3 MiB allocated for this tiny fixture. This is a correctness
diagnostic, not a full-data speed or overfitting result. No HPC diagnostic was
submitted by the assistant.
