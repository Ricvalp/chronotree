# Parallel closed-loop validation

For the shared compact policy's 15k/30k random TREE/FLAT comparison, see
[matched compact checkpoint selection](compact_checkpoint_selection.md).

`hpc/evaluate.py` evaluates one checkpoint using several independent simulator
processes in one Slurm allocation. The default requests **4 GPUs, 64 CPU cores,
128 GB host RAM and six hours**, and runs **16 workers** (four per GPU). Each
worker owns its policy, simulator and episode history. There is one parent W&B
evaluation run, with success rate and overall progress; workers do not create
additional W&B runs.

The default is **four validation episodes per task**: 64 episodes for all 16
tasks. These are the first four official validation configurations, including
the two used previously. Episode/noise seeds and video selection are independent
of worker count. The default video cap is **10 total**, spread across tasks.
Missing work and infrastructure errors remain visible in the result; aggregation
does not silently remove them from the planned denominator.

## Submit an evaluation

Pull the implementation on the HPC first. No new site file is needed:
`--site hpc/site_4gpu.env` supplies installation/data paths and online W&B settings. The
launcher defaults to `gpuq` and **untyped `--gres=gpu:4`**, even if an older site
file contains a B200 partition or typed GPU setting. Account, QOS and constraint
settings still come from the supplied site file.

For example, evaluate the already copied frozen-feature checkpoint using eight
executed targets per generated chunk:

```bash
cd /hpc/home/phi/rvalperga/chronotree

.venv/bin/python hpc/evaluate.py \
  --site hpc/site_4gpu.env \
  --checkpoint runs/inspection/eval_20261031T093812Z/frozen_10hz/checkpoint.pt \
  --name frozen_10hz_exec8_p16_e4 \
  --gpus 4 --workers-per-gpu 4 --cpus 64 \
  --episode-count 4 --video-cap 10 --execute-actions 8 \
  --partition gpuq --walltime 06:00:00 \
  --dry-run
```

Inspect `hpc/generated/frozen_10hz_exec8_p16_e4/plan.json` and
`evaluate.sbatch`. Repeat the same command with **`--submit` instead of
`--dry-run`** to submit one allocation. The launcher creates the required
directories; do not manually copy or edit an sbatch file. Omitting either flag
defaults to dry-run.

Use a different `--name` for another checkpoint or experiment. To evaluate all
three snapshots, use the same options for `history_h16`, `history_h32` and
`frozen_10hz`, changing both the checkpoint directory and name. This submits
three independent allocations if `--submit` is supplied for each; Slurm decides
when resources are available.

The checkpoint's saved configuration supplies its architecture, trained inputs,
task set and observation/action cadence. Optional `--config FILE` allows a
compatible evaluation YAML; incompatible model or geometry changes fail.
`--execute-actions 8` is an explicit validation ablation and leaves the trained
prediction horizon unchanged. It means eight native 20 Hz commands (0.4 s) for
the older H16/H32 checkpoints, or eight 10 Hz targets held for two native
commands each (0.8 s) for the frozen-feature 10 Hz checkpoint. Omit this flag to
use the checkpoint's original execution length.

## Snapshot and output layout

The first invocation, including a dry-run, copies and verifies an **immutable
checkpoint snapshot**. A subsequent `--submit` with the same name uses that
snapshot even if the source `last.pt` has advanced. The plan records the exact
checkpoint step, SHA-256, resolved configuration and source-code identity. A new
name takes a new snapshot.

```text
hpc/generated/NAME/
  evaluation.yaml                 resolved configuration
  evaluate.sbatch                 generated single-allocation launch script
  plan.json                       resources, exact inputs and commands
  submitted.json                  most recent submitted Slurm job ID
  logs/stmem_eval_NAME_JOBID.out   supervisor stdout
  logs/stmem_eval_NAME_JOBID.err   supervisor stderr
  stops/JOBID.stop                 allocation-specific stop request, if any

runs/evaluation_inputs/NAME/
  checkpoint.pt                   immutable copied weights
  checkpoint.json                 source path, hash and training step

runs/evaluation/NAME/
  manifest.json                   complete planned episode set and identity
  allocation.json                 physical GPU identities and CPU allocation
  parallel_status.json            worker states and log paths
  summary.json                    combined success, error and missing counts
  episode_index.json              paths to all recorded episode results
  video_index.json                globally selected videos and their paths
  complete.json                   present only after all shards complete
  workers/shardNNN.out            per-worker stdout
  workers/shardNNN.err            per-worker stderr
  shards/shardNNN/episodes/       durable per-episode results
  shards/shardNNN/videos/         actual MP4 files
  videos/                        convenient symlinks to those MP4s
```

`--output PATH` changes the evaluation result directory; the generated plan and
input snapshot retain the locations above. Keep the `shards/` directories when
copying the result because `videos/` links point into them. Alternatively use
`rsync -L` to copy the contents of the video links. The source training directory
and its checkpoints are not modified. The checkpoint's original run root is
preserved in the configuration for W&B grouping, independently of output paths.

Monitor a run with:

```bash
squeue -u "$USER"
tail -n 50 hpc/generated/frozen_10hz_exec8_p16_e4/logs/*.out
cat runs/evaluation/frozen_10hz_exec8_p16_e4/summary.json
cat runs/evaluation/frozen_10hz_exec8_p16_e4/parallel_status.json
```

The supervisor prints progress periodically; W&B updates when completed episode
count changes. `success_rate_so_far_percent` uses recorded outcomes, while
progress uses all 64 planned episodes. The final summary retains per-task and
overall results, missing counts and worker-completion checks. A worker crash
fails the allocation instead of reporting a partial set as a complete result.

## Choosing 16 or 32 workers

Start with the default 16 workers. To test 32, use a **new name** and
`--workers-per-gpu 8`; keep `--gpus 4 --cpus 64 --threads-per-worker 2`.
Increase `--mem`, for example to `256G`, if the combined host-memory requirement
needs it. Use `--exclusive` when you want the allocation isolated from other
jobs on the node. Without it, the launcher requests the stated resources and
does not reserve any additional GPUs the node may have.

Parallelism overlaps independent simulation, rendering and policy inference.
It does not vectorize the policy forward pass across episodes. Each worker
replicates policy state, simulator resources and any frozen visual backbone;
long histories also consume memory independently. Thirty-two workers can add
contention and memory pressure, so elapsed time and memory measurements should
decide which count is faster. No 16× or 32× speedup is assumed.

Physics uses each official task's normal single-environment simulator. The
supervisor partitions the Slurm CPU affinity among workers, bounds their thread
counts, and binds each worker to an allocated GPU by UUID. Policy inference and
SAPIEN rendering must resolve to that same physical GPU; a PCI-identity mismatch
fails immediately. The sbatch allocation and its single `srun` step both request
64 CPUs explicitly. This is separate-process rollout parallelism, with no DDP
or changes to causal history, action timing or task reset behavior.

## Resume interrupted work

After the original allocation has stopped, repeat the **same named launcher
command and options**, replacing `--dry-run` with **`--resume --submit`**.
Completed episodes are reused; interrupted episodes restart with their original
seed. Keep the original input snapshot and result tree. The launcher refuses a
duplicate submission while its recorded job is still active.

Changing the walltime or requested RAM is allowed for a resume. The partition
can also change, provided the worker's recorded software/hardware environment
remains compatible with its existing evaluation manifest.
Changing weights, scientific configuration, code or worker assignment requires
a new name/output to avoid mixing experiments. Fully completed outputs cannot
be resumed. A queued script checks that source code still matches the rendered
plan before launching; finish code updates before rendering and submission.

Slurm's pre-time-limit signal requests a stop through the existing job wrapper.
The supervisor stops worker process groups and retains completed episode
artifacts. Partial episodes are rerun on resume, rather than treated as new
seeds or successful outcomes.

## Validation and limits

CPU tests exercise resource rendering, immutable snapshots, source guards,
explicit submission, safe resume, shell quoting, and single-supervisor launch:

```bash
uv run pytest tests/test_parallel_evaluation_hpc.py
```

Two concurrent real RoboMME processes passed reset, eight simulation steps and
rendering on the local single RTX 5000 Ada. A separate end-to-end check used the
60k frozen-feature checkpoint on two PickXtimes validation scenes, with eight
executed targets and a deliberately short 24-step simulation cap. One-worker
and two-worker runs produced bit-identical action commands for both scenes
(25 commands per scene, including the wrapper's termination behavior), zero
infrastructure errors, one video each, and one parent W&B run each. These short
runs are integration diagnostics, not task-success estimates. Evidence is in
`runs/inspection/parallel_evaluation_smoke_20261031/comparison.json`.

Four-H200 placement, peak memory and full-evaluation throughput require
measurement on the HPC; no Slurm job was submitted as part of implementation.
The supplied defaults are a starting configuration for that measurement.
