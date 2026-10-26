# Cluster execution

Copy `hpc/site.env.example` to the ignored `hpc/site.env`. Fill absolute shared
repository, data, cache and run paths and the Python executable. If modules are
needed, `STMEM_ACTIVATE` names a shell file to source. The site file itself uses
literal `STMEM_NAME=value` assignments and the four W&B settings documented below;
the planner deliberately does not execute shell substitutions. Set the site's
partition, optional account/QoS/constraint, GPU resource name, memory and walltime.
`gpuq` is only the supplied partition default and does not select H200 hardware.

```bash
cp hpc/site.env.example hpc/site.env
# Edit hpc/site.env with actual site values.
uv run python hpc/plan.py --plan core_seed0 --dry-run
```

The default plan is one seed, all 16 tasks, six methods and validation only.
The renderer writes concrete commands, configuration files, dependencies,
expected inputs/outputs and literal `#SBATCH` resources under `hpc/generated/`.
Inspect its `plan.json` before choosing to submit. Cost is **unknown** until actual
timings exist. Requested walltime is a limit, not an estimate of GPU hours.

```bash
# Only this explicit flag submits allocations.
uv run python hpc/plan.py --plan core_seed0 --submit
```

`hpc/logs` is created by the planner before `sbatch`, because Slurm opens its log
files before starting the shell. Direct submission of a template in `hpc/jobs`
requires running from the repository root and creating that directory first.
The templates carry one-GPU defaults. Site variables in shell code cannot change
`#SBATCH` comments; use the planner or literal `sbatch` resource overrides.

The scheduler uses `afterok` for scientific dependencies and at most two allocation
lanes by default, controlled by `STMEM_MAX_CONCURRENT`. Each lane has an `afterany`
predecessor solely to bound concurrency. The bound includes CPU selection/report
jobs and GPU jobs. No automatic submission retry or self-resubmission occurs.
Submitted job IDs are persisted after every successful `sbatch --parsable` call.

## Directory layout

Keep manually maintained cluster settings directly under `hpc/`. The default is
`hpc/site.env`; optional local variants such as `hpc/site_4gpu.env` are selected
with `--site`. The default file and new `hpc/site_*.env` variants are ignored by
Git; the explicitly tracked `hpc/site_4gpu.env` and `hpc/site_b200_4gpu.env`
contain this cluster's H200 and B200 four-GPU settings respectively.
`hpc/site.env.example` remains the portable template.

```text
hpc/
  site.env.example              tracked settings template
  site.env                      local default settings
  site_4gpu.env                  tracked four-H200 cluster settings
  site_b200_4gpu.env             tracked four-B200 cluster settings
  plan.py                       planner
  jobs/                         maintained job templates
  generated/
    core_seed0_validation_h200_1gpu/
      plan.json                 rendered commands, resources and dependencies
      configs/                  generated stage YAML files
      jobs/                     generated submit-ready .sbatch files
      submitted.json            job IDs and signatures, created on submission
      stops/                    allocation-specific stop requests
    core_seed0_validation_h200_4gpu/
      ...                       same structure for the four-GPU plan
    core_seed0_validation_h200_4gpu_100k/
      ...                       separate plan state for the new 100k teacher run
  logs/                         Slurm stdout/stderr, named with job IDs
```

`hpc/generated/` holds rendered plans and their execution state. It should not
hold manually maintained site settings. Dry-run and submission regenerate
`plan.json`, stage configurations and scripts, so edit their source configs or
site settings instead of editing the generated files. Preserve a plan directory
while its jobs are active or you need its submission/accounting history; in
particular, `submitted.json` and stop files are execution state, not source
configuration. `--output-dir` is an optional override for additional independent
plans; the default `<plan>_<phase>_<profile>` directories suffice for a one-GPU
versus four-GPU comparison.

Training checkpoints and metrics go under `$STMEM_RUN_ROOT/seed<seed>/<stage>/`;
W&B files default there too unless a separate W&B directory is configured.
These training artifacts are separate from `hpc/generated/`. Changing the GPU
profile creates a different default plan directory but does not change the run
root. Use a different `STMEM_RUN_ROOT` for an independent comparison to isolate
its checkpoints, W&B identity and stage markers. Prepared data stays under the
configured data root.

## Launching the 100k teacher run

Use the established `hpc/plan.py` workflow. Training settings live in
`configs/teacher.yaml`, candidate checkpoints in the experiment manifests, and
cluster resources and output paths in `hpc/site_4gpu.env`. The teacher uses:

| Setting | Value |
|---|---|
| Optimizer updates | 100,000 |
| Global effective batch | 256 decisions |
| Configured microbatch cap | 64 decisions per GPU |
| Actual microbatch on four GPUs | 64 decisions per GPU |
| Forward/backward passes per GPU per update | 1 |
| Encoder blocks per call | 128 |
| Immutable checkpoint steps | 50,000 and 100,000 |

The actual microbatch is `min(microbatch, effective_batch / GPU_count)`.
Four GPUs therefore process `4 × 64 = 256` decisions per optimizer update.
This is 25.6 million sampled decisions over the full run, eight times the exposure
of the previous 50,000-update/global-64 configuration. The inherited learning rate is
unchanged; the decay schedule uses the new 100,000-update budget. All 16 tasks,
full causal histories, verified alignment checks, prepared data and eight loading
workers per rank are retained. Larger batches and block groups increase peak
memory; the previous smaller-batch profile does not establish this run's peak.

The tracked site file sets
`STMEM_RUN_ROOT=/hpc/home/phi/rvalperga/chronotree/runs/teacher_b128_m64_blocks128_100k`.
Use that fresh path for this run; if a local site edit was preserved during
syncing, update that assignment before rendering. Data and cache paths stay the
same. The explicit `--output-dir` below keeps this run's generated plan and
submission records separate from previous four-GPU jobs.

After syncing these source files to the HPC, run:

```bash
cd /hpc/home/phi/rvalperga/chronotree
.venv/bin/python hpc/plan.py \
  --site hpc/site_4gpu.env --plan core_seed0 --profile h200_4gpu \
  --only seed0_teacher \
  --output-dir hpc/generated/core_seed0_validation_h200_4gpu_100k \
  --log-every 1 --progress-every-seconds 30 \
  --dry-run
```

Inspect that directory's `plan.json`, `configs/seed0_teacher.yaml` and
`jobs/seed0_teacher.sbatch`. The generated teacher config extends
`configs/teacher.yaml`. Verify one selected job, four GPUs, 64 CPUs, the new run
root, and expected checkpoints `step_00050000.pt` and `step_00100000.pt`.
Then repeat the same arguments with `--submit`:

```bash
.venv/bin/python hpc/plan.py \
  --site hpc/site_4gpu.env --plan core_seed0 --profile h200_4gpu \
  --only seed0_teacher \
  --output-dir hpc/generated/core_seed0_validation_h200_4gpu_100k \
  --log-every 1 --progress-every-seconds 30 \
  --submit
```

This submits one ordinary training allocation, with per-update metrics and
30-second progress messages through the explicit logging flags. W&B settings come
from the site settings; system metrics and the configured checkpoint point-cloud
scenes remain enabled. Its checkpoints and W&B files go under
`runs/teacher_b128_m64_blocks128_100k/seed0/teacher/`. `last.pt` is refreshed every
1,000 updates and on a coordinated stop; training writes `complete.json` only
after completing the requested budget. The planner retains the normal dependency,
signature, submission-record and resume workflow. `--only seed0_teacher` selects
just the teacher allocation; validation selection and downstream stages remain
available through the same plan.

Use the same site file and `--output-dir` when later resuming or selecting other
stages from this run. Omit `--resume` for the first submission. Do not submit the
same fresh run twice or reuse a previous run's output directory. Existing active
jobs keep their separate artifacts. Queued jobs still read their configuration
files when they start, so source configuration edits also affect queued jobs
that extend those files. No interactive-shell export of `STMEM_PYTHON` is needed:
the planner reads the site file directly.

### Four B200s with the balanced dataset

The user confirmed on 22 October 2026 that the cluster advertises
`aiq gpu:b200:7/8`. `hpc/site_b200_4gpu.env` therefore selects partition `aiq`
and GPU type `b200`, with profile `b200_4gpu`. The rendered teacher requests
`--gres=gpu:b200:4`, one node, one Slurm task and four torchrun workers.
The profile controls the worker count; the partition and typed GRES select the
hardware. CPU and host-memory requests remain 64 cores and 128 GB. The inherited
48-hour limit has not been checked against this partition's scheduling policy.

The first supplied failure traceback showed `cudaErrorDevicesUnavailable` at
`torch.cuda.set_device(local)`, before model/data initialization, and a standalone
probe reproduced it on `ainode01`. A subsequent training retry reached roughly
600 updates on `ainode04` before a genuine GPU OOM in the decoder's K/V projection:
162.76 GiB allocated by PyTorch, 13.32 GiB reserved but unused, and a failed
702 MiB request. The earlier device-access failure and this training OOM are
separate problems. The latter run followed the 16/32 retry instructions; its exact
resolved config and per-update memory trace were not supplied.

The batched visual encoder now ranks candidate neighbors without autograd and
computes differentiable Gaussian distances only for the selected neighbors. This
removes dense candidate-distance tensors retained solely for bandwidth gradients,
without changing points, centers, neighborhoods, history coverage or the objective.
The measured local reduction and its scope are recorded in `docs/status.md`.
Long histories and other saved encoder activations still require headroom. Use
`configs/teacher_b200.yaml` for the retry: it extends `teacher.yaml` with
microbatch 8 and encoder block batch 16, retaining effective batch 256 and
100,000 updates. Each GPU accumulates eight passes of 8 decisions per update.
This reduces per-pass activations and temporary encoder tensors; without
activation checkpointing, saved activations across encoder chunks still remain
until backward. Prepared geometry remains `prepared_balanced_v2/`, 2,048 points
and 32 centers per frame; no new preparation is needed.

After pushing and pulling these changes, render the job:

```bash
cd /hpc/home/phi/rvalperga/chronotree
# Select current code, even if this shell previously evaluated an old checkout.
export PYTHONPATH="$PWD/src"
# Optional fragmentation mitigation for variable history lengths; not a memory-fit guarantee.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python hpc/plan.py \
  --site hpc/site_b200_4gpu.env --profile b200_4gpu \
  --teacher-config configs/teacher_b200.yaml \
  --plan core_seed0 --only seed0_teacher \
  --run-root runs/teacher_balanced_v2_b200_b256_m8_blocks16_memoryfix_100k \
  --output-dir hpc/generated/teacher_balanced_v2_b200_b256_m8_blocks16_memoryfix_100k \
  --log-every 1 --progress-every-seconds 30 \
  --dry-run
```

Inspect `plan.json` and `jobs/seed0_teacher.sbatch` in the generated directory,
then repeat exactly the same command with `--submit` instead of `--dry-run`.
The explicit run-root override chooses a fresh directory:
`runs/teacher_balanced_v2_b200_b256_m8_blocks16_memoryfix_100k/seed0/teacher/`.
The existing H200 site file and run roots remain separate. W&B is online in the
same `chronotree` project, including system metrics. The failed 64/128 and 16/32
runs' artifacts stay in their previous directories. No resume is requested for the retry.
If device initialization fails again, inspect GPU visibility and the node before
tuning batch sizes. Preserve Slurm's `CUDA_VISIBLE_DEVICES`; no binding or driver
workaround is established. The smaller batch and encoder fix have not been
benchmarked on full-history B200 training; no automatic OOM reduction exists.
The default rolling checkpoint interval is 1,000 updates, so a failure near 600
may have no `last.pt` to resume. Check for that file before requesting resume.
The existing peak-memory metrics are cumulative high-water marks, not memory
still held after each update; a rising peak alone does not establish a leak.

### Teacher learning-rate comparison

`configs/teacher_lr001.yaml` extends `teacher.yaml` and overrides only
`training.lr: 0.01`. It retains the 1,000-update warmup and the same cosine
multiplier over the 100,000-update schedule. The maximum is 0.01 and the final
learning rate is 0.001 (10% of the maximum). Batch sizes, checkpoint intervals,
model and data settings are inherited unchanged.

Select the variant through the same planner and override its run root explicitly:

```bash
.venv/bin/python hpc/plan.py \
  --site hpc/site_4gpu.env --plan core_seed0 --profile h200_4gpu \
  --teacher-config configs/teacher_lr001.yaml \
  --run-root runs/teacher_lr001 \
  --only seed0_teacher \
  --output-dir hpc/generated/core_seed0_validation_h200_4gpu_lr001 \
  --log-every 1 --progress-every-seconds 30 \
  --dry-run
```

Inspect the rendered plan, then repeat the same arguments with `--submit`.
`--teacher-config` selects the teacher source YAML; `--run-root` overrides the
site's output root for this invocation, including exported job variables and
stage markers. Relative values for both flags resolve against `STMEM_REPO_ROOT`.
This writes to `runs/teacher_lr001/seed0/teacher/`, giving the comparison separate
checkpoints and W&B identity. The baseline source configuration, site file and
previous generated plan are not edited.

Keep the same `--teacher-config`, `--run-root` and `--output-dir` on subsequent
resume or downstream-stage commands for this experiment. Teacher validation and
cache jobs use the selected YAML; compact/router/Perceiver training settings
remain their own. Protocol freeze records the selected teacher source and its
inheritance hashes; official-test planning rejects a different teacher source.

## Parallel data preparation

Run full preparation in a compute allocation. It uses CPU point-cloud geometry,
hashing, HDF5 reads and compression; the GPU does not accelerate this path. Tasks
can be prepared independently by spawned worker processes, each opening its own
raw HDF5 file and writing a separate prepared task file. At most 16 tasks can run
concurrently. This is preprocessing parallelism, not distributed model training.
The number of GPUs in `STMEM_PROFILE` does not set the preparation worker count.

`hpc/jobs/prepare.sbatch` is supplied for the GPU-only `gpuq` queue. Its literal
defaults request one node, one Slurm task, one GPU, 16 CPUs, 64 GB RAM and a 24-hour
time limit. Eight single-threaded CPU workers leave room within that CPU allocation
for the parent process and I/O. The memory request provides headroom for several
Torch processes and their frame-processing buffers; it is not a measured peak or
an assurance for every task. The allocated GPU remains unused and still counts
toward the site's allocation charges.

From the repository root, after configuring `hpc/site.env` and staging the raw
archives, extracted HDF5 files and download integrity manifests:

```bash
mkdir -p hpc/logs
# This command submits an allocation:
sbatch hpc/jobs/prepare.sbatch
```

The template sources `hpc/jobs/common.sh` to load the site paths, Python executable
and optional activation file. It launches Python through `srun` directly; the
training stop-file wrapper is not used. Logs go to
`hpc/logs/stmem_prepare_JOBID.out` and `.err`. It prepares 100 episodes per task
with `configs/teacher.yaml` and enables `--skip-existing`.

The approved balanced preparation writes
`$STMEM_DATA_ROOT/prepared_balanced_v2/<task>.h5`, selected by
`data.prepared_subdir` in `configs/base.yaml`. Existing `prepared/` files are
retained; no re-download is required. The 2,048-point sampler reserves 75% for
geometry above z=5 mm, balancing those points across 10 cm XY cells. Preparation
also stores the 32 spatially balanced center indices per frame, avoiding repeated
clustering during training. These settings and implementation versions enter the
prepared manifest; stale files cannot satisfy the new training configuration.
Changing prepared format requires a different output directory, even when an old
configuration or `--prepared-subdir` points at an existing file.

After pushing the code and pulling it on the HPC, the full migration can be run
without creating another site file:

```bash
# In the repository root, load the existing site paths and Python executable.
export STMEM_SITE_FILE="$PWD/hpc/site.env"
source hpc/jobs/common.sh
mkdir -p hpc/logs
# Reserve one GPU because this site has a GPU-only queue; conversion uses CPUs.
# At most 16 workers are useful because there are 16 independent task files.
STMEM_PREP_WORKERS=16 sbatch --cpus-per-task=16 --mem=64G --time=24:00:00 \
  hpc/jobs/prepare.sbatch --config configs/teacher.yaml
```

If shared-storage throughput is worse with 16 workers, use eight; 64 allocated
cores do not create more than 16 task workers. This CPU implementation uses one
thread per worker. The runtime ETA now includes clustering as well as RGB-D
conversion. An assignment to `STMEM_PREP_WORKERS` in the site file takes
precedence over the submission environment.

Wait for `preparation_complete` and 16 finalized `.h5` files before launching
the teacher. The raw dataset and alignment convention are unchanged, so the
existing verified revision-level `alignment.json` remains applicable. Use a
fresh run root and regenerate training jobs with `hpc/plan.py` afterwards;
existing generated YAML snapshots retain their previous settings. The much
larger leaf count needs a new memory measurement before reusing the previous
microbatch size. No new full-data training throughput is claimed.

`STMEM_PREP_WORKERS` overrides the worker count, subject to the allocated CPU count
and the 16-task maximum. Set it in `hpc/site.env` or the submission environment;
an assignment in the site file takes precedence. Without an override, the default
is the smallest of eight, the allocated CPU count and 16. More workers compete
for shared storage bandwidth, so start with four to eight and measure progress.
Eight workers do not imply an eightfold speedup. Each worker uses one Torch thread,
and the template sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
and `NUMEXPR_NUM_THREADS` to one before Python starts.

The batch defaults do not change when shell code reads `STMEM_MEM` or
`STMEM_WALLTIME`. Override resources explicitly, within the site's limits:

```bash
# Example for a site that permits a 48-hour allocation:
STMEM_PREP_WORKERS=4 sbatch --cpus-per-task=16 --mem=64G --time=48:00:00 \
  hpc/jobs/prepare.sbatch
```

Supply `--account`, `--qos`, `--partition` or a typed `--gres` to `sbatch` when
required by the site. A CPU partition, if available, can avoid reserving an unused
GPU; this particular template is intended for the supplied GPU-only queue. Use
`--tasks VideoUnmask` after the script name for a selected task. `STMEM_CONFIG`
can choose another preparation configuration; the worker controls remain fixed
by the template so optional CLI arguments cannot oversubscribe the allocation.

If already inside a compute allocation with at least eight CPUs, no second batch
job is needed:

```bash
source hpc/jobs/common.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$STMEM_PYTHON" -u -m stmem.data.prepare --config configs/teacher.yaml --episodes 100 \
  --workers 8 --threads-per-worker 1 --skip-existing
```

Use fewer workers for a smaller CPU allocation. Outside this template the CLI
defaults to one worker and one thread per worker. `--workers` parallelizes tasks,
so selecting one task still runs only one worker for that task. Preparation
continues to use the configured point count and the complete causal observation
data; worker count does not reduce data or change preprocessing geometry.

To move a foreground preparation process off a login node, interrupt it once
with Ctrl-C and wait for it to exit before starting another process on the same
outputs. Raw data and finalized task files remain available. A current
`<task>.partial.h5` is not a completed task and cannot resume at its last episode;
that task restarts from the beginning. With `--skip-existing`, completed files are
reused only after checking a fresh raw-file hash, the preprocessing configuration,
the exact episode count and the prepared layout. Merely finding an output file is
insufficient. The 256-point/two-episode debug subset therefore does not satisfy a
2,048-point/100-episode request. Keep interrupted-task files out of training until
preparation completes, and do not run independent writers on the same task output.

Every task freshly hashes the raw file, even when checking a completed output.
The `raw_integrity` record reports whether that digest was also checked against
the download manifest. Current downloads include the extracted-file checksum;
legacy manifests without it remain supported and explicitly report
`download_manifest_verified: false`. Recording a new digest alone does not prove
archive membership. Reuse checks compare the fresh digest with the completed
file's embedded manifest and inspect metadata/layout and small chronology arrays;
they do not checksum every prepared point-cloud/action chunk for storage corruption.

After checking the input paths and manifests, the parent immediately prints a
`preparation_started` JSON record with the active worker count and CPU budget.
Each worker reports `task_started` before hashing its raw file. Hashing and frame
conversion report progress approximately every ten seconds while processing
continues, with additional records at hash and episode completion. Only the
parent prints these JSON records, so concurrent workers do not interleave them.

Frame records include `frames_done`, `frames_total`, `frames_per_second` and
`eta_seconds`. This ETA estimates the remaining frame conversion for that task
from its average rate so far; it excludes hashing and queued tasks and is not a
whole-run estimate. Episode lengths and shared filesystem load vary, so early
estimates can change substantially. Use these records and each task's final
`prepared_balanced_v2/<task>.manifest.json` to assess throughput. The final
`preparation_complete` record reports total elapsed time and the number of
verified files skipped; no measured full-data preparation ETA is claimed.

## Weights & Biases tracking

Tracking is enabled in `configs/base.yaml`, including the inherited debug configs.
The default `offline` mode records W&B run data locally without credentials or
network access. Training logs scalar metrics at the existing configured reporting
cadence; evaluation logs success and progress. Only rank zero opens the
training run under DDP. W&B system metrics are enabled by default through
`tracking.system_stats: true`, which sets `x_disable_stats=False` and also enables
system metadata. The **System** tab shows GPU utilization, allocated memory,
power, temperature and clocks, plus CPU, RAM, disk and network metrics. The
installed SDK samples these in its background service every 15 seconds. Under
DDP, that single service monitors GPUs visible to rank zero, normally all GPUs
in the single-node allocation; it does not create four duplicate W&B runs.
`tracking.system_stats: false` is an explicit opt-out. Gradient watching and
console capture remain disabled. The existing local
checkpoints, metrics and per-episode results remain the durable experiment records.
Model weights, videos and source files are not uploaded automatically. The SDK's
[settings reference](https://docs.wandb.ai/models/ref/python/experiments/settings)
documents the collection controls.

Evaluation has two fixed chart metrics, independent of the number of tasks:

- `evaluation/success_rate_so_far_percent`: successes / recorded episodes, in
  percent. Failures, policy/infrastructure errors and timeouts stay in the
  denominator. No success estimate is logged before the first outcome.
- `evaluation/progress_percent`: recorded / planned episodes in this shard,
  in percent. An empty shard has no work remaining and reports 100% progress,
  with no success estimate.

The initial update includes all saved episodes on resume. Further updates follow
each atomic episode-result write, outside inference timing. The run summary
keeps recorded/planned counts, errors, timeouts and the last episode outcome.
Bookkeeping axes are hidden from automatic evaluation plots. System monitoring
remains enabled, and training logging is unchanged.

This running success rate is for monitoring; it uses the completed episode mix,
which can change while tasks run sequentially. The scientific task-macro score
with fixed planned denominators, per-task/suite outcomes and missing counts
remains in `summary.json`. Full latency/storage/cost aggregates are written to
`diagnostics.json`; raw decisions, commands and errors remain in `episodes/*.json`.
Aggregate research reports still have access to all these measurements.

The logging change applies when a new Python evaluation process starts. It does
not remove existing W&B history or panels. Evaluation source hashes remain
strict: after updating code, use a new `--output` directory for any new rollout
invocation; an old output directory cannot be resumed under changed source.

`hpc/site.env` accepts these nonsecret settings:

| Setting | Default in the example | Meaning |
|---|---|---|
| `WANDB_MODE` | `offline` | `offline` writes locally; `online` sends metrics to W&B; `disabled` opts out. |
| `WANDB_PROJECT` | `chronotree` | W&B project for the run. |
| `WANDB_ENTITY` | empty | Optional account/team; empty uses the tracker/account default. |
| `WANDB_DIR` | empty | Optional absolute durable directory for W&B files; empty uses the stage/evaluation output directory. |

These values override the tracking configuration. The planner imports these four
variables from its environment and then applies assignments from the site file,
so a value written in `hpc/site.env` takes precedence. Empty W&B site assignments
clear inherited values. All four settings are explicitly carried into the
generated batch script. A path containing spaces needs quotes; `$HOME`, `${...}`,
command substitutions and `~` expansion are not supported in the site file.

For live dashboards, authenticate once from a machine that can reach W&B, using
the account that will run the jobs, then update the nonsecret site settings:

```bash
uv run wandb login
```

```bash
# hpc/site.env
WANDB_MODE=online
WANDB_PROJECT=chronotree
WANDB_ENTITY=your-account-or-team
```

The compute node must be able to reach W&B and access that account's login
credentials. An existing virtual environment is sufficient; W&B login is separate
from `STMEM_ACTIVATE`. Do not put API keys in the site file: `WANDB_API_KEY` and
other unapproved W&B assignments are rejected, and ambient credentials are never
copied into generated script text or plan JSON. Authentication stays in W&B's
credential store or the site's external credential mechanism.

Teacher, compact, tree-router, flat-router and Perceiver training retain separate
run identities. Evaluations are separate runs for their method, split and shard.
Training retains the experiment/seed group `<run-root-name>-<path-hash>-seed<N>`.
New closed-loop evaluation runs use `evaluation-<run-root-name>-<path-hash>-seed<N>`
in the same W&B project. The original `job_type=evaluation` field and tag also
remain available for filtering, including on older runs. This metadata change
does not reorganize already-running or historical W&B runs or their panels.
Each output directory has a `tracking.json` record mapping local work to its W&B
run ID and latest local W&B directory. The ID survives a restart; online runs
resume with that ID, while each offline invocation writes a separate segment.
The local `tracking_history.jsonl` journal preserves increasing W&B history indices
across restarts, with optimizer/episode progress logged separately as `progress/step`.
Keep the journal alongside the checkpoints when resuming. An existing run's
project/entity cannot be changed in place; use a new output directory for a new
destination. Logging settings do not select checkpoints, change scientific seeds,
or alter Slurm completion dependencies.

The default `evaluation.video_cap: 2` records at most the first two planned
episodes **per evaluation shard**, rather than two per task. For the standard
unsharded two-episodes-per-task run, these are `videos/BinFill_000.mp4` and
`videos/BinFill_001.mp4` inside the evaluation output directory. Simulator failures
can leave fewer recordings. Resuming skips saved episodes without recreating
missing videos or adding another two; recordings are not uploaded automatically.

For nodes without network access, leave `WANDB_MODE=offline`. After the allocation,
sync a chosen offline directory from a machine with the files and Internet access:

```bash
uv run wandb login
uv run wandb sync /absolute/path/to/wandb/offline-run-TIMESTAMP-RUNID-attempt1
# For later segments of the same ID, sync in chronological order:
uv run wandb sync --append /absolute/path/to/wandb/offline-run-TIMESTAMP-RUNID-attempt2
```

Use the actual directory paths from `tracking.json` and the selected W&B storage
root. Retain all offline directories from resumed work until they have been synced.
Keep the attempt's `files/media` directory alongside its `.wandb` file: scene and
table references in the history require those files during sync. Copying only the
`.wandb` file is insufficient for media.
The installed `uv run wandb sync --help` and W&B's
[sync command reference](https://docs.wandb.ai/models/ref/cli/wandb-sync)
describe the upload command. No login, online upload or sync is performed
automatically by the planner.

### Training startup

Teacher, compact and Perceiver startup reads episode headers and the small
`source`, `actions`, and `action_valid` arrays. One pass builds the deterministic
decision index and gathers action-normalization counts/sums/squared sums. The
normalizer consumes those statistics without a second data scan. Point clouds,
point-validity arrays and proprioception payloads are loaded only when a training
example is requested. Their shapes and validity dtypes are still checked from
HDF5 headers during startup; alignment, preprocessing, episode-count and memory
length checks remain mandatory. The prepared file format is unchanged.

This removes the previous two full-episode startup passes. It does not preload
the complete dataset or change per-example loading during training. Every launch
still performs the small action/header scan, including resume; no persistent
index cache can go stale. The local two-episode measurement and exact parity
checks are in [the startup evidence](evidence/training_startup.json); cluster
filesystem timing still needs to be measured on the full dataset.

Rank zero emits flushed JSON events with `event: training_startup`, including
catalog start, per-task index progress, normalization, and `training_ready`.
W&B summaries show `startup/phase`, episode progress and elapsed time. Once the
checkpoint position is known, startup durations are logged at that optimizer
step (zero for a fresh run). `startup/data_seconds` includes gathering the
normalization statistics; `startup/normalizer_seconds` measures only applying
them to the model. `startup/total_seconds` includes data/model/device setup and
resume loading, but excludes the earlier W&B initialization.

Policy loss/speed/memory metrics are emitted after the first completed optimizer
update of each launch, then at the configured cadence (every 50 updates by
default), final completion and a requested partial stop. This introduces no
per-example W&B calls or extra point-cloud previews.

Prepared data does not need to be regenerated for either the startup improvement
or batched training. A running process cannot pick up source changes. The batching
upgrade changes training configuration and sampler ordering, so start it in a
fresh run directory using the [launch instructions below](#batched-policy-training).
Do not use the older scalar trainer's checkpoints for that new experiment.

### Diagnosing slow training

Both the planner and `python -m stmem.train` accept runtime logging overrides:

```bash
.venv/bin/python hpc/plan.py --plan core_seed0 --profile h200_1gpu \
  --only seed0_teacher \
  --log-every 1 --progress-every-seconds 30 --dry-run
```

`--log-every 1` sends scalar metrics after every completed optimizer update.
For policies, `train/step_seconds` measures the latest update: sample loading,
history encoding, forward/backward passes, optimizer work and existing control
synchronization. It excludes checkpoint writes, metric reporting and point-cloud
diagnostics. `train/data_wait_seconds_rank` measures how long the reporting rank
waits for packed CPU microbatches; `train/data_loading_seconds_rank` is retained
as an alias. With `training.workers: 0`, this includes synchronous reading and
packing. With workers enabled, it measures exposed wait after background work
overlaps previous GPU work. Frozen-cache reads remain outside this counter.
`train/data_worker_load_seconds_rank` and `train/data_worker_pack_seconds_rank`
sum reading and packing durations for consumed batches. Those durations can
overlap each other and GPU work; do not add them to update time.
`train/data_prefetch_workers_rank` and `train/data_prefetch_capacity_rank`
record the actual process count and outstanding microbatch bound.
The cumulative `train/examples_per_second` includes
overhead from previous checkpoints and diagnostics. Under DDP, per-update timing
and heartbeat fields describe rank zero, not a reduction of all ranks' timings.

`--progress-every-seconds 30` enables a small background thread that writes a
flushed JSON heartbeat to stdout and `<stage>/progress.jsonl` even while the main
thread is inside a long operation. It reports `phase` (for example,
`loading_batch`, `encoding_history`, `forward`, `backward` or `checkpoint`),
`optimizer_steps_completed`, the update in progress, completed examples on this
rank, and elapsed phase/update time. With the teacher's microbatch cap of 64,
completed example counts advance after each batched backward pass: 64 and 128 on
one GPU, or 32 on each rank for a four-GPU update. A teacher step is one optimizer
update over 128 examples globally. During batch
encoding, multiple examples and their historical blocks are being processed.
The `batch_tasks` field identifies tasks in the current microbatch. Completed
update metrics include `train/microbatch_size_rank`,
`train/microbatches_per_update_rank`, `train/history_frames_mean_rank`,
`train/history_frames_max_rank`, `train/history_leaves_per_update_rank` and
`train/history_context_tokens_mean_rank`. Compare these alongside step time:
full-history computation still varies with the sampled decisions' history lengths.

Heartbeat timings describe where the host is working or waiting; they are not
synchronized CUDA kernel timings. Heartbeats make no tensor reads, CUDA calls,
collectives, RNG operations or W&B calls. They do small CPU/file writes, so their
overhead is not literally zero. W&B receives completed-update metrics through
the main thread; a heartbeat does not invent a completed step or partial loss.
Watch ongoing work with:

```bash
tail -f "$STMEM_RUN_ROOT/seed0/teacher/progress.jsonl"
```

Without flags, scalar cadence comes from `training.log_every` and periodic
heartbeats are disabled. Both intervals must be positive (seconds may be
fractional). Effective settings are recorded separately in the manifest,
checkpoint and W&B metadata; generated scientific YAML and checkpoint provenance
are unchanged. Router training supports `--log-every` but preserves its original
full-validation cadence. The planner passes heartbeats only to teacher, compact
and Perceiver jobs.

Existing processes cannot pick up these flags. For a later restart of the same
batched configuration, use `--resume` and retain its run root. To preserve progress
from a running generated job, create the exact `STMEM_STOP_FILE` printed in its
generated `.sbatch` script; its allocation-specific filename has this form:

```bash
touch hpc/generated/PLAN_DIRECTORY/stops/seed0_teacher_JOB_ID.stop
```

The request takes effect at the next completed optimizer update, when the job
writes `last.pt` and exits with the intentional partial-run code 75. It can take
time if the current update is slow. Wait for the job to exit and verify that
`$STMEM_RUN_ROOT/seed0/teacher/last.pt` exists. Then copy the updated source to the
cluster, generate a plan with `--resume`, and replace `--dry-run` with `--submit`
to launch. Keep the original run root; never run two writers in that directory.
The planner retains already active allocations when invoked with `--resume`;
it cannot change their logging options in place. No data re-preparation is needed.

### Training point clouds and action comparisons

Teacher, compact and Perceiver policy training reuse two fixed early training
decisions, from distinct episodes when available, to show how predictions change
at scheduled checkpoints. The teacher uses FULL context; the compact diagnostic
uses UNIFORM context at the configured main budget; Perceiver uses its resampled
context. These examples do not select checkpoints or replace the planned
closed-loop validation. Router fitting does not run these action-policy scenes.
Official validation/test episodes have no recorded expert action chunks, so they
do not receive a predicted-versus-ground-truth comparison.

The defaults inherited from `configs/base.yaml` are:

```yaml
tracking:
  scenes:
    enabled: true
    count: 2
    max_points: 1024
```

`count` accepts one or two fixed examples; `max_points` accepts 1–4,096 observed
points per cloud, with small target-path overlays added separately. Set
`enabled: false` to disable scene collection while retaining scalar metrics. Scenes follow
`training.checkpoint_every` (1,000 updates in the core config), declared candidate
checkpoint steps and the final update. There is no separate scene frequency or
extra first-update preview. A partial-stop checkpoint skips scene collection so
the remaining allocation time is available for saving training state.

W&B panels `diagnostics/scene_0` and `diagnostics/scene_1` contain interactive
[Object3D point clouds](https://docs.wandb.ai/models/ref/python/data-types/object3d).
They show only the current observed RGB-D cloud in world coordinates. Green
marks predicted tool-center targets, orange marks recorded targets, and magenta
marks the observed current tool center. The paths come from the pinned Panda
forward kinematics applied to joint commands; dotted connections show command
order, not simulated motion, contact, or task success. Joint angles are not
treated as Cartesian coordinates. Predicted joint targets use the same clipping
as deployment; clipping counts appear in the scene caption and metadata, and raw
predictions remain in the action table. Gripper commands are shown in the accompanying
`diagnostics/actions_0` and `diagnostics/actions_1` tables, alongside raw/deployed
arm commands, recorded targets and validity masks. Missing or padded targets are
excluded from the path. The overlay checks its coordinate alignment against the
observed robot state and skips inconsistent examples with a reason. Synthetic
examples are explicitly skipped because their random proprioception does not
describe a physically consistent robot pose.

Only rank zero prepares scenes and predicts actions, using the unwrapped policy
with gradients disabled and restoring its training modes and RNG state. Other
ranks can wait at the next normal DDP synchronization. Additional inference,
CPU conversion and media serialization have a cost: inspect
`diagnostics/scenes_seconds` and `diagnostics/scenes_bytes` for each scene event.
The scalar logging measurements do not measure this additional work. Scene media
uses the existing W&B run and storage location; its metadata appears in
`tracking_history.jsonl`, and the SDK retains the files under that attempt's
`files/media` directory in offline mode as well.

The final `training/seconds_including_diagnostics` and
`training/allocated_gpu_hours_including_diagnostics` summaries include the final
scene event. Scene totals, like training elapsed time, cover the current process
attempt. A local two-update, 256-point real-data diagnostic measured 0.174–0.339 s
and approximately 201 KB per **pair** of scenes plus action tables. This is a
small debug-model measurement, not a full-size/H200 or online-upload estimate.
The core checkpoint interval is 1,000 updates; measure its actual cost on the
cluster using the emitted counters. Offline media decoding was tested; live
dashboard delivery was not tested against an authenticated W&B account.

For manual inspection of the encoder's input geometry, pooling neighborhoods
and binary memory structure, use the separate
[offline HTML inspector](visualization.md). It exports a selected prepared
episode on CPU, supports optional checkpoint bandwidths, and requires only that
task's final `.h5` file and the verified alignment record. Install the optional
`visualization` extra while retaining `--extra simulator`; no Slurm submission
or training change is part of the export.

## Profiles and distributed training

| Profile | GPU workers per single-node allocation | Intended use |
|---|---:|---|
| `a5000_debug` | 1 local process | Explicit small configs/debug shapes; not a core Slurm plan |
| `h200_1gpu` | 1 | Default experimental profile |
| `h200_4gpu` | 4 | Optional measured-throughput comparison |
| `b200_1gpu` | 1 | Explicit opt-in |
| `b200_4gpu` | 4 | Explicit opt-in; tracked B200 site selects `aiq`, `gpu:b200:4` |
| `b200_8gpu` | 8 | Explicit opt-in |

Choose `--profile h200_4gpu` to render DDP training. One Slurm task runs **one**
torchrun agent, with `--nnodes=1 --nproc-per-node=4`. Evaluation shards each request
one GPU and use independent processes. Four replicas do not pool memory; use the
configured microbatch/accumulation and block checkpointing to address a teacher
replica OOM. Block checkpointing is available but disabled by default. The
effective batch is fixed by the scientific config and must remain
divisible by the world size. Do not alter it silently for throughput.

### Batched policy training

Teacher, compact and Perceiver training use genuine batches of decision examples.
Each example contains its current observation, available conditioning source,
strictly past execution history and target action chunk. It is not a complete
episode presented as one target. Current geometry, text, history blocks and the
action decoder have batched paths; variable context lengths are padded with
explicit validity masks. Padding cannot create occupied visual leaves or let one
example attend to another. Partial blocks retain only their observed prefix;
blocks never cross an example or source boundary. The full teacher history stays
available even when examples have different history lengths.

Teacher settings in `configs/teacher.yaml`, together with its inherited defaults,
are:

```yaml
model:
  checkpoint_blocks: false
training:
  steps: 100000
  effective_batch: 128
  microbatch: 64
  encoder_block_batch_size: 128
  checkpoint_steps: [50000, 100000]
  length_bucket_size: 1
  workers: 8
  prefetch_factor: 1
tracking:
  system_stats: true
```

Compact and Perceiver stages retain the base settings of effective batch 64,
microbatch cap 16 and encoder block batch size 32.

`microbatch` is the maximum number of decision examples processed per GPU in one
forward/backward pass. The trainer caps it at that rank's share of the global
batch. `encoder_block_batch_size` bounds the number of historical visual blocks
in one encoder call, across examples; remaining blocks are processed in further
calls. It does not limit history length or change the optimizer batch size.
Activation checkpointing is disabled by default to avoid recomputing historical
visual features during backward. Set `model.checkpoint_blocks: true` if measured
activation memory requires that trade-off; optimizer checkpoint saving is separate
and remains enabled.

`training.workers` now controls persistent background processes for policy
training. Each rank starts eight CPU workers using the `spawn` start method. They
read observations and pack CPU tensors for upcoming microbatches while that
rank trains on the current one. HDF5 reads take place independently in each
worker; no open HDF5 handle or CUDA context is passed to a worker. The existing
decision index is reused, so workers do not repeat the full startup data scan.
Model execution, random flow draws, context selection and all trainable encoding
stay in the training process. Compact/Perceiver frozen feature-cache reads
also remain there; the workers prepare raw observations and CPU input tensors.

`prefetch_factor` is the number of outstanding microbatches allowed per worker.
The default of one means at most eight queued or in-flight microbatches per rank,
in addition to the batch currently consumed. The queue does not preload the
dataset. Its memory use still depends on history length and multiplies across
ranks. Four ranks use 32 loading workers plus four training processes.
`hpc/site_4gpu.env` requests `STMEM_CPUS=64` for this entire allocation;
it is not a per-GPU allowance. Workers use one Torch CPU thread each.
If using an older site-file copy, change its `STMEM_CPUS` too before rendering.
Eight workers is a throughput experiment; shared-storage bandwidth can still
limit the benefit of additional processes.
Scale CPU resources with the GPU count: eight GPUs would start 64 loading
workers plus eight training processes, so use more CPUs or reduce `workers`.
The FP16 fallback also retains the current optimizer update's CPU batches until
the step succeeds so an overflow retry uses exactly the same inputs; BF16/FP32
do not retain that additional retry set.

Set `training.workers: 0` for synchronous loading and packing; the debug base
config explicitly uses this setting. `prefetch_factor` must be a positive
integer even when unused by the synchronous path. Process startup and filling
the first batch can take longer than later fetches. Compare several completed
updates after startup, and check CPU/RAM use alongside GPU utilization.
`startup/loader_start_seconds` measures initiation of the worker iterator;
the first update's wait also includes worker imports and preparing its inputs.
This change overlaps the existing HDF5 reader with training; it does not add
bulk or prefix-only reads, rebalance rank assignments, or alter the sampler.

| Teacher GPUs | Examples per GPU per update | Examples per batched pass | Passes accumulated per GPU |
|---:|---:|---:|---:|
| 1 | 128 | 64 | 2 |
| 4 | 32 | 32 | 1 |
| 8 | 16 | 16 | 1 |

The teacher optimizer sees 128 examples per update for 100,000 updates:
12.8 million example draws. Each example averages its own valid action dimensions before
examples are averaged, so variable terminal padding does not change its weight.
DDP synchronizes accumulated gradients at the update boundary. Debug configs
retain smaller microbatches for bounded workstation checks.

Task-balanced shuffled sampling replaces the old per-task length sorting. That
sorting made all tasks advance from short to long history together, causing the
observed 16-update cycle at batch 32. `length_bucket_size` must remain 1; the
older ordered buckets are rejected. Batching does not recreate that ordering. History
length still varies naturally, so correlate timings with the emitted history
metrics. No H200/B200 speedup is guaranteed before measuring the new run.

Follow [Launching the 100k teacher run](#launching-the-100k-teacher-run) above.
It uses the existing four-GPU site file with a fresh run root and a separate
generated plan. Prepared data and alignment records are reused.

Compare `train/step_seconds`, `train/examples_per_second`, history lengths and
peak allocated memory after several completed updates, and inspect GPU
utilization in W&B's **System** tab. Four-GPU execution consists of one batched
forward/backward pass per rank with 32 examples, followed by a synchronized
optimizer update. For a one-GPU comparison, use a separate run root and plan,
select `h200_1gpu`, and retain the same global batch and microbatch settings.

No downloads run inside these jobs. Stage the locked environment, pinned benchmark
and simulator, task archives, prepared training data, assets and caches beforehand.
`CUDA_VISIBLE_DEVICES` remains under Slurm's control. Durable checkpoints and results
always use `STMEM_RUN_ROOT`. Node-local scratch is optional; the current launcher
does not copy caches automatically. If a site stages immutable caches, point
`STMEM_CACHE_ROOT` at that verified copy in a site activation script and preserve
the original cache manifest/hashes. `STMEM_SCRATCH_ROOT` documents the available
location but does not infer a mount or silently stage the dataset.

## Checkpoint selection and resume

The actual dependency chain is teacher training → both planned validation
checkpoints → immutable teacher selection → cache → compact training → both compact
checkpoints under **both** tree and flat context families → immutable compact
selection. Labels and the two routers follow. Each router gets 16k initial contexts,
5k fitting steps, one 8k-context refresh and resumed fitting to 10k total updates.
The shared compact policy stays frozen throughout. Perceiver training shares the
selected teacher and cache. Evaluation covers the six methods and writes one report.

`--resume` skips matching completed stages, keeps active jobs, and adds an existing
`last.pt` to incomplete training commands. A marker only matches its concrete
configuration hashes and command signature. To submit only selected stages with
already available dependencies, use `--only`:

```bash
uv run python hpc/plan.py --plan core_seed0 --resume --dry-run
uv run python hpc/plan.py --plan core_seed0 --resume --only seed0_router_tree_final --dry-run
```

`--only` never adds ancestors implicitly. Submission checks external files and
upstream completion before allowing such a stage. Review the printed remaining
work, then replace `--dry-run` with `--submit` if desired. After interrupted chains,
cancel obsolete pending descendants with failed dependencies using the site's
normal Slurm workflow, then resume the appropriate remaining jobs; active job IDs
are retained rather than blindly duplicated.

The allocation shell traps `B:USR1@180` into a fresh allocation-specific stop file.
It does **not** forward USR1 into torchrun. Rank zero checks the marker at optimizer
boundaries and coordinates a save. A partial stage exits nonzero, so downstream
`afterok` jobs cannot start. The wrapper repeats an interrupted Bash `wait` to
retain the child's true exit status. Local subprocess tests exercise the warning,
checkpointed partial exit and resume with a stale marker. Periodic saves use a
same-filesystem atomic replace and retain `<checkpoint>.previous`.

The training checkpoint includes model, optimizer, scheduler/scaler when used,
RNG and sampler state, step, configuration and provenance. One rank writes after
all ranks have coordinated. Resuming on a different world size/device layout may
require a new explicitly documented sampling boundary; it is not an exact resume.
The two-rank CPU check includes a genuinely empty history, which needs dynamic
unused-parameter detection for the event encoder. An actual stop at update 1 and
resume to update 2 matched uninterrupted weights exactly and removed a stale
completion marker. This validates the distributed control path on CPU, not NCCL
throughput or multiple physical GPUs. FP16 skipped steps do not advance the
scheduler or optimizer-step budget; repeated overflow has a bounded failure limit.

## Final test and replication

Official test is a separate invocation after validation and protocol freeze:

```bash
uv run python hpc/plan.py --plan core_seed0 --freeze-protocol runs/frozen_core_seed0.json
uv run python hpc/plan.py --plan core_seed0 --phase test \
  --freeze-manifest runs/frozen_core_seed0.json --dry-run
```

The freeze records checkpoint/configuration/source hashes and the complete
inference contract. The direct evaluation CLI checks the same frozen task set,
budget, episode count, training seed, checkpoint role and observation/model settings,
so bypassing the planner cannot silently tune the test protocol. Changed weights
or configurations fail validation before test jobs are rendered. The test plan
assigns all 50 official test seeds per task exactly once
using deterministic modulo sharding. The separate `central_3seed` plan prepares
only FULL/TREE/FLAT/UNIFORM across seeds 0, 1 and 2. It excludes additional Perceiver,
CURRENT and ablation replication. Existing seed-0 work should be resumed/reused.

## Hardware and measured costs

NVIDIA lists RTX A5000 at capability 8.6, H200 at 9.0 and B200 at 10.0. Its
architecture matrix places initial Blackwell support at CUDA 12.8. PyTorch provides
2.9.1/torchvision 0.24.1 CUDA 12.8 wheels, matching the upstream version pins.
These facts establish a candidate combination, not a successful hardware test:
[NVIDIA capabilities](https://developer.nvidia.com/cuda/gpus),
[NVIDIA architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html),
[official PyTorch wheels](https://pytorch.org/get-started/previous-versions/).

`uv run python -m stmem.doctor --output runs/doctor.json` reports the actual driver,
wheel CUDA build, device and supported architectures, and executes real attention
and matrix multiplication forward/backward tests in FP32, BF16 and FP16. BF16 is
selected only after it passes; FP16 requires GradScaler and FP32 remains the debug
fallback. The installed GPU here was **RTX 5000 Ada**, not the anticipated A5000;
Torch 2.9.1+cu128 passed all three tests on driver 580.159.04. Neither H200 nor B200
has been tested. System drivers are never modified. The CUDA maximum reported by
`nvidia-smi` is a driver capability, not the wheel runtime.

Compute success does not establish Vulkan or headless RGB-D rendering. Run the
prepared doctor allocation first and perform the documented one-episode simulator
check on the actual evaluation node. No H200/B200 rendering success is claimed.
Inside that allocation, the separately bounded rendering/throughput probe is:

```bash
python hpc/smoke.py --render --updates 3 --output "$STMEM_RUN_ROOT/node_smoke.json"
```

It measures the actual device and executes a conditioning-aware reset/step plus
three synthetic debug updates. It does not predict core-shape throughput or cost.
The training and rendering probes create separate W&B runs explicitly tagged
`synthetic-debug` under the output path without its `.json` suffix; the doctor
alone does not create a training or evaluation run. Logging occurs outside the
timed updates.
The Slurm semantics used here follow the [official sbatch manual](https://slurm.schedmd.com/sbatch.html).

Job artifacts record measured process duration separately from full allocation
duration. After jobs finish, collect the latter from Slurm accounting:

```bash
uv run python hpc/accounting.py \
  --submitted hpc/generated/core_seed0_validation_h200_1gpu/submitted.json
```

`sacct` elapsed allocation seconds × allocated GPU count includes reserved GPU time
for preparation and evaluation. H200 and B200 requested-profile totals remain
separate; no credit conversion is invented. Verify the actual GPU model in the
run's doctor artifact because a profile name cannot constrain unspecified site
resources.

## Matched teacher continuations

Use `hpc/continue_teacher.py` to compare additional training with previous joint
commands against removing those seven input channels. Both branches start from
the same immutable copy of a teacher training checkpoint, retaining model,
optimizer, scheduler, scaler, normalization, RNG and sample position. Measured
joint positions and the previous gripper command remain inputs in both branches.
The source checkpoint must use the same GPU count as the new jobs.

After pushing the implementation and pulling it on the HPC:

```bash
cd /hpc/home/phi/rvalperga/chronotree
.venv/bin/python hpc/continue_teacher.py \
  --site hpc/site_b200_4gpu.env \
  --profile b200_4gpu \
  --checkpoint runs/teacher_b200_b128_m8_blocks32/seed0/teacher/last.pt \
  --additional-steps 10000 \
  --run-root runs/teacher_joint_command_continuations_50k \
  --output-dir hpc/generated/teacher_joint_command_continuations_50k \
  --dry-run
```

Inspect `source_step`, `end_step`, `schedule_steps`, resources and both jobs in the
printed plan. `last.pt` means its actual saved step; use `step_00050000.pt` if an
exact 50k snapshot exists and is the intended starting point. A 49,978-step source
produces checkpoints at 54,978 and 59,978. The original 100k LR schedule is kept;
the continuation is not a new warmup or a shortened cosine-decay schedule.

Rerun the same command with `--submit` replacing `--dry-run` to submit both jobs.
The reviewed snapshot remains fixed if the source `last.pt` subsequently advances.
Each job uses four GPUs on `aiq` with untyped `--gres=gpu:4`, 64 CPUs, 128G RAM and
48 hours with the current B200 site file. They can run independently and request
eight GPUs total if scheduled together. No new site file or data preparation is
needed. Training artifacts go to:

```text
runs/teacher_joint_command_continuations_50k/control/seed0/teacher/
runs/teacher_joint_command_continuations_50k/no_previous_joint_commands/seed0/teacher/
```

Both create new W&B runs in the existing project. The original run and the fresh
from-scratch ablation use their own directories. The paired launcher is for fresh
branches and refuses duplicate submissions; do not rerun it to resume an existing
branch. A continuation interrupted before its declared endpoint remains resumable
with `stmem.train --resume <branch>/seed0/teacher/last.pt` and that branch's generated
configuration, without `--fork-from` or `--fork-sha256`. A completed 10k continuation
has its own completion marker; it does not mark the original 100k run complete.
