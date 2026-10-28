# Chronotree

A runnable PyTorch research implementation of query-adaptive space-time memory
for RoboMME. TREE and learned FLAT retrieval consume the same frozen causal leaves
and the same compact action policy. FULL, UNIFORM, PERCEIVER and CURRENT are also
implemented. Bounded active context does **not** imply bounded archive storage.

The implementation brief is [PROMPT_CHRONOTREE.md](PROMPT_CHRONOTREE.md).
Measured checks and outstanding experiments are in [docs/status.md](docs/status.md).

The opt-in [frozen-vision / no-proprioception experiment](docs/frozen_vision_experiment.md)
includes matched teacher configurations, separate feature preparation, HPC launch
commands and closed-loop targeting diagnostics. Existing XYZRGB runs keep their
original input behavior.
No benchmark performance or scaling result is claimed.

## Run these next

The environment and a two-episode VideoUnmask subset are already present on this
workstation. Reproduce the bounded complete synthetic stage cycle, then inspect
the cluster plan:

```bash
.venv/bin/python scripts/smoke.py
.venv/bin/python hpc/plan.py --plan core_seed0 --dry-run
```

The smoke creates a fresh timestamped directory under `runs/smoke/`. It executes
12 teacher, 12 compact and 12 Perceiver updates, two small router fits with one
label refresh each, then all six readers at inference. Its checkpoint selections
are explicitly synthetic debug copies, never official validation selections.

Before cluster execution, copy `hpc/site.env.example` to `hpc/site.env` and set
the shared paths, Python/activation file, partition, account/QoS and actual GPU
resource name. Full data and assets must be staged first. Submission happens only
with an explicit `--submit`; the default plan performs validation, not test tuning.

## Weights & Biases logging

Every training stage and official evaluation shard logs to W&B, including
checkpoint-selection rollouts. Label generation, aggregate reports and the smoke
diagnostics are tracked too. Logging is enabled by default in **offline** mode:
real W&B run files are saved beside each stage's outputs without requiring network
access or credentials. Existing JSON/JSONL results remain available.

For live dashboards, authenticate on the cluster with `.venv/bin/wandb login`
and add these assignments to `hpc/site.env`:

```bash
WANDB_MODE=online
WANDB_PROJECT=chronotree
WANDB_ENTITY=your_wandb_team_or_username
WANDB_DIR=
```

An empty entity uses your account's default; an empty directory stores logs with
each run. Keep API keys out of `site.env`; use W&B login or a credential supplied
through the job environment. For direct local CLI runs, export these variables
in the shell instead. `site.env` is read by the HPC launcher, not local training.

The logger records resolved configuration, source/data/checkpoint provenance,
training losses, LR, gradient norm, throughput, memory and router diagnostics.
Evaluation publishes just two chart metrics: `evaluation/success_rate_so_far_percent`
(successes divided by recorded episodes, including errors and timeouts) and
`evaluation/progress_percent` (recorded divided by planned episodes). Both use
0–100 percent. Counts and the last outcome remain in the run summary; full
per-task results and planned-denominator macro success stay in `summary.json`,
with latency/storage diagnostics in `diagnostics.json` and `episodes/*.json`.
Only rank zero logs under DDP. Policy training logs startup milestones and the
first completed optimizer update of each attempt, then `training.log_every`
(50 updates in the core config), plus final/stop metrics; evaluation logs its
initial progress and after each episode. Training startup builds the decision
index and normalization statistics in one pass over source/action arrays,
without loading point-cloud observations.
No data re-preparation is needed; see [training startup](docs/hpc.md#training-startup).
For a slow run, pass `--log-every 1 --progress-every-seconds 30` to `hpc/plan.py`
or policy training. This logs each completed update to W&B, including
`train/step_seconds`, and prints a heartbeat every 30 seconds while an update
is running. Heartbeats also append to the stage's `progress.jsonl`; they report
the current phase and completed examples without querying CUDA or calling W&B.
These runtime settings can change on resume without changing the scientific
configuration. See [diagnosing slow training](docs/hpc.md#diagnosing-slow-training).
W&B's background service handles writes/uploads and collects system metrics by
default (`tracking.system_stats: true`). Open the run's **System** tab for visible
GPU utilization, memory, power and temperature, plus CPU/RAM metrics. Only rank
zero starts the service under DDP; the service can monitor all GPUs visible in
that allocation. The installed SDK samples system metrics every 15 seconds.
There are no model watch hooks, gradient histograms or automatic checkpoint/video
uploads. Initialization and final flushing occur outside training loops and
policy latency measurements. Collection has some overhead; it does not add a
CUDA synchronization to each training step.

Real-data teacher, compact and Perceiver training also log two fixed training
examples at scheduled checkpoint saves, including candidate and final checkpoints.
In W&B, open `diagnostics/scene_0` and `diagnostics/scene_1` for the current point
cloud and predicted/recorded tool-center targets; the matching
`diagnostics/actions_0` and `diagnostics/actions_1` tables retain raw predictions,
deployed commands, ground truth and validity masks. These are training diagnostics,
not closed-loop validation results. Synthetic examples are skipped because their
random robot states do not satisfy the real robot's kinematics. The default cap
is 1,024 observed points per scene; disable scenes with
`tracking.scenes.enabled: false`. Scene inference and media serialization add work,
recorded separately as `diagnostics/scenes_seconds` and `diagnostics/scenes_bytes`.
The [cluster guide](docs/hpc.md#training-point-clouds-and-action-comparisons)
explains the visualization, schedule and settings.

`tracking.json` retains the W&B ID across resumes, while `tracking_history.jsonl`
maintains monotonic W&B history indices independently of checkpoint progress.
Offline attempts produce separate files with the same ID; sync them in attempt
order, using `wandb sync --append` for subsequent attempts. Keep each attempt's
`files/media` directory with its `.wandb` file so point clouds and action tables
can also be uploaded. Detailed setup and commands are in [docs/hpc.md](docs/hpc.md).
New closed-loop evaluation runs use an `evaluation-` prefix on their W&B
experiment/seed group, separating them from training within the same project.
Existing runs can already be filtered by `job_type=evaluation`.

## Inspect point clouds and encoder structure

The optional offline HTML inspector shows prepared XYZRGB clouds, the encoder's
actual observed-point centers and weighted space-time pooling edges, and the local trees
and temporal forest built from the causal history. It also shows the current
frame separately and marks pending execution blocks. See the
[visualization guide](docs/visualization.md) for installation, cluster export
commands and checkpoint-based inspection. Exporting does not change training
or data loading.

## Install

Linux, Python 3.11–3.12, and `uv` are required. The lock resolves Python 3.11.15,
Torch 2.9.1 + CUDA 12.8, and compatible NumPy 1.26.4. Simulator dependencies are
optional and pinned to the requested benchmark and ManiSkill fork revisions.

```bash
uv sync --locked --group dev                         # core training/tests
uv sync --locked --extra simulator --group dev       # includes pinned RoboMME
source .venv/bin/activate
python scripts/fetch_sources.py
python -m stmem.doctor --output runs/doctor.json
python -m pytest -q
```

The unified environment works locally, so no policy service is needed. The
optional `scripts/setup_sim.sh` installs the same locked simulator extra. No script
changes system drivers. Prepare the environment and assets while Internet is
available; jobs do not download packages on compute nodes. For offline `uv` use,
stage its cache and run `uv sync --locked --offline --extra simulator --group dev`.

The actual workstation is **RTX 5000 Ada, 32 GB**, not the anticipated A5000.
H200/B200 support is documented but still needs an actual node check:

```bash
python hpc/smoke.py --render --output runs/node_smoke.json
```

This performs at most eight debug updates and one reset/step/render. It is not a
core throughput estimate. The first node should also be profiled with the actual
teacher shapes before deciding whether four replicated GPUs are worthwhile.

## Real data and alignment

The downloaded VideoUnmask archive is 1,251,608,544 bytes compressed and expands to
14,445,442,112 bytes. Only two episodes were prepared. Episode limits apply after
downloading an entire task archive. The downloader prints sizes before transfer,
checks SHA256 and rejects unsafe archive members.

```bash
python -m stmem.data.download --tasks VideoUnmask --dry-run
python -m stmem.data.download --tasks VideoUnmask
python -m stmem.data.prepare --tasks VideoUnmask --episodes 2 --config configs/debug/data.yaml
python -m stmem.data.inspect --config configs/debug/data.yaml
python -m stmem.data.alignment --config configs/debug/data.yaml
```

Replay establishes `post_action_v1`: stored observation j predicts command j+1.
The reset command is not replayed a second time. Conditioning/reset commands are
unavailable model inputs; terminal duplicates and command gaps are masked.
Evidence and calibration details are in [docs/data_contract.md](docs/data_contract.md).
Training fails closed if the alignment registry or preprocessing contract does
not match. The committed alignment evidence is documentation; the executable
registry is generated against the actual local dataset under `data/`.

A tiny **real-data diagnostic**, already run locally, is:

```bash
python -m stmem.train --stage teacher --config configs/debug/real_teacher.yaml
python -m stmem.evaluate --config configs/debug/rollout.yaml --split val --method FULL \
  --checkpoint runs/real_debug/seed0/teacher/last.pt --output runs/real_debug/rollout_full_val
```

Existing runs require `--resume` or a new `paths.run_root`. The two-update model
has no demonstrated manipulation capability; the short rollout checks the
conditioning/history/chunking/termination path.

Full-data commands are prepared but **were not executed**:

```bash
python -m stmem.data.download --tasks all --dry-run
# Explicit full transfer, when storage and network are ready:
python -m stmem.data.download --tasks all
# Run preprocessing inside a compute allocation, with at least eight CPUs:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -u -m stmem.data.prepare --episodes 100 --config configs/teacher.yaml \
  --workers 8 --threads-per-worker 1 --skip-existing
python -m stmem.data.alignment --config configs/debug/data.yaml
```

Preparation parallelizes independent tasks across CPU worker processes. It does
not use the GPU. For this cluster's GPU-only queue, the supplied batch template
requests one GPU allocation with 16 CPUs and starts eight single-threaded workers:

```bash
mkdir -p hpc/logs
sbatch hpc/jobs/prepare.sbatch
```

`--skip-existing` rehashes raw inputs and skips only matching, complete outputs.
An interrupted task restarts from its beginning; completed matching tasks can be
reused. Start with four to eight workers and monitor storage throughput before
increasing concurrency. JSON progress identifies the hashing and conversion
phases; frame progress includes an estimated remaining conversion time per task.
Details and resource overrides are in the
[preparation guide](docs/hpc.md#parallel-data-preparation).

Use the core 2,048-point preprocessing config for the actual experiment. It
reserves 75% of the point budget for geometry above the calibrated table plane,
balanced over 10 cm XY cells, and saves spatially balanced clustered centers.
New outputs are in `data.prepared_subdir: prepared_balanced_v2` beneath the site
data root; existing `prepared/` files remain separate. Preparation refuses to
replace a different prepared format. The 256-point two-episode subset uses
`prepared_balanced_v2_debug` and is intentionally incompatible with core
training. Each task is opened independently; there is no framewise train/test split.

## Stages and comparisons

Explore the [interactive teacher architecture](docs/diagrams/teacher_architecture.html)
or open the [editable draw.io diagram](docs/diagrams/teacher_architecture.drawio).
Its four linked pages cover the full dataflow, encoder layers, action decoder,
context shapes and exact parameter counts. See the
[reading and maintenance guide](docs/architecture_diagram.md).

| Stage | Behavior |
|---|---|
| Teacher | Train visual/event/text encoders and action-query full-leaf flow policy. |
| Selection | Evaluate planned 50k/100k teacher candidates on the same two official val seeds/task; freeze the winner. |
| Cache | Immutable HDF5 task shards, completed blocks and only queried causal prefix variants. |
| Compact | Freeze encoders; one decoder on 45% TREE, 45% FLAT, 5% UNIFORM, 5% CURRENT contexts across budgets. |
| Selection | Compare 15k/30k candidates under both fixed tree and flat validation contexts. |
| Labels/router | Signed paired teacher-velocity gains; episode-disjoint scorer diagnostics; one bounded refresh. |
| Perceiver | Separately adapt B learned latents and decoder from teacher with frozen encoders. |

Core configuration uses F=8, 32 visual centers/frame (256 visual leaves/block), 32 current visual tokens,
d=256, H=16, execute=4, eight Euler flow steps and B=128. Training budgets are
64/128/256. The debugging profile is explicitly smaller (d=64). Local Gaussian
pooling uses metre-squared and recorded-frame-squared denominators; source-local
time never depends on final episode duration. Frozen cached and streaming content
is encoded in FP32 for parity; training and online decoding use device-tested
autocast. Sensitive geometry, reductions and aggregates remain FP32.

Policy training batches decision examples with their complete causal histories.
The teacher configuration trains for 100,000 updates with global effective batch
256 and a microbatch cap of 64 examples per GPU per forward/backward pass.
One GPU accumulates four batches of 64; four GPUs process 64 examples each in one
concurrent round. Compact and Perceiver defaults remain global batch 64 with a
microbatch cap of 16. Variable history/context
lengths use padding with explicit validity masks. Each example's valid action
dimensions are averaged before examples are averaged, preserving the loss weight
of short terminal chunks. Teacher history is never truncated to make a profile fit.
These existing batch settings have not been memory-validated with the larger
32-centers/frame encoder; measure on the target GPU before training and reduce
microbatch if needed while retaining the desired effective batch.

Preparation computes the geometry-only center indices once per frame and stores
them with versioned selection settings. The loader carries these small arrays
through CPU packing to the encoder. Online RGB-D ingestion uses the identical
point and center selectors once for each arriving observation; no object masks
or hidden simulator state enter either selector. The resulting local Gaussian
pooling and learned bandwidths remain trainable.

Historical visual blocks are processed in tensor batches of at most 128 for the
teacher, controlled by `training.encoder_block_batch_size`; the shared base
default remains 32. This setting limits an encoder call, not
the number of blocks retained from an example. Activation checkpointing is off
by default (`model.checkpoint_blocks: false`); enabling it trades additional
backward computation for lower activation memory. Task-balanced shuffled sampling
no longer sorts decisions into the
previous periodic short-to-long history cycle (`training.length_bucket_size: 1`).
The plain regression debug head and optional framewise encoder remain available.

Policy training loads and packs upcoming microbatches in persistent CPU worker
processes. The default `training.workers: 8` is **per GPU/rank**;
`training.prefetch_factor: 1` permits eight queued or in-flight microbatches per
rank. Four GPUs therefore use 32 loading workers plus four training processes
within the 64-CPU allocation requested by `hpc/site_4gpu.env`. The queue is
bounded, and training retains
the same sample order, causal histories and configured global batch. Set `workers: 0`
for synchronous loading with the same packing; debug configs do this explicitly.
Worker startup makes the first fetch slower, so assess throughput over completed
updates after startup. No full-data HPC speedup is assumed.
For compact/Perceiver stages, frozen feature-cache reads and context selection
remain in the training process. Loader parity, resume/DDP checks and the bounded
local CUDA diagnostic are recorded in
[the verification status](docs/status.md).

Use a fresh run directory for these changed settings; prepared data remains usable.
See the [batching configuration and fresh four-GPU launch](docs/hpc.md#batched-policy-training).

Examples after official checkpoint selection:

```bash
python -m stmem.data.cache --config configs/teacher.yaml --checkpoint runs/seed0/teacher/selected.pt
python -m stmem.train --stage compact --config configs/compact.yaml
python -m stmem.training.gain_labels --reader tree --config configs/router_tree.yaml
python -m stmem.train --stage router --config configs/router_tree.yaml
python -m stmem.evaluate --config configs/compact.yaml --split val --method TREE
python -m stmem.report --run-root runs/seed0 --split val
```

The planner supplies initial/final router step budgets and the one refresh. Use
it for the complete dependency chain, including checkpoint-selection evaluations:

```bash
python hpc/plan.py --plan core_seed0 --dry-run
python hpc/plan.py --plan core_seed0 --resume --dry-run
# Only when intentionally submitting:
python hpc/plan.py --plan core_seed0 --submit
```

The rendered core plan has 45 jobs and at most two concurrent allocations by
default. Costs remain unknown. Read [docs/hpc.md](docs/hpc.md) for selective stage
resume, signals, resource profiles, final protocol freeze and accounting.

## Analysis and artifacts

The [standalone teacher memory profiler](docs/profiling.md) runs 10 warmup
optimizer updates, 5 measured updates and one separate detailed trace. It reuses
the teacher and data loader, with per-GPU reports and its own W&B run/output
directory. The dedicated HPC launcher defaults to dry-run.

`stmem.evaluate --budget 64|128|256` supports matched reader budget curves;
`--intervention erase_execution|permute_execution|erase_conditioning` supports
fixed-validation memory sensitivity. PERCEIVER has a fixed trained latent count
and rejects budget overrides. `stmem.analyze` performs bounded frozen-checkpoint
diagnostics, including tiny exact search with hard size guards. Commands are in
[docs/analyses.md](docs/analyses.md).

Runs contain resolved config/provenance, masked normalization statistics inside
checkpoints, JSONL metrics, immutable candidate checkpoints, atomic per-episode
actions/status/error/timing results, capped videos, and completion markers.
Partial training exits 75 and cannot release `afterok` dependencies. Reports keep
all planned episode denominators, use paired episode uncertainty, and render
unrun methods as null/“not yet run”. Archive payload, keys, transfers, prefix
updates, cold startup and warm decision latency are recorded. Process RSS and
GPU peak memory are measured separately from logical tensor payload.

Implementation decisions and limitations are recorded in
[docs/decisions.md](docs/decisions.md). Upstream code, pinned revisions and license
attribution are documented under [third_party](third_party/README.md).
