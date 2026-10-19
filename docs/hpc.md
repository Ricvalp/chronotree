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

## Weights & Biases tracking

Tracking is enabled in `configs/base.yaml`, including the inherited debug configs.
The default `offline` mode records W&B run data locally without credentials or
network access. Training logs scalar metrics at the existing configured reporting
cadence; evaluation logs episode outcomes and summaries. Only rank zero opens the
training run under DDP. Gradient watching, automatic system polling and console
capture are disabled to keep collection overhead small. The existing local
checkpoints, metrics and per-episode results remain the durable experiment records.
Model weights, videos and source files are not uploaded automatically. The SDK's
[settings reference](https://docs.wandb.ai/models/ref/python/experiments/settings)
documents the collection controls.

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
Each output directory has a `tracking.json` record mapping local work to its W&B
run ID and latest local W&B directory. The ID survives a restart; online runs
resume with that ID, while each offline invocation writes a separate segment.
The local `tracking_history.jsonl` journal preserves increasing W&B history indices
across restarts, with optimizer/episode progress logged separately as `progress/step`.
Keep the journal alongside the checkpoints when resuming. An existing run's
project/entity cannot be changed in place; use a new output directory for a new
destination. Logging settings do not select checkpoints, change scientific seeds,
or alter Slurm completion dependencies.

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

## Profiles and distributed training

| Profile | GPU workers per single-node allocation | Intended use |
|---|---:|---|
| `a5000_debug` | 1 local process | Explicit small configs/debug shapes; not a core Slurm plan |
| `h200_1gpu` | 1 | Default experimental profile |
| `h200_4gpu` | 4 | Optional measured-throughput comparison |
| `b200_1gpu` | 1 | Explicit opt-in |
| `b200_8gpu` | 8 | Explicit opt-in |

Choose `--profile h200_4gpu` to render DDP training. One Slurm task runs **one**
torchrun agent, with `--nnodes=1 --nproc-per-node=4`. Evaluation shards each request
one GPU and use independent processes. Four replicas do not pool memory; use the
configured microbatch/accumulation and block checkpointing to address a teacher
replica OOM. The effective batch is fixed by the scientific config and must remain
divisible by the world size. Do not alter it silently for throughput.

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
