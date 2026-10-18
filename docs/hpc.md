# Cluster execution

Copy `hpc/site.env.example` to the ignored `hpc/site.env`. Fill absolute shared
repository, data, cache and run paths and the Python executable. If modules are
needed, `STMEM_ACTIVATE` names a shell file to source. The site file itself uses
literal `STMEM_NAME=value` assignments; the planner deliberately does not execute
shell substitutions. Set the site's partition, optional account/QoS/constraint,
GPU resource name, memory and walltime. `gpuq` is only the supplied partition
default and does not select H200 hardware.

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
