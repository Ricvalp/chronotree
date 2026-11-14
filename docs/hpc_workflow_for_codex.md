# HPC experiment workflow: a handoff for another repository

**Reviewed against the Chronotree implementation on 14 November 2026.**

This document explains the workflow we use to prepare, submit, monitor, evaluate,
and resume research jobs on Slurm. It is intended to be given to Codex in another
repository as an implementation reference. The reusable part is the separation
of machine settings, scientific configurations, generated execution plans, and
durable results. The Chronotree model and dataset are not prerequisites.

Examples using `stmem`, `STMEM_`, teacher stages, and our directory names are
**actual Chronotree interfaces**. Replace these with the other project's names.
Sections marked “For a new repository” are recommendations, not claims that every
existing Chronotree launcher already implements them.

## 1. The basic model

There are four main layers:

| Layer | Purpose | Typical files |
|---|---|---|
| Site settings | Where files live and which cluster resources to request | `hpc/site.env`, `hpc/site_b200_4gpu.env` |
| Scientific configuration | Model, dataset, optimizer, batch sizes, seeds, training duration, evaluation protocol | `configs/base.yaml`, experiment-specific YAML |
| Generated execution plan | Concrete commands, dependencies, paths, Slurm scripts, submission receipts | `hpc/generated/<experiment>/` |
| Scientific artifacts | Checkpoints, metrics, manifests, evaluation results, videos | `runs/<experiment>/` |

A Python launcher reads the first two layers and writes the third. Only an
explicit `--submit` sends jobs to Slurm. The submitted jobs produce the fourth.

The usual sequence is:

1. Develop and test code locally.
2. Push the code and pull it on the HPC, or create a fixed checkout for the run.
3. Choose the site file and experiment configuration.
4. Generate a plan with `--dry-run`.
5. Inspect the resolved settings, scripts, destinations, and dependencies.
6. Repeat with `--submit`.
7. Monitor Slurm, local logs, and W&B.
8. Resume interrupted work from a checkpoint, or evaluate an immutable snapshot.

Codex normally prepares the code and gives the user submission commands. It
does not submit long experiments merely because it can access a shell.

## 2. Directory ownership and Git

Our layout is approximately:

~~~text
chronotree/
  pyproject.toml
  uv.lock
  configs/
    base.yaml
    teacher.yaml
    teacher_<variant>.yaml
    experiment_core_seed0.yaml
  src/stmem/
    train.py
    training/checkpoints.py
    tracking.py
  hpc/
    site.env.example
    site.env
    site_b200_4gpu.env
    plan.py
    evaluate.py
    train_compact.py
    student_utility.py
    run_job.sh
    job_status.py
    accounting.py
    jobs/
      common.sh
      teacher.sbatch
      evaluate.sbatch
      ...
    generated/
      <experiment>/
        plan.json
        configs/
        jobs/
        submitted.json
        stops/
    logs/
  runs/
    <experiment>/
      seed0/
        teacher/
          config.json
          manifest.json
          tracking.json
          metrics.jsonl
          progress.jsonl
          last.pt
          step_00100000.pt
          complete.json
    evaluation_inputs/<evaluation-name>/checkpoint.pt
    evaluation/<evaluation-name>/
  docs/
~~~

Not every launcher uses exactly that generated-directory layout. Some put
`compact.yaml` or `evaluate.sbatch` directly in the plan directory, and some put
logs beneath that directory. Read the emitted plan for the authoritative paths.

**The distinction between `hpc/jobs/` and `hpc/generated/` matters:**

- `hpc/jobs/` contains maintained, generic one-stage templates with literal
  default resources.
- `hpc/generated/` contains experiment-specific rendered files and execution
  state. The general planner renders its scripts in Python; it does not simply
  copy the templates.
- `runs/` contains the results. It is independent of the generated-plan location.

Commit source code, reusable configs, documentation, launchers, and the site
template. Ignore weights, caches, logs, generated plans, and ordinary local site
files. We have deliberately tracked a few nonsecret site variants for our shared
cluster, but that is a project choice, not a reason to commit credentials.

Generated does not mean disposable while a job is queued or running. Jobs can
read generated configs and plans at startup; receipts are also needed for
duplicate detection and recovery. Preserve them with the experiment record.

## 3. Site files

A site file is a small set of **literal assignments**, not an executable setup
program. It describes a machine/allocation preset. It should not contain model
architecture or learning-rate choices.

Example for four B200s on our cluster; replace the paths and site-specific values
when porting:

~~~dotenv
STMEM_REPO_ROOT=/shared/project/chronotree
STMEM_DATA_ROOT=/shared/data/robomme-data
STMEM_CACHE_ROOT=/shared/data/chronotree-cache
STMEM_RUN_ROOT=/shared/project/chronotree/runs
STMEM_PYTHON=/shared/project/chronotree/.venv/bin/python
STMEM_ACTIVATE=

STMEM_PARTITION=aiq
STMEM_ACCOUNT=
STMEM_QOS=
STMEM_CONSTRAINT=
STMEM_GPU_TYPE=b200
STMEM_PROFILE=b200_4gpu
STMEM_CPUS=64
STMEM_MEM=128G
STMEM_WALLTIME=48:00:00
STMEM_EVAL_WALLTIME=06:00:00
STMEM_MAX_CONCURRENT=4
STMEM_SCRATCH_ROOT=

WANDB_MODE=online
WANDB_PROJECT=chronotree
WANDB_ENTITY=
WANDB_DIR=
~~~

### Meaning of the variables

| Variable | Meaning |
|---|---|
| `STMEM_REPO_ROOT` | Absolute checkout path visible on the compute nodes. |
| `STMEM_DATA_ROOT` | Shared raw/prepared data root. |
| `STMEM_CACHE_ROOT` | Shared reusable derived-data/cache root. |
| `STMEM_RUN_ROOT` | Default root for training outputs and stage markers. Prefer an explicit per-experiment override when launching independent runs. |
| `STMEM_PYTHON` | Absolute Python executable in the installed environment. |
| `STMEM_ACTIVATE` | Optional shell script sourced by the job for modules or environment setup. Empty means none. |
| `STMEM_PARTITION` | Slurm partition, such as our `aiq` or `gpuq`. |
| `STMEM_ACCOUNT` | Optional Slurm charging account. |
| `STMEM_QOS` | Optional site-authorized QoS. |
| `STMEM_CONSTRAINT` | Optional node-feature constraint. |
| `STMEM_GPU_TYPE` | Scheduler's GPU GRES type, such as `b200`; not a Python device name. |
| `STMEM_PROFILE` | Project hardware profile; in the general planner it determines training GPU count. |
| `STMEM_CPUS` | CPUs for the one Slurm task in the general planner, shared by its trainer and child processes. |
| `STMEM_MEM` | Host RAM requested for the allocation, not GPU VRAM. |
| `STMEM_WALLTIME` | Allocation time limit for training/preprocessing in the general planner. |
| `STMEM_EVAL_WALLTIME` | Evaluation time limit in the general planner. |
| `STMEM_MAX_CONCURRENT` | Allocation-lane limit within a general-planner submission; not a cluster-wide cap. |
| `STMEM_SCRATCH_ROOT` | Optional known node-local location for staging immutable features. Empty keeps shared-storage access. |
| `WANDB_MODE` | `online` for live synchronization, `offline` for local W&B files; `disabled` explicitly turns it off. |
| `WANDB_PROJECT` | W&B project. We use separate projects for some experiment families. |
| `WANDB_ENTITY` | Optional account/team; blank uses the tracking fallback. |
| `WANDB_DIR` | Optional W&B storage location; blank normally keeps files beside run outputs. |

`STMEM_PREP_WORKERS` is an additional override used by the preparation template.
It is not a global data-loader worker setting. Likewise, `STMEM_CONFIG`,
`STMEM_SITE_FILE`, and `STMEM_NPROC` belong to particular direct-template/runtime
interfaces; do not assume every launcher consumes every `STMEM_` variable.

Use full paths. The parser rejects `$HOME`, `$(...)`, and other dollar/backtick
substitutions. Avoid `~` too: the literal parser does not expand it. Quotes and
comments are accepted. Keep activation commands in `STMEM_ACTIVATE`'s separate
script. Never store `WANDB_API_KEY` in a site file; authenticate with the account's
login mechanism or provide credentials through the runtime environment.

### Do I need to source it?

**Not when invoking the Python planner with `--site`.** This works from an ordinary
shell:

~~~bash
.venv/bin/python hpc/plan.py --site hpc/site.env --dry-run
~~~

The planner reads the file itself. However, this does not set variables in the
parent shell. If you want to type `"$STMEM_PYTHON"` interactively, first load the
trusted file:

~~~bash
set -a
source hpc/site.env
set +a
"$STMEM_PYTHON" hpc/plan.py --site hpc/site.env --dry-run
~~~

An unset `STMEM_PYTHON` explains the confusing `Command '' not found` error. Prefer
`.venv/bin/python` for launcher commands to avoid that dependency.

### Precedence

The shared `read_site()` helper starts with supported process-environment
variables, then **file assignments override them**, including explicit blanks.
Launcher-specific CLI overrides are applied afterward where implemented.

For the general planner, `--run-root`, `--profile`, and `--walltime` can override
the corresponding site settings. Changing an exported shell variable will not
override the same assignment in the site file.

Dedicated launchers often have their own resource flags/defaults. For example,
`hpc/evaluate.py` takes `--partition`, `--gpu-type`, `--gpus`, and `--walltime`
directly. A B200-named site file alone does not make that launcher request B200s.
Check `--help` and the rendered plan, rather than assuming identical precedence
across all scripts.

## 4. Scientific configs and experiment identity

`configs/base.yaml` contains shared scientific defaults. Stage/experiment YAML
files inherit and override them. Our `extends` mechanism is implemented by this
repository; it is not standard YAML functionality. Nested mappings are merged,
while scalar/list values are replaced.

For example, a new teacher configuration can contain:

~~~yaml
extends: teacher.yaml
training:
  lr: 0.0001
  effective_batch: 128
  microbatch: 8
~~~

This changes the training experiment, so give it a new run root and generated
plan directory. There is no need to create another site file just to change LR.

Changing `base.yaml` can change every dependent configuration. Some general-planner
generated configs still extend source YAML files by absolute path: rendering is
not equivalent to freezing their contents forever. Preserve source configs used
by active jobs. **For a new repository, write a fully resolved config snapshot
into each immutable attempt directory.**

Keep these identities separate:

| Identity | Example | What changes it |
|---|---|---|
| Scientific run | `teacher_lr_compare_seed0` | Different scientific config, data, initialization, or seed. |
| Execution attempt | A new Slurm job resuming that run | New allocation, restart, or reviewed runtime change. |
| Evaluation | A named snapshot plus fixed protocol | Different checkpoint, reader, episode set, action cadence, etc. |

A resumed run should keep its scientific output directory and W&B identity.
An independent comparison should have different outputs and identity, even if
it starts from the same weights.

## 5. Hardware profiles and Slurm resources

A profile such as `b200_4gpu` is a project label. It does not itself select
hardware. In the general planner:

- `--profile b200_4gpu` selects four training processes/GPUs for applicable stages.
- `STMEM_PARTITION=aiq` selects the partition.
- `STMEM_GPU_TYPE=b200` produces the literal request `--gres=gpu:b200:4`.

Other stages can deliberately use fewer GPUs: the general planner does not
allocate four GPUs to every stage merely because its training profile says four.
The plan lists each job's resources.

Ask the cluster which resources exist:

~~~bash
sinfo -o '%P %G %f'
~~~

Our cluster advertises B200s on `aiq` and H200s on `gpuq`. **As reported by the
user in November 2026, it now requires typed GPU requests and enforces the
associated quotas.** Older checked-in site comments and commands that require
untyped `gpu:N` are historical and must not be copied as current policy.
Configure legitimate account/QoS settings with the administrator.

Slurm's typed GPU syntax and allocation rules are documented in the
[GRES guide](https://slurm.schedmd.com/gres.html). A profile name also does not
prove what ran: record the actual GPU model in the environment manifest.

Render literal directives:

~~~bash
#!/usr/bin/env bash
#SBATCH --partition=aiq
#SBATCH --gres=gpu:b200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
~~~

Loading `site.env` inside a shell script cannot change an earlier directive such
as `#SBATCH --gres=gpu:$GPU_COUNT`. Slurm reads those directives without shell
expansion. It also transfers the batch script, not the referenced checkout,
configs, or datasets. See the
[`sbatch` documentation](https://slurm.schedmd.com/sbatch.html).

We do not request `--exclusive` by default. It is a separate node-sharing choice,
not a prerequisite for multi-GPU execution. Requested walltime is an upper bound,
not an estimate of how long training needs or a promise the partition permits it.

## 6. The general planner and the everyday submission workflow

`hpc/plan.py` expands an experiment manifest into a dependency graph. A job record
contains its name/stage, argv command lists, dependency names, expected inputs
and outputs, seed, GPU count, walltime, script path, and stage signature.

Its main operations are:

1. Read the selected site, profile, and experiment manifest.
2. Build stage-specific configs and commands.
3. Resolve configuration inheritance and calculate signatures.
4. Render literal Slurm scripts and create output/log directories.
5. Write and print `plan.json`.
6. If `--submit` is present, validate dependencies, call `sbatch --parsable`, and
   atomically save every returned job ID.

**Dry-run means no job submission. It can still create or update files**, including
configs, scripts, manifests, and—in evaluation launchers—checkpoint copies.
Do not run a different dry-run into a directory used by an active job.

### A concrete Chronotree training example

Run on the HPC from the checkout root. First configure `hpc/site_b200_4gpu.env`
with the current typed resource `STMEM_GPU_TYPE=b200` and appropriate paths.
This example selects an existing teacher config; it does not change its scientific
settings.

~~~bash
cd /hpc/home/phi/rvalperga/chronotree

STMEM_EXPERIMENT=teacher_workflow_example
STMEM_LAUNCH_ARGS=(
  --site hpc/site_b200_4gpu.env
  --plan core_seed0
  --profile b200_4gpu
  --teacher-config configs/teacher_fv_no_prop_10hz.yaml
  --only seed0_teacher
  --run-root "runs/$STMEM_EXPERIMENT"
  --output-dir "hpc/generated/$STMEM_EXPERIMENT"
  --log-every 1
  --progress-every-seconds 30
)

.venv/bin/python hpc/plan.py "${STMEM_LAUNCH_ARGS[@]}" --dry-run
~~~

The Bash argument array is used to keep dry-run, submit, and resume arguments
identical. In the command above and below, its expansion is `"${STMEM_LAUNCH_ARGS[@]}"`.
Run the blocks in the same shell.

Inspect:

~~~bash
cat "hpc/generated/$STMEM_EXPERIMENT/plan.json"
cat "hpc/generated/$STMEM_EXPERIMENT/configs/seed0_teacher.yaml"
cat "hpc/generated/$STMEM_EXPERIMENT/jobs/seed0_teacher.sbatch"
~~~

Check the run root, interpreter/checkout, config inheritance, input identities,
GPU type/count, CPU/RAM/walltime, W&B project, dependencies, and expected outputs.
A one-job teacher selection should report `job_count: 1`.

Then submit:

~~~bash
.venv/bin/python hpc/plan.py "${STMEM_LAUNCH_ARGS[@]}" --submit
~~~

The array expansion in these examples is ordinary Bash syntax; another repo
must implement its own equivalent flags rather than assume Chronotree stage
names exist. The general planner currently uses `--teacher-config` rather than a
universal `--config`; hyperparameters such as LR belong in that YAML, not
unimplemented `--lr` flags.

`--only` selects matching job names or stage categories. It does not automatically
rerun all their ancestors. Required upstream outputs must already exist or be
provided by the selected dependency chain.

### Running two experiments simultaneously

Use the same site file if resource requirements are the same. Choose a different
scientific YAML when needed, and **different run roots and generated directories**
for both experiments. Then submit both plans. Slurm decides when each can start.

Changing only `--profile` changes the default generated-directory name but does
**not** change the run root. Two jobs with different profile names can otherwise
write the same `seed0/teacher/last.pt` and W&B identity. Do not share a writable
run directory between independent trainers.

## 7. Dependencies, arrays, and concurrency

Our general planner uses:

- `afterok` for scientific dependencies: a downstream stage needs successful
  completion of its prerequisites.
- `afterany` for a separate set of allocation lanes: an earlier allocation must
  finish before another in that lane starts, regardless of success.

`STMEM_MAX_CONCURRENT` controls those lanes for that submission. The printed
`maximum_concurrent_allocations` is at most the number of selected jobs. Thus a
teacher-only plan reports one even when the site limit is four. True dependencies
can reduce the attainable parallelism further. This counts allocations, including
CPU stages, not GPUs, trainers, or data-loader workers.

Separate invocations/studies do not share one global concurrency manager.
Three independent four-GPU evaluations can therefore request twelve GPUs in
total. Scheduler/account limits still apply.

For embarrassingly parallel preparation or measurement, use bounded job arrays:

~~~bash
# Syntax illustration: script must implement the shard-to-array-index mapping.
sbatch --array=0-15%4 path/to/independent_shard_job.sbatch
~~~

This represents sixteen tasks, at most four running simultaneously. Each uses
`SLURM_ARRAY_TASK_ID` to select disjoint work. Use `%A` and `%a` in array log
names. See [Slurm job arrays](https://slurm.schedmd.com/job_array.html).

Independence requires separate output ownership or an explicit safe writer
protocol. Different workers must not concurrently write the same HDF5 file or
completion record. Increase concurrency only after checking aggregate RAM,
storage bandwidth, GPU consumption, and quotas.

The planner submits the graph; it is not a continuously running orchestration
service. A failed prerequisite can leave dependent jobs pending. Recovery must
replace failed dependency IDs through a fresh validated submission.

## 8. Multi-GPU execution and batch sizes

Training uses one node, one Slurm task, and one torchrun agent:

~~~bash
srun --ntasks=1 /shared/project/chronotree/.venv/bin/python \
  -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m stmem.train --stage teacher --config /shared/experiment/config.yaml
~~~

That agent launches four workers, one per assigned GPU. Do not request four
Slurm tasks and have each launch another four-worker agent. The trainer uses
`LOCAL_RANK`, initializes distributed training, and respects Slurm's
`CUDA_VISIBLE_DEVICES`.

For evenly sized microbatches:

~~~text
effective global batch = GPUs × per-GPU microbatch × accumulation passes
128 = 4 × 8 × 4
128 = 2 × 64 × 1
~~~

A logged training step is an **optimizer update**, including its accumulation
passes. The implementation must handle a smaller final microbatch correctly if
the per-rank sample count is not divisible by the microbatch cap.

More GPUs do not increase per-replica VRAM under ordinary DDP. Host RAM is another
resource entirely. Chronotree's `encoder_block_batch_size` controls an internal
encoder work chunk; it is neither the global batch nor the action horizon.

Data-loader workers are per rank. Two GPUs with sixteen workers each create
thirty-two loader processes plus trainers and other services. Allocate CPUs and
host RAM accordingly; increasing prefetch also increases buffered data. Record
actual data-wait and compute times before assuming more workers or GPUs will help.

Exact resume currently checks world size and training configuration. Switching
from two GPUs to four, or changing the effective batch, is not automatically a
valid resume. Implement and document a deliberate migration/fork if needed.

## 9. Checkpoints, stopping, resuming, and continuing

Keep three events distinct:

| Event | Meaning |
|---|---|
| Periodic checkpoint | Save progress and continue training. |
| Checkpointed partial stop | Save coherent progress, exit, but the requested training endpoint is not complete. |
| Scientific completion | The requested endpoint and required artifacts have been produced. |

Checkpoint code writes to a temporary file on the destination filesystem, then
atomically replaces the final path. Rank zero writes; distributed training
collects the required per-rank state first. Our recovery payload includes model,
optimizer, scheduler, AMP scaler, optimizer step, per-rank RNG state, configuration,
data/dependency provenance, and sufficient sampling position to resume the
deterministic stream. Retain a previous known-good checkpoint where supported.

`last.pt` is the rolling recovery checkpoint; numbered `step_XXXXXXXX.pt` files
retain selected snapshots. `complete.json` records actual completion.
`partial.json` records a coordinated early stop; it is not guaranteed to exist
after a hard kill or node failure.

### Graceful stop

The training script requests `#SBATCH --signal=B:USR1@180`. The wrapper
`hpc/run_job.sh` traps the warning and creates an allocation-specific stop file.
The trainer checks it at a safe boundary, coordinates ranks, saves, and exits.
The wrapper preserves the child's exit status instead of mistaking an interrupted
shell `wait` for successful completion.

Request the same stop manually for an active training allocation:

~~~bash
STMEM_JOB_ID=123456
scancel --signal=USR1 --batch "$STMEM_JOB_ID"
squeue -j "$STMEM_JOB_ID"
~~~

Wait for the job to stop and inspect the checkpoint. This is a checkpoint request,
not an immediate-save guarantee: a hung process or failed node may never reach
the save boundary. Choose a warning margin long enough for an update and save.
Plain cancellation can lose work since the last periodic checkpoint.

A checkpointed partial training job exits with code **75** in our implementation.
That prevents downstream `afterok` jobs from treating incomplete training as
success. Slurm may display it as FAILED despite a useful saved checkpoint.

### Resume

Use the same run root, scientific config, generated-plan identity, and compatible
data/model dependencies. For the earlier teacher example, after the job stops:

~~~bash
test -s "runs/$STMEM_EXPERIMENT/seed0/teacher/last.pt"
.venv/bin/python hpc/plan.py "${STMEM_LAUNCH_ARGS[@]}" --resume --dry-run
# Inspect the emitted --resume checkpoint and remaining work before submitting.
.venv/bin/python hpc/plan.py "${STMEM_LAUNCH_ARGS[@]}" --resume --submit
~~~

Check the actual saved step rather than infer it from the last W&B chart point:

~~~bash
.venv/bin/python - "runs/$STMEM_EXPERIMENT/seed0/teacher/last.pt" <<'PY'
import sys
from stmem.training.runtime import load_checkpoint
state = load_checkpoint(sys.argv[1])
print({"stage": state["stage"], "step": state["step"]})
PY
~~~

The general planner's `--resume` skips hash-verified completed stages and adds a
checkpoint argument when `last.pt` exists. **It is not a universal guarantee
that a missing checkpoint is an error**: unfinished stages without one may start
fresh. Dedicated compact/student launchers have stricter resume checks. Inspect
the emitted command. For a new repository, make a requested training resume
fail clearly when its checkpoint is missing.

Resuming means continuing to the original endpoint, not adding that many steps
again. Preserve the run's `tracking.json` to continue the same W&B identity.
Resume cannot recover updates performed after the last completed checkpoint.

### Extending training

Changing the allocation walltime does not change the optimizer-step endpoint.
Changing `training.steps` can change the LR schedule and checkpoint identity.
Chronotree has explicit continuation interfaces, such as teacher
`--train-until-stopped` and compact `--continue-to-step`, with their own restrictions.
They preserve the existing decay horizon and implement the documented continuation
policy. Neither automatically resubmits jobs indefinitely.

For a new repository, expose “resume”, “extend endpoint”, and “initialize a new
experiment from weights” as distinct operations.

## 10. Immutable plans and source changes

Slurm saving a script does not freeze Python imports or YAML referenced by that
script. A queued job—and a requeued copy of an old job—can start after a
`git pull`. A newly spawned worker can also import updated files.

Our stronger launchers pin source/config/checkpoint identities and fail early
when they differ. A real failure we encountered was:

~~~text
ValueError: Source changed since compact job rendering
~~~

The job had been rendered before the checkout changed. A later startup refused
the mismatch. This message describes that startup failure; it does not explain
why an earlier allocation of the same job stopped.

Do not remove a guard or manually replace a hash to make the job run. Either:

- Use the unchanged compatible source the plan was generated for; or
- Deliberately regenerate a validated resume attempt under reviewed code,
  preserving the scientific inputs and recording the transition.

Some compact launchers expose `--upgrade-loader` for an explicitly reviewed runtime
source transition with unchanged configuration/dependencies/world size. It is
not a general declaration that arbitrary code changes are scientifically safe.

**Implementation differences to preserve honestly:** the general planner hashes
merged configuration and `src/**/*.py` into stage signatures, but it does not
install the same comprehensive startup guard as every newer dedicated launcher.
Generated general-planner YAML may still inherit live source configs. Dedicated
launchers have stronger named-plan checks, locks, and sometimes immutable
per-attempt plans. Do not assume all paths have identical protections.

For a new repository, improve this from the start:

1. Give each attempt an immutable config and startup-plan path.
2. Run against a pinned checkout or source snapshot.
3. Record Git revision, dirty status, and relevant source/config fingerprints.
4. Include launcher/runtime code in the appropriate integrity checks, not just
   model files.
5. Acquire the run lock and reject active/foreign owners **before** changing files.
6. Keep immutable receipts for all attempts, not only the latest job ID.

A detached Git worktree is one way to isolate source:

~~~bash
# Illustration: replace the path and commit with the intended values.
git worktree add --detach /shared/checkouts/experiment_a COMMIT_SHA
~~~

It creates another checkout at that commit without creating a branch or switching
the main checkout. Verify that its Python imports point to that checkout:
an editable installation or inherited `PYTHONPATH` can otherwise select another
source tree. Share immutable data/caches through explicit paths, and keep outputs
outside disposable source snapshots.

Some generated scripts export `STMEM_SITE_FILE=/dev/null` and a fixed
`PYTHONPATH` so runtime helpers cannot reload an unrelated site or import a
different checkout. This is intentional; it is not how one normally selects a
site for the initial planner invocation.

## 11. Completion, caching, and provenance

The general planner uses `hpc/job_status.py` to record exit status, required output
presence, and output hashes. A stage is reusable only when its completion marker,
signature, and required output hashes agree. A zero process exit without expected
outputs is not scientific completion.

Record enough identity to distinguish:

- Code/config changes.
- Different datasets, splits, preprocessing, or alignment revisions.
- Different upstream checkpoints.
- Different seeds, sampling conventions, or distributed world sizes.
- A new evaluation protocol applied to the same checkpoint.

Chronotree's scientific consumers also enforce verified action alignment, causal
history, exact evidence coverage, and teacher/cache compatibility. A generic
launcher should call project-specific validation hooks rather than relax these
conditions when scheduling or memory becomes inconvenient.

Caches are shared **immutable inputs** with content/configuration identities,
separate from mutable run outputs. Expensive preprocessing can be partitioned
by task/shard, using lock files and partial outputs, then atomically published.
Do not treat a `.partial` file as a completed cache.

Validate expensive payloads when constructing/publishing them. Reuse trusted
completion manifests, identities, and appropriate integrity checks at startup,
instead of unnecessarily rescanning every byte for every worker. Different
checks protect different invariants; a fast “all task files present” status
report is not the same as content verification.

## 12. Evaluation is a separate experiment

Use a dedicated launcher for evaluation. It should freeze:

- The exact checkpoint and its hash/step.
- Configuration and code.
- Dataset split, episodes, seeds, reader, budget, and action/replanning settings.
- Worker assignment, output location, and video cap.

Our `hpc/evaluate.py` copies the source checkpoint into
`runs/evaluation_inputs/<name>/checkpoint.pt` during preparation, verifies that
the source did not change during copying, and uses that fixed copy later.

**“Latest” means latest at snapshot creation.** Submitting an existing named
dry-run uses its already frozen checkpoint, even if training has since advanced.
Use a new evaluation name to capture a newer `last.pt`. Resuming an evaluation
must retain its original snapshot.

Actual Chronotree example for a full-history teacher:

~~~bash
cd /hpc/home/phi/rvalperga/chronotree
STMEM_EVAL_NAME="teacher_eval_$(date -u +%Y%m%dT%H%M%SZ)"

.venv/bin/python hpc/evaluate.py \
  --site hpc/site_b200_4gpu.env \
  --checkpoint runs/teacher_fv_no_prop_10hz/seed0/teacher/last.pt \
  --name "$STMEM_EVAL_NAME" \
  --method FULL --execute-actions 8 \
  --partition aiq --gpu-type b200 \
  --gpus 4 --workers-per-gpu 4 --cpus 64 --mem 128G \
  --walltime 06:00:00 --episode-count 10 --video-cap 10 \
  --dry-run
~~~

Inspect `hpc/generated/<name>/plan.json`, then repeat with the same name and
arguments, replacing `--dry-run` with `--submit`.

Here four GPUs × four workers/GPU gives sixteen independent simulator processes,
not DDP training. Ten episodes/task across sixteen tasks gives 160 episodes.
The video cap is ten across the evaluation, not ten per worker. The model and
renderer are replicated per process, so worker count affects both RAM and VRAM.

Results live under `runs/evaluation/<name>/`. The aggregate `summary.json` must
report the planned denominator, recorded/missing episodes, success, ordinary
failure, timeout, and infrastructure error. Per-worker logs are under
`workers/`; the generated plan identifies the top-level Slurm logs.
Keep shard artifacts/videos reachable from this evaluation directory.

Shards should publish per-episode results atomically and never duplicate work by
padding a distributed sampler. Distinguish valid task timeouts from missing
results and broken renderers. A partially completed evaluation is not a finished
benchmark with a smaller denominator.

For compact Chronotree policies, `--method TREE --selection-family tree` means
random/gain-blind TREE selection. A learned router is a separately pinned input
passed with `--router-checkpoint`. The reader name alone does not prove that a
trained selector was evaluated.

## 13. Logging and monitoring

Every training/evaluation run should remain understandable without W&B.
Persist local resolved config, provenance, scalar metrics, progress heartbeats,
checkpoint metadata, and per-episode results.

We additionally use W&B with:

- A stable run ID in `tracking.json`, reused across genuine resumes.
- Separate run types/groups for training and evaluation.
- Rank-zero logging under DDP; a controlled aggregate view for parallel evaluation.
- Loss, LR, optimizer step, step duration, data wait, throughput, and CPU/GPU memory.
- System statistics enabled and expensive visual diagnostics at bounded intervals.
- A small evaluation dashboard for progress/success, with detailed tables/files
  kept out of hundreds of automatically generated scalar panels.

The same W&B project/entity must remain compatible on resume. A new run root
normally means a new run identity. A stopped process can leave a stale remote
status, so use local logs and scheduler state to diagnose failures.

The W&B chart step is not necessarily the optimizer step: diagnostic events can
advance its history counter independently. Read the explicit training-step metric
or checkpoint metadata.

Useful HPC commands:

~~~bash
squeue -u "$USER"

STMEM_JOB_ID=123456
scontrol show job "$STMEM_JOB_ID"

sacct -D -j "$STMEM_JOB_ID" \
  --format=JobID,State,ExitCode,Start,End,Elapsed,Timelimit,NodeList,Reason%40

sstat --allsteps --jobs="$STMEM_JOB_ID" \
  --format=JobID,AveCPU,AveRSS,MaxRSS,MaxDiskRead
~~~

`scontrol show job` includes `StdOut` and `StdErr` paths while the controller
still retains the job. `sacct -D` helps recover earlier accounting records when
a job ID has multiple attempts; inspect timestamps and nodes. See
[`sacct` documentation](https://slurm.schedmd.com/sacct.html).

Typical log destinations:

| Launcher | Usual log location |
|---|---|
| General `hpc/plan.py` | `hpc/logs/stmem_<job-name>_<job-id>.out/.err` |
| Compact/student launchers | `hpc/generated/<experiment>/logs/` |
| Parallel evaluation | `hpc/generated/<evaluation-name>/logs/` plus result-directory `workers/` |
| Array stages | Launcher-specific generated logs, usually including array job ID and task index |

Create these directories **before** submission: Slurm opens its output streams
before the batch shell can create them.

### Failure patterns we have encountered

| Symptom | Interpretation and next check |
|---|---|
| `Command '' not found` | An unset interpreter variable; invoke `.venv/bin/python` or source the intended site. |
| `Source changed since ... rendering` | Startup source differs from the pinned plan. Recover with compatible source or a reviewed new attempt. |
| “Plan is immutable” | Same experiment name was reused with changed inputs. Identify the changed input; do not erase provenance. |
| `QOSGrpGRESMinutes` | A scheduler GPU-time quota limit. Inspect account/QoS usage; a shorter request does not replenish an exhausted cumulative quota. |
| GPU type required | Site now requires a typed GRES request; update legitimate resource settings. |
| CUDA device unavailable / reset required / Vulkan device lost | Device or rendering initialization/runtime failure; collect node and scheduler evidence. This alone is not a model-memory OOM. |
| Explicit CUDA out-of-memory | Inspect per-replica activations and long-history cases; host `--mem` does not increase VRAM. |
| Host OOM / killed loader | Check allocation RAM and total workers/prefetch, not just GPU utilization. |
| `ChildFailedError` | Torchrun's wrapper report; find the first worker error or signal above it. |
| Exit 75 with `partial.json` and a valid checkpoint | Expected coordinated partial stop in this project. |
| W&B present but no training metrics | Inspect startup/progress logs and the local optimizer step; do not infer useful training from a live run alone. |
| Last allocation failed in seconds after hours of training elsewhere | Check all attempts with `sacct -D`; the latest startup error may differ from the original interruption. |

Avoid treating `Reason=BeginTime` or a generic final launcher traceback as a
complete diagnosis. Preserve the original error and distinguish scheduler,
infrastructure, data, and model failures.

## 14. Transferring checkpoints through a laptop

Our network path is often **HPC (`peano`) → laptop → workstation (`workstation`)**.
Run both transfers on the laptop, using its SSH host aliases.

Transfer an immutable numbered checkpoint or evaluation snapshot. Avoid directly
transferring a rolling `last.pt` while training may replace it. Preserve a checksum
from the HPC and verify it after the second transfer.

Example using the fixed input snapshot of an evaluation; replace the evaluation
name with the actual one:

~~~bash
# Run on the laptop.
STMEM_EVAL_NAME=teacher_eval_REPLACE_WITH_ACTUAL_TIMESTAMP
STMEM_LOCAL="$HOME/Desktop/chronotree_local/$STMEM_EVAL_NAME"

# Create a checksum for the immutable evaluation input on the HPC.
ssh peano bash -s -- "$STMEM_EVAL_NAME" <<'BASH'
set -euo pipefail
cd /hpc/home/phi/rvalperga/chronotree/runs/evaluation_inputs
cd "$1"
sha256sum checkpoint.pt > SHA256SUMS
BASH

mkdir -p "$STMEM_LOCAL"
rsync -avP \
  "peano:/hpc/home/phi/rvalperga/chronotree/runs/evaluation_inputs/$STMEM_EVAL_NAME/" \
  "$STMEM_LOCAL/"

ssh workstation "mkdir -p /home/rvalperga/chronotree/runs/inspection/$STMEM_EVAL_NAME"
rsync -avP "$STMEM_LOCAL/" \
  "workstation:/home/rvalperga/chronotree/runs/inspection/$STMEM_EVAL_NAME/"

ssh workstation \
  "cd /home/rvalperga/chronotree/runs/inspection/$STMEM_EVAL_NAME && sha256sum -c SHA256SUMS"
~~~

Names here are simple launcher-generated identifiers, not arbitrary shell input.
For a new repository, automate bundling and quote/validate remote arguments.
Include config, checkpoint hash/step, code revision, and relevant evaluation
manifest/results so the receiving machine can identify what it has. A transfer
completing at 100% does not prove the source checkpoint was valid; deserialize and
verify its expected stage/step too. Local inspection launchers may require their
own bundle layout in addition to a valid checkpoint.

## 15. Cost, preparation, and operational boundaries

Compute GPU hours from **allocation elapsed time × allocated GPUs**, including
startup/cleanup and jobs whose GPU was idle during CPU preprocessing. Keep
hardware types separate. Our `hpc/accounting.py` reads the general planner's
submission ledger with `sacct`; dedicated-launcher receipt formats differ.
Its requested profile is not independent proof of the physical GPU model.

Do not invent a walltime estimate from the resource request. Measure representative
startup, step, cache, and evaluation times, including difficult/long examples.
Record unknown cost as unknown.

Full preprocessing and expensive diagnostics belong in compute allocations.
Planning, inspection, and submission belong on the login node. Install/sync the
locked environment and required assets before depending on them on compute nodes;
do not assume network access there or reinstall dependencies in every worker.

Node-local scratch is optional and site-specific. Stage immutable inputs only
when the path, capacity, copy cost, and sharing behavior are known. Keep durable
checkpoints and results on shared storage. Do not invent a scratch mount or change
cluster drivers as part of a training launcher.

## 16. What to implement in another repository

Keep the interface small first; add dedicated launchers when a stage has genuinely
different inputs or lifecycle requirements.

Suggested responsibilities:

| Component | Responsibility |
|---|---|
| `hpc/site.env.example` | Document every supported nonsecret site setting. |
| `hpc/plan.py` | Parse CLI/site/config, validate identities, build jobs, render, explicitly submit. |
| Job/Plan records | Serializable argv, resources, dependency names, outputs, fingerprints, run/attempt paths. |
| Renderer | Literal Slurm directives, quoted paths, fixed working directory/interpreter/import roots. |
| Submission ledger | Atomic per-job receipts and history of attempts; retain submitted jobs if a later submission fails. |
| Startup guard | Revalidate the pinned config/source/input identities before expensive work. |
| Stop wrapper | Signal → unique stop file → coherent application checkpoint; preserve child exit status. |
| Training runtime | Distributed setup, exact sampling/accumulation, checkpoint/resume, progress telemetry. |
| Completion verifier | Required artifacts and hashes; partial completion must not unlock dependencies. |
| Evaluation launcher | Fixed checkpoint/protocol, disjoint shards, resumable results, explicit aggregation. |

A useful preparation/submission transaction is:

~~~text
read supported site settings
resolve experiment configuration and declared CLI overrides
validate names, paths, resources, scientific inputs, and resume compatibility
acquire the experiment lock
check ownership and reject conflicting active attempts
freeze source/config/input identities into a new immutable attempt
render scripts and create log directories
write and print the plan
if not --submit:
    return
for each job in topological order:
    resolve true dependencies and concurrency limits
    call sbatch --parsable with an argv list
    atomically persist the returned job ID immediately
~~~

Use `subprocess.run([...], check=True, capture_output=True, text=True)` for
submission, not an interpolated shell command. Use proper shell quoting when
rendering Bash. Reject newlines in directive values and validate supported
resource/name formats. Do not use `HOME` or other system variables as project
scratch variables.

A submission ledger reduces duplication but is not a perfect transaction with
Slurm: a crash after successful `sbatch` but before receipt persistence needs
reconciliation against the queue. Give attempts recognizable identities and
avoid blindly retrying an ambiguous submission.

Distinguish scientific identity from reviewed runtime knobs. A logging interval
may be safe to change on resume; a dataset, objective, batching convention, or
world-size change may not be. Record runtime changes even when excluded from
scientific signatures. Do not implement “ignore all mismatches” as recovery.

The universal planner should default to render-only. For a useful first version,
support selecting one experiment/stage, a site file, a source config, a run root,
a generated directory, dry-run, submit, and explicit resume. Make later sweeps
bounded manifests, not accidental products of loosely nested shell loops.

## 17. Verification before real cluster experiments

Use CPU-safe tests and fake `sbatch`/`squeue` executables to verify the planner:

1. Dry-run never submits work.
2. Site files are literal, credentials are excluded, and precedence is explicit.
3. Paths with spaces and generated shell scripts are handled correctly.
4. Resource directives are literal and unsupported values fail clearly.
5. Log directories exist before submission.
6. Dependency IDs, array limits, and allocation lanes are correct.
7. Receipts survive partial submission failure.
8. An active job cannot be duplicated or have its inputs rewritten.
9. A changed pinned source/config/checkpoint fails before compute.
10. Resume restores training state and rejects incompatible/missing inputs.
11. Signal handling saves at a coherent boundary and preserves failure status.
12. A partial stop cannot create a successful completion marker.
13. Evaluation snapshots remain fixed, even when `last.pt` changes later.
14. Every planned evaluation episode has one owner; missing/errors are explicit.
15. Genuine resumes retain tracking identity; independent experiments do not.

Validate generated Bash with `bash -n`. Then run bounded device and application
checks on an actual allocated GPU. Our project uses `uv run pytest` for CPU-safe
invariants and `uv run python -m stmem.doctor` for the actual device. Rendering
requires its own simulator/Vulkan check; CUDA availability alone is insufficient.

Source guards, exact scientific inputs, and partial/completion semantics should
remain intact while debugging. A synthetic test verifies implementation behavior,
not real-data scientific performance.

## 18. A concise brief to give Codex in the other repository

> Implement a Slurm workflow using literal site.env settings, scientific YAML
> configurations, and a Python plan.py renderer/submitter. Default to dry-run;
> only --submit launches jobs. Keep source templates under hpc/jobs, generated
> immutable attempts and receipts under hpc/generated, and durable outputs under
> runs. Separate site resources from experiment hyperparameters. Use absolute
> shared paths, a fixed Python environment, typed GPU requests as required by
> the cluster, and one Slurm task spawning one torchrun agent per node.
>
> Give independent experiments separate run roots and tracking IDs. Implement
> pinned input/source/config identities, active-job ownership checks, locks,
> dependency-aware submission, bounded concurrency, atomic checkpoints, graceful
> signal-driven partial stops, and explicit resume. Preserve optimizer/scheduler,
> sampling/RNG state, and provenance. Do not mark partial work complete, bypass
> source mismatches, or silently change a scientific experiment on resume.
>
> Add a separate evaluation launcher that freezes a checkpoint at preparation,
> shards work without duplication, caps videos, writes durable per-item results,
> and reports missing/error outcomes. Provide local logs plus W&B with stable
> resume identities and system metrics. Test dry-run, failure, signal, resume,
> and duplicate-submission behavior before a bounded cluster smoke check.
>
> Prepare code, documentation, and exact commands. Do not submit long training,
> download full datasets, or launch an unbounded sweep without explicit user
> authorization. Treat Chronotree's cluster names and scientific modules as
> examples, not requirements for this repository.

## 19. Chronotree source references

These are useful reading references, not a recommendation to copy every
project-specific branch into a new planner:

- [Site template](../hpc/site.env.example).
- [General planner](../hpc/plan.py): site parsing, config resolution, jobs,
  signatures, dependencies, submission receipts.
- [Direct-template environment helper](../hpc/jobs/common.sh) and
  [teacher template](../hpc/jobs/teacher.sbatch).
- [Signal wrapper](../hpc/run_job.sh) and [stage completion wrapper](../hpc/job_status.py).
- [Checkpoint implementation](../src/stmem/training/checkpoints.py),
  [training runtime](../src/stmem/train.py), and [tracking](../src/stmem/tracking.py).
- [Dedicated compact launcher](../hpc/train_compact.py) and
  [student-study launcher](../hpc/student_utility.py): stricter experiment ownership
  and resume rules.
- [Evaluation launcher](../hpc/evaluate.py): checkpoint snapshots, named plans,
  parallel simulation, and startup checks.
- [Allocation accounting](../hpc/accounting.py).
- [Existing cluster guide](hpc.md) and [scientific contract](../PROMPT_CHRONOTREE.md).
  Older experiment-specific commands can reflect previous cluster policy; the
  current site requirements and emitted plan take precedence over historical examples.
