# Adopt the accepted frozen-vision teacher and cache its encoded evidence

This is the first part of the seed-0 compact-memory experiment. It does not train
a new teacher, compact policy or router. The new entry point is
[`hpc/cache_teacher.py`](../hpc/cache_teacher.py). It reads site paths without
sourcing the `.env` file, adopts the exact evaluated snapshot, and renders one
Slurm task array. Only `--submit` contacts Slurm.

## Resources and data

The defaults are 16 array elements (one per configured task), at most four
simultaneously. Each requests one GPU using **`--gres=gpu:1`**, eight CPU cores,
64 GB host RAM, and 24 hours in `gpuq`. There is no `--exclusive` directive, GPU
type, or distributed training launch. Four CPU threads are used inside each
encoder process. These are independent inference jobs with bounded descriptor
loads, so there is no reason to reserve four GPUs for one task's encoder.

The existing `hpc/site_b200_4gpu.env` supplies shared paths and optional account,
QOS, constraint and activation settings. Its GPU count, partition, training run
root and resource defaults do not override this launcher's explicit settings.
Node-local staging is disabled. The existing prepared point clouds and frozen
CLIP sidecars are reused; no download or point-cloud preparation is launched.

## Commands on the HPC

After pushing these changes and pulling them on the HPC:

```bash
cd /hpc/home/phi/rvalperga/chronotree

.venv/bin/python hpc/cache_teacher.py \
  --site hpc/site_b200_4gpu.env \
  --evaluation runs/evaluation/eval_20261102T120621Z_frozen_10hz \
  --name memory_fv_seed0 \
  --dry-run
```

The named evaluation is the one associated with the transferred frozen-vision
checkpoint at step 104,110. Its expected checkpoint hash is
`eb57ccd4c131888a5f5d6225a3a0cbe5cd14b3afc5c6ff0a1a17ee319cd70918`.
The launcher obtains the snapshot path from the evaluation manifest, rather than
following a mutable training `last.pt`. If the accepted result is in a different
directory, supply that exact directory instead; the entire evaluation evidence
must be present, not just `summary.json`.

The dry-run validates and copies the selected checkpoint and evaluation evidence.
It then writes `hpc/generated/memory_fv_seed0/plan.json`, `cache.yaml`, and
`cache.sbatch`. Check the printed `accepted_macro_success` (expected 0.3125),
checkpoint step/hash, task list and `evaluation_execution` (eight executed targets,
16 native simulator commands). Submit that plan with:

```bash
.venv/bin/python hpc/cache_teacher.py \
  --site hpc/site_b200_4gpu.env \
  --evaluation runs/evaluation/eval_20261102T120621Z_frozen_10hz \
  --name memory_fv_seed0 \
  --submit
```

This command submits only the cache array. It prints its Slurm array ID. Training
and later evaluation jobs are not queued. To check progress:

```bash
squeue -u "$USER"

.venv/bin/python hpc/cache_teacher.py \
  --site hpc/site_b200_4gpu.env \
  --name memory_fv_seed0 \
  --status
```

Status reads the saved plan directly and is unaffected by an inherited training
`STMEM_RUN_ROOT`. It reports the cache directory, completed tasks and missing tasks.
It is a filesystem progress view; cache construction verifies content hashes.

Logs are in `hpc/generated/memory_fv_seed0/logs/`, with names
`stmem_cache_memory_fv_seed0_<array-id>_<task-index>.out` and `.err`. Standard output
contains startup, per-episode completion/reuse, per-task completion and final
cache-completion JSON events. This preprocessing stage does not create training
or evaluation W&B runs.

Once status is `complete` with no missing tasks, continue with the
[shared compact-policy training launcher](compact_training_hpc.md). It derives
the next configuration from this saved cache configuration and reuses its teacher.

If an array stops before completion, wait until all its elements are inactive,
then use the same submission command with `--resume`. Completed episodes and
task shards are reused after integrity checks. The launcher checks the Slurm
array base ID and refuses a duplicate submission while any element remains active.
The 24-hour request can be changed with `--walltime`; concurrency can be changed
with `--max-concurrent`. Neither changes the scientific cache identity.

Retain the original evaluation directory and its checkpoint snapshot for launcher
retries. Keep source files unchanged while array tasks are queued or running;
startup and construction guards reject mixed implementations.

## Outputs and scientific contract

The new run root is `runs/memory_fv_seed0/`, independently of the site's old
training run root:

- `seed0/teacher/selected.pt`: identical checkpoint bytes, never a rewritten model.
- `seed0/teacher/selected.selection.json`: explicit user acceptance, checkpoint
  hash, recomputed success statistics, original training configuration hash,
  accepted evaluation configuration and execution override.
- `seed0/teacher/selected.evaluation/`: immutable manifests, per-episode outcomes,
  summary and completion evidence; videos are not copied.
- `seed0/cache/cache_location.json`: points to the content-addressed cache directory.
- `seed0/cache/complete.json`: published only once all configured task shards verify.

Cache shards live under `STMEM_CACHE_ROOT/<identity-hash>/`, normally
`/hpc/home/phi/rvalperga/data/chronotree-cache/<identity-hash>/`, not in the prepared
dataset directory. The directory contains one HDF5 shard and receipt per task,
plus `manifest.json` after full completion. While constructing a task, at most
one atomic per-episode commit per requested episode is retained in its private
work directory; these are removed after the final task shard is published.

The trained checkpoint retains its original two-target execution setting.
The accepted evaluation's eight-target setting is recorded separately for later
matched closed-loop comparisons. Encoding cadence, no-proprioception/no-command
inputs, CLIP identity, geometry, tokenization, horizon and normalization remain
the teacher's exact settings. Frozen encoding uses FP32, as declared in the cache
identity. Parent aggregation and reconstruction preserve causal leaf coverage.

Completed blocks and every queried incomplete historical prefix are encoded
separately. Cached current observations are supplied once, outside history.
Lazy CLIP rows are hydrated only for one local block/current frame at a time,
through reused file mappings, rather than materializing a descriptor episode.
Task array elements keep the complete dataset identity; they do not create
incompatible per-task configuration hashes by narrowing `data.tasks`.

Cache format 2 records an encoding-code fingerprint. It conservatively covers
encoded-content producers while excluding evaluation orchestration, inspection
tools and report rendering. The 2026-11-03 cached-training update additionally
excludes four exact training-consumer paths through an audited compatibility
record; all other protected source bytes must match before an old cache identity
is retained. This permits worker-side reading without another encoding pass.
See the [compact training verification](compact_training_hpc.md) for that record
and measured checks. The final manifest separately records the full repository
source revision for audit.

Checkpoint adoption verifies completed official all-task FULL validation, exact
checkpoint identity, the declared execution override, each episode and worker,
and recomputes success from saved outcomes. It rejects missing/error episodes or
a summary that disagrees with the records. It records explicit acceptance; it
does not claim to have run the original two-candidate teacher selection.

## Measured local checks

The focused CPU suite passed 223 tests covering adoption and tamper rejection, immutable launch configuration,
untyped GPU/task-array rendering, duplicate submission, interruption/recovery,
partial-cache publication, descriptor hydration and streaming parity. Synthetic
fixtures remain explicitly diagnostic and are never accepted as a real teacher.
Two additional CPU integration checks passed for compact and Perceiver training
consuming the new cache with and without background loading workers.

A bounded real-data GPU probe used the transferred 104,110-step teacher and one
local VideoUnmask episode (173 native execution frames). Cache writing took
about 1.2 seconds on the local RTX 5000 Ada and produced a 21,786,386-byte diagnostic
file. Eight decisions, including incomplete 10 Hz blocks, matched online node
IDs, counts, completion flags and timestamps. Leaf contents, current tokens and
all content sums were bit-exact. Parent means differed by at most 2.98e-8 due to
CPU/GPU FP32 count division. The initial zero-tolerance check caught that rounding
difference; the final check used explicit FP32 tolerance for parent means while
retaining exact leaf/current/sum checks. This is a bounded parity check, not an
HPC throughput estimate or full-dataset cache build. The device doctor and
architecture snapshot check passed; the model architecture was unchanged.

No Slurm job was submitted from the local workspace. Full-array runtime, storage
and GPU-hours remain unmeasured; task logs and final shard receipts provide them
after the HPC run.
