# Shared compact policy after the accepted teacher cache

`hpc/train_compact.py` launches Stage P only. It derives the complete configuration
from `hpc/generated/memory_fv_seed0/cache.yaml`, rather than the generic
`configs/compact.yaml`. The user reported all 16 task shards complete at
`/hpc/home/phi/rvalperga/data/chronotree-cache/3db7b7ce80926d21659863c3`.
This remote status was supplied by the user; the local workspace has not scanned
the HPC cache itself.

## Experiment and allocation

The decoder and context adapter start from the accepted 104,110-step teacher,
with a **new optimizer and 30,000-step learning-rate schedule**. Visual and task
encoders stay frozen. Historical and current encodings come from the completed
cache. This is one shared compact policy, not separate TREE and FLAT policies.

- Mixture: 45% random TREE, 45% FLAT plus residuals, 5% UNIFORM, 5% CURRENT.
- History budgets: 64, 128, 256; intermediate sizes enabled. Current tokens are
  separate from the historical budget. There is no learned router yet.
- Teacher geometry, CLIP features, absence of proprioception and previous
  commands, causal history, horizon 16, and 10 Hz action/observation cadence are
  retained. No data preparation or new encoder-cache build is requested.
- Initial learning rate: 1e-4, 1,000-step warmup followed by the existing cosine
  schedule; schedule horizon explicitly resets to 30,000 compact updates.
- Effective batch 128; microbatch 32 per GPU. With two GPUs this is two
  accumulation microbatches per rank per optimizer update.
- Two GPUs in `gpuq`, untyped `--gres=gpu:2`, no `--exclusive`, 64 CPU cores,
  128 GB host RAM, 48 hours. One Slurm task starts one torchrun agent with two
  training processes. Sixteen loading workers per rank, prefetch factor one.
- W&B online logging and system statistics enabled; scalar metrics every step,
  attention diagnostics every 10 steps. The existing compact attention probe
  uses a fixed UNIFORM context, not the randomly sampled training frontier.
- Resumable `last.pt` every 1,000 steps and on graceful stop; immutable selection
  candidates at 15,000 and 30,000. One run appears as job type `compact` in the
  same W&B project.

The original compact run, Slurm job 66922, reached approximately 317 steps after
nine hours: its last reported updates took 92–108 seconds. Its heartbeat repeatedly
reported `cache_loading`. Cache loading and tree construction ran serially in each
training process, after the already-prefetched raw observation batches arrived.
The user chose a fresh restart rather than preserving that partial checkpoint.

The first worker update moved cache reading, tree construction and context selection
into background processes. The user reports that update settled around 13 seconds
per optimizer update. It still rebuilt many deterministic parents when random
sampling missed its bounded in-process cache.

The new path uses a persistent derived hierarchy cache. Build it once with
[`hpc/cache_hierarchy.py`](../hpc/cache_hierarchy.py) before starting this fresh
compact run. It derives parents and causal-prefix trees from existing encoded
leaves; it does not prepare point clouds or run the teacher encoder again. The
loader still samples the same random TREE/FLAT contexts and budgets. FLAT residuals
continue to exclude the selected leaves exactly. Read-time ages are computed at
the actual decision. Precomputation changes neither grouping nor training diversity.
Workers return selected contiguous tensors, and both in-process reuse and prefetch
remain bounded. The new HPC throughput remains to be measured.

The selected teacher's stored `model.execute: 2` remains unchanged. Its accepted
evaluation used the explicit eight-target execution override. That override is
recorded in the launch plan and must be used for matched downstream evaluation.
The compact configuration records four validation episodes per task for later
selection. This command does not submit those evaluations or router training.
Once both candidates exist, follow [matched TREE/FLAT checkpoint selection](compact_checkpoint_selection.md)
to submit the four parallel evaluations and record the selected compact weights.

## Prepare the reusable hierarchy once

Use the same accepted teacher cache name. This CPU preprocessing reserves one
untyped GPU per task because this cluster provides the required compute cores on
GPU nodes. Defaults: 16 task elements, four simultaneous, `gpuq`, four CPU cores,
32 GB RAM and 24 hours each. It does not start training or create a W&B run.

```bash
cd /hpc/home/phi/rvalperga/chronotree
git pull

.venv/bin/python hpc/cache_hierarchy.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv_seed0 --dry-run

.venv/bin/python hpc/cache_hierarchy.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv_seed0 --submit
```

Check completion before training:

```bash
.venv/bin/python hpc/cache_hierarchy.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv_seed0 --status
```

`--status` only reads filesystem progress; it creates no directories and does not
contact Slurm. Preparation logs are under
`hpc/generated/memory_fv_seed0_hierarchy/logs/`. The printed
`hierarchy_cache_root` is a content-addressed directory beneath the existing leaf
cache's `hierarchy/` directory. The original leaf files remain dependencies; keep
them in place because hierarchy virtual datasets reference their contents.

If an array is interrupted, wait until every element is inactive, then repeat
its command with `--resume --submit`. Verified completed work is reused. Never
change source or generated configuration while jobs are queued or running. A new
builder version has a different derived-cache identity; it does not invalidate
the frozen encoded leaves.

## Submit on the HPC

Push the implementation to the shared Git remote, then on the HPC:

```bash
cd /hpc/home/phi/rvalperga/chronotree
git pull

.venv/bin/python hpc/train_compact.py \
  --site hpc/site_b200_4gpu.env \
  --cache-name memory_fv_seed0 \
  --name memory_fv_seed0_compact_hierarchy \
  --run-root runs/memory_fv_seed0_compact_hierarchy \
  --dry-run
```

Check `cache_path`, `hierarchy_cache_root`, `teacher_step`, `gres: gpu:2`,
`partition: gpuq`, `cpus: 64`, and the training settings in the printed plan. Then:

```bash
.venv/bin/python hpc/train_compact.py \
  --site hpc/site_b200_4gpu.env \
  --cache-name memory_fv_seed0 \
  --name memory_fv_seed0_compact_hierarchy \
  --run-root runs/memory_fv_seed0_compact_hierarchy \
  --submit
```

No environment sourcing or new `.env` file is necessary. Despite the existing
site filename, this launcher uses its interpreter, repository location, optional
activation/account/QOS/constraint and W&B destination. Data/cache paths come from
the saved cache configuration; `--run-root` selects only the new training output.
Allocation defaults above override the site's old B200 training allocation. The
script explicitly exports all three resolved roots so an inherited
`STMEM_RUN_ROOT` cannot redirect this experiment. The original immutable teacher
dependency remains an absolute path under `runs/memory_fv_seed0/`.

Output:

- Generated configuration/script/plan: `hpc/generated/memory_fv_seed0_compact_hierarchy/`.
- Slurm logs: `hpc/generated/memory_fv_seed0_compact_hierarchy/logs/` with `%x_%j.out/.err`.
- Training metrics, W&B identity and weights: `runs/memory_fv_seed0_compact_hierarchy/seed0/compact/`.
- Selection candidates: `step_00015000.pt` and `step_00030000.pt` in that directory.
- Accepted teacher and cache remain at their existing paths; nothing overwrites
  them. The old slow compact run under `runs/memory_fv_seed0/seed0/compact/` is also
  retained. The new output root creates a fresh W&B run and optimizer state.

## Resume and integrity

After the job has stopped and saved `last.pt`:

```bash
.venv/bin/python hpc/train_compact.py \
  --site hpc/site_b200_4gpu.env \
  --cache-name memory_fv_seed0 \
  --name memory_fv_seed0_compact_hierarchy \
  --run-root runs/memory_fv_seed0_compact_hierarchy \
  --resume --submit
```

Resume uses the compact checkpoint, optimizer, sampler, schedule and W&B identity,
not the teacher's optimizer. Keep the original GPU count and batch/training
configuration. If nondefault options were used initially, pass the same values.
Walltime, memory and CPU allocation may be changed without changing scientific
inputs. Active submitted jobs are rejected before resubmission. Existing generated
plans and checkpoint configurations are immutable: the launcher never changes an
old run to the new worker count or hierarchy path. `--legacy-cache` explicitly
retains on-demand hierarchy construction; reproducing an old launch also requires
its original `--name`, `--run-root`, GPU count, batch settings and worker count.
A changed source signature is rejected rather than silently resuming different
code. The commands above create a separate fresh run.

If an allocation fails **before any checkpoint exists**, use
`--restart-unstarted --submit` instead of `--resume --submit`. This explicitly
retries from step zero and refuses to overwrite any existing `.pt` checkpoint.
It retains the run directory's W&B identity. Never retry while the old allocation
is still active.

The launcher verifies teacher/configuration/selection hashes, the final cache
manifest hash, hierarchy manifest/hash and source-cache dependency, cache identity
(including encoding code), task coverage, shard
sizes and receipt consistency. It does not repeat a full byte scan of every HDF5
shard; those hashes were verified by cache construction before final publication.
The trainer independently checks alignment, the dataset identity, teacher input
compatibility, and cache identity when it opens the dataset. All these checks
remain enabled.

Each rendered attempt has an immutable startup-plan snapshot. A later dry-run
cannot change the plan used by a previously submitted script. The generated
compact YAML is immutable, and output ownership prevents two differently named
launch plans from writing the same compact directory. A completion marker prevents
accidentally starting another 30,000-step run over completed results.

The original encoded leaves and causal prefix variants are reused. Runtime-only
cached-data loading changes do not justify another encoding pass; cache
compatibility remains checked against the encoder/preprocessing implementation.
Model architecture, causal boundaries, exact leaf coverage and TREE/FLAT
supervision remain unchanged.

## Local verification

The persistent-hierarchy launcher update passed 60 CPU launch tests on
2026-11-03. These include a production hierarchy build over 16 tiny synthetic
task shards, followed by compact planning, allocation guards and checkpoint
resume with the hierarchy manifest dependency. Status, immutable inputs, active
array rejection, shell syntax, resources and isolated output roots are covered.
Ruff and the owned-file whitespace check passed. No HPC submission was made;
these tests do not measure training throughput or scientific success.

CPU launch tests use explicitly synthetic serialized fixtures and cover scientific
configuration preservation, immutable inputs, hash/coverage failures, shell
quoting, duplicate submissions, checkpoint resume and startup guards. The existing
cached compact/Perceiver training tests exercise the actual training path on tiny
synthetic diagnostic inputs. These are implementation checks, not benchmark
measurements or training results. No HPC job was submitted from this workspace.

Before this pipeline update on 2026-11-03, 39 CPU checks passed (37 launcher tests plus two cached
compact/Perceiver integration tests, with and without background workers).
Ruff, `git diff --check`, and the architecture diagram snapshot check passed.
`uv run python -m stmem.doctor` passed FP32, BF16 and FP16 compute checks on the
local NVIDIA RTX 5000 Ada (32 GB). This does not establish HPC throughput.

### Cached-context worker update, 2026-11-03

108 focused CPU cases passed across cached-context selection, token/gradient
parity, two-rank distributed training, deterministic stop/resume, existing
teacher/Perceiver prefetch, cache compatibility and launch checks. The initial
combined run passed 106 cases; two additional action-alignment/retry cases passed
in the loader suite afterward. A separate synthetic GPU forward/backward probe
using pinned CPU token banks matched the reference predictions and all 43
trainable gradient tensors bit-exactly in both FP32 and BF16. No optimizer updates
were performed in that GPU probe. The local device doctor passed all precisions.

The reproducible read-only benchmark is `scripts/benchmark_compact_loader.py`.
It uses the previously encoded real VideoUnmask training episode and explicitly
does not constitute a training experiment or benchmark-policy evaluation. For
128 examples at microbatch 16 with **block reuse disabled**:

| Input pipeline | Preparation/consumption time |
| --- | ---: |
| Original serial cache path | 15.67 seconds |
| New, zero workers | 14.48 seconds |
| New, four workers | 4.94 seconds |
| New, eight workers | 3.22 seconds |

There were zero block-cache hits in these new-loader runs. Timings include the
first-batch wait and exclude neural computation and worker shutdown; the
eight-worker total including shutdown was 3.56 seconds. Selected tensors,
targets, state, clocks and history counts matched the reference bit-exactly on
32 distinct real decisions. Maximum selected tensor payload was 5.19 MB per
microbatch. Local reports are
`runs/compact_loader_diagnostic_20261103/report_no_reuse_parallel.json` and
`runs/compact_loader_diagnostic_20261103/report.json`.

The separate repeated-episode benchmark was much faster with block reuse, but
does not represent random sampling across 1,600 training episodes. Even the
no-reuse test uses one local episode, warm OS filesystem caches and local
storage. Neither measures HPC shared-storage behavior or complete optimizer
step latency. The initial dataset/alignment/index validation is unchanged; its
roughly 16-minute cost in the stopped HPC run may remain.

To assess the replacement run, compare `train/step_seconds_max_rank` and
`train/data_wait_seconds_max_rank`. Worker effort is reported separately as
`train/cache_worker_seconds_rank`, `train/context_worker_seconds_rank` and
`train/data_worker_pack_seconds_rank`. These are sums of overlapping background
work and must not be added to the wall-clock step time. Block hit/miss counters
show whether reuse helps on the real task mixture. GPU system metrics and
attention diagnostics remain enabled.

### Persistent hierarchy update, 2026-11-03

146 focused CPU cases passed, including the production sidecar writer/reader,
exact coverage and prefix parity, interrupted construction recovery, spawned
workers, two-rank DDP, deterministic checkpoint/resume, launch guards and existing
encoded-cache invariants. The local device doctor passed FP32/BF16/FP16. The
updated architecture diagram was regenerated, checked and visually inspected.

The real one-episode diagnostic used the previously encoded VideoUnmask episode,
with a copied leaf shard and explicitly diagnostic manifests under
`runs/hierarchy_cache_diagnostic_20261103/`. It did not run an encoder or update a
model. For 128 examples at microbatch 16 with **topology reuse disabled**:

| Input preparation | Time |
| --- | ---: |
| Original serial reconstruction and packing | 16.72 seconds |
| Indexed reader, zero workers | 2.53 seconds |
| Indexed reader, eight workers | 1.14 seconds |

The indexed runs had zero topology-cache hits and 122 misses; CURRENT examples
did not load an index. Timings include the first-batch wait, exclude neural work
and shutdown, and use warm OS caches on local storage. The eight-worker total
including shutdown was 1.50 seconds. It has only eight batches, so it does not
establish steady-state throughput. With metadata reuse enabled the zero-worker
run took 1.04 seconds, but that repeated-episode workload favors reuse.

The separate 32-distinct-decision check matched selected content, metadata,
targets, clocks, state and history counts **bit-exactly**. The original 21,786,386
byte leaf file gained a 19,152,360 byte hierarchy file; combined storage was
40.94 MB. Leaf payloads remain referenced in the original file. These counts
are for one episode, not an extrapolated full-dataset storage promise.
Reports: `benchmark.json` and `benchmark_no_index_reuse.json` in the diagnostic
directory. Reproduce by supplying `--hierarchy-root` to
`scripts/benchmark_compact_loader.py`; `--disable-index-reuse` disables only the
in-process metadata cache for the diagnostic.

On the HPC, compare the same step/wait maxima as before. New worker metrics are
`train/hierarchy_read_seconds_rank`, `train/hierarchy_select_seconds_rank`,
`train/hierarchy_residual_seconds_rank`, `train/hierarchy_read_bytes_rank`, and
`train/hierarchy_index_hits_rank`/`train/hierarchy_index_misses_rank`. Read time
covers array retrieval and decoding; byte counts are requested logical bytes,
not physical filesystem traffic. Index hits refer to episode topology, unlike
the earlier block-tree hit counter. Background effort times overlap and must
not be added to step wall time. Full-dataset HPC throughput and the effect of
16 workers per GPU remain unmeasured locally.

The cache compatibility migration excludes exactly `train.py`,
`training/loader.py`, `training/cached_loader.py`, `training/cached_policy.py`,
`training/packed_context.py` and `data/hierarchy_cache.py` from encoded-content
identity. The last two consume frozen leaves; the parent builder is separately
fingerprinted by the derived hierarchy identity, and training records the full
runtime source provenance. It preserves the old identity only when every
other protected source byte matches audited commit `21d0b24`, plus a checked
digest of the fingerprint implementation itself. Changes to encoder weights,
geometry, preparation, alignment, aggregation or unknown source files still
invalidate compatibility. Using the accepted checkpoint and its reported data
identity, the current implementation reproduces the exact existing cache suffix
`3db7b7ce80926d21659863c3`. Full runtime source provenance is recorded separately.
