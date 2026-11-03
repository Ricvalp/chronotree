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
  retained. No data preparation or cache reconstruction is requested.
- Initial learning rate: 1e-4, 1,000-step warmup followed by the existing cosine
  schedule; schedule horizon explicitly resets to 30,000 compact updates.
- Effective batch 128; microbatch 32 per GPU. With two GPUs this is two
  accumulation microbatches per rank per optimizer update.
- Two GPUs in `gpuq`, untyped `--gres=gpu:2`, no `--exclusive`, 32 CPU cores,
  128 GB host RAM, 48 hours. One Slurm task starts one torchrun agent with two
  training processes. Four loading workers per rank, prefetch factor one.
- W&B online logging and system statistics enabled; scalar metrics every step,
  attention diagnostics every 10 steps. The existing compact attention probe
  uses a fixed UNIFORM context, not the randomly sampled training frontier.
- Resumable `last.pt` every 1,000 steps and on graceful stop; immutable selection
  candidates at 15,000 and 30,000. One run appears as job type `compact` in the
  same W&B project.

This is a reasonable initial allocation, **not a measured throughput or memory
result**. Cached encoder outputs remove encoder training, but workers still load
and pack input histories, and each rank reconstructs cached archives per example.
Data loading can still dominate; inspect measured step times before scaling.

The selected teacher's stored `model.execute: 2` remains unchanged. Its accepted
evaluation used the explicit eight-target execution override. That override is
recorded in the launch plan and must be used for matched downstream evaluation.
The compact configuration records four validation episodes per task for later
selection. This command does not submit those evaluations or router training.

## Submit on the HPC

Push the implementation to the shared Git remote, then on the HPC:

```bash
cd /hpc/home/phi/rvalperga/chronotree
git pull

.venv/bin/python hpc/train_compact.py \
  --site hpc/site_b200_4gpu.env \
  --cache-name memory_fv_seed0 \
  --dry-run
```

Check `cache_path`, `teacher_step`, `gres: gpu:2`, `partition: gpuq`, and the
training settings in the printed plan. Then:

```bash
.venv/bin/python hpc/train_compact.py \
  --site hpc/site_b200_4gpu.env \
  --cache-name memory_fv_seed0 \
  --submit
```

No environment sourcing or new `.env` file is necessary. Despite the existing
site filename, this launcher uses its interpreter, repository location, optional
activation/account/QOS/constraint and W&B destination. Paths come from the saved
cache configuration; allocation defaults above override the site's old B200
training allocation. The script explicitly exports all three saved roots so an
inherited `STMEM_RUN_ROOT` cannot redirect this experiment.

Output:

- Generated configuration/script/plan: `hpc/generated/memory_fv_seed0_compact/`.
- Slurm logs: `hpc/generated/memory_fv_seed0_compact/logs/` with `%x_%j.out/.err`.
- Training metrics, W&B identity and weights: `runs/memory_fv_seed0/seed0/compact/`.
- Selection candidates: `step_00015000.pt` and `step_00030000.pt` in that directory.
- Accepted teacher and cache remain at their existing paths; nothing overwrites
  them. Only the compact stage receives new model weights and optimizer state.

## Resume and integrity

After the job has stopped and saved `last.pt`:

```bash
.venv/bin/python hpc/train_compact.py \
  --site hpc/site_b200_4gpu.env \
  --cache-name memory_fv_seed0 \
  --resume --submit
```

Resume uses the compact checkpoint, optimizer, sampler, schedule and W&B identity,
not the teacher's optimizer. Keep the original GPU count and batch/training
configuration. If nondefault options were used initially, pass the same values.
Walltime, memory and CPU allocation may be changed without changing scientific
inputs. Active submitted jobs are rejected before resubmission.

If an allocation fails **before any checkpoint exists**, use
`--restart-unstarted --submit` instead of `--resume --submit`. This explicitly
retries from step zero and refuses to overwrite any existing `.pt` checkpoint.
It retains the run directory's W&B identity. Never retry while the old allocation
is still active.

The launcher verifies teacher/configuration/selection hashes, the final cache
manifest hash, cache identity (including encoding code), task coverage, shard
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

Only HPC orchestration, tests and documentation changed. The encoder-cache code
fingerprint is unchanged, so the completed cache remains compatible. No model or
tensor shapes changed; the teacher architecture diagram needs no content changes.

## Local verification

CPU launch tests use explicitly synthetic serialized fixtures and cover scientific
configuration preservation, immutable inputs, hash/coverage failures, shell
quoting, duplicate submissions, checkpoint resume and startup guards. The existing
cached compact/Perceiver training tests exercise the actual training path on tiny
synthetic diagnostic inputs. These are implementation checks, not benchmark
measurements or training results. No HPC job was submitted from this workspace.

On 2026-11-03, 39 CPU checks passed (37 launcher tests plus two cached
compact/Perceiver integration tests, with and without background workers).
Ruff, `git diff --check`, and the architecture diagram snapshot check passed.
`uv run python -m stmem.doctor` passed FP32, BF16 and FP16 compute checks on the
local NVIDIA RTX 5000 Ada (32 GB). This does not establish HPC throughput.
