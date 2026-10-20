# Teacher memory profiling

The standalone profiler imports the existing teacher, training data loader, loss,
optimizer and precision selection. It does not change the normal training loop or
model, resume an existing run, or write teacher checkpoints. Its output and W&B run
are separate from training. Real data still requires the verified revision-level
alignment record used by training.

The default job runs **10 warmup optimizer updates**, **5 measured updates**, then
**1 separately instrumented update**. Warmup may be set to 5–10 updates. It includes
forward, backward and optimizer updates so AdamW state is allocated before
measurement. The original training schedule remains in the resolved configuration;
the profiling budget does not replace the teacher's 100,000-update schedule.

## Run on the HPC

Run these commands from the repository root after syncing the changes. The launcher
reads the site file itself; it does not require sourcing it first. Its Python must
be the environment in which the project dependencies are installed.

For the current four-GPU setup:

```bash
.venv/bin/python hpc/profile_memory.py \
  --site hpc/site_4gpu.env --profile h200_4gpu \
  --name teacher_4gpu_memory --dry-run
```

Inspect the printed command, resources and these files:

```text
hpc/generated/profiles/teacher_4gpu_memory/config.yaml
hpc/generated/profiles/teacher_4gpu_memory/profile_teacher.sbatch
hpc/generated/profiles/teacher_4gpu_memory/plan.json
```

Then explicitly submit the one job:

```bash
.venv/bin/python hpc/profile_memory.py \
  --site hpc/site_4gpu.env --profile h200_4gpu \
  --name teacher_4gpu_memory --submit
```

For one GPU, use `--site hpc/site.env --profile h200_1gpu` and a new name, such as
`teacher_1gpu_memory`. The defaults come from `configs/teacher.yaml`: effective
batch 128, microbatch cap 64 and encoder block batches of 128. Four GPUs use one
pass of 32 decisions per rank; one GPU uses two accumulation passes of 64.
Use `--config PATH` to match a different training configuration,
or explicitly override `--microbatch` and `--effective-batch` for a later comparison.
Profiles control allocations, not GPU memory pooling: each rank holds its own model.
The recorded H200 profile in [status.md](status.md) predates this configuration:
it used effective batch 64, microbatch 16 and encoder block batches of 32.
Its measured timings and memory peaks do not describe the new settings.

Each submitted name is reserved. Choose a new `--name` to repeat a run; omitting it
generates a unique UTC timestamp and suffix. There is no automatic submission,
resume, deletion or teacher completion marker. The default walltime is two hours;
`--walltime HH:MM:SS` changes only this job's allocation limit. The renderer preserves
site CPU, memory, partition, account, QoS, constraint and GPU type settings.

## Representative and long histories

The default `--scenario representative` follows the normal task-balanced training
sampling. For a separate stress check, use:

```bash
.venv/bin/python hpc/profile_memory.py \
  --site hpc/site_4gpu.env --profile h200_4gpu \
  --name teacher_4gpu_long_history --scenario long-history --dry-run
```

Inspect it and rerun with `--submit` when ready. `long-history` selects the latest
valid decisions from the longest 10% of training episodes **within each task**,
using episode-length metadata for selection and preserving task balance. It never
truncates histories or draws from official validation/test seeds. This is a selected
stress diagnostic, not an estimate of average training throughput or the maximum
possible configuration length.

Use `--warmup-steps 5` for the lower warmup bound, `--steps N` for 1–100 measured
updates, and `--trace-steps 0` to omit the additional detailed trace. A single job
measures one scenario; the long-history command is not submitted automatically.

## Outputs and interpretation

Results are written to:

```text
$STMEM_RUN_ROOT/profiles/<name>/
  config.json
  manifest.json
  summary.json
  rank0/
    steps.jsonl
    steps.csv
    phases.jsonl
    phases.csv
    measurements.json          # Saved before the optional detailed trace
    summary.json
    trace.json                 # Only with --trace-steps 1
    memory_snapshot.pickle     # CUDA allocation history from the traced update
  rank1/ ...                    # One directory per GPU rank
```

The final summary combines all ranks. Per-step memory peaks, live allocated and
reserved bytes, device memory, timing, history lengths and padding distinguish a
typical batch from a longer one. Phase measurements separate loader waits,
action transfer/flow-target construction, history and current encoding, text
encoding, metadata adaptation, decoder work, backward and optimizer work.
Point-cloud transfers are included in their encoder phases; other context assembly
allocations are included in the whole-update peak. Detailed tracing adds operation
names and allocation history on its separate update. Phase peaks include tensors
already live when that phase begins, so they must not be added together.
The allocation-history ring buffer retains at most 100,000 events.

If an update fails, `rankN/failure.json` records the update, phase, error and
available memory counters. A CUDA OOM also triggers a best-effort
`rankN/oom_snapshot.pickle`, including when it occurs during warmup or ordinary
measurement. Baseline OOM snapshots show allocator state; allocation-event history
is recorded only while detailed tracing is active. Completed JSONL records remain available, and
`measurements.json` preserves the baseline summary if the later trace fails.
Trace-export failures are recorded separately in `trace_export_errors.json`;
failed jobs do not receive a successful global summary.

The profiler creates a dedicated W&B run with job type `teacher-memory-profile`,
using the site's existing project/entity and online/offline setting. Offline runs
remain available to sync later. System metrics remain enabled according to the
configuration. Tracking is required for profiling as it is for training.

Detailed traces and memory snapshots are diagnostic artifacts and can be large.
Open `trace.json` with a Chrome trace viewer or Perfetto and
`memory_snapshot.pickle` with [PyTorch's memory visualizer](https://pytorch.org/memory_viz).
The memory snapshot describes allocations visible to PyTorch's allocator; driver,
NCCL and other external allocations also consume GPU memory. Reserved memory
includes reusable allocator cache. See the
[PyTorch 2.9 memory documentation](https://docs.pytorch.org/docs/2.9/torch_cuda_memory.html).
A measured margin applies to the observed
batches and GPU, not to every possible longer history.

## Local verification

CPU diagnostics use only an explicitly synthetic/debug configuration and do not
establish GPU memory headroom. They run the script directly, outside Slurm:

```bash
uv run python scripts/profile_teacher_memory.py \
  --config configs/debug/teacher.yaml --cpu \
  --output runs/profiles/cpu_profile_check \
  --warmup-steps 5 --steps 1 --trace-steps 0
```

Before a GPU diagnostic, run `uv run python -m stmem.doctor` on the actual device.
Launcher tests do not allocate GPUs or submit jobs:

```bash
uv run pytest tests/test_profile_launcher.py tests/test_profile_teacher_memory.py
```
