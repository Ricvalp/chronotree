# Refinement utility study with the accepted 140k teacher

On 11 November 2026 the user selected the 140,000-step teacher and requested
three new matched students, keeping the old S0 separate. The completed
[FULL evaluation](https://forge.coreweave.com/wandb/ricvalp/chronotree/runs/a9e2ab943ccc6d04)
reports 25 successes in 64 episodes, macro success **0.390625**. The accepted
rollout executes eight of sixteen predicted targets, holds each for two native
20 Hz control steps, and uses eight Euler integration steps. This is a small
validation estimate, not an official test result.

The exact evaluated checkpoint SHA256 is
`d17dee6cfe70caae8ff8d1013b0fe50aee01d1cd816feda4a25f0f4d388c0d34`.
The new protocol pins this hash and step 140000. It does not use an advancing
`last.pt`. The accepted evaluation directory is
`runs/evaluation/teacher_latest_execute8_20261111T102411Z` and its fixed snapshot
is under `runs/evaluation_inputs/teacher_latest_execute8_20261111T102411Z/`.

## What changes

- **S_STD:** original compact context sampler, the matched baseline.
- **S_STAGE:** random single-move routing prefixes and the proposed stage/budget
  exposure schedule.
- **S_PACKET:** the same stage/budget/prefix exposure, with coherent allocation
  packets instead of independent single moves.

All three start with identical 140k teacher weights and fresh optimizers. They
train the action decoder, read adapter and clock components for 100k updates;
visual/task encoders remain frozen. Effective batch is 128, microbatch 32 per
GPU, with two GPUs and 16 loader workers per GPU. LR is 1e-4, with 1k warmup and
cosine decay to 1e-5 at 100k. Recovery `last.pt` is saved every 1k and immutable
numbered snapshots every 5k. Earlier snapshots are development-only; 100k is the
predeclared primary endpoint.

The old 90k student was trained with the 104k teacher's learned encodings.
It cannot be inserted into this matched comparison or evaluated using the new
cache. It remains in the original study. `S_INIT` uses the exact new teacher
weights under compact contexts solely for the smoke check; it has **zero compact
training updates**, is not a trained student baseline, and is excluded from the
primary matrix, generated-action selection and router fitting.

Prepared point clouds and frozen CLIP descriptors are reusable. Learned encoded
tokens and their hierarchy must be built for this teacher. They have new
content-addressed cache identities; old caches and runs remain in place.

## HPC commands

Transfer these code changes before starting. Run from the main HPC checkout;
no new site file, environment sourcing or detached worktree is needed. These
launchers read `hpc/site_b200_4gpu.env` for paths/account settings but explicitly
request **untyped GPUs on gpuq**, with no `--exclusive`. Source/configuration
checks pin each submitted plan; keep this checkout unchanged while these jobs
are queued or running. Updating it also changes the source identity expected by
older queued/requeued plans; those need to be regenerated before restarting.
The new names below must be unused.

### 1. Adopt the evaluated snapshot and build its learned token cache

```bash
cd /hpc/home/phi/rvalperga/chronotree

.venv/bin/python hpc/cache_teacher.py \
  --site hpc/site_b200_4gpu.env \
  --evaluation runs/evaluation/teacher_latest_execute8_20261111T102411Z \
  --checkpoint runs/teacher_fv_no_prop_10hz/seed0/teacher/step_00140000.pt \
  --name memory_fv140k_seed0 \
  --partition gpuq --gpu-type "" \
  --cpus 8 --threads 4 --mem 64G --walltime 24:00:00 \
  --max-concurrent 16 --submit
```

Adoption verifies the checkpoint against the complete evaluation and freezes it
at `runs/memory_fv140k_seed0/seed0/teacher/selected.pt`. The array has 16
independent task jobs, each requesting one GPU, 8 CPUs and 64 GB RAM. All may
start concurrently as resources allow. If the numbered training file was moved,
omit `--checkpoint`: the launcher then uses the exact evaluation snapshot.

Check completion without modifying anything:

```bash
.venv/bin/python hpc/cache_teacher.py \
  --site hpc/site_b200_4gpu.env --name memory_fv140k_seed0 --status
```

Wait for `"status": "complete"` and no missing tasks.

### 2. Build the deterministic hierarchy

```bash
.venv/bin/python hpc/cache_hierarchy.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv140k_seed0 \
  --partition gpuq --cpus 4 --threads 1 --mem 32G \
  --walltime 24:00:00 --max-concurrent 16 --submit

.venv/bin/python hpc/cache_hierarchy.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv140k_seed0 --status
```

Wait for completion again. This array also has 16 independent tasks. Hierarchy
construction is CPU work; it requests one untyped GPU per task because this site
uses GPU allocations for these jobs. It computes parent summaries/topology from
the learned token cache, without rerunning the teacher or preparing raw data.

### 3. Freeze the panel and gain-blind candidate contexts

```bash
.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv140k_seed0 \
  --name router_utility_teacher140k \
  --protocol configs/inspection/refinement_utility_teacher140k.yaml \
  --teacher-reference --stage prepare --submit
```

This one-GPU allocation requests 8 CPUs, 64 GB and four hours. It verifies the
accepted teacher/cache identities and freezes the common 256-state diagnostic
panel, episode exclusions and TREE/FLAT candidate definitions. It does not
train a model or choose candidates from losses. Wait for
`runs/router_utility_teacher140k/contexts_complete.json`.

### 4. Render training jobs and run the two independent preflight checks

```bash
.venv/bin/python hpc/student_utility.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv140k_seed0 \
  --manifest runs/router_utility_teacher140k/manifest.json \
  --name utility_students_teacher140k --dry-run

.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_b200_4gpu.env --name router_utility_teacher140k \
  --stage audit --student-study-name utility_students_teacher140k --submit

.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_b200_4gpu.env --name router_utility_teacher140k \
  --stage smoke --student S_INIT --panel-partition train --submit
```

The audit compares 4,096 gain-blind context samples across all three samplers.
The smoke checks frozen predictions, exact coverage, paired noise and numerical
consistency on a bounded multi-task panel. Each has one GPU, 8 CPUs and 64 GB;
they can run simultaneously. No favorable utility result is required to train.
Check that both completed successfully:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
root = Path("runs/router_utility_teacher140k")
for name, path in [
    ("Sampler audit", root / "sampler_audit/summary.json"),
    ("Smoke check", root / "measurements/S_INIT/step_00140000/smoke/train/worker_0.json"),
]:
    result = json.loads(path.read_text())
    print(name, result["status"])
    assert result["status"] == "complete", path
PY
```

### 5. Submit all three independent student jobs

```bash
.venv/bin/python hpc/student_utility.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv140k_seed0 \
  --manifest runs/router_utility_teacher140k/manifest.json \
  --name utility_students_teacher140k \
  --partition gpuq --gpus 2 --cpus 64 --mem 128G \
  --microbatch 32 --workers 16 --walltime 48:00:00 \
  --submit
```

This submits **three jobs**, with two GPUs each (six GPUs total if concurrent).
W&B training runs use **chronotree-routers**. They retain the full designated
training pool while excluding diagnostic episodes and the original router
validation partition. They are not trained on the 256-state diagnostic panel.
All three share state/noise schedules, initialization and compute settings.

Outputs and logs:

```text
runs/utility_students_teacher140k_s_std/seed0/compact/
runs/utility_students_teacher140k_s_stage/seed0/compact/
runs/utility_students_teacher140k_s_packet/seed0/compact/
hpc/generated/utility_students_teacher140k_s_*/logs/
hpc/generated/router_utility_teacher140k/logs/
```

For an inactive job with a saved `last.pt`, repeat the student command with
`--resume --variants S_STD` (or the affected variant). Do not add `--resume` for
already-running or completed arms.

### 6. Measure the three frozen 100k endpoints

After their endpoints exist, submit the development measurements:

```bash
for variant in S_STD S_STAGE S_PACKET; do
  suffix=$(printf '%s' "$variant" | tr '[:upper:]' '[:lower:]')
  .venv/bin/python hpc/refinement_utility.py \
    --site hpc/site_b200_4gpu.env --name router_utility_teacher140k \
    --stage measure --student "$variant" \
    --checkpoint "runs/utility_students_teacher140k_${suffix}/seed0/compact/step_00100000.pt" \
    --panel-partition train --shards 4 --submit
done
```

Each measurement array has four independent one-GPU jobs. After freezing the
analysis protocol, repeat with `--panel-partition validation` for confirmation.
Earlier declared snapshots may be measured on `train` only. There is no S0 or
S_INIT endpoint in this primary three-arm matrix. Reports retain missing arms
and distinguish smoke diagnostics from completed primary evidence.

```bash
.venv/bin/python -m stmem.inspection.refinement_utility \
  --mode report --manifest runs/router_utility_teacher140k/manifest.json
```

## Verification and remaining HPC work

The 140k evaluation was read through the authenticated W&B API. Its actual
checkpoint and full new caches are not available on this workstation; adoption,
full-cache preparation, the multi-task smoke and all new training remain HPC
jobs. No cluster jobs were submitted during implementation. Local verification
uses explicitly synthetic CPU fixtures for provenance and launch behavior;
these are not scientific measurements of the new teacher or students.

Local verification on 11 November: **253 focused CPU tests passed**, covering
dependency/endpoint guards, all three-arm consumers, legacy invariants, teacher
adoption, cache launchers and continuation. Ruff and command-block `bash -n`
checks passed. The architecture artifacts were regenerated and their snapshot/XML
check passed; the visual layout and model layers are unchanged. The device
doctor passed FP32/BF16/FP16 computation on the local RTX 5000 Ada. The protected
encoder-cache code fingerprint is unchanged:
`9cddd3ea461824003854f5b0c42595f5f1f975224d4f59ca0be72d0e1e83dc34`.
New cache identities arise from the selected teacher weights, not from altered
encoder/preprocessing/causal semantics.
