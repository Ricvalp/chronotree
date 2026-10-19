# Implementation and verification status

As of **19 October 2026**, the project installs and its complete bounded local
pipeline runs. The research comparison has **not yet run**. There is no remaining
known local installation, action-alignment, calibration or simulator blocker.

## Implemented

- Locked PyTorch/core and optional pinned simulator environments; actual-device
  doctor and separate bounded HPC compute/rendering probe.
- Task-selective integrity-checked downloads and safe extraction, numeric HDF5
  ordering, two-camera calibrated point clouds, typed allowlisted observations,
  causal aligned chunks, terminal/gap/action-availability masks.
- Joint space-time Gaussian pooling, framewise variant, robot-event leaves,
  deterministic complete/pending hierarchies, budget-filling tree/flat/uniform
  readers, exact residual coverage, cached immutable router keys.
- Full-history flow teacher, frozen task-level caches, shared variable-budget
  compact policy, paired signed-gain supervision, episode-disjoint router-label
  diagnostics, one refresh, Perceiver adaptation and current-only diagnostic.
- Chronological streaming evaluation, conditioning-bank reset, chunked continuous
  joint actions, finite-action checks, empty terminal responses, official statuses,
  per-episode atomic results, videos, timing/storage/transfer counters and reports.
- Bounded memory/budget/routing diagnostics, checkpoint selection, exact resume,
  DDP launch, Slurm dependency/resume planner and frozen official-test protocol.
- W&B logging for all training stages, evaluation/selection shards, label jobs,
  aggregate reports and smoke diagnostics. Default offline recording, optional
  online transport, stable resume identities and one writer per DDP allocation.
- Two fixed W&B point-cloud/action comparisons at policy-training checkpoints,
  calibrated Panda TCP projections, exact action tables, bounded scene size and
  separately measured diagnostic overhead.

## Actually tested locally

The actual GPU is **NVIDIA RTX 5000 Ada Generation, 32 GB**, capability 8.9;
driver 580.159.04, Torch 2.9.1+cu128, Python 3.11.15. BF16, FP16 and FP32 attention
and matmul forward/backward checks pass. Vulkan/RGB-D reset/step works.

| Check | Measured outcome |
|---|---|
| Core test suite | **214 passed**; 33.09s after parallel preparation integration |
| Package lock/install | `uv lock --check`, `uv pip check`, imports and lint pass |
| Real data | One 1,251,608,544-byte VideoUnmask archive; SHA verified; two episodes prepared |
| Expert replay | Both complete sampled training episodes reached official success; exact nonterminal post-command qpos match |
| RGB-D geometry | Front/wrist clouds inspected together; calibration and replay checks pass |
| Real training | Two finite teacher optimizer updates on the real prepared subset |
| Real policy loop | Short official val episode 0, 66 conditioning frames, 13 issued commands, expected short-limit timeout, no errors |
| Complete synthetic cycle | Teacher 12 → cache → compact 12 → both 30-context label jobs → both routers 6 → both 15-context refreshes → both routers 12 total → Perceiver 12 |
| Reader inference | FULL 33 leaves; TREE/FLAT/UNIFORM/PERCEIVER 16 historical tokens; CURRENT 0; all actions finite |
| Causal/coverage invariants | Future perturbation, prefix replacement, masks, exact tree/flat coverage, shared noise/time, cache/stream parity pass |
| Training behavior | Encoder gradients finite; fixed tiny loss decreases; normalization round trip passes |
| Stop/resume | Single-process and two-rank CPU final weights exactly match uninterrupted training; partial exit 75 and stale marker removal verified |
| FP16 recovery | Forced scale 1,048,576 reduced to 65,536; exactly two successful optimizer/scheduler steps |
| Analysis CLI | Synthetic memory interventions and exact tiny searches over 6 TREE and 120 FLAT contexts execute |
| Reports | Measured debug summary/plot generated; absent runs remain null/“not yet run” |
| HPC | 45-job core dry-run, maximum two concurrent allocations; scripts pass syntax checks; fake-submission tests only |
| W&B | Real offline files decoded and checked; rewind/resume, interrupted journal recovery, failure cleanup and single-writer DDP pass |
| CPU preparation | Serial/parallel arrays exactly equal; safe interruption/reuse and CPU limits pass; two real episodes prepared at 1,024 points |

Portable measured summaries are saved in
[evidence/validation.json](evidence/validation.json),
[evidence/data_alignment.json](evidence/data_alignment.json), and
[evidence/real_debug.json](evidence/real_debug.json).
The full local artifacts remain ignored under `runs/final_smoke/`,
`runs/real_debug/`, `runs/ddp_stop_audit/`, `runs/fp16_overflow_audit/`, and
`data/diagnostics/`. Earlier debug artifacts may have different source hashes;
cache validation deliberately rejects them after source edits.

The W&B follow-up is recorded separately in
[evidence/wandb_validation.json](evidence/wandb_validation.json), with fresh local
artifacts under `runs/wandb_verified_smoke/` and `runs/wandb_real/`.
The caller-side logging diagnostic measured 0.139 ms median and 0.205 ms p95 for
1,000 records with 30 scalars each. This excludes startup/shutdown, background SDK
work, network and cluster filesystem costs; it is not an end-to-end training
slowdown estimate. Online authentication, dashboard delivery and offline upload
have not been exercised against a W&B account.

The scene follow-up is recorded in
[evidence/wandb_scenes.json](evidence/wandb_scenes.json). On the existing two
VideoUnmask episodes, enabling checkpoint scenes left model, optimizer,
scheduler, scaler, sampler and all saved RNG states bit-identical to the disabled
run; partial-stop/resume also matched exactly. Real SDK scene and action-table
files were decoded and checked. Both scenes took 0.174–0.339 s and about 201 KB
per checkpoint on this 256-point debug model. These numbers include inference,
CPU conversion and media serialization, but exclude network delivery and run
startup/shutdown; full-size cluster overhead remains unmeasured. All 464 prepared
robot observations agree with the pinned forward kinematics within 0.51 μm and
1.24 μrad. Local GPU artifacts are under `runs/wandb_scenes_verified/`.

A separate two-rank CPU check completed with one W&B writer and two scenes per
checkpoint. Both ranks' RNG states matched exactly. This pair did **not** produce
bit-identical weights: tiny differences already existed in the checkpoint saved
before the first scene callback (maximum 2.91e-11; final maximum 2.98e-8). It
verifies distributed completion and media ownership, not exact CPU numerical
reproducibility. Artifacts are under `runs/wandb_scenes_ddp/`.

Parallel preparation checks are recorded in
[evidence/parallel_preparation.json](evidence/parallel_preparation.json). A local
synthetic four-task/192-frame diagnostic took 14.42 s with one worker and 4.65 s
with four; every prepared array and scientific manifest matched exactly. These
single-pair timings include Python/process startup, hashing, conversion and writes,
and reflect warm local storage, not HPC throughput. Separately, two existing real
VideoUnmask episodes (464 frames) were prepared at 1,024 points in 30.27 s; verified
reuse took 4.98 s and preserved bytes/timestamps. Raw inputs were freshly hashed
both times; the legacy local manifest lacks a member digest, so no new independent
archive verification is claimed. Original prepared data remained unchanged.
Full artifacts are ignored under `runs/preparation_benchmark/` and
`runs/preparation_real/`. No full-data preparation or Slurm job was launched.

These checks do not establish manipulation performance. Expert replay verifies
the data/control interface. A two-update learned model's deliberately short
timeout is a protocol check. Synthetic label quality is not a research result.

## Not yet run / user-supplied prerequisites

- Full 16-task data download/preparation (approximately 56.4 GB compressed listing;
  expanded total must be measured). The existing two-episode 256-point subset
  cannot be substituted for core 1,024-point all-task training.
- Useful teacher/compact/router/Perceiver training, official checkpoint selection,
  full validation, frozen evaluation on 50 test seeds per task, or final replication.
- A5000, H200, B200, multi-GPU NCCL, cluster rendering and actual core-shape
  throughput. CPU two-rank tests establish distributed control correctness only.
- Actual Slurm submission/accounting, site credentials/resource names and costs.
  No approved overall GPU-hour or credit budget is inferred.
- Complete asset availability for every task on offline cluster nodes; only the
  locally exercised VideoUnmask simulator path is verified.

The user needs to supply `hpc/site.env`: shared repo/data/cache/run paths,
Python/activation settings and the site's actual partition/GPU/account/QoS choices.
No dataset credential was needed for the public task archive. Full transfer and
long jobs remain explicit user actions.

## Immediate commands

```bash
.venv/bin/python scripts/smoke.py
.venv/bin/python hpc/plan.py --plan core_seed0 --dry-run
```

For the first allocated HPC node:

```bash
python hpc/smoke.py --render --updates 3 --output "$STMEM_RUN_ROOT/node_smoke.json"
```

The README supplies the installation, full-data preparation and subsequent stage
commands. The planner submits only with `--submit`; standalone templates, including
CPU preparation on the GPU queue, require an explicit user `sbatch` command.
