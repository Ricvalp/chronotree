# Implementation and verification status

As of **17 October 2026**, the project installs and its complete bounded local
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

## Actually tested locally

The actual GPU is **NVIDIA RTX 5000 Ada Generation, 32 GB**, capability 8.9;
driver 580.159.04, Torch 2.9.1+cu128, Python 3.11.15. BF16, FP16 and FP32 attention
and matmul forward/backward checks pass. Vulkan/RGB-D reset/step works.

| Check | Measured outcome |
|---|---|
| Core test suite | **55 passed**; 10.07s in the final recorded run |
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

Portable measured summaries are saved in
[evidence/validation.json](evidence/validation.json),
[evidence/data_alignment.json](evidence/data_alignment.json), and
[evidence/real_debug.json](evidence/real_debug.json).
The full local artifacts remain ignored under `runs/final_smoke/`,
`runs/real_debug/`, `runs/ddp_stop_audit/`, `runs/fp16_overflow_audit/`, and
`data/diagnostics/`. Earlier debug artifacts may have different source hashes;
cache validation deliberately rejects them after source edits.

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
commands. No Slurm job is submitted without `--submit`.
