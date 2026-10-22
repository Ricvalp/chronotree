# Implementation and verification status

As of **22 October 2026**, the project installs and its complete bounded local
pipeline runs. The research comparison has **not yet run**. There is no remaining
known local installation, action-alignment, calibration or simulator blocker.

**Representation correction after manual HTML inspection:** in the original
exported VideoUnmask episode 0 at decision 20, all 344 historical visual leaves
pool only tabletop points (support height at most 1.84 mm). Four FPS centers per
frame select tabletop corners; the hard 0.3 m neighborhoods exclude the elevated
objects. The current-frame 64-center path does cover the elevated observations.
The approved correction now uses 2,048 points, above-table spatially balanced
sampling, and 32 clustered observed-point centers per frame for current and
history. Preparation stores center indices once; training reuses them. Core
outputs use `prepared_balanced_v2/`; old files remain separate and cannot satisfy
the new configuration. The original measurements and their single-episode scope
are in [evidence/encoder_coverage_inspection.json](evidence/encoder_coverage_inspection.json).
Matched previews improved geometric and table coverage but did not improve every
pooling measurement: one PickXtimes history block retained 1,365/1,403 cube-point
observations with clustering versus 1,403/1,403 with FPS, both covering all eight
frames. No new trained-policy success result or full-data throughput is claimed.

**Sampling-cycle correction:** a read-only W&B snapshot of real run
`1919882fa3ac237d` identified synchronized 16-update duration/loss cycles caused
by sorting each task's shuffled buckets by decision index. The batched trainer
now keeps each task's epoch shuffled, preserving candidate coverage and global
task balance. GPU task ownership rotates between updates. The sampler and
training engine are versioned in resume provenance; this upgrade requires a
fresh run. The original diagnosis is recorded in
[evidence/training_sampling_cycle.json](evidence/training_sampling_cycle.json).

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
- True padded policy batches, vectorized local history-block encoders, masked
  text/current/history attention and equal per-example action losses. The current
  teacher configuration uses global batch 256, microbatch cap 64 and 128 encoder
  blocks per chunk; four GPUs process 64 decisions each per update. These batch
  settings need a fresh memory measurement with 32 centers per frame. Compact and
  Perceiver defaults remain batch 64, microbatch 16 and 32 encoder blocks.
  Teacher training constructs tensor leaf banks directly; exact trees remain
  available for compact selection, caching and evaluation.
- Persistent CPU processes load and pack upcoming policy microbatches, with
  eight workers per rank and one prefetched microbatch per worker by default.
  The tracked four-GPU site file requests 64 CPUs for the whole allocation.
  Worker processes reuse the decision index and independently read HDF5;
  debug configurations retain the synchronous `training.workers: 0` path.
  Activation checkpointing is disabled in the core defaults; ordinary saved
  training checkpoints remain enabled. Prefetching preserves the configured
  batch, sampler ordering and complete causal histories.
- Chronological streaming evaluation, conditioning-bank reset, chunked continuous
  joint actions, finite-action checks, empty terminal responses, official statuses,
  per-episode atomic results, videos, timing/storage/transfer counters and reports.
- Bounded memory/budget/routing diagnostics, checkpoint selection, exact resume,
  DDP launch, Slurm dependency/resume planner and frozen official-test protocol.
- Explicit `b200_4gpu` training/profiling with a separate tracked site file using
  user-confirmed `aiq` / `gpu:b200:4`. After a user-reported OOM at microbatch 64
  and encoder block batch 128, `configs/teacher_b200.yaml` reduces these to 16
  and 32, preserving global batch 256 and 100k updates with a fresh retry root.
  Fifty-six config/planner tests and the retry dry-run pass. The traceback has
  not been inspected; smaller-run memory fit and throughput remain unmeasured.
- The current teacher schedule is 100,000 updates with selection candidates at
  50k and 100k, configured consistently in both the core and replication plans.
  It uses the established `hpc/plan.py` workflow and a fresh run root in
  `hpc/site_4gpu.env`. This configuration has not yet been measured; the earlier
  batch-64/microbatch-16 profiling results below retain their original scope.
- Standalone teacher memory profiling with 10 warmup updates, separate baseline
  and detailed trace passes, per-rank memory/phase reports and isolated W&B
  output. A dedicated dry-run HPC launcher and optional long-history training
  sampler are documented in [profiling.md](profiling.md).
- W&B logging for all training stages, evaluation/selection shards, label jobs,
  aggregate reports and smoke diagnostics. Default offline recording, optional
  online transport, stable resume identities and one writer per DDP allocation.
  Background GPU/CPU/memory statistics and hardware metadata are now enabled by
  default (`x_disable_stats=False`).
- One action-only startup pass for deterministic decision indexing and exact
  normalization statistics, without visual payload reads; startup progress and
  first-completed-update metrics for teacher/compact/Perceiver training.
- Runtime scalar logging intervals and optional policy heartbeats exposing the
  current phase and accumulation progress; latest-update and sample-loading
  timings without extra heartbeat CUDA synchronization or W&B calls.
- Two fixed W&B point-cloud/action comparisons at policy-training checkpoints,
  calibrated Panda TCP projections, exact action tables, bounded scene size and
  separately measured diagnostic overhead.
- Offline HTML inspection of actual point-to-center pooling edges, normalized
  weights, local trees and temporal forests, with separate current observations,
  pending-block labels, checkpoint bandwidths and JSON provenance.

## Actually tested locally

The actual GPU is **NVIDIA RTX 5000 Ada Generation, 32 GB**, capability 8.9;
driver 580.159.04, Torch 2.9.1+cu128, Python 3.11.15. BF16, FP16 and FP32 attention
and matmul forward/backward checks pass. Vulkan/RGB-D reset/step works.

| Check | Measured outcome |
|---|---|
| Core test suite | **503 passed, 1 skipped** in 88.64 s on 21 October; optional new-format debug prepared subset absent. Earlier transient worker-shutdown failure and successful isolated repeats are recorded in decision 27. |
| Balanced geometry integration | Reviewed real preview parity; four real RGB-D records match online/offline points and centers exactly. BF16 CUDA forward/backward on two eight-frame 2,048-point/32-center blocks passes; no full-history H200 capacity claim. |
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
| Training startup | Exact candidate/normalizer parity with zero visual payload reads; local two-episode median 115.95 ms → 3.79 ms |
| Frequent training diagnostics | Logging overrides preserve exact model/optimizer/RNG/sampler state and resume; blocked-phase heartbeats, cleanup and planner forwarding pass |
| Full-data H200 run inspection | Read-only W&B snapshot, 262 updates: 90.66 s/update mean, 16-update sampling cycle, duration/history-leaf correlation 0.9981 |
| Batched teacher CUDA diagnostic | Eight synthetic decisions, 3,552 history frames, core model/1,024 points: BF16 median forward/backward 7.755 s serial → 0.433 s batched; disk I/O and optimizer/logging excluded |
| Batched teacher real subset | Two updates, two aligned VideoUnmask episodes, core model, global batch 32/microbatch 8: 1.231 s then 0.615 s/update; finite losses, checkpoint and two scene files |
| Batched training invariants | Serial/batched outputs and gradients, masks, causal boundaries, cached readers, remainder weighting, exact resume and two-rank CPU DDP pass |
| Batched cached pipeline | CPU synthetic teacher → HDF5 cache → compact/Perceiver → TREE/FLAT labels and routers → all six readers passed in 8.79 s; frozen encoders remain identical |
| W&B system telemetry | Actual offline SDK recorded GPU utilization/memory/power/temperature/clocks and CPU/RAM metrics with the new default |
| Background loader invariants | Bounded work queue, persistent worker PIDs, joined workers after exceptions, retry identity and pre/post-action HDF5 parity pass |
| Background training parity | Zero/two workers preserve model, optimizer, RNG and sampler state exactly; stop/resume, two-rank CPU DDP and frozen compact/Perceiver paths pass |
| Background loader CUDA diagnostic | Three real-subset BF16 teacher updates at 256 points, global batch 64/microbatch 16, two workers and no activation checkpointing: 1.550, 0.480, 0.474 s/update; peak allocated memory 5.444 GB |

Portable measured summaries are saved in
[evidence/validation.json](evidence/validation.json),
[evidence/data_alignment.json](evidence/data_alignment.json), and
[evidence/real_debug.json](evidence/real_debug.json).
The full local artifacts remain ignored under `runs/final_smoke/`,
`runs/real_debug/`, `runs/ddp_stop_audit/`, `runs/fp16_overflow_audit/`, and
`data/diagnostics/`. Earlier debug artifacts may have different source hashes;
cache validation deliberately rejects them after source edits.

Batched training evidence is recorded in
[evidence/batched_training.json](evidence/batched_training.json). The reusable
`scripts/benchmark_teacher_batch.py` compares identical weights, decisions and
flow draws without optimizer updates. For eight synthetic examples, BF16 peak
allocated memory was 0.195 GB serial and 1.178 GB batched. A full-shape FP32
confirmation had identical loss, maximum prediction difference 5.96e-7 and
maximum parameter-gradient difference 8.94e-8; BF16 gradient relative L2
difference was 0.23%. These numerical differences reflect precision and
reduction shape, rather than a promise of bitwise equivalence across batch sizes.
The real-data check used only two short, locally cached episodes. Neither check
predicts the full 1,600-episode H200 shared-storage training rate. Local artifacts
remain ignored under `runs/batched_teacher_diagnostic/`,
`runs/batched_real_diagnostic/` and `runs/batching_system_stats_diagnostic/`.
These local measurements used microbatch 8. A later read-only W&B snapshot
of the user's four-H200 run `3c2db37b57ccb22a`, fetched at 11:19:53 UTC on
20 October, contains updates 1–387 with microbatch 16/global batch 64.
Mean update time was 7.685 s; rank-zero synchronous loading alone took
5.746 s, or 74.77% of total update time. Rank zero logged peak allocated
PyTorch memory of 2.058 GB. This motivates overlapping CPU work with GPU
training; the timing remainder includes packing, transfers and distributed
waiting and is not a pure GPU-compute measurement. The snapshot remains
ignored under `runs/gpu_utilization_audit/`. These are measurements from the
user's cluster job, not a locally executed full-data benchmark.

Background-loader checks are recorded in
[evidence/prefetch_loader.json](evidence/prefetch_loader.json). CPU tests verify
that zero versus two workers preserves training state exactly, prefetched
future batches do not alter stop/resume, and two DDP ranks can each run two
workers. Synthetic compact and Perceiver training retains the same model,
optimizer, RNG and sampler states with zero/two workers; frozen encoders stay
identical to the selected teacher. Loader tests cover the bounded queue,
persistent processes, failure cleanup, overflow retry identity and HDF5
pre/post-action alignment parity.

On the RTX 5000 Ada, three real teacher updates used the existing two verified
VideoUnmask episodes, 256 points per observation, width 256, global batch 64,
microbatch 16, BF16, two workers, prefetch factor one and activation checkpointing
disabled. Updates took 1.550, 0.480 and 0.474 s; waits for packed inputs were
0.720, 0.016 and 0.068 s. Peak allocated PyTorch memory was 5.444 GB. The device
doctor was rerun and FP32/BF16/FP16 checks passed. Artifacts remain ignored under
`runs/prefetch_loader_diagnostic/`.

This diagnostic has only two short local episodes, which fit each worker's
episode cache. It is not a full-data speed comparison or a forecast for
1,600 episodes at 1,024 points on four H200s. Worker startup is visible in the
first update. Shared-storage contention may still limit the full run, and
compact/Perceiver frozen-cache reads remain in the training process; background
workers only perform raw observation loading and CPU packing. No Slurm job or
full-data H200/B200 benchmark was launched for this implementation.
The subsequent resource change requests 64 CPUs with four GPUs and raises
the core default to eight workers per rank. The measurements above used two
workers; the eight-worker full-data throughput remains unmeasured.

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

Encoder inspection checks are recorded in
[evidence/encoder_visualization.json](evidence/encoder_visualization.json).
The new optional tracing preserves exact forward outputs and parameter gradients,
including checkpointed blocks. An independent comparison against the prior
committed encoder on a real eight-frame, 1,024-point block was also bit-identical.
Two initial-parameter exports at execution decisions 16 and 20 preserve the
complete causal archive and show the separate 64-center current path; decision
20 includes a four-frame pending prefix. Their HTML files are approximately
9–10 MB. On this local two-episode subset, the final CLI exports took 1.33–1.36 s,
including Python startup, file hashing, inspection and HTML/JSON serialization;
inspection itself took 0.39–0.46 s. Browser rendering is excluded. These timings
do not predict hashing or reading a full task on cluster storage.
A separate export loaded the existing two-update debug teacher and used its
learned bandwidths; that checkpoint remains a diagnostic, not a useful policy.

Offline Chromium checks exercise frame filters/animation, time offsets, center
and event selection, edge filters, current/history separation, exact hierarchy
node/edge counts, group selection and light/dark/mobile views. Screenshots were
inspected. The browser made no external requests and reported no JavaScript
errors. The wheel includes both viewer assets. Local HTML/JSON/screenshots are
ignored under `runs/visualizations/`; reproduction commands are in the
[visualization guide](visualization.md).

Action-only startup checks are recorded in
[evidence/training_startup.json](evidence/training_startup.json). Nineteen new
regression cases cover pre/post alignment, gaps, terminal/optional-gripper masks,
nonfinite values, numeric episode order and caps, length bounds and malformed
visual headers. Read guards prohibit full-episode construction, visual payload
reads and a second action scan during normalization. A tiny CPU training check
also verifies startup milestones, first-update metrics and exactly matching
uninterrupted/stop-resumed final weights.

On the existing two real VideoUnmask episodes at 1,024 points, three alternating
paired warm local-storage repetitions measured median indexing plus normalization
of 115.95 ms before and 3.79 ms after the fix (about 30.6×). All 79 candidate
decisions, data identities, first three epoch permutations and normalization
tensors matched exactly. This excludes imports, W&B/model/device initialization
and training. The old two-episode cache already avoided its second read pass;
these measurements do not predict the full 1,600-episode cluster timing. The
doctor was rerun on the local RTX 5000 Ada and FP32/BF16/FP16 checks passed.

Runtime logging checks use bounded synthetic CPU training and real offline W&B
files. A three-update run logging every update and a stop/resumed run changing
the interval retain bitwise-identical model, optimizer, scheduler, scaler,
sampler and RNG state versus the default cadence. Heartbeats continue while the
main thread is blocked, avoid CUDA/W&B calls and preserve the training exception
if final reporting fails. Nonprimary/disabled heartbeats create no worker or
output. Planner dry-runs forward flags without changing generated scientific
configuration or stage identity; router tests preserve full-validation frequency.
These diagnostics do not measure full-data training throughput or heartbeat
overhead on the cluster filesystem. Commands and timing definitions are in
[diagnosing slow training](hpc.md#diagnosing-slow-training).

Separately, user-provided cluster output reports completed full-data preparation
and a running one-GPU H200 NVL teacher job (58082), with CUDA, BF16 and effective
batch 32. Its original startup took approximately 26 minutes before the training
manifest appeared. This was user-reported evidence, not an agent-executed cluster
benchmark; the later post-fix run is described below.
The subsequent report of an hour without old-job metrics does not identify
whether a specific operation stalled or the original 50-update reporting
interval had not yet completed. The new runtime flags require a new process;
they do not change logging in that already running allocation.

On 20 October, a read-only API inspection of the replacement H200 run
`ricvalp/chronotree/1919882fa3ac237d` (Slurm 58085, source `fd954c7`, one GPU,
effective batch 32) retrieved 263 unsampled history rows: startup plus updates
1–262. Reported startup is 16.76 s, including a 13.80 s data scan. Mean update
time is 90.66 s and mean sample-loading time is 11.29 s (12.45% of update time).
These are measurements recorded by the user's cluster run, not a locally
executed full-data benchmark. The first two roughly 60 s updates were not
representative of the full history-length cycle.

The source sampler predicts a 16-update period. Detrended lag-16 correlations
are 0.9873 for time and 0.8289 for loss. Update duration correlates at 0.9981
with differences of the cumulative history-leaf counter. Across complete
cycles, mean time rises from 55.61 s at phase 1 to 145.66 s at phase 16. Mean
loss over complete cycles still decreases from 2.274 in the first cycle to
0.877 in the sixteenth. Loss is normalized by valid action dimensions, so more
historical leaves do not directly multiply the loss; sorted episode-stage and
target/context composition are a supported explanation, not a measured
decomposition of the loss mechanism.

A production-sampler diagnostic using explicitly synthetic candidate metadata
reproduces the short-to-long ordering without training or observation reads.
Setting `length_bucket_size: 1` removes the synthetic periodic ramp while
preserving exact candidate multisets and task balance. This intervention has
not been applied to real training or tested for learned-policy quality. Changing
it changes batch composition and checkpoint training provenance. The raw snapshot,
plot and reproduction scripts remain ignored under `runs/loss_periodicity/`;
portable measured results are in
[evidence/training_sampling_cycle.json](evidence/training_sampling_cycle.json).

These checks do not establish manipulation performance. Expert replay verifies
the data/control interface. A two-update learned model's deliberately short
timeout is a protocol check. Synthetic label quality is not a research result.

The standalone memory profiler passed exact CPU model/AdamW parity checks,
including detailed tracing, and two-rank Gloo runs with spawned loader workers.
A local CUDA diagnostic completed 10 warmup, two measured and one traced update
on two existing, alignment-verified VideoUnmask training episodes. It used the
core-width teacher with 256 points, global batch 64, microbatch 16 and two loader
workers. The two measured updates averaged 0.497 s and peaked at 5,206,243,328
allocated bytes (4.85 GiB); the longest observed context was 1,197 tokens.
These short local histories do not estimate memory headroom or throughput for
the full 1,024-point, 16-task H200 job. The exported trace contains CUDA kernels
and the allocation snapshot contains recorded allocation events. Outputs and
W&B remain separate from teacher training. Portable evidence and commands are in
[evidence/teacher_memory_profiler.json](evidence/teacher_memory_profiler.json).

The user's subsequent four-H200 run `ricvalp/chronotree/63302afad9af0d6f`
completed on the full prepared training configuration: 16 tasks, 100 episodes
per task, 1,024 points, microbatch 16/global batch 64 and eight loader workers
per rank. Read-only W&B inspection and synced summaries show five measured
updates averaging 0.698 s (slowest rank per update; median 0.489 s), with a
19.42 GiB maximum allocated peak and 22.62 GiB maximum reserved peak against
139.80 GiB device capacity. Measured contexts averaged 1,797 tokens and reached
6,252; this is not a maximum-history stress test.

The first detailed inspection used complete rank 0/1 allocation snapshots and
rank 1's complete trace; rank 0's trace JSON was truncated locally during rsync.
Rank 0's separate traced update peaked at 21.13 GiB. Stack attribution assigns
approximately 9.52 GiB to edge MLP/gather/concatenation, 7.45 GiB to point MLPs,
2.77 GiB to geometry intermediates and 0.59 GiB to decoder K/V projections.
These are approximate live allocations at one traced peak, not independent
phase peaks summed together. Rank 1's collective waits align with rank 0 input
waits and heavier historical encoding; they do not establish a network
bandwidth bottleneck. Baseline history encoding averages 207–214 ms on ranks
0/1, compared with 8–10 ms for the decoder.

The whole W&B run lasted about 752 s: progress records identify roughly 685 s
in the action-only index/statistics scan. All timed updates together took
25.37 s including warmup and tracing. The remaining startup bottleneck is
therefore distinct from update throughput; exact filesystem versus CPU costs
were not instrumented. System telemetry is present, but its 15 s cadence is too
sparse to characterize this short update window. This is a user-executed,
bounded profile, not an agent-submitted job or a long-run scaling result.
Evidence and scope are in
[evidence/teacher_memory_h200.json](evidence/teacher_memory_h200.json).

## Not yet run / user-supplied prerequisites

- Full 16-task data with the original preparation is user-reported available on
  the cluster, but is not present in this workspace. The new 2,048-point balanced
  preparation has not been run on the full dataset. The local two-episode subset
  cannot be substituted for core all-task training.
- Useful teacher/compact/router/Perceiver training, official checkpoint selection,
  full validation, frozen evaluation on 50 test seeds per task, or final replication.
- Agent-executed A5000/H200/B200 checks, multi-GPU NCCL, cluster rendering and
  actual core-shape throughput. CPU two-rank tests establish distributed control
  correctness only; user-provided output and a read-only full-data H200 run
  inspection are described above.
- Agent-executed Slurm submission/accounting, site credentials/resource names and costs.
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
