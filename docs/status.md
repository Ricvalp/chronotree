# Implementation and verification status

## Empty-history sampler audit repair — 11 November 2026

HPC audit job 72962 failed before producing `sampler_audit/summary.json`.
The traceback was reproduced locally for all three student variants: an empty
causal FLAT archive made NumPy infer a floating-point selection mask, so
inverting it for residual accounting raised `TypeError`. The shared archive
helper now declares Boolean masks and integer group/count arrays explicitly.
Zero-history examples remain in the experiment with zero history tokens;
no padding, future leaves, example skipping or coverage relaxation was added.
The same dtype correction covers empty packet-eligibility queries.

Verification: **79 focused CPU tests passed**, including 20 new empty-history
cases across candidate construction, context sampling, audit statistics and
hydration. The original FLAT exception was observed before the fix for S_STD,
S_STAGE and S_PACKET. Device doctor passed FP32/BF16/FP16; lint and architecture
snapshot/XML checks passed. Encoder-cache fingerprint remains unchanged.
The complete 4,096-example production audit still needs its HPC retry; no jobs
were submitted locally. See the [recovery commands](refinement_utility_pilot.md#recovering-the-empty-history-audit-failure).

## Matched student and refinement-utility study — 10 November 2026

Revision 2 of the user's utility addendum, with the user's subsequent 100k-update
amendment, is implemented as an opt-in study.
Three matched teacher-initialized 100k jobs (S_STD, S_STAGE, S_PACKET) preserve
the accepted teacher, production 90k S0 and existing caches. Whole diagnostic
episodes and the original router-validation partition are excluded from new
optimization. Shared per-example FM randomness is independent of context RNG.
Complete local/packet construction routes preserve every integer prefix and
exact coverage. Fixed candidate panels, shared teacher predictions, raw expert
FM utilities, independent confirmation, numerical controls, capability inputs,
episode-clustered reports and bounded generated-action checks are implemented.
The [HPC workflow](refinement_utility_pilot.md) gives exact launch commands.

The amended schedule uses LR 1e-4, 1k warmup and a cosine floor of 1e-5 at 100k,
rolling recovery checkpoints every 1k and numbered snapshots every 5k. Earlier
snapshots are development-only; the three 100k endpoints are the primary
comparison. This is a user-requested plan change, not completed training or new
benchmark evidence. Existing 30k studies are not reinterpreted or overwritten.

Verification: the full CPU suite passed **1,739 tests**, with four expected
skips. After final additions, **108 focused tests** passed. Ruff, whitespace and
architecture snapshot/XML checks pass. The device doctor passes FP32/BF16/FP16.
Bounded real-data training probes preserve cached FULL teacher parity exactly
and leave encoders frozen. Final FP32 inspection of eight VideoUnmask states
measured 9,664 compact and 64 shared teacher model examples in about 4.42 seconds
after startup, including serial cache/context assembly. Maximum serial/batched
loss difference was 2.16e-7. These small local histories cannot predict full-HPC
throughput; the production multi-task smoke remains required.

One gain-blind generated-action case and four exact-state first-chunk
continuations also completed, with four MP4s. All four contexts grasped the
correct container and succeeded after the same 36-command expert prefix. The
expert prefix had already approached that container, and generated TCP paths
differed by only about 1 mm: there is no differential task-choice benefit or
autonomous benchmark claim. See [the continuation record](refinement_continuation.md)
and [machine-readable evidence](evidence/refinement_utility_20261110.json).

The full production caches are on the HPC. The 4,096-state sampler audit,
256-state utility panel and three 100k student runs have **not** been submitted
or completed locally. Conditional rehearsal has an unactivated launcher;
learned-router/adaptation protocols are prepared but still require their
explicitly listed label/controller integration if activated. The main report
does not depend on those later branches.

## Reusable teacher continuation inspection — 10 November 2026

`scripts/inspect_teacher.py` now orchestrates the existing usual checks, 48 fixed
history-control rollouts, four fresh recent-window rollouts, and generic paired
checkpoint reports. `--detach` avoids terminal/tool-session interruption;
`--resume` fills missing trials under a sealed identity without replacing
published outcomes. Comparisons and observer-based behavior summaries are
repository modules rather than checkpoint-specific scripts under ignored runs.
Source/runtime differences remain explicit review items. See the
[workflow](teacher_inspection_workflow.md).

The saved 104k/120k analysis was reproduced on CPU, including the complete
112-video audit and all 240 prior offline metric means. No new simulations or
140k evaluation were launched for this change. 111 targeted CPU tests pass, covering resume integrity,
interruption/locking, copied controls, comparison math and bounded orchestration.
The training architecture and diagnostic scientific protocol are unchanged.

## Fixed-label longer router fit — 4 November 2026

The user-run initial router jobs are now observable through W&B. TREE/velocity
run `10f7ad712dcf45d0` completed 5k updates in 16.4 seconds on an H200, with
50,582 fitting labels and 12,642 held-out labels. Validation loss was 10.373
at step 100 and 10.376 at step 5k; final within-context ranking agreement was
50.15%. The other initial routers ended at 50.33–51.88% ranking agreement.
All initial label shards completed; their reported elapsed generation times
were 425–509 seconds. These are read-only measurements of the user's jobs.
They do not establish useful learned routing or its closed-loop success.

The requested longer-fit diagnostic is implemented in `hpc/overfit_router.py`.
It continues the immutable initial 5k TREE/velocity checkpoint to 100k total
updates on exactly the existing initial labels, with preserved optimizer/RNG
and episode split. Separate outputs and the **`chronotree-routers`** W&B project
keep it outside the original training pipeline. Full fitting/validation losses
and ranking, plus zero/median reference predictors, distinguish fitting from
generalization. See [router_overfit_hpc.md](router_overfit_hpc.md).

Verification: CPU regression tests establish unchanged optimization with
diagnostics on/off, unchanged original inputs, and guarded isolated launch and
resume. A tiny synthetic CUDA continuation matched uninterrupted fitting
exactly; the device doctor passed FP32/BF16/FP16. Architecture snapshot checks
pass without diagram changes because no model/context/batching design changed.
The 100k diagnostic itself has not been run or submitted by the assistant.

## Router objective ablation — 4 November 2026

The velocity/final-denoised-action ablation is implemented. The combined label
generator shares cached evidence and teacher work, uses the persisted causal
hierarchy, batches inference, and resumes exact independent shards. Four
equal-capacity scorers receive matched initial/refresh optimizer exposure and
retain separate objectives/checkpoints. Existing encoded and hierarchy caches
remain compatible through an audited consumer-only source migration.

Verification: 308 focused CPU tests passed; all four scorers completed a tiny
synthetic initial/refresh/resume pipeline; spawned-worker and interruption
recovery checks passed. The RTX 5000 Ada passed the device doctor and a bounded
synthetic CUDA label/optimizer check. The unchanged architecture diagram's
provenance was regenerated. Full-data throughput and HPC execution were
unmeasured at implementation time; subsequent user-run observations are recorded
above. Learned-router closed-loop results remain unmeasured. No Slurm jobs were
submitted by the assistant.

Next command on the HPC (dry-run):

```bash
.venv/bin/python hpc/train_routers.py --site hpc/site_b200_4gpu.env
```

Add `--submit` to launch the dependency chain. It freezes the latest compact
checkpoint from `memory_fv_seed0_compact_hierarchy`, generates 16k initial and
8k refresh contexts per reader/objective, and trains velocity TREE/FLAT plus
eight-step final-action TREE/FLAT. See [router_training_hpc.md](router_training_hpc.md)
for exact resources, outputs, four-step alternative and resume commands.

## Earlier project-wide verification

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
  user-confirmed `aiq` / `gpu:b200:4`. After user-reported OOMs at microbatch /
  encoder block batch 64/128 and 16/32, `configs/teacher_b200.yaml` reduces these
  to 8/16, preserving global batch 256 and 100k updates with a fresh retry root.
  The subsequently supplied traceback fails with `cudaErrorDevicesUnavailable`
  at device selection, before model/data initialization; that attempt is not
  evidence of a batch-size OOM. A user-run standalone four-worker tensor probe
  reproduces the error on `ainode01`, despite four visible GPUs appearing empty
  in Default compute mode; only rank 3's failure is confirmed before the others
  are terminated. A later user-run training job reached roughly 600 updates on
  `ainode04`, then genuinely exhausted GPU memory in the decoder K/V projection
  (162.76 GiB actively allocated by PyTorch). The batched encoder now avoids
  retaining dense candidate distances for bandwidth gradients. Full-history
  B200 memory fit and throughput after that fix remain unmeasured. The 8/16 retry
  passes 30 configuration tests
  and a relocated planner dry-run; previous 16/32 validation included 56
  config/planner tests.
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
| Core test suite | **510 passed, 1 skipped** in 90.99 s on 23 October; optional new-format debug prepared subset absent. Earlier transient worker-shutdown failure and successful isolated repeats are recorded in decision 27. |
| Balanced geometry integration | Reviewed real preview parity; four real RGB-D records match online/offline points and centers exactly. BF16 CUDA forward/backward on two eight-frame 2,048-point/32-center blocks passes; no full-history H200 capacity claim. |
| Saved encoder geometry | Synthetic BF16, four eight-frame blocks at 2,048 points/32 centers: peak additional encoder forward/backward memory 677,018,624 → 487,831,040 bytes (27.9% lower). All outputs bit-identical; maximum parameter-gradient difference 1.49e-8. CPU regression checks dense saved geometry is absent; no full-policy/B200 capacity claim. See [evidence](evidence/encoder_distance_memory.json). |
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

## Router architecture screen — 2026-11-05

Implemented fixed-label dot/MLP, TREE child-feature, current-attention and combined
router variants, verified enriched-input reconstruction, GPU-resident training,
exact resume, and cached online scoring. The new launcher isolates all outputs,
uses chronotree-routers and untyped GPU requests, and submits only with --submit.
See [router architecture commands](router_architectures_hpc.md).

244 focused CPU tests and five four-update explicitly synthetic CUDA fits passed.
The real device doctor passed FP32/BF16/FP16. These checks do not establish improved
router generalization, task success, or H200 throughput. Production label shards
are on the HPC, so the real production replay/training screen remains for the
user to launch. A bounded real-cache replay of 12 VideoUnmask decisions matched
online current/child features bit-exactly, with artificial gain placeholders and
no optimization. No cluster jobs were submitted here.

## Local gain-repeatability inspection — 2026-11-09

Implemented and ran `stmem.inspection.gain_repeatability`: 30 fixed contexts,
both readers, four candidates and 16 paired draws on one real VideoUnmask
demonstration. Both velocity and eight-step final-action labels were measured;
interactive HTML includes every candidate, source RGB/point clouds, parent-child
geometry, action-target paths and separate expert errors. No memorization fit.
See [protocol, findings, and rerun commands](gain_repeatability.md).

Noise-dependent ranking changes coexist with stable signal. Final-action
cross-draw pair-order agreement is 74.2% TREE and 67.2% FLAT in this local sample.
Featured examples include target-container utility, ranking reversal, and a
teacher-agreement/expert-error disagreement. These are not task success rates.
The local 83k compact differs from the 90k compact used for production labels;
the teacher is exact. Exact production-policy reproduction awaits that checkpoint.

The actual-device doctor and 45 focused CPU tests passed, with
real HTML rendering and interaction checks. Identical-input/no-op controls gave
zero differences. Architecture outputs still match the unchanged model code.
Generated measurements, images and HTML remain ignored under
`runs/inspection/gain_repeatability_20261109/`.

## Matched learned-router evaluation launcher — 2026-11-09

`hpc/evaluate.py` now accepts `--router-checkpoint` and `--wandb-project`.
It snapshots and validates the router alongside the exact compact policy,
rejects incompatible reader/provenance or mixed random/learned selection, and
protects both snapshots across resume. Commands for retained FLAT-dot 50k
versus random FLAT with the frozen 90k compact are in
[router architecture instructions](router_architectures_hpc.md#matched-closed-loop-check).

87 CPU-safe launcher, parallel-evaluation, aggregation and execution tests
passed. They include matched seeds/configuration, immutable-input resume and
generated Bash checks. The architecture snapshot was regenerated: only the
launcher source hash changed, with identical HTML/draw.io content; `--check`
passes. No HPC job was submitted and no router closed-loop success is measured
by these implementation checks. The 52.33% ranking peak at 2k was not saved;
the selected retained 50k router recorded 51.98% offline ranking accuracy.

## Visible-cue refinement diagnostic — 2026-11-09

Verified the transferred 90k compact against the production-label SHA256 and ran
`stmem.inspection.cue_refinement` on one real VideoUnmask demonstration. Nine
decision/budget cases, six exact-coverage resolution arms, two separate reference
arms and eight shared noise draws are saved with an interactive HTML viewer.
Primary reset/B128 target-cue exposure changes targets by1.506 mm on average,
not a new movement toward a different object. Later covered-region detail can
cause much larger adverse changes. See `docs/gain_repeatability.md` for the
causal controls, limitations and reproducible command.

44 focused CPU tests passed; doctor passed on the actual GPU. Same-input replay
has zero numerical error. Architecture diagram check passed unchanged. No
training, simulated rollout or HPC submission in this diagnostic.

## Target-appearance counterfactual — 2026-11-09

Implemented and ran `stmem.inspection.cue_counterfactual` with exact 90k compact
and full-history teacher checkpoints. Blue/red appearance is swapped in both
cameras throughout the retained reveal interval, with recomputed CLIP features
and learned history tokens and unchanged current input. Sixteen paired draws
show no intended target redirection in the first chunk of the one tested scene.
Mean target-path changes range from 0.017 to 0.762 mm across tested models/readers.
Encoded evidence changes are verified; unchanged-input rebuild/replay controls
are exact. Results and interactive paired RGB/path views are in
`runs/inspection/cue_counterfactual_verified_20261109/`.

35 focused CPU tests and real HTML browser checks passed. No simulator rollout,
training or Slurm submission. See `docs/gain_repeatability.md` for the measured
scope and limitations; this is not a claim about all episodes or later decisions.

## Normal VideoUnmaskSwap container-choice check — 2026-11-09

Ran the selected full-history frozen-vision teacher (step 104110) on official
validation configurations 0–9 with original reveal/swap histories and ordinary
closed-loop execution: 16 predicted targets, 10 executed, action/observation
stride 2, eight flow-integration steps. Two local simulator workers completed all
10 episodes; all videos and read-only container/contact traces are retained.

Official outcome: 1/10 success, nine failures, no timeouts or infrastructure
errors. Eight episodes first grasp a wrong container. Episode 6 briefly contacts
the correct one without lifting it, then picks a distractor and fails. Episode 8
performs both requested pickups and succeeds. Grasped containers vary in identity
and world position; this contradicts an always-fixed-container explanation but
does not establish reliable use of the reveal history.

Report: `runs/inspection/video_unmask_swap_normal10_20261109/index.html`.
W&B: https://wandb.ai/ricvalp/chronotree-routers/runs/c16dbb018a75f627 .
The new standalone runner is `stmem.inspection.swap_rollouts`; the read-only
observer and offline report have 13 passing CPU tests. Architecture snapshot
check passes unchanged. This is a small validation diagnostic, not test tuning.

## Paired teacher history-use investigation — 2026-11-09

Completed 56 fresh local simulator rollouts with the same full-history frozen
teacher at step 104110: official validation scenes 0–3 on SwingXtimes,
VideoRepick and VideoPlaceOrder with four history reads, plus two four-scene
Swing controls. Settings stay at 16 predicted/10 executed targets, 10 Hz action
and observation sampling, and eight flow steps. No training or Slurm jobs.

FULL succeeds 4/4 Swing, 1/4 Repick and 0/4 PlaceOrder. Erasing execution history
or all history gives 0/4 on each task. Erasing the initial demonstration gives
4/4 Swing (an exact no-op), 0/4 Repick and 3/4 PlaceOrder. Swing's recent-window
and count-preserving payload/time permutation controls both give 0/4 versus
matching FULL 4/4. All 56 episodes have valid task outcomes and no infrastructure
or observer errors. All paired reset inputs match; 52 fixed-current shadow
probes reproduce unmodified actions exactly, with unchanged RNG state.

The concrete positive case is Repick1: FULL performs three correct pick/drop
cycles and presses the button; without the demonstration it carries the same
correct cube toward the button during the first cycle and fails. Swing controls
break pickup or transport before the first completed swing, supporting useful
movement-history dependence without proving an abstract counter. PlaceOrder
provides no consistent semantic benefit: FULL has three wrong-pad drops and
one near-correct-pad timeout, while video-free successes are compatible with a
location preference. Preserve these negative cases alongside the successes.

Reusable runner: `stmem.inspection.history_use`. Reports with videos, explicit
demonstration/policy-start controls, paired outcomes and fixed-current probes:
`runs/inspection/history_use_20261109/index.html`,
`runs/inspection/history_recent_swing_20261109/index.html`, and
`runs/inspection/history_permuted_swing_20261109/index.html`.
All were browser-checked; 52 focused CPU tests, Ruff and unchanged architecture
snapshot checks pass. Details, limitations and commands are recorded in
`docs/gain_repeatability.md`. Diagnostic artifacts and weights remain ignored.

## Teacher continuation with retained 20k checkpoints — 2026-11-10

Implemented resume-safe `--checkpoint-every` in policy training/the planner and
planner `--walltime`. The requested frozen-vision teacher command resumes its
existing `last.pt` and W&B run on four untyped `aiq` GPUs for a requested 72 hours,
retaining `step_XXXXXXXX.pt` every 20k total optimizer steps plus rolling and
checkpointed-stop `last.pt` saves. It preserves the original 1e-5 LR floor and
scientific configuration. Commands are in `docs/hpc.md`.

111 focused CPU tests pass, including exact resumed-state comparison, two-rank
save/resume, W&B identity preservation and literal 72-hour Slurm rendering.
The actual synced teacher's configuration was dry-run with checked shell syntax;
no training or Slurm submission occurred. Doctor passes FP32/BF16/FP16 on the
available RTX 5000 Ada. Ruff and diagram snapshot checks pass; regenerated diagram
layout/content are unchanged, with only the training source hash updated.
