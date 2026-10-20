# Implementation decisions — 17 October 2026

1. **Unified pinned environment.** Core uses Torch 2.9.1 CUDA12.8, Python3.11,
   NumPy1.26.4; the simulator extra pins the requested benchmark/fork and
   torchvision0.24.1. NumPy2 was incompatible with the pinned simulator, so it
   was removed before the real checks. A separate policy service is unnecessary
   in the verified environment. `uv.lock` captures the exact dependency graph.

2. **Measured hardware differs from the brief.** The available device is RTX5000
   Ada32GB, capability8.9, driver580.159.04. A5000/H200/B200 are documented targets,
   not measured machines. The historical profile name `a5000_debug` is retained,
   but actual device identity appears in doctor evidence. Debug width64 keeps
   tests small; scientific width256 is unchanged.

3. **Alignment is post-action.** Two VideoUnmask episodes replayed with exact
   nonterminal qpos agreement after each stored command. The automatic initial
   execution command is consumed by reset. Terminal wrapper extra steps explain
   terminal-only differences. No per-sample nearest-pose offset is used.
   Conditioning actions and initial reset commands stay unavailable. Source-local
   recorded-frame time is used because “NO RECORD” gaps prevent claiming physical
   continuity. See data_contract.md and its measured evidence.

4. **Deterministic evidence hierarchy.** Spatial balanced binary splits inside
   visual blocks, temporal splits inside robot-event blocks, and binary-counter
   merges across completed blocks. Temporary prefix trees are re-encoded only
   from available frames and replaced on completion. Every valid leaf has mass1.
   Parents aggregate FP32 latent sums before nonlinear adapters. Flat residuals
   exclude selected leaves and refill zero-cost singleton replacements.

5. **Compact policy is genuinely shared.** Tree and flat use one frozen encoder,
   one metadata schema, one compact decoder/normalizer, matched scorer capacity,
   and equal context-mixture weights at each training budget. No method ID or
   expert/noisy action enters routing. The full reference only cross-attends from
   action queries, so its historical attention cost is linear in leaf count.

6. **Precision and caching.** Teacher gradients use tested BF16 (FP16+GradScaler
   fallback, then FP32). Frozen cache and online evidence encoding use FP32 to
   maintain numerical parity; online read-adapter/decoder uses tested autocast.
   Geometry, distances, sums/counts, normalization and losses remain FP32.
   Frozen keys are prepared in batches before CPU heap traversal; key construction,
   full archive scans, transfer and storage costs are counted. No GPU key-network
   call is hidden inside an individual tree split.

7. **Explicit simple training implementation.** Variable-length teacher samples
   use microbatch1, block gradient checkpointing and gradient accumulation.
   Task-balanced global sample positions are partitioned across ranks without
   sampler padding. Length buckets only order samples; they do not shorten history.
   Empty-history branches require DDP dynamic unused-parameter detection. CPU2rank
   stop/resume exactly reproduced uninterrupted weights. FP16 skipped steps retry
   the same optimizer-step position without advancing the scheduler; eight repeated
   overflows fail with an explicit FP32 diagnostic suggestion.

8. **Limited action-utility supervision.** Initial16k and refresh8k contexts per
   reader, up to four moves each. Budgets cross early/middle/late strata rather
   than being coupled to them. Zero-cost flat moves are normalized without division.
   One teacher/noise/time sample is reused before/after every candidate move.
   Gain scaling is fitted on scorer-training episodes and retained on refresh.
   These are conditional diagnostics, not unseen-policy-episode generalization.

9. **Storage and durability.** Task-level HDF5 cache shards avoid millions of
   files and full-dataset replication in workers. Cache identity includes code,
   teacher, preprocessing, data/alignment and prefix contracts. Changing source
   content conservatively invalidates caches, including semantically neutral edits.
   Checkpoints contain optimizer/scheduler/scaler/RNG/sample position; one writer
   uses atomic replacement and retains a previous known-good checkpoint. Selected
   checkpoints require immutable validation-selection records outside synthetic
   smoke runs. A partial stage exits nonzero and removes stale completion markers.

10. **Evidence boundaries.** Real expert replay success verifies protocol, not
    learned-policy performance. The two-update real model's short rollout timed
    out as expected and has no useful-control claim. Synthetic cycles validate
    plumbing only. No full download, long training, real Slurm submission, official
    test evaluation, or architecture sweep was launched.

11. **Deliberately limited first implementation.** CPU Python traversal and a
    complete flat scan are measured honestly; there is no ANN/custom CUDA/FP8.
    Process RSS and GPU peak allocation are recorded separately from logical
    payload counters; GPU allocation excludes driver and external library memory.
    Full-scale throughput, long-run learning and H200/B200 rendering remain to be
    measured. The main contribution is the experimental memory interface, with
    no claim of first hierarchical memory or an exact pooled-attention theorem.

12. **Checkpoint-only action visualization (19 October follow-up).** Teacher,
    compact and Perceiver training refresh at most two fixed training examples
    in the existing W&B run. Select the first available decision at least two
    action chunks into execution, or the first valid decision for short episodes;
    selection never ranks policy outputs. Compact previews use the declared
    UNIFORM reader, so no untrained router is presented as learned retrieval.
    Full causal evidence is retained; only the displayed current cloud is capped.
    Fixed noise, target-free inference inputs, restored RNG/modes and rank-zero
    unwrapped inference isolate this diagnostic from optimization. Evidence is
    FP32 as in deployment; decoder autocast matches the run's precision.
    Pinned Panda forward kinematics projects commanded joints into world TCP
    positions, with deployment clipping explicitly recorded alongside raw actions.
    The observed TCP validates the coordinate frame before any scene is emitted.
    Paths are command targets, not simulated trajectories or success measures;
    gripper/joint details remain in tables. Synthetic unphysical states and
    official evaluation without expert targets do not receive these comparisons.
    Partial-stop saves skip visualization; per-event costs and the final loop
    wall time including diagnostics are logged. See
    [evidence/wandb_scenes.json](evidence/wandb_scenes.json) for measured checks.

13. **CPU task-parallel preparation (19 October follow-up).** Spawn one process
    per active task, each opening its own HDF5 files; cap concurrency by requested
    workers, selected task count and the CPU allocation/affinity. Default to one
    Torch/native-library thread per worker. A GPU-only queue may allocate one GPU,
    but preprocessing remains on CPU. Frame geometry, numeric chronology, causal
    action masks and unresolved alignment metadata are unchanged. No episode
    sharding, shared HDF5 writer or CUDA context is introduced.

    A completed task is published by atomic replacement under an exclusive
    per-output OS lock. Workers stop on parent interruption or peer failure;
    cancellation is deferred across child startup so every live writer can be
    joined. Readers exclude partial files. Optional reuse freshly hashes raw
    inputs and checks the exact configuration, episode set, array layouts and
    chronology metadata. It does not scan all prepared payloads for bit rot;
    interrupted tasks restart from their beginning. Legacy download manifests
    without extracted-file digests remain usable, with an explicit unverified
    download-manifest integrity event; fresh hashes alone do not establish archive
    membership. Current download manifests retain their SHA256 integrity check.

    The parent emits serialized JSON progress, including hashing and per-task
    conversion ETA. Local synthetic parallel parity/timing and a bounded real
    two-episode check are recorded in
    [evidence/parallel_preparation.json](evidence/parallel_preparation.json).
    These are not full-data/HPC throughput estimates. No data download, long run
    or Slurm submission was performed for this follow-up.

14. **Offline inspection of exact encoder structure (19 October follow-up).**
    A CPU diagnostic exports standalone HTML and JSON for one selected prepared
    episode/decision. Optional trace callbacks capture actual FPS centers,
    capped neighbor indices, masks and normalized Gaussian weights from the
    existing forward calculation. The standard streaming builder constructs
    the complete source/modality-separated binary archive. Point-level detail
    is limited to the last requested blocks per source plus the separate current
    observation; the full historical hierarchy retains exact leaf coverage.
    Pending execution evidence is explicitly marked and never merged as a
    completed block. Display time offsets change no model inputs or graph edges.

    Verified alignment and preprocessing checks are reused. Only a completed
    selected task file is opened; it is hashed for provenance. Input/output
    aliasing is rejected so an adjacent JSON cannot overwrite alignment or a
    checkpoint. No loader, training schedule, action targets, retrieval policy,
    or scientific preprocessing is changed. Default exports explicitly use
    initial configured bandwidths; an optional compatible trusted checkpoint
    supplies learned parameters. CPU distance ties may differ across devices.

    Plotly is an optional locked extra and is embedded into each HTML alongside
    the data. Browser tests run with external networking blocked. Synthetic
    invariants and bounded real-data exports are recorded in
    [evidence/encoder_visualization.json](evidence/encoder_visualization.json).
    They establish inspectability and forward/gradient parity, not learned
    representation quality or manipulation success.

15. **Action-only training startup (19 October follow-up).** A user-reported
    H200 job spent approximately 26 minutes before writing its training manifest.
    Source inspection found two full-episode scans, including unnecessary visual
    payload reads, to build the decision index and fit action normalization.
    Startup now makes one pass over source/action arrays, retaining only the
    candidate index and small FP64 action moments. It checks visual array headers
    without reading/decompressing the point-cloud, point-mask or proprioception
    payloads. This requires neither data re-preparation nor a persistent cache.

    A candidate is usable exactly when its immediate aligned command has seven
    available joint components; canonical chunk continuity masks prevent a later
    command from rescuing a gap. Normalization still includes every finite valid
    execution action component once, in the same episode order, including partial
    commands. It excludes conditioning actions and does not count overlapping
    chunks. Data identity, alignment verification, sampling order, model and
    optimizer configuration are unchanged, preserving checkpoint compatibility.

    Startup emits bounded per-task progress and measured durations. Policy
    training additionally logs its first completed update per attempt; pre-update
    duration metrics use the validated resume step rather than claiming an
    optimizer update occurred. No extra inference or inner-example synchronization
    is introduced. Full-size HPC startup/throughput remains unmeasured after the
    fix; the bounded local comparison and correctness checks are recorded in
    [evidence/training_startup.json](evidence/training_startup.json).

16. **Runtime training diagnostics (19 October follow-up).** A further report of
    an hour without metrics from the original job cannot distinguish a slow
    update from a stalled operation. `--log-every` overrides scalar cadence for
    policy/router training and the HPC planner without mutating scientific
    configuration or resume provenance. Router held-out validation keeps its
    original cadence, so additional scalar rows do not add validation passes.
    Planner identity excludes only validated logging override pairs; generated
    commands retain them, and effective settings are recorded as runtime metadata.

    Optional `--progress-every-seconds` policy heartbeats report the host phase,
    per-rank example counters and completed/in-progress optimizer steps to stdout
    and `progress.jsonl`. A rank-zero background thread reads Python scalar
    snapshots and monotonic clocks only. It does not query CUDA, synchronize
    tensors, use RNGs, run inference or call W&B. Completed-update W&B rows add
    latest-update wall time and sample-loading time; heartbeat phase times are
    host observations, not kernel profiling. Checkpoint and preview costs are
    excluded from latest-update time and remain in cumulative throughput.

    Bounded CPU tests compare complete model, optimizer, scheduler, scaler,
    sampler and RNG states across different logging frequencies and across
    stop/resume. They also cover heartbeats while the main thread is blocked,
    exception cleanup, nonprimary ranks and dry-run planner propagation. These
    checks establish diagnostic correctness, not full-data H200 throughput.

17. **Sampling-cycle diagnosis (20 October follow-up).** Read-only W&B history
    from real teacher run `1919882fa3ac237d` confirms synchronized 16-update
    cycles in loss and duration. The existing implementation shuffles each
    task's candidates, sorts consecutive groups of 32 by execution decision
    index, and round-robins through 16 tasks. An effective batch of 32 consumes
    two decisions per task, synchronizing all tasks' short-to-long ramps over
    16 optimizer updates. This is an unintended curriculum produced by the
    initial length-bucketing choice, not a useful padding optimization for
    microbatch-one training.

    Timing correlates with per-update history-leaf counts at 0.9981. The loss
    normalization is by valid action dimensions, not history length; batch
    episode-stage/target/context composition provides a supported common cause
    for the observed loss cycle, without claiming its components have been
    causally isolated. Smooth LR warmup has no 16-update reset; checkpoint and
    scalar-reporting work are outside `step_seconds`.

    A synthetic metadata-only control with `length_bucket_size: 1` removes
    the periodic ordering while preserving candidate coverage and task balance.
    No sampler/configuration modification or training intervention was applied
    during this investigation. Such a change must be explicit because it
    changes optimizer batch composition and exact-resume provenance. Reordering
    alone does not reduce total full-history compute. Four GPUs at the same
    global batch size preserve the cycle. Measured scope and reproduction
    information are recorded in
    [evidence/training_sampling_cycle.json](evidence/training_sampling_cycle.json).

18. **True policy batches and system telemetry (20 October follow-up).** The
    teacher, compact policy and Perceiver trainer now execute multiple decisions
    in one forward/backward pass. Default global batch remains 32; microbatch 8
    means four accumulation rounds on one GPU or one round on each of four GPUs.
    The configured microbatch is capped at the per-rank update size. A smaller
    final microbatch receives its proportional weight; each decision first
    averages its valid action dimensions, preserving the original objective.

    Raw histories are packed into independent source-local blocks, with up to
    32 blocks encoded together. Geometry stays FP32 under autocast, checkpointed
    encoding remains available, and decoder/text padding receives explicit
    masks. No history truncation or point/leaf subsampling is introduced.
    The full teacher reads tensor leaf banks directly, avoiding construction of
    unused Python hierarchy parents and per-leaf CPU synchronization. Source
    clocks, completed/pending metadata and current/history separation match the
    reference archive. Cached compact reads still use the exact TREE/FLAT
    selection implementations. Canonical distance ties now use point order as
    an exact secondary key in both serial and batched visual paths, so padded
    shape changes cannot alter neighbor membership.

    The per-task sampler now shuffles without sorting length buckets. Global
    task balance and epoch candidate coverage are preserved; rotating rank
    offsets prevent a permanent task subset on each GPU. Training engine,
    sampler and rank assignment versions are recorded in checkpoint provenance.
    Old runs are intentionally not resumable through this upgrade. New batched
    checkpoints retain deterministic optimizer-boundary resume. DDP encoding
    runs inside the wrapped forward and `no_sync` covers both forward/backward.

    W&B uses `x_disable_stats=False` and hardware metadata by default, controlled
    by `tracking.system_stats`. Its background service collects telemetry on
    rank zero; no model-watch hooks or parameter/gradient histogram uploads are
    introduced. Actual offline SDK files contain GPU utilization, memory, power,
    temperature and CPU/RAM statistics. New scalar metrics expose actual
    microbatch size, accumulation rounds, history lengths and context sizes.

    Synthetic tests compare serial/batched predictions and gradients, invalid
    padding, empty banks, causal perturbations, exact leaf metadata, selected
    TREE/FLAT/Perceiver contexts, remainder weighting and two-rank CPU DDP.
    Full-shape local CUDA measurements and their limited scope are recorded in
    [evidence/batched_training.json](evidence/batched_training.json). They are
    implementation/throughput diagnostics, not evidence of learned policy
    quality, H200/B200 throughput or correction of the separately documented
    historical point-cloud coverage problem.

19. **Larger policy batch requested by the user (20 October follow-up).** Core
    policy defaults are now effective batch 64 and per-GPU microbatch 16. Four
    GPUs each process 16 decisions in one forward/backward pass per update;
    one GPU accumulates four passes of 16. The 50,000-update teacher schedule
    therefore draws 3.2 million examples rather than 1.6 million. This is an
    explicit exposure change, not an equal-exposure speed comparison.

    Learning rate, optimizer-step schedule, encoder block batch size 32 and
    system-statistics collection retain their existing settings. Compact and
    Perceiver policies inherit the core batch defaults; debug configurations
    and the router's explicit effective batch keep their own overrides.
    Existing batch-8 benchmark evidence remains labeled with its measured
    configuration. The new defaults have configuration/inheritance validation;
    no new training or Slurm job was launched for this edit.

20. **Background loading and activation memory (20 October follow-up).** At the
    user's request, policy training now loads and packs upcoming microbatches
    in persistent CPU worker processes. Core defaults are two workers per
    rank and `training.prefetch_factor: 1`, bounding outstanding work at two
    microbatches per rank. Four GPUs use eight loading workers plus four
    training processes within the current 16-CPU allocation. Workers use
    `spawn`, independent HDF5 access and one Torch thread each; the parent
    supplies its existing decision index rather than repeating startup scans.
    They prepare CPU inputs only. Trainable encoding, stochastic policy
    draws and context selection remain in the training process, as do frozen
    feature-cache reads for compact/Perceiver training.

    Core `model.checkpoint_blocks` is now false, explicitly trading additional
    activation memory for avoiding recomputation during backward. Optimizer
    checkpoint saving is unaffected. The global batch remains 64, microbatch
    remains 16, and the sampler, rank assignment, exact history coverage and
    matched TREE/FLAT supervision remain unchanged. This change does not
    implement bulk HDF5 reads, prefix-only loading or rank rebalancing.
    `training.workers: 0` retains synchronous loading with the same packing;
    debug configurations use it to avoid process overhead in tiny checks.

    Process startup affects the first batch. A bounded queue is not a claim
    of bounded bytes independent of history length, nor does it preload the
    full dataset. Use a fresh run directory for the changed core settings;
    prepared observations and verified alignment records are reused.
    The full CPU suite passed 425 tests; a subsequent five-case integration
    run added two compact/Perceiver parity cases, giving 427 distinct passing
    tests across commands. Checks cover identical zero/two-worker training
    state, stop/resume, two-rank CPU DDP, bounded work, persistent processes,
    cleanup, retry inputs and both action-alignment conventions.
    A three-update RTX 5000 Ada diagnostic used two verified real episodes,
    256 points, global batch 64/microbatch 16 and the new defaults. It
    completed with finite losses and a checkpoint; the short episodes fit
    the worker caches. Measurements and scope are in
    [evidence/prefetch_loader.json](evidence/prefetch_loader.json).
    No full-data H200/B200 throughput improvement or model-quality result
    is claimed.

21. **Larger CPU allocation for loading (20 October follow-up).** Following the
    user's request to allocate 64 CPU cores, `hpc/site_4gpu.env` now requests
    `STMEM_CPUS=64`, and core policy configurations use eight loading workers
    per rank with prefetch factor one. Four training ranks therefore have 32
    loading processes and at most 32 queued/in-flight microbatches in total.
    CPU allocation is shared by the single Slurm task's four training processes
    and their workers. Debug configurations retain zero workers.
    This increases available input preparation concurrency; storage bandwidth
    and scheduler resource availability still determine its practical benefit.
    Previous two-worker measurements remain labeled with their actual settings.
    All 28 configuration tests passed. A local planner dry-run produced one
    four-GPU job with literal `--cpus-per-task=64`, eight workers per rank,
    prefetch factor one and valid shell syntax.
    No training or Slurm submission is part of this configuration change.

22. **Standalone teacher memory profiling (20 October).** The profiler lives in
    `scripts/profile_teacher_memory.py`; normal training, model modules and
    configuration defaults are unchanged. It imports the production data index,
    packed loader, model and loss, and matches the teacher's AdamW, schedule,
    equal-example accumulation and DDP synchronization. Ten complete optimizer
    updates warm the model and allocate AdamW state before five measured updates.
    One subsequent update adds an operator trace and bounded CUDA allocation
    history. Warmup and trace measurements remain separate from the baseline.

    Temporary wrappers record encoder, adapter, decoder, backward and optimizer
    phases. CUDA events avoid synchronizing at each phase; synchronization at
    update boundaries makes timings comparable. Peak counters are harvested
    before every reset, including work outside named phases. Phase peaks include
    memory live at entry and cannot be added. PyTorch allocator counters do not
    account for external allocations such as NCCL; end-of-update free memory is
    not an exact peak headroom measurement.

    The normal sampler is the default. An explicit long-history diagnostic
    selects the latest valid decisions in the longest 10% of episodes per task,
    retaining complete causal prefixes and task balance. Length-based selection
    is diagnostic bookkeeping, never a policy feature. Neither mode consumes
    official validation/test seeds, resumes a checkpoint or saves teacher weights.
    Existing output paths are rejected; profiling has a separate W&B identity.
    Failures preserve their original exception and save available phase/memory
    diagnostics. Completed baseline measurements survive a failed detailed trace.

    `hpc/profile_memory.py` renders one allocation under
    `hpc/generated/profiles/<name>`; results go under
    `$STMEM_RUN_ROOT/profiles/<name>`. Only explicit `--submit` calls Slurm.
    Tests establish exact CPU model/optimizer parity, isolation, causal sampler
    behavior, memory accounting and failure recovery. Two-rank Gloo checks
    include spawned loader workers. A bounded real-data RTX 5000 Ada check
    validates CUDA traces and snapshots; its reduced-data/point-count scope and
    measured results are recorded in
    [evidence/teacher_memory_profiler.json](evidence/teacher_memory_profiler.json).
    Full-data H200/B200 capacity and NCCL behavior remain unmeasured locally.
