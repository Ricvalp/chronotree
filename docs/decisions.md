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

23. **Explicit 100k-update teacher experiment (20 October follow-up).** At the
    user's request, `configs/teacher.yaml`
    sets 100,000 optimizer updates, global batch 128, microbatch cap 64 and
    encoder block batch size 128. Four H200 ranks receive 32 decisions each,
    so the actual microbatch is 32 with one forward/backward pass per update.
    This deliberately changes total sampled exposure to 12.8 million decisions,
    four times the previous 50,000-update/global-64 budget. The learning rate stays
    at the inherited value and decay follows the new total update count.
    Immutable validation candidates are fixed at 50,000 and 100,000 updates;
    neither is automatically selected.

    Use the existing `hpc/plan.py --plan core_seed0 --only seed0_teacher`
    workflow. Both core and central-replication manifests now select the same
    50k/100k teacher candidates, so later validation, selection and cache stages
    reference the checkpoints actually produced. The teacher override leaves
    base, compact and Perceiver defaults unchanged. The existing
    `hpc/site_4gpu.env` points to a fresh
    `runs/teacher_b128_m64_blocks128_100k` root. The existing `--output-dir`
    option selects `hpc/generated/core_seed0_validation_h200_4gpu_100k`,
    separating submission records from the earlier four-GPU plan. Resume and
    downstream stages use those same site and plan paths.

    One planner remains responsible for training submission. No new command-line
    options or submission implementation were needed. The redundant teacher
    launcher, its separate experiment YAML and dedicated tests were removed.
    Standard source configs define the run; generated files remain outputs.
    The diagram was regenerated because it displays teacher batching settings.
    The policy architecture and scientific data contracts are unchanged.

    Seventy-five relevant tests passed across configuration contracts, planner
    logging, W&B exports, infrastructure and the existing profiler's config
    preservation. Checks include both experiment manifests' candidate agreement,
    the selected teacher command and output paths, one simulated submission and
    active-job resume without duplicate submission. Shell syntax, Ruff and
    diagram freshness checks passed; the updated diagram was inspected in
    Chromium. No training or real submission ran locally; the user will submit
    on the HPC. These checks do not establish the larger batch's memory peak or
    throughput.

24. **Teacher learning-rate variant through the existing planner.** The requested
    higher-maximum comparison lives in `configs/teacher_lr001.yaml`, inheriting
    `teacher.yaml` and changing only `training.lr` from 0.0001 to 0.01. The
    schedule multiplier is unchanged: 1,000-update warmup, the same cosine decay
    over 100,000 updates and a final rate of 10% of the maximum (0.001).
    Existing teacher batching, data, model and checkpoint candidates are retained.

    `hpc/plan.py` now accepts `--teacher-config` and `--run-root`. The former
    selects the teacher YAML for training, its evaluations and cache generation;
    the latter consistently overrides generated configuration, job environment,
    stage markers and output paths. The existing `--output-dir` separates plan
    and submission records. No additional launcher or site file is introduced,
    and compact/router/Perceiver source settings are unchanged. Teacher candidate
    compatibility is checked before rendering; protocol freeze records and checks
    the selected teacher source and its inherited hashes.

    Sixty-eight configuration, planner, W&B and infrastructure tests passed,
    including variant inheritance, relative paths, output isolation, downstream
    config selection and frozen-protocol checks. A local dry-run of the actual
    variant verified four GPUs and unchanged warmup/batching/checkpoints; shell
    syntax, Ruff and the unchanged architecture diagram check passed. No training
    or real Slurm submission was performed.

25. **Compact live evaluation dashboard.** Evaluation now publishes only running
    success percentage and episode progress percentage, rather than recursively
    expanding every task, outcome, latency distribution and memory measurement
    into W&B history. The running success denominator includes every recorded
    failure, error and timeout. It is explicitly a monitoring statistic, distinct
    from the fixed-plan task-macro score in the scientific `summary.json`.
    An unrun evaluation has no success estimate. Counts, last outcome and errors
    remain in W&B summary metadata, with no per-episode metric namespace.

    Detailed aggregates are retained locally in `diagnostics.json`; per-episode
    decisions, actions, errors and measurements are unchanged. Aggregate reports
    retain the full diagnostics. Logging happens initially and after durable
    episode writes, outside measured inference. Resume progress includes all
    saved outcomes even across gaps, and video selection follows the fixed
    episode positions. Evaluation bookkeeping axes are hidden from automatic
    charts; system monitoring remains enabled. Existing run history/panels and
    running processes are not rewritten, and the source-hash resume guard is
    preserved.

    Sixty-six targeted CPU-safe tracking, evaluation protocol and infrastructure
    tests passed, including real offline W&B serialization/hidden metrics, all
    16 task names without extra chart keys, failure/timeout denominators, empty
    shards and resume across missing episodes. Ruff and diff whitespace checks
    passed. No simulator rollout, training, online W&B upload or Slurm submission
    was performed. The required diagram check reports a pre-existing mismatch:
    committed `configs/teacher.yaml` has effective batch 256, while the committed
    diagram snapshot displays 128. Only that config hash differs from the diagram
    snapshot; this logging change touches none of its fingerprinted sources.

26. **Separate evaluation groups within the existing W&B project.** New
    `job_type=evaluation` runs prefix their existing experiment/path-hash/seed
    group with `evaluation-`. Training and auxiliary-stage groups, project,
    logical run IDs, job types and tags are preserved. Historical or running
    remote runs are not edited; their existing evaluation job type can be used
    for filtering. The video cap remains two per shard, with the existing fixed
    episode selection and local storage.

    Thirty-two tracking/evaluation tests passed, including checks of group names,
    unchanged projects and stable IDs/groups across real offline W&B restart
    segments. Ruff and whitespace checks passed. No online metadata was changed
    and no Slurm job was submitted. The pre-existing diagram mismatch described
    above is unaffected by this tracking-only change.

27. **Approved balanced geometry and 32 centers per frame (21 October 2026).**
    After reviewing matched real-scene HTML previews, the user selected spatially
    balanced clustering with greater above-table point coverage. Core defaults
    now sample 2,048 visible points and reserve 75% for z>5 mm, balanced over
    10 cm XY cells after the same 5 mm voxel reduction. Remaining slots sample
    all unselected geometry; a sparse elevated set returns its unused budget.
    This uses only observed RGB-D and the calibrated table-height prior. It is
    not object segmentation; visible robot points qualify too.

    Both current and historical observations now use 32 observed-point centers:
    24 elevated plus eight global seeds, initialized by stratified FPS. Each
    occupied 10 cm XYZ cell has total weight one. Weighted Lloyd updates refine
    elevated centers, then hold them fixed while refining global centers; each
    centroid snaps to an observed cluster member. The cap is 20 iterations per
    stage. Selection is frame-local and deterministic, with no future frames,
    object identities, labels, learned features or target actions. CPU refinement
    retains the preview's FP64 internal geometry on FP32 inputs; model geometry
    and Gaussian pooling remain FP32. The original neighborhood radius, learned
    bandwidths and 128-neighbor cap are unchanged.

    A complete eight-frame block contains 256 visual and eight event leaves;
    the current context contains 32 visual plus three auxiliary tokens. Exact
    leaf coverage, separate source banks, pending-prefix construction and matched
    TREE/FLAT supervision remain unchanged. The model parameter count remains
    6,098,698; leaf count and activation cost increase. Existing effective-batch
    256/microbatch-cap 64 settings are preserved but require fresh target-device
    memory profiling. No full-history H200 capacity or throughput is claimed.

    `canonical_balanced_v2` preparation writes a separate
    `prepared_balanced_v2/` directory, with the smaller debug data under
    `prepared_balanced_v2_debug/`. It saves fixed-length original-cloud center
    indices for each required quota, padding only with -1. No larger selection
    is truncated to emulate a smaller quota. Both selector recipes, versions and
    quotas enter manifest identity; a different existing prepared format cannot
    be overwritten. Configured preparation, training and checkpoint validation
    require explicit selector settings, so old resolved YAML snapshots cannot
    silently acquire new preprocessing defaults.

    The online adapter computes the same indices once per arriving observation.
    Dataset loading and background packing transport indices and a settings
    signature; batched GPU pooling only remaps/gathers these anchors. Missing or
    incompatible packed indices fail clearly. Synthetic/scalar diagnostic frames
    can compute and cache their indices on CPU. Startup still reads only small
    supervision arrays and center layout metadata, preserving the action-only
    startup optimization. Existing verified raw-revision alignment remains valid.

    Tests cover exact reviewed-cloud/center parity, sampling invariants, prepared
    identity and old-format protection, online/offline equality, padding and
    canonical index remapping, scalar/batched gradients, prefetch/resume/DDP,
    source causality and unchanged scientific invariants. A four-record real
    HDF5 diagnostic matches online ingestion exactly and deliberately remains
    alignment-unresolved. A bounded RTX 5000 Ada BF16 forward/backward used two
    eight-frame real replay blocks at the new 2,048/32 geometry: 512 valid leaves
    and finite gradients, with CPU selection forbidden inside GPU execution.
    Small FP32 CPU/CUDA outputs and gradients agree (maximum output difference
    5.96e-8). These are implementation checks, not a trained-policy evaluation.
    The bounded real-data/device measurements and implementation hashes are in
    [balanced_geometry_integration.json](evidence/balanced_geometry_integration.json).

    Final `uv run pytest -q`: 503 passed, one skipped in 88.64 seconds. The skip
    requires the optional new-format debug prepared subset; the bounded real
    RGB-D integration above runs independently. An earlier full pass attempt
    had one intermittent PyTorch worker SIGABRT during persistent DataLoader
    teardown, after the test body. Its five-test module then passed, all eight
    fresh isolated repetitions passed, and the final full suite passed. The
    local cause is unproven; no error suppression or private queue workaround
    was added. Ruff, whitespace, batch-script syntax, the generated-config
    dry-run and the architecture snapshot/XML checks pass. All four diagram
    pages were browser-inspected; no Slurm submission was made.

    Matched previews improved nearest-center geometry and table coverage but
    showed a trade-off: one PickXtimes history block pooled 1,365/1,403 cube-point
    observations versus FPS's 1,403/1,403, both preserving all eight frames. No
    blanket object-coverage or success-rate improvement is claimed. The diagram,
    data contract and HPC preparation instructions are updated; no full-data
    preparation, long training or real Slurm submission was performed locally.

28. **Explicit four-B200 launch (22 October 2026).** The user requested four
    B200s, microbatch 64 per GPU and 128 encoder blocks per call. Added
    `b200_4gpu` to the shared planner/profiler profile table. The new tracked
    `hpc/site_b200_4gpu.env` uses partition `aiq` and typed GRES `b200`, both
    confirmed by the user's `sinfo -o '%P %G %f'` output. A teacher allocation
    requests `gpu:b200:4` on one node; one Slurm task launches one torchrun agent
    with four workers. GPU type selection remains a site setting, not inferred
    from the profile label. The existing H200 site is unchanged.

    The launch explicitly selects `configs/teacher.yaml`, which already has
    effective batch 256, microbatch 64, encoder block batch 128 and 100,000
    updates. No model, data recipe, learning rate or schedule was changed.
    A fresh B200 run root and generated-plan directory isolate checkpoints,
    tracking identity and submission state. The site retains 64 CPU cores,
    128 GB host memory, a 48-hour requested limit and online W&B. The limit's
    acceptance by `aiq` remains unverified; the supplied node listing does not
    report partition limits. The architecture diagram was reviewed and its
    snapshot check passes; displayed batching and architecture are unchanged.

    Forty-three focused planner/profiler tests passed, including typed GRES,
    one-agent/four-worker launches, fake-srun execution, configuration inheritance
    and artifact isolation. A dry-run using the actual new site with only local
    filesystem/Python paths substituted produced one teacher job with
    `--partition=aiq --gres=gpu:b200:4`, the requested batch sizes and the new
    balanced dataset. Generated batch syntax, lint and whitespace checks pass.
    No Slurm submission, GPU benchmark or training was run; B200 memory fit,
    throughput, scheduler acceptance and full-data execution remain unmeasured.

29. **Smaller B200 retry after reported OOM (22 October 2026).** The user reported
    that the 64-example / 128-encoder-block B200 job exhausted memory and asked
    to reduce it. Added `configs/teacher_b200.yaml`, extending the teacher with
    microbatch 16 and encoder block batch 32. Effective batch 256, all causal
    history, geometry, learning rate, checkpoint schedule and 100k updates remain
    unchanged. Four GPUs accumulate four passes of 16 decisions per update.
    Smaller encoder chunks reduce temporary tensors; saved activations across
    chunks still remain until backward because activation checkpointing is off.

    The B200 site now chooses a fresh `m16_blocks32` run root; documented launch
    commands select the new YAML and a fresh generated directory without resume.
    They set `PYTHONPATH` to the current checkout's `src`, avoiding the older
    evaluation worktree exported by the intervening legacy-evaluation recipe.
    The default `teacher.yaml` and its architecture diagram remain unchanged;
    the diagram source/settings review and snapshot check pass.

    Fifty-six config/planner tests passed. A relocated-site dry-run verified
    `aiq` / `gpu:b200:4`, the requested retry sizes, four accumulation passes,
    unchanged model/data configuration and generated batch syntax. No job was
    submitted or training run locally. The failure traceback was not supplied;
    CUDA versus host-memory exhaustion is unconfirmed. Fit and throughput of
    the smaller configuration still require measurement on the cluster.

30. **Further B200 reduction after a second OOM (22 October 2026).** The user
    reported another OOM at 16 examples / 32 encoder blocks. The same B200
    variant now uses microbatch 8 and encoder block batch 16, with a fresh
    `m8_blocks16` run root and generated directory. Effective batch 256 is
    retained through eight passes per rank on four GPUs. No history, geometry,
    schedule, activation-checkpointing or loader-worker setting was changed.

    A read-only accumulation audit found backward runs after each microbatch
    and loss logging detaches its value; training does not retain all eight
    microbatch graphs. Encoder chunk activations still accumulate within each
    microbatch, so the smaller microbatch is the main retained-activation
    reduction. The error log was requested to distinguish CUDA from host RAM
    exhaustion; neither the failure phase nor memory fit has been measured.

    Thirty configuration tests passed, together with a relocated-site planner
    dry-run verifying four typed B200s, eight accumulation passes, unchanged
    model/data/budget, separate output paths and generated batch syntax. The
    default-teacher architecture diagram remains unchanged and its snapshot
    check passes. No Slurm submission or training was performed locally.

31. **Traceback corrects the B200 failure diagnosis (22 October 2026).** The user
    subsequently supplied `torch.AcceleratorError: CUDA-capable device(s) is/are
    busy or unavailable`, identified as `cudaErrorDevicesUnavailable`, from
    `distributed_setup()` at `torch.cuda.set_device(local)`. This happens before
    the data loader, policy, optimizer or training batch is constructed. The
    supplied trace is not evidence that the requested microbatch exhausted
    device memory. Entries 29/30 describe changes based on the user's initial
    OOM report; earlier allocations' logs have not been compared.

    Read-only review confirms the intended single Slurm task / one torchrun
    agent / four LOCAL_RANK workers, with Slurm's CUDA visibility preserved. No
    definite launcher defect or premature repository-level CUDA context creation
    was found. NVIDIA documents unavailable devices as potentially caused by
    prohibited/exclusive compute mode or already-active device work; the exact
    site cause remains unknown. Allocation accounting, GPU visibility/mode and
    process diagnostics, and a small CUDA initialization test on the B200 node
    are required before further tuning. No speculative binding changes, driver
    changes, checkpointing changes or batch-size reversal were made. The 8/16
    variant remains available but cannot fix a device-selection failure.

    The user's subsequent four-worker standalone probe reproduces the same
    failure on `ainode01` with Torch 2.9.1+cu128, driver 580.126.20 and four
    visible B200s. All workers report `CUDA_VISIBLE_DEVICES=0,1,2,3` and device
    count four. The pre-launch `nvidia-smi` snapshot shows 0 MiB used, Default
    compute mode, disabled MIG and no running processes for all four devices.
    Rank 3 is the first reported failure at `set_device`; torchrun terminates
    ranks 0–2 before any success message, so their usability is undetermined.
    This probe imports no Chronotree modules, creates no model, and initializes
    neither W&B nor an NCCL process group. The issue reproduces independently of
    the training implementation. Fresh sequential Python processes within a
    four-B200 allocation are the next diagnostic, followed by a node comparison
    if needed. Driver/device access and fabric readiness remain hypotheses,
    not established diagnoses. No further batch or launcher changes were made.

32. **Remove saved dense geometry after a real B200 OOM (23 October 2026).**
    The user reports a four-GPU job failing after roughly 600 updates. The log
    identifies `ainode04`, a rank-zero decoder K/V allocation failure requesting
    702 MiB, 162.76 GiB actively allocated by PyTorch, and 13.32 GiB reserved but
    unused on a 178.35 GiB device. This is a genuine training OOM, distinct from
    the earlier device-initialization failure. The job followed the 16/32 retry
    instructions, but its resolved configuration and memory timeline were not
    supplied. Variable history lengths, padded decoder contexts and fragmentation
    can raise the peak; the traceback alone does not establish a growing leak.

    Auditing the trainer found backward after every microbatch, detached metric
    accumulation and bounded CPU-only prefetch. Encoder outputs from every chunk
    of one microbatch remain differentiable until that backward. Consequently,
    lowering encoder chunk size alone does not bound all retained activations.
    In particular, dense point-to-center Gaussian distances retained candidate
    tensors for the two learned bandwidth derivatives even though only the
    selected neighbors contribute. Batched pooling now performs the unchanged
    discrete ranking without autograd, gathers spatial/temporal distances, and
    differentiates their weighted distance only for the selected neighbors.
    Dense temporaries are released before edge feature processing. No points,
    centers, history frames, masks, causal boundaries, supervision, precision,
    model parameters or activation-checkpointing defaults are changed.

    The unchanged scalar encoder supplies an independent dense reference. New
    synthetic CPU checks cover content and occupancy, all output fields, learned
    bandwidth and MLP gradients, input-point/time gradients, equal-distance ties,
    invalid points, absent centers, empty blocks and framewise/joint neighborhoods.
    A saved-tensor regression fails against the previous implementation and
    passes after the fix: the small fixture's dense saved geometry falls from
    14,688 bytes to zero. The architecture diagram's neighborhood tooltip and
    source fingerprints are regenerated with the same mathematical layers,
    tensor shapes and causal boundaries.

    On the available RTX 5000 Ada, the actual-device doctor passes FP32, BF16 and
    FP16. A bounded synthetic BF16 encoder comparison uses four independent
    eight-frame blocks, 2,048 points/frame, 32 observed-point centers/frame,
    128 neighbors/center and width 256, without activation checkpointing. All
    output fields are bit-identical; maximum parameter-gradient difference is
    1.49e-8. Additional live memory at the end of forward falls from 613,761,024
    to 412,368,896 bytes; forward/backward peak above the pre-forward baseline
    falls from 677,018,624 to 487,831,040 bytes (27.9%). This is an encoder-only
    diagnostic, not a full-policy, long-history, DDP or B200 capacity claim.
    See `docs/evidence/encoder_distance_memory.json`. No data preparation,
    cluster submission or long training was performed. The existing 8/16 B200
    variant remains the conservative retry with effective batch 256; memory fit
    over all real histories still requires the user's cluster run.

    The complete CPU-safe invariant suite passes: 510 passed, 1 skipped in
    90.99 seconds; the optional real small prepared subset is absent. The diagram
    generator's snapshot and XML validation pass after regeneration, and the
    generated HTML was opened in headless Firefox and visually inspected.

33. **Selective previous-joint-command ablation (26 October 2026).**
    Following command-sensitivity and expert-prefix handoff diagnostics, the user
    requests a fresh teacher run without previous commanded joint targets.
    `model.condition_on_previous_joint_commands: false` now zeros the seven joint
    command channels before the current state MLP, every historical event MLP,
    and the compact readers' routing query. The eighth command channel (gripper),
    command availability, all measured proprioception, visual observations,
    conditioning video and source clocks remain available. This is a targeted
    input ablation, not removal of execution history or measured joint angles.
    Current, serial history, packed training, streaming evaluation, checkpoint
    inspection and cached-feature construction share the setting. Tensor shapes,
    parameter counts, leaf coverage, causal alignment and eight-dimensional
    supervision are unchanged. Raw prepared data is unchanged; no preparation
    rerun is needed. Checkpoint contract validation rejects switching the flag at
    evaluation, and frozen-cache identity includes it. Missing flags in older
    checkpoints retain the original enabled behavior.

    `configs/teacher_b200_b128_m8_blocks32_no_previous_joint_commands.yaml`
    inherits the existing low-LR recipe and changes only this flag: 100,000
    updates, peak LR 0.0001, effective batch 128, per-GPU microbatch cap 8,
    encoder block batch 32 and checkpointing off. On four GPUs this uses four
    accumulated microbatches per optimizer update. Use a new run root and omit
    `--resume` for the requested random initialization. This run tests whether
    removing command conditioning improves learned visual grounding; the
    diagnostics do not establish that this change will fix covariate shift.
    Autocast scopes, geometry, targets, action normalization and flow sampling
    have not been changed as additional experimental factors.

    The 28 new CPU synthetic diagnostics verify exact prediction invariance to
    commanded-joint perturbations across serial, batched, worker-packed, cached
    and checkpoint-reloaded inference paths; retained gripper/state sensitivity;
    unavailable commands; gradients; routing; inspector parity; and config/cache
    identity. The affected model, training, configuration, inspector and launcher
    suites pass (180 tests). A local-path planner dry-run generated one fresh four-GPU job with
    `--partition=aiq --gres=gpu:4`, 64 CPUs, 128G host RAM and 48-hour walltime;
    resolved hyperparameters and shell syntax passed. The actual-device doctor
    passed FP32, BF16 and FP16 on the workstation's RTX 5000 Ada. Architecture
    artifacts were regenerated and inspected in Chromium. These are correctness
    checks, not a real-data training or policy-quality result. No Slurm job was
    submitted. One initial existing loader test aborted a worker at teardown;
    rerunning its five-test module passed without loader changes.

34. **Matched continuations from one teacher checkpoint (26 October 2026).**
    The fresh command-ablation run is additional to the requested control and
    ablated continuations. `--fork-from` explicitly creates a new teacher run
    while restoring the parent's model, normalizer, Adam moments, scheduler,
    scaler, optimizer position and per-rank RNG. Model changes are restricted
    to `condition_on_previous_joint_commands`; data/geometry/seed, remaining
    model and training settings, effective batch, world size, sampler version
    and next sample position must match. Ordinary `--resume` retains its strict
    provenance check. A fork rejects an occupied destination, records parent
    checkpoint hash/step/W&B identity in its manifest/checkpoints, and creates a
    new W&B run. Subsequent ordinary resume preserves that lineage.

    The standalone `hpc/continue_teacher.py` prepares both branches from one
    immutable snapshot, separate configurations, output roots and Slurm scripts.
    Neither branch depends on the other. The default budget is 10,000 additional
    updates, saving immutable checkpoints at +5,000 and +10,000. The newly
    explicit `training.schedule_steps` retains the parent's cosine horizon;
    `training.steps` declares the bounded experiment endpoint. Thus a parent at
    49,978 with a 100k schedule ends both experiments at 59,978 without compressing
    LR decay into those 10k updates or restarting warmup. Finishing that declared
    experiment exits successfully; interruption before it still checkpoints and
    returns 75. This launcher does not run the core 100k teacher-selection DAG.
    Default planning snapshots/renders only; `--submit` is required for Slurm.
    Submission receipts and locked output reservations prevent duplicate fresh
    runs, and both jobs verify the pinned source hash before loading its state.

    Synthetic CPU integration checks establish exact control-fork parity with
    uninterrupted training at the same step, including parameters, Adam state,
    scheduler, RNG, sampler and losses. The fixture exercises cosine decay after
    warmup, not merely the warmup branch. The ablated fork changes learning while
    keeping normalization and LR schedule. Separate W&B identity, input mismatch
    rejection, and interrupted-fork resume are checked. The continuation and
    launcher checks pass (42 tests), including frozen snapshot reuse after a
    moving `last.pt` advances, source hashing across atomic filename replacement,
    isolated output paths, and mock-only submissions. A local-path dry-run using
    the actual 49,978-step lower-LR inspection checkpoint generated two valid
    shell scripts with a 59,978 endpoint, 100k schedule and the requested B200
    resources. Existing affected training/configuration/tracking suites pass
    (91 tests). These checks do not
    establish B200 throughput or closed-loop success. No real-data training or
    Slurm submission was performed locally.

35. **Recover the three interrupted command-conditioning experiments (27 October 2026).**
    Slurm accounting supplied by the user confirms node-down requeues for the
    matched continuations. Their restarted initial fork scripts correctly refuse
    the occupied run directories. Recovery must use each branch's own saved
    checkpoint, rather than repeat the fork from the parent. The paired launcher
    now supports `--resume` with the original run root and generated directory.
    It validates the original configuration, fork lineage, world size, sampler
    boundary and W&B identity, then writes recovery artifacts under `resume/`
    without changing the original fork plan or any checkpoint. It retains the
    declared 64,978-step endpoint for the 15k experiments and the 100k LR horizon;
    it does not allocate another 15k updates. Existing tracking files preserve
    each branch's separate W&B run.

    Both planners query the current user's full Slurm queue instead of asking
    about expired job IDs individually. Terminal jobs allow recovery; suspended,
    held, requeued and unfamiliar nonterminal states prevent duplicates. A
    failed or malformed queue query aborts submission. Paired submissions hold
    directory locks, retain receipt history and reject ambiguous earlier
    submission results. Dry-run never contacts Slurm. Recovery commands retain
    the live `last.pt` path so a subsequent Slurm requeue reloads the latest
    checkpoint. Paired recovery requires both branches to remain unfinished;
    it does not implement selective recovery of a single completed/unfinished
    pair.

    The separate from-scratch ablation uses the ordinary planner's `--resume`
    and original YAML recipe. Its current model, training, data, geometry, debug
    and seed settings match the W&B configuration recorded for run
    `e69369b3b3887b48`. Its endpoint remains 100,000. Documented shell guards
    require the checkpoint and tracking identity to exist before invoking the
    general planner, which otherwise permits starting unstarted DAG stages.
    All three use their existing run roots; no new data preparation, experiment
    configuration or optimization changes are introduced.

    CPU checks pass: 65 launcher/planner tests, including mocked Slurm states
    and synthetic checkpoint metadata, plus 72 existing fork, tracking and
    training-logging checks. The latter include interrupted-continuation resume
    correctness. Lint, diff checks and the architecture snapshot/XML check pass;
    no model or diagram changes were needed. No Slurm job was submitted locally,
    and the HPC checkpoints remain to be loaded on that system. This recovery
    does not establish the cause of, or fix, the observed GPU/runtime stalls.
