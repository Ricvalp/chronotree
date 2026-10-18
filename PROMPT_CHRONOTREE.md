# Implementation brief: Query-Adaptive Space-Time Supernodes on RoboMME

**Authoritative implementation specification — 17 October 2026.**

Read this file completely, then implement the project in this initially empty repository. The benchmark is **RoboMME** (the user's message spells it RoboMMM). This is an implementation and experiment-infrastructure task, not a request for another proposal.

## 0. Mission, priorities, and scope

Build a runnable, reproducible PyTorch research repository for **history-dependent point-cloud manipulation policies with a bounded active memory context**. Implement the method, strong matched baselines, RoboMME data preparation and closed-loop evaluation, local debugging, and Slurm experiment submission scripts.

The user will develop with you on a Linux workstation with an NVIDIA RTX A5000. Experiments will run later on HPC nodes with up to four H200 GPUs, or optionally eight B200 GPUs. Default to one H200-class GPU per job; make four-GPU training available. Do not assume that allocating more GPUs is faster or more economical. The user has limited time and credits for preliminary experiments and wants to reach the core comparison quickly.

**Priorities, in order:**

1. Correct causal data construction, action alignment, and a working RoboMME evaluation loop.
2. A faithful, compact implementation of the method and a strong flat-retrieval comparison.
3. One single-seed core experiment on all 16 RoboMME tasks.
4. A small number of informative analyses and final replication of the central comparison.
5. Additional sophistication only in response to a diagnosed problem.

Do not turn this into a large architecture search, a foundation-model pretraining effort, or a sequence of multiseed gates. Do not add other benchmarks, real-robot integration, MAML, test-time gradient adaptation, a new action tokenizer, a world model, or a large language QA auxiliary pipeline in this version.

You may make routine implementation choices autonomously. Record significant choices and deviations in `docs/decisions.md`. Fix implementation errors without waiting for permission. Do not stop after writing scaffolding or an implementation plan: deliver working code, tests of the scientific invariants, configuration files, reproducible commands, and job scripts. If a dataset, simulator, credential, or hardware capability is unavailable, finish everything that can be implemented and verified, identify the precise blocked check, and provide its executable command. Do not fabricate successful runs.

**Authorization in this task:** run bounded local debugging and short smoke checks on the A5000; download a small demonstration subset and required public sources/assets when possible. Prepare the full-data download command and HPC jobs. Do not automatically download the entire dataset, run long training, submit Slurm jobs, or launch an unbounded experiment campaign. Job submission must require an explicit `--submit` flag supplied by the user.

This file supersedes earlier informal suggestions where they differ. In particular:

- Use PyTorch, not a port of the complete old ICIL training stack.
- Use deterministic parent aggregation initially.
- Use staged training.
- Use a **shared compact policy trained on both tree and flat contexts** for the primary reader comparison.
- Use the official continuous joint-action interface initially.
- Keep preliminary checks short and targeted.
- Do not hardcode earlier speculative project-wide GPU-hour estimates as an approved spending budget.

## 1. Research hypothesis and honest claim

The policy observes RGB-D point clouds, proprioception, an available task instruction, and its own past commanded actions. Some RoboMME tasks also provide an initial conditioning video. The current observation alone may not determine the correct action.

The hypothesis is that a robot can allocate a limited historical token budget to **different spatial and temporal resolutions of previously observed geometric evidence**, retaining coarse coverage of the past while exposing important regions in detail.

There are three distinct claims:

1. Local space-time supernodes provide useful historical evidence.
2. Query-conditioned resolution allocation improves the use of a fixed active context.
3. A hierarchy provides a useful trade-off relative to flat retrieval from the same evidence.

The initial experiment primarily isolates claims 2 and 3 using the same frozen leaf encoder and policy. It does not, by itself, establish superiority of this tokenizer over every other point-cloud encoder.

A growing archive is permitted. **Bounded active context is not bounded total storage.** All archive, routing, transfer, and encoding costs must be counted. “Recoverable” means relative to the archived leaves, not lossless reconstruction of raw observations.

The decisive comparator is **learned flat retrieval from the exact same leaves, with summaries of the unselected remainder**, trained with comparable action-based supervision. A result against only a current-only policy or FIFO window is insufficient.

## 2. Framework decision: use PyTorch

This is a project-specific engineering choice, not a claim that PyTorch is intrinsically faster.

| Criterion | Weight | PyTorch | JAX |
|---|---:|---|---|
| Fast A5000 debugging | 30% | Eager inspection of geometry, masks, caches, and Python trees | Excellent tools, but tracing/shape issues add initial work |
| Implementation effort for dynamic memory | 25% | Straightforward Python archive plus tensorized policy/router | Feasible with padding and structured control flow, requiring more infrastructure |
| Existing implementation reuse | 20% | Reimplement a small tokenizer; ManiSkill returns Torch tensors | Advantage: original ICIL code and official MME-VLA/OpenPI policy stack |
| Single-node H200/B200 training | 15% | AMP, SDPA, DDP, established tooling | Strong compilation and sharding; no measured speed winner here |
| Dependency management | 10% | Standalone compact policy with optional simulator environment | Also feasible; official JAX policy already uses a separate simulator environment |

Implementation speed and inspectability outweigh reusing an unrelated pretrained VLA stack. The official RoboMME MME-VLA policy implementation is JAX/OpenPI; do not incorrectly describe all official baselines as PyTorch.

Start with eager PyTorch and built-in scaled-dot-product attention. Do not require custom CUDA kernels, Triton, PyTorch3D, or external flash-attention packages. Add `torch.compile` only after correctness and actual profiling; do not compile the Python archive traversal as a prerequisite.

### 2.1 Dependency and hardware setup

- Use a modern, locked Python environment; the audited benchmark requires Python >=3.11.
- Keep core training/data dependencies separate from simulator extras.
- Prefer one compatible environment if it works. If RoboMME's pins conflict with training requirements, provide separate training and simulator environments with a small typed local policy-service interface. Do not force a giant upstream VLA dependency stack into this project.
- Pin the benchmark commit and simulator fork below. Resolve and record a working Torch/CUDA wheel combination on the actual hardware.
- The audited benchmark pins `torch==2.9.1` and `torchvision==0.24.1`. Treat those as the upstream simulator environment's pins, not permission to install arbitrary latest ManiSkill.
- Verify support for A5000, H200, and B200 using current official PyTorch and NVIDIA documentation. Do not infer the Torch runtime CUDA version from the maximum CUDA version displayed by `nvidia-smi`.
- Add a `doctor` command reporting Python, Torch, wheel CUDA build, driver, GPU name/capability, supported architecture list, and a real attention/matmul forward/backward test.
- Use BF16 autocast if an actual device test succeeds. Otherwise use FP16 with GradScaler, with FP32 debugging fallback. Keep geometry, distance computations, normalization statistics, counts, aggregation sums, and sensitive reductions in FP32.
- No FP8 in the first study.
- CUDA availability does not imply working headless rendering. Check Vulkan/RGB-D reset-step-render on the actual evaluation GPU.
- Do not modify system/cluster drivers from installation scripts.
- Compute nodes may have no Internet access. Prepare dependencies, datasets, assets, and optional model weights beforehand. Do not assume Docker is available on the cluster.

## 3. Sources, versions, and reuse

### 3.1 Main benchmark

Audit anchor:

- Repository: https://github.com/RoboMME/robomme_benchmark
- Commit: `6503b26e00ff338611fceb039417fbc64826ccc8`
- ManiSkill fork: https://github.com/YinpeiDai/ManiSkill
- Fork revision: `07be6fbc66350ddca200abfb0a11b692f078f7fd`
- Dataset: https://huggingface.co/datasets/Yinpei/robomme_data_h5
- Paper: https://arxiv.org/abs/2603.04639

Use the pinned snapshot unless a concrete installation or benchmark bug requires changing it. Record patches and the exact revision used. Do not silently switch to upstream ManiSkill or a different action protocol. Inspect licenses and preserve attribution when adapting code.

Important benchmark paths:

- `pyproject.toml`, `readme.md`, `Dockerfile`
- `doc/h5_data_format.md` and `doc/env_format.md`
- `src/robomme/env_record_wrapper/RecordWrapper.py`
- `src/robomme/env_record_wrapper/DemonstrationWrapper.py`
- `src/robomme/env_record_wrapper/EndeffectorDemonstrationWrapper.py`
- `src/robomme/env_record_wrapper/episode_config_resolver.py`
- `scripts/dataset_replay.py` and `scripts/evaluation.py`

Use the official policy code to understand protocol, not as an unquestioned alignment oracle:

https://github.com/RoboMME/robomme_policy_learning

In particular inspect `src/mme_vla_suite/dataset_builder/build_robomme_dataset.py` alongside the recorder: an apparent timing inconsistency is described below.

### 3.2 Original tokenizer reference, optional dependency

The user does **not** need to copy the old repository into this one.

- https://github.com/Ricvalp/icil
- Current local encoder reference:
  https://github.com/Ricvalp/icil/blob/3af009b458c74aa66e695e35b02f9f8801f52138/icil_jax_rlbench/models/ttt_supernode.py
- Older demonstration encoder:
  https://github.com/Ricvalp/icil/blob/187fa8bb353e7bb8cdb80d86651d344083a16d50/icil_jax_rlbench/models/encoders.py

Inspect `SpacetimeEventEncoder` for the Gaussian local pooling idea. Reimplement the useful mechanism in Torch. Do not copy its final register bottleneck into the archive leaves, its full TTT machinery, or any temporally unordered query-history path. Access to this repository is helpful but not required: this specification defines the necessary algorithm.

The earlier research PDF is optional background. Do not depend on an attachment or reproduce all its suggested experiments before the core run.

## 4. Data contract: establish this before training

### 4.1 Dataset and task splits

Use the released training demonstrations: 100 episodes per task, 16 tasks, 1,600 episodes total. The official benchmark has 50 validation and 50 test episode seeds per task. Preserve those split definitions. Do not create train/test splits by randomly splitting frames.

Tasks:

| Suite | Task IDs |
|---|---|
| Counting | `BinFill`, `PickXtimes`, `SwingXtimes`, `StopCube` |
| Permanence | `VideoUnmask`, `VideoUnmaskSwap`, `ButtonUnmask`, `ButtonUnmaskSwap` |
| Reference | `PickHighlight`, `VideoRepick`, `VideoPlaceButton`, `VideoPlaceOrder` |
| Imitation | `MoveCube`, `InsertPeg`, `PatternLock`, `RouteStick` |

Provide task-selective downloads with integrity checks and safe extraction. The released format uses a compressed HDF5 archive per task; do not promise that two episodes can be downloaded independently. An episode limit may restrict preparation/reading after downloading the relevant task archive. Show the required archive size before transferring it, and support an already downloaded file. The dataset listing was about 56 GB compressed at audit time; measure actual download/extracted size before a full transfer. Never extract archive entries outside the configured data directory. Keep large data, caches, assets, and checkpoints out of git.

HDF5 episode/timestep group names must be sorted **numerically**. Do not assume one-based indexing because a schema illustration starts at 1.

### 4.1.1 Validation without assuming unreleased expert data

Official validation/test splits are seeded environment configurations. Do not assume expert HDF5 demonstrations exist for those splits. The default teacher and compact policy use all 100 released training episodes per task.

- Select checkpoints using closed-loop success on one fixed small official validation subset, identical across methods. Start with two seeds per task; this is checkpoint selection within the main run, not a separate pilot campaign.
- Compare at most two planned teacher checkpoints (for example, 25k and 50k updates) and two planned compact checkpoints (15k and 30k). For the shared compact policy use a balanced, fixed tree/flat context protocol at the main budget. Keep this selection rule fixed and record it.
- Split router-label episodes, for example 80/20 within each task's released training episodes, before generating labels. The scorer-fitting set and scorer-validation set must have disjoint episodes, including during refresh. Encoders and policies may already have seen both sets: call this a conditional router-label diagnostic, not unseen-episode end-to-end generalization.
- Report teacher/compact losses on their own training episodes as training diagnostics. Offline held-out action loss requires an explicit episode holdout or separately acquired expert validation data; neither is assumed available by default.
- Use official validation rollouts for end-to-end model selection. Reserve official test seeds for the frozen protocol.

Record the selected immutable checkpoint before dependent cache/training jobs run. The job planner must include checkpoint-selection evaluation and resolution of that checkpoint, rather than assuming the final optimizer step is automatically selected.

### 4.2 A necessary alignment check, not a research gate

At the audited benchmark revision, `RecordWrapper.step(action)` calls `super().step(action)` and then records the returned observation together with that executed action. Under that convention, stored `obs_j` is post-action and should predict `action_{j+1}`.

However, the official MME-VLA converter currently builds an action chunk starting at the same `step_idx` as the observation. The released data may also come from an older recorder.

**Do not blindly copy either offset.** Resolve this once against a few actual downloaded episodes:

1. Inspect the HDF5 record and both source implementations.
2. Replay a short segment using the correct episode configuration and action space.
3. Compare stored proprioception and available observations with pre-step and post-step values.
4. Inspect the conditioning-video/execution boundary and repeated terminal records. Account for reset's automatic initial execution step; blindly replaying the first stored execution command again can create a false alignment mismatch.
5. Choose one explicit, versioned alignment convention for the data revision, and document the evidence in `docs/data_contract.md`.
6. Add a small deterministic regression test.

Do not choose an offset independently per sample by nearest-pose matching. If different dataset revisions have different conventions, register them explicitly. Do not launch substantive training while the alignment remains unresolved. This is a targeted data-correctness check, not a demand for learning curves or multiple seeds.

Canonical internal decision records must mean:

- `observation_t`: available before the predicted command.
- `past_executed_actions`: commands already issued before that observation/decision.
- `target_chunk`: the next H expert commands under the resolved alignment.
- `target_valid`: masks padding, terminal suffixes, unavailable dimensions, and source boundaries.
- `conditioning_bank`: observations legitimately supplied before execution.
- `execution_prefix`: only current-episode observations already available.
- Versioned source timestamps/indices, calibration, and alignment metadata.

### 4.3 Allowed model inputs

Use an explicit allowlist; never pass the full benchmark `obs/info` dictionaries into the policy.

Allowed:

- Front and wrist RGB and depth.
- Camera intrinsics and each frame's extrinsics.
- Observed joint positions, end-effector pose, and gripper state.
- The task-goal text actually supplied by the environment.
- Previously executed commands generated by our own policy, or the correctly aligned expert history in offline training.
- Source tags and recorded-frame timing needed to distinguish initial conditioning from execution.

Do not provide simulator object poses/IDs, hidden contents, oracle subgoal strings, grounded subgoal strings, keyframe/subgoal-boundary annotations, completion flags, episode IDs, seeds, difficulty labels, target actions, or future observations as policy/router features. Those may be retained separately for bookkeeping or explicitly labeled offline diagnostics.

Use `is_video_demo` only to reconstruct the source boundary and action-supervision mask that are observable from the reset/step protocol. Do not use its raw per-step value as a hidden task-progress signal.

### 4.4 Conditioning video is required when RoboMME supplies it

Some tasks supply a conditioning video during reset. This is part of the standard benchmark, not an optional new ICIL experiment.

At the audited continuous-action interface, reset returns conditioning observations followed by one initial execution observation. Preserve the supplied history; do not simply take the final frame and discard the video.

- Store conditioning observations in a separate bank.
- Store execution history in its own bank and reset it on every episode reset.
- Finalize partial conditioning blocks because that video is already complete and available.
- Do not let a local block cross the conditioning/execution boundary.
- Do not use conditioning-video frames as action-training targets.
- Crucially, the online reset supplies video/proprioception but not the demonstrator's commanded actions. **Mask conditioning action fields as unavailable during training too**, even if HDF5 stores them.
- Similarly, the automatic initial reset command is not an observed user-policy action unless the public API explicitly supplies it; use an unavailable-action flag rather than reading an internal reset command.

Process every returned frame once in chronological order. A normal continuous-action step returns one frame; reset and other protocols may return lists. Build an adapter with explicit event identity so repeated service calls do not append the same observation twice.

### 4.5 RGB-D and geometry

The released schema includes:

- RGB: uint8, front/wrist `256 x 256 x 3`.
- Depth: signed int16 millimetres; convert to FP32 metres and reject nonpositive/invalid depths before geometry.
- Intrinsics: `3 x 3` in episode setup.
- Extrinsics: `3 x 4` for each frame and camera.

Use the documented CV camera convention. For an extrinsic mapping world to camera, invert it for backprojection: `p_world = R^T (p_camera - t)`. Verify with both cameras and the actual benchmark convention instead of assuming the matrix direction. Wrist extrinsics change over time.

Build camera-visible point clouds only. Use RGB in point features. Start with deterministic voxel/reservoir sampling or modest FPS and a fixed, documented workspace crop. Do not use simulator masks to find objects or balance point counts in the principal experiment. A crop must be based on calibration/workspace geometry, not hidden target poses.

Render one short diagnostic showing front/wrist clouds aligned in the same world coordinates and the robot/table in sensible locations. This is a necessary calibration check. Save the result, then move on.

Segmentation is supported internally by the simulator, but the documented released HDF5 writer does not save it. Do not make masks a prerequisite or regenerate the entire dataset to obtain them.

### 4.6 Time and sampling

The audited simulator fork defaults to 100 Hz physics and 20 Hz control for checked tasks. Read/log actual per-task frequencies at runtime. Video-export FPS is not control frequency.

HDF5 may omit “NO RECORD” intervals. Unless continuity is verified, distinguish a **recorded-frame clock** from actual elapsed physical time. Keep source-local indices and explicit nominal cadence metadata. Report steps and physical seconds only where the conversion is justified.

Do not normalize execution time by the eventual episode length. Use fixed time scales, source-local timestamps, relative age at read time, and explicit block intervals.

Main runs process every acquired observation at the declared cadence. If downsampling is needed for a specific profile, make it a visible config and apply it to every method; do not silently skip cue frames.

## 5. Action representation and initial policy

Use RoboMME's `action_space="joint_angle"` initially:

- Seven absolute joint position targets.
- Gripper command `-1 = close`, `+1 = open`.
- PatternLock and RouteStick use a stick: handle/mask the unused gripper dimension consistently.
- Preserve the official continuous control semantics and primitive-step count.

This avoids adding a new IK failure mode to a memory paper and matches the official continuous joint-action protocol. Keep observed end-effector pose as useful geometric conditioning.

Do not silently replace this with planner-assisted `waypoint` execution. `ee_pose` may be supported later, but is not needed for this implementation milestone.

Use a compact conditional **flow-matching action-chunk policy** as the default, with a plain regression head for debugging and a later optional decoder check. No new action codec.

Suggested initial defaults, configurable rather than promises of optimality:

| Parameter | Core default | Local debug profile |
|---|---:|---:|
| History block length F | 8 observations | 4 |
| Sampled points per observation, both views combined | 1,024 | 128–256 |
| Visual supernodes per complete block | 32 | 8–16 |
| Current-scene visual tokens C | 64 | 16 |
| Feature width d | 256 | 128 |
| Policy layers | 4 | 2 |
| Action horizon H | 16 primitive commands | 8 |
| Execute before replanning | 4 commands | 1–2 |
| Flow integration steps | 8 | 4 |
| Main historical token budget B | 128 | 16–32 |
| Analysis budgets | 64, 128, 256 | one budget |
| Router key/query width | 128 | 64 |

The debug profile is not a scientific result. Never mix its checkpoints/caches with core runs.

The debug B=16–32 range is valid only if it covers all source/modality/pending roots. Check that bound even in debug mode. For a long diagnostic episode, raise debug B or use an explicitly shortened diagnostic prefix; never silently drop roots.

A simple conditional flow-matching convention is:

- Normalize expert chunk `a` using training-only statistics.
- Draw noise `epsilon ~ N(0,I)` and `u ~ Uniform(0,1)`.
- Form `x_u = (1-u) epsilon + u a`.
- Train `v_theta(x_u, u, context)` against `a - epsilon` with a masked, dimension-normalized squared loss.
- At inference integrate from noise at `u=0` to an action at `u=1`.
- Convert to physical commands, apply documented joint-limit handling and gripper discretization, execute only the first configured number of commands, and replan.

Pick one solver/convention and test it. Use the same decoder, normalization, integration steps, command execution horizon, and noise-seed protocol for matched comparisons. Mask terminal padding instead of training repeated synthetic terminal commands.

Action tokens may self-attend and cross-attend to current/history tokens. **Do not apply dense self-attention over the entire accumulated archive in the full-history reference.** A full-leaf cross-attention teacher is a strong, more practical reference. With a fixed number of action queries its historical read cost is linear in N; do not manufacture a quadratic baseline to inflate speedup.

Use a small task-text encoder trained with the teacher, e.g. a byte/subword encoder plus a small Transformer. Condition on the provided goal text rather than a hidden task/episode identifier. Keep this simple and consistent across methods; no large VLA is required. If a frozen pretrained text encoder is adopted for convenience, record its source, revision, and additional pretraining rather than hiding it.

## 6. Fine space-time evidence encoder

### 6.1 Definition of a leaf supernode

A visual leaf is a token anchored at a spatial location and time, initially aggregating observed point features from a local space-time neighborhood:

`z_m = sum_i alpha_mi h_mi`,

where the weights are normalized across valid neighbors. A practical kernel is

`logit_mi = -||x_i-c_m||^2 / tau_x - (t_i-tau_m)^2 / tau_t`.

Here `tau_x` and `tau_t` are positive denominators with explicit units, not unqualified “radii.” They may be globally learned during teacher training and bounded away from zero.

Use a nonlinear point/relative-edge MLP before pooling. Include RGB, geometry, and local temporal offsets. Temporal stratification of center selection should give each observed frame candidate centers, without claiming every tiny cue is preserved. Make the encoder invariant to permutation of points within a frame.

Start with bounded local neighbor sets and a modest per-center cap. A chunked dense center-to-point reference implementation is acceptable if simpler and fast at these local sizes; count its actual cost. Do not materialize a matrix over the entire episode's points. Keep the locality definition/config explicit.

Masked/all-invalid neighborhoods must return zero content and `valid=False`. Do not let finite negative masks produce uniform attention over padding. Do not fabricate occupied leaves to fill a shape.

Use one internal mask convention and explicitly convert it at API boundaries. PyTorch SDPA boolean masks use `True` for allowed attention, whereas `MultiheadAttention.key_padding_mask` uses `True` for excluded positions. Handle entirely empty history separately so attention never normalizes an all-invalid row.

Retain the resulting anchored leaves. **Do not run them through a global Perceiver register bottleneck before archival.** Small local contextualization within a fully observed block is allowed, but keep center/time support metadata and do not assume contextualized features have strictly point-local semantics.

### 6.2 Separate robot-event leaves

Do not repeat the same robot state across every point.

For each historical observation, create a typed robot-event leaf from observed proprioception and the preceding executed command, with an explicit command-available mask. State and action are distinct input fields; they can share this low-dimensional event token to keep the first implementation small.

Maintain separate visual and robot-event trees within each source. Do not average unrelated modalities into an untyped vector. Conditioning-bank event tokens contain observed state but no unavailable demonstrator commands.

Each valid visual or robot-event leaf has unit mass. Gaussian occupancy is auxiliary metadata, **not** the aggregation mass.

### 6.3 Current scene and cacheable content

Represent the current observation separately at high resolution. At decision t, the execution archive contains observations strictly before the current observation; current input is supplied once through the current-scene path.

Use shared point-feature machinery for current and historical observations where practical. Freeze the historical encoder, current visual encoder, event encoder, and task encoder after teacher training for the primary controlled comparison.

Separate cacheable latent content from read-time geometry/time/type embeddings:

- Cache content, anchor, spatial bounds, observed-time interval, source and modality.
- Compute relative age at the current decision.
- Apply a shared trainable read adapter to content plus metadata inside the policy.

Do not cache decision-dependent relative-age embeddings as immutable historical features.

## 7. Archive, hierarchy, and bounded frontier

### 7.1 Tree construction

For each source (conditioning or execution) and modality (visual or robot-event):

1. Build a balanced binary local tree for each completed block.
2. Visual leaves are grouped spatially, using a deterministic balanced split such as the longest spatial extent. Robot-event leaves are grouped by time.
3. Combine adjacent completed block roots into a binary-counter temporal forest: merge equal-scale neighboring subtrees, retaining both children.
4. Store every valid leaf and parent. Parent summaries never delete descendants.

This is a temporal hierarchy over blocks with spatial refinement inside selected visual blocks. Do not claim arbitrary independent spatial/time subdivision or object tracking.

Parent records contain:

- Child IDs and stable source/modality identity.
- FP32 sum of descendant content vectors and integer descendant-leaf count.
- Mean content = sum / count.
- Spatial bounds and count-weighted mean anchor.
- Minimum/maximum observed time; scale/depth metadata.
- A bounded auxiliary routing summary, initially coordinatewise max of descendant content, if useful.
- Validity and availability/completion metadata.

Content means are built **before** nonlinear policy read adapters. Recompute bounds/type/scale metadata structurally; do not just average nonlinear positional embeddings.

All of this is deterministic aggregation. There is no separately trained “coarse encoder.”

### 7.2 Incomplete execution blocks

Default replanning every four commands with eight-observation blocks creates incomplete historical blocks. Handle them explicitly:

- Encode only the observed prefix of the pending block.
- Construct temporary local prefix trees and include their roots as candidates.
- Count selected prefix nodes within the same B-token history budget.
- When the block completes, replace that temporary prefix representation with the completed-block representation.
- Never use a cached full-block feature at a decision before all its source observations were available.

The current observation remains outside this history block, in the C-token current-scene input.

Frozen-encoder caches may store completed blocks plus the prefix variants actually queried by training. Do not blindly cache every possible prefix of every trajectory if that is unnecessarily large. Match offline and online block construction numerically.

### 7.3 Frontier

The initial frontier is the collection of all nonempty forest roots, including temporary prefix roots. Each split replaces one parent by its two valid children.

A valid frontier:

- Covers every available leaf exactly once.
- Contains no node together with one of its descendants.
- Contains at most B valid historical tokens.
- Contains no node whose evidence was unavailable at the decision.

Split until B is filled or no further split is possible. **Do not stop because predicted gains are negative or near zero.** A low immediate gain does not imply useful descendants are absent.

Root count must fit B. Assert and report this. Include every source/modality/prefix root in the count; do not silently discard an old bank to fit a budget. Declare a maximum supported episode length and verify all configured budgets against it. Empty histories and single-leaf trees must work.

### 7.4 Read-time token metadata and mass

Use one shared metadata schema for tree nodes, individual leaves, and flat residual summaries. Include content mean, bounds, time interval, mean anchor, source/modality, log count, and resolution metadata. No “tree method” versus “flat method” embedding.

A residual summary may cover a disconnected set; represent its count/mean/bounding interval honestly. Do not make it masquerade as a contiguous trajectory segment.

The initial default may use `log(count)` as an additive historical attention-logit bias, consistently for all grouped readers; full leaves have count 1. Make this configurable for a later small ablation. Current-scene/proprio/instruction tokens have their separately defined measures.

Do not present this as an exact whole-policy approximation theorem. Exact grouped-attention results require particular post-projection key/value means; learned adapters and contextual layers do not automatically satisfy them. The scientific correctness tests concern tree coverage and aggregation, not a false claim that pooled-token predictions equal full predictions.

### 7.5 Storage and complexity

Maintain a simple CPU archive with optional GPU-resident mode. Read selected records in batches. Count all archive features, metadata, routing keys, prefix caches, and transferred bytes.

For fixed-query local node scores, a binary frontier with R roots and S splits has R+S nodes. Reaching B permits at most B-R splits and O(B) visited-node score evaluations. A heap adds O(B log B) CPU bookkeeping. Archive storage is O(Nd).

This is only a node-visit/accounting claim. Python traversal, cache misses, small kernel launches, archive updates, and transfers can dominate wall time. Measure them. Do not claim a speedup from asymptotics alone.

## 8. Retriever design

### 8.1 Fixed decision query

Build the routing query from current point-cloud features, current proprioception, available instruction, the current decision/source-local clock, and the target budget. Explicit current-time information is needed for relative-time reasoning when cached candidate keys contain historical timestamps. It must not depend on expert target actions, noisy diffusion/flow action inputs, future frames, or oracle annotations. Never supply the eventual episode duration.

Compute the query once per action chunk. Reuse the selected frontier throughout the flow integration and executed chunk.

Use a lightweight local scorer for both hierarchical and flat retrieval. A two-tower implementation is a good default:

- Query projection from current conditioning.
- Candidate-key projection from content mean/max and source/modality/geometry/time/scale metadata.
- Dot-product compatibility plus a small local metadata bias.

Choose a form that permits immutable candidate-key caching once router weights are frozen. If relative age is used in a key, handle its time dependence explicitly; it cannot be cached forever as a static key.

Avoid one GPU synchronization for every node split. A CPU heap using cached keys and one transferred query is acceptable; so is a carefully batched GPU implementation. Benchmark the actual implementation. Any batched multi-split approximation must be explicit and reproducible.

The true split gain depends on the entire frontier F. A fixed-query local scorer does not observe F, so it predicts **approximate expected gain across its training contexts**, not exact conditional marginal benefit. This approximation is deliberate for a simple, matched first implementation.

### 8.2 Hierarchical retrieval

Start from all roots. Score eligible nodes when encountered. Expand the best-scoring splittable node, insert its children, and continue to the budget. Cache scores for this decision; do not silently rescore all prior nodes while claiming O(B) evaluations.

### 8.3 Strong flat retrieval with residual summaries

Use exactly the same available fine leaves, including valid prefix leaves.

Score all candidate fine leaves once using an equally sized local scorer. Select the highest scoring leaves, then represent every unselected leaf through a residual summary for its source/modality group.

- Reserve up to one residual token per nonempty source/modality group.
- All selected leaves plus residuals must fit B.
- Residual mean/sum/count/bounds/time/type must exclude the selected leaves.
- If all leaves fit, use them directly with no empty residual.
- Refill any capacity freed when selecting the last remaining leaf of a group removes that group's residual. A fixed `B - number_of_groups` selected-leaf count can underuse the budget. Treat singleton residual replacement as a zero-cost representation change, then continue if another positive-cost move fits. Use the same accounting in training contexts, gain labels, and inference.
- Track the actual token count and never hide uncounted residuals.
- Matching the hierarchy's source/type structure matters more than claiming “one” residual when there are actually four.
- Implement a simple exact flat scan first. At the initial archive lengths, an ANN subsystem is not necessary. Do not make the baseline artificially slow with repeated Python per-leaf network calls.
- Include the flat key cache and full scan in measured storage/latency.

Tree splitting and flat selection are different operations but receive comparable action-derived utility supervision.

## 9. Exact staged training algorithm

This is staged training, not end-to-end differentiation through hard branch choices.

### 9.1 Stage T: full-history teacher and encoders

Train the visual/event/current/task encoders and the full-history action policy from expert actions.

At each decision, the teacher sees the same valid sensors and causal sources as the eventual student, but all available fine leaves. It sees no privileged state or future execution history.

Use action-query cross-attention to historical leaves, local block encoding, gradient checkpointing across blocks if needed, length buckets, and gradient accumulation. A smaller microbatch is preferable to silently shortening the teacher's history.

Suggested initial cap: 50k optimizer updates, effective batch 32, AdamW with a sensible warmup/decay schedule and learning rate around 1e-4. These are starting configurations, not proven hyperparameters. Adjust microbatch/accumulation to the measured hardware. Keep effective batch and exposure documented.

For a meaningful teacher, avoid the bootstrap of first freezing a randomly initialized encoder and expecting retrieval to fix it. If full-prefix backpropagation is genuinely infeasible, first try checkpointing/bucketing/microbatch reduction. Any partial-gradient or short-history warmup fallback must be documented and followed by a full-history decoder stage; do not label a truncated window as full history.

Save a resumable training checkpoint and an immutable selected teacher checkpoint. Select checkpoints using validation criteria, never test results.

### 9.2 Stage C: freeze encoders and build caches

Freeze the selected teacher encoders. Cache local leaves and the causal prefix variants needed by sampled decisions. Use the same cache for tree, flat, uniform, full-history evaluation, and compression baselines.

Cache identity must include:

- Encoder checkpoint hash.
- Input/sampling/calibration/action-alignment configuration.
- Training normalization statistics version.
- Dataset revision and episode/source hashes.
- Block length and prefix convention.
- Cache-format version and code revision.

Changing encoder weights or preprocessing invalidates the cache. Do not quietly use stale features. Router keys have an additional router-checkpoint hash.

Use chunked immutable shards/memmaps or another simple efficient format. Avoid millions of tiny files, object pickles, or loading the full dataset into every worker. Open HDF5 independently per worker and close handles reliably.

### 9.3 Stage P: one shared compact policy

Initialize its decoder/read-adapter weights from the teacher where shape-compatible. Keep all encoders frozen.

Train a **single compact policy** on a balanced mixture of:

- Valid random tree frontiers.
- Random selected leaves plus source/type residual summaries.
- A small proportion of uniform-resolution contexts.
- A small proportion of current-only contexts as a diagnostic control.

A useful initial mixture is 45% tree, 45% flat-plus-residual, 5% uniform, 5% current-only. Keep tree and flat equally represented at each budget and across early/late decisions. Sample partially refined contexts as well as near-budget contexts, so intermediate router states are not out of distribution.

Use B in {64,128,256} for training; include valid intermediate sizes between the root count and the selected cap. All readers get identical current observations, instruction, metadata schema, normalization, and action decoder. Do not provide a method ID.

Train on the expert flow target. Teacher distillation in this stage is not required initially; the teacher is used for router labels below. Suggested cap: 30k optimizer updates, initialized from teacher, same task sampling and validation discipline.

Select one checkpoint using an aggregate validation objective covering both tree and flat context families. Freeze it for the **primary reader comparison**.

This design saves training and isolates reader quality under a shared policy. It does not claim to measure each reader's independently optimized ceiling.

### 9.4 Stage L: generate action-utility labels

Teacher and compact policy remain in deterministic evaluation mode with all parameters frozen.

For sampled decision t:

1. Construct a valid random or reader-generated context F below the target budget.
2. Sample one expert chunk, one noise tensor epsilon, and one flow time u.
3. Evaluate the full-history teacher and compact policy using exactly the same x_u and u.
4. Define D_t(F) as the masked, dimension-normalized mean squared difference between their velocity predictions.
5. For a sampled candidate move s, construct F' and compute `g_t(s,F) = [D_t(F)-D_t(F')] / [|F'|-|F|]`.
6. Store signed gain labels and the inputs needed by the local scorer.

For the tree, a move replaces a parent by its children. For flat retrieval, it moves a currently unselected leaf out of its residual summary into an individual token, updating that residual. Skip zero-cost/invalid candidate moves or treat them with an explicitly separate rule; do not divide by zero.

Use the same expert-chunk/noise/time sample for the before/after contexts and teacher. One paired draw is sufficient initially; fresh draws on later visits provide variability. Do not compare independently sampled final action chunks.

The expert future chunk is used only to construct training losses/noised inputs. It is **not an input to the retriever**.

Limit label generation:

- Initial default: about 16k sampled decision-context pairs per reader, task-balanced.
- Balance target B over {64,128,256} within that total, including the label refresh. Do not train labels only at B=128 and then call the other budget curves matched in-distribution evaluations.
- Four candidate moves per context, evaluated in a batch.
- Reuse one teacher result per identical decision/noise sample.
- Include early, middle, and late decisions and exploratory deeply refined contexts.
- Sample without test data or oracle event boundaries.
- Log the gain distribution: positive, negative, near-zero, and variation across noise/context.
- A small gain-based replay buffer may reuse already observed informative examples; it must not introduce test labels or simulator-only supervision.

These defaults produce around 64k move labels per reader, not exhaustive labels for every tree node at every frame. Make budgets explicit in the job manifest. Do not grow this dataset automatically until compute is exhausted.

### 9.5 Stage R: train both local scorers

Train hierarchical and flat scorers with equal parameter capacity, comparable label counts, the same available conditioning, and the same optimization budget.

Use robust regression on appropriately scaled signed gains, with an optional within-context ranking term. Fit target scaling from the training labels only. Report regression/ranking diagnostics without pretending offline gain prediction is task success.

Suggested initial cap: 10k lightweight router updates, batch 256. Validate label prediction on the episode-disjoint router-label split from Section 4.1.1. This does not imply those episodes were unseen by the teacher or compact policy. Use the fixed official validation rollout subset for end-to-end selection; never require nonexistent official validation expert chunks or use official test contexts for model selection.

Then perform **one small label refresh** using contexts visited by each learned reader, mixed with exploratory contexts. A default extra 8k decision-context pairs per reader is sufficient for the initial plan; resume scorer fitting with the mixture. The compact policy remains frozen, so labels change due to context coverage rather than a moving policy.

Do not repeatedly regenerate huge label sets by default.

### 9.6 Optional alternating adaptation, only if justified

Do not make joint/alternating training a prerequisite.

If the learned readers select valid contexts but the shared policy is clearly limited by their distribution, allow one matched policy fine-tuning stage for **both** readers, with equal updates, followed by regenerated gain labels and scorer refresh.

That changes the experimental question: report it separately from the shared-policy comparison. Do not fine-tune only the proposed method and compare against an unadapted flat baseline.

### 9.7 Known routing failure and one targeted fallback

An ancestor split can have negligible immediate benefit while useful evidence exists deeper below it. Gains are not guaranteed to be submodular or greedy-friendly.

Initial safeguards: fill the budget, include deep exploration in label generation, and avoid drowning all training examples in routine memory-insensitive motion.

Only if full leaves and a small oracle/lookahead diagnostic work while learned routing fails, consider depth-two continuation targets or descendant-propagated utility. Label this as a method change, refresh supervision for both readers where applicable, and document the extra search cost.

Do not call a greedy search an exact oracle. Exact enumeration is feasible only on deliberately tiny diagnostic archives.

## 10. Core baselines and run plan

### 10.1 Implement these first

| ID | Method | Additional expensive training |
|---|---|---|
| FULL | Full-leaf teacher/reference | Stage T |
| TREE | Learned adaptive frontier, B=128 | Shared Stage P + small tree scorer |
| FLAT | Learned flat selection + residuals, B=128 | Same Stage P + small flat scorer |
| UNIFORM | Fixed deterministic coverage frontier, B=128 | Reuses Stage P |
| PERCEIVER | Learned fixed-size B-token resampler over all leaves | One adaptation run with frozen encoders and teacher decoder initialization |
| CURRENT | Current-only diagnostic | Reuses the history-dropout-trained shared policy |

For UNIFORM, split the largest/coarsest represented regions deterministically to approximate uniform temporal/spatial resolution. It must satisfy the same frontier invariants.

PERCEIVER is a learned global compression baseline: train its resampler and policy jointly on expert targets while keeping the leaf encoder frozen. Include its O(BN) historical read and total archive in costs. It is not a streaming recurrent memory. Use comparable decoder size, optimization exposure, and current inputs.

CURRENT is explicitly a diagnostic control, not the maximum performance of a separately optimized current-only policy. On video-conditioned tasks it also lacks the supplied reference, so its failure alone does not establish online execution-memory dependence.

Do not require reimplementing every VLA paper. Published external results may be listed in a separate contextual table with their different sensors, pretraining, and action protocols, never mixed into a supposedly matched comparison.

### 10.2 Minimal progression

**A. Local correctness/debugging.** Run only the narrow checks in Section 12. No multiple seeds.

**B. One early end-to-end check, reusing useful work.** Start the actual seed-0 all-task teacher job and examine a short initial segment/checkpoint plus throughput. Do not launch a separate multiseed gate study. If an isolated 1–2 task learning debug is genuinely needed, cap it explicitly and stop once the concrete bug or uncertainty is resolved.

**C. Core seed-0 experiment.** Train the all-task teacher, caches, shared compact policy, two routers, and Perceiver baseline. Produce the six-method validation comparison on all 16 tasks.

**D. Freeze the primary configuration, then evaluate official test episodes.** Keep test results out of tuning.

**E. Final replication of the central comparison.** Prepare seeds 1 and 2 for the shared pipeline and TREE versus FLAT; FULL and UNIFORM can reuse those trained checkpoints. These are final result replications, not preliminary gates. Do not automatically replicate all optional ablations.

The default job plan is `core_seed0`. A separate `central_3seed` plan expands only the important comparison. The user explicitly chooses submission of later phases.

### 10.3 Small analyses with high information value

Implement these analyses without automatically launching a large grid:

1. **Budget curve:** TREE, FLAT, UNIFORM at B={64,128,256}, using the shared variable-budget policy and budget-conditioned scorers. Report success against total latency and storage as well as token count.
2. **Memory dependence:** on a fixed small validation subset, erase or permute execution history while keeping current input and conditioning video fixed; separately erase conditioning history where relevant. These are sensitivity diagnostics, not proof of non-Markovianity.
3. **Timing and storage:** short/medium/long naturally occurring histories, with complete encoding/update/read/decode timings and archive bytes.
4. **Frozen-checkpoint routing diagnosis:** compare learned reads with full leaves and limited exhaustive/search-based refinements on a few small archived prefixes. This does not require another trained model.
5. **Tokenizer attribution, secondary:** prepare a framewise spatial-token encoder variant at approximately equal total leaves and one corresponding teacher/shared-reader seed. Do not run it before the central TREE/FLAT result. This is the main additional experiment needed before making a strong joint-space-time-tokenizer claim.

Implement configurable framewise and joint encoders behind the same interface, but do not multiply experiments by every possible encoder, loss, budget, or seed.

Controlled new tasks, late-revealed queries, real robots, other benchmarks, large ANN indexes, and a second full decoder study belong to a later expansion. Do not claim standard RoboMME alone proves all the broader hypotheses.

### 10.4 What to do when results are weak

A poor result does not authorize an endless search.

- If replay/calibration/alignment fails, repair that specific problem before training.
- If the full-history teacher does not learn, investigate action/control/perception/training with a small targeted diagnostic. Do not train a router to imitate an unusable teacher and present the failure as a retrieval conclusion.
- If the compact policy is poor with both random contexts and all-leaf contexts on small archives within its trained budget range, inspect metadata/normalization and teacher transfer. Feeding it thousands of fine leaves after compact adaptation is out of distribution; poor performance there alone does not establish a bug.
- If useful detail exists but the router misses it, inspect labels/deep-ancestor behavior.
- If FLAT matches or beats TREE, report it honestly. Do not weaken FLAT, change test seeds, remove difficult tasks, or silently add extra information to TREE.

Produce clear evidence and a recommended next step; success means a trustworthy experiment, not manufacturing a favorable result.

## 11. Evaluation contract and reporting

### 11.1 Simulator adapter

Use the official builder and registered environments. At the audited revision the shape is:

~~~python
from robomme.robomme_env import *  # benchmark task registration
from robomme.env_record_wrapper import BenchmarkEnvBuilder

builder = BenchmarkEnvBuilder(
    env_id=task_name,
    dataset="val",  # "test" only for the final frozen protocol
    action_space="joint_angle",
    gui_render=False,
    max_steps=1300,
)
env = builder.make_env_for_episode(
    episode_idx,
    include_front_depth=True,
    include_wrist_depth=True,
    include_front_camera_extrinsic=True,
    include_wrist_camera_extrinsic=True,
    include_front_camera_intrinsic=True,
    include_wrist_camera_intrinsic=True,
    include_maniskill_obs=False,
    include_available_multi_choices=False,
)
~~~

Verify the exact callable against the pinned source; preserve the intent if a compatible signature differs. Optional observation flags default to false, so point-cloud evaluation must explicitly request depth and calibration.

The wrapper may adjust its internal step limit. Log the actual effective limit and use the same one for every method. Preserve the benchmark's primitive control cadence.

For each chunk:

- Retrieve once.
- Predict H actions.
- Execute the configured first actions sequentially.
- Inspect termination/truncation and terminal status before indexing the returned observation: a failure wrapper may return an empty observation dictionary. Record the terminal outcome safely.
- Append each valid resulting observation and issued command exactly once.
- Stop immediately on termination/truncation or the benchmark's terminal status.
- Replan using the updated causal memory.

Reset all execution/cache state between episodes. Reinitialize the conditioning bank from the new reset response. Do not carry learned recurrent episode state across trials.

Count success using the benchmark's `info['status'] == 'success'` convention, verified against the pinned wrapper. `terminated=True` or a nonzero reward alone is not success. Invalid policy actions, NaNs, and action-shape errors count as failed planned episodes and retain their error details; do not exclude them from the denominator.

Keep any policy-service protocol typed and minimal. Send only allowed observations, not the full info dict.

### 11.2 Metrics

Primary:

- Closed-loop success per task, suite, and macro-average over tasks.
- Official test seed count and completed/failed/error/timeout counts.
- Same task/episode seeds across methods.

Secondary:

- Action prediction/flow losses, explicitly labeled as training diagnostics or genuinely held-out loss according to Section 4.1.1.
- Per-decision active token count and resolution distribution.
- Archive bytes, routing key/index bytes, peak allocated/reserved accelerator memory.
- Nodes/leaves scored, records fetched, transferred bytes.
- p50/p95 complete online decision latency and component times.
- Offline preprocessing/cache/label-generation time and actual allocated GPU-hours.
- Per-stage training examples, optimizer steps, effective batch size, and throughput.

Include both warm steady-state and cold/startup costs where relevant. Frozen training caches must not make online encoding/update costs disappear from inference timing. Synchronize GPU measurements correctly. A large archive kept entirely on GPU is not constant accelerator memory.

Use official val/test episodes through the builder. For expensive interim validation, use one fixed, declared small subset identically across methods. Final core results use all 50 test seeds/task. Keep run manifests and denominators explicit.

Use episode-level paired comparisons and uncertainty intervals. With multiple training seeds, report both training-seed variation and paired evaluation variability. Do not treat correlated frames as independent experimental trials. A single training seed can establish an engineering screen; do not label its variation across rollouts as training robustness.

Infrastructure errors are recorded. Retry only a diagnosed infrastructure issue with the same original episode seed. Do not silently replace failed cases with easier ones or exclude policy-induced errors.

### 11.3 Output artifacts

Every run writes:

- Resolved configuration and source/environment/dataset hashes.
- Training metrics JSONL/CSV, checkpoints, normalization statistics.
- Per-episode evaluation JSON with actions/status/runtime/error information.
- Small representative videos, with a configurable recording cap.
- Summary tables and plots generated by code, not hand-edited.
- A run-completion marker only after the intended stage actually finishes.

Reports distinguish measured results from estimates and unrun configurations. Include a “not yet run” status rather than zero or invented scores.

## 12. Minimal required verification

These are implementation tests and short checks, not a preliminary research campaign:

1. **Real data alignment check** from Section 4.2.
2. **One RGB-D calibration check** and one expert-action replay using the chosen action protocol.
3. **Causal-prefix test:** change future execution observations/actions and ensure an earlier decision's available leaves/frontier/prediction remain unchanged under fixed noise. Test partial blocks and conditioning/execution boundaries.
4. **Hierarchy tests:** exact leaf coverage, no overlap, correct sums/counts/bounds, budget compliance, empty history, all-invalid points, multiple roots, prefix replacement, and source resets.
5. **Flat residual tests:** selected plus residual leaves cover the same archive exactly once; count/means agree with direct calculation.
6. **Action/flow tests:** target alignment and masking, normalization round trip, shared noise/time in gains, no target/noisy action passed into router, finite gradients and a tiny-batch loss reduction.
7. **Offline/online parity:** the same observed prefix produces numerically consistent encoded evidence and selected context in cached and streaming implementations.
8. **One actual closed-loop rollout** through reset, conditioning video if present, empty execution history, action chunking, and termination.
9. **Checkpoint/resume and batch-script syntax checks**, including a short local stop-request/resume test.
10. **Hardware doctor** on the A5000, then a short separately prepared HPC rendering/throughput check.

Do not add a full unit test for every trivial helper. Prioritize checks that prevent invalid results. No multiseed pilot suite, no broad grid, no extra benchmark required before the core experiment.

If the A5000 cannot run the full-history core configuration, shrink only the debug profile. Do not alter the scientific configuration silently to make the workstation happy.

## 13. HPC implementation

### 13.1 Directory and configuration conventions

Provide at least:

- `hpc/site.env.example` for repo/data/cache/run paths, environment activation, optional account/QoS/constraints, GPU profile, CPU/memory/time requests.
- `hpc/jobs/teacher.sbatch`
- `hpc/jobs/cache.sbatch`
- `hpc/jobs/compact.sbatch`
- `hpc/jobs/labels.sbatch`
- `hpc/jobs/router.sbatch`
- `hpc/jobs/perceiver.sbatch`
- `hpc/jobs/evaluate.sbatch`
- `hpc/jobs/doctor.sbatch`
- A job-plan renderer/submitter supporting dry-run, explicit submit, dependency tracking, and resume.
- `hpc/logs/.gitkeep` and documentation that the logs directory must exist before sbatch.

Paths must be configurable, quoted, and portable between workstation and cluster. Do not repurpose HOME or other standard environment variables. Use project-specific names such as `STMEM_DATA_ROOT` and `STMEM_RUN_ROOT`.

User-supplied resource template:

~~~bash
#!/usr/bin/env bash
#SBATCH --job-name=stmem_teacher
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
~~~

These lines are defaults, not proof that a particular GPU model is selected. Keep account, QoS, GPU type, and constraints configurable. Do not invent the site's H200/B200 resource names.

Render resource values literally into job files or pass them as `sbatch` command-line arguments. Shell variables loaded from `site.env` are not expanded inside `#SBATCH` comments. Make the single-node assumption explicit with `--nodes=1` in rendered jobs or submission arguments.

Resolve repo working directory and log paths consistently before submission. `mkdir -p hpc/logs` inside the job is too late for Slurm's initial stdout/stderr opening; the submission wrapper must create it first.

### 13.2 GPU profiles and launch

Profiles:

- `a5000_debug`: local, small shape profile.
- `h200_1gpu`: default experiment profile.
- `h200_4gpu`: optional single-node DDP.
- `b200_1gpu` / `b200_8gpu`: opt-in, never selected automatically.

Single node: one Slurm task launches one torchrun agent, which launches one worker per allocated GPU. For four GPUs, keep `--ntasks=1` and use `--nproc-per-node=4`. Do not create four torchrun agents each spawning four workers.

A representative launch pattern is:

~~~bash
srun --ntasks=1 torchrun --standalone --nnodes=1 --nproc-per-node=1 \
  -m stmem.train --config configs/teacher.yaml
~~~

Set CUDA device from LOCAL_RANK, use NCCL where appropriate, and respect Slurm's assigned CUDA_VISIBLE_DEVICES.

DDP does **not** pool GPU memory for a single replica. Use it for measured throughput improvements. Handle teacher memory by microbatching/checkpointing first; do not claim four replicated GPUs solve a per-replica OOM. Explicit sharding would be a separate implementation choice.

Preserve effective batch size across hardware profiles or document a deliberate change. Keep per-task sampling balanced across ranks.

Use rank-sharded training samples, deterministic epoch reseeding, matching optimizer-step counts, and coordinated stop collectives. Evaluation sharding must assign every planned episode exactly once; do not use distributed-sampler padding that silently duplicates episodes.

### 13.3 Submission plan and costs

The planner must print, before submitting:

- Exact commands/configs, training seeds, and task set.
- Requested GPU type/profile/count and walltime.
- Dependencies and expected input/output checkpoints/caches.
- Number of jobs and maximum concurrent allocations.
- Estimated cost based on measured timings, or clearly “unknown.”
- Remaining stage work when resuming.

Default to dry-run. Only `--submit` performs submission. Use `sbatch --parsable` and record returned job IDs. Use `afterok:` dependencies where a completed stage is required.

An illustrative DAG:

- teacher -> immutable encoder/teacher selection -> cache
- cache + teacher -> shared compact policy
- compact + cache + teacher -> tree/flat label jobs -> router jobs
- cache + teacher -> Perceiver baseline
- successful checkpoints -> evaluation shards -> aggregation

Support selectively submitting/resuming one stage without rerunning all ancestors. Avoid duplicating teacher/cache work across methods. Budget/seed sweeps are explicit manifests, not nested shell loops that accidentally launch hundreds of jobs.

Keep H200 and B200 allocated hours separate. Do not invent credit conversion rates. Record elapsed walltime times allocated GPU count for each job, including evaluation and data preparation when they reserve GPUs.

### 13.4 Resumability and signals

Checkpoint atomically on the same filesystem:

- Model/read adapter/router parameters and EMA if used.
- Optimizer, scheduler, AMP scaler if used.
- Step/epoch, RNG states, sampler state or documented resume boundary.
- Config and data/cache/checkpoint hashes.
- Per-rank RNG information where applicable.

Use one writer and retain the previous known-good checkpoint.

Provide periodic checkpoints plus a safe stop-request mechanism. An advance Slurm signal can set a marker file; rank zero polls it at optimizer boundaries, broadcasts a stop, and checkpoints coherently. Do not blindly forward USR1 through torchrun: launcher behavior may terminate workers before they save.

Use a fresh stop-request marker per allocation/job ID, or explicitly clear an old marker after validating the resume checkpoint. A stale marker must not immediately stop every resumed run.

If using `#SBATCH --signal=B:USR1@180`, test shell trapping/wait semantics with a local subprocess. A trapped signal must not cause the wrapper to report false success or discard the child's real exit status.

A checkpointed partial stage must **not** exit successfully as if complete and release downstream afterok jobs. Write completion markers only at actual completion. Provide an explicit resumed job plan; automatic unbounded self-resubmission is forbidden.

### 13.5 Evaluation and I/O

Use independent worker processes or bounded Slurm arrays for task/episode shards. Do not assume the stock benchmark supports large vectorized RL batches.

Every shard writes atomic per-episode results and resumes idempotently. Limit array concurrency through config. Keep model-load/renderer costs in mind; avoid one allocation per tiny episode if that wastes startup time.

If separate policy/simulator processes are used, launch both inside the allocation with health checks, unique ports, cleanup traps, and failure propagation. No orphan servers or interactive terminals.

Use available node-local scratch for immutable cache shards only if configured. Write durable checkpoints/results to the shared configured run directory. Do not assume a scratch mount, unrestricted storage, or Internet access.

## 14. Suggested repository layout and interfaces

Adapt names if there is a clear reason, but keep the components separable:

~~~text
PROMPT.md
README.md
AGENTS.md
pyproject.toml
uv.lock
configs/
  debug/
  teacher.yaml
  compact.yaml
  router_tree.yaml
  router_flat.yaml
  perceiver.yaml
  experiment_core_seed0.yaml
  experiment_central_3seed.yaml
src/stmem/
  data/
    download.py
    inspect.py
    alignment.py
    canonical.py
    pointcloud.py
    dataset.py
    cache.py
  models/
    spacetime_encoder.py
    framewise_encoder.py
    event_encoder.py
    metadata_adapter.py
    action_flow.py
    router.py
    perceiver.py
  memory/
    records.py
    hierarchy.py
    frontier.py
    flat.py
    streaming.py
  training/
    teacher.py
    compact.py
    gain_labels.py
    routers.py
    checkpoints.py
  evaluation/
    robomme_adapter.py
    runner.py
    metrics.py
    timing.py
  cli/
tests/
scripts/
hpc/
docs/
  method.md
  data_contract.md
  decisions.md
  experiments.md
  hpc.md
  status.md
third_party/
~~~

Keep downloaded benchmark code pinned under an ignored checkout or an explicit submodule; record licenses and patches. Do not accidentally commit a nested repository with no reproducible fetch instruction.

Expose equivalents of these commands:

~~~bash
python -m stmem.doctor
python -m stmem.data.download --tasks VideoUnmaskSwap --dry-run
python -m stmem.data.download --tasks VideoUnmaskSwap
python -m stmem.data.prepare --tasks VideoUnmaskSwap --episodes 2
python -m stmem.data.inspect --config configs/debug/data.yaml
python -m stmem.data.alignment --config configs/debug/data.yaml
python -m stmem.train --stage teacher --config configs/debug/teacher.yaml
python -m stmem.data.cache --config configs/teacher.yaml --checkpoint PATH
python -m stmem.train --stage compact --config configs/compact.yaml
python -m stmem.training.gain_labels --reader tree --config configs/router_tree.yaml
python -m stmem.train --stage router --config configs/router_tree.yaml
python -m stmem.evaluate --config CONFIG --split val
python -m stmem.report --run-root PATH
python hpc/plan.py --plan core_seed0 --dry-run
python hpc/plan.py --plan core_seed0 --submit
~~~

These are intended interfaces, not claims that code already exists. Implement them consistently and provide actual tested commands in README. A small number of unified CLIs is preferable to duplicating logic.

Each configuration must record architecture, sensor inputs, timing, action alignment, task set, data fraction, train/eval seeds, token budgets, precision, effective batch, training steps, and checkpoint dependencies. Avoid hidden defaults that alter the scientific comparison.

## 15. Completion criteria and handoff

The implementation is ready for the user's HPC stage when:

- Core package installs and imports in a reproducible environment.
- A5000/CPU-safe structural tests pass.
- A small actual RoboMME dataset is parsed and its alignment/calibration checked, or the exact access blocker is clearly reported.
- The teacher, shared compact policy, both routers, uniform reader, Perceiver, and streaming evaluation paths are implemented.
- A tiny local training/label-generation/inference cycle works.
- No future information reaches the policy or router.
- The same available leaf archive is consumed by TREE and FLAT.
- Every historical token and residual is counted.
- Checkpoint/resume and all batch-script syntaxes are tested.
- The dry-run job plan is concrete, bounded, and dependency-correct.
- README contains a short “run these commands next” sequence for the user.
- `docs/status.md` distinguishes implemented, tested locally, blocked, and not yet run.
- No benchmark success, GPU-hour total, or scalability claim is invented.

End your implementation turn with:
1. What is implemented.
2. What was actually tested on the available hardware/data.
3. Any unresolved scientific/data/installation issue.
4. The exact next local command and the dry-run HPC command.
5. Which files/settings the user must supply (normally only site paths/environment activation/resource selection).

Do not ask the user to read a long internal report to discover the next action.

## 16. References to consult when needed

Use primary sources and official implementations. Read only what resolves a concrete implementation question; do not restart a broad literature survey.

### Benchmark and software

- RoboMME paper: https://arxiv.org/abs/2603.04639
- Benchmark: https://github.com/RoboMME/robomme_benchmark
- HDF5 schema: https://github.com/RoboMME/robomme_benchmark/blob/6503b26e00ff338611fceb039417fbc64826ccc8/doc/h5_data_format.md
- Environment API: https://github.com/RoboMME/robomme_benchmark/blob/6503b26e00ff338611fceb039417fbc64826ccc8/doc/env_format.md
- Dataset: https://huggingface.co/datasets/Yinpei/robomme_data_h5
- Official policy suite/protocol: https://github.com/RoboMME/robomme_policy_learning
- ManiSkill observations: https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html
- ManiSkill installation: https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html
- PyTorch installation: https://pytorch.org/get-started/locally/
- PyTorch AMP: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- Torchrun: https://docs.pytorch.org/docs/stable/elastic/run.html
- NVIDIA architecture/driver compatibility: https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html
- JAX compilation trade-offs: https://docs.jax.dev/en/latest/jit-compilation.html
- Slurm sbatch: https://slurm.schedmd.com/sbatch.html

### Architectural ancestry and close comparators

- Universal Physics Transformers: https://arxiv.org/abs/2402.12365
- Perceiver: https://proceedings.mlr.press/v139/jaegle21a.html
- Instant Policy: https://arxiv.org/abs/2411.12633
- HALO retrieval: https://robin-lab.cs.utexas.edu/HALO/
- StreamForest: https://arxiv.org/abs/2509.24871
- AVQ-Attention: https://arxiv.org/abs/2607.12789
- Atlas, including leave-one-out retrieval supervision: https://www.jmlr.org/papers/v24/23-0037.html
- AGORA action-sensitive retention: https://arxiv.org/abs/2605.26596
- ToMe/proportional attention: https://arxiv.org/abs/2210.09461
- Mem4Act: https://arxiv.org/abs/2610.22323
- LTM: https://arxiv.org/abs/2610.21362

Do not claim first hierarchical memory, first adaptive tokenization, first action-supervised retrieval, or a new general attention theorem. The proposed contribution is the specific memory interface and demonstrated control/efficiency trade-off.

