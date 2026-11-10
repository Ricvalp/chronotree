# Chronotree: research and implementation handoff

**State as of 10 November 2026.** Prepared for an external ChatGPT discussion about which experiments to run next.

**Repository:** `/home/rvalperga/chronotree`. Code revision inspected: `7e1d169e3bbfbffb77dba724cf6b1cb4a7b73184`.

This document is self-contained. Repository links provide supporting detail; generated checkpoints, videos, HTML viewers, and downloaded data are Git-ignored and will not accompany this Markdown file automatically. The original synthesis used existing evidence. The continuation addendum below records newly completed local evaluations of a subsequently transferred 120k checkpoint; no new training was run locally.

Evidence is distinguished throughout:

- **Measured locally:** saved inspection outputs, simulations, or numerical checks available in the workstation repository.
- **HPC evidence:** results, logs, and summaries supplied by the researcher, or recorded in the checkpoint-adoption workflow. These are not newly queried live cluster results.
- **Implemented:** executable code exists; this alone does not establish a successful research result.
- **Hypothesis:** an explanation or possible next experiment, not a demonstrated cause.

Some older documents describe superseded defaults. In particular, the original four historical centers per frame and older batching settings are not the settings of the accepted frozen-vision teacher. The descriptions below use experiment-specific records and implementation rather than treating every old configuration as current.

## 1. Executive assessment

Chronotree studies **query-adaptive compression of visual history for robot manipulation**. A full-history flow-matching policy learns visual encoders and provides a reference. A shared compact policy is then trained to act using a bounded number of historical tokens. Learned TREE and FLAT readers should select useful detail from the same archived evidence, with coarse summaries preserving coverage of everything not selected individually.

The project has a functioning real-data pipeline, GPU training, streaming simulation, parallel HPC evaluation, immutable encoded caches, compact-policy training, two kinds of router labels, several router architectures, and extensive diagnostic tooling. The central scientific comparison is nevertheless **not yet established**: there is no demonstrated closed-loop advantage of learned TREE over a matched learned FLAT reader, nor a completed end-to-end quality/latency/storage comparison.

The most important findings are:

1. **The best accepted teacher is useful but uneven.** A frozen early-CLIP-feature, no-proprioception teacher at 104,110 updates achieved **20/64 = 31.25% success** on four official validation episodes per task, using full history and eight executed targets per chunk. Only **1/16** permanence-task episodes succeeded. This is a useful engineering starting point, not strong evidence that semantic visual memory is solved.
2. **A compact policy can perform meaningful manipulation.** User-supplied evaluations improved from 7/64 around 15k updates to a best reported 15/64 = 23.4375% with ten executed targets. Checkpoints and execution/reader settings vary across these results, so they are not a clean matched learning curve.
3. **History matters, but what it represents is unresolved.** In the latest paired teacher study, removing execution history reduced SwingXtimes from 4/4 to 0/4. Keeping only about two seconds of history also yielded 0/4. However, failures often occurred during grasp/lift/first-swing continuity, not only when deciding the final count. This establishes useful dependence on history, not abstract counting or reliable semantic memory.
4. **Semantic cue use remains weak or inconsistent.** A controlled reveal-video color swap barely changed the first action chunk in one scene. Ordinary VideoUnmaskSwap evaluation yielded 1/10 successes. On VideoPlaceOrder, deleting the initial video improved a four-scene diagnostic from 0/4 to 3/4. These results deserve attention before interpreting teacher agreement as a strong semantic routing target.
5. **One button-hover failure has a clear mechanism.** The compact policy sometimes predicts a button press late in its 16-target chunk, then discards that part when replanning after eight targets. At an exactly replayed state, executing the saved full chunk succeeded while replaying two saved eight-target prefixes left the episode ongoing over the same 1.6-second window. Longer execution helps some cases, but is not a universal fix.
6. **Routers fit training labels without convincing held-out ranking performance.** Both velocity-based and eight-step final-action labels produced roughly 50–52% held-out within-context pair ordering. Longer fits and richer lightweight architectures overfit. The frequently discussed “52%” is label-ranking accuracy, not task success.
7. **Labels contain some repeatable signal, but their meaning is limited.** Repeated-noise diagnostics found roughly 67–74% pair-order repeatability in one episode. Some visually interpretable moves repeatedly scored well, but other moves improved teacher agreement while worsening agreement with expert actions. Reproducible gain is not automatically useful task evidence.

**Current activity:** the researcher reports that a new 72-hour continuation of the frozen-vision teacher is training on four B200s, at the existing ending learning rate `1e-5`, with numbered checkpoints every 20k global updates. The original immutable 104,110-step teacher, cached tokens, compact policy, and router datasets remain separate and unchanged.

**Continuation addendum, later on 10 November:** a checkpoint at **120,000 updates** (+15,890) has now been verified and inspected locally. The fixed three-task history diagnostic was repeated on the same four validation scenes per task, with ten executed targets and eight flow integration steps. FULL success remains **SwingXtimes 4/4, VideoRepick 1/4, VideoPlaceOrder 0/4** at both checkpoints. Removing execution history or all history yields 0/4 on every task for both. Removing the initial video again changes VideoPlaceOrder to 3/4 and VideoRepick to 0/4; Swing has no such video, so its placebo remains 4/4. Keeping only roughly two seconds of Swing history again gives 0/4. All 48 primary plus four fresh recent-window trials completed with no policy errors; normal timeouts reached the simulator command limit.

The aggregate history benefit is unchanged, with mixed changes in individual scenes. VideoRepick gains one successful configuration and loses another; the new successful scene completes three pick/drop cycles, whereas a lost scene now attempts to finish after one. The new VideoPlaceOrder teacher grasps the cube but approaches a wrong destination in all four scenes. Standard expert-prefix checks also change from 10/10 to 9/10 on VideoUnmask because one case now lifts a wrong container. Expert-state action errors remain close, and fresh PickXtimes still grasps/lifts with less accurately centered first closure. Thus these additional updates provide **no evidence of improved semantic history use on the fixed diagnostic subset**, despite clear continued dependence on history. This is not a new 16-task success estimate and does not invalidate or replace the immutable 104k reference. Detailed methods, recovery provenance, paired videos and limitations are in [the 120k inspection note](docs/teacher_120k_inspection.md). The historical studies below retain their original checkpoint identities.

## 2. Research question and intended claims

The scientific contract is [PROMPT_CHRONOTREE.md](PROMPT_CHRONOTREE.md).

At each decision, a robot has a current observation, a public task instruction, and potentially a long history of observations. Some RoboMME tasks also supply a demonstration video before execution starts. The proposed memory keeps local space-time visual evidence in a growing archive and exposes different resolutions of that evidence to an action decoder.

The intended claim is a useful **control-quality versus total computational cost tradeoff**:

- Preserve coarse coverage of the entire available history.
- Spend a fixed active token budget on fine detail where it helps the current action decision.
- Test whether hierarchical refinement is useful compared with strong flat retrieval from exactly the same leaves.

The decisive comparison shares the frozen encoders, compact action policy, current observations, instruction, action normalization, solver, execution horizon, token budget, and evaluation seeds. TREE and FLAT receive comparable action-derived supervision. FLAT retains summaries of unselected evidence; it is not an artificially weakened “throw everything else away” baseline.

There are three distinct questions:

1. Are the local space-time tokens useful historical evidence?
2. Does query-conditioned allocation improve a fixed active memory budget?
3. Does a hierarchy offer an advantage over a flat scan of the same evidence?

The planned main experiment targets questions 2 and 3. It does not by itself prove that this tokenizer is superior to other point-cloud encoders. The researcher does not consider beating a large pretrained VLA such as pi0.5 a prerequisite for a useful memory study. Conversely, a teacher that ignores task-relevant historical cues can make the routing experiment uninformative even if it achieves some aggregate success.

**Bounded active context is not bounded total memory.** Leaves, parents, candidate keys, pending prefixes, and CPU/GPU storage all count. The full-history reference uses action-query cross-attention over history, not quadratic self-attention over the entire archive. Its historical read is already linear in the number of leaves for fixed action queries; the baseline must not be made artificially expensive.

The original progression was one seed-0 pipeline, a controlled validation comparison, a frozen official-test protocol, and final seeds 1 and 2 for the central comparison. The project is still in the seed-0 investigation stage. No completed final test-set or multiseed result is claimed here.

## 3. Benchmark, data, and causal protocol

### 3.1 Data and tasks

Training uses the released **100 demonstrations per task, 16 tasks, 1,600 episodes total**. These are variable-length trajectories, so episode count alone does not determine the number of supervised decisions or an “ideal batch size.” The official benchmark provides 50 validation and 50 test episode configurations per task. Routine interim HPC evaluations currently use four fixed validation episodes per task, totaling 64 episodes.

| Suite | Tasks |
|---|---|
| Counting | BinFill, PickXtimes, SwingXtimes, StopCube |
| Permanence | VideoUnmask, VideoUnmaskSwap, ButtonUnmask, ButtonUnmaskSwap |
| Reference | PickHighlight, VideoRepick, VideoPlaceButton, VideoPlaceOrder |
| Imitation | MoveCube, InsertPeg, PatternLock, RouteStick |

Pinned identities include RoboMME revision `6503b26e00ff338611fceb039417fbc64826ccc8`, its ManiSkill fork revision `07be6fbc66350ddca200abfb0a11b692f078f7fd`, and dataset revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337`. The working environment uses Python 3.11 and PyTorch 2.9.1 with CUDA 12.8 wheels. The current local inspection GPU is an RTX 5000 Ada, rather than the A5000 anticipated by the original brief.

No official validation expert HDF5 is assumed to exist. Teacher/compact action-loss diagnostics use released training demonstrations. Router labels have their own episode-disjoint fitting/validation split within those demonstrations; this is not the same as official unseen-configuration closed-loop validation.

### 3.2 Observation/action alignment

The verified dataset convention is `post_action_v1`: recorded observation `o_j` follows recorded command `a_j`, so the next supervised command is `a_(j+1)`. Simulator reset already executes an initial command; replay must not execute that command twice.

Real replay checks verified the convention, conditioning/execution boundary, initial state and calibration. Two full inspected VideoUnmask expert replays succeeded; median post-action joint RMSE was zero in both, and maximum nonterminal joint RMSE was at most `9.63e-6`. The wrapper's extra terminal step is treated separately from active-decision alignment.

At a policy decision:

- The current observation appears once, through the current-scene path.
- Execution history contains only strictly earlier available observations.
- The supplied initial demonstration is a separate conditioning bank.
- Blocks never cross the conditioning/execution boundary.
- Partial historical blocks contain only frames observed so far.
- Unavailable future targets, terminal padding, gaps, and unavailable coordinates are masked appropriately; valid future expert actions are the supervised targets.
- Conditioning demonstrations do not expose demonstrator commands online; those fields are also unavailable during training.

No repeated fake reset frames are inserted to fill history. At reset, execution history is empty; legitimate conditioning video can still be present. An audit found bitwise raw/prepared parity on 158 inspected frames. Invalid padding remains invalid rather than becoming extra observed evidence.

See [docs/data_contract.md](docs/data_contract.md) and [docs/reset_distribution_audit.md](docs/reset_distribution_audit.md).

### 3.3 Actions and temporal resolution

Actions are **seven absolute Panda joint-angle targets plus a gripper command**, not torques and not Cartesian deltas. The gripper uses `-1` for close and `+1` for open. Stick tasks mask the unavailable gripper dimension. The simulator's low-level joint-position controller tracks these targets; the policy does not execute planner-assisted waypoints.

The audited native cadence is 20 Hz control and 100 Hz physics. The accepted frozen-vision model downsamples both observations/history and action targets by two, giving nominal 10 Hz policy targets while the simulator still steps at 20 Hz. A target is held for two native steps, checking termination on each step. There is no interpolation between the predicted targets.

Recorded source-local indices remain the clock; missing recorded intervals must not be silently interpreted as verified continuous physical time. Training uses the versioned causal sampling rules rather than renumbering every retained frame as an original simulator frame.

| Setting for the accepted H16/10 Hz model | Meaning |
|---|---|
| Predicted targets | 16, nominal 1.6-second horizon |
| Stored checkpoint execution default | 2 targets = 4 native commands = 0.2 seconds |
| Accepted 31.25% evaluation | 8 targets = 16 native commands = 0.8 seconds |
| Many recent diagnostics | 10 targets = 20 native commands = 1 second |
| Full-chunk button diagnostic | 16 targets = 32 native commands = 1.6 seconds |
| Historical block length | 8 retained observations, not eight total blocks |

The teacher sees all available historical blocks. Downsampling reduced tokens per unit of physical time, but no fixed eight-block limit was imposed on full history. An older H32/20 Hz model also predicts approximately 1.6 seconds; H16/20 Hz predicts only 0.8 seconds.

Changing execution horizon changes the controller and can change success substantially. A checkpoint name or training configuration alone is insufficient to identify an evaluation protocol.

## 4. Current visual inputs and architecture

### 4.1 Geometry version 2

Both **front and wrist RGB-D cameras** contribute to a single world-coordinate point cloud. Each camera's calibration is used, including per-frame wrist extrinsics. Only visible depth-derived points enter the model; simulator object masks, hidden contents, object poses, and target IDs are not policy inputs.

The old preprocessing and sparse FPS centers overrepresented the table and corners in inspected scenes. The adopted geometry version uses:

- 2,048 selected points per observation, after calibrated crop and deterministic voxel selection.
- A 5 mm voxel scale and a table plane fixed by workspace calibration, not a target-object detector.
- A 75% above-table quota, with “above” defined using a 5 mm height margin over table `z=0`.
- Spatial balancing in 10 cm cells, plus global coverage for the remaining quota.
- **32 centers per retained frame**, selected by deterministic spatially balanced clustering, rather than the original four historical FPS centers.

The clustering starts from stratified FPS seeds, uses spatially balanced weighted Lloyd updates, and snaps centers to observed points. The configured split is 24 above-table and eight global centers when enough eligible points exist. It does not use object membership or guarantee one center per object. The robot is also above the table, so this is a geometric sampling heuristic, not object segmentation.

Current and historical frames use the same center-selection recipe. Center indices are prepared once and reused. Missing/invalid points do not become fabricated occupied tokens. The older geometry lives under `prepared/`; the new geometry lives under `prepared_balanced_v2/`. Raw data was retained.

### 4.2 Frozen visual descriptors

The primary model uses the **three-convolution stem and first residual stage of CLIP RN50**, independently on each camera image. It does not use the complete CLIP visual encoder or CLIP's global image embedding. These are early pretrained visual features, not guaranteed object-semantic tokens.

For a 256×256 image, the extractor produces a 256×64×64 map. Each selected point receives a 256-value descriptor sampled from its originating camera/pixel using the calibrated correspondence. FP16 descriptor sidecars are cached in `features_clip_rn50_layer1_v1/`. Geometry and descriptor identity/provenance are validated together.

The downstream point encoder trains a LayerNorm and 256→128 descriptor projection. Each point then has XYZ, RGB, and 128 projected descriptor values, totaling 134 inputs. The frozen extractor has 244,576 parameters. Task language uses a separate learned byte-based Transformer, not CLIP's text encoder.

“Trained from scratch” needs qualification: the action policy and learned point-cloud/task modules are trained on RoboMME, but this variant benefits from externally pretrained frozen visual weights. It is not a fully from-scratch perception baseline and is not a pretrained VLA.

### 4.3 Active teacher computation

| Component | Actual primary-model computation |
|---|---|
| Point features | Linear 134→256, SiLU, Linear 256→256 |
| Local edge features | Point features plus relative XYZ/time, Linear 260→256, SiLU, Linear 256→256 |
| Local pooling | Normalized spatial/temporal Gaussian weights over bounded valid neighborhoods; one local pooling stage |
| Historical leaves | 32 centers/frame × 8 retained frames = up to 256 visual tokens per complete block |
| Current scene | One current frame → 32 visual tokens |
| Task | Learned small byte-based Transformer → task representation |
| Read adapter | Shared content/metadata adaptation for leaves and summary records |
| Action input | Noisy 16×8 normalized action chunk, projected to width 256, with position and flow-time embeddings |
| Action decoder | Four pre-LN blocks: action self-attention, context cross-attention, 256→1024→256 GELU feedforward; eight attention heads |
| Output | LayerNorm and Linear 256→8 predict action-space velocity |
| Sampling | Eight explicit Euler flow-integration steps from Gaussian noise |

The visual encoder is **not four stacked graph-convolution layers**; the four layers refer to the action decoder. Local neighborhoods can span observed frames within a block. Leaves remain anchored evidence without a global register bottleneck before archival.

The accepted no-proprioception teacher has **6,164,874 serialized policy parameters**, of which **5,888,650 are active teacher parameters**. The remaining 276,224 belong to disabled state/event modules retained for shared initialization. The 244,576 frozen CLIP parameters are external to that policy count.

### 4.4 What the model does and does not receive

The accepted primary model disables both numerical proprioception and previous-command conditioning. It has no explicit current joint vector, gripper state, end-effector XYZ/pose, previous target commands, or corresponding availability token. Historical numerical robot-event tokens are disabled too. It can still see the robot in RGB-D.

Its full-history decoder context is:

```text
32 current visual tokens
+ 1 task/goal token
+ 1 clock token
+ all causal historical visual tokens, from conditioning and execution banks
```

The compact model substitutes at most B historical records for the final line. At the usual `B=128`, that means up to 162 context tokens in this variant; the 34 current/goal/clock tokens are additional to the history budget.

Tokens contain a 23-value metadata schema covering observed anchor/bounds, source-local times and age, log descendant count, depth/scale, block completion, disconnectedness, source/modality, and leaf status. Visual XYZ metadata comes from observed point neighborhoods, **not hidden end-effector state**. “Completion” here means an observed block is complete, not an oracle task-completion flag.

Metadata provides useful geometric/temporal conditioning, but also potential shortcuts. Its role has been discussed as an ablation subject; a systematic metadata-ablation result is not available. No eventual episode duration is supplied.

See [docs/frozen_vision_experiment.md](docs/frozen_vision_experiment.md), [docs/architecture_diagram.md](docs/architecture_diagram.md), and the experiment-specific variants in [docs/diagrams/teacher_architecture.json](docs/diagrams/teacher_architecture.json).

## 5. Memory representation and reader definitions

### 5.1 Leaves, summaries, and causal partial blocks

Each valid visual leaf has unit aggregation mass, independent of its Gaussian neighborhood occupancy. Parents are deterministic summaries of descendants:

- FP32 sum and integer leaf count; content mean is sum/count.
- Coordinatewise content maximum for routing.
- Count-weighted anchor, spatial bounds, and observed-time interval.
- Child topology, source/modality, depth/scale, and availability metadata.

There is no separately learned coarse encoder. Nonlinear read adapters are applied after aggregation; nonlinear position embeddings are not averaged and called exact summaries. Historical grouped tokens use the configured `log(count)` attention-logit bias. This is not an exact theorem that grouped predictions equal full-leaf predictions.

Local visual trees split spatially within blocks. Completed block roots merge into a binary-counter temporal forest. Conditioning and execution banks remain separate. Temporary trees encode incomplete execution prefixes using only available frames; completed representations replace them when the block finishes. Reusing a future full-block encoding for an earlier partial decision would be leakage and is prohibited.

### 5.2 Readers

| Reader | Historical context |
|---|---|
| FULL | All valid causal fine leaves; the reference teacher |
| TREE | A frontier of parents/leaves; each split replaces one parent with both children |
| FLAT | Individually selected fine leaves plus residual summaries of all unselected leaves, separately by nonempty source/modality group |
| UNIFORM | Deterministic coarse-to-fine splitting based on represented mass/resolution, with fixed tie rules |
| CURRENT | Empty historical context in an existing trained policy; a diagnostic control |
| PERCEIVER | Implemented learned fixed-size resampler over all leaves; no completed central benchmark result recorded here |

TREE frontiers cover every available leaf exactly once, without ancestor/descendant overlap. FLAT selected leaves plus residuals cover exactly the same evidence. Residuals count against B, exclude selected leaves, and honestly represent potentially disconnected support. Singleton/last-leaf replacement has explicit zero-extra-token handling.

Learned TREE continues splitting until the budget is filled or no valid split remains, even if predicted gains are negative. Otherwise a weak ancestor could permanently hide useful descendants. The number of roots must fit the budget; no bank is silently dropped.

“TREE” in compact-policy training or some baseline evaluations can mean **random valid TREE selection**, not a trained router. “FLAT” can likewise mean random leaf selection plus residuals. Reader names must be accompanied by selection mode/router identity.

### 5.3 Efficiency qualification

A fixed-query TREE scorer visits O(B) nodes to construct a budget-B frontier, with heap bookkeeping. This excludes archive construction, candidate-key preparation, data movement, and decoding. FLAT scores all available fine-leaf keys in a batch. Neither current router uses dense cross-attention over the complete archive to create its query.

Candidate keys can be reused while weights and archived records remain unchanged. Query construction happens once per replanning decision, and the chosen context is reused throughout eight flow evaluations. The actual end-to-end benefit still needs measurement; an O(B) traversal statement alone is not a wall-clock speedup result.

## 6. Training stages and current experimental state

| Stage | What is optimized or computed | State |
|---|---|---|
| Teacher | Learned visual/task modules and full-history action policy on expert flow targets; CLIP extractor frozen in the primary variant | Several runs completed; primary teacher now continuing |
| Encoded cache | Frozen accepted teacher encodes historical/current observations and task representations | All 16 tasks complete on HPC |
| Hierarchy cache | Deterministic parents/topology from encoded leaves | All 16 tasks complete on HPC |
| Compact policy | Shared metadata/read adapter, clock projection, action decoder; encoders frozen | Trained through at least the designated 90k snapshot |
| Initial router labels | Frozen teacher–compact gains for candidate TREE/FLAT moves | Velocity and final-action-8 datasets generated |
| Router fits and refresh | Lightweight scorers; policies remain frozen | Initial 5k, refresh, 10k fits completed; longer/architecture diagnostics run |
| Controlled online reader comparison | Same compact checkpoint, seeds, B, solver, execution horizon | Implemented; no confirmed learned-router advantage in the audited results |
| Final benchmark/replication | Frozen configuration, all official test seeds, central training-seed replication | Not established |

The compact policy is trained on **expert demonstration flow-matching targets**, not directly distilled teacher-generated actions. The teacher becomes the utility reference when generating router labels later.

The compact training mixture is 45% random TREE, 45% random FLAT plus residuals, 5% UNIFORM, and 5% CURRENT. Budgets 64, 128, and 256 and valid intermediate refinements are sampled. A single shared decoder avoids confounding reader quality with separately trained action policies.

The original compact schedule ran 30k updates with LR around `1e-4`, warmup, and cosine decay to `1e-5`; it was then continued at the ending rate. The prescribed early selection protocol evaluated 15k and 30k checkpoints with both TREE and FLAT and selected a checkpoint by their mean score, not by choosing whichever reader happened to win. The later 90k router baseline was explicitly designated by the user; it should not be described as the outcome of that earlier formal selection rule.

## 7. Main training experiments and what they established

### 7.1 Early XYZRGB teachers and learning-rate tests

Early runs used XYZRGB points, explicit robot state/command information, and sparse historical centers. Training loss decreased, but closed-loop trajectories often resembled the correct skill while missing the cube. Pressing the button in some scenes did not establish general object grounding.

The researcher compared lower LR around `1e-4` with higher LR `1e-3`, including checkpoints near 50k. Neither eliminated the inspected grounding failure. No controlled all-task LR superiority is established in the retained evidence. Lower flow loss alone was not treated as proof of better closed-loop control.

An apparent periodic loss/step-time oscillation was traced to sampler ordering: task buckets were sorted by decision index, producing periodic easy/short versus hard/long data. Shuffling and rank/task ownership were corrected and versioned. It was not evidence by itself of an unstable optimizer.

### 7.2 Geometry improvement

An old inspected historical encoding could contain hundreds of leaves whose pooled support was effectively table-only. Sparse FPS centers often landed at spatial extremes; local radius limits then excluded small objects. Version-2 above-table sampling and 32 balanced centers/frame addressed this real representation problem.

Afterward, one inspected PickXtimes state retained 123 cube points, with two current tokens and 15 latest-history tokens receiving majority cube pooling mass. Cube and table features were distinguishable. Nevertheless, the model could still target the wrong place. Thus **losing the object during sampling was a plausible contributor, but not a sufficient explanation after v2**.

The geometry change also increased historical token count substantially. It was not a free quality improvement and required batching/loading work to remain practical.

### 7.3 Previous-command ablation

At a matched PickXtimes state, perturbing historical joint-1 command targets by +0.05 radians changed the first four predicted TCP targets by **22.57/28.76 mm** for the old low/high-LR teachers. Perturbing only the separate current latest-command field changed them by **0.0026/0.0214 mm**. A smaller historical +0.01-radian edit changed predictions by 4.49/5.75 mm.

Moving the cube physically by +100 mm in X and rerendering the inspected visual history/current scene changed imminent predictions by only **0.154/0.284 mm**. These interventions have different units and are not a normalized feature-importance ratio. They nevertheless exposed a strong dependence on expert command history relative to the tested visual relocation.

Two matched continuations from the lower-LR approximately 50k checkpoint were trained for another 15k updates: control and removal of previous seven-joint command targets. A separate run started from scratch without those commands. The matched ablation retained measured state/gripper information; it was not yet the later complete no-proprioception model.

Removed-channel invariance was verified. Some handoffs improved, but the short fresh-scene grounding problem remained: first-four response to a 100 mm cube shift was 0.148 mm for the 64,978-step control and 0.083 mm for the ablation, with neither grasping in that short probe. Command removal alone did not solve the problem. The fresh approximately 16k run was not exposure-matched to those continuations.

### 7.4 Longer action horizons, frozen features, and no proprioception

Experiments included H16 and H32 at 20 Hz; frozen early-CLIP features with and without numerical proprioception; and H16 at 10 Hz for both actions and observations/history. The adopted model combines frozen features, no explicit proprioception/previous commands, improved geometry, and the longer physical horizon of H16/10 Hz.

The following local comparison used transferred checkpoints on 2 November. Expert-state error is imminent 0.2-second TCP prediction error on ten training states. First-close error is a separate short fresh PickXtimes validation rollout. These measurements are not aggregate task success.

| Model | Updates | Expert-state TCP error | First-close XY miss | Fresh-scene pickup |
|---|---:|---:|---:|---|
| Frozen vision, H16/10 Hz | 104,110 | 5.40 mm | 4.28 mm | Grasp; cube rose 135.0 mm |
| XYZRGB H16/20 Hz | 114,093 | 6.91 mm | 299.36 mm | No grasp/lift |
| XYZRGB H32/20 Hz | 91,859 | 6.48 mm | 8.02 mm | Grasp; cube rose 125.5 mm |

The frozen and H32 policies improved natural pickup in these scenes. Multiple input, horizon, and exposure factors differ, so this is not an isolated causal demonstration that CLIP alone, removal of proprioception alone, or H32 alone caused the improvement.

Under 100 mm reset cube relocations, frozen/H32 imminent motion still changed only roughly 0.9–1.5 mm, whereas full-chunk endpoints changed about 42–52 mm. A small first-chunk-prefix response does not mean the complete prediction is invariant; successful pickup in an unmodified scene also does not prove robust correction after an arbitrary relocation.

A separate frozen-feature, no-proprioception **current-only model** was trained to 100k updates without history or clock/time metadata. Its initial HPC attempt encountered a GPU-reset fault. It is distinct from deleting history at inference in a history-trained teacher.

In the 30 October local inspection, this current-only model succeeded from reset on both inspected VideoUnmask **training** scenes, and at all five expert-prefix lengths on those scenes. On the fresh PickXtimes validation scene, it missed by 37.30 mm at first closure, with only 1.25 mm maximum cube rise and no sustained grasp within 160 native commands. These probes used execute-two at 10 Hz. Familiar VideoUnmask scenes can therefore be solved without supplied history by this trained policy, which leaves scene priors or other shortcuts plausible; it does not show that history is unnecessary for unseen hidden-object configurations. A matched all-task aggregate result for the current-only model has not been recovered here.

Sources: [docs/command_ablation_inspection.md](docs/command_ablation_inspection.md), [docs/h16_h32_inspection.md](docs/h16_h32_inspection.md), [docs/checkpoint_inspection_20261102.md](docs/checkpoint_inspection_20261102.md), [docs/current_only_experiment.md](docs/current_only_experiment.md), [docs/four_model_inspection_20261030.md](docs/four_model_inspection_20261030.md).

## 8. Closed-loop success ledger

### 8.1 Accepted teacher result

The accepted frozen-vision teacher is step **104,110**, SHA256:

```text
eb57ccd4c131888a5f5d6225a3a0cbe5cd14b3afc5c6ff0a1a17ee319cd70918
```

The supplied completed HPC summary records **20 successes, 38 terminal failures, six timeouts, zero errors, zero missing episodes**, across 64 planned episodes. Since every task has four episodes, macro success and pooled success both equal **0.3125**. Timeouts are failures in that denominator.

Protocol: FULL history; H16/10 Hz; eight executed targets; eight FM steps; four official validation episodes/task. It is not the full 800-episode validation or test suite.

| Suite | Task | Successes / 4 |
|---|---|---:|
| Counting | BinFill | 1 |
| Counting | PickXtimes | 1 |
| Counting | SwingXtimes | 3 |
| Counting | StopCube | 0 |
| Permanence | VideoUnmask | 0 |
| Permanence | VideoUnmaskSwap | 0 |
| Permanence | ButtonUnmask | 1 |
| Permanence | ButtonUnmaskSwap | 0 |
| Reference | PickHighlight | 2 |
| Reference | VideoRepick | 3 |
| Reference | VideoPlaceButton | 1 |
| Reference | VideoPlaceOrder | 2 |
| Imitation | MoveCube | 2 |
| Imitation | InsertPeg | 3 |
| Imitation | PatternLock | 0 |
| Imitation | RouteStick | 1 |

Suite success: Counting 31.25%, Permanence 6.25%, Reference 50%, Imitation 37.5%. Four episodes per task produce coarse 25-point task-rate increments; individual-task percentages should not be overinterpreted.

The teacher was explicitly accepted as sufficient to start the compact-memory study. Acceptance is a research decision, not proof of robust semantic history use or superiority to external baselines with different pretraining, sensors, and protocols.

### 8.2 Compact-policy results

| Checkpoint/evaluation description | Reported aggregate success | Evidence limitation |
|---|---:|---|
| Approximately 15k updates | 7/64 = 10.9375% | Early compact evaluation |
| Approximately 27k updates | 10/64 = 15.625% | User-supplied completed summary |
| Approximately 80k updates | 11/64 = 17.1875% | Reader/protocol must be recovered from evaluation manifest for exact pairing |
| Later execution-horizon comparison; best was execute-ten | 15/64 = 23.4375% | User confirmed ten targets; reader identity was not confirmed in the recorded exchange |

These establish useful compact-policy behavior and motivate further work. They do not establish a matched TREE/FLAT ranking, a clean learning curve, or a fair numerical gap to the teacher's execute-eight score. Video quality and smoother movement are useful diagnostic observations, not substitutes for success counts.

The production router study froze a **90,000-step compact checkpoint**, SHA256:

```text
4fe22fd00fbc64dc740ba046d6ce1e58ade2a70580f8792e152477caad316427
```

Several earlier local diagnostics instead used an **83k** compact checkpoint. Those are identified below and must not be merged as if all experiments used the exact production router policy.

## 9. Behavioral diagnosis: what has and has not been explained

### 9.1 Covariate shift and expert-prefix handoffs

The early policies were much more successful after genuine expert prefixes. On two VideoUnmask training scenes, old near-50k teachers failed from prefixes 0/4/16 within a 320-native-command cap but succeeded after 32 or 64 expert commands. At prefix 32 the expert had not yet grasped, so the policy still completed substantive autonomous manipulation.

This is consistent with ordinary behavioral-cloning covariate shift: training sees expert states and histories; deployment sees the consequences of its own errors. Prefixes change pose, history, task phase, and previous commands together, so this is not an isolated history-channel causal test.

By step 104,110, the frozen teacher succeeded on all ten VideoUnmask handoff cases: two training scenes × prefixes 0/4/16/32/64. PickXtimes showed grasps and lifts and reached later task phases, but did not complete within the short cap. These inspections replanned every four native commands, which is only two targets for the 10 Hz model; they are not directly comparable to the eight/ten-target benchmark evaluations.

The demonstrations are not entirely devoid of recovery: two inspected VideoUnmask experts contain an unsuccessful initial grasp followed by a retry. They still do not cover arbitrary policy-induced states. There is no demonstrated general covariate-shift mitigation in the current project.

### 9.2 Numerical precision and a deferred action-input mismatch

Same-input FP32-encoder/BF16-decoder versus all-BF16 probes changed imminent targets by roughly 0.4–1.8 mm, much smaller than the original approximately 100 mm targeting errors. This makes precision an insufficient explanation for those large misses. It does not rule out accumulated or near-contact effects.

A separate concrete mismatch remains **unfixed by user choice**: training zeroes unavailable/padded future action coordinates before action self-attention, whereas inference initializes all coordinates with Gaussian noise. This concerns terminal chunk tails and unavailable gripper dimensions. A synthetic probe confirmed that those slots can influence valid outputs through self-attention.

That proves a computational path, not that this mismatch causes the trained policy's failures. The correct causal target masks must remain; “fixing” this by supervising unknown future targets or inventing repeated terminal expert actions would be invalid. Any later repair should align decoder input conventions while preserving target validity.

### 9.3 Button hovering and discarded chunk tails

This is the strongest specific causal failure diagnosis so far. The policy can finish much of a task, hover well above the button, and repeatedly postpone the press rather than fail to track a nearby contact command.

Using the **83k compact policy, UNIFORM, B=128**, VideoRepick episode 0 at native command 256:

- Actual TCP–button-cap separation was 121.5 mm.
- Predicted descent by target 8 was only **3.6 mm**.
- Predicted descent by target 16 was **112.3 mm**.
- Replanning after eight targets discarded almost all of the planned descent.

At another inspected state, the corresponding descent was 0.1 mm versus 113.5 mm. Fresh noise draws showed similar postponement. The teacher, conditioned on the student's same observed state/history, usually brought descent earlier in the chunk.

The simulator was reset and the exact saved command prefix replayed, with zero handoff-state/proprioception mismatch. **Executing the saved full 16-target student chunk succeeded** in the first case; replaying the next two baseline eight-target prefixes left the episode ongoing over the same 32-native-command, 1.6-second window. Together with the recorded late-descent chunks, this demonstrates postponement into the discarded tail. The bounded branch itself is not a complete newly replanned rollout or an eventual-failure measurement.

A matched four-scene VideoRepick pilot gave 0/4 with execute-eight and 2/4 with execute-sixteen. Longer execution also changes earlier manipulation, so this does not justify execute-sixteen universally. Some other button-adjacent states were still waiting for another pickup; similar-looking failures need not share one cause.

An important nuisance was found: random reset UUIDs contribute to node IDs, and UNIFORM uses ID-based tie-breaking for equal-mass nodes. Identical sensors/noise can therefore select different compact contexts across resets. An early pilot was confounded. The final paired button experiment used **diagnostic-only stable observation IDs**, verifying identical initial inputs/chunks/native commands. The production ID convention was not silently changed.

See [docs/button_hover_diagnostic.md](docs/button_hover_diagnostic.md).

### 9.4 Count-conditioned expert histories

An ad hoc diagnostic used PickXtimes rather than an ambiguously interpreted BinFill task. Genuine official-planner histories completed `k` pick/place cycles, with requested count `X` in normal task language. One training scene, X∈{1,2,3}, k≤X, and four paired noise seeds were used. Teacher 104,110 FULL and student 15k UNIFORM/B128 generated 72 chunks each across the diagnostic arms.

For natural histories, target-8 movement toward the button was:

| Completed cycles k | Requested X | Teacher progress | Student progress |
|---:|---:|---:|---:|
| 1 | 1 | +60.3 mm | +62.5 mm |
| 1 | 2 | −5.2 mm | −2.3 mm |
| 1 | 3 | −5.5 mm | −3.4 mm |
| 2 | 2 | +48.6 mm | +62.2 mm |
| 2 | 3 | −3.4 mm | +0.5 mm |
| 3 | 3 | +49.1 mm | +66.8 mm |

This is meaningful conditional behavior: after enough cycles the predicted chunk heads toward the button; otherwise it does not. However, these are forward-kinematics predictions, not executed successful presses, and count correlates with duration/history length.

An intermediate same-current/same-clock test retimed the genuine histories. At k=2, X=3, where another pickup should occur, the student predicted +58.9 mm toward the button while the teacher predicted only +0.36 mm. History lengths still differed, and retiming changes consistency; this weakens a claim of robust student counting without isolating a unique shortcut.

Equal-duration “idle padding” repeated reset observations; it was not genuine simulated waiting. It changed history content, mass, block boundaries, and timing and made the student approach the button in all six cells. It is an OOD intervention, not proof that the policy only uses a clock. Corrected final outputs preserve original observation IDs to avoid changing UNIFORM tie-breaking.

See [docs/count_history_diagnostic.md](docs/count_history_diagnostic.md).

## 10. Router supervision: exactly what is being learned

The teacher, compact policy, encoders, normalization, and cache are frozen during label generation and router fitting. Let F be a valid selected history and F′ the context after one candidate refinement. The signed utility is:

```text
gain = (teacher–compact discrepancy before − discrepancy after)
       / additional active history tokens
```

TREE replaces a parent by both children. FLAT promotes a leaf out of a residual and updates the residual. Both changes preserve exact coverage. Invalid or zero-extra-token candidates are handled explicitly rather than dividing by zero.

### 10.1 Velocity labels

With normalized expert chunk `a`, Gaussian noise `ε`, and flow time `u`:

```text
x_u = (1 − u) ε + u a
expert flow target = a − ε
D(F) = masked normalized MSE(v_teacher(x_u,u,FULL), v_compact(x_u,u,F))
```

The same expert/noise interpolation and time are shared across teacher, compact-before, and all compact-after candidates. Expert future actions are used for this offline computation, never as router features.

### 10.2 Final-action labels

Teacher and compact contexts each start from the **same pure Gaussian chunk**, independently integrate eight Euler steps, and compare the final normalized action chunks with masked MSE. Expert actions do not initialize this rollout. Common noise reduces comparison noise; candidates are not independently resampled.

Four integration steps are implemented as a separate cheaper label objective, but no measured four-step label experiment is established here. Using fewer label-generation steps would not silently change the eight-step evaluation solver.

Both objectives optimize **agreement with the full-history policy**, not directly environment success. A teacher can confidently choose a wrong target; matching it can then reward the wrong evidence.

### 10.3 Label datasets and splits

- Initial **16,000 decision-context IDs per reader/objective**, up to four candidate moves each, approximately 64k move labels.
- One refresh of **8,000 contexts** per reader/objective, using the corresponding frozen initial router with 50% exploration.
- Budgets balanced over 64, 128, 256; early/late decisions and partial refinements included.
- Episode-disjoint 80/20 fitting/validation split within released training episodes, maintained through refresh.
- Teacher and compact training may already have seen both partitions. “Held out” here means held out from router fitting, not unseen by the whole policy system.
- Four independent one-GPU label workers share teacher computations where appropriate. Initial user-run shard times were approximately 425–509 seconds.

These are candidate-move labels, not 16k new demonstrations. Adding more distinct contexts is possible with explicit provenance and unchanged splits, but the current evidence does not establish that data quantity alone is the bottleneck.

## 11. Router architectures and fitting results

### 11.1 Original local scorer

The default scorer has width 256 and 128-dimensional keys/queries:

- Query input: mean current visual content (256) + task representation (256) + 27 state/command/mask/clock/budget slots = **539 dimensions**. Numerical state and command fields are zero in the primary no-proprio/no-command experiment; clock and budget remain.
- Candidate input: content mean (256) + coordinatewise max (256) + metadata (23) = **535 dimensions**.
- Each tower: Linear → SiLU → Linear → LayerNorm.
- Score: scaled query–key dot product plus linear metadata bias.
- **171,288 parameters.**

The query is fixed during retrieval. The score does not observe the entire currently selected frontier F, the sampled flow noise/time, or expert future actions. It estimates expected marginal utility across training contexts, not exact conditional gain for every possible frontier.

### 11.2 Implemented architecture screen

Fresh 100k fits used the same initial final-action-8 labels and episode partitions, with extra inputs reconstructed from the encoded cache rather than new teacher inference.

| Variant | Reader applicability | Parameters |
|---|---|---:|
| Dot baseline | TREE, FLAT | 171,288 |
| Nonlinear interaction MLP | TREE, FLAT | 237,081 |
| Child-aware candidate | TREE | 388,505 |
| Current-token attention | TREE, FLAT | 638,745 |
| Child-aware + current attention | TREE | 790,169 |

The interaction MLP uses `[q, k, q*k, abs(q−k)]`. Child-aware variants embed both actual children and fuse parent, mean-child, and max-child information. Current attention uses **eight learned task-conditioned query slots, width 128, four heads, two blocks**, reading all current tokens and their metadata plus auxiliary conditioning. Candidates then interact with those eight query outputs.

This addresses the original mean-pooled-current bottleneck without attending over the full historical archive. There is still no frontier-dependent rescoring. Child-specific variants are architecture diagnostics, not evidence of a matched TREE advantage over an unchanged FLAT scorer.

### 11.3 Measured initial/refresh results

Four scorers used batch 256, AdamW LR `3e-4`, signed Smooth L1 regression with training-fitted median-absolute-gain scaling, 5k initial updates, one refresh, then 10k total updates.

| Reader/objective | Fitting labels | Held-out labels | Held-out pair-ranking agreement | Held-out scaled loss |
|---|---:|---:|---:|---:|
| TREE velocity | 75,896 | 18,984 | 51.227% | 10.953 |
| FLAT velocity | 76,068 | 19,054 | 51.723% | 11.873 |
| TREE final-action-8 | 75,896 | 18,982 | 50.054% | 43.079 |
| FLAT final-action-8 | 76,068 | 19,054 | 51.238% | 64.677 |

Ranking compares candidate pairs within the same context, excluding near-tied labels. Around 50% is near-chance ordering. Scaled loss magnitudes across different objectives should not be compared naively because their target distributions and fitted scales differ.

The very short router jobs were plausible: one initial TREE-velocity 5k fit took 16.4 seconds on H200. Fitting a small scorer from precomputed arrays is much cheaper than visual encoding or label generation. Fast completion did not imply a skipped training stage.

### 11.4 Longer fitting and overfitting

For the TREE-velocity continuation on unchanged initial labels:

| Metric | At 5k | At 100k |
|---|---:|---:|
| Full fitting loss | 10.2237 | 8.9833 |
| Fitting ranking agreement | 52.358% | 60.541% |
| Held-out loss | 10.3758 | 11.3435 |
| Held-out ranking agreement | 50.151% | 50.880% |

The held-out zero-predictor reference loss was 10.3591, better than the 100k fitted result. Best held-out ranking was 51.399% at 8k. The additional 95k updates took about 260 seconds.

The researcher also ran the final-action 100k diagnostic and the architecture screen and observed overfitting without convincing held-out improvement. Complete per-variant numeric snapshots were not found in the audited local records, so a fabricated detailed leaderboard is deliberately not supplied.

A separate 256-context memorization diagnostic was proposed but **declined by the researcher**, since fitting/overfitting was already observed. It was not run and is not a required next step.

### 11.5 The “52.5% checkpoint” and closed-loop status

The documented peak FLAT-dot validation ranking was **52.33% at 2k updates**. That exact checkpoint was not retained; automatic validation-best checkpoint saving was not in place for that run. A retained 50k snapshot scored **51.98%**.

This is label-ranking accuracy, not 52% closed-loop success. The implemented matched online comparison uses the retained FLAT-dot 50k scorer versus random FLAT, the same frozen 90k compact policy, B128, execute-ten, eight integration steps, and the same four validation episodes/task. No measured outcome establishing a learned-router benefit is available in the audited local records.

Sources: [docs/router_training_hpc.md](docs/router_training_hpc.md), [docs/router_overfit_hpc.md](docs/router_overfit_hpc.md), [docs/router_architectures_hpc.md](docs/router_architectures_hpc.md), and saved W&B snapshots under `runs/inspection/router_objectives_status_20261105/` and `runs/inspection/router_overfit_wandb_20261104/`.

## 12. Are high-gain historical tokens actually meaningful?

### 12.1 Repeatability across action-noise draws

A local diagnostic used teacher 104,110 and the then-available **83k compact checkpoint**, one real VideoUnmask training episode, 30 fixed contexts per reader spanning 21 decision indices, budgets 64/128/256, four candidates, and 16 shared noise draws. The first eight draws selected high/low candidates; the last eight tested them.

These are **measured counterfactual gains**, not scores predicted by a trained router.

| Reader/objective | Mean pair-order repeatability | Discovery-high beats discovery-low on held-out mean |
|---|---:|---:|
| TREE velocity | 70.9% | 23/30 |
| TREE final-action-8 | 74.2% | 21/30 |
| FLAT velocity | 69.4% | 27/30 |
| FLAT final-action-8 | 67.2% | 20/30 |

There is repeatable signal and substantial noise. Final-action labels are not uniformly more stable. This is one episode and an 83k compact policy, not a direct diagnosis of all production 90k labels.

Interesting visual examples included:

- Refining the container hiding the target beat a distractor refinement on 8/8 fresh draws, with held-out gains about `+1.79e-4` versus `−6.35e-5`. The cube was already covered in that frame; earlier reveal information remained elsewhere in history.
- A target-region leaf beat a table leaf on 8/8 draws while **both gains were negative**. Higher gain can mean “less harmful,” not helpful.
- One discovery-high candidate lost on 7/8 fresh draws, demonstrating a noise-dependent reversal.
- A table-region leaf won all 16 teacher-agreement comparisons but slightly worsened expert-action error on 7/8 held-out draws. A target-region leaf slightly improved expert error while receiving a poor teacher-agreement score.

The last case directly demonstrates disagreement between teacher agreement and expert-action agreement; neither measurement is itself a closed-loop success result. Featured cases were chosen for interpretation; all generated cases remain available, and selected examples should not be treated as unbiased prevalence estimates.

### 12.2 Exposing finer reveal detail

After the exact 90k compact checkpoint arrived, a stronger cue-resolution study used one VideoUnmask training episode, decisions 0/20/64, budgets 64/128/256, eight draws, and six matched-budget exact-coverage frontiers.

At reset/B128, revealing fine detail near the blue cube instead of leaving the reveal coarse changed predicted TCP paths by **1.506 mm on average**, with a 2.857 mm endpoint change and no nearest-target-region switch. The earlier covered-container refinement changed the path by only 0.354 mm.

This does not prove the cue is absent or unused: the coarse summaries still contain it. It does show that these particular refinements did not cause the large action redirection originally hoped for.

### 12.3 Reveal-video appearance counterfactual

A more decisive intervention swapped saturated blue/red appearance across all 16 retained 10 Hz reveal frames (native frames 0, 2, …, 30) in both cameras. Both source-camera RGB and corresponding canonical point RGB were recolored, then full frozen visual maps and affected historical encodings were recomputed. Geometry, current observation, task text, noise, and selection IDs stayed fixed. The instruction still targeted blue, so the intended target moved by 237.6 mm.

With 16 paired noise draws, mean first-chunk TCP path changes were:

| Policy/context | Mean change |
|---|---:|
| Teacher FULL | 0.524 mm |
| Student UNIFORM | 0.049 mm |
| Student protected coarse cue | 0.017 mm |
| Student fine detail on both cue regions | 0.762 mm |
| Student random FLAT | 0.100 mm |

No intended target redirection appeared in the first chunk. The table reports averages; the largest pointwise trajectory change across the tested cases was 15.73 mm. Controls verified exact unchanged-input reconstruction; 80/512 cue leaves changed, with 14.9% aggregate relative L2 change, and unrelated leaves were unchanged.

This is a negative result for **one scene's initial chunk**, not proof of total history blindness throughout every rollout. Layout/generation correlations and other scene cues remain possible. Auditing 100 training VideoUnmask scenes found 99 distinct target XY locations and varied target colors/ranks, so “the target is always at exactly one position” is false; more subtle shortcuts are not ruled out.

### 12.4 Ordinary VideoUnmaskSwap configurations

The teacher was then evaluated on ten ordinary unmodified validation configurations, episodes 0–9, FULL history, execute-ten, eight flow steps. Result: **1/10 success, nine failures, no infrastructure errors or timeouts**.

Eight first grasps selected the wrong container; one scene briefly contacted the correct one then picked wrongly; the successful scene completed both required pickups. Chosen identities and positions varied, so the policy did not simply always select the same container. A matched random complete-task baseline was not measured; calling the 1/10 result statistically “worse than random” would be unsupported.

See [docs/gain_repeatability.md](docs/gain_repeatability.md) and the visual report map below.

## 13. Latest history-use study: positive evidence and counterexamples

The latest completed local study tested the **teacher only**, using the same 104,110-step checkpoint, H16/10 Hz, execute-ten, eight integration steps, and validation episodes 0–3 on three tasks.

Four arms retained everything, erased the initial demonstration, erased execution history, or erased all history. These are inference interventions in one history-trained policy, not separately retrained alternatives. The main experiment contains 48 simulations; four recent-history and four permutation follow-ups bring the total to **56 fresh simulations**. Copied FULL reference rows are not extra trials.

| Task | FULL | Erase initial demo | Erase execution history | CURRENT |
|---|---:|---:|---:|---:|
| SwingXtimes | 4/4 | 4/4, since no initial demo exists | 0/4 | 0/4 |
| VideoRepick | 1/4 | 0/4 | 0/4 | 0/4 |
| VideoPlaceOrder | 0/4 | 3/4 | 0/4 | 0/4 |

### 13.1 SwingXtimes: substantial execution-history dependence

Keeping only the last two complete eight-frame blocks plus the pending prefix retained 16/18/20/22 sampled frames, roughly two seconds. This **recent-only** arm scored 0/4. All four trials grasped/lifted the correct cube, then opened prematurely before completing the first swing. The first 60 native commands matched FULL exactly; divergence started at command 61 when an old block was first removed.

A count-preserving permutation of execution payload/time associations also scored 0/4, all normal timeouts at the effective command limit. The first 20 commands matched FULL; divergence began at 21 with the first historical read. Three trials failed pickup; one lifted then dropped before completing its first swing.

This is strong evidence that correctly organized past observations support the learned movement sequence. It is not a clean counting test: the failures arise before the policy reaches a final count-dependent button choice. Permutation is OOD, and encoded contents retain local temporal clues even after their outer time association is changed.

### 13.2 VideoRepick: useful sequencing evidence

In episode 1, FULL completed three correct pick/drop cycles and pressed the button, succeeding after 471 commands. Removing the initial video still grasped the correct cube, but carried it toward the button during the first cycle and failed after 153 commands. This is a concrete example where legitimate history changes a useful task sequence, not merely arbitrary sensitivity.

In episode 3, FULL picked the correct cube, ID 14 of 15 in report-only simulator annotations, twice before timing out. Without the initial video, it never lifted and briefly contacted a wrong cube. That is suggestive identity benefit, but not a completed-task success.

### 13.3 VideoPlaceOrder: history can be harmful

FULL had three wrong-pad placements and one placement near the correct pad but 6.03 cm away, outside a 5 cm tolerance. Removing the initial video produced 3/4 successes, compatible with a positive-X/negative-Y location preference in this small set. The failures did not consistently follow a simple “use the last demonstrated pad” rule either.

This is an important counterexample to the claim that the teacher reliably extracts the intended semantics from supplied videos. The earlier accepted benchmark had 2/4 on this task using execute-eight; the latest study uses execute-ten. Those different execution settings prevent treating the outcomes as repeated measurements of an identical controller.

### 13.4 Controls and defensible conclusion

Reset fingerprints and noise seeds were paired; there were no rollout/observer errors. Fifty-two fixed-current shadow probes passed, with zero error when replaying the original FULL input; alternative histories could intentionally change predictions. Erasing the nonexistent SwingXtimes initial video produced bitwise identical complete action sequences, a useful placebo control. Timeouts were ordinary benchmark-limit outcomes.

**Defensible conclusion:** useful history dependence is established for movement continuity and at least some task sequencing. Reliable semantic target/destination recall and abstract counting are not established. History deletion, truncation, and permutation are distribution shifts; their failures should not be equated with an optimal current-only or short-memory policy's capability.

The same matched study has not yet been run for the compact policy or learned routers. A teacher's dependence on history does not automatically show that useful information survives compression or that a router selects it.

## 14. Attention mass and entropy telemetry

The code logs diagnostic decoder attention every ten training updates on fixed scenes/noise. It measures selected action-head cross-attention, including the final sampled integration stage, rather than recording every training tensor or gradient.

For attention probabilities `p_i` over context tokens:

```text
mass(group) = sum of p_i over tokens belonging to that group
conditional p_i = p_i / mass(group)
normalized entropy(group) = −sum(conditional p_i log conditional p_i) / log(n_group)
```

Aggregation averages appropriate heads, action queries, layers, and diagnostic examples. Empty/single-token groups require explicit handling. Mass is also compared with the uniform-attention reference `n_group / n_context`.

History may receive most raw mass simply because it contains most tokens. High conditional entropy means attention is broadly distributed within a group; low entropy means concentration. Neither alone establishes semantic use or causal necessity. Value vectors and downstream residual computation also matter.

An archived **early** frozen-vision snapshot around step 260 had history mass about 0.963 versus a uniform reference 0.964, and normalized entropies near 0.999. That was nearly uniform attention at that early stage, not evidence about the final 104k teacher. Older models showed more concentrated attention and differing event/visual mass, but steps and input configurations were not matched.

The numerical mass/entropy implementation was checked. Interpretation remains observational; the paired history and task-choice interventions provide stronger evidence about behavior than a large attention mass alone.

## 15. Engineering progress and performance limitations

### 15.1 Training and preprocessing

Early startup spent about 26 minutes reading large prepared payloads while constructing decision indices and normalization. That was reduced by selective reads of action/state information and index reuse. This was separate from repeated per-step encoder cost.

Early “batch 32” training accumulated essentially one example at a time. True padded microbatches, validity masks, DDP, shuffled sampling, and batched encoder work were implemented. Later worker processes loaded/packed upcoming batches with persistent handles and bounded prefetch queues. Activation checkpointing was disabled in the main configuration at the user's request; it remains an explicit option rather than an assumed speed improvement.

The accepted teacher's relevant batch settings were effective batch 128, four GPUs, microbatch eight per GPU, encoder block batch 32. This gives four microbatches per rank per optimizer update. The encoder block batch controls how many local historical blocks are encoded together; it is not the predicted-action horizon or the number of episodes in an optimizer batch.

Larger B200 configurations, including microbatch 64 and encoder block batch 128, OOMed. DDP replicates the model and does not pool GPU memory for one replica. Effective batch, per-device microbatch, and encoder block batch must be distinguished when interpreting speed or memory.

Frozen descriptor loading initially caused very slow steps, with major warm-cache effects. Bounded descriptor prefetch, reusable pinned buffers, and asynchronous transfers were added. User reports indicated substantial later improvement, but early approximately 100-second and later approximately 15-second steps are not a controlled benchmark across identical histories/storage states. A local bounded chunk forward/backward probe improved from approximately 0.985 seconds to 0.608 seconds with prefetch; that is not a whole-job B200 throughput claim.

Optional node-local staging exists but remains disabled because the researcher did not know an available scratch path/capacity.

### 15.2 Encoded cache and hierarchy optimization

There are three different preprocessing levels:

1. Geometry preparation: calibrated visible points and selected center indices.
2. Frozen CLIP feature sidecars: early image descriptors aligned with those points.
3. Accepted-teacher encoded cache: final learned local visual/task representations, followed by derived hierarchy summaries.

The latter two are not equivalent. Freezing CLIP does not make learned local encodings independent of the teacher checkpoint.

The first compact run reached only 317 updates in roughly nine hours, with 92–108 seconds/update, little GPU memory use, and most time in cache/hierarchy work. Worker prefetch brought user-observed time to roughly 13 seconds/update. Further changes precomputed deterministic parent/topology arrays, directly gathered required records, avoided building a tree for FLAT and CURRENT examples, and used 64 CPUs with 16 workers per rank on two GPUs.

A bounded local eight-batch input-pipeline benchmark improved from 16.72 seconds serial to 2.53 seconds indexed with zero workers and 1.14 seconds with eight workers, with parity checks. This is not a verified steady-state HPC end-to-end timing. A one-episode hierarchy sidecar added about 19.15 MB to a 21.79 MB encoded file; it is not a full-dataset storage estimate.

Precomputing deterministic parent summaries does not eliminate random frontier/leaf selection during training. It removes repeated arithmetic/topology construction. Exact leaf coverage and separate causal partial-block representations remain required. Derived cache virtual datasets reference original encoded shards, so original shards must be retained.

### 15.3 Evaluation, logging, and infrastructure

Parallel evaluation normally requests four H200s, four independent simulator/policy workers per GPU, 64 CPUs, four episodes/task, and a global ten-video cap. These are 16 separate worker processes, not one batched vectorized simulator or distributed policy forward. Manifests assign each planned episode once, with explicit aggregation and failure accounting.

Training/evaluation W&B runs use `chronotree`; router fits use the separate `chronotree-routers` project. Local JSONL metrics, progress heartbeats, immutable evaluation snapshots, and Slurm stdout/stderr remain available independently of W&B. Frequent action point-cloud and attention diagnostics are bounded rather than logging every batch's full media.

Some historical failures were diagnosed as infrastructure: GPU busy/unavailable, “GPU requires reset,” NodeDown/requeue, SIGILL, and queue/account resource limits. Those should not be interpreted as model OOM or learning failure without the actual error. On this site, the working recipes use untyped `--gres=gpu:N`, `aiq` for B200s or `gpuq` for H200s, and avoid `--exclusive` as requested. This records the site's working configuration, not a general claim about cluster policy.

`hpc/jobs/` contains generic job templates. `hpc/generated/` contains rendered experiment configurations, sbatch scripts, plans, submission receipts, and often logs. Site files belong under `hpc/`; checkpoints/results belong under the chosen run root. Rendered configs freeze a run's settings, so changing `base.yaml` does not retroactively rewrite an already generated job.

## 16. Current runs and immutable dependencies

### 16.1 Teacher continuation now running

The researcher has just reported that the resumed teacher is training. The documented recipe is:

| Field | Value |
|---|---|
| Training root | `runs/teacher_fv_no_prop_10hz/seed0/teacher/` |
| W&B | `chronotree`, run `9cb42f0d49592774` |
| Hardware request | Four GPUs in B200 partition `aiq`, 64 CPUs, 128 GB RAM |
| Allocation duration | 72 hours |
| Learning rate | Continue at `1e-5`; no warmup restart |
| Duration in optimizer steps | Open-ended until stop/time limit, not another fixed 100k cap |
| Numbered checkpoint interval | 20,000 global updates |
| Rolling/safe-stop checkpoint | `last.pt`, including graceful stop between numbered intervals |
| Generated plan | `hpc/generated/teacher_fv_no_prop_10hz_resume72h/` |

This resumes model, optimizer, RNG, sampler, and continuation state rather than merely loading weights into a fresh optimizer. “Every 20k” means global step multiples, not exactly 20k after the resume instant. No current Slurm ID, live step count, or new success score was queried while preparing this report.

Continuing the teacher is an empirical test of whether more exposure at the ending rate improves control/history semantics. It does not guarantee a fix; the existing router results show that training longer can also simply overfit.

### 16.2 Frozen branch used for memory experiments

The accepted snapshot is stored on HPC under:

```text
runs/memory_fv_seed0/seed0/teacher/selected.pt
```

Its selection record retains the exact checkpoint hash and accepted evaluation evidence, including execute-eight. The completed encoded cache is:

```text
/hpc/home/phi/rvalperga/data/chronotree-cache/3db7b7ce80926d21659863c3
```

The completed hierarchy cache is:

```text
/hpc/home/phi/rvalperga/data/chronotree-cache/3db7b7ce80926d21659863c3/hierarchy/3567142ed84d1b30b4a744ca
```

The compact run is:

```text
runs/memory_fv_seed0_compact_hierarchy/seed0/compact/
```

The router objective roots are:

```text
runs/memory_fv_seed0_routers/velocity/seed0/router_{tree,flat}/
runs/memory_fv_seed0_routers/final_action_8/seed0/router_{tree,flat}/
```

Advancing the original teacher run **does not upgrade these frozen dependencies automatically**. If a newer teacher is adopted and learned encoder weights differ, a new encoded-cache identity and downstream experimental branch are needed. Reusing old learned tokens while claiming a new encoder would be inconsistent. Existing frozen studies remain valid as studies of their recorded checkpoint.

## 17. Conclusions, hypotheses, and open questions

### 17.1 Conclusions supported by current evidence

- The causal data/simulation pipeline is functioning on inspected real examples; the original “fake repeated reset history” explanation was not supported.
- Better point sampling preserved small-object evidence, but token presence alone did not ensure grounding.
- Old models depended strongly on historical commanded actions. Removing them was a meaningful ablation but did not alone solve targeting.
- Frozen-feature and longer-horizon variants achieved much better natural pickup in inspected scenes, with confounded training/input differences.
- A particular button-hover mechanism is repeated postponement into an unexecuted chunk tail, demonstrated by exact-state execution.
- The accepted teacher uses history in ways that help some real movement sequences. It is not completely history-blind.
- The teacher does not reliably follow semantic cues across the investigated tasks. History sometimes harms performance.
- Router fitting is computationally cheap relative to label generation/encoding. Longer fitting alone did not yield useful held-out gains.
- Gain labels contain repeatable differences, but teacher agreement and expert-action agreement can disagree.
- A learned-router quality/efficiency advantage is still unproven.

### 17.2 Plausible explanations, not established causes

1. **Teacher utility may not align with intended memory semantics.** The target can preserve confident but incorrect behavior or value movement continuity over task-relevant reveal information.
2. **Local gain targets are noisy and context-dependent.** Utility depends on the rest of F and on action noise/time, while the scorer omits those variables. Similar router inputs can correspond to different target gains.
3. **Regression may not optimize useful choice quality.** Signed gains are often small or heavy-tailed; median scaling and Smooth L1 can emphasize magnitudes rather than robust top-choice improvements. Near-chance pair ranking is not itself a full top-k or rollout diagnosis.
4. **Useful cues may already survive coarse summaries.** Refining them then has little marginal effect. Alternatively the policy may not know how to exploit the extra detail. Those mechanisms need different interventions.
5. **The compact policy may have learned a different control strategy.** Its late-chunk button press suggests decoder adaptation and execution horizon matter, independently of memory selection.
6. **Available information or representation may still be insufficient.** Original mean-pooling loses current geometry; current-attention variants address that limitation but did not establish a convincing held-out improvement. Early CLIP layer-1 descriptors are not automatically semantic object representations.
7. **Temporal/geometry shortcuts may dominate.** Clocks, spatial priors, duration, and robot appearance could support some success without the intended semantic memory. Existing count/padding tests do not isolate these cleanly.
8. **Train/rollout covariate shift remains.** Genuine history deletions, permutations, and own-policy errors can all move conditioning away from training. Poor ablation rollouts do not alone identify what information is fundamentally necessary.

### 17.3 Unresolved experimental and measurement gaps

- A controlled learned TREE versus learned FLAT versus UNIFORM/random-reader closed-loop comparison with exact shared checkpoint/protocol and paired scenes.
- Full online timing/storage accounting, including visual extraction, prefix encoding, archive/key updates, selection, transfer, and eight-step action generation.
- A verified retained-best router checkpoint policy; the historical peak at 2k was not saved.
- Semantic-cue tests across multiple natural configurations through actual later choice/outcome, rather than only a single first chunk.
- Matching the latest teacher history-ablation scenes with the compact and learned-reader policies.
- A clean separation of counting from early movement-control failure and duration/history-length cues.
- The deferred terminal/unavailable-action-input mismatch and its real closed-loop impact.
- Production-level stable identity/tie-breaking treatment for exact reader comparisons.
- A recovered matched result for the independently trained current-only model.
- The Perceiver benchmark, budget/latency curves, final test protocol, and training-seed replication.

None of these gaps licenses weakening FLAT, dropping difficult tasks, leaking hidden state, selecting easier test scenes, or claiming a favorable result from an isolated visualization.

## 18. Questions to ask ChatGPT about next experiments

The researcher wants proposals that are diagnostically decisive, practical to implement, and likely to improve the experiment. Many jobs can run in parallel on the available H200/B200 resources, but engineering and interpretation should not expand into an uncontrolled architecture search.

Useful questions, given the evidence above:

1. **What minimum positive semantic-memory result should we require before investing further in routing?** Propose natural matched scenes and an outcome metric that distinguishes target/sequence recall from smoother motor execution.
2. **How should we evaluate the resumed teacher?** Which fixed tasks, checkpoints, execution horizons, and paired controls would tell us whether more training improves history semantics rather than only lower action loss?
3. **Is teacher-agreement gain the right target?** Compare expert-action loss reduction, noise-averaged teacher gains, ranking/top-choice objectives, uncertainty filtering, and limited lookahead. Explain what each changes scientifically and how TREE/FLAT would remain matched.
4. **Can the router observe enough context without reading the whole archive?** Consider a cheap selected-context summary or limited state information versus full-history attention. Include actual update/scoring costs, not only parameter counts.
5. **Can we distinguish “coarse memory already contains the cue” from “policy ignores the cue”?** Specify a valid intervention and expected results under each explanation, without silently deleting unrelated coverage.
6. **What is the fairest way to address action-tail postponement?** Separate execution-horizon tuning from changing training horizon/targets or adding objectives. Account for failures earlier in manipulation and compare all readers with the same controller.
7. **Which small correctness repairs should precede new modeling?** In particular, assess terminal-slot input parity and deterministic tie-breaking, and define meaningful tests without assuming either is the root cause.
8. **What result would justify stopping or redirecting the hierarchy study?** If FLAT/UNIFORM matches TREE or semantic memory remains weak, how can we report an honest useful result rather than optimize indefinitely?

For each proposed experiment, ask for: hypothesis, minimal intervention, exact controls, primary measurement, expected discriminating outcomes, likely cost, and the claim it would or would not support. Avoid recommending the already-declined tiny memorization fit as a necessary prerequisite. Do not treat another long router fit as a promising default without explaining why it would differ from the completed overfitting experiments.

## 19. Source and artifact map

These links are for the local repository. A ChatGPT session receiving only this Markdown file will need the researcher to provide referenced artifacts if it wants to inspect them directly.

| Purpose | Source |
|---|---|
| Scientific contract and staged plan | [PROMPT_CHRONOTREE.md](PROMPT_CHRONOTREE.md) |
| Causality, action timing, calibration | [Data contract](docs/data_contract.md) |
| Model/data decisions and chronology | [Decisions](docs/decisions.md), [status](docs/status.md) |
| Architecture and frozen-vision variant | [Architecture](docs/architecture_diagram.md), [frozen vision](docs/frozen_vision_experiment.md) |
| Current-only model and measured local outcomes | [Current-only experiment](docs/current_only_experiment.md), [four-model inspection](docs/four_model_inspection_20261030.md) |
| Historical command ablation | [Command ablation inspection](docs/command_ablation_inspection.md) |
| Horizon and latest teacher inspections | [H16/H32](docs/h16_h32_inspection.md), [2 November inspection](docs/checkpoint_inspection_20261102.md) |
| Prefix handoffs and input-shift audit | [Expert-prefix handoff](docs/expert_prefix_handoff.md), [reset audit](docs/reset_distribution_audit.md) |
| Count-history experiment | [Count-history diagnostic](docs/count_history_diagnostic.md) |
| Button-tail experiment | [Button-hover diagnostic](docs/button_hover_diagnostic.md) |
| Immutable teacher/cache adoption | [Teacher cache](docs/teacher_cache_hpc.md) |
| Shared compact training and selection | [Compact training](docs/compact_training_hpc.md), [checkpoint selection](docs/compact_checkpoint_selection.md) |
| Router objectives, long fits, architectures | [Router training](docs/router_training_hpc.md), [overfit](docs/router_overfit_hpc.md), [architectures](docs/router_architectures_hpc.md) |
| Gain/cue/history-use investigations | [Gain repeatability and follow-ups](docs/gain_repeatability.md) |
| Current resume and evaluation operations | [HPC workflow](docs/hpc.md), [parallel evaluation](docs/parallel_evaluation.md) |

Selected local viewers and outputs:

- [Three-teacher inspection report](runs/inspection/eval_20261102T120621Z_analysis/index.html): prediction errors, relocated-cube point clouds, rollouts, and expert-prefix videos.
- [Count-history corrected outputs](runs/inspection/count_probe_teacher_student_15k_v2/): natural and controlled history predictions.
- [Button-hover outputs](runs/inspection/button_83k_20261104/): saved chunks, exact-state continuations, and matched execution-horizon probes.
- [Gain-repeatability viewer](runs/inspection/gain_repeatability_20261109/index.html): high/low candidates, repeatability, and point-cloud context.
- [Latest history-use gallery](runs/inspection/history_use_20261109/index.html): FULL/initial-erased/execution-erased/CURRENT comparisons.
- [Recent-history Swing gallery](runs/inspection/history_recent_swing_20261109/index.html).
- [Permuted-history Swing gallery](runs/inspection/history_permuted_swing_20261109/index.html).

Some rendered videos begin with the supplied expert demonstration. The viewers' “Play policy execution” controls skip that segment; otherwise an expert demonstration can be mistaken for successful policy execution.

Reusable entry points include `scripts/inspect_checkpoints.py`, `hpc/evaluate.py`, `hpc/cache_teacher.py`, `hpc/cache_hierarchy.py`, `hpc/train_routers.py`, and `hpc/router_architectures.py`. Generated data, weights, and media remain ignored. Existing tests cover causal prefixes, exact TREE/FLAT coverage, input omission, cached/streaming parity, deterministic resume, and evaluation accounting. This report is a synthesis of those records, not a claim that every historical test or full dataset was rerun today.

The central decision for the next phase is whether to improve the teacher's task-relevant history use, improve the router's utility target/generalization, or first finish a tightly controlled online reader comparison. The evidence currently supports investigating those separately rather than assuming a single architecture change will solve all three.
