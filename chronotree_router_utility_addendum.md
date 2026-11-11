# Addendum: jointly test student training and refinement-based routing

**Date:** 10 November 2026  
**Revision:** 2 — student training and router design are now explicit experimental factors; amended at the user's request to 100k primary student updates with frequent checkpoints  
**Audience:** Codex working in `Ricvalp/chronotree`  
**Companion report:** `PROJECT_HANDOFF_FOR_CHATGPT.md`, dated 10 November 2026  
**Report/code reference:** `7e1d169e3bbfbffb77dba724cf6b1cb4a7b73184`

**11 November teacher-selection amendment (explicit user instruction):** The new
primary study uses the accepted 140,000-step teacher, SHA256
`d17dee6cfe70caae8ff8d1013b0fe50aee01d1cd816feda4a25f0f4d388c0d34`.
Its completed FULL validation obtained 25/64 successes (39.0625%) with eight
executed targets. Rebuild only its learned encoded-token and hierarchy caches;
reuse prepared observations and frozen CLIP features. Train S_STD, S_STAGE and
S_PACKET from this identical teacher for 100k updates each. Keep the old S0 and
its caches/results as a separate historical study; do not include S0 in the new
matched matrix or feed it the new encodings. A frozen teacher-weight `S_INIT`
compact-context smoke check is diagnostic only, with zero compact updates.
New generated-action/prefix checks use eight executed targets, matching the
accepted rollout. These changes supersede the old fixed dependencies, four-arm
matrix, no-cache-regeneration restriction and execute-ten setting below **for
the new study only**. All other causal, matching, data-split and numerical
requirements remain. See [the 140k workflow](docs/refinement_utility_teacher140k.md)
and `configs/inspection/refinement_utility_teacher140k.yaml`.

## 1. Task and intended outcome

Implement a bounded, reproducible **student-training × routing experiment**. Determine whether current router labels are weak because they measure isolated, immediate refinements in already informative contexts, because the student has not learned to exploit useful allocations, or both.

The immediate research question is:

> Which student-training distributions produce useful policies under the routing strategies we intend to deploy, and do coordinated allocations reveal value that immediate-refinement labels conceal?

Preserve the production 90k compact policy as an immutable reference. Train a small, matched set of additional students using the accepted teacher and existing immutable encoded/hierarchy caches. Freeze each student for its utility measurements. Treat the student and its context distribution as part of the retrieval method, rather than treating one existing student's utility landscape as an intrinsic property of the archive.

Implement the measurement pipeline, student configurations and job launchers, verify them on bounded real data, and provide executable jobs plus a report for measurements actually completed. Follow the user's existing local/HPC launch instructions: this document specifies the jobs to include and does not override a prepare-only instruction. If checkpoints, caches or cluster access are unavailable, finish the code/configurations and exact launch commands, identify missing dependencies, and do not substitute another checkpoint or invent results.

The required first batch is the existing reference plus three matched student jobs, followed by the same frozen diagnostic for each. A bounded router fitting/adaptation stage is specified separately. **Do not require positive refinement gains from the existing student before running the new student jobs.** Whether training can create better use of available information is itself under test.

Deliver the first-batch report independently of the optional later stages. Prepared configurations for an unactivated branch are sufficient; do not delay reporting the primary matrix until every conditional idea has been tried.

### Scope boundaries

- The accepted teacher, visual/task encoders and cached representations stay frozen. New student decoder/adapter training is explicitly in scope; additional teacher training belongs to its existing separate branch.
- No new demonstrations, DAgger, recovery collection, or simulator interaction for the primary offline labels.
- No new SSM, permanent keep/forget memory, differentiable gate system, PPO, or architecture sweep.
- No full-cache regeneration or migration to the teacher currently continuing training.
- Router fitting follows the bounded evidence-driven stage below; no open-ended router sweep or repeated adaptation cycles.
- Attention-derived selection and auxiliary semantic training are separate follow-ups, outside this initial implementation.
- The 256-state panel is for measurement. Student jobs use the full designated training partition, not that panel or the previously declined tiny memorization experiment.

## 2. Hypotheses and discriminating evidence

| Hypothesis | What would support it | What does not establish it |
|---|---|---|
| H1: immediate utility is sparse | Most measured first-split gains are practically small or unstable, while a minority reproduce on independent draws | A histogram of single-draw gains or 50–52% learned ranking |
| H2: background redundancy suppresses utility | The same candidate operation helps substantially more with a coarse background than with a richer one | Comparing different candidate nodes at different budgets |
| H3: refinement has delayed value | Selecting a subtree by its completed packet gives better final-budget outcomes than selecting it by its first-split gain | An eight-token packet having a larger raw gain than a one-token split |
| H4: useful detail is complementary | A controlled pair of allocations has a positive interaction beyond their separate substitutions | Two larger contexts outperforming a smaller context |
| H5: the frozen decoder is insensitive to useful resolution | Broadly different, carefully chosen allocations fail to improve the appropriate task decision | One scene's unchanged initial action chunk |
| H6: teacher agreement is misaligned | Reproducible teacher-agreement preferences conflict with expert-action-loss preferences | A single noisy sign disagreement |
| H7: student training changes usable refinement value | A matched student trained on router-compatible contexts improves absolute expert loss/task choices and the benefit of appropriate allocations | Larger gains caused by a worse coarse baseline, or comparing unmatched training durations |
| H8: student and router specialize to one another | Matched crossed evaluations or a controlled adaptation round reveal an interaction between student training and routing strategy | Comparing only old-student/old-router with new-student/new-router |

Do not assume H1 is true. The report establishes noisy labels and weak held-out prediction, not the prevalence of genuinely useful refinements.

For a common background and equal-cost candidates, ranking `L(before) - L(after_i)` is identical to ranking `-L(after_i)`. Renaming the score or applying a sharper softmax cannot create missing information. The substantive changes here are the background and the unit of allocation.

## 3. Immutable dependencies and invariants

Resolve paths from the current repository/manifests rather than copying workstation paths blindly. Verify these recorded identities:

| Dependency | Recorded identity |
|---|---|
| Teacher | 104,110 updates; SHA256 `eb57ccd4c131888a5f5d6225a3a0cbe5cd14b3afc5c6ff0a1a17ee319cd70918` |
| Compact policy | 90,000 updates; SHA256 `4fe22fd00fbc64dc740ba046d6ce1e58ade2a70580f8792e152477caad316427` |
| Encoded cache | `3db7b7ce80926d21659863c3` |
| Hierarchy cache | `3567142ed84d1b30b4a744ca` under the encoded cache |
| Action model | H16, nominal 10 Hz targets, eight Euler integration steps |
| Executed-prefix diagnostic | First ten targets, matching the production router investigation |

During every utility measurement, the teacher and evaluated student must be in evaluation mode with gradients disabled. During designated student jobs, only the intended decoder/adapter/clock parameters train. Record each student's initialization, parent checkpoint and resulting hash alongside code, configuration and precision. A current implementation revision may differ from the report reference; preserve that distinction.

Maintain all existing causal and representation contracts:

1. Current visual observation appears exactly once and is unchanged between alternatives.
2. Historical execution evidence is strictly earlier than the current observation.
3. Conditioning and execution banks stay separate. No source is silently dropped.
4. Partial blocks use the cached observed prefix, never the future complete block.
5. TREE frontiers partition every available leaf exactly once, without ancestor/descendant overlap.
6. FLAT selected leaves and residual summaries partition exactly the same evidence.
7. Residuals count toward the historical token budget; current/task/clock tokens remain outside that budget.
8. Preserve metadata, aggregation, unit leaf mass, normalization, masks and `log(count)` attention bias.
9. Preserve the no-proprioception/no-previous-command input convention.
10. Gain-blind diagnostic candidate construction may use observed geometry, topology, times and source identity, but no hidden object IDs, simulator masks, target poses, expert future actions or measured model gains. Later learned routing may use its trained observable-input scores, with separately recorded provenance.

Expert actions are legitimate offline loss targets. They must not influence which main-panel candidates are generated or enter retrieval queries.

The terminal/unavailable-action-input mismatch is a separate known issue. **Do not silently repair it in this study.** Use the checkpoint's established FM input convention for offline measurements and its normal inference convention for generated chunks. Keep target masks valid. Record the convention and isolate terminal versus structurally unavailable dimensions in reporting.

## 4. Repository integration

Inspect and reuse existing code before introducing parallel implementations. At the report revision, relevant entry points include:

- `src/stmem/inspection/gain_repeatability.py`: frozen-policy loading, provenance, paired draws and selected generated paths.
- `src/stmem/inspection/gain_statistics.py`: discovery/confirmation separation, ties and descriptive noise statistics.
- `src/stmem/inspection/cue_frontiers.py`: exact-coverage checks and protected/prioritized frontiers.
- `src/stmem/training/label_objectives.py`: batched cached contexts and teacher/compact evaluations.
- `src/stmem/training/label_data.py`, `label_experiment.py`: indexed data access, partitions and provenance.
- `src/stmem/memory/frontier.py`, `flat.py`, `hierarchy.py`: valid operations and actual token counts.
- `src/stmem/models/action_flow.py`: `flow_sample`, `masked_mse` and integration.
- `hpc/train_compact.py`, `src/stmem/train.py`: Stage P initialization, trainable parameters, schedules and immutable run ownership.
- `src/stmem/training/packed_context.py`, `cached_loader.py`, `cached_policy.py`: context-family sampling, indexed assembly and gradients through the shared adapter.
- The corresponding inspection, reader, objective and hierarchy tests.

Important implementation cautions from the inspected code:

- Existing `evaluate_batch` returns teacher-agreement gains. Extend or factor its prediction/loss computation to obtain **raw expert losses**, not merely rename its existing outputs.
- `flow_sample` returns the correct masked expert velocity target. Reuse it rather than rebuilding a subtly different interpolation convention.
- Existing gain paths require strictly positive token increments. Same-budget comparisons and no-op controls need separate semantics; never divide their differences by a fake cost.
- The local `gain_repeatability.py` contains VideoUnmask-specific setup and physical-action/FK calls. Generalize those paths for the actual task instead of copying its hardcoded task name.
- Existing statistics describe repeated noise within a fixed scene. Add episode-aware aggregation for this larger study.

Suggested additions, subject to repository conventions:

- `src/stmem/inspection/refinement_utility.py` for the experiment runner.
- `src/stmem/inspection/refinement_candidates.py` for deterministic constructions.
- `configs/inspection/refinement_utility_pilot.yaml`.
- A small student-study launcher/configuration family reusing the existing compact-training infrastructure.
- Focused tests and `docs/refinement_utility_pilot.md` describing commands/results.

These are proposed names, not claims that these files already exist. Avoid a broad refactor.

## 5. Sampling plan

### Main panel

Default to **256 decision states** spanning all 16 tasks:

- Eight distinct released training episodes per task.
- Two separated decision states per episode.
- Prefer four episodes from each of the existing router fitting/validation partitions, giving 128 decisions per partition.
- Preserve the existing episode partition exactly. These policies may already have trained on both partitions; describe the second partition as held out from router fitting, not unseen by the entire system.
- Distribute decision states across early/middle/late execution, approximately balanced within each task. Use eligible recorded decision indices and deterministic seeded sampling, not gain-based selection.
- Do not sample hundreds of adjacent states from a few trajectories.

The primary panel uses states with a complete H16 target horizon for the task's available action dimensions. This limits the terminal-padding nuisance without excluding stick tasks merely because their gripper is structurally unavailable. Record excluded terminal states and the structural action mask.

Preselect the 256 states before testing packet/background eligibility. If insufficient eligible paired-background states exist, report the shortfall. Keep a separate single-budget stratum when possible, and retain packet-ineligible states for the unrestricted immediate-only audit whenever that audit is feasible. Do not replace hard cases after seeing their losses. Reset states with no execution history can be structurally ineligible; this is not a policy failure. Legitimate conditioning history may still make a reset state eligible.

The primary panel deliberately balances tasks, phases and router partitions. Its gain prevalence describes **this diagnostic sampling distribution**, not an automatically unbiased estimate over all training timesteps.

Create the complete state manifest before evaluating policy losses. Record available leaf count, root count, task, episode, source-local decision index, temporal stratum, partition, action-mask summary and eligibility reasons.

### Data boundaries for new training

Keep original router fitting/validation episode tags intact. Exclude **all episodes represented in the diagnostic panel** from new student optimization and router-label fitting. Also exclude the full existing router-validation partition from new training. The remaining router-fitting episodes form the common student training pool. Under an exact 80/20 split with four diagnostic episodes drawn from the fitting side, this leaves 76 episodes/task; derive the actual count from manifests rather than assuming it.

Use the fitting-tagged half of the diagnostic panel for development and the validation-tagged half for a final frozen-protocol comparison. Both halves are held out from new student updates. The inherited teacher and S0 may already have seen these episodes; describe this accurately as held out from the **new training**, not entirely unseen by the system. Do not use the final half to select a checkpoint, sampler or effect threshold and still call it independent confirmation. If already inspected during design, label it exploratory and reserve further untouched validation episodes for the next comparison.

Independent flow-noise banks separate within-state selection from evaluation; they do not substitute for the episode boundary. Official benchmark validation provides a separate closed-loop check. Official test configurations remain untouched.

### Smoke panel

Use 8–16 states across at least three tasks and multiple episodes, including conditioning/execution banks and one causal partial block. If available, include a task with a structurally unavailable gripper. The smoke run checks correctness and measures throughput; it is not used to choose research conclusions or effect thresholds.

## 6. Student training must match the intended retrieval process

### 6.1 What the existing student actually learned

Verify the recorded recipe before changing it. Stage P initialized the decoder and context adapter from the accepted teacher, froze visual/text encoders, and optimized expert FM loss. Its family proportions were 45% TREE, 45% FLAT, 5% UNIFORM and 5% CURRENT. Nominal budgets cycled through 64/128/256; with probability 0.5 on non-CURRENT examples, the actual budget was sampled uniformly from the root-count minimum to the nominal budget.

Random TREE uses persistent random node priorities, not a uniform distribution over valid frontiers. It can already produce deep, concentrated refinements. Therefore neither small contexts nor coherent detail should be described as completely unseen. The question is their exposure frequency, joint availability and relevance to action decisions. No explicit FULL-history rehearsal family exists for long histories.

Before training, sample 4,096 context manifests without policy evaluation and summarize actual token counts, depths, leaf fractions, source coverage, concentration and contiguous refinement lengths. Use the same source states to compare the proposed samplers. This cheap audit checks whether the intended distributions actually differ.

### 6.2 Required jobs and initialization

The user considers student training affordable and permits multiple students. Prefer a clean comparison of training recipes over treating a short repair of S0 as decisive. The **default is three new teacher-initialized Stage P jobs**, sharing the accepted teacher's weights and frozen caches. These are compact-policy training jobs, not new visual-encoder or teacher runs.

| ID | Initialization and training | Purpose |
|---|---|---|
| S0 | Existing 90k compact checkpoint, no updates | Preserve existing evidence and measure the current utility landscape |
| S_STD | Accepted teacher; original context sampler | Matched baseline for the new training pool and schedule |
| S_STAGE | Same initialization; router-stage exposure with random single-move allocation | Test exposure to the budgets and intermediate states of the contemplated router |
| S_PACKET | Same initialization and exposure schedule as S_STAGE; coherent packet allocation | Isolate allocation structure from budget/exposure changes |

At the user's subsequent request, extend each primary student to **100,000 updates**. Keep fresh optimizers, LR `1e-4`, 1,000-step warmup, the same cosine form reaching `1e-5` at update 100,000, effective batch 128, and the same precision/distributed setup. Save rolling recovery `last.pt` every 1,000 updates and immutable numbered snapshots every 5,000 updates, including 100,000. The primary comparison uses the **predeclared 100k endpoints**. Earlier declared snapshots, including 15k and 30k, support development/learning-curve checks on the fitting partition only, not retrospective selection on final confirmation. This is a requested schedule amendment, not a claim of measured convergence.

Pin `student_training: {steps: 100000, checkpoint_every: 1000, snapshot_every: 5000}` in the protocol, and record the primary endpoint and development snapshot list in each student configuration. Existing immutable 30k study manifests/checkpoints retain their original endpoint. Prepare a fresh named study when changing a pinned schedule or source revision; do not overwrite queued plans or reinterpret earlier checkpoint provenance.

Match initialization, episode/decision ordering, action-noise/time streams, effective batch, updates and optimizer settings across the three jobs. Context construction uses its own random stream so different numbers of selection operations do not change the action-noise stream. Match underlying demonstration exposure, not GPU hours alone; report actual compute separately. S0 has a different optimization/data history and is a reference, not the controlled comparator for recipe effects.

These three jobs are a **paired seed-0 experiment**, not three independent training replications. Episode bootstrap intervals quantify variation over evaluated episodes, not uncertainty across training seeds. Replicate the selected central comparison later before making broad robustness claims.

Provide a lower-cost `continuation` mode if resource planning calls for it: all three arms start from the same immutable S0 weights, use identical optimizer reset/restoration policy, a predeclared common update budget and LR, and the same new training pool. Do not mix teacher-initialized and S0-initialized arms in the main causal comparison. Do not automatically launch both initialization modes. A failed continuation only establishes failure to adapt from S0 under that schedule.

### 6.3 The routers for which we are preparing the students

Define two concrete controller families:

- **Local controller:** repeatedly choose one valid positive-cost refinement/promotion until the historical budget is filled.
- **Packet controller:** repeatedly choose an anchor region and a deterministic coherent completion of up to eight positive-cost moves, then re-score the resulting context. This permits useful descendants to influence a branch choice. It does not imply that a learned completion policy or PPO is necessary.

Both begin from the same valid coarse representation of the causal archive for their reader and retain exact coverage. TREE begins from its causal forest roots; FLAT begins from the existing coarse residual representation. Roots/residuals can differ between readers, so match budgets and evidence while reporting minimum feasible counts. Current observations remain outside the historical allocation process.

The later frozen panel around B56/B120 tests the final eight-token allocation. It is useful but does **not** establish early roots-to-budget credit. Training below includes complete construction trajectories, and the optional learned-router stage measures complete reachable routes separately.

### 6.4 Matched S_STAGE and S_PACKET context distributions

Preserve the original 45/45/5/5 family proportions. For half of TREE/FLAT examples, use the original sampler unchanged. For the other half, use a router-construction template. UNIFORM and CURRENT examples remain unchanged. These proportions are explicit pilot defaults; avoid tuning them on final confirmation.

For each template, draw a common target budget: B128 with probability 0.75, B64 with probability 0.25. Create the same coarse starting context for S_STAGE and S_PACKET. Draw the actual training prefix count independently of allocation: with probability 0.5 use the completed target count; otherwise sample a valid earlier integer count between the initial count and target minus one. A state with too few leaves uses the same attainable count in both arms and is tagged as a saturated short history. If initial and attainable counts coincide, use that context without sampling an empty interval. If the minimum representation exceeds the chosen budget, use the common ordinary-context fallback with a documented feasible legacy budget. States exceeding every configured feasible budget must be reported consistently for all arms, not silently compressed by dropping roots.

Construct a complete gain-blind route and extract the requested prefix:

- **S_STAGE:** apply the existing random positive-cost single-move rule across currently available candidates.
- **S_PACKET:** select a feasible region from observed topology/source/time information, execute its deterministic coarse-to-fine packet, then select the next region from the resulting context. Within TREE, complete an anchor by mass-based descendant splitting with stable ties. Within FLAT, use ordered observed space-time leaf regions and exact residual updates as in the diagnostic construction. All packet operations must have actual cost +1.

Use this concrete initial region sampler: choose uniformly among source/modality groups with feasible regions, then uniformly among that group's eligible region IDs in a stable order. TREE regions are currently active anchors; FLAT regions are canonical observed hierarchy subtrees containing the required unselected leaves, with duplicate descendant-support sets removed. FLAT may use this topology for proposal generation while retaining FLAT record/residual semantics; report its indexing cost. Recompute eligibility after each packet. Record the sampler seed and eligible pool sizes. More elaborate time/depth stratification is a separate configured variant, not an undocumented implementation choice.

At each packet decision, choose a common candidate cost `q = min(8, remaining budget)`. If no valid region supports q, reduce q deterministically to the largest feasible positive cost. Every candidate compared at that decision must then cost exactly q. Record q and apply the same rule during training, labeling and deployment. Do not rank variable-cost completions by raw gain as if their resource use were equal. Prefix sampling can stop inside a packet; save both boundary and within-packet exposure statistics. Do not fill a packet with duplicate or zero-cost records. The same packet constructor must be reused by training, diagnostics and the corresponding deployed controller wherever their starting frontiers are compatible.

S_STAGE and S_PACKET share family, source state, starting context, target budget, actual prefix count and feasible-state/fallback decisions. Only the allocation path differs. If a valid packet route cannot be built for a template, both arms use the identical precomputed ordinary-context fallback. A common capability prepass may compute eligibility from archive structure; never condition it on model losses. Repeated source-local frames remain distinct evidence when genuinely recorded; no fabricated waiting history.

Do not choose regions using target-object identities, future expert commands, hidden masks or measured gains. Conditioning-bank and execution-bank regions should both receive exposure when available; report their realized frequencies. Do not silently replace the region sampler with an oracle because random regions are usually irrelevant. Task-specific privileged regions can be used in a separately labeled diagnostic capability probe only.

The completed routes make this a test of the intended multistage controller distribution. A simpler last-eight-moves implementation is an acceptable explicitly labeled smoke prototype, but it does not fulfill this training specification or justify claims about early routing.

### 6.5 Supervision and protected capabilities

Keep expert FM as the common primary objective and train the existing decoder, read adapter and clock parameters. Preserve frozen encoders and cache identities. The student receives one sampled context per training example in the primary jobs, keeping model-example counts matched. If a future paired-context objective uses two forwards, account for that extra training exposure rather than calling it the same budget.

Do not add a loss requiring fine contexts to outperform coarse contexts, nor force their predictions to agree. Either can distort the measured utility: the former can be satisfied by damaging coarse predictions; the latter can suppress a legitimate difference when the coarse context is ambiguous. Supervise correct actions, then measure whether extra detail helps.

Log task, phase, history length, context family, budget and flow-time-stratified losses. Overall FM loss can conceal rare target/stop/order decisions. Include generated-action checks from pure noise; an FM input near the expert-action endpoint already contains much of the target chunk, so loss averaged over flow time is not itself a semantic-memory test. Preserve the original flow-time distribution in the main comparison.

Do not change decision-state sampling, execution horizon or mask behavior in only one primary arm. If later evidence supports oversampling genuinely memory-dependent decisions, create a separate matched sampling ablation with a training-only phase definition and a representative evaluation distribution. Neither high action speed nor high loss alone identifies a memory-dependent decision.

### 6.6 Common measurement matrix and capability reference

Evaluate S0, S_STD, S_STAGE and S_PACKET with the exact same gain-blind state/context/candidate manifests and the same discovery/confirmation draws. Freeze every evaluated checkpoint. Reuse teacher reference predictions, but recompute compact predictions and utilities for every student.

Every student is evaluated with ordinary controls, immediate-selected completed packets and packet-selected completed packets. The last two use measured offline losses to select candidates and must be labeled **diagnostic selections, not deployed learned routers**. Keep the comparison crossed: do not report only S_STAGE with local choices and S_PACKET with packet choices.

Primary training contrasts are S_STAGE versus S_STD (router-state exposure) and S_PACKET versus S_STAGE (allocation structure). Report absolute expert loss and task behavior alongside each student's own selected-versus-control improvement. Use the same final budgets and same control contexts across students. A larger refinement gain is not an improvement if it comes from a worse background.

Add a bounded **32-state capability panel**, chosen before evaluating new students and spanning episodes/tasks with target, destination or stop/count decisions. Reuse main-panel states when suitable; otherwise predeclare additional episodes and add them to training exclusions. Compare teacher FULL, each student on ordinary B128, detailed B128 allocations, B256, and all cached leaves where feasible. Full-leaf student inputs diagnose retention/distribution effects, not a guaranteed upper bound: long FULL contexts can be outside compact training support. Report omissions rather than truncate history. At teacher initialization, test cached full-leaf parity with the teacher on a small subset before interpreting later drift.

If both teacher and students fail the relevant choice even with informative evidence, do not attribute the failure exclusively to the router. If coarse contexts already yield correct choices, a flat utility landscape is compatible with successful compression.

Run the full frozen panel at the predeclared primary endpoints. Development subsets may monitor earlier declared 5k snapshots, including 15k and 30k, without repeatedly consuming final confirmation. Fit no new student directly to the 256 diagnostic states or their measured winning candidates.

### 6.7 Conditional additional student jobs

Implement configurations for the following, but activate them only for the corresponding documented finding. These are alternatives to an uncontrolled sweep.

**Richer-history rehearsal:** if informative detailed/full inputs demonstrably support better choices, but compact training appears to lose this ability, branch from one selected compact checkpoint into a matched pair. Both use the same 10k-update continuation at LR `1e-5`; one keeps its sampler, while the other replaces 10% of non-CURRENT contexts with all available cached leaves. Keep expert targets, exposure counts and frozen encoders unchanged. Use length-aware microbatches/accumulation to preserve the effective batch. Report the extra compute. Never silently truncate FULL or drop long examples; if this is infeasible, configure a common documented higher-budget alternative and describe it as such. The ratio and duration are pilot defaults. Rehearsal cannot reconstruct information absent from frozen features and does not require teacher-output distillation.

**One router-conditioned adaptation cycle:** after a learned selector has usable evidence, branch the selected student into two matched 10k-update continuations at LR `1e-5`. Both retain 50% of their existing context sampler. In the other half, one uses gain-blind routes and the other uses frozen-router-produced routes, with matched families, actual budget/prefix schedules and episode exposure. Build each new route from the current causal archive; include earlier routing states if the controller operates from coarse roots. Train the action policy on expert FM only, then freeze it and regenerate utility labels before any router refresh. If only terminal contexts are used in a simpler implementation, label that restriction and do not claim intermediate routing-state adaptation.

Both paired continuations start from the identical checkpoint and optimizer policy. A trained router may be allowed into this stage on strong development evidence even if a small closed-loop score is inconclusive, but report that choice; do not use test outcomes. Stop after one such cycle in this addendum. No PPO, environment reward training, permanent forgetting or encoder unfreezing is required.

### 6.8 Bounded learned-router stage

After frozen comparisons, select at most two student variants using development evidence and absolute policy quality. Preserve one matched comparator when testing an allocation-training claim. Generate separate fitting labels from the designated training pool, not diagnostic episodes. Start with a bounded pilot of 4,096 fitting states per selected student, shared across objectives/readers where feasible, with separate development evaluation. Measure actual throughput and confirmation noise before increasing that count.

Fit lightweight scorers for immediate and completed-packet utility under each selected frozen student. Reuse the existing scorer family and training conventions; keep capacity, observable input features and fit budgets matched. Scores may condition on current input, selected-context summaries, remaining budget, candidate metadata and feasible completion cost. No expert action/noise/flow time or offline gain is a deployment feature. Count the cost of constructing packet features, examining descendants and maintaining the selected-context summary.

Compare two different questions explicitly:

1. **Credit target:** on the same candidate packet pool, select by predicted first-move utility or predicted completed-packet utility, then execute the same completed packet. This isolates the supervisory target at equal allocation cost.
2. **Controller:** one-step replanning versus packet commitments from the same valid starting context and final budget. This also changes the sequence of available choices and must be reported as a controller comparison.

For full roots-to-budget routes, collect earlier reachable frontiers as well as late ones, using a recorded mixture of gain-blind trajectories and refreshed model routes. Apply existing gain-blind exploration for coverage. Last-eight-token labels alone do not establish long-range credit. Completed local packets provide bounded lookahead; claims that a terminal reward credits every early decision require a separate full-route return experiment, which is not assumed here.

On a bounded held-out set, evaluate routers both with the student that produced their labels and, when compatible, with the other selected student. Report the crossed student × router table and fixed random/UNIFORM controls. A gain regressor's numerical scale may change across students; report selection performance rather than treating its calibration as transferable. Do not pool utilities from different students unless student identity is explicitly modeled and that is a separate experiment.

For matched closed-loop screening, use the same predeclared official validation episodes, seeds, B128, eight flow steps and execute-ten setting for all arms. A bounded initial default is two episodes/task, with one shared student and multiple readers evaluated under identical conditions. Run fixed controls as well as learned readers; report episode-paired uncertainty and all failure types. These are development results, not final benchmark claims. Expand only to resolve a stated uncertainty, and keep official test evaluation for the final frozen protocol.

## 7. Context construction and the three primary experiment families

### 7.1 Shared parameters

Initial final historical budgets are `B in {64, 128}`. Initial packet cost is `m = 8` additional historical records. Each background therefore has `b = B - m` records, initially 56 and 120.

The compact model was trained with B64/B128/B256 and intermediate frontiers. Audit its sampler to state how well backgrounds of size 56/120 and packet prefixes are represented. Do not claim that every such context is known to be well trained. Roots-only or much smaller contexts are out of the default scope.

For each eligible state and reader, generate up to eight coherent packet candidates. Use the same number where feasible, otherwise retain the actual count and report eligibility. Require at least two distinct candidates for ranking. Do not duplicate candidates to reach eight.

Each candidate has a deterministic first move and a deterministic completion to exactly `m` added tokens. Preserve the sequence and record prefix contexts at increments 1, 2, 4 and 8. Prefixes 1 and 8 are required for the full panel; evaluating 2 and 4 can be restricted to the generated-action subset if cost requires.

### 7.2 TREE: protected anchors and paired backgrounds

Construct a deterministic exact-coverage anchor frontier at size 56. Select a gain-blind candidate pool of splittable frontier nodes, using seeded sampling stratified over available sources and descendant-count ranges. Each candidate must contain at least nine leaves to permit eight splits within its subtree.

Build the richer background of size 120 by refining **only outside the candidate anchor subtrees**. The same anchors must remain present and unchanged. This gives a genuinely paired coarse/rich comparison:

- Same candidate node ID and descendant support.
- Same current input, instruction, clocks and causal archive.
- Same first split and same internal packet completion.
- Different detail only in the rest of the frontier.

Check feasibility before evaluating losses. Protecting a collection of anchors can leave insufficient capacity for 120 records. A bounded, deterministic procedure may reduce/reselect the pool using topology alone. Record this procedure and all reasons for failure. If a common protected pool cannot be constructed, retain the state in a separately labeled single-background analysis.

Do not use a different protected background for each candidate and then call the resulting scores ordinary within-frontier ranking labels. Those candidates would have different conditioning contexts.

For a candidate anchor `n`, complete its packet by splitting valid descendants inside `n` using a fixed mass-based coarse-to-fine rule, with stable tie-breaking. The first operation splits `n`. All later packet operations remain inside its descendant support. Each binary split adds one record, so eight valid splits must produce exactly `B` records from the common `B-8` background.

### 7.3 FLAT: matched residual-preserving analogue

Use the same decision states, B values, packet cost, loss targets and noise budget.

Construct FLAT backgrounds with 56 and 120 records, including all residual summaries. The richer background retains the coarse background's selected leaves and promotes additional leaves outside the protected candidate regions. Recompute residuals with the existing exact aggregation implementation.

Define coherent packet regions using the observed space-time supports of the TREE anchors where feasible. This use of hierarchy regions is only a diagnostic candidate-definition device; it is not a claim about production FLAT search complexity or superiority of either reader.

Within each region, preselect a deterministic ordered sequence of eight previously unselected leaves, using geometry/time-balanced coverage and stable ties. Protect the **entire region** from background promotions where possible, so its detailed support remains unchanged across background levels. If only a weaker construction is feasible, label it separately and do not include it in the strongest paired-background analysis.

Every promotion in the packet must have actual cost +1. Preserve an unselected leaf in each affected residual group throughout the packet. Handle singleton/last-leaf normalization explicitly; exclude zero-cost moves from this positive-cost experiment. Use actual `len(records)` and verified coverage, never infer cost solely from the number of selected IDs.

TREE and FLAT can have different feasible candidate pools. Present the full within-reader findings and a matched-state subset; do not silently drop different states and compare aggregate rates as if they were paired.

### 7.4 Family A: immediate refinement utility

**A1: packet-conditioned first moves.**

For each background `F_b` and candidate packet `i`, evaluate its first-move context `F_i^(1)`:

`gain_1(i) = L(F_b) - L(F_i^(1))`.

This measures immediate signal among packet-eligible anchors. Its final context has `b+1` records. Preserve raw signed effects and noise variation. Because these anchors must support eight refinements, this panel alone cannot establish how common weak gains are among ordinary router candidates.

**A2: unrestricted immediate-only audit.**

For each of the same 256 preselected states and both readers, construct one ordinary background with 127 historical records and sample up to four unique valid single moves, each adding exactly one record to reach B128. Use the existing production-label background/candidate sampling convention where feasible; record the exact sampler and any necessary adaptation to B127. Candidate generation remains gain-blind.

- Do not impose the nine-leaf packet condition or protect a packet region. TREE candidates may be any valid splittable frontier node; FLAT candidates may be any valid positive-cost leaf promotion with correct residual accounting.
- Include packet-ineligible states whenever a valid B127 background and at least one +1 move exist. Do not condition this audit on success of the paired-background construction.
- Retain states with just one valid move for gain distributions; ranking comparisons require at least two.
- Use the same four discovery and four confirmation draws and shared teacher references as the packet panel. Evaluate the base context once per draw and all available candidates.
- Report this audit separately from A1. Its prevalence estimate applies to the audited budget and documented sampling distribution, not automatically to all nodes or deployment decisions.

This small audit directly tests whether ordinary single-refinement labels are mostly negligible, without selecting only regions capable of supporting a larger packet.

### 7.5 Family B: the same operation under coarse versus rich backgrounds

Evaluate each protected first move and packet under both backgrounds. Pair results by the exact operation/support identity, rather than by candidate rank or array position alone.

Compare `gain_1_coarse - gain_1_rich`, and separately the corresponding packet-gain difference. Different baselines can have different absolute losses, so report raw differences and carefully labeled relative effects.

This tests context dependence of marginal utility. A larger coarse-background gain is consistent with redundancy; it does not uniquely prove semantic redundancy, because policy response to context resolution may also contribute.

### 7.6 Family C: packet utility and delayed credit at the same final budget

Evaluate every completed coherent packet `F_i^(8)`. All have exactly `B` records.

Generate two gain-blind controls from the same background:

1. A deterministic UNIFORM/mass-based continuation for eight positive-cost moves.
2. A seeded dispersed continuation for eight positive-cost moves, spreading allocations across available regions when feasible.

For FLAT, the corresponding controls use deterministic time/geometry-balanced promotions and seeded dispersed promotions, with residuals retained. Do not present an adapted FLAT control as the production TREE UNIFORM reader.

Controls may have different support from coherent packets, but must have the same added cost and final budget. Record the number of distinct regions touched and flag cases where topology prevents a meaningfully dispersed control.

Use discovery draws to select:

- `i_immediate`: best first-move expert gain.
- `i_packet`: best completed-packet expert gain.
- `i_teacher`: best completed-packet teacher-agreement gain.
- A discovery-low packet for diagnostic contrast.

On independent confirmation draws, evaluate **the completed packets** selected by all three rules. Compare them with the fixed controls and the mean of all coherent candidates, which represents uniform random choice over this candidate pool.

The crucial delayed-credit contrast is:

`L_confirm(F_i_immediate^(8)) - L_confirm(F_i_packet^(8))`.

Both operands use exactly `B` records. A reproducible positive difference supports evaluating a branch by its downstream allocation value rather than its immediate split effect.

Also compare packet-selected loss with the dispersed/UNIFORM controls at the same B. A packet having eight times the raw gain of a single split is not evidence by itself. Report gain per added token descriptively, but do not use that normalization as a substitute for same-budget controls.

## 8. Primary and secondary loss measurements

### Primary: expert flow-matching loss

For each state and draw, obtain normalized action `a`, validity mask `M`, Gaussian noise `epsilon`, flow time `u`, interpolated action `x_u` and velocity target using the existing `flow_sample`.

On valid coordinates:

`x_u = (1-u) epsilon + u a`  
`v_target = a - epsilon`  
`L_expert(F) = maskedMSE(v_compact(x_u, u, F), v_target, M)`.

Evaluate **one velocity prediction per context/draw**. Do not run a flow solver for the primary panel.

Record full-H16 loss and a separately masked first-ten-target loss from the same prediction. Full-chunk expert loss is the predeclared primary selection objective. Prefix loss is a secondary control for the known discarded-tail failure. Do not change winner selection after inspecting which metric looks better.

### Secondary: teacher agreement

For each state/draw, compute the teacher's full-history velocity once:

`D_teacher(F) = maskedMSE(v_compact(x_u,u,F), v_teacher(x_u,u,FULL), M)`.

Reuse this prediction across both readers, both backgrounds and all candidates. Keep teacher agreement separate from expert loss; do not introduce an arbitrary weighted combined target. Record the teacher's own expert loss as a reference.

The same compact output supports both losses. This makes the objective comparison inexpensive relative to separate label generation runs.

### Costs, signs and units

- Save raw losses and signed differences, not only scaled labels.
- Save `delta_tokens` exactly. First-move cost is 1; packet cost is 8.
- Save gain-per-token only for strictly positive-cost changes.
- Same-budget allocation contrasts have `delta_tokens=0` and use raw loss differences without division.
- A best candidate may still have negative gain relative to the background. Report this explicitly.
- A lower FM loss is a supervised surrogate result, not a measured increase in task success.

## 9. Discovery/confirmation randomness and numerical controls

Default to four discovery and four independent confirmation draws per state. Each draw contains an independently sampled Gaussian action tensor and flow time from the existing distribution.

Derive seeds from `(experiment_seed, task, episode, decision_index, bank, draw_index)`. Do **not** include student identity, reader, budget, candidate or worker order. Every alternative for a state/draw must receive exactly the same `(epsilon,u)`.

Fix candidates and backgrounds before drawing/evaluating losses. Use only discovery results to choose winners, losers and illustrative cases. Do not reselect on confirmation results.

Four confirmation draws are a bounded initial estimate, not enough to make precise per-context significance claims. Expose an additional-draw option with a fixed cap. Any confirmatory extension must be a new independent bank and a separately reported stage, not repeated sampling until a preferred sign appears.

Required numerical controls:

- Duplicate unchanged contexts produce the same prediction within the configured numerical tolerance.
- Reordering candidate evaluation or changing worker assignment preserves results within tolerance.
- Serial/reference versus packed/batched evaluation agrees on a representative subset.
- Reduced-precision measurements have a measured numerical floor; compare a small paired sample in higher precision if effects are of that scale.
- Use at least the existing raw-gain near-tie deadband where appropriate, but distinguish it from a practical effect threshold and from statistical uncertainty.

Ties must not be converted into confident preferences. Average selection credit/evaluation over the tie set or use a documented deterministic representative and report tie frequency. Two ties do not constitute perfect directional ranking agreement.

## 10. Analysis and the balanced-label question

### Required primary summaries

1. Distribution of immediate and packet gains, by reader, task, phase, background, source and descendant-count/depth stratum. Keep the unrestricted A2 audit separate from packet-conditioned A1 gains and show their distinct eligibility denominators.
2. Discovery-versus-confirmation scatter and winner reversal rate.
3. Confirmation loss/gain for immediate-selected packet, packet-selected packet, teacher-selected packet, candidate-pool random choice, UNIFORM and dispersed controls.
4. Same-candidate coarse/rich gain comparisons.
5. Packet-prefix curves for increments 0, 1, 2, 4, 8 on the bounded detailed subset.
6. Expert-versus-teacher preference disagreement, including cases where teacher improvement worsens expert loss.
7. Context/episode eligibility counts and exclusions, including unavailable sources, insufficient candidates, protected-background capacity and short horizons.

Use confirmation performance of a discovery-selected candidate as the main estimate of selection benefit. A maximum computed over confirmation means is an optimistic finite-pool diagnostic, not independently achieved performance and not a global oracle.

### Quantify whether balancing is worthwhile

Estimate the fraction of candidate operations and decision contexts with reproducible positive, harmful and practically small effects. Report raw effects, paired uncertainty and sensitivity to predeclared relative thresholds, for example 0.5%, 1% and 5% of background expert loss. These thresholds are diagnostic choices, not established robotics success thresholds. If the baseline is near zero, flag relative effects as ill-conditioned and rely on raw units rather than dividing by an arbitrary tiny denominator.

Do not define a useful positive solely as the top quantile. A top quantile exists even if all candidates are equivalent. Do not define statistical confidence from a numerical deadband.

Assess whether informative preferences occur across many episodes or are concentrated in a handful. This determines whether balanced sampling would expose a broadly learnable signal or merely repeat a few unusual trajectories. Retain all-negative contexts and report them separately.

No classifier or router is fitted during the frozen measurement phase. The bounded subsequent fitting stage in Section 6.8 must preserve episode boundaries and retain evaluation at representative prevalence if training uses balanced examples. Do not manufacture a positive class from arbitrary top quantiles.

### Statistical unit

Noise draws, candidates and multiple decisions from one episode are correlated measurements. Average over draws first; aggregate state measurements within episodes, then tasks. Provide paired episode-clustered bootstrap intervals for population summaries, stratified by task where feasible, and report the number of episodes. Do not count thousands of candidate/noise comparisons as thousands of independent demonstrations.

Report the router fitting and held-out episode partitions separately. Use the fitting partition for any exploratory construction adjustment, then freeze the protocol before the held-out partition. If both were already inspected, label the results exploratory.

## 11. Bounded generated-action validation

After the primary panel, validate at most **16 shared decision cases** per evaluated student with ordinary eight-step generation from pure Gaussian noise. Define a bounded common case list from development/discovery results across students before examining confirmation, spanning several episodes/tasks. Use a predeclared symmetric capped-union rule and evaluate every student on every retained case; do not give each student its own favorable case set:

- Cases where packet choice is predicted to beat immediate choice.
- Cases where the coarse/rich contrast is large.
- Expert/teacher objective disagreement cases.
- Null or unstable cases for comparison.

These are diagnostic strata, not an unbiased task-success sample. Use fresh generation seeds shared across contexts. Hold one selected context fixed for all eight integration steps, as in the production policy.

For each case, compare the base context, the completed immediate-selected and packet-selected contexts, the dispersed control, and the teacher full-history reference where affordable. Report normalized action differences, task-correct physical/FK paths, gripper timing when applicable, and prefix-versus-tail effects. Expert-action MSE of a generated sample remains a surrogate; a different valid motion may have higher MSE.

If an existing exact-state replay workflow and local assets are available, perform a small task-choice continuation subset. Restore a genuine causal prefix and verify the handoff state. Use the same observed history and fresh noise across branches, execute ten targets held twice at native 20 Hz, and report the selected target, destination or stop/continue behavior. Restrict the intervention to the first selected chunk, then resume one common predefined reader for subsequent replanning unless a fully specified causal continuation rule is implemented. Describe which choice the intervention can actually influence; do not claim long-term routing benefit from a single chunk.

Avoid replaying an old frontier as if it remains valid after new observations arrive. Include new history correctly at later decisions. Do not mistake an expert prefix for autonomous policy success. These continuations are diagnostic handoffs, not benchmark-from-reset scores.

Offline outcomes alone cannot distinguish a correctly used coarse cue from a cue ignored at every resolution. State this limitation if behavioral probes are unavailable or inconclusive.

## 12. Optional small interaction panel

Implement this only after the primary families run correctly and useful packet variation exists. It should not delay the main experiment.

Use up to 16 discovery-selected cases and two allocation slots. Each slot chooses between a candidate detail packet and a matched control packet, all with the same positive cost. Ensure the two slots have disjoint supports and every combination is feasible from a common background.

All four contexts `F00`, `F10`, `F01`, `F11` must have identical final token count and exact coverage. Save the four confirmation losses and compute:

`interaction = L10 + L01 - L11 - L00`.

A positive value indicates that selecting both candidate packets improves loss more than the sum of their separate substitutions under these controls. Call it an **interaction between resolution allocations**. It does not establish semantic complementarity, and its interpretation depends on the controls. When feasible, repeat with a second predeclared control pair.

Do not infer complementarity solely from `base`, `A`, `B`, `A+B` contexts with unequal sizes; general budget effects are then a competing explanation.

## 13. Compute bounds, batching and resumption

Reuse cached features and hierarchy records. Adapt each distinct compact context once and reuse it across noise draws where the existing API permits. Group by context length for bounded batching. Teacher full-history passes need length-aware batching and must never truncate history to fit a batch.

A default main panel with 256 states, two readers, two backgrounds, eight candidates, first/completed packet contexts, two controls and eight draws has an upper planning count of approximately:

`256 * 2 * 2 * (1 + 8 + 8 + 2) * 8 = 155,648`

compact velocity evaluations, before deduplicating identical contexts. The unrestricted immediate-only audit adds at most:

`256 * 2 * (1 + 4) * 8 = 20,480`

compact velocity evaluations, bringing the combined planning total to **176,128 per frozen student**. At the four primary endpoints S0/S_STD/S_STAGE/S_PACKET, this is at most **704,512** compact velocity evaluations before deduplication. Teacher references require approximately `256 * 8 = 2,048` full-history velocity evaluations and are shared across all students and panels; no extra teacher pass is needed for a new context at an already measured state/draw. Calculate actual counts from the manifest, including packet-ineligible states retained only for the audit. Counts refer to individual model-example evaluations, not GPU batch launches. The 32-state capability panel, prefix-2/4 curves, higher-precision probes and generated chunks add explicit costs and must appear in the dry-run estimate.

The default primary training matrix adds three 100k student jobs: **300k optimizer updates total**, or 38.4 million example exposures at effective batch 128. Frozen S0 adds no updates. These are three separate jobs with shared data and encoders, not a requirement for three simultaneous GPU allocations. Time one real-data training segment for each sampler; geometry/packet construction can change CPU loading costs even when decoder FLOPs match. Report GPU hours, wall time and cache-assembly overhead per job. Use the existing checkpoint-and-resume mechanism if an allocation ends; do not shorten only one arm unnoticed.

Prepare all required configurations and their dependency graph before launch. Frozen S0 measurements and new student jobs may proceed independently after their correctness checks. The two conditional paired continuations each add 20k total student updates if activated; router label generation, scoring and simulation have their own separate estimates. They are not silently included in the initial allocation. The user has explicitly permitted multiple student jobs; no positive-S0-result approval gate is needed, while existing resource and prepare-only instructions still apply.

Provide separate modes for manifest construction, smoke measurement, main measurement and report generation. After the real-data smoke run, report observed batch throughput, peak memory, projected model-example counts and estimated total runtime. Do not extrapolate from router-fit seconds or promise a particular HPC duration without measurement.

Make maximum states, candidates, draws, generated cases, wall time and worker count configurable. Use the project's existing resource/job conventions. A time cap should checkpoint completed rows and stop cleanly, not silently shrink histories or change precision. Avoid launching a large resource sweep.

Resumption must verify the complete experiment identity and preserve candidate IDs, seeds and selections. A changed checkpoint, candidate recipe, precision, action convention or cache identity starts a new experiment. Do not overwrite existing router labels or diagnostic results.

## 14. Output schema and reporting requirements

Store compact machine-readable arrays plus a human-readable Markdown report. Use existing project conventions for HDF5/NPZ/JSONL; do not serialize every full point cloud into every result row.

At minimum retain:

- **Experiment manifest:** schema version, code/config/dependency hashes, state sampling, candidate rules, seeds, precision, limits and phase status.
- **Student manifest:** variant, initialization and checkpoint hash, trainable modules, optimizer/schedule, episode exclusions, context/prefix distributions, fallback counts, update count and training compute.
- **State table:** task/episode/decision/partition, eligible archive roots/leaves, source counts, target mask and exclusions.
- **Context table:** stable ID, experiment family/panel, reader, background, candidate/packet ID, ordered moves, exact leaf coverage hash, node/selected-leaf IDs, residual identities, actual cost and count.
- **Raw draw table:** student checkpoint hash, bank, draw seed, flow time, context ID, full/prefix expert losses and teacher discrepancies.
- **Selection table:** student checkpoint hash, discovery choices and tie sets, selection objective, independent confirmation results and fixed-control comparisons.
- **Router/adaptation manifest:** label-generating student hash, fitting episodes, objective, input features, controller/action-space version, scoring compute, any adaptation parent and refreshed label identity.
- **Timing/memory table:** encoding/cache loads, context assembly, teacher/compact forward costs, integration and report generation separately.
- **Behavioral subset:** selection reason, fresh seeds, exact intervention convention, generated actions and observed outcome when run.

The report must answer H1–H8 individually with `supported`, `not supported`, or `inconclusive`, qualified by student, tested tasks/budgets/candidate families. Include negative results and all denominators. Report the complete required student matrix, including failed or inconclusive arms. Link a small selection of interpretable examples while retaining the full machine-readable result set.

Required plots are signed-gain distributions, discovery/confirmation agreement, equal-budget selection comparisons, same-operation coarse/rich comparisons, prefix curves, and expert/teacher disagreement. Plot episode-aware uncertainty where appropriate. Make plots reproducible from stored results without rerunning models.

Add a student × allocation-strategy table of absolute losses/behaviors and a separate table of within-student routing improvements. Distinguish diagnostic loss-selected packets from deployable learned-router results. Include realized training distributions and the fixed-endpoint learning curves. If conditional branches are not activated, record the finding and whether their implementation/configuration is ready.

## 15. Meaningful correctness tests

Add focused tests for scientific invariants rather than reproducing implementation line by line:

1. TREE packet adds exactly m records and preserves exact causal coverage at every prefix.
2. FLAT packet preserves residual accounting and rejects zero-cost promotions in this experiment.
3. Protected candidate identities/support and internal packet operations remain unchanged across coarse/rich backgrounds.
4. Final coherent/control packets have equal actual budgets; invalid protected-background constructions fail explicitly.
5. Same-state alternatives share exactly the same noise/time, independent of batching and worker order.
6. Expert loss uses the real FM target, not teacher velocity; a synthetic example where their preferences disagree produces the expected signs.
7. Discovery-selected indices cannot be changed by modifying confirmation losses; ties follow the documented policy.
8. Serial versus batched losses agree; no-op comparisons establish the numerical floor.
9. Restored runs reproduce manifests and measured rows; dependency changes reject resume.
10. Task-specific action conversion and masks work for at least a gripper task and a stick task when those assets are available.
11. S_STAGE/S_PACKET share state, family, target/actual budget and feasibility/fallback schedules independently of allocation-specific RNG consumption.
12. Complete training routes preserve exact coverage at every intermediate count, including short final packets and FLAT residual normalization.
13. New student/router fitting rejects diagnostic and reserved-validation episodes; no final-confirmation metric influences checkpoint selection.
14. Intended student modules receive gradients while encoders/teacher remain frozen; cached full-leaf initialization agrees with the teacher on a small reference subset.
15. A changed student hash invalidates compact utilities and fitted-router label provenance while allowing compatible teacher reference predictions to be reused.
16. Router-conditioned continuation controls match actual prefix budgets and underlying examples; only context-selection policy differs.

Run the relevant existing causality, frontier, residual and cached-policy tests. Do not introduce unrelated model changes to make diagnostic results look better.

## 16. Decision rules after the experiment

| Result | Recommended next direction | Claim not justified |
|---|---|---|
| Stable packet selection beats immediate selection and dispersed controls at equal B | Train packet/short-lookahead preferences with a lightweight scorer | Globally optimal retrieval or TREE superiority |
| Same candidate helps mostly with coarse background | Condition utility on available detail; investigate the supported smaller-budget regime | Transfer of a roots-only router to B128 |
| Stable positives exist across many episodes but are rare | Test balanced within-context preference training | Balanced accuracy as deployment success |
| Teacher and expert preferences systematically conflict | Use expert FM utility as a primary training target and retain teacher agreement as a diagnostic | Guaranteed improvement in closed-loop success |
| Fine and coarse contexts both produce correct choices | Investigate a lower useful budget and compression efficiency | Need for a complex learned router |
| Broad candidate allocations fail to improve the intended choices | Improve the teacher/compact policy's use of historical information before more router fitting | That no possible retrieval scheme could help |
| FM gains fail to survive generated-action/choice checks | Treat the current surrogate as insufficient for the desired behavior | Robot-control improvement |
| Different readers have different eligibility/candidate support | Report within-reader conclusions and matched subsets | A clean TREE-versus-FLAT winner |
| S_STAGE improves over matched S_STD | Prefer training exposure closer to the intended routing states | That packet structure caused the improvement |
| S_PACKET improves over matched S_STAGE under common controls/choices | Retain coherent allocation exposure and test its interaction with packet routing | That larger utility spread alone implies a better student |
| Only teacher or rich-history controls make correct memory decisions | Consider the separate rehearsal/capability branch; inspect what compression loses | That forcing teacher agreement will solve the problem |
| A matched router-conditioned continuation improves behavior | Freeze the adapted student, refresh its utility labels and evaluate one refreshed router | An unlimited benefit from repeated joint optimization |

Do not require an arbitrary 60% ranking score or a specific success percentage to proceed. A useful positive result is independently confirmed improvement in the actual selection made, at a controlled budget, with some behavioral evidence that the difference matters. Complete the required matched student matrix even if S0 has weak utility; do not expand conditional branches indefinitely when evidence remains weak. A student that performs well with coarse contexts is a valid positive compression result even if routing adds little.

## 17. Literature informing this addendum

These sources motivate the measurement design; none establishes that the proposed changes will work in Chronotree.

- **REPLUG**, NAACL 2024: frozen-reader supervision from reference-output likelihood. Our analogue uses expert FM loss. [Paper](https://aclanthology.org/2024.naacl-long.463/)
- **Atlas**, JMLR 2023, sections 2.2.3–2.2.4 and 4.3: individual-document/leave-one-out likelihood distillation and the conditioning-distribution issue; reader training with retrieval provides substantial benefit even with a fixed retriever during pretraining. This motivates testing the reader rather than assuming immediate joint optimization is mandatory. [Paper](https://jmlr.org/papers/volume24/23-0037/23-0037.pdf)
- **CEIL**, ICML 2023: preferences between complete retrieved sets. Borrow the set-level measurement idea, not a generic diversity penalty that could remove repeated counting evidence. [Paper](https://proceedings.mlr.press/v202/ye23c.html)
- **Learning to Retrieve Iteratively for In-Context Learning**, EMNLP 2024: stateful retrieval and subsequent-return credit. This addendum tests whether delayed value exists before adopting its RL machinery. [Paper](https://aclanthology.org/2024.emnlp-main.406/)
- **Native Sparse Attention**, 2025, section 3.3.3: discusses local patterns dominating learning of compressed/selected information and uses separate attention branches. This motivates checking the shortcut hypothesis, not adding its architecture to the initial experiment. [Paper](https://arxiv.org/html/2502.11089v1)
- **DeepSeek-V3.2**, 2025, section 2.1.1: indexer warm-up with a frozen main model, followed by sparse adaptation of the main model. Borrow the staged adaptation principle; its scale, objective and architecture differ from this robotics study. [Paper](https://arxiv.org/html/2512.02556v1)
- **Fighting Copycat Agents in Behavioral Cloning from Observation Histories**, NeurIPS 2020: past-action shortcuts can be inferred from observation histories and coexist with good imitation loss. Removal of explicit action/proprioception inputs does not by itself establish semantic memory use. [Paper](https://papers.nips.cc/paper_files/paper/2020/hash/1b113258af3968aaf3969ca67e744ff8-Abstract.html)

## 18. Completion checklist for Codex

- [ ] Existing implementations and local instructions inspected; no parallel incompatible data path introduced.
- [ ] Frozen checkpoint/cache identities verified and included in the manifest.
- [ ] Common new-training pool and development/final episode exclusions fixed before model evaluation.
- [ ] S_STD/S_STAGE/S_PACKET training jobs implemented with matched initialization, exposure, optimizer and schedule; S0 preserved.
- [ ] Complete coarse-to-budget random/packet routes and matched prefix/fallback sampling implemented; sampler audit reported.
- [ ] Gain-blind state/candidate manifests and eligibility accounting implemented.
- [ ] Unrestricted B127-to-B128 immediate-only audit implemented independently of packet eligibility.
- [ ] Same-candidate coarse/rich backgrounds, coherent packets and equal-budget controls implemented for TREE and FLAT.
- [ ] Expert FM and teacher-agreement losses measured from shared predictions with paired randomness.
- [ ] Frozen measurements repeated for each primary student, using the same candidate/noise manifests and checkpoint-specific labels.
- [ ] Capability references and absolute student-quality comparisons included, not only refinement-gain comparisons.
- [ ] Discovery and confirmation kept separate; episode-aware summaries and reproducible plots implemented.
- [ ] Focused tests and bounded real-data smoke completed, or exact missing assets documented.
- [ ] Runtime estimate and resumable main configuration supplied; actual completed scope stated accurately.
- [ ] Bounded generated-action validation prepared/completed as resources permit, without claiming benchmark success.
- [ ] Required student jobs completed or exact ready-to-run commands and missing resources reported, following existing launch instructions.
- [ ] Conditional rehearsal and one-cycle router-adaptation configurations provided; activation/omission and actual scope stated.
- [ ] Bounded learned-router stage implemented/prepared with fresh student-specific labels and crossed evaluation when supported.
- [ ] Findings mapped to H1–H8 and next decisions; deployed-router results separated from offline diagnostic selections.
