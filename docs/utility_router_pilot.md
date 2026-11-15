# Expert-loss router follow-up on B200s

This is a new **95k development pilot** derived from the accepted 140k teacher
study. It does not change that study's 100k primary endpoint, consume its final
confirmation partition, reuse its diagnostic episodes for fitting, or relabel
the earlier teacher-agreement router datasets. All new labels and fits live in
a separate run directory. The prior utility measurements remain reusable.

Commit and push these implementation changes, then pull them on the HPC. The
launcher freezes the current committed source for the new pilot and restores
the original committed source for the older generated-action diagnostic. It
creates ignored source copies, not branches or Git worktrees. A commit whose
source does not match the working tree is rejected; commit the intended source
changes instead of bypassing that check.

No new dataset preparation or teacher encoding is needed. The site file supplies
paths, Python, activation and account settings. These launchers explicitly use
the current cluster requirement: **`aiq`, `--gres=gpu:b200:1`**, with no exclusive
node request. They do not inherit the old teacher run directory or W&B project
from `site_b200_4gpu.env`.

## 1. Inspect generated actions from the same development cases

Run from the HPC checkout:

```bash
cd /hpc/home/phi/rvalperga/chronotree
git pull --ff-only

.venv/bin/python hpc/refinement_generated.py \
  --site hpc/site_b200_4gpu.env \
  --manifest runs/router_utility_teacher140k/manifest.json \
  --name utility140k_generated95k \
  --limited-check \
  --checkpoints \
    S_STD=runs/utility_students_teacher140k_m64_s_std/seed0/compact/step_00095000.pt \
    S_STAGE=runs/utility_students_teacher140k_m64_s_stage/seed0/compact/step_00095000.pt \
    S_PACKET=runs/utility_students_teacher140k_m64_s_packet/seed0/compact/step_00095000.pt \
  --partition aiq --gpu-type b200 \
  --walltime 02:00:00 \
  --dry-run
```

The dry-run prepares one immutable case plan on the CPU and renders three batch
scripts. It does not run policy inference or submit anything. The exact original
study code verifies the parent, measured checkpoint hashes and discovery files.
`--limited-check` records the earlier 95k snapshot scope; it does not relabel it
as the declared 100k endpoint. All three students must have the same actual step
and initialization.

Submit the already prepared jobs:

```bash
.venv/bin/python hpc/refinement_generated.py \
  --site hpc/site_b200_4gpu.env \
  --name utility140k_generated95k \
  --stage generate --submit
```

The jobs are independent: one B200, eight CPUs, 64 GB RAM and a two-hour limit
each. They evaluate the same discovery-selected set of at most 16 states, using
two shared fresh Gaussian draws, eight flow integration steps and the parent's
eight-target executed prefix. The compared contexts are base, completed
immediate-selected packet, completed packet-selected packet, dispersed control
and teacher FULL. Case limits and diversity caps are not relaxed to fill a quota.

The results are predicted joint targets and task-correct FK paths, **not
simulator trajectories or success rates**. Per-student generation is parallel;
aggregate HTML writing is serialized so concurrent workers cannot corrupt it.

```bash
.venv/bin/python hpc/refinement_generated.py \
  --site hpc/site_b200_4gpu.env \
  --name utility140k_generated95k --stage status
```

Results:

```text
runs/utility140k_generated95k/generated_plan.json
runs/utility140k_generated95k/generated_plan_results/index.html
runs/utility140k_generated95k/generated_plan_results/report.md
runs/utility140k_generated95k/generated_plan_results/summary.json
hpc/generated/utility140k_generated95k/logs/generated_s_*_<job>_*.{out,err}
```

An interrupted worker resumes completed cases with `--stage generate --submit
--resume`; active workers are not duplicated. To rebuild only presentation on
the CPU, use `--stage report`. `--stage status` only reads files.

## 2. Prepare labels and train the four matched routers

This stage uses S_STD as the frozen action policy, selected from the development
report. It is independent of the generation jobs: scheduling it does not depend
on choosing favorable generated examples. Interpretation of closed-loop claims
still requires behavioral evidence.

```bash
.venv/bin/python hpc/utility_routers.py \
  --site hpc/site_b200_4gpu.env \
  --parent-manifest runs/router_utility_teacher140k/manifest.json \
  --checkpoint runs/utility_students_teacher140k_m64_s_std/seed0/compact/step_00095000.pt \
  --student S_STD --development-step 95000 \
  --name utility_router_std95k \
  --fit-states 4096 --steps 5000 \
  --label-shards 4 --max-concurrent 4 \
  --partition aiq --gpu-type b200 \
  --stage all --dry-run
```

Inspect `hpc/generated/utility_router_std95k/plan.json` and the generated batch
files, then submit that frozen plan:

```bash
.venv/bin/python hpc/utility_routers.py \
  --site hpc/site_b200_4gpu.env \
  --name utility_router_std95k \
  --stage all --submit
```

This schedules a dependency chain:

1. **Preparation:** one B200 allocation with eight CPUs and 64 GB RAM, up to two
   hours. Preparation itself is CPU work. It validates the original study with
   its original code, freezes the S_STD checkpoint and writes a separate child
   manifest. It selects 4,096 unique task/episode/phase-balanced fitting states
   from the permitted pool and retains the original 128 development states.
2. **Labels:** four independent one-B200/eight-CPU/64-GB workers, up to four
   hours each. They share the immutable state manifest and produce both TREE
   and FLAT examples. Earlier reachable routing states and late refinements
   are included. Before/first/complete contexts share action noise; discovery
   and confirmation banks are independent. Raw expert losses and signed gains
   remain available. The original final-confirmation episodes remain unused.
3. **Four fits:** TREE/FLAT × immediate/completed-packet supervision, each with
   one B200, four CPUs and 16 GB RAM, up to two hours. Each job trains the same
   lightweight MLP scorer for 5,000 updates and then runs offline development
   evaluation. Expert actions, action noise and flow time never enter router
   features. Capacity, observable features and training budgets are matched.

The first label set uses gain-blind routing trajectories. No model-route refresh
or student adaptation is launched. The packet controller uses the same packet
construction and actual token-cost accounting as label generation. These new
expert-loss labels cannot be loaded by the earlier teacher-agreement trainer.

The resource limits are scheduling requests, **not measured duration estimates**.
The main chain makes six submissions: preparation, one four-worker array and
four fitting jobs. At most four GPUs are simultaneously eligible within this
chain. The separate generated-action jobs can run concurrently if quota permits.
If an allocation expires, a partial stage does not release dependent jobs as if
it had completed.

All label, training and offline-evaluation tracking goes to **`chronotree-routers`**.

```text
runs/utility_router_std95k/manifest.json
runs/utility_router_std95k/frozen/student.pt
runs/utility_router_std95k/labels/worker_0.json ... worker_3.json
runs/utility_router_std95k/fits/tree_immediate/
runs/utility_router_std95k/fits/tree_packet/
runs/utility_router_std95k/fits/flat_immediate/
runs/utility_router_std95k/fits/flat_packet/
  last.pt
  step_00005000.pt
  complete.json
  evaluation/summary.json
hpc/generated/utility_router_std95k/logs/
```

Read status or resume inactive/partial work:

```bash
.venv/bin/python hpc/utility_routers.py \
  --site hpc/site_b200_4gpu.env \
  --name utility_router_std95k --stage status

.venv/bin/python hpc/utility_routers.py \
  --site hpc/site_b200_4gpu.env \
  --name utility_router_std95k --stage all --submit --resume
```

Completed stages are skipped, active allocations are reused and missing work is
resubmitted once. No automatic repeated resubmission occurs. The source snapshot
and experiment identity remain fixed even if the main checkout later changes.
Use a new name when changing checkpoints, fitting counts, schedules or resources.

## 3. Matched closed-loop screening after the fits

First inspect `fits/*/evaluation/summary.json`, especially selected confirmation
loss versus the equal-cost random and UNIFORM controls. Label ranking alone is
not a task-success result.

```bash
.venv/bin/python hpc/utility_routers.py \
  --site hpc/site_b200_4gpu.env \
  --name utility_router_std95k \
  --stage evaluate \
  --eval-gpus 4 --workers-per-gpu 4 --eval-cpus 64 \
  --episode-count 2 --video-cap 10 \
  --eval-walltime 06:00:00 \
  --dry-run
```

Replace `--dry-run` with `--submit` after reviewing the plans. This renders seven
matched validation jobs: four learned packet selectors, two gain-blind random
packet controls (one per reader), and the existing global UNIFORM baseline. All
use the exact same frozen S_STD checkpoint, budget 128, eight integration steps,
eight executed targets and official validation episode IDs 0 and 1 for each of
the 16 tasks. Each allocation has four typed B200s, 64 CPUs and 128 GB RAM. The
seven jobs need not run simultaneously; Slurm applies the account limits.

In the four learned runs, both supervision variants execute the entire selected
packet: this isolates the credit target at equal packet cost. `--controller
local` is a separately named comparison that executes only the first move and
re-scores. It changes the controller, so do not pool it with the credit-only
comparison. Its one-move feature values also differ from the usual eight-move
packet candidates used for fitting, so it is a transfer diagnostic rather than
the primary learned-controller test.

The existing evaluation launcher snapshots router and policy weights. These
evaluation jobs use the main checkout's guarded source, so keep that checkout
unchanged while they are queued/running. This restriction does not apply to the
new pilot's snapshot-based label/training jobs.

Before rendering any evaluation, the pilot launcher validates its child manifest
under the frozen runtime and verifies all four completed fits and their offline
receipts against exact checkpoint hashes. Evaluation reads the pilot's immutable
`frozen/student.pt`, not the original checkpoint path supplied during preparation.
An explicit policy configuration preserves the declared integration steps and
action cadence; the token budget and executed-target count come from the sealed
child protocol.

Evaluation outputs and videos are under:

```text
runs/evaluation/utility_router_std95k_<variant>_packet_eval/
```

Use the corresponding `summary.json`, per-episode outputs and W&B evaluation
links. Timeouts and infrastructure errors remain in the planned denominator.
This initial 32-episode screen is development evidence, not final confirmation
or a replacement for the existing 160-episode evaluations.

## Verification and current limits

The launchers have CPU-only tests for dry-run behavior, typed GRES, exact source
separation, dependency wiring, active-job-safe resumption, immutable parameters,
checkpoint identity and concurrent report serialization. Batch scripts are
syntax checked when rendered. A 556-test focused CPU regression passed across
utility, refinement, router, evaluation, tracking and cached-policy paths. The
new synthetic integration test exercises actual cached FM inference, label
serialization, all four fits and offline evaluation. Router rollout tests bind
the execution protocol before saving manifests; online/cached feature tests
cover both readers, empty histories and exact leaf coverage. The architecture
snapshot/XML check passed and the updated HTML panel was visually inspected.
The workstation GPU was not used and no Slurm
job was submitted during implementation. Actual B200 throughput, rendered
simulator behavior and router success remain to be measured on the HPC.
