# Student training × refinement utility study

**Current launch instructions:** the user selected the 140k teacher on
11 November. Use [the separate 140k workflow](refinement_utility_teacher140k.md).
The original 104k teacher/S0 study and its measured checks below are retained as
historical evidence; their checkpoint/cache identities must not be substituted
into the new study.

This implements the required first batch in
[`chronotree_router_utility_addendum.md`](../chronotree_router_utility_addendum.md).
The production 90k compact policy is S0. Three new, matched students start from
the accepted 104,110-step teacher: S_STD keeps the original context sampler,
S_STAGE adds random single-move construction prefixes, and S_PACKET uses
coherent packet construction with the same exposure/prefix schedule. At the
user's request, each now trains 100k updates on the same remaining router-fitting
episodes. No teacher training,
encoder update, raw preparation or production-cache regeneration is introduced.

The amended schedule keeps LR 1e-4, 1k warmup and cosine decay to 1e-5 at 100k.
It saves rolling `last.pt` every 1k updates and immutable numbered snapshots
every 5k, including the 100k primary endpoint. Earlier snapshots are for
development on fitting-tagged episodes only. All three jobs total 300k updates
and 38.4 million example exposures at effective batch 128; runtime is unmeasured.
The protocol pins `student_training: {steps: 100000, checkpoint_every: 1000,
snapshot_every: 5000}`.

## Frozen dependencies and scope

| Dependency | Identity |
|---|---|
| Teacher 104110 | `eb57ccd4c131888a5f5d6225a3a0cbe5cd14b3afc5c6ff0a1a17ee319cd70918` |
| S0 compact 90000 | `4fe22fd00fbc64dc740ba046d6ce1e58ade2a70580f8792e152477caad316427` |
| Encoded cache | `3db7b7ce80926d21659863c3` |
| Hierarchy cache | `3567142ed84d1b30b4a744ca` |

The workstation has the exact checkpoints, but only a bounded real VideoUnmask
episode cache. Its identity is deliberately different and is accepted only
with `--diagnostic-assets`. That mode creates no training pool and cannot be
used by the production student launcher. It is an implementation check, not a
substitute for the requested 256-state matrix or multi-task smoke.

Production manifests preserve the original router split. They first select
four fitting and four validation episodes per task, then two separated eligible
states per episode, balancing early/middle/late strata. The whole selected
episode is excluded even if it lacks an eligible state. A complete horizon is
required only on structurally available action coordinates; stick tasks remain
eligible. Every diagnostic episode and the entire original router-validation
partition are excluded from new optimization. No measured loss selects states,
backgrounds, candidate pools or training examples.

## What is measured

For each frozen student, reader and state, the same immutable candidates and
four discovery/four independent confirmation draws are used. Expert FM loss is
computed from `flow_sample`'s true velocity target. Frozen measurements default
to FP32 with TF32 disabled; student optimization stays BF16. Teacher discrepancy is a
separate loss from the same student prediction. Full H16 expert loss selects
candidates; the first ten targets are a secondary measurement. Teacher FULL
predictions are shared across students and protected by evidence/draw/precision
checksums. Their cache excludes student identity; compact rows include it.

The primary panel includes:

- Protected B56/B120 backgrounds with unchanged candidate support and internal
  operations, followed by eight +1-cost refinements/promotions to B64/B128.
- Prefixes 1/2/4/8 and equal-final-budget uniform/dispersed controls. The current
  implementation measures prefixes 2/4 throughout; the exact dry-run estimate
  includes these extra contexts rather than quoting the smaller planning bound.
- An independent unrestricted B127→B128 audit with up to four single moves. It
  remains eligible even when packet construction fails.
- A 32-state gain-blind capability panel: ordinary B128, gain-blind detailed
  B128, B256, all cached leaves, and the teacher reference. These are proxy
  capability comparisons; task-choice correctness still needs behavior.

TREE frontiers partition all available leaves. FLAT records preserve selected
leaf plus source/modality residual coverage, including singleton normalization.
Both preserve causal pending-prefix encodings. Current observation appears once;
task/clock/current tokens remain outside the historical budget. Positive-cost
packet promotions never use a zero-cost final-leaf operation. Same-budget
contrasts have cost zero and are never divided by a fabricated token cost.

Packets are selected gain-blind by observable source/modality and region
identity. TREE descendants split by mass with stable ties. FLAT uses balanced
observed space/time ordering when metadata is present; a topology-only fallback
is explicitly named. Every complete route retains each integer prefix, so a
training sample may stop within a packet. S_STAGE and S_PACKET use the same
feasibility prepass/fallback and actual prefix count.

Discovery ties use average evaluation over the near-tie set. Confirmation
cannot change the selected set. Reports retain all-negative pools, noise
variation, objective disagreements, eligibility denominators, paired background
contrasts, absolute losses and within-student selection improvements. Bootstrap
units are episodes within tasks, after averaging states; they are not candidates
or noise draws and do not estimate variation across training seeds.

An optional [independent confirmation extension](refinement_confirmation_extension.md)
adds at most 16 fresh draws in a separate bank. It preserves the primary
discovery choices and reports its rows separately. The one-task local check
does not produce an HPC duration forecast. After the required multi-task smoke,
`runtime_projection.json` scales measured component costs by actual manifest
work counts and shows minimum/median/maximum cost scenarios, explicitly excluding
training, startup and later behavioral checks.

## HPC sequence

Run from `/hpc/home/phi/rvalperga/chronotree` after transferring the code. No site
sourcing is required. The site file supplies paths/account settings; these
launchers request untyped GPUs on `gpuq`, without `--exclusive`.

The names below assume a fresh start. If these names already pin an old 30k
manifest or source revision, choose new panel and student study names throughout
and prepare again. Do not overwrite an immutable plan or change the checkout
used by queued/running jobs. Old manifests and checkpoints retain their original
declared endpoint; the new defaults do not reinterpret them.

First prepare the common panel/candidates on an allocation. Its default S0 path
is the preserved router-study copy, not a mutable training `last.pt`:

```bash
.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv_seed0 \
  --name router_utility_v1 --stage prepare --dry-run

# Submit the inspected plan (one GPU allocation, 8 CPUs, 64 GB, four hours).
.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv_seed0 \
  --name router_utility_v1 --stage prepare --submit
```

Wait for `runs/router_utility_v1/contexts_complete.json`. This includes the exact
deduplicated model-example counts and eligibility accounting. No model losses
have been evaluated yet. If the preserved S0 file is elsewhere, supply `--compact
/absolute/path/to/checkpoint.pt`; its hash must still be the recorded 90k hash.

Render all three student jobs, then run the sampler audit and S0 smoke:

```bash
.venv/bin/python hpc/student_utility.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv_seed0 \
  --manifest runs/router_utility_v1/manifest.json \
  --name utility_students_v1 --dry-run

.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_b200_4gpu.env --name router_utility_v1 \
  --stage audit --student-study-name utility_students_v1 --submit

.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_b200_4gpu.env --name router_utility_v1 \
  --stage smoke --student S0 --panel-partition train --submit
```

Audit and smoke are independent. The audit samples 4,096 common source states
without policy inference. It records realized source coverage, depth, leaf
fraction, actual counts, concentration, packet lengths, prefix locations and
fallbacks. The smoke requires at least eight states across three tasks, both
history banks and a causal partial block; it does not accept a one-task check
as the required smoke. Inspect failures before proceeding; no positive utility
result is required to start the students.

Submit the three independent student jobs and S0 development measurements:

```bash
.venv/bin/python hpc/student_utility.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv_seed0 \
  --manifest runs/router_utility_v1/manifest.json \
  --name utility_students_v1 --submit

.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_b200_4gpu.env --name router_utility_v1 \
  --stage measure --student S0 --panel-partition train --shards 4 --submit
```

Each student requests two GPUs, 64 CPUs, 128 GB RAM and 48 hours. There are three
allocations, six GPUs if all start together. The measurement array has four
independent one-GPU allocations, each with 8 CPUs/64 GB. These resource requests
are bounded defaults, not a measured cost prediction. The student effective
batch is 128, microbatch 32/GPU, with 16 loading workers/GPU. W&B training runs
use `chronotree-routers`. Do not change source while jobs are queued/running:
startup guards intentionally reject a changed implementation.

At each fixed 100k endpoint, measure that student; for example:

```bash
.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_b200_4gpu.env --name router_utility_v1 \
  --stage measure --student S_STAGE \
  --checkpoint runs/utility_students_v1_s_stage/seed0/compact/step_00100000.pt \
  --panel-partition train --shards 4 --submit
```

Use the corresponding `S_STD`/`_s_std` and `S_PACKET`/`_s_packet` paths for the
other arms. Any earlier declared snapshot, including 15k or 30k, must use the
development partition and is stored separately. After the development protocol
is fixed, repeat the four endpoint measurement
commands with `--panel-partition validation` for the reserved final half.
Checkpoint variant, initialization, study manifest and update count are checked.

Generate a report from whatever has actually completed:

```bash
.venv/bin/python -m stmem.inspection.refinement_utility \
  --mode report --manifest runs/router_utility_v1/manifest.json
```

Reports explicitly include missing student arms. They can be regenerated without
loading policies. The report command is CPU-only. Results live under
`runs/router_utility_v1/`; launch scripts/logs live under
`hpc/generated/router_utility_v1/`. Student runs/scripts have their independent
`utility_students_v1_s_*` names. Add `--resume` to a launcher only after its
previous allocation/array is inactive. Committed state rows are checksum-checked
and reused; changed code/checkpoint/configuration requires a new study identity.

## Recovering the empty-history audit failure

The 11 November fix repairs an empty FLAT Boolean mask and empty group indices.
It preserves zero-history examples, causal boundaries and the existing caches.
Job 72962's failed audit output and the original `router_utility_v1` source-bound
manifest must remain historical artifacts. Use fresh `router_utility_v2` and
`utility_students_v2` names. `--resume` cannot apply the fix to an old source
identity, and the audit refuses to overwrite a nonempty output directory.

After committing/pushing the repair, the following creates an isolated checkout
on the HPC so the still-running teacher can keep its main checkout unchanged.
The detached worktree creates no branch. Links share the environment and output
directories; the prepared data and teacher caches are read from their existing
absolute paths. `git fetch` updates remote references without changing the main
checkout's files or HEAD.

```bash
cd /hpc/home/phi/rvalperga/chronotree
git fetch origin
git worktree add --detach ../chronotree_utility_fix origin/master

cd /hpc/home/phi/rvalperga/chronotree_utility_fix
ln -s /hpc/home/phi/rvalperga/chronotree/.venv .venv
ln -s /hpc/home/phi/rvalperga/chronotree/runs runs
ln -s /hpc/home/phi/rvalperga/chronotree/hpc/generated hpc/generated

cp /hpc/home/phi/rvalperga/chronotree/hpc/site_b200_4gpu.env hpc/site_utility.env
cat >> hpc/site_utility.env <<'ENV'

STMEM_REPO_ROOT=/hpc/home/phi/rvalperga/chronotree_utility_fix
STMEM_PYTHON=/hpc/home/phi/rvalperga/chronotree/.venv/bin/python
ENV
export PYTHONPATH="$PWD/src"

.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_utility.env --cache-name memory_fv_seed0 \
  --name router_utility_v2 --stage prepare --submit
```

This submits only the utility preparation job. Wait for
`runs/router_utility_v2/contexts_complete.json` with `status: complete`, then
render the new student configurations and submit the independent checks:

```bash
cd /hpc/home/phi/rvalperga/chronotree_utility_fix
export PYTHONPATH="$PWD/src"

.venv/bin/python hpc/student_utility.py \
  --site hpc/site_utility.env --cache-name memory_fv_seed0 \
  --manifest runs/router_utility_v2/manifest.json \
  --name utility_students_v2 --steps 100000 --dry-run

.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_utility.env --name router_utility_v2 \
  --stage audit --student-study-name utility_students_v2 --submit

.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_utility.env --name router_utility_v2 \
  --stage smoke --student S0 --panel-partition train --submit
```

After successful checks, use the same student command with `--submit` in place
of `--dry-run`. Continue using this checkout, `hpc/site_utility.env`, and the
v2 names for subsequent measurements. Physical outputs remain under the main
checkout's `runs/` and `hpc/generated/` through the links. The old v1 artifacts
remain available. These recovery commands were source-reviewed and shell syntax
checked locally; no HPC resubmission has been performed here.

## Generated actions and conditional follow-ups

`python -m stmem.inspection.refinement_generated` creates a common at-most-16-case
plan from development/discovery rows only, using a symmetric capped union across
students. Every student receives every retained case. It performs ordinary
eight-step pure-Gaussian generation, retaining one context throughout the
integration, and writes predicted physical/FK paths with ten-target prefix versus
tail. These paths are not executed dynamics or success rates. A limited S0-only
plan must be explicitly marked `--limited-check`; it cannot masquerade as the
crossed primary matrix.

The [exact-state continuation wrapper](refinement_continuation.md) additionally
executes one selected first chunk, then uses the same UNIFORM/B128 reader over
each branch's newly observed history. This bounded local diagnostic verifies
the genuine expert prefix and produces videos; it is not a benchmark evaluation.

The optional interaction panel is unactivated. Full-history rehearsal and
one-cycle router adaptation are separate conditional protocols; no conditional
training or router fitting is launched by the commands above. A complete
first-batch report does not depend on positive S0 gains or on activating them.

The old router labels optimize teacher agreement. They must not be renamed to
expert-FM/packet labels. The later fitting protocol requires fresh labels bound
to each selected student, the same episode exclusions, and the corresponding
controller. Conditional preparation records implementation readiness explicitly.

## Cache compatibility audit

The two new training modules are exact, audited runtime consumers in
`cache_source.py`; all encoders, aggregation, normalization and action-alignment
source remains protected. The previous teacher-continuation commit also added
only `positive_checkpoint_every`/`resolve_checkpointing` helpers to the otherwise
protected checkpoint module. Auditing those exact changes preserves the existing
`9cddd3…` encoded-content fingerprint. No permissive wildcard exclusion, cache
identity rewrite, or representation migration was introduced. New unknown source
files and future protected edits still invalidate compatibility.

## Measurements completed locally

The device doctor passes FP32/BF16/FP16 on the RTX 5000 Ada (32 GB). A bounded
real VideoUnmask-episode-0 training probe performed five microbatch-four updates
per sampler from identical teacher initialization, without saving model weights.
Cached FULL context and velocity parity were exact on 1,056 historical leaves;
frozen encoders received no gradients. Excluding the first update:

| Sampler | Serial cache/context assembly | GPU update | Peak allocated memory |
|---|---:|---:|---:|
| S_STD | 0.06246 s | 0.00796 s | 112.74 MiB |
| S_STAGE | 0.09320 s | 0.00813 s | 112.74 MiB |
| S_PACKET | 0.09369 s | 0.00803 s | 112.74 MiB |

This is 20 example exposures per sampler on one episode, without DDP or worker
overlap. It is not the requested full training pool and cannot forecast H200
runtime. The full all-task production caches, the 4,096-state production sampler
audit, required multi-task smoke, three 100k runs, and their final frozen panels
remain HPC work. No Slurm jobs were submitted by Codex.

The final FP32 local utility check is in
`runs/inspection/refinement_utility_final_20261110/`. Eight states from that one
episode produced 9,664 compact and 64 shared teacher velocity evaluations. The
worker took 4.42 seconds after startup, of which 3.42 seconds were serial
cache/context assembly. Peak allocated/reserved GPU memory was 198/292 MiB.
Two numerical-control states had a maximum serial/batched loss difference of
2.16e-7, below the 1e-6 near-tie deadband; duplicate contexts were identical.
The preceding BF16 pilot exceeded the deadband, motivating FP32 frozen
diagnostics without changing BF16 student training. Resume preserved committed
rows byte-for-byte, and an independent final rerun reproduced the FP32 arrays.

The gain-blind VideoUnmask episode-0/frame-36 generated check used two fresh
noise draws. Packet-selected completion lowered generated-action MSE versus
immediate-selected completion, but first-ten TCP target paths differed by only
1.09–1.26 mm. In four subsequent simulations all branches grasped container 0
and succeeded after the same verified expert prefix. That prefix already brought
the TCP nearest the correct container. These results therefore provide no
differential task-choice advantage; they verify the pipeline and show that all
four tested contexts retained sufficient behavior at this particular handoff.

The episode's original local partition label remains validation; this explicit
single-asset check is exploratory, not production final confirmation. All eight
hypotheses remain inconclusive for the required matched study. The complete
matrix needs the new students and multi-episode HPC measurements. The tracked
[evidence record](evidence/refinement_utility_20261110.json) links all artifacts.
Full CPU validation passed 1,739 tests (four expected skips), followed by 108
focused tests after final additions. No local full-data training was launched.
