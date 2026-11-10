# Matched student training for refinement utility

The opt-in launcher `hpc/student_utility.py` prepares three new Stage P jobs from
the same accepted 104,110-step teacher. It never modifies S0, the teacher, or the
encoded/hierarchy caches. Default submission allocates two untyped GPUs on
`gpuq`, 64 CPUs, 128 GB RAM and 48 hours per independent job, with 16 loader
workers per rank. Runtime remains unmeasured on these new samplers.

At the user's request, each arm now runs **100,000 updates**, effective batch
128, microbatch 32/GPU and fresh AdamW. LR warms up to 1e-4 over 1,000 updates,
then follows the same cosine form to 1e-5 at update 100,000. Rolling `last.pt`
recovery checkpoints are saved every 1,000 updates. Immutable numbered snapshots
are saved every 5,000 updates, including 100,000. Only the predeclared 100k
endpoint is the primary result; earlier snapshots, including 15k and 30k, are
development checkpoints evaluated on the fitting partition only. Recovery
preserves the existing per-rank RNG, optimizer, scheduler and stop-marker state.

The three jobs total 300,000 optimizer updates and 38.4 million example
exposures. This is a requested schedule change, not a measured convergence or
runtime result. The panel protocol records:

```yaml
student_training:
  steps: 100000
  checkpoint_every: 1000
  snapshot_every: 5000
```

`S_STD` preserves the original 45/45/5/5 family mixture and budget sampling.
`S_STAGE` and `S_PACKET` replace half of TREE/FLAT examples with complete
coarse-to-budget construction routes, selecting the same target and training
prefix counts. Single-move and coherent-packet allocation are the only route
difference. The shared constructor handles short packets and FLAT singleton
normalization, and checks exact causal coverage at every prefix. Both route
arms use the same ordinary fallback if packet construction is infeasible.
No model loss or expert future action enters construction.

The immutable panel manifest supplies the training pool. All diagnostic
episodes and the original router-validation partition are excluded from new
optimization. Original catalogue indices and partition tags are retained;
sampling filters only the eligible optimization decisions. The accepted
teacher's normalization statistics remain unchanged. This is a holdout from
new training, not a claim that the inherited models never saw those episodes.

All three arms share per-global-example FM noise/time draws from a separate
generator, independent of context RNG consumption, batching and worker order.
Only the decoder/read adapter/clock train; encoders remain frozen. No action
alignment, mask, terminal-input convention, model shape or online policy input
changes are introduced.

On the HPC, after constructing the common utility panel manifest:

```bash
.venv/bin/python hpc/student_utility.py \
  --site hpc/site_b200_4gpu.env --cache-name memory_fv_seed0 \
  --manifest runs/refinement_utility/panel/manifest.json \
  --name utility_students_v1 --dry-run
```

The manifest path is an example; use the exact path printed by manifest
construction. The launcher verifies the recorded dependency identities. All
three configurations must render successfully before `--submit` submits any
allocation. Replace `--dry-run` with `--submit` after the sampler audit; add
`--variants S_STAGE` to operate on one arm, and `--resume` after its allocation
stops. It creates separate `runs/utility_students_v1_s_std`, `_s_stage` and
`_s_packet` roots. Plans, scripts and logs use corresponding directories in
`hpc/generated/`. W&B uses `chronotree-routers` for all three.

These commands assume fresh study names. If a panel or student plan was already
prepared under the old 30k schedule or an older source revision, use new panel
and student study names and prepare them again. Existing immutable manifests,
queued scripts and run ownership must not be overwritten. Source guards reject
code changed after rendering; wait for active source-bound jobs to finish before
updating their checkout. The launcher takes schedule defaults from the panel;
explicit `--steps`, `--checkpoint-every` and `--snapshot-every` values must match
that protocol. Existing 30k manifests/checkpoints retain their recorded schedule.

Run the common 4,096-state sampling audit on allocated CPU cores before training:

```bash
.venv/bin/python -m stmem.training.utility_sampler_audit \
  --config hpc/generated/utility_students_v1_s_std/compact.yaml \
  --count 4096 --output runs/refinement_utility/sampler_audit
```

This uses the same underlying states for every sampler and no policy inference.
It records actual counts, depths, fine-leaf fractions, source coverage, packet
lengths, within-packet prefixes, fallback frequencies and construction CPU time.
The complete context IDs remain in `contexts.jsonl`. These measurements do not
show whether a context is useful to the policy.

Each training run writes one-way task, phase, history-length, family, actual
budget and flow-time loss strata to `study_strata_rankN.jsonl`. Fixed task/phase/
family/flow-time summaries are also logged to W&B with explicitly rank-local
names. Existing metrics retain exposed loader wait, background construction
cost, GPU memory, examples/second, wall time and allocated GPU hours.

A lower-cost alternative is available with `--initialization continuation
--initial-checkpoint PATH`: all three arms copy the exact 90k S0 weights, reset
their optimizers identically, and by default run 10k updates at constant 1e-5.
Use a distinct `--name`. This alternative is not launched alongside the primary
teacher-initialized matrix.

The separate `configs/inspection/utility_student_followups.yaml` prepares bounded
rehearsal, learned-router and one-cycle adaptation protocols. They are explicitly
unactivated. `hpc/student_utility_rehearsal.py` can prepare the paired 10k-update
rehearsal continuation once a development finding, immutable parent checkpoint
and panel manifest are supplied. It requires `--evidence`, uses singleton
history batches with accumulation to preserve effective batch 128, and never
truncates FULL inputs. The control uses the same accumulation. Long-history
memory and throughput still require measurement before activating this branch.
Router adaptation needs a selected frozen scorer plus a matched route loader;
its configuration is prepared but that loader is not yet integrated. These
conditional branches do not delay the primary report.

## Local verification, 10 November 2026

99 focused CPU tests passed, including exact S_STD parity with the original
indexed sampler, common stage/packet exposure schedules, episode-boundary
rejection, paired per-example FM randomness, gradients through the intended
modules, and complete FULL rehearsal without truncation. Synthetic fixtures
are explicitly diagnostic.

The reusable `python -m stmem.inspection.utility_training_probe` additionally
completed a bounded real-data GPU probe on the accepted teacher and the local
VideoUnmask episode-0 cache. It ran five microbatch-4 updates for each sampler,
restoring the same teacher weights and fresh optimizer between variants.
Cached all-leaf context and velocity at initialization matched the serial
teacher exactly (maximum absolute difference zero, 1,056 historical leaves).
Frozen encoders received no gradients.

| Sampler | Mean serial loading, updates 2–5 | Mean GPU update, updates 2–5 |
|---|---:|---:|
| S_STD | 0.06246 s | 0.00796 s |
| S_STAGE | 0.09320 s | 0.00813 s |
| S_PACKET | 0.09369 s | 0.00803 s |

Peak allocated GPU memory was 118,214,144 bytes for all three, on an RTX 5000
Ada using BF16. This is one episode, 20 example exposures per arm, serial
loading, no DDP, and a small microbatch; it is not a prediction of full-data HPC
throughput. No checkpoint was saved or production student trained. The complete
record is `runs/inspection/utility_student_training_probe_20261110/results/summary.json`.
The all-task 4,096-context sampler audit and full 100k student jobs still require
the production caches on the HPC.

## Schedule amendment verification, 11 November 2026

The 100k change passed 126 targeted CPU tests across student scheduling, HPC
rendering, endpoint/partition guards, generated-action comparisons, router
planning, checkpoint cadence and compact resume. The synthetic launcher check
renders all three arms with 100k decay horizons and numbered snapshots through
`step_00100000.pt`, without contacting Slurm. Ruff and whitespace checks passed.
The immutable encoder-cache source fingerprint remains
`9cddd3ea461824003854f5b0c42595f5f1f975224d4f59ca0be72d0e1e83dc34`.
No substantive training or HPC submission was performed for this amendment.
