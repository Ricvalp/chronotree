# Matched router objectives on the frozen memory policies

`hpc/train_routers.py` prepares the requested velocity versus final-action
ablation using the existing selected full-history teacher, immutable leaf and
hierarchy caches, and one shared frozen compact checkpoint. It creates four
scorers: velocity TREE/FLAT and final-action TREE/FLAT. No policy or encoder is
trained by this launcher.

On the cluster, render the complete experiment with one command:

```bash
uv run python hpc/train_routers.py --site hpc/site_b200_4gpu.env
```

The defaults locate the existing `memory_fv_seed0` teacher-cache plan and
`runs/memory_fv_seed0_compact_hierarchy/seed0/compact/last.pt`. If the compact
checkpoint lives elsewhere, add `--compact /absolute/path/to/last.pt`.
The B200 site file supplies repository/interpreter/account locations; the new
jobs explicitly request partition `gpuq`, **one untyped GPU per allocation**,
and no exclusive node. They do not inherit that file's old teacher run root,
GPU type, CPU count, or `aiq` partition. Data/cache paths and all observation,
encoder, action, and stride settings come from the validated compact/cache
snapshots. The default four concurrent allocations use four GPUs in total.

The dry-run validates dependencies, copies the designated compact checkpoint
once, and writes the configurations, dependency graph, and scripts. It does
not contact Slurm or generate labels. Review the printed plan, then repeat the
command with `--submit` to submit it. There is no automatic resubmission.

The compact snapshot is saved as
`hpc/generated/memory_fv_seed0_routers/frozen/compact.pt`. Its adjacent
`.selection.json` records the original source, optimizer step, and SHA-256,
and explicitly says **user-designated checkpoint, not matched validation
selection**. This supports the requested fixed-checkpoint ablation without
inventing selection rollouts. Later changes to the source `last.pt` do not
change this experiment. Use a new `--name` to choose a different snapshot or
scientific configuration.

| Stage | Work | Allocation |
| --- | --- | --- |
| Initial labels | 16,000 fixed decision-context IDs per reader/objective; four candidates each | Array of four independent jobs, each 1 GPU, 16 CPUs, 32 GB, 12 hours |
| Initial scorers | Four scorers, 5,000 updates each | Four independent jobs, each 1 GPU, 4 CPUs, 16 GB, 1 hour |
| Refresh labels | 8,000 IDs per reader/objective using its own frozen 5k scorer plus exploration | Same four-job label array |
| Final scorers | Resume each scorer to 10,000 total updates on initial plus refresh labels | Same four independent scorer jobs |

This is ten submitted Slurm jobs (two arrays plus eight scorers), representing
16 one-GPU allocations in total, with at most four concurrent allocations.
Every dependency uses `afterok`; a checkpointed partial job exits nonzero so
it cannot release downstream work. CPU preparation defaults to four workers,
a four-context batch, a 32,768 teacher-token budget, a 64-context compact batch
cap, and 512 MB of prefetched prepared data. These are initial resource limits,
not measured throughput claims. Label IDs are partitioned by modulo shard
index; shard count and context totals are immutable within a named experiment.
The main inference batch has a separate 512 MB payload bound; an individually
oversized history runs alone without truncation. Worker topology/action caches,
temporary summary arrays and packed pinned buffers consume additional bounded
host memory. Workers receive the parent's validated small episode catalogue;
they do not rebuild the complete decision index.

Both readers use the same training/diagnostic episode split, task/budget
schedule, frozen policies, four-candidate allowance, and scorer architecture.
The initial context plan is shared across objectives; refresh uses the matching
objective's own TREE or FLAT scorer. Signed utility is discrepancy reduction
per added active token. The existing velocity objective evaluates a paired
expert/noise/time draw. The final-action objective starts teacher and every
compact context at the **same pure Gaussian normalized action chunk**, solves
each flow independently with the same solver, and compares masked normalized
final action chunks. Future expert commands are not used to initialize that
rollout and are never router features. Validity masks still restrict supported
action coordinates. One teacher prediction or rollout is reused across the
paired TREE/FLAT candidate evaluations for that decision and objective.

The default final-action label solve uses eight steps. `--integration-steps 4`
is an explicit cheaper ablation named `final_action_4`; it does not alter the
policy's evaluation solver settings (normally eight steps). Both values are
printed in the plan. This distinction must remain visible in reported results.

For a bounded cost pilot, render a separate named experiment:

```bash
uv run python hpc/train_routers.py --site hpc/site_b200_4gpu.env --name memory_fv_seed0_routers_pilot --contexts 256 --refresh-contexts 128
```

Add `--submit` only when ready to spend the pilot allocation. Context counts
are the totals for **each** reader/objective, split across four workers; they
are not multiplied by the worker count. A pilot does not replace the 16k/8k
protocol and its losses are conditional router diagnostics, not benchmark
success. Full-run GPU hours remain unknown until measured; record allocated
GPU hours by actual GPU type using the existing accounting tools. Changing
context totals, shard count, integration steps, preparation settings, or source
code requires a new named experiment. Walltime, memory, CPUs, and partition may
be adjusted when restarting the same immutable computation.

Artifact locations for the default run are:

- Plan, nine YAML configurations, ten job scripts, submission ledger, and
  immutable startup plans: `hpc/generated/memory_fv_seed0_routers/`.
- Standard output/error: `hpc/generated/memory_fv_seed0_routers/logs/`.
- Labels: `runs/memory_fv_seed0_routers/{velocity,final_action_8}/seed0/router_{tree,flat}/labels_{initial,refresh}/shard_00000.h5`
  through `shard_00003.h5`.
- Initial immutable scorer: each router directory's `step_00005000.pt`.
- Final immutable scorer: each router directory's `step_00010000.pt`; its
  `last.pt`, metrics, and completion records remain in that directory.

For the separately requested longer-fit diagnostic on unchanged initial labels,
see [router_overfit_hpc.md](router_overfit_hpc.md). It starts from the immutable
5k scorer in a new output directory; it does not change this dependency chain.

Inspect artifact completion without contacting Slurm:

```bash
uv run python hpc/train_routers.py --site hpc/site_b200_4gpu.env --status
```

After **all previous allocations are inactive**, resume incomplete stages:

```bash
uv run python hpc/train_routers.py --site hpc/site_b200_4gpu.env --resume --submit
```

The ledger rejects duplicate active jobs. Completed label contexts are reused,
completed stages are skipped, and unfinished scorers resume their own `last.pt`.
The 5k refresh inputs are immutable step checkpoints, so final fitting cannot
change the refresh policy. Allocation startup checks source/configuration,
checkpoint and cache hashes before model work. A signal-safe wrapper writes a
fresh stop marker for each allocation; label workers commit coherent progress
and scorers save at optimizer boundaries. If submission was interrupted after
its intent was recorded but before its returned job ID was saved, the ledger
reports an uncertain submission and refuses another submission until the entry
has been reconciled with Slurm. Pending descendants of a failed attempt also
count as active; resolve them before resuming the chain.

Verification on 4 November: 308 focused CPU-safe tests passed, including indexed
causal bank/reference parity, four/eight-step final-action gain parity, exact
episode partitions, worker spawning, shard interruption recovery, four initial
scorers and objective-specific refresh/resume. Slurm is mocked in orchestration
tests. The actual RTX 5000 Ada passed `stmem.doctor` in FP32/BF16/FP16. A bounded
synthetic CUDA smoke check compared eight gain vectors between batched and
single-example inference with eight integration steps, then performed one finite
router optimizer update. These synthetic checks establish implementation
consistency, not policy quality or full-data speed. Real-data labeling throughput,
multi-GPU allocations and downstream evaluation have not been run.
A revision-level verified alignment record and all completed
frozen caches are still required by the real-data runtime. Evaluation on official
test seeds and any change to the evaluation protocol remain separate actions.

The existing teacher and hierarchy caches remain compatible. Only the exact
label/scorer consumer modules were added to the audited cache-source exclusion
list; encoder/preprocessing/alignment/aggregation source and all teacher/data
identities remain protected. Label files and scorer checkpoints separately record
the full objective, solver, source, sample-plan and frozen-dependency provenance.
