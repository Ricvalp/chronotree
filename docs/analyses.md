# Bounded analyses after the central comparison

These commands are prepared diagnostics. No analysis campaign or benchmark score
is implied by their existence. Use frozen selected checkpoints and the same
declared official validation episode subset. Official test seeds remain reserved
for the frozen protocol.

## Memory sensitivity

`stmem.analyze.intervene_evidence` supports `erase_execution`,
`permute_execution`, and `erase_conditioning`. It preserves the current scene,
proprioception, instruction, source clocks, and target/noise protocol. Execution
interventions preserve all conditioning leaves.

The permutation shuffles execution **leaf payloads across recorded-time slots,
separately within each modality**, then rebuilds deterministic local/temporal
trees. It preserves unit leaf mass and the content multiset but disrupts temporal
association. This is an encoded-evidence sensitivity check, not a physically
plausible alternate trajectory. It differs from merely shuffling the order of an
attention token list, which would not change this policy. New evidence IDs prevent
frozen routing keys from being silently reused after the intervention.

For a cheap offline check, the command below compares fixed-noise predicted
physical commands with UNIFORM contexts. The JSON labels this as action
sensitivity, not closed-loop success or proof of non-Markovianity:

```bash
uv run python -m stmem.analyze --config configs/debug/compact.yaml \
  --mode memory --episode-index 0 --decision-index 5 --budget 16 \
  --output runs/analysis/debug_memory.json
```

The same intervention helper can be applied by the evaluation runner before
routing. A closed-loop comparison must reuse the same official validation seeds,
reader/checkpoint, budget, flow-noise convention, and execution horizon. Erasing
conditioning and erasing execution answer different questions, especially for
video-conditioned tasks.

## Tiny exhaustive routing diagnosis

```bash
uv run python -m stmem.analyze --config configs/debug/compact.yaml \
  --mode routing --episode-index 0 --decision-index 1 --budget 6 \
  --max-leaves 24 --max-states 4096 --seed 0 \
  --output runs/analysis/debug_routing.json
```

The command uses one expert/noise/time draw for the full-leaf teacher and every
compact context. It enumerates every feasible TREE frontier and every FLAT
selected-plus-residual context at the requested budget on the tiny available
archive. Singleton residuals use the same zero-cost accounting as training and
inference. It reports the best teacher-velocity agreement within each reader's
feasible family, compact-full agreement, and UNIFORM agreement.

Optional `--tree-router PATH` and `--flat-router PATH` add learned-reader results
using the same draw and evidence. This helps distinguish a reader's search error
from its constrained context family's best attainable teacher agreement. These
are action-utility diagnostics, not control-optimal or task-success oracles.

`--max-leaves` and `--max-states` are hard guards. Exceeding either fails with a
request to choose a smaller causal prefix or explicitly adjust the diagnostic
cap. The tool does not silently truncate history or call a partial search exact.
No model is trained by this command.

## Budgets, time, and storage

The primary variable-budget policy and both scorers train across B={64,128,256}.
Budget curves should use those same frozen checkpoints and fixed evaluation
seeds, then join success with complete online decision latency and archive/key
storage. A smaller token count by itself is not an efficiency result.

Candidate keys are prepared in batches when the immutable archive changes.
TREE prepares keys for all archive nodes and FLAT for all leaves. Completed keys
are reused, while replaced prefix keys are evicted. The later tree traversal is
CPU-only and does not launch a GPU key tower at each split. The new-key preparation
cost, candidate-feature transfers, key transfers, query transfer, retained key
bytes, pending-prefix encoding, and archive updates all belong in measured costs.
The O(B) tree node-score accounting excludes no hidden key construction cost.

Use naturally occurring short/middle/long histories with the complete online
pipeline; frozen training-cache speed is not online encoding speed. Report warm
and cold measurements, synchronize accelerator timers, and record hardware by
its actual reported model. CPU archive payload is distinct from Python allocator
and tensor backing-storage overhead; report process memory as well as logical
payload counters when measuring storage.

The framewise encoder remains an opt-in attribution configuration with the same
center quotas and event leaves. Its matching teacher/shared-reader run follows
the central TREE/FLAT comparison, rather than expanding the default campaign.
