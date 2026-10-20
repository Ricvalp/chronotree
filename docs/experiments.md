# Reproducible experiment protocol

`configs/experiment_core_seed0.yaml` is the bounded primary manifest. It uses all
100 released training demonstrations for each of the 16 tasks. Validation and
test are official seeded environment configurations; expert HDF5 for those splits
is neither assumed nor synthesized. The debug configs are implementation checks
and cannot supply benchmark scores.

The teacher trains for 100,000 updates with effective batch 128, microbatch cap
64 and encoder block batches of 128. With four GPUs this means 32 decisions per
GPU per update and no gradient accumulation. These settings live in
`configs/teacher.yaml`; compact and Perceiver batching defaults remain separate.
Teacher candidate updates are 50k and 100k in both experiment manifests.
The fixed checkpoint-selection subset is
official validation episode indices 0 and 1 in every task. Select by task-macro
closed-loop success, resolving ties with the earlier checkpoint. Then freeze that
teacher and its encoders, record the hash, and build immutable causal caches.

The compact policy is initialized from the selected teacher and trained with frozen
encoders on 45% random tree, 45% flat with residuals, 5% uniform and 5% current-only
contexts. Budgets 64, 128 and 256 and intermediate valid sizes are represented.
Compact candidate updates are 15k and 30k. Both candidates use the same fixed two
validation indices per task, once with each context family, and aggregate the two
families equally. These deterministic selection contexts are declared by
`--selection-family`; they require no trained router. The selected compact weights
are shared by TREE, FLAT, UNIFORM and CURRENT.

Each reader gets 16k initial decision-context pairs, four candidate moves per pair,
one 8k-context refresh, and 10k total scorer updates split at 5k. Train and diagnostic
label partitions are split by episode within each task (80/20) before generation;
the refresh preserves that partition. Both scorers have equal architecture and
supervision budgets. Label prediction is a conditional router diagnostic: these
episodes may have trained the teacher or compact policy. It is not unseen-episode
end-to-end generalization. Training action losses likewise remain explicitly
training diagnostics.

The primary comparison is FULL, TREE, FLAT, UNIFORM, PERCEIVER and CURRENT at B=128
where applicable. The six-method validation phase uses 50 official seeds/task.
After reviewing validation, freeze the protocol and checkpoint/configuration hashes,
then invoke the separate final-test plan with all 50 official test seeds/task.
Every method receives the same task seeds and noise-seed protocol. Policy-induced
errors remain failed planned episodes. Infrastructure errors retain their original
seed and error details; retry only a diagnosed infrastructure cause.

`python -m stmem.report --run-root runs --split val` produces JSON/CSV tables and a
plot only when measured completed outcomes exist. Missing methods say **not yet
run**, never zero success. Partial results expose their planned denominator and
lower-bound rate. Task-macro averaging gives tasks equal weight. TREE–FLAT intervals
use task-stratified paired episode bootstrap and reject incomplete pairing instead
of silently taking an intersection. With several training seeds, the report also
shows across-training-seed variation separately from evaluation variability. A
single seed supports an engineering screen, not a training-robustness claim.

Per-decision active counts, archive and index bytes, transfers, fetched/scored
records, component times and full online latency accompany outcomes. First
decisions are reported separately from later decisions. Online encoding and archive
updates remain timed even if training uses cached leaves. Reserved and allocated
CUDA peaks are separate measurements. Slurm accounting supplies full allocated
GPU hours; no total or speedup exists before measurement.

`experiment_central_3seed.yaml` prepares seeds 0, 1 and 2 for the central shared
pipeline and FULL/TREE/FLAT/UNIFORM comparison only. It is a final replication plan,
not a preliminary multiseed gate. Do not duplicate already completed seed-0 teacher,
cache or compact work with matching configuration and provenance; use the
planner's resume controls. Runs made with the earlier 50k-update teacher
configuration are not interchangeable with the current 100k-update run.

Analyses are explicit follow-up invocations, not default grids: budget 64/128/256
for TREE/FLAT/UNIFORM; separate execution-history and conditioning-history erasure;
execution permutation with current and conditioning inputs fixed; complete timing
at naturally short/medium/long prefixes; frozen-checkpoint routing diagnosis on
tiny prefixes; and one later matched framewise tokenizer run. No analysis may tune
on official test episodes or use future/oracle features. A poor result motivates
a bounded diagnosis, not unbounded label generation or automatic sweeps.

All scientific training, checkpoint-selection rollouts and benchmark success
comparisons are **not yet run** at implementation time. Consult run artifacts for
measured local tests; synthetic loss reductions are never benchmark evidence.
