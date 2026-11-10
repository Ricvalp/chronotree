# Generated-action follow-up for the student utility study

The primary measurement uses expert FM velocity loss. This separate bounded
follow-up asks whether the measured allocation choices also change ordinary
eight-step generation from pure Gaussian noise. It produces predicted
joint-target/TCP paths, not simulator success rates.

`stmem.inspection.refinement_generated` provides separate `plan`, `generate`,
and `report` modes. Planning reads only the diagnostic panel's **development
episodes and discovery draws**. Confirmation losses and the reserved validation
episodes cannot select an illustrative case. Every requested student receives
every retained state, reader and background; each student's immediate-selected
and packet-selected completed contexts use that student's own discovery losses.
Base and dispersed controls are common. Teacher FULL is the reference.

The bounded union nominates cases symmetrically from each student's rankings of
delayed packet value, paired background contrast, expert/teacher preference
conflict and near-null gains. It keeps at most 16 unique states, at most four
per task and one per episode. It reports shortfalls rather than relaxing those
caps or replacing cases after generation. Discovery near-ties use the first
stable candidate ID as a documented representative; the full tie sets are
retained. This differs from the main loss report's averaging over tie sets.

## Primary matrix after the 100k endpoints

Using the default study names, after development utility measurements have
completed for all four students:

```bash
uv run python -m stmem.inspection.refinement_generated \
  --mode plan \
  --manifest runs/router_utility_v1/manifest.json \
  --plan runs/router_utility_v1/generated_plan.json \
  --checkpoints \
    S_STD=runs/utility_students_v1_s_std/seed0/compact/step_00100000.pt \
    S_STAGE=runs/utility_students_v1_s_stage/seed0/compact/step_00100000.pt \
    S_PACKET=runs/utility_students_v1_s_packet/seed0/compact/step_00100000.pt
```

S0 defaults to the exact frozen checkpoint in the parent manifest. The other
paths must match the hashes of the measured 100k endpoints. Planning supports
the endpoint/partition directory layout and rejects final-only measurements.
An explicitly configured matched continuation study instead uses its declared
common continuation endpoint. The planner rejects mixed initializations,
different declared endpoints, or different observed snapshot steps between new
arms. Existing studies with an immutable 30k declaration retain that endpoint;
use their original matching paths rather than relabeling them as the new study.

Run generation on an allocated GPU:

```bash
uv run python -m stmem.inspection.refinement_generated \
  --mode generate \
  --plan runs/router_utility_v1/generated_plan.json \
  --device cuda:0 --batch-size 16 --max-hours 1
```

This runs the students sequentially. To assign independent GPU allocations,
use the same command with `--student S0`, `--student S_STD`, `--student S_STAGE`
or `--student S_PACKET`; every process uses the same immutable case plan.
The time bound saves completed student/state cells and returns exit code 75
for a partial stage. Repeating the same command validates checksums and resumes
missing cells. Altered checkpoints, inputs or source identities require a new
experiment rather than silently changing an existing result.

Outputs appear beside the plan under `generated_plan_results/`:

- Compact NPZ action/path arrays and per-state metrics.
- `summary.json`, `report.md` and self-contained `index.html`.
- Explicit complete/partial receipts and missing matrix cells.

The HTML shows the same current visual-token anchors for each alternative,
the first ten predicted targets as a solid path and the remaining six as a
dashed tail. Anchors are not object masks or a full point cloud. The FK paths
use the task's actual Panda/stick variant and controller clipping/gripper
conversion. Two fresh Gaussian seeds are shared across every student and
context. Metrics include full/prefix/tail expert errors, teacher discrepancy,
paired normalized-action differences and TCP-target distances.

Regenerate presentation without model inference:

```bash
uv run python -m stmem.inspection.refinement_generated \
  --mode report --plan runs/router_utility_v1/generated_plan.json
```

## Limited local checks

When only S0 and a small local cache are present, an explicitly incomplete
development probe is possible. It does not replace the four-student matrix:

```bash
uv run python -m stmem.inspection.refinement_generated \
  --mode plan \
  --manifest runs/inspection/refinement_utility_final_20261110/manifest.json \
  --plan runs/inspection/refinement_utility_final_20261110/generated_plan.json \
  --students S0 --measurement-mode smoke --limited-check --max-cases 2
```

The generated-action runner does not itself execute simulations. The separate
`stmem.inspection.refinement_continuation` wrapper supports the bounded local
VideoUnmask asset check below. It verifies a genuine expert prefix, executes
one selected chunk's ten targets held twice, then rebuilds a common UNIFORM
reader as new observations arrive. Ordinary from-reset evaluations are not
substituted for this intervention.

If the only local cached episode belongs to the original validation partition,
the development selector correctly produces no ordinary cases. An explicit
asset-only check can instead use **one predeclared state**, without relabeling
the episode or treating it as production confirmation:

```bash
uv run python -m stmem.inspection.refinement_generated \
  --mode plan \
  --manifest runs/inspection/refinement_utility_final_20261110/manifest.json \
  --plan runs/inspection/refinement_utility_final_20261110/generated_asset_plan.json \
  --students S0 --measurement-mode smoke --limited-check --max-cases 1 \
  --diagnostic-state-id 206558cd6af57114245f
```

This option requires a manifest explicitly marked
`limited_real_data_diagnostic`. It verifies that the requested state is the
first cached VideoUnmask decision at or after native frame 32 (the available
episode currently yields frame 36), fixes TREE/background 120 before reading
losses, and uses discovery values only to choose the two completed alternatives.
The HTML and report preserve the original partition and label this as an
exploratory local asset check. The ordinary multi-student generated matrix still
selects only from development episodes.

After generation completes, the separate bounded four-branch continuation is:

```bash
uv run python -m stmem.inspection.refinement_continuation \
  --plan runs/inspection/refinement_utility_final_20261110/generated_asset_plan.json \
  --output runs/inspection/refinement_utility_final_20261110/generated_asset_continuation \
  --assets-config configs/inspection.yaml \
  --draw 0 --cap-native-commands 320 --seconds-per-branch 300 --device cuda:0
```

It compares base, immediate-selected completion, packet-selected completion and
dispersed control from the same state/noise, writes videos and choice diagnostics,
and labels the expert-prefix handoffs separately from autonomous benchmark
success. The intervention affects the first generated chunk only.

## Conditional later router stage

`configs/inspection/refinement_router_conditional.yaml` is a disabled, concrete
follow-up specification, independent of the first student matrix. The planning
CLI binds at most two explicitly selected frozen students, validates the common
fitting pool exclusions, and records 4,096 fitting states/student, matched
scorer budgets, observable features, controller versions, compute bounds and
the crossed student/router matrix:

```bash
uv run python -m stmem.inspection.refinement_router_plan \
  --manifest runs/router_utility_v1/manifest.json \
  --output runs/router_utility_v1/conditional_router_plan.json \
  --checkpoints \
    S_STD=runs/utility_students_v1_s_std/seed0/compact/step_00100000.pt \
    S_PACKET=runs/utility_students_v1_s_packet/seed0/compact/step_00100000.pt
```

This is preparation only, with **no submit or training action**. Activation
still needs a new expert/packet-label schema and writer, observable
cost/remaining-budget feature integration, a deployed scorer-to-packet
controller, and an explicit crossed-student evaluation protocol. The production
router's exact-student checkpoint check remains intact. Existing velocity or
final-action teacher-agreement labels cannot be renamed into expert packet
utilities. Each missing integration is retained in the prepared plan's
`activation_blockers`; no router result is claimed.
