# Frozen student utility measurements on B200s

This runs the development phase of the 140k-teacher utility study on the HPC.
It performs frozen-policy inference, not student or router training. No GPU on
the workstation is used.

Use the **95,000-step numbered snapshot of all three students** for this first
matched comparison. S_STD and S_STAGE reached 100k, but S_PACKET stopped near
96k. Comparing all three at their declared 95k development snapshot preserves
equal training exposure and permits a later common generated-action diagnostic.
The existing latest-checkpoint simulator results remain separate evidence.
The reserved confirmation comparison still requires the declared 100k endpoints;
these jobs do not relabel 95k as a completed primary endpoint.

The fixed panel contains 256 decisions across 16 tasks. These commands use its
128 fitting-tagged **development decisions**, which were excluded from the new
students' optimization. They leave the reserved validation half untouched.
Cache eligibility can reduce the number of measured decisions; the saved
manifests and reports retain those exclusions.

Every student receives the same states, contexts, candidate operations and noise
draws. Measurements include TREE and FLAT, B64/B128 allocation comparisons,
immediate versus completed-packet utility, controls, expert FM loss and teacher
agreement. The 32-state capability panel was selected across the entire
256-state panel: its development subset is included automatically, using
ordinary B128, detailed B128, B256, all available leaves where feasible, and
teacher FULL. These jobs do not run simulator episodes or make videos.

## Source and data identities

Pull the launcher changes on the HPC before running the commands below.
`--study-source` uses an isolated, ignored checkout of the exact Git revision
recorded in the existing study manifest. It verifies that this source matches
the manifest before using it for scientific measurements. It creates no new
branch and leaves the main checkout on its current branch.

The new name `router_utility_teacher140k_b200_dev95k` separates **launch plans
and logs**. The explicit `--manifest` retains the original panel, episode split,
student bindings and output root. There is no new panel, data preparation,
teacher encoding, hierarchy construction or cache migration. Existing study
identity checks remain in force; an unavailable or mismatched recorded revision
is an error, not permission to substitute current measurement code.

## Submit the measurements

Run on the HPC from `/hpc/home/phi/rvalperga/chronotree`. The first pass renders
and checks all three jobs without submitting. Submission begins only after all
three dry runs succeed. Each submitted job is an array of four independent
tasks, each requesting **one typed B200, 8 CPUs and 64 GB RAM on `aiq`**, with a
four-hour allocation limit. All three arrays can run concurrently: at most
12 B200s, 96 CPUs and 768 GB RAM in total. No exclusive-node request is used.

```bash
cd /hpc/home/phi/rvalperga/chronotree

(
set -euo pipefail

for action in --dry-run --submit; do
  for student in S_STD S_STAGE S_PACKET; do
    suffix=$(printf '%s' "$student" | tr '[:upper:]' '[:lower:]')

    .venv/bin/python hpc/refinement_utility.py \
      --site hpc/site_b200_4gpu.env \
      --name router_utility_teacher140k_b200_dev95k \
      --manifest runs/router_utility_teacher140k/manifest.json \
      --stage measure --student "$student" \
      --checkpoint "runs/utility_students_teacher140k_m64_${suffix}/seed0/compact/step_00095000.pt" \
      --panel-partition train \
      --study-source \
      --partition aiq --gpu-type b200 \
      --shards 4 --walltime 04:00:00 \
      "$action"
  done
done
)
```

The walltime is an allocation bound, not a runtime estimate. Actual throughput
has not been measured for this complete panel. The workers share validated
teacher references through per-state locks, so simultaneous students safely
reuse the same reference computations.

## Check progress

```bash
squeue -u "$USER"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

root = Path("runs/router_utility_teacher140k/measurements")
for student in ("S_STD", "S_STAGE", "S_PACKET"):
    directory = root / student / "step_00095000/main/train"
    for worker in range(4):
        path = directory / f"worker_{worker}.json"
        if not path.exists():
            print(student, worker, "no receipt yet; inspect the Slurm log")
            continue
        record = json.loads(path.read_text())
        print(student, worker, record["status"],
              len(record.get("completed_states", [])), "/",
              len(record.get("planned_states", [])), "states")
PY
```

Launch plans, submission receipts and logs:

```text
hpc/generated/router_utility_teacher140k_b200_dev95k/
hpc/generated/router_utility_teacher140k_b200_dev95k/logs/
```

Raw measurements and worker completion receipts:

```text
runs/router_utility_teacher140k/measurements/S_STD/step_00095000/main/train/
runs/router_utility_teacher140k/measurements/S_STAGE/step_00095000/main/train/
runs/router_utility_teacher140k/measurements/S_PACKET/step_00095000/main/train/
```

A completed state is committed as a checksum-validated `.npz` plus its `.json`
metadata. A worker reaching its bounded runtime can stop with a `partial`
receipt and exit code **75**. This means more states remain; it is not a completed
measurement or by itself a numerical failure. After the previous array is
inactive, repeat that student's identical submission command with `--resume`.
Completed states are verified and reused. Keep the checkpoint, shard count,
panel and source unchanged. A failed receipt instead includes the exception to
investigate before retrying.

## Build the report

Once all twelve worker receipts say `complete`, submit the report job. The
report itself is CPU work; this short allocation requests one B200 because the
launcher follows the GPU-partition workflow.

```bash
.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_b200_4gpu.env \
  --name router_utility_teacher140k_b200_dev95k \
  --manifest runs/router_utility_teacher140k/manifest.json \
  --stage report --study-source \
  --partition aiq --gpu-type b200 \
  --walltime 00:30:00 \
  --submit
```

The report writes:

```text
runs/router_utility_teacher140k/report.md
runs/router_utility_teacher140k/analysis.json
```

It can also summarize partial data, but completion and endpoint labels must be
retained when interpreting such a report. The 95k matrix is development evidence,
not the reserved 100k confirmation result.

## What follows

Inspect repeatable gains, absolute losses and capability comparisons before
selecting router training targets. A bounded common generated-action diagnostic
can then compare the matched 95k snapshots with an explicit development scope.
Its predicted paths are not simulator success rates. The later learned-router
pilot uses separate fitting states and student-specific labels; none of these
measurement jobs trains a router or consumes the reserved confirmation panel.

These instructions prepare user-submitted HPC work. No HPC allocation or
workstation GPU measurement was launched while writing this workflow.
