# Portable local router diagnostics

The exact development checks need more than checkpoint files. The workstation's
older VideoUnmask-only diagnostic cache was encoded by a different teacher and
cannot replace the accepted 140k teacher's production evidence.

`scripts/export_utility_router_bundle.py` exports the fixed experiment inputs on
the HPC, using CPU reads of the existing encoded/hierarchy caches. It does not
encode images, optimize weights, select replacement cases, or run simulation.
The export includes the accepted teacher, all three matched 95k students, the
four completed routers, the pinned CLIP backbone, and the fixed decision inputs.
It also includes original provenance records, router fit metrics and available
generated-action results. It avoids copying whole raw/prepared datasets or the
dense visual feature cache.

The causal hierarchy is compacted to active rows for transport. Node identities,
canonical leaf reduction order, metadata, exact coverage, current tokens, task
embeddings, target masks and causal clocks remain unchanged. This is an explicit
portable adoption of sealed inputs, not a path rewrite of the original study.
The bundle has a checksum inventory and the local run has its own identity.

Commit and push the implementation, then pull the same revision on the HPC.
Wait for evaluations using the main checkout to finish before updating that
checkout. Keep both machines on the same scientific source for export and local
execution. The old pilot is validated using its recorded source.

## Transfer through the laptop

Run the following in a **laptop terminal**. It first waits for a compute-node
allocation on the HPC and creates the bundle. The allocation reserves one B200
because that partition supplies the available compute nodes, but the export
itself is CPU-only (`CUDA_VISIBLE_DEVICES` is empty).

```bash
(
set -eu

STMEM_BUNDLE="router_std95k_portable_$(date -u +%Y%m%dT%H%M%SZ)"
STMEM_LOCAL="$HOME/Desktop/chronotree_local/$STMEM_BUNDLE"

ssh peano bash -s -- "$STMEM_BUNDLE" <<'BASH'
set -euo pipefail
cd /hpc/home/phi/rvalperga/chronotree

srun --partition=aiq --gres=gpu:b200:1 \
  --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=32G --time=02:00:00 \
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -u scripts/export_utility_router_bundle.py \
    --pilot-name utility_router_std95k \
    --generated-name utility140k_generated95k \
    --output "runs/inspection/$1"
BASH

mkdir -p "$STMEM_LOCAL"
rsync -avP \
  "peano:/hpc/home/phi/rvalperga/chronotree/runs/inspection/$STMEM_BUNDLE/" \
  "$STMEM_LOCAL/"

ssh workstation \
  "mkdir -p /home/rvalperga/chronotree/runs/inspection/$STMEM_BUNDLE"
rsync -avP "$STMEM_LOCAL/" \
  "workstation:/home/rvalperga/chronotree/runs/inspection/$STMEM_BUNDLE/"

ssh workstation \
  "cd /home/rvalperga/chronotree/runs/inspection/$STMEM_BUNDLE && sha256sum -c SHA256SUMS"

printf '\nPortable inspection bundle: runs/inspection/%s\n' "$STMEM_BUNDLE"
)
```

This does not launch local GPU work. The export reports progress and its actual
size; transfer duration depends on that size and network throughput. Keep the
printed bundle name for resuming a transfer with the same rsync commands.

## Local execution after transfer

Use the printed bundle path instead of the placeholder below. Run GPU work only
when the workstation GPU is available:

```bash
cd /home/rvalperga/chronotree

uv run python scripts/inspect_utility_routers.py \
  --bundle runs/inspection/ROUTER_BUNDLE_NAME \
  --mode all --device cuda:0 --batch-size 16 --max-hours 4
```

`--mode routes` evaluates all seven complete allocations on the original 128
development states. It shares fresh action-noise draws across selectors and
reports full-horizon and executed-prefix FM losses. `--mode generated` reruns
the already fixed generated-action cases for the three students, preserving the
original generation seeds and integration settings. `--mode all` runs both.
These are prediction diagnostics; neither mode reports simulator task success.

Use `--resume` with the same inputs and batch setting after interruption. Use
`--mode status` for progress and `--mode report` to regenerate presentation from
completed results without running inference. Training and inference settings
are recorded separately from the untouched original manifests.

The transferred policy/router weights and CLIP backbone also provide the model
inputs for matched closed-loop evaluation using the workstation's installed
RoboMME simulator. That is a separate run with fixed validation seeds, preserved
action cadence and the same token budget. A single local GPU will not reproduce
the wall-clock throughput of the four-GPU HPC evaluation.

Verification on 16 November: 75 focused CPU tests passed across portable export,
local execution, complete routing, original router fitting/evaluation, cached
measurement and generated-action planning. Tests include the complete synthetic
128-state export, relocation without rewriting original paths, byte-identical
token banks and route identities, actual frozen-model FM and eight-step action
generation, artifact corruption checks, interrupted-export recovery and local
resumption. Ruff, transfer-shell syntax and the architecture snapshot check
passed. No full-data download, GPU evaluation or Slurm submission was performed.
