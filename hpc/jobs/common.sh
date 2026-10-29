#!/usr/bin/env bash
# Loaded by the human-readable one-stage templates. Prefer hpc/plan.py for DAGs.
set -euo pipefail
STMEM_SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STMEM_SITE_FILE=${STMEM_SITE_FILE:-$STMEM_SCRIPT_ROOT/hpc/site.env}
if [[ -f "$STMEM_SITE_FILE" ]]; then
  set -a
  source "$STMEM_SITE_FILE"
  set +a
fi
STMEM_REPO_ROOT=${STMEM_REPO_ROOT:-$STMEM_SCRIPT_ROOT}
STMEM_PYTHON=${STMEM_PYTHON:-$STMEM_REPO_ROOT/.venv/bin/python}
STMEM_RUN_ROOT=${STMEM_RUN_ROOT:-$STMEM_REPO_ROOT/runs}
STMEM_SEED=${STMEM_SEED:-0}
STMEM_NPROC=${STMEM_NPROC:-${SLURM_GPUS_ON_NODE:-1}}
if [[ -n "${STMEM_ACTIVATE:-}" ]]; then source "$STMEM_ACTIVATE"; fi
cd "$STMEM_REPO_ROOT"
STMEM_STOP_FILE="$STMEM_RUN_ROOT/stops/${SLURM_JOB_ID:-local_$$}.stop"
export STMEM_STOP_FILE STMEM_RUN_ROOT
stmem_train() {
  if [[ -n "${STMEM_SCRATCH_ROOT:-}" ]]; then
    STMEM_FEATURE_CACHE_ROOT=$("$STMEM_PYTHON" -m stmem.data.stage_features \
      --config "$2" --scratch-root "$STMEM_SCRATCH_ROOT" --print-path)
    export STMEM_FEATURE_CACHE_ROOT
  fi
  exec "$STMEM_REPO_ROOT/hpc/run_job.sh" "$STMEM_STOP_FILE" \
    srun --ntasks=1 "$STMEM_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
    --nproc-per-node="$STMEM_NPROC" -m stmem.train --stage "$1" --config "$2" --stop-file "$STMEM_STOP_FILE" "${@:3}"
}
stmem_run() {
  exec "$STMEM_REPO_ROOT/hpc/run_job.sh" "$STMEM_STOP_FILE" srun --ntasks=1 "$STMEM_PYTHON" "$@"
}
