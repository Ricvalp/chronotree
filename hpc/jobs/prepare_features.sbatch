#!/usr/bin/env bash
#SBATCH --job-name=stmem_features
#SBATCH --partition=aiq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%A_%a.out
#SBATCH --error=hpc/logs/%x_%A_%a.err
# Each array element prepares one task with one GPU. The task order comes from
# the selected config; both experiment arms share the same immutable sidecars.
source "${STMEM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/hpc/jobs/common.sh"
STMEM_CONFIG=${STMEM_CONFIG:-configs/teacher_fv_prop.yaml}
mapfile -t STMEM_FEATURE_TASKS < <("$STMEM_PYTHON" - "$STMEM_CONFIG" <<'PY'
import sys
from stmem.config import load_config
print("\n".join(load_config(sys.argv[1])["data"]["tasks"]))
PY
)
STMEM_FEATURE_INDEX=${SLURM_ARRAY_TASK_ID:-0}
if [[ "$STMEM_FEATURE_INDEX" -ge "${#STMEM_FEATURE_TASKS[@]}" ]]; then
  echo "Feature task index is outside the selected configuration" >&2
  exit 64
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
stmem_run -u -m stmem.data.prepare_features --config "$STMEM_CONFIG" \
  --tasks "${STMEM_FEATURE_TASKS[$STMEM_FEATURE_INDEX]}" --device cuda \
  --threads 2 --skip-existing "$@"
