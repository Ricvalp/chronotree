#!/usr/bin/env bash
# Execution body for hpc/profile_memory.py's generated, site-specific sbatch file.
# Use that renderer; direct sbatch submission would omit GPU/resource requests.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Run this profiling body inside the allocation rendered by hpc/profile_memory.py." >&2
  exit 64
fi
: "${STMEM_REPO_ROOT:?Missing repository path; use hpc/profile_memory.py}"
: "${STMEM_PYTHON:?Missing Python path; use hpc/profile_memory.py}"
: "${STMEM_ALLOCATED_GPUS:?Missing GPU count; use hpc/profile_memory.py}"
case "$STMEM_ALLOCATED_GPUS" in
  1|4|8) ;;
  *) echo "Profiling requires 1, 4 or 8 allocated GPUs." >&2; exit 64 ;;
esac
cd "$STMEM_REPO_ROOT"
exec srun --ntasks=1 "$STMEM_PYTHON" -u -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node="$STMEM_ALLOCATED_GPUS" \
  "$STMEM_REPO_ROOT/scripts/profile_teacher_memory.py" "$@"
