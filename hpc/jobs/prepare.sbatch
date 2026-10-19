#!/usr/bin/env bash
#SBATCH --job-name=stmem_prepare
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
# CPU preprocessing on a GPU-only queue; the allocated GPU is not used.
# Submit from the repository root after creating hpc/logs. Resource changes
# require literal sbatch overrides; site.env is read after Slurm schedules us.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit prepare.sbatch with sbatch, or run the documented preparation CLI inside an allocation." >&2
  exit 64
fi
source "${STMEM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/hpc/jobs/common.sh"

STMEM_PREP_CPUS=${SLURM_CPUS_PER_TASK:-${STMEM_CPUS:-16}}
if [[ ! "$STMEM_PREP_CPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SLURM_CPUS_PER_TASK (or STMEM_CPUS) must be a positive integer." >&2
  exit 64
fi
# Limit arithmetic to short validated numbers, even for malformed site values.
if (( ${#STMEM_PREP_CPUS} > 2 )) || (( STMEM_PREP_CPUS > 16 )); then
  STMEM_PREP_LIMIT=16
else
  STMEM_PREP_LIMIT=$STMEM_PREP_CPUS
fi
STMEM_PREP_WORKERS=${STMEM_PREP_WORKERS:-$((STMEM_PREP_LIMIT < 8 ? STMEM_PREP_LIMIT : 8))}
if [[ ! "$STMEM_PREP_WORKERS" =~ ^[1-9][0-9]*$ ]] \
  || (( ${#STMEM_PREP_WORKERS} > 2 )) \
  || (( STMEM_PREP_WORKERS > STMEM_PREP_LIMIT )); then
  echo "STMEM_PREP_WORKERS must be an integer from 1 to $STMEM_PREP_LIMIT (allocated CPUs, capped at 16 tasks)." >&2
  exit 64
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# Preparation restarts interrupted tasks; it does not poll the training stop file.
# Keep worker controls last so optional task/config arguments cannot override them.
exec srun --ntasks=1 "$STMEM_PYTHON" -u -m stmem.data.prepare \
  --config "${STMEM_CONFIG:-configs/teacher.yaml}" --episodes 100 "$@" \
  --workers "$STMEM_PREP_WORKERS" --threads-per-worker 1 --skip-existing
