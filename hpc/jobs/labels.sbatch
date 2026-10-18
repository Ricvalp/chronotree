#!/usr/bin/env bash
#SBATCH --job-name=stmem_labels
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --signal=B:USR1@180
# Defaults above are literal. Planner renders site-specific resources before submit.
# Submit from the repository root after creating hpc/logs.
source "${STMEM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/hpc/jobs/common.sh"
stmem_run -m stmem.training.gain_labels --reader "${STMEM_READER:-tree}" --config "${STMEM_CONFIG:-configs/router_${STMEM_READER:-tree}.yaml}" "$@"
