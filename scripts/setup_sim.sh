#!/usr/bin/env bash
set -euo pipefail
STMEM_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$STMEM_REPO_ROOT"
# Core and simulator pins are compatible: NumPy1.26.4, Torch2.9.1/CUDA12.8.
# No system drivers, cluster modules or privileged packages are changed.
uv sync --frozen --extra simulator
