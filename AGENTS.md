# Repository guidance

Read `PROMPT_CHRONOTREE.md` for the scientific contract. Never weaken causal
alignment, exact leaf coverage, or matched TREE/FLAT supervision to make a run pass.
Substantive real-data training requires a verified revision-level alignment record.
Synthetic checks must remain explicitly labeled diagnostics.

Use `uv run pytest` for CPU-safe invariants and `uv run python -m stmem.doctor`
for the actual device. No full-data download, long training, Slurm submission,
or test-seed tuning without explicit user authorization. The planner defaults
to dry-run; only `--submit` submits jobs. Keep generated data and weights ignored.
Record implementation decisions, measured checks, and blockers honestly in docs.
