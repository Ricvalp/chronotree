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

Keep the [interactive teacher architecture diagram](docs/architecture_diagram.md)
in sync with the implementation. When changing encoders, metadata, context
assembly, action decoding, tensor shapes, or displayed batching settings, review
`scripts/generate_teacher_diagram.py` and regenerate the `.drawio`, `.html`, and
`.json` files in `docs/diagrams/` in the same change. Preserve the diagram's causal
boundaries and distinction between the full-history teacher and compact readers.
Run `uv run python scripts/generate_teacher_diagram.py --check`; this checks the
snapshot and XML structure, but does not replace reviewing the layer descriptions
against the code. Open the HTML or draw.io file to check layout after visual edits.
