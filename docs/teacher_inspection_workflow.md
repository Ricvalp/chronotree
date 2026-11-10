# Repeatable teacher continuation inspection

[`scripts/inspect_teacher.py`](../scripts/inspect_teacher.py) runs the usual
checks, history controls, comparisons and reports. Only the transferred bundle
path needs to change. The default reference is the saved **104,110-step**
teacher, declared in [this reference recipe](../configs/inspection_teacher_reference.yaml).
Reference simulations are read, not rerun. A bundle contains `SHA256SUMS` and
one `frozen_10hz/checkpoint.pt`, as in the existing laptop transfer workflow.

## Run and resume

From the workstation repository, substitute the transferred directory name:

```bash
uv run python scripts/inspect_teacher.py \
  --bundle runs/inspection/NEW_BUNDLE \
  --detach
```

`--detach` starts a bounded coordinator independent of the terminal or assistant
tool session and prints its PID and log path. The default uses two workers on
local `cuda:0` for history simulations, runs stages sequentially, and bounds
child-stage execution to four hours per invocation. CPU report generation adds
a small final overhead. `--workers 4` increases concurrency without changing
trials or settings. No training, downloads, Slurm submission or test-seed tuning
is performed.

Outputs default to `runs/inspection/NEW_BUNDLE_teacher_inspection/`; `--output`
can select another directory. Inspect progress with:

```bash
cat runs/inspection/NEW_BUNDLE_teacher_inspection/status.json
tail -n 30 runs/inspection/NEW_BUNDLE_teacher_inspection/logs/history.log
```

The launcher emits progress every 30 seconds. Per-worker logs are under
`history/logs/<attempt>/` and `recent/logs/<attempt>/`. History evaluations use
the `chronotree-routers` W&B project; the usual suite keeps its existing local
logging behavior.

After interruption, repeat the same command with `--resume`:

```bash
uv run python scripts/inspect_teacher.py \
  --bundle runs/inspection/NEW_BUNDLE \
  --resume --detach
```

Resume verifies identities and fills missing history trials only. Completed
usual stages are revalidated through their CPU report path. Published success,
failure and normal-timeout results/media remain unchanged; error records require
diagnosis and are not silently retried. An unpublished interrupted episode
restarts from its original scene/seed, not an arbitrary simulator state.
W&B run IDs are preserved. Worker concurrency may change; source, checkpoint,
assets, reference and scientific settings must remain fixed for one inspection.
Locks prevent overlapping writers. Older unsealed results remain valid
comparison references but are not silently adopted as resumable runs.

## What runs

1. **Usual checks:** checkpoint integrity, fixed expert-state prediction and
   sensitivity, fresh PickXtimes grasp/lift, four-position cube point clouds,
   and expert-prefix handoffs. This retains the usual four-native-command
   replanning interval: two 10 Hz targets, each held for two 20 Hz commands.
2. **History controls:** FULL, erased initial video, erased execution history
   and CURRENT, on validation scenes 0–3 of SwingXtimes, VideoRepick and
   VideoPlaceOrder. These 48 trials predict 16 targets at 10 Hz, execute ten,
   and use eight flow integration steps. Controls use the same teacher.
3. **Recent window:** four fresh Swing trials retaining two completed
   eight-frame blocks plus the pending prefix. Four FULL controls are copied
   and verified, not simulated again.
4. **Audit/comparison:** checkpoint success differences, matched resets/noise,
   actual held-action accounting, same-current probes, subgoal progress,
   grasp/lift and placement measurements, complete video decoding and links.
   Source/runtime differences are flagged for review, not assumed equivalent.
5. **Reports:** landing page, numerical comparisons, paired videos and a list
   of changed outcomes to prioritize for manual inspection.

The simulations were already scripted. This workflow also replaces the old
checkpoint-specific comparison/recovery helpers. Selected videos still require
human review; automatic measurements do not claim semantic recall or pretend
that someone watched a video. These are fixed-scene diagnostics, not a new
all-task success estimate.

## Outputs

| Path within the output directory | Contents |
|---|---|
| `index.html`, `summary.md`, `summary.json` | Main results and linked detailed reports |
| `review_queue.json` | Changed outcomes to inspect |
| `usual/index.html` | Usual checks, point clouds, rollouts and handoffs |
| `history/index.html`, `recent/index.html` | Control videos and reference comparisons |
| `comparison/history_comparison.html` | Reference/current paired videos and history benefit |
| `comparison/usual_comparison.json` | Expert-state, grounding and handoff changes |
| `comparison/behavior_comparison.json` | Automatic behavior measurements and suggested video review |
| `teacher_inspection.json`, `status.json` | Immutable protocol and current progress |

`--dry-run` validates the transfer/reference and prints commands without
creating results. `--audit-only` runs full CPU checkpoint/asset verification;
`--resume` then launches remaining stages. `--report-only` rebuilds and validates
reports in an existing completed suite without simulations.

Use `--reference /path/to/completed-suite` for another reference checkpoint,
or provide a YAML following the default recipe. A standalone CPU comparison
also works for previous outputs:

```bash
uv run python scripts/compare_teacher_history.py \
  --reference-history runs/inspection/history_use_20261109 \
  --history runs/inspection/history_use_teacher120k_20261110_4workers \
  --reference-recent runs/inspection/history_recent_swing_20261109 \
  --recent runs/inspection/history_recent_swing_teacher120k_20261110 \
  --output runs/inspection/history_comparison_rebuilt \
  --require-complete --validate-media
```

Reusable CPU APIs are `history_comparison.compare_history_runs`,
`teacher_comparison.compare_usual`, `teacher_comparison.compare_behavior`, and
`teacher_suite.build_reports`. Generated weights, media and reports remain
ignored; no checkpoint-specific results are embedded in implementation code.

## Verification

111 targeted CPU tests pass, covering input identities, unchanged protocols, incomplete/error records,
copied FULL references, duplicate writers, interrupted processes, time limits,
resume and comparison math. The complete report path was run on the saved
104,110/120,000 outputs without simulations: all 240 exported offline means
match the prior calculations within `1e-10`; success counts are unchanged;
4,869 history/media checks pass, including decoding all 112 referenced videos.
No 140k evaluation has been launched.

The model, causal boundaries, history-read semantics, controller and scenes
are unchanged. The architecture snapshot check and Ruff pass; no diagram update
is needed.
