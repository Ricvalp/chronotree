# Exact-state first-chunk continuation

`stmem.inspection.refinement_continuation` implements the bounded behavioral
check in addendum §11. It reuses the simulator adapter, released expert source
reader, streaming memory, online policy and video helpers. It does not modify
training, generated-action selection or the production evaluation runner.

The local case is chosen before reading losses: the first cache-eligible
VideoUnmask episode-0 native decision at or after step 32. In the available
cache this is decision 36 (`206558cd6af57114245f`). Its original router-panel
partition remains **validation**. Consequently this single-asset case is
explicitly exploratory; it is not silently relabeled as development data and
cannot represent an independently selected final validation result.

The generated-action plan must have scope
`limited_asset_exploratory_single_state`, one case, the accepted S0 checkpoint,
and fixed TREE/background120. Discovery-only losses select the completed
immediate and packet alternatives; the plan records near-tie representatives.
The same fresh Gaussian draw is used for all four first chunks. The wrapper
verifies generated arrays, checkpoint, plan, source and asset identities.

Each of four branches—base, immediate-completed, packet-completed, dispersed—
starts a fresh official training configuration and replays the genuine expert
commands. Reset, every conditioning state, every expert-prefix joint state and
the handoff must match the recorded states within 1e-4 radians. The handoff
source index and number of retained observations are also checked. Hidden
container/grasp labels are saved only for the diagnostic report and never enter
the policy or selection.

At the handoff, the wrapper executes exactly the first ten generated targets,
each held for two native 20-Hz simulator steps. The base context has 120 tokens;
the three completed alternatives have 128. The packet/immediate/dispersed
comparison therefore controls final budget; the base comparison does not.
After this single intervention, **all branches use UNIFORM/B128**, eight flow
integration steps, and ten executed targets held twice. Each branch ingests its
own newly observed frames into the ordinary causal streaming memory. No stored
frontier is reused after new observations arrive. Subsequent noise streams
share a fresh seed and advance once per subsequent chunk.

Once the explicit diagnostic generated plan and arrays have completed:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
.venv-sim/bin/python -m stmem.inspection.refinement_continuation \
  --plan PATH/diagnostic_generated_plan.json \
  --output PATH/first_chunk_continuation \
  --assets-config configs/inspection.yaml \
  --draw 0 --cap-native-commands 320 --seconds-per-branch 300 \
  --device cuda:0
```

The native-command cap includes the expert prefix. Each branch has a separate
wall-time bound and MP4. `manifest.json` freezes the protocol; per-branch
`result.json` retains all commands, observation indices, prefix checks, choices
and outcomes. `summary.json` and `report.md` summarize the observed first grasp
and target height. Existing results are reused only when the experiment identity
matches. A changed source/configuration requires a new output directory.

This can show whether a selected first chunk changes a subsequent choice from a
verified expert state. It cannot establish autonomous benchmark success or
long-term learned-router benefit. Later behavior includes the common UNIFORM
policy's responses to states caused by that first intervention.

CPU protocol tests verify gain-blind state choice, two native holds per target,
replay mismatch rejection, exactly one selected-context chunk, and causal memory
updates before subsequent UNIFORM decoding. Real local execution is recorded
separately after generated arrays are available; tests alone are not simulator
evidence.

## Completed local check, 2026-11-10

Artifacts: `runs/inspection/refinement_utility_final_20261110/first_chunk_continuation/`.
The frozen case above was run with generation draw 0, a 320-native-command cap
and a 300-second bound per branch. All four branches grasped container 0 (the
correct blue-cube container) and reached the simulator's success condition.

| First context | First grasp, native step | Terminal native step | Wall seconds |
|---|---:|---:|---:|
| Base B120 | 75 | 102 | 10.67 |
| Immediate-selected completed B128 | 82 | 99 | 9.31 |
| Packet-selected completed B128 | 75 | 101 | 9.43 |
| Dispersed B128 | 82 | 105 | 9.55 |

Reset, conditioning and every expert-prefix joint comparison had zero error.
The handoff state was identical across branches, with 1,632 historical leaves.
Independent trace validation checked all held commands, a single intervened
chunk, later UNIFORM replanning at native steps 56/76/96, and causal observation
counts. All four MP4s decoded successfully; their hashes are retained in
`validation.json`.

There was **no differential task-choice improvement** in this case. The expert
prefix had already brought the gripper nearest the correct container before the
intervention (77 mm lateral distance, 259 mm three-dimensional distance). All
contexts retained sufficient behavior for this handoff, but the probe cannot
separate historical-cue use from the expert approach's visible positional cue,
and these four branches are not four independent demonstrations.
