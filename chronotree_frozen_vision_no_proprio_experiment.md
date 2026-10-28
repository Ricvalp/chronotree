# ChronoTree experiment: frozen visual features without proprioception

## Goal and implementation scope

Implement an offline imitation-learning experiment in [Ricvalp/chronotree](https://github.com/Ricvalp/chronotree) that combines:

1. A frozen pretrained RGB encoder whose dense spatial features are attached to the corresponding observed 3D points.
2. Removal of all explicit current and historical proprioceptive inputs to the learned policy.

The primary question is whether this combination improves object targeting and autonomous execution from episode reset. Implement the main variant and a matched control with the same frozen vision encoder but with proprioception retained. Preserve the existing memory architecture and action protocol.

This document is an implementation handoff, not a report of completed experiments. Read the repository's current `AGENTS.md`, scientific contract and data-contract documentation before editing. The source review informing this brief used commit `13a5fab241fd0b8d09aaa551da4f2bba492678bc`; inspect current code and incorporate any subsequent changes.

Deliver implementation, focused checks, experiment configurations and runnable launch/evaluation commands. Full cluster training and complete feature-cache generation belong to the subsequent experiment execution phase; a small smoke run is sufficient for implementation validation.

## Motivation and existing evidence

The existing teacher learns plausible pick-and-place motion but frequently misses the object. An expert approach prefix improves subsequent behavior, even before contact. This is consistent with the expert trajectory supplying target information through robot configuration and motion, which the learner can then extrapolate.

Repository diagnostics already report that:

- Removing previous joint commands did not solve targeting. Measured robot-state history remained available.
- At an inspected PickXTimes state, moving the cube by 100 mm produced only a very small change in predicted tool positions.
- The commanded trajectory itself missed the object; controller tracking error was much smaller.
- Cube geometry survived point sampling and pooling.
- Reset representations and packed/streaming inputs matched in the inspected cases.
- Executing all 16 predicted commands instead of four did not solve the problem.
- Some inspected expert prefixes contain substantial initial holding and ramp-up.

These are bounded diagnostic observations, not proof of a single root cause. Do not repeat already completed ablations as though they were new.

The proposed mechanism has two parts: pretrained spatial descriptors reduce the burden of learning visual features from action supervision, and removal of numerical robot state removes an easy competing predictor. Neither guarantees that the policy will use target geometry. In particular, robot appearance and episode time can still support shortcuts.

Relevant precedent:

- RoboMME Appendix A.2.2 reports no consistent benefit from adding historical proprioception. Appendix B.2.3 and Table 6 exclude proprioception from MME-VLA inputs generally, despite joint-space actions.
- Adapt3R aligns pretrained RGB features with 3D points. Its LIBERO/BAKU ablation reports 91.9% original-setting success with image features versus 60.5% with raw RGB point colors. This supports testing the representation; it does not predict the gain for ChronoTree.
- These results do not establish that proprioception is universally harmful or that a frozen backbone is always optimal.

## Experiment arms

| Arm | Frozen image features | Current measured state | Historical measured state | Status |
| --- | --- | --- | --- | --- |
| FV-NoProp | Yes | No | No | Primary candidate |
| FV-Prop | Yes | Yes | Yes | Required matched control |
| FV-CurrentOnly | Yes | Yes | No | Optional follow-up |

Disable **all previous-command conditioning in both required arms**, including previous joint commands, previous gripper commands and command-availability indicators. Previous commands are distinct from measured proprioception. The existing `condition_on_previous_joint_commands: false` setting is insufficient for this stricter experiment because it retains other command information.

FV-NoProp versus FV-Prop isolates the effect of removing explicit proprioception under a common pretrained visual representation. It does **not** isolate the benefit of visual pretraining. Existing XYZRGB checkpoints provide diagnostic context but are not a matched control if their training budget, horizon or command treatment differs. A future XYZRGB arm with otherwise identical settings can test the encoder contribution separately.

Train both required arms freshly. Match initial weights of shared trainable modules, data order, seeds and evaluation sampling where practical. Do not make continuation from the existing teacher the only comparison.

## Frozen visual encoder

Use one backbone initially, without a backbone sweep:

- **Default:** pretrained CLIP ResNet-50, frozen.
- Extract a dense intermediate feature map after the first residual stage, typically `layer1`, before global attention pooling. Verify the actual implementation's feature dimensions and spatial stride.
- Use native 256 × 256 camera images where compatible with this intermediate convolutional path. Apply checkpoint-specific channel ordering and normalization.
- Avoid random geometric image augmentation in the initial experiment. Any future crop or resize must have an explicit coordinate mapping to the original RGB-D pixels.
- Do not use a globally pooled embedding broadcast to every point.
- Keep the backbone in evaluation mode, disable parameter gradients, and exclude it from the optimizer. Calling `train()` on the parent policy must not reactivate backbone BatchNorm updates.

The projection, point encoder, memory modules and action decoder remain trainable. DINOv2 dense patch features are a possible later alternative, not a required second experiment.

For an observed point originating at pixel `(u, v)` in camera `c` and frame `t`, construct:

```text
dense_feature = sample_spatial_map(frozen_encoder(rgb[t, c]), u, v)
point_feature = concat(raw_rgb, projection(LayerNorm(dense_feature)))
```

Use a trainable projection to **128 channels** initially. Retain precise XYZ separately for geometric computations and retain raw RGB alongside the pretrained descriptor. A compatible point MLP can then consume XYZ, RGB and the projected descriptor, while all neighborhood distances and anchors continue to use XYZ only.

Small cubes must remain distinguishable in the feature map. Inspect feature/point overlays on a few existing examples; interpolation of a coarse map cannot restore detail already lost.

## Pixel-to-point correspondence and caching

Preserve the source camera and original pixel coordinates through backprojection, cropping, voxelization and sampling. Prefer sidecar provenance/features indexed by the same selected point indices, leaving canonical XYZRGB geometry intact.

Requirements:

- Attach the descriptor from the camera that actually observed the point.
- Do not reproject a merged point into every camera without checking visibility.
- Do not change point selection because extra feature channels alter sorting, tie-breaking, voxel representatives or sampling.
- Keep point counts, geometry-based anchor-selection rules, visual token quotas, neighborhood rules and temporal grouping matched between the two arms. This does not require trained pooling weights or learned routing decisions to be identical.
- Carry padding/validity masks consistently; invalid points must not contribute features.
- Use exactly the same feature extraction and coordinate convention for training and online rollout.
- Apply the encoder independently to each available frame, including conditioning-video frames and execution history. Never expose future frames or use a bidirectional video encoder.

A frozen feature cache is optional for the first smoke run and desirable for full training. Cache raw frozen feature maps or their correctly sampled descriptors, **before** the trainable projection. Do not cache outputs of a projection whose weights are changing.

Version caches by backbone checkpoint identity, extraction layer, preprocessing, feature shape, and relevant point-provenance/sampling conventions. Existing XYZRGB-only caches must not be silently interpreted as feature-augmented caches. Expose cache size and extraction throughput before scheduling complete generation.

## Exact meaning of no proprioception

FV-NoProp must have no learned dependence on explicit numerical:

- Current joint positions.
- Current end-effector position or orientation.
- Current measured gripper state.
- Historical versions of those states, including conditioning-video state records.
- Previous commanded actions or their availability flags, as specified for both arms.

Audit every path, not only the action decoder:

- Current state embedding.
- Historical robot-state/event tokens.
- Router queries and routing features.
- Serial and packed/batched policy implementations.
- Offline preparation, streaming memory, caches and checkpoint restoration.
- Auxiliary summaries or features computed from state.

Remove state-bearing tokens or make their inputs strictly state-independent through an explicit configuration path. Do not leave an unreviewed state-derived summary available elsewhere.

Retain temporal ordering, timestamps, validity metadata and existing clock features identically across arms. Removing absolute clocks is a separate ablation. Match eligible visual observations, sampling rules and visual quotas: do not spend removed state-token slots on additional visual observations in this comparison. A learned router may legitimately select different records after training. Report actual token counts; use masked padding where needed for matched layouts.

“No explicit proprioception” concerns learned policy inputs. Calibrated depth, camera extrinsics, the existing low-level joint controller and evaluation-only state measurements remain available. The robot remains visible in RGB-D. Do not claim that all information about robot configuration has been removed.

## Configuration and compatibility

Add explicit configuration options for:

- Visual feature mode: legacy XYZRGB or frozen pretrained features.
- Backbone identity, layer, preprocessing and projected feature dimension.
- Proprioception mode: all, current-only, or none.
- Complete previous-command conditioning.
- Feature-cache identity and location.

Use names consistent with the repository's configuration system. Preserve legacy defaults, old checkpoint loading behavior and existing experiments. Reject incompatible cache/checkpoint/configuration combinations clearly.

Likely implementation touchpoints, subject to current source inspection:

- `src/stmem/data/pointcloud.py`: pixel provenance and stable point selection.
- Dataset preparation, frame records and cache serialization.
- `src/stmem/models/spacetime_encoder.py`: feature ingestion and historical state events.
- `src/stmem/models/event_encoder.py`: state/command conditioning.
- `src/stmem/models/policy.py` and `batched_policy.py`: current state and packed paths.
- Router feature construction.
- Online adapter, runner and streaming preparation.

Avoid replacing the memory architecture or adding unrelated objectives in this experiment. Follow repository requirements for architecture diagrams and data-contract updates.

## Matched training protocol

Use the existing RoboMME demonstrations and splits, with pure offline imitation learning. No DAgger, recovery collection, new expert prefixes for training, segmentation masks or simulator object poses as policy inputs.

Recommended common setup, based on the existing H32 configuration:

| Setting | Both required arms |
| --- | --- |
| Prediction horizon | 32 consecutive native-cadence actions |
| Execution horizon | 4 actions |
| Command cadence | Existing 20 Hz |
| Effective batch size | 128 |
| Learning-rate recipe | Existing 1e-4 teacher recipe and scheduler |
| Planned training budget | 100k updates |
| Comparison checkpoints | 25k, 50k and 100k |
| Other sampling, normalization and memory settings | Identical |

Start from the intent of `configs/teacher_b200_b128_m8_blocks32_no_previous_joint_commands_h32.yaml`, then add the new feature/state flags and complete command removal. Inspect the resolved configuration instead of assuming inherited values.

H32 is a common setup choice here, motivated by initial holds. This comparison does not test its effect. Keep execute-four; do not downsample action targets or change controller timing.

Microbatch size may change for memory capacity, but preserve effective batch size and record accumulation. Do not independently tune one arm. Use the same fixed evaluation seeds and checkpoint selection rule, and reserve held-out benchmark evaluation for the final comparison.

## Focused implementation checks

Add meaningful checks for the new contracts:

1. **Alignment and sampling:** synthetic RGB-D correspondences and representative real samples confirm correct feature-to-pixel-to-point mapping; attaching features leaves selected XYZRGB points unchanged.
2. **Frozen/trainable boundary:** the backbone has no gradients or running-statistic updates; the projection and downstream modules receive gradients.
3. **No-state invariance:** with RGB, XYZ, camera calibration, metadata and action-sampling noise fixed, changing unused raw current/historical proprioception and previous commands leaves FV-NoProp predictions unchanged. Include routing when it is active.
4. **Path parity:** offline/packed and streaming/serial paths agree on features and predictions under matched precision and noise.
5. **Causality and reset:** the current frame appears exactly once; unavailable history is represented correctly; future data cannot affect a current prediction.
6. **Compatibility:** legacy configurations still run, and incompatible caches/checkpoints fail explicitly.

The invariance test is an input-path check, not a physically consistent scene intervention: keep geometry/calibration fixed intentionally. Run relevant existing causal/data-contract tests and a bounded forward/backward/rollout smoke test. Report which checks ran and any limitations.

## Evaluation and interpretation

Use a small fixed development set of existing scenes for diagnostics, including PickXTimes and VideoUnmask. Separate training-scene debugging from validation results.

Record:

- Autonomous first-grasp success and first successful lift, where applicable.
- Tool-to-object position error at the grasp attempt.
- Commanded versus observed tool trajectory error.
- Full-task success, with partial pick/place progress reported separately.
- Target-position sensitivity using the existing displacement probe, with matched policy noise and several scenes/phases.
- Behavior-cloning loss, inference latency, peak memory, trainable/frozen parameter counts and cache costs.

Object poses or segmentation may be used by evaluation tooling to calculate metrics or visualize correspondences, but must never enter training inputs or policy features.

Interpret intervention results carefully: changing a target only in the current image while leaving its old history intact creates conflicting evidence. Distinguish that probe from reset-scene or consistently constructed interventions. Initial holding also means a small first-four-command response alone does not prove blindness.

Expert-prefix handoffs may remain diagnostic, but the main outcome is autonomous performance from reset. An improvement only after expert intervention is not a successful resolution.

Decision guidance:

- Better first-grasp success and targeting in FV-NoProp support removing explicit state under the frozen encoder.
- Better targeting but worse precise control motivate FV-CurrentOnly as a follow-up.
- Little change in targeting despite improved imitation loss suggests the relevant visual information is still underused.
- Failure of both arms does not by itself show that pretraining is ineffective; action supervision, feature resolution and state-to-action mapping remain possible limitations.

Once the teacher is useful, apply the selected visual/state recipe consistently across the paper's memory-method comparisons. Do not regenerate costly router supervision merely because the new encoder compiles. Disclose external visual pretraining and include its online computational cost.

## Requested deliverables from Codex

- Implemented encoder/provenance path and complete proprioception/command switches.
- Two runnable configurations: FV-NoProp and FV-Prop.
- Focused validation results and updated relevant documentation.
- Exact feature-preparation, training and evaluation commands with explicit data/checkpoint paths.
- A concise account of changed interfaces, compatibility, expected resource costs and remaining uncertainties.

## Sources and prior repository reports

- [RoboMME, Sections 4.1, A.2.2 and B.2.3 / Table 6](https://arxiv.org/html/2603.04639v3).
- [Adapt3R, Section 3, Table 3 and backbone studies](https://arxiv.org/html/2503.04877v2).
- [3D Diffuser Actor, spatial feature lifting](https://arxiv.org/html/2402.10885v3).
- [Command-ablation and action-cadence report](https://github.com/Ricvalp/chronotree/blob/13a5fab241fd0b8d09aaa551da4f2bba492678bc/docs/command_ablation_inspection.md).
- [Reset-distribution audit](https://github.com/Ricvalp/chronotree/blob/13a5fab241fd0b8d09aaa551da4f2bba492678bc/docs/reset_distribution_audit.md).
- [Expert-prefix handoff report](https://github.com/Ricvalp/chronotree/blob/13a5fab241fd0b8d09aaa551da4f2bba492678bc/docs/expert_prefix_handoff.md).
