# Data contract and completed audit

The released revision `a5e4e25ffe8af34f64944f9533d06455ce5f8337` uses **post-action observations**. A decision made from stored observation j predicts command j+1. This was resolved from actual downloaded data and two complete expert replays on 17 October 2026. The immutable evidence summary is [data_alignment.json](evidence/data_alignment.json); the full per-command report is generated at `data/alignment.json`.

## Source evidence

The pinned benchmark `6503b26e00ff338611fceb039417fbc64826ccc8` calls `super().step(action)` before writing the observation/action pair in `RecordWrapper.py:834`. Its continuous reset generates a conditioning trajectory and executes one initial command before returning (`DemonstrationWrapper.py:146`). The official policy converter at `ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b` instead starts `get_action_chunk` at the observation's own `step_idx` (`build_robomme_dataset.py:256`). That converter offset is not used here.

The downloaded `VideoUnmask` archive is 1,251,608,544 bytes, SHA256 `5069f03981eee6ecf0c7b996ecdfb8ce64a554a9fc0ac4045d2354bab2a4e28e`. Its extracted HDF5 is 14,445,442,112 bytes, SHA256 `05a653a8f8232882f82c84057f328e045f4a875ff5cfcf068738c429c6081427`. Only episodes 0 and 1 were prepared. The full 16-task dataset was not downloaded.

| Measurement | Episode 0 | Episode 1 |
|---|---:|---:|
| Conditioning records | 66 | 66 |
| Execution records | 173 | 159 |
| Commands replayed after reset | 161 | 152 |
| Reset qpos RMSE | 0 | 0 |
| All nonterminal post-step qpos max RMSE | 0 | 0 |
| Median stored-vs-pre-step qpos RMSE (rad) | 0.00361074 | 0.00421149 |
| Official replay outcome | success | success |
| Repeated suffix records after first completed record | 11 | 6 |

Both front and wrist reset extrinsics exactly matched the stored first execution frame, as did all conditioning qpos. Reset already supplies raw execution record 66; replay therefore starts with the command in raw record 67. Replaying record 66 again would introduce a spurious one-command delay.

The continuous wrapper makes an automatic extra terminal step. Only at termination, its reported qpos differed from the stored terminal observation by 0.00621 / 0.00644 rad RMSE. These terminal observations are not used to decide another command. The registry requires every compared **nonterminal** observation to meet tolerance; it does not hide mismatches in an average. The first terminal command is retained as a target for its preceding decision, and every synthetic terminal suffix command is masked.

These are expert replays, not success measurements of the learned policy. The actual workstation is an RTX 5000 Ada Generation, not the A5000 anticipated in the brief. Headless reset, depth rendering, both camera extrinsics and terminal handling worked. Runtime control/physics frequencies are 20 / 100 Hz, and requested 1300 steps produces an effective wrapper limit of 1302.

## Canonical observations and causality

`Frame` has points `[N,6]` (xyz metres and RGB in [0,1]), boolean valid-point mask, 15 observed proprioceptive values (7 joint angles, end-effector xyz, quaternion wxyz, mean finger width), previous commanded action `[8]`, its availability flag, source-local recorded-frame clock, source tag and protocol event identity. Raw timestep/task/episode bookkeeping IDs are not model features; the source tag and recorded-frame clock are allowed features. End-effector input is the public six-dimensional extrinsic-XYZ RPY state, converted to a deterministic quaternion hemisphere.

Conditioning commands are always zero and unavailable, even when the raw demonstration file contains them. The initial automatic execution-reset command is also unavailable and excluded from action statistics. At execution observation j>0, the previous action is command j, because that command has already executed. At decision j, history is execution `[:j]`, the current observation is supplied separately once, and the target is commands `[j+1:j+1+H]`.

Conditioning and execution banks have separate source-local clocks and never share a block. The raw `is_video_demo` flag reconstructs the initial source boundary only. No per-step video/progress flag reaches the model. Policy/router inputs exclude simulator state, object IDs/poses, all oracle/subgoal text, completion flags, seeds and episode IDs. Completion flags are used only to mask training targets.

Joint-angle commands contain 7 absolute positions and a discrete gripper command (-1 close/+1 open). PatternLock/RouteStick mask the unused gripper; their simulator receives seven values. Runtime clipping uses Panda's documented URDF position limits, followed by gripper discretization, and archives the actual clipped command.

All HDF5 episode and timestep names are sorted numerically; data starts at zero. Raw names are preserved. Missing numeric steps break a target chunk's continuity. A missing/unavailable command masks all subsequent positions in that chunk. Padding and terminal suffixes are masked, never repeated as synthetic expert targets.

`EpisodeDataset` reads HDF5 with a local context manager per load, avoiding shared worker file handles. It reads prepared compact arrays; at most a small episode LRU should be used by the trainer. `validate_preprocessing(config)` rejects point-count, crop, voxel, depth or calibration changes without re-preparation. Prepared files contain source/dataset hashes; alignment evidence is loaded separately. Unknown revision/alignment raises before any substantive training sample is exposed. No nearest-pose offset is selected per sample.

`data.alignment_path` selects the evidence registry for both preparation diagnostics and training. Its default `data/alignment.json` follows `paths.data_root` (including `STMEM_DATA_ROOT`); a custom path is preserved. `{data_root}/audit/alignment.json` is also supported. Cache construction and policy loading validate checkpoint sensor, encoder and control settings against the requested configuration. Reader budgets and diagnostic environment limits can change without silently changing point sampling or action cadence. When download manifests provide extracted-file SHA256 values, preparation verifies those values before writing any prepared output.

## Calibration and time

Depth is signed int16 millimetres. Nonpositive/nonfinite values are rejected before conversion to FP32 metres. Camera intrinsics are episode-level; each frame's front and wrist extrinsics are used. CV world-to-camera `[R|t]` is inverted as `world = (camera - t) @ R` in row-vector form.

The fixed crop is x=[-0.9,0.75], y=[-0.75,0.75], z=[-0.05,1.25] metres; the table is z=0 and Panda base is (-0.615,0,0). This comes from benchmark workspace calibration, not hidden object poses. Depth range is (0,4) metres. The balanced-v2 sampler takes deterministic lexicographic 5 mm voxel representatives, then reserves 75% of the 2,048-point budget for z>5 mm. Those elevated samples are balanced over 10 cm XY cells; the remaining budget takes evenly spaced representatives from all unselected points. Unused elevated capacity returns to global coverage. It merges both cameras, uses RGB, and uses no segmentation/object masks. Padding has zero points and `valid=False`; observed points are never duplicated to fill a budget.

Each frame also stores 32 observed-point center indices selected independently
of other frames. Start with 24 elevated and eight global stratified-FPS seeds.
Refine the elevated centers using weighted Lloyd assignments, then hold them
fixed while refining the global centers. Each occupied 10 cm XYZ cell has total
weight one; each centroid is snapped to an observed member point. Selection
stops at stable indices or 20 iterations per stage. RGB only breaks canonical
ties. Robot geometry qualifies as elevated; the rule does not identify objects.
CPU clustering uses FP64 internal distance/centroid accumulations on FP32 input
to reproduce the reviewed preview; stored points, neural geometry, Gaussian
pooling and model reductions remain FP32. No learned feature or target enters
preparation. Gaussian neighborhoods and bandwidths are computed by the encoder,
so they remain trainable even though anchor indices are saved.

`canonical_balanced_v2` files live under `prepared_balanced_v2/` by default,
separate from the original `prepared/`. Manifest identity includes point and
center recipe versions, resolved settings and every required center quota.
Current and historical paths share the 32-center recipe; debug configurations
store their own distinct quotas without truncating a larger center selection.
Online ingestion uses the same selectors once per arriving observation. Indices
carry a selector signature through CPU loading/packing, preventing reuse under
different settings. Prepared indices and selected centers do not encode future
frames or privileged state. Existing raw-data alignment evidence still applies
because the RGB-D source, source clocks and action alignment are unchanged.

`data/diagnostics/VideoUnmask_calibration.png` shows both clouds in a common world frame at conditioning, initial execution and moving-wrist frames. Visual inspection found coincident table/object geometry. Reprojection round-trip residual was below 0.0005 pixels for both cameras. The figure marks the documented robot base; it does not access simulator object state. Camera-visible robot geometry is retained.

The recorded frame clock is deliberately distinct from physical elapsed time. Upstream can omit NO RECORD intervals; contiguous HDF5 numeric names do not prove uninterrupted physics. Metadata logs nominal and measured cadence, but stored indices are not silently treated as physical seconds or divided by eventual episode length.

## Optional frozen visual descriptors

The frozen-vision experiment adds sidecars under
`features_clip_rn50_layer1_v1/<task>.h5`; the existing
`prepared_balanced_v2/<task>.h5` geometry and alignment record remain unchanged.
Each selected point retains its observing camera (front=0, wrist=1) and original
integer `(u,v)`. Online preprocessing carries this identity through XYZRGB
selection. Offline feature preparation reads the saved XYZRGB/validity rows
unchanged and recovers camera/pixel correspondence without rerunning selection.
FP64 reprojection proposes an integer raw pixel; exact RGB and backprojection
of its recorded depth must agree with the canonical point within 1 micrometre
per XYZ coordinate. Missing or ambiguous correspondences are rejected; only
bitwise-identical raw XYZRGB candidates in both cameras, with voxel deduplication
enabled, use the sampler's stable front-camera tie break. The numerical bound
validates correspondence, not a replacement point cloud. Voxel representatives,
point order, centers, validity and quotas remain unchanged. Preparation also
verifies the original raw SHA256.
The bounded `--check-geometry --max-frames N` diagnostic reads at most N frames
per task without writing a cache or hashing the full raw file; it is not a
substitute for production integrity checks or cache completion.

Frozen CLIP RN50 `layer1` maps have shape `[256,64,64]` for native 256×256 RGB.
Sampling uses receptive-field origin `(1,1)` and stride four, with bilinear
border interpolation and `align_corners=False`. The versioned extraction
contract includes checkpoint checksum, CLIP normalization, camera order,
provenance/sampling rules and explicit FP32 backend settings. Sidecars store
FP16 descriptors **before** trainable LayerNorm/projection. Invalid points have
zero descriptors and camera/pixel identity -1. Lightweight row references cross
loader processes; only the next local encoder block batch is read into RAM.
Mismatched, incomplete or `--max-frames` diagnostic caches cannot silently train
a policy. Completed caches are immutable and require explicit verified reuse.

`model.proprioception_mode` selects `all`, `current_only` or `none`;
`model.condition_on_previous_commands: false` removes joint/gripper commands
and availability completely. Defaults preserve legacy behavior. FV-NoProp
removes current state and historical event tokens, including state-derived
end-effector metadata. Both required FV arms disable all previous commands.
Visual observations, timestamps and clocks are unchanged; removed event slots
do not become extra visual slots. Measured state still exists in canonical
records for controller/evaluation bookkeeping, and the robot remains visible
in RGB-D. Evaluation-only target poses/contact flags are kept outside `Frame`
and policy inputs. See [the experiment workflow](frozen_vision_experiment.md)
for preparation, matched training and diagnostic definitions.

## Reproduce

```bash
bash scripts/setup_sim.sh
uv run --extra simulator python -m stmem.data.download --tasks VideoUnmask --dry-run
uv run --extra simulator python -m stmem.data.download --tasks VideoUnmask
uv run --extra simulator python -m stmem.data.prepare --config configs/debug/data.yaml
uv run --extra simulator python -m stmem.data.inspect --config configs/debug/data.yaml
OMP_NUM_THREADS=4 uv run --extra simulator python -m stmem.data.alignment --config configs/debug/data.yaml --episodes 2 --steps 0 --video
```

The archive size is shown before transfer. `--archive PATH` reuses an already downloaded archive and still verifies SHA256. `--episodes 2` restricts preparation, not the downloadable archive. `--steps 0` explicitly runs the short full expert replay; default replay is limited to 32 commands per episode.

Large data and generated videos/figures are ignored by git. Copy raw/prepared data, download manifests and `alignment.json` together for offline HPC execution. Full experiment preparation uses 2,048 points and all 100 released episodes per task; do not reuse the 256-point `prepared_balanced_v2_debug/` data with the core configuration. Official val/test are environment seeds; no validation expert HDF5 is assumed.

## Bounded learned-policy path check

A separate two-optimizer-step real-data diagnostic also completed with `configs/debug/real_teacher.yaml`, then ran the actual `FULL` policy on official validation episode 0 with `configs/debug/rollout.yaml`. It retained all 66 conditioning frames, replanned seven times and issued 13 commands before the official short-limit timeout. The requested limit was 12; the benchmark's effective limit was 14 including the automatic reset command. There were no policy-shape, numerical, renderer or simulator errors. This is an execution-path check of a two-step debug model, not a benchmark success result. Measured artifacts and exact commands are recorded in [real_debug.json](evidence/real_debug.json); actions, timings and video remain under `runs/real_debug/rollout_full_val`.
