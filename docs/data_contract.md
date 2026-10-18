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

The fixed crop is x=[-0.9,0.75], y=[-0.75,0.75], z=[-0.05,1.25] metres; the table is z=0 and Panda base is (-0.615,0,0). This comes from benchmark workspace calibration, not hidden object poses. Depth range is (0,4) metres. Sampling takes deterministic lexicographic 5 mm voxel representatives followed by evenly spaced representatives. It merges both cameras, uses RGB, and uses no segmentation/object masks. Padding has zero points and `valid=False`.

`data/diagnostics/VideoUnmask_calibration.png` shows both clouds in a common world frame at conditioning, initial execution and moving-wrist frames. Visual inspection found coincident table/object geometry. Reprojection round-trip residual was below 0.0005 pixels for both cameras. The figure marks the documented robot base; it does not access simulator object state. Camera-visible robot geometry is retained.

The recorded frame clock is deliberately distinct from physical elapsed time. Upstream can omit NO RECORD intervals; contiguous HDF5 numeric names do not prove uninterrupted physics. Metadata logs nominal and measured cadence, but stored indices are not silently treated as physical seconds or divided by eventual episode length.

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

Large data and generated videos/figures are ignored by git. Copy raw/prepared data, download manifests and `alignment.json` together for offline HPC execution. Full experiment preparation uses 1024 points and all 100 released episodes per task; do not reuse the 256-point debug prepared data with the core configuration. Official val/test are environment seeds; no validation expert HDF5 is assumed.

## Bounded learned-policy path check

A separate two-optimizer-step real-data diagnostic also completed with `configs/debug/real_teacher.yaml`, then ran the actual `FULL` policy on official validation episode 0 with `configs/debug/rollout.yaml`. It retained all 66 conditioning frames, replanned seven times and issued 13 commands before the official short-limit timeout. The requested limit was 12; the benchmark's effective limit was 14 including the automatic reset command. There were no policy-shape, numerical, renderer or simulator errors. This is an execution-path check of a two-step debug model, not a benchmark success result. Measured artifacts and exact commands are recorded in [real_debug.json](evidence/real_debug.json); actions, timings and video remain under `runs/real_debug/rollout_full_val`.
