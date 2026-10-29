# Current observation and task only

`configs/teacher_current_only_fv_no_prop_10hz.yaml` trains a fresh policy to test
whether the current visual representation and flow decoder can ground an
approach to the cube without relying on history or explicit robot state.
It inherits the frozen-vision 10 Hz recipe and changes the policy input contract
to `model.history_mode: none`.

The policy receives one current point cloud combining the front and wrist
RGB-D cameras, aligned frozen CLIP RN50 layer1 descriptors, and the supplied
task-goal text. The same trainable current-scene point/edge encoder generates up
to 32 visual tokens. One goal token gives a maximum of 33 context tokens,
independent of episode length. The four-layer, eight-head, width-256 conditional
flow decoder and its loss remain unchanged.

There is no history encoder invocation, archive accumulation, historical event
encoder, state MLP or clock projection. Previous joint and gripper commands,
their availability flag, measured joint positions, measured TCP pose and measured
gripper state cannot condition the model. Absolute and relative time metadata
are zeroed too; observed spatial metadata remains. Images may still show the
robot, and calibrated geometry remains available. This is not an attempt to
remove visually observable robot state.

The loader selectively reads only the current prepared frame and its feature
row. It retains the verified action-alignment convention and terminal/interval
masks when constructing targets. Online execution applies the same contract,
including omitting the initial conditioning video. Existing prepared geometry
and feature sidecars are reused; no new preparation is required.

## Training recipe

| Setting | Value |
|---|---|
| Allocation | One node, two H200 GPUs in `gpuq`, 32 CPU cores, 128 GB host RAM |
| Slurm GPU request | `--gres=gpu:2` |
| Effective batch | 128 |
| Per-GPU microbatch | 64, one local pass per optimizer update |
| Learning rate | Peak 0.0001, inherited warmup/decay schedule |
| Training | 100,000 optimizer updates, fresh initialization |
| Action chunk | 16 absolute joint/gripper targets at 10 Hz |
| Execution | Hold each target for two native 20 Hz steps; execute two targets before replanning |
| Activation checkpointing | Disabled |
| W&B | Same `chronotree` project; a separate run, system statistics enabled |

The current-only policy has 5,887,881 active trainable parameters and one
inactive temporal-bandwidth scalar. The frozen external RGB encoder contains
244,576 parameters and is excluded from the policy optimizer. The absence of
long-history activations makes the proposed microbatch plausible, but H200
memory consumption and throughput must be measured on the allocation.

A bounded local integration check on the RTX 5000 Ada (32 GB) completed two
BF16 updates with microbatch 64 using two real VideoUnmask training episodes.
Peak allocated GPU memory was 1.39 GiB, with zero history tokens. Fixed-scene
point clouds and attention logging also ran. This verifies this batch shape
locally; it is not an H200 throughput measurement or evidence of learned success.

## Submit on the HPC

After pushing the implementation locally and pulling it on the HPC, run from
the checkout. The site file is supplied with the established HPC paths; no
shell `source` step is needed for these commands.

```bash
cd /hpc/home/phi/rvalperga/chronotree

.venv/bin/python hpc/plan.py \
  --site hpc/site_h200_2gpu.env \
  --profile h200_2gpu \
  --teacher-config configs/teacher_current_only_fv_no_prop_10hz.yaml \
  --plan core_seed0 --only seed0_teacher \
  --run-root runs/teacher_current_only_fv_no_prop_10hz_b128 \
  --output-dir hpc/generated/teacher_current_only_fv_no_prop_10hz_b128 \
  --log-every 1 --progress-every-seconds 30 --attention-every 10 \
  --dry-run
```

This writes a resolved configuration, `plan.json` and one `seed0_teacher.sbatch`
in the specified generated directory. It creates the required output/log
directories without submitting a job. Inspect the generated script for
`--partition=gpuq`, `--gres=gpu:2` and two torchrun workers, then run the same
command with `--submit` replacing `--dry-run`.

Checkpoints and local tracking files go to:

```text
/hpc/home/phi/rvalperga/chronotree/runs/teacher_current_only_fv_no_prop_10hz_b128/seed0/teacher/
```

Slurm stdout/stderr go to `hpc/logs/stmem_seed0_teacher_<jobid>.out` and `.err`.
The existing training runs have different run roots and are not resumed or
overwritten. Use a new run root and generated directory for any additional
independent repeat.

## How to interpret it

Inspect first approach/grasp success and the change in the first action chunk
when the cube moves, alongside ordinary validation outcomes. A successful
first grasp would establish that this sensor/decoder path can ground that
behavior in the tested configurations. A failure does not, by itself, establish
that the architecture cannot learn it: optimization, supervision and visibility
remain possible causes.

Some RoboMME tasks need the initial reference video or prior episode events.
This experiment deliberately withholds that evidence. Overall all-task success
therefore is not a matched memory-benchmark score. It does not alter the main
full-history policy, exact TREE/FLAT coverage or their matched supervision, and
is distinct from the compact reader's existing `CURRENT` context diagnostic.
The planner permits standalone teacher training and teacher checkpoint selection
for this configuration; it rejects using it for the subsequent memory-reader
pipeline.

Implementation verification and any measured local diagnostics are recorded in
`docs/decisions.md`. Training or benchmark success must not be inferred from
unit tests, an architecture diagram or a generated Slurm script.
