# Independent confirmation draws

An optional confirmation extension adds one fresh bank of at most 16 draws to
an already measured frozen student and context panel. It preserves the original
manifest, checkpoints, discovery rows, confirmation rows and candidate choices.
The draw seeds depend on the state and the new `confirmation_extension` bank;
they do not depend on the student or candidate. The same fresh draws are shared
across alternatives and students. The protocol's `additional_draw_cap` can set
a smaller limit.

The extension is explicitly **raw-only, separate evidence**. The primary report
lists its inventory but does not combine it with the original confirmation bank
or reselect candidates. Any subsequent analysis must retain the original
discovery-selected choices and identify the extension bank separately. It is not
an automatic retry-until-significant procedure.

For example, after the S0 development panel has completed on the HPC:

```bash
.venv/bin/python hpc/refinement_utility.py \
  --site hpc/site_b200_4gpu.env \
  --name router_utility_v1 \
  --stage extension --student S0 \
  --manifest runs/router_utility_v1/manifest.json \
  --panel-partition train \
  --extension-base runs/router_utility_v1/measurements/S0/step_00090000/main/train \
  --additional-draws 8 --shards 4 \
  --partition gpuq --submit
```

The launcher requests one untyped GPU, eight CPUs and 64 GB per independent
shard. Omit `--submit` or use `--dry-run` to prepare without submitting. For a new
student, pass its matching `--student` and immutable `--checkpoint`, and point
`--extension-base` at that exact endpoint's completed primary directory. Keep
`--panel-partition` identical to the base directory's partition. For the amended
100k student schedule, every earlier declared snapshot, including 15k and 30k,
accepts only development/training-partition measurements. Extensions cannot turn
a development snapshot into a primary endpoint or change its training schedule.

On an already allocated local GPU, the same operation is available directly:

```bash
uv run python -m stmem.inspection.refinement_utility \
  --mode extension \
  --manifest runs/router_utility_v1/manifest.json \
  --student S0 --partition train \
  --extension-base runs/router_utility_v1/measurements/S0/step_00090000/main/train \
  --additional-draws 8 --device cuda:0 --max-hours 1
```

Rows are saved under
`measurements/STUDENT/step_XXXXXXXX/extension/PARTITION/`. Teacher predictions
use a separate shared reference directory. The extension seals hashes of the
primary artifacts; changing its base, draw count, student, precision or panel
after it starts is rejected. An interrupted run resumes its missing states with
the identical command. There is one extension identity per student endpoint and
partition, so choose the bounded draw count before starting it.

Regenerate the ordinary report with `--mode report`; the extension inventory
appears separately without changing the primary statistics.
