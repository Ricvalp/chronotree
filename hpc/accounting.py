#!/usr/bin/env python3
"""Collect actual Slurm allocation costs without inventing GPU credit rates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submitted", type=Path, required=True, help="Planner submitted.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    submitted = json.loads(args.submitted.read_text())
    if not submitted:
        raise SystemExit("No submitted jobs to account")
    ids = [value["job_id"] for value in submitted.values()]
    completed = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
        "--format=JobIDRaw,State,ElapsedRaw,AllocTRES,Start,End"], capture_output=True, text=True, check=True)
    plan = json.loads((args.submitted.parent / "plan.json").read_text())
    profile = plan["profile"]
    reverse = {value["job_id"]: name for name, value in submitted.items()}
    records = []
    for line in completed.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 6 or fields[0] not in reverse:
            continue
        job_id, status, elapsed, tres, start, end = fields[:6]
        match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
        gpu_count = int(match.group(1)) if match else 0
        seconds = int(elapsed)
        records.append({"name": reverse[job_id], "job_id": job_id, "state": status,
                        "elapsed_seconds": seconds, "allocated_gpus": gpu_count,
                        "allocated_gpu_hours": seconds * gpu_count / 3600,
                        "profile": profile, "alloc_tres": tres, "start": start, "end": end})
    by_hardware = {"H200": 0.0, "B200": 0.0, "other": 0.0}
    for record in records:
        group = "H200" if profile.startswith("h200") else "B200" if profile.startswith("b200") else "other"
        by_hardware[group] += record["allocated_gpu_hours"]
    output = {"source": "sacct allocation records", "jobs": records, "gpu_hours_by_requested_profile": by_hardware,
              "missing_job_ids": sorted(set(ids) - {record["job_id"] for record in records}),
              "gpu_model_note": "Profile is requested hardware; verify actual GPU in each doctor/run artifact",
              "credit_conversion": None}
    path = args.output or args.submitted.parent / "accounting.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
