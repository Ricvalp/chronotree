"""Run one stage, record allocation cost, and mark only verified completion."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    start = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    args.marker.unlink(missing_ok=True)
    result = subprocess.run(command, check=False)
    elapsed = time.perf_counter() - start
    outputs = json.loads(args.outputs)
    missing = [path for path in outputs if not Path(path).exists()]
    code = result.returncode or (66 if missing else 0)
    output_hashes = {}
    if code == 0:
        for output in outputs:
            path = Path(output)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            output_hashes[output] = digest.hexdigest()
    record = {"signature": args.signature, "status": "complete" if code == 0 else "incomplete",
              "exit_code": code, "missing_outputs": missing, "profile": args.profile,
              "output_sha256": output_hashes,
              "allocated_gpus": args.gpus, "process_elapsed_seconds": elapsed,
              "process_gpu_hours": elapsed * args.gpus / 3600,
              "allocation_gpu_hours": None,
              "accounting_note": "Use hpc/accounting.py with sacct for actual allocation walltime, including startup/cleanup",
              "started_utc": started_utc,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "command": command}
    args.marker.parent.mkdir(parents=True, exist_ok=True)
    target = args.marker if code == 0 else args.marker.with_suffix(".incomplete.json")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(target)
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
