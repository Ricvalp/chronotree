"""Set a worker's CPU affinity before importing Torch or the simulator."""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-affinity", required=True)
    parser.add_argument("evaluation_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cpus = {int(value) for value in args.cpu_affinity.split(",")}
    if not cpus or not cpus.issubset(os.sched_getaffinity(0)):
        raise ValueError("Worker CPU affinity is outside the inherited allocation")
    os.sched_setaffinity(0, cpus)
    forwarded = args.evaluation_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if not forwarded:
        raise ValueError("Missing evaluation arguments")
    os.execv(sys.executable, [sys.executable, "-u", "-m", "stmem.evaluate", *forwarded])


if __name__ == "__main__":
    main()
