#!/usr/bin/env python3
"""Fetch the exact audited public sources without committing nested repositories."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT / "third_party/sources.lock.json").read_text())
SOURCES = {
    "robomme_benchmark": ("https://github.com/RoboMME/robomme_benchmark.git", LOCK["benchmark_revision"]),
    "ManiSkill": ("https://github.com/YinpeiDai/ManiSkill.git", LOCK["maniskill_revision"]),
    "robomme_policy_learning": (
        "https://github.com/RoboMME/robomme_policy_learning.git",
        LOCK["policy_revision"],
    ),
}


def run(*cmd):
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()
    destination = ROOT / "third_party/checkouts"
    destination.mkdir(parents=True, exist_ok=True)
    for name in args.sources:
        url, revision = SOURCES[name]
        target = destination / name
        if not target.exists():
            run("git", "clone", "--filter=blob:none", "--no-checkout", url, str(target))
        dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
        if dirty:
            raise RuntimeError(
                f"Checkout has local changes; preserve/review them before refetching: {target}"
            )
        run("git", "-C", str(target), "fetch", "origin", revision)
        run("git", "-C", str(target), "checkout", "--detach", revision)
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Revision mismatch for {name}")
        print(f"{name} {actual}")


if __name__ == "__main__":
    main()
