"""Resolve an immutable checkpoint from the fixed official validation protocol."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
from stmem.training.checkpoints import atomic_json, file_sha256


def select(candidates, evaluations, output):
    candidates = [Path(p) for p in candidates]
    evaluations = [Path(p) for p in evaluations]
    if len(candidates) != 2 or len(evaluations) not in (2, 4):
        raise ValueError(
            "Selection compares exactly two predeclared checkpoints, one or balanced two context families each"
        )
    per_candidate = len(evaluations) // len(candidates)
    scores = []
    plans = []
    evidence = []
    for i, checkpoint in enumerate(candidates):
        values = []
        families = []
        for directory in evaluations[i * per_candidate : (i + 1) * per_candidate]:
            manifest = json.loads((directory / "manifest.json").read_text())
            summary = json.loads((directory / "summary.json").read_text())
            if manifest["split"] != "val" or summary["status"] != "complete":
                raise ValueError("Checkpoint selection requires complete official validation evaluation")
            statuses = summary["overall"]["statuses"]
            if statuses.get("error", 0) or statuses.get("infrastructure_error", 0):
                raise ValueError("Resolve diagnosed infrastructure errors before selecting a checkpoint")
            expected = file_sha256(checkpoint)
            if expected not in manifest["checkpoints"].values():
                raise ValueError("Evaluation checkpoint identity mismatch")
            plans.append(manifest["planned"])
            families.append(manifest["selection_family"])
            values.append(summary["macro_success"])
            evidence.append(str(directory))
        if per_candidate == 2 and set(families) != {"tree", "flat"}:
            raise ValueError("Compact selection must balance tree and flat contexts")
        scores.append(sum(values) / len(values))
    if any(plan != plans[0] for plan in plans):
        raise ValueError("Selection evaluations used different episode seeds")
    # Predeclared tie break: earlier candidate, minimizing unnecessary exposure.
    best = max(range(len(scores)), key=lambda i: (scores[i], -i))
    output = Path(output)
    if output.exists():
        if file_sha256(output) != file_sha256(candidates[best]):
            raise FileExistsError("Immutable selection already exists with different content")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".selecting.tmp")
    shutil.copyfile(candidates[best], temporary)
    temporary.replace(output)
    atomic_json(
        {
            "selected": str(candidates[best]),
            "sha256": file_sha256(output),
            "scores": scores,
            "candidates": list(map(str, candidates)),
            "evaluations": evidence,
            "rule": "macro success, equal tree/flat weight for compact, earliest candidate on ties",
        },
        output.with_suffix(".selection.json"),
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--evaluations", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    print(select(args.candidates, args.evaluations, args.output))


if __name__ == "__main__":
    main()
