"""Generate measured summaries and paired comparisons; missing runs stay missing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

from stmem.config import digest, load_config, source_revision
from stmem.evaluate import evaluation_tracking_scalars, numeric_scalars
from stmem.evaluation.metrics import episode_key, paired_comparison, summarize_episodes
from stmem.evaluation.timing import latency_summary
from stmem.training.checkpoints import atomic_json, file_sha256
from stmem.training.runtime import environment_manifest
from stmem.tracking import RunTracker

METHODS = ("FULL", "TREE", "FLAT", "UNIFORM", "PERCEIVER", "CURRENT")


def collect(directory: Path) -> tuple[list[dict], list[tuple[str, int]] | None]:
    records: list[dict] = []
    planned: set[tuple[str, int]] = set()
    saw_plan = False
    for path in sorted(directory.rglob("*.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read evaluation artifact {path}: {error}") from error
        if not isinstance(value, dict):
            continue
        # A shard records its own subset and the complete comparison manifest.
        # Prefer the latter, so a missing entire shard cannot shrink denominators.
        planned_values = value.get("all_planned", value.get("planned_episodes", value.get("planned")))
        if isinstance(planned_values, list):
            saw_plan = True
            for item in planned_values:
                planned.add(episode_key(item) if isinstance(item, dict) else (str(item[0]), int(item[1])))
        if (
            "task" in value
            and "status" in value
            and any(key in value for key in ("episode_idx", "episode_index", "seed"))
        ):
            value["_evaluation_shard"] = str(
                path.parent.parent if path.parent.name == "episodes" else path.parent
            )
            records.append(value)
    return records, sorted(planned) if saw_plan else None


def build_report(run_root: Path, split: str = "val", *, tracker=None) -> dict[str, Any]:
    # A seed directory is one analysis unit. A run root can contain many seeds.
    seed_roots = sorted(run_root.glob("seed[0-9]*")) or [run_root]
    seeds: dict[str, Any] = {}
    rows = []
    for seed_root in seed_roots:
        seed = seed_root.name
        methods, records_by_method = {}, {}
        for method in METHODS:
            directory = seed_root / "evaluation" / split / method
            records, plan = collect(directory) if directory.exists() else ([], None)
            summary = summarize_episodes(records, plan)
            decision_seconds, cold_seconds, warm_seconds = [], [], []
            for record in records:
                episode_seconds = []
                for decision in record.get("decisions", []):
                    value = decision.get(
                        "complete_online_seconds",
                        decision.get("decision_seconds", decision.get("timing", {}).get("decision_seconds")),
                    )
                    if value is not None:
                        episode_seconds.append(float(value))
                decision_seconds.extend(episode_seconds)
                cold_seconds.extend(episode_seconds[:1])
                warm_seconds.extend(episode_seconds[1:])
            summary["latency"] = {
                "all": latency_summary(decision_seconds, 0)["all"],
                "cold_first_decision_per_episode": latency_summary(cold_seconds, 0)["all"],
                "warm_after_first_decision_per_episode": latency_summary(warm_seconds, 0)["all"],
            }
            summary["startup"] = {
                key: latency_summary([float(row[key]) for row in records if key in row], 0)["all"]
                for key in ("startup_seconds", "reset_encoding_seconds")
            }
            shard_loads = {
                row["_evaluation_shard"]: float(row["model_load_seconds"])
                for row in records
                if "model_load_seconds" in row
            }
            summary["startup"]["model_load_seconds_per_shard"] = latency_summary(
                list(shard_loads.values()), 0
            )["all"]
            if tracker is not None:
                tracker.log(
                    evaluation_tracking_scalars(
                        records, plan, summary=summary, prefix=f"report/{seed}/{method}"
                    )
                )
            methods[method], records_by_method[method] = summary, records
            rows.append(
                {
                    "training_seed": seed,
                    "method": method,
                    "split": split,
                    "status": summary["status"],
                    "macro_success": summary["macro_success"],
                    **{
                        key: summary["overall"][key]
                        for key in ("planned", "recorded", "missing", "successes", "errors", "timeouts")
                    },
                }
            )
        left, right = records_by_method["TREE"], records_by_method["FLAT"]
        if left and right:
            try:
                paired = paired_comparison(left, right)
            except ValueError as error:
                paired = {"status": "incomplete pairing", "reason": str(error)}
        else:
            paired = {"status": "not yet run", "macro_success_difference": None, "ci95": None}
        seeds[seed] = {"methods": methods, "tree_minus_flat": paired}
    variation = {}
    import numpy as np

    for method in METHODS:
        measured = [
            seed["methods"][method]["macro_success"]
            for seed in seeds.values()
            if seed["methods"][method]["status"] == "complete"
            and seed["methods"][method]["macro_success"] is not None
        ]
        variation[method] = {
            "completed_training_seeds": len(measured),
            "mean_macro_success": float(np.mean(measured)) if measured else None,
            "training_seed_standard_deviation": float(np.std(measured, ddof=1))
            if len(measured) > 1
            else None,
        }
    report = {
        "split": split,
        "seeds": seeds,
        "training_seed_variation": variation,
        "claim": "Measured closed-loop episode outcomes only; null means not yet run.",
        "rows": rows,
    }
    output = run_root / "reports" / split
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(report, output / "summary.json")
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Code-generated plot contains only methods with actual completed outcomes.
    measured_rows = [row for row in rows if row["status"] == "complete" and row["macro_success"] is not None]
    if measured_rows:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        figure, axis = plt.subplots(figsize=(max(6, len(measured_rows) * 0.65), 4))
        axis.bar(
            [f"{row['training_seed']}\n{row['method']}" for row in measured_rows],
            [row["macro_success"] for row in measured_rows],
        )
        axis.set(ylabel="Task-macro success", ylim=(0, 1), title=f"Official {split}: measured completed runs")
        figure.tight_layout()
        figure.savefig(output / "success.png", dpi=160)
        plt.close(figure)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--config", default="configs/base.yaml", help="Configuration for run tracking")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_match = re.fullmatch(r"seed(\d+)", args.run_root.name)
    effective_root = args.run_root.parent if seed_match else args.run_root
    cfg.setdefault("paths", {})["run_root"] = str(effective_root.resolve())
    if seed_match:
        cfg["seed"] = int(seed_match.group(1))
    directory = args.run_root / "reports" / args.split
    with RunTracker(
        cfg,
        directory,
        job_type="report",
        name=f"report-{args.run_root.name}-{args.split}",
        metadata={
            "run_root": str(args.run_root.resolve()),
            "split": args.split,
            "source": source_revision(),
            "environment": environment_manifest(),
        },
        resume=(directory / "summary.json").exists(),
    ) as tracker:
        seed_roots = sorted(args.run_root.glob("seed[0-9]*")) or [args.run_root]
        input_manifests = {}
        input_provenance = {}
        configurations = {}
        for seed_root in seed_roots:
            for path in sorted((seed_root / "evaluation" / args.split).glob("*/**/manifest.json")):
                relative = str(path.relative_to(args.run_root))
                manifest = json.loads(path.read_text())
                input_manifests[relative] = file_sha256(path)
                provenance = {
                    key: manifest[key]
                    for key in (
                        "method",
                        "split",
                        "source",
                        "environment",
                        "checkpoints",
                        "shard_index",
                        "num_shards",
                    )
                    if key in manifest
                }
                if "config" in manifest:
                    configuration_hash = digest(manifest["config"])
                    configurations[configuration_hash] = manifest["config"]
                    provenance["config_sha256"] = configuration_hash
                input_provenance[relative] = provenance
        tracker.summary(
            {
                "report/input_manifest_sha256": input_manifests,
                "report/input_provenance": input_provenance,
                "report/input_configurations": configurations,
                "report/training_seeds": [root.name for root in seed_roots],
            }
        )
        report = build_report(args.run_root, args.split, tracker=tracker)
        scalars = numeric_scalars(report, "report")
        for seed, values in report["seeds"].items():
            interval = values["tree_minus_flat"].get("ci95")
            if interval is not None:
                scalars[f"report/seeds/{seed}/tree_minus_flat/ci95_lower"] = float(interval[0])
                scalars[f"report/seeds/{seed}/tree_minus_flat/ci95_upper"] = float(interval[1])
        tracker.log(scalars)
        tracker.summary({**scalars, "report/status": "generated", "report/rows": report["rows"]})
    for row in report["rows"]:
        score = (
            "not yet run" if row["macro_success"] is None else f"{row['macro_success']:.4f} ({row['status']})"
        )
        print(f"{row['training_seed']} {row['method']}: {score}")
    print(f"Written {args.run_root / 'reports' / args.split / 'summary.json'}")


if __name__ == "__main__":
    main()
