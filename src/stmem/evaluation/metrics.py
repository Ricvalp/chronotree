"""Episode-denominator metrics and stratified paired uncertainty intervals."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np

SUITES = {
    "Counting": ("BinFill", "PickXtimes", "SwingXtimes", "StopCube"),
    "Permanence": ("VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"),
    "Reference": ("PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"),
    "Imitation": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}
TASKS = tuple(task for group in SUITES.values() for task in group)


def episode_key(record: dict[str, Any]) -> tuple[str, int]:
    index = record.get("episode_idx", record.get("episode_index", record.get("seed")))
    if index is None:
        raise ValueError("Episode result lacks episode_idx/episode_index/seed")
    return str(record["task"]), int(index)


def _indexed(records: Iterable[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for record in records:
        key = episode_key(record)
        if key in result:
            raise ValueError(f"Duplicate episode result: {key}")
        result[key] = record
    return result


def _counts(records: list[dict[str, Any]], planned: int) -> dict[str, Any]:
    if len(records) > planned:
        raise ValueError("Completed records exceed planned episode denominator")
    counts = Counter(str(row.get("status", "error")) for row in records)
    successes = counts["success"]
    errors = sum(counts[s] for s in ("error", "infrastructure_error", "policy_error"))
    timeouts = sum(counts[s] for s in ("timeout", "truncated"))
    return {
        "planned": planned,
        "recorded": len(records),
        "missing": planned - len(records),
        "successes": successes,
        "errors": errors,
        "timeouts": timeouts,
        "failures": len(records) - successes,
        "completed": len(records) - errors,
        "success_rate": successes / planned if planned and records else None,
        "statuses": dict(counts),
        "complete": len(records) == planned,
    }


def summarize_episodes(
    records: Iterable[dict[str, Any]], planned: Iterable[tuple[str, int]] | None = None
) -> dict[str, Any]:
    """Missing planned episodes remain visible; never silently narrow denominator.

    Partial success rates are explicitly lower bounds until all planned results
    exist. Without a plan, only the recorded denominator is known.
    """
    indexed = _indexed(records)
    keys = list(planned) if planned is not None else list(indexed)
    if len(keys) != len(set(keys)):
        raise ValueError("Planned episode list contains duplicates")
    if set(indexed) - set(keys):
        raise ValueError("Result contains an episode outside the planned manifest")
    task_counts = Counter(task for task, _ in keys)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (task, _), row in indexed.items():
        grouped[task].append(row)
    tasks = {task: _counts(grouped[task], count) for task, count in sorted(task_counts.items())}
    suite_values = {}
    for suite, members in SUITES.items():
        present = [tasks[task]["success_rate"] for task in members if task in tasks]
        suite_values[suite] = {
            "macro_success": float(np.mean([value or 0 for value in present]))
            if any(value is not None for value in present)
            else None,
            "tasks_present": sum(task in tasks for task in members),
            "tasks_expected": len(members),
        }
    rates = [row["success_rate"] or 0 for row in tasks.values()]
    overall = _counts(list(indexed.values()), len(keys))
    return {
        "status": "not yet run" if not indexed else ("complete" if overall["complete"] else "partial"),
        "denominator_source": "planned" if planned is not None else "recorded_only",
        "rates_are_lower_bounds": not overall["complete"],
        "macro_success": float(np.mean(rates)) if rates and indexed else None,
        "tasks": tasks,
        "suites": suite_values,
        "overall": overall,
    }


def paired_comparison(
    left: Iterable[dict[str, Any]],
    right: Iterable[dict[str, Any]],
    *,
    bootstrap_samples: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Task-stratified episode bootstrap; same resample applies to both methods.

    Strict pairing rejects missing episodes instead of selecting a convenient
    intersection. Intervals describe evaluation variability conditional on this
    training seed, not training robustness.
    """
    a, b = _indexed(left), _indexed(right)
    if a.keys() != b.keys():
        raise ValueError("Paired comparison requires exactly the same task/episode seeds")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    groups: dict[str, list[float]] = defaultdict(list)
    for key in sorted(a):
        groups[key[0]].append(
            float(a[key].get("status") == "success") - float(b[key].get("status") == "success")
        )
    if not groups:
        return {"status": "not yet run", "paired_episodes": 0, "macro_success_difference": None, "ci95": None}
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(values) for values in groups.values()]
    replicate = np.zeros(bootstrap_samples)
    for values in arrays:
        replicate += values[rng.integers(0, len(values), size=(bootstrap_samples, len(values)))].mean(axis=1)
    replicate /= len(arrays)
    return {
        "status": "measured",
        "paired_episodes": len(a),
        "task_count": len(arrays),
        "macro_success_difference": float(np.mean([values.mean() for values in arrays])),
        "ci95": np.quantile(replicate, [0.025, 0.975]).tolist(),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "uncertainty": "task-stratified paired episode bootstrap, conditional on training seed",
    }
