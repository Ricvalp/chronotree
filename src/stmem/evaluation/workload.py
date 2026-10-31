"""Deterministic episode and video assignments shared by serial/parallel eval."""

from __future__ import annotations


def planned_episodes(cfg, count=None):
    count = cfg["evaluation"]["episode_count"] if count is None else count
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 50:
        raise ValueError("Official split provides 1–50 episode configurations per task")
    tasks = cfg["data"]["tasks"]
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("Evaluation task list must be nonempty and contain no duplicates")
    return [(task, episode) for task in tasks for episode in range(count)]


def video_episodes(cfg, count=None):
    """One global cap, spread across tasks, independent of shard/worker count."""
    planned = planned_episodes(cfg, count)
    cap = cfg["evaluation"]["video_cap"]
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        raise ValueError("evaluation.video_cap must be a nonnegative integer")
    task_order = {task: index for index, task in enumerate(cfg["data"]["tasks"])}
    return sorted(planned, key=lambda pair: (pair[1], task_order[pair[0]]))[:cap]
