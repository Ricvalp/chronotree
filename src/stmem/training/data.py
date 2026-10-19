"""Deterministic task-balanced sampling, with disjoint rank positions."""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import random
import time
import torch
from stmem.config import digest
from stmem.data.canonical import valid_decision_indices
from stmem.data.dataset import EpisodeDataset, synthetic_episode
from stmem.data.paths import alignment_path


class TrainingData:
    def __init__(self, cfg, *, progress=None):
        started = time.perf_counter()

        def report(phase, **values):
            if progress is not None:
                progress({"phase": phase, "elapsed_seconds": time.perf_counter() - started, **values})

        report("catalog_started")
        self.cfg = cfg
        self.tasks = list(cfg["data"]["tasks"])
        self.synthetic = cfg["data"].get("synthetic", False)
        self.refs = []
        self.by_task = {task: [] for task in self.tasks}
        self.horizon = cfg["model"]["horizon"]
        self.stride = cfg["data"]["decision_stride"]
        if self.synthetic:
            if not cfg.get("debug"):
                raise ValueError("Synthetic input requires debug: true")
            for ti, task in enumerate(self.tasks):
                for i in range(cfg["data"]["episodes_per_task"]):
                    self.by_task[task].append(len(self.refs))
                    self.refs.append((task, i, ti * 10000 + i))
            self.identity = digest({"synthetic": cfg["data"], "points": cfg["model"]["points"]})
        else:
            root = Path(cfg["paths"]["data_root"])
            alignment = alignment_path(cfg)
            self.dataset = EpisodeDataset(
                root / cfg["data"].get("prepared_subdir", "prepared"), self.tasks, self.horizon, alignment
            )
            self.dataset.validate_preprocessing(cfg)
            cap = int(cfg["data"]["episodes_per_task"] * cfg["data"].get("fraction", 1.0))
            for task in self.tasks:
                indices = self.dataset.by_task.get(task, [])
                if len(indices) < cap:
                    raise ValueError(
                        f"{task}: need {cap} prepared episodes, found {len(indices)}; do not silently reduce data"
                    )
                for index in indices[:cap]:
                    self.by_task[task].append(len(self.refs))
                    self.refs.append(index)
            self.identity = digest(
                {
                    "manifests": [r.metadata for r in self.dataset.episodes],
                    "alignment": self.dataset.alignment,
                    "data": cfg["data"],
                }
            )
        self.candidates = {}
        self._action_count = torch.zeros(8, dtype=torch.long)
        self._action_total = torch.zeros(8, dtype=torch.float64)
        self._action_squares = torch.zeros(8, dtype=torch.float64)
        done = 0
        report("index_started", episodes_done=done, episodes_total=len(self.refs))
        # One action-only pass builds the index and training normalization stats.
        # Accumulate in the same task/episode order as the original full reader.
        for task in self.tasks:
            items = []
            for index in self.by_task[task]:
                if self.synthetic:
                    episode = self.episode(index)
                    execution_length = len(episode.execution)
                    conditioning_length = len(episode.conditioning)
                else:
                    episode = self.dataset.load_episode_actions(self.refs[index])
                    execution_length = episode.execution_length
                    conditioning_length = episode.conditioning_length
                if (
                    execution_length > cfg["model"]["max_execution_frames"]
                    or conditioning_length > cfg["model"]["max_conditioning_frames"]
                ):
                    raise ValueError(f"Episode {episode.episode_id} exceeds declared memory length bound")
                items.extend(
                    (index, t)
                    for t in valid_decision_indices(
                        episode.action_valid, episode.alignment, horizon=self.horizon, stride=self.stride
                    )
                )
                # Each issued execution command contributes once, independent of
                # candidate selection, chunk overlaps and partially valid joints.
                mask = episode.action_valid.bool() & torch.isfinite(episode.actions)
                actions = episode.actions.double().masked_fill(~mask, 0)
                self._action_count += mask.sum(0)
                self._action_total += actions.sum(0)
                self._action_squares += actions.square().sum(0)
                done += 1
            if not items:
                raise ValueError(f"No valid decisions for {task}")
            self.candidates[task] = items
            report("index_progress", task=task, episodes_done=done, episodes_total=len(self.refs))
        self.startup_seconds = time.perf_counter() - started
        report(
            "ready",
            episodes_done=done,
            episodes_total=len(self.refs),
            decisions=sum(len(items) for items in self.candidates.values()),
        )

    @lru_cache(maxsize=2)
    def episode(self, index):
        if not self.synthetic:
            return self.dataset.load_episode(self.refs[index])
        task, i, seed = self.refs[index]
        d = self.cfg["data"]
        episode = synthetic_episode(
            d.get("synthetic_execution_frames", 12),
            d.get("synthetic_conditioning_frames", 5),
            self.cfg["model"]["points"],
            seed,
        )
        episode.task = task
        episode.episode_id = f"episode_{i}"
        return episode

    def decision(self, index, time):
        return self.episode(index).decision(time, self.horizon)

    @lru_cache(maxsize=32)
    def _permutation(self, task, epoch):
        items = list(self.candidates[task])
        rng = random.Random(self.cfg["seed"] * 1000003 + self.tasks.index(task) * 1009 + epoch)
        rng.shuffle(items)
        # Shuffled length buckets reduce adjacent accumulation memory variance.
        bucket = self.cfg["training"].get("length_bucket_size", 32)
        return [
            x
            for start in range(0, len(items), bucket)
            for x in sorted(items[start : start + bucket], key=lambda p: p[1])
        ]

    def sample_position(self, position):
        task = self.tasks[position % len(self.tasks)]
        within = position // len(self.tasks)
        n = len(self.candidates[task])
        ep, time = self._permutation(task, within // n)[within % n]
        return self.decision(ep, time)

    def fit_normalizer(self, normalizer):
        # The small sufficient statistics were collected during indexing. No
        # second dataset pass, observation reads, or episode-LRU churn is needed.
        count, total, squares = self._action_count, self._action_total, self._action_squares
        mean = total / count.clamp_min(1)
        var = (squares / count.clamp_min(1) - mean.square()).clamp_min(0)
        normalizer.mean.copy_(mean.float())
        normalizer.std.copy_(var.sqrt().clamp_min(0.01).float())
        normalizer.count.copy_(count)

    def label_partition(self, episode_index):
        task = next(t for t, ids in self.by_task.items() if episode_index in ids)
        ids = list(self.by_task[task])
        random.Random(self.cfg["labels"]["seed"] + self.tasks.index(task)).shuffle(ids)
        cutoff = max(1, min(len(ids) - 1, int(len(ids) * self.cfg["labels"]["split_fraction"])))
        if len(ids) < 2:
            raise ValueError("Router-label diagnostics require at least two episodes per task")
        return "train" if episode_index in ids[:cutoff] else "validation"
