"""Synthetic diagnostics for task-balanced batching, ordering, and causal reads."""

from collections import Counter
import random

import pytest
import torch

from stmem.config import load_config
from stmem.training.data import TrainingData, rank_positions


def _indexed_data(seed=7):
    """Metadata-only catalog with enough candidates to expose a 16-update ramp."""
    data = TrainingData.__new__(TrainingData)
    data.cfg = {"seed": seed, "training": {"length_bucket_size": 1}}
    data.tasks = [f"task_{i}" for i in range(16)]
    data.candidates = {
        task: [(4 * task_index + episode, time) for episode in range(4) for time in range(64)]
        for task_index, task in enumerate(data.tasks)
    }
    return data


def _synthetic_data():
    cfg = load_config("configs/debug/teacher.yaml")
    cfg["data"].update(
        tasks=["VideoUnmask", "PickXtimes", "PatternLock"],
        episodes_per_task=3,
        synthetic_execution_frames=9,
        synthetic_conditioning_frames=2,
        decision_stride=1,
    )
    cfg["model"].update(points=4, horizon=4)
    return TrainingData(cfg)


def test_epoch_shuffle_preserves_every_candidate_and_is_seeded():
    data = _indexed_data()
    for task_index, task in enumerate(data.tasks):
        for epoch in (0, 1, 17):
            expected = list(data.candidates[task])
            random.Random(data.cfg["seed"] * 1000003 + task_index * 1009 + epoch).shuffle(expected)
            actual = data._permutation(task, epoch)
            assert actual == expected
            assert Counter(actual) == Counter(data.candidates[task])
        assert data._permutation(task, 0) != data._permutation(task, 1)
    assert data._permutation(data.tasks[0], 0) != _indexed_data(seed=8)._permutation(data.tasks[0], 0)


def test_randomized_history_order_has_no_synchronized_sixteen_update_ramp():
    data = _indexed_data()
    # The previous sort of 32 candidates/task, at two candidates/task/update,
    # made the mean history length grow monotonically for 16 updates and reset.
    phases = [[] for _ in range(16)]
    for step in range(128):
        histories = [data.sample_reference(step * 32 + offset)[1] for offset in range(32)]
        phases[step % 16].append(sum(histories) / len(histories))
    means = torch.tensor([sum(values) / len(values) for values in phases])
    correlation = torch.corrcoef(torch.stack((torch.arange(16, dtype=torch.float32), means)))[0, 1]
    assert abs(float(correlation)) < 0.4
    assert float(means.max() - means.min()) < 10


def test_global_sample_order_balances_tasks_and_disjoint_rank_positions():
    data = _indexed_data()
    for step in range(8):
        start = step * 32
        references = [data.sample_reference(position) for position in range(start, start + 32)]
        assert Counter(episode_index // 4 for episode_index, _ in references) == dict.fromkeys(range(16), 2)
        rank_references = [
            {data.sample_reference(position) for position in range(start + rank, start + 32, 4)}
            for rank in range(4)
        ]
        assert all(len(items) == 8 for items in rank_references)
        assert len(set.union(*rank_references)) == 32
        assert set.union(*rank_references) == set(references)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_rank_positions_cover_each_global_batch_and_rotate_all_tasks(world):
    batch, task_count = 32, 16
    tasks_by_rank = [Counter() for _ in range(world)]
    for step in range(world):
        positions_by_rank = [rank_positions(step, batch, rank, world) for rank in range(world)]
        assert all(len(positions) == batch // world for positions in positions_by_rank)
        flattened = [position for positions in positions_by_rank for position in positions]
        assert sorted(flattened) == list(range(step * batch, (step + 1) * batch))
        for rank, positions in enumerate(positions_by_rank):
            tasks_by_rank[rank].update(position % task_count for position in positions)
    assert all(counts == dict.fromkeys(range(task_count), 2) for counts in tasks_by_rank)


@pytest.mark.parametrize("args", [(-1, 32, 0, 4), (0, 0, 0, 4), (0, 32, 4, 4), (0, 5, 0, 4)])
def test_rank_positions_reject_invalid_or_unequal_assignments(args):
    with pytest.raises(ValueError):
        rank_positions(*args)


def test_sample_references_resume_without_scans_or_global_rng_changes(monkeypatch):
    data = _indexed_data()
    python_rng, torch_rng = random.getstate(), torch.get_rng_state()
    positions = list(range(3500, 4200))  # Cross each task's epoch boundary.
    expected = [data.sample_reference(position) for position in positions]
    resumed = _indexed_data()

    def forbidden(*args):
        pytest.fail("Resolving sample metadata loaded a point-cloud episode")

    monkeypatch.setattr(resumed, "episode", forbidden)
    # Out-of-order rank reads and cache eviction cannot alter later decisions.
    assert [resumed.sample_reference(position) for position in reversed(positions)] == expected[::-1]
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


def _assert_frame_equal(actual, expected):
    for key in ("points", "valid", "proprio", "previous_action"):
        torch.testing.assert_close(getattr(actual, key), getattr(expected, key), rtol=0, atol=0)
    for key in ("action_available", "time", "source", "event_id", "source_index"):
        assert getattr(actual, key) == getattr(expected, key)


def _assert_decision_equal(actual, expected):
    for key in ("task", "episode_id", "index", "goal"):
        assert getattr(actual, key) == getattr(expected, key)
    torch.testing.assert_close(actual.target, expected.target, rtol=0, atol=0)
    torch.testing.assert_close(actual.target_valid, expected.target_valid)
    _assert_frame_equal(actual.current, expected.current)
    for key in ("conditioning", "execution_prefix"):
        actual_frames, expected_frames = getattr(actual, key), getattr(expected, key)
        assert len(actual_frames) == len(expected_frames)
        for actual_frame, expected_frame in zip(actual_frames, expected_frames):
            _assert_frame_equal(actual_frame, expected_frame)
    assert all(frame.source_index < actual.current.source_index for frame in actual.execution_prefix)
    assert all(not frame.action_available for frame in actual.conditioning)


def test_batched_fetch_preserves_order_targets_masks_and_causal_history(monkeypatch):
    data = _synthetic_data()
    positions_by_episode = {}
    for position in range(sum(map(len, data.candidates.values()))):
        episode_index, _ = data.sample_reference(position)
        positions_by_episode.setdefault(episode_index, []).append(position)
    episodes = list(positions_by_episode)[:4]
    positions = [
        positions_by_episode[episodes[0]][0],
        positions_by_episode[episodes[1]][0],
        positions_by_episode[episodes[2]][0],
        positions_by_episode[episodes[0]][1],
        positions_by_episode[episodes[3]][0],
        positions_by_episode[episodes[1]][1],
        positions_by_episode[episodes[0]][0],  # Duplicate request preserves output multiplicity.
    ]
    expected = [data.sample_position(position) for position in positions]
    loads = []
    original_episode = data.episode

    def load(index):
        loads.append(index)
        return original_episode(index)

    monkeypatch.setattr(data, "episode", load)
    actual = data.sample_positions(iter(positions))
    assert Counter(loads) == dict.fromkeys(episodes, 1)
    assert len(actual) == len(expected)
    for decision, oracle in zip(actual, expected):
        _assert_decision_equal(decision, oracle)


def test_empty_or_invalid_batch_does_not_load_episode(monkeypatch):
    data = _indexed_data()

    def forbidden(*args):
        pytest.fail("Invalid or empty batch loaded an episode")

    monkeypatch.setattr(data, "episode", forbidden)
    assert data.sample_positions([]) == []
    with pytest.raises(ValueError, match="nonnegative"):
        data.sample_positions([0, 1, -1])
    with pytest.raises(TypeError):
        data.sample_positions([0.5])


def test_task_epoch_cycles_independently_when_candidate_counts_differ():
    data = _indexed_data()
    data.candidates[data.tasks[0]] = data.candidates[data.tasks[0]][:3]
    for task_index, task in enumerate(data.tasks):
        count = len(data.candidates[task])
        for epoch in range(3):
            actual = [
                data.sample_reference((epoch * count + index) * len(data.tasks) + task_index)
                for index in range(count)
            ]
            assert Counter(actual) == Counter(data.candidates[task])
