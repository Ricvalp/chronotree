import json

import numpy as np
import pytest

from stmem.inspection.gain_statistics import summarize_gains


def test_stable_ordering_and_paired_difference_cancel_common_noise():
    common = np.arange(16, dtype=float)[:, None]
    report = summarize_gains(common + np.asarray([3, 1, -2]))
    assert report["selection"]["high_index"] == 0
    assert report["selection"]["low_index"] == 2
    assert report["validation"]["high_minus_low_mean_gain"] == 5
    assert report["validation"]["high_minus_low_se"] == 0
    assert report["validation"]["selected_vs_random_se"] == 0
    assert report["pairwise"]["independent_draw_agreement"] == 1
    assert report["pairwise"]["split_mean_agreement"] == 1
    assert report["candidates_stats"]["winner_probability"] == [1, 0, 0]
    json.dumps(report, allow_nan=False)


def test_selection_cannot_peek_at_validation_reversal():
    gains = np.asarray([[10, 0]] * 4 + [[0, 10]] * 4)
    report = summarize_gains(gains)
    assert report["selection"]["high_index"] == 0
    assert report["validation"]["high_mean_gain"] == 0
    assert report["validation"]["high_minus_low_mean_gain"] == -10
    assert report["validation"]["selected_vs_random_mean_gain"] == -5
    assert report["pairwise"]["split_mean_agreement"] == 0
    assert report["pairwise"]["independent_draw_agreement"] == pytest.approx(12 / 28)


def test_ties_have_fair_credit_and_no_artificial_rank_consistency():
    gains = np.zeros((5, 3))
    report = summarize_gains(gains)
    assert report["selection_draws"] == 2
    assert report["validation_draws"] == 3
    assert report["selection"]["high_tie_indices"] == [0, 1, 2]
    assert report["selection"]["low_tie_indices"] == [0, 1, 2]
    assert report["candidates_stats"]["winner_probability"] == pytest.approx([1 / 3] * 3)
    assert report["pairwise"]["independent_draw_agreement"] == 0.5
    assert report["pairwise"]["non_tied_agreement"] is None
    assert report["pairwise"]["non_tied_fraction"] == 0
    assert report["validation"]["high_beats_low_probability"] == 0.5


def test_tied_selection_averages_candidates_without_row_order_advantage():
    gains = np.asarray([[1, 1, 0]] * 2 + [[0, 10, 0]] * 2)
    report = summarize_gains(gains)
    assert report["selection"]["high_tie_indices"] == [0, 1]
    assert report["validation"]["high_mean_gain"] == 5
    assert report["validation"]["high_minus_low_mean_gain"] == 5


def test_numerical_deadband_avoids_counting_roundoff_as_signal():
    gains = np.asarray([[1e-8, -1e-8]] * 4)
    report = summarize_gains(gains, epsilon=1e-6)
    assert report["candidates_stats"]["near_zero_probability"] == [1, 1]
    assert report["candidates_stats"]["winner_probability"] == [0.5, 0.5]
    assert report["pairwise"]["non_tied_agreement"] is None


def test_best_fixed_mean_is_not_optimistic_per_draw_oracle():
    gains = np.asarray([[1, -1], [-1, 1]] * 4)
    report = summarize_gains(gains)
    assert report["validation"]["best_fixed_candidate_mean_gain"] == 0
    assert np.max(gains[4:], axis=1).mean() == 1


def test_one_candidate_has_no_ranking_but_valid_gain_summary():
    report = summarize_gains(np.asarray([[1], [2], [3], [4]]))
    assert report["pairwise"]["independent_draw_agreement"] is None
    assert report["pairwise"]["comparisons"] == 0
    assert report["validation"]["selected_vs_random_mean_gain"] == 0
    assert report["candidates_stats"]["winner_probability"] == [1]


@pytest.mark.parametrize("gains", [[], np.ones(4), np.ones((3, 2)), np.ones((4, 0)), [[np.nan, 0]] * 4, [[np.inf, 0]] * 4])
def test_invalid_measurements_rejected(gains):
    with pytest.raises(ValueError):
        summarize_gains(gains)


@pytest.mark.parametrize("epsilon", [-1, float("nan"), float("inf"), True])
def test_invalid_deadband_rejected(epsilon):
    with pytest.raises(ValueError):
        summarize_gains(np.zeros((4, 2)), epsilon=epsilon)
