"""Descriptive paired-noise diagnostics for fixed memory-refinement moves.

Noise draws are the repeated measurements, not independent scenes.  Candidate
selection uses the first half of the draws; all selection-quality comparisons
use the second half.  These statistics describe teacher-agreement labels and
must not be presented as task success or population-level confidence intervals.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _mean_se(values):
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _sign(values, epsilon):
    return np.where(values > epsilon, 1, np.where(values < -epsilon, -1, 0))


def _agreement(left, right):
    # A tied ranking carries no directional preference: its expected agreement
    # with any ranking is 1/2.  Do not give two ties perfect agreement.
    return np.where((left == 0) | (right == 0), 0.5, (left == right).astype(float))


def summarize_gains(gains, epsilon=1e-6):
    """Summarize a finite ``[noise_draw, candidate_move]`` gain matrix.

    ``epsilon`` is a numerical/near-tie deadband in raw gain units, not a
    statistical significance threshold.  It should be at least the observed
    same-input numerical floor.  There must be at least four draws so both
    halves have enough samples for a standard error.

    Exact and epsilon-near ties share winner/selection credit equally.  The
    representative ``high_index`` and ``low_index`` are deterministic indices
    for visualization only; the held-out selection statistics average the
    entire corresponding tie set.
    """
    values = np.asarray(gains, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 4 or values.shape[1] < 1:
        raise ValueError("gains must have shape [at least four draws, at least one candidate]")
    if not np.isfinite(values).all():
        raise ValueError("gain measurements must be finite")
    if isinstance(epsilon, bool) or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    epsilon = float(epsilon)
    draws, candidates = values.shape
    nselect = draws // 2
    selection, validation = values[:nselect], values[nselect:]
    selection_mean, validation_mean = selection.mean(0), validation.mean(0)
    high = np.flatnonzero(selection_mean >= selection_mean.max() - epsilon)
    low = np.flatnonzero(selection_mean <= selection_mean.min() + epsilon)
    # Shared draws make these paired differences; independent standard errors
    # for two means would throw away that pairing.
    high_draw = validation[:, high].mean(1)
    low_draw = validation[:, low].mean(1)
    random_draw = validation.mean(1)
    contrast = high_draw - low_draw
    selected_vs_random = high_draw - random_draw
    contrast_mean, contrast_se = _mean_se(contrast)
    improvement_mean, improvement_se = _mean_se(selected_vs_random)
    winners = values >= values.max(1, keepdims=True) - epsilon
    winner_probability = (winners / winners.sum(1, keepdims=True)).mean(0)
    signs = _sign(values, epsilon)
    pairs = list(combinations(range(candidates), 2))
    if pairs:
        left, right = np.asarray(pairs).T
        rankings = _sign(values[:, left] - values[:, right], epsilon)
        draw_left, draw_right = np.asarray(list(combinations(range(draws), 2))).T
        a, b = rankings[draw_left], rankings[draw_right]
        directional = (a != 0) & (b != 0)
        split_a = _sign(selection_mean[left] - selection_mean[right], epsilon)
        split_b = _sign(validation_mean[left] - validation_mean[right], epsilon)
        pairwise = {
            "independent_draw_agreement": float(_agreement(a, b).mean()),
            "non_tied_agreement": float((a[directional] == b[directional]).mean()) if directional.any() else None,
            "non_tied_fraction": float(directional.mean()),
            "split_mean_agreement": float(_agreement(split_a, split_b).mean()),
            "comparisons": int(a.size),
            "candidate_pairs": len(pairs),
            "note": "Descriptive correlated comparisons within one fixed scene/frontier; not independent trials.",
        }
    else:
        pairwise = {
            "independent_draw_agreement": None, "non_tied_agreement": None,
            "non_tied_fraction": None, "split_mean_agreement": None,
            "comparisons": 0, "candidate_pairs": 0,
            "note": "Only one candidate; ranking repeatability is undefined.",
        }
    return {
        "draws": draws, "candidates": candidates, "selection_draws": nselect,
        "validation_draws": draws - nselect, "epsilon": epsilon,
        "candidates_stats": {
            "mean": values.mean(0).tolist(), "std": values.std(0, ddof=1).tolist(),
            "se": (values.std(0, ddof=1) / np.sqrt(draws)).tolist(),
            "positive_probability": (signs > 0).mean(0).tolist(),
            "negative_probability": (signs < 0).mean(0).tolist(),
            "near_zero_probability": (signs == 0).mean(0).tolist(),
            "winner_probability": winner_probability.tolist(),
        },
        "selection": {
            "high_index": int(np.argmax(selection_mean)), "low_index": int(np.argmin(selection_mean)),
            "high_tie_indices": high.tolist(), "low_tie_indices": low.tolist(),
            "mean": selection_mean.tolist(),
        },
        "validation": {
            "mean": validation_mean.tolist(),
            "high_mean_gain": float(high_draw.mean()), "low_mean_gain": float(low_draw.mean()),
            "high_minus_low_mean_gain": contrast_mean, "high_minus_low_se": contrast_se,
            "high_beats_low_probability": float(np.where(np.abs(contrast) <= epsilon, 0.5, contrast > epsilon).mean()),
            "random_mean_gain": float(random_draw.mean()),
            "best_fixed_candidate_mean_gain": float(validation_mean.max()),
            "best_fixed_candidate_note": "Maximum validation mean across fixed candidates: descriptive optimistic bound, not an independently selected policy and not a per-draw oracle.",
            "selected_vs_random_mean_gain": improvement_mean, "selected_vs_random_se": improvement_se,
        },
        "pairwise": pairwise,
        "scope": "Noise repeatability conditional on a fixed scene, frontier and candidate set; not task success.",
    }


def summarize_experiment(result):
    """Equal-context descriptive means; never count draws as new scenes."""
    contexts = result["contexts"]
    aggregates = {}
    for reader in sorted({c["reader"] for c in contexts}):
        selected = [c for c in contexts if c["reader"] == reader]
        for objective in selected[0]["statistics"]:
            values = [c["statistics"][objective] for c in selected]
            aggregates[f"{reader}/{objective}"] = {
                "contexts": len(selected),
                "candidate_moves": sum(s["candidates"] for s in values),
                "cross_draw_rank_agreement": float(np.mean([
                    s["pairwise"]["independent_draw_agreement"] for s in values])),
                "split_mean_rank_agreement": float(np.mean([
                    s["pairwise"]["split_mean_agreement"] for s in values])),
                "high_beats_low_on_heldout_mean": sum(
                    s["validation"]["high_minus_low_mean_gain"] > s["epsilon"] for s in values),
                "candidates_with_both_gain_signs": sum(
                    bool(np.any(np.asarray(ca["gains"][objective]) > s["epsilon"]) and
                         np.any(np.asarray(ca["gains"][objective]) < -s["epsilon"]))
                    for c, s in zip(selected, values) for ca in c["candidates"]),
            }
    return {
        "scope": result["protocol"]["scope"], "reader_contexts": len(contexts),
        "unique_decisions": len({(c["task"], c["episode"], c["decision_index"]) for c in contexts}),
        "episodes": sorted({f"{c['task']}/{c['episode']}" for c in contexts}),
        "noise_draws": len(result["protocol"]["noise_seeds"]), "aggregates": aggregates,
        "checks": result.get("checks", {}),
        "exact_hpc_compact_file": result["protocol"].get("exact_hpc_compact_file"),
    }
