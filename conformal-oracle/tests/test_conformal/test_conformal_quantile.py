"""Regression tests for the finite-sample split-conformal quantile.

Pins the order statistic returned by ``conformal_quantile`` on cases with a
known closed-form answer, and guards the property that it is never less
conservative than the plain empirical quantile. This is the guarantee that the
250-day rolling correction lost when it used ``np.quantile(scores, 1 - alpha)``
directly (fixed in v0.3.1).
"""

import numpy as np
import pytest

from conformal_oracle.conformal.quantile import conformal_quantile
from conformal_oracle.conformal.rolling import compute_qv_roll_from_scores


def test_order_statistic_known_case_alpha05():
    # n=100, alpha=0.05 -> k = ceil(101*0.95) = 96 -> 96th smallest = 95.0
    scores = np.arange(100.0)
    assert conformal_quantile(scores, 0.05) == 95.0


def test_order_statistic_known_case_alpha01():
    # n=100, alpha=0.01 -> k = ceil(101*0.99) = 100 -> 100th smallest = 99.0
    scores = np.arange(100.0)
    assert conformal_quantile(scores, 0.01) == 99.0


def test_order_statistic_rolling_window_250():
    # The rolling default: w=250, alpha=0.01 -> k = ceil(251*0.99) = 249
    # -> 249th smallest of 0..249 = 248.0  (plain np.quantile gives 247.02)
    scores = np.arange(250.0)
    assert conformal_quantile(scores, 0.01) == 248.0
    assert conformal_quantile(scores, 0.01) > np.quantile(scores, 0.99)


def test_never_less_conservative_than_plain_quantile():
    rng = np.random.default_rng(2026)
    for _ in range(50):
        scores = rng.standard_normal(rng.integers(120, 800))
        assert conformal_quantile(scores, 0.01) >= np.quantile(scores, 0.99) - 1e-12


def test_plus_infinity_case_returns_max():
    # alpha < 1/(n+1): k = ceil((n+1)(1-alpha)) > n -> clip to the max score
    scores = np.arange(50.0)
    assert conformal_quantile(scores, 0.01) == 49.0  # ceil(51*0.99)=51 -> max


def test_empty_scores_returns_zero():
    assert conformal_quantile(np.array([]), 0.01) == 0.0


def test_rolling_uses_conformal_quantile_per_window():
    rng = np.random.default_rng(7)
    scores = rng.standard_normal(600)
    window = 250
    qv = compute_qv_roll_from_scores(scores, 0.01, window)
    # each entry must be the finite-sample conformal quantile of its window
    for i in (0, 100, len(qv) - 1):
        expected = conformal_quantile(scores[i : i + window], 0.01)
        assert qv[i] == pytest.approx(expected)


def test_finite_sample_coverage_at_short_window():
    """On i.i.d. scores the conformal rolling correction must not under-cover
    at the 250-day window the way the plain quantile did (>=1-alpha coverage)."""
    rng = np.random.default_rng(11)
    n, window, alpha = 4000, 250, 0.01
    scores = rng.standard_normal(n)
    qv = compute_qv_roll_from_scores(scores, alpha, window)
    # coverage of the one-step-ahead score by its rolling threshold
    covered = scores[window:] <= qv[: n - window]
    assert covered.mean() >= 1 - alpha - 0.005
