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
    covered = scores[window:] <= qv[: n - window]
    assert covered.mean() >= 1 - alpha - 0.005


def test_rolling_coverage_panel_near_nominal():
    """Behavioral regression: end-to-end rolling coverage on a fixed synthetic
    panel stays close to nominal.

    Each of the 20 seeded series has N(0,1) returns and a deliberately
    miscalibrated raw lower quantile (-2.0; the true 1% quantile is ~-2.326),
    corrected by the 250-day rolling conformal shift. The old plain-quantile
    code under-covered here at a panel-mean violation of ~0.0135; the
    finite-sample conformal quantile brings it to ~0.008. The upper bound
    0.012 is precisely the guard the pre-fix code violated.
    """
    alpha, window, n, n_series = 0.01, 250, 2000, 20
    violations = []
    for k in range(n_series):
        rng = np.random.default_rng(100 + k)
        r = rng.standard_normal(n)
        q_raw = -2.0  # miscalibrated raw lower quantile (positive-loss VaR = 2.0)
        scores = q_raw - r  # S_t = q_lo - r
        qv = compute_qv_roll_from_scores(scores, alpha, window)
        var_corr = (-q_raw) + qv  # positive-loss corrected VaR path
        viol = (r[window:] < -var_corr[: n - window]).astype(int)
        violations.append(viol.mean())
    panel_mean = float(np.mean(violations))
    assert 0.005 <= panel_mean <= 0.012, f"panel-mean violation {panel_mean:.4f}"
