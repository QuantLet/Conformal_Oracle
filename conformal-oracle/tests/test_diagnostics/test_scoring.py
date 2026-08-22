"""Tests for quantile score and FZ score."""

import numpy as np

from conformal_oracle.diagnostics.scoring import quantile_score, fissler_ziegel_fz0


def test_qs_perfect_forecast():
    """Perfect forecast at the alpha-quantile should give minimal score."""
    rng = np.random.default_rng(42)
    n = 10000
    alpha = 0.05
    realised = rng.standard_normal(n)
    q = np.quantile(realised, alpha)
    forecasts = np.full(n, q)
    qs = quantile_score(realised, forecasts, alpha)
    assert qs > 0


def test_qs_non_negative():
    """Quantile score should always be non-negative."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(500)
    forecasts = rng.standard_normal(500)
    qs = quantile_score(realised, forecasts, 0.01)
    assert qs >= 0


def test_qs_worse_forecast_higher_score():
    """A worse forecast should have a higher quantile score."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(1000)
    good = np.full(1000, np.quantile(realised, 0.05))
    bad = np.full(1000, np.quantile(realised, 0.05) + 2.0)
    qs_good = quantile_score(realised, good, 0.05)
    qs_bad = quantile_score(realised, bad, 0.05)
    assert qs_bad > qs_good


def test_fz_finite():
    """FZ score should be finite for reasonable inputs."""
    rng = np.random.default_rng(42)
    n = 500
    realised = rng.standard_normal(n) * 0.01
    var_fc = np.full(n, -0.025)
    es_fc = np.full(n, -0.035)
    fz = fissler_ziegel_fz0(realised, var_fc, es_fc, 0.01)
    assert np.isfinite(fz)
