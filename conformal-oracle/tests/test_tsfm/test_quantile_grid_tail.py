"""Verify QuantileGridDistribution tail completion for TSFM quantile grids.

This tests the exact integration path that TimesFM 2.5 exercises:
9-decile grid → Student-t tail completion → 1% quantile.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from conformal_oracle._types import QuantileGridDistribution


def test_student_t_tail_completion_on_synthetic_grid():
    """Build a known Student-t distribution, sample 9 deciles,
    query 1% quantile via QuantileGridDistribution, compare
    against true ppf. The quantile-function fit should recover
    the parameters well enough to match within tight tolerance.
    """
    df, loc, scale = 5.0, -0.001, 0.015
    rv = stats.t(df=df, loc=loc, scale=scale)

    levels = np.arange(0.1, 1.0, 0.1)
    quantiles = np.array([rv.ppf(p) for p in levels])

    grid = QuantileGridDistribution(levels=levels, quantiles=quantiles)

    q01_grid = grid.quantile(0.01, completion="student_t")
    q01_true = rv.ppf(0.01)

    assert np.isfinite(q01_grid)
    assert q01_grid < quantiles[0]
    np.testing.assert_allclose(q01_grid, q01_true, atol=0.002)


def test_student_t_parameter_recovery():
    """Fitting from 9 deciles should recover known Student-t params."""
    df_true, loc_true, scale_true = 6.0, -0.002, 0.012
    rv = stats.t(df=df_true, loc=loc_true, scale=scale_true)

    levels = np.arange(0.1, 1.0, 0.1)
    quantiles = np.array([rv.ppf(p) for p in levels])

    grid = QuantileGridDistribution(levels=levels, quantiles=quantiles)
    df_fit, loc_fit, scale_fit = grid._fit_parametric("student_t")

    np.testing.assert_allclose(loc_fit, loc_true, atol=1e-4)
    np.testing.assert_allclose(scale_fit, scale_true, atol=1e-4)
    np.testing.assert_allclose(df_fit, df_true, atol=0.5)


def test_tail_completion_ordering():
    """q(0.01) < q(0.05) < q(0.10) for a realistic return grid."""
    levels = np.arange(0.1, 1.0, 0.1)
    quantiles = np.array([
        -0.025, -0.012, -0.005, -0.001, 0.001,
        0.004, 0.008, 0.015, 0.028,
    ])

    grid = QuantileGridDistribution(levels=levels, quantiles=quantiles)
    q01 = grid.quantile(0.01)
    q05 = grid.quantile(0.05)
    q10 = grid.quantile(0.10)

    assert q01 < q05 < q10


def test_tail_completion_finite_for_degenerate_grid():
    """Even for a nearly-flat grid, tail completion shouldn't NaN."""
    levels = np.arange(0.1, 1.0, 0.1)
    quantiles = np.full(9, 0.001)

    grid = QuantileGridDistribution(levels=levels, quantiles=quantiles)
    q01 = grid.quantile(0.01)
    assert np.isfinite(q01)


def test_es_from_quantile_grid():
    """Expected shortfall should be computable from 9-decile grid."""
    levels = np.arange(0.1, 1.0, 0.1)
    quantiles = np.array([
        -0.025, -0.012, -0.005, -0.001, 0.001,
        0.004, 0.008, 0.015, 0.028,
    ])

    grid = QuantileGridDistribution(levels=levels, quantiles=quantiles)
    es = grid.expected_shortfall(0.01)
    q01 = grid.quantile(0.01)
    assert np.isfinite(es)
    assert es <= q01
