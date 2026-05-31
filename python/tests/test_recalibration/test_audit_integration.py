"""Integration tests: recalibration methods with audit_static."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.recalibration import (
    AdaptiveConformalInference,
    ConformalShift,
    HistoricalQuantileRecalibration,
    IsotonicQuantileRegression,
    LinearQuantileRegression,
    ScaleCorrectionRecalibration,
)


@pytest.fixture(scope="module")
def garch_returns():
    rng = np.random.default_rng(2026)
    n = 2000
    omega, alpha_g, beta_g = 1e-6, 0.05, 0.90
    r = np.empty(n)
    s2 = np.empty(n)
    s2[0] = omega / (1 - alpha_g - beta_g)
    for t in range(n):
        if t > 0:
            s2[t] = omega + alpha_g * r[t - 1] ** 2 + beta_g * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_normal()
    dates = pd.bdate_range("2018-01-02", periods=n)
    return pd.Series(r, index=dates, name="garch")


ALL_METHODS = [
    ConformalShift(),
    HistoricalQuantileRecalibration(),
    ScaleCorrectionRecalibration(),
    LinearQuantileRegression(),
    IsotonicQuantileRegression(),
    AdaptiveConformalInference(gamma=0.05),
]


@pytest.mark.parametrize(
    "method",
    ALL_METHODS,
    ids=[type(m).__name__ for m in ALL_METHODS],
)
def test_audit_static_with_recalibration(garch_returns, method):
    """Every recalibration method should integrate with audit_static."""
    result = audit_static(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        alpha=0.01,
        recalibration=method,
    )
    assert result.n_test > 0
    assert len(result.var_corrected) == result.n_test
    assert 0 <= result.violation_rate_corrected <= 1


@pytest.mark.parametrize(
    "method",
    ALL_METHODS,
    ids=[type(m).__name__ for m in ALL_METHODS],
)
def test_result_fields_populated(garch_returns, method):
    """All standard result fields should be populated."""
    result = audit_static(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        alpha=0.01,
        recalibration=method,
    )
    assert np.isfinite(result.kupiec_pvalue_corrected)
    assert np.isfinite(result.quantile_score_corrected)
    assert np.isfinite(result.fz_score_corrected)
    assert result.regime in ("signal-preserving", "replacement")


def test_default_none_unchanged(garch_returns):
    """recalibration=None should produce identical results to no arg."""
    fc = HistoricalSimulationForecaster(window=250)
    r1 = audit_static(garch_returns, fc, alpha=0.01)
    r2 = audit_static(garch_returns, fc, alpha=0.01, recalibration=None)

    np.testing.assert_allclose(
        r1.var_corrected.values,
        r2.var_corrected.values,
        rtol=1e-10,
    )
