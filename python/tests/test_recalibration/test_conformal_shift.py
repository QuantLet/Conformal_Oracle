"""Tests for ConformalShift RecalibrationMethod wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.recalibration import ConformalShift, RecalibrationMethod


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


def test_conformal_shift_protocol():
    """ConformalShift implements RecalibrationMethod."""
    assert isinstance(ConformalShift(), RecalibrationMethod)


def test_conformal_shift_matches_default(garch_returns):
    """ConformalShift recalibration should produce same results as default."""
    fc = HistoricalSimulationForecaster(window=250)

    result_default = audit_static(
        garch_returns, fc, alpha=0.01,
    )

    result_recal = audit_static(
        garch_returns, fc, alpha=0.01,
        recalibration=ConformalShift(),
    )

    np.testing.assert_allclose(
        result_recal.violation_rate_corrected,
        result_default.violation_rate_corrected,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result_recal.var_corrected.values,
        result_default.var_corrected.values,
        rtol=1e-6,
    )


def test_conformal_shift_additive():
    """ConformalShift adds a constant to VaR."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(500) * 0.01
    var_raw = np.abs(rng.standard_normal(500)) * 0.02

    cs = ConformalShift()
    cs.fit(var_raw, realised, alpha=0.01)

    test1 = np.array([0.01, 0.02, 0.03])
    test2 = np.array([0.04, 0.05])

    corrected1 = cs.apply(test1)
    corrected2 = cs.apply(test2)

    shift1 = corrected1 - test1
    shift2 = corrected2 - test2

    assert np.allclose(shift1, shift1[0])
    assert np.allclose(shift1[0], shift2[0])
