"""T1: HistoricalQuantileRecalibration integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.recalibration import HistoricalQuantileRecalibration


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


def test_hq_discards_base():
    """HQ ignores raw forecasts; output is constant."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(500) * 0.01
    var_raw = np.abs(rng.standard_normal(500)) * 0.02

    hq = HistoricalQuantileRecalibration()
    hq.fit(var_raw, realised, alpha=0.01)
    corrected = hq.apply(var_raw)

    assert np.all(corrected == corrected[0])


def test_hq_violation_rate_near_alpha(garch_returns):
    """Corrected violation rate should be close to alpha."""
    recal = HistoricalQuantileRecalibration()
    result = audit_static(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        alpha=0.01,
        recalibration=recal,
    )
    assert abs(result.violation_rate_corrected - 0.01) < 0.02


def test_hq_protocol_compliance():
    """HQ implements RecalibrationMethod protocol."""
    from conformal_oracle.recalibration import RecalibrationMethod

    assert isinstance(HistoricalQuantileRecalibration(), RecalibrationMethod)


def test_hq_constant_across_test_sets():
    """Output value depends only on calibration data."""
    rng = np.random.default_rng(99)
    realised = rng.standard_normal(500) * 0.01
    var_raw_cal = np.abs(rng.standard_normal(500)) * 0.02

    hq = HistoricalQuantileRecalibration()
    hq.fit(var_raw_cal, realised, alpha=0.01)

    test1 = hq.apply(np.ones(100) * 0.01)
    test2 = hq.apply(np.ones(50) * 0.05)
    assert test1[0] == test2[0]


def test_hq_quantile_sign_convention():
    """VaR output should be positive for typical loss quantiles."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(1000) * 0.01
    var_raw = np.abs(rng.standard_normal(1000)) * 0.02

    hq = HistoricalQuantileRecalibration()
    hq.fit(var_raw, realised, alpha=0.01)
    corrected = hq.apply(var_raw[:10])

    assert corrected[0] > 0
