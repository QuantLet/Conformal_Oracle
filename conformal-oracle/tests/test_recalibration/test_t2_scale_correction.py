"""T2: ScaleCorrectionRecalibration integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.recalibration import ScaleCorrectionRecalibration


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


def test_scale_multiplicative():
    """Scale correction should multiply VaR by c = alpha / pihat."""
    realised = np.concatenate([
        np.full(90, 0.001),
        np.full(10, -0.05),
    ])
    var_raw = np.full(100, 0.02)

    sc = ScaleCorrectionRecalibration()
    sc.fit(var_raw, realised, alpha=0.01)
    corrected = sc.apply(var_raw[:10])

    pihat = 10 / 100
    expected_c = 0.01 / pihat
    np.testing.assert_allclose(corrected, var_raw[:10] * expected_c, rtol=1e-10)


def test_scale_violation_rate_reasonable(garch_returns):
    """Corrected violation rate bounded but not necessarily near alpha.

    Scale Correction uses c = alpha / pihat_cal, a first-order
    multiplicative correction. On a well-calibrated base (HistSim
    on GARCH), pihat_cal is already near alpha, so c < 1 shrinks
    VaR, moving pihat AWAY from alpha. This is the expected
    behaviour documented in the paper's Table 12 where Scale
    Correction gives pihat = 0.264 on the TSFM panel.

    The formula only helps when the base overpredicts risk
    (pihat < alpha). On correctly-calibrated or underpredicting
    bases, it increases the violation rate.
    """
    recal = ScaleCorrectionRecalibration()
    result = audit_static(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        alpha=0.01,
        recalibration=recal,
    )
    assert 0 <= result.violation_rate_corrected <= 0.15


def test_scale_protocol_compliance():
    """Scale implements RecalibrationMethod protocol."""
    from conformal_oracle.recalibration import RecalibrationMethod

    assert isinstance(ScaleCorrectionRecalibration(), RecalibrationMethod)


def test_scale_preserves_relative_ordering():
    """If VaR_a > VaR_b, corrected_a > corrected_b."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(500) * 0.01
    var_raw = np.abs(rng.standard_normal(500)) * 0.02

    sc = ScaleCorrectionRecalibration()
    sc.fit(var_raw, realised, alpha=0.01)

    test_var = np.array([0.01, 0.02, 0.03])
    corrected = sc.apply(test_var)
    assert corrected[0] < corrected[1] < corrected[2]


def test_scale_zero_violations_fallback():
    """When pihat_cal = 0, c should default to 1.0."""
    realised = np.full(100, 0.01)
    var_raw = np.full(100, 0.001)

    sc = ScaleCorrectionRecalibration()
    sc.fit(var_raw, realised, alpha=0.01)
    corrected = sc.apply(var_raw[:10])

    np.testing.assert_allclose(corrected, var_raw[:10])
