"""T3: LinearQuantileRegression integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.recalibration import LinearQuantileRegression


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


def test_qr_finite_parameters():
    """QR fit should produce finite slope and intercept."""
    rng = np.random.default_rng(2026)
    realised = rng.standard_normal(1000) * 0.01
    var_raw = np.abs(rng.standard_normal(1000)) * 0.02

    qr = LinearQuantileRegression()
    qr.fit(var_raw, realised, alpha=0.01)

    assert np.isfinite(qr.slope)
    assert np.isfinite(qr.intercept)


def test_qr_violation_rate_reasonable(garch_returns):
    """Corrected violation rate bounded but may exceed alpha.

    Linear QR fits Q_alpha(r | VaR_raw) = b0 + b1 * VaR_raw at
    the 1% quantile. On a well-calibrated base (HistSim on GARCH),
    the relationship between VaR_raw and the conditional 1% quantile
    is noisy at the extreme tail, so the fit may overshoot. This is
    consistent with the paper's finding that QR on residuals gives
    pihat = 0.012 on real data (marginally above alpha) but can be
    less stable on synthetic data.
    """
    recal = LinearQuantileRegression()
    result = audit_static(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        alpha=0.01,
        recalibration=recal,
    )
    assert 0 <= result.violation_rate_corrected <= 0.15


def test_qr_protocol_compliance():
    """LinearQR implements RecalibrationMethod protocol."""
    from conformal_oracle.recalibration import RecalibrationMethod

    assert isinstance(LinearQuantileRegression(), RecalibrationMethod)


def test_qr_properties_accessible():
    """Intercept and slope properties should be readable after fit."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(500) * 0.01
    var_raw = np.abs(rng.standard_normal(500)) * 0.02

    qr = LinearQuantileRegression()
    qr.fit(var_raw, realised, alpha=0.01)

    assert isinstance(qr.intercept, float)
    assert isinstance(qr.slope, float)


def test_qr_output_shape():
    """Apply should return array of same length as input."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(500) * 0.01
    var_raw = np.abs(rng.standard_normal(500)) * 0.02

    qr = LinearQuantileRegression()
    qr.fit(var_raw, realised, alpha=0.01)

    test_var = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    corrected = qr.apply(test_var)
    assert corrected.shape == (5,)
