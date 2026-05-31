"""T8: FilteredHistoricalSimulation integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.recalibration import FilteredHistoricalSimulation


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


def test_fhs_violation_rate(garch_returns):
    """FHS corrected violation rate should be in a reasonable range."""
    recal = FilteredHistoricalSimulation()
    result = audit_static(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        alpha=0.01,
        recalibration=recal,
    )
    assert 0 <= result.violation_rate_corrected <= 0.10


def test_fhs_output_shape():
    """Apply should return array of same length as input."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(1000) * 0.01
    var_raw = np.abs(rng.standard_normal(1000)) * 0.02

    fhs = FilteredHistoricalSimulation()
    fhs.fit(var_raw, realised, alpha=0.01)

    test_var = np.array([0.01, 0.02, 0.03])
    corrected = fhs.apply(test_var)
    assert corrected.shape == (3,)


def test_fhs_positive_output():
    """Corrected VaR should be positive."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(1000) * 0.01
    var_raw = np.abs(rng.standard_normal(1000)) * 0.02

    fhs = FilteredHistoricalSimulation()
    fhs.fit(var_raw, realised, alpha=0.01)

    test_var = np.array([0.01, 0.02, 0.03])
    corrected = fhs.apply(test_var)
    assert np.all(corrected > 0)


def test_fhs_preserves_ordering():
    """Larger raw VaR should produce larger corrected VaR."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(1000) * 0.01
    var_raw = np.abs(rng.standard_normal(1000)) * 0.02

    fhs = FilteredHistoricalSimulation()
    fhs.fit(var_raw, realised, alpha=0.01)

    test_var = np.array([0.01, 0.02, 0.03])
    corrected = fhs.apply(test_var)
    assert corrected[0] < corrected[1] < corrected[2]


def test_fhs_custom_lambda():
    """Custom EWMA lambda should work."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(1000) * 0.01
    var_raw = np.abs(rng.standard_normal(1000)) * 0.02

    fhs1 = FilteredHistoricalSimulation(ewma_lambda=0.90)
    fhs2 = FilteredHistoricalSimulation(ewma_lambda=0.97)

    fhs1.fit(var_raw, realised, alpha=0.01)
    fhs2.fit(var_raw, realised, alpha=0.01)

    c1 = fhs1.apply(var_raw[:10])
    c2 = fhs2.apply(var_raw[:10])
    assert not np.allclose(c1, c2)
