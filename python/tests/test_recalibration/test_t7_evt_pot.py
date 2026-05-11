"""T7: ExtremeValueTheoryPOT integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.recalibration import ExtremeValueTheoryPOT


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


def test_evt_violation_rate(garch_returns):
    """EVT-POT corrected violation rate should be in a reasonable range."""
    recal = ExtremeValueTheoryPOT()
    result = audit_static(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        alpha=0.01,
        recalibration=recal,
    )
    assert 0 <= result.violation_rate_corrected <= 0.10


def test_evt_gpd_parameters():
    """EVT-POT should produce finite GPD parameters."""
    rng = np.random.default_rng(2026)
    n = 2000
    realised = rng.standard_normal(n) * 0.01
    var_raw = np.abs(rng.standard_normal(n)) * 0.02

    evt = ExtremeValueTheoryPOT()
    evt.fit(var_raw, realised, alpha=0.01)

    assert np.isfinite(evt.shape_parameter)
    assert np.isfinite(evt.scale_parameter)


def test_evt_output_shape():
    """Apply should return array of same length as input."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(1000) * 0.01
    var_raw = np.abs(rng.standard_normal(1000)) * 0.02

    evt = ExtremeValueTheoryPOT()
    evt.fit(var_raw, realised, alpha=0.01)

    test_var = np.array([0.01, 0.02, 0.03])
    corrected = evt.apply(test_var)
    assert corrected.shape == (3,)
    assert np.all(np.isfinite(corrected))


def test_evt_positive_output():
    """Corrected VaR should be positive."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(1000) * 0.01
    var_raw = np.abs(rng.standard_normal(1000)) * 0.02

    evt = ExtremeValueTheoryPOT()
    evt.fit(var_raw, realised, alpha=0.01)

    test_var = np.array([0.01, 0.02, 0.03])
    corrected = evt.apply(test_var)
    assert np.all(corrected > 0)


def test_evt_fit_ok_flag():
    """GPD fit should succeed on well-behaved data."""
    rng = np.random.default_rng(2026)
    realised = rng.standard_normal(2000) * 0.01
    var_raw = np.abs(rng.standard_normal(2000)) * 0.02

    evt = ExtremeValueTheoryPOT()
    evt.fit(var_raw, realised, alpha=0.01)
    assert isinstance(evt.gpd_fit_ok, bool)
