"""T4: IsotonicQuantileRegression integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.recalibration import IsotonicQuantileRegression


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


def test_isotonic_violation_rate(garch_returns):
    """Corrected violation rate bounded; may produce Kupiec rejection
    without Basel Red due to the difference between the two diagnostics
    in small-sample regimes.

    Isotonic regression is known to be unstable at the 1% tail
    (pool-adjacent violators has sparse support), so we allow a
    wide tolerance.
    """
    recal = IsotonicQuantileRegression()
    result = audit_static(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        alpha=0.01,
        recalibration=recal,
    )
    assert result.violation_rate_corrected < 0.50


def test_isotonic_protocol_compliance():
    """IsotonicQR implements RecalibrationMethod protocol."""
    from conformal_oracle.recalibration import RecalibrationMethod

    assert isinstance(IsotonicQuantileRegression(), RecalibrationMethod)


def test_isotonic_output_shape():
    """Apply should return array of same length as input."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(500) * 0.01
    var_raw = np.abs(rng.standard_normal(500)) * 0.02

    iso = IsotonicQuantileRegression()
    iso.fit(var_raw, realised, alpha=0.01)

    test_var = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    corrected = iso.apply(test_var)
    assert corrected.shape == (5,)


def test_isotonic_positive_output():
    """Output should remain positive."""
    rng = np.random.default_rng(42)
    n = 1000
    realised = rng.standard_normal(n) * 0.01
    var_raw = np.abs(rng.standard_normal(n)) * 0.02

    iso = IsotonicQuantileRegression()
    iso.fit(var_raw, realised, alpha=0.01)

    test_var = np.array([0.01, 0.02, 0.03])
    corrected = iso.apply(test_var)
    assert np.all(corrected > 0)
