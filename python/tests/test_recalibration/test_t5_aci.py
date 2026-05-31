"""T5: AdaptiveConformalInference integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.recalibration import AdaptiveConformalInference


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


def test_aci_violation_rate_near_alpha(garch_returns):
    """Long-run average violation rate should converge to alpha."""
    recal = AdaptiveConformalInference(gamma=0.05)
    result = audit_static(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        alpha=0.01,
        recalibration=recal,
    )
    assert abs(result.violation_rate_corrected - 0.01) < 0.02


def test_aci_protocol_compliance():
    """ACI implements RecalibrationMethod protocol."""
    from conformal_oracle.recalibration import RecalibrationMethod

    assert isinstance(AdaptiveConformalInference(), RecalibrationMethod)


def test_aci_gamma_effect():
    """Higher gamma should produce faster adaptation."""
    rng = np.random.default_rng(2026)
    realised = rng.standard_normal(1000) * 0.01
    var_raw = np.abs(rng.standard_normal(1000)) * 0.005

    aci_slow = AdaptiveConformalInference(gamma=0.01)
    aci_fast = AdaptiveConformalInference(gamma=0.10)

    aci_slow.fit(var_raw, realised, alpha=0.01)
    aci_fast.fit(var_raw, realised, alpha=0.01)

    corr_slow = aci_slow.apply(var_raw[:100])
    corr_fast = aci_fast.apply(var_raw[:100])

    assert not np.allclose(corr_slow, corr_fast)


def test_aci_online_shape():
    """apply_online should return same-length array."""
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(500) * 0.01
    var_raw = np.abs(rng.standard_normal(500)) * 0.02

    aci = AdaptiveConformalInference(gamma=0.05)
    aci.fit(var_raw, realised, alpha=0.01)

    corrected = aci.apply_online(var_raw[:100], realised[:100])
    assert corrected.shape == (100,)


def test_aci_default_gamma():
    """Default gamma should be 0.05."""
    aci = AdaptiveConformalInference()
    assert aci.gamma == 0.05
