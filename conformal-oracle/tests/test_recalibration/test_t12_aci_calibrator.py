"""T12: ACICalibrator tests (gamma-grid selection over Gibbs-Candes ACI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.recalibration import ACICalibrator, RecalibrationMethod


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


def test_default_grid_matches_spec():
    assert ACICalibrator.DEFAULT_GAMMA_GRID == (0.001, 0.005, 0.01, 0.05)
    assert ACICalibrator().gamma_grid == (0.001, 0.005, 0.01, 0.05)


def test_protocol_compliance():
    assert isinstance(ACICalibrator(), RecalibrationMethod)


def test_fixed_gamma_skips_selection():
    rng = np.random.default_rng(1)
    realised = rng.standard_normal(800) * 0.01
    var_raw = np.abs(rng.standard_normal(800)) * 0.02
    cal = ACICalibrator(gamma=0.01)
    cal.fit(var_raw, realised, alpha=0.01)
    assert cal.selected_gamma == 0.01


def test_selection_picks_grid_member():
    rng = np.random.default_rng(2)
    realised = rng.standard_normal(1500) * 0.01
    var_raw = np.abs(rng.standard_normal(1500)) * 0.02
    cal = ACICalibrator()  # selection mode
    cal.fit(var_raw, realised, alpha=0.05)
    assert cal.selected_gamma in ACICalibrator.DEFAULT_GAMMA_GRID


def test_apply_and_online_shapes():
    rng = np.random.default_rng(3)
    realised = rng.standard_normal(600) * 0.01
    var_raw = np.abs(rng.standard_normal(600)) * 0.02
    cal = ACICalibrator(gamma=0.05)
    cal.fit(var_raw, realised, alpha=0.01)
    assert cal.apply(var_raw[:50]).shape == (50,)
    assert cal.apply_online(var_raw[:50], realised[:50]).shape == (50,)


def test_apply_before_fit_raises():
    cal = ACICalibrator(gamma=0.05)
    with pytest.raises(RuntimeError):
        cal.apply(np.zeros(10))
    with pytest.raises(RuntimeError):
        cal.apply_online(np.zeros(10), np.zeros(10))


def test_bad_validation_fraction():
    with pytest.raises(ValueError):
        ACICalibrator(validation_fraction=0.0)
    with pytest.raises(ValueError):
        ACICalibrator(validation_fraction=1.0)


def test_shape_mismatch_raises():
    cal = ACICalibrator(gamma=0.01)
    with pytest.raises(ValueError):
        cal.fit(np.zeros(10), np.zeros(9), alpha=0.01)


def test_coverage_near_alpha(garch_returns):
    """Selected-gamma ACI drives corrected coverage close to alpha."""
    cal = ACICalibrator()
    result = audit_static(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        alpha=0.01,
        recalibration=cal,
    )
    assert abs(result.violation_rate_corrected - 0.01) < 0.02
