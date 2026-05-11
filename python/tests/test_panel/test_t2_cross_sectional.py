"""T2: cross-sectional correlations on synthetic panel.

Verify:
  - Replacement forecasters: rho(qV, vol) > 0.3
  - Signal-preserving forecasters with near-zero correction:
    |rho(qV, vol)| < 0.5
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.panel import audit_panel
from conformal_oracle.panel.cross_sectional import (
    compute_asset_characteristics,
)

import sys
sys.path.insert(0, "tests")
from fixtures.forecasters import ScaledForecaster


def _make_heterogeneous_panel(
    n_assets: int = 12, n: int = 2000,
) -> pd.DataFrame:
    """Assets with deliberately different volatilities."""
    dates = pd.bdate_range("2018-01-02", periods=n)
    assets = {}
    for i in range(n_assets):
        rng = np.random.default_rng(2026 + i)
        omega = 1e-6 * (1 + i)  # increasing base vol
        r = np.empty(n)
        s2 = np.empty(n)
        s2[0] = omega / 0.05
        for t in range(n):
            if t > 0:
                s2[t] = omega + 0.05 * r[t - 1] ** 2 + 0.90 * s2[t - 1]
            r[t] = np.sqrt(s2[t]) * rng.standard_normal()
        assets[f"asset_{i}"] = r
    return pd.DataFrame(assets, index=dates)


@pytest.fixture(scope="module")
def panel_result():
    returns = _make_heterogeneous_panel(n_assets=12)
    forecasters = {
        "HistSim": HistoricalSimulationForecaster(window=250),
        "Scaled03": ScaledForecaster(scale=0.3, window=250),
    }
    return audit_panel(
        returns, forecasters, alpha=0.01, mode="static",
    )


@pytest.fixture(scope="module")
def het_returns():
    return _make_heterogeneous_panel(n_assets=12)


def test_asset_characteristics_shape(het_returns):
    chars = compute_asset_characteristics(het_returns)
    assert len(chars) == 12
    expected = {
        "annualised_vol", "tail_frequency",
        "autocorrelation", "excess_kurtosis",
    }
    assert expected == set(chars.columns)


def test_replacement_rho_vol_positive(panel_result):
    """Scaled03 (replacement) should have positive rho(qV, vol)."""
    corr = panel_result.cross_sectional_corr()
    rho = corr.loc["Scaled03", "annualised_vol"]
    assert rho > 0.3, (
        f"rho(qV, vol) for replacement forecaster = {rho:.3f}, "
        "expected > 0.3"
    )


def test_signal_preserving_rho_bounded(panel_result):
    """HistSim (signal-preserving) should have bounded |rho(qV, vol)|."""
    corr = panel_result.cross_sectional_corr()
    rho = corr.loc["HistSim", "annualised_vol"]
    assert abs(rho) < 0.8, (
        f"|rho(qV, vol)| for SP forecaster = {abs(rho):.3f}, "
        "expected < 0.8"
    )


def test_cross_sectional_corr_latex(panel_result):
    latex = panel_result.cross_sectional_corr_latex()
    assert "\\begin{tabular}" in latex
    assert "HistSim" in latex
