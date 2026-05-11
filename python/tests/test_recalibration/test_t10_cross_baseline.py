"""T10: Cross-baseline integration test.

Verifies that audit_with_benchmarks works with recalibration
methods and that the comparison table renders correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit import audit_with_benchmarks
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.recalibration import (
    AdaptiveConformalInference,
    ConformalShift,
    ExtremeValueTheoryPOT,
    FilteredHistoricalSimulation,
    HistoricalQuantileRecalibration,
    LinearQuantileRegression,
    ScaleCorrectionRecalibration,
)


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


def test_audit_with_recalibrations(garch_returns):
    """audit_with_benchmarks should work with recalibration methods."""
    recals = [
        ConformalShift(),
        ScaleCorrectionRecalibration(),
        HistoricalQuantileRecalibration(),
    ]
    comp = audit_with_benchmarks(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        benchmarks=["hist_sim"],
        recalibrations=recals,
        alpha=0.01,
        mode="static",
    )
    table = comp.comparison_table()
    assert len(table) > 0
    assert "violation_rate_corrected" in table.columns


def test_cross_baseline_all_methods(garch_returns):
    """All non-GBM recalibrations should integrate correctly."""
    recals = [
        ConformalShift(),
        ScaleCorrectionRecalibration(),
        HistoricalQuantileRecalibration(),
        LinearQuantileRegression(),
        AdaptiveConformalInference(gamma=0.05),
        ExtremeValueTheoryPOT(),
        FilteredHistoricalSimulation(),
    ]
    comp = audit_with_benchmarks(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        benchmarks=["hist_sim"],
        recalibrations=recals,
        alpha=0.01,
        mode="static",
    )
    table = comp.comparison_table()
    assert len(table) >= 14


def test_comparison_table_has_all_methods(garch_returns):
    """Comparison table should have rows for each combination."""
    recals = [
        ConformalShift(),
        ScaleCorrectionRecalibration(),
    ]
    comp = audit_with_benchmarks(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        benchmarks=["hist_sim"],
        recalibrations=recals,
        alpha=0.01,
        mode="static",
    )
    table = comp.comparison_table()
    assert any("ConformalShift" in idx for idx in table.index)
    assert any("ScaleCorrectionRecalibration" in idx for idx in table.index)


def test_no_recalibrations_unchanged(garch_returns):
    """recalibrations=None should behave like v0.1."""
    comp = audit_with_benchmarks(
        garch_returns,
        HistoricalSimulationForecaster(window=250),
        benchmarks=["hist_sim"],
        alpha=0.01,
        mode="static",
    )
    table = comp.comparison_table()
    assert "user" in table.index
    assert "hist_sim" in table.index
