"""Tests for BenchmarkComparison.diebold_mariano() with HAC variance."""

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit.benchmark import audit_with_benchmarks
from conformal_oracle.forecasters.hist_sim import HistoricalSimulationForecaster

from tests.fixtures.forecasters import ConstantZeroForecaster


def test_dm_on_benchmarks(synthetic_returns):
    """DM test should not blow up on a normal comparison."""
    fc = HistoricalSimulationForecaster(window=250)
    comp = audit_with_benchmarks(
        synthetic_returns, fc,
        benchmarks=["hist_sim"],
        mode="static",
    )
    dm = comp.diebold_mariano(baseline="hist_sim")
    assert "user" in dm
    assert 0.0 <= dm["user"] <= 1.0


def test_dm_replacement_regime_stable(synthetic_returns):
    """DM stat should remain finite even for a replacement-regime forecaster."""
    fc = ConstantZeroForecaster()
    comp = audit_with_benchmarks(
        synthetic_returns, fc,
        benchmarks=["hist_sim"],
        mode="static",
    )
    dm = comp.diebold_mariano(baseline="hist_sim")
    assert np.isfinite(dm["user"])


def test_dm_baseline_not_found(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    comp = audit_with_benchmarks(
        synthetic_returns, fc,
        benchmarks=["hist_sim"],
        mode="static",
    )
    with pytest.raises(ValueError, match="not in benchmarks"):
        comp.diebold_mariano(baseline="gjr_garch")
