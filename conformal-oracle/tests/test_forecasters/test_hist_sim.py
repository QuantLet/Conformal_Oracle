"""Tests for Historical Simulation forecaster."""

import numpy as np
import pandas as pd

from conformal_oracle.forecasters.hist_sim import HistoricalSimulationForecaster
from conformal_oracle._types import SampleDistribution


def test_hist_sim_returns_sample():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(500) * 0.01)
    fc = HistoricalSimulationForecaster(window=250)
    dist = fc.forecast(returns, t=300)
    assert isinstance(dist, SampleDistribution)
    assert len(dist) == 250


def test_hist_sim_quantile_matches_empirical():
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(500) * 0.01)
    fc = HistoricalSimulationForecaster(window=250)
    dist = fc.forecast(returns, t=300)
    q = dist.quantile(0.01)
    expected = np.quantile(returns.iloc[50:300].values, 0.01)
    assert abs(q - expected) < 1e-10


def test_hist_sim_short_history():
    returns = pd.Series([0.01, -0.02, 0.005])
    fc = HistoricalSimulationForecaster(window=250)
    dist = fc.forecast(returns, t=2)
    assert isinstance(dist, SampleDistribution)
    assert len(dist) == 2
