"""Tests for GJR-GARCH forecaster."""

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.forecasters.gjr_garch import GJRGARCHForecaster
from conformal_oracle._types import ParametricDistribution


def test_gjr_garch_returns_parametric(synthetic_returns):
    fc = GJRGARCHForecaster(window=250)
    fc.fit(synthetic_returns)
    dist = fc.forecast(synthetic_returns, t=300)
    assert isinstance(dist, ParametricDistribution)


def test_gjr_garch_quantile_negative(synthetic_returns):
    """1% quantile should be negative (loss in left tail)."""
    fc = GJRGARCHForecaster(window=250)
    dist = fc.forecast(synthetic_returns, t=300)
    q = dist.quantile(0.01)
    assert q < 0


def test_gjr_garch_small_t_fallback():
    """With very few observations, should still return a distribution."""
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(30) * 0.01)
    fc = GJRGARCHForecaster(window=250)
    dist = fc.forecast(returns, t=10)
    assert isinstance(dist, ParametricDistribution)
    assert dist.family == "normal"
