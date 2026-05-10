"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_returns() -> pd.Series:
    """2000-obs GARCH(1,1)-generated returns for testing."""
    rng = np.random.default_rng(2026)
    n = 2000
    omega = 1e-6
    alpha_g = 0.05
    beta_g = 0.90
    mu = 0.0

    returns = np.empty(n)
    sigma2 = np.empty(n)
    sigma2[0] = omega / (1 - alpha_g - beta_g)

    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha_g * returns[t - 1] ** 2 + beta_g * sigma2[t - 1]
        returns[t] = mu + np.sqrt(sigma2[t]) * rng.standard_normal()

    dates = pd.bdate_range("2018-01-02", periods=n)
    return pd.Series(returns, index=dates, name="synthetic_garch")


@pytest.fixture
def short_returns() -> pd.Series:
    """Short 500-obs returns for quick tests."""
    rng = np.random.default_rng(42)
    n = 500
    returns = rng.standard_normal(n) * 0.01
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series(returns, index=dates, name="short")


@pytest.fixture
def miscalibrated_returns() -> pd.Series:
    """Returns with known miscalibration for replacement-regime tests."""
    rng = np.random.default_rng(2026)
    n = 2000
    omega = 1e-6
    alpha_g = 0.05
    beta_g = 0.90

    returns = np.empty(n)
    sigma2 = np.empty(n)
    sigma2[0] = omega / (1 - alpha_g - beta_g)

    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha_g * returns[t - 1] ** 2 + beta_g * sigma2[t - 1]
        returns[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

    dates = pd.bdate_range("2018-01-02", periods=n)
    return pd.Series(returns, index=dates, name="miscalibrated_garch")
