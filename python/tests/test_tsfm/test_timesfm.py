"""T_TIMESFM: TimesFM25Forecaster integration tests.

Skipped if timesfm is not installed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from conformal_oracle._types import QuantileGridDistribution
from conformal_oracle.forecasters.tsfm.timesfm import (
    PAPER_MODEL,
    QUANTILE_LEVELS,
    TimesFM25Forecaster,
)


def _timesfm_25_available() -> bool:
    if importlib.util.find_spec("timesfm") is None:
        return False
    try:
        from timesfm import ForecastConfig  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _timesfm_25_available(),
    reason="timesfm 2.5 API not available (ForecastConfig missing)",
)


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(2026)
    n = 700
    r = rng.standard_normal(n) * 0.01
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.Series(r, index=dates, name="synthetic")


def test_timesfm_forecast_type(returns):
    fc = TimesFM25Forecaster(device="cpu")
    dist = fc.forecast(returns, t=600)
    assert isinstance(dist, QuantileGridDistribution)


def test_timesfm_quantile_grid_shape(returns):
    fc = TimesFM25Forecaster(device="cpu")
    dist = fc.forecast(returns, t=600)
    assert len(dist.levels) == 9
    assert len(dist.quantiles) == 9
    np.testing.assert_array_equal(dist.levels, QUANTILE_LEVELS)


def test_timesfm_tail_completion(returns):
    """Student-t tail completion should produce finite 1% quantile."""
    fc = TimesFM25Forecaster(device="cpu")
    dist = fc.forecast(returns, t=600)
    q01 = dist.quantile(0.01)
    assert np.isfinite(q01)
    q10 = dist.quantile(0.10)
    assert q01 < q10


def test_timesfm_cache_hit(returns, tmp_path):
    fc = TimesFM25Forecaster(device="cpu", cache_dir=tmp_path)
    d1 = fc.forecast(returns, t=600)
    d2 = fc.forecast(returns, t=600)
    np.testing.assert_array_equal(d1.quantiles, d2.quantiles)


def test_timesfm_deterministic(returns):
    """TimesFM is deterministic (no sampling) so seed shouldn't matter."""
    fc1 = TimesFM25Forecaster(seed=42, device="cpu")
    d1 = fc1.forecast(returns, t=600)

    fc2 = TimesFM25Forecaster(seed=99, device="cpu")
    d2 = fc2.forecast(returns, t=600)

    np.testing.assert_array_equal(d1.quantiles, d2.quantiles)


def test_timesfm_paper_model():
    assert PAPER_MODEL == "google/timesfm-2.5-200m-pytorch"


def test_timesfm_quantile_levels():
    expected = np.arange(0.1, 1.0, 0.1)
    np.testing.assert_allclose(QUANTILE_LEVELS, expected)
