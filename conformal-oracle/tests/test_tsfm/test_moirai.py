"""T_MOIRAI: MoiraiForecaster integration tests.

Skipped if uni2ts is not installed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from conformal_oracle._types import QuantileGridDistribution, SampleDistribution
from conformal_oracle.forecasters.tsfm.moirai import (
    PAPER_MODELS,
    MoiraiForecaster,
)


def _moirai_available() -> bool:
    return (
        importlib.util.find_spec("uni2ts") is not None
        and importlib.util.find_spec("gluonts") is not None
        and importlib.util.find_spec("torch") is not None
    )


pytestmark = pytest.mark.skipif(
    not _moirai_available(),
    reason="uni2ts or its dependencies not installed",
)


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(2026)
    n = 700
    r = rng.standard_normal(n) * 0.01
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.Series(r, index=dates, name="synthetic")


def test_moirai11_forecast_shape(returns):
    fc = MoiraiForecaster(version="1.1", n_samples=50, device="cpu")
    dist = fc.forecast(returns, t=600)
    assert isinstance(dist, SampleDistribution)
    assert len(dist.samples) == 50


def test_moirai11_deterministic(returns):
    fc1 = MoiraiForecaster(version="1.1", n_samples=50, seed=42, device="cpu")
    d1 = fc1.forecast(returns, t=600)

    fc2 = MoiraiForecaster(version="1.1", n_samples=50, seed=42, device="cpu")
    d2 = fc2.forecast(returns, t=600)

    np.testing.assert_array_equal(d1.samples, d2.samples)


def test_moirai20_forecast_type(returns):
    fc = MoiraiForecaster(version="2.0", device="cpu")
    dist = fc.forecast(returns, t=600)
    assert isinstance(dist, QuantileGridDistribution)


def test_moirai20_quantile_grid(returns):
    fc = MoiraiForecaster(version="2.0", device="cpu")
    dist = fc.forecast(returns, t=600)
    assert len(dist.levels) == len(dist.quantiles)
    assert len(dist.levels) > 0


def test_moirai20_tail_completion(returns):
    fc = MoiraiForecaster(version="2.0", device="cpu")
    dist = fc.forecast(returns, t=600)
    q01 = dist.quantile(0.01)
    assert np.isfinite(q01)


def test_moirai_cache_hit(returns, tmp_path):
    fc = MoiraiForecaster(
        version="1.1", n_samples=50, seed=42, device="cpu",
        cache_dir=tmp_path,
    )
    d1 = fc.forecast(returns, t=600)
    d2 = fc.forecast(returns, t=600)
    np.testing.assert_array_equal(d1.samples, d2.samples)


def test_moirai_paper_models():
    assert "1.1" in PAPER_MODELS
    assert "2.0" in PAPER_MODELS
    assert "small" in PAPER_MODELS["1.1"]
    assert "large" in PAPER_MODELS["2.0"]


def test_moirai_version_model_id():
    fc11 = MoiraiForecaster(version="1.1", size="small")
    fc20 = MoiraiForecaster(version="2.0", size="small")
    assert "moirai-1.1" in fc11.model_id
    assert "moirai-2.0" in fc20.model_id
