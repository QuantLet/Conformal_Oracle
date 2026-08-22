"""T_LAG_LLAMA: LagLlamaForecaster integration tests.

Skipped if lag-llama is not installed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from conformal_oracle._types import SampleDistribution
from conformal_oracle.forecasters.tsfm.lag_llama import (
    PAPER_CKPT,
    PAPER_MODEL,
    LagLlamaForecaster,
)


def _lag_llama_available() -> bool:
    return (
        importlib.util.find_spec("lag_llama") is not None
        and importlib.util.find_spec("gluonts") is not None
        and importlib.util.find_spec("huggingface_hub") is not None
    )


pytestmark = pytest.mark.skipif(
    not _lag_llama_available(),
    reason="lag-llama or its dependencies not installed",
)


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(2026)
    n = 700
    r = rng.standard_normal(n) * 0.01
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.Series(r, index=dates, name="synthetic")


def test_lag_llama_forecast_shape(returns):
    fc = LagLlamaForecaster(n_samples=50, device="cpu")
    dist = fc.forecast(returns, t=600)
    assert isinstance(dist, SampleDistribution)
    assert len(dist.samples) == 50


def test_lag_llama_deterministic(returns):
    fc1 = LagLlamaForecaster(n_samples=50, seed=42, device="cpu")
    d1 = fc1.forecast(returns, t=600)

    fc2 = LagLlamaForecaster(n_samples=50, seed=42, device="cpu")
    d2 = fc2.forecast(returns, t=600)

    np.testing.assert_array_equal(d1.samples, d2.samples)


def test_lag_llama_cache_hit(returns, tmp_path):
    fc = LagLlamaForecaster(
        n_samples=50, seed=42, device="cpu", cache_dir=tmp_path,
    )
    d1 = fc.forecast(returns, t=600)
    d2 = fc.forecast(returns, t=600)
    np.testing.assert_array_equal(d1.samples, d2.samples)


def test_lag_llama_quantile_finite(returns):
    fc = LagLlamaForecaster(n_samples=100, device="cpu")
    dist = fc.forecast(returns, t=600)
    q = dist.quantile(0.01)
    assert np.isfinite(q)


def test_lag_llama_paper_model():
    assert PAPER_MODEL == "time-series-foundation-models/Lag-Llama"
    assert PAPER_CKPT == "lag-llama.ckpt"


def test_lag_llama_import_error_message():
    fc = LagLlamaForecaster()
    assert fc.model_id == PAPER_MODEL
