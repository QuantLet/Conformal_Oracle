"""Tests for BaseTSFMForecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle._types import SampleDistribution
from conformal_oracle.forecasters.tsfm._base import BaseTSFMForecaster


class DummyTSFM(BaseTSFMForecaster):
    def forecast(self, returns, t):
        ctx = self._get_context(returns, t)
        return SampleDistribution(samples=ctx)


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    n = 600
    r = rng.standard_normal(n) * 0.01
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.Series(r, index=dates, name="test")


def test_context_length(returns):
    fc = DummyTSFM(model_id="test", context_length=100)
    dist = fc.forecast(returns, t=500)
    assert len(dist.samples) == 100


def test_context_capped_at_start(returns):
    fc = DummyTSFM(model_id="test", context_length=512)
    dist = fc.forecast(returns, t=50)
    assert len(dist.samples) == 50


def test_call_seed_deterministic():
    fc = DummyTSFM(model_id="test", seed=2026)
    s1 = fc._call_seed(100)
    s2 = fc._call_seed(100)
    assert s1 == s2


def test_call_seed_varies_with_t():
    fc = DummyTSFM(model_id="test", seed=2026)
    s1 = fc._call_seed(100)
    s2 = fc._call_seed(101)
    assert s1 != s2


def test_fit_is_noop(returns):
    fc = DummyTSFM(model_id="test")
    fc.fit(returns)


def test_device_resolution_cpu():
    fc = DummyTSFM(model_id="test", device="cpu")
    assert fc._resolve_device() == "cpu"


def test_device_resolution_auto():
    fc = DummyTSFM(model_id="test", device="auto")
    device = fc._resolve_device()
    assert device in ("cpu", "cuda", "mps")
