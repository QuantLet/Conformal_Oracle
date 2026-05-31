"""T_CHRONOS: ChronosForecaster integration tests.

Skipped if chronos is not installed or crashes on import.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from conformal_oracle._types import SampleDistribution
from conformal_oracle.forecasters.tsfm.chronos import PAPER_MODELS, ChronosForecaster


def _chronos_importable() -> bool:
    if importlib.util.find_spec("chronos") is None:
        return False
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "from chronos import ChronosPipeline"],
        capture_output=True,
        timeout=30,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _chronos_importable(),
    reason="chronos not installed or import fails",
)


@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(2026)
    n = 700
    r = rng.standard_normal(n) * 0.01
    dates = pd.bdate_range("2021-01-04", periods=n)
    return pd.Series(r, index=dates, name="synthetic")


def test_chronos_forecast_shape(returns):
    fc = ChronosForecaster(size="mini", n_samples=50, device="cpu")
    dist = fc.forecast(returns, t=600)
    assert isinstance(dist, SampleDistribution)
    assert len(dist.samples) == 50


def test_chronos_deterministic(returns):
    fc1 = ChronosForecaster(size="mini", n_samples=50, seed=42, device="cpu")
    d1 = fc1.forecast(returns, t=600)

    fc2 = ChronosForecaster(size="mini", n_samples=50, seed=42, device="cpu")
    d2 = fc2.forecast(returns, t=600)

    np.testing.assert_array_equal(d1.samples, d2.samples)


def test_chronos_different_seeds(returns):
    fc1 = ChronosForecaster(size="mini", n_samples=50, seed=42, device="cpu")
    d1 = fc1.forecast(returns, t=600)

    fc2 = ChronosForecaster(size="mini", n_samples=50, seed=99, device="cpu")
    d2 = fc2.forecast(returns, t=600)

    assert not np.array_equal(d1.samples, d2.samples)


def test_chronos_cache_hit(returns, tmp_path):
    fc = ChronosForecaster(
        size="mini", n_samples=50, seed=42, device="cpu", cache_dir=tmp_path,
    )
    d1 = fc.forecast(returns, t=600)
    d2 = fc.forecast(returns, t=600)
    np.testing.assert_array_equal(d1.samples, d2.samples)


def test_chronos_quantile_finite(returns):
    fc = ChronosForecaster(size="mini", n_samples=100, device="cpu")
    dist = fc.forecast(returns, t=600)
    q = dist.quantile(0.01)
    assert np.isfinite(q)
    assert q < 0


def test_chronos_paper_models():
    assert "small" in PAPER_MODELS
    assert "mini" in PAPER_MODELS
    assert PAPER_MODELS["small"] == "amazon/chronos-t5-small"


def test_chronos_import_error_message():
    fc = ChronosForecaster(size="mini")
    assert fc.model_id == "amazon/chronos-t5-mini"
