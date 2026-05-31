"""Tests for TSFMPredictionCache."""

from __future__ import annotations

import numpy as np
import pytest

from conformal_oracle._types import SampleDistribution
from conformal_oracle.forecasters.tsfm._cache import TSFMPredictionCache


@pytest.fixture
def cache(tmp_path):
    return TSFMPredictionCache(
        cache_dir=tmp_path,
        model_id="test/model",
        model_revision="abc123",
    )


def test_cache_miss_returns_none(cache):
    ctx = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    assert cache.get(ctx, t=100, n_samples=1000, seed=42) is None


def test_cache_roundtrip(cache):
    ctx = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    dist = SampleDistribution(samples=np.array([0.1, 0.2, 0.3]))
    cache.put(ctx, t=100, n_samples=1000, seed=42, dist=dist)

    recovered = cache.get(ctx, t=100, n_samples=1000, seed=42)
    assert recovered is not None
    np.testing.assert_array_equal(recovered.samples, dist.samples)


def test_cache_different_t_is_miss(cache):
    ctx = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    dist = SampleDistribution(samples=np.array([0.1, 0.2]))
    cache.put(ctx, t=100, n_samples=1000, seed=42, dist=dist)

    assert cache.get(ctx, t=101, n_samples=1000, seed=42) is None


def test_cache_different_seed_is_miss(cache):
    ctx = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    dist = SampleDistribution(samples=np.array([0.1, 0.2]))
    cache.put(ctx, t=100, n_samples=1000, seed=42, dist=dist)

    assert cache.get(ctx, t=100, n_samples=1000, seed=99) is None


def test_cache_clear(cache):
    ctx = np.array([0.01, 0.02], dtype=np.float32)
    dist = SampleDistribution(samples=np.array([0.5]))
    cache.put(ctx, t=1, n_samples=10, seed=1, dist=dist)

    assert cache.get(ctx, t=1, n_samples=10, seed=1) is not None
    cache.clear()
    assert cache.get(ctx, t=1, n_samples=10, seed=1) is None


def test_cache_eviction(tmp_path):
    cache = TSFMPredictionCache(
        cache_dir=tmp_path,
        model_id="test/evict",
        max_bytes=500,
    )
    big_samples = np.random.default_rng(42).standard_normal(10000)
    dist = SampleDistribution(samples=big_samples)

    for t in range(20):
        ctx = np.array([float(t)], dtype=np.float32)
        cache.put(ctx, t=t, n_samples=1000, seed=42, dist=dist)

    cache_dir = tmp_path / "test__evict"
    files = list(cache_dir.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in files)
    assert total_size <= 500 + 200000


def test_cache_different_context_is_miss(cache):
    ctx1 = np.array([0.01, 0.02], dtype=np.float32)
    ctx2 = np.array([0.01, 0.03], dtype=np.float32)
    dist = SampleDistribution(samples=np.array([0.5]))
    cache.put(ctx1, t=1, n_samples=10, seed=1, dist=dist)

    assert cache.get(ctx2, t=1, n_samples=10, seed=1) is None


def test_cache_preserves_numpy_dtype(cache):
    ctx = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    samples = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    dist = SampleDistribution(samples=samples)
    cache.put(ctx, t=50, n_samples=3, seed=7, dist=dist)

    recovered = cache.get(ctx, t=50, n_samples=3, seed=7)
    assert recovered.samples.dtype == np.float64
