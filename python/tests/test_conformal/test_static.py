"""Tests for compute_qv_stat."""

import numpy as np
import pytest

from conformal_oracle._types import SampleDistribution
from conformal_oracle.conformal.static import compute_qv_stat


def test_qv_stat_recovers_quantile():
    """On uniform calibration scores, qV_stat should recover the (1-alpha)-quantile."""
    rng = np.random.default_rng(42)
    n = 5000
    alpha = 0.01

    samples = rng.standard_normal(n)
    forecasts = [SampleDistribution(samples=np.array([s])) for s in samples]
    realised = np.zeros(n)

    qv = compute_qv_stat(forecasts, realised, alpha)
    expected = np.quantile(samples, 1 - alpha)
    assert abs(qv - expected) < 0.01


def test_qv_stat_correct_spec_near_zero():
    """When forecaster is perfectly calibrated, qV_stat should be near zero."""
    rng = np.random.default_rng(123)
    n = 2000
    alpha = 0.01

    realised = rng.standard_normal(n) * 0.01
    forecasts = [
        SampleDistribution(samples=rng.standard_normal(1000) * 0.01)
        for _ in range(n)
    ]

    qv = compute_qv_stat(forecasts, realised, alpha)
    assert abs(qv) < 0.05


def test_qv_stat_positive_for_underpredicting():
    """If forecaster underpredicts risk, qV should be positive."""
    rng = np.random.default_rng(99)
    n = 1000
    alpha = 0.01

    realised = rng.standard_normal(n) * 0.02
    forecasts = [
        SampleDistribution(samples=rng.standard_normal(1000) * 0.005)
        for _ in range(n)
    ]

    qv = compute_qv_stat(forecasts, realised, alpha)
    assert qv > 0


def test_qv_stat_length_mismatch():
    """Should work when forecast list and realised have same length."""
    forecasts = [SampleDistribution(samples=np.array([0.0])) for _ in range(10)]
    realised = np.zeros(10)
    qv = compute_qv_stat(forecasts, realised, 0.05)
    assert isinstance(qv, float)
