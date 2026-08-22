"""Tests for cdf() on all distribution types."""

import numpy as np
from scipy import stats

from conformal_oracle._types import (
    SampleDistribution,
    QuantileGridDistribution,
    ParametricDistribution,
)


def test_sample_cdf_boundaries():
    samples = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    dist = SampleDistribution(samples=samples)
    assert dist.cdf(-3.0) == 0.0
    assert dist.cdf(3.0) == 1.0


def test_sample_cdf_median():
    rng = np.random.default_rng(42)
    samples = rng.standard_normal(10000)
    dist = SampleDistribution(samples=samples)
    assert abs(dist.cdf(0.0) - 0.5) < 0.02


def test_parametric_cdf_normal():
    dist = ParametricDistribution(location=0.0, scale=1.0, family="normal")
    assert abs(dist.cdf(0.0) - 0.5) < 1e-10
    assert abs(dist.cdf(1.96) - 0.975) < 0.001


def test_parametric_cdf_student_t():
    dist = ParametricDistribution(location=0.0, scale=1.0, family="student_t", df=5.0)
    assert abs(dist.cdf(0.0) - 0.5) < 1e-10


def test_quantile_grid_cdf():
    levels = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    quantiles = stats.norm.ppf(levels)
    dist = QuantileGridDistribution(levels=levels, quantiles=quantiles)
    cdf_at_zero = dist.cdf(0.0)
    assert abs(cdf_at_zero - 0.5) < 0.05
