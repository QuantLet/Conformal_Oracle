"""Tests for bootstrap_qv_ci."""

import numpy as np

from conformal_oracle.conformal.bootstrap import bootstrap_qv_ci


def test_bootstrap_ci_contains_point_estimate():
    """CI should contain the point estimate of qV."""
    rng = np.random.default_rng(42)
    scores = rng.standard_normal(500)
    alpha = 0.05
    qv_point = float(np.quantile(scores, 1 - alpha))
    lo, hi = bootstrap_qv_ci(scores, alpha, n_boot=199, seed=42)
    assert lo <= qv_point <= hi


def test_bootstrap_ci_ordering():
    """Lower bound should be less than upper bound."""
    scores = np.random.default_rng(42).standard_normal(300)
    lo, hi = bootstrap_qv_ci(scores, 0.01, n_boot=99)
    assert lo < hi


def test_bootstrap_ci_reproducible():
    """Same seed should give same result."""
    scores = np.random.default_rng(42).standard_normal(300)
    ci1 = bootstrap_qv_ci(scores, 0.01, n_boot=99, seed=123)
    ci2 = bootstrap_qv_ci(scores, 0.01, n_boot=99, seed=123)
    assert ci1 == ci2
