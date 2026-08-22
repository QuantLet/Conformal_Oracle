"""Tests for Kupiec POF test."""

import numpy as np
from scipy import stats

from conformal_oracle.diagnostics.kupiec import kupiec_pof_pvalue


def test_kupiec_zero_violations():
    """x=0 edge case should use the (1-alpha)^T convention."""
    violations = np.zeros(250)
    alpha = 0.01
    p = kupiec_pof_pvalue(violations, alpha)
    lr = -2.0 * 250 * np.log(1 - alpha)
    expected = float(stats.chi2.sf(lr, df=1))
    assert abs(p - expected) < 1e-10


def test_kupiec_exact_rate():
    """When violation rate matches alpha exactly, p-value should be ~1."""
    n = 1000
    alpha = 0.05
    x = int(n * alpha)
    violations = np.zeros(n)
    violations[:x] = 1
    p = kupiec_pof_pvalue(violations, alpha)
    assert p > 0.9


def test_kupiec_too_many():
    """Many violations should give low p-value."""
    violations = np.zeros(250)
    violations[:50] = 1
    p = kupiec_pof_pvalue(violations, 0.01)
    assert p < 0.01


def test_kupiec_all_violations():
    """All violations should give very low p-value."""
    violations = np.ones(100)
    p = kupiec_pof_pvalue(violations, 0.01)
    assert p < 0.001
