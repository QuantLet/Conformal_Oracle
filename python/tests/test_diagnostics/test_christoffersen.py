"""Tests for Christoffersen conditional coverage test."""

import numpy as np

from conformal_oracle.diagnostics.christoffersen import christoffersen_pvalue


def test_christoffersen_returns_three_keys():
    violations = np.zeros(250)
    violations[::25] = 1
    result = christoffersen_pvalue(violations, 0.01)
    assert set(result.keys()) == {"unconditional", "independence", "joint"}


def test_christoffersen_iid_high_independence():
    """IID violations should have high independence p-value."""
    rng = np.random.default_rng(42)
    violations = (rng.random(1000) < 0.01).astype(int)
    result = christoffersen_pvalue(violations, 0.01)
    assert result["independence"] > 0.01


def test_christoffersen_clustered_low_independence():
    """Clustered violations should have low independence p-value."""
    violations = np.zeros(500)
    for i in range(0, 500, 50):
        violations[i : i + 5] = 1
    result = christoffersen_pvalue(violations, 0.01)
    assert result["independence"] < 0.05
