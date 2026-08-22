"""Tests for Diebold-Mariano test with HAC variance."""

import numpy as np

from conformal_oracle.diagnostics.diebold_mariano import (
    diebold_mariano_pvalue,
    quantile_score_sequence,
)


def test_dm_identical_losses():
    """Identical loss sequences should give p-value = 1."""
    losses = np.random.default_rng(42).standard_normal(500)
    p = diebold_mariano_pvalue(losses, losses)
    assert p == 1.0


def test_dm_clearly_different():
    """Very different losses should give low p-value."""
    rng = np.random.default_rng(42)
    losses_a = rng.standard_normal(500) * 0.01
    losses_b = rng.standard_normal(500) * 0.01 + 1.0
    p = diebold_mariano_pvalue(losses_a, losses_b)
    assert p < 0.01


def test_dm_hln_correction_reduces_significance():
    """HLN correction should produce a larger (less significant) p-value."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal(50) * 0.01
    b = rng.standard_normal(50) * 0.01 + 0.1
    p_no_hln = diebold_mariano_pvalue(a, b, hln_correction=False)
    p_hln = diebold_mariano_pvalue(a, b, hln_correction=True)
    assert p_hln >= p_no_hln


def test_dm_symmetric():
    """DM test should be symmetric in the two-sided case."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal(200)
    b = rng.standard_normal(200) + 0.5
    p1 = diebold_mariano_pvalue(a, b)
    p2 = diebold_mariano_pvalue(b, a)
    assert abs(p1 - p2) < 1e-10


def test_dm_short_series():
    """Very short series should not crash."""
    p = diebold_mariano_pvalue(np.array([1.0]), np.array([2.0]))
    assert p == 1.0


def test_qs_sequence_shape():
    rng = np.random.default_rng(42)
    realised = rng.standard_normal(100)
    forecasts = rng.standard_normal(100)
    seq = quantile_score_sequence(realised, forecasts, 0.01)
    assert len(seq) == 100
    assert np.all(seq >= 0)
