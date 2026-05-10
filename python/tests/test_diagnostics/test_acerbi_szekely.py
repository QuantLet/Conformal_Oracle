"""Tests for Acerbi-Szekely Z2 statistic."""

import numpy as np

from conformal_oracle.diagnostics.acerbi_szekely import z2_statistic


def test_z2_no_violations():
    """No violations should return 0."""
    violations = np.zeros(100)
    realised = np.random.default_rng(42).standard_normal(100) * 0.01
    es = np.full(100, 0.03)
    z2 = z2_statistic(violations, realised, es, 0.01)
    assert z2 == 0.0


def test_z2_stabilised_vs_unstabilised():
    """Both modes should return a finite number."""
    rng = np.random.default_rng(42)
    violations = np.zeros(250)
    violations[:5] = 1
    realised = rng.standard_normal(250) * 0.02
    realised[:5] = -0.05
    es = np.full(250, 0.03)

    z2_stab = z2_statistic(violations, realised, es, 0.01, stabilised=True)
    z2_unstab = z2_statistic(violations, realised, es, 0.01, stabilised=False)

    assert np.isfinite(z2_stab)
    assert np.isfinite(z2_unstab)
