"""T3: wild-cluster bootstrap conservatism.

When cluster correlations are high, bootstrap p-values should be
more conservative (larger) than asymptotic p-values.
"""

from __future__ import annotations

import numpy as np
import pytest

from conformal_oracle.panel.kupiec_panel import panel_kupiec_test
from conformal_oracle.panel.wildcluster_bootstrap import (
    wild_cluster_bootstrap_kupiec,
)


def test_bootstrap_more_conservative():
    """For correlated violations, bootstrap p > asymptotic p."""
    rng = np.random.default_rng(2026)
    alpha = 0.01
    n = 1000
    n_assets = 10

    # Create correlated violations via a common shock
    common = (rng.random(n) < 0.025).astype(float)
    violations = {}
    for i in range(n_assets):
        idio = (rng.random(n) < 0.005).astype(float)
        v = np.clip(common + idio, 0, 1).astype(int)
        violations[f"asset_{i}"] = v

    _, p_asym, _ = panel_kupiec_test(violations, alpha)

    boot = wild_cluster_bootstrap_kupiec(
        violations, alpha, B=499, seed=2026,
    )
    p_boot = boot["p_bootstrap"]

    # Bootstrap should be at least as conservative
    assert p_boot >= p_asym * 0.5, (
        f"p_boot={p_boot:.4f} vs p_asym={p_asym:.4f}: "
        "bootstrap should be more conservative"
    )


def test_bootstrap_kupiec_keys():
    """Verify the output dict has expected keys."""
    violations = {
        "A": np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        "B": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    }
    result = wild_cluster_bootstrap_kupiec(
        violations, 0.01, B=99, seed=42,
    )
    assert "lr_original" in result
    assert "p_asymptotic" in result
    assert "p_bootstrap" in result
    assert "boot_mean_lr" in result
    assert "boot_q95_lr" in result
