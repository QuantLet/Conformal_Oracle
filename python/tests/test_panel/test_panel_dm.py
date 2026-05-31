"""Tests for panel Diebold-Mariano test."""

from __future__ import annotations

import numpy as np

from conformal_oracle.panel.diebold_mariano import panel_dm_test


def test_identical_forecasters_no_reject():
    """T5: identical forecasters yield stat=0, p=1."""
    rng = np.random.default_rng(2026)
    n = 500
    qs: dict[str, dict[str, np.ndarray]] = {"A": {}, "B": {}}
    for asset in ["x", "y", "z"]:
        base = rng.random(n) * 0.001
        qs["A"][asset] = base
        qs["B"][asset] = base.copy()

    stat, p = panel_dm_test(qs, "A", "B")
    assert abs(stat) < 1e-10
    assert p > 0.99


def test_small_noise_no_reject():
    """Adding mean-zero noise should fail to reject."""
    rng = np.random.default_rng(42)
    n = 500
    qs: dict[str, dict[str, np.ndarray]] = {"A": {}, "B": {}}
    for asset in ["x", "y", "z"]:
        base = rng.random(n) * 0.01
        noise = rng.normal(0, base.std(), n)
        noise -= noise.mean()
        qs["A"][asset] = base
        qs["B"][asset] = base + noise

    stat, p = panel_dm_test(qs, "A", "B")
    assert p > 0.05


def test_large_difference_rejects():
    """Large systematic difference should reject."""
    rng = np.random.default_rng(2026)
    n = 500
    qs: dict[str, dict[str, np.ndarray]] = {"A": {}, "B": {}}
    for asset in ["x", "y", "z"]:
        qs["A"][asset] = rng.random(n) * 0.001
        qs["B"][asset] = qs["A"][asset] + 0.01  # large shift

    stat, p = panel_dm_test(qs, "A", "B")
    assert p < 0.05
