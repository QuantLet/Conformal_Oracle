"""Tests for panel-pooled Kupiec test."""

from __future__ import annotations

import numpy as np

from conformal_oracle.panel.kupiec_panel import panel_kupiec_test


def test_perfect_calibration_not_rejected():
    """T4: exact alpha violation rate should not reject."""
    alpha = 0.01
    n = 10000
    violations = {}
    for i in range(10):
        v = np.zeros(n, dtype=int)
        n_viol = int(n * alpha)
        v[:n_viol] = 1
        violations[f"asset_{i}"] = v

    lr, p, per_asset = panel_kupiec_test(violations, alpha)
    assert p > 0.99


def test_severe_miscalibration_rejected():
    """Panel Kupiec should reject when violations >> alpha."""
    alpha = 0.01
    n = 1000
    violations = {}
    for i in range(5):
        v = np.zeros(n, dtype=int)
        v[:100] = 1  # 10% violation rate vs 1% target
        violations[f"asset_{i}"] = v

    lr, p, per_asset = panel_kupiec_test(violations, alpha)
    assert p < 0.01
    for a in per_asset:
        assert abs(per_asset[a] - 0.10) < 0.001


def test_per_asset_rates_correct():
    """Per-asset violation rates match known values."""
    violations = {
        "A": np.array([0, 0, 0, 0, 1]),
        "B": np.array([0, 0, 1, 1, 1]),
    }
    _, _, per_asset = panel_kupiec_test(violations, 0.01)
    assert abs(per_asset["A"] - 0.2) < 1e-10
    assert abs(per_asset["B"] - 0.6) < 1e-10
