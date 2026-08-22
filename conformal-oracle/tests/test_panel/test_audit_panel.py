"""Tests for audit_panel orchestration."""

from __future__ import annotations

import pandas as pd

from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.panel import audit_panel
from tests.fixtures.forecasters import ScaledForecaster


def test_audit_panel_static_shape(panel_returns):
    """audit_panel returns correct number of results."""
    forecasters = {
        "HistSim": HistoricalSimulationForecaster(window=250),
        "Scaled05": ScaledForecaster(scale=0.5, window=250),
    }
    panel = audit_panel(
        panel_returns, forecasters, alpha=0.01, mode="static",
    )
    assert len(panel.forecaster_names) == 2
    assert len(panel.asset_names) == 5
    assert len(panel.results) == 2
    for fc_name in panel.forecaster_names:
        assert len(panel.results[fc_name]) == 5


def test_audit_panel_rolling_shape(panel_returns):
    """audit_panel rolling mode returns correct structure."""
    forecasters = {
        "HistSim": HistoricalSimulationForecaster(window=250),
    }
    panel = audit_panel(
        panel_returns, forecasters, alpha=0.01, mode="rolling",
    )
    assert panel.mode == "rolling"
    assert len(panel.results["HistSim"]) == 5


def test_audit_panel_deterministic(panel_returns):
    """Same seed produces identical results."""
    forecasters = {
        "HistSim": HistoricalSimulationForecaster(window=250),
    }
    p1 = audit_panel(
        panel_returns, forecasters, alpha=0.01,
        mode="static", seed=42,
    )
    p2 = audit_panel(
        panel_returns, forecasters, alpha=0.01,
        mode="static", seed=42,
    )
    for asset in p1.asset_names:
        r1 = p1.results["HistSim"][asset]
        r2 = p2.results["HistSim"][asset]
        assert r1.q_v_stat == r2.q_v_stat
