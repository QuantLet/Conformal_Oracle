"""Tests for panel LaTeX table emitters."""

from __future__ import annotations

from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.panel import audit_panel
from tests.fixtures.forecasters import ScaledForecaster


def test_master_table_latex(panel_returns):
    forecasters = {
        "HistSim": HistoricalSimulationForecaster(window=250),
        "Scaled05": ScaledForecaster(scale=0.5, window=250),
    }
    panel = audit_panel(
        panel_returns, forecasters, alpha=0.01, mode="static",
    )
    latex = panel.master_table_latex()
    assert "\\begin{tabular}" in latex
    assert "\\toprule" in latex
    assert "\\bottomrule" in latex
    assert "HistSim" in latex
    assert "Scaled05" in latex


def test_regime_summary_latex(panel_returns):
    forecasters = {
        "HistSim": HistoricalSimulationForecaster(window=250),
    }
    panel = audit_panel(
        panel_returns, forecasters, alpha=0.01, mode="static",
    )
    latex = panel.regime_summary_latex()
    assert "\\begin{tabular}" in latex
    assert "HistSim" in latex
