"""Tests for PanelResult.master_table() and regime_summary()."""

from __future__ import annotations

from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.panel import audit_panel
from tests.fixtures.forecasters import ScaledForecaster


def test_master_table_shape(panel_returns):
    forecasters = {
        "HistSim": HistoricalSimulationForecaster(window=250),
        "Scaled05": ScaledForecaster(scale=0.5, window=250),
    }
    panel = audit_panel(
        panel_returns, forecasters, alpha=0.01, mode="static",
    )
    mt = panel.master_table()
    assert len(mt) == 10  # 2 forecasters x 5 assets
    assert "forecaster" in mt.columns
    assert "asset" in mt.columns
    assert "regime" in mt.columns
    assert "q_v" in mt.columns
    assert "R" in mt.columns
    assert "pi_corrected" in mt.columns


def test_master_table_columns(panel_returns):
    forecasters = {
        "HistSim": HistoricalSimulationForecaster(window=250),
    }
    panel = audit_panel(
        panel_returns, forecasters, alpha=0.01, mode="static",
    )
    mt = panel.master_table()
    expected_cols = {
        "forecaster", "asset", "regime", "q_v", "R",
        "pi_raw", "pi_corrected", "basel_raw", "basel_corrected",
        "kupiec_p", "christoffersen_p",
        "qs_raw", "qs_corrected", "fz_raw", "fz_corrected",
    }
    assert expected_cols <= set(mt.columns)


def test_regime_summary_shape(panel_returns):
    forecasters = {
        "HistSim": HistoricalSimulationForecaster(window=250),
        "Scaled05": ScaledForecaster(scale=0.5, window=250),
    }
    panel = audit_panel(
        panel_returns, forecasters, alpha=0.01, mode="static",
    )
    rs = panel.regime_summary()
    assert len(rs) == 2
    assert "n_signal_preserving" in rs.columns
    assert "n_replacement" in rs.columns
    assert "green_corrected" in rs.columns
    for _, row in rs.iterrows():
        assert (
            row["n_signal_preserving"] + row["n_replacement"]
            == 5
        )
