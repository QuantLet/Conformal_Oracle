"""T6: end-to-end PanelResult integration test."""

from __future__ import annotations

import pytest

from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.panel import audit_panel
from tests.fixtures.forecasters import ScaledForecaster


@pytest.fixture
def panel_static(panel_returns):
    """Run a 2-forecaster x 5-asset static panel."""
    forecasters = {
        "HistSim": HistoricalSimulationForecaster(window=250),
        "Scaled05": ScaledForecaster(scale=0.5, window=250),
    }
    return audit_panel(
        panel_returns, forecasters, alpha=0.01, mode="static",
    )


def test_master_table_rows(panel_static):
    mt = panel_static.master_table()
    assert len(mt) == 10  # 2 x 5


def test_regime_summary_consistent(panel_static):
    rs = panel_static.regime_summary()
    assert len(rs) == 2
    for _, row in rs.iterrows():
        total = row["n_signal_preserving"] + row["n_replacement"]
        assert total == 5


def test_cross_sectional_corr_runs(panel_static):
    corr = panel_static.cross_sectional_corr()
    assert len(corr) == 2  # 2 forecasters
    assert len(corr.columns) == 4  # 4 characteristics


def test_diagnostic_regression_runs(panel_static):
    dr = panel_static.diagnostic_regression()
    assert dr.n_obs == 10
    assert len(dr.coefficients) == 3
    assert dr.r_squared >= 0.0


def test_diebold_mariano_runs(panel_static):
    dm = panel_static.diebold_mariano(baseline="HistSim")
    assert len(dm) == 1  # 1 comparison (Scaled05 vs HistSim)
    assert "dm_statistic" in dm.columns
    assert "p_value" in dm.columns


def test_panel_kupiec_runs(panel_static):
    pk = panel_static.panel_kupiec()
    assert len(pk) == 2  # 1 per forecaster
    assert "lr_statistic" in pk.columns
    assert "p_value" in pk.columns


def test_wild_cluster_bootstrap_runs(panel_static):
    wcb = panel_static.wild_cluster_bootstrap(B=99, seed=2026)
    assert len(wcb.kupiec_table) == 2
    assert wcb.n_bootstrap == 99
    assert wcb.n_clusters == 5


def test_master_table_latex_nonempty(panel_static):
    latex = panel_static.master_table_latex()
    assert len(latex) > 0
    assert "\\begin{tabular}" in latex


def test_to_dict_roundtrip(panel_static):
    d = panel_static.to_dict()
    assert len(d["forecasters"]) == 2
    assert len(d["assets"]) == 5
    assert d["alpha"] == 0.01
    assert d["mode"] == "static"
