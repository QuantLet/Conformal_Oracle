"""Tests for LaTeX export and reporting module."""

from conformal_oracle import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster
from conformal_oracle.reporting.latex import (
    audit_result_to_latex_row,
    comparison_to_latex,
)


def test_latex_row_static(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    row = audit_result_to_latex_row(result, "HistSim")
    assert "HistSim" in row
    assert "\\\\" in row
    assert "green" in row or "yellow" in row or "red" in row


def test_latex_table_sorted(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    r1 = audit_static(synthetic_returns, fc, alpha=0.01)
    r2 = audit_static(synthetic_returns, fc, alpha=0.01)
    table = comparison_to_latex(
        {"ModelA": r1, "ModelB": r2},
        caption="Test table",
        label="tab:test",
    )
    assert "\\begin{table}" in table
    assert "\\end{table}" in table
    assert "\\toprule" in table
    assert "\\bottomrule" in table
    assert "Test table" in table
    assert "tab:test" in table


def test_latex_column_order(synthetic_returns):
    """Verify manuscript column order: pi_hat, Kupiec, Chr., Basel, QS, FZ, qV, R, Regime."""
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    row = audit_result_to_latex_row(result, "Test")
    parts = row.split("&")
    assert len(parts) == 10
