"""End-to-end integration test for static audit."""

from conformal_oracle import audit_static
from conformal_oracle.forecasters import HistoricalSimulationForecaster


def test_e2e_static_violation_rate(synthetic_returns):
    """Corrected violation rate should be close to alpha."""
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    assert abs(result.violation_rate_corrected - 0.01) < 0.02


def test_e2e_static_all_fields_populated(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    assert result.q_v_stat_ci[0] < result.q_v_stat_ci[1]
    assert 0.0 <= result.violation_rate_raw <= 1.0
    assert result.kupiec_pvalue_raw >= 0
    assert result.christoffersen_pvalue_raw >= 0
    assert result.basel_zone_raw in ("green", "yellow", "red")
    assert result.n_calibration + result.n_test == len(synthetic_returns)
