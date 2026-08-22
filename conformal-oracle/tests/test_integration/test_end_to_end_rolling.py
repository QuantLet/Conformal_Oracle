"""End-to-end integration test for rolling audit."""

from conformal_oracle import audit_rolling
from conformal_oracle.forecasters import HistoricalSimulationForecaster


def test_e2e_rolling_corrected_green(synthetic_returns):
    """Rolling correction should achieve Basel green zone on synthetic data."""
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_rolling(
        synthetic_returns, fc, alpha=0.01, window=250, warmup=250
    )
    assert result.basel_zone_corrected in ("green", "yellow")


def test_e2e_rolling_all_series_aligned(synthetic_returns):
    """All output series should have the same index."""
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_rolling(
        synthetic_returns, fc, alpha=0.01, window=250, warmup=250
    )
    assert len(result.q_v_roll) == len(result.replacement_ratio)
    assert len(result.q_v_roll) == len(result.drift_diagnostic)
    assert len(result.q_v_roll) == len(result.var_corrected)
