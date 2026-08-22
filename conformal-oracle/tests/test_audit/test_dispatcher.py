"""Tests for the audit() convenience dispatcher."""

from conformal_oracle import audit
from conformal_oracle.audit.single_static import StaticAuditResult
from conformal_oracle.audit.single_rolling import RollingAuditResult
from conformal_oracle.forecasters import HistoricalSimulationForecaster


def test_dispatcher_static(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit(synthetic_returns, fc, mode="static")
    assert isinstance(result, StaticAuditResult)


def test_dispatcher_rolling(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit(synthetic_returns, fc, mode="rolling")
    assert isinstance(result, RollingAuditResult)


def test_dispatcher_invalid_mode(synthetic_returns):
    import pytest
    fc = HistoricalSimulationForecaster(window=250)
    with pytest.raises(ValueError, match="Unknown mode"):
        audit(synthetic_returns, fc, mode="bad")
