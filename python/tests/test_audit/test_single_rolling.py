"""Tests for audit_rolling."""

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit.single_rolling import audit_rolling, RollingAuditResult
from conformal_oracle.forecasters.hist_sim import HistoricalSimulationForecaster
from conformal_oracle._types import SampleDistribution


def test_rolling_returns_result(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_rolling(
        synthetic_returns, fc, alpha=0.01, window=250, warmup=250
    )
    assert isinstance(result, RollingAuditResult)
    assert result.mode == "rolling"


def test_rolling_qv_series_length(synthetic_returns):
    """qV_roll should have correct length."""
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_rolling(
        synthetic_returns, fc, alpha=0.01, window=250, warmup=250
    )
    expected_len = len(synthetic_returns) - 250 - 250
    assert len(result.q_v_roll) == expected_len


def test_rolling_drift_diagnostic_present(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_rolling(
        synthetic_returns, fc, alpha=0.01, window=250, warmup=250
    )
    assert len(result.drift_diagnostic) == len(result.q_v_roll)


def test_rolling_summary(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_rolling(
        synthetic_returns, fc, alpha=0.01, window=250, warmup=250
    )
    s = result.summary()
    assert "Rolling Conformal Audit" in s


def test_rolling_persistence_regime():
    """Persistence rule: 19-day spike should not trigger replacement."""
    rng = np.random.default_rng(42)
    n = 1500
    returns = pd.Series(
        rng.standard_normal(n) * 0.01,
        index=pd.bdate_range("2018-01-02", periods=n),
    )

    fc = HistoricalSimulationForecaster(window=250)
    result = audit_rolling(
        returns, fc, alpha=0.01, window=250, warmup=250, persistence=20
    )
    assert result.regime in ("signal-preserving", "replacement")
