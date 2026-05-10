"""Tests for audit_static."""

import numpy as np
import pandas as pd
import pytest

from conformal_oracle.audit.single_static import audit_static, StaticAuditResult
from conformal_oracle.forecasters.hist_sim import HistoricalSimulationForecaster
from conformal_oracle.forecasters.gjr_garch import GJRGARCHForecaster

from tests.fixtures.forecasters import ScaledForecaster


def test_static_returns_result(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    assert isinstance(result, StaticAuditResult)
    assert result.mode == "static"


def test_static_gjr_garch_qv_near_zero(synthetic_returns):
    """Well-specified GJR-GARCH on GARCH data: qV should be small."""
    fc = GJRGARCHForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    assert abs(result.q_v_stat) < 0.05


def test_static_miscalibrated_replacement(synthetic_returns):
    """Miscalibrated forecaster should be classified as replacement."""
    fc = ScaledForecaster(scale=0.3)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    assert result.q_v_stat > 0
    assert result.regime == "replacement"


def test_static_summary(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    s = result.summary()
    assert "Static Conformal Audit" in s
    assert "Regime" in s


def test_static_to_dict(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    d = result.to_dict()
    assert "q_v_stat" in d
    assert "regime" in d
