"""Tests for the forecast= agnostic audit path."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conformal_oracle import audit
from conformal_oracle.audit.single_rolling import RollingAuditResult
from conformal_oracle.audit.single_static import StaticAuditResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_quantile_data():
    """Returns + a synthetic quantile path that is slightly biased."""
    rng = np.random.default_rng(2026)
    n = 2000
    returns = rng.standard_normal(n) * 0.01
    dates = pd.bdate_range("2018-01-02", periods=n)
    r = pd.Series(returns, index=dates, name="returns")

    # Predict the 1% quantile as mean - 2.3 * std  (slightly off)
    q_lo = pd.Series(
        np.full(n, np.mean(returns) - 2.3 * np.std(returns)),
        index=dates,
        name="q_lo",
    )
    return r, q_lo


# ---------------------------------------------------------------------------
# audit() dispatch validation
# ---------------------------------------------------------------------------

def test_audit_requires_one_of_forecaster_or_forecast(simple_quantile_data):
    r, q_lo = simple_quantile_data
    with pytest.raises(ValueError, match="exactly one"):
        audit(r)


def test_audit_rejects_both_forecaster_and_forecast(simple_quantile_data):
    r, q_lo = simple_quantile_data

    class DummyFC:
        def fit(self, returns): pass
        def forecast(self, returns, t):
            from conformal_oracle._types import SampleDistribution
            return SampleDistribution(samples=np.zeros(10))

    with pytest.raises(ValueError, match="not both"):
        audit(r, DummyFC(), forecast=q_lo)


def test_audit_rejects_recalibration_with_forecast(simple_quantile_data):
    r, q_lo = simple_quantile_data
    with pytest.raises(ValueError, match="recalibration"):
        audit(r, forecast=q_lo, recalibration=object())


# ---------------------------------------------------------------------------
# Static agnostic audit
# ---------------------------------------------------------------------------

def test_static_agnostic_produces_result(simple_quantile_data):
    r, q_lo = simple_quantile_data
    result = audit(r, forecast=q_lo, mode="static")
    assert isinstance(result, StaticAuditResult)
    assert result.mode == "static"
    assert 0 < result.n_test < len(r)
    assert result.n_calibration + result.n_test == len(r)


def test_static_agnostic_es_is_nan(simple_quantile_data):
    """ES-related fields must be NaN in the agnostic path."""
    r, q_lo = simple_quantile_data
    result = audit(r, forecast=q_lo, mode="static")
    assert np.isnan(result.z2_statistic_raw)
    assert np.isnan(result.z2_statistic_corrected)
    assert np.isnan(result.fz_score_raw)
    assert np.isnan(result.fz_score_corrected)


def test_static_agnostic_qs_computed(simple_quantile_data):
    """Quantile scores should be finite."""
    r, q_lo = simple_quantile_data
    result = audit(r, forecast=q_lo, mode="static")
    assert np.isfinite(result.quantile_score_raw)
    assert np.isfinite(result.quantile_score_corrected)
    assert len(result.qs_sequence_raw) == result.n_test
    assert len(result.qs_sequence_corrected) == result.n_test


def test_static_agnostic_qv_stat_finite(simple_quantile_data):
    r, q_lo = simple_quantile_data
    result = audit(r, forecast=q_lo, mode="static")
    assert np.isfinite(result.q_v_stat)
    assert np.isfinite(result.q_v_stat_ci[0])
    assert np.isfinite(result.q_v_stat_ci[1])


# ---------------------------------------------------------------------------
# Rolling agnostic audit
# ---------------------------------------------------------------------------

def test_rolling_agnostic_produces_result(simple_quantile_data):
    r, q_lo = simple_quantile_data
    result = audit(r, forecast=q_lo, mode="rolling", window=250, warmup=250)
    assert isinstance(result, RollingAuditResult)
    assert result.mode == "rolling"
    assert result.n_test > 0


def test_rolling_agnostic_es_is_nan(simple_quantile_data):
    r, q_lo = simple_quantile_data
    result = audit(r, forecast=q_lo, mode="rolling", window=250, warmup=250)
    assert np.isnan(result.z2_statistic_corrected)
    assert np.isnan(result.fz_score_raw)
    assert np.isnan(result.fz_score_corrected)


def test_rolling_agnostic_qs_computed(simple_quantile_data):
    r, q_lo = simple_quantile_data
    result = audit(r, forecast=q_lo, mode="rolling", window=250, warmup=250)
    assert np.isfinite(result.quantile_score_raw)
    assert np.isfinite(result.quantile_score_corrected)
    assert result.n_test == len(result.qs_sequence_corrected)


def test_rolling_agnostic_qv_roll_series(simple_quantile_data):
    r, q_lo = simple_quantile_data
    result = audit(r, forecast=q_lo, mode="rolling", window=250, warmup=250)
    assert isinstance(result.q_v_roll, pd.Series)
    assert len(result.q_v_roll) == result.n_test


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_misaligned_index_raises():
    """If returns and forecast share no index, raise ValueError."""
    r = pd.Series([0.01, -0.02], index=[0, 1])
    q_lo = pd.Series([-0.03, -0.04], index=[10, 11])
    with pytest.raises(ValueError, match="no common index"):
        audit(r, forecast=q_lo, mode="static")


def test_invalid_mode_raises(simple_quantile_data):
    r, q_lo = simple_quantile_data
    with pytest.raises(ValueError, match="Unknown mode"):
        audit(r, forecast=q_lo, mode="bad")
