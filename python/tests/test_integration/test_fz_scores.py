"""Integration tests for FZ score and QS sequence in audit results."""

import numpy as np

from conformal_oracle import audit_static, audit_rolling
from conformal_oracle.forecasters import HistoricalSimulationForecaster


def test_static_fz_scores_finite(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    assert np.isfinite(result.fz_score_raw)
    assert np.isfinite(result.fz_score_corrected)


def test_static_qs_sequence_length(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    assert len(result.qs_sequence_raw) == result.n_test
    assert len(result.qs_sequence_corrected) == result.n_test


def test_static_qs_sequence_mean_matches(synthetic_returns):
    """Mean of QS sequence should match the scalar quantile_score."""
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_static(synthetic_returns, fc, alpha=0.01)
    assert abs(np.mean(result.qs_sequence_raw) - result.quantile_score_raw) < 1e-10
    assert abs(np.mean(result.qs_sequence_corrected) - result.quantile_score_corrected) < 1e-10


def test_rolling_fz_scores_finite(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_rolling(synthetic_returns, fc, alpha=0.01, window=250, warmup=250)
    assert np.isfinite(result.fz_score_raw)
    assert np.isfinite(result.fz_score_corrected)


def test_rolling_qs_sequence_length(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_rolling(synthetic_returns, fc, alpha=0.01, window=250, warmup=250)
    assert len(result.qs_sequence_raw) == result.n_test
    assert len(result.qs_sequence_corrected) == result.n_test
