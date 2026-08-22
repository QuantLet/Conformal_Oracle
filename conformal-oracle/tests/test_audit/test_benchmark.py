"""Tests for audit_with_benchmarks."""

import pytest

from conformal_oracle.audit.benchmark import audit_with_benchmarks, BenchmarkComparison
from conformal_oracle.forecasters.hist_sim import HistoricalSimulationForecaster


def test_benchmark_static(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_with_benchmarks(
        synthetic_returns, fc,
        benchmarks=["hist_sim"],
        mode="static",
    )
    assert isinstance(result, BenchmarkComparison)
    assert "hist_sim" in result.benchmarks


def test_benchmark_comparison_table(synthetic_returns):
    fc = HistoricalSimulationForecaster(window=250)
    result = audit_with_benchmarks(
        synthetic_returns, fc,
        benchmarks=["hist_sim"],
        mode="static",
    )
    df = result.comparison_table()
    assert "user" in df.index
    assert "hist_sim" in df.index


def test_benchmark_unknown_raises():
    import pandas as pd
    returns = pd.Series([0.01] * 100)
    fc = HistoricalSimulationForecaster()
    with pytest.raises(ValueError, match="Unknown benchmark"):
        audit_with_benchmarks(returns, fc, benchmarks=["not_a_model"])
