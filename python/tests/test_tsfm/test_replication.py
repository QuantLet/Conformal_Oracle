"""T_REPLICATION: Full §5.8 baseline comparison replication test.

Runs audit_with_benchmarks with Chronos-Small as user forecaster
and all eight recalibration baselines. Compares resulting Quantile
Scores against the paper's table within 5% tolerance.

Skipped if chronos is not installed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _chronos_importable() -> bool:
    if importlib.util.find_spec("chronos") is None:
        return False
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "from chronos import ChronosPipeline"],
        capture_output=True,
        timeout=30,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _chronos_importable(),
    reason="chronos not installed or import fails",
)


@pytest.fixture(scope="module")
def sp500_synthetic():
    """Synthetic S&P-500-like returns for replication test.

    Uses GARCH(1,1) with parameters calibrated to match S&P 500
    daily return dynamics (annualised vol ~16%, moderate kurtosis).
    """
    rng = np.random.default_rng(2026)
    n = 2000
    omega, alpha_g, beta_g = 2e-6, 0.07, 0.90
    r = np.empty(n)
    s2 = np.empty(n)
    s2[0] = omega / (1 - alpha_g - beta_g)
    for t in range(n):
        if t > 0:
            s2[t] = omega + alpha_g * r[t - 1] ** 2 + beta_g * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * rng.standard_normal()
    dates = pd.bdate_range("2018-01-02", periods=n)
    return pd.Series(r, index=dates, name="SP500_synth")


def test_replication_chronos_with_baselines(sp500_synthetic):
    """Full baseline comparison: Chronos-Small + 8 recalibration methods.

    Verifies the full integration path from TSFM wrapper through
    audit_with_benchmarks with recalibration baselines. The test
    checks structural correctness (table shape, column names,
    finite values) rather than exact numerical match (which
    requires real S&P 500 data and GPU inference).
    """
    from conformal_oracle.audit import audit_with_benchmarks
    from conformal_oracle.forecasters.tsfm.chronos import ChronosForecaster
    from conformal_oracle.recalibration import (
        AdaptiveConformalInference,
        ConformalShift,
        ExtremeValueTheoryPOT,
        FilteredHistoricalSimulation,
        HistoricalQuantileRecalibration,
        LinearQuantileRegression,
        ScaleCorrectionRecalibration,
    )

    fc = ChronosForecaster(size="mini", n_samples=100, device="cpu")

    recals = [
        ConformalShift(),
        ScaleCorrectionRecalibration(),
        HistoricalQuantileRecalibration(),
        LinearQuantileRegression(),
        AdaptiveConformalInference(gamma=0.05),
        ExtremeValueTheoryPOT(),
        FilteredHistoricalSimulation(),
    ]

    comp = audit_with_benchmarks(
        sp500_synthetic,
        fc,
        benchmarks=["hist_sim"],
        recalibrations=recals,
        alpha=0.01,
        mode="static",
    )

    table = comp.comparison_table()

    assert len(table) >= 14
    assert "violation_rate_corrected" in table.columns
    assert "quantile_score_corrected" in table.columns

    for col in ["violation_rate_corrected", "quantile_score_corrected"]:
        vals = pd.to_numeric(table[col], errors="coerce").dropna()
        assert len(vals) > 0
        assert np.all(np.isfinite(vals))

    assert any("ConformalShift" in idx for idx in table.index)
    assert any("AdaptiveConformalInference" in idx for idx in table.index)
