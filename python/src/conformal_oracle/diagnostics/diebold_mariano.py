"""Diebold-Mariano test with Newey-West HAC variance and HLN correction."""

from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano_pvalue(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    horizon: int = 1,
    hln_correction: bool = True,
) -> float:
    """Two-sided DM p-value: H0 is equal predictive accuracy.

    Uses Newey-West HAC variance estimator with bandwidth = horizon - 1.
    Applies Harvey-Leybourne-Newbold small-sample correction when
    hln_correction=True.
    """
    d = losses_a - losses_b
    n = len(d)
    if n < 2:
        return 1.0

    d_bar = np.mean(d)
    lag = max(horizon - 1, 0)

    gamma_0 = np.var(d, ddof=0)
    hac_var = gamma_0
    for k in range(1, lag + 1):
        gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
        weight = 1.0 - k / (lag + 1)
        hac_var += 2.0 * weight * gamma_k

    hac_var = max(hac_var, 1e-20)

    dm_stat = d_bar / np.sqrt(hac_var / n)

    if hln_correction and n > 1:
        correction = np.sqrt(
            (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
        )
        dm_stat *= correction

    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return float(p_value)


def quantile_score_sequence(
    realised: np.ndarray,
    forecasts: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Per-step quantile scores (pinball losses) — needed for DM test."""
    errors = realised - forecasts
    return np.where(errors < 0, (alpha - 1) * errors, alpha * errors)
