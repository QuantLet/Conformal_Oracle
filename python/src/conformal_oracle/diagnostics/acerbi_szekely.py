"""Acerbi-Szekely Z2 backtest for Expected Shortfall."""

from __future__ import annotations

import numpy as np


def z2_statistic(
    violations: np.ndarray,
    realised: np.ndarray,
    es_forecasts: np.ndarray,
    alpha: float,
    stabilised: bool = True,
) -> float:
    """Z2 statistic for ES backtesting.

    When stabilised=True, uses time-averaged ES denominator to avoid
    division by near-zero daily ES on calm days.
    """
    mask = violations.astype(bool)
    n_viol = int(np.sum(mask))

    if n_viol == 0:
        return 0.0

    if stabilised:
        denom = np.mean(np.abs(es_forecasts)) + 1e-12
    else:
        denom = np.abs(es_forecasts[mask]) + 1e-12

    excess = realised[mask] + np.abs(es_forecasts[mask])

    if stabilised:
        z2 = float(np.sum(excess) / (n_viol * denom))
    else:
        z2 = float(np.mean(excess / denom))

    return z2
