"""Scoring rules: quantile score (pinball loss) and Fissler-Ziegel FZ_0."""

from __future__ import annotations

import numpy as np


def quantile_score(
    realised: np.ndarray,
    forecasts: np.ndarray,
    alpha: float,
) -> float:
    """Pinball (quantile) loss averaged over t."""
    errors = realised - forecasts
    score = np.where(
        errors < 0,
        (alpha - 1) * errors,
        alpha * errors,
    )
    return float(np.mean(score))


def fissler_ziegel_fz0(
    realised: np.ndarray,
    var_forecasts: np.ndarray,
    es_forecasts: np.ndarray,
    alpha: float,
) -> float:
    """FZ_0 joint VaR-ES consistent scoring function."""
    violations = (realised < var_forecasts).astype(float)

    es_safe = np.where(np.abs(es_forecasts) < 1e-12, -1e-12, es_forecasts)

    term1 = (1.0 / (alpha * es_safe)) * violations * (var_forecasts - realised)
    term2 = var_forecasts / es_safe
    term3 = -np.log(-es_safe + 1e-12)

    fz = term1 + term2 + term3 - 1.0
    return float(np.mean(fz))
