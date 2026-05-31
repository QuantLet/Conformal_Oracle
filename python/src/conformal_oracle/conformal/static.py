"""Static conformal correction qV_stat."""

from __future__ import annotations

import numpy as np

from conformal_oracle._types import PredictiveDistribution


def compute_qv_stat(
    forecasts: list[PredictiveDistribution],
    realised: np.ndarray,
    alpha: float,
) -> float:
    """Compute the static conformal correction qV_stat.

    Score S_t = forecasts[t].quantile(alpha) - realised[t].
    qV_stat = empirical (1-alpha)-quantile of {S_t}.
    """
    scores = _compute_scores(forecasts, realised, alpha)
    return float(np.quantile(scores, 1 - alpha))


def _compute_scores(
    forecasts: list[PredictiveDistribution],
    realised: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Nonconformity scores: S_t = q_alpha(F_t) - r_t."""
    n = len(forecasts)
    scores = np.empty(n)
    for t in range(n):
        q_t = forecasts[t].quantile(alpha)
        scores[t] = q_t - realised[t]
    return scores
