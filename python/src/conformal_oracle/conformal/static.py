"""Static conformal correction qV_stat."""

from __future__ import annotations

import numpy as np

from conformal_oracle._types import PredictiveDistribution
from conformal_oracle.conformal.quantile import conformal_quantile


def compute_qv_stat(
    forecasts: list[PredictiveDistribution],
    realised: np.ndarray,
    alpha: float,
) -> float:
    """Compute the static conformal correction qV_stat.

    Score S_t = forecasts[t].quantile(alpha) - realised[t].

    qV_stat is the finite-sample split-conformal quantile of {S_t}: the
    ``ceil((n + 1) * (1 - alpha))``-th order statistic of the n calibration
    scores, as computed by :func:`conformal_quantile`. It is NOT the plain
    empirical quantile ``np.quantile(scores, 1 - alpha)``; the two differ by one
    order statistic, an ``O(1/n)`` gap that is material at short windows and can
    change the sign of the correction when the scores straddle zero near the
    (1-alpha) level.
    """
    scores = _compute_scores(forecasts, realised, alpha)
    return conformal_quantile(scores, alpha)


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
