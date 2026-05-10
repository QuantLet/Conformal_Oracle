"""Rolling conformal correction qV_roll and drift diagnostic."""

from __future__ import annotations

import numpy as np

from conformal_oracle._types import PredictiveDistribution


def compute_qv_roll(
    forecasts: list[PredictiveDistribution],
    realised: np.ndarray,
    alpha: float,
    window: int = 250,
) -> np.ndarray:
    """Compute rolling conformal correction qV_roll(t).

    For each t in [window, len(forecasts)), qV_roll(t) is the
    (1-alpha)-empirical quantile of the most recent `window`
    nonconformity scores S_{t-window}, ..., S_{t-1}.

    Returns array of length len(forecasts) - window.
    """
    scores = _compute_scores(forecasts, realised, alpha)
    return compute_qv_roll_from_scores(scores, alpha, window)


def compute_qv_roll_from_scores(
    scores: np.ndarray,
    alpha: float,
    window: int = 250,
) -> np.ndarray:
    """Rolling qV from precomputed nonconformity scores."""
    n = len(scores)
    out_len = n - window
    qv = np.empty(out_len)
    for t in range(window, n):
        qv[t - window] = np.quantile(scores[t - window : t], 1 - alpha)
    return qv


def compute_drift_diagnostic(
    scores: np.ndarray,
    window: int = 250,
    n_bins: int | None = None,
) -> np.ndarray:
    """Empirical TV distance between first and second halves of each window.

    delta_hat_w(t) = 0.5 * sum_b |p_{1,b}(t) - p_{2,b}(t)|
    """
    if n_bins is None:
        n_bins = max(int(np.sqrt(window)), 5)

    n = len(scores)
    out_len = n - window
    delta = np.empty(out_len)

    for t in range(window, n):
        w = scores[t - window : t]
        half = window // 2
        first_half = w[:half]
        second_half = w[half:]

        bins = np.linspace(w.min(), w.max(), n_bins + 1)
        h1, _ = np.histogram(first_half, bins=bins, density=True)
        h2, _ = np.histogram(second_half, bins=bins, density=True)

        h1_norm = h1 / (h1.sum() + 1e-12)
        h2_norm = h2 / (h2.sum() + 1e-12)

        delta[t - window] = 0.5 * np.sum(np.abs(h1_norm - h2_norm))

    return delta


def _compute_scores(
    forecasts: list[PredictiveDistribution],
    realised: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Nonconformity scores: S_t = q_alpha(F_t) - r_t."""
    n = len(forecasts)
    scores = np.empty(n)
    for t in range(n):
        scores[t] = forecasts[t].quantile(alpha) - realised[t]
    return scores
