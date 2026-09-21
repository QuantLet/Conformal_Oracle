"""One-coefficient corrections of a lower quantile forecast.

Four corrections subtract one fitted coefficient from the raw lower quantile
``q_t``. Scores are ``S_t = q_t - r_t`` on the calibration block; a positive
score is a violation of the raw threshold. With ``p = 1 - alpha``:

* ``shift_cp``: the split-conformal order statistic ``S_(k)``,
  ``k = min(n, ceil((n + 1) p))``; corrected quantile ``q_t - c``.
* ``shift_erm``: the inverse empirical CDF of the scores at ``p``, which
  minimises the calibration pinball loss over constant shifts.
* ``vol_cp``: the conformal order statistic of ``S_t / sigma_t``; corrected
  quantile ``q_t - c sigma_t``.
* ``vol_erm``: the ``sigma_t``-weighted empirical ``p``-quantile of
  ``S_t / sigma_t``, which minimises the calibration pinball loss in return
  units over shifts proportional to ``sigma_t``.

The volatility proxy ``sigma_t`` must be known before date ``t``; the
manuscript uses the standard deviation of the preceding twenty returns.
Under known scale, the scaled and constant corrections have different
first-order estimation costs (manuscript Proposition on correction shape);
neither is a coverage guarantee under temporal dependence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from conformal_oracle.conformal.quantile import conformal_quantile

__all__ = [
    "OneCoefficientCorrections",
    "fit_one_coefficient_corrections",
    "shift_cp",
    "shift_erm",
    "vol_cp",
    "vol_erm",
    "weighted_quantile",
]


def _scores(raw_quantiles: ArrayLike, realised: ArrayLike) -> np.ndarray:
    q = np.asarray(raw_quantiles, dtype=float)
    r = np.asarray(realised, dtype=float)
    if q.shape != r.shape or q.ndim != 1:
        raise ValueError("raw_quantiles and realised must be equal-length 1-D arrays")
    if q.size == 0 or not (np.isfinite(q).all() and np.isfinite(r).all()):
        raise ValueError("raw_quantiles and realised must be non-empty and finite")
    return np.asarray(q - r, dtype=float)


def _check_alpha(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return float(alpha)


def _check_sigma(sigma: ArrayLike, n: int) -> np.ndarray:
    s = np.asarray(sigma, dtype=float)
    if s.shape != (n,):
        raise ValueError("sigma must have one positive value per calibration date")
    if not (np.isfinite(s).all() and (s > 0).all()):
        raise ValueError("sigma must be finite and strictly positive")
    return s


def weighted_quantile(values: ArrayLike, weights: ArrayLike, p: float) -> float:
    """Weighted empirical ``p``-quantile: the smallest value whose cumulative
    weight (stable ascending order) reaches ``p`` times the total weight."""
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.shape != w.shape or x.ndim != 1 or x.size == 0:
        raise ValueError("values and weights must be equal-length non-empty 1-D arrays")
    if not ((w > 0).all() and np.isfinite(w).all() and np.isfinite(x).all()):
        raise ValueError("weights must be finite and positive; values finite")
    order = np.argsort(x, kind="stable")
    cumulative = np.cumsum(w[order])
    position = int(np.searchsorted(cumulative, p * cumulative[-1], side="left"))
    return float(x[order[min(position, x.size - 1)]])


def shift_cp(raw_quantiles: ArrayLike, realised: ArrayLike, alpha: float) -> float:
    """Split-conformal constant shift (order statistic with the capped rank)."""
    return conformal_quantile(_scores(raw_quantiles, realised), _check_alpha(alpha))


def shift_erm(raw_quantiles: ArrayLike, realised: ArrayLike, alpha: float) -> float:
    """Constant shift minimising calibration pinball loss (inverse empirical CDF)."""
    s = _scores(raw_quantiles, realised)
    return float(np.quantile(s, 1.0 - _check_alpha(alpha), method="inverted_cdf"))


def vol_cp(
    raw_quantiles: ArrayLike, realised: ArrayLike, sigma: ArrayLike, alpha: float
) -> float:
    """Conformal order statistic of the volatility-standardised scores."""
    s = _scores(raw_quantiles, realised)
    sig = _check_sigma(sigma, s.size)
    return conformal_quantile(s / sig, _check_alpha(alpha))


def vol_erm(
    raw_quantiles: ArrayLike, realised: ArrayLike, sigma: ArrayLike, alpha: float
) -> float:
    """Volatility-weighted quantile of the standardised scores: the coefficient
    of the scaled shift that minimises calibration pinball loss in return units."""
    s = _scores(raw_quantiles, realised)
    sig = _check_sigma(sigma, s.size)
    return weighted_quantile(s / sig, sig, 1.0 - _check_alpha(alpha))


@dataclass(frozen=True)
class OneCoefficientCorrections:
    """Fitted coefficients of the four one-coefficient corrections."""

    alpha: float
    n_calibration: int
    shift_cp: float
    shift_erm: float
    vol_cp: float
    vol_erm: float

    def apply(
        self, raw_quantiles: ArrayLike, sigma: ArrayLike
    ) -> dict[str, np.ndarray]:
        """Corrected lower quantiles on new dates, keyed by correction name.

        ``sigma`` must be the past-only volatility proxy of the new dates.
        """
        q = np.asarray(raw_quantiles, dtype=float)
        sig = _check_sigma(sigma, q.size)
        return {
            "Raw": q.copy(),
            "Shift-CP": q - self.shift_cp,
            "Shift-ERM": q - self.shift_erm,
            "Vol-CP": q - self.vol_cp * sig,
            "Vol-ERM": q - self.vol_erm * sig,
        }


def fit_one_coefficient_corrections(
    raw_quantiles: ArrayLike,
    realised: ArrayLike,
    sigma: ArrayLike,
    alpha: float,
) -> OneCoefficientCorrections:
    """Fit all four corrections on one calibration block."""
    s = _scores(raw_quantiles, realised)
    a = _check_alpha(alpha)
    sig = _check_sigma(sigma, s.size)
    return OneCoefficientCorrections(
        alpha=a,
        n_calibration=int(s.size),
        shift_cp=conformal_quantile(s, a),
        shift_erm=float(np.quantile(s, 1.0 - a, method="inverted_cdf")),
        vol_cp=conformal_quantile(s / sig, a),
        vol_erm=weighted_quantile(s / sig, sig, 1.0 - a),
    )
