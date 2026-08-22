"""Extreme value theory with peaks-over-threshold recalibration."""

from __future__ import annotations

import numpy as np
from scipy.stats import genpareto


class ExtremeValueTheoryPOT:
    """Extreme value theory with peaks-over-threshold.

    Uses EWMA-filtered residuals (McNeil and Frey, 2000):
      1. Standardise calibration returns by EWMA volatility.
      2. Fit GPD to exceedances above the threshold_quantile
         of |standardised residuals|.
      3. Compute VaR_z = u + (beta/xi)*((n/n_u * alpha)^(-xi) - 1).
      4. At test time, scale VaR_z by the base forecaster's
         implicit volatility: VaR_EVT = raw_VaR * (var_z / cal_ratio).

    Falls back to empirical quantile if GPD fit fails or produces
    extreme shape parameter (|xi| > 0.5).
    """

    def __init__(
        self,
        ewma_lambda: float = 0.94,
        threshold_quantile: float = 0.90,
    ) -> None:
        self._ewma_lambda = ewma_lambda
        self._threshold_quantile = threshold_quantile
        self._scale_factor: float = 1.0
        self._xi: float = 0.0
        self._beta: float = 0.0
        self._fit_ok: bool = False

    def _ewma_vol(self, returns: np.ndarray) -> np.ndarray:
        lam = self._ewma_lambda
        n = len(returns)
        sigma2 = np.empty(n)
        sigma2[0] = np.var(returns[:min(20, n)])
        for t in range(1, n):
            sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * returns[t - 1] ** 2
        return np.sqrt(np.maximum(sigma2, 1e-20))

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        sigma = self._ewma_vol(realised)
        z = realised / sigma
        z_valid = z[np.isfinite(z)]

        if len(z_valid) < 50:
            self._scale_factor = 1.0
            self._fit_ok = False
            return

        losses = -z_valid
        u = float(np.quantile(losses, self._threshold_quantile))
        exceedances = losses[losses > u] - u
        n_exc = len(exceedances)

        if n_exc < 10:
            var_z = -float(np.quantile(z_valid, alpha))
        else:
            try:
                xi, _, beta = genpareto.fit(exceedances, floc=0)
                if abs(xi) > 0.5 or beta <= 0:
                    var_z = -float(np.quantile(z_valid, alpha))
                else:
                    self._xi = xi
                    self._beta = beta
                    n_w = len(z_valid)
                    var_z = u + (beta / xi) * (
                        (n_w / n_exc * alpha) ** (-xi) - 1
                    )
                    self._fit_ok = True
            except Exception:
                var_z = -float(np.quantile(z_valid, alpha))

        mean_sigma = float(np.mean(sigma))
        evt_var = mean_sigma * var_z

        mean_raw = float(np.mean(np.abs(raw_var_forecasts)))
        if mean_raw > 1e-12:
            self._scale_factor = evt_var / mean_raw
        else:
            self._scale_factor = 1.0

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        return np.abs(raw_var_forecasts) * self._scale_factor

    @property
    def shape_parameter(self) -> float:
        return self._xi

    @property
    def scale_parameter(self) -> float:
        return self._beta

    @property
    def gpd_fit_ok(self) -> bool:
        return self._fit_ok
