"""Filtered Historical Simulation recalibration."""

from __future__ import annotations

import numpy as np


class FilteredHistoricalSimulation:
    """GARCH/EWMA volatility filter + empirical quantile of
    standardised residuals.

    z_tau = r_tau / sigma_tau
    VaR_FHS_t = sigma_t * Q_alpha({z_tau : tau in calibration})

    Combines GARCH dynamics with non-parametric tail estimation.
    Uses EWMA volatility as the filter (robust to GARCH convergence
    failures that affect ~70% of windows in the paper's experiments).
    """

    def __init__(self, ewma_lambda: float = 0.94) -> None:
        self._ewma_lambda = ewma_lambda
        self._scale_factor: float = 1.0

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

        if len(z_valid) < 20:
            self._scale_factor = 1.0
            return

        q_alpha_z = float(np.quantile(z_valid, alpha))
        mean_sigma = float(np.mean(sigma))
        fhs_var = -mean_sigma * q_alpha_z

        mean_raw = float(np.mean(np.abs(raw_var_forecasts)))
        if mean_raw > 1e-12:
            self._scale_factor = fhs_var / mean_raw
        else:
            self._scale_factor = 1.0

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        return np.abs(raw_var_forecasts) * self._scale_factor
