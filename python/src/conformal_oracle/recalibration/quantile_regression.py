"""Quantile regression recalibration baselines."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class LinearQuantileRegression:
    """Linear quantile regression: Q_alpha(r | VaR_raw) = a + b * VaR_raw.

    Two parameters (a, b) fit via pinball loss minimisation
    (Nelder-Mead). The corrected VaR on the test set is
    -(a + b * (-VaR_raw)), i.e. the negative of the fitted
    conditional quantile.
    """

    def __init__(self) -> None:
        self._b0: float = 0.0
        self._b1: float = 1.0

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        v = -raw_var_forecasts

        def pinball(params: np.ndarray) -> float:
            b0, b1 = params
            pred = b0 + b1 * v
            resid = realised - pred
            return float(np.sum(
                np.where(resid < 0, (alpha - 1) * resid, alpha * resid)
            ))

        res = minimize(
            pinball,
            [0.0, 1.0],
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-8},
        )
        self._b0, self._b1 = res.x

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        v_test = -raw_var_forecasts
        q_hat = self._b0 + self._b1 * v_test
        return -q_hat

    @property
    def intercept(self) -> float:
        return self._b0

    @property
    def slope(self) -> float:
        return self._b1


class IsotonicQuantileRegression:
    """Monotone non-parametric mapping via pool-adjacent violators.

    Fits P(violation | VaR_raw) using isotonic regression, then
    scales VaR_raw by alpha / P_hat(violation | VaR_raw) to
    achieve the target violation rate.

    Unstable at the 1% tail due to sparse violations.
    """

    def __init__(self) -> None:
        self._iso: object | None = None

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        from sklearn.isotonic import IsotonicRegression

        violations = (realised < -raw_var_forecasts).astype(float)
        self._iso = IsotonicRegression(
            y_min=0, y_max=1, out_of_bounds="clip",
        )
        self._iso.fit(-raw_var_forecasts, violations)
        self._alpha = alpha

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        pred_prob = self._iso.predict(-raw_var_forecasts)
        scale = np.where(
            pred_prob > 1e-6,
            self._alpha / np.clip(pred_prob, 1e-6, 1.0),
            1.0,
        )
        return raw_var_forecasts * scale
