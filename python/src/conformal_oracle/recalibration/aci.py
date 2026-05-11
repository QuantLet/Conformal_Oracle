"""Adaptive Conformal Inference (Gibbs and Candes, 2021)."""

from __future__ import annotations

import numpy as np


class AdaptiveConformalInference:
    """Sequential online update of the target miscoverage rate.

    alpha_{t+1} = alpha_t + gamma * (alpha - V_t)

    where V_t is the realised violation indicator at time t.

    The fitted state consists of the final alpha_T from the
    calibration run. At test time, the update continues
    sequentially using the realised test violations.

    Unlike other RecalibrationMethods, ACI needs realised returns
    at test time to continue updating. The apply() method uses
    the calibration-fitted alpha_T as a static shift. For the
    full online update, use apply_online() with test returns.
    """

    def __init__(self, gamma: float = 0.05) -> None:
        self.gamma = gamma
        self._alpha_target: float = 0.01
        self._q_v: float = 0.0

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        self._alpha_target = alpha
        scores = -raw_var_forecasts - realised

        alpha_t = alpha
        for t in range(len(scores)):
            violation = int(realised[t] < -raw_var_forecasts[t])
            alpha_t = alpha_t + self.gamma * (alpha - violation)
            alpha_t = np.clip(alpha_t, 1e-6, 1 - 1e-6)

        self._q_v = float(np.quantile(scores, 1 - alpha_t))

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        return raw_var_forecasts + self._q_v

    def apply_online(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
    ) -> np.ndarray:
        """Full online ACI: update alpha_t at each test step."""
        alpha = self._alpha_target
        scores_cal = -raw_var_forecasts - realised

        n = len(raw_var_forecasts)
        var_corrected = np.empty(n)

        alpha_t = alpha
        for t in range(n):
            q_t = float(np.quantile(scores_cal, 1 - alpha_t))
            var_corrected[t] = raw_var_forecasts[t] + q_t

            violation = int(realised[t] < -raw_var_forecasts[t])
            alpha_t = alpha_t + self.gamma * (alpha - violation)
            alpha_t = np.clip(alpha_t, 1e-6, 1 - 1e-6)

        return var_corrected
