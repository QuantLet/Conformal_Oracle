"""Scale correction recalibration baseline."""

from __future__ import annotations

import numpy as np


class ScaleCorrectionRecalibration:
    """Multiplicative rescaling of raw VaR forecasts.

    c = alpha / pihat_cal, where pihat_cal is the empirical
    violation rate on the calibration set.

    VaR_SC_t = c * VaR_raw_t

    Single parameter c, chosen so that the rescaled calibration
    violation rate equals alpha.
    """

    def __init__(self) -> None:
        self._c: float = 1.0

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        pihat_cal = float(np.mean(realised < -raw_var_forecasts))
        self._c = alpha / pihat_cal if pihat_cal > 0 else 1.0

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        return raw_var_forecasts * self._c
