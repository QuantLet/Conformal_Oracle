"""RecalibrationMethod protocol for post-hoc VaR correction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RecalibrationMethod(Protocol):
    """A method that takes raw VaR forecasts and realised returns
    on a calibration set, and produces corrected VaR forecasts on
    a test set.

    The Forecaster protocol covers base forecasters that produce
    predictive distributions. RecalibrationMethod covers methods
    that adjust the forecaster's lower-tail quantile output.
    """

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        """Fit the recalibration parameters on calibration data.

        Args:
            raw_var_forecasts: Base VaR forecasts (positive = loss).
            realised: Realised returns on calibration set.
            alpha: Target tail probability (e.g. 0.01).
        """
        ...

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        """Apply the fitted recalibration to test-set forecasts.

        Args:
            raw_var_forecasts: Base VaR forecasts on test set.

        Returns:
            Corrected VaR forecasts (positive = loss).
        """
        ...


class ConformalShift:
    """The default conformal correction, wrapped as a RecalibrationMethod.

    Computes qV = quantile(scores, 1-alpha) where scores = -VaR_raw - r,
    then shifts VaR_corrected = VaR_raw + qV.
    """

    def __init__(self) -> None:
        self.q_v_stat: float = 0.0

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        scores = -raw_var_forecasts - realised
        self.q_v_stat = float(np.quantile(scores, 1 - alpha))

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        return raw_var_forecasts + self.q_v_stat
