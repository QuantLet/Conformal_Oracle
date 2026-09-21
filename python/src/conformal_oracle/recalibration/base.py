"""RecalibrationMethod protocol for post-hoc VaR correction."""

from __future__ import annotations

import warnings
from typing import Protocol, runtime_checkable

import numpy as np

from conformal_oracle.conformal.quantile import conformal_quantile


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
    """The conformal correction, wrapped as a RecalibrationMethod.

    Computes qV = quantile(scores, 1-alpha) where scores = -VaR_raw - r,
    then shifts VaR_corrected = VaR_raw + intensity * qV.

    Intensity. The default 1.0 applies the whole fitted shift, which is the
    correction the primary comparisons of the companion study evaluate. The
    whole shift lowers expected loss only when the correction the forecaster
    needs is larger than the standard error of the fitted quantile. At
    intensity 0.5, the average of the raw and the fully corrected threshold,
    the leading coefficient of the local-bias corollary becomes
    ``(f/8)(sigma^2 - 3 delta^2)``: the correction pays over a region three
    times wider in squared bias, costs a quarter as much when no correction
    was needed, and is the optimal intensity at the boundary where the whole
    shift stops paying.

    On the evaluated panels, intensity 0.5 lowered quantile loss against the
    whole shift at every calibration length in every universe, and against the
    raw forecast once the shift was fitted on 1000 calibration pairs. The
    recommended setting is therefore intensity 0.5 with at least 1000
    calibration pairs; that evidence is retrospective.

    Estimating the intensity from the same window that fits the shift did
    worse than the fixed 0.5 in every supported comparison between them, so no
    estimator of it is exposed here.
    :func:`conformal_oracle.diagnostics.optimism.first_order_shrinkage` is a
    different estimator of the same quantity and carries ``validated=False``
    for its own reason.

    Args:
        intensity: Fraction of the fitted shift to apply, in [0, 1].
    """

    def __init__(self, intensity: float = 1.0) -> None:
        if not np.isfinite(intensity) or not 0.0 <= intensity <= 1.0:
            raise ValueError("intensity must be a finite number in [0, 1]")
        self.intensity: float = float(intensity)
        self.q_v_stat: float = 0.0

    @property
    def shift(self) -> float:
        """The correction actually applied, intensity times the fitted shift."""
        return self.intensity * self.q_v_stat

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        scores = -raw_var_forecasts - realised
        n = int(np.asarray(scores).size)
        if n:
            rank = int(np.ceil((n + 1) * (1.0 - alpha)))
            if rank >= n:
                warnings.warn(
                    f"conformal rank {rank} reaches the calibration sample size "
                    f"{n} at alpha={alpha}, so the fitted shift is the largest "
                    "calibration score. Such a window carries no information "
                    "about the noise in its own shift.",
                    UserWarning,
                    stacklevel=2,
                )
        self.q_v_stat = conformal_quantile(scores, alpha)

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        return raw_var_forecasts + self.shift
