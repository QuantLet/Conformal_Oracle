"""Adaptive Conformal Inference (Gibbs and Candes, 2021)."""

from __future__ import annotations

from typing import Sequence

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


class ACICalibrator:
    """Adaptive Conformal Inference calibrator with gamma selection.

    A thin wrapper around :class:`AdaptiveConformalInference` that is
    API-consistent with the rolling ``RecalibrationMethod`` (``fit`` / ``apply``,
    plus the online ``apply_online``) and adds step-size selection over a grid
    by first-half validation, as required for the baseline comparison.

    The ACI update is
    :math:`\\alpha_{t+1} = \\alpha_t + \\gamma\\,(\\alpha - \\mathrm{err}_t)`
    on the score sequence (Gibbs and Candes, 2021). ``gamma`` is chosen from
    ``gamma_grid`` by fitting on the first ``validation_fraction`` of the
    calibration set and picking the value whose online coverage on the held-out
    second half is closest to nominal. Passing an explicit ``gamma`` fixes the
    step size and skips selection (used for the gamma-sensitivity table).

    Attributes:
        selected_gamma: The gamma used after :meth:`fit` (either the fixed
            value or the one chosen by first-half validation).
    """

    DEFAULT_GAMMA_GRID: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05)

    def __init__(
        self,
        gamma: float | None = None,
        gamma_grid: Sequence[float] = DEFAULT_GAMMA_GRID,
        validation_fraction: float = 0.5,
    ) -> None:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        self.gamma = gamma
        self.gamma_grid = tuple(gamma_grid)
        self.validation_fraction = validation_fraction
        self.selected_gamma: float | None = gamma
        self._aci: AdaptiveConformalInference | None = None

    def _validation_error(
        self,
        raw: np.ndarray,
        realised: np.ndarray,
        alpha: float,
        gamma: float,
    ) -> float:
        """Absolute deviation of held-out online coverage from nominal."""
        n = len(raw)
        k = max(1, int(n * self.validation_fraction))
        if k >= n:  # not enough data to hold out; fall back to in-sample
            k = n // 2 or 1
        aci = AdaptiveConformalInference(gamma=gamma)
        aci.fit(raw[:k], realised[:k], alpha)
        corrected = aci.apply_online(raw[k:], realised[k:])
        viol = float(np.mean(realised[k:] < -corrected))
        return abs(viol - alpha)

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        raw = np.asarray(raw_var_forecasts, dtype=float)
        realised = np.asarray(realised, dtype=float)
        if raw.shape != realised.shape:
            raise ValueError("raw_var_forecasts and realised must have equal shape")

        if self.gamma is not None:
            self.selected_gamma = float(self.gamma)
        else:
            errors = {
                g: self._validation_error(raw, realised, alpha, g)
                for g in self.gamma_grid
            }
            # tie-break toward the smaller (smoother) step size
            self.selected_gamma = min(self.gamma_grid, key=lambda g: (errors[g], g))

        self._aci = AdaptiveConformalInference(gamma=self.selected_gamma)
        self._aci.fit(raw, realised, alpha)

    def apply(self, raw_var_forecasts: np.ndarray) -> np.ndarray:
        if self._aci is None:
            raise RuntimeError("ACICalibrator.apply called before fit")
        return self._aci.apply(raw_var_forecasts)

    def apply_online(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
    ) -> np.ndarray:
        """Full online ACI corrected VaR path (for the smoothness comparison)."""
        if self._aci is None:
            raise RuntimeError("ACICalibrator.apply_online called before fit")
        return self._aci.apply_online(raw_var_forecasts, realised)
