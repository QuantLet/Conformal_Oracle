"""Location-scale scale diagnostic for the conformal shift.

DIAGNOSTIC ONLY. This module quantifies how much of the one-parameter
conformal shift could, in principle, be reattributed to a multiplicative
(scale) term by relaxing the shift to the two-parameter location-scale map

    VaR_cp = a_hat + b_hat * VaR_raw.

It is *not* a competing recalibrator: the paper's argument that a single
degree of freedom (k = 1) is dictated by the alpha*T effective sample size
in the 1% tail is unchanged, and adding a second free parameter would only
amplify the small-sample instability that motivates the one-parameter design.
The diagnostic exists to confirm that the estimated conformal shift q_hat_V is
not a disguised scale correction.

The location-scale map is fit by alpha-level linear quantile regression of the
realised return on the raw VaR, reusing ``LinearQuantileRegression``. Writing
its fitted conditional quantile as Q_alpha(r) = b0 + b1 * (-VaR_raw), the
implied location-scale coefficients are a_hat = -b0 and b_hat = b1, so the
one-parameter conformal shift corresponds to the restricted fit (a_hat = q_hat_V,
b_hat = 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conformal_oracle.recalibration.base import ConformalShift
from conformal_oracle.recalibration.quantile_regression import (
    LinearQuantileRegression,
)


@dataclass(frozen=True)
class ScaleDiagnostic:
    """Decomposition of the correction into location and scale parts.

    Attributes:
        a_hat: Location coefficient of ``VaR_cp = a_hat + b_hat * VaR_raw``.
        b_hat: Multiplicative (scale) coefficient; ``b_hat = 1`` means the
            two-parameter fit reduces to a pure location shift.
        q_v_stat: The one-parameter conformal shift (the restricted fit with
            ``b_hat`` fixed to 1).
        loc_magnitude: ``|a_hat|`` -- size of the location component.
        scale_magnitude: ``|(b_hat - 1) * mean(VaR_raw)|`` -- size of the
            multiplicative component evaluated at the mean raw VaR.
        scale_share: ``scale_magnitude / (scale_magnitude + loc_magnitude)`` --
            the share of the total correction attributable to the
            multiplicative term. ``0`` means the correction is pure location
            (the conformal shift is not absorbing scale error); values near
            ``1`` mean the correction is mostly multiplicative.
    """

    a_hat: float
    b_hat: float
    q_v_stat: float
    loc_magnitude: float
    scale_magnitude: float
    scale_share: float


def diagnose_scale(
    raw_var_forecasts: np.ndarray,
    realised: np.ndarray,
    alpha: float,
) -> ScaleDiagnostic:
    """Decompose the correction into location and scale components.

    Args:
        raw_var_forecasts: Raw VaR forecasts (positive = loss) on the
            calibration set.
        realised: Realised returns on the calibration set.
        alpha: Target tail probability (e.g. 0.01).

    Returns:
        A :class:`ScaleDiagnostic` with the two-parameter coefficients, the
        one-parameter conformal shift, and the share of the correction taken
        by the multiplicative term.

    Raises:
        ValueError: If fewer than two observations are supplied.
    """
    raw = np.asarray(raw_var_forecasts, dtype=float)
    realised = np.asarray(realised, dtype=float)
    if raw.shape != realised.shape:
        raise ValueError("raw_var_forecasts and realised must have equal shape")
    if raw.size < 2:
        raise ValueError("diagnose_scale requires at least two observations")

    # One-parameter conformal shift (the restricted, b = 1 fit).
    shift = ConformalShift()
    shift.fit(raw, realised, alpha)
    q_v_stat = float(shift.q_v_stat)

    # Two-parameter location-scale fit, reusing the quantile-regression
    # baseline. Q_alpha(r) = b0 + b1 * (-VaR_raw)  =>  a_hat = -b0, b_hat = b1.
    lqr = LinearQuantileRegression()
    lqr.fit(raw, realised, alpha)
    a_hat = -float(lqr.intercept)
    b_hat = float(lqr.slope)

    mean_raw = float(np.mean(raw))
    loc_magnitude = abs(a_hat)
    scale_magnitude = abs((b_hat - 1.0) * mean_raw)
    denom = scale_magnitude + loc_magnitude
    scale_share = float(scale_magnitude / denom) if denom > 0 else 0.0

    return ScaleDiagnostic(
        a_hat=a_hat,
        b_hat=b_hat,
        q_v_stat=q_v_stat,
        loc_magnitude=loc_magnitude,
        scale_magnitude=scale_magnitude,
        scale_share=scale_share,
    )
