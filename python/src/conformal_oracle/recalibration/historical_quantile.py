"""Historical quantile recalibration baseline."""

from __future__ import annotations

import numpy as np


class HistoricalQuantileRecalibration:
    """Discards base forecast; returns the empirical alpha-quantile
    of calibration returns.

    VaR_HQ_t = -Q_alpha({r_tau : tau in calibration set})

    Zero free parameters. This is the trivial replacement baseline.
    """

    def __init__(self) -> None:
        self._var_hq: float = 0.0

    def fit(
        self,
        raw_var_forecasts: np.ndarray,
        realised: np.ndarray,
        alpha: float,
    ) -> None:
        self._var_hq = -float(np.quantile(realised, alpha))

    def apply(
        self,
        raw_var_forecasts: np.ndarray,
    ) -> np.ndarray:
        return np.full(len(raw_var_forecasts), self._var_hq)
