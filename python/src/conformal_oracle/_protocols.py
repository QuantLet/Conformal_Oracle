"""Forecaster protocol that user implementations must satisfy."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from conformal_oracle._types import PredictiveDistribution


class Forecaster(Protocol):
    """Protocol for one-step-ahead probabilistic forecasters.

    User's forecaster must implement fit() and forecast().
    """

    def fit(self, returns: pd.Series) -> None:
        """Fit on calibration data. May be a no-op for zero-shot models."""
        ...

    def forecast(self, returns: pd.Series, t: int) -> PredictiveDistribution:
        """One-step-ahead predictive distribution at time t.

        The forecaster sees only returns.iloc[:t] (history up to t-1).
        """
        ...
