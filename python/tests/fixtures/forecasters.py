"""Deliberately broken forecasters for edge-case testing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conformal_oracle._types import SampleDistribution


class ConstantZeroForecaster:
    """Predicts zero VaR — guaranteed replacement regime.

    Useful for verifying that diagnostics (FZ score, DM test,
    regime classification) remain stable when the forecaster
    produces degenerate predictions.
    """

    def fit(self, returns: pd.Series) -> None:
        pass

    def forecast(self, returns: pd.Series, t: int) -> SampleDistribution:
        return SampleDistribution(samples=np.zeros(100))


class ScaledForecaster:
    """Deliberately miscalibrated: multiplies historical vol by `scale`.

    scale < 1 underpredicts risk (positive qV, replacement regime).
    scale > 1 overpredicts risk (negative qV, conservative).
    """

    def __init__(self, scale: float = 0.5, window: int = 250) -> None:
        self.scale = scale
        self.window = window

    def fit(self, returns: pd.Series) -> None:
        pass

    def forecast(self, returns: pd.Series, t: int) -> SampleDistribution:
        start = max(0, t - self.window)
        samples = returns.iloc[start:t].values.copy() * self.scale
        if len(samples) == 0:
            samples = np.array([0.0])
        return SampleDistribution(samples=samples)
