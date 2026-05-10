"""Historical Simulation forecaster."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conformal_oracle._types import PredictiveDistribution, SampleDistribution


class HistoricalSimulationForecaster:
    """Empirical distribution from the trailing window."""

    def __init__(self, window: int = 250) -> None:
        self.window = window

    def fit(self, returns: pd.Series) -> None:
        pass

    def forecast(
        self, returns: pd.Series, t: int
    ) -> PredictiveDistribution:
        start = max(0, t - self.window)
        samples = returns.iloc[start:t].values.copy()
        if len(samples) == 0:
            samples = np.array([0.0])
        return SampleDistribution(samples=samples)
