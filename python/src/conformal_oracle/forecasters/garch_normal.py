"""GARCH(1,1) forecaster with normal innovations."""

from __future__ import annotations

from conformal_oracle.forecasters.gjr_garch import GJRGARCHForecaster


class GARCHNormalForecaster(GJRGARCHForecaster):
    """Standard GARCH(1,1) with normal innovations on a rolling window."""

    def __init__(self, window: int = 250) -> None:
        super().__init__(window=window, distribution="normal")
