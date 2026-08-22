"""GJR-GARCH(1,1) forecaster with skewed-t innovations."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

from conformal_oracle._types import ParametricDistribution, PredictiveDistribution


class GJRGARCHForecaster:
    """GJR-GARCH(1,1) with skewed-t innovations on a rolling window."""

    def __init__(self, window: int = 250, distribution: str = "skewt") -> None:
        self.window = window
        self.distribution = distribution

    def fit(self, returns: pd.Series) -> None:
        pass

    def forecast(
        self, returns: pd.Series, t: int
    ) -> PredictiveDistribution:
        start = max(0, t - self.window)
        train = returns.iloc[start:t]

        if len(train) < 50:
            loc = float(train.mean())
            scale = float(train.std()) if len(train) > 1 else 0.01
            return ParametricDistribution(
                location=loc, scale=scale, family="normal"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(
                train * 100,
                vol="Garch",
                p=1,
                o=1,
                q=1,
                dist=self.distribution,
                mean="Constant",
                rescale=False,
            )
            try:
                res = am.fit(disp="off", show_warning=False)
            except Exception:
                loc = float(train.mean())
                scale = float(train.std())
                return ParametricDistribution(
                    location=loc, scale=scale, family="normal"
                )

        fc = res.forecast(horizon=1)
        mu = float(fc.mean.iloc[-1, 0]) / 100
        var = float(fc.variance.iloc[-1, 0]) / (100**2)
        sigma = np.sqrt(max(var, 1e-12))

        params = res.params
        if self.distribution == "skewt" and "eta" in params and "lambda" in params:
            return ParametricDistribution(
                location=mu,
                scale=sigma,
                family="skewed_t",
                df=float(params["eta"]),
                skew=float(params["lambda"]),
            )
        elif self.distribution == "normal":
            return ParametricDistribution(
                location=mu, scale=sigma, family="normal"
            )
        else:
            df_val = float(params.get("nu", params.get("eta", 5.0)))
            return ParametricDistribution(
                location=mu, scale=sigma, family="student_t", df=df_val
            )
