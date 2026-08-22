"""Example 4: Writing a custom forecaster against the protocol.

Implements a placeholder TSFM-like forecaster that returns a
Student-t predictive distribution, and audits it in both modes.
"""

import numpy as np
import pandas as pd

from conformal_oracle import audit_static, audit_rolling
from conformal_oracle._types import SampleDistribution


class FakeTSFMForecaster:
    """Placeholder TSFM-like forecaster for demonstration.

    Returns samples from a Student-t distribution scaled by the
    historical volatility. In practice, replace with calls to
    Chronos, TimesFM, Moirai, etc.
    """

    def __init__(self, df: float = 5.0, window: int = 250):
        self.df = df
        self.window = window

    def fit(self, returns: pd.Series) -> None:
        pass

    def forecast(self, returns: pd.Series, t: int) -> SampleDistribution:
        start = max(0, t - self.window)
        hist = returns.iloc[start:t]
        sigma = float(hist.std()) if len(hist) > 1 else 0.01
        rng = np.random.default_rng(t)
        samples = rng.standard_t(self.df, size=1000) * sigma
        return SampleDistribution(samples=samples)


def make_synthetic_returns(n: int = 2000, seed: int = 2026) -> pd.Series:
    rng = np.random.default_rng(seed)
    omega, alpha_g, beta_g = 1e-6, 0.05, 0.90
    returns = np.empty(n)
    sigma2 = np.empty(n)
    sigma2[0] = omega / (1 - alpha_g - beta_g)
    for t in range(n):
        if t > 0:
            sigma2[t] = omega + alpha_g * returns[t - 1] ** 2 + beta_g * sigma2[t - 1]
        returns[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    return pd.Series(returns, index=pd.bdate_range("2018-01-02", periods=n))


if __name__ == "__main__":
    returns = make_synthetic_returns()
    forecaster = FakeTSFMForecaster(df=5.0)

    print("=== Static Audit ===")
    static_result = audit_static(returns, forecaster, alpha=0.01)
    print(static_result.summary())

    print()
    print("=== Rolling Audit ===")
    rolling_result = audit_rolling(
        returns, forecaster, alpha=0.01, window=250, warmup=250
    )
    print(rolling_result.summary())
