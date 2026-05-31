"""Example 2: Rolling conformal audit with GJR-GARCH.

Generates synthetic returns, runs a rolling audit, prints the summary,
and saves a drift-diagnostic figure.
"""

import numpy as np
import pandas as pd

from conformal_oracle import audit_rolling
from conformal_oracle.forecasters import GJRGARCHForecaster
from conformal_oracle.reporting import plot_rolling_diagnostic


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
    forecaster = GJRGARCHForecaster(window=250)

    result = audit_rolling(returns, forecaster, alpha=0.01, window=250, warmup=250)

    print(result.summary())

    fig = plot_rolling_diagnostic(result, save_path="rolling_audit_diagnostic.png")
    print("\nFigure saved to rolling_audit_diagnostic.png")
