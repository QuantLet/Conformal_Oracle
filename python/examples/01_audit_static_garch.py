"""Example 1: Static conformal audit with built-in GJR-GARCH.

Generates synthetic GARCH(1,1) returns, audits a GJR-GARCH forecaster
in static mode, and prints the result summary.
"""

import numpy as np
import pandas as pd

from conformal_oracle import audit_static
from conformal_oracle.forecasters import GJRGARCHForecaster


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

    result = audit_static(returns, forecaster, alpha=0.01, warmup=250)

    print(result.summary())
    print()
    print("LaTeX row:")
    print(result.to_latex_row("GJR-GARCH"))
