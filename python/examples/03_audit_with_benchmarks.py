"""Example 3: Benchmark comparison in static mode.

Audits a GJR-GARCH forecaster (as the user's model) alongside
built-in GJR-GARCH and Historical Simulation benchmarks. Produces
a comparison table and Diebold-Mariano p-values.
"""

import numpy as np
import pandas as pd

from conformal_oracle import audit_with_benchmarks
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
    user_fc = GJRGARCHForecaster(window=250)

    comparison = audit_with_benchmarks(
        returns, user_fc,
        benchmarks=["gjr_garch", "hist_sim"],
        mode="static",
        warmup=250,
    )

    print("=== Comparison Table ===")
    df = comparison.comparison_table()
    cols = [c for c in df.columns if not c.startswith("qs_") and "sequence" not in c]
    print(df[cols].to_string())

    print()
    print("=== Diebold-Mariano p-values (vs GJR-GARCH) ===")
    dm = comparison.diebold_mariano(baseline="gjr_garch")
    for name, pval in dm.items():
        print(f"  {name}: {pval:.4f}")

    print()
    print("=== LaTeX Table ===")
    print(comparison.comparison_table_latex(
        caption="Conformal audit: user vs benchmarks",
        label="tab:benchmark",
    ))
