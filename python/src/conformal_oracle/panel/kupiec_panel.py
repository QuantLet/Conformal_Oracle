"""Panel-pooled Kupiec test."""

from __future__ import annotations

import numpy as np
from scipy import stats


def panel_kupiec_test(
    violations: dict[str, np.ndarray],
    alpha: float,
) -> tuple[float, float, dict[str, float]]:
    """Panel-pooled Kupiec test.

    Pools violations across assets:
    LR = -2 * log[((1-alpha)^(N-X) * alpha^X) /
                  ((1-X/N)^(N-X) * (X/N)^X)]

    Returns (LR_statistic, asymptotic_p, per_asset_violation_rates).
    """
    total_obs = 0
    total_viol = 0
    per_asset: dict[str, float] = {}

    for asset, v in violations.items():
        n = len(v)
        x = int(np.sum(v))
        total_obs += n
        total_viol += x
        per_asset[asset] = x / n if n > 0 else 0.0

    N = total_obs
    X = total_viol
    pi_hat = X / N if N > 0 else 0.0

    if X == 0 or X == N:
        return 0.0, 1.0, per_asset

    lr = -2.0 * (
        (N - X) * np.log((1 - alpha) / (1 - pi_hat))
        + X * np.log(alpha / pi_hat)
    )
    lr = max(lr, 0.0)
    p = float(stats.chi2.sf(lr, df=1))

    return lr, p, per_asset
