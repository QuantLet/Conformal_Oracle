"""Kupiec Proportion of Failures (POF) test."""

from __future__ import annotations

import numpy as np
from scipy import stats


def kupiec_pof_pvalue(
    violations: np.ndarray,
    alpha: float,
) -> float:
    """Two-sided p-value for the Kupiec POF likelihood-ratio test.

    Edge case x=0: use continuous-limit convention (x/T)^x = 1.
    """
    T = len(violations)
    x = int(np.sum(violations))

    if T == 0:
        return 1.0

    if x == 0:
        lr = -2.0 * T * np.log(1 - alpha)
    elif x == T:
        lr = -2.0 * T * np.log(alpha)
    else:
        pi_hat = x / T
        lr = -2.0 * (
            x * np.log(alpha / pi_hat)
            + (T - x) * np.log((1 - alpha) / (1 - pi_hat))
        )

    return float(stats.chi2.sf(lr, df=1))
