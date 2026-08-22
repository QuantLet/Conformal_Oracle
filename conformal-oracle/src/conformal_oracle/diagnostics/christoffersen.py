"""Christoffersen conditional coverage test."""

from __future__ import annotations

import numpy as np
from scipy import stats

from conformal_oracle.diagnostics.kupiec import kupiec_pof_pvalue


def christoffersen_pvalue(
    violations: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Conditional coverage test.

    Returns {"unconditional": p_uc, "independence": p_ind, "joint": p_cc}.
    The joint statistic is LR_POF + LR_IND ~ chi2(2).
    """
    T = len(violations)

    p_uc = kupiec_pof_pvalue(violations, alpha)

    n00, n01, n10, n11 = 0, 0, 0, 0
    for i in range(1, T):
        prev, curr = int(violations[i - 1]), int(violations[i])
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(T - 1, 1)

    lr_ind = 0.0
    if pi > 0 and pi < 1:
        if n00 + n01 > 0:
            if pi01 > 0 and pi01 < 1:
                lr_ind += 2.0 * (
                    n00 * np.log((1 - pi01) / (1 - pi))
                    + n01 * np.log(pi01 / pi)
                )
            elif pi01 == 0:
                lr_ind += 2.0 * n00 * np.log((1 - pi01 + 1e-12) / (1 - pi))
            else:
                lr_ind += 2.0 * n01 * np.log((pi01 - 1e-12) / pi)

        if n10 + n11 > 0:
            if pi11 > 0 and pi11 < 1:
                lr_ind += 2.0 * (
                    n10 * np.log((1 - pi11) / (1 - pi))
                    + n11 * np.log(pi11 / pi)
                )
            elif pi11 == 0:
                lr_ind += 2.0 * n10 * np.log((1 - pi11 + 1e-12) / (1 - pi))
            else:
                lr_ind += 2.0 * n11 * np.log((pi11 - 1e-12) / pi)

    lr_ind = max(lr_ind, 0.0)
    p_ind = float(stats.chi2.sf(lr_ind, df=1))

    lr_uc = stats.chi2.isf(p_uc, df=1) if p_uc < 1.0 else 0.0
    lr_cc = lr_uc + lr_ind
    p_cc = float(stats.chi2.sf(lr_cc, df=2))

    return {"unconditional": p_uc, "independence": p_ind, "joint": p_cc}
