"""Finite-sample split-conformal quantile of nonconformity scores."""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    r"""Finite-sample split-conformal upper quantile of nonconformity scores.

    Returns the :math:`\lceil (n+1)(1-\alpha) \rceil`-th smallest score, i.e.
    the threshold that guarantees marginal coverage :math:`\ge 1-\alpha` in
    finite samples (Vovk, Gammerman and Shafer 2005; Lei et al. 2018). This is
    one order statistic more conservative than the plain empirical
    ``np.quantile(scores, 1 - alpha)``: the gap is :math:`O(1/n)`, so it is
    negligible on large calibration sets but material at short windows (for the
    250-day rolling correction at :math:`\alpha=0.01` it is the 249th vs the
    plain 247.5th order statistic, worth ~0.6 pp of realised coverage).

    When :math:`\lceil (n+1)(1-\alpha) \rceil > n` (the conformal ``+inf`` case,
    reached when ``alpha < 1 / (n + 1)``) the largest observed score is
    returned as the finite proxy for an unbounded shift.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(k, n)
    return float(s[k - 1])
