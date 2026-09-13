"""Finite-sample split-conformal quantile of nonconformity scores."""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    r"""Finite-sample split-conformal upper quantile of nonconformity scores.

    Returns the :math:`\lceil (n+1)(1-\alpha) \rceil`-th smallest score, i.e.
    the threshold giving marginal coverage :math:`\ge 1-\alpha` for
    exchangeable calibration and evaluation scores when this rank is at most
    ``n`` (Vovk, Gammerman and Shafer 2005; Lei et al. 2018). It is not the
    interpolated empirical quantile ``np.quantile(scores, 1 - alpha)`` and
    need not exactly minimise the empirical Quantile Score over translations.
    This rank convention alone does not establish validity under temporal
    dependence for a contiguous or rolling estimator.

    When :math:`\lceil (n+1)(1-\alpha) \rceil > n` (the conformal ``+inf`` case,
    reached when ``alpha < 1 / (n + 1)``) the largest observed score is
    returned as the finite proxy for an unbounded shift. That proxy does not
    retain the usual finite-sample coverage guarantee. Empty arrays return
    ``0.0`` for compatibility, not as a calibrated estimate.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = s.size
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(k, n)
    return float(s[k - 1])
