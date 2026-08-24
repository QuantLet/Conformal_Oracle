"""Stationary block-bootstrap CI for qV."""

from __future__ import annotations

import numpy as np

from conformal_oracle.conformal.quantile import conformal_quantile


def bootstrap_qv_ci(
    scores: np.ndarray,
    alpha: float,
    confidence: float = 0.95,
    n_boot: int = 999,
    block_length: int = 20,
    seed: int = 2026,
) -> tuple[float, float]:
    """Stationary block-bootstrap CI for qV.

    Uses geometric block lengths with mean `block_length`.
    Returns (lower, upper) bounds of the CI.

    Each bootstrap replicate is the same estimator as the point estimate:
    :func:`conformal_quantile`, the order statistic of equation (8). Through
    version 0.3.1 the replicates were computed with ``np.quantile(sample,
    1 - alpha)`` instead, so the interval was centred on a different estimator
    from the value it was reported around. See CHANGELOG.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    qv_boots = np.empty(n_boot)
    p = 1.0 / block_length

    for b in range(n_boot):
        boot_sample = _stationary_bootstrap_sample(scores, n, p, rng)
        qv_boots[b] = conformal_quantile(boot_sample, alpha)

    tail = (1 - confidence) / 2
    lo = float(np.quantile(qv_boots, tail))
    hi = float(np.quantile(qv_boots, 1 - tail))
    return (lo, hi)


def _stationary_bootstrap_sample(
    data: np.ndarray,
    n: int,
    p: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one stationary bootstrap resample."""
    sample = np.empty(n)
    idx = rng.integers(0, n)
    for i in range(n):
        sample[i] = data[idx]
        if rng.random() < p:
            idx = rng.integers(0, n)
        else:
            idx = (idx + 1) % n
    return sample
