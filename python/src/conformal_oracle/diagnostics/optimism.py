"""Nuisance-free estimators of the optimism of a fitted conformal shift.

For a static shift ``C_n`` fitted on ``n`` calibration scores, the manuscript's
expected-loss expansion gives, to first order, that the in-sample loss change
``I_n(C_n)`` understates the out-of-sample loss change by the penalty
``2 A_0 = Omega / (n f_*)``, with ``Omega`` the long-run variance of the
tail-hit indicator and ``f_*`` the score density at the target quantile. The
two estimators here need neither quantity:

* blocked cross-validation: ``K`` contiguous folds, the shift refitted on the
  complement of each fold, the held-out loss change averaged, and the
  difference from ``I_n(C_n)`` rescaled by ``2 (K - 1) / (2K - 1)`` (the
  first-order factor mapping ``A_{n(K-1)/K} + A_n`` to ``2 A_n`` under
  independence);
* circular block bootstrap: Efron's optimism, the loss change of the resampled
  shift on the original scores minus its training loss change, averaged over
  resamples of block length ``ceil(n^(1/3))``.

On the manuscript's synthetic histories both recover the mean penalty within
the prespecified accuracy criterion at ``n >= 700``. The first-order shrinkage
factor derived from them (:func:`first_order_shrinkage`) FAILED its
prespecified value criterion in the same study: with positive raw bias it had
higher expected loss than full correction in most cells. It is exposed only as
a diagnostic and must not be used as a deployment rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from conformal_oracle.conformal.quantile import conformal_quantile

__all__ = [
    "OptimismEstimate",
    "ShrinkageDiagnostic",
    "block_bootstrap_optimism",
    "blocked_cv_optimism",
    "first_order_shrinkage",
    "pinball_loss",
    "training_loss_change",
]


def pinball_loss(u: ArrayLike, alpha: float) -> np.ndarray:
    """``rho_alpha(u) = u (alpha - 1{u < 0})`` elementwise."""
    x = np.asarray(u, dtype=float)
    return np.asarray(x * (alpha - (x < 0.0)), dtype=float)


def _loss_change(c: float, scores: np.ndarray, alpha: float) -> float:
    """``D(c, S) = rho(c - S) - rho(-S)`` averaged over the scores."""
    return float(
        np.mean(pinball_loss(c - scores, alpha) - pinball_loss(-scores, alpha))
    )


def _check(scores: ArrayLike, alpha: float) -> tuple[np.ndarray, float]:
    s = np.asarray(scores, dtype=float)
    if s.ndim != 1 or s.size < 2 or not np.isfinite(s).all():
        raise ValueError("scores must be a finite 1-D array with at least two values")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return s, float(alpha)


def training_loss_change(scores: ArrayLike, alpha: float) -> float:
    """``I_n(C_n)``: in-sample loss change of the conformal shift."""
    s, a = _check(scores, alpha)
    return _loss_change(conformal_quantile(s, a), s, a)


@dataclass(frozen=True)
class OptimismEstimate:
    """Estimated penalty ``2 A_0`` and the quantities it was built from."""

    method: Literal["blocked_cv", "block_bootstrap"]
    alpha: float
    n_calibration: int
    shift: float
    training_loss_change: float
    penalty: float
    factor: float
    folds: int
    resamples: int
    block_length: int
    seed: int | None

    @property
    def estimated_out_of_sample_loss_change(self) -> float:
        """``I_n(C_n) + penalty``: the bias-corrected loss change."""
        return self.training_loss_change + self.penalty


def blocked_cv_optimism(
    scores: ArrayLike, alpha: float, folds: int = 5
) -> OptimismEstimate:
    """Blocked cross-validation estimate of the optimism penalty ``2 A_0``."""
    s, a = _check(scores, alpha)
    n = s.size
    if folds < 2 or n < 2 * folds:
        raise ValueError("need at least two folds and 2*folds scores")
    edges = np.linspace(0, n, folds + 1).astype(int)
    held_out = []
    for j in range(folds):
        lo, hi = int(edges[j]), int(edges[j + 1])
        rest = np.concatenate([s[:lo], s[hi:]])
        held_out.append(_loss_change(conformal_quantile(rest, a), s[lo:hi], a))
    c_n = conformal_quantile(s, a)
    train = _loss_change(c_n, s, a)
    factor = 2.0 * (folds - 1) / (2.0 * folds - 1.0)
    penalty = factor * (float(np.mean(held_out)) - train)
    return OptimismEstimate(
        method="blocked_cv",
        alpha=a,
        n_calibration=n,
        shift=c_n,
        training_loss_change=train,
        penalty=penalty,
        factor=factor,
        folds=folds,
        resamples=0,
        block_length=0,
        seed=None,
    )


def _block_length(n: int) -> int:
    m = int(round(n ** (1.0 / 3.0)))
    while m**3 < n:
        m += 1
    while m > 1 and (m - 1) ** 3 >= n:
        m -= 1
    return m


def block_bootstrap_optimism(
    scores: ArrayLike,
    alpha: float,
    resamples: int = 200,
    block_length: int | None = None,
    seed: int | Sequence[int] = 20260913,
) -> OptimismEstimate:
    """Circular block-bootstrap (Efron) estimate of the optimism penalty ``2 A_0``."""
    s, a = _check(scores, alpha)
    n = s.size
    blen = _block_length(n) if block_length is None else int(block_length)
    if blen < 1 or blen > n or resamples < 1:
        raise ValueError(
            "block_length must lie in [1, n] and resamples must be positive"
        )
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(resamples, -(-n // blen)))
    idx = ((starts[:, :, None] + np.arange(blen)) % n).reshape(resamples, -1)[:, :n]
    xs = s[idx]
    k = min(n, int(np.ceil((n + 1) * (1.0 - a))))
    c_star = np.partition(xs, k - 1, axis=1)[:, k - 1]
    fit_on_original = pinball_loss(c_star[:, None] - s[None, :], a).mean(axis=1)
    fit_on_sample = pinball_loss(c_star[:, None] - xs, a).mean(axis=1)
    base_original = pinball_loss(-s, a).mean()
    base_sample = pinball_loss(-xs, a).mean(axis=1)
    penalty = float(
        np.mean((fit_on_original - base_original) - (fit_on_sample - base_sample))
    )
    c_n = conformal_quantile(s, a)
    seed_value = int(seed) if isinstance(seed, int) else None
    return OptimismEstimate(
        method="block_bootstrap",
        alpha=a,
        n_calibration=n,
        shift=c_n,
        training_loss_change=_loss_change(c_n, s, a),
        penalty=penalty,
        factor=1.0,
        folds=0,
        resamples=int(resamples),
        block_length=blen,
        seed=seed_value,
    )


@dataclass(frozen=True)
class ShrinkageDiagnostic:
    """First-order shrinkage factor built from an optimism estimate.

    ``validated`` is always ``False``: in the manuscript's synthetic admission
    study the rule raised expected loss above full correction in most biased
    cells. Do not deploy it.
    """

    estimated_cost: float
    estimated_removable_loss: float
    factor: float
    validated: bool = False


def first_order_shrinkage(estimate: OptimismEstimate) -> ShrinkageDiagnostic:
    """``lambda = clip(B / (B + A), 0, 1)`` with ``A = penalty / 2`` and
    ``B = -I_n(C_n) - A``; zero when ``B + A <= 0``. Diagnostic only."""
    a_hat = estimate.penalty / 2.0
    b_hat = -estimate.training_loss_change - a_hat
    denominator = b_hat + a_hat
    lam = float(np.clip(b_hat / denominator, 0.0, 1.0)) if denominator > 0 else 0.0
    return ShrinkageDiagnostic(
        estimated_cost=a_hat, estimated_removable_loss=b_hat, factor=lam
    )
