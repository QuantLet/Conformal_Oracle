"""Explicit separated single-split recalibration and a non-certified gap proxy.

These utilities do not fit a forecaster or infer a coverage guarantee. The
dependent-data theorem additionally requires its maintained assumptions on
the score process, forecaster context and separation. A positive gap alone,
and in particular a proxy-based gap, does not establish those assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from conformal_oracle.conformal.quantile import conformal_quantile


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        raw = np.asarray(values)
        if raw.ndim != 1 or raw.dtype.kind not in "iuf":
            raise ValueError
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{name} must be a one-dimensional real numeric array"
        ) from error
    if not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain nonempty finite data")
    return result


@dataclass(frozen=True)
class GapResult:
    """Operational gap metadata, not an estimate or certificate of mixing.

    ``gap = context_length + log_gap``. ``persistence_proxy`` is the absolute
    Pearson lag-one score autocorrelation, not a mixing-rate estimator.
    ``minimum_log_gap`` is used only at numerically zero persistence; it is
    not a lower bound on the logarithmic term for nonzero persistence.
    """

    gap: int
    context_length: int
    log_gap: int
    persistence_proxy: float
    signed_autocorrelation: float
    safety_factor: float
    n_calibration: int
    minimum_log_gap: int
    numerical_zero_threshold: float
    near_zero: bool
    proxy_based: bool = field(default=True, init=False)
    certified: bool = field(default=False, init=False)

    @property
    def rho_tilde(self) -> float:
        """Alias for the absolute lag-one persistence proxy."""
        return self.persistence_proxy


def proxy_separation_gap(
    scores: ArrayLike,
    *,
    context_length: int,
    safety_factor: float = 1.1,
    minimum_log_gap: int = 5,
    numerical_zero_threshold: float = 1e-12,
) -> GapResult:
    """Compute the R7-style gap from calibration scores only.

    For ``rho_tilde = abs(corr(scores[:-1], scores[1:]))`` above the numerical
    zero threshold, the logarithmic term is
    ``ceil(safety_factor * log(n) / abs(log(rho_tilde)))``. At or below the
    threshold only, use ``minimum_log_gap``. Undefined, constant or
    unit-magnitude proxies are rejected, including negative unit correlation.
    Any finite computed magnitude below one uses the stated formula without
    clipping, even when it produces a gap too large for a particular split.
    The adjacent-slice Pearson calculation agrees with pandas lag-one
    autocorrelation for these finite inputs.

    Supply calibration scores, never evaluation scores. ``safety_factor``
    must exceed one, but applying it to this proxy does not certify the
    theorem's strict inequality for the unknown mixing rate.
    """
    context = _integer(context_length, "context_length")
    floor = _integer(minimum_log_gap, "minimum_log_gap")
    factor = _real(safety_factor, "safety_factor")
    threshold = _real(numerical_zero_threshold, "numerical_zero_threshold")
    if factor <= 1:
        raise ValueError("safety_factor must exceed one")
    if not 0 <= threshold < 1:
        raise ValueError("numerical_zero_threshold must lie in [0, 1)")
    values = _vector(scores, "scores")
    if len(values) < 3:
        raise ValueError("at least three calibration scores are required")
    left, right = values[:-1], values[1:]
    if np.all(left == left[0]) or np.all(right == right[0]):
        raise ValueError("lag-one autocorrelation is undefined for constant slices")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        rho = float(np.corrcoef(left, right)[0, 1])
    persistence = abs(rho)
    if not np.isfinite(persistence) or persistence >= 1:
        raise ValueError("lag-one autocorrelation must be finite and in (-1, 1)")
    near_zero = persistence <= threshold
    if near_zero:
        log_gap = floor
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            requested_gap = factor * np.log(len(values)) / abs(np.log(persistence))
        if not np.isfinite(requested_gap):
            raise ValueError("the requested logarithmic gap is not finite")
        log_gap = int(np.ceil(requested_gap))
    return GapResult(
        gap=context + log_gap,
        context_length=context,
        log_gap=log_gap,
        persistence_proxy=persistence,
        signed_autocorrelation=rho,
        safety_factor=factor,
        n_calibration=len(values),
        minimum_log_gap=floor,
        numerical_zero_threshold=threshold,
        near_zero=near_zero,
    )


@dataclass(frozen=True)
class SeparatedSplitResult:
    """Evaluation-only lower quantiles and explicit split provenance.

    Integer index arrays are zero-based input positions, even for Series.
    ``evaluation_index`` retains Series labels separately. Corrected lower
    quantiles equal ``raw_quantiles - q_v_stat``; positive-loss VaR values
    would be their negatives. No evaluation outcomes enter ``q_v_stat``.

    ``maximum_score_fallback`` denotes a requested conformal rank above the
    calibration size, where the existing helper returns the finite maximum
    instead of infinity. This fallback lacks the usual coverage guarantee.
    No guarantee is certified by the returned result in any case.
    """

    q_v_stat: float
    calibration_indices: NDArray[np.int64]
    gap_indices: NDArray[np.int64]
    evaluation_indices: NDArray[np.int64]
    raw_quantiles: NDArray[np.float64]
    corrected_quantiles: NDArray[np.float64]
    alpha: float
    calibration_fraction: float
    gap: int
    conformal_rank: int
    maximum_score_fallback: bool
    evaluation_index: pd.Index | None = None
    certified: bool = field(default=False, init=False)

    @property
    def n_calibration(self) -> int:
        return len(self.calibration_indices)

    @property
    def n_evaluation(self) -> int:
        return len(self.evaluation_indices)


@dataclass(frozen=True, kw_only=True)
class SeparatedSplitConformalVaR:
    """Apply an explicitly specified gap without changing the static shift.

    ``n_cal = int(calibration_fraction * N)``. Calibration positions are
    ``[0, n_cal)``, gap positions ``[n_cal, n_cal + gap)``, and evaluation
    begins at ``n_cal + gap``. The gap discards observations from evaluation;
    it does not remove any calibration scores or change their order statistic.

    Inputs must be aligned finite one-dimensional numeric arrays, or Series
    with identical, unique, increasing indexes. Forecasts must already be
    predictable from their own past; this helper does not enforce that or fit
    a model. Evaluation returns are accepted for alignment/validation only,
    never for fitting the shift. An explicit ``gap=0`` is permitted as a
    contiguous comparator, not as evidence of the theorem's separation.
    """

    gap: int
    alpha: float = 0.01
    calibration_fraction: float = 0.70
    minimum_evaluation_size: int = 1

    def __post_init__(self) -> None:
        gap = _integer(self.gap, "gap")
        minimum = _integer(
            self.minimum_evaluation_size, "minimum_evaluation_size", minimum=1,
        )
        alpha = _real(self.alpha, "alpha")
        fraction = _real(self.calibration_fraction, "calibration_fraction")
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if not 0 < fraction < 1:
            raise ValueError(
                "calibration_fraction must lie strictly between zero and one"
            )
        object.__setattr__(self, "gap", gap)
        object.__setattr__(self, "minimum_evaluation_size", minimum)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "calibration_fraction", fraction)

    def split(self, returns: ArrayLike, quantiles: ArrayLike) -> SeparatedSplitResult:
        """Compute the calibration shift and return only evaluation forecasts."""
        returns_series = isinstance(returns, pd.Series)
        quantiles_series = isinstance(quantiles, pd.Series)
        if returns_series != quantiles_series:
            raise ValueError("supply either two Series or two positional arrays")
        labels = None
        if isinstance(returns, pd.Series) and isinstance(quantiles, pd.Series):
            if not returns.index.equals(quantiles.index):
                raise ValueError("returns and quantiles must have identical indexes")
            if not returns.index.is_unique or not returns.index.is_monotonic_increasing:
                raise ValueError("Series indexes must be unique and increasing")
            labels = returns.index
        realised = _vector(returns, "returns")
        predicted = _vector(quantiles, "quantiles")
        if len(realised) != len(predicted):
            raise ValueError("returns and quantiles must have equal lengths")
        n = len(realised)
        n_cal = int(self.calibration_fraction * n)
        evaluation_start = n_cal + self.gap
        if n_cal < 1:
            raise ValueError(
                "the split must contain at least one calibration observation"
            )
        if n - evaluation_start < self.minimum_evaluation_size:
            raise ValueError(
                "insufficient evaluation observations after the requested gap"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            scores = predicted[:n_cal] - realised[:n_cal]
        if not np.all(np.isfinite(scores)):
            raise ValueError("calibration scores must be finite")
        shift = conformal_quantile(scores, self.alpha)
        raw = predicted[evaluation_start:].copy()
        with np.errstate(over="ignore", invalid="ignore"):
            corrected = raw - shift
        if not np.all(np.isfinite(corrected)):
            raise ValueError("corrected quantiles must be finite")
        rank = int(np.ceil((n_cal + 1) * (1 - self.alpha)))
        return SeparatedSplitResult(
            q_v_stat=shift,
            calibration_indices=np.arange(n_cal, dtype=np.int64),
            gap_indices=np.arange(n_cal, evaluation_start, dtype=np.int64),
            evaluation_indices=np.arange(evaluation_start, n, dtype=np.int64),
            raw_quantiles=raw,
            corrected_quantiles=corrected,
            alpha=self.alpha,
            calibration_fraction=self.calibration_fraction,
            gap=self.gap,
            conformal_rank=rank,
            maximum_score_fallback=rank > n_cal,
            evaluation_index=(
                None if labels is None else labels[evaluation_start:].copy()
            ),
        )
