"""Calibration-only indication and selective, causal conformal recalibration.

The indication is a frozen decision about whether to apply a correction, not
a model-quality certificate. Contiguous static and rolling corrections here
do not implement the separated estimator of the dependent-data theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from conformal_oracle.conformal.quantile import conformal_quantile
from conformal_oracle.diagnostics.kupiec import kupiec_pof_pvalue


@dataclass(frozen=True)
class RecalibrationDecision:
    """Immutable calibration-window diagnostics and the resulting OR decision.

    ``reasons`` contains ``basel_not_green`` and/or ``kupiec_rejection``;
    an empty tuple means skip. ``calibration_fingerprint`` binds the decision
    to the calibration values, Series index provenance, and decision settings.
    It cannot verify that a caller has truthfully labelled those values as
    calibration rather than evaluation.
    """

    apply: bool
    basel_zone: Literal["green", "yellow", "red"]
    kupiec_statistic: float
    kupiec_pvalue: float
    kupiec_level: float
    reasons: tuple[str, ...]
    information_window: Literal["calibration"]
    alpha: float
    n_calibration: int
    n_violations: int
    scaled_violations_250: float
    rule: Literal["basel_or_kupiec"]
    calibration_fingerprint: str

    @property
    def level(self) -> float:
        """Alias for the Kupiec rejection threshold."""
        return self.kupiec_level


@dataclass(frozen=True)
class SelectiveRecalibrationResult:
    """Raw/final quantiles and the corrections subtracted from the raw series.

    Arrays are independent, read-only copies. A skipped result retains the
    raw dtype and bit pattern exactly. ``finite_sample_rank`` is the requested
    rank before capping at the available score count; ``maximum_score_proxy``
    flags that cap, which does not retain the usual exchangeable finite-sample
    guarantee. Both are ``None`` when the correction is skipped.
    """

    raw_quantiles: np.ndarray
    final_quantiles: np.ndarray
    corrections: np.ndarray
    decision: RecalibrationDecision
    method: Literal["static", "rolling"]
    window: int | None
    n_calibration: int
    calibration_fingerprint: str
    evaluation_index: tuple[object, ...] | None
    finite_sample_rank: int | None
    maximum_score_proxy: bool | None

    @property
    def alpha(self) -> float:
        return self.decision.alpha

    @property
    def applied(self) -> bool:
        return self.decision.apply


def _probability(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and strictly between 0 and 1")
    value = float(value)
    if not np.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1")
    return value


def _vector(value: ArrayLike, name: str) -> tuple[np.ndarray, pd.Index | None]:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite real numbers")
    index = value.index if isinstance(value, pd.Series) else None
    if index is not None and (
        not index.is_unique or not index.is_monotonic_increasing or index.hasnans
    ):
        raise ValueError(f"{name} index must be unique, increasing, and nonmissing")
    return array, index


def _aligned(
    left: np.ndarray,
    left_index: pd.Index | None,
    right: np.ndarray,
    right_index: pd.Index | None,
    name: str,
) -> None:
    if len(left) != len(right):
        raise ValueError(f"{name} lengths must match")
    if (left_index is None) != (right_index is None):
        raise ValueError(f"{name} must both be indexed or both be positional")
    if left_index is not None and not left_index.equals(right_index):
        raise ValueError(f"{name} indices must match exactly")


def _calibration(
    returns: ArrayLike,
    quantiles: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, pd.Index | None, str]:
    r, ri = _vector(returns, "calibration_returns")
    q, qi = _vector(quantiles, "calibration_quantiles")
    _aligned(r, ri, q, qi, "calibration returns and quantiles")
    # Float64 is also the precision used for the conformal score computation.
    with np.errstate(over="raise", invalid="raise"):
        try:
            r, q = r.astype(float), q.astype(float)
        except FloatingPointError as exc:
            raise ValueError("calibration values exceed float64 finite range") from exc
    digest = sha256(r.tobytes() + q.tobytes())
    if ri is not None:
        # Object conversion normalises equivalent Timestamp/Timedelta values
        # across datetime storage units without discarding subsecond precision.
        labels = pd.util.hash_pandas_object(ri.astype(object), index=False)
        dtype = str(ri.dtype)
        if isinstance(ri, pd.DatetimeIndex):
            dtype = f"datetime:{ri.tz}"
        elif isinstance(ri, pd.TimedeltaIndex):
            dtype = "timedelta"
        digest.update(f"{type(ri).__qualname__}:{dtype}".encode("utf-8"))
        digest.update(labels.to_numpy(dtype=np.uint64).tobytes())
    return r, q, ri, digest.hexdigest()


def _configured_fingerprint(digest: str, alpha: float, level: float, rule: str) -> str:
    settings = np.asarray([alpha, level], dtype=">f8").tobytes() + rule.encode("ascii")
    return sha256(bytes.fromhex(digest) + settings).hexdigest()


def recalibration_indication(
    *,
    calibration_returns: ArrayLike,
    calibration_quantiles: ArrayLike,
    alpha: float = 0.01,
    kupiec_level: float = 0.05,
    rule: Literal["basel_or_kupiec"] = "basel_or_kupiec",
) -> RecalibrationDecision:
    """Decide on calibration data alone; no evaluation inputs are accepted.

    Quantiles use the lower-return-tail convention: a violation is strictly
    ``return < quantile``. Apply iff the Basel zone is not Green OR the
    two-sided Kupiec LR p-value is strictly below ``kupiec_level``. Equality
    at the rejection threshold does not reject.

    The R7 zone convention scales the violation count from the *entire*
    calibration window to 250 days, without rounding: Green <= 4, Yellow <= 9,
    otherwise Red. It is not the existing trailing-250 diagnostic. These fixed
    Basel thresholds refer to the 1% convention; they are not rescaled when
    a different ``alpha`` is supplied.

    Arrays are positional and must already be chronological. Paired pandas
    Series must have identical, unique, increasing indices. The API enforces
    this information boundary, but cannot detect falsely labelled caller data.
    """
    alpha = _probability(alpha, "alpha")
    kupiec_level = _probability(kupiec_level, "kupiec_level")
    if rule != "basel_or_kupiec":
        raise ValueError("rule must be 'basel_or_kupiec'")
    r, q, _, digest = _calibration(calibration_returns, calibration_quantiles)
    digest = _configured_fingerprint(digest, alpha, kupiec_level, rule)
    violations = r < q
    n, x = len(r), int(violations.sum())
    scaled = x * 250 / n
    zone: Literal["green", "yellow", "red"] = (
        "green" if scaled <= 4 else "yellow" if scaled <= 9 else "red"
    )
    if x == 0:
        statistic = -2.0 * n * np.log(1 - alpha)
    elif x == n:
        statistic = -2.0 * n * np.log(alpha)
    else:
        rate = x / n
        statistic = -2.0 * (
            x * np.log(alpha / rate) + (n - x) * np.log((1 - alpha) / (1 - rate))
        )
    pvalue = kupiec_pof_pvalue(violations, alpha)
    reasons = tuple(
        reason
        for reason, present in (
            ("basel_not_green", zone != "green"),
            ("kupiec_rejection", pvalue < kupiec_level),
        )
        if present
    )
    return RecalibrationDecision(
        apply=bool(reasons),
        basel_zone=zone,
        kupiec_statistic=float(statistic),
        kupiec_pvalue=pvalue,
        kupiec_level=kupiec_level,
        reasons=reasons,
        information_window="calibration",
        alpha=alpha,
        n_calibration=n,
        n_violations=x,
        scaled_violations_250=scaled,
        rule=rule,
        calibration_fingerprint=digest,
    )


def _readonly(array: np.ndarray) -> np.ndarray:
    array = array.copy()
    array.flags.writeable = False
    return array


def selectively_recalibrate(
    raw_quantiles: ArrayLike,
    *,
    calibration_returns: ArrayLike,
    calibration_quantiles: ArrayLike,
    decision: RecalibrationDecision,
    method: Literal["static", "rolling"] = "static",
    window: int = 250,
    evaluation_returns: ArrayLike | None = None,
) -> SelectiveRecalibrationResult:
    """Apply a frozen indication using a static or causal rolling correction.

    ``alpha`` is taken from the required decision; its calibration fingerprint
    and all calibration-only diagnostics must match the supplied history.
    Validation recomputes the indication from calibration data only; it does
    not replace the supplied decision or use evaluation data to choose policy.
    Static correction uses every
    calibration score. Rolling requires ``window`` calibration observations
    and uses only scores strictly before each evaluation time. Evaluation
    outcomes can update the rolling shift, never the frozen indication.

    If the decision says skip, no score or subtraction is computed and
    ``evaluation_returns`` is ignored (including its length and contents).
    Static correction also does not read evaluation outcomes. Inputs are
    positional arrays or consistently indexed Series; indexed evaluation
    dates must be strictly later than all calibration dates. No separation
    gap or dependent-data coverage guarantee is implied by that ordering.
    """
    if not isinstance(decision, RecalibrationDecision):
        raise TypeError("decision must be a RecalibrationDecision")
    if method not in ("static", "rolling"):
        raise ValueError("method must be 'static' or 'rolling'")
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, Integral):
        raise ValueError("window must be a positive integer")
    if window <= 0:
        raise ValueError("window must be a positive integer")
    alpha = _probability(decision.alpha, "decision.alpha")
    level = _probability(decision.kupiec_level, "decision.kupiec_level")
    if decision.information_window != "calibration":
        raise ValueError("decision information_window must be 'calibration'")
    if decision.rule != "basel_or_kupiec":
        raise ValueError("decision rule must be 'basel_or_kupiec'")
    if (
        decision.basel_zone not in ("green", "yellow", "red")
        or not np.isfinite(decision.kupiec_statistic)
        or not np.isfinite(decision.kupiec_pvalue)
        or not 0 <= decision.kupiec_pvalue <= 1
        or not isinstance(decision.apply, bool)
    ):
        raise ValueError("decision diagnostics must be finite and valid")
    expected_reasons = tuple(
        reason
        for reason, present in (
            ("basel_not_green", decision.basel_zone != "green"),
            ("kupiec_rejection", decision.kupiec_pvalue < level),
        )
        if present
    )
    if decision.reasons != expected_reasons or decision.apply != bool(expected_reasons):
        raise ValueError("decision diagnostics, reasons, and apply flag disagree")
    r, q, ci, digest = _calibration(calibration_returns, calibration_quantiles)
    digest = _configured_fingerprint(digest, alpha, level, decision.rule)
    if decision.n_calibration != len(r) or decision.calibration_fingerprint != digest:
        raise ValueError("decision does not match the supplied calibration history")
    expected_decision = recalibration_indication(
        calibration_returns=calibration_returns,
        calibration_quantiles=calibration_quantiles,
        alpha=alpha,
        kupiec_level=level,
        rule=decision.rule,
    )
    if decision != expected_decision:
        raise ValueError("decision does not match the calibration-only diagnostics")
    raw, ei = _vector(raw_quantiles, "raw_quantiles")
    if (ci is None) != (ei is None):
        raise ValueError(
            "calibration and evaluation must both be indexed or positional"
        )
    if ci is not None and ei is not None:
        try:
            ordered = bool(ci[-1] < ei[0])
        except TypeError as exc:
            raise ValueError(
                "calibration/evaluation indices must be comparable"
            ) from exc
        if not ordered:
            raise ValueError("evaluation must start strictly after calibration")
    rank, proxy, used_window = None, None, None
    if not decision.apply:
        final = raw.copy()
        corrections = np.zeros(len(raw), dtype=float)
    else:
        with np.errstate(over="raise", invalid="raise"):
            try:
                scores = q - r
                if method == "static":
                    corrections = np.full(len(raw), conformal_quantile(scores, alpha))
                    n_scores = len(scores)
                else:
                    if window > len(r):
                        raise ValueError("rolling window exceeds calibration length")
                    if evaluation_returns is None:
                        raise ValueError(
                            "rolling correction requires evaluation_returns"
                        )
                    outcomes, oi = _vector(evaluation_returns, "evaluation_returns")
                    _aligned(raw, ei, outcomes, oi, "evaluation quantiles and returns")
                    corrections = np.empty(len(raw))
                    history = list(scores[-window:])
                    for t in range(len(raw)):
                        corrections[t] = conformal_quantile(np.asarray(history), alpha)
                        if t + 1 < len(raw):
                            score = np.float64(raw[t]) - np.float64(outcomes[t])
                            history = history[1:] + [float(score)]
                    n_scores, used_window = int(window), int(window)
                final = raw.astype(float) - corrections
            except FloatingPointError as exc:
                raise ValueError(
                    "score or corrected quantile exceeds finite range"
                ) from exc
        if not np.isfinite(corrections).all() or not np.isfinite(final).all():
            raise ValueError("score or corrected quantile exceeds finite range")
        rank = int(np.ceil((n_scores + 1) * (1 - alpha)))
        proxy = rank > n_scores
    return SelectiveRecalibrationResult(
        raw_quantiles=_readonly(raw),
        final_quantiles=_readonly(final),
        corrections=_readonly(corrections),
        decision=decision,
        method=method,
        window=used_window,
        n_calibration=len(r),
        calibration_fingerprint=digest,
        evaluation_index=tuple(ei) if ei is not None else None,
        finite_sample_rank=rank,
        maximum_score_proxy=proxy,
    )


@dataclass(frozen=True)
class PastLossSelection:
    """Outcome of selection among candidate policies on an inner validation block.

    ``past_minimum`` is the candidate with the lowest mean validation loss
    (ties resolved by candidate order). ``cautious_gate`` is the candidate
    whose one-sided simultaneous bootstrap upper bound against ``Raw`` is
    negative at both block lengths and lowest, or ``Raw`` when none is. Both
    are operational heuristics on past loss, not guarantees of future loss.
    """

    past_minimum: str
    cautious_gate: str
    candidates: tuple[str, ...]
    mean_differences: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    draws: int
    block_lengths: tuple[int, ...]


def _circular_block_means(
    values: np.ndarray, block: int, draws: int, rng: np.random.Generator
) -> np.ndarray:
    n, d = values.shape
    blocks = int(np.ceil(n / block))
    remainder = n - (blocks - 1) * block
    starts = rng.integers(0, n, size=(draws, blocks))
    extended = np.concatenate([values, values[:block]], axis=0)
    prefix = np.vstack([np.zeros(d), np.cumsum(extended, axis=0)])
    sums = prefix[starts + block] - prefix[starts]
    sums[:, -1] = prefix[starts[:, -1] + remainder] - prefix[starts[:, -1]]
    return np.asarray(sums.sum(axis=1) / n, dtype=float)


def past_loss_selection(
    validation_losses: dict[str, ArrayLike],
    key: str,
    draws: int = 499,
    block_lengths: tuple[int, ...] = (20, 60),
    seed_prefix: str = "20260909",
) -> PastLossSelection:
    """Select a policy from validation-block daily losses (manuscript Section 7).

    ``validation_losses`` maps candidate names to equal-length daily pinball
    losses on the inner validation block; the first key must be ``"Raw"``.
    The seed of each block length is
    ``int.from_bytes(sha256(f"{seed_prefix}/{key}/{block}")[:4], "little")``.
    """
    names = list(validation_losses)
    if not names or names[0] != "Raw":
        raise ValueError("the first candidate must be 'Raw'")
    arrays = [np.asarray(validation_losses[n], dtype=float) for n in names]
    bad = any(
        a.ndim != 1 or a.shape != arrays[0].shape or not np.isfinite(a).all()
        for a in arrays
    )
    if bad:
        raise ValueError("validation losses must be finite 1-D arrays of equal length")
    differences = np.column_stack([a - arrays[0] for a in arrays[1:]])
    means = differences.mean(axis=0)
    uppers = []
    for block in block_lengths:
        digest = sha256(f"{seed_prefix}/{key}/{block}".encode()).digest()[:4]
        seed = int.from_bytes(digest, "little")
        rng = np.random.default_rng(seed)
        boot = _circular_block_means(differences, int(block), int(draws), rng)
        sd = boot.std(axis=0, ddof=1)
        positive = sd > 1e-15
        standard = np.zeros_like(boot)
        standard[:, positive] = (boot[:, positive] - means[positive]) / sd[positive]
        critical = max(0.0, float(np.quantile(standard.max(axis=1), 0.95)))
        uppers.append(means + critical * sd)
    upper = np.max(np.vstack(uppers), axis=0)
    best = int(np.argmin(upper))
    gate = names[best + 1] if upper[best] < 0 else "Raw"
    plain = min(
        names, key=lambda n: (float(np.mean(arrays[names.index(n)])), names.index(n))
    )
    return PastLossSelection(
        past_minimum=plain,
        cautious_gate=gate,
        candidates=tuple(names),
        mean_differences=tuple(float(x) for x in means),
        upper_bounds=tuple(float(x) for x in upper),
        draws=int(draws),
        block_lengths=tuple(int(b) for b in block_lengths),
    )
