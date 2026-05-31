"""Audit API: static, rolling, benchmarks, and convenience dispatcher."""

from __future__ import annotations

from typing import Literal, Union

import pandas as pd

from conformal_oracle._protocols import Forecaster
from conformal_oracle.audit.benchmark import (
    BenchmarkComparison,
    audit_with_benchmarks,
)
from conformal_oracle.audit.single_rolling import (
    RollingAuditResult,
    _audit_rolling_from_quantiles,
    audit_rolling,
)
from conformal_oracle.audit.single_static import (
    StaticAuditResult,
    _audit_static_from_quantiles,
    audit_static,
)


def audit(
    returns: pd.Series,
    forecaster: Forecaster | None = None,
    *,
    forecast: pd.Series | None = None,
    alpha: float = 0.01,
    mode: Literal["static", "rolling"] = "static",
    recalibration: object | None = None,
    **kwargs: object,
) -> Union[StaticAuditResult, RollingAuditResult]:
    """Convenience dispatcher for static or rolling audit.

    Supply exactly one of ``forecaster`` (a Forecaster Protocol
    implementation) or ``forecast`` (a pre-computed lower-alpha
    quantile ``pd.Series``).

    When ``forecast`` is used the audit runs without any forecaster
    dependency, making the core install agnostic.

    Parameters
    ----------
    returns : pd.Series
        Log-return series (decimal, negative = loss).
    forecaster : Forecaster or None
        A fitted or fittable forecaster implementing the Forecaster
        Protocol.
    forecast : pd.Series or None
        Pre-computed predicted alpha-quantile path, aligned to
        ``returns`` by index.  Mutually exclusive with *forecaster*.
    alpha : float
        Tail probability level (default 0.01).
    mode : {"static", "rolling"}
        Audit mode.
    recalibration : RecalibrationMethod or None
        Optional recalibration baseline (only used with *forecaster*).
    **kwargs
        Forwarded to ``audit_static`` / ``audit_rolling``.
    """
    # --- input validation ---
    if forecaster is not None and forecast is not None:
        raise ValueError(
            "Supply exactly one of 'forecaster' or 'forecast', not both."
        )
    if forecaster is None and forecast is None:
        raise ValueError(
            "Supply exactly one of 'forecaster' or 'forecast'."
        )

    # --- agnostic path (pre-computed quantile series) ---
    if forecast is not None:
        if recalibration is not None:
            raise ValueError(
                "'recalibration' is not supported with the 'forecast=' "
                "agnostic path. Apply recalibration before passing the "
                "quantile series."
            )
        if mode == "static":
            return _audit_static_from_quantiles(
                returns, forecast, alpha=alpha, **kwargs,
            )
        elif mode == "rolling":
            return _audit_rolling_from_quantiles(
                returns, forecast, alpha=alpha, **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. Use 'static' or 'rolling'."
            )

    # --- forecaster path (original behaviour) ---
    if mode == "static":
        return audit_static(
            returns, forecaster, alpha=alpha,
            recalibration=recalibration, **kwargs,
        )
    elif mode == "rolling":
        return audit_rolling(
            returns, forecaster, alpha=alpha,
            recalibration=recalibration, **kwargs,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'static' or 'rolling'.")


__all__ = [
    "audit",
    "audit_static",
    "audit_rolling",
    "audit_with_benchmarks",
    "StaticAuditResult",
    "RollingAuditResult",
    "BenchmarkComparison",
]
