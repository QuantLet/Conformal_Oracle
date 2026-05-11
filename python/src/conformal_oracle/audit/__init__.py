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
    audit_rolling,
)
from conformal_oracle.audit.single_static import (
    StaticAuditResult,
    audit_static,
)


def audit(
    returns: pd.Series,
    forecaster: Forecaster,
    alpha: float = 0.01,
    mode: Literal["static", "rolling"] = "static",
    recalibration: object | None = None,
    **kwargs: object,
) -> Union[StaticAuditResult, RollingAuditResult]:
    """Convenience dispatcher for static or rolling audit."""
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
