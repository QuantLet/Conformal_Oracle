"""Deprecated top-level wrappers.

These emit DeprecationWarning and forward to the canonical API.
"""

from __future__ import annotations

import warnings

import pandas as pd

from conformal_oracle._protocols import Forecaster
from conformal_oracle.audit.single_rolling import RollingAuditResult
from conformal_oracle.audit.single_static import StaticAuditResult


def audit_static(
    returns: pd.Series,
    forecaster: Forecaster,
    **kwargs: object,
) -> StaticAuditResult:
    """Deprecated: use ``audit(returns, forecaster, mode='static')``."""
    warnings.warn(
        "audit_static() is deprecated since v0.3.0. "
        "Use audit(returns, forecaster, mode='static') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from conformal_oracle.audit.single_static import (
        audit_static as _audit_static,
    )

    return _audit_static(returns, forecaster, **kwargs)


def audit_rolling(
    returns: pd.Series,
    forecaster: Forecaster,
    **kwargs: object,
) -> RollingAuditResult:
    """Deprecated: use ``audit(returns, forecaster, mode='rolling')``."""
    warnings.warn(
        "audit_rolling() is deprecated since v0.3.0. "
        "Use audit(returns, forecaster, mode='rolling') instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from conformal_oracle.audit.single_rolling import (
        audit_rolling as _audit_rolling,
    )

    return _audit_rolling(returns, forecaster, **kwargs)


def audit_with_benchmarks(
    returns: pd.Series,
    forecaster: Forecaster,
    **kwargs: object,
) -> object:
    """Deprecated: use ``compare_forecasters()`` instead."""
    warnings.warn(
        "audit_with_benchmarks() is deprecated since v0.3.0. "
        "Use compare_forecasters() for the new API.",
        DeprecationWarning,
        stacklevel=2,
    )
    from conformal_oracle.audit.benchmark import (
        audit_with_benchmarks as _audit_with_benchmarks,
    )

    return _audit_with_benchmarks(returns, forecaster, **kwargs)


def audit_panel(
    returns_panel: pd.DataFrame,
    forecasters: object,
    **kwargs: object,
) -> object:
    """Deprecated: use ``conformal_oracle.panel.audit_panel()``."""
    warnings.warn(
        "audit_panel() is deprecated since v0.3.0. "
        "Use conformal_oracle.panel.audit_panel() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from conformal_oracle.panel.audit import audit_panel as _audit_panel

    return _audit_panel(returns_panel, forecasters, **kwargs)
