"""Conformal recalibration audit for tail quantile forecasters."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from conformal_oracle._deprecated import (
    audit_panel,
    audit_rolling,
    audit_static,
    audit_with_benchmarks,
)
from conformal_oracle._protocols import Forecaster
from conformal_oracle._types import (
    ParametricDistribution,
    PredictiveDistribution,
    QuantileGridDistribution,
    SampleDistribution,
)
from conformal_oracle.audit import audit
from conformal_oracle.classify import RegimeVerdict, classify_regime
from conformal_oracle.compare import ComparisonResult, compare_forecasters

if TYPE_CHECKING:
    from conformal_oracle.recalibration import (
        AdaptiveConformalInference,
        ConformalShift,
        ExtremeValueTheoryPOT,
        FilteredHistoricalSimulation,
        GBMQuantileRegression,
        HistoricalQuantileRecalibration,
        IsotonicQuantileRegression,
        LinearQuantileRegression,
        RecalibrationMethod,
        ScaleCorrectionRecalibration,
    )

__version__ = "0.3.0"

_RECALIBRATION_NAMES = {
    "RecalibrationMethod",
    "ConformalShift",
    "HistoricalQuantileRecalibration",
    "ScaleCorrectionRecalibration",
    "LinearQuantileRegression",
    "IsotonicQuantileRegression",
    "AdaptiveConformalInference",
    "GBMQuantileRegression",
    "ExtremeValueTheoryPOT",
    "FilteredHistoricalSimulation",
}

def __getattr__(name: str):
    if name in _RECALIBRATION_NAMES:
        mod = importlib.import_module("conformal_oracle.recalibration")
        return getattr(mod, name)
    raise AttributeError(f"module 'conformal_oracle' has no attribute {name!r}")


__all__ = [
    # Core types
    "SampleDistribution",
    "QuantileGridDistribution",
    "ParametricDistribution",
    "PredictiveDistribution",
    "Forecaster",
    # Main entry points
    "audit",
    "classify_regime",
    "compare_forecasters",
    # Result types
    "RegimeVerdict",
    "ComparisonResult",
    # Deprecated (still importable, emit warnings on call)
    "audit_static",
    "audit_rolling",
    "audit_with_benchmarks",
    "audit_panel",
    # Recalibration (loaded on first access)
    "RecalibrationMethod",
    "ConformalShift",
    "HistoricalQuantileRecalibration",
    "ScaleCorrectionRecalibration",
    "LinearQuantileRegression",
    "IsotonicQuantileRegression",
    "AdaptiveConformalInference",
    "GBMQuantileRegression",
    "ExtremeValueTheoryPOT",
    "FilteredHistoricalSimulation",
]
