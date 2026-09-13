"""Conformal recalibration and backtesting for extreme financial quantiles."""

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
from conformal_oracle.conformal.separated import (
    GapResult,
    SeparatedSplitConformalVaR,
    SeparatedSplitResult,
    proxy_separation_gap,
)
from conformal_oracle.deployment import (
    PastLossSelection,
    RecalibrationDecision,
    SelectiveRecalibrationResult,
    past_loss_selection,
    recalibration_indication,
    selectively_recalibrate,
)
from conformal_oracle.diagnostics.optimism import (
    OptimismEstimate,
    ShrinkageDiagnostic,
    block_bootstrap_optimism,
    blocked_cv_optimism,
    first_order_shrinkage,
)
from conformal_oracle.diagnostics.paired_bootstrap import paired_calendar_bootstrap
from conformal_oracle.recalibration.one_coefficient import (
    OneCoefficientCorrections,
    fit_one_coefficient_corrections,
)

if TYPE_CHECKING:
    from conformal_oracle.recalibration import (
        ACICalibrator,
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
        ScaleDiagnostic,
        diagnose_scale,
    )

__version__ = "0.4.0"

_RECALIBRATION_NAMES = {
    "RecalibrationMethod",
    "ConformalShift",
    "HistoricalQuantileRecalibration",
    "ScaleCorrectionRecalibration",
    "LinearQuantileRegression",
    "IsotonicQuantileRegression",
    "AdaptiveConformalInference",
    "ACICalibrator",
    "GBMQuantileRegression",
    "ExtremeValueTheoryPOT",
    "FilteredHistoricalSimulation",
    "ScaleDiagnostic",
    "diagnose_scale",
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
    "SeparatedSplitConformalVaR",
    "proxy_separation_gap",
    "recalibration_indication",
    "selectively_recalibrate",
    "past_loss_selection",
    "fit_one_coefficient_corrections",
    "blocked_cv_optimism",
    "block_bootstrap_optimism",
    "first_order_shrinkage",
    "paired_calendar_bootstrap",
    # Result types
    "RegimeVerdict",
    "ComparisonResult",
    "SeparatedSplitResult",
    "GapResult",
    "RecalibrationDecision",
    "SelectiveRecalibrationResult",
    "PastLossSelection",
    "OneCoefficientCorrections",
    "OptimismEstimate",
    "ShrinkageDiagnostic",
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
    "ACICalibrator",
    "GBMQuantileRegression",
    "ExtremeValueTheoryPOT",
    "FilteredHistoricalSimulation",
    "ScaleDiagnostic",
    "diagnose_scale",
]
