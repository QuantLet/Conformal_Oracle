"""Recalibration baselines for post-hoc VaR correction."""

from conformal_oracle.recalibration.aci import (
    ACICalibrator,
    AdaptiveConformalInference,
)
from conformal_oracle.recalibration.base import ConformalShift, RecalibrationMethod
from conformal_oracle.recalibration.evt_pot import ExtremeValueTheoryPOT
from conformal_oracle.recalibration.fhs import FilteredHistoricalSimulation
from conformal_oracle.recalibration.gbm_qr import GBMQuantileRegression
from conformal_oracle.recalibration.historical_quantile import (
    HistoricalQuantileRecalibration,
)
from conformal_oracle.recalibration.quantile_regression import (
    IsotonicQuantileRegression,
    LinearQuantileRegression,
)
from conformal_oracle.recalibration.scale_correction import (
    ScaleCorrectionRecalibration,
)
from conformal_oracle.recalibration.scale_diagnostic import (
    ScaleDiagnostic,
    diagnose_scale,
)

__all__ = [
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
