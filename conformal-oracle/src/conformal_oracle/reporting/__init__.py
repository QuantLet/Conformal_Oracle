"""Reporting utilities (LaTeX, plots)."""

from conformal_oracle.reporting.latex import (
    audit_result_to_latex_row,
    comparison_to_latex,
)
from conformal_oracle.reporting.plotting import plot_rolling_diagnostic

__all__ = [
    "audit_result_to_latex_row",
    "comparison_to_latex",
    "plot_rolling_diagnostic",
]
