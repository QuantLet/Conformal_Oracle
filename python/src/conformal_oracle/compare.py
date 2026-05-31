"""First-class forecaster comparison entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

import pandas as pd

from conformal_oracle._protocols import Forecaster
from conformal_oracle.audit.single_rolling import RollingAuditResult
from conformal_oracle.audit.single_static import StaticAuditResult
from conformal_oracle.diagnostics.diebold_mariano import diebold_mariano_pvalue


@dataclass
class ComparisonResult:
    """Result of comparing multiple forecasters or quantile paths."""

    results: dict[str, Union[StaticAuditResult, RollingAuditResult]]
    mode: Literal["static", "rolling"]
    dm_pvalues: dict[tuple[str, str], float] = field(default_factory=dict)

    def comparison_table(self) -> pd.DataFrame:
        """DataFrame with one row per forecaster."""
        rows = {}
        for name, res in self.results.items():
            rows[name] = res.to_dict()
        return pd.DataFrame(rows).T

    def dm_matrix(self) -> pd.DataFrame:
        """Pairwise DM p-value matrix."""
        names = sorted(self.results.keys())
        matrix: dict[str, dict[str, float]] = {
            a: {b: float("nan") for b in names} for a in names
        }
        for (a, b), p in self.dm_pvalues.items():
            matrix[a][b] = p
        return pd.DataFrame(matrix, index=names, columns=names)


def compare_forecasters(
    returns: pd.Series,
    forecasts: dict[str, pd.Series] | dict[str, Forecaster] | None = None,
    *,
    alpha: float = 0.01,
    mode: Literal["static", "rolling"] = "rolling",
    test: Literal["dm_hac"] = "dm_hac",
    **kwargs: object,
) -> ComparisonResult:
    """Compare multiple forecasters or pre-computed quantile paths.

    Parameters
    ----------
    forecasts : dict
        Mapping from name to either a ``pd.Series`` (quantile path)
        or a ``Forecaster`` instance.
    alpha : float
        Tail probability level.
    mode : {"static", "rolling"}
        Audit mode.
    test : {"dm_hac"}
        Pairwise comparison test.
    **kwargs
        Forwarded to :func:`audit`.
    """
    from conformal_oracle.audit import audit

    if forecasts is None or len(forecasts) < 2:
        raise ValueError("Supply at least two forecasters/forecasts to compare.")

    results: dict[str, Union[StaticAuditResult, RollingAuditResult]] = {}
    for name, fc_or_series in forecasts.items():
        if isinstance(fc_or_series, pd.Series):
            results[name] = audit(
                returns, forecast=fc_or_series,
                alpha=alpha, mode=mode, **kwargs,
            )
        else:
            results[name] = audit(
                returns, forecaster=fc_or_series,
                alpha=alpha, mode=mode, **kwargs,
            )

    # Pairwise DM tests
    dm_pvalues: dict[tuple[str, str], float] = {}
    names = sorted(results.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            qs_a = results[a].qs_sequence_corrected
            qs_b = results[b].qs_sequence_corrected
            n = min(len(qs_a), len(qs_b))
            p = diebold_mariano_pvalue(qs_a[:n], qs_b[:n], horizon=1)
            dm_pvalues[(a, b)] = p
            dm_pvalues[(b, a)] = p

    return ComparisonResult(
        results=results,
        mode=mode,
        dm_pvalues=dm_pvalues,
    )
