"""Benchmark comparison: audit user forecaster alongside reference models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import pandas as pd

from conformal_oracle._protocols import Forecaster
from conformal_oracle.audit.single_rolling import RollingAuditResult, audit_rolling
from conformal_oracle.audit.single_static import StaticAuditResult, audit_static
from conformal_oracle.diagnostics.diebold_mariano import diebold_mariano_pvalue
from conformal_oracle.forecasters import (
    GARCHNormalForecaster,
    GJRGARCHForecaster,
    HistoricalSimulationForecaster,
)
from conformal_oracle.reporting.latex import comparison_to_latex

_BENCHMARK_REGISTRY: dict[str, type] = {
    "gjr_garch": GJRGARCHForecaster,
    "garch_normal": GARCHNormalForecaster,
    "hist_sim": HistoricalSimulationForecaster,
}


@dataclass
class BenchmarkComparison:
    """Comparison of user forecaster against reference benchmarks."""

    user: Union[StaticAuditResult, RollingAuditResult]
    benchmarks: dict[str, Union[StaticAuditResult, RollingAuditResult]]
    mode: Literal["static", "rolling"]

    def comparison_table(self) -> pd.DataFrame:
        rows = {"user": self.user.to_dict()}
        for name, result in self.benchmarks.items():
            rows[name] = result.to_dict()
        return pd.DataFrame(rows).T

    def comparison_table_latex(
        self,
        caption: str = "Conformal audit comparison",
        label: str = "tab:audit",
    ) -> str:
        """Full LaTeX table sorted by replacement ratio ascending."""
        all_results: dict[str, Union[StaticAuditResult, RollingAuditResult]] = {
            "User": self.user,
            **self.benchmarks,
        }
        return comparison_to_latex(all_results, caption=caption, label=label)

    def diebold_mariano(
        self, baseline: str = "gjr_garch"
    ) -> dict[str, float]:
        """DM p-values comparing each forecaster's QS against baseline.

        Uses Newey-West HAC variance estimator and Harvey-Leybourne-Newbold
        small-sample correction on per-step quantile score sequences.
        """
        if baseline not in self.benchmarks:
            raise ValueError(f"Baseline '{baseline}' not in benchmarks")
        base_qs_seq = self.benchmarks[baseline].qs_sequence_corrected
        results = {}
        all_items: dict[str, Union[StaticAuditResult, RollingAuditResult]] = {
            "user": self.user, **self.benchmarks
        }
        for name, res in all_items.items():
            if name == baseline:
                continue
            qs_seq = res.qs_sequence_corrected
            n = min(len(qs_seq), len(base_qs_seq))
            results[name] = diebold_mariano_pvalue(
                qs_seq[:n], base_qs_seq[:n], horizon=1, hln_correction=True
            )
        return results


def audit_with_benchmarks(
    returns: pd.Series,
    forecaster: Forecaster,
    benchmarks: list[str] | None = None,
    recalibrations: list[object] | None = None,
    alpha: float = 0.01,
    mode: Literal["static", "rolling"] = "rolling",
    seed: int = 2026,
    **mode_kwargs: object,
) -> BenchmarkComparison:
    """Audit user's forecaster alongside reference benchmarks.

    Args:
        recalibrations: Optional list of RecalibrationMethod instances.
            When provided, each (base_forecaster, recalibration) combination
            is audited. Default (None) uses the conformal shift only.
    """
    if benchmarks is None:
        benchmarks = ["gjr_garch", "hist_sim"]

    for name in benchmarks:
        if name not in _BENCHMARK_REGISTRY:
            raise ValueError(
                f"Unknown benchmark '{name}'. "
                f"Available: {list(_BENCHMARK_REGISTRY.keys())}"
            )

    audit_fn = audit_static if mode == "static" else audit_rolling

    if recalibrations is None:
        user_result = audit_fn(
            returns, forecaster, alpha=alpha, seed=seed, **mode_kwargs,
        )
        bench_results: dict[str, Union[StaticAuditResult, RollingAuditResult]] = {}
        for name in benchmarks:
            bench_fc = _BENCHMARK_REGISTRY[name]()
            bench_results[name] = audit_fn(
                returns, bench_fc, alpha=alpha, seed=seed, **mode_kwargs,
            )
    else:
        user_result = audit_fn(
            returns, forecaster, alpha=alpha, seed=seed, **mode_kwargs,
        )
        bench_results = {}

        all_forecasters: dict[str, Forecaster] = {"user": forecaster}
        for name in benchmarks:
            all_forecasters[name] = _BENCHMARK_REGISTRY[name]()

        for recal in recalibrations:
            recal_name = type(recal).__name__
            for fc_name, fc in all_forecasters.items():
                label = f"{fc_name}+{recal_name}"
                bench_results[label] = audit_fn(
                    returns, fc, alpha=alpha, seed=seed,
                    recalibration=recal, **mode_kwargs,
                )

    return BenchmarkComparison(
        user=user_result,
        benchmarks=bench_results,
        mode=mode,
    )
