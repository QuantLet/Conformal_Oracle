"""PanelResult dataclass for panel-level audit results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Union

import numpy as np
import pandas as pd

from conformal_oracle.audit.single_rolling import RollingAuditResult
from conformal_oracle.audit.single_static import StaticAuditResult

if TYPE_CHECKING:
    from conformal_oracle.panel.diagnostic_regression import (
        DiagnosticRegressionResult,
    )
    from conformal_oracle.panel.wildcluster_bootstrap import (
        WildClusterBootstrapResult,
    )

AuditResult = Union[StaticAuditResult, RollingAuditResult]


def _get_qv(r: AuditResult) -> float:
    if isinstance(r, StaticAuditResult):
        return r.q_v_stat
    return float(r.q_v_roll.mean())


def _get_ratio(r: AuditResult) -> float:
    if isinstance(r, StaticAuditResult):
        return r.replacement_ratio
    return float(r.replacement_ratio.mean())


@dataclass
class PanelResult:
    """Result of a panel-level audit across forecasters and assets."""

    results: dict[str, dict[str, AuditResult]]
    forecaster_names: list[str]
    asset_names: list[str]
    alpha: float
    mode: Literal["static", "rolling"]
    returns: pd.DataFrame = field(repr=False)

    def master_table(self) -> pd.DataFrame:
        rows = []
        for fc_name in self.forecaster_names:
            for asset in self.asset_names:
                r = self.results[fc_name][asset]
                row = {
                    "forecaster": fc_name,
                    "asset": asset,
                    "regime": r.regime,
                    "q_v": _get_qv(r),
                    "R": _get_ratio(r),
                    "pi_raw": r.violation_rate_raw,
                    "pi_corrected": r.violation_rate_corrected,
                    "basel_raw": r.basel_zone_raw,
                    "basel_corrected": r.basel_zone_corrected,
                    "kupiec_p": (
                        r.kupiec_pvalue_corrected
                        if isinstance(r, RollingAuditResult)
                        else r.kupiec_pvalue_corrected
                    ),
                    "christoffersen_p": (
                        r.christoffersen_pvalue_corrected
                        if isinstance(r, RollingAuditResult)
                        else r.christoffersen_pvalue_corrected
                    ),
                    "qs_raw": r.quantile_score_raw,
                    "qs_corrected": r.quantile_score_corrected,
                    "fz_raw": r.fz_score_raw,
                    "fz_corrected": r.fz_score_corrected,
                }
                rows.append(row)
        return pd.DataFrame(rows)

    def regime_summary(self) -> pd.DataFrame:
        rows = []
        for fc_name in self.forecaster_names:
            regimes = [
                self.results[fc_name][a].regime
                for a in self.asset_names
            ]
            basels_raw = [
                self.results[fc_name][a].basel_zone_raw
                for a in self.asset_names
            ]
            basels_corr = [
                self.results[fc_name][a].basel_zone_corrected
                for a in self.asset_names
            ]
            n = len(self.asset_names)
            rows.append({
                "forecaster": fc_name,
                "n_signal_preserving": sum(
                    1 for r in regimes if r == "signal-preserving"
                ),
                "n_replacement": sum(
                    1 for r in regimes if r == "replacement"
                ),
                "frac_signal_preserving": (
                    sum(1 for r in regimes if r == "signal-preserving")
                    / n
                ),
                "green_raw": sum(
                    1 for b in basels_raw if b == "green"
                ),
                "yellow_raw": sum(
                    1 for b in basels_raw if b == "yellow"
                ),
                "red_raw": sum(
                    1 for b in basels_raw if b == "red"
                ),
                "green_corrected": sum(
                    1 for b in basels_corr if b == "green"
                ),
                "yellow_corrected": sum(
                    1 for b in basels_corr if b == "yellow"
                ),
                "red_corrected": sum(
                    1 for b in basels_corr if b == "red"
                ),
            })
        return pd.DataFrame(rows)

    def cross_sectional_corr(
        self,
        characteristics: list[str] | None = None,
    ) -> pd.DataFrame:
        from conformal_oracle.panel.cross_sectional import (
            compute_asset_characteristics,
            compute_cross_sectional_correlations,
        )

        qv_data = {}
        for fc_name in self.forecaster_names:
            qv_data[fc_name] = {
                asset: _get_qv(self.results[fc_name][asset])
                for asset in self.asset_names
            }
        qv_df = pd.DataFrame(qv_data).T
        qv_df.columns = self.asset_names

        char_df = compute_asset_characteristics(self.returns)
        if characteristics is not None:
            char_df = char_df[characteristics]

        return compute_cross_sectional_correlations(qv_df, char_df)

    def diagnostic_regression(self) -> "DiagnosticRegressionResult":
        from conformal_oracle.panel.diagnostic_regression import (
            fit_diagnostic_regression,
        )

        qv_vals = []
        pi_raw_vals = []
        qs_improve = []
        fc_ids = []
        asset_ids = []

        for fc_name in self.forecaster_names:
            for asset in self.asset_names:
                r = self.results[fc_name][asset]
                qv_vals.append(_get_qv(r))
                pi_raw_vals.append(r.violation_rate_raw)
                qs_improve.append(
                    r.quantile_score_raw - r.quantile_score_corrected
                )
                fc_ids.append(fc_name)
                asset_ids.append(asset)

        return fit_diagnostic_regression(
            qv_values=np.array(qv_vals),
            raw_violation_rates=np.array(pi_raw_vals),
            qs_improvement=np.array(qs_improve),
            forecasters=np.array(fc_ids),
            assets=np.array(asset_ids),
        )

    def diebold_mariano(
        self,
        baseline: str | None = None,
    ) -> pd.DataFrame:
        from conformal_oracle.panel.diebold_mariano import (
            panel_dm_test,
        )

        if baseline is None:
            baseline = self.forecaster_names[0]

        qs_seqs: dict[str, dict[str, np.ndarray]] = {}
        for fc_name in self.forecaster_names:
            qs_seqs[fc_name] = {}
            for asset in self.asset_names:
                r = self.results[fc_name][asset]
                qs_seqs[fc_name][asset] = r.qs_sequence_corrected

        rows = []
        for fc_name in self.forecaster_names:
            if fc_name == baseline:
                continue
            stat, pval = panel_dm_test(
                qs_seqs, baseline, fc_name,
            )
            rows.append({
                "forecaster": fc_name,
                "baseline": baseline,
                "dm_statistic": stat,
                "p_value": pval,
            })
        return pd.DataFrame(rows)

    def panel_kupiec(self) -> pd.DataFrame:
        from conformal_oracle.panel.kupiec_panel import (
            panel_kupiec_test,
        )

        rows = []
        for fc_name in self.forecaster_names:
            violations: dict[str, np.ndarray] = {}
            for asset in self.asset_names:
                r = self.results[fc_name][asset]
                realised = self.returns[asset].values
                if isinstance(r, StaticAuditResult):
                    n_test = r.n_test
                    var_corr = r.var_corrected.values
                    real_test = realised[-n_test:]
                    violations[asset] = (
                        real_test < -var_corr
                    ).astype(int)
                else:
                    var_corr = r.var_corrected.values
                    n_eval = len(var_corr)
                    real_test = realised[-n_eval:]
                    violations[asset] = (
                        real_test < -var_corr
                    ).astype(int)

            lr, p, per_asset = panel_kupiec_test(
                violations, self.alpha,
            )
            rows.append({
                "forecaster": fc_name,
                "lr_statistic": lr,
                "p_value": p,
                "n_assets": len(self.asset_names),
                **{
                    f"pi_{a}": v
                    for a, v in per_asset.items()
                },
            })
        return pd.DataFrame(rows)

    def wild_cluster_bootstrap(
        self,
        B: int = 999,
        seed: int = 2026,
    ) -> "WildClusterBootstrapResult":
        from conformal_oracle.panel.wildcluster_bootstrap import (
            WildClusterBootstrapResult,
            wild_cluster_bootstrap_dm,
            wild_cluster_bootstrap_kupiec,
        )

        kupiec_rows = []
        all_violations: dict[str, dict[str, np.ndarray]] = {}
        for fc_name in self.forecaster_names:
            violations: dict[str, np.ndarray] = {}
            for asset in self.asset_names:
                r = self.results[fc_name][asset]
                realised = self.returns[asset].values
                if isinstance(r, StaticAuditResult):
                    n_test = r.n_test
                    var_corr = r.var_corrected.values
                    real_test = realised[-n_test:]
                else:
                    var_corr = r.var_corrected.values
                    n_eval = len(var_corr)
                    real_test = realised[-n_eval:]
                violations[asset] = (
                    real_test < -var_corr
                ).astype(int)
            all_violations[fc_name] = violations

            boot_result = wild_cluster_bootstrap_kupiec(
                violations, self.alpha, B=B, seed=seed,
            )
            kupiec_rows.append({
                "forecaster": fc_name,
                **boot_result,
            })

        kupiec_table = pd.DataFrame(kupiec_rows)

        qs_seqs: dict[str, dict[str, np.ndarray]] = {}
        for fc_name in self.forecaster_names:
            qs_seqs[fc_name] = {}
            for asset in self.asset_names:
                r = self.results[fc_name][asset]
                qs_seqs[fc_name][asset] = r.qs_sequence_corrected

        qs_diffs: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        baseline = self.forecaster_names[0]
        for fc_name in self.forecaster_names[1:]:
            pair_diffs: dict[str, np.ndarray] = {}
            for asset in self.asset_names:
                seq_a = qs_seqs[baseline][asset]
                seq_b = qs_seqs[fc_name][asset]
                min_len = min(len(seq_a), len(seq_b))
                pair_diffs[asset] = (
                    seq_a[-min_len:] - seq_b[-min_len:]
                )
            qs_diffs[(baseline, fc_name)] = pair_diffs

        dm_boot = wild_cluster_bootstrap_dm(
            qs_diffs, B=B, seed=seed,
        )
        dm_rows = []
        for (fc_a, fc_b), pval in dm_boot.items():
            dm_rows.append({
                "baseline": fc_a,
                "forecaster": fc_b,
                "bootstrap_p": pval,
            })
        dm_table = pd.DataFrame(dm_rows)

        return WildClusterBootstrapResult(
            kupiec_table=kupiec_table,
            dm_table=dm_table,
            n_bootstrap=B,
            n_clusters=len(self.asset_names),
            seed=seed,
        )

    def master_table_latex(self) -> str:
        from conformal_oracle.panel.latex import (
            master_table_to_latex,
        )
        return master_table_to_latex(self)

    def regime_summary_latex(self) -> str:
        from conformal_oracle.panel.latex import (
            regime_summary_to_latex,
        )
        return regime_summary_to_latex(self)

    def cross_sectional_corr_latex(self) -> str:
        from conformal_oracle.panel.latex import (
            cross_sectional_corr_to_latex,
        )
        return cross_sectional_corr_to_latex(
            self.cross_sectional_corr()
        )

    def diebold_mariano_latex(
        self,
        baseline: str | None = None,
    ) -> str:
        from conformal_oracle.panel.latex import (
            diebold_mariano_to_latex,
        )
        return diebold_mariano_to_latex(
            self.diebold_mariano(baseline=baseline)
        )

    def to_dict(self) -> dict:
        d: dict = {
            "forecasters": self.forecaster_names,
            "assets": self.asset_names,
            "alpha": self.alpha,
            "mode": self.mode,
            "results": {},
        }
        for fc_name in self.forecaster_names:
            d["results"][fc_name] = {}
            for asset in self.asset_names:
                r = self.results[fc_name][asset]
                d["results"][fc_name][asset] = r.to_dict()
        return d
