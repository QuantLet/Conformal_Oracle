"""Rolling conformal audit pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from conformal_oracle._protocols import Forecaster
from conformal_oracle._types import PredictiveDistribution
from conformal_oracle.audit.regime import classify_regime_rolling
from conformal_oracle.conformal.rolling import (
    compute_drift_diagnostic,
    compute_qv_roll_from_scores,
)
from conformal_oracle.diagnostics.acerbi_szekely import z2_statistic
from conformal_oracle.diagnostics.basel import basel_traffic_light
from conformal_oracle.diagnostics.christoffersen import christoffersen_pvalue
from conformal_oracle.diagnostics.diebold_mariano import quantile_score_sequence
from conformal_oracle.diagnostics.kupiec import kupiec_pof_pvalue
from conformal_oracle.diagnostics.scoring import fissler_ziegel_fz0, quantile_score


@dataclass
class RollingAuditResult:
    """Result of a rolling conformal audit."""

    regime: Literal["signal-preserving", "replacement"]

    q_v_roll: pd.Series
    replacement_ratio: pd.Series
    drift_diagnostic: pd.Series
    var_corrected: pd.Series

    violation_rate_raw: float
    violation_rate_corrected: float
    basel_zone_raw: Literal["green", "yellow", "red"]
    basel_zone_corrected: Literal["green", "yellow", "red"]

    kupiec_pvalue_corrected: float
    christoffersen_pvalue_corrected: float
    z2_statistic_corrected: float

    quantile_score_raw: float
    quantile_score_corrected: float
    fz_score_raw: float
    fz_score_corrected: float

    qs_sequence_raw: np.ndarray
    qs_sequence_corrected: np.ndarray

    alpha: float
    window: int
    warmup: int
    n_test: int
    mode: Literal["rolling"] = "rolling"

    def summary(self) -> str:
        qv_m = self.q_v_roll.mean()
        qv_s = self.q_v_roll.std()
        lines = [
            (f"=== Rolling Conformal Audit"
             f" (alpha={self.alpha}, window={self.window}) ==="),
            f"Regime: {self.regime}",
            f"qV_roll mean: {qv_m:.6f}  std: {qv_s:.6f}",
            f"Replacement ratio mean: {self.replacement_ratio.mean():.3f}",
            f"Drift diagnostic mean: {self.drift_diagnostic.mean():.4f}",
            "",
            (f"Coverage (raw):       {self.violation_rate_raw:.4f}"
             f"  Basel: {self.basel_zone_raw}"),
            (f"Coverage (corrected): {self.violation_rate_corrected:.4f}"
             f"  Basel: {self.basel_zone_corrected}"),
            "",
            (f"Kupiec p (corrected):  "
             f"{self.kupiec_pvalue_corrected:.4f}"),
            (f"Chr. p (corrected):    "
             f"{self.christoffersen_pvalue_corrected:.4f}"),
            (f"Z2 (corrected):        "
             f"{self.z2_statistic_corrected:.4f}"),
            (f"QS:  raw={self.quantile_score_raw:.6f}"
             f"  corrected={self.quantile_score_corrected:.6f}"),
            (f"FZ:  raw={self.fz_score_raw:.6f}"
             f"  corrected={self.fz_score_corrected:.6f}"),
            "",
            f"Test observations: {self.n_test}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (pd.Series, np.ndarray)):
                if isinstance(v, pd.Series):
                    d[f"{k}_mean"] = float(v.mean())
                    d[f"{k}_std"] = float(v.std())
                continue
            else:
                d[k] = v
        return d

    def to_latex_row(self, name: str) -> str:
        return (
            f"{name} & "
            f"{self.violation_rate_corrected:.3f} & "
            f"{self.kupiec_pvalue_corrected:.3f} & "
            f"{self.christoffersen_pvalue_corrected:.3f} & "
            f"{self.basel_zone_corrected} & "
            f"{self.quantile_score_corrected:.4f} & "
            f"{self.fz_score_corrected:.4f} & "
            f"{self.q_v_roll.mean():.4f} & "
            f"{self.replacement_ratio.mean():.3f} & "
            f"{self.regime} \\\\"
        )


def audit_rolling(
    returns: pd.Series,
    forecaster: Forecaster,
    alpha: float = 0.01,
    window: int = 250,
    warmup: int = 250,
    persistence: int = 20,
    seed: int = 2026,
    recalibration: object | None = None,
) -> RollingAuditResult:
    """Run the rolling conformal audit pipeline.

    Args:
        recalibration: Optional RecalibrationMethod to use instead of
                       the rolling conformal shift. If provided, the
                       method is fit on a calibration window and applied
                       to the evaluation period.
    """
    n = len(returns)
    forecaster.fit(returns.iloc[:warmup])

    all_forecasts: list[PredictiveDistribution] = []
    var_raw_all = []
    es_raw_all = []
    for t in range(warmup, n):
        fc = forecaster.forecast(returns, t)
        all_forecasts.append(fc)
        var_raw_all.append(-fc.quantile(alpha))
        es_raw_all.append(-fc.expected_shortfall(alpha))

    n_fc = len(all_forecasts)
    realised_all = returns.iloc[warmup:].values

    scores = np.array([
        fc.quantile(alpha) - r
        for fc, r in zip(all_forecasts, realised_all)
    ])

    qv_roll = compute_qv_roll_from_scores(scores, alpha, window)
    drift = compute_drift_diagnostic(scores, window)

    n_eval = len(qv_roll)
    offset = n_fc - n_eval

    var_raw_eval = np.array(var_raw_all[offset:])
    es_raw_eval = np.array(es_raw_all[offset:])
    realised_eval = realised_all[offset:]

    if recalibration is not None:
        cal_var = np.array(var_raw_all[:offset])
        cal_real = realised_all[:offset]
        recalibration.fit(cal_var, cal_real, alpha)
        var_corrected_eval = recalibration.apply(var_raw_eval)
    else:
        var_corrected_eval = var_raw_eval + qv_roll

    es_corrected_eval = es_raw_eval + (var_corrected_eval - var_raw_eval)

    viol_raw = (realised_eval < -var_raw_eval).astype(int)
    viol_corrected = (realised_eval < -var_corrected_eval).astype(int)

    repl_ratio = np.abs(qv_roll) / (np.abs(var_raw_eval) + 1e-12)
    repl_ratio_series = pd.Series(
        repl_ratio,
        index=returns.index[warmup + offset : warmup + offset + n_eval],
        name="replacement_ratio",
    )

    regime = classify_regime_rolling(repl_ratio_series, persistence=persistence)

    idx = returns.index[warmup + offset : warmup + offset + n_eval]

    chris_corr = christoffersen_pvalue(viol_corrected, alpha)

    return RollingAuditResult(
        regime=regime,
        q_v_roll=pd.Series(qv_roll, index=idx, name="qV_roll"),
        replacement_ratio=repl_ratio_series,
        drift_diagnostic=pd.Series(drift, index=idx, name="drift_TV"),
        var_corrected=pd.Series(var_corrected_eval, index=idx, name="VaR_corrected"),
        violation_rate_raw=float(np.mean(viol_raw)),
        violation_rate_corrected=float(np.mean(viol_corrected)),
        basel_zone_raw=basel_traffic_light(viol_raw),
        basel_zone_corrected=basel_traffic_light(viol_corrected),
        kupiec_pvalue_corrected=kupiec_pof_pvalue(viol_corrected, alpha),
        christoffersen_pvalue_corrected=chris_corr["joint"],
        z2_statistic_corrected=z2_statistic(
            viol_corrected, realised_eval, es_corrected_eval, alpha
        ),
        quantile_score_raw=quantile_score(realised_eval, -var_raw_eval, alpha),
        quantile_score_corrected=quantile_score(
            realised_eval, -var_corrected_eval, alpha
        ),
        fz_score_raw=fissler_ziegel_fz0(
            realised_eval, -var_raw_eval, -es_raw_eval, alpha
        ),
        fz_score_corrected=fissler_ziegel_fz0(
            realised_eval, -var_corrected_eval, -es_corrected_eval, alpha
        ),
        qs_sequence_raw=quantile_score_sequence(
            realised_eval, -var_raw_eval, alpha
        ),
        qs_sequence_corrected=quantile_score_sequence(
            realised_eval, -var_corrected_eval, alpha
        ),
        alpha=alpha,
        window=window,
        warmup=warmup,
        n_test=n_eval,
    )
