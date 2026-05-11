"""Static conformal audit pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from conformal_oracle._protocols import Forecaster
from conformal_oracle._types import PredictiveDistribution
from conformal_oracle.audit.regime import classify_regime_static
from conformal_oracle.conformal.bootstrap import bootstrap_qv_ci
from conformal_oracle.diagnostics.acerbi_szekely import z2_statistic
from conformal_oracle.diagnostics.basel import basel_traffic_light
from conformal_oracle.diagnostics.christoffersen import christoffersen_pvalue
from conformal_oracle.diagnostics.diebold_mariano import quantile_score_sequence
from conformal_oracle.diagnostics.kupiec import kupiec_pof_pvalue
from conformal_oracle.diagnostics.scoring import fissler_ziegel_fz0, quantile_score


@dataclass
class StaticAuditResult:
    """Result of a static conformal audit."""

    regime: Literal["signal-preserving", "replacement"]
    replacement_ratio: float

    q_v_stat: float
    q_v_stat_ci: tuple[float, float]
    var_corrected: pd.Series

    violation_rate_raw: float
    violation_rate_corrected: float
    basel_zone_raw: Literal["green", "yellow", "red"]
    basel_zone_corrected: Literal["green", "yellow", "red"]

    kupiec_pvalue_raw: float
    kupiec_pvalue_corrected: float
    christoffersen_pvalue_raw: float
    christoffersen_pvalue_corrected: float
    z2_statistic_raw: float
    z2_statistic_corrected: float

    quantile_score_raw: float
    quantile_score_corrected: float
    fz_score_raw: float
    fz_score_corrected: float

    qs_sequence_raw: np.ndarray
    qs_sequence_corrected: np.ndarray

    alpha: float
    calibration_split: float
    n_calibration: int
    n_test: int
    mode: Literal["static"] = "static"

    def summary(self) -> str:
        ci_lo = self.q_v_stat_ci[0]
        ci_hi = self.q_v_stat_ci[1]
        lines = [
            f"=== Static Conformal Audit (alpha={self.alpha}) ===",
            f"Regime: {self.regime} (R = {self.replacement_ratio:.3f})",
            f"qV_stat: {self.q_v_stat:.6f}  CI: [{ci_lo:.6f}, {ci_hi:.6f}]",
            "",
            (f"Coverage (raw):       {self.violation_rate_raw:.4f}"
             f"  Basel: {self.basel_zone_raw}"),
            (f"Coverage (corrected): {self.violation_rate_corrected:.4f}"
             f"  Basel: {self.basel_zone_corrected}"),
            "",
            (f"Kupiec p:     raw={self.kupiec_pvalue_raw:.4f}"
             f"  corrected={self.kupiec_pvalue_corrected:.4f}"),
            (f"Chr. p:       raw={self.christoffersen_pvalue_raw:.4f}"
             f"  corrected={self.christoffersen_pvalue_corrected:.4f}"),
            (f"Z2:           raw={self.z2_statistic_raw:.4f}"
             f"  corrected={self.z2_statistic_corrected:.4f}"),
            (f"QS:           raw={self.quantile_score_raw:.6f}"
             f"  corrected={self.quantile_score_corrected:.6f}"),
            (f"FZ:           raw={self.fz_score_raw:.6f}"
             f"  corrected={self.fz_score_corrected:.6f}"),
            "",
            (f"Calibration: {self.n_calibration} obs"
             f" | Test: {self.n_test} obs"),
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, (pd.Series, np.ndarray)):
                continue
            elif isinstance(v, tuple):
                d[f"{k}_lo"] = v[0]
                d[f"{k}_hi"] = v[1]
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
            f"{self.q_v_stat:.4f} & "
            f"{self.replacement_ratio:.3f} & "
            f"{self.regime} \\\\"
        )


def audit_static(
    returns: pd.Series,
    forecaster: Forecaster,
    alpha: float = 0.01,
    calibration_split: float = 0.70,
    warmup: int = 50,
    seed: int = 2026,
    recalibration: object | None = None,
) -> StaticAuditResult:
    """Run the static conformal audit pipeline.

    Args:
        warmup: Minimum observations before the first valid forecast.
                Forecasts at t < warmup are skipped during calibration
                score computation.
        recalibration: Optional RecalibrationMethod to use instead of
                       the built-in conformal shift. If None (default),
                       the conformal shift is used.
    """
    n = len(returns)
    n_cal = int(n * calibration_split)
    n_test = n - n_cal

    cal_returns = returns.iloc[:n_cal]
    test_returns = returns.iloc[n_cal:]

    forecaster.fit(cal_returns)

    warmup_start = min(warmup, n_cal // 4)
    cal_forecasts: list[PredictiveDistribution] = []
    for t in range(warmup_start, n_cal):
        cal_forecasts.append(forecaster.forecast(returns, t))

    cal_realised = cal_returns.values[warmup_start:]
    cal_var_raw = np.array([-f.quantile(alpha) for f in cal_forecasts])
    cal_scores = np.array([
        f.quantile(alpha) - r for f, r in zip(cal_forecasts, cal_realised)
    ])
    q_v_stat = float(np.quantile(cal_scores, 1 - alpha))

    ci = bootstrap_qv_ci(cal_scores, alpha, seed=seed)

    test_forecasts: list[PredictiveDistribution] = []
    var_raw = np.empty(n_test)
    es_raw = np.empty(n_test)

    for i, t in enumerate(range(n_cal, n)):
        fc = forecaster.forecast(returns, t)
        test_forecasts.append(fc)
        var_raw[i] = -fc.quantile(alpha)
        es_raw[i] = -fc.expected_shortfall(alpha)

    if recalibration is not None:
        recalibration.fit(cal_var_raw, cal_realised, alpha)
        var_corrected = recalibration.apply(var_raw)
    else:
        var_corrected = var_raw + q_v_stat

    es_corrected = es_raw + (var_corrected - var_raw)

    test_realised = test_returns.values
    viol_raw = (test_realised < -var_raw).astype(int)
    viol_corrected = (test_realised < -var_corrected).astype(int)

    regime, ratio = classify_regime_static(q_v_stat, -var_raw)

    chris_raw = christoffersen_pvalue(viol_raw, alpha)
    chris_corr = christoffersen_pvalue(viol_corrected, alpha)

    var_corrected_series = pd.Series(
        var_corrected, index=test_returns.index, name="VaR_corrected"
    )

    return StaticAuditResult(
        regime=regime,
        replacement_ratio=ratio,
        q_v_stat=q_v_stat,
        q_v_stat_ci=ci,
        var_corrected=var_corrected_series,
        violation_rate_raw=float(np.mean(viol_raw)),
        violation_rate_corrected=float(np.mean(viol_corrected)),
        basel_zone_raw=basel_traffic_light(viol_raw),
        basel_zone_corrected=basel_traffic_light(viol_corrected),
        kupiec_pvalue_raw=kupiec_pof_pvalue(viol_raw, alpha),
        kupiec_pvalue_corrected=kupiec_pof_pvalue(viol_corrected, alpha),
        christoffersen_pvalue_raw=chris_raw["joint"],
        christoffersen_pvalue_corrected=chris_corr["joint"],
        z2_statistic_raw=z2_statistic(viol_raw, test_realised, es_raw, alpha),
        z2_statistic_corrected=z2_statistic(
            viol_corrected, test_realised, es_corrected, alpha
        ),
        quantile_score_raw=quantile_score(test_realised, -var_raw, alpha),
        quantile_score_corrected=quantile_score(
            test_realised, -var_corrected, alpha
        ),
        fz_score_raw=fissler_ziegel_fz0(
            test_realised, -var_raw, -es_raw, alpha
        ),
        fz_score_corrected=fissler_ziegel_fz0(
            test_realised, -var_corrected, -es_corrected, alpha
        ),
        qs_sequence_raw=quantile_score_sequence(test_realised, -var_raw, alpha),
        qs_sequence_corrected=quantile_score_sequence(
            test_realised, -var_corrected, alpha
        ),
        alpha=alpha,
        calibration_split=calibration_split,
        n_calibration=n_cal,
        n_test=n_test,
    )
