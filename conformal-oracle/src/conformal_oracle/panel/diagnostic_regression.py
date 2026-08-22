"""Diagnostic regression: Delta_QS on qV_stat and pi_raw."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class DiagnosticRegressionResult:
    """Result of the diagnostic regression eq:dqs_regression."""

    coefficients: pd.Series
    se_ols: pd.Series
    se_cluster_asset: pd.Series
    se_cluster_model: pd.Series
    r_squared: float
    partial_r_squared_qv: float
    n_obs: int

    def summary(self) -> str:
        lines = [
            "=== Diagnostic Regression ===",
            (f"Delta_QS = {self.coefficients.iloc[0]:.6f}"
             f" + {self.coefficients.iloc[1]:.6f} * qV"
             f" + {self.coefficients.iloc[2]:.6f} * pi_raw"),
            "",
            f"R^2:            {self.r_squared:.4f}",
            f"Partial R^2 qV: {self.partial_r_squared_qv:.4f}",
            f"N:              {self.n_obs}",
            "",
            "            coef     OLS_SE   Clust_A  Clust_M",
        ]
        for name in self.coefficients.index:
            lines.append(
                f"  {name:10s}"
                f" {self.coefficients[name]:9.6f}"
                f" {self.se_ols[name]:8.6f}"
                f" {self.se_cluster_asset[name]:8.6f}"
                f" {self.se_cluster_model[name]:8.6f}"
            )
        return "\n".join(lines)

    def to_latex(self) -> str:
        header = (
            "\\begin{tabular}{l rrrr}\n"
            "\\toprule\n"
            " & Coef. & OLS SE & Cluster(asset) "
            "& Cluster(model) \\\\\n"
            "\\midrule"
        )
        rows = []
        for name in self.coefficients.index:
            rows.append(
                f"{name} & "
                f"{self.coefficients[name]:.4f} & "
                f"({self.se_ols[name]:.4f}) & "
                f"({self.se_cluster_asset[name]:.4f}) & "
                f"({self.se_cluster_model[name]:.4f}) \\\\"
            )
        footer = (
            "\\midrule\n"
            f"$R^2$ & {self.r_squared:.4f} & & & \\\\\n"
            f"Partial $R^2(\\hat{{q}}_V)$ & "
            f"{self.partial_r_squared_qv:.4f} & & & \\\\\n"
            f"$N$ & {self.n_obs} & & & \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}"
        )
        body = "\n".join(rows)
        return f"{header}\n{body}\n{footer}"


def _cluster_se(
    resid: np.ndarray,
    X: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    """Cluster-robust standard errors (Liang-Zeger)."""
    unique = np.unique(clusters)
    J = len(unique)
    n, k = X.shape

    XtX_inv = np.linalg.inv(X.T @ X)
    meat = np.zeros((k, k))
    for c in unique:
        mask = clusters == c
        Xc = X[mask]
        ec = resid[mask]
        score = Xc.T @ ec
        meat += np.outer(score, score)

    scale = J / (J - 1) * (n - 1) / (n - k)
    V = XtX_inv @ meat @ XtX_inv * scale
    return np.sqrt(np.diag(V))


def fit_diagnostic_regression(
    qv_values: np.ndarray,
    raw_violation_rates: np.ndarray,
    qs_improvement: np.ndarray,
    forecasters: np.ndarray,
    assets: np.ndarray,
) -> DiagnosticRegressionResult:
    """Fit: Delta_QS = beta_0 + beta_1*qV + beta_2*pi_raw + eps.

    Partial R^2 for qV via Frisch-Waugh-Lovell.
    """
    y = qs_improvement
    X = sm.add_constant(
        np.column_stack([qv_values, raw_violation_rates])
    )
    ols = sm.OLS(y, X).fit()

    coef = pd.Series(
        ols.params,
        index=["const", "qV", "pi_raw"],
    )
    se_ols = pd.Series(
        ols.bse,
        index=["const", "qV", "pi_raw"],
    )

    resid = ols.resid
    se_ca = pd.Series(
        _cluster_se(resid, X, assets),
        index=["const", "qV", "pi_raw"],
    )
    se_cm = pd.Series(
        _cluster_se(resid, X, forecasters),
        index=["const", "qV", "pi_raw"],
    )

    # Frisch-Waugh-Lovell partial R^2 for qV
    X_partial = sm.add_constant(raw_violation_rates)
    y_res = sm.OLS(y, X_partial).fit().resid
    qv_res = sm.OLS(qv_values, X_partial).fit().resid
    if np.var(qv_res) > 0 and np.var(y_res) > 0:
        partial_r2 = float(
            np.corrcoef(y_res, qv_res)[0, 1] ** 2
        )
    else:
        partial_r2 = 0.0

    return DiagnosticRegressionResult(
        coefficients=coef,
        se_ols=se_ols,
        se_cluster_asset=se_ca,
        se_cluster_model=se_cm,
        r_squared=float(ols.rsquared),
        partial_r_squared_qv=partial_r2,
        n_obs=len(y),
    )
