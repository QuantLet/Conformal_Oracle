"""LaTeX table emitters matching the manuscript's column order.

Column order for tab:master (static mode):
  Model | pi_hat | Kupiec p | Chr. p | Basel | QS | FZ | qV | R | Regime

Column order for tab:rolling_vs_static (rolling mode):
  Model | pi_hat | Kupiec p | Chr. p | Basel | QS | FZ | mean(qV) | mean(R) | Regime
"""

from __future__ import annotations

from typing import Union

from conformal_oracle.audit.single_rolling import RollingAuditResult
from conformal_oracle.audit.single_static import StaticAuditResult


def audit_result_to_latex_row(
    result: Union[StaticAuditResult, RollingAuditResult],
    name: str,
) -> str:
    """One LaTeX table row matching the manuscript's column order."""
    if isinstance(result, StaticAuditResult):
        return _static_row(result, name)
    else:
        return _rolling_row(result, name)


def comparison_to_latex(
    results: dict[str, Union[StaticAuditResult, RollingAuditResult]],
    caption: str = "Conformal audit comparison",
    label: str = "tab:audit",
) -> str:
    """Full LaTeX table from a dict of named audit results.

    Results are sorted by replacement_ratio ascending, so
    signal-preserving models appear first.
    """
    is_static = any(isinstance(r, StaticAuditResult) for r in results.values())

    if is_static:
        header = _STATIC_HEADER
    else:
        header = _ROLLING_HEADER

    sorted_names = sorted(
        results.keys(),
        key=lambda n: (
            results[n].replacement_ratio
            if isinstance(results[n], StaticAuditResult)
            else results[n].replacement_ratio.mean()
        ),
    )

    rows = []
    for name in sorted_names:
        rows.append(audit_result_to_latex_row(results[name], name))

    body = "\n".join(rows)

    return (
        f"\\begin{{table}}[htbp]\n"
        f"\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\small\n"
        f"{header}\n"
        f"{body}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}"
    )


_STATIC_HEADER = (
    "\\begin{tabular}{l rr r rr rr l}\n"
    "\\toprule\n"
    "Model & $\\hat{\\pi}$ & Kupiec $p$ & Chr.~$p$ & Basel "
    "& QS & FZ & $\\hat{q}_V$ & $\\bar{R}$ & Regime \\\\\n"
    "\\midrule"
)

_ROLLING_HEADER = (
    "\\begin{tabular}{l rr r rr rr l}\n"
    "\\toprule\n"
    "Model & $\\hat{\\pi}$ & Kupiec $p$ & Chr.~$p$ & Basel "
    "& QS & FZ & $\\overline{q_V}$ & $\\bar{R}$ & Regime \\\\\n"
    "\\midrule"
)


def _static_row(r: StaticAuditResult, name: str) -> str:
    return (
        f"{name} & "
        f"{r.violation_rate_corrected:.3f} & "
        f"{r.kupiec_pvalue_corrected:.3f} & "
        f"{r.christoffersen_pvalue_corrected:.3f} & "
        f"{r.basel_zone_corrected} & "
        f"{r.quantile_score_corrected:.4f} & "
        f"{r.fz_score_corrected:.4f} & "
        f"{r.q_v_stat:.4f} & "
        f"{r.replacement_ratio:.3f} & "
        f"{r.regime} \\\\"
    )


def _rolling_row(r: RollingAuditResult, name: str) -> str:
    return (
        f"{name} & "
        f"{r.violation_rate_corrected:.3f} & "
        f"{r.kupiec_pvalue_corrected:.3f} & "
        f"{r.christoffersen_pvalue_corrected:.3f} & "
        f"{r.basel_zone_corrected} & "
        f"{r.quantile_score_corrected:.4f} & "
        f"{r.fz_score_corrected:.4f} & "
        f"{r.q_v_roll.mean():.4f} & "
        f"{r.replacement_ratio.mean():.3f} & "
        f"{r.regime} \\\\"
    )
