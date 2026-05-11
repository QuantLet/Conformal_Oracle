"""LaTeX table emitters for panel-level results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from conformal_oracle.panel.result import PanelResult


def master_table_to_latex(
    panel_result: "PanelResult",
    sort_by: str = "R",
    panel_split: bool = True,
) -> str:
    """LaTeX table matching tab:master format."""
    df = panel_result.master_table()
    agg = (
        df.groupby("forecaster")
        .agg({
            "regime": "first",
            "q_v": "mean",
            "R": "mean",
            "pi_corrected": "mean",
            "kupiec_p": "mean",
            "christoffersen_p": "mean",
            "basel_corrected": lambda x: (x == "green").mean(),
            "qs_corrected": "mean",
            "fz_corrected": "mean",
        })
        .rename(columns={"basel_corrected": "green_frac"})
    )
    agg = agg.sort_values(sort_by)

    header = (
        "\\begin{tabular}{l l rrr rr rr}\n"
        "\\toprule\n"
        "Model & Regime & $\\hat{\\pi}$ & Kupiec $p$ "
        "& Chr.~$p$ & Green\\% & QS & "
        "$\\hat{q}_V$ & $\\bar{R}$ \\\\\n"
        "\\midrule"
    )

    rows = []
    if panel_split:
        sp = agg[agg["regime"] == "signal-preserving"]
        rp = agg[agg["regime"] == "replacement"]
        if len(sp) > 0:
            rows.append(
                "\\multicolumn{9}{l}"
                "{\\textit{Panel A: Signal-preserving}} \\\\"
            )
            for name, r in sp.iterrows():
                rows.append(_master_row(str(name), r))
        if len(rp) > 0:
            rows.append("\\midrule")
            rows.append(
                "\\multicolumn{9}{l}"
                "{\\textit{Panel B: Replacement}} \\\\"
            )
            for name, r in rp.iterrows():
                rows.append(_master_row(str(name), r))
    else:
        for name, r in agg.iterrows():
            rows.append(_master_row(str(name), r))

    body = "\n".join(rows)
    footer = "\\bottomrule\n\\end{tabular}"

    return f"{header}\n{body}\n{footer}"


def _master_row(name: str, r: pd.Series) -> str:
    return (
        f"{name} & {r['regime']} & "
        f"{r['pi_corrected']:.3f} & "
        f"{r['kupiec_p']:.3f} & "
        f"{r['christoffersen_p']:.3f} & "
        f"{r['green_frac']:.0%} & "
        f"{r['qs_corrected']:.4f} & "
        f"{r['q_v']:.4f} & "
        f"{r['R']:.3f} \\\\"
    )


def regime_summary_to_latex(
    panel_result: "PanelResult",
) -> str:
    """LaTeX table for regime summary."""
    df = panel_result.regime_summary()

    header = (
        "\\begin{tabular}{l rr rrr rrr}\n"
        "\\toprule\n"
        "Model & SP & Repl. & "
        "G(raw) & Y(raw) & R(raw) & "
        "G(corr) & Y(corr) & R(corr) \\\\\n"
        "\\midrule"
    )

    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['forecaster']} & "
            f"{r['n_signal_preserving']} & "
            f"{r['n_replacement']} & "
            f"{r['green_raw']} & "
            f"{r['yellow_raw']} & "
            f"{r['red_raw']} & "
            f"{r['green_corrected']} & "
            f"{r['yellow_corrected']} & "
            f"{r['red_corrected']} \\\\"
        )

    body = "\n".join(rows)
    footer = "\\bottomrule\n\\end{tabular}"
    return f"{header}\n{body}\n{footer}"


def cross_sectional_corr_to_latex(
    corr_df: pd.DataFrame,
) -> str:
    """LaTeX table for cross-sectional correlations."""
    header = "\\begin{tabular}{l" + " r" * len(corr_df.columns) + "}\n"
    header += "\\toprule\n"
    header += "Model"
    for col in corr_df.columns:
        header += f" & {col}"
    header += " \\\\\n\\midrule"

    rows = []
    for name, r in corr_df.iterrows():
        parts = [str(name)]
        for col in corr_df.columns:
            parts.append(f"{r[col]:.3f}")
        rows.append(" & ".join(parts) + " \\\\")

    body = "\n".join(rows)
    footer = "\\bottomrule\n\\end{tabular}"
    return f"{header}\n{body}\n{footer}"


def diebold_mariano_to_latex(
    dm_table: pd.DataFrame,
) -> str:
    """LaTeX table for panel DM test results."""
    header = (
        "\\begin{tabular}{l l rr}\n"
        "\\toprule\n"
        "Forecaster & Baseline & DM stat & $p$-value \\\\\n"
        "\\midrule"
    )

    rows = []
    for _, r in dm_table.iterrows():
        rows.append(
            f"{r['forecaster']} & "
            f"{r['baseline']} & "
            f"{r['dm_statistic']:.3f} & "
            f"{r['p_value']:.4f} \\\\"
        )

    body = "\n".join(rows)
    footer = "\\bottomrule\n\\end{tabular}"
    return f"{header}\n{body}\n{footer}"


def wildcluster_kupiec_to_latex(
    kupiec_table: pd.DataFrame,
) -> str:
    """LaTeX table for wild-cluster bootstrap Kupiec results."""
    header = (
        "\\begin{tabular}{l rrr}\n"
        "\\toprule\n"
        "Model & LR & $p$(asym) & $p$(boot) \\\\\n"
        "\\midrule"
    )
    rows = []
    for _, r in kupiec_table.iterrows():
        rows.append(
            f"{r['forecaster']} & "
            f"{r['lr_original']:.3f} & "
            f"{r['p_asymptotic']:.4f} & "
            f"{r['p_bootstrap']:.4f} \\\\"
        )
    body = "\n".join(rows)
    footer = "\\bottomrule\n\\end{tabular}"
    return f"{header}\n{body}\n{footer}"


def wildcluster_dm_to_latex(
    dm_table: pd.DataFrame,
) -> str:
    """LaTeX table for wild-cluster bootstrap DM results."""
    header = (
        "\\begin{tabular}{l l r}\n"
        "\\toprule\n"
        "Forecaster & Baseline & $p$(boot) \\\\\n"
        "\\midrule"
    )
    rows = []
    for _, r in dm_table.iterrows():
        rows.append(
            f"{r['forecaster']} & "
            f"{r['baseline']} & "
            f"{r['bootstrap_p']:.4f} \\\\"
        )
    body = "\n".join(rows)
    footer = "\\bottomrule\n\\end{tabular}"
    return f"{header}\n{body}\n{footer}"
