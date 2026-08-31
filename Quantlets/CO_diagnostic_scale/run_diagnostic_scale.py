"""
CO_diagnostic_scale — run_diagnostic_scale.py
=============================================
Empirical two-parameter scale diagnostic for the conformal shift.

For each of the 216 model-asset pairs (9 forecasters x 24 assets) this fits the
location-scale map VaR_cp = a_hat + b_hat * VaR_raw by alpha-level linear
quantile regression on the calibration split, and reports the share of the
one-parameter conformal shift |q_hat_V| attributable to the multiplicative
(scale) term, aggregated per forecaster. A share near zero indicates a pure
location shift; a larger share indicates a multiplicative (scale) component.
Reported as it falls (no thumb on the scale).

DIAGNOSTIC ONLY: the location-scale map is not proposed as a competing
corrector (the k = 1 argument from the alpha*T constraint stands).

Uses conformal_oracle.recalibration.diagnose_scale.

Outputs:
  tab_scale_diagnostic.tex        LaTeX table (per forecaster) -> rr_material.tex
  scale_diagnostic_results.csv    Full per-pair results

Usage:  python run_diagnostic_scale.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from conformal_oracle.recalibration import diagnose_scale
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "Quantlets"))
from cfp_config import split_indices  # noqa: E402

ALPHA = 0.01
F_CAL = 0.70

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent
DATA_DIR = BASE / "cfp_ijf_data"

ASSETS = [
    "ASX200",
    "AUDUSD",
    "BOVESPA",
    "BTC",
    "FCHI",
    "CBU0",
    "DJCI",
    "ETH",
    "EURUSD",
    "FTSE100",
    "GBPUSD",
    "GDAXI",
    "GOLD",
    "HSI",
    "IBGL",
    "ICLN",
    "NATGAS",
    "NIFTY",
    "NIKKEI",
    "SP500",
    "STOXX",
    "TLT",
    "USDJPY",
    "WTI",
]

TSFM_MODELS = {
    "Chronos-Small": "chronos_small",
    "Chronos-Mini": "chronos_mini",
    "TimesFM-2.5": "timesfm25",
    "Moirai-2.0": "moirai2",
    "Lag-Llama": "lagllama",
}
BENCHMARK_MODELS = {
    "GJR-GARCH": "gjr_garch",
    "GARCH-N": "garch_n",
    "HS": "hs",
    "EWMA": "ewma",
}
ALL_MODELS = {**TSFM_MODELS, **BENCHMARK_MODELS}


def load_pair(model_name, asset, alpha=ALPHA):
    """Aligned (returns, VaR_raw_positive) for a model-asset pair.

    The parquet stores the signed lower quantile in column ``VaR_{alpha}``
    (violation when ``r < VaR``); we return the positive-loss convention
    ``raw_var = -VaR`` expected by ``diagnose_scale``.
    """
    ret_df = pd.read_csv(
        DATA_DIR / "returns" / f"{asset}.csv", index_col=0, parse_dates=True
    )
    ret_df.columns = [c.strip().lower() for c in ret_df.columns]
    r_series = ret_df.iloc[:, 0]

    subdir = ALL_MODELS[model_name]
    if model_name in TSFM_MODELS:
        var_path = DATA_DIR / subdir / f"{asset}.parquet"
    else:
        var_path = DATA_DIR / "benchmarks" / f"{asset}_{subdir}.parquet"
    var_df = pd.read_parquet(var_path)
    var_df.index = pd.to_datetime(var_df.index)
    v_series = var_df[f"VaR_{alpha}"]

    common = r_series.index.intersection(v_series.index).sort_values()
    r = r_series.loc[common].values.astype(float)
    v = v_series.loc[common].values.astype(float)
    mask = ~(np.isnan(r) | np.isnan(v))
    return r[mask], -v[mask]  # positive-loss raw VaR


def run():
    rows = []
    n_skip = 0
    for model in ALL_MODELS:
        for asset in ASSETS:
            try:
                r, raw = load_pair(model, asset)
                _cal, _test, _g = split_indices(len(r), raw - r, f_cal=F_CAL)
                n_cal, t0 = len(_cal), int(_test[0])
                d = diagnose_scale(raw[:n_cal], r[:n_cal], ALPHA)
                # Regime R = |q_v_stat| / mean(|VaR_raw|) on the same
                # calibration split (Section 4; R > 1 => replacement).
                mean_raw = float(np.mean(np.abs(raw[:n_cal])))
                r_ratio = abs(d.q_v_stat) / mean_raw if mean_raw > 1e-12 else np.inf
                rows.append(
                    dict(
                        model=model,
                        asset=asset,
                        q_v_stat=d.q_v_stat,
                        a_hat=d.a_hat,
                        b_hat=d.b_hat,
                        loc_magnitude=d.loc_magnitude,
                        scale_magnitude=d.scale_magnitude,
                        scale_share=d.scale_share,
                        mean_raw=mean_raw,
                        R=r_ratio,
                        replacement=bool(r_ratio > 1.0),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                n_skip += 1
                print(f"  skip {model}/{asset}: {exc}")
    print(f"loaded {len(rows)} pairs, skipped {n_skip}")
    return pd.DataFrame(rows)


def generate_table(df):
    order = list(ALL_MODELS.keys())
    agg = (
        df.groupby("model")
        .agg(
            n=("scale_share", "size"),
            b_hat=("b_hat", "mean"),
            pct_bneg=("b_hat", lambda s: 100.0 * (s < 0).mean()),
            pct_repl=("replacement", lambda s: 100.0 * s.mean()),
            scale_share=("scale_share", "mean"),
            scale_share_med=("scale_share", "median"),
        )
        .reindex(order)
    )
    overall = df["scale_share"].mean()
    sp = df[~df["replacement"]]  # signal-preserving regime (R <= 1)

    lines = [
        r"\begin{table}[htbp]",
        r"	\centering",
        r"	\caption{Two-parameter scale diagnostic on the 216 model--asset "
        r"pairs ($\alpha = 0.01$, calibration split). $\hat b$: mean slope of "
        r"the location-scale fit $\VaR_{\mathrm{cp}}=\hat a+\hat b\,"
        r"\VaR^{\mathrm{raw}}$ ($\hat b=1$ means a pure location shift); "
        r"\%$\hat b{<}0$: share of pairs with a negative (inverted) slope; "
        r"\%$R{>}1$: share in the replacement regime ($R=|\hat q_V|/"
        r"\overline{|\VaR^{\mathrm{raw}}|}$); scale share: mean (median) "
        r"fraction of the correction taken by the multiplicative term, "
        r"$|(\hat b-1)\overline{\VaR^{\mathrm{raw}}}|/"
        r"(|(\hat b-1)\overline{\VaR^{\mathrm{raw}}}|+|\hat a|)$. The scale "
        r"share is interpretable as genuine scale repair only in the "
        r"signal-preserving regime ($R\le1$): under replacement it is inflated "
        r"by slope inversion ($\hat b<0$) or degenerate raw magnitudes "
        r"($\hat a$ dominates). Diagnostic only.}",
        r"	\label{tab:scale_diagnostic}",
        r"	\footnotesize",
        r"	\begin{tabular}{@{}lrrrrrr@{}}",
        r"		\hline\hline",
        r"		Forecaster & $n$ & $\hat b$ & \%$\hat b{<}0$ & \%$R{>}1$ "
        r"& Scale share & (median) \\",
        r"		\hline",
    ]
    for model in order:
        if model not in agg.index or pd.isna(agg.loc[model, "n"]):
            continue
        row = agg.loc[model]
        lines.append(
            f"		{model} & {int(row['n'])} & {row['b_hat']:.3f} "
            f"& {row['pct_bneg']:.0f} & {row['pct_repl']:.0f} "
            f"& {row['scale_share']:.3f} & {row['scale_share_med']:.3f} \\\\"
        )
    lines += [
        r"		\hline",
        f"		\\textit{{All pairs}} & {len(df)} & "
        f"{df['b_hat'].mean():.3f} & {100.0 * (df['b_hat'] < 0).mean():.0f} "
        f"& {100.0 * df['replacement'].mean():.0f} & {overall:.3f} "
        f"& {df['scale_share'].median():.3f} \\\\",
        f"		\\textit{{Signal-preserving}} ($R\\le1$) & {len(sp)} & "
        f"{sp['b_hat'].mean():.3f} & {100.0 * (sp['b_hat'] < 0).mean():.0f} "
        f"& 0 & {sp['scale_share'].mean():.3f} "
        f"& {sp['scale_share'].median():.3f} \\\\",
        r"		\hline\hline",
        r"	\end{tabular}",
        r"\end{table}",
    ]
    out = SCRIPT_DIR / "tab_scale_diagnostic.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


def generate_crosstab(df):
    """Cross-tabulate the two independent regime diagnostics: the sign of the
    quantile-regression slope (b_hat < 0) and the paper's magnitude ratio
    (R > 1). High concordance means b_hat corroborates the R classification."""
    bneg = df["b_hat"] < 0
    repl = df["replacement"]
    n11 = int((bneg & repl).sum())
    n10 = int((bneg & ~repl).sum())
    n01 = int((~bneg & repl).sum())
    n00 = int((~bneg & ~repl).sum())
    concord = 100.0 * ((bneg == repl).mean())
    bneg_implies_repl = 100.0 * (repl[bneg].mean()) if bneg.any() else float("nan")

    lines = [
        r"\begin{table}[htbp]",
        r"	\centering",
        r"	\caption{Concordance of two independent regime diagnostics on the "
        r"216 pairs: the sign of the location-scale slope $\hat b$ (this "
        r"section) and the magnitude ratio $R=|\hat q_V|/"
        r"\overline{|\VaR^{\mathrm{raw}}|}$ from the paper's classification "
        r"(Section~4). Cells count model--asset pairs. Agreement on the "
        f"replacement/signal-preserving split is {concord:.1f}\\%; a negative "
        f"slope implies $R>1$ in {bneg_implies_repl:.0f}\\% of cases, so the "
        r"scale diagnostic independently recovers the paper's regime split.}",
        r"	\label{tab:scale_regime_crosstab}",
        r"	\footnotesize",
        r"	\begin{tabular}{@{}lrr@{}}",
        r"		\hline\hline",
        r"		& $R\le1$ (signal-pres.) & $R>1$ (replacement) \\",
        r"		\hline",
        f"		$\\hat b\\ge0$ & {n00} & {n01} \\\\",
        f"		$\\hat b<0$ & {n10} & {n11} \\\\",
        r"		\hline\hline",
        r"	\end{tabular}",
        r"\end{table}",
    ]
    out = SCRIPT_DIR / "tab_scale_regime_crosstab.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"Saved: {out}")


if __name__ == "__main__":
    df = run()
    df.to_csv(SCRIPT_DIR / "scale_diagnostic_results.csv", index=False)
    print(f"Saved: {SCRIPT_DIR / 'scale_diagnostic_results.csv'}")
    if len(df):
        generate_table(df)
        generate_crosstab(df)
        sp = df[~df["replacement"]]
        print(f"\nOverall mean scale share: {df['scale_share'].mean():.3f}")
        print(
            f"Signal-preserving (R<=1) scale share: {sp['scale_share'].mean():.3f} "
            f"(n={len(sp)})"
        )
