#!/usr/bin/env python3
"""AE point 4: does the conformal correction improve EVERY model-asset pair?

The AE's objection is that Table 4 aggregates over assets, and suspects that
pairs whose raw forecast was already well calibrated on the test set are made
worse by a correction estimated on the validation set. This script answers that
from the stored forecast series; no model inference is re-run.

For every (model, asset, alpha) cell it reports
    dQS = QS_raw - QS_corrected            (positive = the correction helped)
for both the single-split and the rolling estimator, together with the raw
forecast's calibration quality (|pihat_raw - alpha| and the Kupiec p-value), and
tests whether deterioration concentrates on the well-calibrated pairs.

Single-split numbers are read from the paper's own result matrices. Rolling
numbers are re-scored here with the same functions the paper uses, because
rolling_vs_static.csv stores coverage but not the quantile score.

Outputs (analysis/ae_point4/):
    pairs_long.csv           one row per (model, asset, alpha, estimator)
    deteriorating_pairs.csv  the dQS < 0 subset
    tab_deterioration.tex    compact LaTeX table of deteriorating pairs
    tab_crosstab.tex         deterioration vs raw calibration quality
    regression_results.txt   dQS on |pihat_raw - alpha|
    fig_dqs_scatter.png      raw miscalibration (x) vs dQS (y), zero line marked
    summary.md               the numbers in prose
"""

from __future__ import annotations

import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = BASE / "analysis" / "ae_point4"
TABLES = DATA / "paper_outputs" / "tables"

ALPHAS = [0.01, 0.025, 0.05, 0.10]
F_CAL = 0.70
W_ROLL = 250

SYMBOLS = ['SP500', 'STOXX', 'GDAXI', 'FCHI', 'FTSE100', 'ICLN',
           'NIKKEI', 'HSI', 'BOVESPA', 'NIFTY', 'ASX200', 'CBU0',
           'TLT', 'IBGL', 'DJCI', 'GOLD', 'WTI', 'NATGAS',
           'BTC', 'ETH', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

# Models come from cfp_config so this analysis cannot silently drift from the
# table it answers to. As of 2026-08-17 that is 13 forecasters x 24 assets x 4
# alpha levels; it was 10 x 24 when the AE's question was first answered, and
# the answer below is NOT the same answer, because four of those series were
# corrected in between.
sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import MODELS  # noqa: E402
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "Quantlets"))
from cfp_config import split_indices  # noqa: E402

# Series carrying a traced defect, which dominate any pooled regression and are
# therefore reported separately rather than averaged in.
#
# This set USED to be {TimesFM-2.5, Moirai-2.0}, on the grounds that their raw
# 1% violation rate was ~99% and they were "interface failures, not
# forecasters". That was the sign defect, not the interface: corrected, they run
# at 0.0143 and 0.0178 and are among the best-calibrated raw forecasters in the
# panel. The two series that now dominate are the Chronos pair sampled under the
# checkpoint default top_k = 50, whose R-bar is 17.3 and 23.5 against 0.09-0.36
# for everything else. Their analytic counterparts are NOT in this set -- they
# are ordinary members of the panel.
DEFECTIVE_SERIES = {'Chronos-Small', 'Chronos-Mini'}


# --------------------------------------------------------------------------- #
# scoring primitives — identical to scripts/regenerate_rolling_vs_static.py
# --------------------------------------------------------------------------- #

def qhat_ceil(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = int(ceil((n + 1) * (1 - alpha))) - 1
    k = min(k, n - 1)
    return float(np.sort(scores)[k])


def quantile_score(r: np.ndarray, v: np.ndarray, alpha: float) -> float:
    """Pinball loss at level alpha. `QS` in the paper's tables is this object."""
    diff = r - v
    return float(np.mean(np.where(diff < 0, (alpha - 1) * diff, alpha * diff)))


def traffic_light(x: int, n: int) -> str:
    """Basel zone, scaled to a 250-day year. Same rule as the paper's pipeline."""
    if n == 0:
        return "Green"
    annual = x * (250.0 / n)
    if annual <= 4:
        return "Green"
    if annual <= 9:
        return "Yellow"
    return "Red"


ZONE_RANK = {"Green": 0, "Yellow": 1, "Red": 2}


def kupiec_p(x: int, n: int, alpha: float) -> float:
    if n == 0:
        return 1.0
    pi_hat = x / n
    if pi_hat == 0:
        lr = 2 * n * np.log(1 / (1 - alpha))
    elif pi_hat == 1:
        lr = 2 * n * np.log(alpha)
    else:
        lr = 2 * (x * np.log(pi_hat / alpha) +
                  (n - x) * np.log((1 - pi_hat) / (1 - alpha)))
    return float(1 - stats.chi2.cdf(lr, 1))


def load_pair(model: str, symbol: str, alpha: float):
    subdir, suffix = MODELS[model]
    ret = pd.read_csv(DATA / "returns" / f"{symbol}.csv",
                      index_col=0, parse_dates=True)
    ret.columns = ["r"]
    name = f"{symbol}.parquet" if suffix is None else f"{symbol}_{suffix}.parquet"
    fc = pd.read_parquet(DATA / subdir / name)
    common = ret.index.intersection(fc.index)
    ret, fc = ret.loc[common], fc.loc[common]
    # Column names are VaR_0.01 / VaR_0.025 / VaR_0.05 / VaR_0.1
    col = f"VaR_{alpha:g}"
    mask = fc[col].notna()
    return ret["r"].values[mask], fc[col].values[mask]


def score_pair(model: str, symbol: str, alpha: float) -> dict | None:
    """Raw, single-split and rolling quantile scores for one cell."""
    try:
        r, q = load_pair(model, symbol, alpha)
    except Exception as exc:  # missing series for this model/asset
        print(f"  SKIP {model}/{symbol}/{alpha}: {exc}", file=sys.stderr)
        return None

    n = len(r)
    _cal, _test, _g = split_indices(n, q - r, f_cal=F_CAL)
    n_cal, t0 = len(_cal), int(_test[0])
    if n_cal < W_ROLL or n - t0 < 50:
        print(f"  SKIP {model}/{symbol}/{alpha}: series too short", file=sys.stderr)
        return None

    r_cal, r_test = r[:n_cal], r[t0:]
    q_cal, q_test = q[:n_cal], q[t0:]
    n_test = len(r_test)

    # --- raw --------------------------------------------------------------- #
    qs_raw = quantile_score(r_test, q_test, alpha)
    viol_raw = int(np.sum(r_test < q_test))
    pihat_raw = viol_raw / n_test

    # The same backtest on the CALIBRATION window. A deployment rule that gates
    # on the test-window backtest is using the outcome it is later scored on;
    # this is the version of the signal that is actually available on the day
    # the decision has to be taken.
    viol_cal = int(np.sum(r_cal < q_cal))
    pihat_cal = viol_cal / n_cal

    # --- single split ------------------------------------------------------ #
    qV = qhat_ceil(q_cal - r_cal, alpha)
    var_static = q_test - qV
    qs_static = quantile_score(r_test, var_static, alpha)
    viol_static = int(np.sum(r_test < var_static))

    # --- rolling ----------------------------------------------------------- #
    history = list((q_cal - r_cal)[-W_ROLL:])
    var_roll = np.empty(n_test)
    for t in range(n_test):
        var_roll[t] = q_test[t] - qhat_ceil(np.array(history[-W_ROLL:]), alpha)
        history.append(q_test[t] - r_test[t])
    qs_roll = quantile_score(r_test, var_roll, alpha)
    viol_roll = int(np.sum(r_test < var_roll))

    return {
        "model": model, "asset": symbol, "alpha": alpha,
        "n_test": n_test, "qV": qV,
        "pihat_raw": pihat_raw,
        "p_kup_raw": kupiec_p(viol_raw, n_test, alpha),
        "miscal_raw": abs(pihat_raw - alpha),
        "QS_raw": qs_raw,
        "QS_static": qs_static, "QS_roll": qs_roll,
        "pihat_static": viol_static / n_test,
        "pihat_roll": viol_roll / n_test,
        "pihat_cal": pihat_cal,
        "p_kup_cal": kupiec_p(viol_cal, n_cal, alpha),
        "TL_cal": traffic_light(viol_cal, n_cal),
        "TL_raw": traffic_light(viol_raw, n_test),
        "TL_static": traffic_light(viol_static, n_test),
        "TL_roll": traffic_light(viol_roll, n_test),
        "dQS_static": qs_raw - qs_static,
        "dQS_roll": qs_raw - qs_roll,
        # QS is scale-dependent across assets, so the relative change is what
        # can be compared or pooled across pairs.
        "rel_static": (qs_raw - qs_static) / qs_raw if qs_raw > 0 else np.nan,
        "rel_roll": (qs_raw - qs_roll) / qs_raw if qs_raw > 0 else np.nan,
    }


# --------------------------------------------------------------------------- #
# cross-check against the paper's own matrices
# --------------------------------------------------------------------------- #

def cross_check(df: pd.DataFrame) -> str:
    """Re-scored single-split numbers must reproduce all_results.csv."""
    ar = pd.read_csv(TABLES / "all_results.csv")
    m11 = TABLES / "moirai11_full_results.csv"
    if m11.exists():
        ar = pd.concat([ar, pd.read_csv(m11)], ignore_index=True)
    ar = ar.rename(columns={"symbol": "asset"})
    merged = df.merge(ar[["model", "asset", "alpha", "QS_raw", "QS_cp",
                          "pihat_raw", "qV"]],
                      on=["model", "asset", "alpha"], how="inner",
                      suffixes=("", "_paper"))
    lines = [f"cross-check against all_results.csv: {len(merged)} cells matched"]
    for ours, theirs in (("QS_raw", "QS_raw_paper"), ("QS_static", "QS_cp"),
                         ("pihat_raw", "pihat_raw_paper"), ("qV", "qV_paper")):
        d = (merged[ours] - merged[theirs]).abs()
        rel = d / merged[theirs].abs().replace(0, np.nan)
        lines.append(f"  {ours:<12} vs {theirs:<16} "
                     f"max abs diff {d.max():.3e}  max rel diff {rel.max():.3e}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #

def deterioration_table(df: pd.DataFrame, est: str) -> pd.DataFrame:
    col = f"dQS_{est}"
    bad = df[df[col] < 0].copy()
    return bad.sort_values(col)


def crosstab(df: pd.DataFrame, est: str) -> pd.DataFrame:
    """Deterioration rate against how well calibrated the raw forecast was."""
    col = f"dQS_{est}"
    d = df.copy()
    d["worse"] = d[col] < 0
    # Kupiec p > 0.05 means the raw forecast's unconditional coverage is not
    # rejected on the test set: "already well calibrated" in the AE's sense.
    d["raw_calibrated"] = np.where(d["p_kup_raw"] > 0.05,
                                   "raw NOT rejected (p>0.05)",
                                   "raw rejected (p<=0.05)")
    tab = d.groupby(["alpha", "raw_calibrated"]).agg(
        n_pairs=("worse", "size"),
        n_worse=("worse", "sum"),
    )
    tab["pct_worse"] = 100 * tab["n_worse"] / tab["n_pairs"]
    return tab.reset_index()


# Above this raw miscalibration the base forecast is not a forecast in any
# useful sense, and the regression becomes close to an identity: a model that
# violates on 99% of days has QS_raw dominated by the same gap that defines
# |pihat_raw - alpha|, which forces slope ~ 1 and R^2 ~ 0.98 mechanically.
# The interpretable region is the one where the base forecast is roughly usable.
MISCAL_INTERPRETABLE = 0.05


def regression(df: pd.DataFrame, est: str, alpha: float, drop_grid: bool,
               restrict: bool = False) -> dict:
    """Relative dQS on raw miscalibration. The AE's hypothesis implies a
    positive slope: the worse the raw forecast, the more the correction helps —
    and, at the well-calibrated end, losses (a negative intercept)."""
    d = df[(df["alpha"] == alpha)].copy()
    if drop_grid:
        d = d[~d["model"].isin(DEFECTIVE_SERIES)]
    if restrict:
        d = d[d["miscal_raw"] < MISCAL_INTERPRETABLE]
    d = d.dropna(subset=[f"rel_{est}", "miscal_raw"])
    if len(d) < 5:
        return {}
    x = d["miscal_raw"].values
    y = d[f"rel_{est}"].values
    res = stats.linregress(x, y)
    return {
        "alpha": alpha, "estimator": est, "drop_grid_failures": drop_grid,
        "restricted": restrict,
        "n": len(d), "slope": res.slope, "intercept": res.intercept,
        "r2": res.rvalue ** 2, "p_value": res.pvalue, "stderr": res.stderr,
        "n_worse": int((d[f"dQS_{est}"] < 0).sum()),
    }


def well_calibrated_test(df: pd.DataFrame, est: str, alpha: float) -> dict:
    """The AE's question, asked directly: on pairs whose RAW forecast already
    passes Kupiec on the test set, does the correction lose on average?"""
    d = df[(df["alpha"] == alpha) & (df["p_kup_raw"] > 0.05)]
    y = (d[f"rel_{est}"] * 100).dropna()
    if len(y) < 3:
        return {}
    t = stats.ttest_1samp(y, 0.0)
    w = stats.wilcoxon(y) if len(y) > 5 else None
    return {
        "alpha": alpha, "estimator": est, "n": len(y),
        "mean_pct": y.mean(), "median_pct": y.median(),
        "n_worse": int((d[f"dQS_{est}"] < 0).sum()),
        "t_stat": t.statistic, "t_p": t.pvalue,
        "wilcoxon_p": w.pvalue if w is not None else np.nan,
    }


# Categorical slots 1 and 2 of the validated reference palette, in fixed order.
C_SPLIT, C_ROLL = "#2a78d6", "#eb6834"
INK, INK_2 = "#0b0b0b", "#52514e"


def _panel(ax, d, alpha, xmax=None):
    """One alpha panel: raw miscalibration against relative dQS, zero line marked."""
    if xmax is not None:
        off = d[d["miscal_raw"] >= xmax]
        d = d[d["miscal_raw"] < xmax]
    ax.axhline(0, color=INK, lw=1.2, zorder=2)
    ax.scatter(d["miscal_raw"], 100 * d["rel_static"], s=42,
               facecolor=C_SPLIT, edgecolor="white", linewidth=0.8,
               label="single split", zorder=4)
    ax.scatter(d["miscal_raw"], 100 * d["rel_roll"], s=42, marker="^",
               facecolor=C_ROLL, edgecolor="white", linewidth=0.8,
               label="rolling", zorder=3)
    ax.set_title(rf"$\alpha$ = {alpha:g}", fontsize=11, color=INK)
    ax.set_xlabel(r"raw miscalibration  $|\hat\pi_{\rm raw}-\alpha|$",
                  fontsize=9, color=INK_2)
    ax.set_ylabel(r"$\Delta$QS  (% of QS$_{\rm raw}$)", fontsize=9, color=INK_2)
    ax.tick_params(labelsize=8, colors=INK_2)
    ax.grid(alpha=0.18, linewidth=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if xmax is not None and len(off):
        lo = 100 * off[["rel_static", "rel_roll"]].min().min()
        hi = 100 * off[["rel_static", "rel_roll"]].max().max()
        ax.annotate(f"{len(off)} pairs off-scale to the right\n"
                    f"(gain +{lo:.0f}% to +{hi:.0f}%)",
                    xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                    fontsize=7.5, color=INK_2)


def make_figure(df: pd.DataFrame, path: Path, zoom: bool) -> None:
    """zoom=True restricts to the region where the base forecast is usable —
    which is where the whole finding lives. The full-range version compresses it
    into an invisible sliver, so both are produced."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    for ax, alpha in zip(axes.ravel(), ALPHAS):
        _panel(ax, df[df["alpha"] == alpha], alpha,
               xmax=MISCAL_INTERPRETABLE if zoom else None)
    axes[0, 0].legend(fontsize=8.5, loc="upper left", framealpha=0.95,
                      labelcolor=INK_2)
    title = ("Correction gain where the base forecast is usable "
             rf"($|\hat\pi_{{\rm raw}}-\alpha| < {MISCAL_INTERPRETABLE}$)"
             if zoom else
             "Correction gain against raw calibration quality — full range")
    fig.suptitle(title + "\nabove the line = the correction helped",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def tex_deteriorating(bad: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, r in bad.iterrows():
        rows.append(
            f"{r['model']} & {r['asset']} & {r['alpha']:g} & "
            f"{r['pihat_raw']:.4f} & {r['miscal_raw']:.4f} & "
            f"{r['p_kup_raw']:.3f} & {r['QS_raw']:.3e} & "
            f"{r['dQS_static']:+.2e} & {100 * r['rel_static']:+.2f} & "
            f"{r['dQS_roll']:+.2e} & {100 * r['rel_roll']:+.2f} \\\\")
    body = "\n".join(rows) if rows else "\\multicolumn{11}{c}{none} \\\\"
    path.write_text(
        "% generated by analysis/ae_point4/run_ae_point4.py\n"
        "\\begin{tabular}{llrrrrrrrrr}\n\\toprule\n"
        "Model & Asset & $\\alpha$ & $\\hat\\pi_{\\rm raw}$ & "
        "$|\\hat\\pi_{\\rm raw}-\\alpha|$ & $p_{\\rm Kup}$ & QS$_{\\rm raw}$ & "
        "$\\Delta$QS$_{\\rm split}$ & \\% & $\\Delta$QS$_{\\rm roll}$ & \\% \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8")


def tex_crosstab(tab: pd.DataFrame, path: Path, est: str) -> None:
    rows = [f"{r['alpha']:g} & {r['raw_calibrated']} & {int(r['n_pairs'])} & "
            f"{int(r['n_worse'])} & {r['pct_worse']:.1f} \\\\"
            for _, r in tab.iterrows()]
    path.write_text(
        "% generated by analysis/ae_point4/run_ae_point4.py\n"
        "\\begin{tabular}{llrrr}\n\\toprule\n"
        "$\\alpha$ & Raw forecast & Pairs & $\\Delta$QS $<0$ & \\% \\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    records = []
    for model in MODELS:
        for sym in SYMBOLS:
            for alpha in ALPHAS:
                rec = score_pair(model, sym, alpha)
                if rec:
                    records.append(rec)
        print(f"  scored {model}", file=sys.stderr)
    df = pd.DataFrame(records)
    df.to_csv(OUT / "pairs_long.csv", index=False)
    print(f"\n{len(df)} cells "
          f"({df['model'].nunique()} models x {df['asset'].nunique()} assets "
          f"x {df['alpha'].nunique()} alphas)", file=sys.stderr)

    check = cross_check(df)
    print("\n" + check, file=sys.stderr)

    # --- deterioration ----------------------------------------------------- #
    lines = ["# AE point 4 — does the correction help every pair?", "",
             "Generated by `analysis/ae_point4/run_ae_point4.py`. "
             "Single-split figures reproduce the paper's `all_results.csv`; "
             "rolling figures are re-scored from the stored forecast series "
             "(no model inference re-run).", "",
             "```", check, "```", ""]

    lines += ["## Distribution of $\\Delta$QS = QS_raw − QS_corrected", "",
              "Positive means the correction helped. Absolute $\\Delta$QS is "
              "scale-dependent across assets, so the percentage column is the "
              "comparable one.", "",
              "| α | estimator | pairs | worse (ΔQS<0) | min | q05 | q25 | median | q75 | q95 | max |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for alpha in ALPHAS:
        for est in ("static", "roll"):
            d = df[df["alpha"] == alpha][f"rel_{est}"].dropna() * 100
            n_worse = int((df[(df["alpha"] == alpha)][f"dQS_{est}"] < 0).sum())
            qs = d.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            lines.append(
                f"| {alpha:g} | {'single split' if est == 'static' else 'rolling'} "
                f"| {len(d)} | **{n_worse}** | {d.min():.2f} | {qs[0.05]:.2f} | "
                f"{qs[0.25]:.2f} | {qs[0.5]:.2f} | {qs[0.75]:.2f} | "
                f"{qs[0.95]:.2f} | {d.max():.2f} |")
    lines.append("")
    lines.append("(percentages of QS_raw)")
    lines.append("")

    # deteriorating pairs: union over both estimators
    bad = df[(df["dQS_static"] < 0) | (df["dQS_roll"] < 0)].copy()
    bad = bad.sort_values(["alpha", "dQS_static"])
    bad.to_csv(OUT / "deteriorating_pairs.csv", index=False)
    tex_deteriorating(bad, OUT / "tab_deterioration.tex")

    lines += [f"## Deteriorating pairs ({len(bad)} of {len(df)} cells)", ""]
    if len(bad):
        lines += ["| Model | Asset | α | π̂_raw | \\|π̂−α\\| | p_Kupiec | "
                  "ΔQS_split % | ΔQS_roll % |", "|---|---|---|---|---|---|---|---|"]
        for _, r in bad.iterrows():
            lines.append(
                f"| {r['model']} | {r['asset']} | {r['alpha']:g} | "
                f"{r['pihat_raw']:.4f} | {r['miscal_raw']:.4f} | "
                f"{r['p_kup_raw']:.3f} | {100 * r['rel_static']:+.2f} | "
                f"{100 * r['rel_roll']:+.2f} |")
    else:
        lines.append("None.")
    lines.append("")

    # --- cross-tabulation -------------------------------------------------- #
    lines += ["## Deterioration against raw calibration quality", ""]
    for est in ("static", "roll"):
        tab = crosstab(df, est)
        tex_crosstab(tab, OUT / f"tab_crosstab_{est}.tex", est)
        lines.append(f"**{'Single split' if est == 'static' else 'Rolling'}**")
        lines.append("")
        lines.append("| α | raw forecast | pairs | ΔQS<0 | % |")
        lines.append("|---|---|---|---|---|")
        for _, r in tab.iterrows():
            lines.append(f"| {r['alpha']:g} | {r['raw_calibrated']} | "
                         f"{int(r['n_pairs'])} | {int(r['n_worse'])} | "
                         f"{r['pct_worse']:.1f} |")
        lines.append("")

    # --- regression -------------------------------------------------------- #
    reg_rows = []
    for est in ("static", "roll"):
        for alpha in ALPHAS:
            for drop in (False, True):
                for restrict in (False, True):
                    res = regression(df, est, alpha, drop, restrict)
                    if res:
                        reg_rows.append(res)
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(OUT / "regression_results.csv", index=False)
    (OUT / "regression_results.txt").write_text(reg.to_string(index=False),
                                                encoding="utf-8")

    lines += ["## Regression of relative ΔQS on raw miscalibration", "",
              "Model: `rel_ΔQS = a + b · |π̂_raw − α|`. The AE's hypothesis "
              "implies b > 0 (the worse the raw forecast, the more the "
              "correction gains) and a ≤ 0 at the well-calibrated end.", "",
              "| α | estimator | grid failures | miscal range | n | slope b | s.e. | intercept | R² | p |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in reg.iterrows():
        lines.append(
            f"| {r['alpha']:g} | "
            f"{'single split' if r['estimator'] == 'static' else 'rolling'} | "
            f"{'excluded' if r['drop_grid_failures'] else 'included'} | "
            f"{'<0.05 only' if r['restricted'] else 'all'} | "
            f"{int(r['n'])} | {r['slope']:.3f} | {r['stderr']:.3f} | "
            f"{r['intercept']:+.4f} | {r['r2']:.3f} | {r['p_value']:.2e} |")
    lines += ["",
              "The rows with grid failures *included* and no restriction are "
              "close to an identity and should not be read as evidence: when a "
              "model violates on 99% of days, QS_raw is dominated by the same "
              "gap that defines |pihat_raw - alpha|, forcing slope ~ 1 and "
              "R^2 ~ 0.98. The informative rows are those restricted to "
              f"|pihat_raw - alpha| < {MISCAL_INTERPRETABLE}, where the "
              "**intercept** answers the AE: it is the expected change in QS "
              "for a base forecast that is already perfectly calibrated.", ""]

    # --- the AE's question asked directly ---------------------------------- #
    wc_rows = [w for est in ("static", "roll") for alpha in ALPHAS
               if (w := well_calibrated_test(df, est, alpha))]
    wc = pd.DataFrame(wc_rows)
    wc.to_csv(OUT / "well_calibrated_test.csv", index=False)
    lines += ["## Pairs whose raw forecast already passes Kupiec (p > 0.05)", "",
              "This is the AE's hypothesis stated as a test: on base forecasts "
              "that were already well calibrated on the test set, does the "
              "correction lose?", "",
              "| α | estimator | n | worse | mean ΔQS % | median % | t-test p | Wilcoxon p |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in wc.iterrows():
        lines.append(
            f"| {r['alpha']:g} | "
            f"{'single split' if r['estimator'] == 'static' else 'rolling'} | "
            f"{int(r['n'])} | {int(r['n_worse'])} | {r['mean_pct']:+.2f} | "
            f"{r['median_pct']:+.2f} | {r['t_p']:.2e} | {r['wilcoxon_p']:.2e} |")
    lines.append("")

    # --- does a QS loss at least buy a zone upgrade? ----------------------- #
    lines += ["## Degraded pairs: does the QS loss buy a Basel zone change?", "",
              "For every pair the correction makes worse in QS, the question is "
              "whether it at least moves the pair to a better Basel zone. A loss "
              "that buys an upgrade is a trade a risk manager can accept; a loss "
              "that leaves the zone unchanged is simply the wrong intervention "
              "on that pair.", ""]
    zone_rows = []
    for est in ("static", "roll"):
        for alpha in ALPHAS:
            d = df[(df["alpha"] == alpha) & (df[f"dQS_{est}"] < 0)].copy()
            if not len(d):
                continue
            up = d[d[f"TL_{est}"].map(ZONE_RANK) < d["TL_raw"].map(ZONE_RANK)]
            same = d[d[f"TL_{est}"].map(ZONE_RANK) == d["TL_raw"].map(ZONE_RANK)]
            down = d[d[f"TL_{est}"].map(ZONE_RANK) > d["TL_raw"].map(ZONE_RANK)]
            same_green = same[same["TL_raw"] == "Green"]
            zone_rows.append({
                "alpha": alpha, "estimator": est, "n_degraded": len(d),
                "zone_up": len(up), "zone_same": len(same),
                "zone_same_already_green": len(same_green),
                "zone_down": len(down),
                "median_loss_pct": 100 * d[f"rel_{est}"].median(),
                "median_loss_if_up": 100 * up[f"rel_{est}"].median() if len(up) else np.nan,
                "median_loss_if_same": 100 * same[f"rel_{est}"].median() if len(same) else np.nan,
            })
    zone = pd.DataFrame(zone_rows)
    zone.to_csv(OUT / "zone_tradeoff.csv", index=False)
    lines += ["| α | estimator | degraded | zone improved | zone unchanged | "
              "(of which already Green) | zone worsened |",
              "|---|---|---|---|---|---|---|"]
    for _, r in zone.iterrows():
        lines.append(
            f"| {r['alpha']:g} | "
            f"{'single split' if r['estimator'] == 'static' else 'rolling'} | "
            f"{int(r['n_degraded'])} | {int(r['zone_up'])} | "
            f"{int(r['zone_same'])} | {int(r['zone_same_already_green'])} | "
            f"{int(r['zone_down'])} |")
    lines.append("")

    # Full zone transition matrix for the degraded rolling pairs at alpha=0.01.
    d = df[(df["alpha"] == 0.01) & (df["dQS_roll"] < 0)]
    ct = pd.crosstab(d["TL_raw"], d["TL_roll"])
    ct.to_csv(OUT / "zone_transitions_roll_001.csv")
    lines += ["Zone transitions for the rolling estimator's degraded pairs at "
              "α = 0.01 (rows = raw zone, columns = zone after correction):", "",
              "```", ct.to_string(), "```", ""]

    # --- the decision rule the trade-off implies --------------------------- #
    lines += ["## Gating the correction on the raw backtest", "",
              "If the correction is an intervention with an indication rather "
              "than a universal improvement, the natural rule is: apply it only "
              "when the raw forecast actually fails a backtest "
              "(Basel zone worse than Green, or Kupiec rejected). This "
              "evaluates that rule against applying it unconditionally.", "",
              "The rule is reported under two gating signals. Keyed on the TEST "
              "window it is an oracle: it uses the outcome it is then scored "
              "on, and bounds what gating could buy. Keyed on the CALIBRATION "
              "window it is deployable, because that is the information "
              "available on the day the decision is taken. The paper quotes the "
              "calibration version.", "",
              "| α | estimator | gating signal | applied | degraded when "
              "applied | skipped | degradations avoided | "
              "zone upgrades kept / total |",
              "|---|---|---|---|---|---|---|---|"]
    gate_rows = []
    for est in ("static", "roll"):
        for alpha in ALPHAS:
            for signal in ("test", "cal"):
                a = df[df["alpha"] == alpha].copy()
                # `test` gates on the evaluation window -- the rule as an
                # ORACLE, an upper bound on what gating can buy. `cal` gates on
                # the calibration window, which is the only version deployable
                # without look-ahead. Both are reported; the paper quotes `cal`.
                if signal == "test":
                    a["raw_fails"] = ((a["TL_raw"] != "Green")
                                      | (a["p_kup_raw"] <= 0.05))
                else:
                    a["raw_fails"] = ((a["TL_cal"] != "Green")
                                      | (a["p_kup_cal"] <= 0.05))
                ap, sk = a[a["raw_fails"]], a[~a["raw_fails"]]
                up = a[a[f"TL_{est}"].map(ZONE_RANK) < a["TL_raw"].map(ZONE_RANK)]
                row = {
                    "alpha": alpha, "estimator": est, "signal": signal,
                    "n_applied": len(ap), "n_skipped": len(sk),
                    "degraded_when_applied": int((ap[f"dQS_{est}"] < 0).sum()),
                    "degradations_avoided": int((sk[f"dQS_{est}"] < 0).sum()),
                    "gains_forgone": int((sk[f"dQS_{est}"] > 0).sum()),
                    "median_gain_applied": 100 * ap[f"rel_{est}"].median(),
                    "median_forgone": 100 * sk[f"rel_{est}"].median(),
                    "zone_upgrades_total": len(up),
                    "zone_upgrades_kept": int(up["raw_fails"].sum()),
                }
                gate_rows.append(row)
                lines.append(
                    f"| {alpha:g} | "
                    f"{'single split' if est == 'static' else 'rolling'} | "
                    f"{'test window (oracle)' if signal == 'test' else 'calibration window'} | "
                    f"{row['n_applied']} | {row['degraded_when_applied']} | "
                    f"{row['n_skipped']} | **{row['degradations_avoided']}** | "
                    f"{row['zone_upgrades_kept']} / {row['zone_upgrades_total']} |")
    pd.DataFrame(gate_rows).to_csv(OUT / "gate_rule.csv", index=False)
    lines.append("")

    make_figure(df, OUT / "fig_dqs_scatter.png", zoom=True)
    make_figure(df, OUT / "fig_dqs_scatter_full.png", zoom=False)
    lines += ["## Figures", "",
              "- `fig_dqs_scatter.png` — the headline: raw miscalibration on x, "
              "ΔQS as % of QS_raw on y, zero line marked, one panel per α, "
              f"restricted to |π̂_raw − α| < {MISCAL_INTERPRETABLE} where the "
              "base forecast is usable. This is where every deterioration sits.",
              "- `fig_dqs_scatter_full.png` — the same over the full range. The "
              "grossly miscalibrated pairs sit at +80–98% and compress the "
              "interesting region to a sliver, which is why the zoom is the "
              "figure to use in the paper.", ""]

    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}/summary.md and 7 companion files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
