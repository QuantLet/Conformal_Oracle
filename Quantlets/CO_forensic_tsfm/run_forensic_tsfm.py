"""
CO_forensic_tsfm -- run_forensic_tsfm.py
=========================================
Forensic validation of quantile-grid TSFM tail failures.
Four-panel figure: TimesFM 2.5 (left) and Moirai 2.0 (right).
Top row: returns vs native 10th pctile and extrapolated 1st pctile.
Bottom row: full predicted quantile structure.

Input:  cfp_ijf_data/timesfm25/SP500 2.parquet
        cfp_ijf_data/moirai2/SP500 2.parquet
        cfp_ijf_data/returns/SP500.csv
Output: fig_forensic_tsfm.pdf/.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.grid": False,
    "savefig.transparent": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 12,
})

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent
DATA = BASE / "cfp_ijf_data"
FIG_DIR = BASE / "figures"
SLIDE_DIR = BASE / "ICFS 2026"
OUT = SCRIPT_DIR

NAVY     = "#1A3A6E"
IDA_RED  = "#C8102E"
AMBER    = "#E07B39"
TEAL     = "#008080"
PURPLE   = "#6A0DAD"
FOREST   = "#228B22"
GREY     = "#999999"

WINDOW_START = "2022-01-01"
WINDOW_END   = "2022-12-31"


def load_data():
    ret = pd.read_csv(DATA / "returns" / "SP500.csv",
                      index_col=0, parse_dates=True)
    tf = pd.read_parquet(DATA / "timesfm25" / "SP500 2.parquet")
    mo = pd.read_parquet(DATA / "moirai2" / "SP500 2.parquet")
    return ret, tf, mo


def clip_to_window(df, start, end):
    return df.loc[start:end].copy()


def main():
    ret, tf, mo = load_data()

    ret_w = clip_to_window(ret, WINDOW_START, WINDOW_END)
    tf_w  = clip_to_window(tf, WINDOW_START, WINDOW_END)
    mo_w  = clip_to_window(mo, WINDOW_START, WINDOW_END)

    common_tf = ret_w.index.intersection(tf_w.index)
    common_mo = ret_w.index.intersection(mo_w.index)
    r_tf = ret_w.loc[common_tf, "log_return"].values * 100
    r_mo = ret_w.loc[common_mo, "log_return"].values * 100
    dates_tf = common_tf
    dates_mo = common_mo

    tf_q01  = tf_w.loc[common_tf, "VaR_0.01"].values * 100
    tf_q10  = tf_w.loc[common_tf, "VaR_0.1"].values * 100
    tf_q025 = tf_w.loc[common_tf, "VaR_0.025"].values * 100
    tf_q05  = tf_w.loc[common_tf, "VaR_0.05"].values * 100
    tf_mean = tf_w.loc[common_tf, "mean"].values * 100

    mo_q01  = mo_w.loc[common_mo, "VaR_0.01"].values * 100
    mo_q10  = mo_w.loc[common_mo, "VaR_0.1"].values * 100
    mo_q025 = mo_w.loc[common_mo, "VaR_0.025"].values * 100
    mo_q05  = mo_w.loc[common_mo, "VaR_0.05"].values * 100
    mo_mean = np.clip(mo_w.loc[common_mo, "mean"].values * 100, -10, 10)

    print(f"TimesFM window: {len(dates_tf)} days")
    print(f"Moirai  window: {len(dates_mo)} days")

    viol_tf_mean = np.mean(r_tf < tf_mean) * 100
    viol_mo_mean = np.mean(r_mo < mo_mean) * 100
    print(f"TimesFM mean violation rate: {viol_tf_mean:.1f}%")
    print(f"Moirai  mean violation rate: {viol_mo_mean:.1f}%")

    # ── Figure ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10),
                             gridspec_kw={"hspace": 0.28, "wspace": 0.22})
    fig.patch.set_alpha(0.0)

    fig.suptitle(
        "Forensic Validation: Quantile-Grid TSFM Tail Failures "
        "(S&P 500, 2022)",
        fontsize=20, fontweight="bold", y=0.98)

    # ── Top row: returns vs q10 and q01 ──
    for col, (dates, r, q10, q01, label) in enumerate([
        (dates_tf, r_tf, tf_q10, tf_q01, "TimesFM 2.5"),
        (dates_mo, r_mo, mo_q10, mo_q01, "Moirai 2.0"),
    ]):
        ax = axes[0, col]
        ax.fill_between(dates, r, 0, alpha=0.15, color=GREY, linewidth=0)
        ax.plot(dates, r, color=GREY, lw=0.4, alpha=0.6)
        ax.plot(dates, q10, color=IDA_RED, lw=1.5, label="Native 10th pctile")
        ax.plot(dates, q01, color=NAVY, lw=1.5, ls="--",
                label="Extrapolated 1st pctile")
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_title(label, fontsize=16, fontweight="bold", pad=8)
        ax.set_ylabel("Return (%)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # ── Bottom row: full quantile structure ──
    q_colors = [NAVY, TEAL, AMBER, IDA_RED, PURPLE]
    q_labels = ["1% quantile", "2.5% quantile", "5% quantile",
                "10% quantile", "Mean"]

    for col, (dates, r, q01, q025, q05, q10, mn, label) in enumerate([
        (dates_tf, r_tf, tf_q01, tf_q025, tf_q05, tf_q10, tf_mean,
         "TimesFM 2.5: predicted quantile structure"),
        (dates_mo, r_mo, mo_q01, mo_q025, mo_q05, mo_q10, mo_mean,
         "Moirai 2.0: predicted quantile structure"),
    ]):
        ax = axes[1, col]
        ax.fill_between(dates, r, 0, alpha=0.15, color=GREY, linewidth=0)
        ax.plot(dates, r, color=GREY, lw=0.4, alpha=0.6)
        for vals, c, lab in zip(
            [q01, q025, q05, q10, mn], q_colors, q_labels
        ):
            ax.plot(dates, vals, color=c, lw=1.5, label=lab)
        ax.axhline(0, color="black", lw=0.5, alpha=0.4)
        ax.set_title(label, fontsize=16, fontweight="bold", pad=8)
        ax.set_ylabel("Return (%)", fontsize=14)
        ax.set_xlabel("Date (2022)", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # ── Combined legend outside at bottom ──
    h_top, l_top = axes[0, 0].get_legend_handles_labels()
    h_bot, l_bot = axes[1, 0].get_legend_handles_labels()
    from collections import OrderedDict
    seen = OrderedDict()
    for h, l in zip(h_top + h_bot, l_top + l_bot):
        if l not in seen:
            seen[l] = h
    fig.legend(seen.values(), seen.keys(),
               loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=7, fontsize=13, frameon=False)

    # ── Save ──
    FIG_DIR.mkdir(exist_ok=True)
    for d in [OUT, FIG_DIR, SLIDE_DIR]:
        d.mkdir(exist_ok=True)
        for ext in ["pdf", "png"]:
            fig.savefig(d / f"fig_forensic_tsfm.{ext}",
                        dpi=600, bbox_inches="tight", pad_inches=0.05,
                        transparent=True)

    plt.close(fig)
    print("Saved: fig_forensic_tsfm.pdf/.png")


if __name__ == "__main__":
    main()
