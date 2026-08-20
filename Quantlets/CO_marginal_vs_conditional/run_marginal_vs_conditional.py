#!/usr/bin/env python3
"""What the conformal guarantee delivers, and what it does not.

Theorem 3.3 gives finite-sample MARGINAL coverage, distribution-free. It says
nothing about conditional coverage, and distribution-free conditional coverage
is not merely unproven but unattainable. That distinction is usually made in
the abstract. Here it is measured, using the sharpest case available: a
forecaster whose sign was inverted.

Split-conformal subtracts a single constant from the whole series,

    var_cp(t) = v(t) - qV,     qV = s_(k),  k = ceil((n_cal + 1)(1 - alpha))

so qV can fix the violation *count* but cannot undo an error whose size moves
with sigma_t. On the sign-flipped Moirai-2.0 series the recalibrated forecasts
therefore attain nominal coverage while responding to volatility with the wrong
sign: they become LESS conservative as volatility rises.

Panel (a)  per-asset corr(var_cp, sigma_t), the two input series paired
Panel (b)  mean recalibrated VaR by volatility decile, pooled across assets

Both inputs are the same forecaster, the same recalibration and the same test
window. The only difference is the sign defect documented in
`analysis/recompute/SIGN_VERIFICATION.md`.

Colour: Okabe--Ito, a published colour-vision-deficiency-safe qualitative
palette. Series are additionally distinguished by marker and line style, so the
figure carries its information in greyscale print as well.

Inputs
    analysis/recompute/superseded/moirai2/<asset>.parquet   as submitted
    analysis/recompute/corrected/moirai2/<asset>.parquet    sign corrected
Outputs
    fig_marginal_vs_conditional.{png,pdf}
    tab_marginal_vs_conditional.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
DATA = BASE / "cfp_ijf_data"
SUPER = BASE / "analysis" / "recompute" / "superseded" / "moirai2"
CORR = BASE / "analysis" / "recompute" / "corrected" / "moirai2"
ALPHA = 0.01
F_CAL = 0.70

sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import SYMBOLS  # noqa: E402

# Okabe--Ito: vermillion and blue, the canonical CVD-safe opposition.
C_FLIP, C_OK = "#D55E00", "#0072B2"
LBL_FLIP = "as submitted (sign inverted)"
LBL_OK = "sign corrected"

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.linewidth": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 160, "savefig.bbox": "tight",
})


def recalibrate(r, v, alpha=ALPHA, f_cal=F_CAL):
    """Split conformal, exactly as the evaluation pipeline does it."""
    n = len(r)
    n_cal = int(n * f_cal)
    s = np.sort(v[:n_cal] - r[:n_cal])
    k = min(int(np.ceil((n_cal + 1) * (1 - alpha))) - 1, n_cal - 1)
    qV = float(s[k])
    return v[n_cal:] - qV, n_cal, qV


def main() -> int:
    rows, pooled = [], []
    for sym in SYMBOLS:
        fs, fc = SUPER / f"{sym}.parquet", CORR / f"{sym}.parquet"
        if not (fs.exists() and fc.exists()):
            continue
        ret = pd.read_csv(DATA / "returns" / f"{sym}.csv", index_col=0,
                          parse_dates=True)
        ret.columns = ["r"]
        a = pd.read_parquet(fs)
        b = pd.read_parquet(fc)
        i = ret.index.intersection(a.index).intersection(b.index).sort_values()
        r = ret.loc[i, "r"].values
        sd = ret.loc[i, "r"].rolling(250).std().values
        rec = {}
        for lab, src in ((LBL_FLIP, a), (LBL_OK, b)):
            cp, ncal, qV = recalibrate(r, src.loc[i, f"VaR_{ALPHA:g}"].values)
            st, rt = sd[ncal:], r[ncal:]
            m = np.isfinite(cp) & np.isfinite(st) & (st > 0)
            rec[lab] = {"corr": float(np.corrcoef(cp[m], st[m])[0, 1]),
                        "pihat": float(np.mean(rt[m] < cp[m])), "qV": qV}
            # pooled, scaled by each asset's own sigma so assets are comparable
            pooled.append(pd.DataFrame({
                "series": lab, "asset": sym,
                "sigma_rank": pd.Series(st[m]).rank(pct=True).values,
                "var_scaled": cp[m] / np.median(st[m])}))
        rows.append({"asset": sym,
                     "corr_flip": rec[LBL_FLIP]["corr"], "pi_flip": rec[LBL_FLIP]["pihat"],
                     "corr_ok": rec[LBL_OK]["corr"], "pi_ok": rec[LBL_OK]["pihat"]})

    d = pd.DataFrame(rows).sort_values("corr_ok").reset_index(drop=True)
    d.to_csv(HERE / "tab_marginal_vs_conditional.csv", index=False)
    P = pd.concat(pooled, ignore_index=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.9),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- (a) paired per-asset correlations ------------------------------
    y = np.arange(len(d))
    ax1.hlines(y, d["corr_flip"], d["corr_ok"], color="0.75", lw=0.8, zorder=1)
    ax1.scatter(d["corr_flip"], y, s=26, color=C_FLIP, marker="o",
                zorder=3, label=LBL_FLIP, edgecolor="white", linewidth=0.5)
    ax1.scatter(d["corr_ok"], y, s=30, color=C_OK, marker="D",
                zorder=3, label=LBL_OK, edgecolor="white", linewidth=0.5)
    ax1.axvline(0, color="0.25", lw=0.7, zorder=2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(d["asset"], fontsize=6.5)
    ax1.set_xlabel(r"corr$\,(\mathrm{VaR}^{\mathrm{cp}}_t,\ \sigma_t)$")
    ax1.set_title("(a) volatility response, per asset", fontsize=9, loc="left")
    ax1.set_xlim(-0.95, 0.95)

    # ---- (b) pooled response by volatility rank -------------------------
    P["bin"] = pd.cut(P["sigma_rank"], np.linspace(0, 1, 11),
                      labels=False, include_lowest=True)
    for lab, c, mk, ls in ((LBL_FLIP, C_FLIP, "o", "--"), (LBL_OK, C_OK, "D", "-")):
        s = P[P["series"] == lab].groupby("bin")["var_scaled"].mean()
        ax2.plot(s.index + 0.5, s.values, color=c, marker=mk, ms=4.2, lw=1.6,
                 ls=ls, label=lab, markeredgecolor="white", markeredgewidth=0.5)
    ax2.set_xlabel(r"volatility decile of $\sigma_t$  (1 = calmest)")
    ax2.set_ylabel(r"mean $\mathrm{VaR}^{\mathrm{cp}}_t\,/\,\mathrm{med}(\sigma)$")
    ax2.set_title("(b) pooled across 24 assets", fontsize=9, loc="left")
    ax2.set_xticks(np.arange(10) + 0.5)
    ax2.set_xticklabels(range(1, 11), fontsize=7)
    ax2.axhline(0, color="0.25", lw=0.7)

    h, lab = ax1.get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.035))
    note = (f"marginal coverage attained either way:  "
            f"$\\hat\\pi$ = {d['pi_flip'].mean():.4f} (inverted)  vs  "
            f"{d['pi_ok'].mean():.4f} (corrected),  nominal {ALPHA:g}")
    fig.text(0.5, -0.10, note, ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig_marginal_vs_conditional.{ext}")

    print(f"assets                 : {len(d)}")
    print(f"corr, inverted input   : {d['corr_flip'].mean():+.3f}  "
          f"(positive on {int((d['corr_flip'] > 0).sum())}/{len(d)})")
    print(f"corr, corrected input  : {d['corr_ok'].mean():+.3f}  "
          f"(negative on {int((d['corr_ok'] < 0).sum())}/{len(d)})")
    print(f"pihat inverted / corrected : {d['pi_flip'].mean():.4f} / "
          f"{d['pi_ok'].mean():.4f}   (nominal {ALPHA:g})")
    print(f"wrote {HERE / 'fig_marginal_vs_conditional.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
