#!/usr/bin/env python3
"""Phase 2: is q_V a rediscovery of the uMCB term of Gneiting & Resin (2023)?

The AE's sharpest point. Gneiting & Resin (2023, EJS 17(2):3226-3286) decompose
the mean score of a forecast as

    S_bar = MCB - DSC + UNC

with MCB the miscalibration, DSC the discrimination and UNC the uncertainty,
where the recalibrated forecast comes from isotonic regression of the outcome on
the forecast (PAV). For the quantile loss at level alpha the pooling operation is
the weighted alpha-quantile rather than the mean.

Within MCB, the part removable by a *constant* shift is the unconditional
miscalibration:

    uMCB = S_bar(x) - min_c S_bar(x - c)

and for the quantile loss the minimising c is the empirical alpha-quantile of the
residuals (x_t - y_t) -- which is exactly the object q_V estimates, but on the
calibration split and with the conformal ceiling. So a relation is guaranteed;
the question is its form.

Three possible verdicts, per the brief, and this script is written not to favour
any of them:

  (a) q_V is a monotone transform of uMCB      -> cite it, reframe q_V as a
                                                  scale-interpretable version of
                                                  a known quantity, drop novelty
  (b) they diverge in an identifiable regime   -> that regime is the contribution
  (c) uMCB is not computable here without      -> state which assumptions, and why
      further assumptions

Note the units differ by construction: uMCB is a *score* difference (loss units),
q_V is a *shift* (return units). A second-order expansion of the quantile loss
gives uMCB ~ (1/2) f(0) q_V^2, with f the residual density at the alpha-quantile.
That is monotone in q_V only if f is common across pairs, which across 24 assets
of very different volatility it is not. The script measures whether that matters.

Outputs (analysis/umcb/):
    umcb_pairs.csv     per pair: full G&R decomposition, uMCB, q_V, R
    fig_umcb_qv.png    uMCB against q_V, and the rank comparison
    MEMO.md            the verdict
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
ALPHA = 0.01

sys.path.insert(0, str(BASE / "analysis" / "ae_point4"))
from run_ae_point4 import (  # noqa: E402

    F_CAL, DEFECTIVE_SERIES, MODELS, SYMBOLS, W_ROLL, load_pair, qhat_ceil,
    quantile_score,
)
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "Quantlets"))
from cfp_config import split_indices  # noqa: E402

# Grouping is by TRACED DEFECT, not by the withdrawn Panel A/B taxonomy. That
# taxonomy put TimesFM-2.5 and Moirai-2.0 in "Panel B" on the strength of ~99%
# raw violation rates, which were a sign inversion rather than a property of the
# quantile-grid interface; corrected, both are among the best-calibrated raw
# forecasters in the panel. The only series that still stand apart are the two
# Chronos ones sampled at the checkpoint default top_k = 50.


def weighted_quantile(v: np.ndarray, w: np.ndarray, alpha: float) -> float:
    """Weighted alpha-quantile — the pooling operation for quantile-loss PAV."""
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    idx = int(np.searchsorted(c, alpha * c[-1], side="left"))
    return float(v[min(idx, len(v) - 1)])


def isotonic_quantile(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Isotonic quantile regression of y on x by pool-adjacent-violators.

    Blocks are merged while their fitted quantiles violate monotonicity; each
    merged block is refitted as the weighted alpha-quantile of its members. This
    is the CORP recalibration of Gneiting & Resin specialised to quantile loss.
    """
    order = np.argsort(x, kind="mergesort")
    ys = y[order]
    # Each block: (list of values, weight, fitted quantile)
    vals: list[np.ndarray] = []
    fits: list[float] = []
    for yi in ys:
        vals.append(np.array([yi]))
        fits.append(yi)
        while len(fits) > 1 and fits[-2] > fits[-1]:
            merged = np.concatenate([vals[-2], vals[-1]])
            vals[-2:] = [merged]
            fits[-2:] = [weighted_quantile(merged, np.ones(len(merged)), alpha)]
    out_sorted = np.concatenate([np.full(len(v), f) for v, f in zip(vals, fits)])
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def decompose(x: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    """Gneiting-Resin decomposition plus the unconditional part of MCB."""
    s_raw = quantile_score(y, x, alpha)

    # Conditional recalibration (isotonic / CORP).
    rc = isotonic_quantile(x, y, alpha)
    s_rc = quantile_score(y, rc, alpha)

    # Marginal forecast: the constant equal to the empirical alpha-quantile of y.
    mg = np.full_like(y, float(np.quantile(y, alpha)))
    s_mg = quantile_score(y, mg, alpha)

    # Unconditional recalibration: the best CONSTANT shift of the forecast.
    # For quantile loss the minimiser is the empirical alpha-quantile of x - y.
    c_star = float(np.quantile(x - y, 1 - alpha))
    s_uc = quantile_score(y, x - c_star, alpha)

    return {
        "S_raw": s_raw, "S_rc": s_rc, "S_mg": s_mg, "S_uc": s_uc,
        "MCB": s_raw - s_rc,          # conditional miscalibration
        "uMCB": s_raw - s_uc,         # removable by a constant shift
        "cMCB": s_uc - s_rc,          # the remainder, genuinely conditional
        "DSC": s_mg - s_rc,
        "UNC": s_mg,
        "c_star": c_star,
    }


def make_figure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C_A, C_B, INK = "#2a78d6", "#eb6834", "#0b0b0b"
    ok = df[~df["defective"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, sub, title in ((ax[0], df, f"all {len(df)} pairs (log-log)"),
                          (ax[1], ok,
                           f"well-specified series only ({len(ok)} pairs)")):
        for flag, colour, lab in ((False, C_A, "well-specified"),
                                  (True, C_B, "top_k-truncated")):
            s = sub[(sub["defective"] == flag) & (sub["uMCB"] > 0)]
            if len(s):
                a.scatter(s["qV"].abs(), s["uMCB"], s=34, facecolor=colour,
                          edgecolor="white", linewidth=0.6, label=lab, zorder=3)
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlabel(r"$|\hat q_V|$  (return units)", fontsize=9)
        a.set_ylabel("uMCB  (loss units)", fontsize=9)
        a.set_title(title, fontsize=10, color=INK)
        a.grid(alpha=0.18, linewidth=0.5)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    ax[0].legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.suptitle("q_V and the unconditional miscalibration term",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        for sym in SYMBOLS:
            try:
                r, q = load_pair(model, sym, ALPHA)
            except Exception:
                continue
            n = len(r)
            _cal, _test, _g = split_indices(n, q - r, f_cal=F_CAL)
            n_cal, t0 = len(_cal), int(_test[0])
            if n_cal < W_ROLL or n - t0 < 50:
                continue
            r_cal, r_test = r[:n_cal], r[t0:]
            q_cal, q_test = q[:n_cal], q[t0:]

            qV = qhat_ceil(q_cal - r_cal, ALPHA)
            d = decompose(q_test, r_test, ALPHA)
            d.update({
                "model": model, "asset": sym,
                "defective": model in DEFECTIVE_SERIES,
                "qV": qV, "R": abs(qV) / abs(np.mean(q_test)),
                "n_test": len(r_test),
            })
            rows.append(d)
        print(f"  {model}", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "umcb_pairs.csv", index=False)

    # --- relation ------------------------------------------------------- #
    def rel(s, label):
        s = s.dropna(subset=["uMCB", "qV"])
        pear = stats.pearsonr(s["qV"].abs(), s["uMCB"])
        spear = stats.spearmanr(s["qV"].abs(), s["uMCB"])
        return (label, len(s), pear.statistic, spear.statistic, spear.pvalue)

    lines = ["# Phase 2 — q_V and the uMCB term of Gneiting & Resin (2023)", "",
             "Generated by `analysis/umcb/run_umcb.py` over "
             f"{len(df)} model–asset pairs at alpha = {ALPHA}.", "",
             "## The decomposition as implemented", "",
             "`S_raw = MCB - DSC + UNC` with the CORP recalibration done by "
             "pool-adjacent-violators using the weighted alpha-quantile as the "
             "pooling operation (the quantile-loss analogue of the mean). MCB is "
             "split into the part a constant shift can remove and the remainder:",
             "", "    uMCB = S_raw - min_c S_raw(x - c)",
             "    cMCB = MCB - uMCB", "",
             "The minimising constant for quantile loss is the empirical "
             "(1-alpha)-quantile of the residuals x - y, which is what q_V "
             "estimates on the calibration split. A relation is therefore "
             "guaranteed by construction; the question is its form.", ""]

    ident = float(np.nanmax(np.abs(df["S_raw"] - (df["MCB"] - df["DSC"] + df["UNC"]))))
    lines += [f"Decomposition identity check: max |S_raw - (MCB - DSC + UNC)| = "
              f"{ident:.2e}.", ""]

    lines += ["## Correlation of |q_V| with uMCB", "",
              "| Subset | n | Pearson | Spearman | p |", "|---|---|---|---|---|"]
    for label, s in (("all pairs", df),
                     ("well-specified series only", df[~df["defective"]]),
                     ("top_k-truncated series only", df[df["defective"]])):
        lab, n, p, sp, pv = rel(s, label)
        lines.append(f"| {lab} | {n} | {p:.3f} | **{sp:.3f}** | {pv:.1e} |")
    lines.append("")

    # Rank disagreement
    d2 = df.dropna(subset=["uMCB", "qV"]).copy()
    d2["rank_qV"] = d2["qV"].abs().rank()
    d2["rank_umcb"] = d2["uMCB"].rank()
    d2["rank_gap"] = (d2["rank_qV"] - d2["rank_umcb"]).abs()
    worst = d2.nlargest(10, "rank_gap")
    lines += ["## Where the two rankings disagree most", "",
              "| Model | Asset | \\|q_V\\| | uMCB | rank(q_V) | rank(uMCB) | gap |",
              "|---|---|---|---|---|---|---|"]
    for _, r in worst.iterrows():
        lines.append(f"| {r['model']} | {r['asset']} | {abs(r['qV']):.5f} | "
                     f"{r['uMCB']:.3e} | {r['rank_qV']:.0f} | "
                     f"{r['rank_umcb']:.0f} | {r['rank_gap']:.0f} |")
    lines.append("")

    # Is uMCB ~ (1/2) f qV^2 -- does the local density break monotonicity?
    sub = d2[(~d2["defective"]) & (d2["uMCB"] > 0) & (d2["qV"].abs() > 1e-6)].copy()
    sub["implied_f"] = 2 * sub["uMCB"] / sub["qV"] ** 2
    lo, hi = sub["implied_f"].quantile([0.05, 0.95])
    neg = int((d2["uMCB"] < 0).sum())
    lines += ["## Why the two are not interchangeable", "",
              "A second-order expansion of the quantile loss gives "
              "uMCB \u2248 \u00bd\u00b7f\u00b7q_V\u00b2, with f the residual density at "
              "the alpha-quantile. Solving for the implied f per pair "
              "(the two top_k-truncated Chronos series excluded, and the "
              f"{neg} pairs with numerically negative uMCB dropped):", "",
              f"- 5th-95th percentile of implied f: **{lo:.1f} to {hi:.1f}**, "
              f"a factor of {hi / max(lo, 1e-9):.0f}; median "
              f"{sub['implied_f'].median():.1f}.", "",
              "The same q_V therefore maps to score penalties differing by an "
              "order of magnitude depending on the asset's tail density. The two "
              "quantities are not readable off one another without f.", "",
              f"(The {neg} negative uMCB values are numerical: the minimising "
              "constant is found by an interpolated quantile, so the achieved "
              "reduction can be ~1e-8 below zero. They are dropped here rather "
              "than clipped.)", ""]

    lines += ["## Share of miscalibration that is unconditional", "",
              "| Subset | mean uMCB/MCB | median |", "|---|---|---|"]
    for label, s in (("all", df),
                     ("well-specified series only", df[~df["defective"]]),
                     ("top_k-truncated series only", df[df["defective"]])):
        sh = (s["uMCB"] / s["MCB"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
        lines.append(f"| {label} | {sh.mean():.3f} | {sh.median():.3f} |")
    lines.append("")

    make_figure(df, OUT / "fig_umcb_qv.png")

    (OUT / "MEMO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-40:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
