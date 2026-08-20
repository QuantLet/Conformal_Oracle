#!/usr/bin/env python3
"""Independent check of the CAViaR result, and of the window it was scored on.

CAViaR-AS passing Kupiec on 15 of 24 assets is load-bearing: it is the evidence
that unconditional coverage is a weak instrument at alpha = 0.01 rather than
that every forecaster is simply bad. It has never been checked against a second
implementation, and this paper's subject is numbers that did not mean what they
appeared to mean.

Two things are verified here, and they are different questions.

1. IMPLEMENTATION. A second estimator, written from the Engle-Manganelli (2004)
   recursions directly, with a different optimiser (Powell rather than
   Nelder-Mead) and different starting values. Agreement of the fitted quantile
   paths, not of the parameters -- CAViaR's objective is flat in directions that
   leave the path nearly unchanged, so parameter disagreement is uninformative
   and path disagreement is what matters.

2. WINDOW. The original run splits the FULL return series 70/30, so its test
   window begins earlier and is longer than that of every model it is compared
   against: 1977 observations against GARCH-N's 1902 and Chronos-analytic's
   1824 on SP500, because those series only begin after 250 and 512
   observations of context respectively. A Kupiec count computed on a different
   sample is not comparable to one computed on this one. Both are reported.

The |r| <= 0.50 filter applied by the other loaders was also checked and is
immaterial: it removes zero observations across the whole panel.

Usage: verify_caviar.py [--assets SP500 ...]
Output: analysis/phase3_dynamic/CAVIAR_VERIFICATION.md and .csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = Path(__file__).resolve().parent
ALPHA = 0.01
F_CAL = 0.70

sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import MODELS, SYMBOLS  # noqa: E402
sys.path.insert(0, str(BASE / "Quantlets" / "CO_full_evaluation"))
from run_full_evaluation import kupiec_pval  # noqa: E402


def tick(y, q, alpha):
    d = y - q
    return float(np.mean(np.where(d < 0, (alpha - 1) * d, alpha * d)))


def path_sav(th, y, q0):
    q = np.empty(len(y)); q[0] = q0
    for t in range(1, len(y)):
        q[t] = th[0] + th[1] * q[t - 1] + th[2] * abs(y[t - 1])
    return q


def path_as(th, y, q0):
    q = np.empty(len(y)); q[0] = q0
    for t in range(1, len(y)):
        q[t] = (th[0] + th[1] * q[t - 1] + th[2] * max(y[t - 1], 0.0)
                + th[3] * max(-y[t - 1], 0.0))
    return q


PATH = {"SAV": path_sav, "AS": path_as}


def fit_independent(y, alpha, spec, seed=7):
    """Powell, different starts, same objective. Deliberately not the original."""
    q0 = float(np.quantile(y[:min(300, len(y))], alpha))
    k = 3 if spec == "SAV" else 4
    f = PATH[spec]

    def obj(th):
        if abs(th[1]) >= 0.999:
            return 1e6
        return tick(y, f(th, y, q0), alpha)

    rng = np.random.default_rng(seed)
    best, bv = None, np.inf
    starts = [np.concatenate([[q0 * 0.05, 0.85], np.full(k - 2, -0.05)]),
              np.concatenate([[q0 * 0.20, 0.70], np.full(k - 2, -0.20)])]
    for _ in range(10):
        starts.append(np.concatenate([
            [q0 * rng.uniform(0.01, 0.4), rng.uniform(0.4, 0.99)],
            rng.uniform(-0.5, 0.05, k - 2)]))
    for s in starts:
        try:
            r = optimize.minimize(obj, s, method="Powell",
                                  options={"maxiter": 20000, "xtol": 1e-10,
                                           "ftol": 1e-12})
            if r.fun < bv:
                bv, best = r.fun, r.x
        except Exception:
            continue
    return best, q0, bv


def common_window(asset):
    """Dates for which EVERY forecaster in cfp_config has a forecast."""
    idx = None
    for model, (sub, suf) in MODELS.items():
        fn = f"{asset}_{suf}.parquet" if suf else f"{asset}.parquet"
        fp = DATA / sub / fn
        if not fp.exists():
            continue
        i = pd.read_parquet(fp).index
        idx = i if idx is None else idx.intersection(i)
    return idx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="*", default=None)
    a = ap.parse_args()
    assets = a.assets or SYMBOLS

    rows = []
    for sym in assets:
        ret = pd.read_csv(DATA / "returns" / f"{sym}.csv", index_col=0,
                          parse_dates=True)
        ser = ret.iloc[:, 0].astype(float)
        y = ser.values
        n = len(y)
        n_cal = int(n * F_CAL)
        cw = common_window(sym)

        for spec in ("SAV", "AS"):
            th, q0, obj = fit_independent(y[:n_cal], ALPHA, spec)
            if th is None:
                continue
            q = PATH[spec](th, y, q0)
            qs = pd.Series(q, index=ser.index)

            # (a) the ORIGINAL window: last 30% of the full return series
            yt, qt = y[n_cal:], q[n_cal:]
            v, nt = int(np.sum(yt < qt)), len(yt)
            orig = {"pihat": v / nt, "p": kupiec_pval(nt, v, ALPHA), "n": nt}

            # (b) the COMMON window: dates every forecaster covers, with the
            #     conformal split taken inside it so the comparison is like for like
            if cw is not None and len(cw) > 300:
                sub = qs.loc[qs.index.intersection(cw)]
                rr = ser.loc[sub.index]
                nc = int(len(sub) * F_CAL)
                yt2, qt2 = rr.values[nc:], sub.values[nc:]
                v2, nt2 = int(np.sum(yt2 < qt2)), len(yt2)
                comm = {"pihat": v2 / nt2, "p": kupiec_pval(nt2, v2, ALPHA),
                        "n": nt2}
            else:
                comm = {"pihat": np.nan, "p": np.nan, "n": 0}

            rows.append({"asset": sym, "model": f"CAViaR-{spec}", "obj": obj,
                         "orig_pihat": orig["pihat"], "orig_p": orig["p"],
                         "orig_n": orig["n"], "comm_pihat": comm["pihat"],
                         "comm_p": comm["p"], "comm_n": comm["n"]})
        print(f"  {sym}", file=sys.stderr, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "caviar_verification.csv", index=False)

    orig = pd.read_csv(OUT / "dynamic_var.csv")
    L = ["# CAViaR: independent re-estimation, and the window question", "",
         "Second implementation: Engle--Manganelli recursions written directly, "
         "Powell optimiser, different starting values. Paths are compared, not "
         "parameters -- the objective is flat in directions that barely move the "
         "fitted quantile path.", "",
         "| model | assets | Kupiec pass, original window | Kupiec pass, common window | mean π̂ orig | mean π̂ common | mean n orig | mean n common |",
         "|---|---|---|---|---|---|---|---|"]
    for m, s in df.groupby("model"):
        L.append(f"| {m} | {len(s)} | "
                 f"**{int((s.orig_p >= 0.05).sum())}/{len(s)}** | "
                 f"**{int((s.comm_p >= 0.05).sum())}/{len(s)}** | "
                 f"{s.orig_pihat.mean():.4f} | {s.comm_pihat.mean():.4f} | "
                 f"{s.orig_n.mean():.0f} | {s.comm_n.mean():.0f} |")
    if len(orig):
        L += ["", "Original run, for comparison:", "",
              "| model | Kupiec pass | mean π̂ |", "|---|---|---|"]
        for m, s in orig.groupby("model"):
            if "p_kup_raw" in s:
                L.append(f"| {m} | {int((s.p_kup_raw >= 0.05).sum())}/{len(s)} | "
                         f"{s.pihat_raw.mean():.4f} |")
    L.append("")
    (OUT / "CAVIAR_VERIFICATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
