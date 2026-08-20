#!/usr/bin/env python3
"""Correct the sign error in the TimesFM 2.5 and Moirai 2.0 VaR series.

    stored:    VaR_alpha = -student_t.ppf(alpha, df, loc=mu, scale=sigma)
    corrected: VaR_alpha =  student_t.ppf(alpha, df, loc=mu, scale=sigma)

Recomputed from the fitted parameters already in each parquet (`df_student`,
`mean`, `std`), which are finite for 100% of observations, so no re-inference is
needed for this step.

The superseded series are preserved verbatim alongside the corrected ones. This
script does not touch any table, figure or claim; it only produces the corrected
forecasts and reports what they do to coverage.

Outcome readings were fixed in advance: see PREREGISTRATION.md.

Outputs (analysis/recompute/):
    corrected/<model>/<asset>.parquet   corrected series
    superseded/<model>/<asset>.parquet  the shipped series, verbatim
    RECOMPUTE.md
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = Path(__file__).resolve().parent
ALPHAS = [0.01, 0.025, 0.05, 0.1]
AFFECTED = {"Moirai-2.0": "moirai2", "TimesFM-2.5": "timesfm25"}

sys.path.insert(0, str(BASE / "analysis" / "ae_point4"))
from run_ae_point4 import SYMBOLS, kupiec_p, traffic_light  # noqa: E402


def main() -> int:
    (OUT / "corrected").mkdir(parents=True, exist_ok=True)
    (OUT / "superseded").mkdir(parents=True, exist_ok=True)
    rows = []
    for model, sub in AFFECTED.items():
        (OUT / "corrected" / sub).mkdir(exist_ok=True)
        (OUT / "superseded" / sub).mkdir(exist_ok=True)
        for sym in SYMBOLS:
            fp = DATA / sub / f"{sym}.parquet"
            if not fp.exists():
                continue
            fc = pd.read_parquet(fp)
            shutil.copy2(fp, OUT / "superseded" / sub / f"{sym}.parquet")

            new = fc.copy()
            for a in ALPHAS:
                new[f"VaR_{a:g}"] = student_t.ppf(
                    a, df=fc["df_student"].values,
                    loc=fc["mean"].values, scale=fc["std"].values)
            new.to_parquet(OUT / "corrected" / sub / f"{sym}.parquet")

            ret = pd.read_csv(DATA / "returns" / f"{sym}.csv", index_col=0,
                              parse_dates=True)
            ret.columns = ["r"]
            common = ret.index.intersection(fc.index)
            r = ret.loc[common, "r"].values
            for label, src in (("stored", fc), ("corrected", new)):
                q = src.loc[common, "VaR_0.01"].values
                m = np.isfinite(q)
                v = int(np.sum(r[m] < q[m]))
                n = int(m.sum())
                A = np.vstack([src.loc[common, f"VaR_{a:g}"].values for a in ALPHAS]).T
                good = np.all(np.isfinite(A), axis=1)
                rows.append({
                    "model": model, "asset": sym, "series": label,
                    "n": n, "pihat": v / n,
                    "p_kupiec": kupiec_p(v, n, 0.01),
                    "TL": traffic_light(v, n),
                    "median_VaR001": float(np.median(q[m])),
                    "frac_monotone": float(np.mean(
                        np.all(np.diff(A[good], axis=1) > 0, axis=1))),
                })
        print(f"  {model}", file=sys.stderr)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "recompute_coverage.csv", index=False)
    g = df.groupby(["model", "series"]).agg(
        pihat=("pihat", "mean"), median_VaR001=("median_VaR001", "median"),
        monotone=("frac_monotone", "mean"),
        green=("TL", lambda x: int((x == "Green").sum())),
        kup_pass=("p_kupiec", lambda x: int((x > 0.05).sum())),
        n_assets=("asset", "size")).reset_index()

    L = ["# Corrected TSFM VaR series", "",
         "Sign error corrected from stored Student-$t$ parameters; no "
         "re-inference. Readings fixed in advance in `PREREGISTRATION.md`.", "",
         "| Model | series | mean π̂ | median VaR₀.₀₁ | monotone | Kupiec pass | Green |",
         "|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        L.append(f"| {r['model']} | {r['series']} | **{r['pihat']:.4f}** | "
                 f"{r['median_VaR001']:+.5f} | {100 * r['monotone']:.1f}% | "
                 f"{r['kup_pass']}/{r['n_assets']} | {r['green']}/{r['n_assets']} |")
    L.append("")

    # Reference: the sample-based and classical models, unaffected.
    L += ["For comparison, unaffected models at α = 0.01 (from the audit): "
          "Chronos-Mini 0.4188, Chronos-Small 0.3884, Lag-Llama 0.0294, "
          "Moirai-1.1 0.0154, Hist-Sim 0.0158, GARCH-N 0.0193, EWMA 0.0208, "
          "GJR-GARCH 0.0042.", ""]
    (OUT / "RECOMPUTE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
