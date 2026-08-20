#!/usr/bin/env python3
"""Is the ~99% violation rate a property of quantile-grid interfaces, or a sign
error in the forecast-generation step?

Three checks, all from stored files, no grids or samples required.

1. SIGN. Moirai 1.1 writes VaR as np.percentile(samples, alpha*100) -- unnegated,
   so negative for a left tail. Moirai 2.0 and TimesFM 2.5 write
   -student_t.ppf(alpha, ...) -- negated, so positive. Report the sign of the
   stored VaR_0.01 per model.

2. MONOTONICITY. A correct left-tail VaR satisfies
       VaR_0.01 < VaR_0.025 < VaR_0.05 < VaR_0.10.
   A sign flip reverses this exactly. This is decisive on its own: a positive
   VaR_0.01 might be an odd but coherent convention; a reversed ordering cannot.

3. PREDICTED RATE. If the diagnosis holds, the observed violation rate is just
   the empirical CDF of returns evaluated at the stored (positive) threshold.
   Compute that prediction and compare against the published 98.8% / 99.0%. A
   match to within a few tenths of a percent across 24 assets confirms the
   diagnosis quantitatively; a mismatch means something else is going on.

4. COLLINEARITY. Does membership of the paper's Panel B coincide exactly with
   the Student-t code path? If the partition is collinear with a code branch,
   the classification result is a diff, not a finding.

Output: analysis/interface/SIGN_DIAGNOSTIC.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE / "analysis" / "ae_point4"))
from run_ae_point4 import MODELS, SYMBOLS  # noqa: E402

LEVELS = ["VaR_0.01", "VaR_0.025", "VaR_0.05", "VaR_0.1"]
# Which construction wrote the file, read from the pipeline notebooks.
CODE_PATH = {
    "Moirai-2.0": "student_t_negated", "TimesFM-2.5": "student_t_negated",
    "Moirai-1.1": "percentile_raw", "Chronos-Small": "percentile_raw",
    "Chronos-Mini": "percentile_raw", "Lag-Llama": "percentile_raw",
    "GJR-GARCH": "benchmark", "GARCH-N": "benchmark",
    "Hist-Sim": "benchmark", "EWMA": "benchmark",
}
# Was PANEL_B, the "effective replacement" half of the withdrawn taxonomy. The
# membership was set by raw violation rates that turned out to be two defects:
# the sign inversion (TimesFM-2.5, Moirai-2.0) and top_k = 50 (both Chronos).
# Kept as an explicit list of the series this diagnostic was built to examine,
# under a name that says so.
SUSPECT_SERIES = {"Chronos-Small", "Chronos-Mini", "TimesFM-2.5", "Moirai-2.0"}


def main() -> int:
    rows = []
    for model, (subdir, suffix) in MODELS.items():
        for sym in SYMBOLS:
            name = f"{sym}.parquet" if suffix is None else f"{sym}_{suffix}.parquet"
            fp = DATA / subdir / name
            if not fp.exists():
                continue
            fc = pd.read_parquet(fp)
            ret = pd.read_csv(DATA / "returns" / f"{sym}.csv", index_col=0,
                              parse_dates=True)
            ret.columns = ["r"]
            common = ret.index.intersection(fc.index)
            r = ret.loc[common, "r"].values
            q = {L: fc.loc[common, L].values for L in LEVELS if L in fc.columns}
            if "VaR_0.01" not in q:
                continue
            v1 = q["VaR_0.01"]
            m = np.isfinite(v1)
            # monotone increasing in alpha is CORRECT for a left tail
            mono_ok = mono_rev = np.nan
            if len(q) == 4:
                A = np.vstack([q[L] for L in LEVELS]).T
                good = np.all(np.isfinite(A), axis=1)
                A = A[good]
                mono_ok = float(np.mean(np.all(np.diff(A, axis=1) > 0, axis=1)))
                mono_rev = float(np.mean(np.all(np.diff(A, axis=1) < 0, axis=1)))
            # predicted violation rate = empirical CDF of r at the stored threshold
            pred = float(np.mean(r[m] < v1[m]))
            rows.append({
                "model": model, "asset": sym, "code_path": CODE_PATH[model],
                "suspect": model in SUSPECT_SERIES,
                "median_VaR001": float(np.median(v1[m])),
                "pct_positive": float(np.mean(v1[m] > 0)),
                "frac_monotone_correct": mono_ok,
                "frac_monotone_reversed": mono_rev,
                "pihat_at_stored": pred,
            })
        print(f"  {model}", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sign_diagnostic.csv", index=False)

    g = df.groupby(["model", "code_path", "panel"]).agg(
        n=("asset", "size"),
        median_VaR001=("median_VaR001", "median"),
        pct_positive=("pct_positive", "mean"),
        mono_correct=("frac_monotone_correct", "mean"),
        mono_reversed=("frac_monotone_reversed", "mean"),
        pihat=("pihat_at_stored", "mean"),
    ).reset_index().sort_values("pihat", ascending=False)

    L = ["# Sign diagnostic for the TSFM VaR construction", "",
         "All figures from stored forecast files; no grids or samples needed.", "",
         "| Model | code path | Panel | median VaR₀.₀₁ | % positive | monotone correct | monotone reversed | π̂ |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        L.append(f"| {r['model']} | `{r['code_path']}` | {r['panel']} | "
                 f"{r['median_VaR001']:+.5f} | {100 * r['pct_positive']:.1f}% | "
                 f"{100 * r['mono_correct']:.1f}% | "
                 f"**{100 * r['mono_reversed']:.1f}%** | {r['pihat']:.4f} |")
    L.append("")

    neg = g[g["code_path"] == "student_t_negated"]
    oth = g[g["code_path"] != "student_t_negated"]
    L += ["## 1. Sign", "",
          f"- `student_t_negated` path: stored VaR₀.₀₁ positive on "
          f"**{100 * neg['pct_positive'].mean():.1f}%** of observations.",
          f"- every other path: positive on "
          f"{100 * oth['pct_positive'].mean():.1f}%.", "",
          "## 2. Monotonicity", "",
          f"- `student_t_negated`: ordering across α is **reversed** on "
          f"**{100 * neg['mono_reversed'].mean():.1f}%** of days, correct on "
          f"{100 * neg['mono_correct'].mean():.1f}%.",
          f"- every other path: correct on "
          f"{100 * oth['mono_correct'].mean():.1f}%, reversed on "
          f"{100 * oth['mono_reversed'].mean():.1f}%.", ""]

    L += ["## 3. Predicted versus published violation rate", "",
          "| Model | π̂ predicted by the stored threshold | published |",
          "|---|---|---|"]
    pub = {"Moirai-2.0": 0.988, "TimesFM-2.5": 0.990}
    for m, v in pub.items():
        got = float(g[g["model"] == m]["pihat"].iloc[0])
        L.append(f"| {m} | **{got:.4f}** | {v:.3f} | ")
    L.append("")

    L += ["## 4. Is Panel B collinear with the code path?", "",
          "| Model | Panel | code path |", "|---|---|---|"]
    for _, r in g.sort_values(["panel", "model"]).iterrows():
        L.append(f"| {r['model']} | {r['panel']} | `{r['code_path']}` |")
    b = set(g[g["panel"] == "B"]["model"])
    sp = set(g[g["code_path"] == "student_t_negated"]["model"])
    L += ["", f"Panel B = {sorted(b)}", f"Student-t path = {sorted(sp)}",
          "", ("**Collinear**: Panel B is exactly the Student-t path."
               if b == sp else
               f"**Not collinear**: Panel B contains {sorted(b - sp)} which do "
               f"not use the Student-t path. The partition is therefore not a "
               f"pure code-path artefact, though {sorted(b & sp)} are affected."), ""]

    (OUT / "SIGN_DIAGNOSTIC.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
