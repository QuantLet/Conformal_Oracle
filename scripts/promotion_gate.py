#!/usr/bin/env python3
"""Promotion gate: no forecast series enters cfp_ijf_data/ without passing.

Every check here exists because something got through. Two confirmed defects and
one configuration trap were each visible for months as anomalies that were
explained rather than traced:

  sign          Moirai 2.0 / TimesFM 2.5 stored VaR as -ppf(alpha), giving a
                positive threshold and ~99% violations
  monotonicity  the same defect reversed the ordering across alpha on 100% of
                days; no check looked
  dispersion    Chronos sampled under the checkpoint default top_k = 50,
                truncating the predictive support to 50 of 4094 bins
  cardinality   dispersion alone would pass 50 atoms if they were spread wide,
                so the number of distinct values is checked separately

A failing series blocks promotion and gets a written diagnosis. A tolerance is
never widened to accommodate a series.

Usage:
    python scripts/promotion_gate.py                 # all models in cfp_ijf_data
    python scripts/promotion_gate.py --dir PATH      # a candidate series tree
    python scripts/promotion_gate.py --out FILE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "cfp_ijf_data"
ALPHAS = [0.01, 0.025, 0.05, 0.10]

sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import MODELS  # noqa: E402  single source of truth

# The shipped Chronos series are retained deliberately as the artefact of the
# top_k = 50 default, so they are EXPECTED to block. Blocking is their role;
# they are not promoted, they are exhibited. Everything else must pass.
EXHIBIT = {"Chronos-Small", "Chronos-Mini"}

CHECKS = [
    ("sign", "median VaR_a < 0 at every alpha"),
    ("monotonicity", "VaR_.01 < VaR_.025 < VaR_.05 < VaR_.10 on >= 99.9% of days"),
    ("scale", "VaR_0.01 / realised sigma in [-3.5, -1.8]"),
    ("alpha_response", "pihat(0.10)/pihat(0.01) >= 3"),
    ("coverage", "pihat in [0.2a, 5a] at every alpha"),
    ("alignment", "forecast for t uses data through t-1 only"),
    ("dispersion", "predictive std / realised sigma in [0.5, 2.0]"),
    ("cardinality", "distinct VaR_0.01 values > 5% of observations"),
    ("tail_reach", "alpha-quantile strictly above the support minimum, with "
                   ">= 5 distinct sampled values below it (needs sample paths; "
                   "n/a for series that stored quantiles only)"),
    ("extremes", "max |VaR_0.01| <= 50x the asset's own median |VaR_0.01|"),
]

# Every other check is a median or a fraction, and is therefore blind to a small
# number of catastrophic days. A GJR-GARCH-t candidate with 56 non-converged
# fits -- conditional means of +/-12000 and |VaR| up to 7.6e8 on CBU0 -- passed
# all eight of them, 24/24, because 56 days out of 5800 move no median and no
# fraction. They move every mean downstream: that series had a mean width of
# 9493 against a median of 0.008. This check exists for that failure mode.
EXTREME_MULT = 50.0

# Two checks assume a continuous predictive distribution, and are wrong for an
# estimator that returns an order statistic of a rolling window:
#
#   cardinality   the empirical 1% quantile of a 250-day window is unchanged for
#                 as long as the same extreme observations stay inside it, so a
#                 low distinct-value count is the estimator working, not a
#                 truncated sampler
#   monotonicity  adjacent alpha levels can select the SAME order statistic, so
#                 VaR_.01 == VaR_.025 is admissible; only a strict inversion is
#                 evidence of anything
#
# Scoping this by estimator class is not a tolerance being widened to admit a
# series: the checks are inapplicable, and a check that cannot fail informatively
# must say so rather than produce a verdict. Hist-Sim blocked on both before this
# was written, and the block was spurious in both cases.
# Ties across adjacent alpha are admissible (discrete support).
TIES_OK = {"Hist-Sim", "Chronos-Small-A", "Chronos-Mini-A"}
# A low distinct-value count is the estimator working, not a truncated sampler.
# This is a strictly smaller set: the analytic Chronos series pass cardinality
# 24/24 on their own merits and are not excused from it.
ORDER_STATISTIC = {"Hist-Sim"}
# Chronos read analytically is a quantile of a CATEGORICAL distribution over
# 4093 bins, so adjacent alpha can select the same bin exactly as an order
# statistic can. Admitted only because the failures were verified to be ties and
# not inversions: weak monotonicity is 1.0000 on all 24 assets for both sizes,
# with 262 tie-days on CBU0, 5-7 on DJCI and none anywhere else. Cardinality is
# NOT waived for these -- they pass it 24/24 on their own merits.
#
# The coverage failure on CBU0 is deliberately NOT scoped away: pihat(0.01) is
# 0.0559 (small) and 0.0750 (mini) against a nominal 0.01, and that is a result
# about the forecaster, not an artefact of the check. Tokenizer resolution was
# tested as the explanation and rejected -- CBU0 has ~376 bins below its 1%
# quantile against SP500's ~467. CBU0 does carry 9.0% exactly-zero returns,
# three times the next asset, and is the worst asset for ties; the chain from
# stale prices to a shallow predictive tail is associated but not traced, so it
# stays open.


def evaluate(model: str, subdir: str, suffix: str | None, root: Path) -> dict:
    sym_ok, res = 0, {c: [] for c, _ in CHECKS}
    detail = {}
    for f in sorted((root / subdir).glob("*.parquet")):
        sym = f.stem.replace(f"_{suffix}", "") if suffix else f.stem
        if suffix and not f.stem.endswith(suffix):
            continue
        fc = pd.read_parquet(f)
        rp = DATA / "returns" / f"{sym}.csv"
        if not rp.exists():
            continue
        ret = pd.read_csv(rp, index_col=0, parse_dates=True)
        ret.columns = ["r"]
        i = ret.index.intersection(fc.index)
        if len(i) < 300:
            continue
        r = ret.loc[i, "r"].values
        sd = ret.loc[i, "r"].rolling(250).std().values
        cols = [f"VaR_{a:g}" for a in ALPHAS if f"VaR_{a:g}" in fc.columns]
        if f"VaR_0.01" not in cols:
            continue
        Q = {a: fc.loc[i, f"VaR_{a:g}"].values for a in ALPHAS
             if f"VaR_{a:g}" in fc.columns}
        q1 = Q[0.01]
        m = np.isfinite(q1)
        sym_ok += 1

        res["sign"].append(all(np.nanmedian(v) < 0 for v in Q.values()))
        if len(Q) == 4:
            A = np.vstack([Q[a] for a in ALPHAS]).T
            g = np.all(np.isfinite(A), axis=1)
            d = np.diff(A[g], axis=1)
            # order-statistic estimators may tie across adjacent alpha; only a
            # strict inversion is evidence. See ORDER_STATISTIC.
            ok = (d >= 0) if model in TIES_OK else (d > 0)
            res["monotonicity"].append(
                float(np.mean(np.all(ok, axis=1))) >= 0.999)
        ok = m & np.isfinite(sd) & (sd > 0)
        mult = np.median(q1[ok] / sd[ok])
        res["scale"].append(-3.5 <= mult <= -1.8)
        pis = {a: float(np.mean(r[np.isfinite(Q[a])] < Q[a][np.isfinite(Q[a])]))
               for a in Q}
        res["alpha_response"].append(
            pis.get(0.10, np.nan) / pis[0.01] >= 3 if pis[0.01] > 0 else False)
        res["coverage"].append(all(0.2 * a <= pis[a] <= 5 * a for a in pis))
        # alignment: a forecast using r_t would correlate with it far too strongly
        res["alignment"].append(
            abs(np.corrcoef(q1[ok], r[ok])[0, 1]) < 0.30)
        if "std" in fc.columns:
            ps = fc.loc[i, "std"].values
            k = ok & np.isfinite(ps) & (ps > 0)
            res["dispersion"].append(0.5 <= np.median(ps[k] / sd[k]) <= 2.0)
        if model not in ORDER_STATISTIC:
            res["cardinality"].append(
                len(np.unique(np.round(q1[m], 10))) > 0.05 * m.sum())
        aq = np.abs(q1[m])
        med_aq = float(np.median(aq)) if aq.size else np.nan
        res["extremes"].append(
            bool(med_aq > 0 and float(np.max(aq)) <= EXTREME_MULT * med_aq))
        # tail_reach needs the sampled support, which the shipped series did not
        # retain. Cardinality alone would pass 50 atoms spread wide; this is the
        # condition that actually fails when a sampler cannot reach the tail.
        sp = f.parent / "samples" / f"{sym}.npy"
        if sp.exists():
            paths = np.load(sp)
            ok_t = []
            for row_s in paths:
                below = np.unique(row_s[row_s < np.percentile(row_s, 1)])
                ok_t.append(len(below) >= 5)
            res["tail_reach"].append(bool(np.mean(ok_t) >= 0.999))
        detail[sym] = {"scale": float(mult), "pi001": pis[0.01]}
    out = {"model": model, "n_assets": sym_ok}
    for c, _ in CHECKS:
        vals = res[c]
        out[c] = (f"{int(np.sum(vals))}/{len(vals)}" if vals else "n/a")
        out[c + "_pass"] = bool(vals) and all(vals)
    out["PASS"] = all(out[c + "_pass"] for c, _ in CHECKS
                      if out[c] != "n/a")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DATA)
    ap.add_argument("--out", type=Path,
                    default=BASE / "analysis" / "provenance" / "PROMOTION_GATE.md")
    a = ap.parse_args()

    rows = [evaluate(m, sub, suf, a.dir) for m, (sub, suf) in MODELS.items()]
    df = pd.DataFrame(rows)
    df.to_csv(a.out.with_suffix(".csv"), index=False)

    L = ["# Promotion gate", "",
         f"Series tree: `{a.dir}`. A failing series blocks promotion and gets a "
         "written diagnosis; no tolerance is widened to accommodate a series.", "",
         "| Check | Condition |", "|---|---|"]
    L += [f"| `{c}` | {d} |" for c, d in CHECKS]
    L += ["", "| Model | n | " + " | ".join(f"`{c}`" for c, _ in CHECKS) + " | verdict |",
          "|---|---|" + "---|" * (len(CHECKS) + 1)]
    for _, r in df.iterrows():
        cells = " | ".join(
            (f"**{r[c]}**" if not r[c + '_pass'] and r[c] != "n/a" else r[c])
            for c, _ in CHECKS)
        L.append(f"| {r['model']} | {r['n_assets']} | {cells} | "
                 f"{'PASS' if r['PASS'] else '**BLOCK**'} |")
    n_block = int((~df["PASS"]).sum())
    L += ["", f"**{n_block} of {len(df)} series block.**", ""]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
