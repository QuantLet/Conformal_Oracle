#!/usr/bin/env python3
"""E3a: run the structural gate on the ML series, blind.

Written and committed BEFORE the panel finished, so the checks cannot have been
shaped by the series they judge. The gate's own alpha-response and
coverage-plausibility bands compute exceedance rates internally -- the manuscript
says so, they are the two checks that need an evaluation window. What must not
happen before this script runs is a *backtest*: no Kupiec, no Christoffersen, no
Basel zone. Those come afterwards.

Conditions are transcribed from Supplement S.11 of the manuscript. Scoping is
declared in Amendment 3 of drafts/prereg_ml.md and applied here by estimator
class, not by looking at the result.

Reading (v) of Amendment 2 is evaluated at the end: the lower edge of the scale
band, -3.500, has never bound on the 312 cells of the sequence panel.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
RETURNS = BASE / "cfp_ijf_data" / "returns"
SERIES = HERE / "series"
ALPHAS = [0.01, 0.025, 0.05, 0.10]

# estimator class per config, declared in Amendment 3 before the run
CLASS = {"lgbm_default": "leaf", "lgbm_pooled": "leaf",
         "qrf_default": "leaf", "qrf_pooled": "leaf"}
# checks declared inapplicable to leaf estimators, before seeing any output
INAPPLICABLE = {"leaf": {"support_cardinality", "tail_reach"}}
# LightGBM emits a quantile only; there is no predictive distribution to take an
# s.d. of. Declared, not failed.
NO_PREDICTIVE_SD = {"lgbm_default", "lgbm_pooled"}

SCALE_BAND = (-3.5, -1.8)


def rolling_sd(r: pd.Series, w: int = 250) -> pd.Series:
    return r.rolling(w).std()


def gate_one(cfg: str, asset: str) -> dict:
    q = pd.read_parquet(SERIES / cfg / f"{asset}.parquet")
    r = pd.read_csv(RETURNS / f"{asset}.csv", parse_dates=["date"]).set_index("date")["log_return"]
    d = q.join(r, how="inner").dropna()
    if len(d) < 300:
        return {}
    lo = {a: d[f"VaR_{a}"].values for a in ALPHAS}
    y = d["log_return"].values
    sd = rolling_sd(d["log_return"]).values
    m = np.isfinite(sd) & (sd > 0)

    out = {"config": cfg, "asset": asset, "n": int(len(d))}
    # 1 sign
    out["sign"] = bool(all(np.median(lo[a]) < 0 for a in ALPHAS))
    # 2 monotonicity across alpha
    ok = np.ones(len(d), bool)
    for a, b in zip(ALPHAS, ALPHAS[1:]):
        ok &= lo[a] <= lo[b] + 1e-15
    out["monotonicity"] = bool(ok.mean() >= 0.999)
    # 3 alignment proxy
    out["alignment"] = bool(abs(np.corrcoef(lo[0.01], y)[0, 1]) < 0.30)
    # 5 extremes
    med = np.median(np.abs(lo[0.01]))
    out["extremes"] = bool(np.max(np.abs(lo[0.01])) <= 50 * med) if med > 0 else False
    # 6 scale
    ratio = float(np.median(lo[0.01][m] / sd[m]))
    out["scale_ratio"] = ratio
    out["scale"] = bool(SCALE_BAND[0] <= ratio <= SCALE_BAND[1])
    # 8 alpha responsiveness, 9 coverage plausibility
    pi = {a: float((y < lo[a]).mean()) for a in ALPHAS}
    out.update({f"pi_{a}": pi[a] for a in ALPHAS})
    out["alpha_response"] = bool(pi[0.10] / pi[0.01] >= 3) if pi[0.01] > 0 else False
    out["coverage_plausibility"] = bool(all(0.2 * a <= pi[a] <= 5 * a for a in ALPHAS))
    return out


def main() -> int:
    rows = []
    for cfg in sorted(CLASS):
        for f in sorted((SERIES / cfg).glob("*.parquet")):
            g = gate_one(cfg, f.stem)
            if g:
                rows.append(g)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "gate_blind_cells.csv", index=False)

    checks = ["sign", "monotonicity", "alignment", "extremes", "scale",
              "alpha_response", "coverage_plausibility"]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    L = [f"# Blind gate verdict on the ML series\n",
         f"Run {ts}. The gate script was committed before the panel finished.",
         "No Kupiec, Christoffersen or Basel figure has been computed for these",
         "series at the time of writing.\n",
         "## Series hashes\n"]
    for cfg in sorted(CLASS):
        h = hashlib.sha256()
        for f in sorted((SERIES / cfg).glob("*.parquet")):
            h.update(f.read_bytes())
        L.append(f"    {cfg:15s} {h.hexdigest()}")
    L.append("\n## Verdict, cells failing each check (of 24 assets)\n")
    L.append("| config | " + " | ".join(checks) + " | blocked |")
    L.append("|" + "---|" * (len(checks) + 2))
    for cfg, g in df.groupby("config"):
        fails = [int((~g[c]).sum()) for c in checks]
        L.append(f"| `{cfg}` | " + " | ".join(str(x) for x in fails) +
                 f" | {'YES' if any(fails) else 'no'} |")
    L.append("\nDeclared inapplicable before the run: support cardinality and tail")
    L.append("reach for every leaf-based estimator; predictive dispersion for the")
    L.append("two LightGBM configurations, which emit a quantile and no")
    L.append("distribution. Recorded as inapplicable, not as passes.\n")
    L.append("## Reading (v): the lower edge of the scale band\n")
    lo_edge = float(df.scale_ratio.min())
    who = df.loc[df.scale_ratio.idxmin()]
    L.append(f"Most negative scale ratio across all {len(df)} ML cells: "
             f"**{lo_edge:.3f}** (`{who.config}` on {who.asset}).\n")
    if lo_edge < -3.5:
        L.append("**Band 1: the lower edge binds for the first time.** Section 7.2's")
        L.append("statement that it blocks 0 of 312 cells is superseded; the block")
        L.append("count becomes ten checks exercised rather than nine.")
    elif lo_edge < -3.0:
        L.append(f"**Band 2.** The most over-conservative cell in the panel reaches "
                 f"{lo_edge:.3f}, and the lower edge remains unexercised.")
    else:
        L.append(f"**Band 3: reading (v) returns nothing**, which is the outcome the")
        L.append(f"amendment recorded as expected. Margin from the edge: "
                 f"{abs(-3.5 - lo_edge):.3f}.")
    (HERE / "GATE_BLIND.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
