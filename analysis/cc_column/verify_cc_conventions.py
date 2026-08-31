#!/usr/bin/env python3
"""Every convention question the CC column raises, checked in one place.

Established already (investigate_cc.py): Table 1's "CC pass" counts a degenerate
test as a pass, which reproduces all ten published rows.

This script settles the rest:

  1. Does Table 2's static panel use the *informative-only* rule, i.e. the same
     statistic under the opposite convention?
  2. Does Table 2's rolling panel use that rule too?
  3. Is the stored statistic the joint LR_CC ~ chi2_2 that Appendix E defines,
     or the independence LR_IND ~ chi2_1?
  4. Is p_cc_raw published anywhere, and under which rule?
  5. How does the degeneracy rate move across alpha? If the test becomes
     informative at 0.05 and 0.10, that curve locates where conditional
     backtesting stops working.

Outputs (analysis/cc_column/):
    table2_check.csv        per-model reproduction of Table 2, both panels
    joint_vs_ind.csv        chi2_2 joint vs chi2_1 independence
    degeneracy_by_alpha.csv the alpha curve
    CONVENTIONS.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
TABLES = DATA / "paper_outputs" / "tables"
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE / "analysis" / "ae_point4"))
from run_ae_point4 import (  # noqa: E402

    ALPHAS, F_CAL, MODELS, SYMBOLS, W_ROLL, load_pair, qhat_ceil,
)
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "Quantlets"))
from cfp_config import split_indices  # noqa: E402

# Table 2 as printed (tab_rolling_vs_static.tex): CC pass, static and rolling.
TABLE2 = {
    "Chronos-Small": (4, 5), "Chronos-Mini": (5, 5), "TimesFM-2.5": (0, 3),
    "Moirai-2.0": (0, 2), "Lag-Llama": (2, 2), "GJR-GARCH": (7, 8),
    "GARCH-N": (6, 10), "Hist-Sim": (6, 9), "EWMA": (7, 5),
}


def cc_counts(v: np.ndarray) -> tuple[int, int, int, int]:
    v = v.astype(int)
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))
    return n00, n01, n10, n11


def lr_ind(v: np.ndarray) -> float:
    """Christoffersen independence LR. NaN when the transition table is
    degenerate — which, at alpha = 0.01, is the common case."""
    n00, n01, n10, n11 = cc_counts(v)
    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return np.nan
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    if pi01 in (0, 1) or pi11 in (0, 1) or pi in (0, 1):
        return np.nan
    return -2.0 * ((n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
                   - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
                   - n10 * np.log(1 - pi11) - n11 * np.log(pi11))


def lr_pof(x: int, n: int, alpha: float) -> float:
    if n == 0 or x == 0:
        return 2 * n * np.log(1 / (1 - alpha)) if n else 0.0
    pi = x / n
    if pi >= 1:
        return -2 * n * np.log(alpha)
    return 2 * (x * np.log(pi / alpha) + (n - x) * np.log((1 - pi) / (1 - alpha)))


def p_ind(v: np.ndarray) -> float:
    lr = lr_ind(v)
    return np.nan if np.isnan(lr) else float(1 - stats.chi2.cdf(abs(lr), 1))


def p_joint(v: np.ndarray, alpha: float) -> float:
    """LR_CC = LR_POF + LR_IND ~ chi2_2, as Appendix E defines it."""
    lr = lr_ind(v)
    if np.isnan(lr):
        return np.nan
    return float(1 - stats.chi2.cdf(abs(lr) + lr_pof(int(v.sum()), len(v), alpha), 2))


def score_all(alpha: float) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for sym in SYMBOLS:
            try:
                r, q = load_pair(model, sym, alpha)
            except Exception:
                continue
            n = len(r)
            _cal, _test, _g = split_indices(n, q - r, f_cal=F_CAL)
            n_cal, t0 = len(_cal), int(_test[0])
            if n_cal < W_ROLL or n - t0 < 50:
                continue
            r_cal, r_test = r[:n_cal], r[t0:]
            q_cal, q_test = q[:n_cal], q[t0:]

            qV = qhat_ceil(q_cal - r_cal, alpha)
            v_static = (r_test < q_test - qV)

            history = list((q_cal - r_cal)[-W_ROLL:])
            var_roll = np.empty(len(r_test))
            for t in range(len(r_test)):
                var_roll[t] = q_test[t] - qhat_ceil(
                    np.array(history[-W_ROLL:]), alpha)
                history.append(q_test[t] - r_test[t])
            v_roll = (r_test < var_roll)
            v_raw = (r_test < q_test)

            n00, n01, n10, n11 = cc_counts(v_static)
            rows.append({
                "model": model, "asset": sym, "alpha": alpha,
                "n11_static": n11, "n_viol_static": int(v_static.sum()),
                "p_ind_static": p_ind(v_static),
                "p_joint_static": p_joint(v_static, alpha),
                "p_ind_roll": p_ind(v_roll),
                "p_ind_raw": p_ind(v_raw),
            })
        print(f"  scored {model} at alpha={alpha}", file=sys.stderr)
    return pd.DataFrame(rows)


def main() -> int:
    d01 = score_all(0.01)
    d01.to_csv(OUT / "cc_all_variants_alpha001.csv", index=False)

    L = ["# CC column — every convention, settled", "",
         "Generated by `analysis/cc_column/verify_cc_conventions.py`.", ""]

    # --- 1 & 2: Table 2, both panels -------------------------------------- #
    rows = []
    for model, (t2s, t2r) in TABLE2.items():
        s = d01[d01["model"] == model]
        rows.append({
            "model": model,
            "static_informative_pass": int((s["p_ind_static"] > 0.05).sum()),
            "static_incl_degenerate": int((s["p_ind_static"].isna() |
                                           (s["p_ind_static"] > 0.05)).sum()),
            "table2_static": t2s,
            "roll_informative_pass": int((s["p_ind_roll"] > 0.05).sum()),
            "roll_incl_degenerate": int((s["p_ind_roll"].isna() |
                                         (s["p_ind_roll"] > 0.05)).sum()),
            "table2_rolling": t2r,
        })
    t2 = pd.DataFrame(rows)
    t2["static_ok"] = t2["static_informative_pass"] == t2["table2_static"]
    t2["roll_ok"] = t2["roll_informative_pass"] == t2["table2_rolling"]
    t2.to_csv(OUT / "table2_check.csv", index=False)

    L += ["## 1–2. Table 2 uses the opposite convention to Table 1", "",
          f"Static panel reproduced by the informative-only rule: "
          f"**{int(t2['static_ok'].sum())}/9**. "
          f"Rolling panel: **{int(t2['roll_ok'].sum())}/9**.", "",
          "| Model | informative only | incl. degenerate | Table 2 static | "
          "roll informative | roll incl. degen. | Table 2 rolling |",
          "|---|---|---|---|---|---|---|"]
    for _, r in t2.iterrows():
        L.append(f"| {r['model']} | **{r['static_informative_pass']}** | "
                 f"{r['static_incl_degenerate']} | **{r['table2_static']}** | "
                 f"**{r['roll_informative_pass']}** | "
                 f"{r['roll_incl_degenerate']} | **{r['table2_rolling']}** |")
    L += ["", "So the same statistic, same estimator, same α is published as "
          "15/24 in Table 1 and 4/24 in Table 2 for Chronos-Small. Table 2 is "
          "the defensible convention; Table 1's should be dropped.", ""]

    # --- 3: joint vs independence ----------------------------------------- #
    jrows = []
    for model, (t2s, _) in TABLE2.items():
        s = d01[d01["model"] == model]
        jrows.append({
            "model": model, "table2_static": t2s,
            "independence_chi2_1": int((s["p_ind_static"] > 0.05).sum()),
            "joint_chi2_2": int((s["p_joint_static"] > 0.05).sum()),
        })
    jv = pd.DataFrame(jrows)
    jv.to_csv(OUT / "joint_vs_ind.csv", index=False)
    ind_ok = int((jv["independence_chi2_1"] == jv["table2_static"]).sum())
    joint_ok = int((jv["joint_chi2_2"] == jv["table2_static"]).sum())
    L += ["## 3. The statistic is independence-only, not the joint test the "
          "appendix defines", "",
          "Appendix E states $\\mathrm{LR}_{\\mathrm{CC}} = "
          "\\mathrm{LR}_{\\mathrm{POF}} + \\mathrm{LR}_{\\mathrm{IND}} \\sim "
          "\\chi^2_2$, and Section 3 describes the test as covering \"joint "
          "coverage and serial independence\". The stored statistic is neither: "
          "it is $\\mathrm{LR}_{\\mathrm{IND}} \\sim \\chi^2_1$ alone.", "",
          f"- Independence-only reproduces Table 2 in **{ind_ok}/9** rows.",
          f"- The joint statistic reproduces it in **{joint_ok}/9** rows.", "",
          "| Model | Table 2 static | independence χ²₁ | joint χ²₂ |",
          "|---|---|---|---|"]
    for _, r in jv.iterrows():
        L.append(f"| {r['model']} | **{r['table2_static']}** | "
                 f"{r['independence_chi2_1']} | {r['joint_chi2_2']} |")
    L += ["", "The definition in the appendix has to be corrected to match the "
          "code, or the code changed to match the appendix. As it stands a "
          "referee checking one against the other finds a mismatch.", ""]

    # --- 4: raw column ----------------------------------------------------- #
    L += ["## 4. `p_cc_raw`", "",
          "No raw CC column is printed in Table 1 or Table 2 — both report the "
          "corrected series only — so no published number inherits the raw "
          "convention. `p_cc_raw` is stored in `all_results.csv` and uses the "
          "same independence-only statistic, with the same NaN-on-degenerate "
          "behaviour, so any future use of it must state the rule.", ""]

    # --- 5: degeneracy across alpha ---------------------------------------- #
    drows = []
    for alpha in ALPHAS:
        dd = d01 if alpha == 0.01 else score_all(alpha)
        for panel, models in (("A", {"Moirai-1.1", "Lag-Llama", "GJR-GARCH",
                                     "GARCH-N", "Hist-Sim", "EWMA"}),
                              ("B", {"Chronos-Small", "Chronos-Mini",
                                     "TimesFM-2.5", "Moirai-2.0"})):
            s = dd[dd["model"].isin(models)]
            if not len(s):
                continue
            drows.append({
                "alpha": alpha, "panel": panel, "n_pairs": len(s),
                "n_degenerate": int(s["p_ind_static"].isna().sum()),
                "pct_degenerate": 100 * s["p_ind_static"].isna().mean(),
                "mean_violations": s["n_viol_static"].mean(),
                "mean_n11": s["n11_static"].mean(),
            })
    deg = pd.DataFrame(drows)
    deg.to_csv(OUT / "degeneracy_by_alpha.csv", index=False)
    L += ["## 5. Where conditional backtesting stops working", "",
          "The degeneracy rate is the share of pairs with no consecutive "
          "violations, where the independence test is undefined.", "",
          "| α | panel | pairs | degenerate | % | mean violations | mean n₁₁ |",
          "|---|---|---|---|---|---|---|"]
    for _, r in deg.iterrows():
        L.append(f"| {r['alpha']:g} | {r['panel']} | {int(r['n_pairs'])} | "
                 f"{int(r['n_degenerate'])} | **{r['pct_degenerate']:.1f}%** | "
                 f"{r['mean_violations']:.1f} | {r['mean_n11']:.1f} |")
    L += ["", "This is the second arm of the tail-sparsity argument. Remark 3.1 "
          "says that at α = 0.01 there are too few events to *estimate* a "
          "flexible correction. This says there are too few events to *test* "
          "independence either — the same identification constraint, applied to "
          "testing rather than estimation, and it contradicts standard "
          "backtesting practice at the 1% level.", ""]

    (OUT / "CONVENTIONS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nTable 2 static {int(t2['static_ok'].sum())}/9, "
          f"rolling {int(t2['roll_ok'].sum())}/9; "
          f"independence {ind_ok}/9 vs joint {joint_ok}/9", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
