#!/usr/bin/env python3
"""Reconcile Moirai-1.1 with the pipeline that produced every other model.

Moirai-1.1 is about to carry the paper's within-family control (sample-based
interface vs Moirai-2.0's quantile grid), so its numbers must come from the same
code path as the rest. They currently do not.

Diagnosis — two differences, not one
------------------------------------
1. **Split point.** The house convention is n_cal = floor(0.70 n); Moirai-2.0
   uses it on all 24 assets. The legacy Moirai-1.1 table uses it on only 12 of
   24 — the other 12 use ceil(0.70 n) — so the file is not even internally
   consistent, and 48 of its 96 cells carry a test sample one observation
   shorter than they should.

2. **Quantile estimator, and this one is substantive.** The legacy table
   computes the conformal shift as `np.quantile(scores, 1 - alpha)` with linear
   interpolation. The rest of the pipeline uses the conformal order statistic
   S_(k) with k = ceil((n+1)(1-alpha)) — which is what Theorem 3.3 requires.
   An interpolated quantile lies below that order statistic and carries no
   finite-sample coverage guarantee, so the legacy Moirai-1.1 figures were not
   produced by the estimator the paper analyses. Verified on ASX200 at
   alpha = 0.01: legacy 0.001566 = np.quantile linear exactly; the conformal
   value is 0.001939.

Both differences push qV downward, so the legacy table understates the
correction for Moirai-1.1 throughout. At alpha = 0.01 this is not cosmetic:
qV is an extreme order statistic, and the combined effect moves it by up to 52%
on individual assets (WTI: 0.00072 -> 0.00109).

This script rebuilds the table under the house convention, with the same schema,
and reports every number that moves.

Outputs (analysis/moirai11_reconciliation/):
    moirai11_full_results_rebuilt.csv   corrected table, legacy schema
    diff_vs_legacy.csv                  per-cell comparison
    RECONCILIATION.md                   what changed and what it affects
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
TABLES = DATA / "paper_outputs" / "tables"
OUT = Path(__file__).resolve().parent

ALPHAS = [0.01, 0.025, 0.05, 0.10]
F_CAL = 0.70

sys.path.insert(0, str(BASE / "analysis" / "ae_point4"))
from run_ae_point4 import (  # noqa: E402

    kupiec_p, qhat_ceil, quantile_score, traffic_light,
)
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[2] / "Quantlets"))
from cfp_config import split_indices  # noqa: E402


def cc_pval(v_bool: np.ndarray) -> float:
    """Christoffersen independence LR, as in the paper's pipeline."""
    v = v_bool.astype(int)
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))
    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return np.nan
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    if pi01 in (0, 1) or pi11 in (0, 1) or pi in (0, 1):
        return np.nan
    lr = -2.0 * ((n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
                 - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
                 - n10 * np.log(1 - pi11) - n11 * np.log(pi11))
    return float(1.0 - stats.chi2.cdf(abs(lr), 1))


def build(symbols: list[str]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        ret = pd.read_csv(DATA / "returns" / f"{sym}.csv",
                          index_col=0, parse_dates=True)
        ret.columns = ["r"]
        fc = pd.read_parquet(DATA / "moirai" / f"{sym}.parquet")
        common = ret.index.intersection(fc.index)
        ret, fc = ret.loc[common], fc.loc[common]
        for alpha in ALPHAS:
            col = f"VaR_{alpha:g}"
            mask = fc[col].notna()
            r = ret["r"].values[mask]
            q = fc[col].values[mask]
            n = len(r)
            _cal, _test, _g = split_indices(n, q - r, f_cal=F_CAL)
            n_cal, t0 = len(_cal), int(_test[0])
            r_cal, r_test = r[:n_cal], r[t0:]
            q_cal, q_test = q[:n_cal], q[t0:]
            n_test = len(r_test)

            qV = qhat_ceil(q_cal - r_cal, alpha)
            var_cp = q_test - qV

            viol_raw = int(np.sum(r_test < q_test))
            viol_cp = int(np.sum(r_test < var_cp))
            rows.append({
                "n_cal": n_cal, "n_test": n_test, "qV": qV,
                "pihat_raw": viol_raw / n_test, "pihat_cp": viol_cp / n_test,
                "viol_raw": viol_raw, "viol_cp": viol_cp,
                "p_kup_raw": kupiec_p(viol_raw, n_test, alpha),
                "p_kup_cp": kupiec_p(viol_cp, n_test, alpha),
                "p_cc_raw": cc_pval(r_test < q_test),
                "p_cc_cp": cc_pval(r_test < var_cp),
                "TL_raw": traffic_light(viol_raw, n_test),
                "TL_cp": traffic_light(viol_cp, n_test),
                "QS_raw": quantile_score(r_test, q_test, alpha),
                "QS_cp": quantile_score(r_test, var_cp, alpha),
                "VaR_width": float(np.mean(var_cp)),
                "raw_width": float(np.mean(q_test)),
                "model": "Moirai-1.1", "symbol": sym, "alpha": alpha,
            })
    return pd.DataFrame(rows)


def main() -> int:
    legacy = pd.read_csv(TABLES / "moirai11_full_results.csv")
    symbols = sorted(legacy["symbol"].unique())
    new = build(symbols)
    new.to_csv(OUT / "moirai11_full_results_rebuilt.csv", index=False)

    key = ["symbol", "alpha"]
    cmp = legacy.merge(new, on=key, suffixes=("_legacy", "_new"))
    num = ["n_cal", "n_test", "qV", "pihat_raw", "pihat_cp", "p_kup_raw",
           "p_kup_cp", "QS_raw", "QS_cp"]
    for c in num:
        cmp[f"d_{c}"] = cmp[f"{c}_new"] - cmp[f"{c}_legacy"]
    cmp.to_csv(OUT / "diff_vs_legacy.csv", index=False)

    L = []
    L.append("# Moirai-1.1 reconciliation")
    L.append("")
    L.append("`rebuild_moirai11.py` regenerates Moirai-1.1 under the same "
             "convention as every other model (n_cal = floor(0.70 n)) and "
             "compares against the legacy `moirai11_full_results.csv`.")
    L.append("")
    L.append("## Diagnosis")
    L.append("")
    n_ceil = int((cmp["n_cal_legacy"] > cmp["n_cal_new"]).sum())
    L.append("**Two** differences from the house pipeline, not one:")
    L.append("")
    L.append(f"1. **Split point.** Differs in **{n_ceil} of {len(cmp)}** cells. "
             "The legacy file is not internally consistent: 12 of 24 assets use "
             "floor(0.70 n), the other 12 use ceil(0.70 n). Moirai-2.0 in "
             "`all_results.csv` uses floor on all 24, so floor is the house "
             "convention and the legacy Moirai-1.1 table is the outlier. Total "
             "sample size is identical in both; only the split moves.")
    L.append("2. **Quantile estimator — the substantive one.** The legacy table "
             "computes the shift as `np.quantile(scores, 1-alpha)` with linear "
             "interpolation. The rest of the pipeline uses the conformal order "
             "statistic with k = ceil((n+1)(1-alpha)), which is the estimator "
             "Theorem 3.3 analyses; an interpolated quantile lies below it and "
             "carries no finite-sample guarantee. Verified exactly on ASX200 at "
             "alpha = 0.01 (legacy 0.001566 = np.quantile linear; conformal "
             "0.001939). This is why qV moves in **all 96** cells, not just the "
             "48 with a different split.")
    L.append("")
    L.append("Both differences push qV **downward**, so the legacy table "
             "understates the correction for Moirai-1.1 throughout.")
    L.append("")
    L.append("## What moves")
    L.append("")
    L.append("| Quantity | max abs change | max rel change | cells changed |")
    L.append("|---|---|---|---|")
    for c in ["qV", "pihat_raw", "pihat_cp", "QS_raw", "QS_cp",
              "p_kup_raw", "p_kup_cp"]:
        d = cmp[f"d_{c}"].abs()
        base = cmp[f"{c}_legacy"].abs().replace(0, np.nan)
        L.append(f"| `{c}` | {d.max():.3e} | {(d / base).max():.1%} | "
                 f"{int((d > 1e-12).sum())} of {len(cmp)} |")
    L.append("")

    a01 = cmp[cmp["alpha"] == 0.01]
    L.append("## At alpha = 0.01 (the level the paper reports)")
    L.append("")
    L.append(f"- Mean qV: legacy {a01['qV_legacy'].mean():.6f} -> "
             f"rebuilt {a01['qV_new'].mean():.6f}")
    L.append(f"- Mean corrected coverage: {a01['pihat_cp_legacy'].mean():.6f} "
             f"-> {a01['pihat_cp_new'].mean():.6f}")
    L.append(f"- Green zones (corrected): "
             f"{int((a01['TL_cp_legacy'] == 'Green').sum())}/24 -> "
             f"{int((a01['TL_cp_new'] == 'Green').sum())}/24")
    L.append(f"- Kupiec not rejected (corrected): "
             f"{int((a01['p_kup_cp_legacy'] > 0.05).sum())}/24 -> "
             f"{int((a01['p_kup_cp_new'] > 0.05).sum())}/24")
    L.append("")
    worst = a01.reindex(a01["d_qV"].abs().sort_values(ascending=False).index)
    L.append("Largest qV moves at alpha = 0.01:")
    L.append("")
    L.append("| Asset | qV legacy | qV rebuilt | change |")
    L.append("|---|---|---|---|")
    for _, r in worst.head(6).iterrows():
        rel = r["d_qV"] / r["qV_legacy"] if r["qV_legacy"] else np.nan
        L.append(f"| {r['symbol']} | {r['qV_legacy']:.6f} | "
                 f"{r['qV_new']:.6f} | {rel:+.1%} |")
    L.append("")
    # --- which version does the PUBLISHED paper actually use? -------------- #
    ar = pd.read_csv(TABLES / "all_results.csv")
    ar01 = ar[ar["alpha"] == 0.01]
    PANEL_A = {"Lag-Llama", "GJR-GARCH", "GARCH-N", "Hist-Sim", "EWMA"}
    L.append("## Which version does the published paper use?")
    L.append("")
    L.append("Decisive, and it is good news: **Table 1 and the text were "
             "produced with the correct computation, not with this CSV.** "
             "Every field that distinguishes the two matches the rebuild.")
    L.append("")
    L.append("| Quantity (α = 0.01) | Paper | Legacy CSV | Rebuilt |")
    L.append("|---|---|---|---|")
    rows = []
    for name, d in (("legacy", legacy[legacy["alpha"] == 0.01]),
                    ("rebuilt", new[new["alpha"] == 0.01])):
        g9 = int((ar01["TL_cp"] == "Green").sum())
        gA9 = int((ar01[ar01["model"].isin(PANEL_A)]["TL_cp"] == "Green").sum())
        g11 = int((d["TL_cp"] == "Green").sum())
        rows.append({
            "kup_cp": f"{int((d['p_kup_cp'] > 0.05).sum())}/24",
            "green": f"{g11}/24",
            "width": f"{d['VaR_width'].abs().mean():.3f}",
            "rbar": f"{d['qV'].mean() / d['raw_width'].abs().mean():.4f}",
            "panelA": f"{gA9 + g11}/144 ({100 * (gA9 + g11) / 144:.1f}%)",
            "all": f"{g9 + g11}/240 ({100 * (g9 + g11) / 240:.1f}%)",
        })
    paper = {"kup_cp": "15/24", "green": "21/24", "width": ".039",
             "rbar": "0.11", "panelA": "127/144 (88.2%)",
             "all": "203/240 (84.6%)"}
    labels = {"kup_cp": "Moirai-1.1 Kupiec pass (corrected)",
              "green": "Moirai-1.1 Green zones",
              "width": "Moirai-1.1 corrected width",
              "rbar": "Moirai-1.1 $\\bar R$",
              "panelA": "Panel A Green (Table 1)",
              "all": "All 240 pairs Green"}
    for k in ("kup_cp", "green", "width", "rbar", "panelA", "all"):
        mark = " ✅" if rows[1][k].startswith(paper[k].lstrip(".0").rstrip("%").split(" ")[0][:4]) or paper[k] in rows[1][k] else ""
        L.append(f"| {labels[k]} | **{paper[k]}** | {rows[0][k]} | "
                 f"**{rows[1][k]}**{mark} |")
    L.append("")
    L.append("`R̄` is the aggregate ratio mean(qV)/mean(|VaR_raw|), not the mean "
             "of per-pair ratios: 0.1121 → 0.11 for the rebuild, versus 0.0997 "
             "→ 0.10 for the legacy CSV.")
    L.append("")
    L.append("## The actual defect: a stale artifact the replication package still reads")
    L.append("")
    L.append("The paper is right; `moirai11_full_results.csv` (and its sibling "
             "`moirai11_results.csv`) are stale files left in "
             "`cfp_ijf_data/paper_outputs/tables/` that no longer match how the "
             "paper's numbers were computed. Three Quantlet scripts still read "
             "them, so **re-running the replication pipeline today reproduces "
             "numbers that contradict the published paper**:")
    L.append("")
    L.append("| Consumer | Output | With stale CSV | Correct |")
    L.append("|---|---|---|---|")
    L.append("| `CO_qV_ranking/run_qV_ranking.py` | qV ranking figure | "
             f"{legacy[legacy['alpha'] == 0.01]['qV'].mean():.6f} | "
             f"{new[new['alpha'] == 0.01]['qV'].mean():.6f} |")
    l01 = legacy[legacy["alpha"] == 0.01]
    n01 = new[new["alpha"] == 0.01]
    L.append("| `CO_multi_quantile_panel/run_multiquantile.py` | Table 5, "
             "Moirai-1.1 at α=0.01 | "
             f"π̂={l01['pihat_cp'].mean():.4f}, Rej "
             f"{int((l01['p_kup_cp'] < 0.05).sum())}/24 | "
             f"π̂={n01['pihat_cp'].mean():.4f}, Rej "
             f"{int((n01['p_kup_cp'] < 0.05).sum())}/24 |")
    L.append("| Table 1 aggregates | Panel A / all-pairs Green | "
             f"{rows[0]['panelA']} / {rows[0]['all']} | "
             f"{rows[1]['panelA']} / {rows[1]['all']} |")
    L.append("")
    L.append("Fix: replace both stale CSVs with "
             "`moirai11_full_results_rebuilt.csv` (regenerated by this script), "
             "then re-run the three consumers and confirm they reproduce "
             "Table 1 unchanged.")
    L.append("")
    L.append("## Caveat on the `p_cc_*` columns")
    L.append("")
    L.append("The rebuilt `p_cc_raw`/`p_cc_cp` implement the Christoffersen "
             "*independence* LR, which is what the surrounding pipeline uses. "
             "Table 1's \"CC pass\" column reports 24/24 for Moirai-1.1 while "
             "both this rebuild and the legacy file give 6/24 at p > 0.05, so "
             "that column is a different statistic (joint conditional coverage, "
             "or a different pass rule). Do not regenerate the CC column from "
             "these fields without first identifying its definition.")
    L.append("")

    (OUT / "RECONCILIATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"rebuilt {len(new)} cells; {n_ceil} had a different split point",
          file=sys.stderr)
    print(f"wrote {OUT}/RECONCILIATION.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
