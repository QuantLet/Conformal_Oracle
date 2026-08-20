#!/usr/bin/env python3
"""Tasks 1 and 3: can q̂_V detect what the backtests cannot?

Decision rules and branch readings were fixed in advance: PREREGISTRATION.md.
Nothing here is tuned on the labels.

DESIGN POINT. The detection question is counterfactual — would a detector have
flagged the series AS IT WAS when defective? So every labelled forecaster is
scored on its SUPERSEDED series, not on the corrected one now in the tree:

    TimesFM-2.5, Moirai-2.0   analysis/recompute/superseded/
    GJR-GARCH                 analysis/gjr_quantile/superseded/
    Chronos-Small, -Mini      cfp_ijf_data/ (the top_k = 50 series, retained
                              deliberately as the exhibit; never corrected in
                              place)

`none` means NOT KNOWN TO BE DEFECTIVE, never "clean". The unlabelled
forecasters have not been audited to the depth the labelled ones have, so
specificity against them is an upper bound on the true false-positive rate. Every
flagged-but-unlabelled case is written out for tracing rather than counted as an
error.

Output: analysis/detection/DETECTION.md, detection_pairs.csv, detection_models.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = Path(__file__).resolve().parent
ALPHA = 0.01
F_CAL = 0.70
R_THRESHOLD = 1.0          # the paper's existing threshold, unchanged

sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import SYMBOLS  # noqa: E402
sys.path.insert(0, str(BASE / "Quantlets" / "CO_full_evaluation"))
from run_full_evaluation import cc_pval, kupiec_pval  # noqa: E402

# (label, family, directory template). Directory is where the series AS SCORED
# lives — superseded for anything since corrected.
SPEC = {
    # --- labelled: a defect that corrupted a reported number ----------------
    "Chronos-Small":  ("top_k_truncation", "foundation", DATA / "chronos_small"),
    "Chronos-Mini":   ("top_k_truncation", "foundation", DATA / "chronos_mini"),
    "TimesFM-2.5":    ("sign_flip", "foundation",
                       BASE / "analysis/recompute/superseded/timesfm25"),
    "Moirai-2.0":     ("sign_flip", "foundation",
                       BASE / "analysis/recompute/superseded/moirai2"),
    "GJR-GARCH":      ("gjr_quantile_map", "classical",
                       BASE / "analysis/gjr_quantile/superseded"),
    # --- unlabelled: not known to be defective, NOT established as clean ----
    "Moirai-1.1":       ("none", "foundation", DATA / "moirai"),
    "Lag-Llama":        ("none", "foundation", DATA / "lagllama"),
    "GARCH-N":          ("none", "classical", DATA / "benchmarks"),
    "Hist-Sim":         ("none", "classical", DATA / "benchmarks"),
    "EWMA":             ("none", "classical", DATA / "benchmarks"),
    "GJR-GARCH-t":      ("none", "classical", DATA / "benchmarks"),
    "Chronos-Small-A":  ("none", "foundation", DATA / "chronos_small_analytic"),
    "Chronos-Mini-A":   ("none", "foundation", DATA / "chronos_mini_analytic"),
}
SUFFIX = {"GARCH-N": "garch_n", "Hist-Sim": "hs", "EWMA": "ewma",
          "GJR-GARCH-t": "gjr_t"}


def load(model, sym):
    lab, fam, d = SPEC[model]
    suf = SUFFIX.get(model)
    fp = d / (f"{sym}_{suf}.parquet" if suf else f"{sym}.parquet")
    if not fp.exists():
        return None
    fc = pd.read_parquet(fp)
    ret = pd.read_csv(DATA / "returns" / f"{sym}.csv", index_col=0,
                      parse_dates=True)
    ret.columns = ["r"]
    i = ret.index.intersection(fc.index).sort_values()
    if len(i) < 300 or f"VaR_{ALPHA}" not in fc.columns:
        return None
    return ret.loc[i, "r"].values, fc.loc[i, f"VaR_{ALPHA}"].values


def cell(r, v):
    n = len(r)
    nc = int(n * F_CAL)
    s = np.sort(v[:nc] - r[:nc])
    k = min(int(np.ceil((nc + 1) * (1 - ALPHA))) - 1, nc - 1)
    qV = float(s[k])
    rt, vt = r[nc:], v[nc:]
    viol = int(np.sum(rt < vt))
    raw_width = float(np.mean(np.abs(vt)))
    return {"qV": qV, "raw_width": raw_width,
            "R": abs(qV) / raw_width if raw_width > 0 else np.nan,
            "pihat_raw": viol / len(rt),
            "p_kup": kupiec_pval(len(rt), viol, ALPHA),
            "p_cc": cc_pval(rt < vt), "n_test": len(rt)}


def main() -> int:
    rows = []
    for model, (lab, fam, _) in SPEC.items():
        for sym in SYMBOLS:
            got = load(model, sym)
            if got is None:
                continue
            rows.append({"model": model, "asset": sym, "defect_label": lab,
                         "defect_family": fam, **cell(*got)})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "detection_pairs.csv", index=False)

    # ---- forecaster-level detector verdicts (rules fixed in advance) -------
    recs = []
    for m, s in d.groupby("model"):
        lab, fam, _ = SPEC[m]
        n = len(s)
        R_bar = float((s["qV"].abs() / s["raw_width"]).mean())
        recs.append({
            "model": m, "defect_label": lab, "defect_family": fam, "assets": n,
            "R_bar": R_bar,
            "flag_R": bool(R_bar > R_THRESHOLD),
            "n_qV_neg": int((s["qV"] < 0).sum()),
            "flag_qV_sign": bool((s["qV"] < 0).sum() > n / 2),
            "kup_fail": int((s["p_kup"] < 0.05).sum()),
            "flag_kupiec": bool((s["p_kup"] < 0.05).sum() > n / 2),
            "cc_fail": int((s["p_cc"] < 0.05).sum()),
            "cc_undefined": int(s["p_cc"].isna().sum()),
            "flag_cc": bool((s["p_cc"] < 0.05).sum() > n / 2),
            "pihat": float(s["pihat_raw"].mean()),
        })
    m = pd.DataFrame(recs).sort_values(["defect_label", "model"])
    m.to_csv(OUT / "detection_models.csv", index=False)

    DET = [("flag_R", "q̂_V magnitude, R̄ > 1"),
           ("flag_qV_sign", "q̂_V sign, negative on a majority of assets"),
           ("flag_kupiec", "Kupiec, p < 0.05 on a majority of assets"),
           ("flag_cc", "Christoffersen, p < 0.05 on a majority of assets")]
    lab_mask = m["defect_label"] != "none"

    L = ["# Detection: does q̂_V flag what the backtests miss?", "",
         "Rules and branch readings fixed in advance in `PREREGISTRATION.md`. "
         "Labelled forecasters are scored on their **superseded** series — the "
         "counterfactual is whether a detector would have fired when the defect "
         "was live.", "",
         f"Labelled (defective): **{int(lab_mask.sum())}** forecasters. "
         f"Unlabelled: **{int((~lab_mask).sum())}** — *not known to be "
         "defective*, not established as clean.", "",
         "## Forecaster-level verdicts", "",
         "| forecaster | defect | family | R̄ | q̂_V<0 | Kupiec fail | CC fail | CC undef | "
         + " | ".join(n for _, n in DET) + " |",
         "|---|---|---|---|---|---|---|---|" + "---|" * len(DET)]
    for _, r in m.iterrows():
        flags = " | ".join("**FLAG**" if r[k] else "—" for k, _ in DET)
        L.append(f"| {r['model']} | {r['defect_label']} | {r['defect_family']} | "
                 f"{r['R_bar']:.3f} | {r['n_qV_neg']}/{r['assets']} | "
                 f"{r['kup_fail']}/{r['assets']} | {r['cc_fail']}/{r['assets']} | "
                 f"{r['cc_undefined']}/{r['assets']} | {flags} |")

    L += ["", "## Sensitivity and specificity", "",
          "Fisher exact, two-sided, on the 2x2 of flag against label. "
          "Specificity is an **upper bound**: the unlabelled set has not been "
          "audited to the depth the labelled set has.", "",
          "| detector | sensitivity | specificity | Fisher p | foundation sens. | classical sens. |",
          "|---|---|---|---|---|---|"]
    for k, name in DET:
        tp = int((m[k] & lab_mask).sum()); fn = int((~m[k] & lab_mask).sum())
        fp = int((m[k] & ~lab_mask).sum()); tn = int((~m[k] & ~lab_mask).sum())
        _, p = stats.fisher_exact([[tp, fp], [fn, tn]])
        fnd = m[lab_mask & (m.defect_family == "foundation")]
        cls = m[lab_mask & (m.defect_family == "classical")]
        fs = f"{int(fnd[k].sum())}/{len(fnd)}" if len(fnd) else "n/a"
        cs = f"{int(cls[k].sum())}/{len(cls)}" if len(cls) else "n/a"
        L.append(f"| {name} | {tp}/{tp + fn} | {tn}/{tn + fp} | {p:.4f} | {fs} | {cs} |")

    fl = m[m[[k for k, _ in DET]].any(axis=1) & ~lab_mask]
    L += ["", "## Flagged but unlabelled — to be traced, not counted", ""]
    if len(fl):
        for _, r in fl.iterrows():
            which = ", ".join(n for k, n in DET if r[k])
            L.append(f"- **{r['model']}** (R̄ = {r['R_bar']:.3f}, "
                     f"π̂ = {r['pihat']:.4f}) flagged by: {which}")
        L.append("")
        L.append("Each requires a verdict. An untraceable flag is a false "
                 "positive; a traceable one is the next defect.")
    else:
        L.append("None.")
    L.append("")
    (OUT / "DETECTION.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
