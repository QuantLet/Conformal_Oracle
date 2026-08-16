#!/usr/bin/env python3
"""Phase 4 — Table 1 for the R2 manuscript: one table, ordered by R.

The binary signal-preserving / replacement classification is gone. There is no
threshold, no Panel A and Panel B, and no persistence rule. R is retained as a
continuous, signed statistic and the models are simply ordered by it, so the
order-of-magnitude spread is visible in the data rather than asserted by a
cutoff — which is what the AE asked for ("Simply reporting the values of the
audit statistic ... might give more detailed information without the arbitrary
classification").

Thirteen forecasters: the ten of the original analysis plus the dynamic VaR
benchmarks added in Phase 3b (CAViaR-SAV, CAViaR-AS, GAS-t). CAViaR is the left
anchor of the axis -- R = 0.001, already calibrated raw, correction changes
nothing -- and Chronos-Mini the right, at R = 23.5.

Two columns the R1 table did not have:

  R_signed   the published R discards the sign, which a paper describing q_V as
             "a signed, continuous measure" cannot afford: GJR-GARCH is
             over-conservative (q_V < 0 on 23 of 24 assets) and prints next to
             near-calibrated models as though it sat on the same side of nominal.
  CC pass    counted only where the Christoffersen independence test is defined.
             A degenerate test (no consecutive violations) is not a pass; it is
             the absence of information. The R1 Table 1 counted it as a pass,
             while Table 2 did not -- the same statistic under two conventions in
             consecutive tables.

Output: tab_master_results_r2.{tex,csv}
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
TABLES = BASE / "cfp_ijf_data" / "paper_outputs" / "tables"
DYN = BASE / "analysis" / "phase3_dynamic" / "dynamic_var.csv"
ALPHA = 0.01

LABELS = {"TimesFM-2.5": "TimesFM 2.5", "Moirai-2.0": "Moirai 2.0",
          "Moirai-1.1": "Moirai 1.1", "Hist-Sim": r"Hist.\ Sim.",
          "CAViaR-SAV": "CAViaR-SAV", "CAViaR-AS": "CAViaR-AS", "GAS-t": "GAS-$t$"}
# Interface / family annotation, which is what the repositioned paper is about.
KIND = {
    "Chronos-Small": "TSFM, sample", "Chronos-Mini": "TSFM, sample",
    "Lag-Llama": "TSFM, sample", "Moirai-1.1": "TSFM, sample",
    "TimesFM-2.5": "TSFM, grid", "Moirai-2.0": "TSFM, grid",
    "GJR-GARCH": "parametric", "GARCH-N": "parametric", "EWMA": "parametric",
    "Hist-Sim": "nonparametric", "CAViaR-SAV": "dynamic quantile",
    "CAViaR-AS": "dynamic quantile", "GAS-t": "score-driven",
}


def rhu(x, dp):
    return format(Decimal(str(x)).quantize(Decimal(10) ** -dp,
                                           rounding=ROUND_HALF_UP), f".{dp}f")


def strip0(s):
    return s[1:] if s.startswith("0.") else ("-" + s[3:] if s.startswith("-0.") else s)


def load() -> pd.DataFrame:
    a = pd.read_csv(TABLES / "all_results.csv")
    m = pd.read_csv(TABLES / "moirai11_full_results.csv")
    d = pd.concat([a, m], ignore_index=True)
    d = d[d["alpha"] == ALPHA].copy()
    dyn = pd.read_csv(DYN).rename(columns={"asset": "symbol"})
    dyn["alpha"] = ALPHA
    return pd.concat([d, dyn], ignore_index=True)


def build(d: pd.DataFrame) -> pd.DataFrame:
    gjr = d[d["model"] == "GJR-GARCH"]["VaR_width"].abs().mean()
    rows = []
    for model, s in d.groupby("model"):
        width = s["VaR_width"].abs().mean()
        # informative-only CC: a degenerate test is not a pass
        cc_def = s["p_cc_cp"].notna()
        rows.append({
            "model": model, "kind": KIND.get(model, ""), "n": len(s),
            "raw_pi": s["pihat_raw"].mean(), "cor_pi": s["pihat_cp"].mean(),
            "raw_kup": int((s["p_kup_raw"] >= 0.05).sum()),
            "cor_kup": int((s["p_kup_cp"] >= 0.05).sum()),
            "cc_pass": int((s["p_cc_cp"] > 0.05).sum()),
            "cc_defined": int(cc_def.sum()),
            "raw_qs": 1e4 * s["QS_raw"].mean(), "cor_qs": 1e4 * s["QS_cp"].mean(),
            "width": width, "w_gjr": width / gjr,
            "green": int((s["TL_cp"] == "Green").sum()),
            "R": (s["qV"].abs() / s["raw_width"].abs()).mean(),
            "R_signed": (s["qV"] / s["raw_width"].abs()).mean(),
            "n_qV_neg": int((s["qV"] < 0).sum()),
        })
    return pd.DataFrame(rows).sort_values("R").reset_index(drop=True)


def to_tex(r: pd.DataFrame) -> str:
    L = [r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}ll rr rr r rr rr r@{}}", r"\toprule",
         r"& & \multicolumn{2}{c}{$\hat\pi$} & \multicolumn{2}{c}{Kupiec pass}",
         r"& CC pass & \multicolumn{2}{c}{QS} & & & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){8-9}",
         r"Forecaster & Interface & Raw & Corr. & Raw & Corr. & Corr.",
         r"& Raw & Corr. & Width & Green & $\bar{R}$ \\", r"\midrule"]
    for _, x in r.iterrows():
        L.append(
            f"{LABELS.get(x['model'], x['model'])} & {x['kind']} "
            f"& {strip0(rhu(x['raw_pi'], 3))} & {strip0(rhu(x['cor_pi'], 3))} "
            f"& {x['raw_kup']}/{x['n']} & {x['cor_kup']}/{x['n']} "
            f"& {x['cc_pass']}/{x['cc_defined']} "
            f"& {rhu(x['raw_qs'], 1)} & {rhu(x['cor_qs'], 1)} "
            f"& {strip0(rhu(x['width'], 3))} & {x['green']}/{x['n']} "
            f"& {rhu(x['R'], 3) if x['R'] < 1 else rhu(x['R'], 1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main() -> int:
    d = load()
    r = build(d)
    r.to_csv(HERE / "tab_master_results_r2.csv", index=False)
    (HERE / "tab_master_results_r2.tex").write_text(to_tex(r))

    g, n = int(r["green"].sum()), int(r["n"].sum())
    note = (f"Overall: {g}/{n} Green ({100 * g / n:.1f}\\%). "
            f"$\\bar R$ spans {r['R'].min():.3f} ({LABELS.get(r.iloc[0]['model'], r.iloc[0]['model'])}) "
            f"to {r['R'].max():.1f} ({LABELS.get(r.iloc[-1]['model'], r.iloc[-1]['model'])}), "
            "a factor of "
            f"{r['R'].max() / r['R'].min():.0f}, with no gap that would support a "
            "binary split.")
    (HERE / "tab_master_results_r2_note.tex").write_text(note + "\n")

    print(r[["model", "kind", "raw_pi", "cor_pi", "cor_qs", "green", "n",
             "R", "R_signed", "n_qV_neg", "cc_pass", "cc_defined"]]
          .round(4).to_string(index=False))
    print("\n" + note)
    # Is there a gap anywhere that a threshold could exploit?
    rs = r["R"].values
    ratios = rs[1:] / rs[:-1]
    j = int(np.argmax(ratios))
    print(f"\nlargest consecutive ratio in R: {ratios[j]:.2f}x "
          f"between {r.iloc[j]['model']} ({rs[j]:.3f}) and "
          f"{r.iloc[j + 1]['model']} ({rs[j + 1]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
