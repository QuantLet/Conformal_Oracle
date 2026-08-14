#!/usr/bin/env python3
"""Regenerate Table 1 (master results) — the table the shipped script cannot produce.

`run_master_table.py` emits a ten-column table over nine models with TimesFM 2.5
and Moirai 2.0 in Panel A. The published Table 1 has twelve columns over ten
models, adds a CC-pass and an R-bar column, and places those two models in
Panel B. The script predates the finding that they carry ~99% raw violation
rates, and it never reads a Moirai-1.1 input. So the published table's values are
correct but nothing in the package assembles them.

This script does, in two modes:

  --convention published    reproduces the printed table exactly, including its
                            CC rule (a degenerate test counts as a pass)
  --convention informative  the same table with CC counting only pairs where the
                            independence test is defined

Run `published` first: reproducing all ten rows proves the printed table was
internally correct and that the only defect was the missing generator. Only then
is the `informative` output meaningful as the revised table, because the
difference between the two IS the degeneracy result of Section 4.2.

Definitions recovered from the published table (none were documented):
  * Kupiec pass  : p >= 0.05
  * CC pass      : Christoffersen INDEPENDENCE LR ~ chi2_1 (not the joint chi2_2
                   that Appendix G defines), NaN when the transition table is
                   degenerate
  * Width        : mean |VaR_width| over assets
  * W/GJR        : ratio of those means
  * R-bar        : mean over assets of |qV| / |VaR_raw|  -- the per-pair mean of
                   absolute ratios, NOT the ratio of means. The absolute value
                   matters: GJR-GARCH has a negative mean qV.
  * Panel        : A if R-bar < 1, B otherwise

Usage:
    python rebuild_master_table.py [--convention published|informative] [--validate]
"""

from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "cfp_ijf_data"
TABLES = DATA / "paper_outputs" / "tables"

ALPHA = 0.01
PANEL_A = ["Moirai-1.1", "Lag-Llama", "GJR-GARCH", "GARCH-N", "Hist-Sim", "EWMA"]
PANEL_B = ["Chronos-Small", "Chronos-Mini", "TimesFM-2.5", "Moirai-2.0"]
LABELS = {"TimesFM-2.5": "TimesFM 2.5", "Moirai-2.0": "Moirai 2.0",
          "Moirai-1.1": "Moirai 1.1", "Hist-Sim": r"Hist.\ Sim."}

# Table 1 as printed, for validation. Order: raw pi, corr pi, raw Kupiec,
# corr Kupiec, CC, raw QS, corr QS, width, W/GJR, Green, R-bar.
PUBLISHED = {
    "Moirai-1.1":    (".015", ".011", 13, 15, 24, 5.4, 5.3, ".039", 1.00, 21, 0.11),
    "Lag-Llama":     (".029", ".009", 0, 22, 24, 5.9, 5.2, ".041", 1.05, 24, 0.36),
    "GJR-GARCH":     (".004", ".011", 7, 16, 20, 5.0, 4.9, ".039", 1.00, 19, 0.16),
    "GARCH-N":       (".019", ".010", 8, 18, 18, 4.9, 4.7, ".036", 0.93, 21, 0.17),
    "Hist-Sim":      (".016", ".011", 11, 21, 14, 5.7, 5.6, ".039", 1.00, 22, 0.11),
    "EWMA":          (".021", ".011", 5, 16, 21, 5.1, 4.9, ".037", 0.93, 20, 0.18),
    "Chronos-Small": (".388", ".011", 0, 14, 15, 37.9, 5.8, ".040", 1.03, 19, 17.3),
    "Chronos-Mini":  (".419", ".010", 0, 11, 16, 41.1, 5.7, ".041", 1.04, 19, 23.5),
    "TimesFM-2.5":   (".990", ".013", 0, 3, 10, 334.0, 9.5, ".069", 1.75, 19, 3.2),
    "Moirai-2.0":    (".988", ".015", 0, 4, 7, 304.9, 9.1, ".063", 1.60, 19, 3.2),
}


def rhu(x: float, dp: int) -> str:
    return format(Decimal(str(x)).quantize(Decimal(10) ** -dp,
                                           rounding=ROUND_HALF_UP), f".{dp}f")


def strip0(s: str) -> str:
    return s[1:] if s.startswith("0.") else s


def load() -> pd.DataFrame:
    ar = pd.read_csv(TABLES / "all_results.csv")
    m11 = pd.read_csv(TABLES / "moirai11_full_results.csv")
    d = pd.concat([ar, m11], ignore_index=True)
    return d[d["alpha"] == ALPHA].copy()


def compute(d: pd.DataFrame, convention: str) -> pd.DataFrame:
    gjr = d[d["model"] == "GJR-GARCH"]["VaR_width"].abs().mean()
    rows = []
    for model in PANEL_A + PANEL_B:
        s = d[d["model"] == model]
        if not len(s):
            raise SystemExit(f"no rows for {model} at alpha={ALPHA}")
        cc = (s["p_cc_cp"].isna() | (s["p_cc_cp"] > 0.05)) \
            if convention == "published" else (s["p_cc_cp"] > 0.05)
        width = s["VaR_width"].abs().mean()
        rows.append({
            "model": model, "n": len(s),
            "raw_pi": s["pihat_raw"].mean(), "cor_pi": s["pihat_cp"].mean(),
            "raw_kup": int((s["p_kup_raw"] >= 0.05).sum()),
            "cor_kup": int((s["p_kup_cp"] >= 0.05).sum()),
            "cc": int(cc.sum()),
            "cc_degenerate": int(s["p_cc_cp"].isna().sum()),
            "raw_qs": 1e4 * s["QS_raw"].mean(), "cor_qs": 1e4 * s["QS_cp"].mean(),
            "width": width, "w_gjr": width / gjr,
            "green": int((s["TL_cp"] == "Green").sum()),
            "yellow": int((s["TL_cp"] == "Yellow").sum()),
            "red": int((s["TL_cp"] == "Red").sum()),
            "rbar": (s["qV"].abs() / s["raw_width"].abs()).mean(),
        })
    return pd.DataFrame(rows)


def fmt_rbar(x: float) -> str:
    return rhu(x, 2) if x < 1 else rhu(x, 1)


def validate(res: pd.DataFrame) -> list[str]:
    out = []
    for _, r in res.iterrows():
        exp = PUBLISHED[r["model"]]
        got = (strip0(rhu(r["raw_pi"], 3)), strip0(rhu(r["cor_pi"], 3)),
               r["raw_kup"], r["cor_kup"], r["cc"],
               float(rhu(r["raw_qs"], 1)), float(rhu(r["cor_qs"], 1)),
               strip0(rhu(r["width"], 3)), float(rhu(r["w_gjr"], 2)),
               r["green"], float(fmt_rbar(r["rbar"])))
        names = ("raw pi", "corr pi", "raw Kupiec", "corr Kupiec", "CC",
                 "raw QS", "corr QS", "width", "W/GJR", "Green", "R-bar")
        for name, e, g in zip(names, exp, got):
            if e != g:
                out.append(f"{r['model']:15s} {name:12s} published={e}  computed={g}")
    return out


def to_tex(res: pd.DataFrame) -> str:
    L = [r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l rr rr r rr rrr r@{}}", r"\hline\hline",
         r"& \multicolumn{2}{c}{$\hat\pi$}", r"& \multicolumn{2}{c}{Kupiec pass}",
         r"& CC pass", r"& \multicolumn{2}{c}{QS} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"\cmidrule(lr){7-8}",
         r"Model & Raw & Corr. & Raw & Corr.", r"& Corr.",
         r"& Raw & Corr. & Width & W/GJR & Green & $\bar{R}$ \\", r"\hline"]
    for panel, models, title, rel in (
            ("A", PANEL_A, "Signal-preserving recalibration", "<"),
            ("B", PANEL_B, "Effective replacement", ">")):
        L.append(r"\multicolumn{12}{@{}l}{\textit{Panel~" + panel + ": " + title + "}")
        L.append(r"	($|\qVstat|/|\VaR_{\mathrm{raw}}| " + rel + r" 1$)} \\[2pt]")
        for model in models:
            r = res[res["model"] == model].iloc[0]
            L.append(f"{LABELS.get(model, model)} & {strip0(rhu(r['raw_pi'], 3))} "
                     f"& {strip0(rhu(r['cor_pi'], 3))} & {r['raw_kup']}/{r['n']} "
                     f"& {r['cor_kup']}/{r['n']}")
            L.append(f"& {r['cc']}/{r['n']}")
            L.append(f"& {rhu(r['raw_qs'], 1)} & {rhu(r['cor_qs'], 1)} "
                     f"& {strip0(rhu(r['width'], 3))} & {rhu(r['w_gjr'], 2)} "
                     f"& {r['green']}/{r['n']} & {fmt_rbar(r['rbar'])} \\\\")
        if panel == "A":
            L.append(r"\hline")
    L += [r"\hline\hline", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convention", choices=["published", "informative"],
                    default="published")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    d = load()
    res = compute(d, args.convention)
    suffix = "" if args.convention == "published" else "_informative_cc"
    res.to_csv(HERE / f"tab_master_results_rebuilt{suffix}.csv", index=False)
    (HERE / f"tab_master_results_rebuilt{suffix}.tex").write_text(to_tex(res))

    g, y, r_ = int(res["green"].sum()), int(res["yellow"].sum()), int(res["red"].sum())
    n = int(res["n"].sum())
    note = (f"Overall: {g}/{n} Green ({100 * g / n:.1f}\\%), {y} Yellow, {r_} Red.")
    (HERE / f"tab_master_results_note{suffix}.tex").write_text(note + "\n")
    print(f"[{args.convention}] {note}")

    if args.convention == "published" or args.validate:
        diffs = validate(res)
        print(f"\nvalidation against the printed table: "
              f"{'ALL 110 CELLS MATCH' if not diffs else f'{len(diffs)} cell(s) differ'}")
        for line in diffs:
            print("  " + line)
    if args.convention == "informative":
        print("\nCC pass, published rule vs informative-only:")
        pub = compute(d, "published")
        for _, r in res.iterrows():
            p = pub[pub["model"] == r["model"]].iloc[0]
            print(f"  {r['model']:15s} {p['cc']:2d}/24 -> {r['cc']:2d}/24 "
                  f"(degenerate: {r['cc_degenerate']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
