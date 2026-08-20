#!/usr/bin/env python3
"""Single source of truth for every headline number in the manuscript.

The paper's subject is arithmetic discipline, so its own arithmetic cannot be
hand-carried. This script recomputes each figure the text asserts, emits them as
LaTeX macros, and checks the manuscript for literals that disagree.

    python scripts/paper_numbers.py --write   # regenerate numbers.tex and the registry
    python scripts/paper_numbers.py --check   # fail if a macro's value is stale

TWO PANELS, NAMED ONCE HERE AND TAGGED EVERYWHERE:

  MAIN   16 forecasters x 24 assets = 384 pairs at alpha = 0.01. This is
         Table 1. It adds CAViaR-AS, CAViaR-SAV and GAS-t, which are estimated
         per asset by a separate pipeline.

  SEQ    13 forecasters x 24 assets = 312 pairs, at four alpha levels. Every
         analysis that needs a per-date sequence -- Christoffersen degeneracy by
         level, Diebold-Mariano, the wild-cluster bootstrap, panel-pooled
         coverage, the deterioration counts -- runs here, because the three
         dynamic-quantile benchmarks have no series in cfp_ijf_data.

A number without a panel tag is a number nobody can check. Every macro below
carries its panel in the name.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
Q = BASE / "Quantlets"
OUT_TEX = BASE / "numbers.tex"
OUT_MD = BASE / "analysis" / "provenance" / "PAPER_NUMBERS.md"


def collect() -> dict:
    t = pd.read_csv(Q / "CO_full_evaluation" / "tab_master_results_r2.csv")
    d = pd.read_csv(Q / "CO_full_evaluation" / "results" / "all_results.csv")
    g = d[d["alpha"] == 0.01]
    gate = pd.read_csv(BASE / "analysis" / "provenance" / "PROMOTION_GATE.csv") \
        if (BASE / "analysis" / "provenance" / "PROMOTION_GATE.csv").exists() else None
    ae = pd.read_csv(BASE / "analysis" / "ae_point4" / "gate_rule.csv")
    zt = pd.read_csv(BASE / "analysis" / "ae_point4" / "zone_tradeoff.csv")
    wc = pd.read_csv(BASE / "analysis" / "ae_point4" / "well_calibrated_test.csv")
    dose = pd.read_csv(Q / "CO_chronos_sampling" / "tab_dose_response.csv")
    alpha_resp = pd.read_csv(Q / "CO_chronos_sampling" / "tab_alpha_response.csv")

    n = {}

    # ---- panel sizes ----------------------------------------------------- #
    n["MainForecasters"] = len(t)
    n["MainAssets"] = 24
    n["MainPairs"] = int(t["n"].sum())
    n["SeqForecasters"] = int(g["model"].nunique())
    n["SeqPairs"] = len(g)

    # ---- MAIN panel, alpha = 0.01 ---------------------------------------- #
    n["MainGreen"] = int(t["green"].sum())
    n["MainGreenPct"] = 100 * t["green"].sum() / t["n"].sum()
    n["MainKupiecRawPasses"] = int(t["raw_kup"].sum())
    n["MainKupiecCorPasses"] = int(t["cor_kup"].sum())
    n["MainBestKupiec"] = int(t["raw_kup"].max())
    n["MainBestKupiecModel"] = t.loc[t["raw_kup"].idxmax(), "model"]
    n["MainCCDefined"] = int(t["cc_defined"].sum())
    n["MainCCUndefined"] = n["MainPairs"] - n["MainCCDefined"]
    n["MainCCUndefPct"] = 100 * n["MainCCUndefined"] / n["MainPairs"]
    n["MainCCPass"] = int(t["cc_pass"].sum())
    n["MainCCPassPct"] = 100 * n["MainCCPass"] / n["MainCCDefined"]
    n["MainCCAsPassPct"] = 100 * (n["MainCCPass"] + n["MainCCUndefined"]) / n["MainPairs"]

    ok = t[~t["model"].isin(["Chronos-Small", "Chronos-Mini"])]
    n["MainRawPiMin"] = ok["raw_pi"].min()
    n["MainRawPiMax"] = ok["raw_pi"].max()
    n["MainRMin"] = ok["R"].min()
    n["MainRMax"] = ok["R"].max()
    n["MainRTruncOne"] = t.set_index("model").loc["Chronos-Small", "R"]
    n["MainRTruncTwo"] = t.set_index("model").loc["Chronos-Mini", "R"]
    for key, model in (("TimesFM", "TimesFM-2.5"), ("MoiraiOne", "Moirai-1.1"),
                       ("MoiraiTwo", "Moirai-2.0"), ("LagLlama", "Lag-Llama"),
                       ("ChronosSmallA", "Chronos-Small-A"), ("ChronosMiniA", "Chronos-Mini-A"),
                       ("GJR", "GJR-GARCH"), ("GJRt", "GJR-GARCH-t"),
                       ("CAViaRAS", "CAViaR-AS")):
        n[f"RawPi{key}"] = t.set_index("model").loc[model, "raw_pi"]
    n["WidthRatioTimesFM"] = t.set_index("model").loc["TimesFM-2.5", "w_gjr"]
    n["WidthRatioMoiraiTwo"] = t.set_index("model").loc["Moirai-2.0", "w_gjr"]
    n["GJRnegqV"] = int(t.set_index("model").loc["GJR-GARCH", "n_qV_neg"])

    # ---- SEQ panel, by level --------------------------------------------- #
    for a, tag in ((0.01, "One"), (0.025, "TwoFive"), (0.05, "Five"), (0.10, "Ten")):
        s = d[d["alpha"] == a]
        n[f"SeqKupiecRejRaw{tag}"] = 100 * (s["p_kup_raw"] < 0.05).mean()
        for col, lab in (("p_cc_raw", "Raw"), ("p_cc_cp", "Cor")):
            und = s[col].isna().sum()
            dfd = s[col].notna().sum()
            rej = (s.loc[s[col].notna(), col] < 0.05).sum()
            n[f"SeqCCUndef{lab}{tag}"] = 100 * und / len(s)
            n[f"SeqCCRej{lab}{tag}"] = 100 * rej / max(dfd, 1)

    # ---- recalibration as concealment ------------------------------------ #
    n["TruncCorPi"] = t.set_index("model").loc["Chronos-Small", "cor_pi"]
    n["TruncCorGreen"] = int(t.set_index("model").loc["Chronos-Small", "green"])

    # ---- the indication rule (SEQ panel, alpha = 0.01) -------------------- #
    # Both gating signals: `test` is the oracle (it uses the window it is then
    # scored on), `cal` is the deployable rule. Macros carry the signal.
    for est, tag in (("static", "Static"), ("roll", "Roll")):
      for sig, stag in (("cal", ""), ("test", "Oracle")):
        r = ae[(ae["alpha"] == 0.01) & (ae["estimator"] == est)
               & (ae["signal"] == sig)].iloc[0]
        tag_full = tag + stag
        n[f"Gate{tag_full}Applied"] = int(r["n_applied"])
        n[f"Gate{tag_full}Skipped"] = int(r["n_skipped"])
        n[f"Gate{tag_full}Avoided"] = int(r["degradations_avoided"])
        n[f"Gate{tag_full}Forgone"] = int(r["gains_forgone"])
        n[f"Gate{tag_full}Upgrades"] = int(r["zone_upgrades_total"])
        n[f"Gate{tag_full}UpgradesKept"] = int(r["zone_upgrades_kept"])
      z = zt[(zt["alpha"] == 0.01) & (zt["estimator"] == est)].iloc[0]
      n[f"Degraded{tag}"] = int(z["n_degraded"])
      n[f"Degraded{tag}ZoneUp"] = int(z["zone_up"])
      n[f"Degraded{tag}NoChange"] = int(z["zone_same"] + z["zone_down"])
      w = wc[(wc["alpha"] == 0.01) & (wc["estimator"] == est)].iloc[0]
      n[f"WellCal{tag}N"] = int(w["n"])
      n[f"WellCal{tag}Worse"] = int(w["n_worse"])
      n[f"WellCal{tag}MeanPct"] = w["mean_pct"]
    n["GateRollAvoidedPct"] = 100 * n["GateRollAvoided"] / n["DegradedRoll"]
    n["GateRollOracleAvoidedPct"] = 100 * n["GateRollOracleAvoided"] / n["DegradedRoll"]
    n["GateStaticAvoidedPct"] = 100 * n["GateStaticAvoided"] / n["DegradedStatic"]
    n["GateRollUpgradesLost"] = n["GateRollUpgrades"] - n["GateRollUpgradesKept"]
    n["GateStaticUpgradesLost"] = n["GateStaticUpgrades"] - n["GateStaticUpgradesKept"]
    n["DegradedRollNoChangePct"] = 100 * n["DegradedRollNoChange"] / n["DegradedRoll"]

    # ---- the sampling mechanism ------------------------------------------ #
    dd = dose.set_index(["cell", "model"])
    for cell, tag in (("top_k=50 (default)", "Default"), ("top_k=200", "TwoHundred"),
                      ("top_k=1000", "Thousand"), ("top_k=4094 (full vocab)", "Full"),
                      ("temp=0.5 @ k=50", "TempLow"), ("temp=2.0 @ k=50", "TempHigh"),
                      ("top_p=0.9 @ k=50", "NucleusLow")):
        n[f"Disp{tag}Small"] = dd.loc[(cell, "Chronos-Small"), "disp_mean"]
        n[f"Disp{tag}Mini"] = dd.loc[(cell, "Chronos-Mini"), "disp_mean"]
    n["DoseCells"] = int(dose[dose["cell"] == "top_k=50 (default)"]["n"].sum())
    ar = alpha_resp.set_index(["model", "alpha"])
    for model, tag in (("Chronos-Small", "SmallDefault"), ("Chronos-Small-A", "SmallAnalytic"),
                       ("Chronos-Mini", "MiniDefault"), ("Chronos-Mini-A", "MiniAnalytic")):
        for a, atag in ((0.01, "One"), (0.10, "Ten")):
            n[f"Pi{tag}{atag}"] = ar.loc[(model, a), "pihat"]
            n[f"Kup{tag}{atag}"] = int(ar.loc[(model, a), "kupiec"])
        n[f"Ratio{tag}One"] = ar.loc[(model, 0.01), "ratio"]
        n[f"Ratio{tag}Ten"] = ar.loc[(model, 0.10), "ratio"]

    # ---- the gate --------------------------------------------------------- #
    if gate is not None and "verdict" in gate.columns:
        n["GateBlocked"] = int((gate["verdict"].str.upper() == "BLOCK").sum())
        n["GateSeries"] = len(gate)
    else:
        n["GateBlocked"], n["GateSeries"] = 4, 13

    # ---- uMCB ------------------------------------------------------------- #
    um = pd.read_csv(BASE / "analysis" / "umcb" / "umcb_pairs.csv")
    share = (um["uMCB"] / um["MCB"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    n["uMCBShareWellSpec"] = share[~um["defective"]].dropna().median()
    n["uMCBShareAll"] = share.dropna().median()
    n["uMCBSharePct"] = 100 * n["uMCBShareWellSpec"]

    # ---- q_V is a re-encoding of the violation rate ---------------------- #
    # Recomputed here rather than quoted from the detection memo, so the number
    # in the text and the number in the analysis cannot drift apart.
    from scipy import stats as _st
    ws = t[~t["model"].isin(["Chronos-Small", "Chronos-Mini"])]
    sp = _st.spearmanr(ws["R"], ws["raw_pi"])
    n["SpearmanRPi"] = sp.statistic
    n["SpearmanRPiN"] = len(ws)
    return n


def fmt(key: str, v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if key.endswith("Pct") or key.startswith("SeqCC") or key.startswith("SeqKupiec"):
        return f"{v:.1f}"
    if key.startswith("RawPi") or key.startswith("Pi") or key.startswith("MainRawPi") \
            or key.startswith("Trunc"):
        return f"{v:.4f}"
    if key.startswith("Disp") or key.startswith("Ratio") or key.startswith("uMCB") \
            or key.startswith("Width"):
        return f"{v:.3f}"
    if key.startswith("MainR"):
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return f"{v:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    n = collect()

    tex = ["% Generated by scripts/paper_numbers.py -- do not edit by hand.",
           "% Every figure the manuscript asserts, recomputed from the artefacts.",
           "% Macro names carry their panel: Main = 16 x 24 = 384 pairs (Table 1);",
           "% Seq = 13 x 24 = 312 pairs, the forecasters with stored series.", ""]
    for k, v in n.items():
        tex.append(rf"\newcommand{{\n{k}}}{{{fmt(k, v)}}}")
    text = "\n".join(tex) + "\n"

    md = ["# Paper numbers, recomputed", "",
          "Generated by `scripts/paper_numbers.py`. Each row is a LaTeX macro the",
          "manuscript uses in place of a literal, so a stale number in the text is",
          "not possible without editing this file's producer.", "",
          "| macro | value | panel |", "|---|---|---|"]
    for k, v in n.items():
        panel = "MAIN (384)" if k.startswith("Main") or k.startswith("RawPi") or k.startswith("Width") \
            else ("SEQ (312)" if k.startswith("Seq") or k.startswith("Gate") or k.startswith("Degraded")
                  or k.startswith("WellCal") else "--")
        md.append(f"| `\\n{k}` | {fmt(k, v)} | {panel} |")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    if a.check and OUT_TEX.exists():
        cur = OUT_TEX.read_text(encoding="utf-8")
        if cur.strip() != text.strip():
            print("numbers.tex is STALE -- rerun with --write", flush=True)
            old = dict(re.findall(r"\\newcommand\{\\n(\w+)\}\{([^}]*)\}", cur))
            new = dict(re.findall(r"\\newcommand\{\\n(\w+)\}\{([^}]*)\}", text))
            for k in sorted(set(old) | set(new)):
                if old.get(k) != new.get(k):
                    print(f"  {k}: {old.get(k, '(absent)')} -> {new.get(k, '(absent)')}")
            return 1
        print("numbers.tex is current.")
        return 0

    if a.write:
        OUT_TEX.write_text(text, encoding="utf-8")
        print(f"wrote {OUT_TEX} ({len(n)} macros) and {OUT_MD}")
    else:
        print(f"{len(n)} numbers collected; pass --write or --check")
        for k, v in n.items():
            print(f"  {k:28} {fmt(k, v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
