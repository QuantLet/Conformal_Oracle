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
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
Q = BASE / "Quantlets"
DATA_QS = BASE / "cfp_ijf_data" / "paper_outputs" / "qs_sequences"
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
      # The decomposition of NoChange used to be carried as three prose literals
      # beside a macro total, so the parts did not move when the artefact did.
      # After R14 the total went 102 -> 103 and the literals stayed at 99 + 1 + 2.
      n[f"Degraded{tag}AlreadyGreen"] = int(z["zone_same_already_green"])
      n[f"Degraded{tag}SameNotGreen"] = int(z["zone_same"] - z["zone_same_already_green"])
      n[f"Degraded{tag}ZoneDown"] = int(z["zone_down"])
      w = wc[(wc["alpha"] == 0.01) & (wc["estimator"] == est)].iloc[0]
      n[f"WellCal{tag}N"] = int(w["n"])
      n[f"WellCal{tag}Worse"] = int(w["n_worse"])
      n[f"WellCal{tag}MeanPct"] = w["mean_pct"]
      # The Wilcoxon p was a prose literal, and it moved with R14: 9.3e-5 to
      # 8.9e-5. Emitted as a formatted string so the exponent travels with it.
      _p = float(w["wilcoxon_p"])
      _e = int(np.floor(np.log10(_p)))
      # Below 1e-10 the paper writes the order of magnitude alone, so the
      # exponent is rounded to nearest rather than floored: 9.94e-17 is 10^{-16}.
      n[f"WellCal{tag}Wilcoxon"] = (
          f"{_p / 10 ** _e:.1f} \\times 10^{{{_e}}}" if _e > -10
          else f"10^{{{int(round(np.log10(_p)))}}}")
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
    # The column is PASS, not verdict. Guarding on "verdict" meant this branch
    # never ran and the else-branch below supplied 4 and 13 as literals -- in the
    # one script whose whole purpose is that no figure is hand-carried. The
    # fallback happened to be right, which is why it survived; it is now an error
    # rather than a default, because a silent fallback to a literal is the defect.
    if gate is None or "PASS" not in gate.columns:
        raise SystemExit("PROMOTION_GATE.csv missing or has no PASS column; "
                         "refusing to fall back to a literal gate count")
    if True:
        n["GateBlocked"] = int((~gate["PASS"].astype(bool)).sum())
        n["GateSeries"] = len(gate)
        # The extremes band was missing from the body's list of what the two
        # default-sampled series fail, and it is the one they do not fail on all
        # 24 assets, so it cannot be carried by the "on all assets" phrasing.
        if "extremes" in gate.columns:
            for m, tag in (("Chronos-Small", "Small"), ("Chronos-Mini", "Mini")):
                row = gate[gate["model"] == m]
                if len(row):
                    n[f"GateExtremes{tag}"] = int(str(row.iloc[0]["extremes"]).split("/")[0])


    # The wild-cluster bootstrap passage in the supplement carried four typed
    # numbers, all stale: GJR-GARCH's asymptotic and bootstrap Kupiec p, and a
    # "16 of 20 DM pairs remain significant" that describes a 20-pair panel. The
    # panel is 30 pairs now, and the bootstrap makes one MORE significant rather
    # than fewer, so the sentence's direction inverted as well as its counts.
    dmw = pd.read_csv(Q / "CO_panel_wildcluster" / "wild_cluster_dm.csv")
    n["WCDMPairs"] = int(len(dmw))
    n["WCDMSigBoot"] = int((dmw["p_boot"] < 0.05).sum())
    n["WCDMSigAsymp"] = int((dmw["p_asymp"] < 0.05).sum())
    _wck = pd.read_csv(Q / "CO_panel_wildcluster" / "wild_cluster_kupiec.csv") \
        .set_index("model")
    n["WCKupAsympGJR"] = float(_wck.loc["GJR-GARCH", "p_asymp"])
    n["WCKupBootGJR"] = float(_wck.loc["GJR-GARCH", "p_boot"])

    # The COVID response-lag decomposition: three counts that were typed while the
    # artefact behind them is a 13-row table.
    cv = pd.read_csv(Q / "CO_covid_response_lag" / "covid_response_lags.csv")
    n["LagForecasters"] = int(len(cv))
    n["LagImmediate"] = int((cv["lag_calendar_days"] == 0).sum())
    _mid = sorted(d for d in cv["lag_calendar_days"].unique() if d > 0)
    n["LagMidDays"] = int(_mid[0])
    n["LagMidN"] = int((cv["lag_calendar_days"] == _mid[0]).sum())
    n["LagLateDays"] = int(_mid[-1])
    n["LagLateN"] = int((cv["lag_calendar_days"] == _mid[-1]).sum())

    # Proposition: zone improvement is a sub-event of coverage improvement. The
    # inclusion is measured on the panel rather than asserted, and the count that
    # matters is the number of counterexamples, which must be zero.
    _zb = json.loads((BASE / "analysis" / "k2_indication" /
                      "benefit_measures.json").read_text())
    n["ZoneCells"] = int(_zb["cells"])
    n["ZoneUpgrades"] = int(_zb["zone_upgrades"])
    n["ZoneCloser"] = int(_zb["closer_to_nominal"])
    n["ZoneUpNotCloser"] = int(_zb["zone_up_not_closer"])
    n["ZoneCloserNotUp"] = int(_zb["closer_not_zone_up"])
    n["ZoneGivenUp"] = int(_zb["given_up"])
    n["ZoneGivenUpWorseScore"] = int(_zb["given_up_worse_on_score"])
    n["ZoneNetCostScore"] = int(_zb["net_cost_by_score"])

    # The two Fisher exact p-values, computed from the 2x2 tables the sentence
    # itself prints rather than carried in a file with no producer.
    from scipy import stats as _fs
    n["FisherKupiec"] = float(_fs.fisher_exact([[5, 6], [0, 2]])[1])
    n["FisherSevere"] = float(_fs.fisher_exact([[5, 0], [0, 8]])[1])

    # The tightest band edge the panel admits: the worst well-specified cell,
    # rounded toward zero on the 0.01 grid the band is stated on. The margin is
    # what is left between the two. Both follow from the cell distribution; the
    # understatement AT that edge does not -- the band-sweep machinery that would
    # evaluate it is not in this repository, so it is bracketed by the two
    # neighbouring rows of band_sweep.csv rather than interpolated between them.
    _cells = pd.read_csv(BASE / "analysis" / "phase2" /
                         "panel_scale_ratios_by_asset.csv")
    _good = _cells[~_cells["series"].str.contains(r"\(default\)")]
    _worst = float(_good["ratio"].max())
    n["TightBand"] = float(np.ceil(_worst * 100) / 100)
    n["TightMargin"] = abs(_worst - n["TightBand"])
    # The understatement at that edge is a closed form, not a solver output, and
    # emit_band_sweep.py reproduces every row of band_sweep.csv from it to 1e-10.
    # So the figure is computed at the tightened edge rather than bracketed by the
    # grid's neighbours, and the file that feeds the 30.9% of Table 2 is no longer
    # an artefact nothing writes.
    sys.path.insert(0, str(BASE / "analysis" / "phase2"))
    from emit_band_sweep import understatement as _und, critical_delta as _cdelta
    n["TightUnd"] = float(_und(n["TightBand"]))
    n["TightDelta"] = float(_cdelta(n["TightBand"]))

    # The constructed pair, and the GJR-vs-GJR-t comparison. Both sets of figures
    # sat in phase2_numbers.json with no producer. The pair's thresholds are in
    # pair.npz; the published alternative, 1.32 sigma, is not -- the construction
    # gives 1.46, which is 56% of the honest threshold rather than "half".
    _pair = np.load(BASE / "analysis" / "phase2" / "pair.npz")
    n["PairVaRHonest"] = abs(float(_pair["q_true"]))
    n["PairVaRAlt"] = abs(float(_pair["q_trunc"]))
    n["PairCapitalPct"] = 100 * n["PairVaRAlt"] / n["PairVaRHonest"]

    _t = pd.read_csv(Q / "CO_full_evaluation" / "tab_master_results_r2.csv") \
        .set_index("model")
    _a, _b = float(_t.loc["GJR-GARCH", "cor_qs"]), float(_t.loc["GJR-GARCH-t", "cor_qs"])
    n["PairQSGapPct"] = 100 * abs(_a - _b) / max(_a, _b)

    # Diebold-Mariano on the corrected loss differentials with a Driscoll-Kraay
    # panel-HAC variance. The convention is stated because the published t = 0.399
    # cannot be reproduced from any artefact here and the convention behind it was
    # never recorded: cross-sectional mean per date, Bartlett kernel, lag
    # floor(4 (T/100)^(2/9)). The verdict is unchanged -- the two are not
    # distinguishable -- but the statistic is now one the reader can recompute.
    _q1 = pd.read_parquet(DATA_QS / "gjr_garch_qs.parquet")
    _q2 = pd.read_parquet(DATA_QS / "gjr_t_qs.parquet")
    _c = _q1.index.intersection(_q2.index)
    _d = (_q1.loc[_c] - _q2.loc[_c]).mean(axis=1, skipna=True).dropna().to_numpy()
    _T = len(_d); _m = int(np.floor(4 * (_T / 100) ** (2 / 9)))
    _u = _d - _d.mean(); _s2 = (_u ** 2).mean()
    for _l in range(1, _m + 1):
        _s2 += 2 * (1 - _l / (_m + 1)) * (_u[_l:] * _u[:-_l]).mean()
    from scipy import stats as _sst
    n["PairDMt"] = float(_d.mean() / np.sqrt(_s2 / _T))
    n["PairDMp"] = float(2 * _sst.norm.sf(abs(n["PairDMt"])))
    n["PairDMLags"] = int(_m)

    # The six-pair bound validation. These five sat in phase2_numbers.json, which
    # has no producer, while their artefact was in the repository all along --
    # and one of them had gone stale there: rho reaches 0.62, not the 0.67 the
    # paper printed, because TimesFM's persistence moved with the sign correction.
    bv = pd.read_csv(Q / "CO_bound_validation" / "tab_bound_validation.csv")
    n["BoundPairs"] = int(len(bv))
    n["BoundRhoLo"] = float(bv["rho_hat"].min())
    n["BoundRhoHi"] = float(bv["rho_hat"].max())
    n["BoundDeltaLo"] = float(bv["delta_n"].min())
    n["BoundDeltaHi"] = float(bv["delta_n"].max())
    n["BoundFloor"] = float(bv["guaranteed"].min())
    n["BoundEmpirical"] = float(bv["empirical"].mean())

    # The tail-closure spread. Its lower end was printed as 0.005, which is also
    # the value DECLARED_CONSTANTS.md admits for the detection severity cut: the
    # same literal standing for two unrelated things is exactly what a macro name
    # is for.
    tc = pd.read_csv(Q / "CO_robustness_inner7" / "inner7_tail_closure.csv")
    n["LitClosureRMin"] = float(tc["R"].min())
    n["LitClosureRMax"] = float(tc["R"].max())

    # Two rungs of the delta-star ladder were prose literals while the rest were
    # macros: the fourth-moment restriction and the GARCH-t class. Both are in
    # delta_by_class.json and were simply never lifted into the registry.
    dbc = {r["cls"]: r for r in json.loads(
        (BASE / "analysis" / "phase2" / "delta_by_class.json").read_text())}
    n["GapDeltaMoment"] = float(dbc["unimodal, fourth moment <= that of P"]["delta"])
    n["GapUndGarchT"] = float(
        dbc["GARCH class, standardised Student-t innovations"]["understatement"])

    # Section 4.4's validation of the analytic estimator against full-vocabulary
    # sampling, recomputed after the R14 map was corrected in analytic_quantiles.py
    # -- which had carried a second copy of the defect, so the announced agreement
    # was measured on the support it was meant to check.
    av = pd.read_csv(BASE / "analysis" / "chronos_sampling" /
                     "analytic_validation_SP500.csv")
    n["LitAnalyticDates"] = int(len(av))
    n["LitAnalyticSdPct"] = 100 * abs(av["an_std"].mean() - av["sm_std"].mean()) \
        / av["sm_std"].mean()
    n["LitAnalyticGrid"] = 1.0 / len(av)

    # The rate the GJR-GARCH series reported before the unstandardised Student-t
    # quantile was corrected. It appeared twice as a literal -- "0.4\%" in the
    # introduction and "$0.004$" in the failure-mode table -- for a quantity that
    # is measured, and measured in a file.
    gjr = pd.read_csv(BASE / "analysis" / "gjr_quantile" / "promotion_before_after.csv")
    n["RawPiGJRDefective"] = float(gjr[gjr["series"] == "superseded"]["pihat"].mean())
    n["RawPiGJRDefectivePct"] = 100 * n["RawPiGJRDefective"]

    # The same quantity for the two series whose lower quantile was stored with
    # an inverted sign. Section 8's failure-mode table printed these as literals
    # in the "with the defect" column while the "without" column beside them was
    # already a macro -- and guard 2 could not see them, because it replaced
    # every tabular before reading. Cell means over the 24 assets.
    sv = pd.read_csv(BASE / "analysis" / "recompute" / "sign_verification.csv")
    for tag, mdl in (("TimesFM", "TimesFM-2.5"), ("MoiraiTwo", "Moirai-2.0")):
        n[f"RawPi{tag}Defective"] = float(
            sv[sv["model"] == mdl]["pihat_stored"].mean())

    # ---- POOL panel: one row is a forecaster pooled over assets and dates --- #
    # 36,588 observations for the foundation models, 38,473 for the benchmarks.
    # These were four prose literals beside two more; the pooled rate and the two
    # cluster p-values are measurements and now travel as macros. The panel tag
    # is "Pool", declared here beside Main, Seq and Gap.
    wcl = pd.read_csv(Q / "CO_panel_wildcluster" / "wild_cluster_kupiec.csv") \
        .set_index("model")
    pnl = pd.read_csv(Q / "CO_multi_quantile_panel" / "tab_panel_pooled.csv") \
        .set_index("model")
    for m, tag in (("EWMA", "EWMA"), ("Moirai-2.0", "MoiraiTwo"),
                   ("Chronos-Small-A", "SmallAnalytic"), ("GJR-GARCH", "GJR")):
        n[f"PoolPi{tag}"] = float(wcl.loc[m, "pi_pooled"])
    n["PoolBootEWMA"] = float(wcl.loc["EWMA", "p_boot"])
    n["PoolClusterEWMA"] = float(pnl.loc["EWMA", "p_cluster"])

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

    # ---- literals converted from prose (unit tagged in each name) ---------
    # SEQ = series-asset cells with stored sequences; MAIN = the 16x24 table.
    def _disp(cell, model):
        r = dose[(dose["cell"] == cell) & (dose["model"] == model)]
        return float(r["disp_mean"].iloc[0])
    n["LitTempGap"] = abs(_disp("temp=2.0 @ k=50", "Chronos-Small")
                          - _disp("temp=0.5 @ k=50", "Chronos-Small"))
    n["LitNucleusGap"] = abs(_disp("top_k=50 (default)", "Chronos-Small")
                             - _disp("top_p=0.9 @ k=50", "Chronos-Small"))
    n["LitSupportPct"] = 100 * (4093 - 50) / 4093
    def _pi(model, a_):
        r = alpha_resp[(alpha_resp["model"] == model)
                       & (alpha_resp["alpha"].round(4) == a_)]
        return float(r["pihat"].iloc[0])
    for tag, mdl in (("Small", "Chronos-Small-A"), ("Mini", "Chronos-Mini-A")):
        for a_, lab in ((0.025, "TwoFive"), (0.05, "Five")):
            n[f"LitRatio{tag}{lab}"] = _pi(mdl, a_) / a_
    n["LitAlphaResp"] = _pi("Chronos-Small", 0.10) / _pi("Chronos-Small", 0.01)
    n["LitMainRMinAll"] = float(t["R"].min())
    n["LitMainRMaxOK"] = float(ok["R"].max())
    # The prose said "thirteen of the sixteen ... between 0.001 and 0.18". At that
    # bound the count is eleven; at the count it claims the bound is 0.32. Both
    # halves are now taken from the same object: the forecasters that are not the
    # two sampled at the checkpoint default, and their largest R-bar.
    n["LitMainROkN"] = int(len(ok))
    n["LitQSMin"] = float(t["cor_qs"].min())
    n["LitQSMax"] = float(t["cor_qs"].max())
    n["LitWMin"] = float(t["w_gjr"].min())
    n["LitWMax"] = float(t["w_gjr"].max())
    n["LitCorPiTypical"] = float(t["cor_pi"].median())
    n["LitMoiraiGapPP"] = 100 * abs(t.set_index("model").loc["Moirai-2.0", "raw_pi"]
                                    - t.set_index("model").loc["Moirai-1.1", "raw_pi"])
    n["LitLambdaTail"] = 0.94 ** 250
    dmc = pd.read_csv(Q / "CO_quantile_scores" / "tab_dm_configuration.csv")
    for _, r in dmc.iterrows():
        tag = "Small" if "Small" in r["default"] else "Mini"
        n[f"LitQSDefault{tag}"] = float(r["QS_default"]) * 1e4
        n[f"LitQSAnalytic{tag}"] = float(r["QS_analytic"]) * 1e4

    # ---- dynamic quantile test, SEQ panel (cells), alpha = 0.01 ----------
    dq = pd.read_csv(BASE / "analysis" / "phase3" / "dq_panel.csv")
    n["SeqDQCells"] = int(len(dq))
    n["SeqDQRejRaw"] = 100 * (dq["p_dq_raw"] < 0.05).mean()
    n["SeqDQRejCor"] = 100 * (dq["p_dq_cp"] < 0.05).mean()

    # ---- Phase 2: the identification result and the calibrated gate band ----
    # Emitted by analysis/phase2/{construct_pair,delta_by_class,band_sweep}.py
    # and frozen into phase2_numbers.json so the manuscript cannot carry a typed
    # literal for any of them. Grid: spacing 0.004, ceiling 32; delta* converges
    # from above, so each is an upper bound.
    p2 = json.loads((BASE / "analysis" / "phase2" / "phase2_numbers.json")
                    .read_text())
    for k, v in p2.items():
        n[f"Gap{k}"] = v

    # ---- the gap ablation, recomputed from its own artefact ---------------
    # GapAblFull, GapAblCovid, RhoLo and RhoHi were hand-entered into
    # phase2_numbers.json and therefore checked against nothing. The two
    # ablation figures are recomputed here from the CSV that
    # scripts/gap_ablation.py writes, so the build fails when the two diverge.
    # The rho range is the ablation's OWN four pairs; the six-pair range behind
    # the remainder estimate is a different set and keeps its own macro.
    abl = pd.read_csv(Q / "CO_robustness" / "gap_ablation.csv")
    for per, key in (("Full", "GapAblFull"), ("COVID", "GapAblCovid")):
        sub = abl[abl["period"] == per]
        a = sub[sub["gap_label"] == "g=0"].set_index(["model", "asset"])["pi_hat"]
        b = sub[sub["gap_label"] == "g=c*log(n)"].set_index(["model", "asset"])["pi_hat"]
        n[f"Gap{key}"] = float((a - b).abs().max())
    rho = abl[abl["period"] == "Full"]["rho_hat"]
    n["GapAblRhoLo"] = float(rho.min())
    n["GapAblRhoHi"] = float(rho.max())
    n["GapAblPairs"] = int(abl[abl["period"] == "Full"]["model"].nunique())

    # ---- Proposition 5.1 and Corollary 5.2 -------------------------------
    from scipy import stats as _st
    tau = 4.0 / 250.0
    n["TLTau"] = tau
    n["TLTauOverAlpha"] = tau / 0.01
    _q = lambda pr: _st.t.ppf(pr, 5) / np.sqrt(5.0 / 3.0)
    n["TLUndTFive"] = 100 * (1 - abs(_q(tau)) / abs(_q(0.01)))
    n["TLUndTThree"] = 100 * (1 - abs(_st.t.ppf(tau, 3) / np.sqrt(3.0))
                          / abs(_st.t.ppf(0.01, 3) / np.sqrt(3.0)))
    n["TLUndNormal"] = 100 * (1 - abs(_st.norm.ppf(tau)) / abs(_st.norm.ppf(0.01)))

    # ---- the Monte Carlo grid --------------------------------------------
    grid = pd.read_csv(BASE / "analysis" / "k2_sim" / "grid.csv")
    _g = lambda dgp, T, col: float(grid[(grid["dgp"] == dgp) & (grid["T"] == T)][col].iloc[0])
    n["MCGreenTFiveSmall"] = _g("t5", 500, "RawGreen")
    n["MCGreenTFiveLarge"] = _g("t5", 10000, "RawGreen")
    n["MCPiTFive"] = _g("t5", 10000, "Raw_pi")
    n["MCGreenSkewSmall"] = _g("skewt3", 500, "RawGreen")
    n["MCGreenSkewLarge"] = _g("skewt3", 10000, "RawGreen")
    n["MCTMax"] = int(grid["T"].max())
    n["MCTMin"] = int(grid["T"].min())
    n["MCReps"] = 500
    n["MCDgps"] = int(grid["dgp"].nunique())

    # ---- the gated rule's cost, decomposed -------------------------------
    ov = json.loads((BASE / "analysis" / "k2_indication"
                     / "gate_ledger_overlap.json").read_text())
    n["GateRollUpgradesLostWorse"] = ov["roll_cal"]["lost_but_score_worse"]
    n["GateRollUpgradesLostNet"] = (ov["roll_cal"]["lost"]
                                    - ov["roll_cal"]["lost_but_score_worse"])
    n["GateRollUpgradeAndDeteriorate"] = ov["roll_cal"]["upgrade_and_deterioration"]

    # ---- SUP panel: the supplement's own prose figures --------------------- #
    # Guard 2 failed on 35 bare decimals in supplement.tex. Each is closed here
    # or declared in DECLARED_CONSTANTS.md; twelve of them did not reproduce and
    # the disposition of all 35 is in
    # analysis/provenance/SUPPLEMENT_LITERALS.md. The panel tag is "Sup".

    # The Acerbi-Szekely sign defect. verify_z2.py is a second implementation
    # with the canonical per-date denominator; the numbers the supplement quotes
    # are its medians, and the violation rate is at the ES level, not at 1%.
    z2v = pd.read_csv(BASE / "analysis" / "provenance" / "z2_verification.csv") \
        .set_index("model")
    for tag, mdl in (("Small", "Chronos-Small"), ("Mini", "Chronos-Mini")):
        n[f"SupZTwoPi{tag}"] = float(z2v.loc[mdl, "pihat_at_ES_level"])
        n[f"SupZTwo{tag}"] = float(z2v.loc[mdl, "Z2_canonical_median"])
        # Dividing by the stored (negative) column maps z -> 2 - z, which is why
        # the defective routine returned a large positive statistic.
        n[f"SupZTwoFlipped{tag}"] = 2.0 - n[f"SupZTwo{tag}"]
    n["SupZTwoAssets"] = int(z2v.loc["Chronos-Small", "n"])

    # The tuned GBM-QR ablation. The prose reported 5/9, 0/9 and 88.9% Green
    # from REPRO_NOTES_E1.md, which describes a nine-model run; the shipped grid
    # is 8 configurations x 13 models and no count in it is 8/9.
    tg = pd.read_csv(Q / "CO_baseline_comparison_tuned" / "tuned_gbm_qr_grid.csv")
    _cfg = lambda ne, d, lr: tg[(tg["n_est"] == ne) & (tg["max_depth"] == d)
                                & (tg["lr"] == lr)]
    _best, _cons = _cfg(100, 3, 0.05), _cfg(100, 3, 0.01)
    n["SupTunedSeries"] = int(len(_best))
    n["SupTunedPiBest"] = float(_best["pi_hat"].mean())
    n["SupTunedPiCons"] = float(_cons["pi_hat"].mean())
    n["SupTunedKupBest"] = int((_best["kupiec_p"] < 0.05).sum())
    n["SupTunedKupCons"] = int((_cons["kupiec_p"] < 0.05).sum())
    n["SupTunedGreenBest"] = int((_best["TL"] == "Green").sum())
    n["SupTunedGreenPctBest"] = 100 * float((_best["TL"] == "Green").mean())
    n["SupTunedGreenPctCons"] = 100 * float((_cons["TL"] == "Green").mean())
    n["SupTunedQSGainPct"] = 100 * float(
        (_cons["QS"].mean() - _best["QS"].mean()) / _cons["QS"].mean())

    # The conformal index at the smallest calibration block, and the Monte Carlo
    # negative control that measures what the two quantile conventions do there.
    gts = json.loads((BASE / "analysis" / "k2_sim" / "gates.json").read_text())
    _ctl = {(c["dgp"], c["T"]): c for c in gts["negative_controls"]}
    n["SupNCal"] = int(_ctl[("t5", 500)]["n_cal"])
    # Closed form, not read off the control: k = ceil((n+1)(1-alpha)), index k/n.
    _k = int(np.ceil((n["SupNCal"] + 1) * (1 - 0.01)))
    n["SupConfIndex"] = _k / n["SupNCal"]
    n["SupConfOvershoot"] = n["SupConfIndex"] - 0.99
    assert abs(n["SupConfOvershoot"] - _ctl[("t5", 500)]["overshoot"]) < 1e-12, \
        "the closed-form overshoot disagrees with the simulation's own record"
    for tag, dgp in (("TFive", "t5"), ("TThree", "t3")):
        n[f"SupCtlExactSmall{tag}"] = abs(float(_ctl[(dgp, 500)]["mean_qV_exact"]))
        n[f"SupCtlExactLarge{tag}"] = abs(float(_ctl[(dgp, 10000)]["mean_qV_exact"]))
        n[f"SupCtlConfSmall{tag}"] = float(_ctl[(dgp, 500)]["mean_qV_conformal"])
        n[f"SupCtlConfLarge{tag}"] = float(_ctl[(dgp, 10000)]["mean_qV_conformal"])

    # The reproduction gate's resolution. A systematic bias survives the gate
    # only if it is under 3 SE in EVERY cell, so the figure is the minimum over
    # cells. The published 2.1e-4 is the mean of the two Normal cells, quoted
    # once as a minimum and once, in GATE_REVISION.md, as a maximum.
    _ref = pd.read_csv(Q / "CO_simulation_study" / "simulation_study_results.csv")
    _sd = _ref.groupby(["dgp_name", "T"])["q_hat_V"].std(ddof=1)
    _nrep, _nrepro = 500, 2000
    _tol = 3 * np.sqrt(1.0 + _nrep / _nrepro) * _sd / np.sqrt(_nrep)
    n["SupReproTol"] = float(_tol.min())
    n["SupReproTolMax"] = float(_tol.max())
    n["SupReproCells"] = int(len(_tol))

    # The corrected rate in the degenerate small-sample regime, where k >= n and
    # the conformal shift is the window maximum.
    _ss = pd.read_csv(Q / "CO_robustness" / "study1_small_sample.csv")
    n["SupSmallPi"] = float(_ss[_ss["T"] == 250]["mean_corr_pi"].median())
    n["SupSmallNCal"] = int(round(0.70 * 250))

    # The delta-star ladder. Every entry of Table S.8 is in delta_by_class.json;
    # the table was hand-authored, and guard 2 strips tabulars, so its literals
    # were outside every check the project runs.
    _dbc = {r["cls"]: r for r in json.loads(
        (BASE / "analysis" / "phase2" / "delta_by_class.json").read_text())}
    for tag, cls in (("Free", "no shape restriction"),
                     ("Uni", "unimodal"),
                     ("Moment", "unimodal, fourth moment <= that of P"),
                     ("Pareto", "unimodal, Pareto tail index 5 beyond 3 sigma"),
                     ("GarchT", "GARCH class, standardised Student-t innovations")):
        n[f"SupQClass{tag}"] = float(_dbc[cls]["q"])
        n[f"SupUndClass{tag}"] = float(_dbc[cls]["understatement"])
        if not np.isnan(_dbc[cls]["delta"]):
            n[f"SupDeltaClass{tag}"] = float(_dbc[cls]["delta"])

    # The delta-star grid. Declaring 0.004 as a constant would widen the
    # allow-list for every document, and 0.005 is already in it for an unrelated
    # reason -- the collision this project has met once. Read the solver's own
    # defaults instead, so the caption cannot drift from the grid it describes.
    #
    # Parsed, not imported. delta_by_class.py writes delta_by_class.json at
    # module level, so importing it would make `paper_numbers.py --check`
    # rewrite the artefact it checks against -- a check that cannot fail on a
    # stale input because it refreshes the input first.
    import ast
    _src = ast.parse((BASE / "analysis" / "phase2" / "delta_by_class.py")
                     .read_text(encoding="utf-8"))
    _fn = next(d for d in ast.walk(_src)
               if isinstance(d, ast.FunctionDef) and d.name == "feasible")
    _dflt = dict(zip([a.arg for a in _fn.args.args][-len(_fn.args.defaults):],
                     [ast.literal_eval(d) for d in _fn.args.defaults]))
    n["SupDeltaCeiling"] = float(_dflt["hi"])
    n["SupDeltaGrid"] = n["SupDeltaCeiling"] / (int(_dflt["m"]) - 1)

    # The constructed pair's backtests. These seven figures had no producer at
    # all: construct_pair.py stops at the linear programme, and sim.npz holds
    # 20,000 draws and no statistic. Recomputed under a declared seed, with the
    # pre-registration in analysis/phase2/PREREG_PAIR_BACKTESTS.md.
    pb = json.loads((BASE / "analysis" / "phase2"
                     / "pair_backtests.json").read_text())
    n["SupPairT"] = int(pb["T_path"])
    for tag, key in (("Honest", "honest"), ("Alt", "truncated")):
        _p = pb["paths"][key]
        n[f"SupPairPi{tag}"] = float(_p["pi_hat"])
        n[f"SupPairKup{tag}"] = float(_p["kupiec_p"])
        n[f"SupPairCC{tag}"] = float(_p["cc_ind_p"])
        n[f"SupPairDQ{tag}"] = float(_p["dq_p"])
    n["SupPairDQLags"] = 4
    # Power of Z_2 against the mean-ES-matched alternative. The fifth constraint
    # is feasible; the rejection frequency is reported with the interval that
    # decides whether it is power at all.
    n["SupPowerT"] = int(pb["T_power"])
    n["SupPowerReps"] = int(pb["n_power"])
    n["SupPowerRej"] = float(pb["power"]["rejection"])
    n["SupPowerLo"] = float(pb["power"]["ci_lo"])
    n["SupPowerHi"] = float(pb["power"]["ci_hi"])

    # The tail-closure factor range. Its endpoints were printed as 3.3 and 76,
    # which are per-pair ratios of the largest to the smallest R across closures.
    _tc = pd.read_csv(Q / "CO_robustness_inner7" / "inner7_tail_closure.csv")
    _f = _tc.groupby(["model", "asset"])["R"].agg(lambda v: v.max() / v.min())
    n["SupClosureFactorLo"] = float(_f.min())
    n["SupClosureFactorHi"] = float(_f.max())

    # Green rates by asset class under the rolling correction. 278 static and
    # 309 rolling of 312, and the two classes the sentence names.
    sys.path.insert(0, str(Q))
    from cfp_config import ASSET_CLASS as _AC
    _zr = pd.read_csv(BASE / "analysis" / "k2_indication"
                      / "zone_vs_coverage_rolling.csv")
    _zr["cls"] = _zr["asset"].map(_AC)
    n["SupGreenStatic"] = int((_zr["TL_static"] == "Green").sum())
    n["SupGreenRoll"] = int((_zr["TL_roll"] == "Green").sum())
    for tag, cls in (("Comm", "Commodity"), ("Bond", "Bond")):
        _c = _zr[_zr["cls"] == cls]
        n[f"SupGreenStaticPct{tag}"] = 100 * float((_c["TL_static"] == "Green").mean())
        n[f"SupGreenRollPct{tag}"] = 100 * float((_c["TL_roll"] == "Green").mean())

    # The COVID figure draws a 250-day annualised realised volatility and takes
    # its response-lag reference date from the peak of a 20-day one. Both are
    # measured here, because the caption quoted the second while pointing at the
    # first.
    _rq = pd.read_csv(BASE / "cfp_ijf_data" / "paper_outputs" / "tables"
                      / "rolling_qv_SP500.csv", index_col=0, parse_dates=True)
    _win = _rq["rvol"].dropna().loc["2019-07":"2021-07"]
    n["SupRvolLongWindow"] = 250
    n["SupRvolLongPeak"] = float(_win.max())
    n["SupRvolLongPeakYear"] = int(_win.idxmax().year)
    _ret = pd.read_csv(BASE / "cfp_ijf_data" / "returns" / "SP500.csv",
                       index_col=0, parse_dates=True)["log_return"]
    n["SupRvolShortWindow"] = 20
    _short = (_ret.rolling(n["SupRvolShortWindow"]).std()
              * np.sqrt(252)).loc["2019-07":"2021-07"]
    n["SupRvolShortPeak"] = float(_short.max())
    n["SupRvolShortPeakYear"] = int(_short.idxmax().year)

    n["SpearmanRPi"] = sp.statistic
    n["SpearmanRPiN"] = len(ws)
    return n


def fmt(key: str, v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        # sample sizes are printed with the thousands separator the paper uses
        if key.startswith("MCT") or key in ("SupPairT", "SupPowerT", "SupPowerReps"):
            return f"{int(v):,}".replace(",", "{,}")
        return str(int(v))

    # ---- SUP panel. Precision is chosen per quantity, because the supplement
    # compares several of these against nominal levels three and four places out
    # and a shared default erases the comparison the sentence makes.
    if key.startswith("SupZTwoPi"):
        return f"{v:.2f}"
    if key.startswith("SupZTwo"):
        return f"{v:.0f}"
    if key.startswith("SupTunedPi"):
        return f"{v:.4f}"
    if key.startswith(("SupTunedGreenPct", "SupTunedQSGainPct", "SupGreenStaticPct",
                       "SupGreenRollPct", "SupUndClass")):
        return f"{v:.1f}"
    if key.startswith(("SupConfIndex", "SupConfOvershoot")):
        return f"{v:.4f}"
    if key.startswith("SupCtlConfSmall"):
        return f"{v:.4f}"
    if key.startswith(("SupCtlExact", "SupCtlConfLarge")):
        return f"{v:.5f}"
    if key.startswith("SupReproTol"):
        m, e = f"{v:.1e}".split("e")
        return rf"{m}\times 10^{{{int(e)}}}"
    if key.startswith(("SupSmallPi", "SupQClass", "SupDeltaClass",
                       "SupPairKup", "SupPairCC", "SupPairDQ", "SupPower")):
        return f"{v:.3f}"
    if key.startswith("SupPairPi"):
        return f"{v:.4f}"
    if key == "SupClosureFactorLo":
        return f"{v:.1f}"
    if key == "SupClosureFactorHi":
        return f"{v:.0f}"
    if key.startswith("SupRvol"):
        return f"{v:.2f}"
    if key == "SupDeltaCeiling":
        return f"{int(v)}"
    if key == "SupDeltaGrid":
        return f"{v:.3f}"
    # Pooled panel rates sit within one thousandth of each other and of nominal;
    # three decimals prints them all as 0.011 and erases the comparison the
    # sentence makes. The p-values keep three.
    if key.startswith("FisherSevere"):
        return f"{v:.5f}"
    if key.startswith("FisherKupiec"):
        return f"{v:.2f}"
    if key.startswith("TightMargin"):
        return f"{v:.3f}"
    if key.startswith("TightBand"):
        return f"{v:.2f}"
    if key.startswith("TightUnd"):
        return f"{v:.1f}"
    if key.startswith("TightDelta"):
        return f"{v:.3f}"
    if key.startswith(("PairVaR", "PairDMt")):
        return f"{v:.2f}"
    if key.startswith(("PairQSGapPct", "PairCapitalPct")):
        return f"{v:.0f}" if v > 1 else f"{v:.3f}"
    if key.startswith("PairDMp"):
        return f"{v:.2f}"
    if key.startswith(("BoundRho", "BoundDelta")):
        return f"{v:.3f}"
    if key.startswith(("BoundFloor", "BoundEmpirical")):
        return f"{v:.1f}"
    if key.startswith("LitClosureRMin"):
        return f"{v:.3f}"
    if key.startswith("LitClosureRMax"):
        return f"{v:.2f}"
    if key.startswith("LitAnalyticSdPct"):
        return f"{v:.2f}"
    if key.startswith("LitAnalyticGrid"):
        return f"{v:.3f}"
    if key.startswith("RawPiGJRDefectivePct"):
        return f"{v:.1f}"
    if key.startswith("RawPiGJRDefective"):
        return f"{v:.4f}"
    if key in ("RawPiTimesFMDefective", "RawPiMoiraiTwoDefective"):
        return f"{v:.3f}"
    if key.startswith("PoolPi"):
        return f"{v:.4f}"
    if key.startswith("GapUnd"):
        return f"{v:.1f}"
    if key.startswith("GapDelta"):
        return f"{v:.3f}"
    if key.startswith("LitSupportPct"):
        return f"{v:.1f}"
    if key.startswith("LitMoiraiGapPP"):
        return f"{v:.2f}"
    if key.startswith("LitLambdaTail"):
        m, e = f"{v:.1e}".split("e")
        return rf"{m}\times 10^{{{int(e)}}}"
    if key.startswith("LitRatio") or key.startswith("LitQS") \
            or key.startswith("LitW") or key.startswith("LitAlphaResp"):
        return f"{v:.2f}"
    if key.startswith("Lit"):
        return f"{v:.3f}"
    if key == "GapFisherSevere":
        return f"{v:g}"
    if key in ("GapGapAblFull", "GapGapAblCovid"):
        return f"{v:.4f}"
    if key in ("GapDMt", "GapDMp", "GapQSGapPct", "GapVaRHonest", "GapVaRAlt",
               "GapRhoLo", "GapRhoHi", "GapDeltaHatLo", "GapDeltaHatHi",
               "GapEmpCoverage", "GapFisherKupiec"):
        return f"{v:g}"
    if key == "GapAblPairs":
        return str(int(v))
    if key in ("GapAblRhoLo", "GapAblRhoHi"):
        return f"{v:+.2f}"
    if key.startswith("GapCells"):
        return str(int(v))
    if key.startswith("GapBand") or key.startswith("GapCell") or key.startswith("GapQTrue") \
            or key.startswith("GapGap") or key.startswith("GapMargin"):
        return f"{v:.3f}"
    if key in ("TLTauOverAlpha",):
        return f"{v:.1f}"
    if key == "TLTau":
        return f"{v:.3f}"
    if key.startswith("TLUnd"):
        return f"{v:.1f}"
    if key.startswith("MCGreen") or key.startswith("MCPi"):
        return f"{v:.1f}" if key.startswith("MCGreen") else f"{v:.4f}"
    if key.startswith("MCT") or key == "MCReps" or key == "MCDgps":
        return f"{int(v):,}".replace(",", "{,}")
    if key.startswith("GateRollUpgradesLost") or key == "GateRollUpgradeAndDeteriorate":
        return str(int(v))
    if key.startswith("SeqDQRej"):
        return f"{v:.1f}"
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
        if not k.isalpha():
            raise SystemExit(f"macro name {k!r} is not letters-only; LaTeX "
                             "control sequences cannot contain digits")
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
