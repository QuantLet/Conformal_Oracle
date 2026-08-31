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


# Every rate the manuscript prints carries the number of observations it was
# computed over, because a rate on N observations lives on a grid of 1/N and
# cannot carry more resolution than that. Three defects in this project were
# this one shape: a dispersion tolerance on a quantity the defect left
# invariant, "agreement to four decimal places" on 40 dates where the grid is
# 0.025, and a family described as "0.6x to 1.0x nominal" on 200 dates where
# the grid is 0.5x. Three makes it a check rather than a note; guard 6 reads
# what this dict records.
RATE_N: dict[str, int] = {}


def rate(n: dict, key: str, value, n_obs: int):
    """Record a rate together with the sample size that sets its resolution."""
    n[key] = value
    RATE_N[key] = int(n_obs)
    return value


def collect() -> dict:
    RATE_N.clear()
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
    rate(n, "MainRawPiMin", ok["raw_pi"].min(), int(g["n_test"].sum()))
    rate(n, "MainRawPiMax", ok["raw_pi"].max(), int(g["n_test"].sum()))
    n["MainRMin"] = ok["R"].min()
    n["MainRMax"] = ok["R"].max()
    n["MainRTruncOne"] = t.set_index("model").loc["Chronos-Small", "R"]
    n["MainRTruncTwo"] = t.set_index("model").loc["Chronos-Mini", "R"]
    for key, model in (("TimesFM", "TimesFM-2.5"), ("MoiraiOne", "Moirai-1.1"),
                       ("MoiraiTwo", "Moirai-2.0"), ("LagLlama", "Lag-Llama"),
                       ("ChronosSmallA", "Chronos-Small-A"), ("ChronosMiniA", "Chronos-Mini-A"),
                       ("GJR", "GJR-GARCH"), ("GJRt", "GJR-GARCH-t"),
                       ("CAViaRAS", "CAViaR-AS")):
        # The three dynamic-quantile models have no series in the SEQ panel, so
        # their observation counts live in the verification files instead. An
        # absent count is an error, not a zero: a rate with N = 0 would divide
        # by zero in the resolution guard, and silently defaulting it would put
        # the rate back in the state the guard exists to end.
        _n = int(g[g["model"] == model]["n_test"].sum())
        if _n == 0:
            _alt = BASE / "analysis" / "k1_verify" / f"k1c_{model}.csv"
            if not _alt.is_file():
                raise SystemExit(f"no observation count for {model}; a rate "
                                 "cannot be emitted without the N that sets "
                                 "its resolution")
            _n = int(pd.read_csv(_alt)["n_test"].sum())
        rate(n, f"RawPi{key}", t.set_index("model").loc[model, "raw_pi"], _n)
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
    rate(n, "TruncCorPi", t.set_index("model").loc["Chronos-Small", "cor_pi"],
     int(g[g["model"] == "Chronos-Small"]["n_test"].sum()))
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
            rate(n, f"Pi{tag}{atag}", ar.loc[(model, a), "pihat"],
                 int(d[(d["model"] == model) & (d["alpha"] == a)]["n_test"].sum()))
            # tab_alpha_response's "kupiec" column counts assets that PASS, not
            # assets rejected: it is exactly 24 minus the rejection count in
            # all_results, at all four levels and for both series. Every other
            # Kupiec macro in this file counts rejections, so the name says
            # which. The prose using it already read "passing Kupiec on ...",
            # so nothing printed was wrong -- the trap was the name.
            n[f"KupPass{tag}{atag}"] = int(ar.loc[(model, a), "kupiec"])
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
    # What the same programme leaves unidentifiable at other scale edges. The
    # reduction from the unrestricted-shape figure is a property of imposing a
    # scale restriction at all; its size is a smooth function of where the edge
    # sits, so Table 2's row is not a measurement of these particular checks.
    # The trade-off the edge actually buys, measured on the 312 cells. Tightening
    # lowers the analytic residual and raises the false-positive count, and on
    # this panel it buys no additional detection at all: every truncated cell is
    # caught at every edge in the range.
    _sc = pd.read_csv(BASE / "analysis" / "phase2" / "panel_scale_ratios_by_asset.csv")
    _tr = _sc["series"].str.contains("default")
    _good, _bad = _sc[~_tr], _sc[_tr]
    n["BandGoodCells"] = int(len(_good))
    n["BandTruncCells"] = int(len(_bad))
    n["BandGoodWorstRatio"] = float(_good["ratio"].max())
    n["BandFPAtNow"] = int((_good["ratio"] > -1.80).sum())
    n["BandFPAtStrict"] = int((_good["ratio"] > -2.00).sum())
    n["BandFPAtVeryStrict"] = int((_good["ratio"] > -2.20).sum())
    n["BandTruncCaughtNow"] = int((_bad["ratio"] > -1.80).sum())
    n["BandTruncCaughtVeryStrict"] = int((_bad["ratio"] > -2.20).sum())
    n["BandVeryStrict"] = -2.20
    n["UndBandVeryStrict"] = float(_und(-2.20))
    assert n["BandTruncCaughtNow"] == n["BandTruncCaughtVeryStrict"] == len(_bad), \
        "tightening the edge now changes what it detects; the sentence must be rewritten"
    n["UndBandLoose"] = float(_und(-1.70))
    n["UndBandStrict"] = float(_und(-2.00))
    n["BandLoose"] = -1.70
    n["BandStrict"] = -2.00
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
    rate(n, "RawPiGJRDefective",
     float(gjr[gjr["series"] == "superseded"]["pihat"].mean()),
     int(gjr[gjr["series"] == "superseded"]["n"].sum()))
    n["RawPiGJRDefectivePct"] = 100 * n["RawPiGJRDefective"]

    # The same quantity for the two series whose lower quantile was stored with
    # an inverted sign. Section 8's failure-mode table printed these as literals
    # in the "with the defect" column while the "without" column beside them was
    # already a macro -- and guard 2 could not see them, because it replaced
    # every tabular before reading. Cell means over the 24 assets.
    sv = pd.read_csv(BASE / "analysis" / "recompute" / "sign_verification.csv")
    for tag, mdl in (("TimesFM", "TimesFM-2.5"), ("MoiraiTwo", "Moirai-2.0")):
        rate(n, f"RawPi{tag}Defective",
             float(sv[sv["model"] == mdl]["pihat_stored"].mean()),
             int(sv[sv["model"] == mdl]["n"].sum()))

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
        rate(n, f"PoolPi{tag}", float(wcl.loc[m, "pi_pooled"]),
             int(wcl.loc[m, "total_n"]))
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
    rate(n, "LitCorPiTypical", float(t["cor_pi"].median()),
     int(g["n_test"].sum() // t["model"].nunique()))
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
    rate(n, "MCPiTFive", _g("t5", 10000, "Raw_pi"),
     int(grid[(grid["dgp"] == "t5") & (grid["T"] == 10000)]["n_test"].iloc[0]) * 500)
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

    # The Christoffersen degeneracy, decomposed. The manuscript said a 1% tail
    # generates "too few exceedances to populate a transition table". K1b3
    # measured the three states and the table is populated in every case: zero
    # pairs have n11 = n10 = 0, and all of them have n11 = 0 with n10 > 0. What
    # is empty is one cell of the table, not the table.
    k1b3 = json.loads((BASE / "analysis" / "k1_verify"
                       / "k1b3_result.json").read_text())
    for tag, key in (("Raw", "raw"), ("Cor", "cor")):
        n[f"SeqCCBothZero{tag}"] = int(k1b3[key]["A"])
        n[f"SeqCCNoConsec{tag}"] = int(k1b3[key]["B"])
    assert n["SeqCCBothZeroRaw"] == 0 and n["SeqCCBothZeroCor"] == 0, \
        "a pair with an unpopulated transition table would change the sentence"

    # The Gneiting-Resin decomposition, from K0a. Two objects that are not the
    # same one: qV estimates the ARGMIN of the unconditional miscalibration
    # term, uMCB is the score reduction achieved there. The map between them
    # runs through the residual density, which is not constant across the panel.
    _k0 = json.loads((BASE / "analysis" / "k0a_mcb" / "k0a_result.json").read_text())
    n["MCBWellSpec"] = int(_k0["well_specified"]["n"])
    n["MCBRhoCal"] = float(_k0["well_specified"]["spearman_qV_delta_cal"])
    n["MCBRhoTest"] = float(_k0["well_specified"]["spearman_qV_delta_test"])
    # The implied residual density, uMCB ~ f qV^2 / 2. WITHIN one window: the
    # published 489 divided a test-window uMCB by a calibration-window qV, so it
    # carried the estimation noise of both and was a cross-window quantity
    # rather than a property of the density. Within window the span is 15.
    # The truncated series are excluded because their qV is not a small
    # displacement; pairs with non-positive uMCB are dropped, not clipped.
    _kp = pd.read_csv(BASE / "analysis" / "k0a_mcb" / "k0a_pairs.csv")
    _ok = _kp[(~_kp["truncated"].astype(bool)) & (_kp["uMCB_in_cal"] > 0)
              & (_kp["qV"].abs() > 0)].copy()
    _ok["f"] = 2 * _ok["uMCB_in_cal"] / _ok["qV"] ** 2
    _ok["gap"] = (_ok["qV"] - _ok["delta_test"]).abs()
    _ok = _ok[_ok["gap"] > 0]
    n["MCBDensityLo"] = float(_ok["f"].quantile(0.05))
    n["MCBDensityHi"] = float(_ok["f"].quantile(0.95))
    n["MCBDensitySpan"] = n["MCBDensityHi"] / n["MCBDensityLo"]
    n["MCBDensityPairs"] = int(len(_ok))

    # The mechanism: the standard deviation of a sample alpha-quantile is
    # sqrt(a(1-a)/n)/f, so the density predicts the size of the estimation error
    # and therefore the collapse in rank correlation across the window boundary.
    # Slope against a theoretical -1, and the magnitude with no fitted constant.
    from scipy import stats as _s3
    _lr = _s3.linregress(np.log(_ok["f"]), np.log(_ok["gap"]))
    n["MCBGapSlope"] = float(_lr.slope)
    n["MCBGapSlopeSE"] = float(_lr.stderr)
    n["MCBGapSlopeSigma"] = abs(n["MCBGapSlope"] + 1.0) / n["MCBGapSlopeSE"]
    _pred = np.sqrt(0.01 * 0.99 * (1 / _ok["n_cal"] + 1 / _ok["n_test"])) / _ok["f"]
    n["MCBGapMagRatio"] = float(_ok["gap"].median() / _pred.median())
    # And the branch that did NOT hold: supplying the density recovers nothing
    # across the window boundary.
    n["MCBRhoPlain"] = float(_s3.spearmanr(_ok["qV"].abs(),
                                           _ok["uMCB_in_test"]).statistic)
    n["MCBRhoConverted"] = float(_s3.spearmanr(0.5 * _ok["f"] * _ok["qV"] ** 2,
                                               _ok["uMCB_in_test"]).statistic)

    # The one-bin rule and where it fails. The closed-form correction
    # `stored - binwidth * scale` was validated on a 200-date block and holds
    # panel-wide on 99.47% of dates; it fails where the bin the CDF crosses
    # moves index by one, which the 200-date block contains none of. A sample
    # that small could not have exhibited a failure mode this rare.
    _rb = json.loads((BASE / "analysis" / "k1_verify"
                      / "k1a_verify_rebuild.json").read_text())
    _lv = _rb["chronos_small_analytic"]["levels"]["0.01"]
    n["RuleRows"] = int(_rb["chronos_small_analytic"]["n_rows"])
    n["RuleExactPct"] = 100 * float(_lv["frac_exactly_one_bin"])
    n["RuleFailures"] = int(round((1 - _lv["frac_exactly_one_bin"]) * n["RuleRows"]))
    n["RuleFailPct"] = 100 - n["RuleExactPct"]
    n["RuleValidationDates"] = 200
    # The probability that a block of that size drawn from the panel contains
    # none of them, which is what makes the validation uninformative here.
    n["RuleMissProb"] = 100 * float(_lv["frac_exactly_one_bin"]) ** n["RuleValidationDates"]

    # The best raw Kupiec count on the SEQ panel. MainBestKupiec is 15 and
    # belongs to CAViaR-AS, which has no stored series and is therefore not in
    # the 312-pair panel at all. Section 7.4 declares the SEQ panel in its own
    # sentence and then used the MAIN figure -- Rule 1's unit confusion, in the
    # section that argues coverage tests cannot separate.
    _sq = g.assign(_p=g["p_kup_raw"] > 0.05).groupby("model")["_p"].sum()
    n["SeqBestKupiec"] = int(_sq.max())
    # The gap ablation in percentage points, which is the unit S.4.2 states it
    # in. "At most 0.2 pp" is wrong in both directions: 0.05 over the full
    # window and 0.58 once the crisis sub-windows are included.
    n["GapAblFullPP"] = 100 * n["GapGapAblFull"]
    n["GapAblCovidPP"] = 100 * n["GapGapAblCovid"]
    assert n["SeqBestKupiec"] <= n["MainBestKupiec"], \
        "the sequence panel cannot beat the main panel it is a subset of"

    # The frontier figure's own series list, parsed from the script that draws
    # it. Its caption said "four representative forecasters" while the figure
    # showed ten; a caption and a plot drifting apart is the same defect class
    # as a table note and its own column.
    import ast as _ast2
    _fr = _ast2.parse((BASE / "Quantlets" / "CFP_Calibration_Efficiency_Frontier"
                       / "run_frontier.py").read_text(encoding="utf-8"))
    _lists = {t.targets[0].id: _ast2.literal_eval(t.value)
              for t in _fr.body if isinstance(t, _ast2.Assign)
              and getattr(t.targets[0], "id", "") in ("TSFMS", "BENCHMARKS")}
    n["FrontierTsfm"] = len(_lists["TSFMS"])
    n["FrontierBench"] = len(_lists["BENCHMARKS"])
    n["FrontierSeries"] = n["FrontierTsfm"] + n["FrontierBench"]

    # The corrected column's two axes. The rate is flat across forecasters; the
    # Kupiec count is not, so the claim that the corrected column "cannot rank"
    # holds of the violation rate and not of the whole column.
    rate(n, "MainCorPiLo", float(t["cor_pi"].min()), int(g["n_test"].sum()))
    rate(n, "MainCorPiHi", float(t["cor_pi"].max()), int(g["n_test"].sum()))
    n["MainCorKupLo"] = int(t["cor_kup"].min())
    n["MainCorKupHi"] = int(t["cor_kup"].max())

    # The separation gap of Theorem 4.5, applied to all 312 cells. Only the
    # DIFFERENCES enter the manuscript: the levels come from a standalone
    # recomputation in run_gap_panel.py and are not the pipeline's headline
    # counts, which are produced elsewhere and differ in window construction.
    gp = pd.read_csv(BASE / "analysis" / "convention" / "gap_panel.csv")
    n["GapPanelCells"] = int(len(gp))
    n["GapPanelGapLo"] = int(gp["gap"].min())
    n["GapPanelGapHi"] = int(gp["gap"].max())
    n["GapPanelGapMed"] = int(gp["gap"].median())
    n["GapPanelGapPct"] = 100 * float(gp["gap"].median() / gp["g0_n_test"].median())
    n["GapPanelDPiMed"] = float(gp["dpi"].median())
    n["GapPanelDPiMax"] = float(gp["dpi"].max())
    n["GapPanelZoneChanges"] = int((gp["g0_TL"] != gp["gn_TL"]).sum())
    n["GapPanelKupFlips"] = int(((gp["g0_p_kupiec"] > 0.05)
                                 != (gp["gn_p_kupiec"] > 0.05)).sum())

    # ---- Monte Carlo grid cells quoted in Section 5's prose ---------------- #
    # Section 5 lived in an \input file guard 2 never read, so its prose carried
    # 46 typed figures. Every one is a cell of grid.csv. The five sample sizes
    # are tagged One..Five in the order 500, 1,000, 2,000, 5,000, 10,000.
    _mg = pd.read_csv(BASE / "analysis" / "k2_sim" / "grid.csv").set_index(["dgp", "T"])
    _TT = [(500, "One"), (1000, "Two"), (2000, "Three"), (5000, "Four"), (10000, "Five")]
    _DD = [("normal", "Normal"), ("t5", "TFive"), ("t3", "TThree"),
           ("skewt3", "Skew"), ("mixnormal", "Mix")]
    for _d, _dt in _DD:
        for _t, _tt in _TT:
            _row = _mg.loc[(_d, _t)]
            rate(n, f"MCPi{_dt}{_tt}", float(_row["Corr_pi"]), int(_row["n_test"]) * 500)
            n[f"MCGrn{_dt}{_tt}"] = float(_row["RawGreen"])
            n[f"MCQv{_dt}{_tt}"] = float(_row["Mean_qV"])
            n[f"MCSd{_dt}{_tt}"] = float(_row["Std_qV"])
        rate(n, f"MCRaw{_dt}", float(_mg.loc[(_d, 10000), "Raw_pi"]),
             int(_mg.loc[(_d, 10000), "n_test"]) * 500)
    # Derived statements the prose makes about the grid.
    # "From T = 1,000 onward" -- the T = 500 row is excluded by the sentence,
    # and it is the row the small-sample qualification applies to.
    _cp = _mg[_mg.index.get_level_values("T") >= 1000]["Corr_pi"]
    _n_ge = int((_mg[_mg.index.get_level_values("T") >= 1000]["n_test"] * 500).min())
    rate(n, "MCCorPiMaxDev", float((_cp - 0.01).abs().max()), _n_ge)
    _small = _mg.xs(500, level="T")
    _n_sm = int((_small["n_test"] * 500).min())
    rate(n, "MCCorPiSmallLo", float(_small["Corr_pi"].min()), _n_sm)
    rate(n, "MCCorPiSmallHi", float(_small["Corr_pi"].max()), _n_sm)
    n["MCSdRatioNormal"] = float(_mg.loc[("normal", 500), "Std_qV"]
                                 / _mg.loc[("normal", 10000), "Std_qV"])
    n["MCSqrtTwenty"] = float(np.sqrt(20))
    # What a reader gets by dividing the two figures as printed. Emitted so the
    # sentence that warns about it does not itself carry a typed literal.
    n["MCSdRatioNaive"] = (float(f"{_mg.loc[('normal', 500), 'Std_qV']:.5f}")
                           / float(f"{_mg.loc[('normal', 10000), 'Std_qV']:.5f}"))
    for _t, _tt in _TT:
        _ncal = int(0.70 * _t)
        n[f"MCOver{_tt}"] = int(np.ceil((_ncal + 1) * 0.99)) / _ncal - 0.99
    # The traffic light's two boundaries in units of alpha.
    # The zone boundaries as Proposition 5.1 states them: green at most 4
    # exceedances per 250 days, yellow up to 9. TLTauOverAlpha uses 4/250, so
    # the yellow edge is 9/250 and not 9.5/250.
    n["TLTauYellowOverAlpha"] = (9.0 / 250) / 0.01
    # Population violation rates of the five DGPs under a Normal-innovation
    # forecaster: a property of the design, taken from the table's own emitter
    # so prose and table cannot diverge.
    import ast as _ast5
    _et5 = _ast5.parse((BASE / "analysis" / "k2_sim" / "emit_mc_table.py")
                       .read_text(encoding="utf-8"))
    _pop = next(_ast5.literal_eval(t.value) for t in _et5.body
                if isinstance(t, _ast5.Assign) and getattr(t.targets[0], "id", "") == "POP")
    for _d, _dt in _DD:
        n[f"MCPop{_dt}"] = float(_pop[_d])
    # How far the normal approximation to the green-zone probability sits from
    # the exact binomial, averaged over the 25 cells.
    n["MCNormalApproxPP"] = float(json.loads(
        (BASE / "analysis" / "k2_sim" / "prop_tl_check.json").read_text()
    )["mean_abs_gap_vs_normal_approx"])

    # ---- the effective level of the conformal shift ------------------------ #
    # Equation (8) returns the k-th smallest of n scores with
    # k = ceil((n+1)(1-alpha)), capped at n. So the estimator targets k/n, not
    # 1-alpha, and the overshoot is exactly alpha wherever k reaches n. The
    # comparison against the sampling standard deviation is computed rather
    # than asserted because the overshoot is a sawtooth in n -- it drops when
    # the ceiling increments and rises between -- so there is no single
    # crossover and a claim of one would be wrong.
    from math import ceil as _ceil

    def _overshoot(_n, _a=0.01):
        return min(_ceil((_n + 1) * (1 - _a)), _n) / _n - (1 - _a)

    _sd_level = lambda _n, _a=0.01: np.sqrt(_a * (1 - _a) / _n)
    n["LevelBound"] = int(2 / 0.01 - 1 - 1)
    n["LevelMaxOvershoot"] = 0.01
    _dom = [_n for _n in range(20, 3001) if _overshoot(_n) > _sd_level(_n)]
    _runs, _cur = [], [_dom[0]]
    for _x in _dom[1:]:
        if _x == _cur[-1] + 1:
            _cur.append(_x)
        else:
            _runs.append((_cur[0], _cur[-1])); _cur = [_x]
    _runs.append((_cur[0], _cur[-1]))
    n["LevelDomRuns"] = len(_runs)
    n["LevelDomOneLo"], n["LevelDomOneHi"] = _runs[0]
    n["LevelDomTwoLo"], n["LevelDomTwoHi"] = _runs[1]
    assert all(abs(_overshoot(_n) - 0.01) < 1e-12
               for _n in range(20, n["LevelBound"] + 1)), \
        "the overshoot is no longer exactly alpha in the degenerate regime"
    assert n["LevelDomOneHi"] == n["LevelBound"], \
        "the first dominance run no longer ends at the degeneracy boundary"

    # ---- the rolling window sweep, K4b ------------------------------------- #
    # Section 3.2.1 promises Section 7 reports what w = 125 does. The three
    # windows do not estimate the same quantity: k = ceil((w+1)(1-alpha)) makes
    # the effective level k/w depend on w, and the overshoot over nominal falls
    # by a factor of five across the sweep. Unit: one cell is one forecaster x
    # one asset x one window, 312 cells per window.
    _ws = pd.read_csv(BASE / "analysis" / "phase3_windows" / "w_sweep.csv")
    n["WSweepCells"] = int(len(_ws) // _ws["w"].nunique())
    for w in (125, 250, 500):
        _gw = _ws[_ws["w"] == w]
        tag = {125: "Short", 250: "Mid", 500: "Long"}[w]
        n[f"WSweep{tag}W"] = int(w)
        n[f"WSweep{tag}K"] = int(_gw["k"].iloc[0])
        n[f"WSweep{tag}Level"] = float(_gw["k"].iloc[0]) / w
        n[f"WSweep{tag}Overshoot"] = n[f"WSweep{tag}Level"] - 0.99
        n[f"WSweep{tag}Shift"] = float(_gw["mean_shift"].median())
        n[f"WSweep{tag}Sd"] = float(_gw["sd_shift"].median())
        rate(n, f"WSweep{tag}Pi", float(_gw["pi_hat"].median()),
             int(_gw["n"].sum()))
    _pv = _ws.pivot_table(index=["model", "asset"], columns="w", values="sd_shift")
    _sh = _ws.pivot_table(index=["model", "asset"], columns="w", values="mean_shift")
    n["WSweepSdRatioShort"] = float((_pv[125] / _pv[250]).median())
    n["WSweepSdRatioLong"] = float((_pv[250] / _pv[500]).median())
    n["WSweepSqrtTwo"] = float(np.sqrt(2))
    n["WSweepExpShort"] = float(np.log(n["WSweepSdRatioShort"]) / np.log(2))
    n["WSweepExpLong"] = float(np.log(n["WSweepSdRatioLong"]) / np.log(2))
    n["WSweepShiftLargerShort"] = int((_sh[125] > _sh[250]).sum())
    n["WSweepBound"] = int(2 / 0.01 - 1 - 1)   # k >= w whenever w < 2/alpha - 1
    assert n["WSweepShortLevel"] == 1.0, \
        "w = 125 no longer targets the window maximum; the sentence changes"

    # ---- the harness's own defect census ----------------------------------- #
    # Counted from PROTOCOL.md's table rather than typed, so the figure in the
    # manuscript cannot drift from the register it summarises. One row is one
    # defect found during the audit, classified by where it sat.
    _pr = (BASE / "analysis" / "provenance" / "PROTOCOL.md").read_text()
    _tbl = _pr.split("### The defect is in the instrument more often than in the object")[1]
    _tbl = _tbl.split("The fourth is the sharpest")[0]
    _where = [l.split("|")[1].strip() for l in _tbl.splitlines()
              if l.startswith("|") and not set(l) <= set("|- ")
              and l.split("|")[1].strip() in ("instrument", "object")]
    n["HarnessDefectsInstrument"] = _where.count("instrument")
    n["HarnessDefectsObject"] = _where.count("object")
    n["HarnessDefectsTotal"] = len(_where)
    assert n["HarnessDefectsTotal"] >= 4, \
        "the defect census parsed fewer rows than the register carries"

    # ---- ML panel: 40 cells, 2 estimators x 4 assets x 5 leaf settings ----- #
    # A separate unit from MAIN and SEQ, and never pooled with them. It exists to
    # answer one question Section 7 left open: whether anything occupies the
    # range between the well-specified and truncated populations. The panel tag
    # is "ML", and every count below is out of 40 cells at 200 dates.
    mlc = pd.read_csv(BASE / "analysis" / "ml" / "gate_cells.csv")
    n["MLCells"] = int(len(mlc))
    n["MLDates"] = 200
    n["MLAssets"] = int(mlc["asset"].nunique())
    n["MLLeafSettings"] = int(mlc["leaf"].nunique())
    # The coverage statistic's resolution, stated because the sentence that
    # motivated this exercise quoted a value off the grid: at 200 dates and
    # alpha = 0.01 the expected count is 2 and pi-hat/alpha moves in halves.
    n["MLPiGrid"] = 1.0 / (n["MLDates"] * 0.01)
    for tag, est in (("Gbm", "LightGBM"), ("Qrf", "quantile forest")):
        _e = mlc[mlc["est"] == est]
        n[f"ML{tag}Cells"] = int(len(_e))
        n[f"ML{tag}Blocked"] = int(_e["blocked"].sum())
        n[f"ML{tag}RatioLo"] = float(_e["ratio"].min())
        n[f"ML{tag}RatioHi"] = float(_e["ratio"].max())
    n["MLBelowLowerEdge"] = int(mlc["below_lower"].sum())
    n["MLRatioMostNegative"] = float(mlc["ratio"].min())
    n["MLLowerEdgeMargin"] = abs(-3.5 - n["MLRatioMostNegative"])
    _b = mlc[mlc["blocked"]]
    n["MLUnderThreshold"] = 2.5
    n["MLBlockedUnder"] = int((_b["pi_ratio"] >= n["MLUnderThreshold"]).sum())
    n["MLBlockedNotUnder"] = int(len(_b) - n["MLBlockedUnder"])
    n["MLPassedUnder"] = int(((~mlc["blocked"])
                              & (mlc["pi_ratio"] >= n["MLUnderThreshold"])).sum())
    # Cells the tightened edge of Table 2 row 4 adds over the standing edge.
    n["MLNewlyBlockedByTightening"] = int(
        ((mlc["ratio"] <= -1.800) & (mlc["ratio"] > -1.940)).sum())

    # The dose-response itself, which is what places the exhibit beside the
    # tail-sparsity remark rather than beside the sampling section: the tail
    # moves by an order of magnitude across the leaf-size grid while the centre
    # does not move. Aggregation is stated because it changes the answer --
    # these pool over the 4 assets at each leaf setting, then take the ratio;
    # a spread computed over asset-by-leaf cells is a different number.
    _dr = pd.read_csv(BASE / "analysis" / "ml" / "dose_response_raw.csv")
    _pi = _dr.assign(h=(_dr["realised"] < _dr["lgbm_q"]).astype(float)) \
        .groupby("leaf")["h"].mean()
    n["MLGbmLeafDefault"] = 20
    # The two dates that make the ML test out of sample, read from git rather
    # than typed, so the claim "the band was fixed before the family was run"
    # is checkable by the reader with the same command.
    import subprocess as _sp
    def _first_commit_date(path, pickaxe=None):
        cmd = ["git", "log", "--format=%ad", "--date=format:%-d %B %Y"]
        if pickaxe:
            cmd += ["-S", pickaxe]
        cmd += ["--", path]
        out = _sp.run(cmd, cwd=BASE, capture_output=True, text=True).stdout.split("\n")
        out = [x for x in out if x.strip()]
        return out[-1] if out else ""
    n["MLBandDeclared"] = _first_commit_date(
        "analysis/provenance/DECLARED_CONSTANTS.md", "-3.5, -1.8")
    n["MLPanelRun"] = _first_commit_date("analysis/ml/dose_response_raw.csv")
    assert n["MLBandDeclared"] and n["MLPanelRun"], \
        "the out-of-sample claim rests on two git dates and one is missing"
    # The knob threshold, read from the emitter that applies it rather than
    # typed twice. Declared in drafts/prereg_ml.md before the knob arm ran.
    import ast as _ast
    _et = _ast.parse((BASE / "analysis" / "ml" / "emit_dose_tables.py")
                     .read_text(encoding="utf-8"))
    n["MLKnobThreshold"] = float(next(
        _ast.literal_eval(t.value) for t in _et.body
        if isinstance(t, _ast.Assign) and getattr(t.targets[0], "id", "") == "KNOB_THRESHOLD"))
    n["MLPooledObs"] = int((_dr["leaf"] == 20).sum())
    n["MLPooledGrid"] = 1.0 / (n["MLPooledObs"] * 0.01)
    rate(n, "MLGbmPiDefault", float(_pi.loc[n["MLGbmLeafDefault"]]),
         int((_dr["leaf"] == n["MLGbmLeafDefault"]).sum()))
    _mono = _pi[_pi.index >= 5]           # leaf 1 -> 5 is not monotone; reported so
    n["MLGbmTailSpan"] = float(_mono.max() / _mono.min())
    _cen = _dr.assign(c=_dr["lgbm_med"] / _dr["train_sd"]).groupby("leaf")["c"].median()
    n["MLGbmCentreSpread"] = float(_cen.max() - _cen.min())
    _qpi = _dr.assign(h=(_dr["realised"] < _dr["qrf_q"]).astype(float)) \
        .groupby("leaf")["h"].mean()
    n["MLQrfTailSpan"] = float(_qpi.max() / _qpi.min())

    # Where the upper edge would block a CORRECTLY specified forecaster: the
    # 1% quantile of its standardised innovation law, in sigma units. Closed
    # form, so the limitation is bounded rather than left open.
    from scipy import stats as _s2
    n["MLQuantUniform"] = float(_s2.uniform(-np.sqrt(3), 2 * np.sqrt(3)).ppf(0.01))
    n["MLQuantTriangular"] = float(
        _s2.triang(0.5, loc=-np.sqrt(6), scale=2 * np.sqrt(6)).ppf(0.01))
    n["MLQuantNormal"] = float(_s2.norm.ppf(0.01))
    n["MLQuantTFive"] = float(_s2.t.ppf(0.01, 5) / np.sqrt(5 / 3))
    # The worked example at -1.9 sigma, in coverage units rather than threshold
    # units. Section 7 called it under-conservative "by a quarter", which is a
    # statement about the threshold, and concluded the block would be a false
    # positive, which is a statement about coverage. The two differ by a factor.
    n["MLWorkedExampleSigma"] = 1.9
    n["MLWorkedNormal"] = _s2.norm.cdf(-1.9) / 0.01
    n["MLWorkedTFive"] = _s2.t.cdf(-1.9 * np.sqrt(5 / 3), 5) / 0.01
    n["MLWorkedTThree"] = _s2.t.cdf(-1.9 * np.sqrt(3.0), 3) / 0.01

    # ---- claims that were half macro-backed -------------------------------- #
    # Check 5 of audit_structural_claims.py skipped any "N of M" whose M was a
    # macro, on the reasoning that one side came from an artefact. Eleven claims
    # had that shape and none was ever checked. Their typed sides are lifted
    # here; two of the eleven were wrong.

    # The order-statistic convention against the level convention. Neither the
    # median gap nor the "none" is catchable by guard 2: one is 3x10^-4 with an
    # integer mantissa, the other is a word.
    _cv = (BASE / "analysis" / "provenance" / "QV_CONVENTION.md").read_text()
    _row = next(l for l in _cv.splitlines() if l.startswith("| `LEVEL_K_OVER_N`")
                and "e-04" in l)
    _cells = [c.strip() for c in _row.split("|")]
    n["SeqLevelGapMedian"] = float(_cells[2])
    n["SeqLevelChanged"] = int(_cells[4].split("of")[0].replace("*", "").strip())

    # Kupiec rejection counts on the corrected series, across the four levels.
    # "5, 9, 5 and 4" was printed for the analytic series; the panel gives 8 at
    # alpha = 0.025, with ASX200 at p = 0.059 just outside the 5% edge.
    _alphas = ((0.01, "One"), (0.025, "TwoFive"), (0.05, "Five"), (0.10, "Ten"))
    for mdl, tag in (("Chronos-Small-A", "SmallAnalytic"),
                     ("Chronos-Small", "SmallDefault")):
        _m = g if False else d[d["model"] == mdl]
        for a, atag in _alphas:
            _c = _m[_m["alpha"] == a]
            n[f"MainKupRejCor{tag}{atag}"] = int((_c["p_kup_cp"] < 0.05).sum())
        n[f"MainKupRejRaw{tag}One"] = int(
            (_m[_m["alpha"] == 0.01]["p_kup_raw"] < 0.05).sum())

    # The Acerbi-Szekely separation, and the ES-correction counts.
    _es = pd.read_csv(Q / "CFP_ES_Correction_Z2" / "table_c1_es_correction.csv")
    _pv = _es.groupby("model")[["raw_pass", "corr_pass"]].sum()
    _trunc, _anal = ["Chronos-Small", "Chronos-Mini"], ["Chronos-Small-A", "Chronos-Mini-A"]
    assert _pv.loc[_trunc, "raw_pass"].nunique() == 1, "the two truncated series differ"
    assert _pv.loc[_anal, "raw_pass"].nunique() == 1, "the two analytic series differ"
    n["MainZTwoPassTrunc"] = int(_pv.loc[_trunc[0], "raw_pass"])
    n["MainZTwoPassAnalytic"] = int(_pv.loc[_anal[0], "raw_pass"])
    n["MainZTwoRejCorrTrunc"] = int(n["MainAssets"] - _pv.loc[_trunc, "corr_pass"].max())
    _rest = _pv.drop(index=_trunc)
    n["MainZTwoPassRawMin"] = int(_rest["raw_pass"].min())
    n["MainZTwoPassCorrMin"] = int(_rest["corr_pass"].min())

    # Dynamic-quantile rejections on the raw series, the counts Section 8 uses
    # to say the test separates nothing.
    _dq = pd.read_csv(BASE / "analysis" / "phase3" / "dq_panel.csv")
    _dqr = _dq.assign(r=_dq["p_dq_raw"] < 0.05).groupby("model")["r"].sum()
    assert _dqr[["GARCH-N", "EWMA", "GJR-GARCH"]].nunique() == 1, \
        "the three benchmarks no longer share a DQ rejection count"
    n["MainDQRejBenchmark"] = int(_dqr["GARCH-N"])
    n["MainDQRejGjrT"] = int(_dqr["GJR-GARCH-t"])

    # Rolling conditional coverage. "0--5" for the high-R-bar series is not in
    # the artefact: the three largest R-bar -- Lag-Llama and the two truncated
    # Chronos -- pass on 2 to 5 assets, and no series passes on none.
    #
    # The class is the table's own "kind" column, not a cut on R-bar. A cut at
    # 0.35 would have been a constant chosen after seeing the data, and the
    # rank-of-three that would replace it is not a natural grouping either:
    # Lag-Llama sits at 0.357 against 0.184 for the next series down, a factor of
    # 1.9, while the two truncated Chronos are at 17 and 24. An assertion
    # demanding a clear gap below the top three fired, which is the check doing
    # its job on a grouping written before it was measured.
    _rs = pd.read_csv(Q / "CO_garch_conformal" / "tab_rolling_vs_static.csv") \
        .set_index("model")
    _rb = pd.read_csv(Q / "CO_full_evaluation" / "tab_master_results_r2.csv") \
        .set_index("model")["R"]
    n["SupRollCCBest"] = int(_rs["r_cc"].max())
    _kind = pd.read_csv(Q / "CO_full_evaluation" / "tab_master_results_r2.csv") \
        .set_index("model")["kind"]
    _tsfm = [m for m in _kind[_kind.str.startswith("TSFM")].index if m in _rs.index]
    n["SupRollCCTsfmN"] = len(_tsfm)
    n["SupRollCCTsfmLo"] = int(_rs.loc[_tsfm, "r_cc"].min())
    n["SupRollCCTsfmHi"] = int(_rs.loc[_tsfm, "r_cc"].max())
    assert n["SupRollCCTsfmHi"] < n["SupRollCCBest"], \
        "a foundation-model series now matches the best benchmark on rolling CC"
    n["SupHistSimGreenStatic"] = int(_rs.loc["Hist-Sim", "s_grn"])
    n["SupHistSimGreenRoll"] = int(_rs.loc["Hist-Sim", "r_grn"])

    # Forecasters green on every asset after the rolling correction.
    _zr2 = pd.read_csv(BASE / "analysis" / "k2_indication"
                       / "zone_vs_coverage_rolling.csv")
    n["SeqRollAllAssets"] = int(
        (_zr2.assign(x=_zr2["TL_roll"] == "Green").groupby("model")["x"].sum()
         == n["MainAssets"]).sum())

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
        rate(n, f"SupZTwoPi{tag}", float(z2v.loc[mdl, "pihat_at_ES_level"]),
             int(d[(d["model"] == mdl) & (d["alpha"] == 0.025)]["n_test"].sum()))
        n[f"SupZTwo{tag}"] = float(z2v.loc[mdl, "Z2_canonical_median"])
        # Dividing by the stored (negative) column maps z -> 2 - z, which is why
        # the defective routine returned a large positive statistic.
        n[f"SupZTwoFlipped{tag}"] = 2.0 - n[f"SupZTwo{tag}"]
    n["SupZTwoAssets"] = int(z2v.loc["Chronos-Small", "n"])
    # The time-averaged variant, which is what Table S.2 reports. Both the mean
    # and the median are emitted because the reconciliation turns on the fact
    # that switching aggregation does not close the gap -- the denominator does.
    _es2 = pd.read_csv(Q / "CFP_ES_Correction_Z2" / "table_c1_es_correction.csv")
    for tag, mdl in (("Small", "Chronos-Small"), ("Mini", "Chronos-Mini")):
        _r = _es2[_es2["model"] == mdl]["z2_raw"]
        n[f"SupZTwoModMean{tag}"] = float(_r.mean())
        n[f"SupZTwoModMedian{tag}"] = float(_r.median())
        n[f"SupZTwoRatio{tag}"] = abs(n[f"SupZTwo{tag}"]) / abs(n[f"SupZTwoModMean{tag}"])

    # The tuned GBM-QR ablation. The prose reported 5/9, 0/9 and 88.9% Green
    # from REPRO_NOTES_E1.md, which describes a nine-model run; the shipped grid
    # is 8 configurations x 13 models and no count in it is 8/9.
    tg = pd.read_csv(Q / "CO_baseline_comparison_tuned" / "tuned_gbm_qr_grid.csv")
    _cfg = lambda ne, d, lr: tg[(tg["n_est"] == ne) & (tg["max_depth"] == d)
                                & (tg["lr"] == lr)]
    _best, _cons = _cfg(100, 3, 0.05), _cfg(100, 3, 0.01)
    n["SupTunedSeries"] = int(len(_best))
    rate(n, "SupTunedPiBest", float(_best["pi_hat"].mean()),
         int(_best["n_test"].sum()))
    rate(n, "SupTunedPiCons", float(_cons["pi_hat"].mean()),
         int(_cons["n_test"].sum()))
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
    # The same index at the calibration size Section 4.3 uses as its example.
    n["TheoryNCal"] = int(_ctl[("t5", 1000)]["n_cal"])
    n["TheoryConfIndex"] = int(np.ceil((n["TheoryNCal"] + 1) * 0.99)) / n["TheoryNCal"]
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
    # The harness defect the gate revision records: the normal-formula standard
    # error understates the true sampling variability of a 1% order statistic,
    # whose replication distribution is skewed.
    # Parsed by column position from the table's own header, not by pattern
    # matching over the prose: a first attempt matched the kurtosis column and
    # returned 4.16 for the skewness, which is the kind of error a regex over a
    # document makes silently.
    _gr = (BASE / "analysis" / "k2_sim" / "GATE_REVISION.md").read_text().splitlines()
    _hdr = next(i for i, l in enumerate(_gr)
                if l.startswith("| cell") and "skew" in l and "ratio" in l)
    _cols = [c.strip() for c in _gr[_hdr].split("|")[1:-1]]
    _isk, _ira = _cols.index("skew"), _cols.index("ratio")
    _sk, _ra = [], []
    for l in _gr[_hdr + 2:]:
        if not l.startswith("|"):
            break
        f = [c.strip().replace("*", "").replace("\u00d7", "") for c in l.split("|")[1:-1]]
        _sk.append(float(f[_isk])); _ra.append(float(f[_ira]))
    n["MCSkewMax"] = max(_sk)
    n["MCSEUnderstate"] = max(_ra)
    assert 1.0 < n["MCSEUnderstate"] < 3.0 and 0.5 < n["MCSkewMax"] < 3.0, \
        "the gate-revision table no longer parses to plausible values"
    n["SupReproTolMax"] = float(_tol.max())
    n["SupReproCells"] = int(len(_tol))

    # The corrected rate in the degenerate small-sample regime, where k >= n and
    # the conformal shift is the window maximum.
    _ss = pd.read_csv(Q / "CO_robustness" / "study1_small_sample.csv")
    # A Monte Carlo mean, so the resolution is the test window times the number
    # of replications, not the window alone: 75 observations per replication
    # would put the grid at 0.013 and make 0.008 against 0.010 unreadable.
    _n250 = _ss[_ss["T"] == 250]
    rate(n, "SupSmallPi", float(_n250["mean_corr_pi"].median()),
         int((250 - round(0.70 * 250)) * _n250["n_valid"].min()))
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
        rate(n, f"SupPairPi{tag}", float(_p["pi_hat"]), int(pb["T_path"]))
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
        if key.startswith("MCT") or key in ("SupPairT", "SupPowerT",
                                           "SupPowerReps", "RuleRows"):
            return f"{int(v):,}".replace(",", "{,}")
        return str(int(v))

    # delta-star at four decimals in both tables. At three, the gate-band row
    # (0.0244) and the Pareto row (0.0236) both print 0.024 while their
    # understatements differ, which reads as a table error. The bisection
    # resolves delta to about 5e-7, so four places claim nothing the
    # computation does not have.
    if key in ("GapDeltaFree", "GapDeltaUni", "GapDeltaGateNow",
               "GapDeltaMoment", "SupDeltaClassFree", "SupDeltaClassUni",
               "SupDeltaClassMoment", "SupDeltaClassPareto"):
        return f"{v:.4f}"
    # ---- SUP panel. Precision is chosen per quantity, because the supplement
    # compares several of these against nominal levels three and four places out
    # and a shared default erases the comparison the sentence makes.
    if key.startswith("SupZTwoPi"):
        return f"{v:.2f}"
    if key.startswith("SupZTwoRatio"):
        return f"{v:.1f}"
    if key.startswith("SupZTwoMod"):
        return f"{v:.1f}"
    if key.startswith("SupZTwo"):
        return f"{v:.0f}"
    if key.startswith("SupTunedPi"):
        return f"{v:.4f}"
    if key.startswith(("SupTunedGreenPct", "SupTunedQSGainPct", "SupGreenStaticPct",
                       "SupGreenRollPct", "SupUndClass")):
        return f"{v:.1f}"
    if key.startswith(("SupConfIndex", "SupConfOvershoot", "TheoryConfIndex")):
        return f"{v:.4f}"
    if key.startswith("SupCtlConfSmall"):
        return f"{v:.4f}"
    if key.startswith(("SupCtlExact", "SupCtlConfLarge")):
        return f"{v:.5f}"
    if key == "SeqLevelGapMedian":
        m, e = f"{v:.1e}".split("e")
        return rf"{m}\times 10^{{{int(e)}}}"
    if key in ("MCSkewMax", "MCSEUnderstate"):
        return f"{v:.2f}"
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
    if key.startswith(("MLQuant", "MLRatioMostNegative", "MLGbmRatio",
                       "MLQrfRatio", "MLLowerEdgeMargin")):
        return f"{v:.3f}"
    if key.startswith(("MLWorked", "MLPiGrid", "MLUnderThreshold",
                       "MLGbmTailSpan", "MLQrfTailSpan", "MLKnobThreshold")):
        return f"{v:.1f}"
    if key == "MLPooledGrid":
        return f"{v:.3f}"
    if key in ("MCBRhoCal", "MCBRhoTest"):
        return f"{v:.2f}"
    if key == "MCBDensitySpan":
        return f"{v:.0f}"
    if key in ("MCBGapSlope", "MCBGapSlopeSE"):
        return f"{v:.2f}"
    if key in ("MCBGapSlopeSigma", "MCBGapMagRatio"):
        return f"{v:.1f}" if key == "MCBGapSlopeSigma" else f"{v:.2f}"
    if key in ("MCBRhoPlain", "MCBRhoConverted"):
        return f"{v:.2f}"
    if key.endswith("Level"):
        return f"{v:.4f}"
    if key.startswith("WSweep") and key.endswith("Overshoot"):
        return f"{v:.4f}"
    if key.startswith("WSweep") and (key.endswith("Shift") or key.endswith("Sd")):
        return f"{v:.4f}"
    if key.startswith("WSweep") and key.endswith("Pi"):
        return f"{v:.4f}"
    if key.startswith(("WSweepSdRatio", "WSweepExp", "WSweepSqrtTwo")):
        return f"{v:.2f}"
    if key.startswith("GapAbl") and key.endswith("PP"):
        return f"{v:.2f}"
    if key.startswith("MainCorPi"):
        return f"{v:.4f}"
    if key.startswith("GapPanelDPi"):
        return f"{v:.6f}"
    if key == "GapPanelGapPct":
        return f"{v:.2f}"
    if key.startswith("UndBand"):
        return f"{v:.1f}"
    if key in ("BandGoodWorstRatio", "BandVeryStrict"):
        return f"{v:.2f}" if key == "BandVeryStrict" else f"{v:.3f}"
    if key.startswith("BandLoose") or key.startswith("BandStrict"):
        return f"{v:.2f}"
    if key == "MCNormalApproxPP":
        return f"{v:.1f}"
    if key.startswith("MCPop"):
        return f"{v:.4f}"
    if key.startswith("MCPi") or key.startswith("MCRaw"):
        return f"{v:.4f}"
    if key.startswith("MCGrn"):
        return f"{v:.1f}"
    if key.startswith(("MCQv", "MCSd")) and not key.startswith("MCSdRatio"):
        return f"{v:.5f}"
    if key.startswith("MCOver"):
        return f"{v:.4f}"
    if key in ("MCCorPiMaxDev", "MCCorPiSmallLo", "MCCorPiSmallHi"):
        return f"{v:.4f}"
    if key in ("MCSdRatioNormal", "MCSdRatioNaive", "MCSqrtTwenty",
               "TLTauYellowOverAlpha"):
        return f"{v:.2f}"
    if key == "LevelMaxOvershoot":
        return f"{v:.2f}"
    if key == "RuleExactPct":
        return f"{v:.2f}"
    if key == "RuleFailPct":
        return f"{v:.2f}"
    if key == "RuleMissProb":
        return f"{v:.0f}"
    if key.startswith("MCBDensity"):
        return f"{v:.1f}"
    if key == "MLGbmPiDefault":
        # 800 observations, so the grid is 1.25e-3 and four decimals claim a
        # resolution the panel does not have. Guard 6 caught this on its first
        # run, on a number written in this session.
        return f"{v:.2f}"
    if key == "MLGbmCentreSpread":
        return f"{v:.4f}"
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


def check_double_rounding(n: dict) -> list[str]:
    """Ratios a reader could reconstruct from two printed figures.

    Three defects in this project were a quantity computed from already-rounded
    displays instead of from the values behind them, and the error inherits the
    direction of the first rounding rather than averaging away. Each entry names
    the ratio and its two operands; the check reports where dividing the printed
    operands disagrees with the emitted ratio past its last printed place, which
    is the gap a reader dividing the table falls into.
    """
    pairs = [("MCSdRatioNormal", "MCSdNormalOne", "MCSdNormalFive"),
             ("MLQrfTailSpan", None, None),
             ("SupZTwoRatioMini", "SupZTwoMini", "SupZTwoModMeanMini")]
    out = []
    for key, a, b in pairs:
        if a is None or key not in n or a not in n or b not in n:
            continue
        va, vb = float(fmt(a, n[a])), float(fmt(b, n[b]))
        if vb == 0:
            continue
        naive = abs(va / vb)
        emitted = float(fmt(key, n[key]))
        step = 10 ** -len(fmt(key, n[key]).split(".")[-1])
        if abs(naive - emitted) > step / 2:
            out.append(f"{key}: emitted {emitted}, but dividing the printed "
                       f"{a}={va} by {b}={vb} gives {naive:.4f} "
                       f"-- a reader reconstructing it lands {abs(naive-emitted):.4f} away")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    n = collect()

    for msg in check_double_rounding(n):
        print(f"  double rounding: {msg}")

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

    # The resolution registry. Every rate carries the number of observations it
    # summarises, so guard 6 can check that nothing is printed to a precision
    # finer than the grid 1/N the rate actually lives on.
    res = ["# macro\tvalue\tn_obs\tgrid\tprinted_dp",
           "# Written by scripts/paper_numbers.py. A rate computed from N",
           "# observations moves in steps of 1/N and cannot carry more",
           "# resolution than that. Guard 6 fails the build when a printed",
           "# figure claims more. Three defects in this project were this one",
           "# shape before it became a check.", ""]
    for k in sorted(RATE_N):
        v = fmt(k, n[k])
        dp = len(v.split(".")[1]) if "." in v else 0
        res.append(f"{k}\t{v}\t{RATE_N[k]}\t{1.0/RATE_N[k]:.3e}\t{dp}")
    (BASE / "analysis" / "provenance" / "RATE_RESOLUTION.tsv").write_text(
        "\n".join(res) + "\n", encoding="utf-8")

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
