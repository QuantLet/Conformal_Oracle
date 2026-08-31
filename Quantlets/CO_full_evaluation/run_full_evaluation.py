#!/usr/bin/env python3
"""Regenerate `all_results.csv`, the input every downstream table reads.

Until now this file existed only as the output of `CO_full_evaluation.ipynb`.
Table 1, the master table, the panel tests and the qV ranking all read it, so a
notebook-only producer meant the single most-consumed artefact in the repository
had no committed script. That is the same class of gap that let GJR-GARCH ship
for months with no producer at all.

Every definition below is transcribed from that notebook, not reinvented:

    calibration split   cfp_config.split_indices, floor, chronological
    nonconformity       s_i = v_i - r_i  (one-sided, lower tail)
    conformal quantile  the ORDER STATISTIC s_(k), k = ceil((n_cal+1)(1-alpha)),
                        never np.quantile -- interpolation carries no
                        finite-sample guarantee
    corrected VaR       var_cp = v_test - qV
    Kupiec              POF LR, chi2_1, with the v=0 and v=n branches
    Christoffersen      INDEPENDENCE LR, chi2_1, NaN when the transition table
                        is degenerate (not the joint chi2_2 of Appendix G)
    Traffic Light       violations scaled to 250 days: <=4 Green, <=9 Yellow
    Quantile score      mean (alpha - 1{r<v})(r - v)
    widths              mean |VaR|, corrected and raw

`--verify` is the reason to trust the rest. It regenerates every model and
compares against the committed `all_results.csv`. Models whose input series have
not changed must reproduce bit-for-bit; if they do not, this transcription is
wrong and nothing it produces should be used. Only GJR-GARCH is expected to
differ, because its series was corrected on 2026-08-17 (raw t_5 quantile with df
hard-coded, replaced by the Gaussian innovation the manuscript describes).

Usage:
    python run_full_evaluation.py --verify        # reproduce, compare, write nothing
    python run_full_evaluation.py --write         # regenerate the artefact
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent
DATA = BASE / "cfp_ijf_data"
TARGETS = [DATA / "paper_outputs" / "tables" / "all_results.csv",
           HERE / "results" / "all_results.csv"]

sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import (MODELS, SYMBOLS, ALPHAS, F_CAL,  # noqa: E402
                        split_indices)

ASSETS = sorted(SYMBOLS)


def load_pair(model, asset, alpha):
    subdir, suffix = MODELS[model]
    fname = f"{asset}_{suffix}.parquet" if suffix else f"{asset}.parquet"
    ret = pd.read_csv(DATA / "returns" / f"{asset}.csv", index_col=0,
                      parse_dates=True)
    fc = pd.read_parquet(DATA / subdir / fname)
    col = f"VaR_{alpha}"
    if col not in fc.columns:
        raise KeyError(f"{col} not in {subdir}/{fname}")
    common = ret.index.intersection(fc.index).sort_values()
    if len(common) < 50:
        raise ValueError(f"only {len(common)} overlapping dates for {model}/{asset}")
    return ret.loc[common, "log_return"].values, fc.loc[common, col].values


def kupiec_pval(n, v, a):
    if v == 0:
        lr = -2.0 * n * np.log(1.0 - a)
    elif v == n:
        lr = -2.0 * n * np.log(a)
    else:
        pihat = v / n
        lr = -2.0 * (v * np.log(a / pihat)
                     + (n - v) * np.log((1.0 - a) / (1.0 - pihat)))
    return 1.0 - stats.chi2.cdf(abs(lr), 1)


def cc_pval(violations_bool):
    """Christoffersen INDEPENDENCE LR. NaN when the test is not defined.

    A degenerate transition table is an absence of evidence, not a pass. Table 1
    counted it as a pass and Table 2 did not, which is where the two tables
    disagreed.
    """
    v = violations_bool.astype(int)
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
    return 1.0 - stats.chi2.cdf(abs(lr), 1)


def basel_tl(n_viol, n_days):
    scaled = n_viol * 250.0 / n_days
    return "Green" if scaled <= 4 else ("Yellow" if scaled <= 9 else "Red")


def quantile_score(r, v, a):
    return float(np.mean((a - (r < v).astype(float)) * (r - v)))


def conformal_backtest(returns, var_raw, alpha, f_cal=F_CAL, gap=None):
    """One split-conformal backtest on one (forecaster, asset, level) cell.

    The split comes from cfp_config.split_indices, which owns both the
    calibration fraction and the separation of Assumption (A3). gap=None takes
    the house setting; the measurement scripts pass it explicitly to compute
    both arms. The gap is taken from the FRONT of the test block, so the
    calibration sample and the shift are unchanged by it and only the evaluated
    window moves.
    """
    r, v = np.asarray(returns, dtype=float), np.asarray(var_raw, dtype=float)
    n = len(r)
    cal, test, g = split_indices(n, v - r, f_cal=f_cal, gap=gap)
    n_cal, n_test = len(cal), len(test)
    r_cal, v_cal = r[cal], v[cal]
    r_test, v_test = r[test], v[test]

    scores = np.sort(v_cal - r_cal)
    k = min(int(np.ceil((n_cal + 1) * (1 - alpha))) - 1, n_cal - 1)
    qV = float(scores[k])
    var_cp = v_test - qV

    viol_raw = int(np.sum(r_test < v_test))
    viol_cp = int(np.sum(r_test < var_cp))
    return {
        "n_cal": n_cal, "n_test": n_test, "gap": g, "qV": qV,
        "pihat_raw": viol_raw / n_test, "pihat_cp": viol_cp / n_test,
        "viol_raw": viol_raw, "viol_cp": viol_cp,
        "p_kup_raw": kupiec_pval(n_test, viol_raw, alpha),
        "p_kup_cp": kupiec_pval(n_test, viol_cp, alpha),
        "p_cc_raw": cc_pval(r_test < v_test),
        "p_cc_cp": cc_pval(r_test < var_cp),
        "TL_raw": basel_tl(viol_raw, n_test), "TL_cp": basel_tl(viol_cp, n_test),
        "QS_raw": quantile_score(r_test, v_test, alpha),
        "QS_cp": quantile_score(r_test, var_cp, alpha),
        "VaR_width": float(np.mean(np.abs(var_cp))),
        "raw_width": float(np.mean(np.abs(v_test))),
    }


def compute(models, gap: bool | None = None) -> pd.DataFrame:
    rows, errors = [], []
    for model in models:
        for asset in ASSETS:
            for alpha in ALPHAS:
                try:
                    r, v = load_pair(model, asset, alpha)
                    res = conformal_backtest(r, v, alpha, gap=gap)
                    res.update({"model": model, "symbol": asset, "alpha": alpha})
                    rows.append(res)
                except Exception as e:                       # noqa: BLE001
                    errors.append(f"{model}/{asset}/{alpha}: {e}")
    if errors:
        print(f"{len(errors)} cell(s) failed:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)
    return pd.DataFrame(rows)


NUM = ["qV", "pihat_raw", "pihat_cp", "p_kup_raw", "p_kup_cp", "p_cc_raw",
       "p_cc_cp", "QS_raw", "QS_cp", "VaR_width", "raw_width"]
KEY = ["model", "symbol", "alpha"]


def verify(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    m = old.merge(new, on=KEY, suffixes=("_old", "_new"), how="outer",
                  indicator=True)
    out = []
    for model, s in m.groupby("model"):
        both = s[s["_merge"] == "both"]
        # Moirai-1.1 was added after this artefact was written and lives in
        # moirai11_full_results.csv, so it has no committed row here to compare
        # against. That is NOT_IN_BASELINE, which is a different thing from
        # agreeing, and must not be reported as a pass.
        if both.empty:
            out.append({"model": model, "cells": len(s),
                        "only_committed": int((s["_merge"] == "left_only").sum()),
                        "only_rebuilt": int((s["_merge"] == "right_only").sum()),
                        "max_abs_diff": np.nan, "verdict": "NOT_IN_BASELINE"})
            continue
        d = {c: float(np.nanmax(np.abs(both[f"{c}_old"] - both[f"{c}_new"])))
             for c in NUM if f"{c}_old" in both}
        worst = max(d.values()) if d else np.nan
        out.append({"model": model, "cells": len(s),
                    "only_committed": int((s["_merge"] == "left_only").sum()),
                    "only_rebuilt": int((s["_merge"] == "right_only").sum()),
                    "max_abs_diff": worst,
                    "verdict": ("REPRODUCES" if worst == 0 else
                                ("ROUNDOFF" if worst < 1e-12 else "DIFFERS"))})
    return pd.DataFrame(out).sort_values("model")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--gap", dest="gap", action="store_true", default=None,
                    help="impose the separation of Corollary 4.6 for this run. "
                         "The default is cfp_config.SEPARATION, which governs "
                         "every site at once -- the panel must move as a unit "
                         "or it mixes two estimators. "
                         "See analysis/convention/GAP_SWITCH_SCOPE.md.")
    ap.add_argument("--no-gap", dest="gap", action="store_false",
                    help="force the contiguous split for this run.")
    a = ap.parse_args()

    models = a.models or list(MODELS)
    new = compute(models, gap=a.gap)
    print(f"computed {len(new)} cells over {new['model'].nunique()} models",
          file=sys.stderr)

    if a.verify:
        old = pd.read_csv(TARGETS[0])
        v = verify(new, old)
        print(v.to_string(index=False))
        # Models whose SERIES were corrected on 2026-08-17 and are therefore
        # expected to differ from the committed baseline. Every other model must
        # reproduce; if one does not, this transcription is wrong. Each entry
        # here is a promotion that went through scripts/promotion_gate.py, with
        # the superseded series preserved:
        #   GJR-GARCH    raw t_5 quantile, df hard-coded -> Gaussian innovation
        #   Moirai-2.0   -ppf(alpha) sign inversion -> ppf(alpha)
        #   TimesFM-2.5  same sign inversion
        CORRECTED = {"GJR-GARCH", "Moirai-2.0", "TimesFM-2.5"}
        bad = v[(v["verdict"] == "DIFFERS") & (~v["model"].isin(CORRECTED))]
        if len(bad):
            print("\nUnchanged models must reproduce exactly. They do not, so "
                  "this transcription is wrong and its output must not be used.",
                  file=sys.stderr)
            return 1
        print("\nAll models whose series were untouched reproduce exactly; "
              f"{', '.join(sorted(CORRECTED))} differ by construction "
              "(series corrected 2026-08-17).")

    if a.write:
        for t in TARGETS:
            t.parent.mkdir(parents=True, exist_ok=True)
            new.to_csv(t, index=False)
            print(f"wrote {t}", file=sys.stderr)
    elif not a.verify:
        print("nothing to do: pass --verify or --write", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
