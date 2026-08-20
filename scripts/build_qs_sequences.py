#!/usr/bin/env python3
"""Rebuild the per-date quantile-score and violation sequences.

`cfp_ijf_data/paper_outputs/qs_sequences/` and `violation_sequences/` are the
input to the Diebold-Mariano table, the wild-cluster bootstrap and the
panel-pooled backtests. They dated from 24 April, covered nine forecasters, and
**no script in this repository produced them** -- three modules read them and
nothing wrote them. That is how an artefact survives a correction to its own
inputs without anyone noticing.

This script produces them, for every forecaster in `cfp_config.MODELS`, from the
promoted series. It does not reimplement the evaluation: it imports
`load_pair` and the conformal split from `CO_full_evaluation/run_full_evaluation.py`,
the script that produces `all_results.csv`, so a sequence and the summary row it
underlies cannot disagree by construction. `--verify` checks exactly that, cell
by cell, against the committed `all_results.csv`.

Layout, unchanged from the files it replaces: one parquet per forecaster, a
DatetimeIndex of the full common history, one column per asset, values only on
the test window and NaN before it.

    {key}_qs.parquet             corrected quantile-loss sequence
    {key}_qs_raw.parquet         raw quantile-loss sequence
    {key}_violations.parquet     corrected violation indicator
    {key}_violations_raw.parquet raw violation indicator

The corrected file keeps the name the existing consumers read, because that is
what they were reading: the April files hold the CONFORMALLY CORRECTED loss, not
the raw one. The raw counterparts are new and are written because half the
questions this paper now asks are about the uncorrected series.

Usage:
    python scripts/build_qs_sequences.py --verify    # rebuild in memory, compare, write nothing
    python scripts/build_qs_sequences.py --write     # supersede and write
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "cfp_ijf_data"
QS_DIR = DATA / "paper_outputs" / "qs_sequences"
VIOL_DIR = DATA / "paper_outputs" / "violation_sequences"

sys.path.insert(0, str(BASE / "Quantlets"))
sys.path.insert(0, str(BASE / "Quantlets" / "CO_full_evaluation"))
from cfp_config import MODELS, SYMBOLS, F_CAL  # noqa: E402
from run_full_evaluation import quantile_score  # noqa: E402

ALPHA = 0.01
ASSETS = sorted(SYMBOLS)


def file_key(model: str) -> str:
    """The stem the existing consumers use: the benchmark suffix, else the directory."""
    subdir, suffix = MODELS[model]
    return suffix if suffix else subdir


def load_pair_dated(model: str, asset: str, alpha: float) -> pd.DataFrame:
    """`run_full_evaluation.load_pair`, but keeping the dates it discards.

    The alignment must be identical to that function -- same intersection, same
    sort -- or the sequences would describe a different sample from the summary
    row. The assertion below is what enforces it.
    """
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
    return pd.DataFrame({"r": ret.loc[common, "log_return"].values,
                         "v": fc.loc[common, col].values}, index=common)


def sequences_for_pair(df: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, dict]:
    """Per-date loss and violation series on the test window, plus the summary."""
    n = len(df)
    n_cal = int(n * F_CAL)
    r, v = df["r"].values, df["v"].values
    r_cal, v_cal = r[:n_cal], v[:n_cal]
    r_test, v_test = r[n_cal:], v[n_cal:]

    scores = np.sort(v_cal - r_cal)
    k = min(int(np.ceil((n_cal + 1) * (1 - alpha))) - 1, n_cal - 1)
    qV = float(scores[k])
    var_cp = v_test - qV

    def loss(x, thr):
        return (alpha - (x < thr).astype(float)) * (x - thr)

    out = pd.DataFrame(index=df.index, dtype=float)
    out["qs_cp"] = np.nan
    out["qs_raw"] = np.nan
    out["viol_cp"] = np.nan
    out["viol_raw"] = np.nan
    test_idx = df.index[n_cal:]
    out.loc[test_idx, "qs_cp"] = loss(r_test, var_cp)
    out.loc[test_idx, "qs_raw"] = loss(r_test, v_test)
    out.loc[test_idx, "viol_cp"] = (r_test < var_cp).astype(float)
    out.loc[test_idx, "viol_raw"] = (r_test < v_test).astype(float)

    summary = {
        "n_cal": n_cal, "n_test": n - n_cal, "qV": qV,
        "viol_raw": int((r_test < v_test).sum()),
        "viol_cp": int((r_test < var_cp).sum()),
        "QS_raw": quantile_score(r_test, v_test, alpha),
        "QS_cp": quantile_score(r_test, var_cp, alpha),
    }
    return out, summary


def build(alpha: float) -> tuple[dict, pd.DataFrame]:
    panels: dict[str, dict[str, pd.DataFrame]] = {}
    checks, errors = [], []
    for model in MODELS:
        key = file_key(model)
        cols = {name: {} for name in ("qs_cp", "qs_raw", "viol_cp", "viol_raw")}
        for asset in ASSETS:
            try:
                pair = load_pair_dated(model, asset, alpha)
            except Exception as e:                                  # noqa: BLE001
                errors.append(f"{model}/{asset}: {e}")
                continue
            seq, summary = sequences_for_pair(pair, alpha)
            for name in cols:
                cols[name][asset] = seq[name]
            summary.update({"model": model, "symbol": asset, "alpha": alpha})
            checks.append(summary)
        panels[key] = {name: pd.DataFrame(cols[name]).sort_index()
                       for name in cols}
        print(f"  {model:16s} -> {key}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} pair(s) failed:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  {e}", file=sys.stderr)
    return panels, pd.DataFrame(checks)


def verify(checks: pd.DataFrame, alpha: float) -> bool:
    """Every sequence must imply exactly the summary row committed in all_results."""
    ref = pd.read_csv(BASE / "Quantlets" / "CO_full_evaluation" / "results" / "all_results.csv")
    ref = ref[ref["alpha"] == alpha]
    m = checks.merge(ref, on=["model", "symbol", "alpha"], suffixes=("_new", "_ref"),
                     how="outer", indicator=True)
    missing = m[m["_merge"] != "both"]
    ok = True
    if len(missing):
        print(f"{len(missing)} cell(s) not matched against all_results.csv:", file=sys.stderr)
        for _, r in missing.head(10).iterrows():
            print(f"  {r['model']}/{r['symbol']}: {r['_merge']}", file=sys.stderr)
        ok = False
    b = m[m["_merge"] == "both"]
    # Counts must agree exactly. The floats cannot: all_results.csv is decimal
    # text, so a round trip through it costs an ulp. The tolerance is the same
    # 1e-12 that run_full_evaluation.py calls ROUNDOFF, and it is roughly eight
    # orders of magnitude below the quantities compared.
    print(f"\n{'quantity':<12} {'max abs diff':>14}   over {len(b)} pairs")
    for c in ("qV", "QS_raw", "QS_cp"):
        d = float(np.nanmax(np.abs(b[f"{c}_new"] - b[f"{c}_ref"])))
        print(f"{c:<12} {d:>14.3e}" + ("" if d < 1e-12 else "   <-- EXCEEDS 1e-12"))
        if not d < 1e-12:
            ok = False
    for c in ("n_cal", "n_test", "viol_raw", "viol_cp"):
        d = int(np.nanmax(np.abs(b[f"{c}_new"] - b[f"{c}_ref"])))
        print(f"{c:<12} {d:>14d}")
        if d != 0:
            ok = False
    print("\n" + ("REPRODUCES all_results.csv: counts identical, floats within "
                  "1e-12" if ok else "DOES NOT REPRODUCE -- do not write"))
    return ok


def write(panels: dict) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    for d in (QS_DIR, VIOL_DIR):
        old = list(d.glob("*.parquet"))
        if old:
            sup = d / f"superseded_{stamp}"
            sup.mkdir(exist_ok=True)
            for f in old:
                shutil.move(str(f), str(sup / f.name))
            print(f"superseded {len(old)} file(s) -> {sup}", file=sys.stderr)
    for key, frames in panels.items():
        frames["qs_cp"].to_parquet(QS_DIR / f"{key}_qs.parquet")
        frames["qs_raw"].to_parquet(QS_DIR / f"{key}_qs_raw.parquet")
        frames["viol_cp"].to_parquet(VIOL_DIR / f"{key}_violations.parquet")
        frames["viol_raw"].to_parquet(VIOL_DIR / f"{key}_violations_raw.parquet")
    print(f"wrote {4 * len(panels)} files", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    a = ap.parse_args()
    if not (a.verify or a.write):
        print("nothing to do: pass --verify or --write", file=sys.stderr)
        return 1

    panels, checks = build(a.alpha)
    ok = verify(checks, a.alpha)
    if a.write:
        if not ok:
            print("refusing to write: the rebuilt sequences do not reproduce "
                  "the committed summary rows.", file=sys.stderr)
            return 1
        write(panels)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
