#!/usr/bin/env python3
"""Rebuild `analysis/phase3/dq_panel.csv`, which had no producer.

The panel is read by `paper_numbers.py` for three macros (`SeqDQCells`,
`SeqDQRejRaw`, `SeqDQRejCor`) and nothing in the repository wrote it — the same
gap `build_qs_sequences.py` was written to close, and the reason R14's correction
to the analytic series could not otherwise reach it.

The statistic is transcribed from `analysis/phase3/PREREG_DQ.md`, not reinvented:
Engle–Manganelli dynamic quantile regression on the test window, Hit_t − alpha on
a constant, four lagged Hits and the reported lower quantile; Wald statistic,
chi-squared with 6 degrees of freedom. The split, the alignment and the conformal
order statistic are imported from the scripts that produce `all_results.csv`, so
a cell here and the summary row it belongs to cannot describe different samples.

`--verify` is what makes the rest usable: it rebuilds every cell and compares
against the committed file. The 264 cells whose series R14 did not touch must
reproduce, or this transcription is wrong and its 48 new cells mean nothing.

Usage:
    python scripts/build_dq_panel.py --verify
    python scripts/build_dq_panel.py --write
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "analysis" / "phase3" / "dq_panel.csv"

sys.path.insert(0, str(BASE / "Quantlets"))
sys.path.insert(0, str(BASE / "scripts"))
from cfp_config import (MODELS, SYMBOLS, F_CAL,  # noqa: E402
                        conformal_quantile, split_indices)
from build_qs_sequences import load_pair_dated  # noqa: E402

ALPHA, LAGS = 0.01, 4
ASSETS = sorted(SYMBOLS)


def dq_pvalue(r: np.ndarray, v: np.ndarray, alpha: float, lags: int = LAGS) -> float:
    """Engle–Manganelli DQ: Wald on [1, Hit_{t-1..t-4}, VaR_t], chi2 with 6 df."""
    hit = (r < v).astype(float) - alpha
    T = len(hit)
    if T <= lags + 1:
        return float("nan")
    y = hit[lags:]
    X = [np.ones(T - lags)]
    for L in range(1, lags + 1):
        X.append(hit[lags - L:T - L])
    X.append(v[lags:])
    X = np.column_stack(X)
    XtX = X.T @ X
    try:
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return float("nan")
    stat = float(beta @ XtX @ beta / (alpha * (1 - alpha)))
    return float(stats.chi2.sf(max(stat, 0.0), X.shape[1]))


def build() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for asset in ASSETS:
            df = load_pair_dated(model, asset, ALPHA)
            n = len(df)
            r, v = df["r"].to_numpy(), df["v"].to_numpy()
            cal, test, _g = split_indices(n, v - r, f_cal=F_CAL)
            n_cal, t0 = len(cal), int(test[0])
            # The shift comes from cfp_config, not from a local order statistic:
            # the convention lived in a docstring and was reimplemented by every
            # producer, which is what QV_CONVENTION.md was written about.
            qv = conformal_quantile(v[:n_cal] - r[:n_cal], ALPHA)
            r_t, v_t = r[t0:], v[t0:]
            rows.append({"model": model, "symbol": asset,
                         "p_dq_raw": dq_pvalue(r_t, v_t, ALPHA),
                         "p_dq_cp": dq_pvalue(r_t, v_t - qv, ALPHA)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    new = build()
    print(f"{len(new)} cells over {new['model'].nunique()} models")
    assert len(new) == 312, f"expected 312 cells, built {len(new)}"

    if OUT.exists():
        old = pd.read_csv(OUT)
        j = old.merge(new, on=["model", "symbol"], suffixes=("_old", "_new"))
        assert len(j) == len(old) == len(new), "cell sets differ"
        touched = j["model"].isin(["Chronos-Small-A", "Chronos-Mini-A"])
        for tag, m in (("untouched by R14", ~touched), ("the analytic series", touched)):
            sub = j[m]
            d_raw = np.abs(sub["p_dq_raw_old"] - sub["p_dq_raw_new"])
            d_cp = np.abs(sub["p_dq_cp_old"] - sub["p_dq_cp_new"])
            flip_raw = int(((sub["p_dq_raw_old"] < 0.05) != (sub["p_dq_raw_new"] < 0.05)).sum())
            flip_cp = int(((sub["p_dq_cp_old"] < 0.05) != (sub["p_dq_cp_new"] < 0.05)).sum())
            print(f"  {tag:20s} {len(sub):3d} cells | max |dp| raw {d_raw.max():.2e} "
                  f"cp {d_cp.max():.2e} | rejection flips raw {flip_raw} cp {flip_cp}")
        rep = j[~touched]
        ok = (np.abs(rep["p_dq_raw_old"] - rep["p_dq_raw_new"]).max() < 1e-9
              and np.abs(rep["p_dq_cp_old"] - rep["p_dq_cp_new"]).max() < 1e-9)
        print(f"  transcription {'REPRODUCES' if ok else 'DOES NOT REPRODUCE'} "
              f"the cells R14 did not touch")
        if not ok and a.write:
            raise SystemExit("refusing to write: the untouched cells do not reproduce")

    print(f"  rejection at 5%: raw {100 * (new['p_dq_raw'] < 0.05).mean():.1f}%  "
          f"corrected {100 * (new['p_dq_cp'] < 0.05).mean():.1f}%")
    if a.write:
        new.to_csv(OUT, index=False)
        print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
