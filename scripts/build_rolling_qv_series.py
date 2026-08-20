#!/usr/bin/env python3
"""Rebuild the rolling conformal threshold series for the monitoring figures.

`cfp_ijf_data/paper_outputs/tables/rolling_qv_SP500.csv` is read by the COVID
response figure and by CO_rolling_qV, and -- like the QS sequences -- had no
producer under version control. It dated from 30 April and covered the ten
forecasters of that vintage.

Definition, matching the rolling estimator of
`scripts/regenerate_rolling_vs_static.py`: at each date t,

    qhat_V(t) = S_(k) of the trailing w = 250 nonconformity scores,
                k = ceil((w + 1)(1 - alpha)),  scores s_i = VaR_i - r_i

computed over the whole history rather than only the test window, because the
figures it feeds are monitoring plots rather than backtests. `rvol` is the
annualised 250-day realised volatility of the asset.

Usage:
    python scripts/build_rolling_qv_series.py [--asset SP500]
"""

from __future__ import annotations

import argparse
import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = DATA / "paper_outputs" / "tables"

sys.path.insert(0, str(BASE / "Quantlets"))
from cfp_config import MODELS  # noqa: E402

ALPHA = 0.01
W_ROLL = 250


def load_pair(model: str, asset: str) -> pd.DataFrame:
    subdir, suffix = MODELS[model]
    fname = f"{asset}_{suffix}.parquet" if suffix else f"{asset}.parquet"
    ret = pd.read_csv(DATA / "returns" / f"{asset}.csv", index_col=0,
                      parse_dates=True)
    fc = pd.read_parquet(DATA / subdir / fname)
    common = ret.index.intersection(fc.index).sort_values()
    return pd.DataFrame({"r": ret.loc[common, "log_return"].values,
                         "v": fc.loc[common, "VaR_0.01"].values}, index=common)


def rolling_qhat(scores: np.ndarray, w: int = W_ROLL, alpha: float = ALPHA):
    """Trailing-window conformal quantile at every date; NaN before the window fills."""
    out = np.full(len(scores), np.nan)
    k = min(int(ceil((w + 1) * (1 - alpha))) - 1, w - 1)
    for t in range(w, len(scores)):
        out[t] = np.sort(scores[t - w:t])[k]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="SP500")
    a = ap.parse_args()

    cols = {}
    for model in MODELS:
        try:
            pair = load_pair(model, a.asset)
        except Exception as e:                                   # noqa: BLE001
            print(f"  skip {model}: {e}", file=sys.stderr)
            continue
        scores = (pair["v"] - pair["r"]).values
        cols[model] = pd.Series(rolling_qhat(scores), index=pair.index)
        print(f"  {model:16s} {int(np.isfinite(cols[model]).sum())} dated values",
              file=sys.stderr)

    ret = pd.read_csv(DATA / "returns" / f"{a.asset}.csv", index_col=0,
                      parse_dates=True)["log_return"]
    cols["rvol"] = ret.rolling(W_ROLL).std() * np.sqrt(252)

    df = pd.DataFrame(cols).sort_index()
    path = OUT / f"rolling_qv_{a.asset}.csv"
    df.to_csv(path)
    print(f"wrote {path} ({df.shape[0]} dates, {df.shape[1]} columns)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
