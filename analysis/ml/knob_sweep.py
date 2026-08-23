#!/usr/bin/env python3
"""Reading (ii) of drafts/prereg_ml.md: the knobs an analyst would actually vary.

The dose-response arm varied leaf size. This arm varies learning rate, tree count
and depth -- the parameters that are documented, discussed and swept in practice
-- holding the leaf size at the library default. The pre-registered reading is
that they do not move the 1% tail.

Only LightGBM: a quantile random forest has no learning rate, and its tree count
and depth were covered by the leaf arm.

Unit: config x asset x date cell. Expected rows 7 x 4 x 200 = 5,600.
Varies between rows: one knob at a time away from the library default, and the
asset. Features, seed, window, dates and leaf size identical throughout.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import dose_response as dr  # noqa: E402  (same directory, shares loader/features)

OUT = Path(__file__).resolve().parent
DEFAULT_LEAF = 20

# one knob moved at a time; the default cell is already in the leaf arm
CONFIGS = [
    ("learning_rate", 0.03), ("learning_rate", 0.30),
    ("n_estimators", 50), ("n_estimators", 400),
    ("max_depth", 3), ("max_depth", 6),
    ("num_leaves", 127),
]


def run() -> pd.DataFrame:
    rows = []
    for asset in dr.ASSETS:
        X, y = dr.load(asset)
        idx = np.linspace(dr.WINDOW, len(X) - 1, dr.N_DATES).astype(int)
        for knob, val in CONFIGS:
            for t in idx:
                Xtr, ytr = X.iloc[t - dr.WINDOW:t], y.iloc[t - dr.WINDOW:t]
                Xte, yte = X.iloc[t], y.iloc[t]
                q, med = dr.fit_lgbm(Xtr, ytr, Xte, DEFAULT_LEAF, **{knob: val})
                rows.append(dict(asset=asset, knob=knob, value=val,
                                 date=X.index[t], realised=float(yte),
                                 train_sd=float(ytr.std()), lgbm_q=q, lgbm_med=med))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    df.to_csv(OUT / "knob_sweep_raw.csv", index=False)
    print(f"cells written: {len(df)}  (pre-registered 5,600)")
