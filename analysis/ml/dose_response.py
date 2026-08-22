#!/usr/bin/env python3
"""E1: leaf-resolution dose-response for LightGBM and a quantile random forest.

Readings fixed in drafts/prereg_ml.md before this ran:
  (i)   dispersion and pi-hat move monotonically in the leaf-size parameter;
  (ii)  learning rate, tree count and depth do not move the tail;
  (iii) the predictive median is materially unchanged across the grid.

Unit: model-config x asset x date cell. Expected 2 x 5 x 4 x 200 = 8,000.
Varies between rows: the leaf-size parameter and the asset. Features, seed,
window and split are identical everywhere.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data" / "returns"
OUT = Path(__file__).resolve().parent
ALPHA = 0.01
ASSETS = ["SP500", "GOLD", "BTC", "EURUSD"]
LEAF_GRID = [1, 5, 20, 100, 500]
N_DATES = 200
WINDOW = 1000
SEED = 20260822


def features(r: pd.Series) -> pd.DataFrame:
    """Nine features, fixed in P4: lags 1-5, RV over 5/22/250, sign of lag 1."""
    X = pd.DataFrame(index=r.index)
    for k in range(1, 6):
        X[f"lag{k}"] = r.shift(k)
    for w in (5, 22, 250):
        X[f"rv{w}"] = r.shift(1).rolling(w).std()
    X["sign1"] = np.sign(r.shift(1))
    return X


def load(asset: str):
    r = pd.read_csv(DATA / f"{asset}.csv", parse_dates=["date"]).set_index("date")["log_return"]
    X = features(r)
    d = pd.concat([X, r.rename("y")], axis=1).dropna()
    return d.drop(columns="y"), d["y"]


def fit_lgbm(Xtr, ytr, Xte, leaf, **kw):
    import lightgbm as lgb
    p = dict(objective="quantile", alpha=ALPHA, min_data_in_leaf=leaf,
             n_estimators=kw.get("n_estimators", 100),
             learning_rate=kw.get("learning_rate", 0.1),
             max_depth=kw.get("max_depth", -1),
             num_leaves=31, verbose=-1, seed=SEED, deterministic=True,
             force_row_wise=True)
    m = lgb.LGBMRegressor(**p).fit(Xtr, ytr)
    q = float(m.predict(Xte.to_frame().T if Xte.ndim == 1 else Xte)[0])
    med = lgb.LGBMRegressor(**{**p, "alpha": 0.5}).fit(Xtr, ytr)
    return q, float(med.predict(Xte.to_frame().T if Xte.ndim == 1 else Xte)[0])


def fit_qrf(Xtr, ytr, Xte, leaf, **kw):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=kw.get("n_estimators", 200),
                              min_samples_leaf=leaf, random_state=SEED,
                              max_depth=kw.get("max_depth", None), n_jobs=-1)
    m.fit(Xtr, ytr)
    x = Xte.values.reshape(1, -1)
    # empirical conditional distribution of in-leaf training targets
    leaves_tr = m.apply(Xtr)
    leaves_te = m.apply(x)[0]
    vals = []
    for j in range(m.n_estimators):
        vals.append(ytr.values[leaves_tr[:, j] == leaves_te[j]])
    pool = np.concatenate(vals)
    return float(np.quantile(pool, ALPHA)), float(np.median(pool)), float(pool.std())


def run() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for asset in ASSETS:
        X, y = load(asset)
        idx = np.linspace(WINDOW, len(X) - 1, N_DATES).astype(int)
        for leaf in LEAF_GRID:
            for t in idx:
                Xtr, ytr = X.iloc[t - WINDOW:t], y.iloc[t - WINDOW:t]
                Xte, yte = X.iloc[t], y.iloc[t]
                ql, ml = fit_lgbm(Xtr, ytr, Xte, leaf)
                qf, mf, sf = fit_qrf(Xtr, ytr, Xte, leaf)
                sd = float(ytr.std())
                rows.append(dict(asset=asset, leaf=leaf, date=X.index[t],
                                 realised=float(yte), train_sd=sd,
                                 lgbm_q=ql, lgbm_med=ml,
                                 qrf_q=qf, qrf_med=mf, qrf_sd=sf))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run()
    df.to_csv(OUT / "dose_response_raw.csv", index=False)
    print(f"cells written: {len(df)}  (pre-registered 8,000)")
