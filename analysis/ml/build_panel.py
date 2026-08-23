#!/usr/bin/env python3
"""E2: the four ML series on 24 assets.

Produces, for each series and asset, a date-indexed table of lower quantiles at
alpha in {0.01, 0.025, 0.05, 0.10} and NOTHING ELSE. No violation indicator, no
coverage, no Kupiec, no Basel zone. Amendment 3 fixes the order: the structural
gate runs blind on these files before any backtest, and computing coverage here
would destroy the only out-of-sample test that gate will get.

Configurations, fixed in Amendment 3:
    lgbm_default   LightGBM        min_data_in_leaf = 20   (library default)
    lgbm_pooled    LightGBM        min_data_in_leaf = 500
    qrf_default    Quantile RF     min_samples_leaf = 1    (library default)
    qrf_pooled     Quantile RF     min_samples_leaf = 500

Unit: date within one series-asset. Varies between rows: the date only.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data" / "returns"
OUT = Path(__file__).resolve().parent / "series"
ALPHAS = [0.01, 0.025, 0.05, 0.10]
WINDOW = 1000
REFIT_EVERY = 25
SEED = 20260822

CONFIGS = {
    "lgbm_default": ("lgbm", 20),
    "lgbm_pooled": ("lgbm", 500),
    "qrf_default": ("qrf", 1),
    "qrf_pooled": ("qrf", 500),
}


def features(r: pd.Series) -> pd.DataFrame:
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


def _fit_lgbm(Xtr, ytr, leaf):
    import lightgbm as lgb
    models = {}
    for a in ALPHAS:
        models[a] = lgb.LGBMRegressor(
            objective="quantile", alpha=a, min_data_in_leaf=leaf,
            n_estimators=100, learning_rate=0.1, num_leaves=31,
            verbose=-1, seed=SEED, deterministic=True, force_row_wise=True
        ).fit(Xtr, ytr)
    return models


def _predict_lgbm(models, Xte):
    return {a: models[a].predict(Xte) for a in ALPHAS}


def _fit_qrf(Xtr, ytr, leaf):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=200, min_samples_leaf=leaf,
                              random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    return m, m.apply(Xtr), ytr.values


def _predict_qrf(fitted, Xte):
    m, leaves_tr, ytr = fitted
    leaves_te = m.apply(Xte)
    out = {a: np.empty(len(Xte)) for a in ALPHAS}
    for i in range(len(Xte)):
        pool = np.concatenate([ytr[leaves_tr[:, j] == leaves_te[i, j]]
                               for j in range(m.n_estimators)])
        for a in ALPHAS:
            out[a][i] = np.quantile(pool, a)
    return out


def build(name: str, kind: str, leaf: int) -> None:
    (OUT / name).mkdir(parents=True, exist_ok=True)
    assets = sorted(p.stem for p in DATA.glob("*.csv"))
    for asset in assets:
        dest = OUT / name / f"{asset}.parquet"
        if dest.exists():
            continue
        X, y = load(asset)
        if len(X) <= WINDOW + REFIT_EVERY:
            continue
        rows, fitted, last_fit = [], None, -10**9
        t = WINDOW
        while t < len(X):
            if t - last_fit >= REFIT_EVERY:
                Xtr, ytr = X.iloc[t - WINDOW:t], y.iloc[t - WINDOW:t]
                fitted = (_fit_lgbm(Xtr, ytr, leaf) if kind == "lgbm"
                          else _fit_qrf(Xtr, ytr, leaf))
                last_fit = t
            hi = min(t + REFIT_EVERY, len(X))
            Xte = X.iloc[t:hi]
            q = (_predict_lgbm(fitted, Xte) if kind == "lgbm"
                 else _predict_qrf(fitted, Xte))
            for k, idx in enumerate(X.index[t:hi]):
                rows.append({"date": idx, **{f"VaR_{a}": float(q[a][k]) for a in ALPHAS}})
            t = hi
        pd.DataFrame(rows).set_index("date").to_parquet(dest)
        print(f"  {name:14s} {asset:8s} {len(rows):5d} dates", flush=True)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    t0 = time.time()
    for name, (kind, leaf) in CONFIGS.items():
        if only and name != only:
            continue
        build(name, kind, leaf)
    print(f"done in {time.time() - t0:.0f}s")
