#!/usr/bin/env python3
"""Phase 3b — Referee 1 point ix: dynamic VaR benchmarks.

Exactly two, not ten. CAViaR (Engle & Manganelli 2004) in its symmetric absolute
value and asymmetric slope specifications, and a score-driven (GAS) VaR model.
These are the two a finance referee names; adding LSTM/XGBoost/TCN would concede
the framing that this paper proposes a forecaster, which it does not.

Protocol matches the rest of the paper: the same 24 assets, the same 70/30
calibration/test split, alpha = 0.01, and the same conformal correction applied
on top so the comparison is like for like.

Specifications
--------------
CAViaR-SAV   q_t = b1 + b2 q_{t-1} + b3 |y_{t-1}|
CAViaR-AS    q_t = b1 + b2 q_{t-1} + b3 max(y_{t-1},0) + b4 max(-y_{t-1},0)
GAS-t        a score-driven location-scale filter with Student-t innovations,
             VaR_t = mu + sigma_t * t_nu^{-1}(alpha); sigma_t updated by the
             scaled score of the observation density (Creal, Koopman & Lucas
             2013), which is the standard score-driven VaR construction.

Parameters are estimated once on the calibration segment by minimising the
quantile (tick) loss for CAViaR and the t log-likelihood for GAS, then the
recursion is filtered through the test segment without re-estimation. This is
the out-of-sample protocol used for the classical benchmarks.

Output: analysis/phase3_dynamic/
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = Path(__file__).resolve().parent
ALPHA = 0.01
F_CAL = 0.70

sys.path.insert(0, str(BASE / "analysis" / "ae_point4"))
from run_ae_point4 import (  # noqa: E402
    SYMBOLS, kupiec_p, qhat_ceil, quantile_score, traffic_light,
)


def tick(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = y - q
    return float(np.mean(np.where(d < 0, (alpha - 1) * d, alpha * d)))


# --------------------------------------------------------------------------- #
# CAViaR
# --------------------------------------------------------------------------- #

def caviar_path(theta: np.ndarray, y: np.ndarray, spec: str, q0: float) -> np.ndarray:
    n = len(y)
    q = np.empty(n)
    q[0] = q0
    for t in range(1, n):
        if spec == "SAV":
            q[t] = theta[0] + theta[1] * q[t - 1] + theta[2] * abs(y[t - 1])
        else:
            q[t] = (theta[0] + theta[1] * q[t - 1]
                    + theta[2] * max(y[t - 1], 0.0)
                    + theta[3] * max(-y[t - 1], 0.0))
    return q


def fit_caviar(y: np.ndarray, alpha: float, spec: str):
    q0 = float(np.quantile(y[:min(300, len(y))], alpha))
    k = 3 if spec == "SAV" else 4

    def obj(th):
        if abs(th[1]) >= 0.999:
            return 1e6
        return tick(y, caviar_path(th, y, spec, q0), alpha)

    best, best_v = None, np.inf
    rng = np.random.default_rng(42)
    starts = [np.concatenate([[q0 * 0.1, 0.9], np.full(k - 2, -0.1)])]
    for _ in range(8):
        starts.append(np.concatenate([
            [q0 * rng.uniform(0.02, 0.3), rng.uniform(0.5, 0.98)],
            rng.uniform(-0.4, 0.0, k - 2)]))
    for s in starts:
        try:
            r = optimize.minimize(obj, s, method="Nelder-Mead",
                                  options={"maxiter": 4000, "fatol": 1e-10})
            if r.fun < best_v:
                best_v, best = r.fun, r.x
        except Exception:
            continue
    return best, q0, (best is not None)


# --------------------------------------------------------------------------- #
# GAS-t
# --------------------------------------------------------------------------- #

def gas_sigma(theta: np.ndarray, y: np.ndarray) -> np.ndarray:
    omega, a, b, nu = theta
    n = len(y)
    lsig = np.empty(n)
    lsig[0] = np.log(max(np.std(y[:min(250, n)]), 1e-8))
    for t in range(1, n):
        e = y[t - 1] / np.exp(lsig[t - 1])
        # scaled score of the log-scale for a Student-t density
        w = (nu + 1) / (nu + e ** 2)
        s = w * e ** 2 - 1.0
        lsig[t] = omega + b * lsig[t - 1] + a * s
    return np.exp(lsig)


def fit_gas(y: np.ndarray):
    def nll(th):
        omega, a, b, nu = th
        if not (0 < a < 1 and 0 < b < 1.0 and 2.05 < nu < 60):
            return 1e8
        sig = gas_sigma(th, y)
        if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
            return 1e8
        z = y / sig
        ll = (stats.t.logpdf(z, df=nu) - np.log(sig)).sum()
        return -ll if np.isfinite(ll) else 1e8

    best, best_v = None, np.inf
    for s in ([0.01, 0.05, 0.95, 6.0], [0.02, 0.10, 0.90, 4.0],
              [0.005, 0.03, 0.97, 10.0]):
        try:
            r = optimize.minimize(nll, np.array(s), method="Nelder-Mead",
                                  options={"maxiter": 6000, "fatol": 1e-8})
            if r.fun < best_v:
                best_v, best = r.fun, r.x
        except Exception:
            continue
    return best, (best is not None)


# --------------------------------------------------------------------------- #

def evaluate(r_test, q_test, r_cal, q_cal, alpha) -> dict:
    qV = qhat_ceil(q_cal - r_cal, alpha)
    cp = q_test - qV
    v_raw = int(np.sum(r_test < q_test))
    v_cp = int(np.sum(r_test < cp))
    n = len(r_test)
    return {
        "n_test": n, "qV": qV,
        "pihat_raw": v_raw / n, "pihat_cp": v_cp / n,
        "p_kup_raw": kupiec_p(v_raw, n, alpha),
        "p_kup_cp": kupiec_p(v_cp, n, alpha),
        "QS_raw": quantile_score(r_test, q_test, alpha),
        "QS_cp": quantile_score(r_test, cp, alpha),
        "TL_raw": traffic_light(v_raw, n), "TL_cp": traffic_light(v_cp, n),
        "R": abs(qV) / abs(np.mean(q_test)) if np.mean(q_test) else np.nan,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in SYMBOLS:
        ret = pd.read_csv(DATA / "returns" / f"{sym}.csv", index_col=0,
                          parse_dates=True)
        y = ret.iloc[:, 0].values.astype(float)
        n = len(y)
        n_cal = int(n * F_CAL)
        y_cal, y_test = y[:n_cal], y[n_cal:]

        for spec in ("SAV", "AS"):
            th, q0, ok = fit_caviar(y_cal, ALPHA, spec)
            if not ok:
                print(f"  FAIL CAViaR-{spec} {sym}", file=sys.stderr)
                continue
            q_full = caviar_path(th, y, spec, q0)
            res = evaluate(y_test, q_full[n_cal:], y_cal, q_full[:n_cal], ALPHA)
            res.update({"model": f"CAViaR-{spec}", "asset": sym, "converged": True})
            rows.append(res)

        th, ok = fit_gas(y_cal)
        if ok:
            sig = gas_sigma(th, y)
            q_full = sig * stats.t.ppf(ALPHA, df=th[3]) / np.sqrt(th[3] / (th[3] - 2))
            res = evaluate(y_test, q_full[n_cal:], y_cal, q_full[:n_cal], ALPHA)
            res.update({"model": "GAS-t", "asset": sym, "converged": True,
                        "nu": th[3]})
            rows.append(res)
        else:
            print(f"  FAIL GAS {sym}", file=sys.stderr)
        print(f"  {sym}", file=sys.stderr, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "dynamic_var.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dynamic_var.csv", index=False)
    print(f"\n{len(df)} rows, {df.model.nunique()} models", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
