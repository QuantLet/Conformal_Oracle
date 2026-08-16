#!/usr/bin/env python3
"""Phase 3a — AE point 7: does a longer estimation window remove the need for
the correction?

The AE's hypothesis: "Estimating GARCH models by only 250 observations might be
quite noisy. The results might stabilize a lot (such that re-calibration has a
lower (or even negative) effect) when longer estimation windows for the model
parameters are employed."

This tests it rather than assuming it away. All four classical benchmarks are
re-estimated on rolling windows of w in {250, 500, 1000} and evaluated under the
same protocol: raw and corrected coverage, q_V, quantile score, Basel zones.
Convergence failures are counted per window length and the fallback recorded.

Reported per (model, window):
    pihat_raw, pihat_cp, qV, QS_raw, QS_cp, Basel zone counts, convergence
    failures and how they were handled.

Output: analysis/phase3_windows/
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / "cfp_ijf_data"
OUT = Path(__file__).resolve().parent

ALPHA = 0.01
F_CAL = 0.70
WINDOWS = [250, 500, 1000]
EWMA_LAMBDA = 0.94

sys.path.insert(0, str(BASE / "analysis" / "ae_point4"))
from run_ae_point4 import (  # noqa: E402
    SYMBOLS, kupiec_p, qhat_ceil, quantile_score, traffic_light,
)


def fit_garch_var(window: np.ndarray, alpha: float, kind: str):
    """One-step-ahead VaR from a GARCH-family fit. Returns (var, converged)."""
    from arch import arch_model
    try:
        if kind == "gjr_garch":
            am = arch_model(window * 100, vol="GARCH", p=1, o=1, q=1,
                            mean="Zero", dist="normal", rescale=False)
        else:
            am = arch_model(window * 100, vol="GARCH", p=1, q=1,
                            mean="Zero", dist="normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        f = res.forecast(horizon=1, reindex=False)
        sigma = float(np.sqrt(f.variance.values[-1, 0])) / 100
        if not np.isfinite(sigma) or sigma <= 0:
            return None, False
        return sigma * stats.norm.ppf(alpha), True
    except Exception:
        return None, False


def ewma_var(window: np.ndarray, alpha: float) -> float:
    n = len(window)
    w = np.array([(1 - EWMA_LAMBDA) * EWMA_LAMBDA ** (n - 1 - i) for i in range(n)])
    w /= w.sum()
    sigma = float(np.sqrt(np.sum(w * window ** 2)))
    return sigma * stats.norm.ppf(alpha)


def hist_sim_var(window: np.ndarray, alpha: float) -> float:
    return float(np.quantile(window, alpha))


def var_path(r: np.ndarray, w: int, model: str, alpha: float):
    """Rolling one-step VaR over the whole series; NaN before the first window."""
    n = len(r)
    var = np.full(n, np.nan)
    fails = 0
    prev = None
    for t in range(w, n):
        win = r[t - w:t]
        if model in ("gjr_garch", "garch_n"):
            v, ok = fit_garch_var(win, alpha, model)
            if not ok:
                fails += 1
                v = ewma_var(win, alpha)          # documented fallback
                if prev is not None and not np.isfinite(v):
                    v = prev
            prev = v
        elif model == "ewma":
            v = ewma_var(win, alpha)
        else:
            v = hist_sim_var(win, alpha)
        var[t] = v
    return var, fails


def evaluate(r: np.ndarray, var: np.ndarray, alpha: float) -> dict:
    m = np.isfinite(var)
    r_v, q_v = r[m], var[m]
    n = len(r_v)
    n_cal = int(n * F_CAL)
    if n_cal < 100 or n - n_cal < 50:
        return {}
    r_cal, r_test = r_v[:n_cal], r_v[n_cal:]
    q_cal, q_test = q_v[:n_cal], q_v[n_cal:]
    qV = qhat_ceil(q_cal - r_cal, alpha)
    cp = q_test - qV
    v_raw = int(np.sum(r_test < q_test))
    v_cp = int(np.sum(r_test < cp))
    nt = len(r_test)
    return {
        "n_test": nt, "qV": qV,
        "pihat_raw": v_raw / nt, "pihat_cp": v_cp / nt,
        "p_kup_raw": kupiec_p(v_raw, nt, alpha),
        "p_kup_cp": kupiec_p(v_cp, nt, alpha),
        "QS_raw": quantile_score(r_test, q_test, alpha),
        "QS_cp": quantile_score(r_test, cp, alpha),
        "TL_raw": traffic_light(v_raw, nt), "TL_cp": traffic_light(v_cp, nt),
        "dQS": quantile_score(r_test, q_test, alpha) - quantile_score(r_test, cp, alpha),
    }


MODELS = {"GJR-GARCH": "gjr_garch", "GARCH-N": "garch_n",
          "EWMA": "ewma", "Hist-Sim": "hs"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sym in SYMBOLS:
        ret = pd.read_csv(DATA / "returns" / f"{sym}.csv", index_col=0,
                          parse_dates=True)
        r = ret.iloc[:, 0].values.astype(float)
        for label, key in MODELS.items():
            for w in WINDOWS:
                if len(r) - w < 200:
                    continue
                var, fails = var_path(r, w, key, ALPHA)
                res = evaluate(r, var, ALPHA)
                if not res:
                    continue
                res.update({"model": label, "asset": sym, "w": w,
                            "convergence_failures": fails,
                            "n_fits": max(len(r) - w, 0)})
                rows.append(res)
        print(f"  {sym}", file=sys.stderr, flush=True)
        pd.DataFrame(rows).to_csv(OUT / "window_sensitivity.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "window_sensitivity.csv", index=False)
    print(f"\n{len(df)} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
