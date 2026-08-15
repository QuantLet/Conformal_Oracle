#!/usr/bin/env python3
"""Duration-based independence tests, defined where the Markov test is not.

The Christoffersen independence LR is undefined on 126 of 240 pairs at
alpha = 0.01 because n11 = 0 -- no two violations land on consecutive days. That
rules out *consecutive* violations; it does not rule out clustering. Fifteen
exceedances packed into two months but never adjacent give n11 = 0 and are
visibly clustered. A first-order Markov transition test cannot see that by
construction: it looks only at immediate transitions, never at the spacing
between violations.

Two standard tests do use the spacing, are defined when n11 = 0, and need no
defending to a finance referee:

  * Christoffersen-Pelletier (2004) duration test. Under independence the
    durations between violations are geometric, hence memoryless. Fitted against
    a Weibull alternative, H0: b = 1. LR ~ chi2_1. First and last durations are
    censored, following the paper.
  * Engle-Manganelli (2004) dynamic quantile test. Regresses the demeaned hit
    indicator on its own lags and on the VaR level; DQ ~ chi2_(p+2) under
    correct conditional coverage.

Three outcomes, all reportable:
  - both reject on a large share of the 126  -> clustering is present and the
    standard Markov test is blind to it by construction, not for lack of power;
  - neither rejects -> those pairs genuinely have well-spaced violations and the
    aggregate 4.47x n11 excess overstates the picture;
  - mixed -> a partition of the pairs by failure type, which is more informative
    than a binary column either way.

Outputs (analysis/duration_tests/):
    duration_tests.csv   per pair: CP and DQ statistics and p-values
    SUMMARY.md
"""

from __future__ import annotations

import sys
from math import ceil, log
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

BASE = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent
ALPHA = 0.01
DQ_LAGS = 4

sys.path.insert(0, str(BASE / "analysis" / "ae_point4"))
from run_ae_point4 import (  # noqa: E402
    F_CAL, MODELS, SYMBOLS, W_ROLL, load_pair, qhat_ceil,
)

PANEL_A = {"Moirai-1.1", "Lag-Llama", "GJR-GARCH", "GARCH-N", "Hist-Sim", "EWMA"}


# --------------------------------------------------------------------------- #
# Christoffersen-Pelletier (2004)
# --------------------------------------------------------------------------- #

def durations(hits: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Durations between violations, plus the two censored end spells."""
    idx = np.flatnonzero(hits)
    if len(idx) == 0:
        return np.array([]), float(len(hits)), 0.0
    first = float(idx[0] + 1)              # censored: time to the first hit
    last = float(len(hits) - idx[-1])      # censored: time after the last hit
    return np.diff(idx).astype(float), first, last


def _weib_ll(params: np.ndarray, d: np.ndarray, c0: float, c1: float) -> float:
    a, b = params
    if a <= 0 or b <= 0:
        return 1e10
    # Weibull with scale 1/a: S(t) = exp(-(a t)^b), f(t) = a b (a t)^(b-1) S(t)
    ll = 0.0
    if len(d):
        ll += np.sum(np.log(a) + np.log(b) + (b - 1) * np.log(a * d) - (a * d) ** b)
    for c in (c0, c1):                     # censored contributions: log S(c)
        if c > 0:
            ll -= (a * c) ** b
    return -ll


def cp_duration_test(hits: np.ndarray) -> tuple[float, float, int]:
    """LR of H0: b = 1 (memoryless) against a Weibull alternative."""
    d, c0, c1 = durations(hits)
    n_viol = int(hits.sum())
    if n_viol < 2 or len(d) < 2:
        return np.nan, np.nan, n_viol
    a0 = 1.0 / max(np.mean(np.concatenate([d, [c0, c1]])), 1e-9)
    ll_exp = -_weib_ll(np.array([a0, 1.0]), d, c0, c1)
    # Profile out `a` under the alternative.
    best = None
    for b_start in (0.5, 0.8, 1.0, 1.5, 2.0):
        try:
            r = optimize.minimize(_weib_ll, np.array([a0, b_start]),
                                  args=(d, c0, c1), method="Nelder-Mead",
                                  options={"maxiter": 2000, "xatol": 1e-8,
                                           "fatol": 1e-10})
            if r.success and (best is None or r.fun < best.fun):
                best = r
        except Exception:
            continue
    if best is None:
        return np.nan, np.nan, n_viol
    lr = 2.0 * (-best.fun - ll_exp)
    lr = max(lr, 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, 1)), n_viol


# --------------------------------------------------------------------------- #
# Engle-Manganelli (2004) DQ
# --------------------------------------------------------------------------- #

def dq_test(hits: np.ndarray, var: np.ndarray, lags: int = DQ_LAGS):
    """Out-of-sample DQ: demeaned hits on their lags and the VaR level."""
    n = len(hits)
    if n <= lags + 3 or hits.sum() < 1:
        return np.nan, np.nan
    h = hits.astype(float) - ALPHA
    rows, y = [], []
    for t in range(lags, n):
        rows.append([1.0] + [h[t - k] for k in range(1, lags + 1)] + [var[t]])
        y.append(h[t])
    X = np.asarray(rows)
    y = np.asarray(y)
    try:
        XtX = X.T @ X
        beta = np.linalg.solve(XtX, X.T @ y)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    stat = float(beta @ XtX @ beta / (ALPHA * (1 - ALPHA)))
    dof = X.shape[1]
    return stat, float(1 - stats.chi2.cdf(stat, dof))


# --------------------------------------------------------------------------- #

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        for sym in SYMBOLS:
            try:
                r, q = load_pair(model, sym, ALPHA)
            except Exception:
                continue
            n = len(r)
            n_cal = int(n * F_CAL)
            if n_cal < W_ROLL or n - n_cal < 50:
                continue
            r_cal, r_test = r[:n_cal], r[n_cal:]
            q_cal, q_test = q[:n_cal], q[n_cal:]
            qV = qhat_ceil(q_cal - r_cal, ALPHA)
            var = q_test - qV
            hits = (r_test < var).astype(int)

            v = hits.astype(bool)
            n11 = int(np.sum(v[:-1] & v[1:]))
            cp_lr, cp_p, n_viol = cp_duration_test(hits)
            dq_stat, dq_p = dq_test(hits, var)
            d, _, _ = durations(hits)
            rows.append({
                "model": model, "asset": sym,
                "panel": "A" if model in PANEL_A else "B",
                "n_test": len(r_test), "n_viol": n_viol, "n11": n11,
                "markov_defined": n11 >= 1,
                "cp_lr": cp_lr, "cp_p": cp_p,
                "dq_stat": dq_stat, "dq_p": dq_p,
                "mean_duration": float(np.mean(d)) if len(d) else np.nan,
                "cv_duration": (float(np.std(d) / np.mean(d))
                                if len(d) and np.mean(d) > 0 else np.nan),
            })
        print(f"  {model}", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "duration_tests.csv", index=False)

    deg = df[~df["markov_defined"]]
    dfn = df[df["markov_defined"]]

    def rate(s, col):
        s = s.dropna(subset=[col])
        return (len(s), int((s[col] <= 0.05).sum()),
                100 * (s[col] <= 0.05).mean() if len(s) else np.nan)

    L = ["# Duration-based independence tests", "",
         "Generated by `analysis/duration_tests/run_duration_tests.py`. "
         f"alpha = {ALPHA}, static conformal correction, {len(df)} pairs.", "",
         "`n11 = 0` rules out *consecutive* violations, not clustering. These "
         "two tests use the spacing between violations and are defined anyway.",
         "", "## Rejection rates at 5%", "",
         "| Subset | pairs | CP duration rejects | DQ rejects |",
         "|---|---|---|---|"]
    for name, s in (("all pairs", df),
                    ("Markov test DEFINED (n11 >= 1)", dfn),
                    ("**Markov test UNDEFINED (n11 = 0)**", deg)):
        n_cp, r_cp, p_cp = rate(s, "cp_p")
        n_dq, r_dq, p_dq = rate(s, "dq_p")
        L.append(f"| {name} | {len(s)} | {r_cp}/{n_cp} ({p_cp:.1f}%) | "
                 f"{r_dq}/{n_dq} ({p_dq:.1f}%) |")
    L.append("")
    both = deg.dropna(subset=["cp_p", "dq_p"])
    if len(both):
        b = int(((both["cp_p"] <= 0.05) & (both["dq_p"] <= 0.05)).sum())
        e = int(((both["cp_p"] <= 0.05) | (both["dq_p"] <= 0.05)).sum())
        L += [f"Among the {len(deg)} pairs the Markov test cannot assess: "
              f"**{e} are flagged by at least one duration-based test**, "
              f"{b} by both.", ""]
    L += ["## Duration dispersion", "",
          "Under independence durations are geometric, so the coefficient of "
          "variation is near 1. Values above 1 indicate over-dispersed spacing "
          "— bursts separated by long quiet spells.", "",
          "| Subset | mean duration | mean CV | share with CV > 1 |",
          "|---|---|---|---|"]
    for name, s in (("Markov defined", dfn), ("Markov undefined", deg)):
        cv = s["cv_duration"].dropna()
        L.append(f"| {name} | {s['mean_duration'].mean():.1f} | {cv.mean():.2f} | "
                 f"{100 * (cv > 1).mean():.1f}% |")
    L.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print((OUT / "SUMMARY.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
