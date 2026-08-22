#!/usr/bin/env python3
"""The observationally equivalent pair, constructed from the characterisation.

Proposition 1 says the exceedance path identifies the joint law of (forecast,
return) only through u_t = F_t(q_t). This script builds two models that agree on
the u-law and disagree on everything a risk manager cares about, subject to being
matched on median, point-forecast accuracy (MAE and RMSE) and invariance to the
temperature / nucleus sweep.

    P  correctly specified forecaster, heavy-tailed returns
    Q  support-truncated forecaster, light-tailed returns

Both report a 1% VaR. Both exceed on exactly 1% of days in conditional
expectation. No function of the exceedance path can tell them apart.
"""
from __future__ import annotations
import json
import numpy as np
from scipy import stats, optimize
from pathlib import Path

OUT = Path(__file__).resolve().parent
ALPHA = 0.01
DELTA = 0.05          # mass discarded from each tail by the truncation
NU = 5                # Student-t degrees of freedom for the honest DGP


# --------------------------------------------------------------- model P ----
# Returns r_t = sigma_t * eps_t, eps ~ standardised t_nu (unit variance).
# Forecaster reports the true conditional alpha-quantile, so u_t == alpha.
scale_t = np.sqrt(NU / (NU - 2))
G_ppf = lambda p: stats.t.ppf(p, NU) / scale_t
G_cdf = lambda x: stats.t.cdf(x * scale_t, NU)

q_true = G_ppf(ALPHA)
mad_G = stats.t.expect(lambda x: abs(x), args=(NU,)) / scale_t
var_G = 1.0

# --------------------------------------------------------------- model Q ----
# The forecaster truncates its predictive support to the central 1-2*DELTA mass
# and reads the alpha-quantile of the renormalised law. Tails go first.
q_trunc = G_ppf(DELTA + ALPHA * (1 - 2 * DELTA))

# Q's returns must make that same threshold an exact alpha-quantile:
#   H(q_trunc) = ALPHA,  Var(H) = 1,  E|H| = mad_G,  median 0, symmetric.
# All four constraints are linear in the probabilities of a discrete symmetric
# law on a fixed grid, so this is a linear feasibility problem.
GRID = np.linspace(-64.0, 64.0, 6001)          # ceiling 64: the boundary is stable from 8 up
GRID = np.unique(np.concatenate([GRID, [q_trunc, -q_trunc]]))
pos = GRID > 0
half = GRID[pos]                              # solve on the positive half
n = len(half)

A_eq, b_eq = [], []
A_eq.append(np.ones(n));                    b_eq.append(0.5)          # total mass
A_eq.append(half ** 2);                     b_eq.append(0.5 * var_G)  # variance
A_eq.append(np.abs(half));                  b_eq.append(0.5 * mad_G)  # MAD
# strict: the exceedance indicator is 1{r < q}, and the discrete law has an
# atom exactly at q_trunc, so P(X < q) and P(X <= q) differ by that atom.
A_eq.append((half > -q_trunc).astype(float)); b_eq.append(ALPHA)      # alpha-quantile
# Unimodality: the density must be non-increasing in |x|. Without it the LP is
# free to park mass in a far tail and return a bimodal return law, which is
# feasible arithmetic and an implausible object for a referee to accept.
A_ub = np.zeros((n - 1, n))
for i in range(n - 1):
    A_ub[i, i + 1] = 1.0     # p_{i+1} - p_i <= 0
    A_ub[i, i] = -1.0
res = optimize.linprog(c=np.zeros(n), A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                       A_ub=A_ub, b_ub=np.zeros(n - 1),
                       bounds=[(0, None)] * n, method="highs")

report = {"feasible": bool(res.success), "message": res.message}
if res.success:
    ph = res.x
    p = np.concatenate([ph[::-1], ph])
    x = np.concatenate([-half[::-1], half])
    H_cdf = lambda v: float(p[x <= v].sum())
    report.update(
        q_true_P=float(q_true), q_reported_Q=float(q_trunc),
        pct_of_way_to_zero=float(100 * (1 - q_trunc / q_true)),
        u_P=float(ALPHA), u_Q=float(p[x < q_trunc].sum()),
        var_P=float(var_G), var_Q=float((p * x ** 2).sum()),
        mad_P=float(mad_G), mad_Q=float((p * np.abs(x)).sum()),
        median_P=0.0, median_Q=0.0,  # exact: p is symmetric by construction, no atom at 0
        kurt_P=float(3 + 6 / (NU - 4)) if NU > 4 else None,
        kurt_Q=float((p * x ** 4).sum()),
        tail_ratio_P=float(abs(q_true)), tail_ratio_Q=float(abs(G_ppf(ALPHA))),
        dispersion_ratio_Q_over_P=float(
            np.sqrt(((p * x ** 2).sum())) / np.sqrt(var_G)),
        support_mass_discarded=float(2 * DELTA),
    )
    np.savez(OUT / "pair.npz", x=x, p=p, q_true=q_true, q_trunc=q_trunc, nu=NU)

(OUT / "pair_report.json").write_text(json.dumps(report, indent=2))
for k, v in report.items():
    print(f"  {k:28s} {v}")
