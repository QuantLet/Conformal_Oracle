#!/usr/bin/env python3
"""Backtests on the constructed pair, and the power of Z_2 against an ES-matched
alternative.

Supplement S.8 printed seven figures that no script in this repository produced:
six p-values from a T = 500,000 simulation and one rejection probability at
T = 1,500. `construct_pair.py` writes the pair and stops; `sim.npz` holds 20,000
draws and no statistic. The objects existed, so the figures are recomputed here
rather than removed.

Pre-registration, written before this ran: analysis/phase2/PREREG_PAIR_BACKTESTS.md

WHAT THE SIX P-VALUES ARE EVIDENCE OF, stated where it cannot be lost. Both laws
put exactly alpha mass strictly below their own reported threshold, so the
exceedance indicator is Bernoulli(alpha) and serially independent under BOTH, and
every exceedance-path test has the *identical* null law under P and under Q. That
is a property of the construction, proved from the constraints, not a simulation
result. What the simulation checks is that the sampled law reproduces its own
design constraints -- that the linear programme's equalities survive being drawn
from. Six unremarkable p-values are that check passing. They are not evidence
that the tests are weak; the equality of the two u-processes is.

    python analysis/phase2/pair_backtests.py
    python analysis/phase2/pair_backtests.py --controls   # negative controls only

Rule 2: every check below runs its negative control first, and the control must
fail before the check may report anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import optimize, stats

OUT = Path(__file__).resolve().parent
ALPHA = 0.01
DELTA = 0.05
NU = 5

# Declared once, never re-drawn to move a p-value. See the pre-registration.
SEED_PATHS = 20260828
SEED_POWER = 20260829
T_PATH = 500_000
T_POWER = 1_500
N_POWER = 4_000          # replications for the power estimate
N_NULL = 20_000          # replications calibrating the Z_2 critical value
SIZE = 0.05

RED, GRN, YEL = "\033[31m", "\033[32m", "\033[33m"
OFF = "\033[0m"


def _ok(m):  print(f"  {GRN}pass{OFF}   {m}")
def _bad(m): print(f"  {RED}FAIL{OFF}   {m}")
def _ctl(m): print(f"  {YEL}ctrl{OFF}   {m}")


# ------------------------------------------------------------------ tests ----
def kupiec_p(hits: np.ndarray, alpha: float) -> float:
    """Unconditional coverage LR, Kupiec (1995). Chi-square with one degree."""
    n, x = len(hits), int(hits.sum())
    if x == 0:
        lr = -2 * n * np.log(1 - alpha)
    else:
        pi = x / n
        lr = -2 * (x * np.log(alpha) + (n - x) * np.log(1 - alpha)
                   - x * np.log(pi) - (n - x) * np.log(1 - pi))
    return float(stats.chi2.sf(lr, 1))


def christoffersen_ind_p(hits: np.ndarray) -> float:
    """Independence LR against a first-order Markov alternative, one degree.

    Returns nan when the transition table is degenerate -- n11 = n10 = 0 -- which
    is the K1b3 case: reporting a pass there would count an unpopulated table as
    evidence. The panel result is that this never happens at T = 500,000.
    """
    a, b = hits[:-1].astype(int), hits[1:].astype(int)
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    if (n10 + n11) == 0 or (n00 + n01) == 0:
        return float("nan")
    p01 = n01 / (n00 + n01); p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p in (0.0, 1.0):
        return float("nan")

    def _ll(k0, k1, q):
        if q <= 0 or q >= 1:
            return 0.0
        return k0 * np.log(1 - q) + k1 * np.log(q)

    lr = -2 * ((_ll(n00 + n10, n01 + n11, p))
               - (_ll(n00, n01, p01) + _ll(n10, n11, p11)))
    return float(stats.chi2.sf(lr, 1))


def dq_p(hits: np.ndarray, var: np.ndarray, alpha: float, lags: int = 4) -> float:
    """Engle-Manganelli dynamic quantile test.

    Instruments: a constant, `lags` lagged exceedance indicators, and the
    reported quantile -- the specification the supplement names.
    """
    h = hits.astype(float) - alpha
    n = len(h)
    cols = [np.ones(n - lags)]
    for l in range(1, lags + 1):
        cols.append(h[lags - l:n - l])
    cols.append(np.asarray(var, dtype=float)[lags:n])
    X = np.column_stack(cols)
    y = h[lags:]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    stat = float(coef @ (X.T @ X) @ coef / (alpha * (1 - alpha)))
    return float(stats.chi2.sf(stat, X.shape[1]))


def basel_zone(hits: np.ndarray, alpha: float) -> str:
    """Basel traffic light, scaled to a 250-day year at the nominal level.

    The published boundaries are counts over 250 days at alpha = 0.01: Green up
    to 4, Yellow 5 to 9, Red from 10. They are converted to rates once, here, so
    a path of any length is classified on the same thresholds.
    """
    rate = hits.mean()
    return "Green" if rate <= 4.5 / 250 else ("Yellow" if rate < 9.5 / 250 else "Red")


# ------------------------------------------------------------- the two laws --
def _honest():
    """Standardised Student-t_NU: unit variance, and its own 1% quantile."""
    scale = np.sqrt(NU / (NU - 2))
    return (lambda rng, n: stats.t.rvs(NU, size=n, random_state=rng) / scale,
            float(stats.t.ppf(ALPHA, NU) / scale))


def _truncated():
    """The linear programme's discrete law, and the truncated threshold."""
    z = np.load(OUT / "pair.npz")
    x, p = z["x"], z["p"]
    p = np.clip(p, 0, None); p = p / p.sum()
    return (lambda rng, n: rng.choice(x, size=n, p=p)), float(z["q_trunc"])


def _es_matched(grid_ceiling: float = 64.0, n_grid: int = 6001):
    """The pair with mean expected shortfall as a FIFTH equality constraint.

    Same programme as construct_pair.py plus

        E[X 1{X < q}] / alpha = ES_G(alpha),

    which is linear in the atoms. Returns (sampler, threshold, report). R6 is the
    reason the grid range is reported beside a verdict rather than after it: an
    infeasibility on a support too narrow to carry the mass is not an
    infeasibility of the programme.
    """
    scale = np.sqrt(NU / (NU - 2))
    G_ppf = lambda pr: stats.t.ppf(pr, NU) / scale
    q_true = G_ppf(ALPHA)
    q_trunc = G_ppf(DELTA + ALPHA * (1 - 2 * DELTA))
    mad_G = stats.t.expect(lambda v: abs(v), args=(NU,)) / scale
    # ES of the honest law at alpha, as a mean of the lower tail.
    es_G = stats.t.expect(lambda v: v / scale, args=(NU,),
                          ub=q_true * scale, conditional=True)

    grid = np.linspace(-grid_ceiling, grid_ceiling, n_grid)
    grid = np.unique(np.concatenate([grid, [q_trunc, -q_trunc]]))
    half = grid[grid > 0]
    n = len(half)

    # Symmetric law: mass p_i at +half_i and at -half_i. The lower tail below
    # q_trunc is the mirror of the upper tail above -q_trunc.
    tail = (half > -q_trunc).astype(float)
    # E[X 1{X < q}] = alpha * ES_G. The lower tail of the symmetric law is
    # {-half_i : half_i > |q|}, so E[X 1{X<q}] = -sum(half_i p_i tail_i) and the
    # constraint is sum(half_i p_i tail_i) = alpha * |ES_G|, a POSITIVE target.
    # Writing it with the sign of ES instead asks a sum of non-negative terms to
    # come out negative, which is infeasible for every grid and reports as a
    # property of the support -- the R6 failure exactly.
    A_eq = [np.ones(n), half ** 2, np.abs(half), tail, half * tail]
    b_eq = [0.5, 0.5, 0.5 * mad_G, ALPHA, ALPHA * abs(es_G)]
    A_ub = np.zeros((n - 1, n))
    for i in range(n - 1):
        A_ub[i, i + 1] = 1.0
        A_ub[i, i] = -1.0
    res = optimize.linprog(c=np.zeros(n), A_eq=np.array(A_eq), b_eq=np.array(b_eq),
                           A_ub=A_ub, b_ub=np.zeros(n - 1),
                           bounds=[(0, None)] * n, method="highs")
    rep = {"feasible": bool(res.success), "message": res.message,
           "grid_ceiling": grid_ceiling, "n_grid": int(n_grid),
           "es_honest": float(es_G), "q_trunc": float(q_trunc)}
    if not res.success:
        return None, float(q_trunc), rep
    ph = np.clip(res.x, 0, None)
    p = np.concatenate([ph[::-1], ph]); p = p / p.sum()
    x = np.concatenate([-half[::-1], half])
    rep["es_alternative"] = float((p * x)[x < q_trunc].sum() / ALPHA)
    rep["u_alternative"] = float(p[x < q_trunc].sum())
    return (lambda rng, m: rng.choice(x, size=m, p=p)), float(q_trunc), rep


def z2(r: np.ndarray, var: float, es: float, alpha: float) -> float:
    """Acerbi-Szekely test 2, ES entered as a positive magnitude.

    The sign convention is the one the project has already been wrong about once
    (Supplement S.2): the stored columns are lower-tail quantities and negative,
    and feeding them in unchanged flips every term.
    """
    ind = (r < var).astype(float)
    return float(np.sum(ind * r) / (len(r) * alpha * abs(es)) + 1.0)


# --------------------------------------------------------------- controls ----
def control_kupiec() -> bool:
    """A series exceeding at 3% must be rejected at the 1% level."""
    rng = np.random.default_rng(1)
    return kupiec_p(rng.random(50_000) < 0.03, ALPHA) < 1e-6


def control_independence() -> bool:
    """A clustered exceedance path must be rejected by the independence test."""
    rng = np.random.default_rng(2)
    h = np.zeros(50_000, dtype=bool)
    i = 0
    while i < len(h) - 10:                    # exceedances arrive in blocks of 5
        if rng.random() < 0.002:
            h[i:i + 5] = True; i += 5
        i += 1
    return christoffersen_ind_p(h) < 1e-6


def control_dq() -> bool:
    """The DQ test must reject a path whose exceedances follow the threshold."""
    rng = np.random.default_rng(3)
    n = 50_000
    v = -2.0 - np.sin(np.arange(n) / 500.0)   # a slowly moving reported quantile
    h = rng.random(n) < np.where(v > -2.0, 0.05, 0.002)
    return dq_p(h, v, ALPHA) < 1e-6


def control_degenerate_cc() -> bool:
    """An unpopulated transition table must return nan, not a pass.

    K1b3: 108 of the panel's cells have n11 = 0 with n10 > 0, and the code that
    called those a CC pass is the reason this control exists.
    """
    h = np.zeros(1000, dtype=bool)
    return np.isnan(christoffersen_ind_p(h))


def control_basel() -> bool:
    """The zone must move with the rate, in both directions."""
    n = 250
    green = np.zeros(n, dtype=bool); green[:4] = True
    red = np.zeros(n, dtype=bool); red[:12] = True
    return basel_zone(green, ALPHA) == "Green" and basel_zone(red, ALPHA) == "Red"


def control_z2() -> bool:
    """Z_2 must go negative when realised losses exceed the forecast ES."""
    rng = np.random.default_rng(4)
    r = rng.standard_normal(200_000)
    var, es = -2.326, -2.665                  # Normal 1% VaR and ES
    honest = z2(r, var, es, ALPHA)
    heavy = z2(r * 3.0, var, es, ALPHA)       # losses three times too large
    return abs(honest) < 0.1 and heavy < -1.0


CONTROLS = [("Kupiec rejects a 3% series", control_kupiec),
            ("independence rejects a clustered path", control_independence),
            ("DQ rejects a threshold-following path", control_dq),
            ("a degenerate transition table returns nan", control_degenerate_cc),
            ("the Basel zone moves with the rate", control_basel),
            ("Z_2 goes negative on losses beyond ES", control_z2)]


def run_controls() -> bool:
    good = True
    for name, fn in CONTROLS:
        if fn():
            _ctl(f"negative control fires: {name}")
        else:
            _bad(f"control does NOT reproduce the failure: {name}"); good = False
    return good


# ------------------------------------------------------------------- main ----
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", action="store_true")
    a = ap.parse_args()

    print("negative controls")
    if not run_controls():
        print("\nBROKEN -- a check that cannot fail is not evidence.")
        return 2
    if a.controls:
        return 0

    out: dict = {"seed_paths": SEED_PATHS, "seed_power": SEED_POWER,
                 "T_path": T_PATH, "T_power": T_POWER,
                 "n_power": N_POWER, "n_null": N_NULL, "size": SIZE,
                 "alpha": ALPHA}

    # ---- Exercise A: the two paths ------------------------------------------
    print("\nexercise A -- the two paths at T = %s" % f"{T_PATH:,}")
    rng = np.random.default_rng(SEED_PATHS)
    rows = {}
    for label, (draw, thr) in (("honest", _honest()), ("truncated", _truncated())):
        r = draw(rng, T_PATH)
        hits = r < thr
        rows[label] = {
            "threshold": float(abs(thr)),
            "pi_hat": float(hits.mean()),
            "kupiec_p": kupiec_p(hits, ALPHA),
            "cc_ind_p": christoffersen_ind_p(hits),
            "dq_p": dq_p(hits, np.full(T_PATH, thr), ALPHA),
            "basel": basel_zone(hits, ALPHA),
        }
        d = rows[label]
        _ok(f"{label:10s} |q| = {d['threshold']:.4f}  pi-hat = {d['pi_hat']:.5f}  "
            f"Kupiec {d['kupiec_p']:.3f}  CC {d['cc_ind_p']:.3f}  "
            f"DQ {d['dq_p']:.3f}  {d['basel']}")
    out["paths"] = rows

    # The falsification condition from the pre-registration, checked before the
    # numbers are used for anything.
    blocked = [k for k, v in rows.items()
               if v["basel"] != "Green"
               or min(v["kupiec_p"], v["cc_ind_p"], v["dq_p"]) < SIZE]
    out["exercise_a_blocked"] = blocked
    if blocked:
        _bad(f"pre-registered falsification met on {blocked}: the sampled law "
             "does not reproduce its own design constraints")
    else:
        _ok("both paths pass all three tests and both are Green, as the "
            "equality of the two u-processes requires")

    # ---- Exercise B: power of Z_2 against the ES-matched alternative ---------
    print("\nexercise B -- Z_2 against an alternative matched on mean ES")
    draw_alt, thr_alt, rep = _es_matched()
    out["es_matched"] = rep
    if not rep["feasible"]:
        # R6: widen the support once before accepting an infeasibility.
        _bad(f"infeasible on |x| <= {rep['grid_ceiling']}; widening once")
        draw_alt, thr_alt, rep = _es_matched(grid_ceiling=256.0, n_grid=12001)
        out["es_matched"] = rep
    if not rep["feasible"]:
        _bad("the fifth constraint is infeasible on a widened support; the "
             "alternative the sentence describes does not exist")
        out["power"] = None
    else:
        _ok(f"fifth constraint feasible on |x| <= {rep['grid_ceiling']}; "
            f"ES matched to {abs(rep['es_alternative'] - rep['es_honest']):.2e}")
        draw_h, thr_h = _honest()
        es_h = float(stats.t.expect(lambda v: v / np.sqrt(NU / (NU - 2)),
                                    args=(NU,), ub=thr_h * np.sqrt(NU / (NU - 2)),
                                    conditional=True))
        rng = np.random.default_rng(SEED_POWER)
        null = np.array([z2(draw_h(rng, T_POWER), thr_h, es_h, ALPHA)
                         for _ in range(N_NULL)])
        crit = float(np.quantile(null, SIZE))          # one-sided, lower tail
        alt = np.array([z2(draw_alt(rng, T_POWER), thr_alt, es_h, ALPHA)
                        for _ in range(N_POWER)])
        rej = float((alt < crit).mean())
        se = float(np.sqrt(rej * (1 - rej) / N_POWER))
        lo, hi = rej - 1.96 * se, rej + 1.96 * se
        covers = lo <= SIZE <= hi
        out["power"] = {"critical_value": crit, "rejection": rej,
                        "se": se, "ci_lo": lo, "ci_hi": hi,
                        "covers_size": bool(covers)}
        _ok(f"critical value {crit:.4f}; rejection {rej:.4f} "
            f"[{lo:.4f}, {hi:.4f}] at size {SIZE}")
        if covers:
            _bad("the interval covers the nominal size: this is NOT evidence of "
                 f"power at {N_POWER} replications, and must not be printed as though it were")
        else:
            _ok("the interval excludes the nominal size: Z_2 retains power "
                "against an alternative matched on mean ES")

    (OUT / "pair_backtests.json").write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {OUT / 'pair_backtests.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
