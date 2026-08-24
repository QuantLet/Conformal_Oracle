"""The traffic-light proposition, checked against the grid.

Claim: P(green) = P(pi_hat_T <= tau) -> 1{pi < tau}, and at finite T the green
share tracks the normal approximation Phi( sqrt(n) (tau - pi) / sd(pi_hat) ).
"""
import numpy as np, pandas as pd, json
from scipy import stats
TAU = 4/250
d = pd.read_csv("analysis/k2_sim/grid_per_rep.csv")
POP = {"normal": 0.0100, "t5": 0.0150, "t3": 0.0136, "skewt3": 0.0233, "mixnormal": 0.0125}

print("(i) green share equals the share of replications with pi_hat <= tau, cell by cell")
bad = 0
for (k, T), g in d.groupby(["dgp", "T"]):
    a, b = 100*g.raw_green.mean(), 100*(g.raw_pi <= TAU + 1e-12).mean()
    bad += abs(a - b) > 1e-9
print(f"    {len(d.groupby(['dgp','T']))} cells, disagreements: {bad}")

print("\n(ii) the finite-T green share against the normal approximation")
print(f"    {'dgp':10s} {'T':>6s} {'pop pi':>8s} {'sd(pi_hat)':>11s} {'observed':>9s} {'normal approx':>14s}")
rows = []
for (k, T), g in d.groupby(["dgp", "T"]):
    sd = g.raw_pi.std(ddof=1)
    obs = 100*g.raw_green.mean()
    approx = 100*stats.norm.cdf((TAU - g.raw_pi.mean())/sd)
    rows.append({"dgp": k, "T": int(T), "pop_pi": float(g.raw_pi.mean()), "sd": float(sd),
                 "observed": float(obs), "normal_approx": float(approx)})
    print(f"    {k:10s} {T:6d} {g.raw_pi.mean():8.4f} {sd:11.5f} {obs:8.1f}% {approx:13.1f}%")
r = pd.DataFrame(rows)
print(f"\n    mean |observed - approx| = {np.abs(r.observed - r.normal_approx).mean():.1f} points")

print("\n(iii) the limit, by the sign of pi - tau")
for k, p in POP.items():
    lim = "1 (green)" if p < TAU else "0 (not green)"
    obs = r[(r.dgp == k) & (r["T"] == 10000)].observed.iloc[0]
    print(f"    {k:10s} population pi = {p:.4f}  {'<' if p<TAU else '>'} tau = {TAU:.4f}"
          f"   ->  limit {lim:14s}  observed at T = 10,000: {obs:.1f}%")

print("\n(iv) NEGATIVE CONTROL: a process whose population pi sits ON the threshold")
print("     should give a green share stuck near 50% at every T, not converging.")
rng = np.random.default_rng(3)
for T in [1000, 5000, 10000]:
    n = T - int(T*0.70)
    pis = rng.binomial(n, TAU, 2000)/n
    print(f"     T={T:6d} n_test={n:5d}  green share {100*(pis <= TAU + 1e-12).mean():.1f}%")
r.to_csv("analysis/k2_sim/prop_tl_check.csv", index=False)
json.dump({"cells": len(r), "identity_disagreements": int(bad),
           "mean_abs_gap_vs_normal_approx": float(np.abs(r.observed-r.normal_approx).mean())},
          open("analysis/k2_sim/prop_tl_check.json","w"), indent=2)
