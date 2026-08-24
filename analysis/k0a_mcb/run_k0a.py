"""K0a. q̂_V against the score-optimal shift, and against the uMCB term.

The pre-registered question: is q̂_V a rewriting of the unconditional
miscalibration component? The decomposition has two separable pieces and they are
not the same object --

    delta*   = argmin_c  mean L_alpha(q_t - c, y_t)      -- a shift, in return units
    uMCB     = S_raw - min_c mean L_alpha(q_t - c, y_t)  -- a score reduction

q̂_V is an estimate of the first, computed out of sample. This script measures how
close it is to the in-sample argmin, and how the second is related to it.
"""
import numpy as np, pandas as pd, json, math, pathlib
from scipy import stats

ALPHA, FC = 0.01, 0.70
SERIES = {"Chronos-Small": ("dir", "chronos_small"), "Chronos-Mini": ("dir", "chronos_mini"),
          "Chronos-Small-A": ("dir", "chronos_small_analytic"), "Chronos-Mini-A": ("dir", "chronos_mini_analytic"),
          "TimesFM-2.5": ("dir", "timesfm25"), "Moirai-2.0": ("dir", "moirai2"), "Moirai-1.1": ("dir", "moirai"),
          "Lag-Llama": ("dir", "lagllama"), "GJR-GARCH": ("bm", "gjr_garch"), "GJR-GARCH-t": ("bm", "gjr_t"),
          "GARCH-N": ("bm", "garch_n"), "Hist-Sim": ("bm", "hs"), "EWMA": ("bm", "ewma")}
TRUNCATED = {"Chronos-Small", "Chronos-Mini"}
assets = sorted(p.stem for p in pathlib.Path("cfp_ijf_data/returns").glob("*.csv"))

def pinball(y, q, a=ALPHA):
    return float(np.mean((a - (y < q).astype(float)) * (y - q)))

def conformal_q(s, a=ALPHA):
    s = np.sort(np.asarray(s)); n = len(s)
    return float(s[min(math.ceil((n + 1) * (1 - a)), n) - 1])

def opt_shift(y, q, a=ALPHA):
    """argmin_c mean L_a(q - c, y). For pinball loss this is the (1-a)-quantile of
    the scores q - y; verified numerically below rather than asserted."""
    return float(np.quantile(q - y, 1 - a, method="linear"))

rows = []
for model, (kind, tag) in SERIES.items():
    for a in assets:
        r = pd.read_csv(f"cfp_ijf_data/returns/{a}.csv", parse_dates=["date"]).set_index("date")["log_return"]
        p = f"cfp_ijf_data/{tag}/{a}.parquet" if kind == "dir" else f"cfp_ijf_data/benchmarks/{a}_{tag}.parquet"
        j = pd.read_parquet(p)[[f"VaR_{ALPHA}"]].join(r, how="inner").dropna()
        n = len(j); ncal = int(math.floor(n * FC))
        cal, test = j.iloc[:ncal], j.iloc[ncal:]
        qc, yc = cal[f"VaR_{ALPHA}"].to_numpy(), cal["log_return"].to_numpy()
        qt, yt = test[f"VaR_{ALPHA}"].to_numpy(), test["log_return"].to_numpy()
        qV = conformal_q(qc - yc)
        d_cal, d_test = opt_shift(yc, qc), opt_shift(yt, qt)
        S_raw_t = pinball(yt, qt)
        rows.append({
            "model": model, "symbol": a, "n_cal": ncal, "n_test": len(test),
            "qV": qV, "delta_cal": d_cal, "delta_test": d_test,
            "S_raw_test": S_raw_t,
            "uMCB_in_test": S_raw_t - pinball(yt, qt - d_test),
            "uMCB_oos_test": S_raw_t - pinball(yt, qt - qV),
            "S_raw_cal": pinball(yc, qc),
            "uMCB_in_cal": pinball(yc, qc) - pinball(yc, qc - d_cal),
            "improve_cal_qV": pinball(yc, qc) - pinball(yc, qc - qV),
            "truncated": model in TRUNCATED})
d = pd.DataFrame(rows)
d.to_csv("analysis/k0a_mcb/k0a_pairs.csv", index=False)
print(f"{len(d)} pairs = {d.model.nunique()} forecasters x {d.symbol.nunique()} assets, alpha = {ALPHA}\n")

# --- is the closed form for the argmin right? grid check on a few pairs --------
print("--- argmin closed form, checked numerically on 5 pairs ---")
for _, row in d.sample(5, random_state=0).iterrows():
    m, a = row.model, row.symbol
    kind, tag = SERIES[m]
    r = pd.read_csv(f"cfp_ijf_data/returns/{a}.csv", parse_dates=["date"]).set_index("date")["log_return"]
    p = f"cfp_ijf_data/{tag}/{a}.parquet" if kind == "dir" else f"cfp_ijf_data/benchmarks/{a}_{tag}.parquet"
    j = pd.read_parquet(p)[[f"VaR_{ALPHA}"]].join(r, how="inner").dropna()
    t = j.iloc[int(math.floor(len(j)*FC)):]
    q, y = t[f"VaR_{ALPHA}"].to_numpy(), t["log_return"].to_numpy()
    grid = np.linspace(row.delta_test - 0.01, row.delta_test + 0.01, 4001)
    best = grid[int(np.argmin([pinball(y, q - c) for c in grid]))]
    print(f"  {m:16s} {a:8s} closed form {row.delta_test:+.6f}  grid argmin {best:+.6f}  "
          f"gap {abs(best-row.delta_test):.2e}")

def report(sub, name):
    sp_ct = stats.spearmanr(sub.qV, sub.delta_test).statistic
    sp_cc = stats.spearmanr(sub.qV, sub.delta_cal).statistic
    sp_um = stats.spearmanr(sub.qV.abs(), sub.uMCB_in_test).statistic
    iqr = float(sub.qV.quantile(0.75) - sub.qV.quantile(0.25))
    med = float(np.median(np.abs(sub.qV - sub.delta_test)))
    rule = sp_ct > 0.95 and med < 0.25 * iqr
    print(f"\n[{name}]  n = {len(sub)}")
    print(f"  Spearman(qV, delta_cal   in-sample on calibration) {sp_cc:.4f}")
    print(f"  Spearman(qV, delta_test  in-sample on test)        {sp_ct:.4f}")
    print(f"  Spearman(|qV|, uMCB on test)                       {sp_um:.4f}")
    print(f"  median |qV - delta_test| {med:.6f}   IQR(qV) {iqr:.6f}   ratio {med/iqr:.3f}")
    print(f"  pre-registered rule -> {'REWRITING' if rule else 'DIFFERS'}")
    return {"n": len(sub), "spearman_qV_delta_cal": float(sp_cc), "spearman_qV_delta_test": float(sp_ct),
            "spearman_absqV_uMCB": float(sp_um), "median_abs_gap": med, "iqr_qV": iqr,
            "gap_over_iqr": med/iqr, "verdict": "rewriting" if rule else "differs"}

out = {"all": report(d, "all 312 pairs"),
       "well_specified": report(d[~d.truncated], "well-specified series only"),
       "truncated": report(d[d.truncated], "top_k-truncated series only")}

print("\n--- AE-3: in-sample improvement against the uMCB term, on the calibration window ---")
ratio = d.improve_cal_qV / d.uMCB_in_cal
print(f"  mean QS improvement from q̂_V on the calibration window: {d.improve_cal_qV.mean():.3e}")
print(f"  in-sample uMCB on the same window:                      {d.uMCB_in_cal.mean():.3e}")
print(f"  ratio, per pair: median {ratio.median():.6f}  5th pct {ratio.quantile(0.05):.6f}  "
      f"95th pct {ratio.quantile(0.95):.6f}  min {ratio.min():.6f}")
print(f"  pairs where the achieved improvement falls short of uMCB by >1%: "
      f"{int((ratio < 0.99).sum())}/{len(d)}")
out["AE3"] = {"mean_improvement_cal": float(d.improve_cal_qV.mean()),
              "mean_uMCB_in_cal": float(d.uMCB_in_cal.mean()),
              "ratio_median": float(ratio.median()), "ratio_p05": float(ratio.quantile(0.05)),
              "ratio_p95": float(ratio.quantile(0.95)), "ratio_min": float(ratio.min()),
              "n_short_by_1pct": int((ratio < 0.99).sum())}

print("\n--- out-of-sample: how much of the in-sample uMCB the conformal shift actually gets ---")
frac = d.uMCB_oos_test / d.uMCB_in_test
print(f"  median {frac.median():.4f}   5th pct {frac.quantile(0.05):.4f}   "
      f"pairs where the correction makes the test score worse: {int((d.uMCB_oos_test < 0).sum())}/{len(d)}")
out["oos"] = {"frac_median": float(frac.median()), "frac_p05": float(frac.quantile(0.05)),
              "n_worse": int((d.uMCB_oos_test < 0).sum())}

print("\n--- NEGATIVE CONTROLS ---")
rng = np.random.default_rng(0)
y = rng.standard_t(5, 4000) / 100
q0 = np.quantile(y, ALPHA)
u0 = pinball(y, np.full_like(y, q0)) - pinball(y, np.full_like(y, q0) - opt_shift(y, np.full_like(y, q0)))
print(f"  perfectly shifted constant forecast: uMCB {u0:.3e}  delta* {opt_shift(y, np.full_like(y,q0)):+.3e}  "
      f"-> {'near zero as required' if abs(u0) < 1e-8 else '!! CONTROL FAILED'}")
KNOWN = 0.01
qb = np.full_like(y, q0 + KNOWN)
ub = pinball(y, qb) - pinball(y, qb - opt_shift(y, qb))
print(f"  forecast shifted by a known {KNOWN}: recovered delta* {opt_shift(y, qb):+.6f} "
      f"(known {KNOWN})  uMCB {ub:.3e} -> "
      f"{'recovers the known shift' if abs(opt_shift(y,qb)-KNOWN) < 1e-3 and ub > 1e-6 else '!! CONTROL FAILED'}")
out["negative_controls"] = {"zero_uMCB": float(u0), "known_shift_recovered": float(opt_shift(y, qb)),
                           "known_shift_uMCB": float(ub)}
json.dump(out, open("analysis/k0a_mcb/k0a_result.json", "w"), indent=2)
