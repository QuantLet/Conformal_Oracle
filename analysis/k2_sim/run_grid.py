"""K2 §5 — Monte Carlo on a convergence grid in T.

Recursions and estimator written fresh and vectorised across replications. The
committed study (Quantlets/CO_simulation_study) is reproduced at its two sample
sizes before any new sample size is reported.
"""
import numpy as np, pandas as pd, json, zlib
from scipy import stats
from scipy.special import gamma as gammafn

ALPHA, FC, NREP = 0.01, 0.70, 500
NREP_REPRO = 2000   # 4x the committed run, so its 500-rep error dominates the comparison
OMEGA, A1, B1 = 1e-5, 0.10, 0.85
GRID = [500, 1000, 2000, 5000, 10000]
DGPS = ["normal", "t5", "t3", "skewt3", "mixnormal"]
Z = stats.norm.ppf(ALPHA)

def seed_for(*parts):
    """Deterministic across processes; hash() is salted per interpreter."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFFFFFF

def innovations(kind, shape, rng):
    if kind == "normal":
        return rng.standard_normal(shape)
    if kind == "t5":
        return stats.t.rvs(df=5, size=shape, random_state=rng) / np.sqrt(5/3)
    if kind == "t3":
        return stats.t.rvs(df=3, size=shape, random_state=rng) / np.sqrt(3.0)
    if kind == "skewt3":
        df, eta = 3, -0.5
        c = gammafn((df+1)/2) / (np.sqrt(np.pi*(df-2))*gammafn(df/2))
        a = 4*eta*c*(df-2)/(df-1); b = np.sqrt(1 + 3*eta**2 - a**2)
        sc = np.sqrt((df-2)/df)
        u = rng.uniform(size=shape); out = np.zeros(shape)
        left = u < (1-eta)/2
        out[left] = ((1-eta)*stats.t.ppf(u[left]/(1-eta), df)*sc - a)/b
        out[~left] = ((1+eta)*stats.t.ppf((u[~left]+eta)/(1+eta), df)*sc - a)/b
        return out
    if kind == "mixnormal":
        u = rng.uniform(size=shape)
        e = np.where(u < 0.95, rng.normal(0, 1, shape), rng.normal(0, 5, shape))
        return (e - e.mean(axis=1, keepdims=True)) / e.std(axis=1, keepdims=True)
    raise ValueError(kind)

def simulate(kind, T, nrep, rng):
    """GARCH(1,1) paths and the true-parameter Normal-innovation VaR forecast."""
    eps = innovations(kind, (nrep, T), rng)
    r = np.zeros((nrep, T)); s2 = np.zeros((nrep, T))
    s2[:, 0] = OMEGA / (1 - A1 - B1)                    # r[:,0] = 0, as in the reference
    for t in range(1, T):
        s2[:, t] = OMEGA + A1 * r[:, t-1]**2 + B1 * s2[:, t-1]
        r[:, t] = np.sqrt(s2[:, t]) * eps[:, t]
    f2 = np.zeros((nrep, T)); f2[:, 0] = OMEGA / (1 - A1 - B1)
    for t in range(1, T):
        f2[:, t] = OMEGA + A1 * r[:, t-1]**2 + B1 * f2[:, t-1]
    return r, -np.sqrt(f2) * Z                           # VaR as a positive threshold

def evaluate(r, var_f, convention):
    nrep, T = r.shape
    ncal = int(T * FC)
    q_lo = -var_f
    s = np.sort(q_lo[:, :ncal] - r[:, :ncal], axis=1)
    n = ncal
    k = int(np.ceil((n + 1) * (1 - ALPHA)))
    if convention == "order":                            # eq. (8) of the manuscript
        qV = s[:, min(k, n) - 1]
    elif convention == "committed":                      # np.quantile at level k/n, linear
        lvl = min(k / n, 1.0)
        qV = np.quantile(s, lvl, axis=1, method="linear")
    else:
        raise ValueError(convention)
    rt, vt = r[:, ncal:], var_f[:, ncal:]
    raw_pi = (rt < -vt).mean(axis=1)
    corr_pi = (rt < -(vt + qV[:, None])).mean(axis=1)
    green = lambda p: (p * 250 <= 4 + 1e-12)
    return pd.DataFrame({"qV": qV, "raw_pi": raw_pi, "corr_pi": corr_pi,
                         "raw_green": green(raw_pi), "corr_green": green(corr_pi)})

def summarise(d):
    return dict(Mean_qV=d.qV.mean(), Std_qV=d.qV.std(ddof=1),
                Raw_pi=d.raw_pi.mean(), Corr_pi=d.corr_pi.mean(),
                RawGreen=100*d.raw_green.mean(), CorrGreen=100*d.corr_green.mean())

# ── Gate 1: the conformal index, exactly, with no Monte Carlo in it ──────────
print("--- convention check: the committed script's quantile level against eq. (8) ---")
rng0 = np.random.default_rng(7)
gaps = []
for T in [1000, 5000]:
    n = int(T * FC); k = int(np.ceil((n + 1) * (1 - ALPHA)))
    s = np.sort(rng0.standard_normal(n) * 0.01)
    exact = s[min(k, n) - 1]
    comm = np.quantile(s, min(k / n, 1.0), method="linear")
    idx = min(k / n, 1.0) * (n - 1)
    print(f"  T={T:5d} n={n:4d} k={k:4d}: eq.(8) takes 0-based index {k-1}, "
          f"np.quantile at level k/n takes {idx:.3f} -> interpolates. "
          f"gap {abs(comm-exact):.3e} ({abs(comm-exact)/abs(exact):.2%} of the value)")
    gaps.append(float(abs(comm - exact)))

# ── Gate 2: reproduce the committed cells ───────────────────────────────────
ref = pd.read_csv("Quantlets/CO_simulation_study/simulation_study_results.csv")
refsum = ref.groupby(["dgp_name", "T"]).apply(
    lambda g: pd.Series(dict(Mean_qV=g.q_hat_V.mean(), Std_qV=g.q_hat_V.std(ddof=1),
                             Raw_pi=g.raw_pi.mean(), Corr_pi=g.corr_pi.mean(),
                             RawGreen=100*(g.raw_TL == "G").mean(),
                             CorrGreen=100*(g.corr_TL == "G").mean())),
    include_groups=False)

print("\n--- reproduction of the committed study, 3 Monte Carlo SE tolerance ---")
rows, ok_all = [], True
for kind in DGPS:
    for T in [1000, 5000]:
        rng = np.random.default_rng(seed_for(kind, T, "repro"))
        r, vf = simulate(kind, T, NREP_REPRO, rng)
        mine = summarise(evaluate(r, vf, "committed"))
        want = refsum.loc[(kind, T)]
        xb = ref[(ref.dgp_name == kind) & (ref["T"] == T)].q_hat_V.to_numpy()
        rb = np.random.default_rng(0)
        boot_se = np.array([xb[rb.integers(0, len(xb), len(xb))].std(ddof=1)
                            for _ in range(4000)]).std()
        # SE of the DIFFERENCE of two independent estimates: var adds.
        f = np.sqrt(1.0 + NREP/NREP_REPRO)
        pg = want.RawGreen/100.0; pc = want.CorrGreen/100.0
        se = {"Mean_qV": f*want.Std_qV/np.sqrt(NREP), "Std_qV": f*boot_se,
              "Raw_pi": f*0.007/np.sqrt(NREP), "Corr_pi": f*0.007/np.sqrt(NREP),
              "RawGreen": f*100*np.sqrt(max(pg*(1-pg), 0.01)/NREP),
              "CorrGreen": f*100*np.sqrt(max(pc*(1-pc), 0.01)/NREP)}
        bad = [q for q in mine if abs(mine[q] - want[q]) > 3*se[q]]
        ok_all &= not bad
        print(f"  {kind:10s} T={T:5d}  qV {mine['Mean_qV']:.4f} vs {want.Mean_qV:.4f} | "
              f"raw pi {mine['Raw_pi']:.4f} vs {want.Raw_pi:.4f} | "
              f"green {mine['RawGreen']:5.1f} vs {want.RawGreen:5.1f}  "
              f"{'OK' if not bad else 'OUTSIDE 3 SE: ' + ','.join(bad)}")
        rows.append({"dgp": kind, "T": T, **{f"mine_{q}": v for q, v in mine.items()},
                     **{f"ref_{q}": float(want[q]) for q in mine}})
pd.DataFrame(rows).to_csv("analysis/k2_sim/reproduction.csv", index=False)
print(f"\nreproduction gate: {'PASSED' if ok_all else 'FAILED — extension abandoned'}")

# ── Gate 3: negative control ────────────────────────────────────────────────
print("\n--- NEGATIVE CONTROL: forecaster given the correct innovation law ---")
print("    tests the harness, at the exact (1-alpha) quantile. The conformal")
print("    convention's own small-n bias is reported beside it as a result, not")
print("    as a control failure -- see GATE_REVISION.md.")
ctrl = []
for kind, q in [("t5", stats.t.ppf(ALPHA, 5)/np.sqrt(5/3)), ("t3", stats.t.ppf(ALPHA, 3)/np.sqrt(3.0))]:
    for T in GRID:
        rng = np.random.default_rng(99)
        r, vf = simulate(kind, T, NREP, rng)
        vf_true = vf * (q / Z)                    # same sigma path, correct innovation quantile
        ncal = int(T * FC)
        srt = np.sort(-vf_true[:, :ncal] - r[:, :ncal], axis=1)
        qv_exact = np.quantile(srt, 1 - ALPHA, axis=1, method="linear")
        kk = int(np.ceil((ncal + 1) * (1 - ALPHA)))
        qv_conf = srt[:, min(kk, ncal) - 1]
        se_e = qv_exact.std(ddof=1)/np.sqrt(NREP); se_c = qv_conf.std(ddof=1)/np.sqrt(NREP)
        flag = abs(qv_exact.mean()) < 3*se_e
        print(f"  {kind:5s} T={T:6d} n_cal={ncal:5d}  harness (exact 1-a): {qv_exact.mean():+.6f} "
              f"(3SE {3*se_e:.6f}) {'OK' if flag else '!! CONTROL FAILED'}   |   "
              f"eq.(8) bias: {qv_conf.mean():+.6f}  overshoot k/n-(1-a) = "
              f"{kk/ncal - (1-ALPHA):.6f}")
        ctrl.append({"dgp": kind, "T": T, "n_cal": ncal,
                     "mean_qV_exact": float(qv_exact.mean()), "three_se_exact": float(3*se_e),
                     "mean_qV_conformal": float(qv_conf.mean()), "three_se_conformal": float(3*se_c),
                     "overshoot": float(kk/ncal - (1-ALPHA)), "harness_passes": bool(flag)})

if not ok_all:
    raise SystemExit("reproduction gate failed; no grid reported")

# ── The grid ────────────────────────────────────────────────────────────────
print("\n--- convergence grid, eq. (8) convention, 500 replications per cell ---")
grid, per_rep = [], []
for kind in DGPS:
    for T in GRID:
        rng = np.random.default_rng(seed_for(kind, T, "grid"))
        r, vf = simulate(kind, T, NREP, rng)
        d = evaluate(r, vf, "order")
        d.insert(0, "T", T); d.insert(0, "dgp", kind); per_rep.append(d)
        s = summarise(d); s.update(dgp=kind, T=T, n_test=T - int(T*FC))
        grid.append(s)
        print(f"  {kind:10s} T={T:6d}  qV {s['Mean_qV']:+.5f} (sd {s['Std_qV']:.5f})  "
              f"raw pi {s['Raw_pi']:.4f}  corr pi {s['Corr_pi']:.4f}  "
              f"green raw {s['RawGreen']:5.1f}%  corr {s['CorrGreen']:5.1f}%")
g = pd.DataFrame(grid)[["dgp","T","n_test","Mean_qV","Std_qV","Raw_pi","Corr_pi","RawGreen","CorrGreen"]]
g.to_csv("analysis/k2_sim/grid.csv", index=False)
pd.concat(per_rep).to_csv("analysis/k2_sim/grid_per_rep.csv", index=False)
json.dump({"reproduction_passed": bool(ok_all), "negative_controls": ctrl,
           "convention_gap_abs": gaps}, open("analysis/k2_sim/gates.json", "w"), indent=2)
print("\nwritten: analysis/k2_sim/grid.csv, grid_per_rep.csv, reproduction.csv, gates.json")
