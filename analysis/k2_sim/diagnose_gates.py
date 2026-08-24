"""Why the two gates failed. Neither is loosened until the cause is identified."""
import numpy as np, pandas as pd, json
from scipy import stats
exec(open("analysis/k2_sim/run_grid.py").read().split("# ── Gate 1")[0])

# ---- Failure A: Std_qV outside 3 SE on 2 of 10 cells ------------------------
print("=== A. the Std_qV tolerance, not the implementation ===")
print("The declared SE for a standard deviation, Std/sqrt(2N), assumes a normal")
print("replication distribution. q_V is a 1% order statistic of heavy-tailed")
print("scores; its replication distribution is skewed. Measuring the true")
print("sampling variability of Std_qV by bootstrapping the committed replications:")
ref = pd.read_csv("Quantlets/CO_simulation_study/simulation_study_results.csv")
rng = np.random.default_rng(0)
for kind, T in [("t5", 5000), ("skewt3", 1000), ("normal", 1000)]:
    x = ref[(ref.dgp_name == kind) & (ref["T"] == T)].q_hat_V.to_numpy()
    boot = np.array([x[rng.integers(0, len(x), len(x))].std(ddof=1) for _ in range(4000)])
    naive = x.std(ddof=1)/np.sqrt(2*len(x))
    print(f"  {kind:9s} T={T:5d}  skew {stats.skew(x):+.2f}  kurtosis {stats.kurtosis(x):+.2f}  "
          f"Std_qV {x.std(ddof=1):.5f}  declared SE {naive:.6f}  bootstrap SE {boot.std():.6f}  "
          f"ratio {boot.std()/naive:.2f}x")

# ---- Failure B: negative control at T = 1000 --------------------------------
print("\n=== B. the negative control: convention, not harness ===")
print("The conformal index k = ceil((n+1)(1-alpha)) sits above the (1-alpha)")
print("sample percentile by k/n - (1-alpha), which shrinks with n:")
for T in [1000, 5000]:
    n = int(T*0.70); k = int(np.ceil((n+1)*0.99))
    print(f"  T={T:5d}  n={n:4d}  k={k:4d}  k/n = {k/n:.6f}  overshoot {k/n - 0.99:.6f}")
o1 = (int(np.ceil((700+1)*0.99))/700) - 0.99
o5 = (int(np.ceil((3500+1)*0.99))/3500) - 0.99
print(f"  predicted bias ratio T=1000 : T=5000  =  {o1/o5:.2f}x")

print("\nRe-running the control with the plain empirical quantile at exactly 1-alpha,")
print("which removes the convention and leaves only the harness:")
out = []
for kind, qtrue in [("t5", stats.t.ppf(0.01, 5)/np.sqrt(5/3)), ("t3", stats.t.ppf(0.01, 3)/np.sqrt(3.0))]:
    for T in [1000, 5000]:
        r, vf = simulate(kind, T, NREP, np.random.default_rng(99))
        vf_true = vf * (qtrue/Z)
        ncal = int(T*FC)
        s = np.sort(-vf_true[:, :ncal] - r[:, :ncal], axis=1)
        qV_exact = np.quantile(s, 0.99, axis=1, method="linear")
        k = int(np.ceil((ncal+1)*0.99)); qV_conf = s[:, min(k, ncal)-1]
        se_e, se_c = qV_exact.std(ddof=1)/np.sqrt(NREP), qV_conf.std(ddof=1)/np.sqrt(NREP)
        print(f"  {kind:5s} T={T:5d}  exact (1-a) quantile: {qV_exact.mean():+.6f} (3SE {3*se_e:.6f}) "
              f"{'ZERO' if abs(qV_exact.mean())<3*se_e else 'NONZERO'}   |   "
              f"conformal eq.(8): {qV_conf.mean():+.6f} (3SE {3*se_c:.6f}) "
              f"{'ZERO' if abs(qV_conf.mean())<3*se_c else 'NONZERO'}")
        out.append({"dgp": kind, "T": T, "mean_qV_exact": float(qV_exact.mean()),
                    "three_se_exact": float(3*se_e), "mean_qV_conformal": float(qV_conf.mean()),
                    "three_se_conformal": float(3*se_c),
                    "ratio_conf_over_exact": float(qV_conf.mean()/max(abs(qV_exact.mean()), 1e-12))})
json.dump(out, open("analysis/k2_sim/gate_diagnosis.json", "w"), indent=2)
