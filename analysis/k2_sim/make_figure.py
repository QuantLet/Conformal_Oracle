import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

g = pd.read_csv("analysis/k2_sim/grid.csv")
LAB = {"normal": "Normal (correct)", "t5": "Student-$t$(5)", "t3": "Student-$t$(3)",
       "skewt3": "Skewed-$t$(3, $-0.5$)", "mixnormal": "Mixture of Normals"}
COL = {"normal": "#4C6EF5", "t5": "#12B886", "t3": "#F59F00", "skewt3": "#E03131", "mixnormal": "#845EF7"}
fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.1))

for k in LAB:
    d = g[g.dgp == k].sort_values("T")
    ax[0].plot(d["T"], d.Corr_pi, "o-", color=COL[k], lw=1.8, ms=5, label=LAB[k])
    ax[1].plot(d["T"], d.RawGreen, "o-", color=COL[k], lw=1.8, ms=5)
    ax[2].plot(d["T"], d.Std_qV, "o-", color=COL[k], lw=1.8, ms=5)

ax[0].axhline(0.01, color="0.35", ls="--", lw=1.2)
ax[0].annotate("nominal $\\alpha = 0.01$", (520, 0.0102), fontsize=8, color="0.35")
ax[0].set_ylabel("corrected $\\hat\\pi$"); ax[0].set_ylim(0.006, 0.0125)
ax[0].set_title("(a) The correction converges to nominal", fontsize=10.5)

ax[1].axhline(50, color="0.85", lw=0.8)
ax[1].set_ylabel("raw forecasts in the Basel green zone (\\%)"); ax[1].set_ylim(-4, 104)
ax[1].set_title("(b) The raw traffic light diverges", fontsize=10.5)
ax[1].annotate("miscovering by 51\\%,\nand greener with more data", (1050, 60),
               fontsize=8, color=COL["t5"])

t = np.array([500, 10000])
ax[2].plot(t, g[g.dgp=="skewt3"].sort_values("T").Std_qV.iloc[0]*np.sqrt(500/t),
           color="0.45", ls=":", lw=1.4)
ax[2].annotate("$T^{-1/2}$", (3000, 0.010), fontsize=9, color="0.45")
ax[2].set_ylabel("sd of $\\hat q_V$ across replications")
ax[2].set_title("(c) And its variance falls at the parametric rate", fontsize=10.5)
ax[2].set_yscale("log")

for a in ax:
    a.set_xscale("log"); a.set_xlabel("sample size $T$")
    a.set_xticks([500, 1000, 2000, 5000, 10000])
    a.set_xticklabels(["500", "1{,}000", "2{,}000", "5{,}000", "10{,}000"], fontsize=8)
    a.grid(alpha=0.25, lw=0.6); a.spines[["top", "right"]].set_visible(False)
ax[0].legend(fontsize=8, frameon=False, loc="lower right")
fig.suptitle("Monte Carlo: what converges with more data, and what does not "
             "(500 replications per cell, $\\alpha = 0.01$)", fontsize=11.5, y=1.01)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(f"figures/fig_mc_convergence.{e}", dpi=200, bbox_inches="tight")
print("written figures/fig_mc_convergence.{pdf,png}")
