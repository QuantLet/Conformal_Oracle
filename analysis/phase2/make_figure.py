#!/usr/bin/env python3
"""One figure: two models no backtest can separate, and what separates them."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

d = np.load("analysis/phase2/pair.npz")
x, p, q_true, q_trunc, nu = d["x"], d["p"], float(d["q_true"]), float(d["q_trunc"]), int(d["nu"])
sc = np.sqrt(nu/(nu-2))
rng = np.random.default_rng(20260822); T = 250_000
eP = stats.t.rvs(nu, size=T, random_state=1)/sc
eQ = rng.choice(x, size=T, p=p/p.sum())
VP, VQ = eP < q_true, eQ < q_trunc

fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))
fig.suptitle("Two models with identical exceedance processes and different predictive objects",
             fontsize=12.5, y=1.00)

# -- left: the predictive laws and the two reported thresholds
g = np.linspace(-6, 6, 1200)
ax[0].plot(g, stats.t.pdf(g*sc, nu)*sc, lw=1.8, label="model $P$ (honest)")
m = np.abs(x) <= 6
ax[0].plot(x[m], p[m]/np.gradient(x)[m], lw=1.8, ls="--", label="model $Q$ (compensating)")
ax[0].axvline(q_true, color="C0", lw=1.2, ls=":")
ax[0].axvline(q_trunc, color="C1", lw=1.2, ls=":")
ax[0].annotate(f"$P$ reports\n{-q_true:.2f}$\\sigma$", (q_true, 0.30), (-5.6, 0.34),
               fontsize=8.5, color="C0", arrowprops=dict(arrowstyle="->", color="C0", lw=.8))
ax[0].annotate(f"$Q$ reports\n{-q_trunc:.2f}$\\sigma$", (q_trunc, 0.16), (-4.2, 0.20),
               fontsize=8.5, color="C1", arrowprops=dict(arrowstyle="->", color="C1", lw=.8))
ax[0].set_xlim(-6, 6); ax[0].set_ylim(0, 0.46)
ax[0].set_title("Predictive laws differ", fontsize=10)
ax[0].set_xlabel("standardised return"); ax[0].set_ylabel("density")
ax[0].legend(fontsize=8.5, frameon=False)

# -- middle: everything the exceedance path can see
def kup(V, a=0.01):
    n=len(V); k=V.sum(); pi=k/n
    lr=-2*((n-k)*np.log(1-a)+k*np.log(a)-((n-k)*np.log(1-pi)+k*np.log(pi)))
    return 1-stats.chi2.cdf(lr,1)
lab = ["exceedance\nrate / $\\alpha$", "Kupiec $p$", "1-step\ntransition $\\hat\\pi_{11}$"]
def tr11(V):
    a=V[:-1].astype(int); b=V[1:].astype(int)
    return np.sum((a==1)&(b==1))/max(np.sum(a==1),1)
vals = np.array([[VP.mean()/0.01, kup(VP), tr11(VP)],
                 [VQ.mean()/0.01, kup(VQ), tr11(VQ)]])
xx = np.arange(3); w = 0.35
ax[1].bar(xx-w/2, vals[0], w, label="$P$"); ax[1].bar(xx+w/2, vals[1], w, label="$Q$")
for i in range(3):
    ax[1].text(xx[i]-w/2, vals[0,i]+.02, f"{vals[0,i]:.3f}", ha="center", fontsize=7.5)
    ax[1].text(xx[i]+w/2, vals[1,i]+.02, f"{vals[1,i]:.3f}", ha="center", fontsize=7.5)
ax[1].set_xticks(xx); ax[1].set_xticklabels(lab, fontsize=8.5)
ax[1].set_title("Everything the exceedance path sees agrees", fontsize=10)
ax[1].legend(fontsize=8.5, frameon=False); ax[1].set_ylim(0, 1.25)

# -- right: what the structural checks see
names = ["reported VaR\n(units of $\\sigma$)", "predictive sd\n/ realised sd", "point-forecast\nMAE"]
P = [-q_true, 1.0, np.abs(eP).mean()]
Q = [-q_trunc, np.sqrt((p*x**2).sum()), np.abs(eQ).mean()]
xx = np.arange(3)
ax[2].bar(xx-w/2, P, w, label="$P$"); ax[2].bar(xx+w/2, Q, w, label="$Q$")
for i in range(3):
    ax[2].text(xx[i]-w/2, P[i]+.03, f"{P[i]:.3f}", ha="center", fontsize=7.5)
    ax[2].text(xx[i]+w/2, Q[i]+.03, f"{Q[i]:.3f}", ha="center", fontsize=7.5)
ax[2].set_xticks(xx); ax[2].set_xticklabels(names, fontsize=8.5)
ax[2].set_title("Only the reported threshold separates them", fontsize=10)
ax[2].legend(fontsize=8.5, frameon=False); ax[2].set_ylim(0, 3.2)
for a in ax: a.spines[["top","right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/fig_nonidentification.pdf", bbox_inches="tight")
fig.savefig("analysis/phase2/fig_nonidentification.png", dpi=150, bbox_inches="tight")
print("wrote figures/fig_nonidentification.pdf")
