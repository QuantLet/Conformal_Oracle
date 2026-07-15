"""
CO_identification — run_identification.py
=========================================
Monte Carlo identification panel for the conformal shift.

Takes the ORACLE Value-at-Risk under each of the five data generating
processes of the main simulation study (CO_simulation_study) and contaminates
it along one channel at a time, to show that the conformal shift q_hat_V and
the audit ratio R separate location/scale misspecification (signal-preserving,
R < 1, fully repaired) from replacement of the tail signal (R > 1):

  * pure bias   VaR_raw = VaR_true + a,   a in {-1, -0.5, +0.5, +1} * sigma,  b = 1
  * pure scale  VaR_raw = b * VaR_true,   b in {0.5, 0.8, 1.25, 2},           a = 0
  * pure noise  VaR_raw = signal-free white noise of matched variance (no
                correct tail level; rectified to positive-loss convention)

DGPs (all GARCH(1,1), omega=1e-5, alpha=0.10, beta=0.85):
  1 Normal (correct)   2 Student-t(5)   3 Student-t(3)
  4 Skewed-t(3,-0.5)   5 Mixture 0.95 N(0,1)+0.05 N(0,25)

The "oracle" forecaster knows the DGP: VaR_true_t = -sigma_t * Q_alpha(eps),
where Q_alpha(eps) is the alpha-quantile of the standardized (unit-variance)
innovation. Contamination is applied to this oracle series.

For each (DGP, channel, level, T) cell we record q_hat_V, R, and the raw and
post-correction violation rates, averaged over 500 replications at both
T in {1,000; 5,000} (alpha=0.01, f_c=0.70), matching the main MC design.

Outputs:
  tab_identification.tex          LaTeX table (Table for rr_material.tex)
  fig_identification.pdf/.png     (q_hat_V, R) identification trajectories
  identification_results.csv      Full per-cell summary (all DGPs)

Usage:  python run_identification.py
"""

from __future__ import annotations

import os
import time
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── Configuration (matches CO_simulation_study) ─────────────────────────────
ALPHA = 0.01
FC = 0.70
N_REP = 500
GARCH_OMEGA = 1e-5
GARCH_ALPHA = 0.10
GARCH_BETA = 0.85
SAMPLE_SIZES = [1000, 5000]
SEED = 42

# Unconditional return std of the GARCH(1,1) (E[sigma_t^2] = omega/(1-a-b)):
SIGMA_UNCOND = np.sqrt(GARCH_OMEGA / (1.0 - GARCH_ALPHA - GARCH_BETA))

BIAS_LEVELS = [-1.0, -0.5, 0.5, 1.0]  # multiples of SIGMA_UNCOND, b = 1
SCALE_LEVELS = [0.5, 0.8, 1.25, 2.0]  # multiplicative, a = 0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
FIG_DIR = os.path.join(ROOT_DIR, "figures")

DGPS = [
    (1, "Normal (correct)", "normal"),
    (2, "Student-$t$(5)", "t5"),
    (3, "Student-$t$(3)", "t3"),
    (4, "Skewed-$t$(3)", "skewt3"),
    (5, "Mix.\\ Normals", "mixnormal"),
]


# ── Innovation sampling (standardized to unit variance) ─────────────────────
def _hansen_skewt_rvs(df, eta, size, rng):
    """Hansen (1994) skewed-t, standardized (unit variance), via inverse CDF."""
    from scipy.special import gamma as gammafn

    c = gammafn((df + 1) / 2) / (np.sqrt(np.pi * (df - 2)) * gammafn(df / 2))
    a = 4 * eta * c * (df - 2) / (df - 1)
    b = np.sqrt(1 + 3 * eta**2 - a**2)
    scale = np.sqrt((df - 2) / df)

    u = rng.uniform(size=size)
    threshold = (1 - eta) / 2
    result = np.zeros(size)

    left = u < threshold
    if left.any():
        p = u[left] / (1 - eta)
        w = stats.t.ppf(p, df) * scale
        result[left] = ((1 - eta) * w - a) / b
    right = ~left
    if right.any():
        p = (u[right] + eta) / (1 + eta)
        w = stats.t.ppf(p, df) * scale
        result[right] = ((1 + eta) * w - a) / b
    return result


def sample_innovations(innov, shape, rng):
    """Standardized (mean 0, variance 1) innovations of the requested family."""
    if innov == "normal":
        return rng.standard_normal(shape)
    if innov == "t5":
        return stats.t.rvs(df=5, size=shape, random_state=rng) / np.sqrt(5.0 / 3.0)
    if innov == "t3":
        return stats.t.rvs(df=3, size=shape, random_state=rng) / np.sqrt(3.0)
    if innov == "skewt3":
        flat = _hansen_skewt_rvs(df=3, eta=-0.5, size=int(np.prod(shape)), rng=rng)
        return flat.reshape(shape)
    if innov == "mixnormal":
        u = rng.uniform(size=shape)
        eps = np.where(u < 0.95, rng.normal(0, 1, shape), rng.normal(0, 5, shape))
        return eps / np.sqrt(2.2)  # Var = .95*1 + .05*25 = 2.2
    raise ValueError(f"Unknown innovation type: {innov}")


def oracle_innovation_quantile(innov, alpha=ALPHA):
    """Q_alpha of the standardized innovation (the oracle's tail multiplier)."""
    rng = np.random.default_rng(20260715)
    big = sample_innovations(innov, (5_000_000,), rng)
    return float(np.quantile(big, alpha))


# ── Vectorized GARCH(1,1) simulation across replications ────────────────────
def simulate_garch_panel(T, n_rep, innov, rng):
    """Return (r, sigma) arrays of shape (n_rep, T)."""
    eps = sample_innovations(innov, (n_rep, T), rng)
    sigma2 = np.empty((n_rep, T))
    r = np.empty((n_rep, T))
    sigma2[:, 0] = GARCH_OMEGA / (1.0 - GARCH_ALPHA - GARCH_BETA)
    r[:, 0] = np.sqrt(sigma2[:, 0]) * eps[:, 0]
    for t in range(1, T):
        sigma2[:, t] = (
            GARCH_OMEGA + GARCH_ALPHA * r[:, t - 1] ** 2 + GARCH_BETA * sigma2[:, t - 1]
        )
        r[:, t] = np.sqrt(sigma2[:, t]) * eps[:, t]
    return r, np.sqrt(sigma2)


# ── Conformal correction (single-split, matches CO_simulation_study) ────────
def conformal_stats(returns, var_raw, alpha=ALPHA, f_cal=FC):
    """q_hat_V, R, and raw/post-correction violation rates on the test split.

    var_raw is the (contaminated) VaR in positive-loss convention.
    """
    T = returns.shape[-1]
    n_cal = int(T * f_cal)

    q_lo = -var_raw
    s_v = q_lo[:n_cal] - returns[:n_cal]
    n = len(s_v)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat_v = np.quantile(s_v, q_level)

    corrected_var = var_raw + q_hat_v
    test_r = returns[n_cal:]
    raw_pi = float((test_r < -var_raw[n_cal:]).mean())
    corr_pi = float((test_r < -corrected_var[n_cal:]).mean())
    r_ratio = abs(q_hat_v) / abs(np.mean(var_raw))
    return q_hat_v, r_ratio, raw_pi, corr_pi


def contaminate(var_true, channel, level, rng):
    """Return contaminated raw VaR for the requested channel/level."""
    if channel == "bias":
        return var_true + level * SIGMA_UNCOND
    if channel == "scale":
        return level * var_true
    if channel == "noise":
        # Signal-free forecast: white noise carrying no correct tail level.
        # Variance matched to the oracle VaR series; rectified to the
        # positive-loss convention so the ratio R stays finite. The forecast
        # keeps a plausible scale but no relationship to sigma_t, so it
        # under-states the tail on average and the conformal correction must
        # supply the level -> replacement regime.
        s = var_true.std()
        return np.abs(rng.normal(0.0, s, size=var_true.shape))
    raise ValueError(channel)


# ── Main loop ───────────────────────────────────────────────────────────────
def run():
    print("=" * 72)
    print("Identification panel — contaminating the oracle VaR")
    print(f"  {N_REP} reps, alpha={ALPHA}, f_c={FC}, T in {SAMPLE_SIZES}")
    print(f"  sigma_uncond = {SIGMA_UNCOND:.5f}")
    print("=" * 72)

    q_alpha = {innov: oracle_innovation_quantile(innov) for _, _, innov in DGPS}
    for _, name, innov in DGPS:
        print(f"  Q_alpha[{innov:10s}] = {q_alpha[innov]:.4f}")

    cells = (
        [("bias", lv) for lv in BIAS_LEVELS]
        + [("scale", lv) for lv in SCALE_LEVELS]
        + [("noise", 0.0)]
    )

    rows = []
    for dgp_idx, dgp_name, innov in DGPS:
        for T in SAMPLE_SIZES:
            t0 = time.time()
            rng = np.random.default_rng(SEED + 1000 * dgp_idx + T)
            r, sigma = simulate_garch_panel(T, N_REP, innov, rng)
            var_true = -sigma * q_alpha[innov]  # positive-loss oracle VaR

            for channel, level in cells:
                q_list, R_list, rawpi_list, corrpi_list = [], [], [], []
                for rep in range(N_REP):
                    var_raw = contaminate(var_true[rep], channel, level, rng)
                    q_hat_v, R, raw_pi, corr_pi = conformal_stats(r[rep], var_raw)
                    q_list.append(q_hat_v)
                    R_list.append(R)
                    rawpi_list.append(raw_pi)
                    corrpi_list.append(corr_pi)
                rows.append(
                    dict(
                        dgp=dgp_idx,
                        dgp_name=dgp_name,
                        innov=innov,
                        channel=channel,
                        level=level,
                        T=T,
                        q_hat_V=np.mean(q_list),
                        R=np.mean(R_list),
                        raw_pi=np.mean(rawpi_list),
                        corr_pi=np.mean(corrpi_list),
                        R_gt1_share=np.mean(np.array(R_list) > 1.0),
                    )
                )
            print(f"  DGP {dgp_idx} {innov:10s} T={T:5d}  ({time.time() - t0:.1f}s)")

    return pd.DataFrame(rows)


# ── LaTeX table (averaged across the five DGPs, per channel/level/T) ─────────
def _channel_label(channel, level):
    if channel == "bias":
        sign = "+" if level > 0 else "-"
        return f"Bias $a={sign}{abs(level):g}\\sigma$"
    if channel == "scale":
        return f"Scale $b={level:g}$"
    return "Noise (matched var.)"


def generate_table(df):
    agg = (
        df.groupby(["channel", "level", "T"], sort=False)[
            ["q_hat_V", "R", "raw_pi", "corr_pi"]
        ]
        .mean()
        .reset_index()
    )
    order = (
        [("bias", lv) for lv in BIAS_LEVELS]
        + [("scale", lv) for lv in SCALE_LEVELS]
        + [("noise", 0.0)]
    )

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"	\centering")
    lines.append(
        r"	\caption{Identification of the conformal shift under controlled "
        r"contamination of the oracle Value-at-Risk (500 replications, "
        r"$\alpha = 0.01$, $f_c = 0.70$; averages across the five DGPs of "
        r"Table~\ref{tab:simulation_extended}). $\qVstat$: mean conformal "
        r"shift; $R=|\qVstat|/|\overline{\VaR^{\mathrm{raw}}}|$: mean audit "
        r"ratio; $\hat\pi_{\mathrm{raw}}$ / $\hat\pi_{\mathrm{cp}}$: raw and "
        r"post-correction violation rates. Bias and scale cells stay at or "
        r"below the replacement threshold ($R\le 1$; the scale cell $b=\tfrac12$ "
        r"sits at the boundary $R=1$ predicted by "
        r"Corollary~\ref{cor:rr_R_response}) and are repaired to nominal "
        r"coverage; only the noise channel enters the replacement regime "
        r"($R>1$).}"
    )
    lines.append(r"	\label{tab:identification}")
    lines.append(r"	\footnotesize")
    lines.append(r"	\begin{tabular}{@{}l c rrrr@{}}")
    lines.append(r"		\hline\hline")
    lines.append(
        r"		Contamination & $T$ & $\qVstat$ & $R$ "
        r"& $\hat\pi_{\mathrm{raw}}$ & $\hat\pi_{\mathrm{cp}}$ \\"
    )
    lines.append(r"		\hline")

    last_channel = None
    for channel, level in order:
        block = agg[(agg["channel"] == channel) & (agg["level"] == level)]
        block = block.set_index("T")
        if channel != last_channel:
            titles = {
                "bias": "Pure bias ($b=1$)",
                "scale": "Pure scale ($a=0$)",
                "noise": "Pure noise",
            }
            lines.append(
                rf"		\multicolumn{{6}}{{@{{}}l}}{{\textit{{{titles[channel]}}}}} \\[1pt]"
            )
            last_channel = channel
        label = _channel_label(channel, level)
        for i, T in enumerate(SAMPLE_SIZES):
            row = block.loc[T]
            lab = label if i == 0 else ""
            lines.append(
                f"		{lab} & {T:,d} & {row['q_hat_V']:.4f} & {row['R']:.3f} "
                f"& {row['raw_pi']:.3f} & {row['corr_pi']:.3f} \\\\"
            )
    lines.append(r"		\hline\hline")
    lines.append(r"	\end{tabular}")
    lines.append(r"	\begin{minipage}{\linewidth}\scriptsize")
    lines.append(r"		\smallskip")
    lines.append(
        r"		$\sigma$ is the unconditional return standard deviation "
        r"$\sqrt{\omega/(1-\alpha_g-\beta_g)}$. The pure-bias shift matches "
        r"$\qVstat=-a$ and the pure-scale ratio matches $R=|1-b|/b$ of "
        r"Proposition~\ref{prop:decomposition} and "
        r"Corollary~\ref{cor:rr_R_response} up to finite-sample error."
    )
    lines.append(r"	\end{minipage}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines) + "\n"
    out = os.path.join(SCRIPT_DIR, "tab_identification.tex")
    with open(out, "w") as f:
        f.write(tex)
    print(f"\nSaved: {out}")


# ── Figure: (q_hat_V, R) identification trajectories ────────────────────────
def generate_figure(df):
    d = df[df["T"] == 5000]
    agg = d.groupby(["channel", "level"], sort=False)[["q_hat_V", "R"]].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    c_bias, c_scale, c_noise = "#1f77b4", "#d62728", "#7f7f7f"

    # Left: bias channel — q_hat_V identifies -a
    ax = axes[0]
    a_grid = np.array(BIAS_LEVELS)
    qv = [agg.loc[("bias", lv), "q_hat_V"] for lv in BIAS_LEVELS]
    xs = a_grid * SIGMA_UNCOND
    line = np.linspace(xs.min() * 1.15, xs.max() * 1.15, 50)
    ax.plot(line, -line, color="black", lw=1.3, ls="--", label=r"theory $\hat q_V=-a$")
    ax.plot(xs, qv, "o-", color=c_bias, ms=8, lw=2, label="simulated (mean)")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel(r"additive bias $a$", fontsize=13)
    ax.set_ylabel(r"conformal shift $\hat q_V^{\,\mathrm{stat}}$", fontsize=13)
    ax.set_title("Pure bias: the shift identifies $-a$", fontsize=13)
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(alpha=0.2)

    # Right: scale channel — R = |1-b|/b, and the noise channel above R=1
    ax = axes[1]
    b_grid = np.array(SCALE_LEVELS)
    Rs = [agg.loc[("scale", lv), "R"] for lv in SCALE_LEVELS]
    R_noise = agg.loc[("noise", 0.0), "R"]
    ymax = max(3.9, R_noise * 1.12)

    ax.axhspan(1.0, ymax, color=c_scale, alpha=0.05, zorder=0)
    ax.text(0.47, ymax * 0.93, "replacement regime ($R>1$)", fontsize=9, color="0.35")
    bb = np.linspace(0.45, 2.05, 100)
    ax.plot(
        bb,
        np.abs(1 - bb) / bb,
        color="black",
        lw=1.3,
        ls="--",
        label=r"theory $R=|1-b|/b$",
    )
    ax.plot(b_grid, Rs, "o-", color=c_scale, ms=8, lw=2, label="scale (simulated)")
    ax.axhline(
        R_noise,
        color=c_noise,
        lw=2,
        ls="-.",
        label=f"noise channel ($R\\approx{R_noise:.1f}$)",
    )
    ax.axhline(1.0, color="0.4", lw=1.0, ls=":")
    ax.text(1.72, 1.06, "threshold $R=1$", fontsize=9, color="0.35")
    ax.set_xlim(0.42, 2.08)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"scale factor $b$", fontsize=13)
    ax.set_ylabel(r"audit ratio $R$", fontsize=13)
    ax.set_title("Pure scale vs. noise: the audit ratio", fontsize=13)
    ax.legend(fontsize=10.5, framealpha=0.9, loc="upper right")
    ax.grid(alpha=0.2)

    fig.suptitle(
        "Identification of the conformal shift across contamination channels "
        "($T=5{,}000$)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ["pdf", "png"]:
        out = os.path.join(SCRIPT_DIR, f"fig_identification.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ["pdf", "png"]:
        shutil.copy2(
            os.path.join(SCRIPT_DIR, f"fig_identification.{ext}"),
            os.path.join(FIG_DIR, f"fig_identification.{ext}"),
        )
    print(f"Saved: {os.path.join(SCRIPT_DIR, 'fig_identification.pdf/.png')}")
    plt.close(fig)


if __name__ == "__main__":
    t_total = time.time()
    df = run()
    csv = os.path.join(SCRIPT_DIR, "identification_results.csv")
    df.to_csv(csv, index=False)
    print(f"\nSaved: {csv}")
    generate_table(df)
    generate_figure(df)
    print(f"\nTotal runtime: {time.time() - t_total:.0f}s")
