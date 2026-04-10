"""
Figure 3: Two-Panel Calibration-Efficiency Frontier
=====================================================
Produces a two-panel subfigure:
  (a) Zoomed view near 99% target
  (b) Full scale including Chronos-Small (raw pi_hat = 38.8%)

Data computed from the all_results.csv backtest outputs.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / 'cfp_ijf_data'
RES  = DATA / 'paper_outputs' / 'tables'
FIG  = Path(__file__).resolve().parent
FIG.mkdir(exist_ok=True)

ALPHA = 0.01
F_CAL = 0.70

# ── Style (CLAUDE.md) ────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.grid': False,
    'savefig.transparent': True, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
})

C_RED = '#A32D2D'; C_BLUE = '#185FA5'; C_TEAL = '#0F6E56'
C_PURPLE = '#534AB7'; C_GRAY = '#888780'

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_csv(RES / 'all_results.csv')
d01 = df[df['alpha'] == ALPHA].copy()

# Four representative models
SHOW = ['Chronos-Small', 'TimesFM-2.5', 'Lag-Llama', 'GJR-GARCH']
COLORS = {
    'Chronos-Small': C_RED,
    'TimesFM-2.5': C_BLUE,
    'Lag-Llama': C_PURPLE,
    'GJR-GARCH': C_TEAL,
}
DISPLAY = {
    'Chronos-Small': 'Chronos-Small',
    'TimesFM-2.5': 'TimesFM 2.5',
    'Lag-Llama': 'Lag-Llama',
    'GJR-GARCH': 'GJR-GARCH',
}

# Compute mean across 24 assets for each model
data = {}
for model in SHOW:
    mdf = d01[d01['model'] == model]
    raw_pi = mdf['pihat_raw'].mean()
    corr_pi = mdf['pihat_cp'].mean()
    raw_cov = 1 - raw_pi
    corr_cov = 1 - corr_pi
    corr_width = mdf['VaR_width'].mean()

    # Raw width: need to compute from forecast files
    # VaR_width in all_results is the corrected width.
    # Raw width = corrected width - |qV| approximately,
    # but more precisely: raw_width = mean(|VaR_raw|), corr_width = mean(|VaR_corr|)
    # Since VaR_corr = VaR_raw - qV, |VaR_corr| = |VaR_raw| + qV (if qV > 0)
    # So raw_width ≈ corr_width - qV
    mean_qv = mdf['qV'].mean()
    raw_width = corr_width - mean_qv  # approximate

    data[model] = {
        'raw_width': raw_width,
        'raw_cov': raw_cov,
        'corr_width': corr_width,
        'corr_cov': corr_cov,
    }

# ── Print data points ────────────────────────────────────────────────────
print("Data points used for Figure 3:")
print(f"{'Model':20s} {'Raw Width':>10s} {'Raw Cov':>10s} {'Corr Width':>10s} {'Corr Cov':>10s}")
for m in SHOW:
    d = data[m]
    print(f"{m:20s} {d['raw_width']:10.4f} {d['raw_cov']:10.4f} "
          f"{d['corr_width']:10.4f} {d['corr_cov']:10.4f}")


def draw_panel(ax, xlim, ylim, title, show_labels=True):
    """Draw one panel of the frontier plot."""
    # 99% target line
    ax.axhline(y=0.99, color=C_GRAY, ls='--', lw=1, zorder=1)
    if ylim[0] < 0.99 < ylim[1]:
        ax.text(xlim[0] + 0.001, 0.9905, '99\\% target', fontsize=8,
                color=C_GRAY, style='italic')

    for model in SHOW:
        d = data[model]
        c = COLORS[model]

        # Check if raw point is within axes limits
        raw_in = (xlim[0] <= d['raw_width'] <= xlim[1] and
                  ylim[0] <= d['raw_cov'] <= ylim[1])
        corr_in = (xlim[0] <= d['corr_width'] <= xlim[1] and
                   ylim[0] <= d['corr_cov'] <= ylim[1])

        if raw_in:
            ax.plot(d['raw_width'], d['raw_cov'], 'o',
                    mfc='white', mec=c, ms=10, mew=2, zorder=5)
        if corr_in:
            ax.plot(d['corr_width'], d['corr_cov'], 'o',
                    color=c, ms=10, zorder=5)

        # Arrow
        if raw_in and corr_in:
            ax.annotate('',
                        xy=(d['corr_width'], d['corr_cov']),
                        xytext=(d['raw_width'], d['raw_cov']),
                        arrowprops=dict(arrowstyle='->', lw=2, color=c))
        elif not raw_in and corr_in:
            # Arrow from edge
            clip_x = max(xlim[0], min(d['raw_width'], xlim[1]))
            clip_y = max(ylim[0], min(d['raw_cov'], ylim[1]))
            ax.annotate('',
                        xy=(d['corr_width'], d['corr_cov']),
                        xytext=(clip_x, clip_y),
                        arrowprops=dict(arrowstyle='->', lw=2, color=c,
                                        connectionstyle='arc3,rad=0'))

        # Labels — only on corrected point
        if show_labels and corr_in:
            name = DISPLAY[model]
            # Offset depends on panel scale
            is_zoomed = xlim[1] - xlim[0] < 0.03
            if model == 'GJR-GARCH':
                off = (-15, -18) if is_zoomed else (-80, -15)
            elif model == 'Chronos-Small':
                off = (10, -5) if is_zoomed else (10, -15)
            elif model == 'Lag-Llama':
                off = (10, 8) if is_zoomed else (10, 8)
            else:  # TimesFM
                off = (-15, 10) if is_zoomed else (-80, 8)
            ax.annotate(name,
                        (d['corr_width'], d['corr_cov']),
                        xytext=off, textcoords='offset points',
                        fontsize=9, fontweight='bold', color=c)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r'Mean VaR width (narrower $\rightarrow$ more efficient)', fontsize=10)
    ax.set_ylabel(r'Empirical coverage $1 - \hat{\pi}$', fontsize=10)
    ax.set_title(title, fontsize=11)


# ── Create figure ─────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

draw_panel(ax1, xlim=(0.0275, 0.0475), ylim=(0.965, 1.000),
           title='(a) Zoomed view near 99\\% target')

draw_panel(ax2, xlim=(0.005, 0.050), ylim=(0.55, 1.005),
           title='(b) Full scale including Chronos-Small')

# Legend
hollow = plt.Line2D([0], [0], marker='o', color=C_GRAY,
                     mfc='white', ms=9, mew=2, ls='')
filled = plt.Line2D([0], [0], marker='o', color=C_GRAY,
                     ms=9, ls='')
fig.legend([hollow, filled], ['Raw', 'Corrected'],
           loc='upper center', bbox_to_anchor=(0.5, -0.02),
           ncol=2, fontsize=9, frameon=False)

plt.tight_layout()
fig.savefig(FIG / 'calibration_efficiency_frontier_v2.pdf',
            dpi=300, bbox_inches='tight')
fig.savefig(FIG / 'calibration_efficiency_frontier_v2.png',
            dpi=150, bbox_inches='tight')
print(f"\nSaved: {FIG / 'calibration_efficiency_frontier_v2.pdf'}")
print(f"Saved: {FIG / 'calibration_efficiency_frontier_v2.png'}")
