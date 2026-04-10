import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style (CLAUDE.md conventions) ──────────────────────────────────────────
plt.rcParams.update({
    "savefig.transparent": True,
    "axes.grid": False,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, ax = plt.subplots(figsize=(8, 5))

# Data from Table 4:
# raw pi_hat -> raw coverage = 1 - pi_hat
# corrected pi_hat -> corrected coverage = 1 - pi_hat
# VaR width from Table 4

# Data from all_results.csv (mean across 24 assets, alpha=0.01)
models = {
    'TimesFM 2.5': {
        'raw_width': 0.0338, 'raw_cov': 1 - 0.014,     # 0.986
        'corr_width': 0.0366, 'corr_cov': 1 - 0.010,   # 0.990
        'color': '#c41e3a',  # dark red
    },
    'Lag-Llama': {
        'raw_width': 0.0301, 'raw_cov': 1 - 0.030,     # 0.970
        'corr_width': 0.0409, 'corr_cov': 1 - 0.010,   # 0.990
        'color': '#6a0dad',  # purple
    },
    'GJR-GARCH': {
        'raw_width': 0.0457, 'raw_cov': 1 - 0.004,     # 0.996
        'corr_width': 0.0392, 'corr_cov': 1 - 0.011,   # 0.989
        'color': '#1f77b4',  # blue
    },
    'Chronos-Small': {
        'raw_width': 0.0027, 'raw_cov': 1 - 0.388,     # 0.612
        'corr_width': 0.0405, 'corr_cov': 1 - 0.011,   # 0.989
        'color': '#8b0000',  # dark red
    },
}

# --- Main plot ---
for name, d in models.items():
    if name == 'Chronos-Small':
        # Only corrected point in main plot
        ax.plot(d['corr_width'], d['corr_cov'], 'o',
                color=d['color'], ms=11, zorder=5)
        ax.annotate(name, (d['corr_width'], d['corr_cov']),
                    xytext=(12, -5), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    color=d['color'])
    else:
        # Raw (hollow)
        ax.plot(d['raw_width'], d['raw_cov'], 'o',
                mfc='white', mec=d['color'], ms=11, mew=2,
                zorder=5)
        # Corrected (filled)
        ax.plot(d['corr_width'], d['corr_cov'], 'o',
                color=d['color'], ms=11, zorder=5)
        # Arrow
        ax.annotate('',
                    xy=(d['corr_width'], d['corr_cov']),
                    xytext=(d['raw_width'], d['raw_cov']),
                    arrowprops=dict(arrowstyle='->', lw=2,
                                   color=d['color']))

# Labels — manually positioned to avoid overlap
ax.annotate('TimesFM 2.5',
            (models['TimesFM 2.5']['corr_width'],
             models['TimesFM 2.5']['corr_cov']),
            xytext=(-15, 10), textcoords='offset points',
            fontsize=10, fontweight='bold',
            color=models['TimesFM 2.5']['color'])

ax.annotate('Lag-Llama',
            (models['Lag-Llama']['corr_width'],
             models['Lag-Llama']['corr_cov']),
            xytext=(12, 5), textcoords='offset points',
            fontsize=10, fontweight='bold',
            color=models['Lag-Llama']['color'])

ax.annotate('GJR-GARCH',
            (models['GJR-GARCH']['raw_width'],
             models['GJR-GARCH']['raw_cov']),
            xytext=(10, 3), textcoords='offset points',
            fontsize=10, fontweight='bold',
            color=models['GJR-GARCH']['color'])

# 99% target
ax.axhline(y=0.99, color='grey', ls='--', lw=1, zorder=1)
ax.text(0.0277, 0.9905, '99% target', fontsize=9,
        color='grey', style='italic')

# Axes
ax.set_xlabel(r'Mean VaR width (narrower $\rightarrow$ more efficient)',
              fontsize=11)
ax.set_ylabel(r'Empirical coverage $1 - \hat{\pi}$',
              fontsize=11)
ax.set_xlim(0.027, 0.048)
ax.set_ylim(0.963, 1.001)

# Legend
hollow = plt.Line2D([0], [0], marker='o', color='grey',
                     mfc='white', ms=9, mew=2, ls='')
filled = plt.Line2D([0], [0], marker='o', color='grey',
                     ms=9, ls='')
ax.legend([hollow, filled], ['Raw', 'Corrected'],
          loc='upper center', bbox_to_anchor=(0.5, -0.1),
          ncol=2, fontsize=9, frameon=False)

# --- Inset: Chronos-Small full scale ---
axins = ax.inset_axes([0.02, 0.02, 0.25, 0.35])
d = models['Chronos-Small']
axins.plot(d['raw_width'], d['raw_cov'], 'o',
           mfc='white', mec=d['color'], ms=7, mew=1.5)
axins.plot(d['corr_width'], d['corr_cov'], 'o',
           color=d['color'], ms=7)
axins.annotate('',
               xy=(d['corr_width'], d['corr_cov']),
               xytext=(d['raw_width'], d['raw_cov']),
               arrowprops=dict(arrowstyle='->', lw=1.5,
                               color=d['color']))
axins.axhline(y=0.99, color='grey', ls='--', lw=0.8)
axins.set_xlim(0.0, 0.05)
axins.set_ylim(0.55, 1.02)
axins.text(0.5, 0.97, 'Chronos-Small\nraw: $\\hat{\\pi}=38.8\\%$',
           transform=axins.transAxes, fontsize=7,
           color=d['color'], fontweight='bold',
           ha='center', va='top')
axins.tick_params(labelsize=6)
axins.spines['top'].set_visible(False)
axins.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/fig_frontier_killer.pdf', dpi=300,
            bbox_inches='tight')
plt.savefig('figures/fig_frontier_killer.png', dpi=150,
            bbox_inches='tight')
print("Saved figures/fig_frontier_killer.pdf")
print("Saved figures/fig_frontier_killer.png")
