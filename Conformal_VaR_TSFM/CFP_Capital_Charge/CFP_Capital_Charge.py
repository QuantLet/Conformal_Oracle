"""
Capital Charge Comparison
=========================
Computes cumulative capital charges for S&P 500 under three configurations:
  (i)   Raw Lag-Llama (Yellow zone, k=3.65)
  (ii)  Conformally corrected Lag-Llama (Green zone, k=3.00)
  (iii) Conformally corrected GJR-GARCH (Green zone, k=3.00)

Produces:
  - figures/capital_charge_cumulative.pdf
  - figures/capital_charge_cumulative.png
  - LaTeX-ready summary sentence
"""

import numpy as np
import pandas as pd
from pathlib import Path
from math import ceil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / 'cfp_ijf_data'
FIG  = Path(__file__).resolve().parent
FIG.mkdir(exist_ok=True)

ALPHA = 0.01
F_CAL = 0.70
SYMBOL = 'SP500'

K_YELLOW = 3.65  # Basel Yellow zone multiplier
K_GREEN  = 3.00  # Basel Green zone multiplier

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.grid': False,
    'savefig.transparent': True, 'savefig.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load_and_split(subdir, suffix=None):
    """Load returns and forecasts, apply 70/30 cal/test split."""
    ret = pd.read_csv(DATA / 'returns' / f'{SYMBOL}.csv',
                      index_col=0, parse_dates=True)
    ret.columns = ['r']

    if suffix:
        fc = pd.read_parquet(DATA / subdir / f'{SYMBOL}_{suffix}.parquet')
    else:
        fc = pd.read_parquet(DATA / subdir / f'{SYMBOL}.parquet')

    common = ret.index.intersection(fc.index)
    ret = ret.loc[common]
    fc = fc.loc[common]

    N = len(common)
    n_cal = int(N * F_CAL)

    return (ret.iloc[:n_cal], ret.iloc[n_cal:],
            fc.iloc[:n_cal], fc.iloc[n_cal:])


def conformal_correct_var(r_cal, fc_cal, fc_test):
    """Apply one-sided conformal VaR correction."""
    var_cal = fc_cal[f'VaR_{ALPHA}'].values
    r_cal_v = r_cal['r'].values
    s_V = var_cal - r_cal_v
    k = ceil((len(s_V) + 1) * (1 - ALPHA))
    k = min(k, len(s_V))
    q_hat_V = np.sort(s_V)[k - 1]

    var_test_corr = fc_test[f'VaR_{ALPHA}'].values - q_hat_V
    return var_test_corr, q_hat_V


# ── Load data ─────────────────────────────────────────────────────────────
# Lag-Llama
r_cal_ll, r_test_ll, fc_cal_ll, fc_test_ll = load_and_split('lagllama')
var_raw_ll = fc_test_ll[f'VaR_{ALPHA}'].values
var_corr_ll, qv_ll = conformal_correct_var(r_cal_ll, fc_cal_ll, fc_test_ll)
dates_test = r_test_ll.index

# GJR-GARCH
r_cal_gj, r_test_gj, fc_cal_gj, fc_test_gj = load_and_split('benchmarks', 'gjr_garch')
var_corr_gj, qv_gj = conformal_correct_var(r_cal_gj, fc_cal_gj, fc_test_gj)

# Align test periods (may differ slightly in length)
# Use the Lag-Llama test dates as primary
dates_gj = r_test_gj.index
common_test = dates_test.intersection(dates_gj)

# Re-index to common dates
idx_ll = [i for i, d in enumerate(dates_test) if d in common_test]
idx_gj = [i for i, d in enumerate(dates_gj) if d in common_test]

var_raw_ll_c = var_raw_ll[idx_ll]
var_corr_ll_c = var_corr_ll[idx_ll]
var_corr_gj_c = var_corr_gj[idx_gj]
dates_common = common_test.sort_values()

N_test = len(dates_common)

# ── Compute capital charges ───────────────────────────────────────────────
# The bank must hold the corrected (wider) VaR for compliance.
# The question is: what multiplier applies?
# Yellow (uncorrected): k=3.65 × |VaR_corr|
# Green (corrected):    k=3.00 × |VaR_corr|
# The saving is from the zone reclassification, at the SAME VaR level.
charge_yellow_ll = K_YELLOW * np.abs(var_corr_ll_c)  # Yellow on corrected VaR
charge_green_ll  = K_GREEN  * np.abs(var_corr_ll_c)  # Green on corrected VaR
charge_green_gj  = K_GREEN  * np.abs(var_corr_gj_c)  # Green on corrected GJR VaR

cumul_yellow_ll = np.cumsum(charge_yellow_ll)
cumul_green_ll  = np.cumsum(charge_green_ll)
cumul_green_gj  = np.cumsum(charge_green_gj)

saving_pct = (cumul_yellow_ll[-1] - cumul_green_ll[-1]) / cumul_yellow_ll[-1] * 100

# ── Print results ─────────────────────────────────────────────────────────
print(f"Asset: {SYMBOL}")
print(f"Test period: {dates_common[0].strftime('%Y-%m-%d')} to {dates_common[-1].strftime('%Y-%m-%d')}")
print(f"Number of test trading days: {N_test}")
print()
var_corr_abs_ll = np.mean(np.abs(var_corr_ll_c))
var_corr_abs_gj = np.mean(np.abs(var_corr_gj_c))

print(f"Lag-Llama corrected |VaR| (mean daily): {var_corr_abs_ll:.6f}")
print(f"  If Yellow (k={K_YELLOW}): daily charge = {K_YELLOW * var_corr_abs_ll:.6f}")
print(f"  If Green  (k={K_GREEN}):  daily charge = {K_GREEN * var_corr_abs_ll:.6f}")
print(f"  Cumulative (Yellow): {cumul_yellow_ll[-1]:.4f}")
print(f"  Cumulative (Green):  {cumul_green_ll[-1]:.4f}")
print(f"  q_hat_V:             {qv_ll:.6f}")
print()
print(f"GJR-GARCH corrected |VaR| (mean daily): {var_corr_abs_gj:.6f}")
print(f"  Cumulative (Green):  {cumul_green_gj[-1]:.4f}")
print(f"  q_hat_V:             {qv_gj:.6f}")
print()
print(f"CAPITAL SAVING from zone reclassification: {saving_pct:.1f}%")
print(f"  (= {K_YELLOW - K_GREEN:.2f} / {K_YELLOW} = {(K_YELLOW-K_GREEN)/K_YELLOW*100:.1f}% "
      f"of charge at corrected VaR level)")
print()

# LaTeX sentence
print("% ── LaTeX sentence ──────────────────────────────────────────")
print(f"Over the {N_test}-day test window, the cumulative capital saving from "
      f"reclassifying Lag-Llama from Yellow to Green amounts to "
      f"{saving_pct:.1f}\\% of the raw cumulative charge.")

# ── Generate figure ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(dates_common, cumul_yellow_ll, color='#A32D2D', ls='--', lw=1.5,
        label=f'Lag-Llama Yellow ($k={K_YELLOW}$)')
ax.plot(dates_common, cumul_green_ll, color='#185FA5', ls='-', lw=1.5,
        label=f'Lag-Llama Green ($k={K_GREEN}$)')
ax.plot(dates_common, cumul_green_gj, color='#0F6E56', ls='-', lw=1.5,
        label=f'GJR-GARCH Green ($k={K_GREEN}$)')

# COVID shading
covid_start = pd.Timestamp('2020-02-01')
covid_end = pd.Timestamp('2020-04-30')
ax.axvspan(covid_start, covid_end, alpha=0.15, color='grey',
           label='COVID-19 (Feb--Apr 2020)')

ax.set_xlabel('Date')
ax.set_ylabel('Cumulative capital charge')
ax.set_title('Cumulative Capital Charge: S\\&P 500', fontsize=12)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=2, fontsize=9, frameon=False)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
fig.savefig(FIG / 'capital_charge_cumulative.pdf', dpi=300, bbox_inches='tight')
fig.savefig(FIG / 'capital_charge_cumulative.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {FIG / 'capital_charge_cumulative.pdf'}")
print(f"Saved: {FIG / 'capital_charge_cumulative.png'}")
