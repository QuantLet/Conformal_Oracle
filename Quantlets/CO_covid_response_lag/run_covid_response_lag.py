"""
CO_covid_response_lag — run_covid_response_lag.py
=================================================
Computes per-model response lag of the rolling conformal threshold
q̂_{V,t} to the COVID-19 volatility shock on S&P 500, and
reproduces fig_covid_response.pdf.

Response lag is measured from the realised-volatility peak
(2020-03-27) to the date when q̂_{V,t} first reaches its
post-shock maximum. This captures how long the trailing-window
quantile takes to fully incorporate the new regime.

Inputs:  cfp_ijf_data/paper_outputs/tables/rolling_qv_SP500.csv
Outputs: covid_response_lags.csv, fig_covid_response.pdf/.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent
DATA = BASE / 'cfp_ijf_data' / 'paper_outputs' / 'tables'
FIG_DIR = BASE / 'figures'
OUT = SCRIPT_DIR

RVOL_PEAK = pd.Timestamp('2020-03-27')
SHOCK_START = pd.Timestamp('2020-02-20')
SHOCK_END = pd.Timestamp('2020-04-30')
PLOT_START = '2019-07-01'
PLOT_END = '2021-07-31'

TSFM_MODELS = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
               'Moirai-1.1', 'Moirai-2.0', 'Lag-Llama']
PARAM_MODELS = ['GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']
ALL_MODELS = TSFM_MODELS + PARAM_MODELS
DISPLAY = {'Chronos-Small': 'Chronos-S', 'Chronos-Mini': 'Chronos-M',
           'TimesFM-2.5': 'TimesFM 2.5', 'Moirai-1.1': 'Moirai 1.1',
           'Moirai-2.0': 'Moirai 2.0',
           'Lag-Llama': 'Lag-Llama', 'GJR-GARCH': 'GJR-GARCH',
           'GARCH-N': 'GARCH-N', 'Hist-Sim': 'Hist-Sim', 'EWMA': 'EWMA'}


def compute_lags(df):
    """Compute per-model response lag from the realised-vol peak."""
    rows = []
    for model in ALL_MODELS:
        series = df[model].dropna()
        post = series[(series.index >= RVOL_PEAK) &
                      (series.index <= '2021-01-01')]
        if len(post) == 0:
            continue
        post_max = post.max()
        first_max_date = post[post == post_max].index[0]
        lag_cal = (first_max_date - RVOL_PEAK).days
        lag_biz = len(df.loc[RVOL_PEAK:first_max_date]) - 1

        pre = series[(series.index >= '2019-11-01') &
                     (series.index < SHOCK_START)]
        pre_level = pre.mean() if len(pre) > 0 else np.nan

        shock_window = series[(series.index >= SHOCK_START) &
                              (series.index <= SHOCK_END)]
        shock_max = shock_window.max() if len(shock_window) > 0 else np.nan
        shock_max_date = (shock_window.idxmax().strftime('%Y-%m-%d')
                          if len(shock_window) > 0 else '')
        ratio = shock_max / pre_level if pre_level != 0 else np.nan

        group = 'TSFM' if model in TSFM_MODELS else 'Parametric'
        rows.append({
            'model': model, 'group': group,
            'pre_shock_mean': pre_level,
            'shock_window_max': shock_max,
            'shock_max_date': shock_max_date,
            'shock_ratio': ratio,
            'post_shock_max': post_max,
            'first_max_date': first_max_date.strftime('%Y-%m-%d'),
            'lag_calendar_days': lag_cal,
            'lag_business_days': lag_biz,
        })
    return pd.DataFrame(rows)


def make_figure(df):
    """Reproduce the COVID response figure."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': True,
    })

    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    plot_df = df.loc[PLOT_START:PLOT_END]

    FEATURED = {
        'Chronos-Small': '#C8102E',
        'TimesFM-2.5':   '#DC143C',
        'Lag-Llama':      '#003DA5',
        'GJR-GARCH':     '#228B22',
    }
    BACKGROUND = [m for m in ALL_MODELS if m not in FEATURED]

    for model in BACKGROUND:
        s = plot_df[model].dropna()
        if len(s) > 0:
            ax.plot(s.index, s.values, lw=0.8, alpha=0.2,
                    color='#888888', zorder=2)

    LINESTYLES = {
        'Chronos-Small': '-',
        'TimesFM-2.5':   '--',
        'Lag-Llama':     '-.',
        'GJR-GARCH':     ':',
    }
    feat_handles = []
    for model, color in FEATURED.items():
        s = plot_df[model].dropna()
        if len(s) > 0:
            ls = LINESTYLES[model]
            line, = ax.plot(s.index, s.values, lw=2.2, alpha=0.9,
                            color=color, ls=ls, zorder=4)
            feat_handles.append(line)

    ax2 = ax.twinx()
    rvol = plot_df['rvol'].dropna()
    ax2.fill_between(rvol.index, 0, rvol.values,
                     color='#CCCCCC', alpha=0.25, zorder=1)
    ax2.set_ylabel('Realised volatility', fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis='y', labelsize=13)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    bg_handle = Line2D([0], [0], color='#888888', lw=0.8, alpha=0.4,
                       label='Other models')
    rvol_handle = Patch(facecolor='#CCCCCC', alpha=0.4,
                        label='Realised volatility (right)')
    all_handles = feat_handles + [bg_handle, rvol_handle]
    all_labels = [DISPLAY[m] for m in FEATURED] + ['Other models',
                  'Realised vol. (right)']
    fig.legend(handles=all_handles, labels=all_labels,
               loc='lower center', bbox_to_anchor=(0.5, -0.08),
               fontsize=11, frameon=False, ncol=6,
               handletextpad=0.4, columnspacing=1.2)

    ax.axvspan(SHOCK_START, SHOCK_END,
               alpha=0.15, color='#FFB3B3', zorder=0)
    ax.annotate('COVID-19 shock', xy=(pd.Timestamp('2020-03-15'), 0),
                xytext=(pd.Timestamp('2020-03-15'), ax.get_ylim()[1] * 1.02),
                fontsize=12, color='#C8102E', ha='center', style='italic',
                va='bottom', zorder=10,
                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                          ec='none', alpha=0.8))

    ax.set_ylabel(r'$\hat{q}_{V,t}$', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    ax.tick_params(axis='both', labelsize=13)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    ax.grid(axis='y', alpha=0.2)

    SLIDE_DIR = BASE / 'ICFS 2026'
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'fig_covid_response.{ext}',
                    dpi=300, bbox_inches='tight', pad_inches=0.05)
        fig.savefig(FIG_DIR / f'fig_covid_response.{ext}',
                    dpi=300, bbox_inches='tight', pad_inches=0.05)
        fig.savefig(SLIDE_DIR / f'fig_covid_response.{ext}',
                    dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────
print("CO_covid_response_lag")
print(f"  Data: {DATA / 'rolling_qv_SP500.csv'}")
print(f"  Reference date (rvol peak): {RVOL_PEAK.date()}")
print()

df = pd.read_csv(DATA / 'rolling_qv_SP500.csv',
                 index_col=0, parse_dates=True)

lags = compute_lags(df)
lags.to_csv(OUT / 'covid_response_lags.csv', index=False)

print("Per-model response lags (calendar days from rvol peak):")
print(f"{'Model':20s} {'Group':12s} {'Lag (cal)':>10s} {'Lag (biz)':>10s} "
      f"{'First max':>12s}")
print("-" * 70)
for _, row in lags.iterrows():
    print(f"{row['model']:20s} {row['group']:12s} "
          f"{row['lag_calendar_days']:10d} {row['lag_business_days']:10d} "
          f"{row['first_max_date']:>12s}")

tsfm_mean = lags[lags['group'] == 'TSFM']['lag_calendar_days'].mean()
param_mean = lags[lags['group'] == 'Parametric']['lag_calendar_days'].mean()
print(f"\nTSFM mean lag:       {tsfm_mean:.0f} calendar days")
print(f"Parametric mean lag: {param_mean:.0f} calendar days")

tsfm_median = lags[lags['group'] == 'TSFM']['lag_calendar_days'].median()
param_median = lags[lags['group'] == 'Parametric']['lag_calendar_days'].median()
print(f"TSFM median lag:     {tsfm_median:.0f} calendar days")
print(f"Parametric median lag: {param_median:.0f} calendar days")

print(f"\nManuscript claims: TSFM = 77 days, Parametric = 161 days")
print(f"Note: 77 days matches TimesFM-2.5 and Lag-Llama individually;")
print(f"      161 days matches GJR-GARCH individually.")

make_figure(df)
print(f"\nSaved: fig_covid_response.pdf/.png")
print(f"Saved: covid_response_lags.csv")
