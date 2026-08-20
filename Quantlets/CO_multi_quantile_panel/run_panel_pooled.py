"""
CO_multi_quantile_panel — Panel-pooled backtest at alpha=0.01.
Produces tab_panel_pooled.tex  (Table 6 in the paper).

Every cell is computed from reproducible inputs:
  - Violation indicators: forecast parquets + returns + qV
  - N_panel, violations, pi_pooled: from violation counts
  - HAC SE: Driscoll-Kraay panel estimator
      Kernel:    Bartlett (Newey-West 1987)
      Bandwidth: Andrews (1991) AR(1) automatic plug-in
      Panel:     NW on S_t (cross-sectional sum of violations at
                 each date), scaled by T/N to obtain SE(pi_pooled)
  - p_Kupiec:   Kupiec (1995) LR test, chi2(1)
  - p_cluster:  two-sided z-test with cluster SE =
                 sqrt(var(pihat_i, ddof=1) / J), J = 24 assets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from scipy.stats import chi2, norm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'cfp_ijf_data'
RES_DIR  = DATA_DIR / 'paper_outputs' / 'tables'
RET_DIR  = DATA_DIR / 'returns'
OUT_DIR  = Path(__file__).resolve().parent

# The model set and the file keys come from cfp_config, so this table cannot
# describe a different panel from Table 1. It used to carry its own dictionary of
# ten, which is how it stayed a ten-forecaster table after the panel became
# thirteen.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cfp_config import MODELS  # noqa: E402

MODEL_ORDER = list(MODELS)
ALPHA = 0.01
VIOL_DIR = DATA_DIR / 'paper_outputs' / 'violation_sequences'


def viol_key(model):
    subdir, suffix = MODELS[model]
    return suffix if suffix else subdir


# ── Load all_results ─────────────────────────────────────────────
# Moirai-1.1 is in all_results.csv since the 17 August rebuild; concatenating
# the legacy file would duplicate its rows.
ar = pd.read_csv(RES_DIR / 'all_results.csv')
dup = ar.duplicated(subset=['model', 'symbol', 'alpha']).sum()
assert dup == 0, f'{dup} duplicated (model, symbol, alpha) rows in all_results.csv'
d01 = ar[ar['alpha'] == ALPHA].copy()

# ── Reconstruct violations & compute statistics ──────────────────
print('Computing panel-pooled backtest (Driscoll-Kraay HAC SE) ...')
rows = []

for model in MODEL_ORDER:
    sub = d01[d01['model'] == model]
    assets = sorted(sub['symbol'].unique())
    J = len(assets)

    # Read the committed violation sequences rather than reconstructing them.
    # This script used to rebuild the conformal correction itself from the
    # forecast parquets -- a second implementation of the evaluation, which is
    # exactly how two tables come to disagree. scripts/build_qs_sequences.py
    # writes these and verifies that they reproduce all_results.csv.
    seq = pd.read_parquet(VIOL_DIR / f'{viol_key(model)}_violations.parquet')
    viols = {}
    for sym in assets:
        v = seq[sym].dropna().astype(int)
        expected = int(sub[sub['symbol'] == sym].iloc[0]['n_test'])
        assert len(v) == expected, (
            f'{model}/{sym}: sequence has {len(v)} test days, '
            f'all_results.csv says {expected}')
        viols[sym] = v

    n_panel    = sum(len(v) for v in viols.values())
    total_viol = int(sum(v.sum() for v in viols.values()))
    pi_pooled  = total_viol / n_panel

    # Kupiec LR
    lr = -2 * (total_viol * np.log(ALPHA / pi_pooled)
               + (n_panel - total_viol) * np.log((1 - ALPHA) / (1 - pi_pooled)))
    p_kupiec = 1 - chi2.cdf(lr, 1)

    # Driscoll-Kraay HAC SE
    all_dates = sorted(set().union(*(v.index for v in viols.values())))
    panel_df = pd.DataFrame(index=all_dates)
    for sym, v in viols.items():
        panel_df[sym] = v
    S_t = panel_df.sum(axis=1).values.astype(float)
    T = len(S_t)
    ols = OLS(S_t, np.ones((T, 1))).fit()
    hac_se = np.sqrt(cov_hac(ols)[0, 0]) * T / n_panel

    # Cluster-robust z and p
    pihat_assets = np.array([
        sub[sub['symbol'] == sym].iloc[0]['pihat_cp'] for sym in assets])
    cluster_se = np.sqrt(np.var(pihat_assets, ddof=1) / J)
    z_cluster = (pi_pooled - ALPHA) / cluster_se
    p_cluster = 2 * (1 - norm.cdf(abs(z_cluster)))

    rows.append({
        'model': model, 'N_panel': n_panel,
        'total_viol': total_viol, 'pi_pooled': pi_pooled,
        'HAC_SE': hac_se, 'p_kupiec': p_kupiec,
        'z_cluster': z_cluster, 'p_cluster': p_cluster,
    })

    print(f'  {model:16s}  N={n_panel}  V={total_viol}  '
          f'pi={pi_pooled:.6f}  HAC_SE={hac_se:.6f}  '
          f'p_kup={p_kupiec:.4f}  p_cl={p_cluster:.4f}')

result = pd.DataFrame(rows).set_index('model')

# ── Save CSV ─────────────────────────────────────────────────────
result.to_csv(OUT_DIR / 'tab_panel_pooled.csv')

# ── Format helpers (round-half-up) ───────────────────────────────
def _rhu(x, dp):
    d = Decimal(str(x)).quantize(Decimal(10) ** -dp, rounding=ROUND_HALF_UP)
    return format(d, f'.{dp}f')

def fmt_n(n):
    return f'{int(n):,}'.replace(',', '{,}')

def fmt_pi(x):
    return '.' + _rhu(x, 4)[2:]

def fmt_se(x):
    return '.' + _rhu(x, 4)[2:]

def fmt_p(x):
    s3 = _rhu(x, 3)
    if s3 == '0.000':
        return '.' + _rhu(x, 4)[2:]
    return '.' + s3[2:]

# ── Build LaTeX ──────────────────────────────────────────────────
lines = [
    r'\begin{tabular}{@{}lrrrrrr@{}}',
    r'\toprule',
    r'Model & $N_{\text{panel}}$ & Violations',
    r'& Corr.\ $\hat\pi$ & HAC SE',
    r'& $p_{\text{Kup}}$',
    r'& $p_{\text{cluster}}$ \\',
    r'\midrule',
]

for i, model in enumerate(MODEL_ORDER):
    r = result.loc[model]

    label = model
    if model == 'TimesFM-2.5':
        label = 'TimesFM 2.5'
    elif model == 'Moirai-1.1':
        label = 'Moirai 1.1'
    elif model == 'Moirai-2.0':
        label = 'Moirai 2.0'
    elif model == 'Hist-Sim':
        label = r'Hist.\ Sim.'

    line = (f'{label} & {fmt_n(r["N_panel"])} & {int(r["total_viol"])}'
            f' & {fmt_pi(r["pi_pooled"])}\n'
            f'& {fmt_se(r["HAC_SE"])} & {fmt_p(r["p_kupiec"])}'
            f' & {fmt_p(r["p_cluster"])} \\\\')
    lines.append(line)
    if i == 5:
        lines.append(r'\midrule')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = OUT_DIR / 'tab_panel_pooled.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)
