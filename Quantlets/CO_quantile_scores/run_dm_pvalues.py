"""
CO_quantile_scores — Diebold-Mariano test p-values (Table 8).
Produces tab_dm_pvalues.tex and tab_dm_pvalues.csv.

Computes the DM test for pairwise Quantile Score (QS) comparisons
across 9 models (5 TSFMs + 4 parametric benchmarks), 24 assets.

Primary specification: Driscoll-Kraay panel HAC estimator
  - S_t = cross-sectional sum of QS differences at each date
  - HAC variance of S_bar via statsmodels cov_hac
    (Bartlett kernel, Andrews 1991 automatic bandwidth)
  - Harvey-Leybourne-Newbold small-sample correction (h = 1)
  - Two-sided p-values (direction varies across pairs)
  - Consistent with the panel-pooled HAC in Table 6

Supplementary: cluster-robust DM (per-asset mean, t-test, J-1 df)
  - Used only for footnote count comparison, not displayed in table

The 4 × 5 sub-table reports parametric-vs-TSFM comparisons.
The footnote reports significance counts across all 36 pairwise
comparisons under both specifications.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.sandwich_covariance import cov_hac

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'cfp_ijf_data'
QS_DIR   = DATA_DIR / 'paper_outputs' / 'qs_sequences'
OUT_DIR  = Path(__file__).resolve().parent

# The two Chronos series sampled at the checkpoint default are NOT forecasters
# and are excluded from the forecast comparison: their loss is a property of a
# sampling parameter. The comparison that concerns them -- default against the
# same checkpoint read analytically -- is reported separately at the foot of this
# script, because it is the one quantitative statement about the defect that
# belongs in a forecast-comparison framework.
ROWS = ['GJR-GARCH-t', 'GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']
COLS = ['Chronos-Small-A', 'Chronos-Mini-A', 'TimesFM-2.5',
        'Moirai-1.1', 'Moirai-2.0', 'Lag-Llama']
COL_SHORT = ['Chr-S', 'Chr-M', 'TFM', 'Moirai 1.1', 'Moirai 2.0', 'L-Llama']

ROW_LABELS = {
    'Hist-Sim': r'Hist.\ Sim.',
    'GJR-GARCH-t': r'GJR-GARCH-$t$',
}

MODEL_ORDER = ['Chronos-Small-A', 'Chronos-Mini-A', 'TimesFM-2.5',
               'Moirai-1.1', 'Moirai-2.0', 'Lag-Llama',
               'GJR-GARCH', 'GJR-GARCH-t', 'GARCH-N', 'Hist-Sim', 'EWMA']

MODEL_FILES = {
    'Chronos-Small-A': 'chronos_small_analytic_qs.parquet',
    'Chronos-Mini-A':  'chronos_mini_analytic_qs.parquet',
    'TimesFM-2.5':     'timesfm25_qs.parquet',
    'Moirai-1.1':      'moirai_qs.parquet',
    'Moirai-2.0':      'moirai2_qs.parquet',
    'Lag-Llama':       'lagllama_qs.parquet',
    'GJR-GARCH':       'gjr_garch_qs.parquet',
    'GJR-GARCH-t':     'gjr_t_qs.parquet',
    'GARCH-N':         'garch_n_qs.parquet',
    'Hist-Sim':        'hs_qs.parquet',
    'EWMA':            'ewma_qs.parquet',
}

# Read separately, for the default-versus-analytic comparison only.
DEFAULT_FILES = {
    'Chronos-Small': 'chronos_small_qs.parquet',
    'Chronos-Mini':  'chronos_mini_qs.parquet',
}

# ── Load QS sequences ────────────────────────────────────────────
qs = {m: pd.read_parquet(QS_DIR / f) for m, f in MODEL_FILES.items()}
qs.update({m: pd.read_parquet(QS_DIR / f) for m, f in DEFAULT_FILES.items()})
assets = list(qs['GJR-GARCH'].columns)
print(f'Loaded {len(qs)} QS sequences, {len(assets)} assets')

# ── DM test: Driscoll-Kraay panel HAC ────────────────────────────
# S_t = sum across assets of [QS_t(m1, a) - QS_t(m2, a)].
# HAC variance via statsmodels cov_hac (Bartlett kernel, Andrews
# 1991 AR(1) plug-in bandwidth).  HLN correction: t *= sqrt((T-1)/T).
def dm_driscoll_kraay(m1, m2):
    d1, d2 = qs[m1], qs[m2]
    common_dates = d1.index.intersection(d2.index)
    common_assets = [a for a in assets
                     if a in d1.columns and a in d2.columns]
    diffs = {a: d1.loc[common_dates, a] - d2.loc[common_dates, a]
             for a in common_assets}
    diff_df = pd.DataFrame(diffs, index=common_dates)
    S_t = diff_df.sum(axis=1).values.astype(float)
    T = len(S_t)
    S_bar = S_t.mean()
    ols_model = OLS(S_t, np.ones((T, 1))).fit()
    hac_var = cov_hac(ols_model)[0, 0]
    if hac_var <= 0:
        return 1.0
    t_stat = S_bar / np.sqrt(hac_var)
    hln = np.sqrt((T - 1) / T)
    t_adj = t_stat * hln
    return 2 * (1 - stats.t.cdf(abs(t_adj), df=T - 1))

# ── DM test: cluster-robust ─────────────────────────────────────
def dm_cluster_robust(m1, m2):
    d1, d2 = qs[m1], qs[m2]
    common_dates = d1.index.intersection(d2.index)
    common_assets = [a for a in assets
                     if a in d1.columns and a in d2.columns]
    cluster_means = []
    for a in common_assets:
        s1, s2 = d1.loc[common_dates, a], d2.loc[common_dates, a]
        mask = s1.notna() & s2.notna()
        diff = s1[mask] - s2[mask]
        if len(diff) > 0:
            cluster_means.append(diff.mean())
    cluster_means = np.array(cluster_means)
    J = len(cluster_means)
    d_bar = cluster_means.mean()
    se = np.std(cluster_means, ddof=1) / np.sqrt(J)
    if se <= 0:
        return 1.0
    t_stat = d_bar / se
    return 2 * (1 - stats.t.cdf(abs(t_stat), df=J - 1))

# ── Compute all pairwise comparisons among the forecasters ──────
pairs_36 = [(MODEL_ORDER[i], MODEL_ORDER[j])
            for i in range(len(MODEL_ORDER))
            for j in range(i + 1, len(MODEL_ORDER))]

pvals_dk = {}
pvals_cl = {}
for m1, m2 in pairs_36:
    pvals_dk[(m1, m2)] = dm_driscoll_kraay(m1, m2)
    pvals_cl[(m1, m2)] = dm_cluster_robust(m1, m2)

def get_p(store, m1, m2):
    if (m1, m2) in store:
        return store[(m1, m2)]
    return store[(m2, m1)]

# ── Significance counts ─────────────────────────────────────────
n_all = len(pairs_36)
n_sub = len(ROWS) * len(COLS)
n_dk_all = sum(1 for p in pvals_dk.values() if p < 0.05)
n_cl_all = sum(1 for p in pvals_cl.values() if p < 0.05)
n_dk_sub = sum(1 for r in ROWS for c in COLS
               if get_p(pvals_dk, r, c) < 0.05)
n_cl_sub = sum(1 for r in ROWS for c in COLS
               if get_p(pvals_cl, r, c) < 0.05)

print(f'Significant at 5%: {n_dk_sub}/{n_sub} (DK, benchmark-vs-TSFM), '
      f'{n_dk_all}/{n_all} (DK, all)')
print(f'                   {n_cl_sub}/{n_sub} (cluster, benchmark-vs-TSFM), '
      f'{n_cl_all}/{n_all} (cluster, all)')

# ── Format helpers (round-half-up) ───────────────────────────────
def fmt_p(x):
    s3 = Decimal(str(x)).quantize(Decimal('0.001'),
                                   rounding=ROUND_HALF_UP)
    if str(s3) == '0.000':
        return r'$<$.001'
    return '.' + format(s3, '.3f')[2:]

# ── Build LaTeX ──────────────────────────────────────────────────
col_header = ' & '.join(COL_SHORT)
lines = [
    r'\begin{tabular}{@{}l' + 'c' * len(COLS) + r'@{}}',
    r'\toprule',
    f'& {col_header} \\\\',
    r'\midrule',
]

for row_model in ROWS:
    label = ROW_LABELS.get(row_model, row_model)
    cells = [fmt_p(get_p(pvals_dk, row_model, c)) for c in COLS]
    line = f'{label:16s} & ' + ' & '.join(cells) + r' \\'
    lines.append(line)

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = OUT_DIR / 'tab_dm_pvalues.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)

# ── Save CSV (full sub-table) ───────────────────────────────────
sub = pd.DataFrame(
    {c: [get_p(pvals_dk, r, c) for r in ROWS] for c in COLS},
    index=ROWS)
sub.to_csv(OUT_DIR / 'tab_dm_pvalues.csv')

# ── Print all 20 cells for verification ──────────────────────────
print(f'\nAll {n_sub} DK p-values (unrounded):')
for r in ROWS:
    for c in COLS:
        p = get_p(pvals_dk, r, c)
        print(f'  {r:16s} vs {c:16s}: {p:.6f}')

# ── The configuration comparison ────────────────────────────────
# Default sampling against the same checkpoint read analytically. Same weights,
# same contexts, same dates: the only thing that differs is the code that turns
# the predictive distribution into a threshold.
print('\nDefault sampling versus analytic reading, same checkpoint:')
config_rows = []
for default, analytic in (('Chronos-Small', 'Chronos-Small-A'),
                          ('Chronos-Mini', 'Chronos-Mini-A')):
    p_dk = dm_driscoll_kraay(default, analytic)
    p_cl = dm_cluster_robust(default, analytic)
    mean_default = float(np.nanmean(qs[default].values))
    mean_analytic = float(np.nanmean(qs[analytic].values))
    config_rows.append({'default': default, 'analytic': analytic,
                        'QS_default': mean_default, 'QS_analytic': mean_analytic,
                        'p_dk': p_dk, 'p_cluster': p_cl})
    print(f'  {default:15s} {mean_default:.6e} vs {mean_analytic:.6e} '
          f'  p_DK = {p_dk:.4f}  p_cluster = {p_cl:.4f}')
pd.DataFrame(config_rows).to_csv(OUT_DIR / 'tab_dm_configuration.csv', index=False)
