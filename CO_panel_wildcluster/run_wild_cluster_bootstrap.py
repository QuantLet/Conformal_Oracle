"""
Wild-cluster bootstrap of panel-pooled Kupiec and DM statistics
(TODO[EMPIRICAL] E3).

Implements Rademacher-weighted cluster bootstrap (B = 999, J = 24
asset clusters) for:
  1. Panel-pooled Kupiec LR statistic — tests H0: π = α for each model.
  2. Panel DM t-statistic — tests H0: equal predictive ability
     (GJR-GARCH vs each TSFM, using QS differences).

Under the wild-cluster bootstrap, the cluster-level statistic is
multiplied by w_j ∈ {−1, +1} (Rademacher) before re-aggregation.
The bootstrap p-value is the proportion of |t*_b| ≥ |t_obs|.

Output:
  - wild_cluster_kupiec.csv              per-model bootstrap results
  - wild_cluster_dm.csv                  per-pair bootstrap results
  - tab_panel_wildcluster_kupiec.tex     LaTeX table for Kupiec
  - tab_panel_wildcluster_dm.tex         LaTeX table for DM
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2, norm
from decimal import Decimal, ROUND_HALF_UP

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / 'cfp_ijf_data'
VIOL_DIR = DATA / 'paper_outputs' / 'violation_sequences'
QS_DIR   = DATA / 'paper_outputs' / 'qs_sequences'
OUT  = Path(__file__).resolve().parent

ALPHA = 0.01
B = 999
SEED = 42

MODELS = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
          'Moirai-2.0', 'Lag-Llama',
          'GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']

MODEL_FILES = {
    'Chronos-Small': 'chronos_small',
    'Chronos-Mini':  'chronos_mini',
    'TimesFM-2.5':   'timesfm25',
    'Moirai-2.0':    'moirai2',
    'Lag-Llama':     'lagllama',
    'GJR-GARCH':     'gjr_garch',
    'GARCH-N':       'garch_n',
    'Hist-Sim':      'hs',
    'EWMA':          'ewma',
}

DM_PAIRS = [(ref, tsfm)
            for ref in ['GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']
            for tsfm in ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
                         'Moirai-2.0', 'Lag-Llama']]


def load_violations(model):
    return pd.read_parquet(VIOL_DIR / f'{MODEL_FILES[model]}_violations.parquet')


def load_qs(model):
    return pd.read_parquet(QS_DIR / f'{MODEL_FILES[model]}_qs.parquet')


def kupiec_lr(x, n, alpha=ALPHA):
    if n == 0:
        return 0.0
    pi = x / n
    if pi == 0:
        return 2 * n * np.log(1 / (1 - alpha))
    if pi == 1:
        return 2 * n * np.log(alpha)
    return 2 * (x * np.log(pi / alpha) +
                (n - x) * np.log((1 - pi) / (1 - alpha)))


def wild_cluster_kupiec(viol_df, rng):
    assets = list(viol_df.columns)
    J = len(assets)

    per_asset_x = {a: viol_df[a].sum() for a in assets}
    per_asset_n = {a: viol_df[a].notna().sum() for a in assets}

    total_x = sum(per_asset_x.values())
    total_n = sum(per_asset_n.values())
    lr_obs = kupiec_lr(total_x, total_n)

    pi_pool = total_x / total_n
    centered_x = {a: per_asset_x[a] - ALPHA * per_asset_n[a]
                  for a in assets}

    boot_count = 0
    for _ in range(B):
        w = rng.choice([-1, 1], size=J)
        x_star = sum(ALPHA * per_asset_n[a] + w[j] * centered_x[a]
                     for j, a in enumerate(assets))
        x_star = max(0, min(x_star, total_n))
        lr_star = kupiec_lr(x_star, total_n)
        if lr_star >= lr_obs:
            boot_count += 1

    p_boot = (boot_count + 1) / (B + 1)
    p_asymp = 1 - chi2.cdf(lr_obs, 1)

    return {
        'total_x': int(total_x),
        'total_n': int(total_n),
        'pi_pooled': pi_pool,
        'LR_obs': lr_obs,
        'p_asymp': p_asymp,
        'p_boot': p_boot,
    }


def wild_cluster_dm(qs1_df, qs2_df, rng):
    assets = sorted(set(qs1_df.columns) & set(qs2_df.columns))
    J = len(assets)
    common_dates = qs1_df.index.intersection(qs2_df.index)

    cluster_means = []
    for a in assets:
        d = qs1_df.loc[common_dates, a] - qs2_df.loc[common_dates, a]
        mask = d.notna()
        if mask.sum() > 0:
            cluster_means.append(d[mask].mean())
        else:
            cluster_means.append(0.0)
    cluster_means = np.array(cluster_means)

    d_bar = cluster_means.mean()
    se = np.std(cluster_means, ddof=1) / np.sqrt(J)
    if se <= 0:
        return {'t_obs': 0.0, 'p_asymp': 1.0, 'p_boot': 1.0}
    t_obs = d_bar / se

    centered = cluster_means - d_bar
    boot_count = 0
    for _ in range(B):
        w = rng.choice([-1, 1], size=J)
        d_star = (centered * w).mean()
        se_star = np.std(centered * w, ddof=1) / np.sqrt(J)
        if se_star > 0:
            t_star = d_star / se_star
        else:
            t_star = 0.0
        if abs(t_star) >= abs(t_obs):
            boot_count += 1

    from scipy.stats import t as t_dist
    p_asymp = 2 * (1 - t_dist.cdf(abs(t_obs), df=J - 1))
    p_boot = (boot_count + 1) / (B + 1)

    return {'t_obs': t_obs, 'p_asymp': p_asymp, 'p_boot': p_boot,
            'd_bar': d_bar, 'se': se}


def rhup(val, ndigits):
    return float(Decimal(str(val)).quantize(
        Decimal(10) ** -ndigits, rounding=ROUND_HALF_UP))


def fmt_p(p):
    r = rhup(p, 3)
    if r == 0.0:
        r4 = rhup(p, 4)
        return f'{r4:.4f}'[1:]
    return f'{r:.3f}'[1:]


def main():
    rng = np.random.default_rng(SEED)

    print("=" * 70)
    print("Part 1: Wild-cluster bootstrap of panel-pooled Kupiec LR")
    print("=" * 70)

    kupiec_rows = []
    for model in MODELS:
        viol_df = load_violations(model)
        res = wild_cluster_kupiec(viol_df, rng)
        res['model'] = model
        kupiec_rows.append(res)
        print(f"  {model:16s}  π̂={res['pi_pooled']:.4f}  "
              f"LR={res['LR_obs']:.2f}  "
              f"p_asymp={res['p_asymp']:.4f}  p_boot={res['p_boot']:.4f}")

    kdf = pd.DataFrame(kupiec_rows)
    kdf.to_csv(OUT / 'wild_cluster_kupiec.csv', index=False)

    print("\n" + "=" * 70)
    print("Part 2: Wild-cluster bootstrap of panel DM t-statistic")
    print("=" * 70)

    qs_cache = {}
    dm_rows = []
    for m1, m2 in DM_PAIRS:
        if m1 not in qs_cache:
            qs_cache[m1] = load_qs(m1)
        if m2 not in qs_cache:
            qs_cache[m2] = load_qs(m2)
        res = wild_cluster_dm(qs_cache[m1], qs_cache[m2], rng)
        res['model_ref'] = m1
        res['model_tsfm'] = m2
        dm_rows.append(res)
        print(f"  {m1:12s} vs {m2:16s}  t={res['t_obs']:+.3f}  "
              f"p_asymp={res['p_asymp']:.4f}  p_boot={res['p_boot']:.4f}")

    ddf = pd.DataFrame(dm_rows)
    ddf.to_csv(OUT / 'wild_cluster_dm.csv', index=False)

    write_kupiec_tex(kdf)
    write_dm_tex(ddf)
    print("\nDone.")


def write_kupiec_tex(kdf):
    lines = []
    lines.append(r'\begin{tabular}{@{}lrrrr@{}}')
    lines.append(r'\toprule')
    lines.append(r'Model & $N_{\text{panel}}$ & Corr.\ $\hat\pi$ '
                 r'& $p_{\text{Kup}}$ & $p_{\text{boot}}$ \\')
    lines.append(r'\midrule')

    for i, row in kdf.iterrows():
        model = row['model'].replace('-', '~') if '-' in row['model'] else row['model']
        if row['model'] == 'Hist-Sim':
            model = r'Hist.\ Sim.'
        n_fmt = f"{int(row['total_n']):,}".replace(',', r'{,}')
        pi_fmt = f"{rhup(row['pi_pooled'], 4):.4f}"[1:]
        p_a = fmt_p(row['p_asymp'])
        p_b = fmt_p(row['p_boot'])
        lines.append(f'{model} & {n_fmt} & {pi_fmt} & {p_a} & {p_b} \\\\')
        if i == 4:
            lines.append(r'\midrule')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    tex = '\n'.join(lines)
    (OUT / 'tab_panel_wildcluster_kupiec.tex').write_text(tex + '\n')
    print(f"Saved {OUT / 'tab_panel_wildcluster_kupiec.tex'}")


def write_dm_tex(ddf):
    refs = ['GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']
    tsfms = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
             'Moirai-2.0', 'Lag-Llama']
    tsfm_short = ['Chr-S', 'Chr-M', 'TFM', 'Moirai', 'L-Llama']

    lines = []
    hdr = ' & '.join(tsfm_short)
    lines.append(r'\begin{tabular}{@{}l' + 'c' * len(tsfms) + r'@{}}')
    lines.append(r'\toprule')
    lines.append(f' & {hdr} \\\\')
    lines.append(r'\midrule')

    for ref in refs:
        ref_disp = ref.replace('-', '~')
        if ref == 'Hist-Sim':
            ref_disp = r'Hist.\ Sim.'
        cells = []
        for tsfm in tsfms:
            row = ddf[(ddf['model_ref'] == ref) &
                      (ddf['model_tsfm'] == tsfm)].iloc[0]
            p_b = rhup(row['p_boot'], 3)
            p_str = fmt_p(row['p_boot'])
            if p_b < 0.01:
                cells.append(f'\\textbf{{{p_str}}}')
            elif p_b < 0.05:
                cells.append(f'\\textit{{{p_str}}}')
            else:
                cells.append(p_str)
        lines.append(f"{ref_disp} & {' & '.join(cells)} \\\\")

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    tex = '\n'.join(lines)
    (OUT / 'tab_panel_wildcluster_dm.tex').write_text(tex + '\n')
    print(f"Saved {OUT / 'tab_panel_wildcluster_dm.tex'}")


if __name__ == '__main__':
    main()
