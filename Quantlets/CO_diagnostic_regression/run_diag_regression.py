"""
Diagnostic regression of ΔQS on qV_stat and pi_raw (TODO[EMPIRICAL] E4).

Reproduces the headline R²=0.822 and partial R²=61.6% from §5.1 and
emits the full coefficient table with clustered standard errors.

Output:
  - diag_regression_results.csv     coefficient estimates + SEs
  - tab_diag_regression.tex         LaTeX table for appendix
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / 'cfp_ijf_data' / 'paper_outputs' / 'tables'
OUT  = Path(__file__).resolve().parent

def rhup(val, ndigits):
    return float(Decimal(str(val)).quantize(
        Decimal(10) ** -ndigits, rounding=ROUND_HALF_UP))


def main():
    # Moirai-1.1 has been part of all_results.csv since the 17 August rebuild;
    # concatenating the legacy file would duplicate its 24 rows and weight that
    # model double in a pooled regression.
    df = pd.read_csv(DATA / 'all_results.csv')
    dup = df.duplicated(subset=['model', 'symbol', 'alpha']).sum()
    assert dup == 0, f'{dup} duplicated (model, symbol, alpha) rows'
    df = df[df['alpha'] == 0.01].copy()

    cols = ['model', 'symbol', 'qV', 'pihat_raw', 'QS_raw', 'QS_cp']
    df = df[cols].copy()
    df['delta_qs'] = df['QS_raw'] - df['QS_cp']
    df = df.dropna(subset=['delta_qs', 'qV', 'pihat_raw'])

    print(f"n = {len(df)} model-asset pairs")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Assets: {len(df['symbol'].unique())}")

    y = df['delta_qs'].values
    X = sm.add_constant(df[['qV', 'pihat_raw']].values)
    col_names = ['const', 'qV_stat', 'pi_raw']

    ols = sm.OLS(y, X).fit()
    print(f"\nOLS R² = {ols.rsquared:.6f}")

    ols_asset = sm.OLS(y, X).fit(
        cov_type='cluster', cov_kwds={'groups': df['symbol'].values})
    ols_model = sm.OLS(y, X).fit(
        cov_type='cluster', cov_kwds={'groups': df['model'].values})

    y_on_pi = sm.OLS(y, sm.add_constant(df['pihat_raw'].values)).fit()
    x_on_pi = sm.OLS(df['qV'].values,
                     sm.add_constant(df['pihat_raw'].values)).fit()
    partial_ols = sm.OLS(y_on_pi.resid, sm.add_constant(x_on_pi.resid)).fit()
    partial_r2 = partial_ols.rsquared

    print(f"Partial R² (qV_stat) = {partial_r2:.6f}")
    print(f"\nOLS coefficients: {dict(zip(col_names, ols.params))}")
    print(f"OLS SEs:          {dict(zip(col_names, ols.bse))}")
    print(f"Cluster(asset) SEs: {dict(zip(col_names, ols_asset.bse))}")
    print(f"Cluster(model) SEs: {dict(zip(col_names, ols_model.bse))}")

    # The manuscript's figures were R2 = 0.822 and partial R2 = 0.616 on the
    # ten-forecaster panel, then 0.782 and 0.534 as reprinted. Both were computed
    # with four defective series in the sample, where delta_QS is dominated by
    # the size of the defect. There is no target to match any more: what the
    # regression now reports is what the corrected panel says, and the body text
    # takes its number from here rather than the other way round.
    print(f"\n--- On the corrected panel ---")
    print(f"R2 = {rhup(ols.rsquared, 3)}   partial R2 (qV) = {rhup(partial_r2, 3)}")
    print("Superseded figures, defective panel: R2 = 0.822 / 0.782, "
          "partial R2 = 0.616 / 0.534.")

    results = []
    for i, name in enumerate(col_names):
        results.append({
            'variable': name,
            'coef': ols.params[i],
            'se_ols': ols.bse[i],
            'se_cluster_asset': ols_asset.bse[i],
            'se_cluster_model': ols_model.bse[i],
            'pval_ols': ols.pvalues[i],
            'pval_asset': ols_asset.pvalues[i],
            'pval_model': ols_model.pvalues[i],
        })
    rdf = pd.DataFrame(results)
    rdf.to_csv(OUT / 'diag_regression_results.csv', index=False)
    print(f"\nSaved results to {OUT / 'diag_regression_results.csv'}")

    # Same regression with the two top_k-truncated series dropped. Their qV is
    # two orders of magnitude larger than anything else in the panel, so leaving
    # them in makes the fit a statement about a sampling parameter rather than
    # about recalibration. Both numbers are reported; neither is hidden.
    sub = df[~df['model'].isin(['Chronos-Small', 'Chronos-Mini'])]
    y_s = sub['delta_qs'].values
    X_s = sm.add_constant(sub[['qV', 'pihat_raw']].values)
    ols_s = sm.OLS(y_s, X_s).fit()
    y_on_pi_s = sm.OLS(y_s, sm.add_constant(sub['pihat_raw'].values)).fit()
    x_on_pi_s = sm.OLS(sub['qV'].values,
                       sm.add_constant(sub['pihat_raw'].values)).fit()
    partial_s = sm.OLS(y_on_pi_s.resid, sm.add_constant(x_on_pi_s.resid)).fit().rsquared
    print(f"Excluding the two truncated series (n = {len(sub)}): "
          f"R2 = {rhup(ols_s.rsquared, 3)}   partial R2 (qV) = {rhup(partial_s, 3)}")

    write_tex(ols, ols_asset, ols_model, col_names,
              ols.rsquared, partial_r2, len(df))


def stars(p):
    if p < 0.01:
        return '^{***}'
    if p < 0.05:
        return '^{**}'
    if p < 0.10:
        return '^{*}'
    return ''


def fmt_coef(val, ndigits=4):
    return f"{rhup(val, ndigits):.{ndigits}f}"


def fmt_se(val, ndigits=4):
    return f"({rhup(val, ndigits):.{ndigits}f})"


def write_tex(ols, ols_asset, ols_model, col_names, r2, partial_r2, n):
    var_labels = {
        'const': 'Intercept',
        'qV_stat': '$\\qVstat$',
        'pi_raw': '$\\hat\\pi^{\\mathrm{raw}}$',
    }

    lines = []
    lines.append(r'\begin{tabular}{@{}l ccc@{}}')
    lines.append(r'\hline\hline')
    lines.append(r' & OLS & Cluster (asset) & Cluster (model) \\')
    lines.append(r'\hline')

    for i, name in enumerate(col_names):
        label = var_labels.get(name, name)
        coef = fmt_coef(ols.params[i])
        s_ols = stars(ols.pvalues[i])
        s_ast = stars(ols_asset.pvalues[i])
        s_mod = stars(ols_model.pvalues[i])

        lines.append(f'{label} & {coef}${s_ols}$ & {coef}${s_ast}$ & {coef}${s_mod}$ \\\\')

        se1 = fmt_se(ols.bse[i])
        se2 = fmt_se(ols_asset.bse[i])
        se3 = fmt_se(ols_model.bse[i])
        lines.append(f' & {se1} & {se2} & {se3} \\\\[3pt]')

    lines.append(r'\hline')
    r2_s = f"{rhup(r2, 3):.3f}"
    pr2_s = f"{rhup(partial_r2, 3):.3f}"
    lines.append(f'$R^{{2}}$ & {r2_s} & {r2_s} & {r2_s} \\\\')
    lines.append(f'Partial $R^{{2}}$ ($\\qVstat$) & {pr2_s} & {pr2_s} & {pr2_s} \\\\')
    lines.append(f'$n$ & {n} & {n} & {n} \\\\')
    lines.append(r'\hline\hline')
    lines.append(r'\end{tabular}')

    tex = '\n'.join(lines)
    (OUT / 'tab_diag_regression.tex').write_text(tex + '\n')
    print(f"Saved to {OUT / 'tab_diag_regression.tex'}")


if __name__ == '__main__':
    main()
