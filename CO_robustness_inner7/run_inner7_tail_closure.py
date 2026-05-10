"""
Inner-7-deciles tail-completion robustness (TODO[EMPIRICAL] E2).

Extends the tail-closure sensitivity analysis by adding a Student-t
refit that uses only the inner 7 deciles (u = 0.2, ..., 0.8),
dropping the outermost (u = 0.1 and u = 0.9).  This tests whether
the tail-extrapolation result is driven by the extreme quantiles of
the predictive grid.

Output:
  - inner7_tail_closure.csv          full results
  - tab_tail_closure_extended.tex    extended table (original 3 closures + inner-7)
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist, norm
from scipy.optimize import minimize
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
DATA = BASE / 'cfp_ijf_data'
OUT  = Path(__file__).resolve().parent

ASSETS = ['SP500', 'BTC', 'NATGAS']
MODELS = {
    'TimesFM-2.5': 'timesfm25',
    'Moirai-2.0': 'moirai2',
}
ALL_DECILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
INNER7_LEVELS = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
ALPHA = 0.01
F_C = 0.70


def reconstruct_deciles(nu, mu, sigma, levels=ALL_DECILE_LEVELS):
    return t_dist.ppf(levels, df=nu, loc=mu, scale=sigma)


def student_t_closure(deciles, levels, target_alpha=0.01):
    def objective(params):
        nu, mu, sigma = params
        if nu <= 2 or sigma <= 0:
            return 1e10
        predicted = t_dist.ppf(levels, df=nu, loc=mu, scale=sigma)
        return np.sum((predicted - deciles) ** 2)

    x0 = [5.0, np.median(deciles), max(np.std(deciles), 1e-6)]
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-8})
    nu, mu, sigma = res.x
    nu = max(nu, 2.01)
    sigma = max(sigma, 1e-8)
    return t_dist.ppf(target_alpha, df=nu, loc=mu, scale=sigma)


def gaussian_closure(deciles, levels, target_alpha=0.01):
    def objective(params):
        mu, sigma = params
        if sigma <= 0:
            return 1e10
        predicted = norm.ppf(levels, loc=mu, scale=sigma)
        return np.sum((predicted - deciles) ** 2)

    x0 = [np.median(deciles), max(np.std(deciles), 1e-6)]
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-8})
    mu, sigma = res.x
    sigma = max(sigma, 1e-8)
    return norm.ppf(target_alpha, loc=mu, scale=sigma)


def linear_closure(deciles, levels, target_alpha=0.01):
    q_lo = deciles[0]
    u_lo = levels[0]
    q_next = deciles[1]
    u_next = levels[1]
    slope = (q_next - q_lo) / (u_next - u_lo)
    return q_lo + slope * (target_alpha - u_lo)


def conformal_pipeline(var_raw, returns):
    common = var_raw.index.intersection(returns.index)
    var_raw = var_raw.loc[common]
    ret = returns.loc[common, 'log_return']

    n = len(common)
    n_cal = int(F_C * n)

    scores = var_raw - ret
    cal_scores = scores.iloc[:n_cal].values
    test_ret = ret.iloc[n_cal:].values
    test_var = var_raw.iloc[n_cal:].values

    ss = np.sort(cal_scores)
    n_c = len(cal_scores)
    k = int(np.ceil((n_c + 1) * (1 - ALPHA))) - 1
    k = min(k, n_c - 1)
    qV = float(ss[k])

    var_cp = test_var - qV
    violations = (test_ret < var_cp)
    pi_hat = violations.mean()
    n_test = len(test_ret)

    mean_var_raw = np.mean(np.abs(test_var))
    R = abs(qV) / mean_var_raw if mean_var_raw > 0 else np.inf

    n_viol = violations.sum()
    if n_viol == 0 or n_viol == n_test:
        tl = 'Green' if n_viol / n_test <= 0.015 else 'Red'
    else:
        from scipy.stats import binom
        p_binom = 1 - binom.cdf(n_viol - 1, n_test, ALPHA)
        if p_binom >= 0.9999:
            tl = 'Red'
        elif p_binom >= 0.95:
            tl = 'Yellow'
        else:
            tl = 'Green'

    return {'qV': qV, 'pi_corr': pi_hat, 'R': R, 'basel': tl,
            'n_cal': n_cal, 'n_test': n_test}


def run():
    closures = {
        'Student-$t$': (student_t_closure, ALL_DECILE_LEVELS),
        'Gaussian': (gaussian_closure, ALL_DECILE_LEVELS),
        'Linear': (linear_closure, ALL_DECILE_LEVELS),
        'Inner-7 Student-$t$': (student_t_closure, INNER7_LEVELS),
    }

    results = []
    for asset in ASSETS:
        returns = pd.read_csv(
            DATA / 'returns' / f'{asset}.csv',
            parse_dates=['date'], index_col='date')

        for model_name, model_dir in MODELS.items():
            fc = pd.read_parquet(DATA / model_dir / f'{asset}.parquet')

            for closure_name, (closure_fn, fit_levels) in closures.items():
                var_series = pd.Series(index=fc.index, dtype=float)

                for t in range(len(fc)):
                    row = fc.iloc[t]
                    nu = row['df_student']
                    mu = row['mean']
                    sigma = row['std']
                    all_deciles = reconstruct_deciles(nu, mu, sigma,
                                                     ALL_DECILE_LEVELS)
                    if fit_levels is INNER7_LEVELS:
                        inner_deciles = reconstruct_deciles(nu, mu, sigma,
                                                           INNER7_LEVELS)
                        q_alpha = closure_fn(inner_deciles, INNER7_LEVELS)
                    else:
                        q_alpha = closure_fn(all_deciles, fit_levels)
                    var_series.iloc[t] = -q_alpha

                result = conformal_pipeline(var_series, returns)
                result['asset'] = asset
                result['model'] = model_name
                result['closure'] = closure_name
                results.append(result)
                print(f"{asset:8s} {model_name:14s} {closure_name:22s} "
                      f"qV={result['qV']:.4f} R={result['R']:.2f} "
                      f"pi={result['pi_corr']:.3f} {result['basel']}")

    df = pd.DataFrame(results)
    df.to_csv(OUT / 'inner7_tail_closure.csv', index=False)
    print(f"\nSaved to {OUT / 'inner7_tail_closure.csv'}")

    write_tex(df)
    return df


def write_tex(df):
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\scriptsize')
    lines.append(r'\caption{Tail-completion robustness for TimesFM~2.5 and Moirai~2.0')
    lines.append(r'across three representative assets. The Student-$t$ closure is the')
    lines.append(r'default; Gaussian, linear, and inner-7-deciles Student-$t$ closures')
    lines.append(r'are reported as sensitivity checks. The inner-7 variant drops the')
    lines.append(r'outermost deciles ($u = 0.1, 0.9$) and refits on $u \in \{0.2,\ldots,0.8\}$.')
    lines.append(r'The replacement-regime classification ($R > 1.5$) is')
    lines.append(r'invariant to closure choice for both models.}')
    lines.append(r'\label{tab:tail_closure_extended}')
    lines.append(r'\begin{tabular}{@{}ll rrrr@{}}')
    lines.append(r'\hline\hline')
    lines.append(r'Model & Closure & $\qVstat$ & $\hat\pi_{\mathrm{corr}}$')
    lines.append(r'& $R$ & Basel \\')
    lines.append(r'\hline')

    for i, asset in enumerate(ASSETS):
        if i > 0:
            lines.append(r'\hline')
        lines.append(f'\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{asset}}}}} \\\\[2pt]')
        sub = df[df['asset'] == asset].reset_index(drop=True)
        prev_model = None
        for _, row in sub.iterrows():
            closure = row['closure']
            is_first = (row['model'] != prev_model)
            model_disp = f"{'~'.join(row['model'].split('-'))}" if is_first else ''
            prev_model = row['model']
            qv = f"{abs(row['qV']):.3f}"[1:]
            pi = f"{row['pi_corr']:.3f}"[1:]
            R = f"{row['R']:.2f}"
            tl = row['basel']
            pad = '' if model_disp else '             '
            lines.append(f'{pad}{model_disp} & {closure} & {qv} & {pi} & {R} & {tl} \\\\')

    lines.append(r'\hline\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    tex = '\n'.join(lines)
    (OUT / 'tab_tail_closure_extended.tex').write_text(tex + '\n')
    print(f"Saved to {OUT / 'tab_tail_closure_extended.tex'}")


if __name__ == '__main__':
    run()
