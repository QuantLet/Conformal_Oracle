"""
Adaptive Conformal Inference (ACI) Baseline
============================================
Implements rolling-window ACI (Gibbs & Candès, 2021) for comparison
with the static/rolling conformal method in Table 12 (tab:baselines).

For each model × asset:
  1. Use SAME test split and forecasts as Table 12
  2. Select gamma on calibration data only
  3. Run ACI on test set with rolling W=250 quantile computation
  4. Compute identical metrics: violation rate, Kupiec, QS, width, Green %
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm, chi2
from math import ceil
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'cfp_ijf_data'

ALPHA = 0.01
F_CAL = 0.70
W = 250          # rolling window size (Basel convention)
GAMMAS = [0.001, 0.005, 0.01]  # candidate learning rates

SYMBOLS = ['SP500', 'STOXX', 'GDAXI', 'FCHI', 'FTSE100', 'ICLN',
           'NIKKEI', 'HSI', 'BOVESPA', 'NIFTY', 'ASX200', 'CBU0',
           'TLT', 'IBGL', 'DJCI', 'GOLD', 'WTI', 'NATGAS',
           'BTC', 'ETH', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

MODELS = {
    'Chronos-Small': ('chronos_small', None),
    'Chronos-Mini':  ('chronos_mini',  None),
    'TimesFM-2.5':   ('timesfm25',     None),
    'Moirai-2.0':    ('moirai2',       None),
    'Lag-Llama':     ('lagllama',      None),
    'GJR-GARCH':     ('benchmarks',    'gjr_garch'),
    'GARCH-N':       ('benchmarks',    'garch_n'),
    'Hist-Sim':      ('benchmarks',    'hs'),
    'EWMA':          ('benchmarks',    'ewma'),
}


def load_data(model_key, symbol):
    """Load returns and VaR forecasts for a model-asset pair."""
    subdir, suffix = MODELS[model_key]
    ret = pd.read_csv(DATA / 'returns' / f'{symbol}.csv',
                      index_col=0, parse_dates=True)
    ret.columns = ['r']
    if suffix is None:
        fc = pd.read_parquet(DATA / subdir / f'{symbol}.parquet')
    else:
        fc = pd.read_parquet(DATA / subdir / f'{symbol}_{suffix}.parquet')
    common = ret.index.intersection(fc.index)
    ret = ret.loc[common]
    fc = fc.loc[common]
    var_col = f'VaR_{ALPHA}'
    mask = fc[var_col].notna()
    ret = ret.loc[mask]
    fc = fc.loc[mask]
    return ret['r'].values, fc[var_col].values


def kupiec_test(violations, n, alpha):
    """Kupiec POF test. Returns (p_value, traffic_light)."""
    x = int(np.sum(violations))
    if n == 0:
        return 1.0, 'Green'
    pi_hat = x / n
    if pi_hat == 0:
        lr = 2 * n * np.log(1 / (1 - alpha))
    elif pi_hat == 1:
        lr = 2 * n * np.log(alpha)
    else:
        lr = 2 * (x * np.log(pi_hat / alpha) +
                   (n - x) * np.log((1 - pi_hat) / (1 - alpha)))
    p_val = 1 - chi2.cdf(lr, 1)
    # Basel traffic light (scaled to 250 days)
    annual_viol = x * (250 / n) if n > 0 else 0
    if annual_viol <= 4:
        tl = 'Green'
    elif annual_viol <= 9:
        tl = 'Yellow'
    else:
        tl = 'Red'
    return p_val, tl


def quantile_score(r, var_neg, alpha):
    """Quantile score (pinball loss). var_neg is the negative VaR forecast."""
    # Convert to positive VaR magnitude for standard formula
    var_pos = -var_neg
    violations = (r < -var_pos).astype(float)
    return np.mean((alpha - violations) * (r + var_pos))


def run_aci_on_segment(r, q_lo, alpha_init, gamma, w):
    """
    Run ACI on a data segment. Returns corrected VaR (as negative values).

    r: realized returns
    q_lo: raw VaR forecasts (negative)
    alpha_init: initial alpha level
    gamma: learning rate
    w: rolling window size
    """
    T = len(r)
    alpha_t = alpha_init
    scores = []
    corrected_var = np.full(T, np.nan)
    violations = np.zeros(T)

    for t in range(T):
        # Compute nonconformity score
        s_t = q_lo[t] - r[t]
        scores.append(s_t)

        # Use rolling window of scores for quantile computation
        if len(scores) <= 1:
            # Not enough history; use raw forecast
            corrected_var[t] = q_lo[t]
        else:
            window_scores = np.array(scores[max(0, len(scores) - w):])
            # Compute (1 - alpha_t) quantile of scores
            clipped_alpha = np.clip(alpha_t, 0.001, 0.10)
            q_level = 1 - clipped_alpha
            q_hat = np.quantile(window_scores, min(q_level, 1.0))
            # Corrected VaR: -(q_lo - q_hat)  =>  stored as negative: q_lo - q_hat
            corrected_var[t] = q_lo[t] - q_hat

        # Check violation: r < -VaR  (VaR stored as negative, so r < corrected_var)
        viol = 1.0 if r[t] < corrected_var[t] else 0.0
        violations[t] = viol

        # Update alpha
        alpha_t = alpha_t + gamma * (alpha_init - viol)
        alpha_t = np.clip(alpha_t, 0.001, 0.10)

    return corrected_var, violations


def select_gamma_on_cal(r_cal, q_lo_cal, alpha, gammas, w):
    """Select gamma using calibration data only."""
    best_gamma = gammas[0]
    best_gap = np.inf
    for g in gammas:
        _, viol = run_aci_on_segment(r_cal, q_lo_cal, alpha, g, w)
        viol_rate = np.mean(viol)
        gap = abs(viol_rate - alpha)
        if gap < best_gap:
            best_gap = gap
            best_gamma = g
    return best_gamma


def main():
    results = []

    for model_name in MODELS:
        for symbol in SYMBOLS:
            try:
                r, q_lo = load_data(model_name, symbol)
            except Exception:
                continue

            T = len(r)
            n_cal = int(T * F_CAL)
            if n_cal < W or T - n_cal < 50:
                continue

            r_cal, r_test = r[:n_cal], r[n_cal:]
            q_cal, q_test = q_lo[:n_cal], q_lo[n_cal:]

            # Select gamma on calibration data
            gamma = select_gamma_on_cal(r_cal, q_cal, ALPHA, GAMMAS, W)

            # Run ACI on test set (start fresh with selected gamma)
            # We need initial scores — use calibration scores as history
            scores_cal = q_cal - r_cal
            n_test = len(r_test)
            corrected_var_test = np.full(n_test, np.nan)
            violations_test = np.zeros(n_test)
            alpha_t = ALPHA

            # Initialize score window from tail of calibration
            score_history = list(scores_cal[-W:])

            for t in range(n_test):
                s_t = q_test[t] - r_test[t]

                # Rolling quantile from last W scores
                window = np.array(score_history[-W:])
                clipped_alpha = np.clip(alpha_t, 0.001, 0.10)
                q_level = 1 - clipped_alpha
                q_hat = np.quantile(window, min(q_level, 1.0))
                corrected_var_test[t] = q_test[t] - q_hat

                viol = 1.0 if r_test[t] < corrected_var_test[t] else 0.0
                violations_test[t] = viol

                alpha_t = alpha_t + gamma * (ALPHA - viol)
                alpha_t = np.clip(alpha_t, 0.001, 0.10)

                score_history.append(s_t)

            pi_hat = np.mean(violations_test)
            p_val, tl = kupiec_test(violations_test, n_test, ALPHA)
            qs = quantile_score(r_test, corrected_var_test, ALPHA) * 1e4
            width = np.mean(np.abs(corrected_var_test))

            results.append({
                'model': model_name,
                'symbol': symbol,
                'pi_hat': pi_hat,
                'kupiec_p': p_val,
                'kupiec_rej': p_val < 0.05,
                'qs': qs,
                'width': width,
                'traffic_light': tl,
                'gamma': gamma,
            })

    df = pd.DataFrame(results)

    # Aggregate across all 216 model-asset pairs
    n_total = len(df)
    mean_pi = df['pi_hat'].mean()
    kupiec_rej = df['kupiec_rej'].sum()
    mean_qs = df['qs'].mean()
    mean_width = df['width'].mean()
    green_pct = (df['traffic_light'] == 'Green').mean() * 100

    print("=" * 60)
    print("ACI Baseline Results (Rolling W=250)")
    print("=" * 60)
    print(f"Total pairs:     {n_total}")
    print(f"Mean pi_hat:     {mean_pi:.3f}")
    print(f"Kupiec rej:      {int(kupiec_rej)}/{n_total}")
    print(f"Mean QS (1e-4):  {mean_qs:.2f}")
    print(f"Mean width:      {mean_width:.3f}")
    print(f"Green %:         {green_pct:.1f}")
    print()
    print("LaTeX row for Table 12:")
    print(f"ACI \\citep{{gibbs2021adaptive}} & .{int(mean_pi*1000):03d} & "
          f"{int(kupiec_rej)}/{n_total} & {mean_qs:.2f} & "
          f".{int(mean_width*1000):03d} & {green_pct:.1f} \\\\")

    # Save detailed results
    out_dir = BASE / 'results'
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'aci_baseline_results.csv', index=False)

    # Per-model summary
    print("\nPer-model breakdown:")
    for m in MODELS:
        sub = df[df['model'] == m]
        if len(sub) == 0:
            continue
        print(f"  {m:16s}: pi={sub['pi_hat'].mean():.3f}, "
              f"Green={int((sub['traffic_light']=='Green').sum())}/{len(sub)}, "
              f"QS={sub['qs'].mean():.2f}, "
              f"W={sub['width'].mean():.3f}")

    return df


if __name__ == '__main__':
    main()
