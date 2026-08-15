"""
R4 revisions: compute rolling w=250 pooled stats for Table 12,
commodity Green at w=500, and the all-9-model calibration-efficiency
frontier figure for Appendix I.

Run from project root.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from math import ceil
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


def qhat_ceil(scores, alpha):
    """Conformal ceiling-quantile: sorted-scores[ceil((n+1)(1-alpha))-1]."""
    n = len(scores)
    k = int(ceil((n + 1) * (1 - alpha))) - 1
    k = min(k, n - 1)
    return np.sort(scores)[k]

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'cfp_ijf_data'
FIG = BASE / 'figures'
FIG.mkdir(exist_ok=True)

ALPHA = 0.01
F_CAL = 0.70

SYMBOLS = ['SP500', 'STOXX', 'GDAXI', 'FCHI', 'FTSE100', 'ICLN',
           'NIKKEI', 'HSI', 'BOVESPA', 'NIFTY', 'ASX200', 'CBU0',
           'TLT', 'IBGL', 'DJCI', 'GOLD', 'WTI', 'NATGAS',
           'BTC', 'ETH', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

COMMODITIES = ['DJCI', 'GOLD', 'WTI', 'NATGAS']

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


def load_pair(model_key, symbol):
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
    return ret['r'].values[mask], fc[var_col].values[mask]


def kupiec_p(x, n, alpha=ALPHA):
    if n == 0:
        return 1.0
    pi_hat = x / n
    if pi_hat == 0:
        lr = 2 * n * np.log(1 / (1 - alpha))
    elif pi_hat == 1:
        lr = 2 * n * np.log(alpha)
    else:
        lr = 2 * (x * np.log(pi_hat / alpha) +
                  (n - x) * np.log((1 - pi_hat) / (1 - alpha)))
    return 1 - chi2.cdf(lr, 1)


def traffic_light(x, n):
    if n == 0:
        return 'Green'
    annual = x * (250.0 / n)
    if annual <= 4:
        return 'Green'
    if annual <= 9:
        return 'Yellow'
    return 'Red'


def pinball_score(r, var_neg, alpha=ALPHA):
    """Pinball loss using positive VaR convention."""
    var_pos = -var_neg
    violations = (r < -var_pos).astype(float)
    return np.mean((alpha - violations) * (r + var_pos))


def rolling_conformal(r, q_lo, w, alpha=ALPHA):
    """
    Rolling w-day conformal correction on test set.
    Uses calibration-set tail as initial score history, then rolls.
    Returns test-set arrays for realised returns, corrected VaR (neg).
    """
    T = len(r)
    n_cal = int(T * F_CAL)
    if n_cal < w or T - n_cal < 50:
        return None
    r_cal, r_test = r[:n_cal], r[n_cal:]
    q_cal, q_test = q_lo[:n_cal], q_lo[n_cal:]

    scores_cal = q_cal - r_cal  # tail-score: positive when r << q_lo
    history = list(scores_cal[-w:])

    n_test = len(r_test)
    corrected_var = np.full(n_test, np.nan)
    for t in range(n_test):
        window = np.array(history[-w:])
        q_hat = qhat_ceil(window, alpha)
        corrected_var[t] = q_test[t] - q_hat  # negative-VaR convention
        # update history AFTER this timestep
        s_t = q_test[t] - r_test[t]
        history.append(s_t)
    return r_test, corrected_var


def eval_rolling(r_test, corrected_var):
    """Given test-set r and corrected VaR (neg), return metrics."""
    n = len(r_test)
    violations = (r_test < corrected_var).astype(int)
    x = int(violations.sum())
    pi_hat = x / n
    kup = kupiec_p(x, n)
    tl = traffic_light(x, n)
    qs = pinball_score(r_test, corrected_var)
    width = float(np.mean(np.abs(corrected_var)))
    return {
        'n_test': n, 'viol': x, 'pi_hat': pi_hat,
        'kupiec_p': kup, 'TL': tl, 'QS': qs, 'width': width,
    }


def run_grid(symbols, window, label=''):
    rows = []
    for mname in MODELS:
        for sym in symbols:
            try:
                r, q = load_pair(mname, sym)
            except Exception as e:
                print(f"  skip {mname}/{sym}: {e}")
                continue
            roll = rolling_conformal(r, q, w=window)
            if roll is None:
                continue
            r_t, v_t = roll
            m = eval_rolling(r_t, v_t)
            m.update({'model': mname, 'symbol': sym, 'w': window})
            rows.append(m)
    df = pd.DataFrame(rows)
    print(f"\n=== Rolling w={window} | {label} | {len(df)} pairs ===")
    return df


def task1_table12_row():
    print("\n" + "=" * 70)
    print("TASK 1: Rolling pooled numbers for Table 12 (w=250, all 216 pairs)")
    print("=" * 70)
    df = run_grid(SYMBOLS, window=250, label='ALL 24 ASSETS')
    n = len(df)
    pi = df['pi_hat'].mean()
    kup_rej = int((df['kupiec_p'] < 0.05).sum())
    qs_mean = df['QS'].mean() * 1e4
    width = df['width'].mean()
    green = int((df['TL'] == 'Green').sum())
    print()
    print(f"TABLE 12 ROW -- Conformal rolling (w=250):")
    print(f"  pi_hat:       {pi:.3f}")
    print(f"  Kupiec rej:   {kup_rej}/{n}")
    print(f"  QS (x1e-4):   {qs_mean:.2f}")
    print(f"  Width:        {width:.3f}")
    print(f"  Green:        {green}/{n} ({100*green/n:.1f}%)")
    df.to_csv(BASE / 'results' / 'rolling_w250_pooled.csv', index=False)
    return df


def task2_commodity_w500():
    print("\n" + "=" * 70)
    print("TASK 2: Commodity Green% at w=500 (DJCI/GOLD/WTI/NATGAS x 9 models)")
    print("=" * 70)
    df = run_grid(COMMODITIES, window=500, label='COMMODITIES ONLY')
    n = len(df)
    green = int((df['TL'] == 'Green').sum())
    wti = df[df['symbol'] == 'WTI']
    wti_green = int((wti['TL'] == 'Green').sum())
    print()
    print(f"COMMODITY w=500:")
    print(f"  Overall Green: {green}/{n} ({100*green/n:.0f}%)")
    print(f"  WTI Green:     {wti_green}/{len(wti)}")
    # also compare to w=250 on same commodity set for reference
    df250 = run_grid(COMMODITIES, window=250, label='COMMODITIES w=250 (ref)')
    g250 = int((df250['TL'] == 'Green').sum())
    wti250 = df250[df250['symbol'] == 'WTI']
    wti_g250 = int((wti250['TL'] == 'Green').sum())
    print(f"\nReference -- Commodities at w=250:")
    print(f"  Overall Green: {g250}/{len(df250)} ({100*g250/len(df250):.0f}%)")
    print(f"  WTI Green:     {wti_g250}/{len(wti250)}")
    df.to_csv(BASE / 'results' / 'commodities_w500.csv', index=False)
    df250.to_csv(BASE / 'results' / 'commodities_w250.csv', index=False)
    return df, df250


def task3_frontier_all9():
    print("\n" + "=" * 70)
    print("TASK 3: Appendix I - full 9-model calibration-efficiency frontier")
    print("=" * 70)
    src = DATA / 'paper_outputs' / 'tables' / 'all_results.csv'
    if not src.exists():
        print(f"MISSING: {src}")
        return None
    df = pd.read_csv(src)
    d01 = df[df['alpha'] == ALPHA].copy()

    MODEL_ORDER = ['Chronos-Small', 'Chronos-Mini', 'TimesFM-2.5',
                   'Moirai-2.0', 'Lag-Llama',
                   'GJR-GARCH', 'GARCH-N', 'Hist-Sim', 'EWMA']
    COLORS = {
        'Chronos-Small': '#A32D2D', 'Chronos-Mini': '#CC5555',
        'TimesFM-2.5':   '#185FA5', 'Moirai-2.0': '#0F6E56',
        'Lag-Llama':     '#534AB7',
        'GJR-GARCH':     '#555555', 'GARCH-N': '#777777',
        'Hist-Sim':      '#999999', 'EWMA': '#BBBBBB',
    }

    # per-model aggregates
    agg = d01.groupby('model').agg(
        raw_cov=('pihat_raw', lambda s: 1 - s.mean()),
        corr_cov=('pihat_cp', lambda s: 1 - s.mean()),
        raw_width=('VaR_width', 'mean'),
        corr_width=('VaR_width', 'mean'),
    ).reindex(MODEL_ORDER)

    # raw VaR width = VaR_width column times raw violation ratio? Actually
    # VaR_width is the corrected width in all_results.csv. For the raw width,
    # we need a separate computation. Approximate raw width from raw forecasts
    # directly below.

    # Compute raw width per model by averaging |VaR_raw| over calibration+test
    print("  Computing raw widths from forecasts...")
    raw_widths = {}
    raw_covs = {}
    for m in MODEL_ORDER:
        ws, cvs = [], []
        for s in SYMBOLS:
            try:
                r, q = load_pair(m, s)
                ws.append(float(np.mean(np.abs(q))))
                # raw coverage = 1 - fraction(r < q)
                cvs.append(1.0 - float(np.mean(r < q)))
            except Exception:
                continue
        raw_widths[m] = float(np.mean(ws)) if ws else np.nan
        raw_covs[m] = float(np.mean(cvs)) if cvs else np.nan

    # corrected widths from all_results.csv
    corr_widths = d01.groupby('model')['VaR_width'].mean().reindex(MODEL_ORDER)
    corr_covs = 1 - d01.groupby('model')['pihat_cp'].mean().reindex(MODEL_ORDER)

    # Plot
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 10,
        'axes.grid': False,
        'savefig.transparent': True, 'savefig.dpi': 300,
        'axes.spines.top': False, 'axes.spines.right': False,
    })
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m in MODEL_ORDER:
        rw, rc = raw_widths[m], raw_covs[m]
        cw, cc = corr_widths[m], corr_covs[m]
        color = COLORS[m]
        # Clip raw coverage for display (Chronos can be 0.6)
        rc_disp = max(rc, 0.55)
        ax.annotate("", xy=(cw, cc), xytext=(rw, rc_disp),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.8))
        ax.scatter([rw], [rc_disp], facecolors='none', edgecolors=color,
                   s=55, linewidths=1.4, zorder=5)
        ax.scatter([cw], [cc], color=color, s=55, zorder=5)
        ax.text(cw + 0.0008, cc, m, fontsize=8, color=color, va='center')
    ax.axhline(1 - ALPHA, ls='--', c='#A32D2D', lw=0.9,
               label=f'{int((1-ALPHA)*100)}% coverage target')
    ax.set_xlabel(r'Mean VaR width $|\widehat{\mathrm{VaR}}|$ (lower = sharper)')
    ax.set_ylabel(r'Empirical coverage $1 - \hat\pi$')
    ax.set_title(r'Calibration--Efficiency Frontier (all nine forecasters, $\alpha = 0.01$)',
                 fontsize=11)
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    out = FIG / 'frontier_all9.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(FIG / 'frontier_all9.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")
    return out


if __name__ == '__main__':
    (BASE / 'results').mkdir(exist_ok=True)
    task1_table12_row()
    task2_commodity_w500()
    task3_frontier_all9()
    print("\nAll three R4 tasks complete.")
