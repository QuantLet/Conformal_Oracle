"""
EVT-POT and Filtered Historical Simulation baselines.
Daily re-estimation on 250-day rolling windows.
Produces the EVT-POT and FHS rows of tab_baselines.tex.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import genpareto
from scipy import stats as sp_stats
from arch import arch_model
import warnings, sys, time
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR.parent.parent
DATA = BASE / 'cfp_ijf_data' / 'returns'
OUT  = SCRIPT_DIR

ALPHA = 0.01
F_CAL = 0.70
W = 250  # rolling window
GPD_THRESH_QUANTILE = 0.95
EWMA_LAMBDA = 0.94

SYMBOLS = ['SP500','STOXX','GDAXI','FCHI','FTSE100','ICLN',
           'NIKKEI','HSI','BOVESPA','NIFTY','ASX200','CBU0',
           'TLT','IBGL','DJCI','GOLD','WTI','NATGAS',
           'BTC','ETH','EURUSD','GBPUSD','USDJPY','AUDUSD']


def kupiec_pof(violations, n_test, alpha):
    """Kupiec proportion-of-failures LR test. Returns p-value."""
    x = int(np.sum(violations))
    n = int(n_test)
    if x == 0:
        return 1.0
    pi_hat = x / n
    if pi_hat >= 1.0:
        return 0.0
    lr = 2 * (x * np.log(pi_hat / alpha) +
              (n - x) * np.log((1 - pi_hat) / (1 - alpha)))
    return 1 - sp_stats.chi2.cdf(lr, 1)


def quantile_score(r, var_pos, alpha):
    """Pinball / quantile score. var_pos is positive VaR magnitude."""
    violations = (r < -var_pos).astype(float)
    return np.mean((alpha - violations) * (r + var_pos))


def fit_garch_window(returns_window):
    """Fit GARCH(1,1)-N on a window. Returns (sigma_forecast, resid_std, success).
    sigma_forecast = next-period vol forecast.
    resid_std = standardised residuals for the window.
    """
    try:
        am = arch_model(returns_window * 100, vol='GARCH', p=1, q=1,
                        mean='Zero', dist='normal', rescale=False)
        res = am.fit(disp='off', show_warning=False)
        cond_vol = res.conditional_volatility.values / 100
        cond_vol = np.where(cond_vol <= 0, np.nan, cond_vol)
        # Standardised residuals
        resid_std = returns_window / cond_vol
        # Next-period forecast
        fcast = res.forecast(horizon=1)
        sigma_next = np.sqrt(fcast.variance.values[-1, 0]) / 100
        if sigma_next <= 0 or np.isnan(sigma_next):
            return None, None, False
        return sigma_next, resid_std, True
    except Exception:
        return None, None, False


def ewma_sigma(returns_window, lam=EWMA_LAMBDA):
    """EWMA volatility forecast for next period."""
    r = returns_window
    n = len(r)
    weights = np.array([(1 - lam) * lam**(n - 1 - i) for i in range(n)])
    weights /= weights.sum()
    var = np.sum(weights * r**2)
    return np.sqrt(var)


def compute_baselines(symbol):
    """Compute EVT-POT and FHS VaR for one asset. Returns dict of results."""
    ret = pd.read_csv(DATA / f'{symbol}.csv', index_col=0, parse_dates=True)
    r = ret.iloc[:, 0].values
    N = len(r)
    n_cal = int(N * F_CAL)

    # Test set indices
    test_start = max(n_cal, W)  # need at least W observations before first test day
    n_test = N - test_start

    if n_test < 50:
        print(f"  {symbol}: SKIP (only {n_test} test days)")
        return None

    var_evt = np.full(n_test, np.nan)
    var_fhs = np.full(n_test, np.nan)
    garch_fails = 0
    gpd_fails = 0
    prev_sigma = None

    for i in range(n_test):
        t = test_start + i
        window = r[t - W:t]

        # Fit GARCH
        sigma_next, z_std, garch_ok = fit_garch_window(window)

        if not garch_ok:
            garch_fails += 1
            sigma_next = ewma_sigma(window)
            z_std = window / sigma_next  # crude standardisation
            if sigma_next <= 0 or np.isnan(sigma_next):
                if prev_sigma is not None:
                    sigma_next = prev_sigma
                else:
                    continue

        prev_sigma = sigma_next

        # Clean z_std
        z_valid = z_std[np.isfinite(z_std)]
        if len(z_valid) < 50:
            continue

        # --- FHS VaR ---
        q_alpha = np.quantile(z_valid, ALPHA)  # alpha-quantile of standardised residuals
        var_fhs[i] = -sigma_next * q_alpha  # positive

        # --- EVT-POT VaR ---
        losses = -z_valid  # positive losses
        u = np.quantile(losses, GPD_THRESH_QUANTILE)
        exceedances = losses[losses > u] - u
        n_exc = len(exceedances)

        if n_exc < 10:
            # Fallback to empirical
            gpd_fails += 1
            var_evt[i] = var_fhs[i]  # use FHS as fallback
            continue

        try:
            xi, _, beta = genpareto.fit(exceedances, floc=0)
            if xi > 0.5 or xi < -0.5 or beta <= 0:
                gpd_fails += 1
                var_evt[i] = var_fhs[i]
                continue

            n_w = len(z_valid)
            n_u = n_exc
            # McNeil-Frey formula
            var_z = u + (beta / xi) * ((n_w / n_u * ALPHA)**(-xi) - 1)
            var_evt[i] = sigma_next * var_z  # positive

            if var_evt[i] <= 0 or np.isnan(var_evt[i]) or var_evt[i] > 1.0:
                gpd_fails += 1
                var_evt[i] = var_fhs[i]

        except Exception:
            gpd_fails += 1
            var_evt[i] = var_fhs[i]

    # Compute metrics on test set
    r_test = r[test_start:]
    results = {}

    for method, var_pos in [('EVT-POT', var_evt), ('FHS', var_fhs)]:
        valid = np.isfinite(var_pos)
        if valid.sum() < 50:
            print(f"  {symbol}/{method}: too few valid VaR ({valid.sum()})")
            continue

        r_v = r_test[valid]
        v_v = var_pos[valid]
        n_v = len(r_v)

        violations = (r_v < -v_v).astype(float)
        pi_hat = violations.mean()
        p_kup = kupiec_pof(violations, n_v, ALPHA)
        qs = quantile_score(r_v, v_v, ALPHA)
        width = np.mean(v_v)

        # Green zone: scale violations to 250 days
        viol_250 = pi_hat * 250
        green = viol_250 <= 4

        results[method] = {
            'symbol': symbol,
            'method': method,
            'n_test': n_v,
            'pi_hat': pi_hat,
            'p_kupiec': p_kup,
            'kup_pass': p_kup >= 0.05,
            'QS': qs,
            'width': width,
            'green': green,
            'viol_250': viol_250,
            'garch_fails': garch_fails,
            'gpd_fails': gpd_fails if method == 'EVT-POT' else 0,
            'fallback_pct': (gpd_fails / n_test * 100) if method == 'EVT-POT' else (garch_fails / n_test * 100),
        }

    return results


# ── Main ──────────────────────────────────────────────────────────────────
print(f"Computing EVT-POT and FHS baselines for {len(SYMBOLS)} assets...")
print(f"Window={W}, alpha={ALPHA}, threshold_q={GPD_THRESH_QUANTILE}")
print()

all_results = []
t0 = time.time()

for idx, sym in enumerate(SYMBOLS):
    t_sym = time.time()
    print(f"[{idx+1:2d}/{len(SYMBOLS)}] {sym}...", end=' ', flush=True)

    res = compute_baselines(sym)
    if res is None:
        print("SKIPPED")
        continue

    for method, d in res.items():
        all_results.append(d)
        status = 'G' if d['green'] else 'Y/R'
        fb = f"fb={d['fallback_pct']:.0f}%" if method == 'EVT-POT' else ''
        print(f"{method}: pi={d['pi_hat']:.3f} QS={d['QS']*1e4:.1f} w={d['width']:.4f} {status} {fb}", end='  ')

    elapsed = time.time() - t_sym
    print(f"({elapsed:.0f}s)")

total_time = time.time() - t0
print(f"\nTotal time: {total_time/60:.1f} minutes")

# ── Results ───────────────────────────────────────────────────────────────
df = pd.DataFrame(all_results)
df.to_csv(OUT / 'baselines_evt_fhs.csv', index=False)

print("\n" + "="*70)
print("PER-ASSET RESULTS")
print("="*70)
for method in ['EVT-POT', 'FHS']:
    mdf = df[df['method'] == method]
    print(f"\n{method}:")
    print(f"{'Asset':10s} {'pi_hat':>8s} {'Kup_p':>8s} {'QS(1e-4)':>10s} {'Width':>8s} {'Green':>6s} {'Fallback%':>10s}")
    for _, row in mdf.iterrows():
        g = 'Y' if row['green'] else 'N'
        print(f"{row['symbol']:10s} {row['pi_hat']:8.4f} {row['p_kupiec']:8.4f} {row['QS']*1e4:10.2f} {row['width']:8.5f} {g:>6s} {row['fallback_pct']:10.1f}")

print("\n" + "="*70)
print("SUMMARY (means across 24 assets)")
print("="*70)

summary = []
for method in ['EVT-POT', 'FHS']:
    mdf = df[df['method'] == method]
    n_assets = len(mdf)
    s = {
        'method': method,
        'pi_hat': mdf['pi_hat'].mean(),
        'kup_pass': f"{int(mdf['kup_pass'].sum())}/{n_assets}",
        'QS': mdf['QS'].mean() * 1e4,
        'width': mdf['width'].mean(),
        'green_pct': mdf['green'].mean() * 100,
        'n_green': int(mdf['green'].sum()),
    }
    summary.append(s)
    print(f"{method:22s} | pi={s['pi_hat']:.3f} | Kup pass={s['kup_pass']} | "
          f"QS={s['QS']:.1f} | w={s['width']:.4f} | Green={s['green_pct']:.1f}%")

pd.DataFrame(summary).to_csv(OUT / 'baselines_evt_fhs_summary.csv', index=False)

# Fallback stats
evt_df = df[df['method'] == 'EVT-POT']
n_high_fb = (evt_df['fallback_pct'] > 10).sum()
n_very_high_fb = (evt_df['fallback_pct'] > 50).sum()
total_garch_fails = df['garch_fails'].sum()
print(f"\nFallback stats:")
print(f"  EVT: {n_high_fb} assets had >10% GPD fallback, {n_very_high_fb} had >50%")
print(f"  GARCH: {int(total_garch_fails)} total convergence failures across all assets")

# LaTeX
print("\n% ── LaTeX rows for Table 12 ──────────────────────────────────")
for s in summary:
    name = 'EVT-POT' if s['method'] == 'EVT-POT' else r"Filtered Hist.\ Sim."
    print(f"\t\t\t{name:25s} & .{s['pi_hat']*1000:.0f} & {s['kup_pass']} & "
          f"{s['QS']:.1f} & .{s['width']*1000:.0f} & {s['green_pct']:.1f} \\\\")
