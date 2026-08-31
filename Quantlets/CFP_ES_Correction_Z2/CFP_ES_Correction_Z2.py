"""
Table C.1: ES Correction Z2 Results
====================================
Computes raw and conformally corrected Z2 (Acerbi-Székely) statistics
for all 9 models × 24 assets at alpha = 0.01.

ES is derived from each model's mean/std output using the Gaussian ES formula:
    ES_t = mean_t - std_t * phi(z_alpha) / alpha
This is exact for GARCH-N and EWMA; an approximation for others.
The conformal ES correction (Appendix C) shifts ES by q_hat_E computed
on calibration-set VaR violation days.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from math import ceil
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = REPO_ROOT / 'cfp_ijf_data'
OUT  = Path(__file__).resolve().parent

ALPHA = 0.01
F_CAL = 0.70
Z2_THRESHOLD = -1.96  # 5% significance (one-sided)

SYMBOLS = ['SP500', 'STOXX', 'GDAXI', 'FCHI', 'FTSE100', 'ICLN',
           'NIKKEI', 'HSI', 'BOVESPA', 'NIFTY', 'ASX200', 'CBU0',
           'TLT', 'IBGL', 'DJCI', 'GOLD', 'WTI', 'NATGAS',
           'BTC', 'ETH', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

# Model set from cfp_config. This file carried its own nine-model dictionary,
# which is how it kept describing a nine-forecaster panel after the panel became
# thirteen -- the same drift that had to be fixed independently in five other
# scripts.
import sys
_here = Path(__file__).resolve()
sys.path.insert(0, str(_here.parent.parent if _here.parent.name != 'scripts'
                       else _here.parent.parent / 'Quantlets'))
try:
    from cfp_config import MODELS, split_indices  # noqa: E402
except ModuleNotFoundError:
    sys.path.insert(0, str(_here.parent.parent.parent / 'Quantlets'))
    from cfp_config import MODELS  # noqa: E402

# Gaussian ES multiplier: phi(z_alpha) / alpha
z_alpha = norm.ppf(ALPHA)
ES_MULT = norm.pdf(z_alpha) / ALPHA  # ≈ 2.6652


def load_data(model_key, symbol):
    """Load returns, VaR forecasts, and compute ES for a model-asset pair."""
    subdir, suffix = MODELS[model_key]

    # Load returns
    ret = pd.read_csv(DATA / 'returns' / f'{symbol}.csv',
                      index_col=0, parse_dates=True)
    ret.columns = ['r']

    # Load forecasts
    if suffix is None:
        fc = pd.read_parquet(DATA / subdir / f'{symbol}.parquet')
    else:
        fc = pd.read_parquet(DATA / subdir / f'{symbol}_{suffix}.parquet')

    # Align dates
    common = ret.index.intersection(fc.index)
    ret = ret.loc[common]
    fc = fc.loc[common]

    # VaR at alpha level (negative number)
    var_t = fc[f'VaR_{ALPHA}'].values

    # ES from Gaussian formula: ES_t = mean_t - std_t * ES_MULT (negative)
    mean_t = fc['mean'].values
    std_t = fc['std'].values
    es_t = mean_t - std_t * ES_MULT

    r_t = ret['r'].values
    dates = common

    return r_t, var_t, es_t, dates


def compute_z2(r_t, var_t, es_t):
    """
    Acerbi-Szekely test 2:

        Z2 = (1/(N*alpha)) * sum_t [ I_t * X_t / ES_t ] + 1,
        I_t = 1{X_t < -VaR_t},

    with X_t the P&L and **ES_t a POSITIVE magnitude**, as in the source. Under
    the null E[Z2] = 0 and the test is one-sided: reject for Z2 sufficiently
    negative, meaning realised losses on violation days exceed the forecast ES.

    CORRECTED 2026-08-21. This routine previously divided by the stored ES
    column directly. That column is a lower-tail quantity and is NEGATIVE, so
    every term of the sum had its sign flipped and Z2 came out large and
    POSITIVE -- on the Chronos series sampled at top_k = 50, around +146 and
    +391. A one-sided lower-tail test can never reject such a value, so the
    routine reported 24/24 passes on a series whose violation rate at the same
    level is 0.39. That was an artefact of this function, not a property of the
    test: with ES entered as a magnitude the statistic is -144 and -389 and the
    test rejects on all 24 assets, as it should.

    Verified against an independent implementation in
    analysis/provenance/verify_z2.py, which also uses the canonical per-date
    denominator rather than the time-averaged one below.
    """
    N = len(r_t)
    I_t = (r_t < var_t).astype(float)

    # ES as a positive magnitude. The stored column is negative.
    es_bar = np.mean(-np.asarray(es_t, dtype=float))

    if es_bar == 0 or np.isnan(es_bar):
        return np.nan

    return np.sum(I_t * r_t) / (N * ALPHA * es_bar) + 1


def conformal_es_correction(r_cal, var_cal, es_cal, es_test):
    """
    Heuristic ES correction (Appendix C):
    1. On calibration VaR violation days, compute s_E_t = r_t + ES_t
       (Note: ES_t is negative, so s_E_t = r_t + ES_t; if r_t is worse than ES,
        s_E_t < 0)
       Actually per paper: s_E_t = r_t + ES_t^{alpha;M}
       With sign convention: ES is stored as negative.
       s_E_t = r_t - |ES_t| = r_t + ES_t  (since ES_t < 0, this works)
    2. q_hat_E = (1-alpha)-quantile of {s_E_t}
    3. Corrected ES: ES_t^cqr = ES_t + q_hat_E
    """
    # Calibration violations
    viol_mask = r_cal < var_cal
    n_viol = np.sum(viol_mask)

    if n_viol == 0:
        return es_test.copy(), 0.0

    # ES residual scores on violation days
    s_E = r_cal[viol_mask] + np.abs(es_cal[viol_mask])
    # Paper formula: s_t^E = r_t + ES_t^{alpha;M}
    # But ES is stored negative. The paper defines ES as a positive loss measure
    # internally. Let me re-read: "s_E = r_t + ES_hat"
    # If ES_hat is the predicted mean tail loss (positive), then
    # s_E = r_t + ES_hat: on a violation day r_t is very negative,
    # ES_hat is positive → s_E could be positive or negative.
    # A positive s_E means the loss was less severe than predicted.
    #
    # With our convention (ES_t negative):
    # |ES_t| is the magnitude of predicted tail loss
    # s_E = r_t + |ES_t| = r_t - ES_t (since ES_t < 0)
    s_E = r_cal[viol_mask] - es_cal[viol_mask]  # = r_t + |ES_t|

    # (1-alpha)-quantile
    k = ceil((n_viol + 1) * (1 - ALPHA))
    k = min(k, n_viol)
    q_hat_E = np.sort(s_E)[k - 1]

    # Corrected ES: ES_t^cqr = ES_t - q_hat_E
    # (shifting ES more negative if q_hat_E > 0, i.e., making it more conservative)
    # Actually: ES_cqr = ES_hat + q_E (paper notation with ES positive)
    # In our negative convention: ES_cqr = ES_t - q_hat_E
    # Wait, let me think again:
    # Paper: ES_cqr_t = ES_hat_t^{alpha;M} + q_hat_E
    # If ES_hat is positive (predicted loss): ES_cqr = ES_hat + q_hat_E
    # In our negative convention, ES_t = -|ES_hat|
    # So ES_cqr = -(|ES_hat| - q_hat_E) = ES_t + q_hat_E? No...
    #
    # Let's be precise. The paper says:
    # s_t^E = r_t + ES_hat (where ES_hat is the predicted ES, positive)
    # q_hat_E is the (1-alpha)-quantile of {s_t^E}
    # ES_cqr = ES_hat + q_hat_E (still positive)
    #
    # Our data: es_t is negative (like VaR). So |es_t| = ES_hat (positive).
    # s_E = r_t + |es_t| = r_t - es_t (computed above correctly)
    # ES_cqr (positive) = |es_t| + q_hat_E
    # ES_cqr (negative, our convention) = -(|es_t| + q_hat_E) = es_t - q_hat_E
    es_corrected = es_test - q_hat_E

    return es_corrected, q_hat_E


# ── Main computation ──────────────────────────────────────────────────────
results = []

for model_name in MODELS:
    raw_z2_list = []
    corr_z2_list = []
    raw_pass = 0
    corr_pass = 0
    n_computed = 0

    for symbol in SYMBOLS:
        try:
            r_t, var_t, es_t, dates = load_data(model_name, symbol)
        except Exception as e:
            print(f"  SKIP {model_name}/{symbol}: {e}")
            continue

        N = len(r_t)
        _cal, _test, _g = split_indices(N, var_t - r_t, f_cal=F_CAL)
        n_cal, t0 = len(_cal), int(_test[0])

        # Split
        r_cal, r_test = r_t[:n_cal], r_t[t0:]
        var_cal, var_test = var_t[:n_cal], var_t[t0:]
        es_cal, es_test = es_t[:n_cal], es_t[t0:]

        # Conformal VaR correction (to get corrected violations for corrected Z2)
        s_V = var_cal - r_cal  # = q_hat_lo - r_t (one-sided score)
        k_v = ceil((n_cal + 1) * (1 - ALPHA))
        k_v = min(k_v, n_cal)
        q_hat_V = np.sort(s_V)[k_v - 1]
        var_test_corr = var_test - q_hat_V  # corrected VaR (more negative = wider)

        # ES correction
        es_test_corr, q_hat_E = conformal_es_correction(
            r_cal, var_cal, es_cal, es_test
        )

        # Raw Z2
        z2_raw = compute_z2(r_test, var_test, es_test)

        # Corrected Z2 (using corrected VaR for violations, corrected ES)
        z2_corr = compute_z2(r_test, var_test_corr, es_test_corr)

        if not np.isnan(z2_raw):
            raw_z2_list.append(z2_raw)
            raw_pass += int(z2_raw >= Z2_THRESHOLD)
            n_computed += 1

        if not np.isnan(z2_corr):
            corr_z2_list.append(z2_corr)
            corr_pass += int(z2_corr >= Z2_THRESHOLD)

        results.append({
            'model': model_name,
            'symbol': symbol,
            'z2_raw': z2_raw,
            'z2_corr': z2_corr,
            'raw_pass': z2_raw >= Z2_THRESHOLD if not np.isnan(z2_raw) else None,
            'corr_pass': z2_corr >= Z2_THRESHOLD if not np.isnan(z2_corr) else None,
            'q_hat_E': q_hat_E,
            'n_test': N - int(N * F_CAL),
        })

    mean_raw = np.mean(raw_z2_list) if raw_z2_list else np.nan
    mean_corr = np.mean(corr_z2_list) if corr_z2_list else np.nan

    print(f"{model_name:18s}  Z2_raw={mean_raw:+.3f}  Z2_corr={mean_corr:+.3f}  "
          f"raw_pass={raw_pass}/{n_computed}  corr_pass={corr_pass}/{n_computed}")

# ── Save CSV ──────────────────────────────────────────────────────────────
df_results = pd.DataFrame(results)
df_results.to_csv(OUT / 'table_c1_es_correction.csv', index=False)
print(f"\nSaved: {OUT / 'table_c1_es_correction.csv'}")

# ── Generate LaTeX ────────────────────────────────────────────────────────
print("\n% ── LaTeX for Table C.1 ──────────────────────────────────────")

MODEL_DISPLAY = {
    'Chronos-Small': 'Chronos-Small (default)',
    'Chronos-Small-A': 'Chronos-Small (analytic)',
    'Chronos-Mini': 'Chronos-Mini (default)',
    'Chronos-Mini-A': 'Chronos-Mini (analytic)',
    'TimesFM-2.5': 'TimesFM 2.5',
    'Moirai-1.1': 'Moirai 1.1',
    'Moirai-2.0': 'Moirai 2.0',
    'Lag-Llama': 'Lag-Llama',
    'GJR-GARCH': 'GJR-GARCH',
    'GJR-GARCH-t': r'GJR-GARCH-$t$',
    'GARCH-N': 'GARCH-N',
    'Hist-Sim': r'Hist.\ Sim.',
    'EWMA': 'EWMA',
}
assert set(MODEL_DISPLAY) == set(MODELS), 'display map out of step with cfp_config'

for model_name in MODELS:
    mdf = df_results[df_results['model'] == model_name]
    n_assets = len(mdf)
    mean_raw = mdf['z2_raw'].mean()
    mean_corr = mdf['z2_corr'].mean()
    n_raw_pass = mdf['raw_pass'].sum()
    n_corr_pass = mdf['corr_pass'].sum()

    display = MODEL_DISPLAY[model_name]
    print(f"\t\t\t{display:20s} & {mean_raw:+.2f} & {mean_corr:+.2f} "
          f"& {int(n_raw_pass)}/{n_assets} & {int(n_corr_pass)}/{n_assets} \\\\")

# ── Summary .tex (tabular-only, for \input in manuscript) ────────────────
from decimal import Decimal, ROUND_HALF_UP

def rhu(x, places):
    fmt = '0.' + '0' * places
    return Decimal(str(x)).quantize(Decimal(fmt), rounding=ROUND_HALF_UP)

MODEL_ORDER = list(MODELS.keys())
lines = [
    r'\begin{tabular}{@{}l rr rr@{}}',
    r'\toprule',
    r'& \multicolumn{2}{c}{Mean $Z_2$}',
    r'& \multicolumn{2}{c}{Pass ($x$/24)} \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
    r'Model & Raw & Corrected & Raw & Corrected \\',
    r'\midrule',
]

for i, model_name in enumerate(MODEL_ORDER):
    mdf = df_results[df_results['model'] == model_name]
    n_assets = len(mdf)
    mean_raw = mdf['z2_raw'].mean()
    mean_corr = mdf['z2_corr'].mean()
    n_raw_pass = int(mdf['raw_pass'].sum())
    n_corr_pass = int(mdf['corr_pass'].sum())

    display = MODEL_DISPLAY[model_name]

    def signed(x):
        # A hardcoded '$+$' was correct only while the statistic was positive
        # by construction -- which it was, because of the sign defect this
        # function used to carry.
        v = rhu(x, 2)
        return f'$-${abs(float(v)):.2f}' if float(v) < 0 else f'$+${float(v):.2f}'

    line = (f'{display} & {signed(mean_raw)} & {signed(mean_corr)}'
            f' & {n_raw_pass}/{n_assets} & {n_corr_pass}/{n_assets} \\\\')
    lines.append(line)
    if i == 4:
        lines.append(r'\midrule')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')

tex = '\n'.join(lines) + '\n'
tex_path = OUT / 'tab_es_correction.tex'
tex_path.write_text(tex)
print(f'\nSaved {tex_path.name}')
print(tex)
