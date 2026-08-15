"""
Fissler–Ziegel (FZ0) Joint VaR-ES Scores
==========================================
Computes the FZ0 consistent scoring function for joint (VaR, ES)
evaluation, comparing raw vs conformally corrected forecasts.

FZ0 specification (Fissler & Ziegel, 2016):
  S(v, e, r) = (1/alpha) * 1(r < -v) * (-v - r)
             + (-v)/alpha * (1(r < -v) - alpha)
             + e^{-1} * (1(r < -v) * (-r) - e * alpha)
             - log(-e)

where v = VaR forecast (positive), e = ES forecast (negative), r = return.

Note: VaR and ES are stored as NEGATIVE values in our data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from math import ceil
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA = BASE / 'cfp_ijf_data'

ALPHA = 0.01
F_CAL = 0.70

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

# Gaussian ES multiplier: phi(z_alpha) / alpha
z_alpha = norm.ppf(ALPHA)
ES_MULT = norm.pdf(z_alpha) / ALPHA


def load_data(model_key, symbol):
    """Load returns, VaR, and ES for a model-asset pair."""
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

    r = ret['r'].values
    var_t = fc[var_col].values           # negative
    mean_t = fc['mean'].values
    std_t = fc['std'].values
    es_t = mean_t - std_t * ES_MULT      # negative (more negative than VaR)

    return r, var_t, es_t


def conformal_correction(r, var_t, es_t, f_cal, alpha):
    """Apply conformal correction, return corrected VaR/ES on test set."""
    T = len(r)
    n_cal = int(T * f_cal)
    r_cal, r_test = r[:n_cal], r[n_cal:]
    v_cal, v_test = var_t[:n_cal], var_t[n_cal:]
    e_cal, e_test = es_t[:n_cal], es_t[n_cal:]

    # VaR correction
    s_V = v_cal - r_cal
    q_hat_V = np.quantile(s_V, ceil((n_cal + 1) * (1 - alpha)) / n_cal)
    corrected_var = v_test - q_hat_V

    # ES correction (heuristic, Appendix C)
    violations_cal = r_cal < v_cal  # violation when r < VaR (both negative)
    if np.sum(violations_cal) > 0:
        s_E = r_cal[violations_cal] - e_cal[violations_cal]
        # Use median of ES residuals as shift
        q_hat_E = np.median(s_E)
        corrected_es = e_test + q_hat_E
    else:
        corrected_es = e_test.copy()

    # Ensure ES is more extreme than VaR
    corrected_es = np.minimum(corrected_es, corrected_var)

    return r_test, v_test, e_test, corrected_var, corrected_es


def fz0_score(r, var_neg, es_neg, alpha):
    """
    FZ0 consistent scoring function for joint (VaR, ES).

    r: realized returns
    var_neg: VaR forecasts (stored as negative)
    es_neg: ES forecasts (stored as negative, more negative than VaR)

    The FZ0 score from Fissler & Ziegel (2016), Equation (5.16):
      S(v, e, r) = -v/e * 1(r<v)*(v-r) + v + log(-e) - 1
    where v = VaR (negative), e = ES (negative).
    """
    T = len(r)
    v = var_neg  # negative
    e = es_neg   # negative, more extreme

    # Avoid division by zero or log of non-positive
    e_safe = np.where(e < -1e-10, e, -1e-10)

    indicator = (r < v).astype(float)

    # FZ0: simplified consistent score
    # S = (1/(alpha*e)) * indicator * (r - v) + v/e + log(-e) - 1
    term1 = (1.0 / (ALPHA * e_safe)) * indicator * (r - v)
    term2 = v / e_safe
    term3 = np.log(-e_safe)
    scores = term1 + term2 + term3 - 1.0

    return np.mean(scores)


def main():
    results = []

    for model_name in MODELS:
        raw_fz_list = []
        corr_fz_list = []
        count = 0

        for symbol in SYMBOLS:
            try:
                r, var_t, es_t = load_data(model_name, symbol)
            except Exception:
                continue

            T = len(r)
            n_cal = int(T * F_CAL)
            if n_cal < 100 or T - n_cal < 50:
                continue

            # Get test-set raw and corrected forecasts
            r_test, v_raw, e_raw, v_corr, e_corr = conformal_correction(
                r, var_t, es_t, F_CAL, ALPHA)

            # Compute FZ0 scores on test set
            fz_raw = fz0_score(r_test, v_raw, e_raw, ALPHA)
            fz_corr = fz0_score(r_test, v_corr, e_corr, ALPHA)

            raw_fz_list.append(fz_raw)
            corr_fz_list.append(fz_corr)
            count += 1

        if count > 0:
            mean_raw = np.mean(raw_fz_list)
            mean_corr = np.mean(corr_fz_list)
            if mean_raw != 0:
                improvement = (mean_raw - mean_corr) / abs(mean_raw) * 100
            else:
                improvement = 0.0

            results.append({
                'model': model_name,
                'raw_fz': mean_raw,
                'corr_fz': mean_corr,
                'improvement_pct': improvement,
                'n_assets': count,
            })

    df = pd.DataFrame(results)

    print("=" * 70)
    print("Fissler-Ziegel FZ0 Joint VaR-ES Scores")
    print("=" * 70)
    print(f"{'Model':18s} {'Raw FZ':>10s} {'Corr FZ':>10s} {'Improv %':>10s}")
    print("-" * 52)
    for _, row in df.iterrows():
        print(f"{row['model']:18s} {row['raw_fz']:10.4f} {row['corr_fz']:10.4f} "
              f"{row['improvement_pct']:9.1f}%")

    # Separate genuine recalibration from replacement regime
    # (Chronos models have essentially zero raw VaR => FZ blows up)
    genuine = df[df['raw_fz'].abs() < 1000].copy()
    replacement = df[df['raw_fz'].abs() >= 1000].copy()

    print("-" * 52)
    if len(genuine) > 0:
        print(f"{'Mean (genuine)':18s} {genuine['raw_fz'].mean():10.4f} "
              f"{genuine['corr_fz'].mean():10.4f} "
              f"{genuine['improvement_pct'].mean():9.1f}%")
    if len(replacement) > 0:
        print(f"\nReplacement regime (raw FZ undefined due to severe miscalibration):")
        for _, row in replacement.iterrows():
            print(f"  {row['model']:18s}: Corr FZ = {row['corr_fz']:.4f}")

    # LaTeX table
    print("\nLaTeX table:")
    print(r"\begin{table}[H]")
    print(r"  \centering")
    print(r"  \caption{Fissler--Ziegel FZ$_0$ joint VaR-ES scores (mean across 24 assets, $\alpha = 0.01$). Lower is better. The ES component uses the Gaussian heuristic (Appendix~\ref{app:es}). Chronos raw scores are undefined due to severe miscalibration ($|\qV|/|\mathrm{VaR}_{\mathrm{raw}}| > 1$).}")
    print(r"  \label{tab:fz_scores}")
    print(r"  \footnotesize")
    print(r"  \begin{tabular}{@{}lrrr@{}}")
    print(r"    \toprule")
    print(r"    Model & Raw FZ$_0$ & Corrected FZ$_0$ & Improvement\,\% \\")
    print(r"    \midrule")
    for _, row in df.iterrows():
        if row['raw_fz'] > 1000:
            # Replacement regime: raw is undefined
            print(f"    {row['model']:18s} & --- & {row['corr_fz']:.2f} & --- \\\\")
        else:
            print(f"    {row['model']:18s} & {row['raw_fz']:.2f} & {row['corr_fz']:.2f} "
                  f"& {row['improvement_pct']:.1f} \\\\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")

    # Save
    out_dir = BASE / 'results'
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / 'fz_scores_summary.csv', index=False)

    return df


if __name__ == '__main__':
    main()
